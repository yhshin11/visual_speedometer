"""Processing"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from simple_parsing import ArgumentParser
from simple_parsing.helpers import list_field
import torch

from visual_speedometer import data_utils

@dataclass
class Config:
    """ Help string for this group of command-line arguments """
    # Steps to execute
    steps: List[str] = list_field('read_frames', 'compute_flow', 'visualize')
    # Directory containing Frame images
    image_dir: str = 'data/custom/processed/train/'
    # Path to ground truth label text file
    label_path: str = 'data/custom/train.txt'
    # Path to predicted label text file
    pred_label_path: str = 'data/custom/train.pred.txt'
    # Testing flag
    testing: bool = False


def preprocess():
    """Preprocess function"""
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")

    args = parser.parse_args()
    cfg = args.config
    print("config:", cfg)

    frame_path = Path(cfg.image_dir).with_suffix('.frames.pt')
    flow_path = Path(cfg.image_dir).with_suffix('.flows.pt')

    processing = data_utils.FrameProcessing()

    if 'read_frames' in cfg.steps:
        # Convert video to frames
        frames = processing.load_frames_from_images(image_dir=cfg.image_dir, save_path=frame_path)
    else:
        return

    if 'compute_flow' in cfg.steps:
        if cfg.testing:
            frames = frames[:10+1]
            test_flows_file = '_flows_test.pt'
            if Path(test_flows_file).exists():
                flows = torch.load(test_flows_file)
            else:
                flows = processing.frames_to_flow(frames=frames, save_path=flow_path)
                torch.save(flows[:10], test_flows_file)
        else:
            # Compute flow from frames
            flows = processing.frames_to_flow(frames=frames, save_path=flow_path)
    else:
        return
    

    if 'visualize' in cfg.steps:
        flow_images = processing.flows_to_image(flows)
        if cfg.testing:
            flow_images = flow_images[:10]
        processing.output_combined_images(cfg.image_dir, frames=frames, flow_images=flow_images, labels=cfg.label_path, predicted_labels=cfg.pred_label_path)
    else:
        return


if __name__=='__main__':
    preprocess()
    print()
