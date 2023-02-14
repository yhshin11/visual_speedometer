"""Train on input data"""
from pathlib import Path
from dataclasses import dataclass
from simple_parsing import ArgumentParser

import cv2
import numpy as np

from visual_speedometer import data_utils

@dataclass
class Config:
    """ Help string for this group of command-line arguments """
    # Data root directory
    data_root: str = './data'
    # Input video
    video_file: str = 'mars/train.mp4'
    # Maximum frames to process
    max_frames: int = 1e9
    # Suffix for output directory
    suffix: str = ''

def preprocess():
    """Preprocess function"""
    parser = ArgumentParser()
    parser.add_arguments(Config, dest="config")

    args = parser.parse_args()
    cfg = args.config
    print("config:", cfg)

    # Convert video to optical flow images
    data_utils.video_to_flow(Path(cfg.data_root)/cfg.video_file, suffix=cfg.suffix, max_frames=cfg.max_frames)


if __name__=='__main__':
    preprocess()
