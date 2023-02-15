"""Data utils"""
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from imutils.video import FileVideoStream, FPS
from PIL import Image, ImageDraw, ImageFont
import tqdm
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
from torchvision.models.optical_flow import raft_large, raft_small
from torchvision.utils import flow_to_image

from .cv_utils import video_output
from .utils import timer
from .utils import append_suffix

def _print(obj, *args, **kwargs):
    tqdm.tqdm.write(str(obj), *args, **kwargs)

def read_labels(path, scale=1.0):
    """Read labels from text file.
    Returned labels are multiplied by `scale`. Use `scale=3.6` to convert m/s to km/h."""
    path = Path(path)
    labels = torch.Tensor(np.loadtxt(path)[1:]) # All but first element
    labels *= scale
    return labels

class FrameProcessing():
    """Class to handle various processing tasks"""
    def __init__(self) -> None:
        pass

    @timer
    def load_frames_from_images(self, image_dir, save_path=None):
        """Load image frames from images and return single tensor containing all of them"""
        if save_path:
            save_path = Path(save_path)
        else:
            save_path = Path(image_dir).with_suffix('.frames.pt')
        if save_path.exists():
            _print(f"💬: {str(save_path)} already exists. Reading instead...")
            frames = torch.load(str(save_path))
            return frames

        # Find all files
        image_paths = sorted(Path(image_dir).glob('*.png'))
        # TODO Check that all files exist and have correct name
        # num_files = len(image_paths)
        tensors = []
        for image_path in tqdm.tqdm(image_paths):
            image_tensor = torchvision.io.read_image(str(image_path))
            tensors.append(image_tensor)
        frames = torch.stack(tensors)
        torch.save(frames, save_path)
        return frames

    @timer
    @torch.no_grad()
    def frames_to_flow(self, frames, save_path=None, batch_size=16):
        """Use pytorch to compute optical flow from frames in batches"""
        if save_path.exists():
            _print(f"💬: {str(save_path)} already exists. Reading instead...")
            flows = torch.load(str(save_path))
            return flows

        weights = Raft_Small_Weights.DEFAULT
        transforms = weights.transforms()

        def preprocess(img1_batch, img2_batch):
            img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
            img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
            return transforms(img1_batch, img2_batch)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(device)
        model = model.eval()

        num_batches = int(np.ceil(frames.shape[0] / batch_size))
        flows = []
        print("⏳: Processing frames...")
        for i in tqdm.trange(num_batches):
            img1_batch = frames[i*batch_size:(i+1)*batch_size, :]
            img2_batch = frames[i*batch_size+1:(i+1)*batch_size+1, :]
            # Trim to same size
            if img1_batch.shape != img2_batch.shape:
                img1_batch = img1_batch[:img2_batch.shape[0], :]
            # Predict on batch
            img1_batch, img2_batch = preprocess(img1_batch, img2_batch)
            list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
            # RAFT model returns optical flow at different iterations, but we only care about the last iteration
            predicted_flows = list_of_flows[-1]
            flows.append(predicted_flows.to("cpu"))
        print("✅: Done processing frames!")
        print("⏳: Saving to file..")
        torch.save(flows, save_path)
        print("✅: Saved to file!")

        return flows

    @timer
    def flows_to_image(self, flows):
        """Convert batches of optical flow tensors to [N, C, H, W] optical flow image tensors"""
        image_batches = []
        for flow_batch in tqdm.tqdm(flows):
            image_batch = flow_to_image(flow_batch)
            image_batches.append(image_batch)
        flow_images = torch.vstack(image_batches)
        return flow_images

    @timer
    def output_combined_images(self, image_dir, save_path=None, frames=None, flow_images=None, labels=None, predicted_labels=None):
        """beepboopbop"""
        if save_path:
            save_path = Path(save_path)
        else:
            save_path = append_suffix(Path(image_dir), "__combined")
        if save_path.exists():
            _print(f"Warning: Overwriting {str(save_path)}")
            shutil.rmtree(save_path)
            # raise FileExistsError()
        save_path.mkdir()
        _print(f"Saving to {str(save_path)}")

        num_frames = frames.shape[0]
        assert frames.shape[0] - 1 == flow_images.shape[0], "Number of flow images does not match number of frames"

        _resize = False
        if frames.shape[-2:] != flow_images.shape[-2:]:
            _resize = True
            _print("Warning: Frames and flow images are different shapes. Resizing...")

        labels = read_labels(labels)
        predicted_labels = read_labels(predicted_labels)
        xy = [50, 50]
        font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
        text_color = (0, 255, 0) # Green
        stroke_color = (0, 0, 0) # Black
        stroke_width = 1

        def draw_text_annotation(image, label, predicted_label):
            text = f"""Ground truth | Predicted
            {label:0.2f} | {predicted_label:0.2f}
            """
            draw = ImageDraw.Draw(image)
            draw.text(
                (xy[0], xy[1]),
                text,
                font=font,
                fill=text_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color,
            )

        for iframe in tqdm.trange(num_frames-1):
            frame = frames[iframe+1, :]
            flow_image = flow_images[iframe, :]
            if _resize:
                flow_image = F.resize(flow_image, frames.shape[-2:])
            # Concatenate vertically
            combined_image = torch.concat((frame, flow_image), dim=1).permute((1, 2, 0)).cpu().numpy()
            image = Image.fromarray(combined_image, mode="RGB")
            draw_text_annotation(image, labels[iframe+1], predicted_labels[iframe+1])
            image_path = save_path / f"{iframe:010d}.png"
            image.save(image_path)


def flow_to_rgb(flow, dst_array):
    """Convert 2D optical flow array to frame"""
    INPUT_MIN = -1 
    INPUT_MAX = 10 
    OUTPUT_MIN = 0
    OUTPUT_MAX = 255
    EPS = 1e-20

    x = flow[..., 0] 
    y = flow[..., 1] 
    # Apply linear scaling
    dst_array[..., 0] = ((x > 0) * np.log(np.abs(x) + EPS) - INPUT_MIN) * (OUTPUT_MAX - OUTPUT_MIN) / (INPUT_MAX - INPUT_MIN) + OUTPUT_MIN
    dst_array[..., 2] = ((x < 0) * np.log(np.abs(x) + EPS) - INPUT_MIN) * (OUTPUT_MAX - OUTPUT_MIN) / (INPUT_MAX - INPUT_MIN) + OUTPUT_MIN
    dst_array[..., 1] = 0

    dst_array = np.clip(dst_array, 0, 255)
    bgr = dst_array.astype(np.uint8)
    # bgr = cv2.cvtColor(dst_array.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return bgr


def flow_to_hsv(flow, hsv):
    """Convert 2D optical flow array to frame"""
    # INPUT_MIN = -500
    # INPUT_MAX = 100
    # OUTPUT_MIN = 0
    # OUTPUT_MAX = 255
    INPUT_MIN = -20
    INPUT_MAX = 20
    OUTPUT_MIN = 0
    OUTPUT_MAX = 255
    EPS = 1e-20

    x = flow[..., 0] 
    y = flow[..., 1] 
    r = np.sqrt(x**2 + y**2)
    t = np.arctan2(y, x) # Return values in [-pi, +pi]
    # OpenCV HSV hue values go from 0 to 180
    hsv[..., 0] = (t - (-np.pi)) * 180 / 2 / np.pi
    hsv[..., 1] = 255
    v = (np.log(r + EPS) - INPUT_MIN) * (OUTPUT_MAX - OUTPUT_MIN) / (INPUT_MAX - INPUT_MIN) + OUTPUT_MIN
    hsv[..., 2] = np.clip(v, 0, 255)
    # hsv[..., 2] = np.clip(r / INPUT_MAX * OUTPUT_MAX, 0, 255)
    # hsv[..., 2] = cv2.normalize(r / INPUT_MAX * OUTPUT_MAX, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return bgr

    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # # hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 0] = ang * 180 / np.pi
    # min = np.min(mag)
    # max = np.max(mag)
    # _print(f'{min}, {max}')
    # # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # hsv[..., 2] = mag
    # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # output_image_path = output_dir / f'{i:010d}.png'
    # cv2.imwrite(str(output_image_path), bgr)

def video_to_flow(input_video, suffix='', mode='default', max_frames=1e9):
    """Take a video and output optical flow images based on consecutive frames.
    Optical flow at frame N is calculating using frames N and N-1."""
    video_path = Path(input_video)
    video_name = video_path.stem
    # output_dir = Path(video_path.parent / video_name)
    output_dir_name = str(video_name) + suffix
    output_dir = Path(video_path.parent / output_dir_name)
    output_dir.mkdir(exist_ok=True)
    _print(output_dir)

    if mode=='default':
        # Single-thread
        cap = cv2.VideoCapture(str(video_path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = FPS().start()
        ret, frame = cap.read()
        H, W, C = (frame.shape[0], frame.shape[1], frame.shape[2])
        video_frame_size = (W, 2*H) # Stack vertically
        # video_frame_size = (W, H) # Stack vertically
        prvs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame, dtype=np.float32)
        hsv[..., 1] = 255
        hsv = np.zeros_like(frame)
        i = 1
        with video_output(frameSize=video_frame_size) as video, tqdm.tqdm(total=num_frames, colour='green') as pbar:
            while(1):
                # _print(f'Converting frame number: {i}')
                ret, frame = cap.read()
                if not ret:
                    _print('No frames grabbed!')
                    break
                next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs_frame, next_frame, None, 0.5, 3, 50, 3, 5, 1.1, 0)
                output_path = output_dir / f'{i:010d}.npy'
                np.save(output_path, flow)
                # Save to video
                # flow_frame = flow_to_hsv(flow, hsv)
                flow_frame = flow_to_rgb(flow, hsv)
                # output_frame = cv2.vconcat(frame, flow_frame)
                output_frame = np.vstack((frame, flow_frame))
                video.write(output_frame)
                # video.write(frame)
                # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                # # hsv[..., 0] = ang * 180 / np.pi / 2
                # hsv[..., 0] = ang * 180 / np.pi
                # min = np.min(mag)
                # max = np.max(mag)
                # _print(f'{min}, {max}')
                # # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                # hsv[..., 2] = mag
                # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # output_image_path = output_dir / f'{i:010d}.png'
                # cv2.imwrite(str(output_image_path), bgr)

                if i >= max_frames:
                    break
                pbar.update(1)
                fps.update()
                prvs_frame = next_frame
                i += 1
    else:
        stream = FileVideoStream(str(video_path))
        stream.start()
        fps = FPS().start()
        frame = stream.read()
        prvs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        i = 0
        while True:
            t0 = time.time()
            _print(f'Converting frame number: {i}')
            try:
                frame = stream.read()
            except Exception as e:
                break
            next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t1 = time.time()
            flow = cv2.calcOpticalFlowFarneback(prvs_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            t2 = time.time()
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            output_image_path = output_dir / f'{i:010d}.png'
            t3 = time.time()
            cv2.imwrite(str(output_image_path), bgr)
            t4 = time.time()
            prvs_frame = next_frame

            if i >= max_frames:
                break
            fps.update()
            i += 1
            _print(f'{t1-t0}, {t2-t1}, {t3-t2}, {t4-t3}')
        stream.stop()

    fps.stop()
    _print(f"[INFO] elasped time: {fps.elapsed():.2f}")
    _print(f"[INFO] approx. FPS: {fps.fps():.2f}")

    cv2.destroyAllWindows()


class SpeedometerDataset(torch.utils.data.Dataset):
    """Dataset to load speedometer data"""
    def __init__(self, label_file, array_dir) -> None:
        # Single column text file, with row N containing the speed of car at frame N.
        self.label_file = label_file
        # Directory where numpy arrays are saved. File name should be in the format f'{N:010d}.npy' where N is frame index.
        self.array_dir = array_dir
        self.labels = torch.Tensor(np.loadtxt(self.label_file)[1:]) # All but first element
        self.array_files = []
        # Check that files exist in array_dir
        for i in range(len(self.labels)):
            # Starting from frame 1
            array_file = Path(self.array_dir) / f"{i+1:010d}.npy"
            assert array_file.exists(), f"File for frame {i+1} - {str(array_file)} does not exist!"
            self.array_files.append(array_file)
        assert len(self.labels) == len(self.array_files)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        array = np.load(self.array_files[idx])
        array_transposed = np.transpose(array, [2, 0, 1])
        tensor = torch.Tensor(array_transposed)
        tensor = F.normalize(tensor, mean=[-2.0, 2.0], std=[10.0, 10.0])
        return (tensor, label)

class FlowDataset(torch.utils.data.Dataset):
    """Dataset to load optical flow data.
    Assumes optical flow is stored in a list of tensors, with each tensor representing a batch of optical flow."""
    def __init__(self, label_file, flow_file) -> None:
        # Single column text file, with row N containing the speed of car at frame N.
        self.label_file = label_file
        # .pt file where flows are saved
        self.flow_file = flow_file
        self.labels = torch.Tensor(np.loadtxt(self.label_file)[1:]) # All but first element

        # Load file
        self.flow_batches = torch.load(str(self.flow_file))
        self.batch_size = self.flow_batches[0].shape[0]
        num_flows = 0
        for flow_batch in self.flow_batches:
            num_flows += flow_batch.shape[0]
        assert num_flows == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        batch_idx = idx // self.batch_size
        flow_idx = idx % self.batch_size
        flow = self.flow_batches[batch_idx][flow_idx, :]
        return (flow, label)
