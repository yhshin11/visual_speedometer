"""Data utils"""
from pathlib import Path
import time
import cv2
import numpy as np
from imutils.video import FileVideoStream, FPS
from tqdm import tqdm

def _print(object, *args, **kwargs):
    tqdm.write(str(object), *args, **kwargs)

def load_video(cfg):
    """Load a video with an accompanying text file, with each line of the text file containing the speed in each frame of the video."""
    video_path = Path(cfg.data_root) / cfg.dataset
    train_video_path = video_path / 'train.mp4'
    train_text_path = video_path / 'train.txt'

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
        ret, frame_prev = cap.read()
        prvs = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame_prev)
        hsv[..., 1] = 255
        i = 1
        with tqdm(total=num_frames, colour='green') as pbar:
            while(1):
                # _print(f'Converting frame number: {i}')
                ret, frame = cap.read()
                if not ret:
                    _print('No frames grabbed!')
                    break
                next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                output_path = output_dir / f'{i:010d}.npy'
                np.save(output_path, flow)
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
                prvs = next
                i += 1
    else:
        stream = FileVideoStream(str(video_path))
        stream.start()
        fps = FPS().start()
        frame_prev = stream.read()
        prvs = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame_prev)
        hsv[..., 1] = 255
        i = 0
        while(1):
            t0 = time.time()
            _print(f'Converting frame number: {i}')
            try:
                frame = stream.read()
            except:
                break
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            t1 = time.time()
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            t2 = time.time()
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            output_image_path = output_dir / f'{i:010d}.png'
            t3 = time.time()
            cv2.imwrite(str(output_image_path), bgr)
            t4 = time.time()
            prvs = next

            if i >= max_frames:
                break
            fps.update()
            i += 1
            _print(f'{t1-t0}, {t2-t1}, {t3-t2}, {t4-t3}')
        stream.stop()

    fps.stop()
    _print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    _print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()

