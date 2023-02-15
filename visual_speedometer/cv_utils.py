"""OpenCV utils"""
from contextlib import contextmanager
import cv2
import numpy as np

WIDTH = 1280
HEIGHT = 720
CHANNEL = 3
FPS = 30
LENGTH = 5  # Length in seconds


@contextmanager
def video_output(filename='out.mp4', fourcc=None, fps=30, frameSize=None, ):
    """Context manager for video output"""
    if fourcc == None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    if frameSize == None:
        frameSize = (1280, 720)
    video = cv2.VideoWriter("test.mp4", fourcc, float(fps), frameSize)
    try:
        yield video
    finally:
        pass
        # video.release() # Automatically called by destructor


def main():
    with video_output('test.mp4') as video:
        for _ in range(FPS * LENGTH):
            image = np.random.randint(0, 255, (HEIGHT, WIDTH, CHANNEL), dtype=np.uint8)
            video.write(image)


if __name__ == "__main__":
    main()
