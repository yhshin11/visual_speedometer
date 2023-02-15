# visual_speedometer

Measure speed of a car from video.

## Instructions

Use `ffmpeg` to convert video to numbered frame images:

```bash
ffmpeg -r 1 -i INPUT_VIDEO FRAME_DIR/%10d.png
```

Convert all frames to pytorch tensor and save to `.pt`:

```bash
python processing.py --steps read_frames -i FRAME_DIR -o FRAME_DIR.pt
```

Compute optical flow using `torchvision` and save batches of optical flow to `.pt`:

```bash
python processing.py --steps compute_flow -i FRAME_DIR.pt -o FRAME_DIR.flow.pt
```

Train model on optical flow (optionally stacked with original frame) to predict labels:

```bash
python train.py --flow FRAME_DIR.flow.pt --frame FRAME_DIR.pt --labels LABELS.txt
```

### [Optional] Visualize results

Output images with original frame, optical flow and ground truth label:

```bash
python processing.py --steps visualize --flow FRAME_DIR.flow.pt --frame FRAME_DIR.pt --labels LABELS.txt -o FRAME_DIR_viz
```

Convert back to video:

```bash
ffmpeg -f image2 -i FRAME_DIR_viz/%10d.png FRAME_DIR_viz.mp4
```