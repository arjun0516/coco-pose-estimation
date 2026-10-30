# COCO Human Pose Estimation

## Project Overview
This project performs human pose estimation on a video using a pretrained YOLO pose model trained for COCO keypoints.

## Features
- Detects 17 human body keypoints
- Uses COCO human pose estimation
- Displays white skeleton lines and keypoints
- Uses a black background
- Processes video frame by frame
- Saves the result as output.mp4

## Technologies Used
- Python
- Ultralytics YOLO Pose
- OpenCV
- NumPy

## How to Run
1. Install Python.
2. Install dependencies using `pip install -r requirements.txt`.
3. Place a video named `input.mp4` in the project folder.
4. Run `python pose_estimation.py`.
5. Open `output.mp4` to view the result.

## Output
The processed video shows estimated human body keypoints and skeleton connections in white against a black background.