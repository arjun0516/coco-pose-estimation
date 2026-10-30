
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Input and output files
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"

# COCO pose model: predicts 17 human body keypoints
MODEL_NAME = "yolo26n-pose.pt"

# Confidence threshold for drawing keypoints
CONF_THRESHOLD = 0.30

# COCO skeleton connections (keypoint index pairs)
SKELETON = [
    (5, 6),                   # shoulders
    (5, 7), (7, 9),            # left arm
    (6, 8), (8, 10),           # right arm
    (5, 11), (6, 12),          # torso sides
    (11, 12),                  # hips
    (11, 13), (13, 15),        # left leg
    (12, 14), (14, 16),        # right leg
    (0, 1), (0, 2),            # nose to eyes
    (1, 3), (2, 4),            # eyes to ears
]

def main():
    if not Path(INPUT_VIDEO).exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_VIDEO}. "
            "Place your sample video in this folder."
        )

    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        raise RuntimeError("Could not open the input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    # MP4 output
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Estimate COCO body keypoints
            result = model.predict(frame, verbose=False)[0]

            # Start with a completely black frame
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()

                for person in keypoints:
                    # XY coordinates and confidence per keypoint
                    xy = person[:, :2]
                    if person.shape[1] >= 3:
                        confidence = person[:, 2]
                    else:
                        confidence = np.ones(len(xy))

                    visible = {}
                    for i, (x, y) in enumerate(xy):
                        if (
                            confidence[i] >= CONF_THRESHOLD
                            and 0 <= x < width
                            and 0 <= y < height
                        ):
                            visible[i] = (int(x), int(y))

                    # Draw white connecting lines
                    for a, b in SKELETON:
                        if a in visible and b in visible:
                            cv2.line(
                                canvas,
                                visible[a],
                                visible[b],
                                (255, 255, 255),
                                2,
                                cv2.LINE_AA
                            )

                    # Draw white keypoint dots
                    for point in visible.values():
                        cv2.circle(
                            canvas,
                            point,
                            4,
                            (255, 255, 255),
                            -1,
                            cv2.LINE_AA
                        )

            writer.write(canvas)

    finally:
        cap.release()
        writer.release()

    print(f"Finished! Output saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
