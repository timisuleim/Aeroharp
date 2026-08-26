"""
record.py

Records short labeled video clips of hand gestures using the webcam.
Each clip is saved to data/gestures/<gesture_name>/<timestamp>.avi

Usage:
    python record.py

Controls (while the webcam window is focused):
    r   - start/stop recording the current clip
    n   - move on to the next gesture in GESTURES
    q   - quit
"""

import cv2
import os
import time

# ---- Config: edit this list to match the gestures you actually plan to use ----
GESTURES = ["open_palm", "fist", "point", "peace_sign"]
REPS_PER_GESTURE = 10          # how many clips you want per gesture
OUTPUT_DIR = "data/gestures"
FPS = 20
CLIP_SECONDS = 2               # how long each recorded clip should be


def ensure_dirs():
    for gesture in GESTURES:
        os.makedirs(os.path.join(OUTPUT_DIR, gesture), exist_ok=True)


def record_clip(cap, gesture_name, rep_index):
    """Records a single fixed-length clip for the given gesture."""
    filename = f"{gesture_name}_{rep_index:02d}_{int(time.time())}.avi"
    filepath = os.path.join(OUTPUT_DIR, gesture_name, filename)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(filepath, fourcc, FPS, (frame_width, frame_height))

    num_frames = FPS * CLIP_SECONDS
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Show a countdown/progress overlay so you know it's recording
        seconds_left = CLIP_SECONDS - (i / FPS)
        cv2.putText(
            frame,
            f"RECORDING: {gesture_name}  ({seconds_left:.1f}s left)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        writer.write(frame)
        cv2.imshow("AeroHarp - Gesture Recorder", frame)
        cv2.waitKey(1)

    writer.release()
    print(f"Saved: {filepath}")


def main():
    ensure_dirs()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam. Check that it's connected and not in use by another app.")
        return

    gesture_index = 0
    rep_index = 1

    print("Controls: 'r' = record a clip, 'n' = next gesture, 'q' = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_gesture = GESTURES[gesture_index]
        cv2.putText(
            frame,
            f"Current gesture: {current_gesture}  (rep {rep_index}/{REPS_PER_GESTURE})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press 'r' to record, 'n' for next gesture, 'q' to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow("AeroHarp - Gesture Recorder", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            record_clip(cap, current_gesture, rep_index)
            rep_index += 1
            if rep_index > REPS_PER_GESTURE:
                rep_index = 1
                gesture_index = (gesture_index + 1) % len(GESTURES)

        elif key == ord("n"):
            gesture_index = (gesture_index + 1) % len(GESTURES)
            rep_index = 1

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
