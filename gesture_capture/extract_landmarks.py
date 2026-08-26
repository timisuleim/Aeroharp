"""
extract_landmarks.py

Runs MediaPipe hand tracking on every recorded gesture video and saves the
hand landmark coordinates to a single CSV file that the classifier can train on.

Each video produces one row per frame where a hand was detected, with:
    gesture label, clip_id (which video it came from), then 21 landmarks x (x, y, z).

The clip_id column lets the classifier split train/test by whole video clips
instead of by individual frames, which matters because frames from the same
clip look nearly identical to each other (see train_classifier.py).

Usage:
    python extract_landmarks.py

Input:  data/gestures/<gesture_name>/*.avi   (created by record.py)
Output: data/landmarks.csv
"""

import cv2
import mediapipe as mp
import os
import csv

INPUT_DIR = "data/gestures"
OUTPUT_FILE = "data/landmarks.csv"

mp_hands = mp.solutions.hands


def landmarks_to_row(hand_landmarks, gesture_label, clip_id):
    """Flattens MediaPipe's 21 hand landmarks into a single row: [label, clip_id, x0, y0, z0, ...]"""
    row = [gesture_label, clip_id]
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return row


def process_video(video_path, gesture_label, clip_id, hands, writer):
    """Runs MediaPipe on every frame of one video, writes a row per frame with a detected hand."""
    cap = cv2.VideoCapture(video_path)
    frames_written = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe expects RGB, OpenCV gives us BGR by default
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Only use the first detected hand (we only expect one hand in frame)
            hand_landmarks = results.multi_hand_landmarks[0]
            row = landmarks_to_row(hand_landmarks, gesture_label, clip_id)
            writer.writerow(row)
            frames_written += 1

    cap.release()
    return frames_written


def build_header():
    """Column names: label, clip_id, then lm0_x, lm0_y, lm0_z, lm1_x, ... for all 21 landmarks."""
    header = ["gesture", "clip_id"]
    for i in range(21):
        header.extend([f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"])
    return header


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    gestures = [
        name for name in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, name))
    ]
    print(f"Found gestures: {gestures}")

    with open(OUTPUT_FILE, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(build_header())

        # static_image_mode=False lets MediaPipe track across frames within a video,
        # which is faster and more stable than treating every frame as brand new.
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:

            total_frames = 0
            for gesture in gestures:
                gesture_dir = os.path.join(INPUT_DIR, gesture)
                video_files = [
                    f for f in os.listdir(gesture_dir) if f.endswith(".avi")
                ]

                for video_file in video_files:
                    video_path = os.path.join(gesture_dir, video_file)
                    # clip_id uniquely identifies this video, e.g. "fist/fist_01_1234567890.avi"
                    clip_id = f"{gesture}/{video_file}"
                    frames_written = process_video(video_path, gesture, clip_id, hands, writer)
                    total_frames += frames_written
                    print(f"  {video_file}: {frames_written} frames with a detected hand")

    print(f"\nDone. Wrote {total_frames} total frames to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
