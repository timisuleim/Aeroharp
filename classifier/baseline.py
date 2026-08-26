"""
baseline.py

A deliberately "dumb" baseline to compare against the real classifier
(train_classifier.py). Instead of using MediaPipe hand landmarks, this
splits each video frame into a grid of regions and measures how much raw
pixel motion happens in each region, frame to frame. It then guesses the
gesture by matching a clip's motion pattern to whichever gesture had the
most similar average motion pattern in training.

This mirrors the "split the frame into regions and track pixel change"
baseline described in the project proposal, adapted to work with
shape-based hand gestures (fist, open_palm, peace_sign, point) instead
of gestures performed in different screen locations.

If this baseline scores much lower than train_classifier.py, that's
evidence the real classifier (using actual hand landmark shape data)
is adding real value over a naive motion-based guess.

Usage:
    python baseline.py

Input:  ../gesture_capture/data/gestures/<gesture_name>/*.avi
"""

import cv2
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

INPUT_DIR = "../gesture_capture/data/gestures"
GRID_ROWS = 2
GRID_COLS = 2          # splits each frame into a 2x2 grid = 4 regions
TEST_FRACTION = 0.2     # ~20% of clips per gesture held out for testing
RANDOM_SEED = 42


def region_motion_profile(video_path):
    """
    Reads a video and returns a single vector: the average amount of
    frame-to-frame pixel change in each grid region, across the whole clip.
    This is the "dumb" feature representation this baseline uses instead
    of MediaPipe hand landmarks.
    """
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    region_totals = np.zeros(GRID_ROWS * GRID_COLS)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            h, w = diff.shape
            row_h, col_w = h // GRID_ROWS, w // GRID_COLS

            region_idx = 0
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    region = diff[r * row_h:(r + 1) * row_h, c * col_w:(c + 1) * col_w]
                    region_totals[region_idx] += region.mean()
                    region_idx += 1

            frame_count += 1

        prev_gray = gray

    cap.release()

    if frame_count == 0:
        return None
    return region_totals / frame_count


def collect_clips():
    """Returns {gesture: [video_path, ...]} for every recorded clip."""
    gestures = [
        name for name in os.listdir(INPUT_DIR)
        if os.path.isdir(os.path.join(INPUT_DIR, name))
    ]
    clips_by_gesture = {}
    for gesture in gestures:
        gesture_dir = os.path.join(INPUT_DIR, gesture)
        clips_by_gesture[gesture] = [
            os.path.join(gesture_dir, f)
            for f in os.listdir(gesture_dir) if f.endswith(".avi")
        ]
    return clips_by_gesture


def split_train_test(clips_by_gesture):
    """Splits each gesture's clips into train/test, holding whole clips out (not frames)."""
    rng = random.Random(RANDOM_SEED)
    train, test = {}, []

    for gesture, paths in clips_by_gesture.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(len(shuffled) * TEST_FRACTION))
        test_paths = shuffled[:n_test]
        train_paths = shuffled[n_test:]

        train[gesture] = train_paths
        for p in test_paths:
            test.append((gesture, p))

    return train, test


def main():
    clips_by_gesture = collect_clips()
    train_clips, test_clips = split_train_test(clips_by_gesture)

    # Build a "profile" for each gesture: the average motion pattern across
    # its training clips. This is the entire "model" -- just an average.
    gesture_profiles = {}
    for gesture, paths in train_clips.items():
        profiles = [region_motion_profile(p) for p in paths]
        profiles = [p for p in profiles if p is not None]
        if profiles:
            gesture_profiles[gesture] = np.mean(profiles, axis=0)

    print("Learned motion profiles (region averages) per gesture:")
    for gesture, profile in gesture_profiles.items():
        print(f"  {gesture}: {np.round(profile, 3)}")

    # Classify each test clip by nearest gesture profile (simple 1-nearest-centroid)
    y_true, y_pred = [], []
    for gesture, path in test_clips:
        clip_profile = region_motion_profile(path)
        if clip_profile is None:
            continue

        distances = {
            g: np.linalg.norm(clip_profile - profile)
            for g, profile in gesture_profiles.items()
        }
        predicted = min(distances, key=distances.get)

        y_true.append(gesture)
        y_pred.append(predicted)

    accuracy = accuracy_score(y_true, y_pred)
    labels = sorted(gesture_profiles.keys())

    print(f"\nBaseline test accuracy: {accuracy:.2%}  (on {len(y_true)} held-out clips)")
    print("\nConfusion matrix (rows = actual gesture, columns = predicted gesture):")
    print(f"Labels order: {labels}")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    print("\nPer-gesture precision/recall:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))


if __name__ == "__main__":
    main()
