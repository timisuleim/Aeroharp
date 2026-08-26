"""
map_to_mrt2.py

Takes a gesture video, runs it through MediaPipe + the trained classifier
(model.pkl), and turns each frame into a control signal for Magenta
RealTime 2:

    - discrete gesture prediction  -> a text style prompt (e.g. "fist" -> "ambient")
    - continuous hand openness     -> intensity (0-127, like a MIDI CC value)
    - continuous hand height       -> pitch (mapped to a MIDI note range)

A temporal smoothing pass is applied to the raw per-frame gesture predictions
before assigning style prompts: each frame's gesture is replaced with the
majority vote across a small surrounding window. Frame-by-frame classification
has no memory, so a single noisy frame (mid-transition, odd angle, etc.) can
briefly "flip" to a different predicted gesture and back -- smoothing collapses
these flickers into the surrounding, intended gesture instead of letting them
spawn spurious short segments in generate.py.

The output is a JSON file listing one control event per frame, with a
timestamp. This file is what mrt2_runner/generate.py (running on a cloud
GPU) actually reads to drive Magenta RealTime 2's offline inference --
this script itself does NOT run MRT2, since that needs a GPU this laptop
doesn't have.

Output filename is derived from the input video's filename, so running
this on multiple clips doesn't overwrite previous results.

Usage:
    python map_to_mrt2.py path/to/some_gesture_video.avi

Output:
    <video_name>_control_sequence.json
"""

import sys
import cv2
import mediapipe as mp
import numpy as np
import pickle
import json
import os
from collections import Counter

MODEL_PATH = "../classifier/model.pkl"

# Maps each discrete gesture to a text style prompt MRT2 can condition on.
# Edit these to taste -- this is the actual "creative" mapping decision of the project.
STYLE_MAP = {
    "fist": "dark ambient drone",
    "open_palm": "bright upbeat synth pop",
    "peace_sign": "playful jazzy piano",
    "point": "sharp percussive electronic",
}

# MIDI note range that hand height gets mapped onto (low hand -> low note, high hand -> high note)
MIDI_NOTE_MIN = 48   # roughly C3
MIDI_NOTE_MAX = 84   # roughly C6

# Temporal smoothing: each frame's gesture becomes the majority vote across
# this many frames centered on it (must be odd). At 20fps, 5 frames = 0.25s --
# short enough to preserve real gesture changes, long enough to kill 1-2 frame
# flicker noise.
SMOOTHING_WINDOW = 5

mp_hands = mp.solutions.hands


def load_classifier():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def landmarks_to_feature_vector(hand_landmarks):
    """Same 63-number flattening used in extract_landmarks.py, so it matches what the classifier was trained on."""
    features = []
    for lm in hand_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features


def compute_hand_openness(hand_landmarks):
    """
    Rough measure of how 'open' the hand is: average distance from the
    wrist (landmark 0) to each fingertip (landmarks 4, 8, 12, 16, 20).
    A fist has short distances (closed), an open palm has long ones.
    """
    wrist = hand_landmarks.landmark[0]
    fingertip_ids = [4, 8, 12, 16, 20]

    distances = []
    for tip_id in fingertip_ids:
        tip = hand_landmarks.landmark[tip_id]
        dist = np.sqrt(
            (tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2 + (tip.z - wrist.z) ** 2
        )
        distances.append(dist)

    return float(np.mean(distances))


def compute_hand_height(hand_landmarks):
    """
    Hand height in frame, 0 (bottom of frame) to 1 (top of frame).
    MediaPipe's y coordinate is 0 at the top and 1 at the bottom, so we flip it.
    """
    wrist_y = hand_landmarks.landmark[0].y
    return 1.0 - wrist_y


def scale(value, in_min, in_max, out_min, out_max):
    """Linearly rescales value from [in_min, in_max] into [out_min, out_max], clamped to the output range."""
    value = max(in_min, min(in_max, value))
    scaled = (value - in_min) / (in_max - in_min)
    return out_min + scaled * (out_max - out_min)


def smooth_gestures(raw_gestures, window):
    """
    Replaces each entry in raw_gestures with the majority vote across a
    window of size `window` centered on it (clipped at the sequence edges).
    raw_gestures is a plain list of gesture label strings, one per
    hand-detected frame (NOT one per video frame -- gaps where no hand was
    detected are simply absent, so this smooths over detected frames only).
    """
    n = len(raw_gestures)
    half = window // 2
    smoothed = []

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window_slice = raw_gestures[start:end]
        majority = Counter(window_slice).most_common(1)[0][0]
        smoothed.append(majority)

    return smoothed


def main():
    if len(sys.argv) < 2:
        print("Usage: python map_to_mrt2.py path/to/gesture_video.avi")
        return

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    # Derive output filename from the input video, so different clips don't
    # overwrite each other's control_sequence.json
    output_file = os.path.splitext(os.path.basename(video_path))[0] + "_control_sequence.json"

    clf = load_classifier()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20  # fall back to 20 if the video doesn't report FPS

    # First pass: collect raw per-detected-frame data (gesture, openness,
    # height, frame_index, time) WITHOUT assigning style prompts yet -- we
    # need the whole sequence before we can smooth it.
    raw_frames = []
    frame_index = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                feature_vector = landmarks_to_feature_vector(hand_landmarks)
                predicted_gesture = clf.predict([feature_vector])[0]

                openness = compute_hand_openness(hand_landmarks)
                height = compute_hand_height(hand_landmarks)

                intensity = round(scale(openness, 0.05, 0.35, 0, 127))
                pitch = round(scale(height, 0.0, 1.0, MIDI_NOTE_MIN, MIDI_NOTE_MAX))

                raw_frames.append({
                    "frame": frame_index,
                    "time_seconds": round(frame_index / fps, 3),
                    "raw_gesture": predicted_gesture,
                    "pitch_midi": pitch,
                    "intensity": intensity,
                })

            frame_index += 1

    cap.release()

    # Second pass: smooth the gesture sequence, then assign style prompts
    # based on the SMOOTHED gesture, not the raw per-frame prediction.
    raw_gesture_list = [f["raw_gesture"] for f in raw_frames]
    smoothed_gesture_list = smooth_gestures(raw_gesture_list, SMOOTHING_WINDOW)

    flips_corrected = sum(1 for r, s in zip(raw_gesture_list, smoothed_gesture_list) if r != s)

    control_events = []
    for raw, smoothed_gesture in zip(raw_frames, smoothed_gesture_list):
        control_events.append({
            "frame": raw["frame"],
            "time_seconds": raw["time_seconds"],
            "gesture": smoothed_gesture,
            "style_prompt": STYLE_MAP.get(smoothed_gesture, "ambient"),
            "pitch_midi": raw["pitch_midi"],
            "intensity": raw["intensity"],
        })

    with open(output_file, "w") as f:
        json.dump(control_events, f, indent=2)

    print(f"Processed {frame_index} frames, {len(control_events)} had a detected hand.")
    print(f"Smoothing (window={SMOOTHING_WINDOW}) changed {flips_corrected}/{len(control_events)} frame predictions.")
    print(f"Saved control sequence to {output_file}")


if __name__ == "__main__":
    main()
