"""
train_classifier.py

Trains a classifier to predict hand gesture from MediaPipe landmark data,
using the CSV produced by extract_landmarks.py.

IMPORTANT: splits train/test by whole video CLIP, not by individual frame.
Frames from the same clip look nearly identical (your hand barely moves in
2 seconds), so a random frame-level split would let the model "cheat" by
training on frames from the same clip its test frames come from. Splitting
by clip means the test set contains gestures performed in reps the model
has never seen any part of, which is a fairer test of real generalization.

Prints accuracy and a confusion matrix, and saves the trained model to disk
so it can be reused later (e.g. by control_mapping/) without retraining.

Usage:
    python train_classifier.py

Input:  ../gesture_capture/data/landmarks.csv
Output: model.pkl   (the trained classifier, saved with pickle)
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

INPUT_FILE = "../gesture_capture/data/landmarks.csv"
MODEL_OUTPUT = "model.pkl"
TEST_SIZE = 0.2       # ~20% of CLIPS held out for testing (not 20% of frames)
RANDOM_SEED = 42       # fixed seed so results are reproducible between runs


def load_data():
    df = pd.read_csv(INPUT_FILE)

    # Columns: gesture, clip_id, then 63 landmark coordinates
    X = df.drop(columns=["gesture", "clip_id"])
    y = df["gesture"]
    groups = df["clip_id"]  # which video clip each row came from

    return X, y, groups


def main():
    X, y, groups = load_data()
    print(f"Loaded {len(X)} labeled frames from {groups.nunique()} clips, gestures: {sorted(y.unique())}")

    # GroupShuffleSplit ensures every row from a given clip_id goes ENTIRELY
    # into either train or test, never split across both.
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Train set: {len(X_train)} frames from {groups.iloc[train_idx].nunique()} clips")
    print(f"Test set:  {len(X_test)} frames from {groups.iloc[test_idx].nunique()} clips")
    print(f"Test clips: {sorted(groups.iloc[test_idx].unique())}")

    # RandomForest is a solid default for small tabular datasets like this one:
    # not much tuning needed, and it handles the 63 landmark features fine.
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    # Evaluate on the held-out test set (whole clips the model never saw during training)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {accuracy:.2%}")
    print("\nConfusion matrix (rows = actual gesture, columns = predicted gesture):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"Labels order: {labels}")
    print(cm)

    print("\nPer-gesture precision/recall:")
    print(classification_report(y_test, y_pred))

    # Save the trained model so control_mapping/ can load it later without retraining
    with open(MODEL_OUTPUT, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nSaved trained model to {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
