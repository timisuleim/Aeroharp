"""
train_classifier.py

Trains a classifier to predict hand gesture from MediaPipe landmark data,
using the CSV produced by extract_landmarks.py.

Splits the data into train/test sets, trains a classifier, prints accuracy
and a confusion matrix, and saves the trained model to disk so it can be
reused later (e.g. by control_mapping/) without retraining.

Usage:
    python train_classifier.py

Input:  ../gesture_capture/data/landmarks.csv
Output: model.pkl   (the trained classifier, saved with pickle)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

INPUT_FILE = "../gesture_capture/data/landmarks.csv"
MODEL_OUTPUT = "model.pkl"
TEST_SIZE = 0.2       # 20% of the data held out for testing
RANDOM_SEED = 42       # fixed seed so results are reproducible between runs


def load_data():
    df = pd.read_csv(INPUT_FILE)

    # First column is the gesture label, the rest are the 63 landmark coordinates
    X = df.drop(columns=["gesture"])
    y = df["gesture"]

    return X, y


def main():
    X, y = load_data()
    print(f"Loaded {len(X)} labeled frames across gestures: {sorted(y.unique())}")

    # Split into train/test sets. stratify=y keeps the gesture proportions
    # roughly equal in both the train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train set: {len(X_train)} frames, Test set: {len(X_test)} frames")

    # RandomForest is a solid default for small tabular datasets like this one:
    # not much tuning needed, and it handles the 63 landmark features fine.
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    clf.fit(X_train, y_train)

    # Evaluate on the held-out test set (frames the model never saw during training)
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
