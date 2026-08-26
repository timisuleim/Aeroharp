"""
record_freeform.py

Records ONE longer, continuous webcam clip -- unlike record.py, this is not
for building the labeled training dataset. It's for testing generate.py's
multi-segment logic on a real video where you deliberately switch gestures
partway through (e.g. start with a fist, switch to open_palm halfway).

Usage:
    python record_freeform.py [duration_seconds]

Output:
    data/freeform_test.avi
"""

import cv2
import sys
import time

DEFAULT_DURATION = 10  # seconds
OUTPUT_PATH = "data/freeform_test.avi"
FPS = 20


def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DURATION

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (frame_width, frame_height))

    print(f"Recording {duration}s. Switch gestures partway through! Press 'q' to stop early.")
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        cv2.putText(frame, f"RECORDING: {elapsed:.1f}s / {duration}s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        writer.write(frame)
        cv2.imshow("Freeform Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
