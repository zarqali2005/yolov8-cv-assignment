"""
02_live_track.py (Python 3.12.0)
---------------------------------
Runs live webcam object detection and ByteTrack tracking (`persist=True`)
using `weights/best.pt`. Press 'r' to save a 5.0-second clip to
`outputs/demo_5sec_tracking.mp4`, or press 'q' to quit.
"""

from pathlib import Path
import time
import cv2
import numpy as np
from ultralytics import YOLO


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    weights_path = repo_root / "weights" / "best.pt"
    output_path = repo_root / "outputs" / "demo_5sec_tracking.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}. Run scripts/01_train_and_eval.py first.")

    model = YOLO(str(weights_path))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (camera index 0).")

    duration_sec = 5.0
    recording = False
    start_ts = 0.0
    frames: list[np.ndarray] = []

    print("Press 'r' to record a 5.0s clip | Press 'q' to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.track(
                source=frame,
                persist=True,
                conf=0.35,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            annotated: np.ndarray = results[0].plot()

            if recording:
                elapsed = time.perf_counter() - start_ts
                frames.append(annotated.copy())
                cv2.putText(
                    annotated,
                    f"REC {elapsed:.1f}s / {duration_sec:.1f}s",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

                if elapsed >= duration_sec:
                    recording = False
                    fps = len(frames) / max(elapsed, 0.1)
                    h, w = frames[0].shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                    for f in frames:
                        writer.write(f)
                    writer.release()
                    print(f"Saved {elapsed:.2f}s video ({len(frames)} frames @ {fps:.1f} FPS) -> {output_path}")
                    frames.clear()

            cv2.imshow("YOLOv8 Tracking (bicycle, car, person)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("r") and not recording:
                recording = True
                frames.clear()
                start_ts = time.perf_counter()
                print("Recording 5.0s clip...")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()