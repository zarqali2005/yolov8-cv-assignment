"""
01_train_and_eval.py (Python 3.12.0)
-------------------------------------
Trains YOLOv8n on `person.v1i.yolov8` (`0: bicycle`, `1: car`, `2: person`),
copies the best checkpoint to `weights/best.pt`, and evaluates validation metrics.
"""

from pathlib import Path
import shutil
from ultralytics import YOLO


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_yaml = repo_root / "data" / "data.yaml"
    weights_dir = repo_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")

    train_results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        workers=0,
        project=str(repo_root / "runs" / "detect"),
        name="person_car_bicycle_yolov8n",
        exist_ok=True,
    )

    best_src = Path(train_results.save_dir) / "weights" / "best.pt"
    best_dst = weights_dir / "best.pt"
    shutil.copy2(best_src, best_dst)
    print(f"\nSaved best weights to: {best_dst}")

    best_model = YOLO(str(best_dst))
    metrics = best_model.val(data=str(data_yaml), imgsz=640, split="val", workers=0)

    print("\n=== Validation Metrics ===")
    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP@50-95 : {metrics.box.map:.4f}")

    for idx, cname in best_model.names.items():
        print(f"  Class {idx} ({cname}) mAP@50: {metrics.box.ap50[idx]:.4f}")


if __name__ == "__main__":
    main()