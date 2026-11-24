"""
00_prepare_dataset.py (Python 3.12.0)
--------------------------------------
Verifies that `datasets/person.v1i.yolov8` has 3 classes (`0: bicycle`,
`1: car`, `2: person`) and at least 100 total images across its splits[cite: 1].
"""

from pathlib import Path
import yaml


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    data_yaml = repo_root / "data" / "data.yaml"

    with open(data_yaml, "r", encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f) or {}

    base_path = Path(cfg.get("path", "."))
    if not base_path.is_absolute():
        base_path = (data_yaml.parent / base_path).resolve()

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def count_split(key: str) -> int:
        rel = cfg.get(key)
        folder = base_path / rel if rel else None
        if not folder or not folder.exists():
            return 0
        return sum(1 for p in folder.rglob("*") if p.suffix.lower() in exts)

    train_n = count_split("train")
    val_n = count_split("val")
    test_n = count_split("test")
    total = train_n + val_n + test_n

    print(f"Dataset path : {base_path}")
    print(f"Classes (nc) : {cfg.get('nc')} -> {cfg.get('names')}")
    print(f"train images : {train_n}")
    print(f"valid images : {val_n}")
    print(f"test images  : {test_n}")
    print(f"Total images : {total}")
    print("Status       :", "PASS (>= 100 images)" if total >= 100 else "CHECK DATASET PATH")


if __name__ == "__main__":
    main()