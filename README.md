# YOLOv8 3-Class Object Detection & Tracking (Python 3.12.0)

Assignment for **Computer Vision – IV** (Prof. Lee Kun-Myung, Chungbuk National University) using the `person.v1i.yolov8` dataset (`4,626` images across `bicycle`, `car`, and `person`)[cite: 1].

## Quick Start
```bash
pip install -r requirements.txt
python scripts/00_prepare_dataset.py
python scripts/01_train_and_eval.py
python scripts/02_live_track.py