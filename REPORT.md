# YOLOv8 Object Detection and Tracking Report

* **Author:** Zarq Ali
* **Student ID:** `2024042025`
* **Repository:** `https://github.com/zarqali2005/yolov8-cv-assignment/tree/master`

## 1. Overview & Objective
This report documents training a custom **YOLOv8n** object detector on `person.v1i.yolov8` across three classes (`0: bicycle`, `1: car`, `2: person`) and running live object detection and tracking with `scripts/02_live_track.py` to record a 5-second demonstration clip (`outputs/demo_5sec_tracking.mp4`).

---

## 2. Dataset & Environment Setup
* **Dataset:** `person.v1i.yolov8` (`4,626` total images across splits; validation split: `922` images, `6,605` instances)
* **Classes:** `0: bicycle`, `1: car`, `2: person`
* **Hardware:** HP Victus laptop with **NVIDIA GeForce RTX 4060 Laptop GPU** (`8,188 MiB` VRAM)
* **Software Stack:** Windows, Python `3.12.0rc2` (`.venv`), PyTorch `2.11.0+cu128`, Ultralytics `8.4.218` (`yolov8n.pt`, fused summary: `73` layers, `3,133,193` parameters, `8.6` GFLOPs)

---

## 3. Training & Execution Notes
* **GPU Acceleration:** Trained on the RTX 4060 GPU with CUDA (`imgsz=640`, `batch=16`).
* **Windows Multiprocessing Fix (`WinError 1455`):** Set `workers=0` in `scripts/01_train_and_eval.py` to avoid `OSError: [WinError 1455] The paging file is too small for this operation to complete` during training/validation on Windows.
* **Training Progress & Early Stop:** Out of `50` planned epochs, training progressed through epoch `29` in `runs/detect/person_car_bicycle_yolov8n` before being stopped early due to time constraints. The best checkpoint (`best.pt`) was copied to `weights/best.pt` and evaluated (`runs/detect/val-3`).
* **Live Tracking:** Ran `scripts/02_live_track.py` (installed `lap-0.5.14`), recorded `outputs/demo_5sec_tracking.mp4` by pressing `r`, and exited with `q`.

---

## 4. Validation Results (`weights/best.pt`)
Evaluated on the `valid` split (`922` images, `6,605` instances, `imgsz=640`, `workers=0`) with speed `0.5ms` preprocess, `6.8ms` inference, `0.0ms` loss, and `1.7ms` postprocess per image:

| Class | Images | Instances | Box(P) | Recall (R) | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **all** | `922` | `6,605` | `0.807` (`0.8070`) | `0.564` (`0.5636`) | `0.668` (`0.6682`) | `0.437` (`0.4367`) |
| **0: bicycle** | `378` | `1,170` | `0.719` | `0.538` | `0.628` (`0.6284`) | `0.398` |
| **1: car** | `308` | `881` | `0.817` | `0.562` | `0.670` (`0.6698`) | `0.456` |
| **2: person** | `878` | `4,554` | `0.885` | `0.591` | `0.706` (`0.7062`) | `0.457` |

### Key Observations
* **Precision vs. Recall:** Overall box precision reached `0.8070` (`0.807` in the table), while recall was `0.5636` (`0.564` in the table), indicating detections were fairly reliable when made, with room for higher recall given more epochs or tuning.
* **Class Performance:** `person` had the highest `mAP@50` (`0.7062` / `0.706`) and precision (`0.885`), followed by `car` (`0.6698` / `0.670` `mAP@50`) and `bicycle` (`0.6284` / `0.628` `mAP@50`).

---

## 5. Deliverables
* `scripts/01_train_and_eval.py`
* `scripts/02_live_track.py`
* `weights/best.pt`
* `outputs/demo_5sec_tracking.mp4`
* `REPORT.md`
* GitHub Repository (`master` branch): `https://github.com/zarqali2005/yolov8-cv-assignment/tree/master`
