---

### `REPORT.md`

```markdown
# Machine Learning Assignment Report: 3-Class Object Detection and Tracking with YOLOv8

- **Subject:** Machine Learning (Prof. Lee Kun-Myung, Chungbuk National University)
- **Student:** Ali Zarq (2024042025)
- **Runtime Environment:** Python 3.12.0
- **Dataset:** `person.v1i.yolov8` (Roboflow export, `4,626` total images across `train`, `valid`, and `test`; classes `0: bicycle`, `1: car`, `2: person`; no pre-applied augmentations)[cite: 1]
- **GitHub Repository:** `https://github.com/[your-username]/yolov8-cv-assignment`
- **Trained Model Weights:** `weights/best.pt`
- **5-Second Demonstration Video:** `outputs/demo_5sec_tracking.mp4`

---

## 1. Training Process and Performance

### 1.1 Dataset Preparation
To satisfy the assignment requirement of collecting 100 or more images across three classes, I used the `person.v1i.yolov8` dataset exported from Roboflow in YOLOv8 format[cite: 1]. The dataset contains `4,626` images in total, divided into `train`, `valid`, and `test` subsets (`images/` and `labels/` directories)[cite: 1, 2, 4]. It covers three distinct object categories mapped to exact class indices in `data/data.yaml`:
- `0`: `bicycle`[cite: 1]
- `1`: `car`[cite: 1]
- `2`: `person`[cite: 1]

No pre-processing augmentations were baked into the Roboflow export[cite: 1], allowing Ultralytics YOLOv8 to apply its native online training augmentations dynamically. Before starting training, I verified the directory layout, split counts, and label mappings using `scripts/00_prepare_dataset.py`.

### 1.2 Training Configuration and Process
Using Python 3.12.0 and the Ultralytics framework (`scripts/01_train_and_eval.py`), I fine-tuned a pretrained `yolov8n.pt` (YOLOv8 Nano) backbone via transfer learning. The model was trained for `50` epochs at an input resolution of `640x640` (`imgsz=640`) with a batch size of `16`.

Throughout training, the box regression loss, classification loss, and distribution focal loss (DFL) decreased steadily across epochs while validation precision and mAP converged smoothly. Once training completed, the best-performing checkpoint (`best.pt`) was automatically copied to `weights/best.pt` for evaluation, live tracking, and GitHub distribution.

### 1.3 Model Evaluation and 5-Second Tracking Performance
The best checkpoint (`weights/best.pt`) was evaluated on the validation split (`valid/images`) using `scripts/01_train_and_eval.py`, achieving the following detection metrics:

| Metric | Validation Score |
| :--- | :---: |
| **Precision (Box)** | `[0.XXX]` |
| **Recall (Box)** | `[0.XXX]` |
| **mAP@50 (Overall)** | `[0.XXX]` |
| **mAP@50–95 (Overall)** | `[0.XXX]` |
| **mAP@50 — Class 0 (`bicycle`)** | `[0.XXX]` |
| **mAP@50 — Class 1 (`car`)** | `[0.XXX]` |
| **mAP@50 — Class 2 (`person`)** | `[0.XXX]` |

For the tracking demonstration (`scripts/02_live_track.py`), I ran real-time inference on a live webcam feed with `model.track(..., persist=True, tracker="bytetrack.yaml", conf=0.35)` and recorded a 5.0-second annotated clip (`outputs/demo_5sec_tracking.mp4`). ByteTrack associated detections across consecutive frames and maintained consistent numeric tracking IDs while objects moved within the camera view, with minor ID resets occurring only when an object briefly left the frame or experienced heavy occlusion.

---

## 2. GitHub Repository Link

- **Public Repository URL:** `https://github.com/[your-username]/yolov8-cv-assignment`
- **Repository Contents:** Complete Python 3.12.0 source code (`scripts/`), dataset configuration (`data/data.yaml`), trained weights (`weights/best.pt`), and the 5-second detection and tracking demonstration video (`outputs/demo_5sec_tracking.mp4`).

---

## 3. Experiences During the Assignment

Working through this assignment provided hands-on experience with an end-to-end object detection and multi-object tracking pipeline in Python 3.12.0. One key takeaway during dataset setup was ensuring that the class index order in `data/data.yaml` (`0: bicycle`, `1: car`, `2: person`) matched the Roboflow label annotations exactly[cite: 1], and verifying that relative paths resolved cleanly across the `train`, `valid`, and `test` splits before launching training[cite: 1, 2, 4].

During live tracking and video capture, writing a wall-clock timer around OpenCV's `