# Object Detection for Dishes using YOLOv8

A simple and powerful Python tool for training and running object detection on dishware using the YOLOv8 model. This project provides a streamlined pipeline to train custom YOLO models and run real-time inference on video streams.

## Features

* **YOLOv8 Training**: Custom training with configurable hyperparameters
* **Real-Time Inference**: Run object detection on video files using pretrained models
* **Simple CLI Interface**: Use command-line arguments via [`fire`](https://github.com/google/python-fire)
* **Logging & Output Management**: All logs and results are stored neatly for further analysis
* **Live Visualization**: View results live with bounding boxes drawn on frames
* **Pythonic & Maintainable Codebase with Poetry**

---

## Installation

### ! Make sure you have Python 3.11 and Poetry installed

1. **Clone the repository**

```bash
git clone https://github.com/IvanArsenev/cv_test_task
cd cv_test_task
```

2. **Install dependencies**

```bash
poetry install
```

3. **Install torch with cuda**

```bash
poetry run pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

P.s. if some errors: install PyTorch manually with correct CUDA version
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

---

## Usage

### Run Inference on a Video

```bash
python dishes_detector.py demo --model_path ./models/best.pt --video_path ./sample_video.mp4
```

### Train a Model

```bash
python dishes_detector.py train --data_path ./datasets/data.yaml --model_path ./model/yolov8n.pt --results_path ./results_old --epochs 200 --batch_size 16 --img_size 1024
```

### Further training

```bash
 python dishes_detector.py fine_tune --data_path ./datasets/data.yaml --pretrained_model_path ./results_old/yolo_training/weights/best.pt --results_path ./fine_tune --epochs 10 --batch_size 16 --img_size 1024
```

### Metrics

```bash
 python dishes_detector.py metrics --model_path ./results/yolo_training/weights/best.pt --data_path ./dataset/data.yaml
```

---

## Project Structure

```
cv_test_task/
│
├── dishes_detector.py       # Main CLI app with train & demo logic
├── pyproject.toml           # Poetry config with dependencies
├── ./datasets/data.yaml     # Dataset config for YOLO
├── models/                  # Pretrained or trained model weights
├── videos/                  # Video files for inference
├── results/                 # Output from training
└── README.md
```
---

## Parameters

### Training

| Argument        | Description                              | Default      |
|-----------------|------------------------------------------|--------------|
| `data_path`     | Path to `data.yaml` config file          | —            |
| `model_path`    | Base YOLO model path (e.g. `yolov9e.pt`) | —            |
| `results_path`  | Directory to save training results       | `./results`  |
| `epochs`        | Number of training epochs                | `50`         |
| `batch_size`    | Number of samples per batch              | `16`         |
| `img_size`      | Image size for training                  | `640`        |

### Demo

| Argument      | Description                         | Default  |
|---------------|-------------------------------------|----------|
| `model_path`  | Path to trained YOLO model (`.pt`)  | —        |
| `video_path`  | Input video file path               | —        |

### Metrics

| Argument     | Description                          | Default  |
|--------------|--------------------------------------|----------|
| `model_path` | Path to trained YOLO model (`.pt`)   | —        |
| `data_path`  | Path to `data.yaml` config file      | —        |

### Fine tune

| Argument                | Description                              | Default     |
|-------------------------|------------------------------------------|-------------|
| `data_path`             | Path to `data.yaml` config file          | —           |
| `pretrained_model_path` | Base YOLO model path (e.g. `yolov9e.pt`) | —           |
| `results_path`          | Directory to save fine-tuning results    | `./results` |
| `epochs`                | Number of training epochs                | `50`        |
| `batch_size`            | Number of samples per batch              | `16`        |
| `img_size`              | Image size for training                  | `640`       |

---

## Preview

Here’s an example of detection in action:

![Demo Output](./docs/demo_1.gif)

![Demo Output](./docs/demo_2.gif)

![Demo Output](./docs/demo_3.gif)

---

## Dataset

Open dataset to download: `https://app.roboflow.com/etbx/test_task_for_cv/4`

---

## Contributing

Contributions are welcome!
Feel free to open issues, request features, or submit pull requests

---

## License

This project is no licensed

---

## Credits

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* [Python Fire](https://github.com/google/python-fire)
