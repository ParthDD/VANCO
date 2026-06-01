# ASL Detection Project

A real-time American Sign Language (ASL) detection system built with **YOLOv8** and **OpenCV**. This project captures ASL sign images, trains a custom YOLO model, and provides live inference via webcam.

---

## Overview

The ASL Detection project implements a complete machine learning pipeline for detecting and classifying American Sign Language hand gestures:

- **Data Collection**: Capture webcam frames for each ASL sign (A-Z).
- **Data Preparation**: Split raw data into train/validation/test sets.
- **Auto-Labeling**: Generate YOLO-format annotations using model predictions.
- **Model Training**: Fine-tune YOLOv8 on your ASL dataset.
- **Inference & Demo**: Validate the model and run live webcam detection.

---

## Project Structure

```
asl_detection/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── data.yaml                      # YOLO dataset configuration
├── yolov8n.pt                     # Pre-trained YOLOv8 nano weights
├── __init__.py                    # Package initializer
├── scripts/                       # Main workflow scripts
│   ├── collect_data.py            # Webcam data capture
│   ├── data_pipeline.py           # Auto-labeling via YOLO
│   ├── data_split.py              # Train/val/test split
│   ├── train.py                   # Model training
│   ├── inference.py               # Validation & metrics
│   └── demo.py                    # Live webcam demo
├── data/                          # Dataset folder
│   ├── raw/                       # Raw captured images
│   ├── train/                     # Training split
│   ├── val/                       # Validation split
│   └── test/                      # Test split
└── models/                        # Trained model outputs
    └── vanco_asl_model/           # Default model directory
```

---

## Setup

### 1. Prerequisites

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- Webcam (for data collection and demo)

### 2. Install Dependencies

From the repository root, navigate to `asl_detection` and install:

```powershell
cd d:\Parth\vanco-solution-architecture\asl_detection

# Option A: Using global Python (packages installed globally)
python -m pip install -r requirements.txt

# Option B: Using a virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script, run:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Or use the venv Python directly without activating:
```powershell
.\venv\Scripts\python.exe -m asl_detection.scripts.train
```

---

## Quick Start

### 1. Collect Data

Capture webcam frames for all 26 ASL letters (A-Z):

```powershell
python -m asl_detection.scripts.collect_data
```

Press **Space** to capture each frame. Images are saved to `data/raw/Sign_A/`, `data/raw/Sign_B/`, etc.

### 2. Split Data into Train/Val/Test

Organize collected images into train (70%), validation (15%), and test (15%) folders:

```powershell
python -m asl_detection.scripts.data_split
```

Output goes to `data/train/`, `data/val/`, `data/test/`.

### 3. Auto-Label Dataset

Generate YOLO-format labels using the pre-trained YOLO model:

```powershell
python -m asl_detection.scripts.data_pipeline
```

This creates `.txt` files alongside each image in the format:
```
<class_id> <x_center> <y_center> <width> <height>
```

### 4. Train the Model

Fine-tune YOLOv8 on your labeled ASL dataset:

```powershell
python -m asl_detection.scripts.train
```

Training outputs are saved to `models/vanco_asl_model/`. This may take 10-30 minutes depending on your hardware.

**Note**: To use GPU, edit `train.py` and change `device='cpu'` to `device=0`.

### 5. Validate the Model

Evaluate the trained model on the test set:

```powershell
python -m asl_detection.scripts.inference
```

This prints metrics like `mAP@50`.

### 6. Run Live Demo

See real-time ASL detection from your webcam:

```powershell
python -m asl_detection.scripts.demo
```

Press **q** to exit.

---

## Configuration

### `data.yaml`

Defines the dataset structure and class names for YOLO training:

```yaml
path: D:\Parth\vanco-solution-architecture\asl_detection\data
train: train/images
val: val/images
test: test/images

nc: 26  # Number of classes (A-Z)
names: ['A', 'B', 'C', ..., 'Z']
```

Update paths as needed for your environment.

### `requirements.txt`

Core dependencies:
- `ultralytics` — YOLOv8 framework
- `opencv-python` — Image processing
- `torch` & `torchvision` — Deep learning backend
- `numpy`, `matplotlib`, `pyyaml` — Utilities

---

## File Descriptions

**Quick Reference:**

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `collect_data.py` | Webcam capture | Webcam stream | Raw images in `data/raw/` |
| `data_split.py` | Dataset split | Raw images | Organized train/val/test folders |
| `data_pipeline.py` | Auto-labeling | Images (split) | YOLO `.txt` labels |
| `train.py` | Model training | Images + labels | `models/vanco_asl_model/` |
| `inference.py` | Validation | Test images | mAP metrics |
| `demo.py` | Live detection | Webcam stream | Annotated video display |

---

## Troubleshooting

### ImportError when running scripts

**Symptom**: `ModuleNotFoundError: No module named 'asl_detection'`

**Solution**: Always run scripts from the repository root using the module syntax:
```powershell
cd d:\Parth\vanco-solution-architecture
python -m asl_detection.scripts.train
```

Do NOT run `python scripts/train.py` from inside `asl_detection/scripts/`.

### CUDA not available

**Symptom**: "CUDA not available" message, but you have a GPU.

**Solution**: Install the CUDA-enabled PyTorch version:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then update `train.py` to use `device=0` instead of `device='cpu'`.

### Webcam not detected

**Symptom**: "Error: Could not open webcam" in `collect_data.py` or `demo.py`.

**Solution**:
- Ensure your camera is not in use by another application.
- Check device manager that your camera is recognized.
- Try `cv2.VideoCapture(1)` instead of `0` if multiple cameras exist.

### Out of memory errors

**Symptom**: Memory error during training.

**Solution**: Reduce batch size in `train.py`:
```python
results = model.train(
    ...
    batch=8,  # Reduce from 16
    ...
)
```

---

## Next Steps

1. **Expand the dataset**: Capture more frames per sign and from different angles/lighting.
2. **Tune hyperparameters**: Modify `train.py` learning rate, epochs, and augmentation.
3. **Deploy**: Export the model to ONNX or TFLite for production inference.
4. **Multi-sign recognition**: Extend to word-level ASL translation.

---

## Resources

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [PyTorch Docs](https://pytorch.org/docs/)

---

## License

This project is part of the VANCO solution architecture repository.

---


