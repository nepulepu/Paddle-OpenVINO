# Paddle-OpenVINO

**paddleOpenVINO** is a lightweight **OpenVINO-based replacement for PaddleOCR inference**.

It reproduces PaddleOCR’s preprocessing, model execution, and post-processing pipeline **without requiring PaddlePaddle**.

The implementation closely mirrors PaddleOCR behavior for:

- **Text Detection** (DB / DB++)
- **Text Classification** (0° / 180° orientation)
- **Text Recognition** (SVTR_LCNet with CTC decoding)

All inference runs through **OpenVINO**, enabling deployment in environments where PaddlePaddle is unavailable.

---

# Features

- No PaddlePaddle dependency
- Uses **OpenVINO runtime** for inference
- Supports **OpenVINO IR (.xml + .bin)** and **ONNX models**
- Implements the **full PaddleOCR pipeline**
- Detection algorithms:
  - DB
  - DB++
- Optional **text orientation classifier**
- **CTC decoding identical to PaddleOCR**
- Dynamic batching for recognition
- Drop-in style API similar to PaddleOCR

---

# Requirements

Install dependencies:

```bash
pip install numpy opencv-python openvino shapely pyclipper
```

Python version:

```
Python >= 3.8
```

---

# Supported Models

The pipeline expects PaddleOCR models converted to:

- **OpenVINO IR (.xml + .bin)**  
or
- **ONNX (.onnx)**

Example structure:

```
models/
├── paddle_ov_det/
│   ├── ppocrv3-en-det.xml
│   └── ppocrv3-en-det.bin
│
├── paddle_ov_rec/
│   ├── ppocrv4-en-rec.xml
│   └── ppocrv4-en-rec.bin
│
├── paddle_ov_cls/
│   ├── ch_ppocr_mobile_v2.0_cls.xml
│   └── ch_ppocr_mobile_v2.0_cls.bin
│
└── en_dict.txt
```

Classifier model is **optional**.

---

# Converting Models

If your models are in **ONNX format**, you can convert them to OpenVINO IR.

Using OpenVINO Converter:

```bash
ovc model.onnx --output_model model.xml
```

Or run directly using ONNX:

```python
det_model_path="det_model.onnx"
```

---

# Quick Start

Example usage:

```python
import cv2
from paddleOpenVino import paddleOpenVINO

ocr = paddleOpenVINO(
    det_model_path="models/paddle_ov_det/ppocrv3-en-det.xml",
    rec_model_path="models/paddle_ov_rec/ppocrv4-en-rec.xml"
)

image = cv2.imread("example.jpg")

results = ocr.ocr(image)

for box, (text, score) in results[0]:
    print(text, score)
```

---

# API

## Initialize

```python
paddleOpenVINO(
    det_model_path,
    rec_model_path,
    cls_model_path=None
)
```

### Parameters

| Parameter | Description |
|---|---|
| det_model_path | detection model (.xml or .onnx) |
| rec_model_path | recognition model |
| cls_model_path | optional orientation classifier |
| det_algorithm | DB or DB++ |
| det_limit_side_len | resize limit before detection |
| det_db_thresh | pixel threshold |
| det_db_box_thresh | box confidence threshold |
| det_db_unclip_ratio | bounding box expansion |
| rec_batch_num | recognition batch size |
| drop_score | filter low-confidence results |

---

# Running OCR

```python
results = model.ocr(image)
```

### Arguments

| Argument | Description |
|---|---|
| img | image path or numpy array |
| det | enable text detection |
| rec | enable recognition |
| cls | enable orientation classifier |

---

# Output Format

The OCR result format matches PaddleOCR:

```
[
  [
    [box, (text, confidence)],
    ...
  ]
]
```

Example:

```
[
  [
    [[10,20],[100,20],[100,50],[10,50]], ("TOTAL", 0.98),
    [[10,60],[120,60],[120,90],[10,90]], ("12.50", 0.96)
  ]
]
```

Where:

| Field | Description |
|---|---|
| box | 4-point bounding box |
| text | recognized text |
| confidence | recognition confidence |

---

# OCR Pipeline

The processing pipeline mirrors PaddleOCR:

```
Input Image
    ↓
Resize + Normalize
    ↓
Text Detection (DB / DB++)
    ↓
Filter + Sort boxes
    ↓
Perspective Crop
    ↓
Direction Classification (optional)
    ↓
Text Recognition (SVTR_LCNet)
    ↓
CTC Decode
    ↓
Final OCR Results
```

---

# Example Output

```
TOTAL 12.50
DATE 2024-03-18
STORE ABC
```

---

# Project Structure

```
Paddle-OpenVINO/
│
├── README.md
├── paddleOpenVino.py
└── paddleOpenVino.md
```

---

# Notes

- The implementation mirrors PaddleOCR preprocessing and postprocessing logic.
- Recognition uses **dynamic width resizing** for efficient batching.
- The classifier rotates text crops if the predicted orientation is **180°**.

---

# Limitations

- Only **CPU inference via OpenVINO** is currently implemented
- Polygon box output is not supported (quad only)
- Benchmarking and GPU execution are not included

---

# Credits

This project reimplements PaddleOCR inference using OpenVINO.

Original project:

**PaddleOCR**  
https://github.com/PaddlePaddle/PaddleOCR