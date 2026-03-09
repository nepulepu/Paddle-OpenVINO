# Paddle-OpenVINO

Implementation of **PaddleOCR using the OpenVINO runtime**, allowing OCR inference **without requiring PaddlePaddle**.

This project recreates the PaddleOCR detection and recognition pipeline entirely in Python while using **OpenVINO IR models for inference**, making it lightweight and easier to deploy in environments where PaddlePaddle is not available.

Based on **PaddleOCR v2.7.3**  
https://github.com/PaddlePaddle/PaddleOCR/tree/v2.7.3

---

# Features

- No PaddlePaddle dependency
- Uses **OpenVINO IR models** for fast inference
- Complete OCR pipeline implementation
  - Text detection (DB detector)
  - Text cropping
  - Text recognition (SVTR-LCNet)
- CPU inference via OpenVINO
- Batch recognition for better throughput
- API similar to PaddleOCR

---

# Requirements

Install dependencies:

```bash
pip install openvino opencv-python numpy pyclipper shapely
```

Python version:

```
Python >= 3.8
```

---

# Model Setup

The pipeline expects **OpenVINO IR models** (`.xml` and `.bin`).

Example directory structure:

```
models/
 ├─ det/
 │   ├─ ppocrv3-en-det.xml
 │   └─ ppocrv3-en-det.bin
 │
 ├─ rec/
 │   ├─ ppocrv4-en-rec.xml
 │   └─ ppocrv4-en-rec.bin
 │
 └─ en_dict.txt
```

---

# Converting Models to OpenVINO

If you only have **ONNX models**, convert them using OpenVINO tools.

Using **Model Optimizer (legacy)**:

```bash
mo --input_model model.onnx --output_dir models/det/
```

Using **OpenVINO Converter (recommended)**:

```bash
ovc model.onnx --output_model models/det/ppocrv3-en-det
```

---

# Usage

Example:

```python
import cv2
from paddleOpenVino import OpenVinoOCR

ocr = OpenVinoOCR(
    det_model_xml="models/det/ppocrv3-en-det.xml",
    rec_model_xml="models/rec/ppocrv4-en-rec.xml",
    rec_char_dict_path="models/en_dict.txt",
    device="CPU"
)

img = cv2.imread("example.jpg")

# Return plain text
text = ocr.run(img)
print(text)

# Return structured OCR results
results = ocr(img)

for box, (text, conf) in results:
    print(f"[{conf:.2f}] {text}")
```

---

# Output Format

The main OCR call returns:

```python
[
  [box, (text, confidence)],
  ...
]
```

Example:

```
[
 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ("TOTAL", 0.98),
 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], ("12.50", 0.96)
]
```

Where:

| Field | Description |
|------|-------------|
| box | 4-point bounding box |
| text | recognized text |
| confidence | mean recognition probability |

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

# Pipeline Overview

The OCR pipeline follows these steps:

1. **Text Detection**
   - DB detection model identifies text regions

2. **Text Cropping**
   - Perspective transform normalizes each detected text region

3. **Text Recognition**
   - Recognition model processes each cropped region

4. **CTC Decoding**
   - Converts model outputs into readable text

---

# Example Output

```
TOTAL 12.50
DATE 2024-03-18
STORE ABC
```

---

# Performance Notes

OpenVINO enables optimized CPU inference.

Important parameters:

```python
device="CPU"
num_threads=4
rec_batch_size=6
```

Batching recognition significantly improves throughput when multiple text regions are detected.

---

# Limitations

- Angle classifier from PaddleOCR is **not implemented**
- Only **OpenVINO runtime** supported
- Requires **pre-converted OpenVINO IR models**

---

# Credits

This project reimplements the PaddleOCR pipeline using OpenVINO.

Original project:

**PaddleOCR**  
https://github.com/PaddlePaddle/PaddleOCR