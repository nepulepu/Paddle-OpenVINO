# 🧠 paddleOpenVino.py — Reimplementation Reference

> This document traces what each part of `paddleOpenVino.py` is reimplementing from the original PaddleOCR codebase, where that original logic lives, and what coverage each flow stage has.

---

## 📊 Overall Flow Coverage

| # | Flow Stage | Coverage | Status |
|---|---|---|---|
| 1 | Top-Level Pipeline Orchestration | 70% | 🟡 Partial |
| 2 | Detection Pre-processing | 85% | 🟡 Partial |
| 3 | Detection Post-processing (DBPostProcess) | 90% | 🟢 Mostly Complete |
| 4 | Box Filtering & Sorting | 100% | 🟢 Complete |
| 5 | Crop Helper | 100% | 🟢 Complete |
| 6 | Recognition Pre-processing | 80% | 🟡 Partial |
| 7 | Recognition Batching | 100% | 🟢 Complete |
| 8 | CTC Decoding | 100% | 🟢 Complete |
| 9 | Character Dictionary Builder | 95% | 🟢 Mostly Complete |
| 10 | Angle Classification | 0% | 🔴 Not Implemented |
| 11 | Inference Backend | 100%* | 🔵 Replaced (OpenVINO) |

> **Weighted Average across inference-critical flow: ~74%**
> The missing 26% is primarily the angle classifier, `box_score_slow`, small-image padding, and multi-algorithm support.

---

## 📦 Overall Intent

`paddleOpenVino.py` reimplements the **end-to-end PaddleOCR inference pipeline** — detection → crop → recognition — as a single self-contained class (`OpenVinoOCR`), replacing the PaddlePaddle runtime with OpenVINO. It draws from logic spread across multiple files and classes in the original codebase.

---

## 🔷 1. Top-Level Pipeline Orchestration

> **Coverage: 70%** 🟡

**Reimplements:** `PaddleOCR.__call__()` in `paddleocr.py` and the wiring between `TextDetector`, `TextRecognizer` inside `tools/infer/predict_system.py`

| `.py` method | Original source |
|---|---|
| `OpenVinoOCR.__call__(img)` | `PaddleOCR.__call__()` → calls `TextDetector` → crops → calls `TextRecognizer` |
| `OpenVinoOCR.run(img)` | `run_paddle()` utility helper pattern used in inference scripts |

The original separates these responsibilities across `TextDetector`, `TextClassifier`, and `TextRecognizer` as independent objects. Here they are collapsed into one class.

### ❌ Not Implemented from this stage

| Missing | Original source |
|---|---|
| `TextClassifier` angle correction step between detection and recognition | `tools/infer/predict_cls.py` → `TextClassifier.__call__()` |
| Benchmark timing wrappers (`autolog`) around each stage | `tools/infer/predict_system.py` — `autolog.times.start/stamp/end` |
| `save_crop_res` / `draw_ocr_box_txt` visualisation utilities | `tools/infer/predict_system.py` |

---

## 🔷 2. Detection Pre-processing

> **Coverage: 85%** 🟡

### 2a. Image Resize

**Reimplements:** `DetResizeForTest.resize_image_type0()` in `ppocr/data/imaug/operators.py`

`OpenVinoOCR._det_resize()` ports the `limit_type = "max"` and `limit_type = "min"` branching logic, the ratio calculation, and the **snap-to-multiple-of-32** formula:

```
max(int(round(h * ratio / 32) * 32), 32)
```

It also returns `shape_list` as `[orig_h, orig_w, ratio_h, ratio_w]`, matching what `DetResizeForTest.__call__()` stores in `data['shape']`.

### 2b. Normalize Image

**Reimplements:** `NormalizeImage.__call__()` in `ppocr/data/imaug/operators.py`

`OpenVinoOCR._det_normalize()` uses the same ImageNet mean/std values and the same HWC → CHW transpose:

- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`

The original `NormalizeImage` receives these via a config dict; here they are hardcoded inline.

### ❌ Not Implemented from this stage

| Missing | Original source |
|---|---|
| Small image zero-padding when `h + w < 64` | `DetResizeForTest.image_padding()` in `ppocr/data/imaug/operators.py` |
| `resize_image_type1` (fixed shape resize with `keep_ratio`) | `DetResizeForTest.resize_image_type1()` in `ppocr/data/imaug/operators.py` |
| `resize_image_type2` (resize_long / stride-128 snapping) | `DetResizeForTest.resize_image_type2()` in `ppocr/data/imaug/operators.py` |
| `DB++` alternate ImageNet mean (different values) | `TextDetector.__init__()` pre-process override in `tools/infer/predict_det.py` |

---

## 🔷 3. Detection Post-processing (DBPostProcess)

> **Coverage: 90%** 🟢

**Reimplements:** `DBPostProcess` class in `ppocr/postprocess/db_postprocess.py`

This is the most directly ported section. The following methods map 1-to-1:

| `.py` method | Original `DBPostProcess` method |
|---|---|
| `_db_postprocess()` | `DBPostProcess.__call__()` |
| `_boxes_from_bitmap()` | `DBPostProcess.boxes_from_bitmap()` |
| `_polygons_from_bitmap()` | `DBPostProcess.polygons_from_bitmap()` |
| `_unclip()` | `DBPostProcess.unclip()` |
| `_get_mini_boxes()` | `DBPostProcess.get_mini_boxes()` |
| `_box_score_fast()` | `DBPostProcess.box_score_fast()` |

All of the following are reproduced from the original `DBPostProcess`:

- Sigmoid threshold binarisation: `pred > det_thresh`
- Dilation kernel `[[1,1],[1,1]]` when `use_dilation=True`
- `max_candidates = 1000` contour cap
- Shapely `poly.area * ratio / poly.length` distance in `_unclip`
- `cv2.minAreaRect` + `cv2.boxPoints` corner ordering in `_get_mini_boxes`
- `cv2.fillPoly` + `cv2.mean` mask scoring in `_box_score_fast`

### ❌ Not Implemented from this stage

| Missing | Original source |
|---|---|
| `box_score_slow` (contour polygon mean score) | `DBPostProcess.box_score_slow()` in `ppocr/postprocess/db_postprocess.py` — parameter `det_score_mode` is accepted but only `fast` is ever called |
| `len(outs) == 3` branch in `findContours` (OpenCV 3 compat) | `DBPostProcess.boxes_from_bitmap()` — the `.py` file only handles the OpenCV 4 `len == 2` case |

---

## 🔷 4. Box Filtering & Sorting

> **Coverage: 100%** 🟢

### 4a. Filter Boxes

**Reimplements:** `TextDetector.filter_tag_det_res()` and `TextDetector.order_points_clockwise()` in `tools/infer/predict_det.py`

`OpenVinoOCR._filter_boxes()` and `_order_points_clockwise()` replicate:

- The sum/diff method for finding TL, TR, BR, BL corners
- `np.clip` to image bounds
- `np.linalg.norm` check dropping boxes ≤ 3px in either dimension

### 4b. Sort Boxes

**Reimplements:** `sorted_boxes()` utility function in `tools/infer/predict_system.py`

`OpenVinoOCR._sort_boxes()` ports the two-pass sort: primary sort by `(y, x)` of first corner, then a bubble-swap pass reordering boxes within **10px Y proximity** by their X coordinate.

---

## 🔷 5. Crop Helper (Between Detection and Recognition)

> **Coverage: 100%** 🟢

**Reimplements:** `get_rotate_crop_image()` in `ppocr/utils/utility.py`

`OpenVinoOCR._crop_text_region()` ports:

- `np.linalg.norm` for computing `crop_w` and `crop_h` from the 4 corner points
- `cv2.getPerspectiveTransform` + `cv2.warpPerspective` with `BORDER_REPLICATE` and `INTER_CUBIC`
- The `rot90` rotation when `h / w >= 1.5` (vertical strip detection)

---

## 🔷 6. Recognition Pre-processing

> **Coverage: 80%** 🟡

**Reimplements:** `TextRecognizer.resize_norm_img()` — the standard SVTR_LCNet path — in `tools/infer/predict_rec.py`

`OpenVinoOCR._rec_resize_norm()` ports:

- Aspect-ratio-preserving resize to fixed height, width capped at `rec_img_w`
- Normalisation: `/ 255 → − 0.5 → / 0.5` (values in `[−1, 1]`)
- Zero-padding on the right side to fill a `[3, H, W]` canvas

### ❌ Not Implemented from this stage

| Missing | Original source |
|---|---|
| Grayscale conversion + LANCZOS resize path for NRTR / ViTSTR | `TextRecognizer.resize_norm_img()` NRTR/ViTSTR branch in `tools/infer/predict_rec.py` |
| `resize_norm_img_srn()` for SRN model | `TextRecognizer.resize_norm_img_srn()` in `tools/infer/predict_rec.py` |
| `resize_norm_img_vl()` for VisionLAN model | `TextRecognizer.resize_norm_img_vl()` in `tools/infer/predict_rec.py` |
| CAN model inverse image option | `TextRecognizer.__call__()` `self.inverse` handling in `tools/infer/predict_rec.py` |

---

## 🔷 7. Recognition Batching

> **Coverage: 100%** 🟢

**Reimplements:** `TextRecognizer.__call__()` batch loop in `tools/infer/predict_rec.py`

`OpenVinoOCR._recognize()` ports:

- Sort crops by aspect ratio before batching (for efficiency)
- Chunk into batches of `rec_batch_size`
- Compute effective width as `min(rec_img_w, ceil(rec_img_h × max_wh_ratio))`
- Restore results to the original crop order after batching

---

## 🔷 8. CTC Decoding

> **Coverage: 100%** 🟢

**Reimplements:** `CTCLabelDecode.__call__()` and `BaseRecLabelDecode.decode()` in `ppocr/postprocess/rec_postprocess.py`

`OpenVinoOCR._ctc_decode()` ports:

- Greedy `argmax` over the time axis: `preds.argmax(axis=2)`
- Skip blank token (index `0`) and consecutive duplicates
- Return `(text, mean_confidence)` per sample, or `(text, 0.0)` if no characters decoded

---

## 🔷 9. Character Dictionary Builder

> **Coverage: 95%** 🟢

**Reimplements:** `BaseRecLabelDecode.__init__()` in `ppocr/postprocess/rec_postprocess.py`

`OpenVinoOCR._build_character_list()` ports:

- Default vocab `"0123456789abcdefghijklmnopqrstuvwxyz"` when no dict path is provided
- UTF-8 line-by-line reading of a char dict file
- Appending `" "` when `use_space_char=True`
- Prepending `"blank"` at index `0` (mirroring `CTCLabelDecode.add_special_char()`)

### ❌ Not Implemented from this stage

| Missing | Original source |
|---|---|
| Arabic RTL text reversal (`self.reverse`) | `BaseRecLabelDecode.__init__()` — checks if `'arabic' in character_dict_path` and sets `self.reverse = True` |

---

## 🔷 10. Angle Classification

> **Coverage: 0%** 🔴

**Not implemented.** The original has a full `TextClassifier` stage that runs between detection and recognition to detect and correct 180°-rotated text crops.

### ❌ Not Implemented from this stage

| Missing | Original source |
|---|---|
| Full `TextClassifier` class | `tools/infer/predict_cls.py` → `TextClassifier.__init__()`, `__call__()` |
| Angle threshold + flip logic for crops | `TextClassifier.__call__()` — flips crops where predicted angle label is `180` and confidence ≥ `cls_thresh` |
| `use_angle_cls` gate in the main pipeline | `predict_system.py` — conditional call to `TextClassifier` between `TextDetector` and `TextRecognizer` |

> `self.args.use_angle_cls` is set to `False` in the `.py` file, acknowledging the absence.

---

## 📁 Original Source File Map

| Original file | What is reimplemented from it | Coverage |
|---|---|---|
| `ppocr/data/imaug/operators.py` | `DetResizeForTest` (type0 only), `NormalizeImage` | 85% |
| `ppocr/postprocess/db_postprocess.py` | `DBPostProcess` — all methods except `box_score_slow` | 90% |
| `ppocr/postprocess/rec_postprocess.py` | `BaseRecLabelDecode`, `CTCLabelDecode` (CTC path only) | 95% |
| `ppocr/utils/utility.py` | `get_rotate_crop_image` | 100% |
| `tools/infer/predict_det.py` | `TextDetector` inference path, `filter_tag_det_res`, `order_points_clockwise` | 85% |
| `tools/infer/predict_rec.py` | `TextRecognizer` — SVTR path only, batch loop | 80% |
| `tools/infer/predict_cls.py` | `TextClassifier` | 0% |
| `tools/infer/predict_system.py` | `sorted_boxes`, system orchestration | 70% |
| `paddleocr.py` | `PaddleOCR.__call__()` top-level flow | 70% |