"""
paddleOpenVINO — drop-in OpenVINO replacement for PaddleOCR inference.

Replicates PaddleOCR's exact pre/post-processing for:
  - Detection   : DB / DB++
  - Classification : direction classifier (0° / 180°)
  - Recognition : SVTR_LCNet (CTC decode)

No PaddlePaddle dependency required.

Dependencies:
  numpy, opencv-python, openvino, shapely, pyclipper
"""

import os
import math
import copy

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Helpers — load an OpenVINO compiled model from either .xml or .onnx
# ---------------------------------------------------------------------------

def _load_ov_model(model_path: str):
    """
    Accepts either:
      - path/to/model.xml   (OpenVINO IR, companion .bin must sit next to it)
      - path/to/model.onnx  (loaded directly by OpenVINO's ONNX frontend)

    Returns (compiled_model, input_layer, output_layers).
    """
    from openvino.runtime import Core
    ie = Core()

    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".xml":
        model = ie.read_model(model=model_path)
    elif ext == ".onnx":
        model = ie.read_model(model=model_path)
    else:
        raise ValueError(
            f"Unsupported model format '{ext}'. Use .xml or .onnx"
        )

    compiled = ie.compile_model(model=model, device_name="CPU")
    input_layer = compiled.input(0)
    output_layers = [compiled.output(i) for i in range(len(compiled.outputs))]
    return compiled, input_layer, output_layers


# ---------------------------------------------------------------------------
# Detection pre-processing  (mirrors DetResizeForTest + NormalizeImage)
# ---------------------------------------------------------------------------

_DET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DBPP_MEAN = np.array(
    [0.48109378172549, 0.45752457890196, 0.40787054090196], dtype=np.float32
)
_DBPP_STD = np.array([1.0, 1.0, 1.0], dtype=np.float32)


def _det_resize_type0(img, limit_side_len, limit_type):
    """Paddle's resize_image_type0: resize to multiple-of-32 dims."""
    h, w = img.shape[:2]

    if limit_type == "max":
        if max(h, w) > limit_side_len:
            ratio = float(limit_side_len) / h if h > w else float(limit_side_len) / w
        else:
            ratio = 1.0
    elif limit_type == "min":
        if min(h, w) < limit_side_len:
            ratio = float(limit_side_len) / h if h < w else float(limit_side_len) / w
        else:
            ratio = 1.0
    elif limit_type == "resize_long":
        ratio = float(limit_side_len) / max(h, w)
    else:
        raise ValueError(f"Unknown limit_type: {limit_type}")

    resize_h = max(int(round(h * ratio / 32) * 32), 32)
    resize_w = max(int(round(w * ratio / 32) * 32), 32)

    img = cv2.resize(img, (resize_w, resize_h))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return img, ratio_h, ratio_w


def _det_preprocess(img, limit_side_len, limit_type, mean, std):
    """
    Returns:
      blob       : float32 ndarray [1, 3, H, W]
      shape_list : ndarray [1, 4]  = [[src_h, src_w, ratio_h, ratio_w]]
    """
    src_h, src_w = img.shape[:2]

    # Pad tiny images
    if src_h + src_w < 64:
        pad_h = max(32, src_h)
        pad_w = max(32, src_w)
        padded = np.zeros((pad_h, pad_w, img.shape[2]), dtype=np.uint8)
        padded[:src_h, :src_w] = img
        img = padded

    img, ratio_h, ratio_w = _det_resize_type0(img, limit_side_len, limit_type)

    # Normalize (HWC)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std

    # HWC → CHW, add batch dim
    blob = img.transpose(2, 0, 1)[np.newaxis, :]
    shape_list = np.array([[src_h, src_w, ratio_h, ratio_w]], dtype=np.float32)
    return blob, shape_list


# ---------------------------------------------------------------------------
# Detection post-processing  (mirrors DBPostProcess)
# ---------------------------------------------------------------------------

def _unclip(box, unclip_ratio):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(
        box.astype(np.int32).tolist(),
        pyclipper.JT_ROUND,
        pyclipper.ET_CLOSEDPOLYGON,
    )
    expanded = np.array(offset.Execute(distance))
    return expanded


def _get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        idx1, idx4 = 0, 1
    else:
        idx1, idx4 = 1, 0
    if points[3][1] > points[2][1]:
        idx2, idx3 = 2, 3
    else:
        idx2, idx3 = 3, 2

    box = [points[idx1], points[idx2], points[idx3], points[idx4]]
    return box, min(bounding_box[1])


def _box_score_fast(bitmap, box):
    h, w = bitmap.shape[:2]
    box = box.copy()
    xmin = int(np.clip(np.floor(box[:, 0].min()), 0, w - 1))
    xmax = int(np.clip(np.ceil(box[:, 0].max()),  0, w - 1))
    ymin = int(np.clip(np.floor(box[:, 1].min()), 0, h - 1))
    ymax = int(np.clip(np.ceil(box[:, 1].max()),  0, h - 1))

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] -= xmin
    box[:, 1] -= ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _boxes_from_bitmap(pred, bitmap, dest_w, dest_h,
                        box_thresh, unclip_ratio, max_candidates, min_size):
    height, width = bitmap.shape

    outs = cv2.findContours(
        (bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = outs[0] if len(outs) == 2 else outs[1]

    boxes, scores = [], []
    for contour in contours[:max_candidates]:
        points, sside = _get_mini_boxes(contour)
        if sside < min_size:
            continue
        points = np.array(points)
        score = _box_score_fast(pred, points.reshape(-1, 2))
        if score < box_thresh:
            continue

        box = _unclip(points, unclip_ratio).reshape(-1, 1, 2)
        box, sside = _get_mini_boxes(box)
        if sside < min_size + 2:
            continue
        box = np.array(box)
        box[:, 0] = np.clip(np.round(box[:, 0] / width  * dest_w), 0, dest_w)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_h), 0, dest_h)
        boxes.append(box.astype(np.int32))
        scores.append(score)

    return np.array(boxes, dtype=np.int32), scores


def _det_postprocess(outputs, shape_list,
                      thresh, box_thresh, unclip_ratio,
                      use_dilation, score_mode, max_candidates=1000, min_size=3):
    """
    Mirrors DBPostProcess.__call__.
    Returns list of dicts [{'points': boxes_array}].
    """
    pred = outputs[0]               # [1, 1, H, W]
    pred = pred[:, 0, :, :]        # [1, H, W]
    segmentation = pred > thresh

    dilation_kernel = np.array([[1, 1], [1, 1]]) if use_dilation else None
    boxes_batch = []

    for i in range(pred.shape[0]):
        src_h, src_w, ratio_h, ratio_w = shape_list[i]
        src_h, src_w = int(src_h), int(src_w)

        mask = segmentation[i].astype(np.uint8)
        if dilation_kernel is not None:
            mask = cv2.dilate(mask, dilation_kernel)

        boxes, scores = _boxes_from_bitmap(
            pred[i], mask, src_w, src_h,
            box_thresh, unclip_ratio, max_candidates, min_size
        )
        boxes_batch.append({"points": boxes})

    return boxes_batch


# ---------------------------------------------------------------------------
# Detection post-processing — filter / order boxes
# ---------------------------------------------------------------------------

def _order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(tmp, axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]
    return rect


def _clip_det_res(points, img_h, img_w):
    for p in range(points.shape[0]):
        points[p, 0] = int(min(max(points[p, 0], 0), img_w - 1))
        points[p, 1] = int(min(max(points[p, 1], 0), img_h - 1))
    return points


def _filter_tag_det_res(dt_boxes, image_shape):
    img_h, img_w = image_shape[:2]
    dt_boxes_new = []
    for box in dt_boxes:
        if isinstance(box, list):
            box = np.array(box)
        box = _order_points_clockwise(box)
        box = _clip_det_res(box, img_h, img_w)
        rect_w = int(np.linalg.norm(box[0] - box[1]))
        rect_h = int(np.linalg.norm(box[0] - box[3]))
        if rect_w <= 3 or rect_h <= 3:
            continue
        dt_boxes_new.append(box)
    return np.array(dt_boxes_new)


# ---------------------------------------------------------------------------
# Crop helper  (mirrors get_rotate_crop_image)
# ---------------------------------------------------------------------------

def _get_rotate_crop_image(img, points):
    assert len(points) == 4, "points must be 4×2"
    img_crop_width  = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3])
    ))
    img_crop_height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2])
    ))
    pts_std = np.float32([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height],
    ])
    M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)
    dst = cv2.warpPerspective(
        img, M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if dst.shape[0] * 1.0 / dst.shape[1] >= 1.5:
        dst = np.rot90(dst)
    return dst


# ---------------------------------------------------------------------------
# Classification pre/post-processing  (mirrors TextClassifier)
# ---------------------------------------------------------------------------

def _cls_resize_norm(img, cls_image_shape):
    imgC, imgH, imgW = cls_image_shape
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = imgW if math.ceil(imgH * ratio) > imgW else int(math.ceil(imgH * ratio))
    resized = cv2.resize(img, (resized_w, imgH)).astype(np.float32)

    if imgC == 1:
        resized = resized / 255.0
        resized = resized[np.newaxis, :]
    else:
        resized = resized.transpose(2, 0, 1) / 255.0

    resized = (resized - 0.5) / 0.5
    padded = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    return padded


def _cls_postprocess(prob_out, label_list, cls_thresh):
    """
    prob_out : ndarray [N, 2]
    Returns  : list of (label_str, score_float)
    """
    pred_idxs = prob_out.argmax(axis=1)
    return [(label_list[idx], float(prob_out[i, idx]))
            for i, idx in enumerate(pred_idxs)]


# ---------------------------------------------------------------------------
# Recognition pre-processing  (mirrors SVTR_LCNet path in TextRecognizer)
# ---------------------------------------------------------------------------

def _rec_resize_norm(img, imgH, imgW):
    """
    Standard CTC rec preprocessing (non-NRTR/RFL/SPIN path).
    img  : BGR uint8 crop
    Returns: float32 CHW padded to [3, imgH, imgW]
    """
    h, w = img.shape[:2]
    imgC = img.shape[2] if img.ndim == 3 else 1

    ratio      = w / float(h)
    resized_w  = imgW if math.ceil(imgH * ratio) > imgW else int(math.ceil(imgH * ratio))
    resized    = cv2.resize(img, (resized_w, imgH)).astype(np.float32)

    resized = resized.transpose(2, 0, 1) / 255.0   # CHW
    resized = (resized - 0.5) / 0.5

    padded = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padded[:, :, :resized_w] = resized
    return padded


# ---------------------------------------------------------------------------
# Recognition post-processing  (mirrors CTCLabelDecode)
# ---------------------------------------------------------------------------

# Exact contents of ppocr/utils/en_dict.txt read line-by-line.
# Paddle strips "\n" and "\r\n" from each line, so the space line becomes "".
# That empty string is included here as-is to mirror Paddle's list exactly.
# use_space_char=True then appends a real " " at the end.
# Result: 95 file chars + "" (stripped space line) + " " = 96 chars.
# With CTC blank prepended at index 0: 97 total → matches model output dim.
_EN_DICT_DEFAULT = [
    '0','1','2','3','4','5','6','7','8','9',
    ':',';','<','=','>','?','@',
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    '[','\\',']','^','_','`',
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z',
    '{','|','}','~','!','"','#','$','%','&',"'",'(',')','*','+',',','-','.','/',
    '',   # space line stripped to "" by Paddle's file reader
]


class _CTCLabelDecode:
    def __init__(self, character_dict_path=None, use_space_char=True):
        self.reverse = False

        if character_dict_path is None:
            # Mirror Paddle's en_dict.txt loading exactly.
            # The "" entry above corresponds to the stripped space line in the file.
            character_str = list(_EN_DICT_DEFAULT)
            if use_space_char:
                character_str.append(" ")
        else:
            character_str = []
            with open(character_dict_path, "rb") as f:
                for line in f:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    character_str.append(line)
            if use_space_char:
                character_str.append(" ")
            if "arabic" in character_dict_path:
                self.reverse = True

        # prepend CTC blank token at index 0
        self.character = ["blank"] + character_str

    def _get_ignored_tokens(self):
        return [0]  # CTC blank

    def decode(self, text_index, text_prob):
        result_list = []
        ignored = self._get_ignored_tokens()
        for b in range(len(text_index)):
            sel = np.ones(len(text_index[b]), dtype=bool)
            sel[1:] = text_index[b][1:] != text_index[b][:-1]   # remove duplicates
            for tok in ignored:
                sel &= (text_index[b] != tok)

            chars  = [self.character[idx] for idx in text_index[b][sel]]
            probs  = text_prob[b][sel]
            conf   = float(np.mean(probs)) if len(probs) > 0 else 0.0
            text   = "".join(chars)
            if self.reverse:
                text = text[::-1]
            result_list.append((text, conf))
        return result_list

    def __call__(self, preds):
        """
        preds : ndarray [N, T, num_classes]
        Returns: list of (text, confidence)
        """
        if isinstance(preds, (list, tuple)):
            preds = preds[-1]
        preds_idx  = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        return self.decode(preds_idx, preds_prob)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class paddleOpenVINO:
    """
    Drop-in OpenVINO replacement for PaddleOCR.

    Parameters
    ----------
    det_model_path : str
        Path to detection model (.xml or .onnx).
    rec_model_path : str
        Path to recognition model (.xml or .onnx).
    cls_model_path : str or None
        Path to direction-classifier model (.xml or .onnx).
        Pass None to skip classification.
    det_algorithm : str
        'DB' or 'DB++'. Controls normalization mean/std for detection.
    rec_char_dict_path : str or None
        Path to character dict .txt file.
        None → default alphanumeric set (0-9 a-z).
    use_space_char : bool
        Append space character to dict. Default True (same as PaddleOCR).
    det_limit_side_len : float
        Max/min side length before detection resize. Default 960.
    det_limit_type : str
        'max' or 'min'. Default 'max'.
    det_db_thresh : float
        Pixel threshold on probability map. Default 0.3.
    det_db_box_thresh : float
        Box score threshold. Default 0.6.
    det_db_unclip_ratio : float
        Unclip expansion ratio. Default 1.5.
    det_db_score_mode : str
        'fast' (bbox mean) or 'slow' (polygon mean). Default 'fast'.
    use_dilation : bool
        Dilate segmentation mask before contour finding. Default False.
    det_box_type : str
        'quad' only (poly not supported here). Default 'quad'.
    rec_image_shape : str
        'C,H,W' string. Default '3,48,320'.
    rec_batch_num : int
        Recognition batch size. Default 6.
    cls_image_shape : str
        'C,H,W' string for classifier. Default '3,48,192'.
    cls_batch_num : int
        Classifier batch size. Default 6.
    cls_thresh : float
        Score threshold to apply 180° rotation. Default 0.9.
    drop_score : float
        Discard recognition results with confidence below this. Default 0.5.
    """

    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        cls_model_path: str = None,
        det_algorithm: str = "DB",
        rec_char_dict_path: str = None,
        use_space_char: bool = True,
        # det params
        det_limit_side_len: float = 960,
        det_limit_type: str = "max",
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.6,
        det_db_unclip_ratio: float = 1.5,
        det_db_score_mode: str = "fast",
        use_dilation: bool = False,
        det_box_type: str = "quad",
        # rec params
        rec_image_shape: str = "3, 48, 320",
        rec_batch_num: int = 6,
        # cls params
        cls_image_shape: str = "3, 48, 192",
        cls_batch_num: int = 6,
        cls_thresh: float = 0.9,
        # output filter
        drop_score: float = 0.5,
    ):
        # --- validate det_algorithm ---
        det_algorithm = det_algorithm.upper()
        if det_algorithm not in ("DB", "DB++"):
            raise ValueError("det_algorithm must be 'DB' or 'DB++'")
        self.det_algorithm = det_algorithm

        # --- load models ---
        self._det_model, self._det_input, self._det_outputs = _load_ov_model(det_model_path)
        self._rec_model, self._rec_input, self._rec_outputs = _load_ov_model(rec_model_path)

        self._use_cls = cls_model_path is not None
        if self._use_cls:
            self._cls_model, self._cls_input, self._cls_outputs = _load_ov_model(cls_model_path)

        # --- detection params ---
        self.det_limit_side_len  = det_limit_side_len
        self.det_limit_type      = det_limit_type
        self.det_db_thresh       = det_db_thresh
        self.det_db_box_thresh   = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.det_db_score_mode   = det_db_score_mode
        self.use_dilation        = use_dilation
        self.det_box_type        = det_box_type

        # pick mean/std based on algorithm
        if det_algorithm == "DB++":
            self._det_mean = _DBPP_MEAN
            self._det_std  = _DBPP_STD
        else:
            self._det_mean = _DET_MEAN
            self._det_std  = _DET_STD

        # --- recognition params ---
        self.rec_image_shape = [int(v) for v in rec_image_shape.replace(" ", "").split(",")]
        self.rec_batch_num   = rec_batch_num
        self._ctc_decode     = _CTCLabelDecode(rec_char_dict_path, use_space_char)
        self.drop_score      = drop_score

        # --- classification params ---
        self.cls_image_shape = [int(v) for v in cls_image_shape.replace(" ", "").split(",")]
        self.cls_batch_num   = cls_batch_num
        self.cls_thresh      = cls_thresh
        self.label_list      = ["0", "180"]

        # ── args namespace (mirrors PaddleOCR reader.args) ────────────────
        # Stored so external code that does vars(reader.args) or
        # reader.args.some_param keeps working without modification.
        import argparse
        self.args = argparse.Namespace(
            # model paths
            det_model_dir        = det_model_path,
            rec_model_dir        = rec_model_path,
            rec_char_dict_path   = rec_char_dict_path,
            # detection
            det_limit_side_len   = det_limit_side_len,
            det_limit_type       = det_limit_type,
            det_db_thresh        = det_db_thresh,
            det_db_box_thresh    = det_db_box_thresh,
            det_db_unclip_ratio  = det_db_unclip_ratio,
            use_dilation         = use_dilation,
            det_db_score_mode    = det_db_score_mode,
            det_box_type         = det_box_type,
            det_algorithm        = "DB",
            # recognition
            # rec_image_shape      = f"3, {rec_img_h}, {rec_img_w}",
            rec_image_shape      = f"3, 48, 320",
            rec_batch_num        = rec_batch_num,
            rec_algorithm        = "SVTR_LCNet",
            use_space_char       = use_space_char,
            drop_score           = drop_score,
            # classifier (disabled)
            use_angle_cls        = False,
            cls_model_dir        = None,
            # runtime
            use_gpu              = False,
            use_onnx             = False,
            use_openvino         = True,
            openvino_device      = "CPU",
            # cpu_threads          = num_threads,
            enable_mkldnn        = False,
            lang                 = "en",
            show_log             = False,
            benchmark            = False,)

    # ------------------------------------------------------------------
    # Internal: detection
    # ------------------------------------------------------------------

    def _detect(self, img):
        """
        img : BGR uint8 ndarray (H, W, 3)
        Returns: ndarray of shape [N, 4, 2] int32 quad boxes
                 in original image coordinates.
        """
        blob, shape_list = _det_preprocess(
            img,
            self.det_limit_side_len,
            self.det_limit_type,
            self._det_mean,
            self._det_std,
        )

        infer_req = self._det_model.create_infer_request()
        infer_req.infer({self._det_input.any_name: blob})
        outputs = [infer_req.get_output_tensor(i).data for i in range(len(self._det_outputs))]

        boxes_batch = _det_postprocess(
            outputs,
            shape_list,
            thresh=self.det_db_thresh,
            box_thresh=self.det_db_box_thresh,
            unclip_ratio=self.det_db_unclip_ratio,
            use_dilation=self.use_dilation,
            score_mode=self.det_db_score_mode,
        )

        dt_boxes = boxes_batch[0]["points"]
        if dt_boxes is None or len(dt_boxes) == 0:
            return np.array([])

        dt_boxes = _filter_tag_det_res(dt_boxes, img.shape)
        return dt_boxes

    # ------------------------------------------------------------------
    # Internal: classification
    # ------------------------------------------------------------------

    def _classify(self, img_list):
        """
        img_list : list of BGR uint8 crops
        Returns  : (rotated_img_list, cls_res)
          cls_res: list of (label_str, score_float)
        """
        img_list  = copy.deepcopy(img_list)
        img_num   = len(img_list)
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices    = np.argsort(np.array(width_list))

        cls_res   = [["", 0.0]] * img_num
        infer_req = self._cls_model.create_infer_request()

        for beg in range(0, img_num, self.cls_batch_num):
            end = min(img_num, beg + self.cls_batch_num)
            batch = []
            for ino in range(beg, end):
                normed = _cls_resize_norm(img_list[indices[ino]], self.cls_image_shape)
                batch.append(normed[np.newaxis, :])

            batch = np.concatenate(batch).copy()
            infer_req.infer({self._cls_input.any_name: batch})
            prob_out = infer_req.get_output_tensor(0).data.copy()

            results = _cls_postprocess(prob_out, self.label_list, self.cls_thresh)
            for rno, (label, score) in enumerate(results):
                idx = indices[beg + rno]
                cls_res[idx] = [label, score]
                if "180" in label and score > self.cls_thresh:
                    img_list[idx] = cv2.rotate(img_list[idx], cv2.ROTATE_90_CLOCKWISE)

        return img_list, cls_res

    # ------------------------------------------------------------------
    # Internal: recognition
    # ------------------------------------------------------------------

    def _recognize(self, img_list):
        """
        img_list : list of BGR uint8 crops (already orientation-corrected)
        Returns  : list of (text_str, confidence_float)
        """
        img_num    = len(img_list)
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices    = np.argsort(np.array(width_list))

        rec_res   = [["", 0.0]] * img_num
        _, imgH, imgW = self.rec_image_shape
        infer_req = self._rec_model.create_infer_request()

        for beg in range(0, img_num, self.rec_batch_num):
            end = min(img_num, beg + self.rec_batch_num)

            # compute max_wh_ratio for this batch to set dynamic width
            max_wh_ratio = imgW / imgH
            for ino in range(beg, end):
                h, w = img_list[indices[ino]].shape[:2]
                max_wh_ratio = max(max_wh_ratio, w / float(h))

            batch = []
            for ino in range(beg, end):
                dyn_imgW = int(math.ceil(imgH * max_wh_ratio))
                normed   = _rec_resize_norm(img_list[indices[ino]], imgH, dyn_imgW)
                batch.append(normed[np.newaxis, :])

            batch = np.concatenate(batch).copy()
            infer_req.infer({self._rec_input.any_name: batch})
            preds = infer_req.get_output_tensor(0).data.copy()  # [N, T, classes]

            results = self._ctc_decode(preds)
            for rno, (text, conf) in enumerate(results):
                rec_res[indices[beg + rno]] = [text, conf]

        return rec_res

    # ------------------------------------------------------------------
    # Public API  —  .ocr()
    # ------------------------------------------------------------------

    def ocr(self, img, det=True, rec=True, cls=True):
        """
        Run the full OCR pipeline.

        Parameters
        ----------
        img : str or ndarray
            Image path or BGR uint8 ndarray.
        det : bool
            Run text detection. Default True.
        rec : bool
            Run text recognition. Default True.
        cls : bool
            Run direction classification (only when cls_model_path was given
            at init). Default True.

        Returns
        -------
        list of [box, (text, score)]
          box   : ndarray [4, 2] int32  — quad corners in original image space
          text  : str
          score : float  — recognition confidence

        If det=False, returns list of (text, score) directly.
        If rec=False, returns list of boxes only.
        """
        # --- load image ---
        if isinstance(img, str):
            img = cv2.imread(img)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img}")

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        ori_img = img.copy()

        # --- detection ---
        if det:
            dt_boxes = self._detect(img)
            if dt_boxes is None or len(dt_boxes) == 0:
                return []
        else:
            # treat whole image as single crop
            h, w = img.shape[:2]
            dt_boxes = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.int32)

        if not rec:
            return [box for box in dt_boxes]

        # --- crop boxes ---
        crops = []
        for box in dt_boxes:
            crop = _get_rotate_crop_image(ori_img, box.astype(np.float32))
            crops.append(crop)

        # --- classification ---
        use_cls = cls and self._use_cls
        if use_cls:
            crops, _cls_res = self._classify(crops)

        # --- recognition ---
        rec_res = self._recognize(crops)

        # --- assemble result ---
        results = []
        for box, (text, score) in zip(dt_boxes, rec_res):
            if score >= self.drop_score:
                results.append([box, (text, score)])

        return [results]
    
if __name__ == "__main__":
    model = paddleOpenVINO(
        det_model_path="models/paddle_ov_det/ppocrv3-en-det.xml",
        rec_model_path="models/paddle_ov_rec/ppocrv4-en-rec.xml",
    )
    imr_src=r"data\readablev3\courtesy_amount\set_10_1.jpg"
    results = model.ocr(imr_src)
    for box, (text, score) in results[0]:
        print(f"{text}  ({score:.3f})")

    print(results)
    # print("RAW OCR:", r)
    # if not r or not results[0]:
    #     return ""
    print(" ".join(line[1][0] for line in results[0] if line))