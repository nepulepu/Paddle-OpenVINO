"""
Standalone PaddleOCR OpenVINO IR pipeline — no paddle/paddlepaddle required.

Dependencies:
    pip install openvino opencv-python numpy pyclipper shapely

Model files expected per model:
    models/det/ppocrv3-en-det.xml  +  models/det/ppocrv3-en-det.bin
    models/rec/ppocrv4-en-rec.xml  +  models/rec/ppocrv4-en-rec.bin

Convert from ONNX if needed:
    mo --input_model model.onnx --output_dir models/det/
    # or with the newer CLI:
    ovc model.onnx --output_model models/det/ppocrv3-en-det
"""

import math
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
from openvino.runtime import Core


class OpenVinoOCR:
    """
    Self-contained PaddleOCR pipeline backed by OpenVINO IR models.

    All pre-processing, post-processing, and helper logic lives inside
    this single class — no external utility functions needed.

    Usage
    -----
        ocr = OpenVinoOCR(
            det_model_xml="models/det/ppocrv3-en-det.xml",
            rec_model_xml="models/rec/ppocrv4-en-rec.xml",
            rec_char_dict_path="en_dict.txt",   # None → digits+lowercase
        )
        text = ocr.run(img)          # returns a plain string
        results = ocr(img)           # returns list of [box, (text, conf)]
    """

    # ─────────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        det_model_xml: str,
        rec_model_xml: str,
        rec_char_dict_path: str = None,
        # detection params
        det_limit_side_len: int = 960,
        det_limit_type: str = "max",        # "max" | "min"
        det_thresh: float = 0.3,            # binarisation threshold
        det_box_thresh: float = 0.6,        # box confidence threshold
        det_unclip_ratio: float = 1.5,      # how much to expand boxes
        det_use_dilation: bool = False,
        det_score_mode: str = "fast",       # "fast" | "slow"
        det_box_type: str = "quad",         # "quad" | "poly"
        # recognition params
        rec_img_h: int = 48,
        rec_img_w: int = 320,
        rec_batch_size: int = 6,
        use_space_char: bool = True,
        drop_score: float = 0.5,
        # openvino params
        device: str = "CPU",
        num_streams: int = 1,
        num_threads: int = 1,
    ):
        ie = Core()

        base_config = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": str(num_streams)}
        if num_threads > 0:
            base_config["INFERENCE_NUM_THREADS"] = str(num_threads)

        # ── compile detection model ──────────────────────────────────────
        det_model = ie.read_model(det_model_xml)
        self._det_compiled = ie.compile_model(det_model, device, base_config)
        self._det_infer    = self._det_compiled.create_infer_request()
        self._det_input    = self._det_compiled.input(0)
        self._det_output   = self._det_compiled.output(0)

        # ── recognition model — kept as XML path for lazy reshape+compile ─
        # reshape() must be called on the Model object BEFORE compile_model.
        # We re-read + compile once per unique (batch, width) shape seen at
        # runtime, caching compiled models to avoid redundant recompilation.
        self._ie          = ie
        self._rec_xml     = rec_model_xml        # path kept for re-read
        self._rec_model   = ie.read_model(rec_model_xml)
        self._rec_device  = device
        self._rec_config  = base_config
        self._rec_input_name  = self._rec_model.input(0).any_name
        self._rec_output_name = self._rec_model.output(0).any_name
        # cache: shape_tuple → (compiled_model, infer_request)
        self._rec_cache: dict = {}

        # ── store params ─────────────────────────────────────────────────
        self._det_limit_side_len = det_limit_side_len
        self._det_limit_type     = det_limit_type
        self._det_thresh         = det_thresh
        self._det_box_thresh     = det_box_thresh
        self._det_unclip_ratio   = det_unclip_ratio
        self._det_min_size       = 3
        self._det_use_dilation   = det_use_dilation
        self._det_score_mode     = det_score_mode
        self._det_box_type       = det_box_type
        self._dilation_kernel    = np.array([[1, 1], [1, 1]]) if det_use_dilation else None

        self._rec_img_h      = rec_img_h
        self._rec_img_w      = rec_img_w
        self._rec_batch_size = rec_batch_size
        self._drop_score     = drop_score

        # ── build character table ─────────────────────────────────────────
        self._character = self._build_character_list(rec_char_dict_path, use_space_char)

        # ── args namespace (mirrors PaddleOCR reader.args) ────────────────
        # Stored so external code that does vars(reader.args) or
        # reader.args.some_param keeps working without modification.
        import argparse
        self.args = argparse.Namespace(
            # model paths
            det_model_dir        = det_model_xml,
            rec_model_dir        = rec_model_xml,
            rec_char_dict_path   = rec_char_dict_path,
            # detection
            det_limit_side_len   = det_limit_side_len,
            det_limit_type       = det_limit_type,
            det_db_thresh        = det_thresh,
            det_db_box_thresh    = det_box_thresh,
            det_db_unclip_ratio  = det_unclip_ratio,
            use_dilation         = det_use_dilation,
            det_db_score_mode    = det_score_mode,
            det_box_type         = det_box_type,
            det_algorithm        = "DB",
            # recognition
            rec_image_shape      = f"3, {rec_img_h}, {rec_img_w}",
            rec_batch_num        = rec_batch_size,
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
            openvino_device      = device,
            cpu_threads          = num_threads,
            enable_mkldnn        = False,
            lang                 = "en",
            show_log             = False,
            benchmark            = False,
        )


    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────

    def __call__(self, img: np.ndarray) -> list:
        """
        Full OCR pass.

        Parameters
        ----------
        img : BGR numpy array  (H, W, 3)

        Returns
        -------
        list of  [box_4x2_array, (text_str, confidence_float)]
        Empty list if nothing detected.
        """
        dt_boxes = self._detect(img)
        if not dt_boxes:
            return []

        crops = [self._crop_text_region(img, np.float32(box)) for box in dt_boxes]
        rec_results = self._recognize(crops)

        return [
            [box, (text, conf)]
            for box, (text, conf) in zip(dt_boxes, rec_results)
            if conf >= self._drop_score
        ]

    def run(self, img) -> str:
        """
        Accepts either:
          - a file path (str or pathlib.Path)  -- mirrors run_paddle(reader, img)
          - a BGR numpy array                  -- direct inference

        Returns all recognised text joined by spaces, or "" if nothing found.
        """
        if not isinstance(img, np.ndarray):
            loaded = cv2.imread(str(img))
            if loaded is None:
                raise FileNotFoundError(f"Could not read image: {img}")
            img = loaded
        results = self(img)
        if not results:
            return ""
        return " ".join(text for _, (text, _conf) in results)

    # ─────────────────────────────────────────────────────────────────────
    # Detection
    # ─────────────────────────────────────────────────────────────────────

    def _detect(self, img: np.ndarray) -> list:
        """Run detection model; return sorted, filtered list of 4-pt boxes."""
        blob, shape_list = self._det_preprocess(img)
        # OpenVINO inference
        result = self._det_infer.infer({self._det_input: blob})
        maps   = result[self._det_output]          # [1, 1, H, W]

        dt_boxes = self._db_postprocess(maps, shape_list)
        if dt_boxes is None or len(dt_boxes) == 0:
            return []
        dt_boxes = self._filter_boxes(dt_boxes, img.shape)
        if len(dt_boxes) == 0:
            return []
        return self._sort_boxes(dt_boxes)

    def _det_preprocess(self, img: np.ndarray):
        """
        DetResizeForTest → NormalizeImage (ImageNet) → HWC→CHW → batch dim.
        Returns (blob [1,3,H,W], shape_list [1,4]).
        """
        resized, shape_list = self._det_resize(img)
        normed = self._det_normalize(resized)
        blob = np.expand_dims(normed, axis=0).astype(np.float32)
        shape_list = np.expand_dims(shape_list, axis=0)
        return blob, shape_list

    def _det_resize(self, img: np.ndarray):
        """
        Resize so the chosen side doesn't exceed limit_side_len,
        then snap H and W to the nearest multiple of 32.
        """
        h, w   = img.shape[:2]
        ratio  = 1.0
        ls     = self._det_limit_side_len

        if self._det_limit_type == "max":
            if max(h, w) > ls:
                ratio = ls / h if h >= w else ls / w
        elif self._det_limit_type == "min":
            if min(h, w) < ls:
                ratio = ls / h if h <= w else ls / w

        rh = max(int(round(h * ratio / 32) * 32), 32)
        rw = max(int(round(w * ratio / 32) * 32), 32)

        resized    = cv2.resize(img, (rw, rh))
        shape_list = [h, w, rh / h, rw / w]   # orig_h, orig_w, ratio_h, ratio_w
        return resized, shape_list

    @staticmethod
    def _det_normalize(img: np.ndarray) -> np.ndarray:
        """
        float32 / 255  →  subtract ImageNet mean  →  divide ImageNet std
        →  HWC to CHW
        """
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = img.astype(np.float32) / 255.0
        img  = (img - mean) / std          # still HWC
        return img.transpose(2, 0, 1)      # → CHW

    # ── DBPostProcess ──────────────────────────────────────────────────

    def _db_postprocess(self, maps: np.ndarray, shape_list: np.ndarray) -> np.ndarray:
        """
        maps        : [1, 1, H, W]  sigmoid probability heatmap
        shape_list  : [1, 4]  [orig_h, orig_w, ratio_h, ratio_w]
        Returns array of boxes in original-image coordinates.
        """
        pred        = maps[:, 0, :, :]          # [1, H, W]
        segmentation = (pred > self._det_thresh).astype(np.uint8)

        if self._dilation_kernel is not None:
            segmentation[0] = cv2.dilate(segmentation[0], self._dilation_kernel)

        src_h = int(shape_list[0][0])
        src_w = int(shape_list[0][1])

        if self._det_box_type == "poly":
            boxes, _ = self._polygons_from_bitmap(pred[0], segmentation[0], src_w, src_h)
        else:
            boxes, _ = self._boxes_from_bitmap(pred[0], segmentation[0], src_w, src_h)

        return np.array(boxes, dtype="int32") if len(boxes) else np.array([])

    def _boxes_from_bitmap(self, pred, bitmap, dest_w, dest_h):
        h, w = bitmap.shape
        outs = cv2.findContours((bitmap * 255).astype(np.uint8),
                                cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = outs[0] if len(outs) == 2 else outs[1]

        boxes, scores = [], []
        for contour in contours[:1000]:   # max_candidates = 1000
            points, sside = self._get_mini_boxes(contour)
            if sside < self._det_min_size:
                continue
            points = np.array(points)
            score  = self._box_score_fast(pred, points.reshape(-1, 2))
            if score < self._det_box_thresh:
                continue
            box = self._unclip(points).reshape(-1, 1, 2)
            box, sside = self._get_mini_boxes(box)
            if sside < self._det_min_size + 2:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / w * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * dest_h), 0, dest_h)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return boxes, scores

    def _polygons_from_bitmap(self, pred, bitmap, dest_w, dest_h):
        h, w = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8),
                                       cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes, scores = [], []
        for contour in contours[:1000]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx  = cv2.approxPolyDP(contour, epsilon, True)
            points  = approx.reshape(-1, 2)
            if points.shape[0] < 4:
                continue
            score = self._box_score_fast(pred, points)
            if score < self._det_box_thresh:
                continue
            box = self._unclip(points)
            if len(box) > 1:
                continue
            box = box.reshape(-1, 2)
            _, sside = self._get_mini_boxes(box.reshape(-1, 1, 2))
            if sside < self._det_min_size + 2:
                continue
            box[:, 0] = np.clip(np.round(box[:, 0] / w * dest_w), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] / h * dest_h), 0, dest_h)
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def _unclip(self, box: np.ndarray) -> np.ndarray:
        """Expand a polygon outward by unclip_ratio using pyclipper."""
        poly     = Polygon(box)
        distance = poly.area * self._det_unclip_ratio / poly.length
        offset   = pyclipper.PyclipperOffset()
        offset.AddPath(
            box.astype(np.int32).tolist(),
            pyclipper.JT_ROUND,
            pyclipper.ET_CLOSEDPOLYGON,
        )
        return np.array(offset.Execute(distance))

    @staticmethod
    def _get_mini_boxes(contour):
        """Fit minimum-area rectangle; return 4 ordered corner points + short side."""
        bounding_box = cv2.minAreaRect(contour)
        pts = sorted(cv2.boxPoints(bounding_box).tolist(), key=lambda x: x[0])
        i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
        i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
        return [pts[i1], pts[i2], pts[i3], pts[i4]], min(bounding_box[1])

    @staticmethod
    def _box_score_fast(bitmap: np.ndarray, box: np.ndarray) -> float:
        """Mean probability inside the bounding rectangle of box."""
        h, w  = bitmap.shape[:2]
        box   = box.copy().astype(np.float32)
        xmin  = max(0,     int(np.floor(box[:, 0].min())))
        xmax  = min(w - 1, int(np.ceil(box[:, 0].max())))
        ymin  = max(0,     int(np.floor(box[:, 1].min())))
        ymax  = min(h - 1, int(np.ceil(box[:, 1].max())))
        mask  = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    # ── box geometry helpers ───────────────────────────────────────────

    @staticmethod
    def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
        rect    = np.zeros((4, 2), dtype="float32")
        s       = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp     = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff    = np.diff(tmp, axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def _filter_boxes(self, dt_boxes: np.ndarray, img_shape) -> np.ndarray:
        """
        Reorder each box's corners clockwise, clip to image bounds,
        and drop boxes that are too small (≤ 3 px in either dimension).
        """
        h, w = img_shape[:2]
        out  = []
        for box in dt_boxes:
            box = self._order_points_clockwise(np.array(box, dtype=np.float32))
            box[:, 0] = np.clip(box[:, 0], 0, w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, h - 1)
            if (int(np.linalg.norm(box[0] - box[1])) <= 3 or
                    int(np.linalg.norm(box[0] - box[3])) <= 3):
                continue
            out.append(box)
        return np.array(out)

    @staticmethod
    def _sort_boxes(dt_boxes) -> list:
        """
        Sort text boxes top-to-bottom, left-to-right.
        Boxes within 10 px of the same Y are reordered by X.
        """
        boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        n     = len(boxes)
        for i in range(n - 1):
            for j in range(i, -1, -1):
                if (abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10 and
                        boxes[j + 1][0][0] < boxes[j][0][0]):
                    boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
                else:
                    break
        return boxes

    # ─────────────────────────────────────────────────────────────────────
    # Crop helper (between detection and recognition)
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _crop_text_region(img: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Perspective-warp the 4-point polygon into a straight rectangle.
        Rotates 90° if the crop looks like a vertical strip (h/w ≥ 1.5).
        """
        points    = np.float32(points)
        crop_w    = int(max(np.linalg.norm(points[0] - points[1]),
                            np.linalg.norm(points[2] - points[3])))
        crop_h    = int(max(np.linalg.norm(points[0] - points[3]),
                            np.linalg.norm(points[1] - points[2])))
        dst       = np.float32([[0, 0], [crop_w, 0],
                                 [crop_w, crop_h], [0, crop_h]])
        M         = cv2.getPerspectiveTransform(points, dst)
        crop      = cv2.warpPerspective(
            img, M, (crop_w, crop_h),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        if crop.shape[0] and crop.shape[0] / crop.shape[1] >= 1.5:
            crop = np.rot90(crop)
        return crop

    # ─────────────────────────────────────────────────────────────────────
    # Recognition
    # ─────────────────────────────────────────────────────────────────────

    def _get_rec_request(self, batch_shape: tuple):
        """
        Return a cached infer_request for the given input shape (B, 3, H, W).

        reshape() must be called on the *uncompiled* Model object before
        compile_model() — you cannot reshape a CompiledModel.
        We re-read the XML each time a new shape is needed (cheap: disk read
        only, no re-compilation of already-cached shapes).
        """
        if batch_shape not in self._rec_cache:
            # Re-read fresh so we don't mutate the stored _rec_model
            model_copy = self._ie.read_model(self._rec_xml)
            model_copy.reshape({self._rec_input_name: list(batch_shape)})
            compiled = self._ie.compile_model(model_copy, self._rec_device, self._rec_config)
            request  = compiled.create_infer_request()
            self._rec_cache[batch_shape] = (compiled, request)
        return self._rec_cache[batch_shape][1]

    def _recognize(self, img_crops: list) -> list:
        """
        Recognise text in a list of BGR crop arrays.
        Sorts by aspect ratio for efficient batching, then runs in batches.
        Returns list of (text, confidence) in the original crop order.
        """
        n = len(img_crops)
        if n == 0:
            return []

        ratios  = [im.shape[1] / float(im.shape[0]) for im in img_crops]
        indices = np.argsort(ratios)
        results = [None] * n

        for batch_start in range(0, n, self._rec_batch_size):
            batch_idx  = indices[batch_start: batch_start + self._rec_batch_size]
            batch_imgs = [img_crops[i] for i in batch_idx]

            # Effective width: widest aspect ratio in this batch, capped at rec_img_w
            max_wh = max(
                self._rec_img_w / self._rec_img_h,
                max(im.shape[1] / float(im.shape[0]) for im in batch_imgs),
            )
            eff_w = min(self._rec_img_w, int(math.ceil(self._rec_img_h * max_wh)))

            norm_batch = np.stack([
                self._rec_resize_norm(im, self._rec_img_h, eff_w)
                for im in batch_imgs
            ]).astype(np.float32)   # [B, 3, H, W]

            # Get (or create) a compiled model sized for this exact shape
            req   = self._get_rec_request(tuple(norm_batch.shape))
            out   = req.infer({self._rec_input_name: norm_batch})
            preds = out[self._rec_output_name]   # [B, T, num_classes]

            decoded = self._ctc_decode(preds)
            for local_i, orig_i in enumerate(batch_idx):
                results[orig_i] = decoded[local_i]

        return results

    @staticmethod
    def _rec_resize_norm(img: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
        """
        Resize crop to fixed height, keep aspect ratio capped at img_w,
        pad right side with zeros.
        Normalise: / 255 → − 0.5 → / 0.5   (values in [−1, 1])
        Output shape: [3, img_h, img_w]
        """
        h, w      = img.shape[:2]
        resized_w = min(img_w, int(math.ceil(img_h * (w / float(h)))))
        resized   = cv2.resize(img, (resized_w, img_h))
        normed    = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        normed    = (normed - 0.5) / 0.5
        canvas    = np.zeros((3, img_h, img_w), dtype=np.float32)
        canvas[:, :, :resized_w] = normed
        return canvas

    # ── CTC decoder ───────────────────────────────────────────────────

    @staticmethod
    def _build_character_list(char_dict_path, use_space_char: bool) -> list:
        """
        Build the character vocabulary.
        Index 0 is always the CTC blank token.
        """
        if char_dict_path is None:
            chars = list("0123456789abcdefghijklmnopqrstuvwxyz")
        else:
            with open(char_dict_path, "rb") as f:
                chars = [line.decode("utf-8").rstrip("\r\n") for line in f]
            if use_space_char:
                chars.append(" ")
        return ["blank"] + chars

    def _ctc_decode(self, preds: np.ndarray) -> list:
        """
        Greedy CTC decode.

        preds : float32  [B, T, num_classes]
        Returns list of (text, mean_confidence) per sample.
        """
        pred_idx  = preds.argmax(axis=2)   # [B, T]
        pred_prob = preds.max(axis=2)      # [B, T]
        out       = []
        for b in range(pred_idx.shape[0]):
            chars, confs, prev = [], [], -1
            for ci, cp in zip(pred_idx[b], pred_prob[b]):
                ci = int(ci)
                if ci == 0 or ci == prev:   # blank or duplicate → skip
                    prev = ci
                    continue
                chars.append(self._character[ci])
                confs.append(float(cp))
                prev = ci
            text = "".join(chars)
            conf = float(np.mean(confs)) if confs else 0.0
            out.append((text, conf))
        return out



# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ocr = OpenVinoOCR(
        det_model_xml="models/paddle_ov_det/ppocrv3-en-det.xml",
        rec_model_xml="models/paddle_ov_rec/ppocrv4-en-rec.xml",
        rec_char_dict_path="models/en_dict.txt",   # replace with path to your char dict .txt
        device="CPU",
    )

    img = cv2.imread(r"data\readablev3\courtesy_amount\set_10_1.jpg")

    # Option A — get plain text string
    print(ocr.run(img))

    print(ocr(img))
    # Option B — get structured results
    for box, (text, conf) in ocr(img):
        print(f"[{conf:.2f}] {text}  @  {box.tolist()}")