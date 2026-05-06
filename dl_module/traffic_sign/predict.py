import cv2
import numpy as np
import tensorflow as tf
import os
import csv

from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Model paths & lazy-load handles
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join("models", "traffic_classifier.h5")
YOLO_MODEL_PATH = os.path.join("models", "yolo_signs.pt")

model      = None   # Keras CNN  — classifier
yolo_model = None   # YOLO model — region proposer

# ── Confidence thresholds ─────────────────────────────────────────────────────
# Minimum YOLO box confidence to accept a region proposal as a candidate crop.
YOLO_CONF_THRESHOLD:  float = 0.40
# Per-frame Keras CNN confidence needed to start or extend a consecutive streak.
FRAME_CONF_THRESHOLD: float = 0.75
# Single-image static API threshold (more lenient — no temporal smoothing).
SIGN_CONF_THRESHOLD:  float = 0.80

# ── Pre-processing heuristic thresholds ──────────────────────────────────────
# These two cheap checks run on the YOLO crop BEFORE the Keras forward pass.
# They act as a second guard against YOLO false positives (brake lights,
# coloured billboards) that are geometrically box-shaped but not traffic signs.
MIN_VARIANCE:     float = 150.0   # grayscale variance floor
MIN_EDGE_DENSITY: float = 0.05    # fraction of 30×30 pixels that must be edges

# ── Consecutive-frame temporal lock ──────────────────────────────────────────
CONSECUTIVE_REQUIRED: int = 3    # back-to-back frames a class must hold
_STALE_THRESHOLD:     int = 20   # non-detection frames before confirmed clears

# Internal mutable state — one dict keeps the module namespace clean.
_consecutive: dict = {
    "class_index":     None,
    "count":           0,
    "confs":           [],
    "confirmed_label": None,
    "confirmed_conf":  0.0,
    "confirmed_class": None,
    "no_detect_count": 0,
    "latest_bbox":     None,
}

_roi_index = None   # lazily built from CSV metadata

# ─────────────────────────────────────────────────────────────────────────────
# Class map (GTSRB — 43 classes)
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = {
    0:  'Speed limit (20km/h)',          1:  'Speed limit (30km/h)',
    2:  'Speed limit (50km/h)',          3:  'Speed limit (60km/h)',
    4:  'Speed limit (70km/h)',          5:  'Speed limit (80km/h)',
    6:  'End of speed limit (80km/h)',   7:  'Speed limit (100km/h)',
    8:  'Speed limit (120km/h)',         9:  'No passing',
    10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road',                13: 'Yield',
    14: 'Stop',                         15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',    17: 'No entry',
    18: 'General caution',              19: 'Dangerous curve left',
    20: 'Dangerous curve right',        21: 'Double curve',
    22: 'Bumpy road',                   23: 'Slippery road',
    24: 'Road narrows on the right',    25: 'Road work',
    26: 'Traffic signals',              27: 'Pedestrians',
    28: 'Children crossing',            29: 'Bicycles crossing',
    30: 'Beware of ice/snow',           31: 'Wild animals crossing',
    32: 'End speed + passing limits',   33: 'Turn right ahead',
    34: 'Turn left ahead',              35: 'Ahead only',
    36: 'Go straight or right',         37: 'Go straight or left',
    38: 'Keep right',                   39: 'Keep left',
    40: 'Roundabout mandatory',         41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons',
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_traffic_model() -> bool:
    """
    Lazily load both models on first call.

    Keras CNN  — required.  Returns False and disables all detection if the
                 weights file is missing or corrupt.

    YOLO model — optional.  If `models/yolo_signs.pt` does not exist yet
                 (e.g. not yet downloaded), a warning is printed and
                 yolo_model is left as None.  The pipeline then degrades
                 gracefully to the heuristic-gated centre-crop fallback
                 used by predict_traffic_sign (static image API).  No crash.
    """
    global model, yolo_model

    # ── Keras sign classifier ─────────────────────────────────────────────────
    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Traffic Sign Classifier (Keras) loaded.")
        except Exception as e:
            print(f"❌ Failed to load Keras model: {e}")
            return False

    # ── YOLO region proposer ──────────────────────────────────────────────────
    if yolo_model is None:
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print("✅ Traffic Sign Detector (YOLO) loaded.")
        except Exception as e:
            # Non-fatal: pipeline falls back to centre-crop path automatically.
            print(f"⚠️  YOLO sign detector unavailable ({e}). "
                  "Centre-crop fallback active.")
            yolo_model = None

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_yolo_crop(frame: np.ndarray):
    if yolo_model is None:
        return None, None

    try:
        # NOTE: If you haven't added imgsz=1024 here yet, do it!
        results = yolo_model.predict(source=frame, verbose=False,
                                     conf=YOLO_CONF_THRESHOLD, imgsz=1024)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None, None

        confs    = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].cpu().numpy())

        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None, None

        # Return BOTH the crop and the coordinates
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    except Exception:
        return None, None


def _is_sign_like(crop: np.ndarray) -> bool:
    """
    Pre-processing heuristic gate — runs BEFORE the Keras forward pass.

    Even a tight YOLO crop can still be a false positive (a brake light,
    a coloured sticker, a road marking).  Two cheap checks on the 30×30
    thumbnail catch most of these cases before wasting a CNN forward pass:

    1. Grayscale variance — uniform / near-uniform regions fail this.
    2. Canny edge density — signs have a hard border + internal symbols;
       soft blobs (reflections, single-colour patches) do not.
    """
    if crop is None or crop.size == 0:
        return False

    small = cv2.resize(crop, (30, 30))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if float(np.var(gray)) < MIN_VARIANCE:
        return False

    edges        = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / edges.size
    if edge_density < MIN_EDGE_DENSITY:
        return False

    return True


def _run_keras_on_crop(crop: np.ndarray):
    """Resize crop to 30×30, normalise, run Keras forward pass."""
    img         = cv2.resize(crop, (30, 30))
    img         = img / 255.0
    img         = np.expand_dims(img, axis=0)
    predictions = model.predict(img, verbose=0)
    class_index = int(np.argmax(predictions))
    confidence  = float(np.max(predictions))
    return class_index, confidence


def _reset_streak() -> None:
    """Reset the active streak without touching the confirmed label."""
    _consecutive["class_index"] = None
    _consecutive["count"]       = 0
    _consecutive["confs"]       = []


def _record_no_detection() -> None:
    """
    Called when a frame yields no valid detection after all stages.
    Resets the streak and advances the stale counter.
    Clears the confirmed label after _STALE_THRESHOLD consecutive misses.
    """
    _reset_streak()
    _consecutive["no_detect_count"] += 1
    if _consecutive["no_detect_count"] >= _STALE_THRESHOLD:
        _consecutive["confirmed_label"] = None
        _consecutive["confirmed_conf"]  = 0.0
        _consecutive["confirmed_class"] = None


def _advance_streak(class_index: int, conf: float) -> None:
    """
    Extend or restart the consecutive-frame detection streak.

    Same class  → increment count, accumulate confidence.
    New class   → discard old streak, restart at count = 1.
    count ≥ CONSECUTIVE_REQUIRED → promote to confirmed (stable) state that
    the renderer sees.  A single lucky high-confidence frame cannot promote.
    """
    if _consecutive["class_index"] == class_index:
        _consecutive["count"] += 1
        _consecutive["confs"].append(conf)
    else:
        _consecutive["class_index"] = class_index
        _consecutive["count"]       = 1
        _consecutive["confs"]       = [conf]

    if _consecutive["count"] >= CONSECUTIVE_REQUIRED:
        avg_conf = sum(_consecutive["confs"]) / len(_consecutive["confs"])
        _consecutive["confirmed_label"] = CLASSES.get(class_index, "Unknown")
        _consecutive["confirmed_conf"]  = avg_conf
        _consecutive["confirmed_class"] = class_index


def _current_confirmed():
    label = _consecutive["confirmed_label"]
    if label is None:
        return None, None, None, None
    return label, _consecutive["confirmed_conf"], _consecutive["confirmed_class"], _consecutive["latest_bbox"]


def _build_roi_index() -> dict:
    index = {}
    for csv_path in [os.path.join("data", "Test.csv"),
                     os.path.join("data", "Train.csv")]:
        if not os.path.exists(csv_path):
            continue
        try:
            with open(csv_path, "r", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    rel_path = str(row.get("Path", "")).replace("\\", "/")
                    if not rel_path:
                        continue
                    key = os.path.basename(rel_path).lower()
                    try:
                        x1 = int(float(row.get("Roi.X1", 0)))
                        y1 = int(float(row.get("Roi.Y1", 0)))
                        x2 = int(float(row.get("Roi.X2", 0)))
                        y2 = int(float(row.get("Roi.Y2", 0)))
                    except Exception:
                        continue
                    index[key] = (x1, y1, x2, y2)
        except Exception:
            continue
    return index


def _roi_from_metadata(image_path: str):
    global _roi_index
    if _roi_index is None:
        _roi_index = _build_roi_index()
    return _roi_index.get(os.path.basename(image_path).lower())


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_traffic_sign(image_path: str) -> str:
    """
    Single-image prediction used by the GUI's static-image path.

    Tries up to three crops (full frame, centre quarter, CSV ROI) and picks
    the one that passes the heuristic gate and scores highest with the Keras
    CNN.  YOLO is intentionally skipped here: static images already provide
    the final answer and there is no temporal history to smooth over.
    """
    if not load_traffic_model():
        return "Model Failed to Load"

    img = cv2.imread(image_path)
    if img is None:
        return "Could not read image"

    h, w = img.shape[:2]
    candidates = [img]

    centre = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    if centre is not None and centre.size > 0:
        candidates.append(centre)

    roi = _roi_from_metadata(image_path)
    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(x1 + 1, min(w, x2)), max(y1 + 1, min(h, y2))
        roi_crop = img[y1:y2, x1:x2]
        if roi_crop is not None and roi_crop.size > 0:
            candidates.append(roi_crop)

    best_class, best_conf = None, -1.0
    for patch in candidates:
        if not _is_sign_like(patch):
            continue
        idx, conf = _run_keras_on_crop(patch)
        if conf > best_conf:
            best_class, best_conf = idx, conf

    if best_class is None or best_conf < SIGN_CONF_THRESHOLD:
        return "No confident sign detected"

    return f"{CLASSES.get(best_class, 'Unknown')} ({best_conf * 100:.1f}%)"


def predict_traffic_sign_frame(frame: np.ndarray):
    # Added a 4th None to match the new coordinates return
    if not load_traffic_model() or frame is None:
        return None, None, None, None

    try:
        # ── Stage 1: YOLO region proposal ────────────────────────────────────
        # Now catches the crop AND the coordinates
        crop, bbox = _best_yolo_crop(frame)
        _consecutive["latest_bbox"] = bbox

        # Fallback: YOLO not loaded or no boxes above threshold.
        if crop is None:
            h, w = frame.shape[:2]
            crop = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
            _consecutive["latest_bbox"] = None

        # ── Stage 2a: Heuristic gate ──────────────────────────────────────────
        if not _is_sign_like(crop):
            _record_no_detection()
            return _current_confirmed()

        # ── Stage 2b: Keras CNN forward pass ─────────────────────────────────
        class_index, confidence = _run_keras_on_crop(crop)

        # ── Stage 3: Per-frame confidence gate ────────────────────────────────
        if confidence < FRAME_CONF_THRESHOLD:
            _record_no_detection()
            return _current_confirmed()

        # ── Stage 3: Advance consecutive-frame streak ─────────────────────────
        _consecutive["no_detect_count"] = 0
        _advance_streak(class_index, confidence)
        return _current_confirmed()

    except Exception:
        # Added a 4th None to match the new coordinates return
        return None, None, None, None