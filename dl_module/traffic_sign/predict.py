import cv2
import numpy as np
import tensorflow as tf
import os
import csv

from ultralytics import YOLO

MODEL_PATH      = os.path.join("models", "traffic_classifier.h5")
YOLO_MODEL_PATH = os.path.join("models", "yolo_signs.pt")

model      = None
yolo_model = None

YOLO_CONF_THRESHOLD:  float = 0.40

FRAME_CONF_THRESHOLD: float = 0.75

SIGN_CONF_THRESHOLD:  float = 0.80

MIN_VARIANCE:     float = 150.0
MIN_EDGE_DENSITY: float = 0.05

CONSECUTIVE_REQUIRED: int = 3
_STALE_THRESHOLD:     int = 20

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

_roi_index = None

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

def load_traffic_model() -> bool:

    global model, yolo_model

    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print(" Traffic Sign Classifier (Keras) loaded.")
        except Exception as e:
            print(f" Failed to load Keras model: {e}")
            return False

    if yolo_model is None:
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print(" Traffic Sign Detector (YOLO) loaded.")
        except Exception as e:

            print(f"  YOLO sign detector unavailable ({e}). "
                  "Centre-crop fallback active.")
            yolo_model = None

    return True

def reload_traffic_model() -> bool:

    global model
    if not os.path.exists(MODEL_PATH):
        print(f"  Weight file not found at {MODEL_PATH}.")
        return False
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(" Traffic Sign Classifier reloaded from disk.")
        return True
    except Exception as e:
        print(f" Failed to reload Keras model: {e}")
        return False

def _best_yolo_crop(frame: np.ndarray):
    if yolo_model is None:
        return None, None

    try:

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

        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    except Exception:
        return None, None

def _is_sign_like(crop: np.ndarray) -> bool:

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

    img         = cv2.resize(crop, (30, 30))
    img         = img / 255.0
    img         = np.expand_dims(img, axis=0)
    predictions = model.predict(img, verbose=0)
    class_index = int(np.argmax(predictions))
    confidence  = float(np.max(predictions))
    return class_index, confidence

def _reset_streak() -> None:

    _consecutive["class_index"] = None
    _consecutive["count"]       = 0
    _consecutive["confs"]       = []

def _record_no_detection() -> None:

    _reset_streak()
    _consecutive["no_detect_count"] += 1
    if _consecutive["no_detect_count"] >= _STALE_THRESHOLD:
        _consecutive["confirmed_label"] = None
        _consecutive["confirmed_conf"]  = 0.0
        _consecutive["confirmed_class"] = None

def _advance_streak(class_index: int, conf: float) -> None:

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

def predict_traffic_sign(image_path: str) -> str:

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

    if not load_traffic_model() or frame is None:
        return None, None, None, None

    try:

        crop, bbox = _best_yolo_crop(frame)
        _consecutive["latest_bbox"] = bbox

        if crop is None:
            h, w = frame.shape[:2]
            crop = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
            _consecutive["latest_bbox"] = None

        if not _is_sign_like(crop):
            _record_no_detection()
            return _current_confirmed()

        class_index, confidence = _run_keras_on_crop(crop)

        if confidence < FRAME_CONF_THRESHOLD:
            _record_no_detection()
            return _current_confirmed()

        _consecutive["no_detect_count"] = 0
        _advance_streak(class_index, confidence)
        return _current_confirmed()

    except Exception:

        return None, None, None, None
