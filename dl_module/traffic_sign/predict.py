import cv2
import numpy as np
import tensorflow as tf
import os
import csv
from collections import deque

MODEL_PATH = os.path.join("models", "traffic_classifier.h5")
model = None
SIGN_CONF_THRESHOLD = 0.8
_sign_history = deque(maxlen=5)
_roi_index = None

CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)', 
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)', 
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)', 
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection', 
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 
    16: 'Veh > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution', 
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve', 
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right', 
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing', 
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing', 
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory', 
    41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}

def load_traffic_model():
    global model
    if model is None:
        try:
            # compile=False avoids loading training state and is faster for inference-only use.
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ Traffic Sign Model Loaded!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    return True


def _predict_patch(patch):
    img = cv2.resize(patch, (30, 30))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img, verbose=0)
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions))
    return class_index, confidence


def _build_roi_index():
    index = {}
    csv_paths = [
        os.path.join("data", "Test.csv"),
        os.path.join("data", "Train.csv"),
    ]

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            continue

        try:
            with open(csv_path, "r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
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


def _roi_from_metadata(image_path):
    global _roi_index
    if _roi_index is None:
        _roi_index = _build_roi_index()

    key = os.path.basename(image_path).lower()
    return _roi_index.get(key)

def predict_traffic_sign(image_path):
    if not load_traffic_model():
        return "Model Failed to Load"

    # 1. Read Image
    img = cv2.imread(image_path)
    if img is None:
        return "Could not read image"

    h, w, _ = img.shape
    candidates = [img]

    center = img[h//4:3*h//4, w//4:3*w//4]
    if center is not None and center.size > 0:
        candidates.append(center)

    roi = _roi_from_metadata(image_path)
    if roi is not None:
        x1, y1, x2, y2 = roi
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        roi_crop = img[y1:y2, x1:x2]
        if roi_crop is not None and roi_crop.size > 0:
            candidates.append(roi_crop)

    class_index = None
    confidence = -1.0
    for patch in candidates:
        idx, conf = _predict_patch(patch)
        if conf > confidence:
            class_index = idx
            confidence = conf

    if confidence < SIGN_CONF_THRESHOLD:
        return "No confident sign detected"
    
    result_text = CLASSES.get(class_index, "Unknown")
    return f"{result_text} ({confidence*100:.1f}%)"


def predict_traffic_sign_frame(frame):
    if not load_traffic_model():
        return None, None, None

    if frame is None:
        return None, None, None

    try:
        h, w, _ = frame.shape
        crop = frame[h//4:3*h//4, w//4:3*w//4]

        img = cv2.resize(crop, (30, 30))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img, verbose=0)
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        if confidence < SIGN_CONF_THRESHOLD:
            return None, None, None

        label = CLASSES.get(class_index, "Unknown")
        _sign_history.append((label, confidence, class_index))

        # Smooth label jitter in videos using short temporal history.
        counts = {}
        for lbl, _, _ in _sign_history:
            counts[lbl] = counts.get(lbl, 0) + 1
        smoothed_label = max(counts, key=counts.get)

        smoothed_candidates = [entry for entry in _sign_history if entry[0] == smoothed_label]
        avg_conf = sum(entry[1] for entry in smoothed_candidates) / max(1, len(smoothed_candidates))
        class_idx = smoothed_candidates[-1][2] if smoothed_candidates else class_index

        return smoothed_label, avg_conf, class_idx

    except Exception:
        return None, None, None