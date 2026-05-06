import cv2
import numpy as np
from ultralytics import YOLO

_model: YOLO | None = None

def _get_model() -> YOLO:

    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model

_RED_LOW1 = np.array([0, 100, 100])
_RED_HI1  = np.array([10, 255, 255])
_RED_LOW2 = np.array([170, 100, 100])
_RED_HI2  = np.array([180, 255, 255])

_YEL_LOW = np.array([15, 100, 100])
_YEL_HI  = np.array([35, 255, 255])

_GRN_LOW = np.array([40, 80, 80])
_GRN_HI  = np.array([90, 255, 255])

def _classify_state(crop: np.ndarray) -> str:

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, _RED_LOW1, _RED_HI1) |               cv2.inRange(hsv, _RED_LOW2, _RED_HI2)
    yel_mask = cv2.inRange(hsv, _YEL_LOW, _YEL_HI)
    grn_mask = cv2.inRange(hsv, _GRN_LOW, _GRN_HI)

    counts = {
        "Red":    int(cv2.countNonZero(red_mask)),
        "Yellow": int(cv2.countNonZero(yel_mask)),
        "Green":  int(cv2.countNonZero(grn_mask)),
    }

    best = max(counts, key=counts.get)

    if counts[best] == 0:
        return "Unknown"
    return best

def detect_traffic_lights(frame: np.ndarray):

    model = _get_model()
    results = model.predict(source=frame, classes=[9], conf=0.3, verbose=False)

    detections: list[tuple] = []
    boxes = results[0].boxes

    h, w = frame.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        state = _classify_state(crop)
        detections.append((x1, y1, x2, y2, state, conf))

    return detections
