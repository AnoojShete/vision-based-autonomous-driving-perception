import cv2
import numpy as np
from ultralytics import YOLO

try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Warning: YOLO model failed to load. {e}")
    model = None

_OBSTACLE_CLASSES = [0, 2, 3, 5, 7]

_CLASS_LABELS = {
    0: "Ped",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

def detect_pedestrians(image_path):

    if model is None:
        return None, "Error: Model not loaded."

    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Could not read image."

    results = model.predict(source=image, classes=_OBSTACLE_CLASSES, conf=0.4, verbose=False)

    result = results[0]
    boxes = result.boxes
    annotated_img = image.copy()

    for box in boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        label_name = _CLASS_LABELS.get(cls_id, "Object")

        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"{label_name}: {float(box.conf[0]):.2f}"
        cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    count = len(boxes)
    status = f"Found {count} obstacle(s)" if count > 0 else "No obstacles found."

    return annotated_img, status

def detect_pedestrians_frame(frame, draw=True):

    if model is None:
        return frame, "Error: Model not loaded."
    if frame is None:
        return frame, "Error: Empty frame."

    try:
        results = model.predict(source=frame, classes=_OBSTACLE_CLASSES, conf=0.4, verbose=False)
        result = results[0]
        boxes = result.boxes
        annotated = frame.copy()

        if draw:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label_name = _CLASS_LABELS.get(cls_id, "Object")
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{label_name}: {float(box.conf[0]):.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(14, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        count = len(boxes)
        status = f"Found {count} obstacle(s)" if count > 0 else "No obstacles found."
        return annotated, status
    except Exception as exc:
        return frame, f"Error: {exc}"

def detect_pedestrians_data(frame):
    if model is None:
        return []

    results = model.predict(source=frame, classes=_OBSTACLE_CLASSES, conf=0.3, verbose=False)
    boxes = results[0].boxes

    output = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = _CLASS_LABELS.get(cls_id, "Object")

        output.append((x1, y1, x2, y2, conf, label))

    return output
