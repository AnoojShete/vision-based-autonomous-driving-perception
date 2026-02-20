import cv2
import numpy as np
from ultralytics import YOLO

# Load the model once. 'yolov8n.pt' is the "Nano" version (fastest).
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Warning: YOLO model failed to load. {e}")
    model = None

def detect_pedestrians(image_path):
    """
    Detects pedestrians using YOLOv8. Returns annotated image.
    """
    if model is None:
        return None, "Error: Model not loaded."

    # 1. Load Image
    image = cv2.imread(image_path)
    if image is None:
        return None, "Error: Could not read image."

    # 2. Predict
    # classes=0 -> Only detect 'Person'
    # conf=0.4 -> Only accept 40%+ confidence
    results = model.predict(source=image, classes=0, conf=0.4, verbose=False)
    
    # 3. Annotate
    result = results[0]
    boxes = result.boxes
    annotated_img = image.copy()
    
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Draw Red Box (BGR: 0, 0, 255)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Add Label
        label = f"Pedestrian: {float(box.conf[0]):.2f}"
        cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    count = len(boxes)
    status = f"Found {count} Pedestrian(s)" if count > 0 else "No pedestrians found."
    
    return annotated_img, status