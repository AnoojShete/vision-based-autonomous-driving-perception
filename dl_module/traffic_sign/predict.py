import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = os.path.join("models", "traffic_classifier.h5")
model = None

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

def predict_traffic_sign(image_path):
    if not load_traffic_model():
        return "Model Failed to Load"

    # 1. Read Image
    img = cv2.imread(image_path)
    if img is None:
        return "Could not read image"
    
    img = cv2.resize(img, (30, 30))
    
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    # Normalize pixel values to 0-1
    img = img / 255.0
    
    # 3. Reshape for Model (1 image, 30, 30, 3)
    img = np.expand_dims(img, axis=0)
    
    # 4. Predict
    predictions = model.predict(img, verbose=0)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    
    result_text = CLASSES.get(class_index, "Unknown")
    return f"{result_text} ({confidence*100:.1f}%)"


def predict_traffic_sign_frame(frame):
    """
    Frame-based traffic-sign prediction.
    Returns (label, confidence, class_index).
    """
    if not load_traffic_model():
        return None, None, None
    if frame is None:
        return None, None, None

    try:
        img = cv2.resize(frame, (30, 30))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img, verbose=0)
        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        label = CLASSES.get(class_index, "Unknown")
        return label, confidence, class_index
    except Exception:
        return None, None, None