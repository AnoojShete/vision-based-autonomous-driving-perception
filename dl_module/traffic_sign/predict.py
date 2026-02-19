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
            model = tf.keras.models.load_model(MODEL_PATH)
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
    
    # 2. Preprocess
    img = cv2.resize(img, (32, 32))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # 3. Predict & DEBUG
    predictions = model.predict(img)
    
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]
    
    print(f"\n🔍 DEBUG INFO for {os.path.basename(image_path)}:")
    for i in top_3_indices:
        class_name = CLASSES.get(i, "Unknown")
        confidence = predictions[0][i] * 100
        print(f"   - {class_name} (Class {i}): {confidence:.2f}%")
    print("--------------------------------------------------\n")

    best_index = top_3_indices[0]
    result_text = CLASSES.get(best_index, "Unknown")
    confidence = predictions[0][best_index]
    
    return f"{result_text} ({confidence*100:.1f}%)"