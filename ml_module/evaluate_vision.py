import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Paths
MODEL_PATH = os.path.join("models", "traffic_classifier.h5")
TEST_CSV_PATH = os.path.join("data", "Test.csv")
TEST_IMG_DIR = os.path.join("data")

def evaluate_traffic_model():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found!")
        return
    
    print("⏳ Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Load Answer Key
    if not os.path.exists(TEST_CSV_PATH):
        print(f"❌ Test.csv not found at {TEST_CSV_PATH}")
        return
    
    df = pd.read_csv(TEST_CSV_PATH)
    print(f"✅ Loaded Test Data: {len(df)} images to check.")

    predictions = []
    actuals = []
    
    print("🚀 Starting Batch Evaluation (This might take a moment)...")
    
    for index, row in df.iterrows():
        img_path = os.path.join(TEST_IMG_DIR, row['Path'])
        class_id = row['ClassId']
        
        if not os.path.exists(img_path):
            continue
            
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            pred_probs = model.predict(img, verbose=0)
            pred_class = np.argmax(pred_probs)
            
            predictions.append(pred_class)
            actuals.append(class_id)
            
            if index % 1000 == 0 and index > 0:
                print(f"   Checked {index} images...")
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    acc = accuracy_score(actuals, predictions) * 100
    print("\n" + "="*30)
    print(f"📊 FINAL EVALUATION REPORT")
    print("="*30)
    print(f"Total Images Tested: {len(actuals)}")
    print(f"✅ Classification Accuracy: {acc:.2f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate_traffic_model()