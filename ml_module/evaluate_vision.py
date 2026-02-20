import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

MODEL_PATH = os.path.join("models", "traffic_classifier.h5")
TEST_CSV_PATH = os.path.join("data", "Test.csv")
DATA_ROOT = "data"

def evaluate_traffic_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    print(f"⏳ Loading Model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded!")

    if not os.path.exists(TEST_CSV_PATH):
        print(f"❌ Error: Answer key (CSV) not found at {TEST_CSV_PATH}")
        print("   Please find 'Test.csv' and place it in the 'data' folder.")
        return
    
    df = pd.read_csv(TEST_CSV_PATH)
    
    valid_rows = []
    print(f"🔍 Found {len(df)} entries in CSV. Verifying files...")
    
    predictions = []
    actuals = []
    
    for index, row in df.iterrows():
        relative_path = row['Path'] 
        full_path = os.path.join(DATA_ROOT, relative_path)
        class_id = row['ClassId']
        
        if not os.path.exists(full_path):
            continue

        try:
            img = cv2.imread(full_path)
            if img is None: 
                continue
                
            IMG_HEIGHT = 30
            IMG_WIDTH = 30
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            
            img = img / 255.0
            
            img = np.expand_dims(img, axis=0)
            
            pred_probs = model.predict(img, verbose=0)
            pred_class = np.argmax(pred_probs)
            
            predictions.append(pred_class)
            actuals.append(class_id)
            
            if len(predictions) % 1000 == 0:
                print(f"   Tested {len(predictions)} images...")

        except Exception as e:
            print(f"Error on {full_path}: {e}")

    if len(predictions) == 0:
        print("❌ No images were tested. Check your paths!")
        return

    accuracy = accuracy_score(actuals, predictions) * 100
    print("\n" + "="*40)
    print(f"📊 AUTOMATED TEST RESULTS")
    print("="*40)
    print(f"Total Images Tested: {len(predictions)}")
    print(f"✅ Final Accuracy:   {accuracy:.2f}%")
    print("="*40)

if __name__ == "__main__":
    evaluate_traffic_model()