import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

# --- CONFIGURATION ---
MODEL_PATH = os.path.join("models", "traffic_classifier.h5")
TEST_CSV = os.path.join("data", "Test.csv")
DATA_DIR = "data"

def generate_visuals():
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print("❌ Model not found. Train it first!")
        return
    
    print("⏳ Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Load Test Data
    print("⏳ Loading Test Data...")
    df = pd.read_csv(TEST_CSV)
    
    X_test = []
    y_true = []
    
    # We will test on a subset (e.g., 1000 images) to be fast, 
    # or remove .head() to test ALL 12,000 images (takes longer)
    for index, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, row['Path'])
        class_id = row['ClassId']
        
        if os.path.exists(img_path):
            try:
                img = cv2.imread(img_path)
                img = cv2.resize(img, (30, 30)) # Match training size
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Keep BGR if trained on BGR
                img = img / 255.0
                X_test.append(img)
                y_true.append(class_id)
            except:
                pass

    X_test = np.array(X_test)
    y_true = np.array(y_true)
    
    # 3. Predict
    print("⏳ Running Predictions...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 4. Generate Confusion Matrix
    print("📊 Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.title('Traffic Sign Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.savefig('confusion_matrix.png') 
    print("✅ Saved 'confusion_matrix.png'")
    
    # 5. Generate Text Report
    print("📝 Generating Classification Report...")
    report = classification_report(y_true, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    print("✅ Saved 'classification_report.txt'")

if __name__ == "__main__":
    generate_visuals()