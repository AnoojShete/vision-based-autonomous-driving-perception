import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

# --- CONFIGURATION ---
DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "Train")
EPOCHS = 15
IMG_HEIGHT = 30
IMG_WIDTH = 30
CHANNELS = 3

def load_and_preprocess():
    print("1️⃣ [LOAD] Loading Dataset from 'data/Train'...")
    images = []
    labels = []
    classes = 43
    
    # Iterate through all 43 folders (0 to 42)
    for i in range(classes):
        path = os.path.join(TRAIN_PATH, str(i))
        images_list = os.listdir(path)
        
        # Load every image in the folder
        for img_name in images_list:
            try:
                # 1. LOAD
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                
                # 2. PREPROCESS (Resize)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(image)
                labels.append(i)
            except:
                pass
        print(f"   - Loaded Class {i}...")

    # 3. PREPROCESS (Normalize & Encoding)
    print("2️⃣ [PREPROCESS] Normalizing and Encoding...")
    images = np.array(images)
    labels = np.array(labels)
    
    # Shuffle indices
    s = np.arange(images.shape[0])
    np.random.seed(42)
    np.random.shuffle(s)
    images = images[s]
    labels = labels[s]
    
    # Normalize pixel values (0-255 -> 0-1)
    images = images / 255.0
    
    # One-Hot Encoding (e.g., Class 3 -> [0,0,0,1,0...])
    labels = to_categorical(labels, classes)
    
    return images, labels

def build_model(input_shape, classes):
    print("3️⃣ [ML MODEL] Building CNN Architecture...")
    model = Sequential()
    
    # Layer 1: Convolution
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Layer 2: Convolution
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Layer 3: Dense (Classification)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax')) # Output Layer
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def plot_history(history):
    print("5️⃣ [GRAPHS] Generating Training Graphs...")
    plt.figure(figsize=(12, 5))
    
    # Accuracy Graph
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss Graph
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Check if Public exists, if not create it
    if not os.path.exists("Public"): os.makedirs("Public")
    plt.savefig('Public/training_graphs.png')

def main():
    # Step 1 & 2: Load & Preprocess
    X, y = load_and_preprocess()
    
    # Split Data (80% Train, 20% Val)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Build Model
    model = build_model(X_train.shape[1:], 43)
    
    # Step 4: Train
    print(f"4️⃣ [TRAIN] Training for {EPOCHS} Epochs...")
    history = model.fit(X_train, y_train, batch_size=32, epochs=EPOCHS, validation_data=(X_val, y_val))
    
    # Save Model
    if not os.path.exists("models"): os.makedirs("models")
    model.save("models/traffic_classifier.h5")
    print("✅ Model Saved to 'models/traffic_classifier.h5'")
    
    # Step 5: Graphs
    plot_history(history)

if __name__ == "__main__":
    main()