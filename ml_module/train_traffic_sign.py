import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

IMG_HEIGHT = 30
IMG_WIDTH = 30
CHANNELS = 3

class GuiLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_func):
        super().__init__()
        self.log_func = log_func

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        self.log_func(f"   Epoch {epoch+1}: Acc={acc:.2f} | Val_Acc={val_acc:.2f}")

def load_and_preprocess(data_path, limit_per_class=None, log_func=print):
    log_func(f"1️⃣ [LOAD] Loading from: {data_path}")
    log_func(f"   (Limit: {limit_per_class if limit_per_class else 'All'} per class)")
    
    images = []
    labels = []
    classes = 43
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Path not found: {data_path}")

    for i in range(classes):
        path = os.path.join(data_path, str(i))
        if not os.path.exists(path):
            continue
            
        images_list = os.listdir(path)
        
        if limit_per_class:
            images_list = images_list[:limit_per_class]

        for img_name in images_list:
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(image)
                labels.append(i)
            except:
                pass
    
    log_func(f"   - Loaded {len(images)} images total.")
    
    if len(images) == 0:
        raise ValueError("No images loaded! Check folder structure (must contain 0, 1, 2... folders).")

    images = np.array(images)
    labels = np.array(labels)
    
    s = np.arange(images.shape[0])
    np.random.seed(42)
    np.random.shuffle(s)
    images = images[s]
    labels = labels[s]
    
    images = images / 255.0
    labels = to_categorical(labels, classes)
    
    return images, labels

def build_model(input_shape, classes):
    model = Sequential()
    model.add(Conv2D(32, (5,5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def start_training(data_path, limit=50, epochs=5, log_func=print):
    try:
        log_func(f"🚀 Starting Training Config: {epochs} Epochs, {limit} Img/Class")
        
        X, y = load_and_preprocess(data_path, limit, log_func)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        log_func("2️⃣ [MODEL] Constructing CNN Architecture...")
        model = build_model(X_train.shape[1:], 43)
        
        log_func(f"3️⃣ [TRAIN] Running {epochs} Epochs on {len(X_train)} samples...")
        history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, 
                            validation_data=(X_val, y_val),
                            callbacks=[GuiLogger(log_func)], verbose=0)
        
        if not os.path.exists("models"): os.makedirs("models")
        model.save("models/traffic_demo.h5")
        log_func("✅ Model Saved to 'models/traffic_demo.h5'")
        
        if not os.path.exists("Public"): os.makedirs("Public")
        
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Accuracy (Live Run)')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss (Live Run)')
        plt.xlabel('Epochs')
        plt.legend()
        
        save_path = "Public/training_demo_graph.png"
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        log_func(f"📊 Graph saved to {save_path}")
        return save_path

    except Exception as e:
        log_func(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None