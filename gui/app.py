import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import ttkbootstrap as tb  # Modern UI Library
from ttkbootstrap.constants import *
import pandas as pd
import cv2
import os

# --- IMPORT MODULES ---
from ml_module.train import train_model
from dl_module.traffic_sign.predict import predict_traffic_sign
from dl_module.lane_detection import detect_lanes_image as run_basic_lane
from dl_module.pipeline import run_advanced_pipeline
from dl_module.pedestrian_detection import detect_pedestrians

# --- GLOBAL VARIABLES ---
df = None
ml_columns_ui = None
ml_algo_ui = None

# ==========================================
# 📊 MODULE 1: CLASSICAL ML
# ==========================================
def load_dataset():
    global df
    file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file: return
    try:
        df = pd.read_csv(file)
        ml_columns_ui['values'] = list(df.columns)
        messagebox.showinfo("Success", f"Loaded {len(df)} rows.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def run_training():
    if df is None:
        messagebox.showerror("Error", "Load a dataset first!")
        return
    target = ml_columns_ui.get()
    algo = ml_algo_ui.get()
    if not target:
        messagebox.showerror("Error", "Select a target column.")
        return
    try:
        acc = train_model(df, target, algo)
        messagebox.showinfo("Result", f"Algorithm: {algo}\nAccuracy: {acc:.2f}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# ==========================================
# 👁️ MODULE 2: PERCEPTION STUDIO
# ==========================================
def run_traffic_sign():
    file = filedialog.askopenfilename(title="Select Sign Image", filetypes=[("Images", "*.jpg;*.png")])
    if not file: return
    try:
        result = predict_traffic_sign(file)
        messagebox.showinfo("Traffic Sign", f"Detected:\n{result}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def run_full_pipeline():
    file = filedialog.askopenfilename(title="Select Drive Image", filetypes=[("Images", "*.jpg;*.png")])
    if not file: return
    
    try:
        loading_popup = tb.Toplevel()
        loading_popup.title("Processing")
        loading_popup.geometry("300x100")
        tb.Label(loading_popup, text="Running AI Models...\nPlease Wait...", font=("Helvetica", 12)).pack(expand=True)
        loading_popup.update() # Force it to show immediately

        # 2. Run the heavy code
        result_img, status = run_advanced_pipeline(file)
        
        # 3. Close loading popup
        loading_popup.destroy()
        
        # 4. Display Result
        cv2.imshow(f"Autonomous Vision System - {status}", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("Error", str(e))

def run_pedestrian_detection():
    file = filedialog.askopenfilename(title="Select Street Image", filetypes=[("Images", "*.jpg;*.png")])
    if not file: return
    try:
        result_img, status = detect_pedestrians(file)
        if result_img is None:
            messagebox.showerror("Error", status)
            return
        
        messagebox.showinfo("YOLOv8 Report", status)
        cv2.imshow(f"Pedestrian Detection - {status}", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# ==========================================
# 🚀 MAIN DASHBOARD LAUNCHER
# ==========================================
def launch_app():
    # THEME: 'cyborg' is a cool dark theme. 
    # Try 'solar', 'superhero', or 'darkly' if you prefer others.
    root = tb.Window(themename="cyborg")
    root.title("AutoDrive AI - Perception Toolkit")
    root.geometry("1100x750")

    # --- HEADER ---
    header_frame = tb.Frame(root, bootstyle="dark")
    header_frame.pack(fill="x", pady=20)
    tb.Label(header_frame, text="🚗 Autonomous Driving Perception System", 
             font=("Helvetica", 24, "bold"), bootstyle="inverse-dark").pack()

    # --- TABS ---
    notebook = tb.Notebook(root, bootstyle="primary")
    notebook.pack(pady=20, padx=20, expand=True, fill="both")

    # Tab 1: Vision (The Cool Stuff)
    tab_vision = tb.Frame(notebook)
    notebook.add(tab_vision, text="👁️ Perception Studio")

    # Tab 2: Classical ML (The Data Stuff)
    tab_ml = tb.Frame(notebook)
    notebook.add(tab_ml, text="📊 Data Mining (ML)")

    # --- PERCEPTION TAB CONTENT ---
    vision_container = tb.Frame(tab_vision)
    vision_container.pack(expand=True)

    tb.Label(vision_container, text="Select a Vision Module", font=("Helvetica", 16)).pack(pady=30)

    # Grid for Buttons
    btn_grid = tb.Frame(vision_container)
    btn_grid.pack()

    # Button 1: Traffic Sign
    btn_sign = tb.Button(btn_grid, text="🛑 Traffic Sign Recognition", command=run_traffic_sign, 
                         bootstyle="danger-outline", width=30)
    btn_sign.grid(row=0, column=0, padx=20, pady=20, ipady=10)

    # Button 2: Lane Detection
    # btn_lane = tb.Button(btn_grid, text="🛣️ Lane Detection", command=run_lane_detection, 
    #                      bootstyle="success-outline", width=30)
    # btn_lane.grid(row=0, column=1, padx=20, pady=20, ipady=10)

    # Button 3: Pedestrian (New!)
    btn_ped = tb.Button(btn_grid, text="🚶 Pedestrian Detection (YOLO)", command=run_pedestrian_detection, 
                        bootstyle="warning-outline", width=30)
    btn_ped.grid(row=1, column=0, columnspan=2, pady=20, ipady=10)
    
    # Button 4: FULL PIPELINE (The "Wow" Factor)
    btn_full = tb.Button(btn_grid, text="🚀 Full Autonomous Pipeline", command=run_full_pipeline, 
                        bootstyle="primary", width=62)
    btn_full.grid(row=2, column=0, columnspan=2, pady=20, ipady=15)

    # --- ML TAB CONTENT ---
    ml_container = tb.Frame(tab_ml)
    ml_container.pack(expand=True)

    tb.Label(ml_container, text="Structured Data Analysis", font=("Helvetica", 16)).pack(pady=20)

    tb.Button(ml_container, text="📂 Load CSV Dataset", command=load_dataset, bootstyle="secondary").pack(pady=10)
    
    tb.Label(ml_container, text="Target Column:", font=("Helvetica", 10)).pack(pady=5)
    global ml_columns_ui
    ml_columns_ui = tb.Combobox(ml_container, width=30, bootstyle="info")
    ml_columns_ui.pack(pady=5)

    tb.Label(ml_container, text="Algorithm:", font=("Helvetica", 10)).pack(pady=5)
    global ml_algo_ui
    ml_algo_ui = tb.Combobox(ml_container, values=["Decision Tree", "Naive Bayes", "SVM"], width=30, bootstyle="info")
    ml_algo_ui.current(0)
    ml_algo_ui.pack(pady=5)

    tb.Button(ml_container, text="🚀 Train Model", command=run_training, bootstyle="success", width=20).pack(pady=30)

    # Footer
    tb.Label(root, text="v1.0 | Ready for Viva", font=("Arial", 8), bootstyle="secondary").pack(side="bottom", pady=10)

    root.mainloop()