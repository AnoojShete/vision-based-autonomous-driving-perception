import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import os

from ml_module.train import train_model
from dl_module.traffic_sign.predict import predict_traffic_sign

df = None
ml_columns_ui = None
ml_algo_ui = None

# MODULE 1: ML
def load_dataset():
    global df
    file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file:
        return
    
    try:
        df = pd.read_csv(file)
        ml_columns_ui['values'] = list(df.columns)
        messagebox.showinfo("Dataset Loaded", f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        messagebox.showerror("Error", f"Could not load CSV: {e}")

def run_training():
    global df
    if df is None:
        messagebox.showerror("Error", "Please load a dataset first.")
        return

    target = ml_columns_ui.get()
    algo = ml_algo_ui.get()

    if target == "":
        messagebox.showerror("Error", "Please select a target column.")
        return

    try:
        acc = train_model(df, target, algo)
        messagebox.showinfo("Training Result", f"Algorithm: {algo}\nAccuracy: {acc:.2f}")
    except Exception as e:
        messagebox.showerror("Training Error", str(e))

# MODULE 2: PERCEPTION STUDIO (DEEP LEARNING)
def run_traffic_sign():
    file = filedialog.askopenfilename(
        title="Select Traffic Sign Image",
        filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not file:
        return
    try:
        result = predict_traffic_sign(file)
        messagebox.showinfo("Traffic Sign Recognition", f"🚗 The AI sees:\n\n{result}")
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Failed to classify image.\nError: {str(e)}")

def run_lane_detection():
    messagebox.showinfo("Perception Module", "Lane Detection is currently under construction (Phase 3).")

def run_pedestrian_detection():
    messagebox.showinfo("Perception Module", "Pedestrian Detection is currently under construction (Phase 4).")

# MAIN GUI LAUNCHER
def launch_app():
    root = tk.Tk()
    root.title("Autonomous Driving Perception Toolkit")
    root.geometry("1000x700")

    # --- TABS ---
    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, expand=True, fill="both")

    # Tab 1: Classical ML
    tab_ml = ttk.Frame(notebook)
    notebook.add(tab_ml, text="📊 Classical ML (WEKA-like)")

    # Tab 2: Vision Perception
    tab_dl = ttk.Frame(notebook)
    notebook.add(tab_dl, text="👁️ Perception Studio (Deep Learning)")

    # --- TAB 1 CONTENT (ML) ---
    tk.Label(tab_ml, text="Structured Data Mining", font=("Arial", 16, "bold")).pack(pady=20)
    
    tk.Button(tab_ml, text="📂 Load CSV Dataset", command=load_dataset, width=30, height=2, bg="#e1e1e1").pack(pady=10)
    
    tk.Label(tab_ml, text="Select Target Column:", font=("Arial", 10)).pack(pady=5)
    global ml_columns_ui
    ml_columns_ui = ttk.Combobox(tab_ml, width=40)
    ml_columns_ui.pack(pady=5)

    tk.Label(tab_ml, text="Select Algorithm:", font=("Arial", 10)).pack(pady=5)
    global ml_algo_ui
    ml_algo_ui = ttk.Combobox(tab_ml, values=["Decision Tree", "Naive Bayes", "SVM"], width=40)
    ml_algo_ui.current(0)
    ml_algo_ui.pack(pady=5)

    tk.Button(tab_ml, text="🚀 Train Model", command=run_training, bg="#4CAF50", fg="white", width=30, height=2).pack(pady=30)

    # --- TAB 2 CONTENT (DL) ---
    tk.Label(tab_dl, text="Autonomous Vehicle Perception Layer", font=("Arial", 16, "bold")).pack(pady=20)
    tk.Label(tab_dl, text="Select a perception task to simulate vehicle sensors:", font=("Arial", 11)).pack(pady=10)

    btn_frame = tk.Frame(tab_dl)
    btn_frame.pack(pady=30)

    # Button 1: Traffic Sign (NOW ACTIVE!)
    tk.Button(btn_frame, text="🛑 Traffic Sign Recognition", command=run_traffic_sign, width=30, height=3, bg="#FF5722", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=20, pady=20)
    
    # Button 2: Lane Detection (Placeholder)
    tk.Button(btn_frame, text="🛣️ Lane Detection", command=run_lane_detection, width=30, height=3, bg="#2196F3", fg="white", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=20, pady=20)
    
    # Button 3: Pedestrian Detection (Placeholder)
    tk.Button(btn_frame, text="🚶 Pedestrian Detection", command=run_pedestrian_detection, width=30, height=3, bg="#FFC107", fg="black", font=("Arial", 10, "bold")).grid(row=1, column=0, columnspan=2, pady=20)

    root.mainloop()