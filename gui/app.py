import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
from ml_module.train import train_model

df = None

def load_dataset():
    global df
    file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not file:
        return

    df = pd.read_csv(file)
    columns['values'] = list(df.columns)
    messagebox.showinfo("Dataset Loaded", f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

def run_training():
    global df
    if df is None:
        messagebox.showerror("Error", "Load dataset first")
        return

    target = columns.get()
    algo = algorithm.get()

    if target == "":
        messagebox.showerror("Error", "Select target column")
        return

    acc = train_model(df, target, algo)
    messagebox.showinfo("Result", f"Accuracy: {acc:.2f}")

def launch_app():
    root = tk.Tk()
    root.title("Autonomous Driving Perception Toolkit")
    root.geometry("900x600")

    title = tk.Label(root, text="ML Module (WEKA-like)", font=("Arial", 16))
    title.pack(pady=10)

    tk.Button(root, text="Load CSV Dataset", command=load_dataset).pack(pady=5)

    global columns
    columns = ttk.Combobox(root, width=40)
    columns.pack(pady=5)

    global algorithm
    algorithm = ttk.Combobox(root, values=["Decision Tree", "Naive Bayes", "SVM"], width=40)
    algorithm.pack(pady=5)

    tk.Button(root, text="Train Model", command=run_training).pack(pady=10)

    root.mainloop()
