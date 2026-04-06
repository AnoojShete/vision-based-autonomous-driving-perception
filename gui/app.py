import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import threading
import datetime
import os

# ─────────────────────────────────────────
# BACKEND IMPORTS
# ─────────────────────────────────────────
try:
    import pandas as pd
    PANDAS_OK = True
except ImportError:
    PANDAS_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except ImportError:
    PIL_OK = False

try:
    from ml_module.train import get_driving_dataset_summary, train_driving_model, train_model
    ML_OK = True
except ImportError:
    ML_OK = False
    def get_driving_dataset_summary(task="metadata", data_dir="data", max_samples=5000):
        raise NotImplementedError("ml_module not available")
    def train_driving_model(task="metadata", algorithm="Decision Tree", data_dir="data", test_size=0.2, random_state=42, max_samples=5000):
        raise NotImplementedError("ml_module not available")
    def train_model(df, target, algo):
        raise NotImplementedError("ml_module not available")

try:
    from dl_module.traffic_sign.predict import predict_traffic_sign
    from dl_module.lane_detection import detect_lanes_image as run_basic_lane
    from dl_module.pipeline import run_advanced_pipeline
    from dl_module.pedestrian_detection import detect_pedestrians
    from ml_module.train_traffic_sign import start_training as train_cnn_demo
    DL_OK = True
except ImportError:
    DL_OK = False
    def predict_traffic_sign(path): raise NotImplementedError("dl_module not available")
    def run_basic_lane(path): raise NotImplementedError("dl_module not available")
    def run_advanced_pipeline(path): raise NotImplementedError("dl_module not available")
    def detect_pedestrians(path): raise NotImplementedError("dl_module not available")
    def train_cnn_demo(data_path, limit, epochs, log_func): raise NotImplementedError("ml_module not available")


# ─────────────────────────────────────────
# COLOUR & STYLE CONSTANTS
# ─────────────────────────────────────────
SIDEBAR_BG   = "#0d0f14"
SIDEBAR_W    = 300
ACCENT       = "#00d4ff" 
ACCENT2      = "#ff4f5e"
ACCENT3      = "#f5c518" 
CONTENT_BG   = "#12151c"
CARD_BG      = "#1a1e2a"
LOG_BG       = "#0a0c10"
TEXT_PRIMARY = "#e8eaf0"
TEXT_MUTED   = "#5c6370"
NAV_ACTIVE   = "#1e2230"
NAV_HOVER    = "#161924"
FONT_TITLE   = ("Courier New", 13, "bold")
FONT_NAV     = ("Courier New", 11)
FONT_LABEL   = ("Courier New", 10)
FONT_LOG     = ("Courier New", 9)
FONT_HEADING = ("Courier New", 18, "bold")
FONT_SUB     = ("Courier New", 11)


class AutoDriveApp(tb.Window):

    def __init__(self):
        super().__init__(themename="cyborg")
        self.title("AutoDrive AI — Perception Toolkit")
        self.geometry("1400x850")
        self.minsize(960, 640)
        self.configure(bg=SIDEBAR_BG)

        self.df            = None
        self._adv_df       = None
        self._adv_loaded_name = ""
        self._driving_task = tk.StringVar(value="metadata")
        self._last_driving_result = None
        self._last_adv_result = None
        self._active_nav   = tk.StringVar(value="dashboard")

        if not os.path.exists("Public"):
            os.makedirs("Public")

        self._build_sidebar()
        self._build_main_area()

        self.show_view("dashboard")
        self.log("[INFO] AutoDrive AI started.")
        if not DL_OK:
            self.log("[WARN] dl_module not found — using placeholders.")
        if not ML_OK:
            self.log("[WARN] ml_module not found — using placeholders.")

    # ─────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────
    def _build_sidebar(self):
        sb = tk.Frame(self, bg=SIDEBAR_BG, width=SIDEBAR_W)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        logo_frame = tk.Frame(sb, bg=SIDEBAR_BG)
        logo_frame.pack(fill="x", padx=16, pady=(28, 8))

        tk.Label(logo_frame, text="⬡", font=("Courier New", 28, "bold"),
                 fg=ACCENT, bg=SIDEBAR_BG).pack(side="left")
        tk.Label(logo_frame, text=" AutoDrive\n AI",
                 font=FONT_TITLE, fg=TEXT_PRIMARY, bg=SIDEBAR_BG,
                 justify="left").pack(side="left", padx=6)

        tk.Frame(sb, bg="#1e2230", height=1).pack(fill="x", padx=16, pady=10)

        tk.Label(sb, text="NAVIGATION", font=("Courier New", 8),
                 fg=TEXT_MUTED, bg=SIDEBAR_BG).pack(anchor="w", padx=20, pady=(4, 8))

        nav_items = [
            ("dashboard",  "🏠  Dashboard"),
            ("vision",     "👁️  Vision Studio"),
            ("datalab",    "📊  Data Lab"),
        ]
        self._nav_buttons = {}
        for key, label in nav_items:
            btn = self._make_nav_btn(sb, key, label)
            self._nav_buttons[key] = btn

        tk.Frame(sb, bg=SIDEBAR_BG).pack(expand=True, fill="both")

        tk.Frame(sb, bg="#1e2230", height=1).pack(fill="x", padx=16, pady=6)
        self._status_dot = tk.Label(sb, text="● System Ready",
                                    font=("Courier New", 9), fg="#00ff88",
                                    bg=SIDEBAR_BG)
        self._status_dot.pack(anchor="w", padx=18, pady=(0, 20))

    def _make_nav_btn(self, parent, key, label):
        frame = tk.Frame(parent, bg=SIDEBAR_BG, cursor="hand2")
        frame.pack(fill="x", padx=10, pady=2)

        indicator = tk.Frame(frame, bg=SIDEBAR_BG, width=3)
        indicator.pack(side="left", fill="y")

        btn = tk.Label(frame, text=label, font=FONT_NAV,
                       fg=TEXT_MUTED, bg=SIDEBAR_BG,
                       anchor="w", padx=12, pady=10)
        btn.pack(side="left", fill="x", expand=True)

        def on_enter(e):
            if self._active_nav.get() != key:
                frame.config(bg=NAV_HOVER)
                btn.config(bg=NAV_HOVER)
                indicator.config(bg=NAV_HOVER)

        def on_leave(e):
            if self._active_nav.get() != key:
                frame.config(bg=SIDEBAR_BG)
                btn.config(bg=SIDEBAR_BG)
                indicator.config(bg=SIDEBAR_BG)

        def on_click(e):
            self.show_view(key)

        for widget in (frame, btn):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", on_click)

        frame._indicator = indicator
        frame._btn       = btn
        return frame

    def _set_active_nav(self, key):
        self._active_nav.set(key)
        for k, frame in self._nav_buttons.items():
            if k == key:
                frame.config(bg=NAV_ACTIVE)
                frame._btn.config(bg=NAV_ACTIVE, fg=ACCENT)
                frame._indicator.config(bg=ACCENT)
            else:
                frame.config(bg=SIDEBAR_BG)
                frame._btn.config(bg=SIDEBAR_BG, fg=TEXT_MUTED)
                frame._indicator.config(bg=SIDEBAR_BG)

    # ─────────────────────────────────────────
    # MAIN AREA
    # ─────────────────────────────────────────
    def _build_main_area(self):
        right = tk.Frame(self, bg=CONTENT_BG)
        right.pack(side="left", fill="both", expand=True)

        self._content_host = tk.Frame(right, bg=CONTENT_BG)
        self._content_host.pack(fill="both", expand=True, padx=0, pady=0)

        tk.Frame(right, bg="#1e2230", height=1).pack(fill="x")

        console_frame = tk.Frame(right, bg=LOG_BG, height=160)
        console_frame.pack(fill="x", side="bottom")
        console_frame.pack_propagate(False)

        hdr = tk.Frame(console_frame, bg=LOG_BG)
        hdr.pack(fill="x", padx=12, pady=(6, 0))
        tk.Label(hdr, text="▸ CONSOLE", font=("Courier New", 8, "bold"),
                 fg=ACCENT, bg=LOG_BG).pack(side="left")
        tk.Button(hdr, text="CLEAR", font=("Courier New", 7),
                  fg=TEXT_MUTED, bg=LOG_BG, bd=0, activebackground=LOG_BG,
                  cursor="hand2", command=self._clear_log).pack(side="right")

        self._log_text = tk.Text(console_frame, bg=LOG_BG, fg="#7ec8a0",
                                  font=FONT_LOG, bd=0, wrap="word",
                                  state="disabled", cursor="arrow",
                                  insertbackground=LOG_BG)
        scroll = tk.Scrollbar(console_frame, command=self._log_text.yview,
                               bg=LOG_BG, troughcolor=LOG_BG, bd=0)
        self._log_text.config(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=12, pady=6)

        self._views = {}
        self._views["dashboard"] = self._build_dashboard_view(self._content_host)
        self._views["vision"]    = self._build_vision_view(self._content_host)
        self._views["datalab"]   = self._build_datalab_view(self._content_host)

    def show_view(self, key):
        for k, frame in self._views.items():
            frame.pack_forget()
        self._views[key].pack(fill="both", expand=True)
        self._set_active_nav(key)

    def log(self, msg):
        ts  = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self._log_text.config(state="normal")
        self._log_text.insert("end", line)
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    # ─────────────────────────────────────────
    # ── VIEW: DASHBOARD ──────────────────────
    # ─────────────────────────────────────────
    def _build_dashboard_view(self, parent):
        view = tk.Frame(parent, bg=CONTENT_BG)

        tk.Label(view, text="🚗  Autonomous Driving\nPerception System",
                 font=("Courier New", 22, "bold"), fg=TEXT_PRIMARY,
                 bg=CONTENT_BG, justify="left").pack(anchor="w", padx=40, pady=(40, 4))
        tk.Label(view, text="Real-time vision, detection & classification — all in one toolkit.",
                 font=FONT_SUB, fg=TEXT_MUTED, bg=CONTENT_BG).pack(anchor="w", padx=42, pady=(0, 30))

        cards_row = tk.Frame(view, bg=CONTENT_BG)
        cards_row.pack(anchor="w", padx=40, pady=4)

        modules = [
            ("👁️  Vision Studio",    "DL / CV Pipeline",    ACCENT,  DL_OK),
            ("📊  Data Lab",          "Classical ML",         ACCENT3, ML_OK),
            ("🎯  YOLOv8",            "Object Detection",     ACCENT2, DL_OK),
            ("🛣️  Lane Detection",    "Geometric CV",         "#a78bfa", DL_OK),
        ]
        for title, sub, color, ok in modules:
            self._status_card(cards_row, title, sub, color, ok)

        tip = tk.Frame(view, bg=CARD_BG, pady=16)
        tip.pack(fill="x", padx=40, pady=30)
        tk.Label(tip, text="  ⚡  QUICK START",
                 font=("Courier New", 9, "bold"), fg=ACCENT, bg=CARD_BG).pack(anchor="w", padx=20)
        tk.Label(tip,
                 text=("  1. Navigate to 👁️ Vision Studio and load an image.\n"
                       "  2. Run the 🚀 Full Autonomous Pipeline for a complete perception pass.\n"
                       "  3. Or head to 📊 Data Lab to train a classical ML model on a CSV dataset."),
                 font=FONT_LOG, fg=TEXT_PRIMARY, bg=CARD_BG, justify="left").pack(anchor="w", padx=20, pady=6)
        return view

    def _status_card(self, parent, title, sub, color, ok):
        card = tk.Frame(parent, bg=CARD_BG, width=240, height=100)
        card.pack(side="left", padx=10, pady=4)
        card.pack_propagate(False)

        tk.Label(card, text=title, font=("Courier New", 10, "bold"),
                 fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(card, text=sub, font=("Courier New", 8),
                 fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w", padx=14)

        status_text  = "● READY" if ok else "● UNAVAILABLE"
        status_color = "#00ff88" if ok else ACCENT2
        tk.Label(card, text=status_text, font=("Courier New", 8),
                 fg=status_color, bg=CARD_BG).pack(anchor="w", padx=14, pady=(6, 0))

        tk.Frame(card, bg=color, height=3).pack(side="bottom", fill="x")

    # ─────────────────────────────────────────
    # ── VIEW: VISION STUDIO ──────────────────
    # ─────────────────────────────────────────
    def _build_vision_view(self, parent):
        view = tk.Frame(parent, bg=CONTENT_BG)

        topbar = tk.Frame(view, bg=CONTENT_BG)
        topbar.pack(fill="x", padx=30, pady=(28, 0))
        tk.Label(topbar, text="👁️  Vision Studio", font=FONT_HEADING,
                 fg=TEXT_PRIMARY, bg=CONTENT_BG).pack(side="left")

        body = tk.Frame(view, bg=CONTENT_BG)
        body.pack(fill="both", expand=True, padx=30, pady=14)

        img_area = tk.Frame(body, bg=CARD_BG, bd=0)
        img_area.pack(side="left", fill="both", expand=True, padx=(0, 16))

        tk.Label(img_area, text="▸ IMAGE DISPLAY", font=("Courier New", 8, "bold"),
                 fg=ACCENT, bg=CARD_BG).pack(anchor="w", padx=14, pady=(12, 0))
        tk.Frame(img_area, bg="#1e2230", height=1).pack(fill="x", padx=14, pady=6)

        self._img_label = tk.Label(img_area,
                                   text="No image loaded.\nResults will appear here.",
                                   font=FONT_SUB, fg=TEXT_MUTED, bg=CARD_BG,
                                   justify="center")
        self._img_label.pack(expand=True)

        action_panel = tk.Frame(body, bg=CONTENT_BG, width=300)
        action_panel.pack(side="right", fill="y", padx=(20, 0))
        action_panel.pack_propagate(False)

        tk.Label(action_panel, text="▸ MODULES", font=("Courier New", 8, "bold"),
                 fg=ACCENT, bg=CONTENT_BG).pack(anchor="w", pady=(4, 10))

        vision_btns = [
            ("🛑  Traffic Sign Recognition", ACCENT2,   self._cmd_traffic_sign),
            ("⚡  Train Model (Live Demo)",   "#e17055", self._cmd_train_cnn_demo),
            ("📈  View Training History",     "#2d3436", self._cmd_view_training_graph),
            ("🚶  Pedestrian Detection",      ACCENT3,   self._cmd_pedestrian),
            ("🛣️  Lane Detection (Basic)",    "#a78bfa", self._cmd_lane),
            ("🚀  Full Autonomous Pipeline",  ACCENT,    self._cmd_full_pipeline),
        ]
        for text, color, cmd in vision_btns:
            self._action_btn(action_panel, text, color, cmd)

        return view

    def _action_btn(self, parent, text, color, command):
        outer = tk.Frame(parent, bg=color, pady=1)
        outer.pack(fill="x", pady=6)

        btn = tk.Label(outer, text=text, font=("Courier New", 10),
                       fg=TEXT_PRIMARY, bg=CARD_BG,
                       anchor="w", padx=14, pady=11, cursor="hand2")
        btn.pack(fill="x")

        def on_enter(e): btn.config(bg=NAV_ACTIVE)
        def on_leave(e): btn.config(bg=CARD_BG)
        def on_click(e): command()

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.bind("<Button-1>", on_click)

    # ─────────────────────────────────────────
    # ── COMMANDS: VISION & TRAINING ──────────
    # ─────────────────────────────────────────
    def _cmd_traffic_sign(self):
        self.log("[INFO] Traffic Sign Recognition — select an image...")
        file = filedialog.askopenfilename(title="Select Sign Image", filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file: return
        self.log(f"[INFO] Processing: {file}")
        try:
            result = predict_traffic_sign(file)
            self.log(f"[OK]   Detected: {result}")
            messagebox.showinfo("Traffic Sign", f"Detected:\n{result}")
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", str(e))

    def _cmd_train_cnn_demo(self):
        popup = tk.Toplevel(self)
        popup.title("Training Configuration")
        popup.geometry("500x350")
        popup.configure(bg=CARD_BG)
        popup.resizable(False, False)

        path_var = tk.StringVar(value=os.path.join(os.getcwd(), "data", "Train"))
        epoch_var = tk.IntVar(value=5)
        limit_var = tk.IntVar(value=50)

        p_frm = tk.Frame(popup, bg=CARD_BG, padx=20, pady=20)
        p_frm.pack(fill="both", expand=True)

        tk.Label(p_frm, text="Live Training Setup", font=FONT_TITLE, fg=ACCENT, bg=CARD_BG).pack(anchor="w", pady=(0, 15))

        tk.Label(p_frm, text="Dataset Path (contains class folders):", font=FONT_LABEL, fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w")
        row1 = tk.Frame(p_frm, bg=CARD_BG)
        row1.pack(fill="x", pady=(2, 10))
        
        path_entry = tb.Entry(row1, textvariable=path_var, bootstyle="secondary")
        path_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        def browse():
            d = filedialog.askdirectory()
            if d: path_var.set(d)
        
        tb.Button(row1, text="Browse", bootstyle="outline-secondary", command=browse).pack(side="right")

        tk.Label(p_frm, text="Epochs (Recommended: 5-15):", font=FONT_LABEL, fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w")
        tb.Spinbox(p_frm, textvariable=epoch_var, from_=1, to=50, bootstyle="secondary").pack(fill="x", pady=(2, 10))

        tk.Label(p_frm, text="Images per Class (Limit for speed):", font=FONT_LABEL, fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w")
        tb.Spinbox(p_frm, textvariable=limit_var, from_=10, to=1000, bootstyle="secondary").pack(fill="x", pady=(2, 20))

        def start_thread():
            d_path = path_var.get()
            ep = epoch_var.get()
            lim = limit_var.get()
            popup.destroy()
            self.log("[INFO] Initializing Live Training Demo...")

            def run():
                graph_path = train_cnn_demo(data_path=d_path, limit=lim, epochs=ep, log_func=self.log)
                if graph_path:
                    self.after(0, lambda: self._show_demo_graph(graph_path))
            threading.Thread(target=run, daemon=True).start()

        tb.Button(p_frm, text="🚀 START TRAINING", bootstyle="success", command=start_thread).pack(fill="x", ipady=5)

    def _show_demo_graph(self, path):
        messagebox.showinfo("Training Complete", "Demo Training Finished!\nShowing results graph.")
        if PIL_OK:
            popup = tk.Toplevel(self)
            popup.title("Live Training Results")
            popup.geometry("1000x500")
            popup.configure(bg=CARD_BG)
            img = Image.open(path)
            img = img.resize((980, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(popup, image=photo, bg=CARD_BG)
            lbl.image = photo
            lbl.pack(expand=True, fill="both")

    def _cmd_view_training_graph(self):
        possible_paths = ["Public/training_graphs.png", "training_graphs.png"]
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
        if not found_path:
            messagebox.showerror("Error", "Graph not found.\nRun 'train_traffic_sign.py' first.")
            return
        
        self.log(f"[INFO] Opening training graph: {found_path}")
        popup = tk.Toplevel(self)
        popup.title("Deep Learning Training History")
        popup.geometry("1000x500")
        popup.configure(bg=CARD_BG)
        
        if PIL_OK:
            img = Image.open(found_path)
            img = img.resize((980, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(popup, image=photo, bg=CARD_BG)
            lbl.image = photo
            lbl.pack(expand=True, fill="both")
        else:
            tk.Label(popup, text="Pillow (PIL) missing.", fg=ACCENT2, bg=CARD_BG).pack()

    def _cmd_pedestrian(self):
        self.log("[INFO] Pedestrian Detection — select an image...")
        file = filedialog.askopenfilename(title="Select Street Image", filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file: return
        self.log(f"[INFO] Processing: {file}")
        try:
            result_img, status = detect_pedestrians(file)
            self.log(f"[OK]   {status}")
            if CV2_OK and result_img is not None:
                import cv2
                cv2.imshow(f"Pedestrian Detection — {status}", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            messagebox.showinfo("YOLOv8 Report", status)
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", str(e))

    def _cmd_lane(self):
        self.log("[INFO] Lane Detection (Basic) — select an image...")
        file = filedialog.askopenfilename(title="Select Drive Image", 
                                          filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file: return
        self.log(f"[INFO] Processing: {file}")
        
        try:
            # Run the basic lane detection
            result = run_basic_lane(file)
            
            # 🔧 FIX: Handle case where function returns (image, lines) tuple
            if isinstance(result, tuple) or isinstance(result, list):
                final_img = result[0] # Take the first item (the image)
            else:
                final_img = result    # It was just an image
            
            self.log("[OK]   Lane detection complete.")
            
            if CV2_OK and final_img is not None:
                import cv2
                cv2.imshow("Lane Detection (Basic)", final_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                messagebox.showerror("Error", "Result image is empty or OpenCV missing.")
                
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", f"Lane Detection Failed:\n{str(e)}")

    def _cmd_full_pipeline(self):
        self.log("[INFO] Full Autonomous Pipeline — select an image...")
        file = filedialog.askopenfilename(title="Select Drive Image", filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file: return
        self.log(f"[INFO] Running pipeline on: {file}")
        popup = tk.Toplevel(self)
        popup.title("Processing")
        popup.geometry("300x100")
        popup.configure(bg=CARD_BG)
        tk.Label(popup, text="Running AI Models…\nPlease wait.", font=FONT_SUB, fg=TEXT_PRIMARY, bg=CARD_BG).pack(expand=True)
        self.update()
        def run():
            try:
                result_img, status = run_advanced_pipeline(file)
                self.after(0, lambda: self._pipeline_done(popup, result_img, status))
            except Exception as e:
                self.after(0, lambda: self._pipeline_error(popup, str(e)))
        threading.Thread(target=run, daemon=True).start()

    def _pipeline_done(self, popup, result_img, status):
        popup.destroy()
        self.log(f"[OK]   Pipeline complete — {status}")
        if CV2_OK and result_img is not None:
            import cv2
            cv2.imshow(f"Autonomous Vision System — {status}", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            messagebox.showinfo("Pipeline", status)

    def _pipeline_error(self, popup, msg):
        popup.destroy()
        self.log(f"[ERR]  {msg}")
        messagebox.showerror("Error", msg)

    # ─────────────────────────────────────────
    # DATA LAB COMMANDS
    # ─────────────────────────────────────────
    def _build_datalab_view(self, parent):
        view = tk.Frame(parent, bg=CONTENT_BG)
        tk.Label(view, text="📊  Driving ML Lab",
                 font=FONT_HEADING, fg=TEXT_PRIMARY, bg=CONTENT_BG
                 ).pack(anchor="w", padx=30, pady=(24, 2))
        tk.Label(view, text="Traffic-sign analytics pipeline — Select task → Prepare features → Train → Evaluate.",
                 font=FONT_SUB, fg=TEXT_MUTED, bg=CONTENT_BG
                 ).pack(anchor="w", padx=32, pady=(0, 14))

        canvas = tk.Canvas(view, bg=CONTENT_BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(view, orient="vertical", command=canvas.yview,
                                  bg=CONTENT_BG, troughcolor=CONTENT_BG, bd=0)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        pipeline = tk.Frame(canvas, bg=CONTENT_BG)
        canvas_window = canvas.create_window((0, 0), window=pipeline, anchor="nw")

        def _on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())
        pipeline.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))

        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        PAD = dict(padx=30, pady=8, fill="x")

        shell1, body1 = self._make_card(pipeline, "STEP 01", "TASK + DATA SOURCE", ACCENT)
        shell1.pack(**PAD)
        tk.Label(body1, text="DRIVING TASK", font=("Courier New", 8, "bold"), fg=ACCENT, bg=CARD_BG).pack(anchor="w", pady=(2, 4))
        self._task_combo = tb.Combobox(
            body1,
            textvariable=self._driving_task,
            values=["metadata", "robustness"],
            width=28,
            bootstyle="info",
            state="readonly",
        )
        self._task_combo.pack(anchor="w", pady=(0, 8))
        file_row = tk.Frame(body1, bg=CARD_BG)
        file_row.pack(fill="x", pady=(4, 6))
        self._dataset_label = tk.Label(file_row, text="No driving dataset loaded.", font=FONT_LABEL, fg=TEXT_MUTED, bg=CARD_BG, width=42, anchor="w")
        self._dataset_label.pack(side="left")
        self._flat_btn(file_row, "📂  Load Driving Data", ACCENT, self._cmd_load_csv).pack(side="left", padx=10)
        self._ingest_stats = tk.Label(body1, text="", font=("Courier New", 8), fg=ACCENT, bg=CARD_BG, anchor="w")
        self._ingest_stats.pack(anchor="w", pady=(0, 4))

        shell2, body2 = self._make_card(pipeline, "STEP 02", "FEATURE PREP", "#a78bfa")
        shell2.pack(**PAD)
        tk.Label(body2, text="Build traffic-sign features and validate schema for the selected task.", font=FONT_LABEL, fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w", pady=(4, 8))
        pre_row = tk.Frame(body2, bg=CARD_BG)
        pre_row.pack(fill="x")
        self._flat_btn(pre_row, "⚙️  Prepare Features", "#a78bfa", self._cmd_preprocess).pack(side="left")
        self._preprocess_status = tk.Label(pre_row, text="  ○  Pending", font=("Courier New", 9), fg=TEXT_MUTED, bg=CARD_BG)
        self._preprocess_status.pack(side="left", padx=16)

        shell3, body3 = self._make_card(pipeline, "STEP 03", "MODEL TRAINING", ACCENT3)
        shell3.pack(**PAD)
        train_left  = tk.Frame(body3, bg=CARD_BG)
        train_left.pack(side="left", fill="x", expand=True)
        train_right = tk.Frame(body3, bg=CARD_BG)
        train_right.pack(side="right", anchor="s", padx=(20, 0))
        tk.Label(train_left, text="ALGORITHM", font=("Courier New", 8, "bold"), fg=ACCENT3, bg=CARD_BG).grid(row=0, column=0, sticky="w", pady=(4, 4))
        self._algo_var = tk.StringVar(value="Decision Tree")
        tb.Combobox(train_left, textvariable=self._algo_var, values=["Decision Tree", "Naive Bayes", "SVM", "Random Forest"], width=28, bootstyle="warning", state="readonly").grid(row=1, column=0, sticky="w", pady=(0, 8))
        self._train_progress = tb.Progressbar(train_left, bootstyle="warning-striped", mode="indeterminate", length=220)
        self._train_progress.grid(row=2, column=0, sticky="w", pady=(4, 4))
        self._flat_btn(train_right, "🚀  TRAIN MODEL", ACCENT3, self._cmd_train).pack(ipadx=10, ipady=8)

        shell4, body4 = self._make_card(pipeline, "STEP 04", "EVALUATION", ACCENT2)
        shell4.pack(**PAD)
        acc_frame = tk.Frame(body4, bg=CARD_BG)
        acc_frame.pack(side="left", padx=(0, 40))
        tk.Label(acc_frame, text="ACCURACY", font=("Courier New", 8, "bold"), fg=ACCENT2, bg=CARD_BG).pack(anchor="w")
        self._result_var = tk.StringVar(value="--%")
        tk.Label(acc_frame, textvariable=self._result_var, font=("Courier New", 30, "bold"), fg=ACCENT3, bg=CARD_BG).pack(anchor="w")
        self._runtime_var = tk.StringVar(value="Runtime: --")
        tk.Label(acc_frame, textvariable=self._runtime_var, font=("Courier New", 9), fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w", pady=(2, 0))
        report_frame = tk.Frame(body4, bg=CARD_BG)
        report_frame.pack(side="left", anchor="center")
        self._flat_btn(report_frame, "📉  Confusion Matrix", ACCENT2, self._cmd_confusion_matrix).pack(fill="x", pady=5, ipadx=6, ipady=5)
        self._flat_btn(report_frame, "📄  Classification Report", "#5c6370", self._cmd_classification_report).pack(fill="x", pady=5, ipadx=6, ipady=5)

        shell5, body5 = self._make_card(pipeline, "STEP 05", "CSV DATASET LAB", "#4f5d75")
        shell5.pack(**PAD)
        tk.Label(
            body5,
            text="Use generic CSV training for experimentation. Driving workflow above remains the default path.",
            font=FONT_LABEL,
            fg=TEXT_MUTED,
            bg=CARD_BG,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        adv_top = tk.Frame(body5, bg=CARD_BG)
        adv_top.pack(fill="x", pady=(0, 6))
        self._adv_dataset_label = tk.Label(
            adv_top,
            text="No CSV imported.",
            font=FONT_LABEL,
            fg=TEXT_MUTED,
            bg=CARD_BG,
            width=42,
            anchor="w",
        )
        self._adv_dataset_label.pack(side="left")
        self._flat_btn(adv_top, "📂  Import CSV", "#4f5d75", self._cmd_adv_import_csv).pack(side="left", padx=10)

        adv_mid = tk.Frame(body5, bg=CARD_BG)
        adv_mid.pack(fill="x", pady=(4, 6))
        tk.Label(adv_mid, text="TARGET", font=("Courier New", 8, "bold"), fg="#9fb3c8", bg=CARD_BG).grid(row=0, column=0, sticky="w")
        self._adv_target_var = tk.StringVar()
        self._adv_target_combo = tb.Combobox(
            adv_mid,
            textvariable=self._adv_target_var,
            width=24,
            bootstyle="secondary",
            state="readonly",
        )
        self._adv_target_combo.grid(row=1, column=0, sticky="w", pady=(2, 0))
        tk.Label(adv_mid, text="ALGORITHM", font=("Courier New", 8, "bold"), fg="#9fb3c8", bg=CARD_BG).grid(row=0, column=1, sticky="w", padx=(20, 0))
        self._adv_algo_var = tk.StringVar(value="Decision Tree")
        tb.Combobox(
            adv_mid,
            textvariable=self._adv_algo_var,
            values=["Decision Tree", "Naive Bayes", "SVM", "Random Forest"],
            width=24,
            bootstyle="secondary",
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=(20, 0), pady=(2, 0))

        adv_actions = tk.Frame(body5, bg=CARD_BG)
        adv_actions.pack(fill="x", pady=(6, 2))
        self._flat_btn(adv_actions, "⚙️  Preprocess", "#6c7a89", self._cmd_adv_preprocess).pack(side="left")
        self._flat_btn(adv_actions, "🚀  Train", "#6c7a89", self._cmd_adv_train).pack(side="left", padx=8)
        self._flat_btn(adv_actions, "📉  CM", "#6c7a89", self._cmd_adv_confusion_matrix).pack(side="left", padx=8)
        self._flat_btn(adv_actions, "📄  Report", "#6c7a89", self._cmd_adv_classification_report).pack(side="left", padx=8)

        self._adv_status_var = tk.StringVar(value="CSV dataset lab ready.")
        tk.Label(body5, textvariable=self._adv_status_var, font=("Courier New", 9), fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w", pady=(6, 0))

        return view

    def _make_card(self, parent, step_num, title, accent_color):
        shell = tk.Frame(parent, bg=accent_color, pady=1, padx=1)
        card = tk.Frame(shell, bg=CARD_BG)
        card.pack(fill="both", expand=True)
        header = tk.Frame(card, bg=CARD_BG)
        header.pack(fill="x", padx=14, pady=(12, 0))
        tk.Label(header, text=step_num, font=("Courier New", 8, "bold"), fg=accent_color, bg=CARD_BG).pack(side="left")
        tk.Label(header, text=f"  {title}", font=("Courier New", 10, "bold"), fg=TEXT_PRIMARY, bg=CARD_BG).pack(side="left")
        tk.Frame(card, bg="#252a38", height=1).pack(fill="x", padx=14, pady=(8, 0))
        body = tk.Frame(card, bg=CARD_BG)
        body.pack(fill="both", expand=True, padx=16, pady=10)
        return shell, body

    def _flat_btn(self, parent, text, color, command):
        btn = tk.Label(parent, text=text, font=("Courier New", 10, "bold"), fg="#0a0c10", bg=color, padx=14, pady=6, cursor="hand2")
        def on_enter(e): btn.config(bg=TEXT_PRIMARY)
        def on_leave(e): btn.config(bg=color)
        def on_click(e): command()
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.bind("<Button-1>", on_click)
        return btn

    def _cmd_load_csv(self):
        task = self._driving_task.get().strip().lower()
        try:
            summary = get_driving_dataset_summary(task=task, data_dir="data", max_samples=5000)
            self.df = summary
            self._dataset_label.config(text=f"Task: {task}  ({summary['samples']:,} samples)", fg=TEXT_PRIMARY)
            self._ingest_stats.config(
                text=f"Features: {summary['features']}    |    Classes: {summary['classes']}"
            )
            self._preprocess_status.config(text="  ○  Pending", fg=TEXT_MUTED)
            self._result_var.set("--%")
            self._runtime_var.set("Runtime: --")
            self._last_driving_result = None
            self.log(f"[OK]   Loaded driving task '{task}'")
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", str(e))

    def _cmd_preprocess(self):
        if self.df is None:
            messagebox.showerror("Error", "Load a driving task first.")
            return
        self._preprocess_status.config(text="  ◌  Running…", fg=ACCENT3)
        self.update()

        def run():
            try:
                task = self._driving_task.get().strip().lower()
                summary = get_driving_dataset_summary(task=task, data_dir="data", max_samples=5000)
                self.df = summary
                text = f"Task={task}, Samples={summary['samples']}, Features={summary['features']}"
                self.after(0, lambda: self._preprocess_done(text))
            except Exception as e:
                self.after(0, lambda: self._preprocess_fail(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _preprocess_done(self, summary):
        self._preprocess_status.config(text="  ✔  Done", fg="#00ff88")
        self.log(f"[OK]   Preprocessing — {summary}")

    def _preprocess_fail(self, msg):
        self._preprocess_status.config(text="  ✖  Failed", fg=ACCENT2)
        messagebox.showerror("Error", msg)

    def _cmd_train(self):
        if self.df is None:
            messagebox.showerror("Error", "Load a driving task first.")
            return
        task = self._driving_task.get().strip().lower()
        algo = self._algo_var.get()
        self.log(f"[INFO] Training {algo}…")
        self._result_var.set("…")
        self._runtime_var.set("Runtime: measuring...")
        self._train_progress.start(12)
        self.update()

        def run():
            try:
                result = train_driving_model(task=task, algorithm=algo, data_dir="data", test_size=0.2, max_samples=5000)
                self.after(0, lambda: self._train_done(result))
            except Exception as e:
                self.after(0, lambda: self._train_error(str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _train_done(self, result):
        self._train_progress.stop()
        self._last_driving_result = result
        self._result_var.set(f"{result['accuracy']:.1%}")
        self._runtime_var.set(
            f"Runtime: train {result['train_seconds']:.2f}s | infer {result['ms_per_sample']:.2f} ms/sample"
        )
        self.log(
            f"[OK]   {result['algorithm']} ({result['task']}) Acc={result['accuracy']:.4f}, "
            f"F1w={result['f1_weighted']:.4f}"
        )

    def _train_error(self, msg):
        self._train_progress.stop()
        self._result_var.set("Err")
        self._runtime_var.set("Runtime: --")
        messagebox.showerror("Error", msg)

    def _cmd_confusion_matrix(self):
        if not self._last_driving_result:
            messagebox.showinfo("Info", "Train a driving model first.")
            return

        matrix_path = self._last_driving_result.get("confusion_matrix_png_path")
        if not matrix_path or not os.path.exists(matrix_path):
            messagebox.showerror("Error", "Confusion matrix image not found.")
            return

        if not PIL_OK:
            messagebox.showinfo("Confusion Matrix", f"Saved at: {matrix_path}")
            return

        try:
            image = Image.open(matrix_path)
            image.thumbnail((900, 640))
            win = tk.Toplevel(self)
            win.title("Confusion Matrix")
            win.configure(bg=CARD_BG)
            photo = ImageTk.PhotoImage(image)
            label = tk.Label(win, image=photo, bg=CARD_BG)
            label.image = photo
            label.pack(fill="both", expand=True, padx=8, pady=8)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _cmd_classification_report(self):
        if not self._last_driving_result:
            messagebox.showinfo("Info", "Train a driving model first.")
            return

        report_path = self._last_driving_result.get("classification_report_path")
        grouped_path = self._last_driving_result.get("grouped_error_report_path")
        if not report_path or not os.path.exists(report_path):
            messagebox.showerror("Error", "Classification report not found.")
            return

        try:
            with open(report_path, "r", encoding="utf-8") as report_file:
                report = report_file.read()

            grouped_block = ""
            if grouped_path and os.path.exists(grouped_path):
                grouped_block = f"\n\nGrouped Error Report CSV:\n{grouped_path}\n"

            win = tk.Toplevel(self)
            win.title("Classification Report")
            win.configure(bg=LOG_BG)
            txt = tk.Text(win, bg=LOG_BG, fg="#7ec8a0", font=("Courier New", 9))
            txt.insert("1.0", report + grouped_block)
            txt.pack(fill="both", expand=True)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _cmd_adv_import_csv(self):
        if not PANDAS_OK:
            messagebox.showerror("Error", "pandas is not installed.")
            return
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file:
            return
        try:
            self._adv_df = pd.read_csv(file)
            cols = list(self._adv_df.columns)
            self._adv_target_combo["values"] = cols
            if cols:
                self._adv_target_combo.current(0)
            short = file.replace("\\", "/").split("/")[-1]
            self._adv_loaded_name = short
            self._adv_dataset_label.config(text=f"{short}  ({len(self._adv_df):,} rows)", fg=TEXT_PRIMARY)
            self._adv_status_var.set("CSV imported. Choose target and algorithm.")
            self._last_adv_result = None
            self.log(f"[OK]   CSV dataset loaded '{short}'")
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", str(e))

    def _cmd_adv_preprocess(self):
        if self._adv_df is None:
            messagebox.showerror("Error", "Import a CSV first in Dataset Lab.")
            return

        def run():
            try:
                work_df = self._adv_df.copy()
                nulls_before = int(work_df.isnull().sum().sum())
                for col in work_df.select_dtypes(include=["number"]).columns:
                    work_df[col] = work_df[col].fillna(work_df[col].median())
                for col in work_df.select_dtypes(include=["object"]).columns:
                    mode_value = work_df[col].mode()[0] if not work_df[col].mode().empty else "Unknown"
                    work_df[col] = work_df[col].fillna(mode_value)

                self._adv_df = work_df
                self.after(
                    0,
                    lambda: self._adv_status_var.set(
                        f"Preprocess complete. Null cells fixed: {nulls_before}."
                    ),
                )
                self.log(f"[OK]   Dataset preprocess complete (nulls fixed: {nulls_before})")
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _cmd_adv_train(self):
        if self._adv_df is None:
            messagebox.showerror("Error", "Import a CSV first in Dataset Lab.")
            return
        target = self._adv_target_var.get()
        algo = self._adv_algo_var.get()
        if not target:
            messagebox.showerror("Error", "Select a target column.")
            return

        self._adv_status_var.set("Training model...")

        def run():
            try:
                acc = train_model(self._adv_df.copy(), target, algo)
                self._last_adv_result = {"target": target, "algorithm": algo, "accuracy": acc}
                self.after(0, lambda: self._adv_status_var.set(f"Model trained. Accuracy: {acc:.1%}"))
                self.log(f"[OK]   Dataset {algo} Accuracy: {acc:.4f}")
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _cmd_adv_confusion_matrix(self):
        if self._adv_df is None:
            messagebox.showerror("Error", "Import a CSV first in Dataset Lab.")
            return
        target = self._adv_target_var.get()
        algo = self._adv_algo_var.get()
        if not target:
            messagebox.showerror("Error", "Select a target column.")
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import ConfusionMatrixDisplay
            from sklearn.model_selection import train_test_split
            from sklearn.naive_bayes import GaussianNB
            from sklearn.preprocessing import LabelEncoder
            from sklearn.svm import SVC
            from sklearn.tree import DecisionTreeClassifier

            work_df = self._adv_df.copy().dropna(subset=[target])
            X = work_df.drop(columns=[target]).copy()
            y = work_df[target].copy()
            X = X.fillna(0)
            for col in X.select_dtypes(include=["object"]).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            if y.dtype == "object":
                y = LabelEncoder().fit_transform(y.astype(str))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            models = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Naive Bayes": GaussianNB(),
                "SVM": SVC(),
                "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            }
            clf = models.get(algo, DecisionTreeClassifier(random_state=42))
            clf.fit(X_train, y_train)

            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor(CARD_BG)
            ax.set_facecolor(CONTENT_BG)
            ConfusionMatrixDisplay.from_estimator(
                clf,
                X_test,
                y_test,
                ax=ax,
                colorbar=True,
                cmap="Blues",
                include_values=False,
            )
            ax.set_title(f"Dataset Confusion Matrix: {algo}", color=TEXT_PRIMARY)
            if not os.path.exists("Public"):
                os.makedirs("Public")
            safe_algo = algo.replace(" ", "_")
            save_path = f"Public/dataset_confusion_matrix_{safe_algo}.png"
            plt.savefig(save_path)

            popup = tk.Toplevel(self)
            popup.geometry("750x600")
            popup.configure(bg=CARD_BG)
            canvas = FigureCanvasTkAgg(fig, master=popup)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self._adv_status_var.set(f"Confusion matrix saved: {save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _cmd_adv_classification_report(self):
        if self._adv_df is None:
            messagebox.showerror("Error", "Import a CSV first in Dataset Lab.")
            return
        target = self._adv_target_var.get()
        algo = self._adv_algo_var.get()
        if not target:
            messagebox.showerror("Error", "Select a target column.")
            return

        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import classification_report
            from sklearn.model_selection import train_test_split
            from sklearn.naive_bayes import GaussianNB
            from sklearn.preprocessing import LabelEncoder
            from sklearn.svm import SVC
            from sklearn.tree import DecisionTreeClassifier

            work_df = self._adv_df.copy().dropna(subset=[target])
            X = work_df.drop(columns=[target]).copy()
            y = work_df[target].copy()
            X = X.fillna(0)
            for col in X.select_dtypes(include=["object"]).columns:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            if y.dtype == "object":
                y = LabelEncoder().fit_transform(y.astype(str))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            models = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Naive Bayes": GaussianNB(),
                "SVM": SVC(),
                "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            }
            clf = models.get(algo, DecisionTreeClassifier(random_state=42))
            clf.fit(X_train, y_train)
            report = classification_report(y_test, clf.predict(X_test), zero_division=0)

            if not os.path.exists("Public"):
                os.makedirs("Public")
            safe_algo = algo.replace(" ", "_")
            save_path = f"Public/dataset_report_{safe_algo}.txt"
            with open(save_path, "w", encoding="utf-8") as handle:
                handle.write(report)

            win = tk.Toplevel(self)
            win.title("Dataset Classification Report")
            win.configure(bg=LOG_BG)
            txt = tk.Text(win, bg=LOG_BG, fg="#7ec8a0", font=("Courier New", 9))
            txt.insert("1.0", report)
            txt.pack(fill="both", expand=True)
            self._adv_status_var.set(f"Classification report saved: {save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = AutoDriveApp()
    app.mainloop()