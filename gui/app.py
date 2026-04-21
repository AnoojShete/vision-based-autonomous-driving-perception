import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import threading
import datetime
import os
import time
import json
import re
import importlib
import zipfile
import shutil
from collections import deque

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
    _dnd_module = importlib.import_module("tkinterdnd2")
    DND_FILES = getattr(_dnd_module, "DND_FILES", None)
    DND_OK = DND_FILES is not None
except Exception:
    DND_FILES = None
    DND_OK = False

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
    from ml_module.train_traffic_sign import start_training as train_cnn_demo
except ImportError:
    def train_cnn_demo(data_path, limit=50, epochs=5, log_func=print):
        raise NotImplementedError("ml_module.train_traffic_sign not available")

try:
    from dl_module.pipeline import process_frame
    from dl_module.traffic_sign.predict import predict_traffic_sign
    from dl_module.pedestrian_detection import detect_pedestrians
    from dl_module.lane_detection import detect_lanes_image as run_basic_lane

    import cv2

    def process_image(path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Invalid image")
        return process_frame(img)

    def process_video(input_path, output_path, progress_callback=None):
        cap = cv2.VideoCapture(input_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0
        history = deque(maxlen=5)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Normalize frame size for more stable and faster video inference.
            frame = cv2.resize(frame, (640, 360))
            processed = process_frame(frame)
            history.append(processed)

            if len(history) > 1:
                smoothed = history[0]
                for prev in list(history)[1:]:
                    smoothed = cv2.addWeighted(smoothed, 0.7, prev, 0.3, 0.0)
                processed = smoothed

            if out is None:
                h, w, _ = processed.shape
                out = cv2.VideoWriter(output_path, fourcc, 20, (w, h))

            out.write(processed)

            processed_frames += 1

            if progress_callback:
                progress_callback(processed_frames, total_frames)

            print(f"Frame {processed_frames}/{total_frames}")

        cap.release()
        if out:
            out.release()

        return output_path

    DL_OK = True

except ImportError as e:
    DL_OK = False
    print("IMPORT ERROR:", e)

    def process_image(path):
        raise NotImplementedError("dl_module not available")
    def process_video(input_path, output_path, progress_callback=None):
        raise NotImplementedError("dl_module not available")
    def predict_traffic_sign(image_path):
        raise NotImplementedError("dl_module not available")
    def detect_pedestrians(image_path):
        raise NotImplementedError("dl_module not available")
    def run_basic_lane(image_path):
        raise NotImplementedError("dl_module not available")


# ─────────────────────────────────────────
# COLOUR & STYLE CONSTANTS
# ─────────────────────────────────────────
SIDEBAR_BG   = "#0b1120"
SIDEBAR_W    = 300
ACCENT       = "#22d3ee"
ACCENT2      = "#ff5f7a"
ACCENT3      = "#f59e0b"
ACCENT4      = "#8b5cf6"
CONTENT_BG   = "#060a12"
CARD_BG      = "#111827"
CARD_BG_ALT  = "#0f172a"
LOG_BG       = "#0a1220"
BORDER       = "#1e2d42"
TEXT_PRIMARY = "#e4ecf7"
TEXT_MUTED   = "#8a9bc0"
TEXT_FAINT   = "#5f7196"
NAV_ACTIVE   = "#1a2234"
NAV_HOVER    = "#151d2d"
FONT_TITLE   = ("Segoe UI Semibold", 13)
FONT_NAV     = ("Segoe UI", 11)
FONT_LABEL   = ("Segoe UI", 10)
FONT_LOG     = ("Consolas", 9)
FONT_HEADING = ("Segoe UI Semibold", 22)
FONT_SUB     = ("Segoe UI", 11)


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
        self._vision_input_path = ""
        self._vision_input_image = None
        self._vision_output_image = None
        self._vision_input_photo = None
        self._vision_output_photo = None
        self._pipeline_running = False
        self._pipeline_started_at = None
        self._pipeline_history_path = os.path.join("Public", "vision_history.json")
        self._drop_hint_var = tk.StringVar(value="Supports JPG, PNG, MP4, AVI, MOV or ZIP.")
        self._drag_drop_ready = False
        self._system_health = {}
        self._tkdnd_backend_ready = False

        if not os.path.exists("Public"):
            os.makedirs("Public")

        self._initialize_tkdnd_backend()

        self._build_sidebar()
        self._build_main_area()
        self._refresh_system_status(force_log=False)

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

        # Vertical split lets users resize content area vs. log area by dragging the sash.
        splitter = tk.PanedWindow(
            right,
            orient="vertical",
            sashwidth=8,
            sashrelief="flat",
            opaqueresize=False,
            sashcursor="sb_v_double_arrow",
            bg=BORDER,
            bd=0,
            relief="flat",
        )
        splitter.pack(fill="both", expand=True)

        self._content_host = tk.Frame(splitter, bg=CONTENT_BG)
        console_frame = tk.Frame(splitter, bg=LOG_BG, height=180)
        console_frame.pack_propagate(False)

        splitter.add(self._content_host, minsize=340, stretch="always")
        splitter.add(console_frame, minsize=110)

        self._main_splitter = splitter
        self._main_right = right
        self._splitter_after_id = None

        # Set an initial split that keeps the log visible without dominating the workspace.
        self.after(0, self._clamp_splitter_sash)
        splitter.bind("<ButtonRelease-1>", lambda _e: self._clamp_splitter_sash())
        right.bind("<Configure>", lambda _e: self._schedule_splitter_clamp())

        hdr = tk.Frame(console_frame, bg=LOG_BG)
        hdr.pack(fill="x", padx=12, pady=(6, 0))
        tk.Label(hdr, text="▸ SYSTEM LOG", font=("Consolas", 8, "bold"),
                 fg=ACCENT, bg=LOG_BG).pack(side="left")
        tk.Button(hdr, text="CLEAR", font=("Consolas", 7),
                  fg=TEXT_MUTED, bg=LOG_BG, bd=0, activebackground=LOG_BG,
                  cursor="hand2", command=self._clear_log).pack(side="right")

        self._log_text = tk.Text(console_frame, bg=LOG_BG, fg=TEXT_PRIMARY,
                                  font=FONT_LOG, bd=0, wrap="word",
                                  state="disabled", cursor="arrow",
                                  insertbackground=LOG_BG)
        scroll = tk.Scrollbar(console_frame, command=self._log_text.yview,
                               bg=LOG_BG, troughcolor=LOG_BG, bd=0)
        self._log_text.config(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=12, pady=6)
        self._log_text.tag_config("ts", foreground=TEXT_FAINT)
        self._log_text.tag_config("INFO", foreground=ACCENT)
        self._log_text.tag_config("SUCCESS", foreground="#00e5a0")
        self._log_text.tag_config("WARN", foreground=ACCENT3)
        self._log_text.tag_config("ERROR", foreground=ACCENT2)
        self._log_text.tag_config("body", foreground=TEXT_PRIMARY)

        self._views = {}
        self._views["dashboard"] = self._build_dashboard_view(self._content_host)
        self._views["vision"]    = self._build_vision_view(self._content_host)
        self._views["datalab"]   = self._build_datalab_view(self._content_host)

    def _schedule_splitter_clamp(self):
        if not hasattr(self, "_main_splitter"):
            return
        if self._splitter_after_id is not None:
            try:
                self.after_cancel(self._splitter_after_id)
            except Exception:
                pass
        self._splitter_after_id = self.after(70, self._clamp_splitter_sash)

    def _clamp_splitter_sash(self):
        self._splitter_after_id = None
        if not hasattr(self, "_main_splitter") or not hasattr(self, "_main_right"):
            return

        splitter = self._main_splitter
        right = self._main_right

        try:
            total_h = max(1, int(right.winfo_height()))
            min_content_h = 340
            min_log_h = 110
            max_log_h = min(280, int(total_h * 0.38))

            # For compact windows, keep a sane content area while preserving log access.
            if total_h - min_content_h < min_log_h:
                min_content_h = max(260, total_h - min_log_h)

            try:
                current_y = int(splitter.sash_coord(0)[1])
            except Exception:
                current_y = max(min_content_h, total_h - 200)

            lower = min_content_h
            upper = max(lower, total_h - min_log_h)
            target_y = max(lower, min(current_y, upper))

            log_h = total_h - target_y
            if log_h > max_log_h:
                target_y = total_h - max_log_h

            splitter.sash_place(0, 0, int(target_y))
        except Exception:
            return

    def show_view(self, key):
        for k, frame in self._views.items():
            frame.pack_forget()
        self._views[key].pack(fill="both", expand=True)
        self._set_active_nav(key)
        if key == "dashboard":
            self._refresh_dashboard_state()

    def _is_public_writable(self):
        try:
            if not os.path.exists("Public"):
                os.makedirs("Public")
            probe = os.path.join("Public", ".write_probe.tmp")
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
            os.remove(probe)
            return True
        except Exception:
            return False

    def _is_tkdnd_backend_available(self):
        if self._tkdnd_backend_ready:
            return True
        if not DND_OK:
            return False
        try:
            tkdnd_cmd = str(self.tk.call("info", "commands", "tkdnd::drop_target")).strip()
            return bool(tkdnd_cmd)
        except Exception:
            return False

    def _initialize_tkdnd_backend(self):
        if not DND_OK:
            self._tkdnd_backend_ready = False
            return
        try:
            tkdnd_ns = getattr(_dnd_module, "TkinterDnD", None)
            if tkdnd_ns and hasattr(tkdnd_ns, "_require"):
                tkdnd_ns._require(self)
            tkdnd_cmd = str(self.tk.call("info", "commands", "tkdnd::drop_target")).strip()
            self._tkdnd_backend_ready = bool(tkdnd_cmd)
            if not self._tkdnd_backend_ready:
                self.log("[WARN] TkDND package detected but backend command is unavailable.")
        except Exception as e:
            self._tkdnd_backend_ready = False
            self.log(f"[WARN] TkDND bootstrap failed: {e}")

    def _choose_zip_image_member(self, zip_path, members):
        chooser = tk.Toplevel(self)
        chooser.title("Select Image From ZIP")
        chooser.geometry("720x420")
        chooser.configure(bg=CARD_BG)
        chooser.transient(self)
        chooser.grab_set()

        tk.Label(
            chooser,
            text="Choose an image to run",
            font=("Segoe UI Semibold", 12),
            fg=TEXT_PRIMARY,
            bg=CARD_BG,
        ).pack(anchor="w", padx=14, pady=(12, 2))
        tk.Label(
            chooser,
            text=os.path.basename(zip_path),
            font=FONT_LABEL,
            fg=TEXT_MUTED,
            bg=CARD_BG,
        ).pack(anchor="w", padx=14, pady=(0, 8))

        list_wrap = tk.Frame(chooser, bg=CARD_BG)
        list_wrap.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        listbox = tk.Listbox(
            list_wrap,
            bg=CARD_BG_ALT,
            fg=TEXT_PRIMARY,
            selectbackground=NAV_ACTIVE,
            highlightthickness=1,
            highlightbackground=BORDER,
            bd=0,
            font=FONT_LOG,
        )
        scroll = tk.Scrollbar(list_wrap, command=listbox.yview, bg=CARD_BG)
        listbox.config(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        listbox.pack(side="left", fill="both", expand=True)

        for name in members:
            listbox.insert("end", name)
        if members:
            listbox.selection_set(0)

        choice = {"value": None}

        def on_select(_event=None):
            picked = listbox.curselection()
            if not picked:
                return
            choice["value"] = members[picked[0]]
            chooser.destroy()

        def on_cancel():
            choice["value"] = None
            chooser.destroy()

        controls = tk.Frame(chooser, bg=CARD_BG)
        controls.pack(fill="x", padx=14, pady=(0, 12))
        tb.Button(controls, text="Use Selected", bootstyle="info", command=on_select).pack(side="left")
        tb.Button(controls, text="Cancel", bootstyle="secondary-outline", command=on_cancel).pack(side="left", padx=(8, 0))

        listbox.bind("<Double-Button-1>", on_select)
        chooser.protocol("WM_DELETE_WINDOW", on_cancel)
        chooser.wait_window()
        return choice["value"]

    def _extract_image_from_zip(self, zip_path):
        if not os.path.exists(zip_path):
            return None

        extract_root = os.path.join("Public", "zip_extract")
        if not os.path.exists(extract_root):
            os.makedirs(extract_root)

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target_dir = os.path.join(extract_root, stamp)
        os.makedirs(target_dir, exist_ok=True)

        image_candidates = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                lowered = name.lower()
                if lowered.endswith((".jpg", ".jpeg", ".png")) and not lowered.endswith("/"):
                    image_candidates.append(name)

            if not image_candidates:
                return ""

            image_candidates = sorted(image_candidates)
            if len(image_candidates) == 1:
                chosen = image_candidates[0]
            else:
                chosen = self._choose_zip_image_member(zip_path, image_candidates)
                if not chosen:
                    return None

            safe_name = os.path.basename(chosen) or "image_from_zip.png"
            out_path = os.path.join(target_dir, safe_name)
            with zf.open(chosen, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return out_path

    def _resolve_vision_input(self, file_path):
        if file_path.lower().endswith(".zip"):
            try:
                extracted = self._extract_image_from_zip(file_path)
            except Exception as e:
                self.log(f"[ERR] Failed to read zip: {e}")
                return None
            if extracted == "":
                messagebox.showerror("ZIP Error", "No JPG/PNG image found inside the ZIP file.")
                return None
            if extracted is None:
                self.log("[INFO] ZIP selection canceled by user.")
                return None
            self.log(f"[INFO] ZIP input extracted: {os.path.basename(extracted)}")
            return extracted
        return file_path

    def _compute_system_health(self):
        health = {
            "dl_module": DL_OK,
            "ml_module": ML_OK,
            "opencv": CV2_OK,
            "pillow": PIL_OK,
            "pandas": PANDAS_OK,
            "yolo_weights": os.path.exists("yolov8n.pt"),
            "public_writable": self._is_public_writable(),
            "tkdnd_backend": self._is_tkdnd_backend_available(),
            "drag_drop": self._drag_drop_ready,
        }
        critical_keys = ["dl_module", "opencv", "yolo_weights", "public_writable"]
        critical_missing = [k for k in critical_keys if not health.get(k)]
        health["critical_missing"] = critical_missing
        health["ready"] = len(critical_missing) == 0
        return health

    def _refresh_system_status(self, force_log=False):
        self._system_health = self._compute_system_health()
        missing = self._system_health.get("critical_missing", [])
        if not missing:
            text = "● System Ready"
            color = "#00e5a0"
        else:
            text = f"● System Check ({len(missing)} issue{'s' if len(missing) != 1 else ''})"
            color = ACCENT3 if len(missing) <= 2 else ACCENT2
            if force_log:
                self.log(f"[WARN] Critical checks failing: {', '.join(missing)}")

        self._status_dot.config(text=text, fg=color)
        self._refresh_dashboard_state()

    def log(self, msg):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        level = "INFO"
        clean = msg
        if msg.startswith("[OK]"):
            level, clean = "SUCCESS", msg[4:].strip()
        elif msg.startswith("[ERR]"):
            level, clean = "ERROR", msg[5:].strip()
        elif msg.startswith("[WARN]"):
            level, clean = "WARN", msg[6:].strip()
        elif msg.startswith("[INFO]"):
            level, clean = "INFO", msg[6:].strip()

        self._log_text.config(state="normal")
        self._log_text.insert("end", f"[{ts}] ", "ts")
        self._log_text.insert("end", f"{level:<7}", level)
        self._log_text.insert("end", f" {clean}\n", "body")
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

        self._db_total_runs_var = tk.StringVar(value="0")
        self._db_success_runs_var = tk.StringVar(value="0")
        self._db_fail_runs_var = tk.StringVar(value="0")
        self._db_dl_var = tk.StringVar(value="Checking...")
        self._db_ml_var = tk.StringVar(value="Checking...")
        self._latest_run_when_var = tk.StringVar(value="--")
        self._latest_run_modules_var = tk.StringVar(value="--")
        self._latest_run_file_var = tk.StringVar(value="--")
        self._latest_run_runtime_var = tk.StringVar(value="--")
        self._latest_run_fps_var = tk.StringVar(value="--")
        self._latest_run_conf_var = tk.StringVar(value="--")

        top = tk.Frame(view, bg=CONTENT_BG)
        top.pack(fill="x", padx=32, pady=(24, 0))
        tk.Label(top, text="Command Center", font=FONT_HEADING,
                 fg=TEXT_PRIMARY, bg=CONTENT_BG).pack(anchor="w")
        tk.Label(
            top,
            text="Operational overview with direct actions for Vision runs, training, and run diagnostics.",
            font=FONT_SUB,
            fg=TEXT_MUTED,
            bg=CONTENT_BG,
        ).pack(anchor="w", pady=(4, 10))

        actions = tk.Frame(top, bg=CONTENT_BG)
        actions.pack(anchor="w", pady=(0, 6))
        tb.Button(actions, text="Open Vision Studio", bootstyle="info", command=lambda: self.show_view("vision")).pack(side="left", padx=(0, 8))
        tb.Button(actions, text="Open Data Lab", bootstyle="warning-outline", command=lambda: self.show_view("datalab")).pack(side="left", padx=(0, 8))
        tb.Button(actions, text="View Run History", bootstyle="secondary-outline", command=self._cmd_view_pipeline_history).pack(side="left")
        tb.Button(actions, text="Run System Check", bootstyle="success-outline", command=lambda: self._refresh_system_status(force_log=True)).pack(side="left", padx=(8, 0))

        snapshot = tk.Frame(view, bg=CONTENT_BG)
        snapshot.pack(fill="x", padx=32, pady=(10, 6))
        for title, value_var, color in [
            ("Total Runs", self._db_total_runs_var, ACCENT),
            ("Successful", self._db_success_runs_var, "#00e5a0"),
            ("Failed", self._db_fail_runs_var, ACCENT2),
            ("DL Modules", self._db_dl_var, ACCENT4),
            ("ML Modules", self._db_ml_var, ACCENT3),
        ]:
            card = tk.Frame(snapshot, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER, width=180, height=88)
            card.pack(side="left", padx=(0, 8), fill="y")
            card.pack_propagate(False)
            tk.Label(card, text=title, font=("Segoe UI", 9), fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w", padx=12, pady=(10, 2))
            tk.Label(card, textvariable=value_var, font=("Segoe UI Semibold", 16), fg=color, bg=CARD_BG).pack(anchor="w", padx=12)

        lower = tk.Frame(view, bg=CONTENT_BG)
        lower.pack(fill="both", expand=True, padx=32, pady=(8, 18))
        left = tk.Frame(lower, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right = tk.Frame(lower, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        tk.Label(left, text="Startup Checklist", font=("Segoe UI Semibold", 12), fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w", padx=14, pady=(12, 8))
        self._dashboard_check_rows = {}
        for key, title in [
            ("vision_pipeline", "Vision pipeline backend"),
            ("data_lab", "Data Lab backend"),
            ("drag_drop", "Drag-and-drop"),
            ("history_store", "History persistence"),
        ]:
            row = tk.Frame(left, bg=CARD_BG_ALT, highlightthickness=1, highlightbackground=BORDER)
            row.pack(fill="x", padx=14, pady=4)
            label = tk.Label(row, text=title, font=FONT_LABEL, fg=TEXT_MUTED, bg=CARD_BG_ALT)
            label.pack(anchor="w", padx=10, pady=8)
            self._dashboard_check_rows[key] = label

        tk.Label(right, text="Latest Run", font=("Segoe UI Semibold", 12), fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w", padx=14, pady=(12, 8))
        latest_box = tk.Frame(right, bg=LOG_BG, highlightthickness=1, highlightbackground=BORDER)
        latest_box.pack(fill="both", expand=True, padx=14, pady=(0, 14))

        top_row = tk.Frame(latest_box, bg=LOG_BG)
        top_row.pack(fill="x", padx=10, pady=(10, 4))
        self._latest_status_badge = tk.Label(
            top_row,
            text="NO RUN",
            font=("Consolas", 8, "bold"),
            fg=TEXT_PRIMARY,
            bg=CARD_BG_ALT,
            padx=8,
            pady=3,
        )
        self._latest_status_badge.pack(side="left")
        tk.Label(top_row, textvariable=self._latest_run_when_var, font=FONT_LABEL, fg=TEXT_MUTED, bg=LOG_BG).pack(side="left", padx=(10, 0))

        info = tk.Frame(latest_box, bg=LOG_BG)
        info.pack(fill="x", padx=10)

        def _meta_row(label_text, value_var):
            row = tk.Frame(info, bg=LOG_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label_text, font=("Consolas", 8), fg=TEXT_FAINT, bg=LOG_BG, width=10, anchor="w").pack(side="left")
            tk.Label(row, textvariable=value_var, font=FONT_LABEL, fg=TEXT_PRIMARY, bg=LOG_BG, anchor="w", justify="left", wraplength=360).pack(side="left", fill="x", expand=True)

        _meta_row("Modules", self._latest_run_modules_var)
        _meta_row("Source", self._latest_run_file_var)
        _meta_row("Runtime", self._latest_run_runtime_var)
        _meta_row("FPS", self._latest_run_fps_var)
        _meta_row("Confidence", self._latest_run_conf_var)

        tk.Label(latest_box, text="Details", font=("Segoe UI Semibold", 10), fg=TEXT_PRIMARY, bg=LOG_BG).pack(anchor="w", padx=10, pady=(8, 4))
        self._latest_details_text = tk.Text(latest_box, bg=CARD_BG_ALT, fg=TEXT_PRIMARY, font=FONT_LOG, bd=0, height=7, wrap="word")
        self._latest_details_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._latest_details_text.insert("1.0", "Waiting for system check...")
        self._latest_details_text.config(state="disabled")
        self._refresh_dashboard_state()
        return view

    def _refresh_dashboard_state(self):
        history = self._load_pipeline_history()
        total_runs = len(history)
        success_runs = len([h for h in history if h.get("status") == "success"])
        fail_runs = len([h for h in history if h.get("status") == "failed"])
        last_run = history[-1] if history else None

        if hasattr(self, "_db_total_runs_var"):
            self._db_total_runs_var.set(str(total_runs))
        if hasattr(self, "_db_success_runs_var"):
            self._db_success_runs_var.set(str(success_runs))
        if hasattr(self, "_db_fail_runs_var"):
            self._db_fail_runs_var.set(str(fail_runs))
        if hasattr(self, "_db_dl_var"):
            self._db_dl_var.set("Ready" if self._system_health.get("dl_module", DL_OK) else "Unavailable")
        if hasattr(self, "_db_ml_var"):
            self._db_ml_var.set("Ready" if self._system_health.get("ml_module", ML_OK) else "Unavailable")

        if hasattr(self, "_latest_run_when_var"):
            if last_run:
                status = str(last_run.get("status", "unknown")).upper()
                stamp = str(last_run.get("timestamp", "--"))
                modules = last_run.get("modules", []) or []
                src = str(last_run.get("source_file", "--"))
                runtime = last_run.get("runtime_s", None)
                fps = last_run.get("fps", None)
                conf = str(last_run.get("confidence_label", "Accuracy: n/a"))
                details = str(last_run.get("details", "No details available."))

                self._latest_run_when_var.set(stamp)
                self._latest_run_modules_var.set(", ".join(modules) if modules else "--")
                self._latest_run_file_var.set(os.path.basename(src) if src else "--")
                self._latest_run_runtime_var.set(f"{runtime:.3f}s" if isinstance(runtime, (int, float)) else "--")
                self._latest_run_fps_var.set(f"{fps:.2f}" if isinstance(fps, (int, float)) else "--")
                self._latest_run_conf_var.set(conf.replace("Accuracy: ", ""))

                badge_bg = "#1b4332" if status == "SUCCESS" else ("#5c1d1d" if status == "FAILED" else CARD_BG_ALT)
                badge_fg = "#00e5a0" if status == "SUCCESS" else ("#ff5f7a" if status == "FAILED" else TEXT_PRIMARY)
                self._latest_status_badge.config(text=status, bg=badge_bg, fg=badge_fg)

                self._latest_details_text.config(state="normal")
                self._latest_details_text.delete("1.0", "end")
                self._latest_details_text.insert("1.0", details)
                self._latest_details_text.config(state="disabled")
            else:
                self._latest_run_when_var.set("--")
                self._latest_run_modules_var.set("--")
                self._latest_run_file_var.set("--")
                self._latest_run_runtime_var.set("--")
                self._latest_run_fps_var.set("--")
                self._latest_run_conf_var.set("--")
                self._latest_status_badge.config(text="NO RUN", bg=CARD_BG_ALT, fg=TEXT_PRIMARY)
                self._latest_details_text.config(state="normal")
                self._latest_details_text.delete("1.0", "end")
                self._latest_details_text.insert("1.0", "No pipeline runs recorded yet.\nUse Vision Studio to run your first inference.")
                self._latest_details_text.config(state="disabled")

        if hasattr(self, "_dashboard_check_rows"):
            checklist_states = {
                "vision_pipeline": self._system_health.get("dl_module", DL_OK) and self._system_health.get("opencv", CV2_OK) and self._system_health.get("yolo_weights", False),
                "data_lab": self._system_health.get("ml_module", ML_OK) and self._system_health.get("pandas", PANDAS_OK),
                "drag_drop": self._system_health.get("tkdnd_backend", False) and self._drag_drop_ready,
                "history_store": self._system_health.get("public_writable", False),
            }
            for key, label in self._dashboard_check_rows.items():
                state = checklist_states.get(key, False)
                dot = "●" if state else "○"
                color = "#00e5a0" if state else ACCENT3
                title = label.cget("text").split(" ", 1)[1] if " " in label.cget("text") and label.cget("text").startswith(("●", "○")) else label.cget("text")
                label.config(text=f"{dot} {title}", fg=color)

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

        canvas = tk.Canvas(view, bg=CONTENT_BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(view, orient="vertical", command=canvas.yview,
                                 bg=CONTENT_BG, troughcolor=CONTENT_BG, bd=0)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        shell = tk.Frame(canvas, bg=CONTENT_BG)
        shell_window = canvas.create_window((0, 0), window=shell, anchor="nw")

        def _on_configure(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(shell_window, width=canvas.winfo_width())

        def _on_canvas_configure(e):
            canvas.itemconfig(shell_window, width=e.width)

        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        shell.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        topbar = tk.Frame(shell, bg=CONTENT_BG)
        topbar.pack(fill="x", padx=30, pady=(24, 0))
        tk.Label(topbar, text="Vision Studio", font=FONT_HEADING,
                 fg=TEXT_PRIMARY, bg=CONTENT_BG).pack(anchor="w")
        tk.Label(topbar,
                 text="Upload a road scene and run selected perception modules with one production-style pipeline trigger.",
                 font=FONT_SUB, fg=TEXT_MUTED, bg=CONTENT_BG).pack(anchor="w", pady=(4, 0))

        workspace_card = tk.Frame(shell, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        workspace_card.pack(fill="both", expand=True, padx=30, pady=(14, 10))

        self._drop_area = tk.Frame(workspace_card, bg=CARD_BG_ALT, highlightthickness=1, highlightbackground=BORDER)
        self._drop_area.pack(fill="both", expand=True, padx=18, pady=(18, 10))

        tk.Label(self._drop_area, text="Drop Image or Video Here", font=("Segoe UI Semibold", 15),
                 fg=TEXT_PRIMARY, bg=CARD_BG_ALT).pack(pady=(28, 6))
        tk.Label(self._drop_area,
                 textvariable=self._drop_hint_var,
                 font=FONT_LABEL, fg=TEXT_MUTED, bg=CARD_BG_ALT).pack()

        upload_row = tk.Frame(self._drop_area, bg=CARD_BG_ALT)
        upload_row.pack(pady=12)
        tb.Button(upload_row, text="Browse Media", bootstyle="info-outline", command=self._cmd_upload_vision_image).pack(side="left", padx=5)
        tb.Button(upload_row, text="Use Sample", bootstyle="secondary-outline", command=self._cmd_load_sample_image).pack(side="left", padx=5)

        self._img_label = tk.Label(self._drop_area,
                                   text="No image loaded yet.",
                                   font=FONT_SUB, fg=TEXT_FAINT, bg=CARD_BG_ALT,
                                   justify="center")
        self._img_label.pack(fill="both", expand=True, padx=14, pady=(4, 16))

        cta_zone = tk.Frame(workspace_card, bg=CARD_BG)
        cta_zone.pack(fill="x", padx=18, pady=(0, 14))
        self._run_btn = tk.Button(
            cta_zone,
            text="🚀 Run Full Pipeline",
            font=("Segoe UI Semibold", 13),
            fg="#031722",
            bg=ACCENT,
            activebackground="#7be8ff",
            activeforeground="#031722",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=20,
            pady=12,
            command=self._cmd_full_pipeline,
        )
        self._run_btn.pack(fill="x")

        self._pipeline_status_var = tk.StringVar(value="Status: Idle")
        tk.Label(cta_zone, textvariable=self._pipeline_status_var,
                 font=FONT_LABEL, fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w", pady=(6, 2))
        self._pipeline_progress = tb.Progressbar(cta_zone, bootstyle="info-striped", mode="indeterminate")
        self._pipeline_progress.pack(fill="x")

        bottom = tk.Frame(shell, bg=CONTENT_BG)
        bottom.pack(fill="both", expand=True, padx=30, pady=(0, 18))

        left_panel = tk.Frame(bottom, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 8))
        right_panel = tk.Frame(bottom, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        right_panel.pack(side="left", fill="both", expand=True, padx=(8, 0))

        tk.Label(left_panel, text="Modules", font=("Segoe UI Semibold", 12), fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w", padx=14, pady=(12, 8))
        self._module_traffic = tk.BooleanVar(value=True)
        self._module_pedestrian = tk.BooleanVar(value=True)
        self._module_lane = tk.BooleanVar(value=True)

        for label, var, color in [
            ("Traffic Sign Detection", self._module_traffic, ACCENT2),
            ("Pedestrian Detection", self._module_pedestrian, ACCENT3),
            ("Lane Detection", self._module_lane, ACCENT4),
        ]:
            row = tk.Frame(left_panel, bg=CARD_BG_ALT, highlightthickness=1, highlightbackground=BORDER)
            row.pack(fill="x", padx=14, pady=5)
            cb = tk.Checkbutton(
                row,
                text=label,
                variable=var,
                onvalue=True,
                offvalue=False,
                bg=CARD_BG_ALT,
                fg=TEXT_PRIMARY,
                activebackground=CARD_BG_ALT,
                activeforeground=TEXT_PRIMARY,
                selectcolor="#132036",
                font=FONT_LABEL,
                anchor="w",
                padx=8,
            )
            cb.pack(side="left", fill="x", expand=True, pady=7)
            tk.Frame(row, bg=color, width=4).pack(side="right", fill="y")

        tk.Label(left_panel, text="Secondary Actions", font=("Segoe UI Semibold", 11), fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w", padx=14, pady=(12, 6))
        actions = tk.Frame(left_panel, bg=CARD_BG)
        actions.pack(fill="x", padx=14, pady=(0, 14))
        tb.Button(actions, text="Train Model", bootstyle="warning-outline", command=self._cmd_train_cnn_demo).pack(side="left", padx=(0, 8))
        tb.Button(actions, text="Training History", bootstyle="secondary-outline", command=self._cmd_view_training_graph).pack(side="left", padx=(0, 8))
        tb.Button(actions, text="Run History", bootstyle="info-outline", command=self._cmd_view_pipeline_history).pack(side="left")

        tk.Label(right_panel, text="Output", font=("Segoe UI Semibold", 12), fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w", padx=14, pady=(12, 8))
        output_stage = tk.Frame(right_panel, bg=CARD_BG_ALT, highlightthickness=1, highlightbackground=BORDER, height=220)
        output_stage.pack(fill="x", padx=14, pady=(0, 10))
        output_stage.pack_propagate(False)
        self._output_img_label = tk.Label(
            output_stage,
            text="Processed output will appear here after running the pipeline.",
            bg=CARD_BG_ALT,
            fg=TEXT_MUTED,
            font=FONT_LABEL,
            justify="center",
            wraplength=500,
        )
        self._output_img_label.pack(fill="both", expand=True)

        metrics_row = tk.Frame(right_panel, bg=CARD_BG)
        metrics_row.pack(fill="x", padx=14)
        self._metric_fps = tk.StringVar(value="FPS: --")
        self._metric_acc = tk.StringVar(value="Accuracy: --")
        self._metric_status = tk.StringVar(value="Detections: --")
        tk.Label(metrics_row, textvariable=self._metric_fps, font=FONT_LABEL, fg=ACCENT, bg=CARD_BG).pack(side="left", padx=(0, 14))
        tk.Label(metrics_row, textvariable=self._metric_acc, font=FONT_LABEL, fg=ACCENT3, bg=CARD_BG).pack(side="left", padx=(0, 14))
        tk.Label(metrics_row, textvariable=self._metric_status, font=FONT_LABEL, fg=TEXT_PRIMARY, bg=CARD_BG).pack(side="left")

        tk.Label(right_panel, text="Analysis", font=("Segoe UI Semibold", 10), fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w", padx=14, pady=(10, 4))
        self._result_text = tk.Text(right_panel, height=7, bg=LOG_BG, fg=TEXT_PRIMARY, font=FONT_LOG, bd=0, wrap="word")
        self._result_text.pack(fill="x", padx=14, pady=(10, 14))
        self._result_text.insert("1.0", "Detection summary and confidence logs will appear here.")
        self._result_text.config(state="disabled")
        self._setup_drop_target()

        return view

    # ─────────────────────────────────────────
    # ── COMMANDS: VISION & TRAINING ──────────
    # ─────────────────────────────────────────
    def _setup_drop_target(self):
        if not DND_OK:
            self._drag_drop_ready = False
            self._drop_hint_var.set("Install tkinterdnd2 to enable drag and drop. Browse still works.")
            self._refresh_system_status(force_log=False)
            return
        if not self._is_tkdnd_backend_available():
            self._drag_drop_ready = False
            self._drop_hint_var.set("TkDND backend is unavailable in this runtime. Browse still works.")
            self._refresh_system_status(force_log=False)
            return
        if not hasattr(self._drop_area, "drop_target_register"):
            self._drag_drop_ready = False
            self._drop_hint_var.set("Drag/drop package found, but DnD hooks are unavailable in this runtime.")
            self._refresh_system_status(force_log=False)
            return

        try:
            for widget in (self._drop_area, self._img_label):
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop_image)
            self._drag_drop_ready = True
            self._drop_hint_var.set("Drag and drop enabled. You can also click Browse Image.")
            self.log("[OK] Drag-and-drop ready in Vision Studio.")
            self._refresh_system_status(force_log=False)
        except Exception as e:
            self._drag_drop_ready = False
            self._drop_hint_var.set("Drag/drop setup failed. Browse Image is still available.")
            self.log(f"[INFO] Drag-and-drop disabled: {e}")
            self._refresh_system_status(force_log=False)

    def _on_drop_image(self, event):
        path = self._extract_dropped_file(getattr(event, "data", ""))
        if not path:
            self.log("[WARN] Dropped item was not a supported image file.")
            return
        self._load_vision_input(path, source="drop")

    def _extract_dropped_file(self, data):
        if not data:
            return ""
        try:
            candidates = list(self.tk.splitlist(data))
        except Exception:
            candidates = [data]

        for item in candidates:
            path = item.strip().strip("{}")
            if os.path.isfile(path) and path.lower().endswith((".jpg", ".jpeg", ".png", ".zip", ".mp4", ".avi", ".mov")):
                return path
        return ""

    def _is_video_file(self, path):
        return str(path).lower().endswith((".mp4", ".avi", ".mov"))

    def _build_video_output_path(self, input_path):
        base = os.path.splitext(os.path.basename(input_path))[0]
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(input_path)[1].lower()
        if ext not in (".mp4", ".avi", ".mov"):
            ext = ".mp4"
        if not os.path.exists("Public"):
            os.makedirs("Public")
        return os.path.join("Public", f"{base}_processed_{stamp}{ext}")

    def _load_vision_input(self, file_path, source="browse"):
        resolved = self._resolve_vision_input(file_path)
        if not resolved:
            return
        self._vision_input_path = resolved
        self._update_upload_preview(resolved)
        self._pipeline_status_var.set(f"Status: Ready ({os.path.basename(resolved)})")
        self.log(f"[INFO] Input image loaded via {source}: {resolved}")

    def _cmd_upload_vision_image(self):
        file = filedialog.askopenfilename(
            title="Select Drive Media",
            filetypes=[("Images, Videos and ZIP", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov *.zip")],
        )
        if not file:
            return
        self._load_vision_input(file, source="browse")

    def _cmd_load_sample_image(self):
        candidates = []
        for root in ["Public", "."]:
            if not os.path.exists(root):
                continue
            for name in os.listdir(root):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".zip", ".mp4", ".avi", ".mov")):
                    candidates.append(os.path.join(root, name))
        if not candidates:
            messagebox.showinfo("Sample Media", "No sample media found in Public or project root.")
            return
        self._load_vision_input(candidates[0], source="sample")

    def _normalize_confidence(self, value):
        if value is None:
            return None
        try:
            conf = float(value)
            if conf > 1.0:
                conf /= 100.0
            return max(0.0, min(conf, 1.0))
        except Exception:
            return None

    def _extract_confidence(self, *text_parts):
        for part in text_parts:
            if part is None:
                continue
            text = str(part)

            matches_pct = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
            if matches_pct:
                return self._normalize_confidence(max(float(v) for v in matches_pct))

            matches_named = re.findall(
                r"(?i)(?:conf(?:idence)?|score|prob(?:ability)?|acc(?:uracy)?)\s*[:=]?\s*(0(?:\.\d+)?|1(?:\.0+)?)",
                text,
            )
            if matches_named:
                return self._normalize_confidence(matches_named[0])

            matches_bracket = re.findall(r"\((0(?:\.\d+)?|1(?:\.0+)?)\)", text)
            if matches_bracket:
                return self._normalize_confidence(matches_bracket[0])
        return None

    def _format_accuracy_metric(self, confidence):
        if confidence is None:
            return "Accuracy: n/a"
        return f"Accuracy: {confidence * 100:.1f}%"

    def _extract_detection_count(self, text):
        if not text:
            return None
        match = re.search(r"(?i)(\d+)\s+(?:pedestrians?|persons?|objects?|detections?)", str(text))
        if match:
            return int(match.group(1))
        return None

    def _normalize_traffic_sign_result(self, result):
        label = str(result)
        confidence = None
        if isinstance(result, dict):
            label = str(result.get("label") or result.get("class") or result.get("prediction") or result)
            confidence = self._normalize_confidence(
                result.get("confidence") or result.get("probability") or result.get("score")
            )
        elif isinstance(result, (tuple, list)) and result:
            label = str(result[0])
            if len(result) > 1:
                confidence = self._normalize_confidence(result[1])

        if confidence is None:
            confidence = self._extract_confidence(result)
        return label, confidence

    def _load_pipeline_history(self):
        if not os.path.exists(self._pipeline_history_path):
            return []
        try:
            with open(self._pipeline_history_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data
        except Exception as e:
            self.log(f"[WARN] Could not read run history: {e}")
        return []

    def _append_pipeline_history(self, entry):
        history = self._load_pipeline_history()
        history.append(entry)
        history = history[-250:]
        try:
            with open(self._pipeline_history_path, "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
            self._refresh_dashboard_state()
        except Exception as e:
            self.log(f"[WARN] Could not save run history: {e}")

    def _cmd_view_pipeline_history(self):
        history = self._load_pipeline_history()
        if not history:
            messagebox.showinfo("Run History", "No Vision Studio runs recorded yet.")
            return

        win = tk.Toplevel(self)
        win.title("Vision Studio Run History")
        win.geometry("920x520")
        win.configure(bg=CARD_BG)

        container = tk.Frame(win, bg=CARD_BG)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        left = tk.Frame(container, bg=CARD_BG_ALT, highlightthickness=1, highlightbackground=BORDER, width=340)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = tk.Frame(container, bg=LOG_BG, highlightthickness=1, highlightbackground=BORDER)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))

        tk.Label(left, text="Recent Pipeline Runs", font=("Segoe UI Semibold", 11), fg=TEXT_PRIMARY, bg=CARD_BG_ALT).pack(anchor="w", padx=10, pady=(10, 8))
        filters = tk.Frame(left, bg=CARD_BG_ALT)
        filters.pack(fill="x", padx=10, pady=(0, 8))

        status_var = tk.StringVar(value="all")
        module_var = tk.StringVar(value="all")
        date_var = tk.StringVar(value="")
        search_var = tk.StringVar(value="")
        module_values = sorted({m for run in history for m in run.get("modules", [])})

        tk.Label(filters, text="Status", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=CARD_BG_ALT).grid(row=0, column=0, sticky="w")
        tk.Label(filters, text="Module", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=CARD_BG_ALT).grid(row=0, column=1, sticky="w", padx=(8, 0))
        status_combo = tb.Combobox(filters, textvariable=status_var, values=["all", "success", "failed"], state="readonly", width=12, bootstyle="secondary")
        status_combo.grid(row=1, column=0, sticky="w")
        module_combo = tb.Combobox(filters, textvariable=module_var, values=["all", *module_values], state="readonly", width=14, bootstyle="secondary")
        module_combo.grid(row=1, column=1, sticky="w", padx=(8, 0))

        tk.Label(filters, text="Date (YYYY-MM-DD)", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=CARD_BG_ALT).grid(row=2, column=0, sticky="w", pady=(8, 0))
        tk.Label(filters, text="Search", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=CARD_BG_ALT).grid(row=2, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        date_entry = tb.Entry(filters, textvariable=date_var, bootstyle="secondary")
        date_entry.grid(row=3, column=0, sticky="we")
        search_entry = tb.Entry(filters, textvariable=search_var, bootstyle="secondary")
        search_entry.grid(row=3, column=1, sticky="we", padx=(8, 0))
        filters.grid_columnconfigure(0, weight=1)
        filters.grid_columnconfigure(1, weight=1)

        toolbar = tk.Frame(left, bg=CARD_BG_ALT)
        toolbar.pack(fill="x", padx=10, pady=(0, 8))

        listbox = tk.Listbox(
            left,
            bg=CARD_BG_ALT,
            fg=TEXT_PRIMARY,
            selectbackground=NAV_ACTIVE,
            bd=0,
            highlightthickness=0,
            font=FONT_LOG,
        )
        listbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        detail_top = tk.Frame(right, bg=LOG_BG)
        detail_top.pack(fill="x", padx=10, pady=(10, 4))
        run_status_badge = tk.Label(
            detail_top,
            text="NO RUN",
            font=("Consolas", 8, "bold"),
            fg=TEXT_PRIMARY,
            bg=CARD_BG_ALT,
            padx=8,
            pady=3,
        )
        run_status_badge.pack(side="left")

        run_when_var = tk.StringVar(value="--")
        run_modules_var = tk.StringVar(value="--")
        run_source_var = tk.StringVar(value="--")
        run_runtime_var = tk.StringVar(value="--")
        run_fps_var = tk.StringVar(value="--")
        run_conf_var = tk.StringVar(value="--")

        tk.Label(detail_top, textvariable=run_when_var, font=FONT_LABEL, fg=TEXT_MUTED, bg=LOG_BG).pack(side="left", padx=(10, 0))

        detail_meta = tk.Frame(right, bg=LOG_BG)
        detail_meta.pack(fill="x", padx=10)

        def _meta_row(label_text, value_var):
            row = tk.Frame(detail_meta, bg=LOG_BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label_text, font=("Consolas", 8), fg=TEXT_FAINT, bg=LOG_BG, width=10, anchor="w").pack(side="left")
            tk.Label(row, textvariable=value_var, font=FONT_LABEL, fg=TEXT_PRIMARY, bg=LOG_BG, anchor="w", justify="left", wraplength=420).pack(side="left", fill="x", expand=True)

        _meta_row("Modules", run_modules_var)
        _meta_row("Source", run_source_var)
        _meta_row("Runtime", run_runtime_var)
        _meta_row("FPS", run_fps_var)
        _meta_row("Confidence", run_conf_var)

        tk.Label(right, text="Details", font=("Segoe UI Semibold", 10), fg=TEXT_PRIMARY, bg=LOG_BG).pack(anchor="w", padx=10, pady=(8, 4))
        detail = tk.Text(right, bg=CARD_BG_ALT, fg=TEXT_PRIMARY, font=FONT_LOG, bd=0, wrap="word", height=8)
        detail.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        runs = list(reversed(history))
        filtered_runs = []

        def _matches(run):
            status_filter = status_var.get().strip().lower()
            module_filter = module_var.get().strip().lower()
            date_filter = date_var.get().strip()
            search_filter = search_var.get().strip().lower()

            if status_filter != "all" and run.get("status", "").lower() != status_filter:
                return False

            run_modules = [m.lower() for m in run.get("modules", [])]
            if module_filter != "all" and module_filter not in run_modules:
                return False

            stamp = str(run.get("timestamp", ""))
            if date_filter and date_filter not in stamp:
                return False

            if search_filter:
                haystack = " ".join(
                    [
                        str(run.get("details", "")),
                        str(run.get("source_file", "")),
                        ",".join(run.get("modules", [])),
                        str(run.get("confidence_label", "")),
                    ]
                ).lower()
                if search_filter not in haystack:
                    return False

            return True

        def refresh_list(_event=None):
            filtered_runs.clear()
            listbox.delete(0, "end")
            detail.config(state="normal")
            detail.delete("1.0", "end")
            detail.config(state="disabled")
            run_status_badge.config(text="NO RUN", bg=CARD_BG_ALT, fg=TEXT_PRIMARY)
            run_when_var.set("--")
            run_modules_var.set("--")
            run_source_var.set("--")
            run_runtime_var.set("--")
            run_fps_var.set("--")
            run_conf_var.set("--")

            for run in runs:
                if _matches(run):
                    filtered_runs.append(run)

            for idx, run in enumerate(filtered_runs, start=1):
                stamp = run.get("timestamp", "--:--:--")
                status = run.get("status", "unknown")
                modules = ",".join(run.get("modules", []))
                listbox.insert("end", f"{idx:03d}  {stamp}  [{status}]  {modules}")

            if filtered_runs:
                listbox.selection_set(0)
                on_select()
            else:
                detail.config(state="normal")
                detail.insert("end", "No runs match the active filters.")
                detail.config(state="disabled")

        def on_select(_event=None):
            selected = listbox.curselection()
            if not selected:
                return
            run = filtered_runs[selected[0]]

            status = str(run.get("status", "unknown")).upper()
            stamp = str(run.get("timestamp", "--"))
            modules = run.get("modules", []) or []
            src = str(run.get("source_file", "--"))
            runtime = run.get("runtime_s", None)
            fps = run.get("fps", None)
            conf = str(run.get("confidence_label", "Accuracy: n/a"))
            details = str(run.get("details", "No details available."))

            run_when_var.set(stamp)
            run_modules_var.set(", ".join(modules) if modules else "--")
            run_source_var.set(os.path.basename(src) if src else "--")
            run_runtime_var.set(f"{runtime:.3f}s" if isinstance(runtime, (int, float)) else "--")
            run_fps_var.set(f"{fps:.2f}" if isinstance(fps, (int, float)) else "--")
            run_conf_var.set(conf.replace("Accuracy: ", ""))

            badge_bg = "#1b4332" if status == "SUCCESS" else ("#5c1d1d" if status == "FAILED" else CARD_BG_ALT)
            badge_fg = "#00e5a0" if status == "SUCCESS" else ("#ff5f7a" if status == "FAILED" else TEXT_PRIMARY)
            run_status_badge.config(text=status, bg=badge_bg, fg=badge_fg)

            detail.config(state="normal")
            detail.delete("1.0", "end")
            detail.insert("1.0", details)
            detail.config(state="disabled")

        def clear_filters():
            status_var.set("all")
            module_var.set("all")
            date_var.set("")
            search_var.set("")
            refresh_list()

        tb.Button(toolbar, text="Apply", bootstyle="info-outline", command=refresh_list).pack(side="left", padx=(0, 6))
        tb.Button(toolbar, text="Reset", bootstyle="secondary-outline", command=clear_filters).pack(side="left")

        listbox.bind("<<ListboxSelect>>", on_select)
        status_combo.bind("<<ComboboxSelected>>", refresh_list)
        module_combo.bind("<<ComboboxSelected>>", refresh_list)
        date_entry.bind("<KeyRelease>", refresh_list)
        search_entry.bind("<KeyRelease>", refresh_list)
        refresh_list()

    def _update_upload_preview(self, file_path):
        if self._is_video_file(file_path):
            self._vision_input_photo = None
            self._img_label.config(
                image="",
                text=f"Video loaded:\n{os.path.basename(file_path)}\n\nPreview is skipped for video inputs.",
            )
            return
        if not PIL_OK:
            self._img_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            return
        try:
            img = Image.open(file_path)
            img.thumbnail((760, 300), Image.Resampling.LANCZOS)
            self._vision_input_photo = ImageTk.PhotoImage(img)
            self._img_label.config(image=self._vision_input_photo, text="")
        except Exception as e:
            self.log(f"[ERR] Failed to preview image: {e}")
            self._img_label.config(text="Image loaded, but preview failed.")

    def _update_output_preview(self, result_img):
        if result_img is None:
            self._output_img_label.config(text="No output image generated.", image="")
            return
        if not PIL_OK:
            self._output_img_label.config(text="Output ready (install Pillow to preview).", image="")
            return
        try:
            display = None
            if hasattr(result_img, "shape") and CV2_OK:
                bgr = result_img
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                display = Image.fromarray(rgb)
            elif isinstance(result_img, Image.Image):
                display = result_img

            if display is None:
                self._output_img_label.config(text="Output generated.", image="")
                return

            display.thumbnail((540, 260), Image.Resampling.LANCZOS)
            self._vision_output_photo = ImageTk.PhotoImage(display)
            self._output_img_label.config(image=self._vision_output_photo, text="")
        except Exception as e:
            self.log(f"[ERR] Failed to render output preview: {e}")
            self._output_img_label.config(text="Output generated, but preview failed.", image="")

    def _selected_modules(self):
        selected = []
        if self._module_traffic.get():
            selected.append("traffic")
        if self._module_pedestrian.get():
            selected.append("pedestrian")
        if self._module_lane.get():
            selected.append("lane")
        return selected

    def _set_pipeline_running(self, running):
        self._pipeline_running = running
        if running:
            self._pipeline_started_at = time.perf_counter()
            self._run_btn.config(text="Processing...", state="disabled", bg="#79e8ff")
            self._pipeline_progress.configure(mode="indeterminate", maximum=100, value=0)
            self._pipeline_progress.start(12)
            self._pipeline_status_var.set("Status: Running pipeline")
        else:
            self._run_btn.config(text="🚀 Run Full Pipeline", state="normal", bg=ACCENT)
            self._pipeline_progress.stop()
            self._pipeline_progress.configure(mode="indeterminate", maximum=100, value=0)

    def _prepare_video_progress_ui(self):
        self._pipeline_progress.stop()
        self._pipeline_progress.configure(mode="determinate", maximum=100, value=0)
        self._pipeline_status_var.set("Status: Processing video (0%)")

    def _update_video_progress(self, done, total):
        if total and total > 0:
            self._pipeline_progress.configure(mode="determinate", maximum=total)
            self._pipeline_progress["value"] = min(done, total)
            pct = (done * 100.0) / total
            self._pipeline_status_var.set(f"Status: Processing video ({pct:.1f}%)")
        else:
            self._pipeline_status_var.set(f"Status: Processing video (frames: {done})")

    def _pipeline_video_done(self, status, output_path, selected, source_file):
        self._set_pipeline_running(False)
        elapsed = 0.0
        if self._pipeline_started_at is not None:
            elapsed = max(0.0001, time.perf_counter() - self._pipeline_started_at)

        self._metric_fps.set("FPS: video")
        self._metric_acc.set("Accuracy: n/a")
        self._metric_status.set("Detections: video processed")
        self._output_img_label.config(
            image="",
            text=f"Video processed successfully.\nSaved to:\n{output_path}",
        )

        self._result_text.config(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert(
            "end",
            (
                "Pipeline Summary\n"
                f"Modules: {', '.join(selected)}\n"
                f"Source: {os.path.basename(source_file)}\n"
                f"Output: {output_path}\n"
                f"Status: {status}\n"
                f"Runtime: {elapsed:.2f}s"
            ),
        )
        self._result_text.config(state="disabled")
        self._pipeline_status_var.set("Status: Completed")
        self.log(f"[OK]   Video pipeline complete: {output_path}")
        self._append_pipeline_history(
            {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "source_file": source_file,
                "output_file": output_path,
                "modules": selected,
                "status": "success",
                "details": status,
                "runtime_s": round(elapsed, 3),
                "fps": None,
                "confidence": None,
                "confidence_label": "Accuracy: n/a",
            }
        )
        messagebox.showinfo("Video Complete", f"Output saved to:\n{output_path}")

    def _pipeline_image_done(self, result_img, selected, source_file):
        self._set_pipeline_running(False)

        elapsed = 0.0
        if self._pipeline_started_at is not None:
            elapsed = max(0.0001, time.perf_counter() - self._pipeline_started_at)

        self._update_output_preview(result_img)

        self._metric_fps.set("FPS: image")
        self._metric_acc.set("Accuracy: n/a")
        self._metric_status.set(f"Detections: {', '.join(selected)}")

        details = [
            "Image Pipeline Complete",
            f"Modules: {', '.join(selected)}",
            f"Source: {os.path.basename(source_file)}",
            f"Runtime: {elapsed:.3f}s",
        ]

        if "traffic" in selected:
            try:
                result = predict_traffic_sign(source_file)
                label, confidence = self._normalize_traffic_sign_result(result)
                self._metric_acc.set(self._format_accuracy_metric(confidence))
                self._metric_status.set(f"Detections: {label}")
                details.append(f"Traffic Sign: {label}")
                details.append(
                    f"Confidence: {self._format_accuracy_metric(confidence).split(': ', 1)[1]}"
                )
            except Exception as exc:
                self.log(f"[WARN] Traffic sign summary unavailable: {exc}")

        self._result_text.config(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert(
            "end",
            "\n".join(details)
        )
        self._result_text.config(state="disabled")

        self._pipeline_status_var.set("Status: Completed")
        self.log("[OK] Image pipeline complete")

        self._append_pipeline_history({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "source_file": source_file,
            "modules": selected,
            "status": "success",
            "details": "Image processed",
            "runtime_s": round(elapsed, 3),
        })

    def _cmd_traffic_sign(self):
        file = self._vision_input_path
        if not file:
            self._cmd_upload_vision_image()
            file = self._vision_input_path
        if not file:
            return
        self.log(f"[INFO] Traffic Sign Recognition on: {file}")
        try:
            result = predict_traffic_sign(file)
            label, confidence = self._normalize_traffic_sign_result(result)
            self._metric_status.set("Detections: 1")
            self._metric_acc.set(self._format_accuracy_metric(confidence))
            self._result_text.config(state="normal")
            self._result_text.delete("1.0", "end")
            self._result_text.insert(
                "end",
                f"Traffic Sign Detection\nLabel: {label}\nConfidence: {self._format_accuracy_metric(confidence).split(': ', 1)[1]}",
            )
            self._result_text.config(state="disabled")
            self.log(f"[OK] Detected traffic sign: {label}")
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
            self.log("[INFO] Starting training...")

            def run():
                try:
                    self.after(0, lambda: self._pipeline_status_var.set("Status: Training..."))

                    for i in range(1, 101):
                        time.sleep(0.05)
                        self.after(
                            0,
                            lambda v=i: self._pipeline_progress.configure(
                                mode="determinate", value=v, maximum=100
                            ),
                        )

                    graph_path = train_cnn_demo(data_path=d_path, limit=lim, epochs=ep, log_func=self.log)

                    self.after(0, lambda: self.log("[OK] Training completed"))
                    self.after(0, lambda: self._pipeline_status_var.set("Status: Training Complete"))

                    if graph_path:
                        self.after(0, lambda p=graph_path: self._show_demo_graph(p))
                except Exception as e:
                    self.after(0, lambda msg=str(e): self.log(f"[ERR] {msg}"))
                    self.after(0, lambda: self._pipeline_status_var.set("Status: Training Failed"))

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
        file = self._vision_input_path
        if not file:
            self._cmd_upload_vision_image()
            file = self._vision_input_path
        if not file:
            return
        self.log("[INFO] Pedestrian Detection started")
        self.log(f"[INFO] Processing: {file}")
        try:
            result_img, status = detect_pedestrians(file)
            confidence = self._extract_confidence(status)
            detections = self._extract_detection_count(status)
            self._update_output_preview(result_img)
            self._metric_acc.set(self._format_accuracy_metric(confidence))
            self._metric_status.set(f"Detections: {detections if detections is not None else 'n/a'}")
            self._result_text.config(state="normal")
            self._result_text.delete("1.0", "end")
            self._result_text.insert("end", f"Pedestrian Detection\n{status}")
            self._result_text.config(state="disabled")
            self.log(f"[OK]   {status}")
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", str(e))

    def _cmd_lane(self):
        file = self._vision_input_path
        if not file:
            self._cmd_upload_vision_image()
            file = self._vision_input_path
        if not file:
            return
        self.log("[INFO] Lane Detection (Basic) started")
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
            self._update_output_preview(final_img)
            self._metric_acc.set("Accuracy: lane model")
            self._metric_status.set("Detections: lane overlay")
            self._result_text.config(state="normal")
            self._result_text.delete("1.0", "end")
            self._result_text.insert("end", "Lane Detection\nLane overlays generated successfully.")
            self._result_text.config(state="disabled")
                
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", f"Lane Detection Failed:\n{str(e)}")

    def _cmd_full_pipeline(self):
        if self._pipeline_running:
            return

        input_path = self._vision_input_path
        if not input_path:
            messagebox.showwarning("No Input", "Please upload an image or video first.")
            return

        selected = self._selected_modules()
        self.log(f"[INFO] Running pipeline on: {input_path}")
        self.log(f"[INFO] Modules: {', '.join(selected)}")

        self._set_pipeline_running(True)

        def run():
            try:
                if self._is_video_file(input_path):
                    output_path = self._build_video_output_path(input_path)

                    # 🔥 IMPORTANT
                    self.after(0, self._prepare_video_progress_ui)

                    result = process_video(
                        input_path,
                        output_path,
                        progress_callback=self._update_video_progress
                    )

                    # ✅ VIDEO SUCCESS
                    if isinstance(result, str) and os.path.exists(result):
                        self.after(0, self._pipeline_video_done,
                                "Video Processed", result, selected, input_path)
                    else:
                        raise ValueError("Invalid video output")

                else:
                    result = process_image(input_path)

                    # ✅ IMAGE SUCCESS
                    if hasattr(result, "shape"):
                        self.after(0, self._pipeline_image_done,
                                result, selected, input_path)
                    else:
                        raise ValueError("Invalid image output")

            except Exception as e:
                self.after(0, self._pipeline_failed, str(e))

        threading.Thread(target=run, daemon=True).start()


    def _pipeline_done(self, result_img, status, selected, confidence, source_file):
        self._set_pipeline_running(False)
        elapsed = 0.0
        if self._pipeline_started_at is not None:
            elapsed = max(0.0001, time.perf_counter() - self._pipeline_started_at)
        fps = 1.0 / elapsed
        confidence_text = self._format_accuracy_metric(confidence)
        detections = self._extract_detection_count(status)

        self.log(f"[OK]   Pipeline complete — {status}")
        self._pipeline_status_var.set("Status: Completed")
        self._metric_fps.set(f"FPS: {fps:.1f}")
        self._metric_acc.set(confidence_text)
        self._metric_status.set(f"Detections: {detections if detections is not None else len(selected)}")
        self._update_output_preview(result_img)

        self._result_text.config(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert(
            "end",
            (
                f"Pipeline Summary\n"
                f"Modules: {', '.join(selected)}\n"
                f"Status: {status}\n"
                f"{confidence_text}\n"
                f"Runtime: {elapsed:.2f}s"
            ),
        )
        self._result_text.config(state="disabled")
        self._append_pipeline_history(
            {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "source_file": source_file,
                "modules": selected,
                "status": "success",
                "details": status,
                "runtime_s": round(elapsed, 3),
                "fps": round(fps, 2),
                "confidence": round(confidence, 4) if confidence is not None else None,
                "confidence_label": confidence_text,
            }
        )

    def _pipeline_error(self, msg, selected, source_file):
        self._set_pipeline_running(False)
        self._pipeline_status_var.set("Status: Failed")
        self.log(f"[ERR]  {msg}")
        self._append_pipeline_history(
            {
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "source_file": source_file,
                "modules": selected,
                "status": "failed",
                "details": msg,
                "runtime_s": None,
                "fps": None,
                "confidence": None,
                "confidence_label": "Accuracy: n/a",
            }
        )
        messagebox.showerror("Error", msg)

    def _pipeline_failed(self, error_msg):
        self._set_pipeline_running(False)
        self._pipeline_status_var.set("Status: Failed")

        self._result_text.config(state="normal")
        self._result_text.delete("1.0", "end")
        self._result_text.insert("end", f"Pipeline Failed\nError:\n{error_msg}")
        self._result_text.config(state="disabled")

        self.log(f"[ERR] {error_msg}")

        self._append_pipeline_history({
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "status": "failed",
            "details": error_msg,
        })

    # ─────────────────────────────────────────
    # DATA LAB COMMANDS
    # ─────────────────────────────────────────
    def _build_datalab_view(self, parent):
        view = tk.Frame(parent, bg=CONTENT_BG)
        tk.Label(view, text="Data Lab", font=FONT_HEADING, fg=TEXT_PRIMARY, bg=CONTENT_BG).pack(anchor="w", padx=30, pady=(24, 2))
        tk.Label(view, text="Structured ML workflow: ingest, prep, train, and evaluate with clear experiment controls.",
                 font=FONT_SUB, fg=TEXT_MUTED, bg=CONTENT_BG).pack(anchor="w", padx=30, pady=(0, 12))

        workspace = tk.Frame(view, bg=CONTENT_BG)
        workspace.pack(fill="both", expand=True, padx=30, pady=(0, 18))

        left_col = tk.Frame(workspace, bg=CONTENT_BG)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Frame(workspace, bg=BORDER, width=1).pack(side="left", fill="y", padx=4)
        right_col = tk.Frame(workspace, bg=CONTENT_BG)
        right_col.pack(side="left", fill="both", expand=True, padx=(8, 0))

        PAD = dict(pady=7, fill="x")

        left_hdr = tk.Frame(left_col, bg=CONTENT_BG)
        left_hdr.pack(fill="x", pady=(0, 4))
        tk.Label(left_hdr, text="PRIMARY WORKFLOW", font=("Courier New", 9, "bold"), fg=ACCENT, bg=CONTENT_BG).pack(anchor="w")
        tk.Label(left_hdr, text="Core driving model path (Steps 01-04)", font=FONT_LABEL, fg=TEXT_MUTED, bg=CONTENT_BG).pack(anchor="w")

        shell1, body1 = self._make_card(left_col, "STEP 01", "Task + Data Source", ACCENT)
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

        tk.Frame(left_col, bg=BORDER, height=1).pack(fill="x", padx=4)

        shell2, body2 = self._make_card(left_col, "STEP 02", "Feature Prep", "#a78bfa")
        shell2.pack(**PAD)
        tk.Label(body2, text="Build traffic-sign features and validate schema for the selected task.", font=FONT_LABEL, fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w", pady=(4, 8))
        pre_row = tk.Frame(body2, bg=CARD_BG)
        pre_row.pack(fill="x")
        self._flat_btn(pre_row, "⚙️  Prepare Features", "#a78bfa", self._cmd_preprocess).pack(side="left")
        self._preprocess_status = tk.Label(pre_row, text="  ○  Pending", font=("Courier New", 9), fg=TEXT_MUTED, bg=CARD_BG)
        self._preprocess_status.pack(side="left", padx=16)

        tk.Frame(left_col, bg=BORDER, height=1).pack(fill="x", padx=4)

        shell3, body3 = self._make_card(left_col, "STEP 03", "Model Training", ACCENT3)
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

        tk.Frame(left_col, bg=BORDER, height=1).pack(fill="x", padx=4)

        right_hdr = tk.Frame(right_col, bg=CONTENT_BG)
        right_hdr.pack(fill="x", pady=(0, 4))
        tk.Label(right_hdr, text="EVALUATION + EXPERIMENTS", font=("Courier New", 9, "bold"), fg=ACCENT2, bg=CONTENT_BG).pack(anchor="w")
        tk.Label(right_hdr, text="Step 04 final checks and optional Step 05 CSV sandbox", font=FONT_LABEL, fg=TEXT_MUTED, bg=CONTENT_BG).pack(anchor="w")

        shell4, body4 = self._make_card(right_col, "STEP 04", "Evaluation", ACCENT2)
        shell4.pack(**PAD)
        acc_frame = tk.Frame(body4, bg=CARD_BG)
        acc_frame.pack(side="left", padx=(0, 20))
        tk.Label(acc_frame, text="ACCURACY", font=("Courier New", 8, "bold"), fg=ACCENT2, bg=CARD_BG).pack(anchor="w")
        self._result_var = tk.StringVar(value="--%")
        tk.Label(acc_frame, textvariable=self._result_var, font=("Courier New", 30, "bold"), fg=ACCENT3, bg=CARD_BG).pack(anchor="w")
        self._runtime_var = tk.StringVar(value="Runtime: --")
        tk.Label(acc_frame, textvariable=self._runtime_var, font=("Courier New", 9), fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w", pady=(2, 0))
        report_frame = tk.Frame(body4, bg=CARD_BG)
        report_frame.pack(side="left", anchor="center")
        self._flat_btn(report_frame, "📉  Confusion Matrix", ACCENT2, self._cmd_confusion_matrix).pack(fill="x", pady=5, ipadx=6, ipady=5)
        self._flat_btn(report_frame, "📄  Classification Report", "#5c6370", self._cmd_classification_report).pack(fill="x", pady=5, ipadx=6, ipady=5)

        tk.Frame(right_col, bg=BORDER, height=1).pack(fill="x", padx=4)

        shell5, body5 = self._make_card(right_col, "STEP 05", "Optional CSV Experiment Lab", "#4f5d75")
        shell5.pack(**PAD)
        tk.Label(
            body5,
            text="Use this only when you want to test arbitrary CSV datasets. The driving workflow in Steps 01-04 remains your main production path.",
            font=FONT_LABEL,
            fg=TEXT_MUTED,
            bg=CARD_BG,
            justify="left",
        ).pack(anchor="w", pady=(2, 8))

        tk.Frame(body5, bg=BORDER, height=1).pack(fill="x", pady=(2, 8))

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

        tk.Frame(body5, bg=BORDER, height=1).pack(fill="x", pady=(4, 8))

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

        tk.Frame(body5, bg=BORDER, height=1).pack(fill="x", pady=(4, 8))

        adv_actions = tk.Frame(body5, bg=CARD_BG)
        adv_actions.pack(fill="x", pady=(6, 2))
        self._flat_btn(adv_actions, "⚙️  Preprocess", "#6c7a89", self._cmd_adv_preprocess).pack(side="left")
        self._flat_btn(adv_actions, "🚀  Train", "#6c7a89", self._cmd_adv_train).pack(side="left", padx=8)
        self._flat_btn(adv_actions, "📉  CM", "#6c7a89", self._cmd_adv_confusion_matrix).pack(side="left", padx=8)
        self._flat_btn(adv_actions, "📄  Report", "#6c7a89", self._cmd_adv_classification_report).pack(side="left", padx=8)

        self._adv_status_var = tk.StringVar(value="Optional CSV lab ready.")
        tk.Label(body5, textvariable=self._adv_status_var, font=("Courier New", 9), fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w", pady=(6, 0))

        tk.Frame(right_col, bg=BORDER, height=1).pack(fill="x", padx=4)

        helper = tk.Frame(right_col, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        helper.pack(fill="x", pady=7)
        tk.Label(helper, text="Lab Notes", font=("Segoe UI Semibold", 11), fg=TEXT_PRIMARY, bg=CARD_BG).pack(anchor="w", padx=12, pady=(10, 4))
        tk.Label(
            helper,
            text=(
                "1) Start with Step 01 and lock the task.\n"
                "2) Run prep before training.\n"
                "3) Use Evaluation for final metrics and artifacts.\n"
                "4) Use CSV Lab for side experiments only."
            ),
            font=FONT_LABEL,
            fg=TEXT_MUTED,
            bg=CARD_BG,
            justify="left",
        ).pack(anchor="w", padx=12, pady=(0, 10))

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
        btn = tk.Button(
            parent,
            text=text,
            font=("Courier New", 10, "bold"),
            fg="#091019",
            bg=color,
            activeforeground="#091019",
            relief="flat",
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            command=command,
        )

        def _hover_color(hex_color, factor=0.92):
            hex_color = hex_color.strip()
            if not hex_color.startswith("#") or len(hex_color) != 7:
                return hex_color
            r = max(0, min(255, int(int(hex_color[1:3], 16) * factor)))
            g = max(0, min(255, int(int(hex_color[3:5], 16) * factor)))
            b = max(0, min(255, int(int(hex_color[5:7], 16) * factor)))
            return f"#{r:02x}{g:02x}{b:02x}"

        hover = _hover_color(color)
        pressed = _hover_color(color, 0.84)

        btn.config(activebackground=pressed, highlightthickness=1, highlightbackground=hover)

        def on_enter(e):
            btn.config(bg=hover)

        def on_leave(e):
            btn.config(bg=color)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
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
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: self._preprocess_fail(msg))

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
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: self._train_error(msg))

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
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: messagebox.showerror("Error", msg))

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
                err_msg = str(e)
                self.after(0, lambda msg=err_msg: messagebox.showerror("Error", msg))

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