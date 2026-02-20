import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import threading
import datetime

# ─────────────────────────────────────────
# BACKEND IMPORTS (safe – commented out for UI testing)
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
    from ml_module.train import train_model
    ML_OK = True
except ImportError:
    ML_OK = False
    def train_model(df, target, algo):
        raise NotImplementedError("ml_module not available")

try:
    # Pointing to the correct files we fixed earlier
    from dl_module.traffic_sign.predict import predict_traffic_sign
    from dl_module.lane_detection import detect_lanes_image as run_basic_lane
    from dl_module.pipeline import run_advanced_pipeline
    from dl_module.pedestrian_detection import detect_pedestrians
    DL_OK = True
except ImportError:
    DL_OK = False
    def predict_traffic_sign(path): raise NotImplementedError("dl_module not available")
    def run_basic_lane(path): raise NotImplementedError("dl_module not available")
    def run_advanced_pipeline(path): raise NotImplementedError("dl_module not available")
    def detect_pedestrians(path): raise NotImplementedError("dl_module not available")


# ─────────────────────────────────────────
# COLOUR & STYLE CONSTANTS
# ─────────────────────────────────────────
SIDEBAR_BG   = "#0d0f14"
SIDEBAR_W    = 300       # Wider sidebar
ACCENT       = "#00d4ff" # cyan glow
ACCENT2      = "#ff4f5e" # red accent
ACCENT3      = "#f5c518" # yellow
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


# ═══════════════════════════════════════════════════════════════
#  MAIN APPLICATION CLASS
# ═══════════════════════════════════════════════════════════════
class AutoDriveApp(tb.Window):

    def __init__(self):
        super().__init__(themename="cyborg")
        self.title("AutoDrive AI — Perception Toolkit")
        self.geometry("1400x850") # Bigger window
        self.minsize(960, 640)
        self.configure(bg=SIDEBAR_BG)

        # State
        self.df            = None
        self._active_nav   = tk.StringVar(value="dashboard")

        # Build layout
        self._build_sidebar()
        self._build_main_area()

        # Initial view
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

        # ── Logo ──────────────────────────────
        logo_frame = tk.Frame(sb, bg=SIDEBAR_BG)
        logo_frame.pack(fill="x", padx=16, pady=(28, 8))

        tk.Label(logo_frame, text="⬡", font=("Courier New", 28, "bold"),
                 fg=ACCENT, bg=SIDEBAR_BG).pack(side="left")
        tk.Label(logo_frame, text=" AutoDrive\n AI",
                 font=FONT_TITLE, fg=TEXT_PRIMARY, bg=SIDEBAR_BG,
                 justify="left").pack(side="left", padx=6)

        # Divider
        tk.Frame(sb, bg="#1e2230", height=1).pack(fill="x", padx=16, pady=10)

        # ── Nav label ─────────────────────────
        tk.Label(sb, text="NAVIGATION", font=("Courier New", 8),
                 fg=TEXT_MUTED, bg=SIDEBAR_BG).pack(anchor="w", padx=20, pady=(4, 8))

        # ── Nav buttons ───────────────────────
        nav_items = [
            ("dashboard",  "🏠  Dashboard"),
            ("vision",     "👁️  Vision Studio"),
            ("datalab",    "📊  Data Lab"),
        ]
        self._nav_buttons = {}
        for key, label in nav_items:
            btn = self._make_nav_btn(sb, key, label)
            self._nav_buttons[key] = btn

        # ── Spacer + status footer ─────────────
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
    # MAIN AREA  (content + console)
    # ─────────────────────────────────────────
    def _build_main_area(self):
        right = tk.Frame(self, bg=CONTENT_BG)
        right.pack(side="left", fill="both", expand=True)

        # Top content area
        self._content_host = tk.Frame(right, bg=CONTENT_BG)
        self._content_host.pack(fill="both", expand=True, padx=0, pady=0)

        # Divider
        tk.Frame(right, bg="#1e2230", height=1).pack(fill="x")

        # Console / Log
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

        # Build all views (hidden initially)
        self._views = {}
        self._views["dashboard"] = self._build_dashboard_view(self._content_host)
        self._views["vision"]    = self._build_vision_view(self._content_host)
        self._views["datalab"]   = self._build_datalab_view(self._content_host)

    # ─────────────────────────────────────────
    # VIEW SWITCHER
    # ─────────────────────────────────────────
    def show_view(self, key):
        for k, frame in self._views.items():
            frame.pack_forget()
        self._views[key].pack(fill="both", expand=True)
        self._set_active_nav(key)

    # ─────────────────────────────────────────
    # LOG HELPERS
    # ─────────────────────────────────────────
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

        # Heading
        tk.Label(view, text="🚗  Autonomous Driving\nPerception System",
                 font=("Courier New", 22, "bold"), fg=TEXT_PRIMARY,
                 bg=CONTENT_BG, justify="left").pack(anchor="w", padx=40, pady=(40, 4))
        tk.Label(view, text="Real-time vision, detection & classification — all in one toolkit.",
                 font=FONT_SUB, fg=TEXT_MUTED, bg=CONTENT_BG).pack(anchor="w", padx=42, pady=(0, 30))

        # Status cards
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

        # Quick-start tip
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
        card = tk.Frame(parent, bg=CARD_BG, width=240, height=100) # WIDER CARDS
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

        # Accent bar at bottom
        tk.Frame(card, bg=color, height=3).pack(side="bottom", fill="x")

    # ─────────────────────────────────────────
    # ── VIEW: VISION STUDIO ──────────────────
    # ─────────────────────────────────────────
    def _build_vision_view(self, parent):
        view = tk.Frame(parent, bg=CONTENT_BG)

        # ── Title bar ──
        topbar = tk.Frame(view, bg=CONTENT_BG)
        topbar.pack(fill="x", padx=30, pady=(28, 0))
        tk.Label(topbar, text="👁️  Vision Studio", font=FONT_HEADING,
                 fg=TEXT_PRIMARY, bg=CONTENT_BG).pack(side="left")

        # ── Main body ──
        body = tk.Frame(view, bg=CONTENT_BG)
        body.pack(fill="both", expand=True, padx=30, pady=14)

        # LEFT: image display area
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

        # RIGHT: action panel
        # Increased width to match left sidebar (300)
        action_panel = tk.Frame(body, bg=CONTENT_BG, width=300)
        action_panel.pack(side="right", fill="y", padx=(20, 0))
        action_panel.pack_propagate(False)

        tk.Label(action_panel, text="▸ MODULES", font=("Courier New", 8, "bold"),
                 fg=ACCENT, bg=CONTENT_BG).pack(anchor="w", pady=(4, 10))

        vision_btns = [
            ("🛑  Traffic Sign Recognition", ACCENT2,   self._cmd_traffic_sign),
            ("🚶  Pedestrian Detection",      ACCENT3,   self._cmd_pedestrian),
            ("🛣️  Lane Detection (Basic)",    "#a78bfa", self._cmd_lane),
            ("🚀  Full Autonomous Pipeline",  ACCENT,    self._cmd_full_pipeline),
        ]
        for text, color, cmd in vision_btns:
            self._action_btn(action_panel, text, color, cmd)

        return view

    def _action_btn(self, parent, text, color, command):
        """Custom styled action button."""
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
    # ── VIEW: DATA LAB (PIPELINE) ────────────
    # ─────────────────────────────────────────
    def _build_datalab_view(self, parent):
        view = tk.Frame(parent, bg=CONTENT_BG)

        # ── Page heading ────────────────────────────────────────────
        tk.Label(view, text="📊  Data Lab",
                 font=FONT_HEADING, fg=TEXT_PRIMARY, bg=CONTENT_BG
                 ).pack(anchor="w", padx=30, pady=(24, 2))
        tk.Label(view, text="4-step machine learning pipeline — Load → Preprocess → Train → Evaluate.",
                 font=FONT_SUB, fg=TEXT_MUTED, bg=CONTENT_BG
                 ).pack(anchor="w", padx=32, pady=(0, 14))

        # ── Scrollable pipeline area ─────────────────────────────────
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

        # Mousewheel scroll
        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        PAD = dict(padx=30, pady=8, fill="x")   # uniform card spacing

        # ══════════════════════════════════════════════════════════════
        # CARD 1 — DATA INGESTION
        # ══════════════════════════════════════════════════════════════
        shell1, body1 = self._make_card(pipeline, "STEP 01", "DATA INGESTION", ACCENT)
        shell1.pack(**PAD)

        file_row = tk.Frame(body1, bg=CARD_BG)
        file_row.pack(fill="x", pady=(4, 6))

        self._dataset_label = tk.Label(
            file_row, text="No file loaded.",
            font=FONT_LABEL, fg=TEXT_MUTED, bg=CARD_BG,
            width=42, anchor="w"
        )
        self._dataset_label.pack(side="left")
        self._flat_btn(file_row, "📂  Browse CSV", ACCENT, self._cmd_load_csv).pack(side="left", padx=10)

        # Mini stats row (hidden until a file is loaded)
        self._ingest_stats = tk.Label(
            body1, text="",
            font=("Courier New", 8), fg=ACCENT, bg=CARD_BG, anchor="w"
        )
        self._ingest_stats.pack(anchor="w", pady=(0, 4))

        # ══════════════════════════════════════════════════════════════
        # CARD 2 — PREPROCESSING
        # ══════════════════════════════════════════════════════════════
        shell2, body2 = self._make_card(pipeline, "STEP 02", "PREPROCESSING", "#a78bfa")
        shell2.pack(**PAD)

        tk.Label(body2, text="Handle missing values, encode labels & normalise features.",
                 font=FONT_LABEL, fg=TEXT_MUTED, bg=CARD_BG).pack(anchor="w", pady=(4, 8))

        pre_row = tk.Frame(body2, bg=CARD_BG)
        pre_row.pack(fill="x")

        self._flat_btn(pre_row, "⚙️  Run Preprocessing", "#a78bfa",
                       self._cmd_preprocess).pack(side="left")

        self._preprocess_status = tk.Label(
            pre_row, text="  ○  Pending",
            font=("Courier New", 9), fg=TEXT_MUTED, bg=CARD_BG
        )
        self._preprocess_status.pack(side="left", padx=16)

        # ══════════════════════════════════════════════════════════════
        # CARD 3 — MODEL TRAINING
        # ══════════════════════════════════════════════════════════════
        shell3, body3 = self._make_card(pipeline, "STEP 03", "MODEL TRAINING", ACCENT3)
        shell3.pack(**PAD)

        # Two-column sub-grid: dropdowns on left, button on right
        train_left  = tk.Frame(body3, bg=CARD_BG)
        train_left.pack(side="left", fill="x", expand=True)
        train_right = tk.Frame(body3, bg=CARD_BG)
        train_right.pack(side="right", anchor="s", padx=(20, 0))

        # Target column
        tk.Label(train_left, text="TARGET COLUMN",
                 font=("Courier New", 8, "bold"), fg=ACCENT3, bg=CARD_BG
                 ).grid(row=0, column=0, sticky="w", pady=(4, 4))
        self._col_var   = tk.StringVar()
        self._col_combo = tb.Combobox(
            train_left, textvariable=self._col_var,
            width=28, bootstyle="warning", state="readonly"
        )
        self._col_combo.grid(row=1, column=0, sticky="w", pady=(0, 10))

        # Algorithm
        tk.Label(train_left, text="ALGORITHM",
                 font=("Courier New", 8, "bold"), fg=ACCENT3, bg=CARD_BG
                 ).grid(row=2, column=0, sticky="w", pady=(0, 4))
        self._algo_var = tk.StringVar(value="Decision Tree")
        tb.Combobox(
            train_left, textvariable=self._algo_var,
            values=["Decision Tree", "Naive Bayes", "SVM"],
            width=28, bootstyle="warning", state="readonly"
        ).grid(row=3, column=0, sticky="w", pady=(0, 8))

        # Progress bar (indeterminate, shown during training)
        self._train_progress = tb.Progressbar(
            train_left, bootstyle="warning-striped",
            mode="indeterminate", length=220
        )
        self._train_progress.grid(row=4, column=0, sticky="w", pady=(4, 4))

        # Train button (right column)
        self._flat_btn(
            train_right, "🚀  TRAIN MODEL", ACCENT3, self._cmd_train
        ).pack(ipadx=10, ipady=8)

        # ══════════════════════════════════════════════════════════════
        # CARD 4 — EVALUATION
        # ══════════════════════════════════════════════════════════════
        shell4, body4 = self._make_card(pipeline, "STEP 04", "EVALUATION", ACCENT2)
        shell4.pack(**PAD)

        # Big accuracy display
        acc_frame = tk.Frame(body4, bg=CARD_BG)
        acc_frame.pack(side="left", padx=(0, 40))

        tk.Label(acc_frame, text="ACCURACY",
                 font=("Courier New", 8, "bold"), fg=ACCENT2, bg=CARD_BG
                 ).pack(anchor="w")
        self._result_var = tk.StringVar(value="--%")
        tk.Label(acc_frame, textvariable=self._result_var,
                 font=("Courier New", 30, "bold"), fg=ACCENT3, bg=CARD_BG
                 ).pack(anchor="w")

        # Report buttons
        report_frame = tk.Frame(body4, bg=CARD_BG)
        report_frame.pack(side="left", anchor="center")

        self._flat_btn(report_frame, "📉  Confusion Matrix",
                       ACCENT2, self._cmd_confusion_matrix).pack(fill="x", pady=5, ipadx=6, ipady=5)
        self._flat_btn(report_frame, "📄  Classification Report",
                       "#5c6370", self._cmd_classification_report).pack(fill="x", pady=5, ipadx=6, ipady=5)

        return view

    def _make_card(self, parent, step_num: str, title: str, accent_color: str):
        """
        Returns (card_frame, body_frame).
        card_frame  — the outer bordered shell
        body_frame  — the inner content area to pack widgets into
        """
        CARD_BORDER = "#252a38"

        shell = tk.Frame(parent, bg=accent_color, pady=1, padx=1)  # 1-px accent border

        card = tk.Frame(shell, bg=CARD_BG)
        card.pack(fill="both", expand=True)

        # ── Header row ──────────────────────────────────────────────
        header = tk.Frame(card, bg=CARD_BG)
        header.pack(fill="x", padx=14, pady=(12, 0))

        tk.Label(header, text=step_num,
                 font=("Courier New", 8, "bold"), fg=accent_color, bg=CARD_BG
                 ).pack(side="left")

        tk.Label(header, text=f"  {title}",
                 font=("Courier New", 10, "bold"), fg=TEXT_PRIMARY, bg=CARD_BG
                 ).pack(side="left")

        # Thin rule
        tk.Frame(card, bg=CARD_BORDER, height=1).pack(fill="x", padx=14, pady=(8, 0))

        # ── Body ────────────────────────────────────────────────────
        body = tk.Frame(card, bg=CARD_BG)
        body.pack(fill="both", expand=True, padx=16, pady=10)

        return shell, body

    def _flat_btn(self, parent, text, color, command):
        """Inline flat label-button."""
        btn = tk.Label(parent, text=text, font=("Courier New", 10, "bold"),
                       fg="#0a0c10", bg=color, padx=14, pady=6, cursor="hand2")
        def on_enter(e): btn.config(bg=TEXT_PRIMARY)
        def on_leave(e): btn.config(bg=color)
        def on_click(e): command()
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        btn.bind("<Button-1>", on_click)
        return btn

    # ─────────────────────────────────────────
    # VISION COMMANDS  (placeholders)
    # ─────────────────────────────────────────
    def _cmd_traffic_sign(self):
        self.log("[INFO] Traffic Sign Recognition — select an image...")
        file = filedialog.askopenfilename(title="Select Sign Image",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file:
            self.log("[INFO] Cancelled.")
            return
        self.log(f"[INFO] Processing: {file}")
        try:
            result = predict_traffic_sign(file)
            self.log(f"[OK]   Detected: {result}")
            messagebox.showinfo("Traffic Sign", f"Detected:\n{result}")
        except NotImplementedError as e:
            self.log(f"[WARN] {e}")
            messagebox.showwarning("Placeholder", str(e))
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", str(e))

    def _cmd_pedestrian(self):
        self.log("[INFO] Pedestrian Detection — select an image...")
        file = filedialog.askopenfilename(title="Select Street Image",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file:
            self.log("[INFO] Cancelled.")
            return
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
        except NotImplementedError as e:
            self.log(f"[WARN] {e}")
            messagebox.showwarning("Placeholder", str(e))
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", str(e))

    def _cmd_lane(self):
        self.log("[INFO] Lane Detection (Basic) — select an image...")
        file = filedialog.askopenfilename(title="Select Drive Image",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file:
            self.log("[INFO] Cancelled.")
            return
        self.log(f"[INFO] Processing: {file}")
        try:
            result = run_basic_lane(file)
            self.log("[OK]   Lane detection complete.")
            if CV2_OK and result is not None:
                import cv2
                cv2.imshow("Lane Detection", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except NotImplementedError as e:
            self.log(f"[WARN] {e}")
            messagebox.showwarning("Placeholder", str(e))
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", str(e))

    def _cmd_full_pipeline(self):
        self.log("[INFO] Full Autonomous Pipeline — select an image...")
        file = filedialog.askopenfilename(title="Select Drive Image",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not file:
            self.log("[INFO] Cancelled.")
            return
        self.log(f"[INFO] Running pipeline on: {file}")

        # Show loading popup on main thread, run pipeline in background
        popup = tk.Toplevel(self)
        popup.title("Processing")
        popup.geometry("300x100")
        popup.configure(bg=CARD_BG)
        popup.resizable(False, False)
        tk.Label(popup, text="Running AI Models…\nPlease wait.",
                 font=FONT_SUB, fg=TEXT_PRIMARY, bg=CARD_BG).pack(expand=True)
        self.update()

        def run():
            try:
                result_img, status = run_advanced_pipeline(file)
                self.after(0, lambda: self._pipeline_done(popup, result_img, status))
            except NotImplementedError as e:
                self.after(0, lambda: self._pipeline_error(popup, str(e), warn=True))
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

    def _pipeline_error(self, popup, msg, warn=False):
        popup.destroy()
        self.log(f"[{'WARN' if warn else 'ERR'}]  {msg}")
        if warn:
            messagebox.showwarning("Placeholder", msg)
        else:
            messagebox.showerror("Error", msg)

    # ─────────────────────────────────────────
    # DATA LAB COMMANDS
    # ─────────────────────────────────────────
    def _cmd_load_csv(self):
        if not PANDAS_OK:
            self.log("[ERR]  pandas not installed.")
            messagebox.showerror("Error", "pandas is not installed.")
            return
        import pandas as pd
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file:
            return
        try:
            self.df = pd.read_csv(file)
            cols    = list(self.df.columns)
            self._col_combo["values"] = cols
            self._col_combo.current(0)

            short = file.replace("\\", "/").split("/")[-1]
            self._dataset_label.config(
                text=f"{short}  ({len(self.df):,} rows)", fg=TEXT_PRIMARY
            )
            self._ingest_stats.config(
                text=f"Columns: {len(cols)}    |    Null cells: {int(self.df.isnull().sum().sum())}"
            )
            # Reset downstream state
            self._preprocess_status.config(text="  ○  Pending", fg=TEXT_MUTED)
            self._result_var.set("--%")
            self._train_progress.stop()
            self.log(f"[OK]   Loaded '{short}' — {len(self.df):,} rows, {len(cols)} columns.")
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", str(e))

    def _cmd_preprocess(self):
        if self.df is None:
            self.log("[ERR]  Load a dataset first.")
            messagebox.showerror("Error", "Load a CSV dataset before preprocessing.")
            return

        self._preprocess_status.config(text="  ◌  Running…", fg=ACCENT3)
        self.update()
        self.log("[INFO] Running preprocessing…")

        def run():
            try:
                import pandas as pd

                # ── Missing values ─────────────────────────────────
                n_missing_before = int(self.df.isnull().sum().sum())
                for col in self.df.select_dtypes(include=["number"]).columns:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                for col in self.df.select_dtypes(include=["object"]).columns:
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else "Unknown",
                                        inplace=True)
                n_missing_after = int(self.df.isnull().sum().sum())

                # ── Label encoding for object columns ──────────────
                from sklearn.preprocessing import LabelEncoder
                encoded_cols = []
                for col in self.df.select_dtypes(include=["object"]).columns:
                    self.df[col] = LabelEncoder().fit_transform(self.df[col].astype(str))
                    encoded_cols.append(col)

                summary = (f"Nulls fixed: {n_missing_before} → {n_missing_after}  |  "
                           f"Encoded: {len(encoded_cols)} col(s)")
                self.after(0, lambda: self._preprocess_done(summary))
            except Exception as e:
                self.after(0, lambda: self._preprocess_fail(str(e)))

        import threading
        threading.Thread(target=run, daemon=True).start()

    def _preprocess_done(self, summary: str):
        self._preprocess_status.config(text=f"  ✔  Done", fg="#00ff88")
        self.log(f"[OK]   Preprocessing complete — {summary}")

    def _preprocess_fail(self, msg: str):
        self._preprocess_status.config(text="  ✖  Failed", fg=ACCENT2)
        self.log(f"[ERR]  Preprocessing failed — {msg}")
        messagebox.showerror("Preprocessing Error", msg)

    def _cmd_train(self):
        if self.df is None:
            self.log("[ERR]  No dataset loaded.")
            messagebox.showerror("Error", "Load a CSV dataset first.")
            return
        target = self._col_var.get()
        algo   = self._algo_var.get()
        if not target:
            self.log("[ERR]  No target column selected.")
            messagebox.showerror("Error", "Select a target column.")
            return

        self.log(f"[INFO] Training {algo} — target='{target}'…")
        self._result_var.set("…")
        self._train_progress.start(12)
        self.update()

        def run():
            try:
                acc = train_model(self.df, target, algo)
                self.after(0, lambda: self._train_done(acc, algo))
            except NotImplementedError as e:
                self.after(0, lambda: self._train_error(str(e), warn=True))
            except Exception as e:
                self.after(0, lambda: self._train_error(str(e)))

        import threading
        threading.Thread(target=run, daemon=True).start()

    def _train_done(self, acc, algo):
        self._train_progress.stop()
        pct = f"{acc:.1%}"
        self._result_var.set(pct)
        self.log(f"[OK]   {algo} → Accuracy = {acc:.4f} ({pct})")

    def _train_error(self, msg, warn=False):
        self._train_progress.stop()
        self._result_var.set("Err")
        tag = "WARN" if warn else "ERR"
        self.log(f"[{tag}]  {msg}")
        if warn:
            messagebox.showwarning("Placeholder", msg)
        else:
            messagebox.showerror("Training Error", msg)

    def _cmd_confusion_matrix(self):
        if self.df is None:
            messagebox.showerror("Error", "Train a model first.")
            return
        target = self._col_var.get()
        algo   = self._algo_var.get()
        if not target:
            messagebox.showerror("Error", "Select a target column.")
            return
        self.log(f"[INFO] Generating confusion matrix for {algo}…")
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import ConfusionMatrixDisplay
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.svm import SVC

            X = self.df.drop(columns=[target])
            y = self.df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Naive Bayes":   GaussianNB(),
                "SVM":           SVC(),
            }
            clf = models.get(algo, DecisionTreeClassifier())
            clf.fit(X_train, y_train)

            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor("#12151c")
            ax.set_facecolor("#1a1e2a")
            ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, ax=ax,
                                                  colorbar=False, cmap="Blues")
            ax.set_title(f"Confusion Matrix — {algo}", color="#e8eaf0")
            ax.tick_params(colors="#e8eaf0")
            ax.xaxis.label.set_color("#e8eaf0")
            ax.yaxis.label.set_color("#e8eaf0")
            plt.tight_layout()
            plt.show()
            self.log("[OK]   Confusion matrix displayed.")
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", str(e))

    def _cmd_classification_report(self):
        if self.df is None:
            messagebox.showerror("Error", "Train a model first.")
            return
        target = self._col_var.get()
        algo   = self._algo_var.get()
        if not target:
            messagebox.showerror("Error", "Select a target column.")
            return
        self.log(f"[INFO] Generating classification report for {algo}…")
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.naive_bayes import GaussianNB
            from sklearn.svm import SVC

            X = self.df.drop(columns=[target])
            y = self.df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Naive Bayes":   GaussianNB(),
                "SVM":           SVC(),
            }
            clf = models.get(algo, DecisionTreeClassifier())
            clf.fit(X_train, y_train)
            report = classification_report(y_test, clf.predict(X_test))

            # ── Pop-up window ──────────────────────────────────────
            win = tk.Toplevel(self)
            win.title(f"Classification Report — {algo}")
            win.configure(bg=LOG_BG)
            win.geometry("560x400")
            win.resizable(True, True)

            tk.Label(win, text=f"Classification Report  ·  {algo}",
                     font=("Courier New", 10, "bold"), fg=ACCENT, bg=LOG_BG
                     ).pack(anchor="w", padx=16, pady=(14, 6))

            txt = tk.Text(win, bg=LOG_BG, fg="#7ec8a0", font=("Courier New", 9),
                          bd=0, wrap="none", state="normal")
            txt.insert("1.0", report)
            txt.config(state="disabled")
            txt.pack(fill="both", expand=True, padx=16, pady=(0, 16))

            self.log("[OK]   Classification report displayed.")
        except Exception as e:
            self.log(f"[ERR]  {e}")
            messagebox.showerror("Error", str(e))