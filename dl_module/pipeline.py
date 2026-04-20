import os
import time
from typing import Callable, Optional

import cv2
import numpy as np

from dl_module.lane_detection import detect_lanes
from dl_module.pedestrian_detection import detect_pedestrians_frame
from dl_module.traffic_sign.predict import predict_traffic_sign_frame


_video_progress_callback: Optional[Callable[[int, int], None]] = None


def set_video_progress_callback(callback: Optional[Callable[[int, int], None]]) -> None:
    """Register an optional progress callback receiving (processed_frames, total_frames)."""
    global _video_progress_callback
    _video_progress_callback = callback


def _video_fourcc_for_path(output_path: str) -> int:
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".avi":
        return cv2.VideoWriter_fourcc(*"XVID")
    return cv2.VideoWriter_fourcc(*"mp4v")


def process_frame(frame: np.ndarray) -> np.ndarray:
    """Run the full perception stack on a single frame and return an annotated frame."""
    if frame is None or not hasattr(frame, "shape"):
        raise ValueError("Invalid input frame")

    # Start from a copy to avoid mutating the caller's frame buffer.
    working = frame.copy()

    # Lane overlay.
    try:
        lane_output = detect_lanes(working)
        if lane_output is not None and hasattr(lane_output, "shape"):
            working = lane_output
    except Exception:
        pass

    # Pedestrian boxes and counts.
    ped_status = ""
    try:
        ped_output, ped_status = detect_pedestrians_frame(working, draw=True)
        if ped_output is not None and hasattr(ped_output, "shape"):
            working = ped_output
    except Exception:
        ped_status = ""

    # Traffic sign label overlay.
    try:
        sign_label, sign_confidence, _ = predict_traffic_sign_frame(working)
        if sign_label:
            sign_text = f"Traffic Sign: {sign_label}"
            if sign_confidence is not None:
                sign_text += f" ({sign_confidence * 100:.1f}%)"
            cv2.putText(
                working,
                sign_text,
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
    except Exception:
        pass

    if ped_status:
        cv2.putText(
            working,
            ped_status,
            (18, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    return working


def process_video(input_path: str, output_path: str) -> str:
    """Process a full video and write an annotated output video."""
    if not os.path.exists(input_path):
        return f"Error: Input video not found: {input_path}"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return f"Error: Could not open input video: {input_path}"

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            return "Error: Invalid input video resolution"

        writer = cv2.VideoWriter(
            output_path,
            _video_fourcc_for_path(output_path),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            return f"Error: Could not open output writer: {output_path}"

        processed = 0
        t0 = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame is None:
                continue

            out = process_frame(frame)
            writer.write(out)
            processed += 1

            if _video_progress_callback is not None:
                _video_progress_callback(processed, total_frames)

        elapsed = max(0.0001, time.perf_counter() - t0)
        writer.release()
        cap.release()

        if processed == 0:
            return "Error: No frames processed"
        return (
            f"OK: Video processed successfully | frames={processed}"
            f" | fps_in={fps:.2f} | fps_proc={processed / elapsed:.2f}"
            f" | output={output_path}"
        )
    except Exception as exc:
        cap.release()
        return f"Error: {exc}"


def process_image(image_path: str):
    """Process a single image path via the frame-based pipeline."""
    frame = cv2.imread(image_path)
    if frame is None:
        return None, "Error loading image"
    out = process_frame(frame)
    return out, "Advanced Pipeline Success"


def run_advanced_pipeline(image_path):
    """Backward-compatible alias used by the GUI."""
    return process_image(image_path)