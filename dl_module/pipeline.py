from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
 
from dl_module.lane_detection import detect_lanes_data
from dl_module.pedestrian_detection import detect_pedestrians_data
from dl_module.traffic_sign.predict import predict_traffic_sign_frame
from dl_module.fusion import fuse
from dl_module.renderer import draw
 
# ── Module-level thread pool ─────────────────────────────────────────────────
# Created once at import time and reused for every frame.  Spawning a new pool
# per frame adds ~2–5 ms of overhead per call on most platforms.
_executor = ThreadPoolExecutor(max_workers=3)
 
# ── Sign-detection interval cache ────────────────────────────────────────────
# The traffic-sign CNN is the bottleneck inside the pipeline: it runs a full
# forward pass through a Keras model for every frame.  Road signs don't change
# every frame, so we gate it to run at most once per _SIGN_INTERVAL frames.
# Tune this constant: lower = more responsive, higher = faster throughput.
_SIGN_INTERVAL: int = 4
 
_sign_state: dict = {
    "counter": 0,
    "label":   None,
    "conf":    0.0,
}
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def draw_smooth_dashed_lane(output, lane_points,
                            color=(0,255,0),
                            thickness=6):

    if lane_points is None or len(lane_points) < 6:
        return

    # Convert (N,1,2) -> (N,2)
    pts = lane_points.reshape(-1, 2)

    x = pts[:, 0]
    y = pts[:, 1]

    try:
        # Smooth curve fit
        poly = np.polyfit(y, x, 2)

        # Generate smooth lane points
        y_new = np.linspace(y.min(), y.max(), 80)

        x_new = (
            poly[0] * y_new**2 +
            poly[1] * y_new +
            poly[2]
        )

        smooth_pts = np.stack(
            [x_new, y_new],
            axis=1
        ).astype(np.int32)

        # Overlay for transparency
        overlay = output.copy()

        # Draw dashed lane
        for i in range(0, len(smooth_pts)-1, 6):

            p1 = tuple(smooth_pts[i])
            p2 = tuple(smooth_pts[min(i+2, len(smooth_pts)-1)])

            cv2.line(
                overlay,
                p1,
                p2,
                color,
                thickness
            )

        # Blend overlay
        cv2.addWeighted(
            overlay,
            0.7,
            output,
            0.3,
            0,
            output
        )

    except Exception as e:
        print(f"[lane smooth] {e}")

def process_frame(frame: np.ndarray, *, force_sign: bool = False) -> np.ndarray:
    """
    Run the full perception pipeline on a single frame.
 
    Concurrency model
    -----------------
    Lane detection and pedestrian (YOLO) detection are submitted to the shared
    thread pool simultaneously.  On a machine with ≥ 2 logical cores they
    execute in parallel; even with GIL contention, OpenCV and PyTorch both
    release the GIL during their heavy compute sections, so real concurrency
    is achieved in practice.
 
    The sign CNN future is only submitted when the frame counter hits a
    multiple of _SIGN_INTERVAL (or force_sign=True).  On all other frames the
    previous cached result is returned instantly at zero cost.
 
    Parameters
    ----------
    frame       Raw BGR ndarray from cv2.VideoCapture / cv2.imread.
    force_sign  Set True to bypass the interval gate (e.g. first frame).
 
    Returns
    -------
    Annotated BGR ndarray — same shape as `frame`.
    """
    if frame is None:
        raise ValueError("Invalid frame: received None")
 
    _sign_state["counter"] += 1
    run_sign = force_sign or (_sign_state["counter"] % _SIGN_INTERVAL == 0)
 
    # ── 1. Dispatch independent tasks concurrently ────────────────────────────
    f_lanes = _executor.submit(detect_lanes_data, frame)
    f_peds  = _executor.submit(detect_pedestrians_data, frame)
    f_sign  = _executor.submit(predict_traffic_sign_frame, frame) if run_sign else None
 
    # ── 2. Collect results ────────────────────────────────────────────────────
    # .result() blocks until the future completes.  Because both f_lanes and
    # f_peds were submitted before either .result() call, they run in parallel
    # while this thread waits.
    lane_data   = f_lanes.result()
    pedestrians = f_peds.result()
 
    if f_sign is not None:
        sign_label, sign_conf, _, sign_bbox = f_sign.result()
        _sign_state["label"] = sign_label
        _sign_state["conf"]  = sign_conf
        _sign_state["bbox"]  = sign_bbox
    else:
        # Reuse the last good sign result
        sign_label = _sign_state["label"]
        sign_conf  = _sign_state["conf"]
        sign_bbox  = _sign_state.get("bbox", None)

    sign = (sign_label, sign_conf) if sign_label else None
 
    # ── 3. Fusion + Render (always sequential — depends on steps 1 & 2) ───────
    info   = fuse(lane_data, pedestrians, frame.shape)
    output = draw(frame, lane_data, pedestrians, sign, info)

    if lane_data: # If it found lanes
        if lane_data:
            for lane_id, lane_points in lane_data.items():
                draw_smooth_dashed_lane(output, lane_points)

    if sign_label and sign_bbox:
        x1, y1, x2, y2 = sign_bbox
        # Draw a bright red rectangle (BGR format: 0, 0, 255)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw the label and confidence above the box
        text = f"{sign_label} {sign_conf*100:.0f}%"
        cv2.putText(output, text, (x1, max(20, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
    return output
 
 
def reset_pipeline_state() -> None:
    """
    Reset per-video cached state.
 
    Call this before starting each new video so the sign CNN runs on frame 0
    instead of waiting _SIGN_INTERVAL frames to produce its first result.
    Also clears the stale label from any previous video.
    """
    _sign_state["counter"] = 0
    _sign_state["label"]   = None
    _sign_state["conf"]    = 0.0
 
 
def shutdown_pipeline() -> None:
    """
    Gracefully shut down the shared thread pool.
 
    Call once when the application exits (e.g. in the Tkinter on_closing
    handler).  Without this, Python's atexit may raise harmless but noisy
    exceptions on some platforms.
    """
    _executor.shutdown(wait=False)