from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

from dl_module.lane_detection import detect_lanes_data
from dl_module.pedestrian_detection import detect_pedestrians_data
from dl_module.traffic_sign.predict import predict_traffic_sign_frame
from dl_module.traffic_light import detect_traffic_lights
from dl_module.fusion import fuse
from dl_module.renderer import draw

_executor = ThreadPoolExecutor(max_workers=4)

_SIGN_INTERVAL: int = 4

_sign_state: dict = {
    "counter": 0,
    "label":   None,
    "conf":    0.0,
}

def draw_smooth_dashed_lane(output, lane_points,
                            color=(0,255,0),
                            thickness=6):

    if lane_points is None or len(lane_points) < 6:
        return

    pts = lane_points.reshape(-1, 2)

    x = pts[:, 0]
    y = pts[:, 1]

    try:

        poly = np.polyfit(y, x, 2)

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

        overlay = output.copy()

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

def process_frame(frame: np.ndarray, *,
                  force_sign: bool = False,
                  run_lanes: bool = True,
                  run_peds: bool = True,
                  run_signs: bool = True,
                  run_lights: bool = True) -> np.ndarray:

    if frame is None:
        raise ValueError("Invalid frame: received None")

    _sign_state["counter"] += 1
    run_sign_this_frame = run_signs and (
        force_sign or (_sign_state["counter"] % _SIGN_INTERVAL == 0)
    )

    f_lanes = _executor.submit(detect_lanes_data, frame) if run_lanes else None
    f_peds  = _executor.submit(detect_pedestrians_data, frame) if run_peds else None
    f_sign  = _executor.submit(predict_traffic_sign_frame, frame) if run_sign_this_frame else None
    f_tl    = _executor.submit(detect_traffic_lights, frame) if run_lights else None

    lane_data   = f_lanes.result() if f_lanes is not None else {}
    pedestrians = f_peds.result()  if f_peds  is not None else []

    sign_label = None
    sign_conf  = 0.0
    sign_bbox  = None

    if f_sign is not None:
        sign_label, sign_conf, _, sign_bbox = f_sign.result()
        _sign_state["label"] = sign_label
        _sign_state["conf"]  = sign_conf
        _sign_state["bbox"]  = sign_bbox
    elif run_signs:

        sign_label = _sign_state["label"]
        sign_conf  = _sign_state["conf"]
        sign_bbox  = _sign_state.get("bbox", None)

    sign = (sign_label, sign_conf) if sign_label else None

    info   = fuse(lane_data, pedestrians, frame.shape)
    output = draw(frame, lane_data, pedestrians, sign, info)

    if run_lanes and lane_data:
        for lane_id, lane_points in lane_data.items():
            draw_smooth_dashed_lane(output, lane_points)

    if run_signs and sign_label and sign_bbox:
        x1, y1, x2, y2 = sign_bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{sign_label} {sign_conf*100:.0f}%"
        cv2.putText(output, text, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    traffic_lights = f_tl.result() if f_tl is not None else []
    _TL_COLORS = {
        "Red":    (0, 0, 255),
        "Yellow": (0, 255, 255),
        "Green":  (0, 255, 0),
    }
    for (tx1, ty1, tx2, ty2, state, tconf) in traffic_lights:
        color = _TL_COLORS.get(state, (200, 200, 200))
        cv2.rectangle(output, (tx1, ty1), (tx2, ty2), color, 2)
        tl_text = f"{state} {tconf*100:.0f}%"
        cv2.putText(output, tl_text, (tx1, max(20, ty1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output

def reset_pipeline_state() -> None:

    _sign_state["counter"] = 0
    _sign_state["label"]   = None
    _sign_state["conf"]    = 0.0

def shutdown_pipeline() -> None:

    _executor.shutdown(wait=False)
