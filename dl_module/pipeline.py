from dl_module.lane_detection import detect_lanes_data
from dl_module.pedestrian_detection import detect_pedestrians_data
from dl_module.traffic_sign.predict import predict_traffic_sign_frame
from dl_module.fusion import fuse
from dl_module.renderer import draw
import numpy as np

def process_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("Invalid frame")

    # 1. Extract data
    lane_data = detect_lanes_data(frame)
    pedestrians = detect_pedestrians_data(frame)

    sign_label, sign_conf, _ = predict_traffic_sign_frame(frame)
    sign = (sign_label, sign_conf) if sign_label else None

    # 2. Fusion
    info = fuse(lane_data, pedestrians, frame.shape)

    # 3. Render
    output = draw(frame, lane_data, pedestrians, sign, info)

    return output