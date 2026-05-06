import cv2
import numpy as np

def _build_ego_lane_polygon(lane_data):

    keys = sorted(lane_data.keys())

    polylines = []
    for k in keys:
        pts = lane_data[k]
        if pts is not None and len(pts) >= 2:
            polylines.append(pts.reshape(-1, 2))
    if len(polylines) < 2:
        return None

    left = polylines[0]
    right = polylines[1]

    polygon = np.vstack([left, right[::-1]])
    return polygon.reshape(-1, 1, 2).astype(np.int32)

def fuse(lane_data, pedestrians, frame_shape):
    height, width = frame_shape[:2]
    info = {}

    left_base = lane_data.get("left_base")
    right_base = lane_data.get("right_base")

    if left_base is not None and right_base is not None:

        lane_center = (left_base + right_base) // 2
        warped_center = width // 2

        if lane_center < warped_center - 40:
            info["direction"] = "LEFT"
        elif lane_center > warped_center + 40:
            info["direction"] = "RIGHT"
        else:
            info["direction"] = "STRAIGHT"

    warning = False
    ego_polygon = _build_ego_lane_polygon(lane_data)

    for det in pedestrians:
        x1, y1, x2, y2 = det[:4]

        bc_x = (x1 + x2) // 2
        bc_y = y2

        if ego_polygon is not None:

            dist = cv2.pointPolygonTest(
                ego_polygon,
                (float(bc_x), float(bc_y)),
                False,
            )
            if dist >= 0:
                warning = True
        else:

            frame_area = width * height
            box_area = (x2 - x1) * (y2 - y1)
            center_third_lo = width // 3
            center_third_hi = 2 * width // 3
            if (center_third_lo <= bc_x <= center_third_hi
                    and box_area / frame_area > 0.08):
                warning = True

    info["collision_warning"] = warning
    return info
