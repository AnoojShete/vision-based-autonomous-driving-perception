def fuse(lane_data, pedestrians, frame_shape):
    height, width = frame_shape[:2]
    info = {}

    # Direction
    left = lane_data.get("left_lane")
    right = lane_data.get("right_lane")

    if left is not None and right is not None:
        lane_center = (left[0] + right[0]) // 2
        frame_center = width // 2

        if lane_center < frame_center - 30:
            info["direction"] = "LEFT"
        elif lane_center > frame_center + 30:
            info["direction"] = "RIGHT"
        else:
            info["direction"] = "STRAIGHT"

    # Collision warning
    warning = False
    for (x1, y1, x2, y2, _) in pedestrians:
        if y2 > height * 0.7:
            warning = True

    info["collision_warning"] = warning

    return info