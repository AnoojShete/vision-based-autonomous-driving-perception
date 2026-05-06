def fuse(lane_data, pedestrians, frame_shape):
    height, width = frame_shape[:2]
    info = {}
    frame_area = width * height

    # 1. Direction based on Bird's-Eye View bases
    left_base = lane_data.get("left_base")
    right_base = lane_data.get("right_base")

    if left_base is not None and right_base is not None:
        # Check if the warped center is drifting
        lane_center = (left_base + right_base) // 2
        warped_center = width // 2 

        if lane_center < warped_center - 40:
            info["direction"] = "LEFT"
        elif lane_center > warped_center + 40:
            info["direction"] = "RIGHT"
        else:
            info["direction"] = "STRAIGHT"

    # 2. Collision Warning (Depth/Area Based)
    warning = False
    for (x1, y1, x2, y2, _) in pedestrians:
        box_area = (x2 - x1) * (y2 - y1)
        # If a pedestrian takes up more than 8% of the camera's total area, they are dangerously close
        if box_area / frame_area > 0.08:
            warning = True

    info["collision_warning"] = warning
    return info