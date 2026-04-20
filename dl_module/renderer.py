import cv2

def draw(frame, lane_data, pedestrians, sign, info):
    output = frame.copy()

    # Draw lanes
    for lane in ["left_lane", "right_lane"]:
        line = lane_data.get(lane)
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 8)

    # Draw pedestrians
    for (x1, y1, x2, y2, conf) in pedestrians:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(output, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw traffic sign
    if sign:
        label, conf = sign
        cv2.putText(output, f"{label} ({conf*100:.1f}%)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

    # Draw direction
    if "direction" in info:
        cv2.putText(output, f"Direction: {info['direction']}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    # Draw warning
    if info.get("collision_warning"):
        cv2.putText(output, "⚠ COLLISION WARNING",
                    (200, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    return output