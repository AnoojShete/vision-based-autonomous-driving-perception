import cv2

def draw(frame, lane_data, pedestrians, sign, info):
    output = frame.copy()
    vehicles = info.get("vehicles", []) if isinstance(info, dict) else []

    # Draw lanes
    for lane in ["left_lane", "right_lane"]:
        line = lane_data.get(lane)
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 8)

    # Draw pedestrians
    for (x1, y1, x2, y2, conf) in pedestrians:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"Pedestrian {conf:.2f}", (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw vehicles, if upstream modules provide them.
    for vehicle in vehicles:
        if len(vehicle) < 5:
            continue
        x1, y1, x2, y2, conf = vehicle[:5]
        label = vehicle[5] if len(vehicle) > 5 else "Vehicle"
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
        cv2.putText(
            output,
            f"{label} {float(conf):.2f}",
            (int(x1), max(15, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            2,
        )

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