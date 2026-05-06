import cv2

def draw_text_with_bg(img, text, pos, font_scale, text_color, bg_color, thickness=2):
    """Helper to draw high-contrast text with a background rectangle."""
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw dark semi-transparent background (optional: make it a solid block for now)
    cv2.rectangle(img, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

def draw(frame, lane_data, pedestrians, sign, info):
    output = frame.copy()
    vehicles = info.get("vehicles", []) if isinstance(info, dict) else []

    # 1. Draw Drivable Lane Area
    overlay = lane_data.get("overlay")
    if overlay is not None and overlay.any():
        output = cv2.addWeighted(output, 1.0, overlay, 0.4, 0)

    # 2. Draw Pedestrians
    for (x1, y1, x2, y2, conf) in pedestrians:
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Ped {conf:.2f}"
        draw_text_with_bg(output, label, (x1, max(20, y1 - 10)), 0.5, (0, 255, 0), (0, 0, 0), 1)

    # 3. Draw Vehicles
    for vehicle in vehicles:
        if len(vehicle) < 5: continue
        x1, y1, x2, y2, conf = vehicle[:5]
        label = vehicle[5] if len(vehicle) > 5 else "Vehicle"
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 2)
        draw_text_with_bg(output, f"{label} {float(conf):.2f}", (int(x1), max(20, int(y1) - 10)), 0.5, (255, 200, 0), (0, 0, 0), 1)

    # UI Overlay Offsets
    ui_y = 40

    # 4. Draw Traffic Sign
    if sign:
        label, conf = sign
        text = f"Sign: {label} ({conf*100:.1f}%)"
        draw_text_with_bg(output, text, (20, ui_y), 0.7, (255, 255, 0), (0, 0, 0))
        ui_y += 40

    # 5. Draw Direction
    if "direction" in info:
        draw_text_with_bg(output, f"Turn: {info['direction']}", (20, ui_y), 0.7, (0, 255, 255), (0, 0, 0))
        ui_y += 40

    # 6. Draw Warning
    if info.get("collision_warning"):
        draw_text_with_bg(output, "⚠ COLLISION WARNING", (20, ui_y), 0.8, (0, 0, 255), (20, 20, 20), 3)

    return output