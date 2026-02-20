import cv2
import numpy as np

def color_selection(image):
    """
    KAGGLE LOGIC: Filter for White and Yellow colors.
    """
    # Convert to HLS
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    # White Mask (High Lightness)
    # Kaggle used RGB>200. In HLS, L>190 is similar.
    lower_white = np.array([0, 190, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    
    # Yellow Mask (Hue 10-40)
    lower_yellow = np.array([10, 0, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    
    combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)
    return masked_image

def canny_edge_detector(image):
    """
    KAGGLE LOGIC: Canny Edges
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Kaggle parameters: 50, 150
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    
    # --- WIDE TRIANGLE FIX ---
    # We go from 0% to 100% width at the bottom to ensure we catch the Right Lane.
    # We stop at 85% height to cut off the dashboard.
    polygons = np.array([
        [
            (0, int(height * 0.85)),                 # Bottom Left (Corner)
            (width, int(height * 0.85)),             # Bottom Right (Corner)
            (int(width * 0.5), int(height * 0.55))   # Top Middle (Horizon)
        ]
    ])
    
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            # Draw Thick Green Lines (Thickness 10)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image

def detect_lanes_image(image_path):
    image = cv2.imread(image_path)
    if image is None: return None, "Error"
    
    lane_image = np.copy(image)
    
    # 1. Color Selection (Kaggle Logic)
    color_filtered = color_selection(lane_image)
    
    # 2. Canny Edges
    canny = canny_edge_detector(color_filtered)
    
    # 3. ROI Masking (Wider)
    cropped_image = region_of_interest(canny)
    
    # 4. Hough Transform
    # Kaggle used: threshold=15, min_line_length=40, max_line_gap=20
    # WE USE:      threshold=15, min_line_length=40, max_line_gap=150 (To connect dashes!)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 15, np.array([]), minLineLength=40, maxLineGap=150)
    
    if lines is None:
        # Debug View: Shows the edges inside the wide triangle
        debug_view = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
        return debug_view, "Warning: No lanes found. Showing Debug View."

    # 5. Draw and Combine
    line_image = display_lines(lane_image, lines)
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    
    return combo_image, "Success: Lanes Detected."