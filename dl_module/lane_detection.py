import cv2
import numpy as np

# -------------------------------
# 1. COLOR FILTER (Same as yours)
# -------------------------------
def color_selection(image):
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower_white = np.array([0, 190, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)

    lower_yellow = np.array([10, 0, 100])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

# -------------------------------
# 2. EDGE DETECTION
# -------------------------------
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 50, 150)

# -------------------------------
# 3. REGION OF INTEREST
# -------------------------------
def region_of_interest(image):
    height, width = image.shape

    polygon = np.array([[
        (0, int(height * 0.85)),
        (width, int(height * 0.85)),
        (int(width * 0.5), int(height * 0.55))
    ]])

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(image, mask)

# -------------------------------
# 4. LINE SEPARATION + AVERAGING
# -------------------------------
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)

        # Avoid division by zero
        if x1 == x2:
            continue

        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Filter nearly horizontal lines
        if abs(slope) < 0.5:
            continue

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = make_coordinates(image, np.mean(left_fit, axis=0)) if left_fit else None
    right_line = make_coordinates(image, np.mean(right_fit, axis=0)) if right_fit else None

    return left_line, right_line

# -------------------------------
# 5. CREATE FULL LANE LINE
# -------------------------------
def make_coordinates(image, line_params):
    slope, intercept = line_params
    height, width, _ = image.shape

    y1 = height
    y2 = int(height * 0.6)

    # x = (y - b) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

# -------------------------------
# 6. DRAW FINAL LANES
# -------------------------------
def draw_lines(image, lines):
    line_image = np.zeros_like(image)

    if lines[0] is not None:
        cv2.line(line_image, tuple(lines[0][:2]), tuple(lines[0][2:]), (255, 0, 0), 10)

    if lines[1] is not None:
        cv2.line(line_image, tuple(lines[1][:2]), tuple(lines[1][2:]), (0, 0, 255), 10)

    return line_image

# -------------------------------
# 7. MAIN PIPELINE
# -------------------------------
def detect_lanes(image):
    lane_image = np.copy(image)

    color_filtered = color_selection(lane_image)
    edges = canny(color_filtered)
    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi,
        2,
        np.pi / 180,
        threshold=15,
        minLineLength=40,
        maxLineGap=150
    )

    averaged_lines = average_slope_intercept(lane_image, lines)

    line_image = draw_lines(lane_image, averaged_lines)

    combo = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)

    return combo


def detect_lanes_image(image_path):
    """Compatibility wrapper expected by the GUI import path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return detect_lanes(image)

def detect_lanes_data(image):
    lane_image = np.copy(image)

    color_filtered = color_selection(lane_image)
    edges = canny(color_filtered)
    roi = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        roi, 2, np.pi/180, 15,
        np.array([]), minLineLength=40, maxLineGap=150
    )

    lanes = average_slope_intercept(lane_image, lines)

    if lanes is None:
        return {"left_lane": None, "right_lane": None}

    left, right = lanes

    return {
        "left_lane": left,
        "right_lane": right
    }