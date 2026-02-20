import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 Model once
try:
    yolo_model = YOLO("yolov8n.pt")
except:
    yolo_model = None

def get_birds_eye_view(image):
    """
    Transforms the image to a top-down 'Bird's Eye' view.
    Updated to use the FULL WIDTH of the camera to catch both lanes.
    """
    height, width = image.shape[:2]

    # SOURCE POINTS: WIDER TRAPEZOID
    # Bottom: 0 to Width (Uses the full bottom of the screen)
    # Top: Wider spread (40% to 60%) to see more of the road ahead
    src = np.float32([
        [int(width * 0.35), int(height * 0.60)],  # Top Left (Wider)
        [int(width * 0.65), int(height * 0.60)],  # Top Right (Wider)
        [0, int(height * 0.85)],                  # Bottom Left
        [width, int(height * 0.85)]               # Bottom Right
    ])

    # DESTINATION POINTS: A flat rectangle
    # We add a small padding (0.1) so the lanes don't touch the edge of the screen
    dst = np.float32([
        [int(width * 0.1), 0],
        [int(width * 0.9), 0],
        [int(width * 0.1), height],
        [int(width * 0.9), height]
    ])

    # Get the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warped = cv2.warpPerspective(image, M, (width, height))
    return warped, Minv

def color_threshold(image):
    """
    Isolate White and Yellow using HLS color space.
    """
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    # White Mask
    lower_white = np.array([0, 180, 0])
    upper_white = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    
    # Yellow Mask
    lower_yellow = np.array([10, 0, 90])
    upper_yellow = np.array([40, 255, 255])
    yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
    
    combined = cv2.bitwise_or(white_mask, yellow_mask)
    return combined

def sliding_window_search(binary_warped):
    """
    The Core 'Advanced' Logic: Finds lane pixels using a histogram and sliding windows.
    """
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Find the peak of the left and right halves of the histogram
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Hyperparameters
    nwindows = 9
    window_height = int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    current_leftx = leftx_base
    current_rightx = rightx_base
    margin = 100
    minpix = 50
    
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        
        # Identify window boundaries
        win_xleft_low = current_leftx - margin
        win_xleft_high = current_leftx + margin
        win_xright_low = current_rightx - margin
        win_xright_high = current_rightx + margin
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            current_leftx = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            current_rightx = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def run_advanced_pipeline(image_path):
    # 1. Load Image
    original_img = cv2.imread(image_path)
    if original_img is None: return None, "Error loading image"
    
    # --- PART A: LANE DETECTION ---
    try:
        # 2. Warp to Bird's Eye View
        warped_img, Minv = get_birds_eye_view(original_img)
        
        # 3. Color Thresholding
        binary_warped = color_threshold(warped_img)
        
        # 4. Sliding Window Search
        leftx, lefty, rightx, righty = sliding_window_search(binary_warped)
        
        # 5. Polynomial Fit (Curve Fitting)
        if len(leftx) > 0 and len(rightx) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            
            # Generate x and y values for plotting
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            # 6. Draw the Lane Polygon
            warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
            color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
            
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            
            # Draw the lane onto the warped blank image (Green)
            cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
            
            # 7. Unwarp back to original perspective
            newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0])) 
            result = cv2.addWeighted(original_img, 1, newwarp, 0.5, 0)
        else:
            result = original_img # Fallback if lanes not found
    except Exception as e:
        print(f"Lane detection failed: {e}")
        result = original_img

    # --- PART B: OBJECT DETECTION (YOLO) ---
    if yolo_model:
        # Run YOLO on the result image (Overlay boxes on top of lanes)
        # classes=[0, 1, 2, 3, 5, 7, 9] limits detection to:
        # Person, Bicycle, Car, Motorcycle, Bus, Truck, Traffic Light
        results = yolo_model.predict(result, classes=[0, 1, 2, 3, 5, 7, 9], conf=0.4, verbose=False)
        result = results[0].plot() # Draws the boxes

    return result, "Advanced Pipeline Success"