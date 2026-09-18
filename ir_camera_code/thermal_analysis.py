import os
import cv2
import numpy as np
import json

def save_background(background_frame, filename="background.npy"):
    """
    Saves the captured thermal background array to a binary .npy file for later use.
    """
    if background_frame is not None:
        np.save(filename, background_frame)
        print(f"Background saved successfully to {filename}")

def load_background(filename="background.npy"):
    """
    Loads the thermal background array from a .npy file.
    Returns None if the file does not exist.
    """
    if os.path.exists(filename):
        print(f"Loading background from {filename}...")
        return np.load(filename)
    print(f"Background file {filename} not found.")
    return None

def subtract_background(current_frame, background_frame):
    """
    Subtracts a static thermal baseline (e.g., warm stepper motors) from the current frame.
    This eliminates static heat sources and isolates the calibration target.
    """
    if background_frame is None:
        return current_frame
    
    clean_frame = current_frame - background_frame
    return np.clip(clean_frame, a_min=0, a_max=None)

def create_roi_mask(frame_shape, transform_matrix, roi_mm_polygon):
    """
    Generates a 2D binary uint8 mask (255 inside ROI, 0 outside) by projecting
    real-world millimeter coordinates back into camera pixel space via inverse homography.
    
    Inputs:
        frame_shape: Tuple (height, width) of the FLIR thermal image array.
        transform_matrix: 3x3 perspective transformation matrix.
        roi_mm_polygon: List or array of [X, Y] millimeter coordinates defining the ROI.
                        e.g. [[10.0, 10.0], [240.0, 10.0], [240.0, 240.0], [10.0, 240.0]]
    """
    if transform_matrix is None or roi_mm_polygon is None:
        return None

    # Calculate inverse matrix (mm -> pixels)
    inv_matrix = np.linalg.inv(transform_matrix)
    
    # Format points for OpenCV perspectiveTransform
    pts_mm = np.array(roi_mm_polygon, dtype=np.float32).reshape(-1, 1, 2)
    pts_px = cv2.perspectiveTransform(pts_mm, inv_matrix)
    pts_px_int = np.int32(pts_px.reshape(-1, 2))

    # Create binary mask
    mask = np.zeros(frame_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts_px_int], 255)
    
    return mask

def get_hot_spot_centroid(temp_array, threshold=0.75, max_temp_cutoff=None, roi_mask=None):
    """
    Calculates the sub-pixel centroid of the hottest surface region above threshold,
    while filtering out the superheated inductive element (pixels >= max_temp_cutoff).

    Parameters:
        temp_array: 2D floating point temperature array in °C.
        threshold: Relative fraction (e.g. 0.75) or absolute temperature threshold.
        max_temp_cutoff: Temperature ceiling (°C). Pixels at or above this value 
                         (e.g., the coil body) are zeroed out.
        roi_mask: Binary 2D mask restricting the active bed workspace.
    """
    temp_array_float = temp_array.astype(np.float32)

    # 1. Restrict to Bed Operating Area (if ROI mask provided)
    if roi_mask is not None:
        temp_array_float = np.where(roi_mask == 255, temp_array_float, 0.0)

    # 2. Filter out Inductive Element Heat (Temperature Ceiling)
    if max_temp_cutoff is not None:
        # Zero out superheated coil pixels so they don't distort centroid or max search
        temp_array_float = np.where(temp_array_float < max_temp_cutoff, temp_array_float, 0.0)

    max_val = np.max(temp_array_float)
    if max_val < 3.0:  # Noise floor
        return None, None, None

    if threshold < 1.0:
        actual_threshold = max_val * threshold
    else:
        actual_threshold = threshold

    # 3. Create Binary Mask for Surface Hotspot
    _, mask = cv2.threshold(temp_array_float, actual_threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)

    # Ensure coil pixels remain suppressed in the binary mask
    if max_temp_cutoff is not None:
        coil_pixels = (temp_array >= max_temp_cutoff)
        mask[coil_pixels] = 0

    # 4. Calculate Center of Mass (Centroid) of Surface Hotspot
    M = cv2.moments(mask)
    if M["m00"] != 0:
        c_x = M["m10"] / M["m00"]
        c_y = M["m01"] / M["m00"]
        max_surface_temp = np.max(temp_array_float[mask == 255])
        return max_surface_temp, c_x, c_y

    return None, None, None

def create_combined_element_mask(frame_shape, transform_matrix, gantry_x_mm, gantry_y_mm, coil_radius_mm=20.0):
    """
    Creates a binary mask that blocks out the physical footprint of the coil 
    based on live gantry position (mm -> pixels via inverse homography).
    """
    if transform_matrix is None:
        return None

    inv_matrix = np.linalg.inv(transform_matrix)
    pt_mm = np.array([[[gantry_x_mm, gantry_y_mm]]], dtype=np.float32)
    pt_px = cv2.perspectiveTransform(pt_mm, inv_matrix)
    u_coil, v_coil = int(pt_px), int(pt_px[1])

    # Convert mm coil radius to approximate pixel radius
    pt_edge_mm = np.array([[[gantry_x_mm + coil_radius_mm, gantry_y_mm]]], dtype=np.float32)
    pt_edge_px = cv2.perspectiveTransform(pt_edge_mm, inv_matrix)
    r_px = int(np.hypot(pt_edge_px - u_coil, pt_edge_px[1] - v_coil))

    mask = np.ones(frame_shape, dtype=np.uint8) * 255
    cv2.circle(mask, (u_coil, v_coil), r_px, 0, -1)  # Zero out coil area
    return mask

def get_cold_spot_centroid(temp_array, threshold=0.05, roi_mask=None):
    """
    Calculates the sub-pixel centroid of the coldest localized region.
    Uses percentile thresholding to ignore extreme hot outliers.
    """
    temp_array_float = temp_array.astype(np.float32)

    # Ignore pixels outside the active ROI
    if roi_mask is not None:
        search_array = np.where(roi_mask == 255, temp_array_float, 999.0)
    else:
        search_array = temp_array_float

    # Flatten and remove excluded pixels to get the valid temperature distribution
    valid_pixels = search_array[search_array < 900.0]
    if len(valid_pixels) == 0:
        return None, None, None

    # Robust Percentile Thresholding
    if threshold < 1.0:
        # Isolate the absolute coldest X% of the board's physical pixels (e.g., 0.05 = 5th percentile)
        actual_threshold = np.percentile(valid_pixels, threshold * 100)
    else:
        actual_threshold = threshold

    # Create mask for pixels colder than the calculated percentile
    mask = np.zeros_like(temp_array_float, dtype=np.uint8)
    mask[(search_array <= actual_threshold)] = 255

    M = cv2.moments(mask)
    if M["m00"] != 0:
        c_x = M["m10"] / M["m00"]
        c_y = M["m01"] / M["m00"]
        
        # Sample the local temperature around the centroid (5x5 pixel patch)
        ix, iy = int(round(c_x)), int(round(c_y))
        h, w = temp_array_float.shape
        y1, y2 = max(0, iy - 2), min(h, iy + 3)
        x1, x2 = max(0, ix - 2), min(w, ix + 3)
        centroid_temp = float(np.mean(temp_array_float[y1:y2, x1:x2]))
        
        return centroid_temp, c_x, c_y

    return None, None, None

def remove_hot_spot(temp_array, h_px_x, h_px_y, radius=10):
    """
    Removes the hottest region above a given threshold from the thermal array
    This is to allow for detection of multiple hot spots in the same frame, such as when a heater is on and the gantry is moving.
    Adjustable radius allows for a larger or smaller area to be removed around the detected hot spot.
    """
    if h_px_x is not None and h_px_y is not None:
        # Create a mask to zero out the hot spot region
        mask = np.zeros_like(temp_array, dtype=np.uint8)
        cv2.circle(mask, (int(h_px_x), int(h_px_y)), radius, 255, -1)
        
        # Set the hot spot region to zero in the original array
        temp_array[mask == 255] = 0
        
    return temp_array

def remove_cold_spot(temp_array, c_px_x, c_px_y, radius=10):
    """
    Removes the coldest region below a given threshold from the thermal array.
    This is to allow for detection of multiple cold spots in the same frame, such as when a cold object is present.
    Adjustable radius allows for a larger or smaller area to be removed around the detected cold spot.
    """
    
    if c_px_x is not None and c_px_y is not None:
        # Create a mask to zero out the cold spot region
        mask = np.zeros_like(temp_array, dtype=np.uint8)
        cv2.circle(mask, (int(c_px_x), int(c_px_y)), radius, 255, -1)
        
        # Set the cold spot region to zero in the original array
        temp_array[mask == 255] = 0
        
    return temp_array

def calibrate_camera_perspective(pixel_points, mm_points, filename="transform_matrix.json"):
    """
    Calculates a 3x3 transformation matrix to convert pixels to mm, 
    accounting for camera tilt and perspective distortion.
    """
    pts_pixel = np.array(pixel_points, dtype=np.float32)
    pts_mm = np.array(mm_points, dtype=np.float32)
    
    pixel_points_avg = np.mean(pts_pixel, axis=0)
    print(f"Average pixel points: {pixel_points_avg}")
    
    matrix = cv2.getPerspectiveTransform(pixel_points_avg, pts_mm)
    
    if os.path.exists(filename):
        os.remove(filename)
        
    with open(filename, "w") as f:
        json.dump(matrix.tolist(), f)
        
    return matrix

def get_mm_from_pixels(pixel_x, pixel_y, matrix):
    """
    Converts a single (x, y) pixel coordinate to mm using the provided transformation matrix.
    
    Parameters:
    - pixel_x: The x-coordinate in pixels.
    - pixel_y: The y-coordinate in pixels.
    - matrix: A 3x3 numpy array representing the transformation matrix.
    
    Returns:
    - mm_x: The x-coordinate in millimeters.
    - mm_y: The y-coordinate in millimeters.
    """
    pt_pixel = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    pt_mm = cv2.perspectiveTransform(pt_pixel, matrix)
    mm_x, mm_y = pt_mm[0][0]
    
    return mm_x, mm_y

def load_transform_matrix(filename="transform_matrix.json"):
    """
    Loads the transformation matrix from a JSON file.
    """
    with open(filename, "r") as f:
        matrix = np.array(json.load(f), dtype=np.float32)
    return matrix

def calibrate_with_checkerboard(image_array, board_dims=(10, 8), square_size_mm=30.0, filename="transform_matrix.json"):
    """
    Finds a thermal checkerboard in the image and computes a highly accurate homography matrix.
    """
    # --- TUNE THESE TO ALIGN THE GREEN HUD BOX ---
    # The matrix currently anchors (0,0) to the first checkerboard corner.
    # Increase X to slide the green box LEFT across the image.
    # Increase Y to slide the green box UP across the image.
    OFFSET_X_MM = 0.0  
    OFFSET_Y_MM = 0.0  
    
    # If the box is drawn rotated 90-degrees compared to your bed, change to True
    SWAP_AXES = False
    # ---------------------------------------------
    
    vmin, vmax = np.percentile(image_array, (2, 98))
    clipped_array = np.clip(image_array, a_min=vmin, a_max=vmax)
    img_norm = cv2.normalize(clipped_array, None, 0, 255, cv2.NORM_MINMAX)
    gray_img = np.uint8(img_norm)
    gray_img = cv2.bitwise_not(gray_img)
    
    cv2.imshow("OpenCV Debug View", gray_img)
    cv2.waitKey(500) 
    
    obj_points = np.zeros((board_dims[0] * board_dims[1], 3), np.float32)
    grid = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)
    
    # Apply the offsets and axis orientation
    if SWAP_AXES:
        obj_points[:, 0] = (grid[:, 1] * square_size_mm) + OFFSET_X_MM
        obj_points[:, 1] = (grid[:, 0] * square_size_mm) + OFFSET_Y_MM
    else:
        obj_points[:, 0] = (grid[:, 0] * square_size_mm) + OFFSET_X_MM
        obj_points[:, 1] = (grid[:, 1] * square_size_mm) + OFFSET_Y_MM
        
    pts_mm = obj_points[:, :2]
    
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray_img, board_dims, flags)
    
    if found:
        print("Checkerboard detected! Refining sub-pixel coordinates...")
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
        
        pts_pixel = corners_subpix.reshape(-1, 2)
        matrix, status = cv2.findHomography(pts_pixel, pts_mm, cv2.RANSAC, 5.0)
        
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, "w") as f:
            json.dump(matrix.tolist(), f)
            
        print("Checkerboard calibration complete. Matrix saved.")
        return matrix
    else:
        print("Failed to detect checkerboard. Ensure the thermal contrast is high enough.")
        return None