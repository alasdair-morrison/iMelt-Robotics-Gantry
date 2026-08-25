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

def get_hot_spot_centroid(temp_array, threshold=23.0):
    """
    Calculates the sub-pixel centroid of the hottest region above a given threshold.
    If threshold is less than 1.0 (e.g., 0.75), it is treated as a relative fraction of the maximum value.
    Otherwise, it is treated as an absolute temperature threshold.
    Returns the max temperature and the (X, Y) sub-pixel coordinates.
    """
    temp_array_float = temp_array.astype(np.float32)
    max_val = np.max(temp_array_float)
    
    if threshold < 1.0:
        actual_threshold = max_val * threshold
        if max_val < 3.0: 
            return None, None, None
    else:
        actual_threshold = threshold
    
    _, mask = cv2.threshold(temp_array_float, actual_threshold, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    
    M = cv2.moments(mask)
    
    if M["m00"] != 0:
        c_x = M["m10"] / M["m00"]
        c_y = M["m01"] / M["m00"]
        
        max_temp = np.max(temp_array[mask == 255])
        return max_temp, c_x, c_y
        
    return None, None, None

def get_cold_spot_centroid(temp_array, threshold=22.0):
    """
    Calculates the sub-pixel centroid of the coldest region below a given threshold.
    If threshold is less than 1.0 (e.g., 0.20), it is treated as a relative fraction 
    of the temperature range (e.g., isolating the bottom 20% of the heat spread).
    Otherwise, it is treated as an absolute temperature threshold.
    Returns the min temperature and the (X, Y) sub-pixel coordinates.
    """
    temp_array_float = temp_array.astype(np.float32)
    min_val = np.min(temp_array_float)
    max_val = np.max(temp_array_float)
    
    # If using relative thresholding
    if threshold < 1.0:
        # Calculate the threshold to capture the bottom X% of the temperature range
        actual_threshold = min_val + ((max_val - min_val) * threshold)
    else:
        actual_threshold = threshold
    
    # Create a binary mask of pixels BELOW the temperature threshold
    # cv2.THRESH_BINARY_INV makes pixels colder than the threshold white (255) 
    _, mask = cv2.threshold(temp_array_float, actual_threshold, 255, cv2.THRESH_BINARY_INV)
    mask = mask.astype(np.uint8)
    
    # Calculate image moments to find the center of mass of the cold signature
    M = cv2.moments(mask)
    
    # Ensure the area (m00) is not zero to prevent division by zero errors
    if M["m00"] != 0:
        # Calculate precise sub-pixel coordinates
        c_x = M["m10"] / M["m00"]
        c_y = M["m01"] / M["m00"]
        
        # Get the actual minimum temperature within the isolated region
        min_temp = np.min(temp_array[mask == 255])
        return min_temp, c_x, c_y
        
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

def calibrate_with_checkerboard(image_array, board_dims=(7, 7), square_size_mm=30.0, filename="transform_matrix.json"):
    """
    Finds a thermal checkerboard in the image and computes a highly accurate homography matrix.
    
    Inputs:
        image_array: The 2D temperature array (or radiometric counts) from the camera.
        board_dims: The number of INTERIOR corners on the checkerboard (columns, rows).
        square_size_mm: The physical size of one side of a printed square in millimeters.
    """
    # Calculate the 2nd and 98th percentiles to ignore extreme hot/cold noise spikes
    vmin, vmax = np.percentile(image_array, (2, 98))
    
    # Clip the array to these limits
    clipped_array = np.clip(image_array, a_min=vmin, a_max=vmax)
    
    # Normalize the clipped array to 8-bit (0-255)
    img_norm = cv2.normalize(clipped_array, None, 0, 255, cv2.NORM_MINMAX)
    gray_img = np.uint8(img_norm)
    gray_img = cv2.bitwise_not(gray_img)
    
    # --- DEBUG VIEW ---
    # This pops up a window showing exactly what OpenCV is trying to process.
    # If this window looks like a solid gray blob, your thermal delta is still too low.
    cv2.imshow("OpenCV Debug View", gray_img)
    cv2.waitKey(500) # Pause for half a second to let the window render
    # ------------------
    
    # Generate the ideal real-world coordinates for the checkerboard corners
    # This creates a grid of points like (0,0,0), (30,0,0), (60,0,0)...
    obj_points = np.zeros((board_dims[0] * board_dims[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:board_dims[0], 0:board_dims[1]].T.reshape(-1, 2)
    obj_points *= square_size_mm
    
    # Drop the Z-axis (since the gantry bed is flat) so it matches the 2D pixel coordinates
    pts_mm = obj_points[:, :2] 

    # Find the checkerboard corners in the thermal image
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray_img, board_dims, flags)
    
    if found:
        print("Checkerboard detected! Refining sub-pixel coordinates...")
        
        # Refine the corner detection to sub-pixel accuracy for maximum precision
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
        
        # Reshape the corners array to match the (N, 2) shape of pts_mm
        pts_pixel = corners_subpix.reshape(-1, 2)
        
        # Calculate the Homography matrix utilizing all points
        # RANSAC ignores any falsely detected corner outliers
        matrix, status = cv2.findHomography(pts_pixel, pts_mm, cv2.RANSAC, 5.0)
        
        # Save the matrix using your existing JSON logic
        if os.path.exists(filename):
            os.remove(filename)
        with open(filename, "w") as f:
            json.dump(matrix.tolist(), f)
            
        print("Checkerboard calibration complete. Matrix saved.")
        return matrix
    else:
        print("Failed to detect checkerboard. Ensure the thermal contrast is high enough.")
        return None