import os
import sys
import time
import threading
import numpy as np
import tabu_controller as tc
import cv2
import asyncio
import PySpin
import math
from zaber_motion import Units
from zaber_motion.ascii import Connection

# Ensure thermal_analysis can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import ir_camera_code.thermal_analysis as ta
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# GLOBAL CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BED_SIZE_MM = 250.0
WORKING_RADIUS_MM = 75.0   # mm
TARGET_TEMP = 50.0
MIN_SPEED = 10.0   # mm/s - Speed when over a cold spot (depositing heat)
MAX_SPEED = 40.0   # mm/s - Speed when over a hot spot (sprinting)
RASTER_STEP = 20.0 # mm - Distance between sweep passes
SPIRAL_STEP = 5.0  # mm - Distance between spiral loops
TOOL_RADIUS = 25.0 # mm - Radius to mask out the physical induction wand

class ThermalTwin:
    def __init__(self, size_mm=BED_SIZE_MM, resolution_mm=1.0, radius_mm=WORKING_RADIUS_MM):
        self.size = int(size_mm / resolution_mm)
        self.center = self.size // 2
        self.radius = int(radius_mm / resolution_mm)
        self.grid = np.full((self.size, self.size), 25.0, dtype=np.float32)
        self.lock = threading.Lock()
        
        # Pre-compute a static circular boundary mask
        self.boundary_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        cv2.circle(self.boundary_mask, (self.center, self.center), self.radius, 1, thickness=-1)

    def update_from_camera(self, raw_thermal_frame, homography_matrix, g_x, g_y):
        top_down_map = cv2.warpPerspective(
            raw_thermal_frame, 
            homography_matrix, 
            (self.size, self.size),
            borderValue=25.0
        )

        cam_mask = np.ones(raw_thermal_frame.shape[:2], dtype=np.uint8)
        valid_cam_area = cv2.warpPerspective(cam_mask, homography_matrix, (self.size, self.size), borderValue=0)

        valid_mask = np.ones((self.size, self.size), dtype=np.uint8)
        if 0 <= g_x < self.size and 0 <= g_y < self.size:
            cv2.circle(valid_mask, (int(g_x), int(g_y)), int(TOOL_RADIUS), 0, thickness=-1)

        with self.lock:
            # ONLY update pixels that are unoccluded AND inside the circular working boundary
            valid_pixels = (valid_mask == 1) & (valid_cam_area == 1) & (self.boundary_mask == 1)
            self.grid[valid_pixels] = (0.8 * top_down_map[valid_pixels]) + (0.2 * self.grid[valid_pixels])
    def get_local_temperature(self, x, y):
        """ Returns the temperature at a specific physical coordinate. """
        ix, iy = int(np.clip(x, 0, self.size-1)), int(np.clip(y, 0, self.size-1))
        with self.lock:
            return float(self.grid[iy, ix])

# Global instances
thermal_twin = ThermalTwin()
current_gantry_pos = [0.0, 0.0]  # [x, y]
system_running = True

def generate_circular_raster_path(size_mm=BED_SIZE_MM, step_mm=RASTER_STEP, radius_mm=WORKING_RADIUS_MM):
    """ Generates a deterministic zig-zag pattern constrained to a circular boundary. """
    waypoints = []
    
    # Calculate the exact center of the mapped area
    cx = size_mm / 2.0
    cy = size_mm / 2.0
    
    # The sweep only needs to run from the top of the circle to the bottom
    y_start = cy - radius_mm
    y_end = cy + radius_mm
    
    for y in np.arange(y_start, y_end + step_mm, step_mm):
        y_val = min(y, y_end)
        
        # Distance from the center Y
        dy = y_val - cy 
        
        # Prevent math domain errors at the exact poles
        if radius_mm**2 < dy**2:
            continue 
            
        # Calculate the X boundary at this specific Y height using Pythagorean theorem
        dx = math.sqrt(radius_mm**2 - dy**2)
        
        x_left = cx - dx
        x_right = cx + dx
        
        # Alternate left-to-right and right-to-left sweeping
        iteration = int(round((y_val - y_start) / step_mm))
        if iteration % 2 == 0:
            waypoints.append([x_left, y_val])
            waypoints.append([x_right, y_val])
        else:
            waypoints.append([x_right, y_val])
            waypoints.append([x_left, y_val])
            
    return waypoints

def generate_spiral_path(size_mm=BED_SIZE_MM, step_mm=SPIRAL_STEP, radius_mm=WORKING_RADIUS_MM):
    """ Generates a spiral path constrained to a circular boundary. """
    waypoints = []
    
    # Calculate the exact center of the mapped area
    cx = size_mm / 2.0
    cy = size_mm / 2.0
    
    # Start from the outer edge and spiral inward
    r = radius_mm
    angle = 0.0
    
    while r > 0:
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        
        # Only add points that are within the circular boundary
        if (x - cx)**2 + (y - cy)**2 <= radius_mm**2:
            waypoints.append([x, y])
        
        # Increment angle and decrease radius for the spiral effect
        angle += step_mm / r  # Adjust angle increment based on current radius
        r -= step_mm / (2 * math.pi)  # Decrease radius gradually
        
    return waypoints

def camera_thread_function():
    """ 
    Runs parallel to motion. Grabs frames, converts to temp, and updates the Twin.
    Keep all your specific PySpin initialization here.
    """
    global system_running, current_gantry_pos
    
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    print('Number of cameras detected: %d' % num_cameras)
    
    if num_cameras == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False
    cam = cam_list.GetByIndex(0)
    try:
        result = True
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        cam.Init()
        nodemap = cam.GetNodeMap()

        # Setup Buffer and Pixel Formats
        sNodemap = cam.GetTLStreamNodeMap()
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))

        node_pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode('PixelFormat'))
        node_pixel_format.SetIntValue(node_pixel_format.GetEntryByName('Mono16').GetValue())

        # Force Radiometric Mode
        node_IRFormat = PySpin.CEnumerationPtr(nodemap.GetNode('IRFormat'))
        node_IRFormat.SetIntValue(node_IRFormat.GetEntryByName('Radiometric').GetValue())

        node_bufferhandling_mode.SetIntValue(node_bufferhandling_mode.GetEntryByName('NewestOnly').GetValue())

        print('*** IMAGE ACQUISITION ***\n')
        try:
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            node_acquisition_mode.SetIntValue(node_acquisition_mode.GetEntryByName('Continuous').GetValue())
            
            cam.BeginAcquisition()
            print('Acquiring images...')

            # Retrieve Calibration constants from Camera Firmware
            R = PySpin.CFloatPtr(nodemap.GetNode('R')).GetValue()
            B = PySpin.CFloatPtr(nodemap.GetNode('B')).GetValue()
            F = PySpin.CFloatPtr(nodemap.GetNode('F')).GetValue()
            X = PySpin.CFloatPtr(nodemap.GetNode('X')).GetValue()
            A1 = PySpin.CFloatPtr(nodemap.GetNode('alpha1')).GetValue()
            A2 = PySpin.CFloatPtr(nodemap.GetNode('alpha2')).GetValue()
            B1 = PySpin.CFloatPtr(nodemap.GetNode('beta1')).GetValue()
            B2 = PySpin.CFloatPtr(nodemap.GetNode('beta2')).GetValue()
            J1 = PySpin.CFloatPtr(nodemap.GetNode('J1')).GetValue()
            J0 = PySpin.CIntegerPtr(nodemap.GetNode('J0')).GetValue()

            print(f"Calibration Constants Loaded. R={R}, B={B}, Gain={J1}, Offset={J0}")

            # Atmospheric Math (Radiometric Physics)
            Emiss, TRefl, TAtm, Humidity = 0.97, 293.15, 293.15, 0.55
            Dist, ExtOpticsTransmission = 2, 1
            TAtmC = TAtm - 273.15
            H2O = Humidity * np.exp(1.5587 + 0.06939 * TAtmC - 0.00027816 * TAtmC**2 + 0.00000068455 * TAtmC**3)
            Tau = X * np.exp(-np.sqrt(Dist) * (A1 + B1 * np.sqrt(H2O))) + (1 - X) * np.exp(-np.sqrt(Dist) * (A2 + B2 * np.sqrt(H2O)))
            r1 = ((1 - Emiss) / Emiss) * (R / (np.exp(B / TRefl) - F))
            r2 = ((1 - Tau) / (Emiss * Tau)) * (R / (np.exp(B / TAtm) - F))
            r3 = ((1 - ExtOpticsTransmission) / (Emiss * Tau * ExtOpticsTransmission)) * (R / (np.exp(B / TAtm) - F))
            K2 = r1 + r2 + r3

            if os.path.exists("transform_matrix.json"):
                transform_matrix = ta.load_transform_matrix("transform_matrix.json")
            else:
                print("No transform matrix found. Please run the calibration script first.")
                return False

            # Define active operating area in real-world mm (e.g. 20mm to 230mm within 250mm bed)
            OPERATING_SURFACE_MM = [
                [20.0, 20.0],
                [230.0, 20.0],
                [230.0, 230.0],
                [20.0, 230.0]
            ]

            # Inside acquire_and_display_images right after loading transform_matrix:
            transform_matrix = ta.load_transform_matrix("transform_matrix.json")
            if transform_matrix is not None:
                # Build binary ROI mask for 2D thermal searches
                roi_mask = ta.create_roi_mask((480, 640), transform_matrix, OPERATING_SURFACE_MM)
                
                # Project MM polygon back to pixels for live HUD display
                inv_matrix = np.linalg.inv(transform_matrix)
                closed_polygon_mm = np.array(OPERATING_SURFACE_MM + [OPERATING_SURFACE_MM[0]], dtype=np.float32).reshape(-1, 1, 2)
                roi_pixels = cv2.perspectiveTransform(closed_polygon_mm, inv_matrix).reshape(-1, 2)

            while system_running:
                try:
                                image_result = cam.GetNextImage(1000)
                                if image_result.IsIncomplete():
                                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                                else:
                                    image_data = image_result.GetNDArray()
                
                                    # Apply Radiometric Math
                                    image_Radiance = (image_data - J0) / J1
                                    image_Temp = (B / np.log(R / ((image_Radiance / Emiss / Tau) - K2) + F)) - 273.15
                                    # Get current gantry position
                                    gx, gy = current_gantry_pos[0], current_gantry_pos[1]
                                    
                                    # Update the Digital Twin
                                    thermal_twin.update_from_camera(image_Temp, transform_matrix, gx, gy)
                                    
                                    # Display the Twin via cv2.imshow for debugging
                                    # Normalize grid temps (20C to 120C) into 0-255 for the HUD
                                    with thermal_twin.lock:
                                        grid_copy = thermal_twin.grid.copy()
                                        
                                    norm_grid = np.clip((grid_copy - 20.0) * (255.0 / (120.0 - 20.0)), 0, 255).astype(np.uint8)
                                    twin_display = cv2.applyColorMap(norm_grid, cv2.COLORMAP_INFERNO)
                                    
                                    # Draw a green circle showing the physical operation boundary
                                    cx, cy = int(BED_SIZE_MM / 2), int(BED_SIZE_MM / 2)
                                    cv2.circle(twin_display, (cx, cy), int(WORKING_RADIUS_MM), (0, 255, 0), 2)
                                    
                                    # Draw a blue dot showing where the software thinks the gantry is
                                    cv2.circle(twin_display, (int(gx), int(gy)), 3, (255, 0, 0), -1)
                                    cv2.imshow("Thermal Twin", twin_display)
                                    cv2.waitKey(1)
                                    
                                    time.sleep(0.05) # ~20 FPS
                except PySpin.SpinnakerException as ex:
                                print('Error: %s' % ex)
                                return False
                
            cam.EndAcquisition()
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False
        cam.DeInit()
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
    # Cleanup
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    return result

async def execute_modulated_sweep(axis_x, axis_y):
    """ 
    The main control loop. Moves the gantry along the raster path, 
    modulating speed based on the Thermal Twin's data. 
    """
    global current_gantry_pos, system_running
    
    waypoints = generate_circular_raster_path()
    # waypoints = generate_spiral_path()
    print(f"Generated {len(waypoints)} sweep waypoints.")
    
    for target_x, target_y in waypoints:
        if not system_running:
            break
            
        print(f"Sweeping to: X:{target_x}, Y:{target_y}")
        
        # Command the Zaber axes to move to the waypoint asynchronously
        move_x_task = asyncio.create_task(axis_x.move_absolute_async(target_x, Units.LENGTH_MILLIMETRES))
        move_y_task = asyncio.create_task(axis_y.move_absolute_async(target_y, Units.LENGTH_MILLIMETRES))
        
        # While the movement tasks are running, constantly modulate speed based on the twin
        while not move_x_task.done() or not move_y_task.done():
            if not system_running:
                # Stop the axes if a keyboard interrupt occurred
                await axis_x.stop_async()
                await axis_y.stop_async()
                break
            
            # Update current physical position from encoders (Async prevents micro-stutters)
            current_gantry_pos[0] = await axis_x.get_position_async(Units.LENGTH_MILLIMETRES)
            current_gantry_pos[1] = await axis_y.get_position_async(Units.LENGTH_MILLIMETRES)
            
            # Read the temperature of the substrate directly underneath the coil
            local_temp = thermal_twin.get_local_temperature(current_gantry_pos[0], current_gantry_pos[1])
            
            # Calculate speed proportionality: 
            # If temp is low, ratio is near 0 -> speed drops to MIN_SPEED
            # If temp is near TARGET, ratio is near 1.0 -> speed hits MAX_SPEED
            temp_ratio = np.clip(local_temp / TARGET_TEMP, 0.0, 1.0)
            modulated_speed = MIN_SPEED + ((MAX_SPEED - MIN_SPEED) * temp_ratio)
            
            # Send live speed update to the axes asynchronously via the settings object
            await axis_x.settings.set_async("maxspeed", modulated_speed, Units.VELOCITY_MILLIMETRES_PER_SECOND)
            await axis_y.settings.set_async("maxspeed", modulated_speed, Units.VELOCITY_MILLIMETRES_PER_SECOND)
            
            await asyncio.sleep(0.05) # 20Hz control loop

    print("Sweep complete.")
    system_running = False

if __name__ == "__main__":
    # Start the camera mapping thread
    cam_thread = threading.Thread(target=camera_thread_function)
    cam_thread.start()

    async def main_motion_routine():
        global system_running
        with Connection.open_serial_port("COM6") as connection:
            connection.enable_alerts()
            device_list = connection.detect_devices()
            device = device_list[0]
            
            x = device.get_axis(2)
            y = device.get_axis(1)
            
            print("[MOTION] Homing Gantry...")
            # You can leave homing synchronous, it's safer to wait before starting
            x.home(wait_until_idle=False)
            y.home(wait_until_idle=False)
            x.wait_until_idle()
            y.wait_until_idle()
            
            # Start the deterministic gantry sweep
            await execute_modulated_sweep(x, y)

    try:
        # Run the async routine
        asyncio.run(main_motion_routine())
    except KeyboardInterrupt:
        print("\nInterrupt received. Shutting down...")
        system_running = False
        
    cam_thread.join()