import os
import sys
import time
import math
import threading
import keyboard
import numpy as np
import tabu_controller as tc

# Force Matplotlib interactive backend
import matplotlib
matplotlib.use('Qt5Agg')  
import matplotlib.pyplot as plt

import PySpin
from zaber_motion import Units
from zaber_motion.ascii import Connection

# Ensure thermal_analysis can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import ir_camera_code.thermal_analysis as ta

# ==========================================
# GLOBAL THREAD-SAFE STATE MANAGEMENT
# ==========================================
state_lock = threading.Lock()
CONTINUE_RECORDING = True

GLOBAL_THERMAL_STATE = {
    'target_temp': 180.0,
    'current_min_temp': 25.0,
    'cold_centroid': None,  # (mm_x, mm_y)
    'hot_centroid': None,   # (mm_x, mm_y)
    'cold_centroids': [],  # List of (mm_x, mm_y) for multiple cold points
    'hot_centroids': [],   # List of (mm_x, mm_y) for multiple hot points
    'hotspot_intensity': 0.0 
}

def get_thermal_state():
    with state_lock:
        return GLOBAL_THERMAL_STATE.copy()

def update_thermal_state(new_data):
    with state_lock:
        GLOBAL_THERMAL_STATE.update(new_data)

# ==========================================
# GANTRY MOTION CONTROL
# ==========================================
CONTROL_HZ = 20                 
LOOP_DELAY = 1.0 / CONTROL_HZ
BASE_SPEED = 15.0               
MIN_SPEED = 2.0                 
MAX_SPEED = 40.0                
MAX_X, MAX_Y = 250.0, 250.0
K_p = 0.5   
K_d = 0.1   

def execute_constant_velocity_spiral(axis_x, axis_y, max_radius=120.0, pitch=15.0, v_target=15.0):
    """Executes a smooth Archimedean spiral with uniform linear surface velocity."""
    print("[MOTION] Starting Constant Velocity Spiral Phase...")
    k = pitch / (2 * math.pi)
    theta = 0.1 
    
    # Move to center
    axis_x.move_absolute(125.0, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    axis_y.move_absolute(125.0, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    axis_x.wait_until_idle()
    axis_y.wait_until_idle()
    
    while CONTINUE_RECORDING:
        r = k * theta
        if r > max_radius:
            print("[MOTION] Edge reached. Switching to Reactive Phase...")
            break
            
        # Calculate Angular Velocity for constant linear speed
        theta_dot = v_target / math.sqrt(r**2 + k**2)
        
        # Derive X and Y Cartesian Velocities
        r_dot = k * theta_dot
        v_x = r_dot * math.cos(theta) - r * theta_dot * math.sin(theta)
        v_y = r_dot * math.sin(theta) + r * theta_dot * math.cos(theta)
        
        axis_x.move_velocity(v_x, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        axis_y.move_velocity(v_y, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        
        theta += theta_dot * LOOP_DELAY
        time.sleep(LOOP_DELAY)
        
    axis_x.move_velocity(0, Units.VELOCITY_MILLIMETRES_PER_SECOND)
    axis_y.move_velocity(0, Units.VELOCITY_MILLIMETRES_PER_SECOND)

def reactive_thermal_loop(axis_x, axis_y):
    """Closed-loop phase. Steers via potential fields and modulates speed based on thermal error."""
    print("[MOTION] Entering Reactive Heating Phase...")
    prev_error = 0.0
    
    # Start tracking physical location from the end of the spiral
    curr_x = axis_x.get_position(Units.LENGTH_MILLIMETRES)
    curr_y = axis_y.get_position(Units.LENGTH_MILLIMETRES)
    
    while CONTINUE_RECORDING:
        state = get_thermal_state()
        
        # If no valid thermal signatures are detected, hold position
        if state['cold_centroid'] is None and state['hot_centroid'] is None:
            axis_x.move_velocity(0, Units.VELOCITY_MILLIMETRES_PER_SECOND)
            axis_y.move_velocity(0, Units.VELOCITY_MILLIMETRES_PER_SECOND)
            time.sleep(LOOP_DELAY)
            continue
            
        # --- PD Feedrate Control ---
        error = state['target_temp'] - state['current_min_temp']
        error_derivative = (error - prev_error) / LOOP_DELAY
        prev_error = error
        
        modulated_speed = BASE_SPEED - (K_p * error) + (K_d * error_derivative)
        current_speed = max(MIN_SPEED, min(MAX_SPEED, modulated_speed))
        
        # --- Potential Field Vectors ---
        dir_x, dir_y = 0.0, 0.0
        
        # Attraction to cold
        if state['cold_centroid'] is not None:
            v_cold_x = state['cold_centroid'][0] - curr_x
            v_cold_y = state['cold_centroid'][1] - curr_y
            dist_cold = math.hypot(v_cold_x, v_cold_y) + 1e-5
            dir_x += (v_cold_x / dist_cold) * 1.0
            dir_y += (v_cold_y / dist_cold) * 1.0
            
        # Repulsion from hot
        if state['hot_centroid'] is not None:
            v_hot_x = curr_x - state['hot_centroid'][0]
            v_hot_y = curr_y - state['hot_centroid'][1]
            dist_hot = math.hypot(v_hot_x, v_hot_y) + 1e-5
            w_hot = state['hotspot_intensity'] * 2.5 
            dir_x += (v_hot_x / dist_hot) * w_hot
            dir_y += (v_hot_y / dist_hot) * w_hot
            
        mag = math.hypot(dir_x, dir_y) + 1e-5
        heading_x, heading_y = dir_x / mag, dir_y / mag
        
        # --- Boundary Safety Limits ---
        if curr_x <= 5.0 and heading_x < 0: heading_x = 0
        if curr_x >= (MAX_X - 5.0) and heading_x > 0: heading_x = 0
        if curr_y <= 5.0 and heading_y < 0: heading_y = 0
        if curr_y >= (MAX_Y - 5.0) and heading_y > 0: heading_y = 0
        
        # --- Stream Kinematics ---
        command_vx = heading_x * current_speed
        command_vy = heading_y * current_speed
        
        axis_x.move_velocity(command_vx, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        axis_y.move_velocity(command_vy, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        
        curr_x += command_vx * LOOP_DELAY
        curr_y += command_vy * LOOP_DELAY
        time.sleep(LOOP_DELAY)

def reactive_thermal_loop_multipoint(axis_x, axis_y):
    """Closed-loop phase with multiple thermal points. Steers via potential fields and modulates speed based on thermal error."""
    print("[MOTION] Entering Reactive Heating Phase (Multipoint)...")
    prev_error = 0.0
    
    # Start tracking physical location from the end of the spiral
    curr_x = axis_x.get_position(Units.LENGTH_MILLIMETRES)
    curr_y = axis_y.get_position(Units.LENGTH_MILLIMETRES)
    
    while CONTINUE_RECORDING:
        state = get_thermal_state()
        
        # If no valid thermal signatures are detected, hold position
        if state['cold_centroids'] is None and state['hot_centroids'] is None:
            axis_x.move_velocity(0, Units.VELOCITY_MILLIMETRES_PER_SECOND)
            axis_y.move_velocity(0, Units.VELOCITY_MILLIMETRES_PER_SECOND)
            time.sleep(LOOP_DELAY)
            continue
            
        # --- PD Feedrate Control ---
        error = state['target_temp'] - state['current_min_temp']
        error_derivative = (error - prev_error) / LOOP_DELAY
        prev_error = error
        
        modulated_speed = BASE_SPEED - (K_p * error) + (K_d * error_derivative)
        current_speed = max(MIN_SPEED, min(MAX_SPEED, modulated_speed))
        
        # --- Potential Field Vectors ---
        dir_x, dir_y = 0.0, 0.0
        
        # Attraction to cold
        if state['cold_centroids'] is not None:
            for centroid in state['cold_centroids']:
                v_cold_x = centroid[0] - curr_x
                v_cold_y = centroid[1] - curr_y
                dist_cold = math.hypot(v_cold_x, v_cold_y) + 1e-5
                dir_x += (v_cold_x / dist_cold) * 1.0
                dir_y += (v_cold_y / dist_cold) * 1.0

        # Repulsion from hot
        if state['hot_centroids'] is not None:
            for centroid in state['hot_centroids']:
                v_hot_x = curr_x - centroid[0]
                v_hot_y = curr_y - centroid[1]
                dist_hot = math.hypot(v_hot_x, v_hot_y) + 1e-5
                dir_x -= (v_hot_x / dist_hot) * 1.0
                dir_y -= (v_hot_y / dist_hot) * 1.0

        # Normalize the direction vector
        mag = math.hypot(dir_x, dir_y) + 1e-5
        heading_x, heading_y = dir_x / mag, dir_y / mag
        
        # --- Boundary Safety Limits ---
        if curr_x <= 5.0 and heading_x < 0: heading_x = 0
        if curr_x >= (MAX_X - 5.0) and heading_x > 0: heading_x = 0
        if curr_y <= 5.0 and heading_y < 0: heading_y = 0
        if curr_y >= (MAX_Y - 5.0) and heading_y > 0: heading_y = 0
        
        # --- Stream Kinematics ---
        command_vx = heading_x * current_speed
        command_vy = heading_y * current_speed
        
        axis_x.move_velocity(command_vx, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        axis_y.move_velocity(command_vy, Units.VELOCITY_MILLIMETRES_PER_SECOND)
        
        curr_x += command_vx * LOOP_DELAY
        curr_y += command_vy * LOOP_DELAY
        time.sleep(LOOP_DELAY)

def gantry_worker(x, y):
    try:
        execute_constant_velocity_spiral(x, y)
        reactive_thermal_loop(x, y)
        #reactive_thermal_loop_multipoint(x, y)
    except Exception as e:
        print(f"[MOTION ERROR] {e}")

# ==========================================
# CAMERA ACQUISITION & PROCESSING
# ==========================================

def handle_close(evt):
    global CONTINUE_RECORDING
    CONTINUE_RECORDING = False
    print("Window closed. Stopping stream...")

def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
    global CONTINUE_RECORDING

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

        # Load Computer Vision Files
        if os.path.exists("background.npy"):
            background_Temp = ta.load_background(filename="background.npy")
        else:
            print("No background frame found. Please run the calibration script first.")
            return False
            
        if os.path.exists("transform_matrix.json"):
            transform_matrix = ta.load_transform_matrix("transform_matrix.json")
        else:
            print("No transform matrix found. Please run the calibration script first.")
            return False

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle('A700 Temperature Radiometric')
        fig.canvas.mpl_connect('close_event', handle_close)
        
        im_display = None
        hot_plot, = ax.plot([], [], marker='+', color='red', markersize=15, linestyle='None')
        cold_plot, = ax.plot([], [], marker='+', color='cyan', markersize=15, linestyle='None')
        hot_text = ax.text(0, 0, '', color='white', fontsize=12, weight='bold')
        cold_text = ax.text(0, 0, '', color='white', fontsize=12, weight='bold')

        print('Press Enter to stop streaming')

        while CONTINUE_RECORDING:
            try:
                image_result = cam.GetNextImage(1000)
                if image_result.IsIncomplete():
                    print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
                else:
                    image_data = image_result.GetNDArray()

                    # Apply Radiometric Math
                    image_Radiance = (image_data - J0) / J1
                    image_Temp = (B / np.log(R / ((image_Radiance / Emiss / Tau) - K2) + F)) - 273.15
                    
                    clean_temp_array = ta.subtract_background(image_Temp, background_Temp)
                    #"""
                    # Find Centroids
                    hot_max, h_px_x, h_px_y = ta.get_hot_spot_centroid(clean_temp_array, threshold=0.75)
                    cold_min, c_px_x, c_px_y = ta.get_cold_spot_centroid(clean_temp_array, threshold=0.15)

                    update_data = {
                        'target_temp': 180.0,
                        'current_min_temp': cold_min if cold_min is not None else 25.0,
                        'cold_centroid': None,
                        'hot_centroid': None,
                        'cold_centroids': [],
                        'hot_centroids': [],
                        'hotspot_intensity': 0.0
                    }

                    # Update Visuals and State
                    if hot_max is not None:
                        hot_mm_x, hot_mm_y = ta.get_mm_from_pixels(h_px_x, h_px_y, transform_matrix)
                        update_data['hot_centroid'] = (hot_mm_x, hot_mm_y)
                        update_data['hotspot_intensity'] = max(0.0, min(1.0, (hot_max - 160.0) / 20.0))
                        
                        hot_plot.set_data([h_px_x], [h_px_y])
                        hot_text.set_position((h_px_x + 5, h_px_y))
                        hot_text.set_text(f"{hot_max:.2f}°C")
                    else:
                        hot_plot.set_data([], [])
                        hot_text.set_text("")

                    if cold_min is not None:
                        cold_mm_x, cold_mm_y = ta.get_mm_from_pixels(c_px_x, c_px_y, transform_matrix)
                        update_data['cold_centroid'] = (cold_mm_x, cold_mm_y)
                        
                        cold_plot.set_data([c_px_x], [c_px_y])
                        cold_text.set_position((c_px_x + 5, c_px_y))
                        cold_text.set_text(f"{cold_min:.2f}°C")
                    else:
                        cold_plot.set_data([], [])
                        cold_text.set_text("")

                    """
                    hotSpots, coldSpots = True, True
                    update_data = {
                                    'target_temp': 180.0,
                                    'current_min_temp': None,
                                    'cold_centroid': None,
                                    'hot_centroid': None,
                                    'cold_centroids': [],
                                    'hot_centroids': [],
                                    'hotspot_intensity': 0.0
                                }
                    
                    while hotSpots or coldSpots:
                        hot_max, h_px_x, h_px_y = ta.get_hot_spot_centroid(clean_temp_array, threshold=0.75)
                        cold_min, c_px_x, c_px_y = ta.get_cold_spot_centroid(clean_temp_array, threshold=0.15)
                        update_data['current_min_temp'] = cold_min if cold_min is not None else 25.0
                        if hot_max is not None:
                            hotSpots = True
                        if cold_min is not None:
                            coldSpots = True
                        if hotSpots:
                            hot_mm_x, hot_mm_y = ta.get_mm_from_pixels(h_px_x, h_px_y, transform_matrix)
                            update_data['hot_centroids'].append((hot_mm_x, hot_mm_y))
                            update_data['hotspot_intensity'] = max(0.0, min(1.0, (hot_max - 160.0) / 20.0))
                            
                            hot_plot.set_data([h_px_x], [h_px_y])
                            hot_text.set_position((h_px_x + 5, h_px_y))
                            hot_text.set_text(f"{hot_max:.2f}°C")
                            remove_hot_spot(clean_temp_array, h_px_x, h_px_y, radius=10)
                        else:
                            hot_plot.set_data([], [])
                            hot_text.set_text("")
                        if coldSpots:
                            cold_mm_x, cold_mm_y = ta.get_mm_from_pixels(c_px_x, c_px_y, transform_matrix)
                            update_data['cold_centroids'].append((cold_mm_x, cold_mm_y))

                            cold_plot.set_data([c_px_x], [c_px_y])
                            cold_text.set_position((c_px_x + 5, c_px_y))
                            cold_text.set_text(f"{cold_min:.2f}°C")
                            remove_cold_spot(clean_temp_array, c_px_x, c_px_y, radius=10)
                        else:
                            cold_plot.set_data([], [])
                            cold_text.set_text("")
                    """

                    # Push to Motion Thread safely
                    update_thermal_state(update_data)

                    # Render Image Efficiently
                    if im_display is None:
                        im_display = ax.imshow(image_Temp, cmap='inferno', aspect='auto')
                        fig.colorbar(im_display, ax=ax, format='%.2f')
                    else:
                        im_display.set_data(image_Temp)
                        im_display.set_clim(vmin=np.min(image_Temp), vmax=np.max(image_Temp))

                plt.pause(0.001)

                if keyboard.is_pressed('ENTER'):
                    print('Program is closing...')
                    plt.close('all')
                    CONTINUE_RECORDING = False

                image_result.Release()

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False

        cam.EndAcquisition()

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        return False

    return True


def run_single_camera(cam):
    try:
        result = True
        nodemap_tldevice = cam.GetTLDeviceNodeMap()
        cam.Init()
        nodemap = cam.GetNodeMap()
        result &= acquire_and_display_images(cam, nodemap, nodemap_tldevice)
        cam.DeInit()
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
    return result

# ==========================================
# MAIN ENTRY POINT
# ==========================================

def main():
    result = True
    system = PySpin.System.GetInstance()
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    print('Number of cameras detected: %d' % num_cameras)

    if num_cameras == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        print('Not enough cameras!')
        input('Done! Press Enter to exit...')
        return False

    # Grab the first available camera
    cam = cam_list.GetByIndex(0)
    
    print('Running camera thermal control program')
    try:
        with Connection.open_serial_port("COM6") as connection:
            connection.enable_alerts()
            device_list = connection.detect_devices()
            device = device_list[0]
            
            x = device.get_axis(1)
            y = device.get_axis(2)
            
            print("[MOTION] Homing Gantry...")
            if not x.is_homed(): x.home(wait_until_idle=False)
            if not y.is_homed(): y.home(wait_until_idle=False)
            x.wait_until_idle()
            y.wait_until_idle()
            
            # Start the background gantry thread
            gantry_thread = threading.Thread(target=gantry_worker, args=(x, y), daemon=True)
            gantry_thread.start()
            
            # Start the blocking camera loop on the main thread
            result &= run_single_camera(cam)
            
        print("Gantry thread has been stopped. Control Program is exiting...")
        
    except Exception as e:
        print(f"[SYSTEM ERROR] Could not initialize Gantry: {e}")

    # Cleanup
    del cam
    cam_list.Clear()
    system.ReleaseInstance()
    return result

if __name__ == '__main__':
    main()