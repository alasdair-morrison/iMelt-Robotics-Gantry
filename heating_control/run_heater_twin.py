import os
import sys
import time
import math
import threading
import keyboard
import numpy as np
import tabu_controller as tc
import thermal_twin as tt

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

def synthetic_thermal_worker(axis_x, axis_y):
    """
    Drop-in replacement for the PySpin / FLIR camera capture loop.
    Simulates thermal diffusion and pushes extracted centroids to the gantry loop.
    """
    twin = tt.NumericalThermalTwin(size_mm=250, res_mm=1.0, dt=LOOP_DELAY)
    
    while CONTINUE_RECORDING:
        # Query instantaneous position from Zaber controllers
        curr_x = axis_x.get_position(Units.LENGTH_MILLIMETRES)
        curr_y = axis_y.get_position(Units.LENGTH_MILLIMETRES)
        
        # Advance PDE solver by LOOP_DELAY (50ms)
        sim_temp_array = twin.step(curr_x, curr_y, heater_on=True)
        
        # Analyze synthetic frame using your existing thermal_analysis methods
        hot_max, h_x, h_y = ta.get_hot_spot_centroid(sim_temp_array, threshold=160.0)
        cold_min, c_x, c_y = ta.get_cold_spot_centroid(sim_temp_array, threshold=0.15)
        
        update_data = {
            'target_temp': 180.0,
            'current_min_temp': float(np.min(sim_temp_array)),
            'cold_centroid': (c_x, c_y) if c_x is not None else None,
            'hot_centroid': (h_x, h_y) if h_x is not None else None,
            'hotspot_intensity': max(0.0, min(1.0, (np.max(sim_temp_array) - 160.0) / 20.0))
        }
        
        # Safely publish to the motion controller
        update_thermal_state(update_data)
        time.sleep(LOOP_DELAY)

# ==========================================
# MAIN ENTRY POINT
# ==========================================

def main():
    result = True
    global CONTINUE_RECORDING
    print('Running thermal twin control program')
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
            
            # Start background worker threads
            gantry_thread = threading.Thread(target=gantry_worker, args=(x, y), daemon=True)
            gantry_thread.start()
            
            thermal_twin_thread = threading.Thread(target=synthetic_thermal_worker, args=(x, y), daemon=True)
            thermal_twin_thread.start()
            
            print("[SYSTEM] Simulation and motion running. Press Enter or Ctrl+C to stop...")
            
            # --- KEEP-ALIVE LOOP (Replaces the blocking camera call) ---
            while CONTINUE_RECORDING:
                if keyboard.is_pressed('ENTER'):
                    print("[SYSTEM] Enter pressed. Terminating run...")
                    break
                time.sleep(0.1)

            # Signal worker threads to halt before closing port
            CONTINUE_RECORDING = False
            
            # Give gantry motors a moment to decelerate to a stop
            x.move_velocity(0, Units.VELOCITY_MILLIMETRES_PER_SECOND)
            y.move_velocity(0, Units.VELOCITY_MILLIMETRES_PER_SECOND)
            time.sleep(0.2)
            
        print("Gantry thread has been stopped. Control Program is exiting...")
        
    except KeyboardInterrupt:
        print("\n[SYSTEM] Keyboard interrupt detected. Shutting down safely...")
    except Exception as e:
        print(f"[SYSTEM ERROR] Could not initialize Gantry: {e}")
        
    except Exception as e:
        print(f"[SYSTEM ERROR] Could not initialize Gantry: {e}")

    # Cleanup
    #del cam
    #cam_list.Clear()
    system.ReleaseInstance()
    return result

if __name__ == '__main__':
    main()