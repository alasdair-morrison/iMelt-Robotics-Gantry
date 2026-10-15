import os
import queue
import sys
import time
import matplotlib
import threading
import numpy as np
import cv2
import math
from zaber_motion import Units
from zaber_motion.ascii import Connection
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use TkAgg backend for matplotlib

BED_SIZE_MM = 250.0
WORKING_RADIUS_MM = 100.0   # mm
TARGET_TEMP = 200.0
MIN_SPEED = 10.0   # mm/s - Speed when over a cold spot (depositing heat)
MAX_SPEED = 250.0   # mm/s - Speed when over a hot spot (sprinting)
RASTER_STEP = 20.0 # mm - Distance between sweep passes
SPIRAL_STEP = 5.0  # mm - Distance between spiral loops
TOOL_RADIUS = 55.0 # mm - Radius to mask out the physical induction wand
ACTIVE_SHAPE = 'square'      # Set to 'circle' or 'square'
ACTIVE_PATTERN = 'lawnmower' # Set to 'lawnmower' or 'concentric'
MARGIN_MM = 5.0              # Safety air-gap from the physical wall

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

class HeaterSimulation:
    def __init__(self, heater_radius_mm=TOOL_RADIUS):
        self.thermal_twin = ThermalTwin()
        self.heater_radius = heater_radius_mm  

    def apply_heating(self, g_x, g_y, voltage, dt=0.1):
        """ Simulates the heating effect of the induction wand at a specific location. """
        # --- Physical & Tunable Parameters ---
        eta = 0.85
        r_eq = 5.0
        rho = 7850.0
        c_p = 480.0
        thickness_m = 0.005
        h_conv = 15.0
        t_amb = 25.0
        alpha_diff = 0.25  # mm^2 / s thermal diffusivity

        power_watts = eta * (voltage ** 2) / r_eq

        size = self.thermal_twin.size
        y_indices, x_indices = np.ogrid[:size, :size]
        dist_sq = (x_indices - g_x) ** 2 + (y_indices - g_y) ** 2

        wand_mask = dist_sq <= (self.heater_radius ** 2)
        sigma_mm = self.heater_radius / 2.0
        spatial_profile = np.zeros((size, size), dtype=np.float32)
        spatial_profile[wand_mask] = np.exp(-dist_sq[wand_mask] / (2.0 * (sigma_mm ** 2)))

        profile_sum = np.sum(spatial_profile)
        if profile_sum > 0:
            spatial_profile /= profile_sum

        pixel_volume_m3 = (1e-6) * thickness_m
        pixel_thermal_mass = rho * c_p * pixel_volume_m3

        # Sub-step dt so explicit diffusion never exceeds Von Neumann stability limit (alpha * sub_dt <= 0.20)
        max_stable_dt = 0.15 / max(alpha_diff, 1e-6)
        num_substeps = max(1, int(math.ceil(dt / max_stable_dt)))
        sub_dt = dt / num_substeps

        with self.thermal_twin.lock:
            grid = self.thermal_twin.grid.copy()
            b_mask = self.thermal_twin.boundary_mask == 1

            for _ in range(num_substeps):
                # ksize=1 uses the standard 5-point stencil: [[0,1,0],[1,-4,1],[0,1,0]]
                laplacian = cv2.Laplacian(grid, cv2.CV_32F, ksize=1)
                
                delta_T_heating = (power_watts * sub_dt / pixel_thermal_mass) * spatial_profile
                delta_T_diffusion = alpha_diff * laplacian * sub_dt
                delta_T_cooling = (h_conv * (1e-6) / pixel_thermal_mass) * (grid - t_amb) * sub_dt

                grid = grid + delta_T_heating + delta_T_diffusion - delta_T_cooling
                # Keep outside boundary fixed at ambient
                grid[~b_mask] = t_amb

            self.thermal_twin.grid = grid
    
    @staticmethod
    def generate_concentric_path(size_mm=BED_SIZE_MM, radius_mm=WORKING_RADIUS_MM, puck_radius=TOOL_RADIUS, margin_mm=5.0, shape='circle'):
        """ Generates concentric paths for either 'circle' or 'square' samples with boundary safety. """
        waypoints = []
        cx = size_mm / 2.0
        cy = size_mm / 2.0
        
        # Bounding limit guaranteeing the physical tool never touches the chamber walls
        safe_max_radius = radius_mm - puck_radius - margin_mm
        
        if safe_max_radius < 0:
            return waypoints 
            
        step_mm = puck_radius 
        radii = np.arange(0, safe_max_radius + 0.1, step_mm)
        arc_resolution = 5.0
        
        for r in radii:
            if r == 0:
                waypoints.append([cx, cy])
            else:
                if shape == 'circle':
                    num_points = int(math.ceil((2 * math.pi * r) / arc_resolution))
                    for i in range(num_points):
                        angle = (2 * math.pi * i) / num_points
                        x = cx + r * math.cos(angle)
                        y = cy + r * math.sin(angle)
                        waypoints.append([x, y])
                        
                elif shape == 'square':
                    # Determine how many points are needed per side to maintain constant surface velocity
                    side_points = int(math.ceil((2 * r) / arc_resolution))
                    if side_points < 1: 
                        side_points = 1
                    
                    # Top edge (Left to Right)
                    for x in np.linspace(cx - r, cx + r, side_points, endpoint=False):
                        waypoints.append([x, cy - r])
                    # Right edge (Top to Bottom)
                    for y in np.linspace(cy - r, cy + r, side_points, endpoint=False):
                        waypoints.append([cx + r, y])
                    # Bottom edge (Right to Left)
                    for x in np.linspace(cx + r, cx - r, side_points, endpoint=False):
                        waypoints.append([x, cy + r])
                    # Left edge (Bottom to Top)
                    for y in np.linspace(cy + r, cy - r, side_points, endpoint=False):
                        waypoints.append([cx - r, y])
                    
        return waypoints
    
    @staticmethod
    def generate_lawnmower_path(size_mm=BED_SIZE_MM, step_mm=RASTER_STEP, radius_mm=WORKING_RADIUS_MM, puck_radius=TOOL_RADIUS, margin_mm=5.0, shape='circle'):
        """ Generates a zig-zag raster constrained to either a 'circle' or 'square' boundary. """
        waypoints = []
        cx = size_mm / 2.0
        cy = size_mm / 2.0
        
        safe_max_radius = radius_mm - puck_radius - margin_mm
        
        if safe_max_radius < 0:
            return waypoints

        y_start = cy - safe_max_radius
        y_end = cy + safe_max_radius
        
        for y in np.arange(y_start, y_end + step_mm, step_mm):
            y_val = min(y, y_end)
            dy = y_val - cy 
            
            if shape == 'circle':
                if safe_max_radius**2 < dy**2:
                    continue 
                dx = math.sqrt(safe_max_radius**2 - dy**2)
            elif shape == 'square':
                dx = safe_max_radius
            else:
                raise ValueError("Shape argument must be 'circle' or 'square'")
                
            x_left = cx - dx
            x_right = cx + dx
            
            # Alternate sweep directions
            iteration = int(round((y_val - y_start) / step_mm))
            if iteration % 2 == 0:
                waypoints.append([x_left, y_val])
                waypoints.append([x_right, y_val])
            else:
                waypoints.append([x_right, y_val])
                waypoints.append([x_left, y_val])
                
        return waypoints

    def execute_path(self, waypoints, speed_min=MIN_SPEED, speed_max=MAX_SPEED, target_temp=TARGET_TEMP, voltage=5.0):
        """ Simulates the gantry moving along the specified waypoints and applying heat. """
        speed_K = speed_max - speed_min
        step_resolution_mm = 5.0
        size = self.thermal_twin.size
        y_indices, x_indices = np.ogrid[:size, :size]

        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i]
            x1, y1 = waypoints[i + 1]
            segment_dist = math.hypot(x1 - x0, y1 - y0)
            if segment_dist == 0:
                continue
            num_steps = max(1, int(math.ceil(segment_dist / step_resolution_mm)))

            xs = np.linspace(x0, x1, num_steps)
            ys = np.linspace(y0, y1, num_steps)
            ds = segment_dist / num_steps

            for x, y in zip(xs, ys):
                # Check both local center temp and peak temp under the wand's footprint
                dist_sq = (x_indices - x) ** 2 + (y_indices - y) ** 2
                wand_mask = dist_sq <= (self.heater_radius ** 2)
                with self.thermal_twin.lock:
                    footprint_max_temp = float(np.max(self.thermal_twin.grid[wand_mask]))
                    local_temp = float(self.thermal_twin.grid[int(np.clip(y, 0, size-1)), int(np.clip(x, 0, size-1))])

                # Control speed using a blend of local temp and footprint peak so we sprint if overlapping a hot zone
                effective_temp = 0.5 * local_temp + 0.5 * footprint_max_temp
                delta_temp = target_temp - effective_temp

                # Sprint at speed_max when approaching/exceeding target_temp; crawl at speed_min when cold
                temp_ratio = np.clip(delta_temp / (target_temp - 25.0), 0.0, 1.0)
                speed = speed_max - (speed_K * temp_ratio)
                speed = float(np.clip(speed, speed_min, speed_max))

                # True physical dwell time over distance ds: NO 0.05s minimum floor!
                dt = ds / speed
                self.apply_heating(x, y, voltage=voltage, dt=dt)

    def cold_point_loop(self, speed_min=MIN_SPEED, speed_max=MAX_SPEED, target_temp=TARGET_TEMP, voltage=5.0, max_iterations=10):
        """ Continuously identifies the coldest point and moves the gantry there to apply heat. """
        size = self.thermal_twin.size
        cx, cy = size / 2.0, size / 2.0
        safe_max_radius = max(0.0, self.thermal_twin.radius - self.heater_radius - MARGIN_MM)
        y_indices, x_indices = np.ogrid[:size, :size]

        for _ in range(max_iterations):
            with self.thermal_twin.lock:
                # Only look for cold points within the reachable heated zone
                reach_mask = ((x_indices - cx)**2 + (y_indices - cy)**2) <= ((safe_max_radius + self.heater_radius * 0.5)**2)
                masked_grid = np.where(reach_mask, self.thermal_twin.grid, np.inf)
                coldest_idx = np.unravel_index(np.argmin(masked_grid), masked_grid.shape)
                coldest_x, coldest_y = float(coldest_idx[1]), float(coldest_idx[0])
                local_temp = float(self.thermal_twin.grid[coldest_idx])

            # Clamp gantry target to safe mechanical radius
            dx, dy = coldest_x - cx, coldest_y - cy
            dist_from_center = math.hypot(dx, dy)
            if dist_from_center > safe_max_radius and dist_from_center > 0:
                scale = safe_max_radius / dist_from_center
                target_x = cx + dx * scale
                target_y = cy + dy * scale
            else:
                target_x, target_y = coldest_x, coldest_y

            # Check if the wand footprint at (target_x, target_y) is already at/above target_temp
            wand_mask = ((x_indices - target_x)**2 + (y_indices - target_y)**2) <= (self.heater_radius**2)
            with self.thermal_twin.lock:
                footprint_peak = float(np.max(self.thermal_twin.grid[wand_mask]))

            if footprint_peak >= target_temp:
                # Do not dwell here if part of the wand is already at 200°C!
                break

            delta_temp = target_temp - local_temp
            speed_K = speed_max - speed_min
            temp_ratio = np.clip(delta_temp / (target_temp - 25.0), 0.0, 1.0)
            speed = float(np.clip(speed_max - (speed_K * temp_ratio), speed_min, speed_max))

            # Dwell time over a 5mm virtual step at the calculated speed
            dt = 5.0 / speed
            self.apply_heating(target_x, target_y, voltage=voltage, dt=dt)

    def run(self):
        """ Main loop to run the heater simulation. """
        voltage = 70.0
        waypoints = self.generate_lawnmower_path()
        average_temp = np.mean(self.thermal_twin.grid)

        cmap = matplotlib.colormaps['hot'].copy()
        cmap.set_over('cyan')

        plt.ion()
        fig, ax = plt.subplots()
        im = ax.imshow(
            self.thermal_twin.grid, 
            cmap=cmap, 
            interpolation='nearest',
            vmin=25.0,          # Lock color scale minimum to ambient temp
            vmax=TARGET_TEMP + 5.0    # Lock color scale maximum to target temp
        )
        fig.colorbar(im, ax=ax, label='Temperature (°C)')
        ax.set_title("Thermal Twin Live Map")
        plt.show(block=False)

        while max(self.thermal_twin.grid[self.thermal_twin.boundary_mask == 1]) < TARGET_TEMP:
            self.execute_path(waypoints, speed_min=MIN_SPEED, speed_max=MAX_SPEED, target_temp=TARGET_TEMP, voltage=voltage)
            b_mask = self.thermal_twin.boundary_mask == 1
            average_temp = np.mean(self.thermal_twin.grid[b_mask])
            self.cold_point_loop(speed_min=MIN_SPEED, speed_max=MAX_SPEED, target_temp=TARGET_TEMP, voltage=voltage, max_iterations=5)
            b_mask = self.thermal_twin.boundary_mask == 1
            average_temp = np.mean(self.thermal_twin.grid[b_mask])
            print(f"Average Temperature: {average_temp:.2f} °C")
            im.set_data(self.thermal_twin.grid)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.01)
        plt.ioff()
        plt.show()
        return True

heater_Sim = HeaterSimulation()
heater_Sim.run()