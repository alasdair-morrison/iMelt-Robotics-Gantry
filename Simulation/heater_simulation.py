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
from Simulation.hilbert_path_generator import generate_dynamic_hilbert_waypoints, fillet_corners

BED_SIZE_MM = 250.0
WORKING_RADIUS_MM = 75.0   # mm
TARGET_TEMP = 245.0
SAFTEY_MAX_TEMP = 270.0
MIN_SPEED = 10.0   # mm/s - Speed when over a cold spot (depositing heat)
MAX_SPEED = 250.0   # mm/s - Speed when over a hot spot (sprinting)
RASTER_STEP = 12.0 # mm - Distance between sweep passes
SPIRAL_STEP = 5.0  # mm - Distance between spiral loops
TOOL_RADIUS = 20.0 # mm - Radius to mask out the physical induction wand
ACTIVE_SHAPE = 'square'       # Set to 'circle' or 'square'
ACTIVE_PATTERN = 'hilbert' # Set to 'lawnmower' or 'concentric' or 'spiral' or 'hilbert'
MARGIN_MM = 5.0              # Safety air-gap from the physical wall

class ThermalTwin:
    def __init__(self, size_mm=BED_SIZE_MM, resolution_mm=1.0, radius_mm=WORKING_RADIUS_MM, shape=ACTIVE_SHAPE):
        self.size = int(size_mm / resolution_mm)
        self.center = self.size // 2
        self.radius = int(radius_mm / resolution_mm)
        self.lock = threading.Lock()

        # State 1: True internal temperature (Tin + Polyplastic composite core)
        self.grid_core = np.full((self.size, self.size), 25.0, dtype=np.float32)
        # State 2: Outer surface temperature (High-temp Silicone Rubber diaphragm)
        self.grid_surface = np.full((self.size, self.size), 25.0, dtype=np.float32)
        # Alias self.grid to grid_core so existing path/control methods read internal temp
        self.grid = self.grid_core

        # Pixel-wise 2x2 Covariance matrices: P_cc (core variance), P_cs (cross), P_ss (surface variance)
        self.P_cc = np.full((self.size, self.size), 4.0, dtype=np.float32)
        self.P_cs = np.full((self.size, self.size), 1.0, dtype=np.float32)
        self.P_ss = np.full((self.size, self.size), 4.0, dtype=np.float32)

        # Kalman noise parameters (°C^2)
        self.q_core = 0.25      # Process noise variance for core heating/conduction
        self.q_surf = 0.10      # Process noise variance for silicone surface
        self.r_cam = 1.50       # IR camera measurement noise variance

        # Circular working boundary mask
        self.boundary_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        if shape == 'circle': 
            cv2.circle(self.boundary_mask, (self.center, self.center), self.radius, 1, thickness=-1) 
        elif shape == 'square': 
            margin_px = int(MARGIN_MM / resolution_mm) 
            cv2.rectangle(self.boundary_mask, (margin_px, margin_px), (self.size - margin_px, self.size - margin_px), 1, thickness=-1 )

    def kalman_update_from_camera(self, raw_thermal_frame, homography_matrix, g_x, g_y):
        """
        Kalman Filter Measurement Update: Uses IR camera readings of the outer 
        silicone surface (T_s) to correct BOTH T_s and the hidden internal core temp (T_c).
        """
        top_down_map = cv2.warpPerspective(
            raw_thermal_frame, homography_matrix, (self.size, self.size), borderValue=25.0
        )
        cam_mask = np.ones(raw_thermal_frame.shape[:2], dtype=np.uint8)
        valid_cam_area = cv2.warpPerspective(cam_mask, homography_matrix, (self.size, self.size), borderValue=0)

        # Mask out pixels occluded by the physical induction wand
        valid_mask = np.ones((self.size, self.size), dtype=np.uint8)
        if 0 <= g_x < self.size and 0 <= g_y < self.size:
            cv2.circle(valid_mask, (int(g_x), int(g_y)), int(TOOL_RADIUS), 0, thickness=-1)

        with self.lock:
            obs = (valid_mask == 1) & (valid_cam_area == 1) & (self.boundary_mask == 1)
            
            # Innovation (measurement residual): y = z_cam - T_surface
            innovation = top_down_map[obs] - self.grid_surface[obs]
            
            # Innovation covariance: S = P_ss + R_cam
            S = self.P_ss[obs] + self.r_cam
            
            # Kalman Gains for hidden core (K_c) and observed silicone surface (K_s)
            K_c = self.P_cs[obs] / S
            K_s = self.P_ss[obs] / S

            # Update temperature states
            self.grid_core[obs] += K_c * innovation
            self.grid_surface[obs] += K_s * innovation

            # Update error covariance matrix P = (I - K*H) * P
            p_cc_old = self.P_cc[obs].copy()
            p_cs_old = self.P_cs[obs].copy()
            p_ss_old = self.P_ss[obs].copy()

            self.P_cc[obs] = p_cc_old - K_c * p_cs_old
            self.P_cs[obs] = (1.0 - K_s) * p_cs_old
            self.P_ss[obs] = (1.0 - K_s) * p_ss_old

    def simulate_noisy_camera_reading(self, g_x, g_y):
        """
        For simulation testing without hardware: generates a synthetic noisy IR camera 
        frame of the outer silicone surface and feeds it into the Kalman update.
        """
        with self.lock:
            synthetic_ir = self.grid_surface + np.random.normal(0.0, math.sqrt(self.r_cam), self.grid_surface.shape).astype(np.float32)
        identity_H = np.eye(3, dtype=np.float32)
        self.kalman_update_from_camera(synthetic_ir, identity_H, g_x, g_y)

    def get_local_temperature(self, x, y, layer='core'):
        ix = int(np.clip(x, 0, self.size - 1))
        iy = int(np.clip(y, 0, self.size - 1))
        with self.lock:
            target_grid = self.grid_core if layer == 'core' else self.grid_surface
            return float(target_grid[iy, ix])

class HeaterSimulation:
    def __init__(self, heater_radius_mm=TOOL_RADIUS):
        self.thermal_twin = ThermalTwin()
        self.heater_radius = heater_radius_mm  

    def apply_heating(self, g_x, g_y, voltage, dt=0.1):
        """
        2-Layer Electro-Thermal Prediction Step:
        - Layer 1 (Core): Tin + Polyplastic composite (receives induction eddy-current heating).
        - Layer 2 (Surface): High-temp Silicone Rubber diaphragm (insulates core, observed by IR camera).
        Also propagates the 2x2 Kalman filter state covariance matrices (P_cc, P_cs, P_ss).
        """
        # Tin + Polyplastic Composite Core Properties
        # Rule of mixtures (~30% Tin layers, ~70% Thermoplastic composite)
        rho_core = 3100.0       # kg/m^3 (effective composite density)
        cp_core = 950.0         # J/(kg*K) (effective specific heat)
        k_core = 18.0           # W/(m*K) (in-plane thermal conductivity boosted by tin layers)
        d_core = 0.0025         # 2.5 mm composite thickness (m)
        alpha_core = (k_core / (rho_core * cp_core)) * 1e6  # mm^2/s lateral diffusivity (~6.1 mm^2/s)

        # High-Temp Silicone Rubber Diaphragm Properties (Silex 1-3mm sheet)
        rho_sil = 1100.0        # kg/m^3
        cp_sil = 1250.0         # J/(kg*K)
        k_sil = 0.22            # W/(m*K) (low conductivity thermal insulator)
        d_sil = 0.0015          # 1.5 mm silicone diaphragm thickness (m)
        alpha_sil = (k_sil / (rho_sil * cp_sil)) * 1e6      # mm^2/s lateral diffusivity (~0.16 mm^2/s)

        # Inter-Layer & Boundary Heat Transfer
        # Vertical conductance U_int across Core <-> Silicone interface (W/(m^2*K))
        r_contact = 0.001       # Inter-ply contact resistance (m^2*K/W)
        u_int = 1.0 / ((d_core / (2.0 * k_core)) + r_contact + (d_sil / (2.0 * k_sil)))
        h_conv = 12.0           # Natural convection + radiation from outer silicone surface (W/(m^2*K))
        t_amb = 25.0

        # Areal heat capacities per 1mm x 1mm pixel (J/K)
        pixel_area_m2 = 1e-6
        C_core = rho_core * cp_core * d_core * pixel_area_m2
        C_sil = rho_sil * cp_sil * d_sil * pixel_area_m2

        # Fixed-voltage induction power deposited ONLY in the conductive Tin/Composite core
        eta = 0.80
        r_eq = 6.5
        power_watts = eta * (voltage ** 2) / r_eq

        size = self.thermal_twin.size
        y_indices, x_indices = np.ogrid[:size, :size]
        dist_sq = (x_indices - g_x) ** 2 + (y_indices - g_y) ** 2
        wand_mask = dist_sq <= (self.heater_radius ** 2)

        # Ring-peaked induction profile (models real pancake coil eddy-current distribution)
        r_dist = np.sqrt(dist_sq)
        r_peak = 0.60 * self.heater_radius   # Peak eddy current ring at 60% of coil radius
        ring_width = 0.30 * self.heater_radius
        spatial_profile = np.zeros((size, size), dtype=np.float32)
        spatial_profile[wand_mask] = np.exp(-((r_dist[wand_mask] - r_peak) ** 2) / (2.0 * (ring_width ** 2)))
        profile_sum = np.sum(spatial_profile)
        if profile_sum > 0:
            spatial_profile /= profile_sum

        # Sub-stepping for numerical stability of explicit Laplacian
        max_alpha = max(alpha_core, alpha_sil, 1e-6)
        max_stable_dt = 0.15 / max_alpha
        num_substeps = max(1, int(math.ceil(dt / max_stable_dt)))
        sub_dt = dt / num_substeps

        # Discrete state-transition coupling rates for Kalman covariance propagation
        a_c = (u_int * pixel_area_m2) / C_core
        a_s = (u_int * pixel_area_m2) / C_sil
        b_s = (h_conv * pixel_area_m2) / C_sil

        with self.thermal_twin.lock:
            Tc = self.thermal_twin.grid_core.copy()
            Ts = self.thermal_twin.grid_surface.copy()
            b_mask = self.thermal_twin.boundary_mask == 1

            for _ in range(num_substeps):
                lap_c = cv2.Laplacian(Tc, cv2.CV_32F, ksize=1)
                lap_s = cv2.Laplacian(Ts, cv2.CV_32F, ksize=1)

                # Heat flux from Tin/Plastic Core -> Outer Silicone Rubber
                q_vertical = u_int * pixel_area_m2 * (Tc - Ts)
                q_conv = h_conv * pixel_area_m2 * (Ts - t_amb)

                # Core temperature update (Induction + lateral diffusion - vertical transfer to silicone)
                dTc = ((power_watts * spatial_profile - q_vertical) / C_core + alpha_core * lap_c) * sub_dt
                # Silicone surface update (Vertical transfer from core + lateral diffusion - ambient cooling)
                dTs = ((q_vertical - q_conv) / C_sil + alpha_sil * lap_s) * sub_dt

                Tc += dTc
                Ts += dTs
                Tc[~b_mask] = t_amb
                Ts[~b_mask] = t_amb

            self.thermal_twin.grid_core[:] = Tc
            self.thermal_twin.grid_surface[:] = Ts

            # Kalman Filter Covariance Prediction: P = F * P * F^T + Q
            f11 = 1.0 - a_c * dt
            f12 = a_c * dt
            f21 = a_s * dt
            f22 = 1.0 - (a_s + b_s) * dt

            p_cc, p_cs, p_ss = self.thermal_twin.P_cc, self.thermal_twin.P_cs, self.thermal_twin.P_ss
            self.thermal_twin.P_cc = (f11**2)*p_cc + 2*f11*f12*p_cs + (f12**2)*p_ss + self.thermal_twin.q_core * dt
            self.thermal_twin.P_cs = f11*f21*p_cc + (f11*f22 + f12*f21)*p_cs + f12*f22*p_ss
            self.thermal_twin.P_ss = (f21**2)*p_cc + 2*f21*f22*p_cs + (f22**2)*p_ss + self.thermal_twin.q_surf * dt

        # Simulate noisy IR camera measurement & Kalman correction step
        self.thermal_twin.simulate_noisy_camera_reading(g_x, g_y)
    
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

    @staticmethod
    def generate_spiral_path(size_mm=BED_SIZE_MM, step_mm=12.0, radius_mm=WORKING_RADIUS_MM, puck_radius=TOOL_RADIUS, margin_mm=MARGIN_MM):
        """
        Generates a continuous Archimedean spiral from the center out to safe_max_radius,
        followed by a full perimeter equalization ring to combat edge heat losses.
        """
        waypoints = []
        cx, cy = size_mm / 2.0, size_mm / 2.0
        safe_max_radius = radius_mm - puck_radius - margin_mm
        if safe_max_radius <= 0:
            return waypoints

        # Archimedean spiral: r(theta) = b * theta, where b = step_mm / (2 * pi)
        b = step_mm / (2.0 * math.pi)
        theta_max = safe_max_radius / b
        arc_step_mm = 4.0  # Constant 4mm spacing along the spiral curve

        theta = 0.0
        while theta <= theta_max:
            r = b * theta
            x = cx + r * math.cos(theta)
            y = cy + r * math.sin(theta)
            waypoints.append([x, y])
            # Advance theta to keep constant linear step ds = sqrt(b^2 + r^2) * dtheta
            dtheta = arc_step_mm / math.sqrt(b**2 + r**2)
            theta += dtheta

        # Add 1 full perimeter pass right at safe_max_radius to boost the cold outer ring
        num_rim_pts = int(math.ceil((2.0 * math.pi * safe_max_radius) / arc_step_mm))
        for i in range(num_rim_pts):
            angle = theta_max + (2.0 * math.pi * i) / num_rim_pts
            waypoints.append([cx + safe_max_radius * math.cos(angle),
                              cy + safe_max_radius * math.sin(angle)])

        return waypoints

    @staticmethod
    def generate_hilbert_path(size_mm=BED_SIZE_MM, puck_radius=TOOL_RADIUS, margin_mm=MARGIN_MM):
        """ Generates a dynamically scaled, filleted Hilbert space-filling curve conforming to a square workpiece boundary. """
        # Compute dynamic waypoints, optimal order n, and pitch
        raw_pts, order, margin, pitch = generate_dynamic_hilbert_waypoints( sample_size=size_mm, induction_radius=puck_radius, safety_clearance=margin_mm )
        # Insert circular arc fillets at 90-degree corners to allow continuous gantry feed rates without corner deceleration stalls
        fillet_r = min(3.5, pitch / 3.0)
        smooth_waypoints = fillet_corners(raw_pts, fillet_radius=fillet_r, arc_resolution=5)
        return smooth_waypoints

    def execute_path(self, waypoints, speed_min=MIN_SPEED, speed_max=MAX_SPEED, target_temp=TARGET_TEMP, voltage=70.0):
        """
        Inverse-speed (linear energy) controller with radial perimeter compensation.
        """
        step_resolution_mm = 4.0
        size = self.thermal_twin.size
        cx, cy = size / 2.0, size / 2.0
        safe_max_radius = max(1.0, self.thermal_twin.radius - self.heater_radius - MARGIN_MM)
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
                dist_sq = (x_indices - x) ** 2 + (y_indices - y) ** 2
                wand_mask = dist_sq <= (self.heater_radius ** 2)
                with self.thermal_twin.lock:
                    footprint_max_temp = float(np.max(self.thermal_twin.grid[wand_mask]))
                    footprint_mean_temp = float(np.mean(self.thermal_twin.grid[wand_mask]))

                # If any pixel under the wand is within 2°C of target_temp, sprint at max speed
                if footprint_max_temp >= (target_temp - 2.0):
                    speed = speed_max
                else:
                    effective_temp = 0.6 * footprint_max_temp + 0.4 * footprint_mean_temp
                    delta_temp = max(0.0, target_temp - effective_temp)
                    temp_ratio = np.clip(delta_temp / (target_temp - 25.0), 0.0, 1.0)

                    # Radial compensation: boost dwell time by up to 1.6x near the outer perimeter
                    r_norm = min(1.0, math.hypot(x - cx, y - cy) / safe_max_radius)
                    radial_boost = 0.70 + 0.90 * (r_norm ** 2)  # Less heat in center (0.7x), more at rim (1.6x)

                    # Interpolate INVERSE speed (s/mm) so energy deposited scales linearly with temp_ratio
                    min_dwell_s_per_mm = 1.0 / speed_max
                    max_dwell_s_per_mm = 1.0 / speed_min
                    desired_dwell = min_dwell_s_per_mm + (max_dwell_s_per_mm - min_dwell_s_per_mm) * (temp_ratio ** 1.5) * radial_boost
                    speed = float(np.clip(1.0 / desired_dwell, speed_min, speed_max))

                dt = ds / speed
                self.apply_heating(x, y, voltage=voltage, dt=dt)

    def cold_point_loop(self, speed_min=MIN_SPEED, speed_max=MAX_SPEED, target_temp=TARGET_TEMP, voltage=5.0, max_iterations=25):
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
        """ Main loop displaying both Internal Core Temp and Outer Silicone Surface Temp. """
        voltage = 70.0
        if ACTIVE_PATTERN == 'concentric':
            waypoints = self.generate_concentric_path(shape=ACTIVE_SHAPE)
        elif ACTIVE_PATTERN == 'spiral':
            waypoints = self.generate_spiral_path()
        elif ACTIVE_PATTERN == 'hilbert':
            waypoints = self.generate_hilbert_path()
        else:
            waypoints = self.generate_lawnmower_path(shape=ACTIVE_SHAPE)
        b_mask = self.thermal_twin.boundary_mask == 1

        size = self.thermal_twin.size
        cx, cy = size / 2.0, size / 2.0
        y_idx, x_idx = np.ogrid[:size, :size]
        reachable_radius = self.thermal_twin.radius - MARGIN_MM
        active_zone_mask = ((x_idx - cx)**2 + (y_idx - cy)**2) <= (reachable_radius**2)

        cmap = matplotlib.colormaps['hot'].copy()
        cmap.set_over('cyan')

        plt.ion()
        fig, (ax_core, ax_surf) = plt.subplots(1, 2, figsize=(12, 5))

        im_core = ax_core.imshow(
            self.thermal_twin.grid_core, cmap=cmap, interpolation='nearest',
            vmin=25.0, vmax=TARGET_TEMP + 5.0
        )
        fig.colorbar(im_core, ax=ax_core, label='Internal Core Temp (°C)', extend='max')
        ax_core.set_title("True Internal (Tin/Polyplastic)")

        im_surf = ax_surf.imshow(
            self.thermal_twin.grid_surface, cmap=cmap, interpolation='nearest',
            vmin=25.0, vmax=TARGET_TEMP + 5.0
        )
        fig.colorbar(im_surf, ax=ax_surf, label='Silicone Surface Temp (°C)', extend='max')
        ax_surf.set_title("Outer Silicone (IR Camera View)")

        plt.tight_layout()
        plt.show(block=False)

        avg_core = float(np.mean(self.thermal_twin.grid_core[active_zone_mask]))
        max_core_temp = float(np.max(self.thermal_twin.grid_core[b_mask]))
        while avg_core < (0.95 * TARGET_TEMP):
            self.execute_path(waypoints, speed_min=MIN_SPEED, speed_max=MAX_SPEED, target_temp=TARGET_TEMP, voltage=voltage)
            self.cold_point_loop(speed_min=MIN_SPEED, speed_max=MAX_SPEED, target_temp=TARGET_TEMP, voltage=voltage, max_iterations=30)

            avg_core = float(np.mean(self.thermal_twin.grid_core[active_zone_mask]))
            max_core_temp = float(np.max(self.thermal_twin.grid_core[b_mask]))
            avg_surf = float(np.mean(self.thermal_twin.grid_surface[active_zone_mask]))
            max_surf = float(np.max(self.thermal_twin.grid_surface[b_mask]))

            print(f"Core [Avg: {avg_core:.1f}°C, Max: {max_core_temp:.1f}°C] | "
                  f"Silicone Surface [Avg: {avg_surf:.1f}°C, Max: {max_surf:.1f}°C]")

            im_core.set_data(self.thermal_twin.grid_core)
            ax_core.set_title(f"Internal Core | Avg: {avg_core:.1f}°C | Max: {max_core_temp:.1f}°C")

            im_surf.set_data(self.thermal_twin.grid_surface)
            ax_surf.set_title(f"Silicone Surface | Avg: {avg_surf:.1f}°C | Max: {max_surf:.1f}°C")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.01)

        plt.ioff()
        plt.show()
        return True

heater_Sim = HeaterSimulation()
heater_Sim.run()