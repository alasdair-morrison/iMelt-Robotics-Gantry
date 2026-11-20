import os
import queue
import sys
import csv
from datetime import datetime
import time
import matplotlib
import threading
import math
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')  # Use TkAgg backend for matplotlib
import os
import queue
import sys
import time
import threading
import numpy as np
import cv2
import asyncio
import PySpin
from zaber_motion import Units
from zaber_motion.ascii import Connection
import json

# Default Parameters
PORT = '/dev/ttyACM0'
# PORT = 'USB5'
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
ACTIVE_PATTERN = 'lissajous' # Set to 'more', 'hilbert', or 'lissajous'
MARGIN_MM = 5.0              # Safety air-gap from the physical wall

class IntegratedThermalTwin:
    def __init__(
        self,
        size_mm=250.0,
        resolution_mm=1.0,
        radius_mm=75.0,
        tool_radius=25.0,
        shape='circle',
    ):
        """
        Initializes the 2-Layer Thermal Twin with Kalman Filter Covariances.
        1 pixel = 1 mm by default.
        """
        self.size = int(size_mm / resolution_mm)
        self.center = self.size // 2
        self.radius = int(radius_mm / resolution_mm)
        self.tool_radius = int(tool_radius / resolution_mm)
        self.lock = threading.Lock()

        # State 1: True internal temperature (Tin + Polyplastic composite core)
        self.grid_core = np.full((self.size, self.size), 25.0, dtype=np.float32)
        # State 2: Outer surface temperature (High-temp Silicone Rubber diaphragm)
        self.grid_surface = np.full((self.size, self.size), 25.0, dtype=np.float32)
        
        # Alias for legacy compatibility
        self.grid = self.grid_core

        # Pixel-wise 2x2 Covariance matrices: P_cc (core), P_cs (cross), P_ss (surface)
        self.P_cc = np.full((self.size, self.size), 4.0, dtype=np.float32)
        self.P_cs = np.full((self.size, self.size), 1.0, dtype=np.float32)
        self.P_ss = np.full((self.size, self.size), 4.0, dtype=np.float32)

        # Kalman noise parameters (°C^2)
        self.q_core = 0.25      # Process noise variance for core heating
        self.q_surf = 0.10      # Process noise variance for silicone surface
        self.r_cam = 1.50       # IR camera measurement noise variance

        # Physical boundary mask (0 outside the clamping ring, 1 inside)
        self.boundary_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        shape = shape.lower()
        if shape == 'circle':
            cv2.circle(
                self.boundary_mask,
                (self.center, self.center),
                self.radius,
                1,
                thickness=-1,
            )
        elif shape == 'square':
            cv2.rectangle(
                self.boundary_mask,
                (self.center - self.radius, self.center - self.radius),
                (self.center + self.radius, self.center + self.radius),
                1,
                thickness=-1,
            )
        else:
            raise ValueError("shape must be either 'circle' or 'square'.")

        # --- Physical Material Properties ---
        # Composite Core (Tin + Thermoplastic)
        self.rho_core = 3100.0       # kg/m^3
        self.cp_core = 950.0         # J/(kg*K)
        self.k_core = 18.0           # W/(m*K)
        self.d_core = 0.0025         # 2.5 mm thickness
        self.alpha_core = (self.k_core / (self.rho_core * self.cp_core)) * 1e6  # mm^2/s

        # Silicone Rubber Diaphragm
        self.rho_sil = 1100.0        # kg/m^3
        self.cp_sil = 1250.0         # J/(kg*K)
        self.k_sil = 0.22            # W/(m*K)
        self.d_sil = 0.0015          # 1.5 mm thickness
        self.alpha_sil = (self.k_sil / (self.rho_sil * self.cp_sil)) * 1e6      # mm^2/s

        # Inter-Layer Coupling & Boundaries
        r_contact = 0.001            # Inter-ply contact resistance
        self.u_int = 1.0 / ((self.d_core / (2.0 * self.k_core)) + r_contact + (self.d_sil / (2.0 * self.k_sil)))
        self.h_conv = 12.0           # W/(m^2*K)
        self.t_amb = 25.0            # Ambient °C

        # Areal heat capacities per 1mm x 1mm pixel
        self.pixel_area_m2 = 1e-6
        self.C_core = self.rho_core * self.cp_core * self.d_core * self.pixel_area_m2
        self.C_sil = self.rho_sil * self.cp_sil * self.d_sil * self.pixel_area_m2

    def predict_step(self, g_x, g_y, voltage, dt):
        """
        Kalman Filter Prediction Step (Driven by Zaber Motion Thread).
        Calculates Joule heating, 2D diffusion, vertical heat transfer, and covariance propagation.
        """
        # Induction Power Calculation
        eta = 0.80
        r_eq = 6.5
        power_watts = eta * (voltage ** 2) / r_eq

        # Spatial Gaussian footprint of the induction coil
        y_indices, x_indices = np.ogrid[:self.size, :self.size]
        dist_sq = (x_indices - g_x) ** 2 + (y_indices - g_y) ** 2
        wand_mask = dist_sq <= (self.tool_radius ** 2)
        sigma_mm = self.tool_radius / 2.0
        
        spatial_profile = np.zeros((self.size, self.size), dtype=np.float32)
        spatial_profile[wand_mask] = np.exp(-dist_sq[wand_mask] / (2.0 * (sigma_mm ** 2)))
        
        profile_sum = np.sum(spatial_profile)
        if profile_sum > 0:
            spatial_profile /= profile_sum

        # Sub-stepping for explicit Laplacian numerical stability
        max_alpha = max(self.alpha_core, self.alpha_sil, 1e-6)
        max_stable_dt = 0.15 / max_alpha
        num_substeps = max(1, int(math.ceil(dt / max_stable_dt)))
        sub_dt = dt / num_substeps

        # Discrete coupling rates for covariance update
        a_c = (self.u_int * self.pixel_area_m2) / self.C_core
        a_s = (self.u_int * self.pixel_area_m2) / self.C_sil
        b_s = (self.h_conv * self.pixel_area_m2) / self.C_sil

        with self.lock:
            Tc = self.grid_core.copy()
            Ts = self.grid_surface.copy()
            b_mask = self.boundary_mask == 1

            # Physics Execution
            for i in range(num_substeps):
                # ksize=1 ensures standard stable 5-point stencil
                lap_c = cv2.Laplacian(Tc, cv2.CV_32F, ksize=1)
                lap_s = cv2.Laplacian(Ts, cv2.CV_32F, ksize=1)

                q_vertical = self.u_int * self.pixel_area_m2 * (Tc - Ts)
                q_conv = self.h_conv * self.pixel_area_m2 * (Ts - self.t_amb)

                dTc = ((power_watts * spatial_profile - q_vertical) / self.C_core + self.alpha_core * lap_c) * sub_dt
                dTs = ((q_vertical - q_conv) / self.C_sil + self.alpha_sil * lap_s) * sub_dt

                Tc += dTc
                Ts += dTs
                
                # Keep physical clamp edges clamped to ambient
                Tc[~b_mask] = self.t_amb
                Ts[~b_mask] = self.t_amb

            self.grid_core[:] = Tc
            self.grid_surface[:] = Ts

            # Covariance Matrix Propagation (P = F*P*F^T + Q)
            f11 = 1.0 - a_c * dt
            f12 = a_c * dt
            f21 = a_s * dt
            f22 = 1.0 - (a_s + b_s) * dt

            p_cc, p_cs, p_ss = self.P_cc, self.P_cs, self.P_ss
            self.P_cc = (f11**2)*p_cc + 2*f11*f12*p_cs + (f12**2)*p_ss + self.q_core * dt
            self.P_cs = f11*f21*p_cc + (f11*f22 + f12*f21)*p_cs + f12*f22*p_ss
            self.P_ss = (f21**2)*p_cc + 2*f21*f22*p_cs + (f22**2)*p_ss + self.q_surf * dt

    def measurement_update(self, raw_thermal_frame, homography_matrix, g_x, g_y):
        """
        Kalman Filter Update Step (Driven by PySpin Camera Thread).
        Uses IR camera readings of the outer silicone to correct BOTH the silicone and hidden core states.
        """
        # Warp perspective and fill unknown outside bounds with ambient
        top_down_map = cv2.warpPerspective(
            raw_thermal_frame, homography_matrix, (self.size, self.size), borderValue=25.0
        )
        
        # Extract the true valid bounds of the camera's physical vision
        cam_mask = np.ones(raw_thermal_frame.shape[:2], dtype=np.uint8)
        valid_cam_area = cv2.warpPerspective(cam_mask, homography_matrix, (self.size, self.size), borderValue=0)

        # Blackout mask for the physical toolhead to prevent Parallax lock-on
        valid_mask = np.ones((self.size, self.size), dtype=np.uint8)
        if 0 <= g_x < self.size and 0 <= g_y < self.size:
            cv2.circle(valid_mask, (int(g_x), int(g_y)), self.tool_radius, 0, thickness=-1)

        with self.lock:
            # Only apply observations inside chamber bounds, in camera view, and outside toolhead shadow
            obs = (valid_mask == 1) & (valid_cam_area == 1) & (self.boundary_mask == 1)
            
            # Innovation (Residual): z_cam - T_surface
            innovation = top_down_map[obs] - self.grid_surface[obs]
            
            # Innovation Covariance: S = P_ss + R_cam
            S = self.P_ss[obs] + self.r_cam
            
            # Kalman Gains for core (K_c) and surface (K_s)
            K_c = self.P_cs[obs] / S
            K_s = self.P_ss[obs] / S

            # State Updates
            self.grid_core[obs] += K_c * innovation
            self.grid_surface[obs] += K_s * innovation

            # Covariance Updates: P = (I - K*H) * P
            p_cc_old = self.P_cc[obs].copy()
            p_cs_old = self.P_cs[obs].copy()
            p_ss_old = self.P_ss[obs].copy()

            self.P_cc[obs] = p_cc_old - K_c * p_cs_old
            self.P_cs[obs] = (1.0 - K_s) * p_cs_old
            self.P_ss[obs] = (1.0 - K_s) * p_ss_old

    def get_local_temperature(self, x, y, layer='core'):
        """ Extracts instantaneous temperature at a specific coordinate for gantry speed modulation. """
        ix = int(np.clip(x, 0, self.size - 1))
        iy = int(np.clip(y, 0, self.size - 1))
        with self.lock:
            target_grid = self.grid_core if layer == 'core' else self.grid_surface
            return float(target_grid[iy, ix])

class AsynchronousVideoCapture:
    def __init__(self, filename="iMelt_Trial_", fps=20, frame_size=(640, 480), max_queue_size=30):
        self.filename = filename + time.strftime("%Y%m%d_%H%M%S") + ".mp4"
        self.fps = fps
        self.frame_size = frame_size
        self.max_queue_size = max_queue_size
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        # Initialize OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.filename, fourcc, self.fps, self.frame_size)
        # Start consumer thread
        self.worker_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.worker_thread.start()
        
    def _writer_loop(self):
        """Background thread worker that writes frames from the queue to disk."""
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                # Wait up to 0.5s for a frame
                frame = self.queue.get(timeout=0.5)
                # Check for shutdown sentinel
                if frame is None:
                    self.queue.task_done()
                    break
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue
        # Release file resource when loop terminates
        if self.writer is not None:
            self.writer.release()

    def add_frame(self, frame):
        """Non-blocking method to add a frame to the queue. Drops frame if queue is full to preserve real-time gantry performance."""
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            # Queue is full — drop video frame to prevent blocking control loop
            pass

    def stop(self):
        """Gracefully flushes remaining queued frames and closes video file."""
        self.stop_event.set()
        try:
            self.queue.put_nowait(None)  # Enqueue sentinel to wake worker
        except queue.Full:
            pass
        if self.worker_thread.is_alive():
            self.worker_thread.join()

class SystemLogger:
    def __init__(self, subfolder="experiment_logs", base_dir=None):
        """
        Initializes a CSV telemetry logger in a designated subfolder.
        
        :param subfolder: Name or nested path of the target directory (e.g., 'logs/reactive_tests')
        :param base_dir: Root directory anchor. Defaults to the folder where this script lives.
        """
        # Anchor path to the script folder to avoid dumping loose files in the project root
        if base_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
        self.log_dir = os.path.join(base_dir, subfolder)
        os.makedirs(self.log_dir, exist_ok=True)
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = os.path.join(self.log_dir, f"telemetry_{session_id}.csv")
        
        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)
        self.start_time = time.time()
        self.flush_counter = 0
        
        self.writer.writerow([
            "timestamp_s", "gantry_x", "gantry_y", "cmd_vx", "cmd_vy", "speed",
            "heading_x", "heading_y", "cold_temp", "cold_x", "cold_y",
            "hot_temp", "hot_x", "hot_y", "error", "error_deriv",
            "bed_min", "bed_mean", "bed_max"
        ])
        print(f"[LOGGING] Session active: {self.filename}")

    def log_step(self, curr_x, curr_y, vx, vy, speed, hx, hy, state, error, error_d):
        elapsed = time.time() - self.start_time
        raw_frame = state.get('raw_temp_frame')
        
        bed_min = float(np.min(raw_frame)) if raw_frame is not None else 0.0
        bed_mean = float(np.mean(raw_frame)) if raw_frame is not None else 0.0
        bed_max = float(np.max(raw_frame)) if raw_frame is not None else 0.0
        
        c_pos = state.get('cold_centroid') or (0.0, 0.0)
        h_pos = state.get('hot_centroid') or (0.0, 0.0)
        
        self.writer.writerow([
            f"{elapsed:.3f}", f"{curr_x:.2f}", f"{curr_y:.2f}",
            f"{vx:.2f}", f"{vy:.2f}", f"{speed:.2f}",
            f"{hx:.3f}", f"{hy:.3f}",
            f"{state.get('current_min_temp', 0.0):.2f}", f"{c_pos[0]:.2f}", f"{c_pos[1]:.2f}",
            f"{state.get('hotspot_intensity', 0.0):.2f}", f"{h_pos[0]:.2f}", f"{h_pos[1]:.2f}",
            f"{error:.2f}", f"{error_d:.2f}",
            f"{bed_min:.2f}", f"{bed_mean:.2f}", f"{bed_max:.2f}"
        ])
        
        # Flush every 20 iterations (1 second) to minimize disk latency
        self.flush_counter += 1
        if self.flush_counter % 20 == 0:
            self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()
        print(f"[LOGGING] Log saved and closed: {self.filename}")

class GantryController:
    """
    Master class encapsulating the Zaber gantry hardware, PySpin FLIR camera, 
    and the 2-layer Kalman physics Thermal Twin.
    """
    def __init__(self, bed_size_mm=250.0, safety_margin_mm=5.0, tool_radius_mm=25.0, port='/dev/ttyACM0'):
        self.bed_size = bed_size_mm
        self.margin = safety_margin_mm
        self.tool_radius = tool_radius_mm
        self.port = port
        self.camera_connected = False
        # Encapsulated state
        self.thermal_twin = None
        self.connection = None
        self.axis_x = None
        self.axis_y = None
        self.axis_z = None
        self.system_running = False
        self.telemetry = []
        self.transform_matrix_filename = "transform_matrix.json"
        self.transform_matrix = None
        if os.path.exists(self.transform_matrix_filename):
            self.transform_matrix = self.load_transform_matrix(self.transform_matrix_filename)

    @staticmethod
    def load_transform_matrix(filename="transform_matrix.json"):
        """
        Loads the transformation matrix from a JSON file.
        """
        with open(filename, "r") as f:
            matrix = np.array(json.load(f), dtype=np.float32)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid 3x3 transform matrix in {filename}.")
        return matrix

    def _calibrate_with_checkerboard(
        self,
        image_array,
        board_dims=(8, 10),
        square_size_mm=30.0,
        filename="transform_matrix.json",
        report_failure=True,
    ):
        """
        Detects a checkerboard in a thermal image and creates a pixel-to-mm homography.

        The values in board_dims are the number of internal corners, matching OpenCV's
        findChessboardCorners convention.
        """
        if image_array.ndim != 2:
            raise ValueError("Checkerboard calibration requires a 2D thermal frame.")
        if len(board_dims) != 2 or min(board_dims) < 2:
            raise ValueError("board_dims must contain at least 2x2 internal corners.")
        if square_size_mm <= 0:
            raise ValueError("square_size_mm must be positive.")

        vmin, vmax = np.percentile(image_array, (2, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            if report_failure:
                print("[CALIBRATION] Thermal frame has insufficient contrast.")
            return None, np.zeros_like(image_array, dtype=np.uint8)

        clipped_array = np.clip(image_array, vmin, vmax)
        gray_img = cv2.normalize(
            clipped_array, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        gray_img = cv2.bitwise_not(gray_img)

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray_img, board_dims, flags)
        if not found:
            if report_failure:
                print(
                    "[CALIBRATION] Checkerboard not detected. "
                    "Increase the thermal contrast and try again."
                )
            return None, gray_img

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners = cv2.cornerSubPix(
            gray_img, corners, (11, 11), (-1, -1), criteria
        )

        object_points = np.zeros(
            (board_dims[0] * board_dims[1], 2), dtype=np.float32
        )
        object_points[:] = (
            np.mgrid[0:board_dims[0], 0:board_dims[1]]
            .T
            .reshape(-1, 2)
            .astype(np.float32)
            * square_size_mm
        )

        pixel_points = corners.reshape(-1, 2)
        matrix, inliers = cv2.findHomography(
            pixel_points, object_points, cv2.RANSAC, 5.0
        )
        if matrix is None or not np.isfinite(matrix).all() or matrix.shape != (3, 3):
            if report_failure:
                print("[CALIBRATION] OpenCV could not calculate a valid homography.")
            return None, gray_img

        if inliers is not None and int(np.count_nonzero(inliers)) < 4:
            if report_failure:
                print("[CALIBRATION] Not enough checkerboard corners were inliers.")
            return None, gray_img

        matrix = matrix.astype(np.float32)
        output_path = os.path.abspath(filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temporary_path = f"{output_path}.tmp"
        with open(temporary_path, "w") as f:
            json.dump(matrix.tolist(), f)
        os.replace(temporary_path, output_path)

        self.transform_matrix_filename = output_path
        self.transform_matrix = matrix
        cv2.drawChessboardCorners(gray_img, board_dims, corners, found)
        print(f"[CALIBRATION] Matrix saved to {output_path}.")
        return matrix, gray_img

    def calibrate_transform_matrix(
        self,
        filename="transform_matrix.json",
        board_dims=(8, 10),
        square_size_mm=30.0,
        auto_calibrate=True,
        max_frames=300,
        show_preview=True,
    ):
        """
        Streams radiometric frames and calibrates the thermal camera-to-bed transform.

        In automatic mode, each frame is checked until a checkerboard is detected or
        max_frames is reached. In manual mode, press 'c' to attempt calibration and
        'q' to stop. The successful matrix is saved to filename and stored in
        self.transform_matrix.
        """
        if max_frames <= 0:
            raise ValueError("max_frames must be positive.")

        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        cam = None
        initialized = False
        acquiring = False

        if cam_list.GetSize() == 0:
            cam_list.Clear()
            system.ReleaseInstance()
            raise RuntimeError("No FLIR camera was detected.")

        try:
            cam = cam_list.GetByIndex(0)
            cam.Init()
            initialized = True
            nodemap = cam.GetNodeMap()

            ir_format = PySpin.CEnumerationPtr(nodemap.GetNode("IRFormat"))
            ir_format.SetIntValue(
                ir_format.GetEntryByName("Radiometric").GetValue()
            )

            stream_nodemap = cam.GetTLStreamNodeMap()
            buffer_mode = PySpin.CEnumerationPtr(
                stream_nodemap.GetNode("StreamBufferHandlingMode")
            )
            buffer_mode.SetIntValue(
                buffer_mode.GetEntryByName("NewestOnly").GetValue()
            )

            acquisition_mode = PySpin.CEnumerationPtr(
                nodemap.GetNode("AcquisitionMode")
            )
            acquisition_mode.SetIntValue(
                acquisition_mode.GetEntryByName("Continuous").GetValue()
            )

            R = PySpin.CFloatPtr(nodemap.GetNode("R")).GetValue()
            B = PySpin.CFloatPtr(nodemap.GetNode("B")).GetValue()
            F = PySpin.CFloatPtr(nodemap.GetNode("F")).GetValue()
            X = PySpin.CFloatPtr(nodemap.GetNode("X")).GetValue()
            A1 = PySpin.CFloatPtr(nodemap.GetNode("alpha1")).GetValue()
            A2 = PySpin.CFloatPtr(nodemap.GetNode("alpha2")).GetValue()
            B1 = PySpin.CFloatPtr(nodemap.GetNode("beta1")).GetValue()
            B2 = PySpin.CFloatPtr(nodemap.GetNode("beta2")).GetValue()
            J1 = PySpin.CFloatPtr(nodemap.GetNode("J1")).GetValue()
            J0 = PySpin.CIntegerPtr(nodemap.GetNode("J0")).GetValue()

            emissivity = 0.25
            reflected_temp = 293.15
            atmospheric_temp = 293.15
            humidity = 0.55
            distance = 2.5
            atmospheric_temp_c = atmospheric_temp - 273.15
            water_vapour = humidity * np.exp(
                1.5587
                + 0.06939 * atmospheric_temp_c
                - 0.00027816 * atmospheric_temp_c**2
                + 0.00000068455 * atmospheric_temp_c**3
            )
            transmission = X * np.exp(
                -np.sqrt(distance) * (A1 + B1 * np.sqrt(water_vapour))
            ) + (1 - X) * np.exp(
                -np.sqrt(distance) * (A2 + B2 * np.sqrt(water_vapour))
            )
            radiance_reflection = (
                R / (np.exp(B / reflected_temp) - F)
            )
            radiance_atmosphere = (
                R / (np.exp(B / atmospheric_temp) - F)
            )
            K2 = (
                ((1 - emissivity) / emissivity) * radiance_reflection
                + ((1 - transmission) / (emissivity * transmission))
                * radiance_atmosphere
            )

            cam.BeginAcquisition()
            acquiring = True
            if show_preview:
                cv2.namedWindow("Thermal Calibration", cv2.WINDOW_NORMAL)

            if auto_calibrate:
                print(
                    f"[CALIBRATION] Searching up to {max_frames} frames "
                    "for the checkerboard."
                )
            else:
                print(
                    "[CALIBRATION] Press 'c' to calibrate or 'q' to cancel."
                )

            for frame_number in range(1, max_frames + 1):
                image_result = cam.GetNextImage(1000)
                try:
                    if image_result.IsIncomplete():
                        continue

                    image_data = image_result.GetNDArray()
                    image_radiance = (image_data - J0) / J1
                    denominator = (
                        (image_radiance / emissivity / transmission) - K2
                    )
                    image_temp = (
                        B / np.log(R / denominator + F)
                    ) - 273.15
                finally:
                    image_result.Release()

                key = -1
                if show_preview:
                    vmin, vmax = np.percentile(image_temp, (2, 98))
                    if vmax > vmin:
                        normalized = np.clip(
                            (image_temp - vmin) * (255.0 / (vmax - vmin)),
                            0,
                            255,
                        ).astype(np.uint8)
                        display = cv2.applyColorMap(
                            normalized, cv2.COLORMAP_INFERNO
                        )
                        cv2.imshow("Thermal Calibration", display)
                    key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("[CALIBRATION] Calibration cancelled.")
                    return None

                should_attempt = auto_calibrate or key == ord("c")
                if not should_attempt:
                    continue

                matrix, debug_img = self._calibrate_with_checkerboard(
                    image_temp,
                    board_dims=board_dims,
                    square_size_mm=square_size_mm,
                    filename=filename,
                    report_failure=not auto_calibrate,
                )
                if matrix is not None:
                    if show_preview:
                        cv2.imshow("Calibration Result", debug_img)
                        cv2.waitKey(500)
                    return matrix

                if auto_calibrate and frame_number % 50 == 0:
                    print(
                        f"[CALIBRATION] No checkerboard found after "
                        f"{frame_number} frames."
                    )

            raise RuntimeError(
                "Calibration failed: no valid checkerboard was detected."
            )
        except PySpin.SpinnakerException as ex:
            raise RuntimeError(f"Thermal camera calibration failed: {ex}") from ex
        finally:
            if show_preview:
                cv2.destroyAllWindows()
            if acquiring and cam is not None:
                cam.EndAcquisition()
            if initialized and cam is not None:
                cam.DeInit()
            del cam
            cam_list.Clear()
            system.ReleaseInstance()

    def prepRun(self, sample_radius, sample_shape):
        """
        Instantiates the Digital Twin for the specific sample geometry and 
        establishes hardware connections prior to the heating run.
        """
        # Initialize the physical state estimator
        self.thermal_twin = IntegratedThermalTwin(
            size_mm=self.bed_size, 
            radius_mm=sample_radius, 
            tool_radius=self.tool_radius,
            shape=sample_shape
        )
        
        # Setup Zaber Hardware
        print("[SYSTEM] Homing Zaber Gantry...")
        self.connection = Connection.open_serial_port(self.port)
        self.connection.enable_alerts()
        device_list = self.connection.detect_devices()
        
        # Assuming Axis 1 and 2 map to X and Z
        self.axis_x = device_list[0].get_axis(1)
        self.axis_z = device_list[0].get_axis(2) 
        try:
            self.axis_y = device_list[1].get_axis(0)
        except IndexError:
            self.axis_y = None
        
        if not self.axis_x.is_homed(): 
            self.axis_x.home(wait_until_idle=False)
        if not self.axis_z.is_homed(): 
            self.axis_z.home(wait_until_idle=False)
        if self.axis_y is not None and not self.axis_y.is_homed():
            self.axis_y.home(wait_until_idle=False)
            
        self.axis_x.wait_until_idle()
        self.axis_z.wait_until_idle()
        if self.axis_y is not None:
            self.axis_y.wait_until_idle()
        print("[SYSTEM] System Ready.")

    def run(self, path, target_temp, safety_temp, min_speed, max_speed, const_voltage, record_video=False):
        """
        Executes the provided trajectory path, modulates speed based on the twin, 
        and collects comprehensive telemetry data for post-run analysis.
        """
        self.system_running = True
        self.telemetry = []
        start_time = time.time()
        
        # Spawn encapsulated camera thread
        cam_thread = threading.Thread(target=self._camera_worker, args=(record_video,), daemon=True)
        cam_thread.start()
        
        last_time = time.time()
        
        # Execute Path Navigation
        for target_x, target_y in path:
            if not self.system_running:
                break
                
            self.axis_x.move_absolute(target_x, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
            self.axis_z.move_absolute(target_y, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
            
            # 20Hz Control Loop during segment execution
            while (self.axis_x.is_busy() or self.axis_z.is_busy()) and self.camera_connected:
                if not self.system_running:
                    self.axis_x.stop()
                    self.axis_z.stop()
                    break
                    
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time
                
                # Retrieve true instantaneous position
                curr_x = self.axis_x.get_position(Units.LENGTH_MILLIMETRES)
                curr_y = self.axis_z.get_position(Units.LENGTH_MILLIMETRES)
                
                # Advance the Thermal Twin PDE
                self.thermal_twin.predict_step(curr_x, curr_y, const_voltage, dt)
                core_temp = self.thermal_twin.get_local_temperature(curr_x, curr_y, layer='core')
                surf_temp = self.thermal_twin.get_local_temperature(curr_x, curr_y, layer='surface')
                
                # Safety Interlock
                if core_temp >= safety_temp:
                    print(f"[ALARM] Safety threshold exceeded! Core: {core_temp:.1f}°C")
                    self.system_running = False
                    self.axis_x.stop()
                    self.axis_z.stop()
                    break
                
                # Proportional Speed Modulation (Inverse PD mapping)
                temp_ratio = np.clip(core_temp / target_temp, 0.0, 1.0)
                modulated_speed = min_speed + ((max_speed - min_speed) * temp_ratio)
                
                self.axis_x.settings.set("maxspeed", modulated_speed, Units.VELOCITY_MILLIMETRES_PER_SECOND)
                self.axis_z.settings.set("maxspeed", modulated_speed, Units.VELOCITY_MILLIMETRES_PER_SECOND)
                
                # Log system state telemetry
                self.telemetry.append({
                    'timestamp_s': current_time - start_time,
                    'x': curr_x,
                    'y': curr_y,
                    'speed': modulated_speed,
                    'core_temp': core_temp,
                    'surface_temp': surf_temp
                })
                
                time.sleep(0.05) 
                
        # Graceful shutdown
        self.system_running = False
        cam_thread.join()
        
        if self.connection:
            self.connection.close()
            
        return self.telemetry

    def _camera_worker(self, record_video):
        """
        Private worker method isolating the PySpin API and radiometric math 
        from the global namespace, feeding the twin's Kalman filter.
        """
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        if cam_list.GetSize() == 0:
            cam_list.Clear()
            system.ReleaseInstance()
            print("[CAMERA] No FLIR detected. Running in pure-simulation mode.")
            return

        cam = cam_list.GetByIndex(0)
        recorder = None
        acquiring = False
        
        try:
            cam.Init()
            nodemap = cam.GetNodeMap()
            
            # Configure Radiometric Node Setup
            node_IRFormat = PySpin.CEnumerationPtr(nodemap.GetNode('IRFormat'))
            node_IRFormat.SetIntValue(node_IRFormat.GetEntryByName('Radiometric').GetValue())
            
            # Retrieve Calibration constants from Firmware
            R = PySpin.CFloatPtr(nodemap.GetNode('R')).GetValue()
            B = PySpin.CFloatPtr(nodemap.GetNode('B')).GetValue()
            F = PySpin.CFloatPtr(nodemap.GetNode('F')).GetValue()
            X = PySpin.CFloatPtr(nodemap.GetNode('X')).GetValue()
            A1, A2 = PySpin.CFloatPtr(nodemap.GetNode('alpha1')).GetValue(), PySpin.CFloatPtr(nodemap.GetNode('alpha2')).GetValue()
            B1, B2 = PySpin.CFloatPtr(nodemap.GetNode('beta1')).GetValue(), PySpin.CFloatPtr(nodemap.GetNode('beta2')).GetValue()
            J1, J0 = PySpin.CFloatPtr(nodemap.GetNode('J1')).GetValue(), PySpin.CIntegerPtr(nodemap.GetNode('J0')).GetValue()
            
            # Atmospheric Constants
            Emiss, TRefl, TAtm, Humidity = 0.97, 293.15, 293.15, 0.55
            TAtmC = TAtm - 273.15
            H2O = Humidity * np.exp(1.5587 + 0.06939 * TAtmC - 0.00027816 * TAtmC**2 + 0.00000068455 * TAtmC**3)
            Tau = X * np.exp(-math.sqrt(2) * (A1 + B1 * math.sqrt(H2O))) + (1 - X) * np.exp(-math.sqrt(2) * (A2 + B2 * math.sqrt(H2O)))
            K2 = (((1 - Emiss) / Emiss) * (R / (np.exp(B / TRefl) - F))) + (((1 - Tau) / (Emiss * Tau)) * (R / (np.exp(B / TAtm) - F)))
            self.camera_connected = True

            if self.transform_matrix is None and os.path.exists(
                self.transform_matrix_filename
            ):
                self.transform_matrix = self.load_transform_matrix(
                    self.transform_matrix_filename
                )
            if self.transform_matrix is None:
                print(
                    "[CAMERA] No transform matrix is available. "
                    "Call calibrate_transform_matrix() first."
                )
                return

            cam.BeginAcquisition()
            acquiring = True
            
            if record_video:
                recorder = AsynchronousVideoCapture(filename="Gantry_Run_", fps=20, frame_size=(640, 480))

            while self.system_running:
                image_result = cam.GetNextImage(1000)
                if not image_result.IsIncomplete():
                    image_data = image_result.GetNDArray()
                    
                    # Process Radiometric Math
                    image_Radiance = (image_data - J0) / J1
                    image_Temp = (B / np.log(R / ((image_Radiance / Emiss / Tau) - K2) + F)) - 273.15
                    
                    # Fetch kinematics without stalling the camera
                    try:
                        curr_x = self.axis_x.get_position(Units.LENGTH_MILLIMETRES)
                        curr_z = self.axis_z.get_position(Units.LENGTH_MILLIMETRES)
                    except Exception:
                        curr_x, curr_z= 125.0, 125.0

                    # Execute Kalman Measurement Update
                    self.thermal_twin.measurement_update(
                        image_Temp,
                        self.transform_matrix,
                        curr_x,
                        curr_z,
                    )
                    
                    if record_video and recorder:
                        norm_temp = np.clip((image_Temp - 20.0) / 180.0 * 255.0, 0, 255).astype(np.uint8)
                        color_frame = cv2.applyColorMap(norm_temp, cv2.COLORMAP_INFERNO)
                        recorder.add_frame(color_frame)
                        
                image_result.Release()
                time.sleep(0.01)

        except PySpin.SpinnakerException as ex:
            print(f"[CAMERA ERROR] {ex}")
            self.camera_connected = False
        finally:
            if recorder: recorder.stop()
            if acquiring:
                cam.EndAcquisition()
            cam.DeInit()
            del cam
            cam_list.Clear()
            system.ReleaseInstance()