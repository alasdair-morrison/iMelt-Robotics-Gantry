import os
import csv
import time
from datetime import datetime
import numpy as np

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