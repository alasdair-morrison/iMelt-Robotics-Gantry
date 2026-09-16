import numpy as np

class ThermalTabuFlowController:
    def __init__(self, grid_size=(250, 250), tabu_duration=8.0):
        self.grid_size = grid_size
        self.tabu_duration = tabu_duration
        self.tabu_map = np.zeros(grid_size, dtype=np.float32)
        
    def update_tabu_memory(self, curr_x, curr_y, radius=20.0, dt=0.05):
        self.tabu_map = np.maximum(0.0, self.tabu_map - dt)
        gx, gy = int(np.clip(curr_x, 0, 249)), int(np.clip(curr_y, 0, 249))
        y_idx, x_idx = np.ogrid[:self.grid_size[1], :self.grid_size[0]]
        mask = (x_idx - gx)**2 + (y_idx - gy)**2 <= radius**2
        self.tabu_map[mask] = self.tabu_duration

    def compute_nav_vector(self, curr_x, curr_y, cold_target, hot_target=None, hot_intensity=0.0):
        """
        Computes a heading vector that directly pursues cold targets,
        actively flees hot spots, and steers away from recently visited tabu zones.
        """
        dir_x, dir_y = 0.0, 0.0

        # Global Attraction to Cold Centroid (Weight: 1.5)
        if cold_target is not None:
            cx, cy = cold_target
            v_cx = cx - curr_x
            v_cy = cy - curr_y
            dist_c = np.hypot(v_cx, v_cy) + 1e-5
            dir_x += (v_cx / dist_c) * 1.5
            dir_y += (v_cy / dist_c) * 1.5

        # Global Repulsion from Overheated Hotspot (Weight: up to 3.0)
        if hot_target is not None and hot_intensity > 0.05:
            hx, hy = hot_target
            v_hx = curr_x - hx
            v_hy = curr_y - hy
            dist_h = np.hypot(v_hx, v_hy) + 1e-5
            w_hot = min(3.0, hot_intensity * 3.0)
            dir_x += (v_hx / dist_h) * w_hot
            dir_y += (v_hy / dist_h) * w_hot

        # Local Tabu Field Repulsion (Normalized Weight: 0.8)
        gx, gy = int(np.clip(curr_x, 0, 249)), int(np.clip(curr_y, 0, 249))
        r = 15
        x_min, x_max = max(0, gx - r), min(250, gx + r + 1)
        y_min, y_max = max(0, gy - r), min(250, gy + r + 1)
        local_tabu = self.tabu_map[y_min:y_max, x_min:x_max]
        
        if np.any(local_tabu > 0):
            gy_local, gx_local = np.ogrid[y_min:y_max, x_min:x_max]
            dx = curr_x - gx_local
            dy = curr_y - gy_local
            dist_sq = dx**2 + dy**2 + 1e-5
            
            # Calculate raw unnormalized repulsion
            repel_x = np.sum((dx / dist_sq) * local_tabu)
            repel_y = np.sum((dy / dist_sq) * local_tabu)
            
            # Normalize and cap Tabu force
            repel_mag = np.hypot(repel_x, repel_y) + 1e-5
            dir_x += (repel_x / repel_mag) * 0.8
            dir_y += (repel_y / repel_mag) * 0.8

        # Final normalization for Zaber velocity stream
        mag = np.hypot(dir_x, dir_y) + 1e-5
        return dir_x / mag, dir_y / mag