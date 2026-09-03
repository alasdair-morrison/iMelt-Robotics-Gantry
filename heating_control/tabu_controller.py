import numpy as np

class ThermalTabuFlowController:
    def __init__(self, grid_size=(250, 250), tabu_duration=8.0):
        self.grid_size = grid_size
        self.tabu_duration = tabu_duration
        self.tabu_map = np.zeros(grid_size, dtype=np.float32)
        
    def update_tabu_memory(self, curr_x, curr_y, radius=15.0, dt=0.05):
        # Decay existing tabu tenure
        self.tabu_map = np.maximum(0.0, self.tabu_map - dt)
        
        # Mark current heater footprint as tabu
        gx, gy = int(np.clip(curr_x, 0, 249)), int(np.clip(curr_y, 0, 249))
        y_idx, x_idx = np.ogrid[:self.grid_size[1], :self.grid_size[0]]
        mask = (x_idx - gx)**2 + (y_idx - gy)**2 <= radius**2
        self.tabu_map[mask] = self.tabu_duration

    def compute_flow_vector(self, temp_array, curr_x, curr_y, target_temp=180.0):
        # Calculate raw thermal deficit
        deficit = np.maximum(0.0, target_temp - temp_array)
        
        # Apply Tabu Search Memory (inhibit recently visited areas)
        active_deficit = np.where(self.tabu_map > 0, 0.0, deficit)
        
        # Create Potential Field (Attractive = Cold Deficit, Repulsive = Hot Spots)
        repulsion = np.maximum(0.0, temp_array - (target_temp - 10.0)) ** 2
        potential = -active_deficit + (2.0 * repulsion)
        
        # Generate Vector Flow Field via Spatial Gradients
        grad_y, grad_x = np.gradient(potential)
        
        # Sample Flow Vector at current gantry position
        gx, gy = int(np.clip(curr_x, 0, 249)), int(np.clip(curr_y, 0, 249))
        vx = -grad_x[gy, gx]
        vy = -grad_y[gy, gx]
        
        norm = np.hypot(vx, vy) + 1e-6
        return vx / norm, vy / norm