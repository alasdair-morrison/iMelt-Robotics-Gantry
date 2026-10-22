import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def hilbert_d2xy(order, d):
    """
    Map 1D distance d (0 to 4^order - 1) to 2D grid coordinates (x, y)
    for a Hilbert curve of given order (grid size N = 2^order).
    """
    n = 2 ** order
    rx, ry, s, t = 0, 0, 1, d
    x, y = 0, 0
    
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def calculate_optimal_hilbert_order(sample_size, induction_radius, safety_clearance=2.0):
    """
    Dynamically computes the optimal Hilbert curve order based on sample size and coil radius.
    Ensures path pitch matches the effective thermal diameter of the induction coil (~2 * radius).
    """
    margin = induction_radius + safety_clearance
    usable_span = sample_size - 2 * margin
    if usable_span <= 0:
        raise ValueError(f"Sample size ({sample_size} mm) is too small for coil radius ({induction_radius} mm) + clearance ({safety_clearance} mm).")

    # Ideal pitch between passes ~ 2 * induction_radius
    target_pitch = 2.0 * induction_radius
    
    # N_intervals = 2^order - 1 = usable_span / target_pitch
    ideal_intervals = usable_span / target_pitch
    
    # Choose order such that 2^order is closest to ideal_intervals + 1
    if ideal_intervals < 1:
        order = 1
    else:
        order = int(np.round(np.log2(ideal_intervals + 1)))
        order = max(1, min(order, 6))  # Cap order between 1 and 6 for gantry practicality
        
    return order, margin

def generate_dynamic_hilbert_waypoints(sample_size=250.0, induction_radius=10.0, safety_clearance=2.0, order=None):
    """
    Generates parametrically scaled (x, y) Hilbert waypoints adapted to variable sample size & coil radius.
    """
    if order is None:
        order, margin = calculate_optimal_hilbert_order(sample_size, induction_radius, safety_clearance)
    else:
        margin = induction_radius + safety_clearance

    n_points = 4 ** order
    raw_pts = np.array([hilbert_d2xy(order, i) for i in range(n_points)], dtype=np.float64)
    
    grid_max = (2 ** order) - 1
    usable_span = sample_size - 2 * margin
    
    if grid_max == 0:
        scaled_x = np.full(n_points, sample_size / 2.0)
        scaled_y = np.full(n_points, sample_size / 2.0)
    else:
        scaled_x = margin + (raw_pts[:, 0] / grid_max) * usable_span
        scaled_y = margin + (raw_pts[:, 1] / grid_max) * usable_span
        
    grid_pitch = usable_span / grid_max if grid_max > 0 else usable_span
    
    return np.column_stack((scaled_x, scaled_y)), order, margin, grid_pitch

def fillet_corners(waypoints, fillet_radius=3.5, arc_resolution=5):
    """
    Inserts circular arc fillets at 90-degree corners to allow continuous
    gantry feedrates without corner deceleration stalls.
    """
    if len(waypoints) < 3:
        return waypoints

    filleted_pts = [waypoints[0]]

    for i in range(1, len(waypoints) - 1):
        p_prev = waypoints[i - 1]
        p_curr = waypoints[i]
        p_next = waypoints[i + 1]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 < 1e-5 or len2 < 1e-5:
            filleted_pts.append(p_curr)
            continue

        u1 = v1 / len1
        u2 = v2 / len2

        dot = np.dot(u1, u2)
        if abs(dot) < 0.1:  # Right-angle corner
            r = min(fillet_radius, len1 / 2.5, len2 / 2.5)
            
            start_tangent = p_curr + u1 * r
            end_tangent = p_curr + u2 * r

            bisector = u1 + u2
            bisector_norm = np.linalg.norm(bisector)
            if bisector_norm > 1e-5:
                bisector /= bisector_norm
            
            center = p_curr + bisector * (r * np.sqrt(2))

            angle_start = np.arctan2(start_tangent[1] - center[1], start_tangent[0] - center[0])
            angle_end = np.arctan2(end_tangent[1] - center[1], end_tangent[0] - center[0])

            d_angle = angle_end - angle_start
            if d_angle > np.pi:
                d_angle -= 2 * np.pi
            elif d_angle < -np.pi:
                d_angle += 2 * np.pi

            angles = np.linspace(angle_start, angle_start + d_angle, arc_resolution)
            arc_pts = np.column_stack((center[0] + r * np.cos(angles), center[1] + r * np.sin(angles)))
            
            filleted_pts.extend(arc_pts)
        else:
            filleted_pts.append(p_curr)

    filleted_pts.append(waypoints[-1])
    return np.array(filleted_pts)

def plot_dynamic_hilbert_path(raw_waypoints, filleted_waypoints, sample_size, induction_radius, margin, order, output_img="/workspace/scratch/hilbert_path_v2.png"):
    """
    Plots the parametrically scaled Hilbert path showing the sample boundary and coil footprint.
    """
    plt.figure(figsize=(8, 8))
    
    # Sample Perimeter
    plt.plot([0, sample_size, sample_size, 0, 0], [0, 0, sample_size, sample_size, 0], 'k--', alpha=0.6, label=f'Sample Perimeter ({sample_size}x{sample_size} mm)')
    
    # Margin boundary
    plt.plot([margin, sample_size-margin, sample_size-margin, margin, margin], 
             [margin, margin, sample_size-margin, sample_size-margin, margin], 
             'g:', alpha=0.5, label=f'Margin ({margin:.1f} mm = R_coil + Clearance)')
    
    # Sharp Grid
    plt.plot(raw_waypoints[:, 0], raw_waypoints[:, 1], 'r:', alpha=0.4, linewidth=1.0, label='Sharp Grid Path')
    
    # Filleted Path
    plt.plot(filleted_waypoints[:, 0], filleted_waypoints[:, 1], 'b-', linewidth=1.5, label='Filleted Gantry Path')
    
    # Draw representative coil footprint circle at start
    start_x, start_y = filleted_waypoints[0]
    coil_circle = plt.Circle((start_x, start_y), induction_radius, color='orange', alpha=0.3, label=f'Induction Coil (R = {induction_radius} mm)')
    plt.gca().add_patch(coil_circle)
    
    plt.scatter(start_x, start_y, color='green', s=80, zorder=5, label='Start Point')
    plt.scatter(filleted_waypoints[-1, 0], filleted_waypoints[-1, 1], color='purple', s=80, zorder=5, label='End Point')
    
    plt.xlim(-10, sample_size + 10)
    plt.ylim(-10, sample_size + 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Parametric Hilbert Trajectory (Order {order})\nSample: {sample_size}x{sample_size} mm | Coil Radius: {induction_radius} mm', fontsize=11)
    plt.xlabel('X Position (mm)', fontsize=10)
    plt.ylabel('Y Position (mm)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    print(f"Saved dynamic visualization to {output_img}")
"""
# Test with custom global variables
SAMPLE_SIZE = 250.0      # mm (can change per run)
INDUCTION_RADIUS = 10.0  # mm (coil radius)
SAFETY_CLEARANCE = 2.0  # mm

raw_pts, order, margin, pitch = generate_dynamic_hilbert_waypoints(
    sample_size=SAMPLE_SIZE,
    induction_radius=INDUCTION_RADIUS,
    safety_clearance=SAFETY_CLEARANCE
)

fillet_r = min(3.5, pitch / 3.0)
filleted_pts = fillet_corners(raw_pts, fillet_radius=fillet_r, arc_resolution=5)

plot_dynamic_hilbert_path(
    raw_pts, filleted_pts, 
    sample_size=SAMPLE_SIZE, 
    induction_radius=INDUCTION_RADIUS, 
    margin=margin, 
    order=order, 
    output_img="/workspace/scratch/hilbert_path_v2.png"
)

shutil.copy("/workspace/scratch/hilbert_path_generator-v2.py", "/workspace/out/hilbert_path_generator-v2.py") if os.path.exists("/workspace/scratch/hilbert_path_generator-v2.py") else None
shutil.copy("/workspace/scratch/hilbert_path_v2.png", "/workspace/out/hilbert_path_v2.png") if os.path.exists("/workspace/scratch/hilbert_path_v2.png") else None
"""