import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_lissajous_waypoints(
    sample_size=250.0,
    induction_radius=10.0,
    safety_clearance=2.0,
    freq_x=7,
    freq_y=8,
    phase_shift=np.pi / 2,
    num_cycles=1,
    target_step_mm=3.0
):
    """
    Generates waypoints for a smooth Lissajous Harmonic Trajectory (Sinusoidal Mesh)
    fitting within a square sample size with safety margins.
    
    x(t) = cx + A * sin(freq_x * t + phase_shift)
    y(t) = cy + B * sin(freq_y * t)
    """
    margin = induction_radius + safety_clearance
    usable_span = sample_size - 2.0 * margin
    
    if usable_span <= 0:
        raise ValueError(f"Sample size ({sample_size} mm) too small for coil radius + clearance.")
        
    cx = sample_size / 2.0
    cy = sample_size / 2.0
    amplitude = usable_span / 2.0
    
    # Total parameter range: t in [0, 2 * pi * num_cycles]
    t_max = 2.0 * np.pi * num_cycles
    
    # Dense initial sampling to compute arc-length parameterization
    t_dense = np.linspace(0, t_max, 20000)
    x_dense = cx + amplitude * np.sin(freq_x * t_dense + phase_shift)
    y_dense = cy + amplitude * np.sin(freq_y * t_dense)
    
    # Calculate cumulative distance along path
    dx = np.diff(x_dense)
    dy = np.diff(y_dense)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate(([0], np.cumsum(ds)))
    
    # Resample at uniform step size (target_step_mm)
    total_path_length = s[-1]
    num_waypoints = max(2, int(np.ceil(total_path_length / target_step_mm)))
    s_uniform = np.linspace(0, total_path_length, num_waypoints)
    
    # Interpolate x and y at uniform spatial steps
    x_uniform = np.interp(s_uniform, s, x_dense)
    y_uniform = np.interp(s_uniform, s, y_dense)
    
    waypoints = np.column_stack((x_uniform, y_uniform))
    return waypoints, margin, amplitude, total_path_length

def export_waypoints_csv(waypoints, filename="/workspace/scratch/lissajous_waypoints.csv", target_speed_mms=20.0):
    steps = np.arange(1, len(waypoints) + 1)
    speeds = np.full(len(waypoints), target_speed_mms)
    data = np.column_stack((steps, waypoints[:, 0], waypoints[:, 1], speeds))
    
    header = "step,x_mm,y_mm,target_speed_mms"
    np.savetxt(filename, data, delimiter=",", header=header, comments="", fmt=["%d", "%.3f", "%.3f", "%.1f"])
    print(f"Exported {len(waypoints)} waypoints to {filename}")

def plot_lissajous_path(waypoints, sample_size, induction_radius, margin, freq_x, freq_y, output_img="lissajous_path.png"):
    plt.figure(figsize=(8, 8))
    
    # Sample Perimeter
    plt.plot([0, sample_size, sample_size, 0, 0], [0, 0, sample_size, sample_size, 0], 'k--', alpha=0.6, label=f'Sample Perimeter ({sample_size}x{sample_size} mm)')
    
    # Margin Boundary
    plt.plot([margin, sample_size-margin, sample_size-margin, margin, margin],
             [margin, margin, sample_size-margin, sample_size-margin, margin],
             'g:', alpha=0.5, label=f'Margin ({margin:.1f} mm)')
             
    # Trajectory
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'b-', linewidth=1.2, label=f'Lissajous Mesh ({freq_x}:{freq_y})')
    
    # Start and End points
    plt.scatter(waypoints[0, 0], waypoints[0, 1], color='green', s=80, zorder=5, label='Start Point')
    plt.scatter(waypoints[-1, 0], waypoints[-1, 1], color='purple', s=80, zorder=5, label='End Point')
    
    # Toolhead circle at start
    coil_circle = plt.Circle((waypoints[0, 0], waypoints[0, 1]), induction_radius, color='orange', alpha=0.3, label=f'Induction Coil (R = {induction_radius} mm)')
    plt.gca().add_patch(coil_circle)
    
    plt.xlim(-10, sample_size + 10)
    plt.ylim(-10, sample_size + 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'Lissajous Harmonic Trajectory ({freq_x}:{freq_y} Ratio)\nSample: {sample_size}x{sample_size} mm | Smooth Sinusoidal Mesh', fontsize=11)
    plt.xlabel('X Position (mm)', fontsize=10)
    plt.ylabel('Y Position (mm)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_img, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_img}")


SAMPLE_SIZE = 250.0
INDUCTION_RADIUS = 10.0
SAFETY_CLEARANCE = 2.0
FREQ_X = 7
FREQ_Y = 8

waypoints, margin, amp, path_len = generate_lissajous_waypoints(
    sample_size=SAMPLE_SIZE,
    induction_radius=INDUCTION_RADIUS,
    safety_clearance=SAFETY_CLEARANCE,
    freq_x=FREQ_X,
    freq_y=FREQ_Y,
    phase_shift=np.pi / 2,
    num_cycles=1,
    target_step_mm=3.0
)

export_waypoints_csv(waypoints, filename="/workspace/scratch/lissajous_waypoints.csv")
plot_lissajous_path(
    waypoints,
    sample_size=SAMPLE_SIZE,
    induction_radius=INDUCTION_RADIUS,
    margin=margin,
    freq_x=FREQ_X,
    freq_y=FREQ_Y,
    output_img="/workspace/scratch/lissajous_path.png"
)

# Copy output files to /workspace/out/
shutil.copy("/workspace/scratch/lissajous_path_generator.py", "/workspace/out/lissajous_path_generator.py") if os.path.exists("/workspace/scratch/lissajous_path_generator.py") else None
shutil.copy("/workspace/scratch/lissajous_waypoints.csv", "/workspace/out/lissajous_waypoints.csv") if os.path.exists("/workspace/scratch/lissajous_waypoints.csv") else None
shutil.copy("/workspace/scratch/lissajous_path.png", "/workspace/out/lissajous_path.png") if os.path.exists("/workspace/scratch/lissajous_path.png") else None
print("Published Lissajous artifacts to /workspace/out/")