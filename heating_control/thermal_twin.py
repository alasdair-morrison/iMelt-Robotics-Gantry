import numpy as np

class NumericalThermalTwin:
    def __init__(self, size_mm=250, res_mm=1.0, dt=0.05, t_ambient=23.0):
        self.dim = int(size_mm / res_mm)
        self.dx = res_mm * 1e-3  # Grid pitch in meters
        self.dt = dt
        self.t_ambient = t_ambient
        self.T = np.full((self.dim, self.dim), t_ambient, dtype=np.float32)
        
        # Polymer thermal properties (Polypropylene / PEEK)
        self.k = 0.22             # Thermal conductivity (W/m*K)
        self.rho = 1300.0         # Density (kg/m^3)
        self.cp = 1400.0          # Specific heat capacity (J/kg*K)
        self.thickness = 0.003    # 3mm polymer sheet thickness
        self.alpha = self.k / (self.rho * self.cp)
        
        # Surface dissipation parameters
        self.h = 12.0             # Convective loss (W/m^2*K)
        self.emiss = 0.95         # Surface emissivity
        self.sigma_sb = 5.67e-8   # Stefan-Boltzmann constant
        
        # Induction coil profile
        self.sigma_heat = 12.0    # Footprint radius in mm
        self.p_absorbed = 250.0   # Absorbed power in Watts
        
        # Precomputed coordinate mesh for induction flux
        y_idx, x_idx = np.indices((self.dim, self.dim), dtype=np.float32)
        self.mesh_x = x_idx * res_mm
        self.mesh_y = y_idx * res_mm

    def step(self, gantry_x_mm, gantry_y_mm, heater_on=True):
        # 2D Diffusion via Pure NumPy (Neumann zero-flux boundary via edge padding)
        T_pad = np.pad(self.T, pad_width=1, mode='edge')
        laplacian = (
            T_pad[2:, 1:-1] +    # bottom neighbor (i+1, j)
            T_pad[:-2, 1:-1] +   # top neighbor (i-1, j)
            T_pad[1:-1, 2:] +    # right neighbor (i, j+1)
            T_pad[1:-1, :-2] -   # left neighbor (i, j-1)
            4.0 * self.T         # center cell (i, j)
        ) / (self.dx ** 2)
        
        d_diff = self.alpha * laplacian
        
        # Convective and Radiative Losses
        t_kelvin = self.T + 273.15
        t_amb_kelvin = self.t_ambient + 273.15
        q_loss = (self.h * (self.T - self.t_ambient) + 
                  self.emiss * self.sigma_sb * (t_kelvin**4 - t_amb_kelvin**4))
        d_loss = q_loss / (self.rho * self.cp * self.thickness)
        
        # Induction Heat Input (Gaussian source)
        d_source = np.zeros_like(self.T)
        if heater_on:
            dist_sq = (self.mesh_x - gantry_x_mm)**2 + (self.mesh_y - gantry_y_mm)**2
            flux_spatial = np.exp(-dist_sq / (2.0 * (self.sigma_heat ** 2)))
            q_in_vol = (self.p_absorbed * flux_spatial) / (2.0 * np.pi * (self.sigma_heat * 1e-3)**2 * self.thickness)
            d_source = q_in_vol / (self.rho * self.cp)
            
        # Explicit Forward Euler Update
        self.T += (d_diff - d_loss + d_source) * self.dt
        return self.T