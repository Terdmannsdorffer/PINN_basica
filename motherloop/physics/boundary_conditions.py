# physics/boundary_conditions.py
import torch
import numpy as np

class BoundaryConfig:
    """Configuration for boundary conditions with different inlet velocities"""
    
    def __init__(self, inlet_velocity=-0.005):
        # Domain dimensions (from your existing code)
        self.L_up = 0.097      # Upper horizontal length  
        self.L_down = 0.174    # Total horizontal length
        self.H_left = 0.119    # Left vertical height
        self.H_right = 0.019   # Right vertical height
        
        # Physical parameters
        self.rho = 0.101972  # Fluid density
        self.g_x, self.g_y = 0.0, -9.81
        
        # Flow parameters
        self.u_in = 0.0  # No horizontal flow at inlet
        self.v_in = inlet_velocity  # Variable inlet velocity
        
        # Calculate outlet velocity from continuity
        inlet_area = self.L_up * 1.0
        outlet_area = self.H_right * 1.0
        self.u_out = abs(self.v_in) * (inlet_area / outlet_area)
        self.v_out = 0.0
        
        # Carbopol rheology parameters
        self.tau_y = 30.0
        self.k = 2.8
        self.n = 0.65
        
        print(f"Boundary Config - Inlet velocity: {self.v_in:.4f} m/s, "
              f"Outlet velocity: {self.u_out:.4f} m/s")

class BoundaryLoss:
    """Compute boundary condition losses"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
    
    def compute_inlet_loss(self, model, inlet_points):
        """Compute inlet boundary condition loss"""
        if len(inlet_points) == 0:
            return torch.tensor(0.0, device=self.device)
        
        inlet = torch.tensor(inlet_points, dtype=torch.float32, device=self.device)
        output = model(inlet)
        u_inlet, v_inlet = output[:, 0], output[:, 1]
        
        # Multiple loss terms for better enforcement
        u_mse = torch.mean((u_inlet - self.config.u_in)**2)
        v_mse = torch.mean((v_inlet - self.config.v_in)**2)
        u_mae = torch.mean(torch.abs(u_inlet - self.config.u_in))
        v_mae = torch.mean(torch.abs(v_inlet - self.config.v_in))
        
        return 50.0 * (u_mse + v_mse) + 25.0 * (u_mae + v_mae)
    
    def compute_outlet_loss(self, model, outlet_points):
        """Compute outlet boundary condition loss"""
        if len(outlet_points) == 0:
            return torch.tensor(0.0, device=self.device)
        
        outlet = torch.tensor(outlet_points, dtype=torch.float32, device=self.device)
        output = model(outlet)
        u_outlet, v_outlet, p_outlet = output[:, 0], output[:, 1], output[:, 2]
        
        u_mse = torch.mean((u_outlet - self.config.u_out)**2)
        v_mse = torch.mean((v_outlet - self.config.v_out)**2)
        u_mae = torch.mean(torch.abs(u_outlet - self.config.u_out))
        v_mae = torch.mean(torch.abs(v_outlet - self.config.v_out))
        p_loss = torch.mean(p_outlet**2)
        
        return 40.0 * (u_mse + v_mse) + 20.0 * (u_mae + v_mae) + 2.0 * p_loss
    
    def compute_wall_loss(self, model, wall_points):
        """Compute wall boundary condition loss (no-slip)"""
        if len(wall_points) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Use subset for efficiency and convert to numpy first
        n_wall = min(150, len(wall_points))
        wall_indices = torch.randperm(len(wall_points))[:n_wall]
        wall_subset = wall_points[wall_indices.numpy()]  # Convert indices to numpy first
        
        wall = torch.tensor(wall_subset, dtype=torch.float32, device=self.device)
        output = model(wall)
        u_wall, v_wall = output[:, 0], output[:, 1]
        
        wall_mse = torch.mean(u_wall**2 + v_wall**2)
        wall_mae = torch.mean(torch.abs(u_wall) + torch.abs(v_wall))
        
        return 20.0 * wall_mse + 10.0 * wall_mae
    
    def compute_total_boundary_loss(self, model, inlet_points, outlet_points, wall_points):
        """Compute total boundary loss"""
        inlet_loss = self.compute_inlet_loss(model, inlet_points)
        outlet_loss = self.compute_outlet_loss(model, outlet_points)
        wall_loss = self.compute_wall_loss(model, wall_points)
        
        return inlet_loss + outlet_loss + wall_loss

class PhysicsLoss:
    """Compute physics-based losses (Navier-Stokes equations)"""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
    
    def compute_physics_loss(self, model, domain_points):
        """Compute physics loss with stable Carbopol rheology"""
        if len(domain_points) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Use subset for efficiency
        n_points = min(1500, len(domain_points))
        indices = torch.randperm(len(domain_points))[:n_points]
        selected_points = domain_points[indices.numpy()]  # Convert to numpy first
        
        xy = torch.tensor(selected_points, dtype=torch.float32, device=self.device, requires_grad=True)
        output = model(xy)
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]

        # Compute gradients
        u_x, u_y = torch.autograd.grad(u.sum(), xy, create_graph=True)[0].split(1, dim=1)
        v_x, v_y = torch.autograd.grad(v.sum(), xy, create_graph=True)[0].split(1, dim=1)
        p_x, p_y = torch.autograd.grad(p.sum(), xy, create_graph=True)[0].split(1, dim=1)

        # Carbopol rheology
        shear_rate = torch.sqrt(2*((u_x)**2 + (v_y)**2) + (u_y + v_x)**2 + 1e-8)
        eta_eff = (self.config.tau_y / (shear_rate + 1e-8)) + self.config.k * torch.pow(shear_rate + 1e-8, self.config.n - 1)
        eta_eff = torch.clamp(eta_eff, min=0.01, max=500.0)

        # Physics equations
        continuity = u_x + v_y
        momentum_x = p_x - eta_eff * (u_x + u_y) + self.config.rho * self.config.g_x
        momentum_y = p_y - eta_eff * (v_x + v_y) + self.config.rho * self.config.g_y

        return torch.mean(continuity**2) + torch.mean(momentum_x**2 + momentum_y**2)