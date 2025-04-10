import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import time
import os
from datetime import datetime

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories for saving results
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Physical parameters for progressive training
rho = 1000.0  # Density (kg/m³)
K_init = 0.01  # Initial consistency index (Newtonian phase)
K_final = 1.0  # Final consistency index
n_init = 1.0   # Initial flow index (Newtonian phase)
n_final = 0.5  # Final flow index
tau_y_init = 0.0  # Initial yield stress (Newtonian phase)
tau_y_final = 0.1  # Final yield stress
u_in = 0.1  # Inlet velocity (m/s)
g = -9.81  # Gravity (negative for downward direction)

# Geometry parameters
L_vertical = 2.0  # Length of vertical section
L_horizontal = 3.0  # Length of horizontal section
W = 0.5  # Width of the pipe

class PINN(nn.Module):
    def __init__(self, hidden_layers, neurons_per_layer):
        super(PINN, self).__init__()

        # Input layer (2 features: x, y)
        layers = [nn.Linear(2, neurons_per_layer), nn.Tanh()]

        # Hidden layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())

        # Output layer (3 outputs: u, v, p)
        layers.append(nn.Linear(neurons_per_layer, 3))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def inside_L_pipe(x, y):
    """Check if point (x, y) is inside the L-shaped pipe."""
    in_vertical = (-W/2 <= x <= W/2) and (-W/2 <= y <= L_vertical)
    in_horizontal = (-W/2 <= x <= L_horizontal) and (-W/2 <= y <= W/2)
    return in_vertical or in_horizontal

def generate_domain_points(num_points):
    """Generate random points inside the L-shaped domain."""
    # Create a bounding box and filter points
    x_min, x_max = -W/2, L_horizontal
    y_min, y_max = -W/2, L_vertical

    # Generate more points than needed to ensure enough remain after filtering
    factor = 3
    x_rand = np.random.uniform(x_min, x_max, num_points * factor)
    y_rand = np.random.uniform(y_min, y_max, num_points * factor)

    # Filter points to only those inside the L-shape
    inside_points = [(x, y) for x, y in zip(x_rand, y_rand) if inside_L_pipe(x, y)]

    # Take only the required number of points
    inside_points = inside_points[:num_points] if len(inside_points) >= num_points else inside_points

    return np.array(inside_points)

def generate_boundary_points(num_points):
    """Generate points on the boundary of the L-shaped pipe."""
    # Define the segments of the L-shaped pipe boundary
    segments = [
        # Outer boundary (counter-clockwise)
        [(-W/2, -W/2), (-W/2, L_vertical)],   # Left vertical edge
        [(-W/2, L_vertical), (W/2, L_vertical)],  # Top horizontal edge
        [(W/2, L_vertical), (W/2, W/2)],      # Right vertical upper edge
        [(W/2, W/2), (L_horizontal, W/2)],    # Top horizontal right edge
        [(L_horizontal, W/2), (L_horizontal, -W/2)],  # Right edge
        [(L_horizontal, -W/2), (-W/2, -W/2)]  # Bottom edge
    ]

    # Calculate the total length of the boundary
    total_length = 0
    for (x1, y1), (x2, y2) in segments:
        total_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    # Allocate points based on segment length
    points = []
    for (x1, y1), (x2, y2) in segments:
        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        segment_points = int(num_points * segment_length / total_length)

        # Generate points along this segment
        for i in range(segment_points):
            t = i / float(segment_points)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            points.append((x, y))

    return np.array(points)

def classify_boundary_points(boundary_points):
    """Classify boundary points as inlet, outlet, or wall."""
    inlet_points = []
    outlet_points = []
    wall_points = []

    for x, y in boundary_points:
        # Inlet: top of vertical section
        if np.isclose(y, L_vertical, rtol=1e-5, atol=1e-8) and abs(x) <= W/2:
            inlet_points.append((x, y))
        # Outlet: right end of horizontal section
        elif np.isclose(x, L_horizontal, rtol=1e-5, atol=1e-8) and abs(y) <= W/2:
            outlet_points.append((x, y))
        # All other boundary points are walls
        else:
            wall_points.append((x, y))

    return np.array(inlet_points), np.array(outlet_points), np.array(wall_points)

def compute_pde_residuals(model, x, y, K, n, tau_y):
    """
    Compute the PDE residuals with progressive non-Newtonian parameters.
    Parameters:
    - K: consistency index
    - n: flow index
    - tau_y: yield stress
    """
    # Convert to tensors and set requires_grad
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    xy = torch.cat([x_tensor.unsqueeze(1), y_tensor.unsqueeze(1)], dim=1)
    xy.requires_grad_(True)

    # Forward pass to get u, v, p
    uvp = model(xy)
    u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]

    # First order derivatives
    grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    grad_v = torch.autograd.grad(v.sum(), xy, create_graph=True)[0]
    grad_p = torch.autograd.grad(p.sum(), xy, create_graph=True)[0]

    du_dx, du_dy = grad_u[:, 0:1], grad_u[:, 1:2]
    dv_dx, dv_dy = grad_v[:, 0:1], grad_v[:, 1:2]
    dp_dx, dp_dy = grad_p[:, 0:1], grad_p[:, 1:2]

    # Second order derivatives for u
    grad_du_dx = torch.autograd.grad(du_dx.sum(), xy, create_graph=True)[0]
    grad_du_dy = torch.autograd.grad(du_dy.sum(), xy, create_graph=True)[0]
    du_dxx = grad_du_dx[:, 0:1]
    du_dyy = grad_du_dy[:, 1:2]

    # Second order derivatives for v
    grad_dv_dx = torch.autograd.grad(dv_dx.sum(), xy, create_graph=True)[0]
    grad_dv_dy = torch.autograd.grad(dv_dy.sum(), xy, create_graph=True)[0]
    dv_dxx = grad_dv_dx[:, 0:1]
    dv_dyy = grad_dv_dy[:, 1:2]

    # For Newtonian fluid (n=1, tau_y=0), mu_eff is just K
    if n == 1.0 and tau_y == 0.0:
        mu_eff = K * torch.ones_like(u)
    else:
        # Calculate strain rate tensor components
        e_xx = du_dx
        e_yy = dv_dy
        e_xy = 0.5 * (du_dy + dv_dx)

        # Calculate the magnitude of the strain rate tensor (shear rate)
        shear_rate = torch.sqrt(2*(e_xx**2 + e_yy**2 + 2*e_xy**2))

        # Add a small value to avoid division by zero with the non-Newtonian model
        epsilon = 1e-8
        shear_rate_safe = shear_rate + epsilon

        # Calculate the effective viscosity using Herschel-Bulkley model
        mu_eff = tau_y / shear_rate_safe + K * (shear_rate_safe ** (n-1))
    conv_weight = 0.5
    # Momentum equations
    pde_u = rho * conv_weight * (u * du_dx + v * du_dy) + dp_dx - mu_eff * (du_dxx + du_dyy)
    pde_v = rho * conv_weight * (u * dv_dx + v * dv_dy) + dp_dy - mu_eff * (dv_dxx + dv_dyy) + rho * g

    # Continuity equation (incompressible flow)
    pde_cont = du_dx + dv_dy

    return pde_u, pde_v, pde_cont

def compute_total_loss(model, domain_points, inlet_points, outlet_points, wall_points, 
                      K, n, tau_y, lambda_pde=1.0, lambda_bc=10.0):
    """Compute the total loss with current rheological parameters."""
    # Domain loss (PDE residuals)
    x_domain, y_domain = domain_points[:, 0], domain_points[:, 1]
    pde_u, pde_v, pde_cont = compute_pde_residuals(model, x_domain, y_domain, K, n, tau_y)

    loss_pde = (torch.mean(pde_u**2) + torch.mean(pde_v**2) + torch.mean(pde_cont**2))

    # Boundary conditions loss
    loss_bc = 0.0

    # Wall boundary condition (no-slip): u = v = 0
    if len(wall_points) > 0:
        x_wall, y_wall = wall_points[:, 0], wall_points[:, 1]
        xy_wall = torch.tensor(np.column_stack([x_wall, y_wall]), dtype=torch.float32, device=device)
        uvp_wall = model(xy_wall)
        u_wall, v_wall = uvp_wall[:, 0], uvp_wall[:, 1]
        loss_bc += torch.mean(u_wall**2) + torch.mean(v_wall**2)

    # Inlet boundary condition: u = 0, v = -u_in
    if len(inlet_points) > 0:
        x_inlet, y_inlet = inlet_points[:, 0], inlet_points[:, 1]
        xy_inlet = torch.tensor(np.column_stack([x_inlet, y_inlet]), dtype=torch.float32, device=device)
        uvp_inlet = model(xy_inlet)
        u_inlet, v_inlet = uvp_inlet[:, 0], uvp_inlet[:, 1]
        loss_bc += (torch.mean(u_inlet**2) + torch.mean((v_inlet + u_in)**2))

    # Outlet boundary condition: p = 0
    if len(outlet_points) > 0:
        x_outlet, y_outlet = outlet_points[:, 0], outlet_points[:, 1]
        xy_outlet = torch.tensor(np.column_stack([x_outlet, y_outlet]), dtype=torch.float32, device=device)
        uvp_outlet = model(xy_outlet)
        p_outlet = uvp_outlet[:, 2]
        loss_bc += torch.mean(p_outlet**2)

    # Total loss = PDE loss + BC loss
    total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc

    return total_loss, loss_pde, loss_bc

def plot_L_shaped_pipe():
    """Visualize the L-shaped pipe geometry."""
    # Define the corners of the L-shaped pipe
    corners = np.array([
        [-W/2, -W/2],          # Bottom-left corner of the L
        [-W/2, L_vertical],    # Top-left of vertical section
        [W/2, L_vertical],     # Top-right of vertical section
        [W/2, W/2],            # Corner point where the outer right edge turns
        [L_horizontal, W/2],   # Top-right of horizontal section
        [L_horizontal, -W/2],  # Bottom-right of horizontal section
        [-W/2, -W/2]           # Back to start to close the polygon
    ])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(corners[:, 0], corners[:, 1], 'r-', linewidth=2)

    # Generate points for visualization
    points = generate_domain_points(5000)
    ax.scatter(points[:, 0], points[:, 1], s=0.5)

    # Add annotations
    ax.annotate("Inlet", xy=(0, L_vertical), xytext=(0, L_vertical+0.2),
                arrowprops=dict(arrowstyle="->"), ha='center')
    ax.annotate("Outlet", xy=(L_horizontal, 0), xytext=(L_horizontal+0.2, 0),
                arrowprops=dict(arrowstyle="->"), ha='center')

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("L-shaped Pipe Geometry")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/L_shaped_pipe_geometry.png", dpi=300)
    plt.close(fig)

def plot_boundary(ax, boundary_points):
    """Plot the boundary of the L-shaped pipe on the given axis."""
    # Connect boundary points in the order they were generated
    boundary_closed = np.vstack([boundary_points, boundary_points[0]])
    ax.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'k-', linewidth=1)

def visualize_results(model, boundary_points, K, n, tau_y, num_points=10000, phase=""):
    """Visualize the flow field results with current rheological parameters."""
    # Generate a grid of points inside the L-shaped pipe
    points = generate_domain_points(num_points)

    # Convert to tensor for prediction
    xy_tensor = torch.tensor(points, dtype=torch.float32, device=device)

    # Get predictions
    with torch.no_grad():
        uvp_pred = model(xy_tensor).cpu().numpy()

    u_pred = uvp_pred[:, 0]
    v_pred = uvp_pred[:, 1]
    p_pred = uvp_pred[:, 2]
    vel_mag = np.sqrt(u_pred**2 + v_pred**2)

    # Create a larger figure with 6 subplots (3x2 layout)
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f"Flow Field - K={K:.4f}, n={n:.2f}, τ_y={tau_y:.4f}", fontsize=16)

    # 1. Velocity magnitude
    ax1 = plt.subplot(3, 2, 1)
    sc = ax1.scatter(points[:, 0], points[:, 1], c=vel_mag, cmap='viridis', s=3)
    plt.colorbar(sc, ax=ax1, label='Velocity magnitude')
    ax1.set_title('Velocity Magnitude')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    plot_boundary(ax1, boundary_points)

    # 2. Pressure
    ax2 = plt.subplot(3, 2, 2)
    sc = ax2.scatter(points[:, 0], points[:, 1], c=p_pred, cmap='coolwarm', s=3)
    plt.colorbar(sc, ax=ax2, label='Pressure')
    ax2.set_title('Pressure')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plot_boundary(ax2, boundary_points)

    # 3. U-velocity (x-direction)
    ax3 = plt.subplot(3, 2, 3)
    sc = ax3.scatter(points[:, 0], points[:, 1], c=u_pred, cmap='coolwarm', s=3)
    plt.colorbar(sc, ax=ax3, label='U-velocity (x-direction)')
    ax3.set_title('U-Velocity Component')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    plot_boundary(ax3, boundary_points)

    # 4. V-velocity (y-direction)
    ax4 = plt.subplot(3, 2, 4)
    sc = ax4.scatter(points[:, 0], points[:, 1], c=v_pred, cmap='coolwarm', s=3)
    plt.colorbar(sc, ax=ax4, label='V-velocity (y-direction)')
    ax4.set_title('V-Velocity Component')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    plot_boundary(ax4, boundary_points)

    # 5. Vector field
    ax5 = plt.subplot(3, 2, 5)
    skip = max(1, len(points) // 200)  # Reduce number of vectors for clarity
    ax5.quiver(points[::skip, 0], points[::skip, 1],
               u_pred[::skip], v_pred[::skip],
               vel_mag[::skip], cmap='viridis',
               angles='xy', scale_units='xy', scale=5)
    ax5.set_title('Velocity Field (Vector Plot)')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    plot_boundary(ax5, boundary_points)

    # 6. Streamlines
    ax6 = plt.subplot(3, 2, 6)

    # Create a finer regular grid for interpolation
    x_grid = np.linspace(-W/2, L_horizontal, 150)
    y_grid = np.linspace(-W/2, L_vertical, 150)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Use linear interpolation for velocities
    u_interp = griddata((points[:, 0], points[:, 1]), u_pred, (X, Y), method='linear', fill_value=0)
    v_interp = griddata((points[:, 0], points[:, 1]), v_pred, (X, Y), method='linear', fill_value=0)

    # Create a mask for the L-shape
    mask = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            mask[i, j] = inside_L_pipe(X[i, j], Y[i, j])

    # Apply mask to velocity fields
    u_masked = np.ma.array(u_interp, mask=~mask)
    v_masked = np.ma.array(v_interp, mask=~mask)

    # Smooth the velocity fields for better streamlines
    from scipy.ndimage import gaussian_filter
    u_smooth = gaussian_filter(u_masked.filled(0), sigma=1.5)
    v_smooth = gaussian_filter(v_masked.filled(0), sigma=1.5)
    u_smooth = np.ma.array(u_smooth, mask=~mask)
    v_smooth = np.ma.array(v_smooth, mask=~mask)

    # Plot streamlines with improved parameters
    ax6.streamplot(X, Y, u_smooth, v_smooth, 
                  density=[1.0, 2.0],  # Higher density for more streamlines
                  color='black',
                  linewidth=0.8,
                  arrowsize=1.0)
    ax6.set_title('Streamlines')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    plot_boundary(ax6, boundary_points)
    
    # Ensure axes have same scale
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_aspect('equal')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(f"plots/flow_field_{phase}.png", dpi=300)
    plt.close(fig)

    # Create a second figure for velocity profiles
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"Velocity Profiles - K={K:.4f}, n={n:.2f}, τ_y={tau_y:.4f}", fontsize=16)

    # Find cross-section points for velocity profiles
    x_cross = 0
    y_cross = 0

    # Filter points close to cross-sections
    x_tol = 0.02
    y_tol = 0.02

    # Vertical cross-section
    v_cross_idx = np.where(np.abs(points[:, 0] - x_cross) < x_tol)[0]
    if len(v_cross_idx) > 0:
        v_cross_points = points[v_cross_idx]
        v_cross_u = u_pred[v_cross_idx]
        v_cross_v = v_pred[v_cross_idx]

        # Sort by y-coordinate
        sort_idx = np.argsort(v_cross_points[:, 1])
        v_cross_points = v_cross_points[sort_idx]
        v_cross_u = v_cross_u[sort_idx]
        v_cross_v = v_cross_v[sort_idx]

        # Plot U velocity at x=0
        plt.subplot(2, 2, 1)
        plt.plot(v_cross_u, v_cross_points[:, 1], 'b-', linewidth=2)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
        plt.title(f'U-Velocity Profile at x={x_cross} (Vertical Section)')
        plt.xlabel('U-Velocity')
        plt.ylabel('y')
        plt.grid(True)

        # Plot V velocity at x=0
        plt.subplot(2, 2, 2)
        plt.plot(v_cross_v, v_cross_points[:, 1], 'g-', linewidth=2)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
        plt.title(f'V-Velocity Profile at x={x_cross} (Vertical Section)')
        plt.xlabel('V-Velocity')
        plt.ylabel('y')
        plt.grid(True)

    # Horizontal cross-section
    h_cross_idx = np.where(np.abs(points[:, 1] - y_cross) < y_tol)[0]
    if len(h_cross_idx) > 0:
        h_cross_points = points[h_cross_idx]
        h_cross_u = u_pred[h_cross_idx]
        h_cross_v = v_pred[h_cross_idx]

        # Sort by x-coordinate
        sort_idx = np.argsort(h_cross_points[:, 0])
        h_cross_points = h_cross_points[sort_idx]
        h_cross_u = h_cross_u[sort_idx]
        h_cross_v = h_cross_v[sort_idx]

        # Plot U velocity at y=0
        plt.subplot(2, 2, 3)
        plt.plot(h_cross_points[:, 0], h_cross_u, 'b-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
        plt.title(f'U-Velocity Profile at y={y_cross} (Horizontal Section)')
        plt.xlabel('x')
        plt.ylabel('U-Velocity')
        plt.grid(True)

        # Plot V velocity at y=0
        plt.subplot(2, 2, 4)
        plt.plot(h_cross_points[:, 0], h_cross_v, 'g-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
        plt.title(f'V-Velocity Profile at y={y_cross} (Horizontal Section)')
        plt.xlabel('x')
        plt.ylabel('V-Velocity')
        plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(f"plots/velocity_profiles_{phase}.png", dpi=300)
    plt.close(fig)

def train_progressive_model():
    """Train the model progressively from Newtonian to non-Newtonian behavior."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize the neural network
    model = PINN(hidden_layers=8, neurons_per_layer=256).to(device)

    # Generate domain and boundary points
    num_domain = 10000
    num_boundary = 1000

    domain_points = generate_domain_points(num_domain)
    boundary_points = generate_boundary_points(num_boundary)
    inlet_points, outlet_points, wall_points = classify_boundary_points(boundary_points)

    print(f"Domain points: {len(domain_points)}")
    print(f"Inlet points: {len(inlet_points)}")
    print(f"Outlet points: {len(outlet_points)}")
    print(f"Wall points: {len(wall_points)}")

    # Plot the geometry with points
    plot_L_shaped_pipe()

    # Define progressive training phases
    phases = [
        # Phase 1: Newtonian fluid
        {"name": "newtonian", "epochs": 1000, "K": K_init, "n": 1.0, "tau_y": 0.0, "lr": 1e-3, "lambda_bc": 100.0},
        
        # Phase 2: Introduce slight non-Newtonian behavior
        {"name": "slight_non_newtonian", "epochs": 500, "K": 0.05, "n": 0.8, "tau_y": 0.0, "lr": 5e-4, "lambda_bc": 20.0},
        
        # Phase 3: Moderate non-Newtonian behavior
        {"name": "moderate_non_newtonian", "epochs": 500, "K": 0.2, "n": 0.6, "tau_y": 0.05, "lr": 2e-4, "lambda_bc": 50.0},
        
        # Phase 4: Full non-Newtonian behavior
        {"name": "full_non_newtonian", "epochs": 1000, "K": K_final, "n": n_final, "tau_y": tau_y_final, "lr": 1e-4, "lambda_bc": 100.0}
    ]

    # For tracking overall progress
    overall_loss_history = []
    best_loss = float('inf')
    start_time = time.time()

    # Train through each phase
    for phase_idx, phase in enumerate(phases):
        phase_name = phase["name"]
        epochs = phase["epochs"]
        K = phase["K"]
        n = phase["n"]
        tau_y = phase["tau_y"]
        lr = phase["lr"]
        lambda_bc = phase["lambda_bc"]
        
        print(f"\n{'='*80}")
        print(f"Phase {phase_idx+1}/{len(phases)}: {phase_name}")
        print(f"Parameters: K={K}, n={n}, tau_y={tau_y}, lr={lr}, lambda_bc={lambda_bc}")
        print(f"Training for {epochs} epochs")
        print(f"{'='*80}\n")

        # Adam optimizer with specified learning rate
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=200, factor=0.5, min_lr=lr/10
        )

        # For tracking phase progress
        phase_loss_history = []
        phase_best_loss = float('inf')
        phase_start_time = time.time()

        last_lr = optimizer.param_groups[0]['lr']

        # Display frequency
        display_every = min(500, epochs // 10)


        
        # Training loop for this phase
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute loss with current rheological parameters
            loss, loss_pde, loss_bc = compute_total_loss(
                model, domain_points, inlet_points, outlet_points, wall_points,
                K=K, n=n, tau_y=tau_y, lambda_pde=1.0, lambda_bc=lambda_bc
            )

            # Backpropagation
            loss.backward()
            
            # Gradient clipping to prevent instability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            # Store loss value
            loss_value = loss.item()
            phase_loss_history.append(loss_value)
            overall_loss_history.append(loss_value)

            # Learning rate scheduling
            scheduler.step(loss_value)

            current_lr = optimizer.param_groups[0]['lr']
            if current_lr != last_lr:
                print(f"Learning rate changed: {last_lr:.1e} → {current_lr:.1e}")
                last_lr = current_lr

            # Save best model for this phase
            if loss_value < phase_best_loss:
                phase_best_loss = loss_value
                torch.save(model.state_dict(), f"models/pinn_phase_{phase_idx+1}_{phase_name}_best.pth")
                
                # Also track overall best model
                if loss_value < best_loss:
                    best_loss = loss_value
                    torch.save(model.state_dict(), "models/pinn_overall_best.pth")

            # Display progress
            if (epoch + 1) % display_every == 0:
                current_lr = optimizer.param_groups[0]['lr']
            
                elapsed = time.time() - phase_start_time
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value:.6f}, PDE Loss: {loss_pde.item():.6f}, BC Loss: {loss_bc.item():.6f}, Time: {elapsed:.2f}s")
                
                # Visualize intermediate results occasionally
                if (epoch + 1) % (display_every * 5) == 0:
                    visualize_results(model, boundary_points, K, n, tau_y, num_points=5000, 
                                     phase=f"{phase_name}_epoch_{epoch+1}")

        # End of phase
        phase_elapsed = time.time() - phase_start_time
        print(f"\nPhase {phase_idx+1} completed in {phase_elapsed:.2f} seconds")
        print(f"Best loss for this phase: {phase_best_loss:.6f}")
        
        # Save final model for this phase
        torch.save(model.state_dict(), f"models/pinn_phase_{phase_idx+1}_{phase_name}_final.pth")
        
        # Plot loss history for this phase
        plt.figure(figsize=(10, 6))
        plt.semilogy(phase_loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Phase {phase_idx+1}: {phase_name} - Training Loss History')
        plt.grid(True)
        plt.savefig(f"plots/loss_history_phase_{phase_idx+1}.png", dpi=300)
        plt.close()
        
        # Visualize results at the end of this phase
        visualize_results(model, boundary_points, K, n, tau_y, num_points=10000, 
                         phase=f"{phase_idx+1}_{phase_name}_final")
        
        # Update domain points occasionally to better explore the domain
        if phase_idx < len(phases) - 1:
            domain_points = generate_domain_points(num_domain)
            print(f"Generated new set of {len(domain_points)} domain points for next phase")

    # End of all phases
    total_elapsed = time.time() - start_time
    print(f"\nProgressive training completed in {total_elapsed:.2f} seconds")
    print(f"Best overall loss: {best_loss:.6f}")
    
    # Plot overall loss history
    plt.figure(figsize=(12, 6))
    plt.semilogy(overall_loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Overall Training Loss History')
    
    # Add vertical lines to mark phase transitions
    cumulative_iters = 0
    for i, phase in enumerate(phases[:-1]):
        cumulative_iters += phase["epochs"]
        plt.axvline(x=cumulative_iters, color='r', linestyle='--')
        plt.text(cumulative_iters, max(overall_loss_history), f" Phase {i+1} → {i+2}", 
                rotation=90, verticalalignment='top')
    
    plt.grid(True)
    plt.savefig("plots/overall_loss_history.png", dpi=300)
    plt.close()
    
    # Load the best overall model for final visualization
    model.load_state_dict(torch.load("models/pinn_overall_best.pth"))
    
    # Final visualization with the best model and final rheological parameters
    visualize_results(model, boundary_points, K_final, n_final, tau_y_final, 
                     num_points=20000, phase="final_best")
    
    return model

def test_trained_model(model_path="models/pinn_overall_best.pth"):
    """Load and test a trained model with various rheological parameters"""
    
    # Initialize the neural network
    model = PINN(hidden_layers=8, neurons_per_layer=256).to(device)
    
    # Load trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Generate boundary points for visualization
    boundary_points = generate_boundary_points(1000)
    
    # Test with different rheological parameters
    test_params = [
        {"K": 0.01, "n": 1.0, "tau_y": 0.0, "name": "newtonian"},
        {"K": 1.0, "n": 0.5, "tau_y": 0.1, "name": "herschel_bulkley"},
        {"K": 1.0, "n": 1.0, "tau_y": 0.1, "name": "bingham_plastic"},
        {"K": 1.0, "n": 0.5, "tau_y": 0.0, "name": "power_law"}
    ]
    
    for params in test_params:
        print(f"Testing with parameters: K={params['K']}, n={params['n']}, tau_y={params['tau_y']}")
        visualize_results(model, boundary_points, 
                         params['K'], params['n'], params['tau_y'], 
                         num_points=15000, phase=f"test_{params['name']}")

if __name__ == "__main__":
    # Train the model with progressive approach
    model = train_progressive_model()
    
    # Test the trained model with different rheological parameters
    test_trained_model()
    
    print("Progressive training and testing completed successfully!")