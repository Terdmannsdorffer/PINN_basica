import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
import time
import os

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories for saving results
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Physical parameters
rho = 1000.0   # Water density (kg/m³)
mu = 1.0e-3    # Water dynamic viscosity (kg/(m·s))
u_in = 0.5     # Increased inlet velocity (m/s)
g = -9.81      # Real gravity    

# Geometry parameters
L_vertical = 2.0  # Length of vertical section (m)
L_horizontal = 3.0  # Length of horizontal section (m)
W = 0.5  # Width of the pipe (m)

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for fluid flow problems
    """
    def __init__(self, hidden_layers=6, neurons_per_layer=50):
        super(PINN, self).__init__()
        
        # Input layer (2 features: x, y coordinates)
        layers = [nn.Linear(2, neurons_per_layer)]
        layers.append(nn.Tanh())
        
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
    """Generate random points inside the L-shaped pipe with more concentration near the corner."""
    # Standard points for most of the domain
    regular_points = []
    # Create a bounding box and filter points
    x_min, x_max = -W/2, L_horizontal
    y_min, y_max = -W/2, L_vertical
    
    # Generate points
    factor = 2
    x_rand = np.random.uniform(x_min, x_max, num_points * factor)
    y_rand = np.random.uniform(y_min, y_max, num_points * factor)
    
    # Filter to domain
    for i in range(len(x_rand)):
        if inside_L_pipe(x_rand[i], y_rand[i]):
            regular_points.append([x_rand[i], y_rand[i]])
            if len(regular_points) >= int(num_points * 0.7):  # 70% regular points
                break
    
    # Additional points concentrated near the corner
    corner_points = []
    corner_x = W/2
    corner_y = W/2
    radius = W * 0.75  # Area around corner to concentrate points
    
    # Generate more points around the corner
    for _ in range(num_points - len(regular_points)):
        # Random distance from corner (higher density closer to corner)
        r = radius * np.random.power(2.0)  # Power distribution for more points near corner
        theta = np.random.uniform(0, 2*np.pi)
        x = corner_x + r * np.cos(theta)
        y = corner_y + r * np.sin(theta)
        if inside_L_pipe(x, y):
            corner_points.append([x, y])
    
    # Combine points
    all_points = np.array(regular_points + corner_points)
    
    # If we have more points than needed, randomly subsample
    if len(all_points) > num_points:
        indices = np.random.choice(len(all_points), size=num_points, replace=False)
        all_points = all_points[indices]
    
    return all_points

def generate_boundary_points(num_points):
    """
    Generate points on the boundary of the L-shaped pipe.
    Returns points specifically at inlet, outlet, and walls.
    """
    # Define the segments of the L-shaped pipe boundary
    segments = [
        # Left wall (bottom to top)
        [(-W/2, -W/2), (-W/2, L_vertical)],
        
        # Top wall (left to right)
        [(-W/2, L_vertical), (W/2, L_vertical)],
        
        # Right upper wall (top to corner)
        [(W/2, L_vertical), (W/2, W/2)],
        
        # Top horizontal wall (corner to right)
        [(W/2, W/2), (L_horizontal, W/2)],
        
        # Right wall (top to bottom)
        [(L_horizontal, W/2), (L_horizontal, -W/2)],
        
        # Bottom wall (right to left)
        [(L_horizontal, -W/2), (-W/2, -W/2)]
    ]
    
    # Calculate the total boundary length
    total_length = 0
    for (x1, y1), (x2, y2) in segments:
        total_length += np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Allocate points based on segment length
    boundary_points = []
    
    for (x1, y1), (x2, y2) in segments:
        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        segment_points = max(2, int(num_points * segment_length / total_length))
        
        # Generate points along this segment
        for i in range(segment_points):
            t = i / float(segment_points - 1)
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            boundary_points.append([x, y])
    
    # Classify boundary points
    inlet_points = []
    outlet_points = []
    wall_points = []
    
    for x, y in boundary_points:
        # Inlet: top of vertical section
        if np.isclose(y, L_vertical, rtol=1e-5, atol=1e-8) and abs(x) <= W/2:
            inlet_points.append([x, y])
        # Outlet: right end of horizontal section
        elif np.isclose(x, L_horizontal, rtol=1e-5, atol=1e-8) and abs(y) <= W/2:
            outlet_points.append([x, y])
        # Wall: all other boundary points
        else:
            wall_points.append([x, y])
    
    return np.array(boundary_points), np.array(inlet_points), np.array(outlet_points), np.array(wall_points)

def NS_residual(model, x, y):
    """
    Compute the Navier-Stokes residuals for incompressible flow.
    For water (Newtonian fluid), the viscosity is constant.
    
    Returns:
    - momentum_x: x-momentum equation residual
    - momentum_y: y-momentum equation residual
    - continuity: continuity equation residual
    """
    # Convert to tensor
    xy_tensor = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32, requires_grad=True, device=device)
    
    # Forward pass
    outputs = model(xy_tensor)
    u = outputs[:, 0:1]  # x-velocity
    v = outputs[:, 1:2]  # y-velocity
    p = outputs[:, 2:3]  # pressure
    
    # Compute gradients
    # First derivatives
    u_grad = torch.autograd.grad(u.sum(), xy_tensor, create_graph=True)[0]
    v_grad = torch.autograd.grad(v.sum(), xy_tensor, create_graph=True)[0]
    p_grad = torch.autograd.grad(p.sum(), xy_tensor, create_graph=True)[0]
    
    u_x = u_grad[:, 0:1]
    u_y = u_grad[:, 1:2]
    v_x = v_grad[:, 0:1]
    v_y = v_grad[:, 1:2]
    p_x = p_grad[:, 0:1]
    p_y = p_grad[:, 1:2]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x.sum(), xy_tensor, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), xy_tensor, create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x.sum(), xy_tensor, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y.sum(), xy_tensor, create_graph=True)[0][:, 1:2]
    
    # Scale factor for better numerical conditioning
    scale_factor = 10.0  # Helps balance the equation terms
    
    # Detect points near the corner for special handling
    corner_x = W/2
    corner_y = W/2
    distance_to_corner = torch.sqrt((xy_tensor[:, 0] - corner_x)**2 + (xy_tensor[:, 1] - corner_y)**2)
    near_corner = distance_to_corner < (W * 0.3)
    
    # Apply stronger viscous effects near corner
    mu_effective = mu * torch.ones_like(u)
    mu_effective[near_corner] = mu * 2.0  # Increased viscosity near corner
    
    # Create masks for different sections of the pipe
    # For vertical section (exclude the corner region)
    in_vertical = ((xy_tensor[:, 0] >= -W/2) & 
                  (xy_tensor[:, 0] <= W/2) & 
                  (xy_tensor[:, 1] >= W/2 + 0.1) & 
                  (xy_tensor[:, 1] <= L_vertical))
    
    # For horizontal section (exclude the corner region)
    in_horizontal = ((xy_tensor[:, 0] >= W/2 + 0.1) & 
                    (xy_tensor[:, 1] >= -W/2) & 
                    (xy_tensor[:, 1] <= W/2))
    
    # Base momentum equations
    momentum_x = rho * (u * u_x + v * u_y) + p_x - mu_effective * (u_xx + u_yy)
    momentum_y = rho * (u * v_x + v * v_y) + p_y - mu_effective * (v_xx + v_yy) + (rho * g) / scale_factor
    
    # Add flow guidance terms to encourage correct flow direction
    flow_guide_strength = 0.05  # Small bias to guide but not dominate
    
    # Create flow guidance masks
    vertical_guide = torch.zeros_like(u)
    vertical_guide[in_vertical] = flow_guide_strength
    
    horizontal_guide = torch.zeros_like(u)
    horizontal_guide[in_horizontal] = flow_guide_strength
    
    # Apply flow guidance: 
    # - In vertical section: encourage downward flow (negative v)
    # - In horizontal section: encourage rightward flow (positive u)
    momentum_x = momentum_x - horizontal_guide * u  # Penalize negative u in horizontal section
    momentum_y = momentum_y + vertical_guide * v    # Penalize positive v in vertical section
    
    # Continuity: u_x + v_y = 0
    continuity = u_x + v_y
    
    return momentum_x, momentum_y, continuity

def compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, 
                lambda_physics=1.0, lambda_bc=10.0):
    """
    Compute the total loss for the PINN with enhanced outlet conditions.
    
    Args:
        model: Neural network model
        domain_points: Points inside the domain
        inlet_points, outlet_points, wall_points: Boundary points
        lambda_physics: Weight for physics loss
        lambda_bc: Weight for boundary condition loss
    
    Returns:
        total_loss: Combined loss from physics and boundary conditions
    """
    # Physics loss (Navier-Stokes residuals in the domain)
    if len(domain_points) > 0:
        x_domain = domain_points[:, 0]
        y_domain = domain_points[:, 1]
        
        momentum_x, momentum_y, continuity = NS_residual(model, x_domain, y_domain)
        
        # Scale the momentum_y loss term if gravity is active
        g_scale = 1.0 if abs(g) < 0.1 else (1.0 / (1.0 + abs(g)))
        
        physics_loss = (torch.mean(momentum_x**2) + 
                       g_scale * torch.mean(momentum_y**2) + 
                       torch.mean(continuity**2))
    else:
        physics_loss = torch.tensor(0.0, device=device)
    
    # Boundary condition loss
    bc_loss = torch.tensor(0.0, device=device)
    
    # Inlet BC: u = 0, v = -u_in (flow enters from top)
    if len(inlet_points) > 0:
        x_inlet = inlet_points[:, 0]
        y_inlet = inlet_points[:, 1]
        
        xy_inlet = torch.tensor(np.stack([x_inlet, y_inlet], axis=1), 
                               dtype=torch.float32, device=device)
        outputs_inlet = model(xy_inlet)
        u_inlet = outputs_inlet[:, 0]
        v_inlet = outputs_inlet[:, 1]
        
        inlet_loss = torch.mean(u_inlet**2) + 5.0 * torch.mean((v_inlet + u_in)**2)
        bc_loss += inlet_loss
    
    # Outlet BC: p = 0 AND positive horizontal flow (u > 0, v ≈ 0)
    # This strongly encourages flow to exit horizontally
    if len(outlet_points) > 0:
        x_outlet = outlet_points[:, 0]
        y_outlet = outlet_points[:, 1]
        
        xy_outlet = torch.tensor(np.stack([x_outlet, y_outlet], axis=1), 
                                dtype=torch.float32, device=device)
        outputs_outlet = model(xy_outlet)
        u_outlet = outputs_outlet[:, 0]
        v_outlet = outputs_outlet[:, 1]
        p_outlet = outputs_outlet[:, 2]
        
        # Target horizontal outflow velocity (positive u, zero v)
        target_u = 0.5 * u_in  # Reasonable target horizontal velocity
        
        # Enhanced outlet conditions:
        # 1. Zero pressure
        # 2. Positive horizontal velocity (u > 0)
        # 3. Near-zero vertical velocity (v ≈ 0)
        outlet_loss = (torch.mean(p_outlet**2) + 
                      3.0 * torch.mean((u_outlet - target_u)**2) + 
                      2.0 * torch.mean(v_outlet**2))
        
        # Apply higher weight to outlet conditions (2-3x more than other BCs)
        bc_loss += 3.0 * outlet_loss
    
    # Wall BC: u = v = 0 (no-slip condition)
    if len(wall_points) > 0:
        x_wall = wall_points[:, 0]
        y_wall = wall_points[:, 1]
        
        xy_wall = torch.tensor(np.stack([x_wall, y_wall], axis=1), 
                             dtype=torch.float32, device=device)
        outputs_wall = model(xy_wall)
        u_wall = outputs_wall[:, 0]
        v_wall = outputs_wall[:, 1]
        
        wall_loss = 3.0 * (torch.mean(u_wall**2) + torch.mean(v_wall**2))
        bc_loss += wall_loss
    
    # Special treatment for the corner region - enforce flow continuity
    corner_x = W/2
    corner_y = W/2
    corner_radius = W * 0.3
    
    # Generate corner points from domain points
    corner_points = []
    for i in range(len(domain_points)):
        x, y = domain_points[i, 0], domain_points[i, 1]
        dist = np.sqrt((x - corner_x)**2 + (y - corner_y)**2)
        if dist < corner_radius:
            corner_points.append([x, y])
    
    if corner_points:
        corner_points = np.array(corner_points)
        xy_corner = torch.tensor(corner_points, dtype=torch.float32, device=device)
        
        # Get divergence at corner points
        x_corner = corner_points[:, 0]
        y_corner = corner_points[:, 1]
        _, _, div_corner = NS_residual(model, x_corner, y_corner)
        
        # Enforce stronger continuity constraint at corner
        corner_loss = 5.0 * torch.mean(div_corner**2)
        bc_loss += corner_loss
    
    # Total loss
    total_loss = lambda_physics * physics_loss + lambda_bc * bc_loss
    
    return total_loss, physics_loss, bc_loss

def train_model(model, optimizer, scheduler, domain_points, inlet_points, outlet_points, wall_points, 
              epochs=5000, lambda_physics=1.0, lambda_bc=10.0, display_every=100):
    """
    Train the PINN model.
    """
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute loss
        loss, physics_loss, bc_loss = compute_loss(
            model, domain_points, inlet_points, outlet_points, wall_points,
            lambda_physics=lambda_physics, lambda_bc=lambda_bc
        )
        
        # Backpropagation
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Update weights
        optimizer.step()
        
        # Update learning rate
        scheduler.step(loss)
        
        # Store loss
        loss_value = loss.item()
        loss_history.append(loss_value)
        
        # Save best model
        if loss_value < best_loss:
            best_loss = loss_value
            torch.save(model.state_dict(), "models/best_model.pth")
        
        # Display progress
        if (epoch + 1) % display_every == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss_value:.6f}, "
                  f"Physics Loss: {physics_loss.item():.6f}, BC Loss: {bc_loss.item():.6f}, "
                  f"LR: {current_lr:.1e}, Time: {elapsed:.2f}s")
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True)
    plt.savefig("plots/loss_history.png", dpi=300)
    plt.close()
    
    # Load best model
    model.load_state_dict(torch.load("models/best_model.pth"))
    
    return model, loss_history

def visualize_results(model, num_points=10000):
    """Visualize the flow field results."""
    # Generate a grid of points inside the L-shaped pipe
    points = generate_domain_points(num_points)
    
    # Generate boundary points for visualization
    boundary_points, _, _, _ = generate_boundary_points(500)
    
    # Predict using the model
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        xy_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        outputs = model(xy_tensor).cpu().numpy()
    
    u_pred = outputs[:, 0]
    v_pred = outputs[:, 1]
    p_pred = outputs[:, 2]
    vel_mag = np.sqrt(u_pred**2 + v_pred**2)
    
    # Create a larger figure with 6 subplots (3x2 layout)
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle("L-shaped Pipe Flow (Water - Newtonian Fluid)", fontsize=16)
    
    # 1. Velocity magnitude
    ax1 = plt.subplot(3, 2, 1)
    sc = ax1.scatter(points[:, 0], points[:, 1], c=vel_mag, cmap='viridis', s=3)
    plt.colorbar(sc, ax=ax1, label='Velocity magnitude (m/s)')
    ax1.set_title('Velocity Magnitude')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    
    # Draw boundary
    ax1.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
    ax1.set_aspect('equal')
    
    # 2. Pressure
    ax2 = plt.subplot(3, 2, 2)
    sc = ax2.scatter(points[:, 0], points[:, 1], c=p_pred, cmap='coolwarm', s=3)
    plt.colorbar(sc, ax=ax2, label='Pressure (Pa)')
    ax2.set_title('Pressure')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
    ax2.set_aspect('equal')
    
    # 3. U-velocity (x-direction)
    ax3 = plt.subplot(3, 2, 3)
    sc = ax3.scatter(points[:, 0], points[:, 1], c=u_pred, cmap='coolwarm', s=3)
    plt.colorbar(sc, ax=ax3, label='U-velocity (m/s)')
    ax3.set_title('U-Velocity Component')
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
    ax3.set_aspect('equal')
    
    # 4. V-velocity (y-direction)
    ax4 = plt.subplot(3, 2, 4)
    sc = ax4.scatter(points[:, 0], points[:, 1], c=v_pred, cmap='coolwarm', s=3)
    plt.colorbar(sc, ax=ax4, label='V-velocity (m/s)')
    ax4.set_title('V-Velocity Component')
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    ax4.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
    ax4.set_aspect('equal')
    
    # 5. Vector field
    ax5 = plt.subplot(3, 2, 5)
    # Reduce number of vectors for clarity
    skip = max(1, len(points) // 200)
    ax5.quiver(points[::skip, 0], points[::skip, 1], 
              u_pred[::skip], v_pred[::skip], 
              vel_mag[::skip], cmap='viridis',
              angles='xy', scale_units='xy', scale=5)
    ax5.set_title('Velocity Field (Vector Plot)')
    ax5.set_xlabel('x (m)')
    ax5.set_ylabel('y (m)')
    ax5.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
    ax5.set_aspect('equal')
    
    # 6. Streamlines
    ax6 = plt.subplot(3, 2, 6)
    
    # Create a regular grid for streamlines
    x_grid = np.linspace(-W/2, L_horizontal, 150)
    y_grid = np.linspace(-W/2, L_vertical, 150)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Interpolate velocity components onto the grid
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
    
    # Apply smoothing for better streamlines
    from scipy.ndimage import gaussian_filter
    u_smooth = gaussian_filter(u_masked.filled(0), sigma=1.0)
    v_smooth = gaussian_filter(v_masked.filled(0), sigma=1.0)
    u_smooth = np.ma.array(u_smooth, mask=~mask)
    v_smooth = np.ma.array(v_smooth, mask=~mask)
    
    # Generate streamline starting points
    # 1. At the inlet (vertical section)
    x_start_inlet = np.linspace(-W/2 + 0.05, W/2 - 0.05, 8)
    y_start_inlet = np.ones_like(x_start_inlet) * (L_vertical - 0.05)

    # 2. Near the corner (to ensure flow continuation)
    x_start_corner = np.ones(5) * (W/2 - 0.05)
    y_start_corner = np.linspace(W/2 + 0.05, L_vertical/2, 5)

    # 3. In the horizontal section
    x_start_horiz = np.linspace(W/2 + 0.1, L_horizontal - 0.1, 8)
    y_start_horiz = np.ones_like(x_start_horiz) * (W/2 - 0.05)

    # Combine all starting points
    start_x = np.concatenate([x_start_inlet, x_start_corner, x_start_horiz])
    start_y = np.concatenate([y_start_inlet, y_start_corner, y_start_horiz])
    start_points = np.column_stack([start_x, start_y])
    
    # Plot streamlines with custom starting points
    streamplot = ax6.streamplot(X, Y, u_smooth, v_smooth, 
                              color='black',
                              linewidth=0.8,
                              arrowsize=1.0,
                              start_points=start_points)
    
    ax6.set_title('Streamlines')
    ax6.set_xlabel('x (m)')
    ax6.set_ylabel('y (m)')
    ax6.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
    ax6.set_aspect('equal')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig("plots/flow_field.png", dpi=300)
    plt.show()
    
    # Create a second figure for velocity profiles
    plt.figure(figsize=(15, 10))
    plt.suptitle("Velocity Profiles (Water - Newtonian Fluid)", fontsize=16)
    
    # Find cross-section points
    x_cross = 0  # Vertical cross-section at x=0
    y_cross = 0  # Horizontal cross-section at y=0
    
    x_tol = 0.02  # Tolerance for finding points near cross-sections
    y_tol = 0.02
    
    # Vertical cross-section (x=0)
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
        plt.title(f'U-Velocity Profile at x={x_cross}m (Vertical Section)')
        plt.xlabel('U-Velocity (m/s)')
        plt.ylabel('y (m)')
        plt.grid(True)
        
        # Plot V velocity at x=0
        plt.subplot(2, 2, 2)
        plt.plot(v_cross_v, v_cross_points[:, 1], 'g-', linewidth=2)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
        plt.title(f'V-Velocity Profile at x={x_cross}m (Vertical Section)')
        plt.xlabel('V-Velocity (m/s)')
        plt.ylabel('y (m)')
        plt.grid(True)
    
    # Horizontal cross-section (y=0)
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
        plt.title(f'U-Velocity Profile at y={y_cross}m (Horizontal Section)')
        plt.xlabel('x (m)')
        plt.ylabel('U-Velocity (m/s)')
        plt.grid(True)
        
        # Plot V velocity at y=0
        plt.subplot(2, 2, 4)
        plt.plot(h_cross_points[:, 0], h_cross_v, 'g-', linewidth=2)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
        plt.title(f'V-Velocity Profile at y={y_cross}m (Horizontal Section)')
        plt.xlabel('x (m)')
        plt.ylabel('V-Velocity (m/s)')
        plt.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig("plots/velocity_profiles.png", dpi=300)
    plt.show()

def main():
    """Main function to run the PINN training and visualization with full gravity."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize model with larger capacity
    model = PINN(hidden_layers=8, neurons_per_layer=128).to(device)
    
    # Generate training points
    num_domain = 8000  # Increased for better coverage
    num_boundary = 1500  # More boundary points
    
    domain_points = generate_domain_points(num_domain)
    _, inlet_points, outlet_points, wall_points = generate_boundary_points(num_boundary)
    
    print(f"Domain points: {len(domain_points)}")
    print(f"Inlet points: {len(inlet_points)}")
    print(f"Outlet points: {len(outlet_points)}")
    print(f"Wall points: {len(wall_points)}")
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=200, factor=0.5, min_lr=1e-6
    )
    
    # Define training parameters - use full gravity from the start
    # g is already set to -9.81 in the global parameters
    lambda_physics = 1.0
    lambda_bc = 200.0  # Strong boundary condition enforcement
    
    # Train model with full gravity
    print("\nTraining model with full gravity (g = -9.81 m/s²)...")
    model, loss_history = train_model(
        model, optimizer, scheduler, 
        domain_points, inlet_points, outlet_points, wall_points,
        epochs=5000,  # Longer training time to compensate for no progressive approach
        lambda_physics=lambda_physics, 
        lambda_bc=lambda_bc,
        display_every=100
    )
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(model, num_points=15000)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()