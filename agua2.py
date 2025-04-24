import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import time
import os
import sys
import traceback

# Basic error handling
try:
    from scipy.interpolate import griddata
    from scipy.ndimage import gaussian_filter
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install missing packages with: pip install scipy matplotlib")
    sys.exit(1)

# Debug output
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)
try:
    import scipy
    print("SciPy version:", scipy.__version__)
except:
    print("SciPy not found")

# Check if CUDA is available with more details
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Try to allocate a small tensor to verify CUDA works
    try:
        test_tensor = torch.zeros(10, 10).to(device)
        print("CUDA test allocation successful")
    except RuntimeError as e:
        print(f"CUDA test allocation failed: {e}")
        print("Falling back to CPU")
        device = torch.device('cpu')
else:
    device = torch.device('cpu')
    print(f"CUDA not available, using device: {device}")

# Create directories for saving results (with error handling)
try:
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    print("Created output directories successfully")
except Exception as e:
    print(f"Error creating directories: {e}")
    print("Will continue but saving results may fail")

# Physical parameters
rho = 1000.0   # Water density (kg/m³)
mu = 1.0e-3    # Water dynamic viscosity (kg/(m·s))
u_in = 0.5     # Inlet velocity (m/s)
g = -9.81      # Gravity (acting in negative y-direction)

# Geometry parameters
L_vertical = 2.0   # Length of vertical section (m)
L_horizontal = 3.0   # Length of horizontal section (m)
W = 0.5    # Width of the pipe (m)

# Momentum conservation parameters
restitution_coef = 1.0  # Coefficient of restitution (1.0 = perfect elastic collision)
friction_coef = 0.0     # Friction coefficient (0.0 = frictionless)

#--------------------------------------------------------------
# NETWORK ARCHITECTURE
#--------------------------------------------------------------
# Simplified network architecture for debugging
class PINN(nn.Module):
    """Physics-Informed Neural Network (PINN) for fluid flow problems"""
    def __init__(self, hidden_layers=3, neurons_per_layer=40):
        super(PINN, self).__init__()

        # Input layer (2 features: x, y coordinates)
        self.input_layer = nn.Linear(2, neurons_per_layer)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))

        # Output layer (3 outputs: u, v, p)
        self.output_layer = nn.Linear(neurons_per_layer, 3)

    def forward(self, x):
        x = torch.tanh(self.input_layer(x))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        x = self.output_layer(x)
        return x

#--------------------------------------------------------------
# DOMAIN AND BOUNDARY POINTS
#--------------------------------------------------------------
def inside_L_pipe(x, y):
    """Check if point (x, y) is inside the L-shaped pipe."""
    in_vertical = (-W/2 <= x <= W/2) and (-W/2 <= y <= L_vertical)
    in_horizontal = (-W/2 <= x <= L_horizontal) and (-W/2 <= y <= W/2)
    # Ensure the corner region is included correctly
    is_inside = in_vertical or in_horizontal
    return is_inside

def generate_domain_points(num_points, randomize=True):
    """Generate points inside the L-shaped pipe."""
    print(f"Generating {num_points} domain points")
    
    # Simple approach for debugging
    if randomize:
        points = []
        attempts = 0
        max_attempts = num_points * 10  # Limit the number of attempts
        
        while len(points) < num_points and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(-W/2, L_horizontal)
            y = np.random.uniform(-W/2, L_vertical)
            if inside_L_pipe(x, y):
                points.append([x, y])
                
        points = np.array(points)
        
        # If we couldn't generate enough points, warn but continue
        if len(points) < num_points:
            print(f"Warning: Could only generate {len(points)} domain points")
    else:
        # Grid-based point generation
        x_grid = np.linspace(-W/2, L_horizontal, int(np.sqrt(num_points)))
        y_grid = np.linspace(-W/2, L_vertical, int(np.sqrt(num_points)))
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # Filter points inside the pipe
        valid_points = []
        for i in range(len(xx.flat)):
            x, y = xx.flat[i], yy.flat[i]
            if inside_L_pipe(x, y):
                valid_points.append([x, y])
                
        points = np.array(valid_points)
        
        if len(points) < num_points:
            print(f"Warning: Grid approach generated {len(points)} domain points")
    
    print(f"Generated {len(points)} domain points")
    return points

def generate_boundary_points(num_points):
    """Generate points on the boundary with normal vectors."""
    print(f"Generating {num_points} boundary points")
    
    # Define the segments of the L-shaped pipe boundary
    segments = [
        # Left wall (bottom to top)
        [(-W/2, -W/2), (-W/2, L_vertical)], # Wall
        # Top wall (left to right) -> Inlet
        [(-W/2, L_vertical), (W/2, L_vertical)], # Inlet
        # Right upper wall (top to corner)
        [(W/2, L_vertical), (W/2, W/2)], # Wall
        # Top horizontal wall (corner to right)
        [(W/2, W/2), (L_horizontal, W/2)], # Wall
        # Right wall (top to bottom) -> Outlet
        [(L_horizontal, W/2), (L_horizontal, -W/2)], # Outlet
        # Bottom wall (right to left)
        [(L_horizontal, -W/2), (-W/2, -W/2)] # Wall
    ]
    segment_types = ['wall', 'inlet', 'wall', 'wall', 'outlet', 'wall']
    
    # Calculate normal vectors for each wall segment
    normal_vectors = []
    for (x1, y1), (x2, y2) in segments:
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            # Normal vector (90 degrees counterclockwise rotation)
            nx, ny = -dy/length, dx/length
        else:
            nx, ny = 0, 0
        normal_vectors.append((nx, ny))
    
    # Points per segment (simplified for debugging)
    points_per_segment = [num_points // 6] * 6
    
    # Generate and classify points
    inlet_points = []
    outlet_points = []
    wall_points = []
    wall_normals = []  # Store normal vectors for wall points

    for i, ((x1, y1), (x2, y2)) in enumerate(segments):
        segment_points = points_per_segment[i]
        segment_type = segment_types[i]
        normal_vector = normal_vectors[i]

        # Generate points along this segment
        x_coords = np.linspace(x1, x2, segment_points)
        y_coords = np.linspace(y1, y2, segment_points)

        for j in range(segment_points):
            x = x_coords[j]
            y = y_coords[j]
            point = [x, y]

            if segment_type == 'inlet':
                inlet_points.append(point)
            elif segment_type == 'outlet':
                outlet_points.append(point)
            else: # wall
                wall_points.append(point)
                wall_normals.append(normal_vector)

    print(f"Generated {len(inlet_points)} inlet points")
    print(f"Generated {len(outlet_points)} outlet points")
    print(f"Generated {len(wall_points)} wall points")
    
    return (np.array(inlet_points + outlet_points + wall_points), 
            np.array(inlet_points), 
            np.array(outlet_points), 
            np.array(wall_points), 
            np.array(wall_normals))

#--------------------------------------------------------------
# PDE RESIDUALS
#--------------------------------------------------------------
def NS_residual(model, x, y):
    """
    Compute the Navier-Stokes residuals for incompressible flow.
    """
    try:
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

        # Navier-Stokes equations (Incompressible, Newtonian)
        # x-momentum
        momentum_x = rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)

        # y-momentum (including gravity force)
        momentum_y = rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy) - (rho * g)

        # Continuity
        continuity = u_x + v_y

        return momentum_x, momentum_y, continuity
    except Exception as e:
        print(f"Error in NS_residual: {e}")
        traceback.print_exc()
        # Return zeros with appropriate shape
        zero = torch.zeros(len(x), 1).to(device)
        return zero, zero, zero

#--------------------------------------------------------------
# MOMENTUM-PRESERVING BOUNDARY CONDITIONS
#--------------------------------------------------------------
def compute_momentum_preserving_bc(model, wall_points, wall_normals, 
                                  restitution_coef=restitution_coef, 
                                  friction_coef=friction_coef):
    """
    Compute boundary conditions that preserve momentum during collisions with walls.
    """
    try:
        if len(wall_points) == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Convert to tensors
        xy_wall = torch.tensor(wall_points, dtype=torch.float32, device=device)
        wall_normals_tensor = torch.tensor(wall_normals, dtype=torch.float32, device=device)
        
        # Get velocities at wall
        with torch.no_grad():
            wall_outputs = model(xy_wall)
        
        u_wall = wall_outputs[:, 0:1]
        v_wall = wall_outputs[:, 1:2]
        
        # Extract wall velocity components into normal and tangential directions
        vel_wall = torch.cat([u_wall, v_wall], dim=1)
        
        # Normal and tangential components at wall
        normal_vel_wall = torch.sum(vel_wall * wall_normals_tensor, dim=1, keepdim=True)
        tangent_normals = torch.stack([-wall_normals_tensor[:, 1], wall_normals_tensor[:, 0]], dim=1)
        tangent_vel_wall = torch.sum(vel_wall * tangent_normals, dim=1, keepdim=True)
        
        # Robin boundary condition for momentum preservation
        # For normal component: should be zero at wall (impermeability)
        normal_bc_loss = normal_vel_wall**2  # Normal velocity should be zero at wall
        
        # For tangential component: simple friction model
        tangent_bc_loss = (friction_coef * tangent_vel_wall)**2
        
        return normal_bc_loss, tangent_bc_loss
    except Exception as e:
        print(f"Error in momentum BC: {e}")
        traceback.print_exc()
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

#--------------------------------------------------------------
# LOSS FUNCTION
#--------------------------------------------------------------
def compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals,
                 lambda_physics=1.0, lambda_bc=10.0, lambda_inlet=5.0, lambda_outlet=3.0, 
                 lambda_wall_normal=10.0, lambda_wall_tangent=1.0):
    """
    Compute the total loss with balanced weighting between different components.
    """
    try:
        # Initialize losses
        physics_loss = torch.tensor(0.0, device=device)
        bc_loss = torch.tensor(0.0, device=device)
        
        # --- Physics Loss (Navier-Stokes residuals in the domain) ---
        if len(domain_points) > 0:
            # Take a subset for debugging if needed
            max_points = min(len(domain_points), 1000)  # Limit points for speed
            indices = np.random.choice(len(domain_points), max_points, replace=False)
            x_domain = domain_points[indices, 0]
            y_domain = domain_points[indices, 1]

            momentum_x, momentum_y, continuity = NS_residual(model, x_domain, y_domain)

            loss_mx = torch.mean(momentum_x**2)
            loss_my = torch.mean(momentum_y**2)
            loss_cont = torch.mean(continuity**2)

            physics_loss = loss_mx + loss_my + loss_cont

        # --- Boundary Condition Loss ---
        # Inlet BC: u = 0, v = -u_in (flow enters vertically downwards)
        if len(inlet_points) > 0:
            max_inlet = min(len(inlet_points), 100)  # Limit points for debug
            indices = np.random.choice(len(inlet_points), max_inlet, replace=False)
            xy_inlet = torch.tensor(inlet_points[indices], dtype=torch.float32, device=device)
            outputs_inlet = model(xy_inlet)
            u_inlet = outputs_inlet[:, 0:1]
            v_inlet = outputs_inlet[:, 1:2]

            inlet_loss = torch.mean(u_inlet**2) + lambda_inlet * torch.mean((v_inlet + u_in)**2)
            bc_loss += inlet_loss

        # Outlet BC: simplified - just encourage outflow
        if len(outlet_points) > 0:
            max_outlet = min(len(outlet_points), 100)  # Limit points for debug
            indices = np.random.choice(len(outlet_points), max_outlet, replace=False)
            xy_outlet = torch.tensor(outlet_points[indices], dtype=torch.float32, device=device)
            outputs_outlet = model(xy_outlet)
            u_outlet = outputs_outlet[:, 0:1]
            p_outlet = outputs_outlet[:, 2:3]

            outlet_loss = torch.mean(p_outlet**2) + lambda_outlet * torch.mean(torch.relu(-u_outlet))  # Encourage positive x-flow
            bc_loss += outlet_loss

        # Wall BC: Using momentum-preserving boundary conditions
        if len(wall_points) > 0:
            max_wall = min(len(wall_points), 200)  # Limit points for debug
            indices = np.random.choice(len(wall_points), max_wall, replace=False)
            normal_bc_loss, tangent_bc_loss = compute_momentum_preserving_bc(
                model, 
                wall_points[indices], 
                wall_normals[indices]
            )
            
            wall_loss = lambda_wall_normal * torch.mean(normal_bc_loss) + lambda_wall_tangent * torch.mean(tangent_bc_loss)
            bc_loss += wall_loss

        # --- Total Loss ---
        total_loss = lambda_physics * physics_loss + lambda_bc * bc_loss

        return total_loss, physics_loss.detach(), bc_loss.detach()
    except Exception as e:
        print(f"Error in compute_loss: {e}")
        traceback.print_exc()
        return torch.tensor(1.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

#--------------------------------------------------------------
# TRAINING FUNCTION
#--------------------------------------------------------------
def train_model(model, optimizer, domain_points, inlet_points, outlet_points, wall_points, wall_normals,
                epochs=50, lambda_physics=1.0, lambda_bc=10.0, display_every=5):
    """
    Train the PINN model.
    """
    try:
        loss_history = []
        start_time = time.time()
        
        print("Starting training loop...")
        for epoch in range(epochs):
            model.train()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            try:
                total_loss, physics_loss, bc_loss = compute_loss(
                    model, domain_points, inlet_points, outlet_points, wall_points, wall_normals,
                    lambda_physics=lambda_physics, lambda_bc=lambda_bc
                )
            except Exception as e:
                print(f"Error computing loss: {e}")
                traceback.print_exc()
                continue
            
            # Backpropagation
            try:
                total_loss.backward()
            except Exception as e:
                print(f"Error in backward pass: {e}")
                traceback.print_exc()
                continue
            
            # Gradient clipping to prevent exploding gradients
            try:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            except Exception as e:
                print(f"Error in gradient clipping: {e}")
                
            # Update weights
            try:
                optimizer.step()
            except Exception as e:
                print(f"Error in optimizer step: {e}")
                continue
            
            # Record loss
            loss_history.append(total_loss.item())
            
            # Display progress
            if (epoch + 1) % display_every == 0 or epoch == 0:
                elapsed = time.time() - start_time
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.6f}, "
                      f"Physics Loss: {physics_loss.item():.6f}, BC Loss: {bc_loss.item():.6f}, "
                      f"Time: {elapsed:.2f}s")
                
        print(f"Training finished. Final Loss: {loss_history[-1]:.6f}")

        # Plot loss history
        plt.figure(figsize=(8, 5))
        plt.plot(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.title('Training Loss History')
        plt.grid(True)
        plt.savefig("plots/loss_history.png", dpi=150)
        plt.close()

        # Save trained model
        torch.save(model.state_dict(), "models/pinn_momentum.pth")
        print("Model saved to models/pinn_momentum.pth")

        return model, loss_history
    except Exception as e:
        print(f"Error in training loop: {e}")
        traceback.print_exc()
        return model, []

#--------------------------------------------------------------
# VISUALIZATION FUNCTION
#--------------------------------------------------------------
def visualize_results(model, num_points=2000):
    """Visualize the flow field results with streamlines"""
    try:
        print("Creating visualization...")
        
        # Generate points in the domain
        vis_points = generate_domain_points(num_points)
        
        # Generate boundary points for drawing the outline
        boundary_points, _, _, _, _ = generate_boundary_points(300)
        
        # Predict using the trained model
        model.eval()
        xy_tensor = torch.tensor(vis_points, dtype=torch.float32, device=device)
        with torch.no_grad():
            outputs = model(xy_tensor).cpu().numpy()
        
        u_pred = outputs[:, 0]
        v_pred = outputs[:, 1]
        p_pred = outputs[:, 2]
        vel_mag = np.sqrt(u_pred**2 + v_pred**2)
        
        # Create grid for streamlines and contour plots
        x_min, x_max = -W/2, L_horizontal
        y_min, y_max = -W/2, L_vertical
        
        # Create a regular grid for streamplot function
        grid_size = 40  # Adjust for desired resolution
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Interpolate velocity components to grid
        points = vis_points
        U = griddata(points, u_pred, (X, Y), method='linear', fill_value=0)
        V = griddata(points, v_pred, (X, Y), method='linear', fill_value=0)
        P = griddata(points, p_pred, (X, Y), method='linear', fill_value=0)
        
        # Mask grid points outside the L-pipe
        mask = np.zeros_like(X, dtype=bool)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                mask[i, j] = inside_L_pipe(X[i, j], Y[i, j])
        
        # Apply mask - set velocities outside domain to NaN
        U_masked = np.where(mask, U, np.nan)
        V_masked = np.where(mask, V, np.nan)
        P_masked = np.where(mask, P, np.nan)
        
        # Calculate velocity magnitude on grid
        Vel_mag = np.sqrt(U_masked**2 + V_masked**2)
        
        # Apply slight smoothing to improve streamlines appearance
        U_smooth = gaussian_filter(U_masked, sigma=0.8)
        V_smooth = gaussian_filter(V_masked, sigma=0.8)
        
        # Create figure with 2x3 layout to include streamlines
        plt.figure(figsize=(18, 12))
        
        # 1. Velocity magnitude with streamlines
        plt.subplot(2, 3, 1)
        plt.contourf(X, Y, Vel_mag, levels=50, cmap='viridis')
        plt.colorbar(label='Velocity magnitude (m/s)')
        # Add streamlines
        plt.streamplot(X, Y, U_smooth, V_smooth, density=1.5, color='white', linewidth=0.8)
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
        plt.title('Velocity Magnitude with Streamlines')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.grid(True)
        
        # 2. Pressure field
        plt.subplot(2, 3, 2)
        plt.contourf(X, Y, P_masked, levels=50, cmap='coolwarm')
        plt.colorbar(label='Pressure (Pa)')
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
        plt.title('Pressure Field')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.grid(True)
        
        # 3. Velocity vector field (quiver plot)
        plt.subplot(2, 3, 3)
        # Plot only a subset of vectors for clarity
        skip = max(1, len(vis_points) // 200)
        plt.quiver(vis_points[::skip, 0], vis_points[::skip, 1], 
                  u_pred[::skip], v_pred[::skip], 
                  vel_mag[::skip], cmap='viridis', scale=20)
        plt.colorbar(label='Velocity magnitude (m/s)')
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
        plt.title('Velocity Field (Vector Plot)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.grid(True)
        
        # 4. U-velocity component
        plt.subplot(2, 3, 4)
        plt.contourf(X, Y, U_masked, levels=50, cmap='RdBu_r')
        plt.colorbar(label='U-velocity (m/s)')
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
        plt.title('U-Velocity Component')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.grid(True)
        
        # 5. V-velocity component
        plt.subplot(2, 3, 5)
        plt.contourf(X, Y, V_masked, levels=50, cmap='RdBu_r')
        plt.colorbar(label='V-velocity (m/s)')
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
        plt.title('V-Velocity Component')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.grid(True)
        
        # 6. Streamlines with color mapped to velocity magnitude
        plt.subplot(2, 3, 6)
        plt.contourf(X, Y, Vel_mag, levels=20, cmap='viridis', alpha=0.3)
        # Use streamlines with color mapped to velocity magnitude
        strm = plt.streamplot(X, Y, U_smooth, V_smooth, 
                            density=2, 
                            color=Vel_mag,
                            cmap='viridis',
                            linewidth=1,
                            arrowsize=1.2)
        plt.colorbar(strm.lines, label='Velocity magnitude (m/s)')
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=1)
        plt.title('Streamlines (Flow Paths)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("plots/flow_field_with_streamlines.png", dpi=200)
        plt.close()
        
        # Create a focused streamline plot
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Vel_mag, levels=20, cmap='viridis', alpha=0.4)
        plt.streamplot(X, Y, U_smooth, V_smooth, 
                     density=2.5, 
                     color='white',
                     linewidth=1.5,
                     arrowsize=1.5)
        plt.plot(boundary_points[:, 0], boundary_points[:, 1], 'k-', linewidth=2)
        plt.colorbar(label='Velocity magnitude (m/s)')
        plt.title('Flow Streamlines in L-shaped Pipe')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.axis('equal')
        plt.grid(True)
        plt.savefig("plots/streamlines_focused.png", dpi=200)
        plt.close()
        
        # Continue with the existing momentum reflection analysis
        analyze_reflection_behavior(model, num_points=4)
        
        print("Enhanced visualizations with streamlines saved to plots/ directory")
    except Exception as e:
        print(f"Error in visualization: {e}")
        traceback.print_exc()

def analyze_reflection_behavior(model, num_points=5):
    """Analyze the reflection behavior at walls"""
    try:
        print("Analyzing reflection behavior at walls...")
        
        # Generate wall points with normals
        _, _, _, wall_points, wall_normals = generate_boundary_points(6*num_points)
        
        # Select a few representative wall points
        indices = np.linspace(0, len(wall_points)-1, num_points, dtype=int)
        test_wall_points = wall_points[indices]
        test_wall_normals = wall_normals[indices]
        
        plt.figure(figsize=(15, 10))
        
        for i in range(num_points):
            plt.subplot(2, 3, i+1)
            
            wall_point = test_wall_points[i]
            normal = test_wall_normals[i]
            tangent = np.array([-normal[1], normal[0]])  # 90 degree rotation
            
            # Draw the wall
            wall_length = 0.3
            plt.plot([wall_point[0] - tangent[0]*wall_length, wall_point[0] + tangent[0]*wall_length],
                    [wall_point[1] - tangent[1]*wall_length, wall_point[1] + tangent[1]*wall_length],
                    'k-', linewidth=2, label='Wall')
            
            # Draw the normal vector
            plt.arrow(wall_point[0], wall_point[1], 
                    normal[0]*0.1, normal[1]*0.1, 
                    head_width=0.02, color='blue', label='Normal')
            
            # Create test points at different angles
            angles = np.linspace(0, np.pi*0.75, 5)
            distances = [0.05, 0.1, 0.15]
            
            for dist in distances:
                for angle in angles:
                    # Direction vector from the wall into the fluid
                    direction = -normal*np.cos(angle) + tangent*np.sin(angle)
                    test_point = wall_point - direction * dist
                    
                    # Check if point is inside domain
                    if inside_L_pipe(test_point[0], test_point[1]):
                        # Get velocity at test point
                        model.eval()
                        with torch.no_grad():
                            xy_tensor = torch.tensor([[test_point[0], test_point[1]]], 
                                                  dtype=torch.float32, device=device)
                            output = model(xy_tensor).cpu().numpy()[0]
                            u, v = output[0], output[1]
                        
                        # Plot velocity vector
                        plt.arrow(test_point[0], test_point[1],
                                u*0.05, v*0.05,
                                head_width=0.01, color='red', alpha=0.7)
                        
                        # Calculate reflection
                        vel_vec = np.array([u, v])
                        vel_normal = np.dot(vel_vec, normal) * normal
                        vel_tangent = vel_vec - vel_normal
                        
                        # Expected reflected velocity
                        refl_vel = -restitution_coef * vel_normal + (1-friction_coef) * vel_tangent
                        
                        # Plot expected reflection (from wall point)
                        plt.arrow(wall_point[0], wall_point[1],
                                refl_vel[0]*0.05, refl_vel[1]*0.05,
                                head_width=0.01, color='green', alpha=0.4)
            
            plt.title(f'Wall Point {i+1}')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.axis('equal')
            plt.grid(True)
            
            # Add a legend to the first subplot
            if i == 0:
                plt.legend(['Wall', 'Normal', 'Velocity', 'Expected Reflection'])
                
        plt.tight_layout()
        plt.savefig("plots/reflection_analysis.png", dpi=200)
        plt.close()
        
        print("Reflection analysis completed")
    except Exception as e:
        print(f"Error in reflection analysis: {e}")
        traceback.print_exc()

#--------------------------------------------------------------
# MAIN FUNCTION
#--------------------------------------------------------------
def main():
    """Main function with error handling"""
    try:
        print("\n===== PINN with Momentum-Preserving Boundary Conditions =====")
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Initialize model
        print("Initializing model...")
        model = PINN(hidden_layers=3, neurons_per_layer=40).to(device)
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Generate training points
        print("\nGenerating training points...")
        domain_points = generate_domain_points(num_points=2000)
        all_boundary_points, inlet_points, outlet_points, wall_points, wall_normals = generate_boundary_points(num_points=300)
        
        # Test model forward pass
        print("\nTesting model forward pass...")
        test_input = torch.tensor([[0.0, 0.0]], dtype=torch.float32).to(device)
        with torch.no_grad():  # Important fix: detach gradients for debug output
            test_output = model(test_input)
            print(f"Test output shape: {test_output.shape}")
            print(f"Test output values: {test_output.cpu().numpy()}")
        
        # Initialize optimizer
        print("\nInitializing optimizer...")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Train model
        print("\n===== Starting Training =====")
        model, loss_history = train_model(
            model, 
            optimizer,
            domain_points, 
            inlet_points, 
            outlet_points, 
            wall_points, 
            wall_normals,
            epochs=100,  # More epochs but still reasonable for testing
            lambda_physics=1.0,
            lambda_bc=10.0,
            display_every=10
        )
        
        # Visualize results
        if loss_history:  # Only if training succeeded
            print("\n===== Creating Visualization =====")
            visualize_results(model, num_points=2000)
        
        print("\n===== Execution completed successfully =====")
        return model
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()
        return None  # 