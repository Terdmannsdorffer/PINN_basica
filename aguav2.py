import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

print("===== Starting PINN script with fixed gradients =====")
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)

# Check CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: {device}")
else:
    device = torch.device('cpu')
    print(f"Using device: {device}")

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
print("Created directories")

# Define a very simple PINN model
class SimplePINN(nn.Module):
    def __init__(self):
        super(SimplePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 3)  # u, v, p outputs
        )
    
    def forward(self, x):
        return self.net(x)

# Initialize model
print("Creating model...")
model = SimplePINN().to(device)
print("Model created")

# Generate a few simple points
print("Generating points...")

# Create a basic L-shaped domain
def inside_L(x, y, W=0.5, L_v=2.0, L_h=3.0):
    vertical = (-W/2 <= x <= W/2) and (-W/2 <= y <= L_v)
    horizontal = (-W/2 <= x <= L_h) and (-W/2 <= y <= W/2)
    return vertical or horizontal

# Generate a few internal points
print("Generating internal points...")
domain_points = []
for _ in range(200):
    x = np.random.uniform(-0.5, 3.0)
    y = np.random.uniform(-0.5, 2.0)
    if inside_L(x, y):
        domain_points.append([x, y])

domain_points = np.array(domain_points)
print(f"Generated {len(domain_points)} internal points")

# Define wall segments
print("Defining walls...")
W = 0.5
L_v = 2.0
L_h = 3.0
wall_segments = [
    [(-W/2, -W/2), (-W/2, L_v)],      # Left wall
    [(-W/2, L_v), (W/2, L_v)],        # Top wall (inlet)
    [(W/2, L_v), (W/2, W/2)],         # Right upper wall
    [(W/2, W/2), (L_h, W/2)],         # Top horizontal wall
    [(L_h, W/2), (L_h, -W/2)],        # Right wall (outlet)
    [(L_h, -W/2), (-W/2, -W/2)]       # Bottom wall
]

# Generate a few boundary points 
print("Generating boundary points...")
wall_points = []
wall_normals = []
inlet_points = []
outlet_points = []

for i, ((x1, y1), (x2, y2)) in enumerate(wall_segments):
    # Generate 10 points per segment
    t_vals = np.linspace(0, 1, 10)
    for t in t_vals:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        # Calculate normal (90° counterclockwise)
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        normal = (-dy/length, dx/length) if length > 0 else (0, 0)
        
        if i == 1:  # Top wall (inlet)
            inlet_points.append([x, y])
        elif i == 4:  # Right wall (outlet)
            outlet_points.append([x, y])
        else:  # Other walls
            wall_points.append([x, y])
            wall_normals.append(normal)

wall_points = np.array(wall_points)
wall_normals = np.array(wall_normals)
inlet_points = np.array(inlet_points)
outlet_points = np.array(outlet_points)

print(f"Generated {len(wall_points)} wall points")
print(f"Generated {len(inlet_points)} inlet points")
print(f"Generated {len(outlet_points)} outlet points")

# Plot the domain and points to verify
plt.figure(figsize=(8, 6))
plt.scatter(domain_points[:, 0], domain_points[:, 1], s=1, alpha=0.5, label='Domain')
plt.scatter(wall_points[:, 0], wall_points[:, 1], s=5, color='black', label='Wall')
plt.scatter(inlet_points[:, 0], inlet_points[:, 1], s=5, color='blue', label='Inlet')
plt.scatter(outlet_points[:, 0], outlet_points[:, 1], s=5, color='red', label='Outlet')

# Plot wall normals
scale = 0.1
for i, (p, n) in enumerate(zip(wall_points, wall_normals)):
    plt.arrow(p[0], p[1], scale * n[0], scale * n[1], head_width=0.03, color='green', alpha=0.5)

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.title('L-Shaped Domain')
plt.grid(True)
plt.savefig('plots/domain.png')
plt.close()
print("Domain plot saved")

# Fixed momentum-preserving boundary condition function
def compute_momentum_bc(model, wall_points, wall_normals, restitution_coef=1.0, friction_coef=0.0):
    print("Computing momentum BC...")
    
    # Convert to tensors WITH gradient tracking
    xy_wall = torch.tensor(wall_points, dtype=torch.float32, device=device, requires_grad=True)
    normals = torch.tensor(wall_normals, dtype=torch.float32, device=device)
    
    # Forward pass - NO torch.no_grad() to allow gradient tracking
    outputs = model(xy_wall)
    
    # Extract velocities
    u_wall = outputs[:, 0:1]
    v_wall = outputs[:, 1:2]
    vel_wall = torch.cat([u_wall, v_wall], dim=1)
    
    # Compute normal and tangential components
    normal_vel = torch.sum(vel_wall * normals, dim=1, keepdim=True)
    
    # Normal component should be zero (impermeability)
    normal_loss = normal_vel**2
    
    # For simplified debugging, just return normal loss
    return torch.mean(normal_loss)

# Fixed loss function with proper gradient tracking
def compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals):
    print("Computing loss...")
    
    # Initialize losses
    physics_loss = torch.tensor(0.0, device=device, requires_grad=True)
    bc_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Simple inlet condition (v = -0.5, u = 0)
    if len(inlet_points) > 0:
        xy_inlet = torch.tensor(inlet_points, dtype=torch.float32, device=device)
        outputs = model(xy_inlet)
        u_inlet = outputs[:, 0]
        v_inlet = outputs[:, 1]
        inlet_loss = torch.mean(u_inlet**2) + torch.mean((v_inlet + 0.5)**2)
        bc_loss = bc_loss + inlet_loss  # Use addition instead of += for gradient tracking
    
    # Simple outlet condition (pressure = 0)
    if len(outlet_points) > 0:
        xy_outlet = torch.tensor(outlet_points, dtype=torch.float32, device=device)
        outputs = model(xy_outlet)
        p_outlet = outputs[:, 2]
        outlet_loss = torch.mean(p_outlet**2)
        bc_loss = bc_loss + outlet_loss
    
    # Wall boundary conditions with momentum preservation
    if len(wall_points) > 0:
        wall_loss = compute_momentum_bc(model, wall_points, wall_normals)
        bc_loss = bc_loss + 10.0 * wall_loss  # Stronger weight for wall conditions
    
    # Total loss (simplified for debugging)
    total_loss = physics_loss + bc_loss
    
    return total_loss

# Training loop with proper gradient tracking
def train_model(model, optimizer, domain_points, inlet_points, outlet_points, wall_points, wall_normals, epochs=2000):
    print(f"Starting training for {epochs} epochs...")
    
    loss_history = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Set model to training mode
        model.train()
        
        # Compute loss
        loss = compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss
        loss_history.append(loss.item())
        print(f"  Loss: {loss.item():.6f}")
    
    # Plot loss history
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig('plots/loss_history.png')
    plt.close()
    
    print("Training completed")
    
    return model

# Visualization function
# Update the visualize_results function to include streamlines
def visualize_results(model, domain_points):
    print("Creating visualization...")
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        xy_tensor = torch.tensor(domain_points, dtype=torch.float32, device=device)
        outputs = model(xy_tensor).cpu().numpy()
    
    u = outputs[:, 0]
    v = outputs[:, 1]
    p = outputs[:, 2]
    vel_mag = np.sqrt(u**2 + v**2)
    
    # First plot: Velocity magnitude
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=vel_mag, cmap='viridis')
    plt.colorbar(scatter, label='Velocity magnitude')
    plt.title('Flow Field - Velocity Magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('plots/flow_field.png')
    plt.close()
    
    # Second plot: Streamlines
    # For streamlines, we need a regular grid
    print("Creating streamline plot...")
    
    # Define grid boundaries based on L-shape domain
    x_min, x_max = -0.5, 3.0
    y_min, y_max = -0.5, 2.0
    
    # Create a regular grid
    nx, ny = 40, 30  # Number of points in x and y directions
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Flatten the grid for model prediction
    grid_points = np.column_stack((X.flatten(), Y.flatten()))
    
    # Filter points to only include those inside the L-shaped domain
    inside_mask = np.array([inside_L(px, py) for px, py in grid_points])
    grid_inside = grid_points[inside_mask]
    
    # Get predictions for the grid points
    with torch.no_grad():
        xy_grid = torch.tensor(grid_inside, dtype=torch.float32, device=device)
        grid_outputs = model(xy_grid).cpu().numpy()
    
    # Extract u and v components
    u_grid = np.zeros(X.shape)
    v_grid = np.zeros(X.shape)
    
    # Assign predicted values to the correct positions in the grid
    inside_indices = np.where(inside_mask)[0]
    for i, idx in enumerate(inside_indices):
        row, col = np.unravel_index(idx, X.shape)
        u_grid[row, col] = grid_outputs[i, 0]
        v_grid[row, col] = grid_outputs[i, 1]
    
    # Create streamline plot
    plt.figure(figsize=(10, 8))
    
    # First plot the domain outline
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    
    # Plot streamlines
    streamlines = plt.streamplot(X, Y, u_grid, v_grid, 
                                density=1.5, 
                                color='blue',
                                linewidth=1,
                                arrowsize=1.5)
    
    # Highlight inlet and outlet
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=30, label='Inlet')
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=30, label='Outlet')
    
    plt.title('Flow Field - Streamlines')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/streamlines.png')
    
    # Combined plot: Streamlines colored by velocity magnitude
    plt.figure(figsize=(12, 9))
    
    # Plot the domain outline
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    
    # Calculate velocity magnitude for coloring
    vel_mag_grid = np.sqrt(u_grid**2 + v_grid**2)
    
    # Create streamlines colored by velocity magnitude
    streamlines = plt.streamplot(X, Y, u_grid, v_grid, 
                                density=1.5,
                                color=vel_mag_grid,
                                cmap='viridis',
                                linewidth=1.5,
                                arrowsize=1.5)
    
    # Add colorbar
    cbar = plt.colorbar(streamlines.lines, label='Velocity magnitude')
    
    # Highlight inlet and outlet
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=30, label='Inlet')
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=30, label='Outlet')
    
    plt.title('Flow Field - Streamlines Colored by Velocity Magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/streamlines_colored.png')
    plt.close()
    
    print("Visualizations saved")

# Reflect analysis (simplified)
def analyze_reflection(model, wall_points, wall_normals):
    print("Analyzing reflection at walls...")
    
    # Skip if not enough wall points
    if len(wall_points) < 5:
        print("Not enough wall points for reflection analysis")
        return
    
    # Select a few wall points
    indices = np.linspace(0, len(wall_points)-1, 5, dtype=int)
    test_points = wall_points[indices]
    test_normals = wall_normals[indices]
    
    plt.figure(figsize=(15, 6))
    for i in range(len(test_points)):
        plt.subplot(1, 5, i+1)
        
        wall_pt = test_points[i]
        normal = test_normals[i]
        tangent = [-normal[1], normal[0]]  # 90° rotation
        
        # Get model prediction
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor([wall_pt], dtype=torch.float32, device=device))
            u, v = pred[0, 0].item(), pred[0, 1].item()
        
        # Draw wall
        plt.plot([wall_pt[0]-tangent[0]*0.2, wall_pt[0]+tangent[0]*0.2],
                [wall_pt[1]-tangent[1]*0.2, wall_pt[1]+tangent[1]*0.2],
                'k-', linewidth=2)
        
        # Draw normal
        plt.arrow(wall_pt[0], wall_pt[1], 
                normal[0]*0.1, normal[1]*0.1, 
                head_width=0.02, color='blue')
        
        # Draw velocity
        plt.arrow(wall_pt[0], wall_pt[1],
                u*0.1, v*0.1,
                head_width=0.02, color='red')
        
        plt.title(f'Point {i+1}')
        plt.xlim(wall_pt[0]-0.3, wall_pt[0]+0.3)
        plt.ylim(wall_pt[1]-0.3, wall_pt[1]+0.3)
        plt.axis('equal')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/reflection_analysis.png')
    plt.close()
    print("Reflection analysis saved")

# Main function
def main():
    print("Setting up optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting main training loop...")
    trained_model = train_model(
        model, optimizer, domain_points, inlet_points, outlet_points, 
        wall_points, wall_normals, epochs=200
    )
    
    print("Creating visualizations...")
    visualize_results(trained_model, domain_points)
    analyze_reflection(trained_model, wall_points, wall_normals)
    
    print("Saving model...")
    torch.save(trained_model.state_dict(), 'models/simple_pinn.pth')
    
    print("All done!")
    return trained_model

if __name__ == "__main__":
    try:
        print("Starting main function...")
        main()
        print("Script completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()