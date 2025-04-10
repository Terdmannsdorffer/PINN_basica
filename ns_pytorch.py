import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Problem parameters
rho = 1.0
mu = 1.0
u_in = 1.0
D = 1.0  # Height of the pipe
L = 5.0  # Length of the pipe (increased from 1.0 to 5.0)

# Define the neural network architecture
class PINN(nn.Module):
    def __init__(self, hidden_layers=5, neurons_per_layer=64):
        super(PINN, self).__init__()
        
        # Input layer
        layers = [nn.Linear(2, neurons_per_layer), nn.Tanh()]
        
        # Hidden layers
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
        
        # Output layer (u, v, p)
        layers.append(nn.Linear(neurons_per_layer, 3))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights (Glorot uniform initialization)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        return self.net(x)

# Function to compute gradients
def compute_gradients(y, x):
    """Compute dy/dx for each component of y with respect to x"""
    if len(y.shape) == 1:
        y = y.unsqueeze(1)  # Convert to shape [batch_size, 1]
    
    gradients = []
    for i in range(y.shape[1]):
        grad_outputs = torch.zeros_like(y)
        grad_outputs[:, i] = 1
        grads = grad(outputs=y, inputs=x, 
                     grad_outputs=grad_outputs, 
                     create_graph=True, 
                     retain_graph=True)[0]
        gradients.append(grads)
    return gradients

# Define the PDE residuals
def navier_stokes_residual(x, model):
    x.requires_grad_(True)
    
    # Forward pass
    uvp = model(x)
    u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
    
    # First-order derivatives for u
    du_grad = grad(u.sum(), x, create_graph=True)[0]
    du_dx, du_dy = du_grad[:, 0:1], du_grad[:, 1:2]
    
    # First-order derivatives for v
    dv_grad = grad(v.sum(), x, create_graph=True)[0]
    dv_dx, dv_dy = dv_grad[:, 0:1], dv_grad[:, 1:2]
    
    # First-order derivatives for p
    dp_grad = grad(p.sum(), x, create_graph=True)[0]
    dp_dx, dp_dy = dp_grad[:, 0:1], dp_grad[:, 1:2]
    
    # Second-order derivatives for u (x direction)
    du_xx = grad(du_dx.sum(), x, create_graph=True)[0][:, 0:1]
    
    # Second-order derivatives for u (y direction)
    du_yy = grad(du_dy.sum(), x, create_graph=True)[0][:, 1:2]
    
    # Second-order derivatives for v (x direction)
    dv_xx = grad(dv_dx.sum(), x, create_graph=True)[0][:, 0:1]
    
    # Second-order derivatives for v (y direction)
    dv_yy = grad(dv_dy.sum(), x, create_graph=True)[0][:, 1:2]
    
    # PDE residuals
    momentum_x = u * du_dx + v * du_dy + (1/rho) * dp_dx - (mu/rho) * (du_xx + du_yy)
    momentum_y = u * dv_dx + v * dv_dy + (1/rho) * dp_dy - (mu/rho) * (dv_xx + dv_yy)
    continuity = du_dx + dv_dy
    
    return momentum_x, momentum_y, continuity

# Generate domain points
def generate_domain_points(num_points):
    # Generate uniform random points in the domain [-L/2, L/2] × [-D/2, D/2]
    x = torch.FloatTensor(num_points, 2).uniform_(-L/2, L/2).to(device)
    # The second column (y-coordinates) needs to be resampled
    x[:, 1] = torch.FloatTensor(num_points).uniform_(-D/2, D/2).to(device)
    return x

# Generate boundary points
def generate_boundary_points(num_boundary_points):
    # Wall points (top and bottom)
    num_wall = num_boundary_points // 2
    num_wall_top = num_wall // 2
    num_wall_bottom = num_wall - num_wall_top
    
    # Top wall points
    x_wall_top = torch.FloatTensor(num_wall_top).uniform_(-L/2, L/2)
    y_wall_top = torch.ones_like(x_wall_top) * (D/2)
    wall_points_top = torch.stack([x_wall_top, y_wall_top], dim=1)
    
    # Bottom wall points
    x_wall_bottom = torch.FloatTensor(num_wall_bottom).uniform_(-L/2, L/2)
    y_wall_bottom = torch.ones_like(x_wall_bottom) * (-D/2)
    wall_points_bottom = torch.stack([x_wall_bottom, y_wall_bottom], dim=1)
    
    wall_points = torch.cat([wall_points_top, wall_points_bottom], dim=0)
    
    # Inlet points
    num_inlet = num_boundary_points // 4
    y_inlet = torch.FloatTensor(num_inlet).uniform_(-D/2, D/2)
    x_inlet = torch.ones_like(y_inlet) * (-L/2)
    inlet_points = torch.stack([x_inlet, y_inlet], dim=1)
    
    # Outlet points
    num_outlet = num_boundary_points - num_wall - num_inlet
    y_outlet = torch.FloatTensor(num_outlet).uniform_(-D/2, D/2)
    x_outlet = torch.ones_like(y_outlet) * (L/2)
    outlet_points = torch.stack([x_outlet, y_outlet], dim=1)
    
    return wall_points.to(device), inlet_points.to(device), outlet_points.to(device)

# Boundary conditions
def boundary_loss(model, wall_points, inlet_points, outlet_points):
    # Wall BC: u=0, v=0
    wall_pred = model(wall_points)
    wall_u_loss = torch.mean(torch.square(wall_pred[:, 0]))
    wall_v_loss = torch.mean(torch.square(wall_pred[:, 1]))
    
    # Inlet BC: u=u_in, v=0
    inlet_pred = model(inlet_points)
    inlet_u_loss = torch.mean(torch.square(inlet_pred[:, 0] - u_in))
    inlet_v_loss = torch.mean(torch.square(inlet_pred[:, 1]))
    
    # Outlet BC: p=0, v=0
    outlet_pred = model(outlet_points)
    outlet_p_loss = torch.mean(torch.square(outlet_pred[:, 2]))
    outlet_v_loss = torch.mean(torch.square(outlet_pred[:, 1]))
    
    return wall_u_loss + wall_v_loss + inlet_u_loss + inlet_v_loss + outlet_p_loss + outlet_v_loss

# Training function
def train_pinn(model, optimizer, num_epochs, domain_points, wall_points, inlet_points, outlet_points, scheduler=None):
    loss_history = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # PDE residuals
        momentum_x, momentum_y, continuity = navier_stokes_residual(domain_points, model)
        pde_loss = torch.mean(torch.square(momentum_x)) + \
                   torch.mean(torch.square(momentum_y)) + \
                   torch.mean(torch.square(continuity))
        
        # Boundary conditions
        bc_loss = boundary_loss(model, wall_points, inlet_points, outlet_points)
        
        # Total loss
        total_loss = pde_loss + bc_loss
        
        # Backpropagation
        total_loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Record loss
        loss_history.append(total_loss.item())
        
        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.6f}, "
                  f"PDE Loss: {pde_loss.item():.6f}, BC Loss: {bc_loss.item():.6f}")
    
    return loss_history

# Function for LBFGS optimization
def train_lbfgs(model, domain_points, wall_points, inlet_points, outlet_points, maxiter=3000):
    # Define closure function for LBFGS
    def closure():
        optimizer.zero_grad()
        
        # PDE residuals
        momentum_x, momentum_y, continuity = navier_stokes_residual(domain_points, model)
        pde_loss = torch.mean(torch.square(momentum_x)) + \
                   torch.mean(torch.square(momentum_y)) + \
                   torch.mean(torch.square(continuity))
        
        # Boundary conditions
        bc_loss = boundary_loss(model, wall_points, inlet_points, outlet_points)
        
        # Total loss
        total_loss = pde_loss + bc_loss
        total_loss.backward()
        
        # Print progress
        if closure.counter % 10 == 0:
            print(f"LBFGS Iteration {closure.counter}, Loss: {total_loss.item():.6f}")
        
        closure.counter += 1
        return total_loss
    
    closure.counter = 0
    
    # Initialize optimizer
    optimizer = optim.LBFGS(model.parameters(), 
                           lr=1.0, 
                           max_iter=maxiter,
                           max_eval=maxiter*2,
                           tolerance_grad=1e-9,
                           tolerance_change=1e-12,
                           history_size=50,
                           line_search_fn="strong_wolfe")
    
    # Optimize
    optimizer.step(closure)

# Generate visualization points
def generate_visualization_grid(nx=200, ny=80):
    # More points in x direction, fewer in y direction
    x = np.linspace(-L/2, L/2, nx)
    y = np.linspace(-D/2, D/2, ny)
    X, Y = np.meshgrid(x, y)
    
    # Reshape to Nx2 array
    grid_points = np.column_stack((X.flatten(), Y.flatten()))
    return torch.FloatTensor(grid_points).to(device), X, Y, nx, ny

# Function to plot the results
def plot_results(model, device):
    # Generate grid for visualization
    grid_points, X, Y, nx, ny = generate_visualization_grid()
    
    # Predict on the grid
    with torch.no_grad():
        uvp = model(grid_points)
    
    u = uvp[:, 0].cpu().numpy().reshape(ny, nx)
    v = uvp[:, 1].cpu().numpy().reshape(ny, nx)
    p = uvp[:, 2].cpu().numpy().reshape(ny, nx)
    velocity_mag = np.sqrt(u**2 + v**2)
    
    # Plot u velocity
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.contourf(X, Y, u, cmap='jet', levels=50)
    plt.colorbar(label='u velocity')
    plt.title('u velocity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    
    # Plot v velocity
    plt.subplot(1, 3, 2)
    plt.contourf(X, Y, v, cmap='jet', levels=50)
    plt.colorbar(label='v velocity')
    plt.title('v velocity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    
    # Plot pressure
    plt.subplot(1, 3, 3)
    plt.contourf(X, Y, p, cmap='jet', levels=50)
    plt.colorbar(label='pressure')
    plt.title('pressure')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Plot streamlines - fix the array shapes for streamplot
    x = np.linspace(-L/2, L/2, nx)
    y = np.linspace(-D/2, D/2, ny)
    
    plt.figure(figsize=(15, 5))
    
    # Create a separate streamplot figure for velocity magnitude and u velocity
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    strm = plt.streamplot(x, y, u.T, v.T, density=1.5, color=velocity_mag.T, cmap='jet')
    plt.colorbar(strm.lines, label='velocity magnitude')
    plt.title('Streamlines colored by velocity magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    
    # Add another subplot showing streamlines colored by u velocity
    plt.subplot(1, 2, 2)
    # For coloring by u velocity, matplotlib expects a 2D array with same shape as x/y meshgrid
    lw = 5*velocity_mag.T/velocity_mag.T.max() + 0.5  # Linewidth based on velocity magnitude
    strm2 = plt.streamplot(x, y, u.T, v.T, density=1.5, linewidth=lw, color='red')
    plt.title('Streamlines with width proportional to velocity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Another visualization showing only horizontal flow with arrows
    plt.figure(figsize=(15, 5))
    # Downsample for clearer arrow plot
    skip = (slice(None, None, 5), slice(None, None, 5))
    plt.quiver(X[skip], Y[skip], u[skip], v[skip], u[skip], cmap='jet')
    plt.colorbar(label='u velocity')
    plt.title('Vector field colored by u velocity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Main function to run the PINN
def run_pinn():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create the model
    model = PINN(hidden_layers=5, neurons_per_layer=64).to(device)
    
    # Generate training points - increase number for a larger domain
    num_domain = 5000  # Increased from 2000
    num_boundary = 500  # Increased from 200
    
    domain_points = generate_domain_points(num_domain)
    wall_points, inlet_points, outlet_points = generate_boundary_points(num_boundary)
    
    # Plot training points
    plt.figure(figsize=(15, 5))  # Wider figure for longer pipe
    plt.scatter(domain_points.cpu().numpy()[:, 0], domain_points.cpu().numpy()[:, 1], 
                s=0.5, label='Domain')
    plt.scatter(wall_points.cpu().numpy()[:, 0], wall_points.cpu().numpy()[:, 1], 
                s=2, label='Wall')
    plt.scatter(inlet_points.cpu().numpy()[:, 0], inlet_points.cpu().numpy()[:, 1], 
                s=2, label='Inlet')
    plt.scatter(outlet_points.cpu().numpy()[:, 0], outlet_points.cpu().numpy()[:, 1], 
                s=2, label='Outlet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.axis('equal')  # Keep aspect ratio equal
    plt.show()
    
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Train with Adam
    print("Training with Adam optimizer...")
    loss_history = train_pinn(model, optimizer, 10000, domain_points, wall_points, inlet_points, outlet_points)
    
    # Plot loss history
    plt.figure(figsize=(10, 6))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History (Adam)')
    plt.grid(True)
    plt.show()
    
    # Train with L-BFGS
    print("Training with L-BFGS optimizer...")
    train_lbfgs(model, domain_points, wall_points, inlet_points, outlet_points, maxiter=3000)
    
    # Visualize results
    plot_results(model, device)
    
    return model

if __name__ == "__main__":
    model = run_pinn()