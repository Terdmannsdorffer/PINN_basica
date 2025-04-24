import torch

# Check CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

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