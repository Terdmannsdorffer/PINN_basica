#boundary_conditions.py
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def compute_momentum_bc(model, wall_points, wall_normals, restitution_coef=1.0, friction_coef=0.05):
    
    
    # Convert wall points and normals to tensors
    xy_wall = torch.tensor(wall_points, dtype=torch.float32, device=device, requires_grad=True)
    normals = torch.tensor(wall_normals, dtype=torch.float32, device=device)
    
    # Get model predictions at wall points
    outputs = model(xy_wall)
    
    # Extract velocity components
    u_wall = outputs[:, 0:1]
    v_wall = outputs[:, 1:2]
    vel_wall = torch.cat([u_wall, v_wall], dim=1)
    
    # Compute normal and tangential velocity components
    normal_vel = torch.sum(vel_wall * normals, dim=1, keepdim=True)
    tangential_vel = vel_wall - normal_vel * normals
    tangential_vel_mag = torch.sqrt(torch.sum(tangential_vel**2, dim=1, keepdim=True) + 1e-10)
    
    # Calculate shear rate at the wall - adjusted for smaller domain
    # Assuming characteristic length scale of 0.005 m for boundary layer
    boundary_layer_thickness = 0.005
    shear_rate = tangential_vel_mag / boundary_layer_thickness
    
    # Carbopol wall slip behavior with updated rheological parameters
    tau_y = 35.55  # Updated yield stress in Pa
    slip_factor = torch.sigmoid(shear_rate - tau_y)
    slip_loss = torch.mean((1.0 - slip_factor) * tangential_vel_mag**2)
    
    # No-penetration constraint at walls (normal velocity = 0)
    normal_loss = 10 * torch.mean(normal_vel**2)
    
    # Return combined loss
    return normal_loss + 0.5 * slip_loss