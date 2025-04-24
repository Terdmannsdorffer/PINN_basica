# Modified boundary_conditions.py to fix flow direction
import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def compute_momentum_bc(model, wall_points, wall_normals, restitution_coef=1.0, friction_coef=0.05):
    print("Computing Carbopol momentum BC...")
    
    xy_wall = torch.tensor(wall_points, dtype=torch.float32, device=device, requires_grad=True)
    normals = torch.tensor(wall_normals, dtype=torch.float32, device=device)
    
    outputs = model(xy_wall)
    
    u_wall = outputs[:, 0:1]
    v_wall = outputs[:, 1:2]
    vel_wall = torch.cat([u_wall, v_wall], dim=1)
    
    normal_vel = torch.sum(vel_wall * normals, dim=1, keepdim=True)
    tangential_vel = vel_wall - normal_vel * normals
    tangential_vel_mag = torch.sqrt(torch.sum(tangential_vel**2, dim=1, keepdim=True) + 1e-10)
    
    # Calculate shear rate at the wall
    shear_rate = tangential_vel_mag / 0.01
    
    # Carbopol wall slip behavior
    tau_y = 5.0
    slip_factor = torch.sigmoid(shear_rate - tau_y)
    slip_loss = torch.mean((1.0 - slip_factor) * tangential_vel_mag**2)
    
    normal_loss = torch.mean(normal_vel**2)
    
    return normal_loss + 0.5 * slip_loss