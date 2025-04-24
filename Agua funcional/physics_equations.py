import torch

# Check CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def compute_physics_loss(model, domain_points, density=1.0, viscosity=0.01, gravity=(0, -9.81)):
    """
    Compute physics loss based on Navier-Stokes equations with gravity.
    
    Args:
        model: The neural network model
        domain_points: Interior points within the fluid domain
        density: Fluid density (default: 1.0)
        viscosity: Fluid viscosity (default: 0.01)
        gravity: Gravity vector as (gx, gy) (default: (0, -9.81))
    
    Returns:
        Physics loss incorporating Navier-Stokes equations with gravity
    """
    # Convert gravity to tensor
    g_x, g_y = gravity
    g = torch.tensor([g_x, g_y], dtype=torch.float32, device=device)
    
    # Convert to tensor with gradient tracking
    xy = torch.tensor(domain_points, dtype=torch.float32, device=device, requires_grad=True)
    
    # Forward pass
    outputs = model(xy)
    
    # Extract velocity and pressure
    u = outputs[:, 0:1]  # x-velocity
    v = outputs[:, 1:2]  # y-velocity
    p = outputs[:, 2:3]  # pressure
    
    # Calculate spatial derivatives
    grad_u = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
    grad_v = torch.autograd.grad(v.sum(), xy, create_graph=True)[0]
    grad_p = torch.autograd.grad(p.sum(), xy, create_graph=True)[0]
    
    # First derivatives
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    v_x = grad_v[:, 0:1]
    v_y = grad_v[:, 1:2]
    p_x = grad_p[:, 0:1]
    p_y = grad_p[:, 1:2]
    
    # Second derivatives for viscosity terms
    u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x.sum(), xy, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y.sum(), xy, create_graph=True)[0][:, 1:2]
    
    # Continuity equation: div(v) = 0
    continuity = u_x + v_y
    continuity_loss = torch.mean(continuity**2)
    
    # Momentum equations with gravity
    # x-momentum: ρ(u*u_x + v*u_y) = -p_x + μ(u_xx + u_yy) + ρ*g_x
    # y-momentum: ρ(u*v_x + v*v_y) = -p_y + μ(v_xx + v_yy) + ρ*g_y
    
    x_momentum = density * (u * u_x + v * u_y) + p_x - viscosity * (u_xx + u_yy) - density * g[0]
    y_momentum = density * (u * v_x + v * v_y) + p_y - viscosity * (v_xx + v_yy) - density * g[1]
    
    momentum_x_loss = torch.mean(x_momentum**2)
    momentum_y_loss = torch.mean(y_momentum**2)
    
    # Total physics loss
    physics_loss = continuity_loss + momentum_x_loss + momentum_y_loss
    
    return physics_loss