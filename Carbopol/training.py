import torch
import matplotlib.pyplot as plt
from boundary_conditions import compute_momentum_bc

# Check CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def compute_shear_rate(u, v, xy):
    # Compute velocity gradients with respect to xy
    du = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    dv = torch.autograd.grad(v, xy, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

    du_dx = du[:, 0:1]
    du_dy = du[:, 1:2]
    dv_dx = dv[:, 0:1]
    dv_dy = dv[:, 1:2]

    # Compute shear rate (second invariant of strain rate tensor)
    gamma_dot = torch.sqrt(2 * ((du_dx)**2 + (dv_dy)**2 + 0.5*(du_dy + dv_dx)**2))
    return gamma_dot, du_dx, du_dy, dv_dx, dv_dy

def compute_physics_loss(model, domain_points, tau_y=33.0, k=8.66, n=0.47, rho=1.0):
    xy = torch.tensor(domain_points, dtype=torch.float32, device=device, requires_grad=True)
    outputs = model(xy)
    u = outputs[:, 0:1]
    v = outputs[:, 1:2]
    p = outputs[:, 2:3]

    gamma_dot, du_dx, du_dy, dv_dx, dv_dy = compute_shear_rate(u, v, xy)

    # Effective viscosity from Herschel-Bulkley model
    eta_eff = (tau_y / (gamma_dot + 1e-6)) + k * (gamma_dot ** (n - 1))

    # Velocity gradients
    u_grad = torch.cat([du_dx, du_dy], dim=1)
    v_grad = torch.cat([dv_dx, dv_dy], dim=1)

    # Compute second derivatives (Laplacians) for viscous terms
    du_grad = torch.autograd.grad(du_dx, xy, grad_outputs=torch.ones_like(du_dx), retain_graph=True, create_graph=True)[0][:, 0:1] + \
              torch.autograd.grad(du_dy, xy, grad_outputs=torch.ones_like(du_dy), retain_graph=True, create_graph=True)[0][:, 1:2]
    dv_grad = torch.autograd.grad(dv_dx, xy, grad_outputs=torch.ones_like(dv_dx), retain_graph=True, create_graph=True)[0][:, 0:1] + \
              torch.autograd.grad(dv_dy, xy, grad_outputs=torch.ones_like(dv_dy), retain_graph=True, create_graph=True)[0][:, 1:2]

    # Pressure gradients
    dp = torch.autograd.grad(p, xy, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    dp_dx = dp[:, 0:1]
    dp_dy = dp[:, 1:2]

    # Momentum residuals (simplified NS with non-Newtonian viscosity)
    momentum_u = rho * (u * du_dx + v * du_dy) + dp_dx - eta_eff * du_grad
    momentum_v = rho * (u * dv_dx + v * dv_dy) + dp_dy - eta_eff * dv_grad

    momentum_loss = torch.mean(momentum_u**2 + momentum_v**2)

    # Continuity residual
    continuity = torch.mean((du_dx + dv_dy)**2)

    return momentum_loss + continuity

# Fixed loss function with proper gradient tracking
def compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals):
    print("Computing loss...")

    physics_loss = compute_physics_loss(model, domain_points)
    bc_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # Simple inlet condition (v = -0.5, u = 0)
    if len(inlet_points) > 0:
        xy_inlet = torch.tensor(inlet_points, dtype=torch.float32, device=device)
        outputs = model(xy_inlet)
        u_inlet = outputs[:, 0]
        v_inlet = outputs[:, 1]

        # Generate parabolic inlet profile (Poiseuille-like)
        y_coords = torch.tensor(inlet_points, dtype=torch.float32, device=device)[:, 1:2]
        y_min, y_max = torch.min(y_coords), torch.max(y_coords)
        y_center = (y_max + y_min) / 2
        H = y_max - y_min

        # Parabolic profile: v = -Vmax * (1 - (2(y - yc)/H)^2)
        v_target = -0.5 * (1 - ((2 * (y_coords - y_center) / H) ** 2))

        inlet_loss = torch.mean(u_inlet**2 + (v_inlet - v_target)**2)

        bc_loss = bc_loss + inlet_loss

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
        bc_loss = bc_loss + 10.0 * wall_loss

    total_loss = physics_loss + bc_loss
    return total_loss

def train_model(model, optimizer, domain_points, inlet_points, outlet_points, wall_points, wall_normals, epochs=2000):
    print(f"Starting training for {epochs} epochs...")

    loss_history = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        model.train()
        loss = compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print(f"  Loss: {loss.item():.6f}")

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
