import torch
from boundary_conditions import compute_momentum_bc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals):
    physics_loss = torch.tensor(0.0, device=device)
    bc_loss = torch.tensor(0.0, device=device)

    if len(domain_points) > 0:
        xy = torch.tensor(domain_points, dtype=torch.float32, device=device, requires_grad=True)
        output = model(xy)
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]

        grads = torch.autograd.grad(outputs=u.sum() + v.sum() + p.sum(), inputs=xy, create_graph=True)[0]
        u_x, u_y = torch.autograd.grad(u.sum(), xy, create_graph=True)[0].split(1, dim=1)
        v_x, v_y = torch.autograd.grad(v.sum(), xy, create_graph=True)[0].split(1, dim=1)
        p_x, p_y = torch.autograd.grad(p.sum(), xy, create_graph=True)[0].split(1, dim=1)

        shear_rate = torch.sqrt(2*((u_x)**2 + (v_y)**2) + (u_y + v_x)**2 + 1e-8)
        tau_y, k, n = 30.0, 2.8, 0.65
        eta_eff = (tau_y / (shear_rate + 1e-6)) + k * (shear_rate ** (n - 1))

        continuity = u_x + v_y
        f_x = p_x - (eta_eff * u_x + eta_eff * u_y)
        f_y = p_y - (eta_eff * v_x + eta_eff * v_y)

        physics_loss = torch.mean(continuity**2) + torch.mean(f_x**2 + f_y**2)

    if len(inlet_points) > 0:
        inlet = torch.tensor(inlet_points, dtype=torch.float32, device=device)
        output = model(inlet)
        u_inlet, v_inlet = output[:, 0], output[:, 1]
        u_in = 0.05
        inlet_loss = torch.mean(u_inlet**2) + 10 * torch.mean((v_inlet + u_in)**2)
        bc_loss += 5.0 * inlet_loss

    if len(outlet_points) > 0:
        outlet = torch.tensor(outlet_points, dtype=torch.float32, device=device)
        output = model(outlet)
        u_outlet, v_outlet, p_outlet = output[:, 0], output[:, 1], output[:, 2]

        L_up, L_down = 0.097, 0.157
        H_left, H_right = 0.3, 0.1
        u_in = 0.05
        u_out = u_in * (H_left / H_right) * (L_up / L_down)

        outlet_loss = torch.mean((u_outlet - u_out)**2) + torch.mean(v_outlet**2)
        bc_loss += torch.mean(p_outlet**2) + 5.0 * outlet_loss

    if len(wall_points) > 0:
        wall_loss = compute_momentum_bc(model, wall_points, wall_normals)
        bc_loss += 10.0 * wall_loss

    total_loss = 0.5 * physics_loss + bc_loss
    return total_loss

def train_model(model, optimizer, domain_points, inlet_points, outlet_points, wall_points, wall_normals, epochs=3000):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, verbose=True)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        loss = compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        loss_history.append(loss.item())

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"[{epoch:04d}] Loss = {loss.item():.6f}")

        if epoch > 500 and loss.item() < 1e-4:
            print("Early stopping: convergence reached.")
            break

    import matplotlib.pyplot as plt
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.savefig("plots/carbopol_loss_history.png")
    plt.close()

    return model
