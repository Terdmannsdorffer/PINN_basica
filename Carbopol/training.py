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
        bc_loss += 20.0 * wall_loss

    total_loss = 0.5 * physics_loss + bc_loss
    return total_loss

def train_model(model, optimizer, domain_points, inlet_points, outlet_points, wall_points, wall_normals, epochs=3000):

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=300,  # Restart every 300 epochs
        T_mult=2,  # Double period after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    loss_history = []
    physics_loss_history = []
    bc_loss_history = []

    for epoch in range(epochs):
        model.train()
        
        # Compute losses
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
            bc_loss += 30.0 * wall_loss

        total_loss = 0.5 * physics_loss + bc_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(total_loss)

        # Store loss history
        loss_history.append(total_loss.item())
        physics_loss_history.append(physics_loss.item())
        bc_loss_history.append(bc_loss.item())

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"[{epoch:04d}] Total Loss = {total_loss.item():.6f}, Physics Loss = {physics_loss.item():.6f}, BC Loss = {bc_loss.item():.6f}")

        if epoch > 500 and total_loss.item() < 1e-4:
            print("Early stopping: convergence reached.")
            break

    # Create a comprehensive loss visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    
    # Plot all losses
    plt.subplot(2, 1, 1)
    plt.semilogy(loss_history, 'b-', label='Total Loss')
    plt.semilogy(physics_loss_history, 'g--', label='Physics Loss')
    plt.semilogy(bc_loss_history, 'r-.', label='Boundary Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss History (Log Scale)')
    plt.legend()
    plt.grid(True)
    
    # Plot all losses on linear scale
    plt.subplot(2, 1, 2)
    plt.plot(loss_history, 'b-', label='Total Loss')
    plt.plot(physics_loss_history, 'g--', label='Physics Loss')
    plt.plot(bc_loss_history, 'r-.', label='Boundary Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (linear scale)')
    plt.title('Training Loss History (Linear Scale)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("plots/carbopol_loss_detailed.png")
    
    # Create a separate visualization to check for boundary condition enforcement
    plt.figure(figsize=(10, 6))
    plt.semilogy(bc_loss_history, 'r-', label='Boundary Loss')
    
    # Check if wall loss is being well-enforced - if it's too high at the end of training,
    # it would explain streamlines passing through walls
    if len(bc_loss_history) > 100:
        # Calculate moving average to smooth the curve
        window_size = min(50, len(bc_loss_history) // 10)
        bc_loss_smooth = []
        for i in range(len(bc_loss_history) - window_size + 1):
            bc_loss_smooth.append(sum(bc_loss_history[i:i+window_size]) / window_size)
        
        plt.semilogy(range(window_size-1, len(bc_loss_history)), bc_loss_smooth, 'k--', 
                   label=f'Boundary Loss (Moving Avg, {window_size} epochs)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Boundary Condition Loss (log scale)')
    plt.title('Boundary Condition Enforcement Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/boundary_loss_analysis.png")
    plt.close()

    return model
def train_staged(model, optimizer, domain_points, inlet_points, outlet_points, wall_points, wall_normals):
    """
    Staged training approach for improved convergence:
    1. Focus on boundary conditions first
    2. Gradually introduce physics
    3. Full optimization with adaptive weights
    """
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    loss_history = []
    physics_loss_history = []
    bc_loss_history = []
    
    # Stage 1: Train with boundary conditions only
    print("Stage 1: Training with boundary conditions only (500 epochs)...")
    scheduler1 = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=1, eta_min=1e-5)
    
    for epoch in range(500):
        model.train()
        
        # Skip physics loss in this stage
        physics_loss = torch.tensor(0.0, device=device)
        bc_loss = torch.tensor(0.0, device=device)
        
        # Apply strong boundary conditions with higher weights
        if len(inlet_points) > 0:
            inlet = torch.tensor(inlet_points, dtype=torch.float32, device=device)
            output = model(inlet)
            u_inlet, v_inlet = output[:, 0], output[:, 1]
            u_in = 0.05
            inlet_loss = torch.mean(u_inlet**2) + 10 * torch.mean((v_inlet + u_in)**2)
            bc_loss += 10.0 * inlet_loss  # Increased weight
        
        if len(outlet_points) > 0:
            outlet = torch.tensor(outlet_points, dtype=torch.float32, device=device)
            output = model(outlet)
            u_outlet, v_outlet, p_outlet = output[:, 0], output[:, 1], output[:, 2]
            
            L_up, L_down = 0.097, 0.157
            H_left, H_right = 0.3, 0.1
            u_in = 0.05
            u_out = u_in * (H_left / H_right) * (L_up / L_down)
            
            outlet_loss = torch.mean((u_outlet - u_out)**2) + torch.mean(v_outlet**2)
            bc_loss += 2.0 * torch.mean(p_outlet**2) + 10.0 * outlet_loss  # Increased weight
        
        if len(wall_points) > 0:
            wall_loss = compute_momentum_bc(model, wall_points, wall_normals)
            bc_loss += 30.0 * wall_loss  # Much higher weight in stage 1
        
        # Only BC loss in stage 1
        total_loss = bc_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler1.step()
        
        loss_history.append(total_loss.item())
        physics_loss_history.append(0.0)  # No physics loss in stage 1
        bc_loss_history.append(bc_loss.item())
        
        if epoch % 100 == 0:
            print(f"[Stage 1: {epoch:04d}] BC Loss = {bc_loss.item():.6f}")
    
    # Stage 2: Introduce physics with lower weight
    print("Stage 2: Introducing physics equations (1000 epochs)...")
    optimizer.param_groups[0]['lr'] = 1e-3  # Reset learning rate
    scheduler2 = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=1e-5)
    
    for epoch in range(1000):
        model.train()
        
        # Now compute physics loss with domain points
        physics_loss = torch.tensor(0.0, device=device)
        bc_loss = torch.tensor(0.0, device=device)
        
        if len(domain_points) > 0:
            xy = torch.tensor(domain_points, dtype=torch.float32, device=device, requires_grad=True)
            output = model(xy)
            u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]

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
        
        # Apply boundary conditions with slightly reduced weights
        if len(inlet_points) > 0:
            inlet = torch.tensor(inlet_points, dtype=torch.float32, device=device)
            output = model(inlet)
            u_inlet, v_inlet = output[:, 0], output[:, 1]
            u_in = 0.05
            inlet_loss = torch.mean(u_inlet**2) + 10 * torch.mean((v_inlet + u_in)**2)
            bc_loss += 8.0 * inlet_loss  # Slightly reduced from stage 1
        
        if len(outlet_points) > 0:
            outlet = torch.tensor(outlet_points, dtype=torch.float32, device=device)
            output = model(outlet)
            u_outlet, v_outlet, p_outlet = output[:, 0], output[:, 1], output[:, 2]
            
            L_up, L_down = 0.097, 0.157
            H_left, H_right = 0.3, 0.1
            u_in = 0.05
            u_out = u_in * (H_left / H_right) * (L_up / L_down)
            
            outlet_loss = torch.mean((u_outlet - u_out)**2) + torch.mean(v_outlet**2)
            bc_loss += 1.5 * torch.mean(p_outlet**2) + 8.0 * outlet_loss
        
        if len(wall_points) > 0:
            wall_loss = compute_momentum_bc(model, wall_points, wall_normals)
            bc_loss += 25.0 * wall_loss  # Still high but slightly reduced
        
        # Low weight for physics in stage 2
        total_loss = 0.2 * physics_loss + bc_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler2.step()
        
        loss_history.append(total_loss.item())
        physics_loss_history.append(physics_loss.item())
        bc_loss_history.append(bc_loss.item())
        
        if epoch % 100 == 0:
            print(f"[Stage 2: {epoch:04d}] Total Loss = {total_loss.item():.6f}, Physics Loss = {physics_loss.item():.6f}, BC Loss = {bc_loss.item():.6f}")
    
    # Stage 3: Full physics with balanced weights
    print("Stage 3: Full physics optimization (2000 epochs)...")
    optimizer.param_groups[0]['lr'] = 5e-4  # Lower learning rate for final stage
    scheduler3 = CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=2, eta_min=1e-6)
    
    for epoch in range(2000):
        model.train()
        
        physics_loss = torch.tensor(0.0, device=device)
        bc_loss = torch.tensor(0.0, device=device)
        
        if len(domain_points) > 0:
            xy = torch.tensor(domain_points, dtype=torch.float32, device=device, requires_grad=True)
            output = model(xy)
            u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]

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
        
        # Apply boundary conditions with balanced weights
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
            bc_loss += 20.0 * wall_loss  # Still high but balanced with physics
        
        # Balanced weights in stage 3
        total_loss = 0.5 * physics_loss + bc_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler3.step()
        
        loss_history.append(total_loss.item())
        physics_loss_history.append(physics_loss.item())
        bc_loss_history.append(bc_loss.item())
        
        if epoch % 100 == 0 or epoch == 1999:
            print(f"[Stage 3: {epoch:04d}] Total Loss = {total_loss.item():.6f}, Physics Loss = {physics_loss.item():.6f}, BC Loss = {bc_loss.item():.6f}")
        
        # Early stopping with stricter criteria
        if epoch > 1000 and total_loss.item() < 1e-5:
            print("Early stopping: high-quality convergence reached.")
            break
    
    # Create detailed loss visualizations
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 10))
    
    # Mark the different training stages
    stage1_end = 500
    stage2_end = stage1_end + 1000
    
    # Plot log scale losses
    plt.subplot(2, 1, 1)
    plt.semilogy(loss_history, 'b-', label='Total Loss')
    plt.semilogy(physics_loss_history, 'g--', label='Physics Loss')
    plt.semilogy(bc_loss_history, 'r-.', label='Boundary Loss')
    
    # Add vertical lines for stage transitions
    plt.axvline(x=stage1_end, color='k', linestyle='--', alpha=0.5, label='Stage 1→2')
    plt.axvline(x=stage2_end, color='k', linestyle=':', alpha=0.5, label='Stage 2→3')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss History (Log Scale) - Staged Training')
    plt.legend()
    plt.grid(True)
    
    # Plot linear scale
    plt.subplot(2, 1, 2)
    plt.plot(loss_history, 'b-', label='Total Loss')
    plt.plot(physics_loss_history, 'g--', label='Physics Loss')
    plt.plot(bc_loss_history, 'r-.', label='Boundary Loss')
    
    # Add vertical lines for stage transitions
    plt.axvline(x=stage1_end, color='k', linestyle='--', alpha=0.5, label='Stage 1→2')
    plt.axvline(x=stage2_end, color='k', linestyle=':', alpha=0.5, label='Stage 2→3')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (linear scale)')
    plt.title('Training Loss History (Linear Scale) - Staged Training')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("plots/staged_training_loss.png")
    plt.close()
    
    return model