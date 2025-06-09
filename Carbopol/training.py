# training.py - CORRECTED VERSION (Physics-Free Approach)
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_staged(model, optimizer, domain_points, inlet_points, outlet_points, wall_points, wall_normals):
    """
    IMPROVED 3-stage training with better boundary conditions
    Based on the approach that achieved 50% directional accuracy
    """
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    loss_history = []
    physics_loss_history = []
    bc_loss_history = []
    
    # Physical parameters
    rho = 0.101972  # Fluid density in kg/m^3
    g_x, g_y = 0.0, -9.81
    
    # CORRECTED geometry parameters (experimental values)
    L_up, L_down = 0.097, 0.174
    H_left, H_right = 0.119, 0.019  # EXPERIMENTAL VALUES
    
    # IMPROVED flow parameters (the ones that worked before)
    u_in = 0.0      # No horizontal flow at inlet
    v_in = -0.015   # Downward flow (this worked before)
    
    # Continuity equation with experimental areas
    inlet_area = L_up * 1.0     # 0.097 m²
    outlet_area = H_right * 1.0  # 0.019 m² 
    u_out = abs(v_in) * (inlet_area / outlet_area)  # = 0.015 * (0.097/0.019) = 0.0766 m/s
    v_out = 0.0
    
    print(f"IMPROVED 3-Stage Flow parameters:")
    print(f"  Inlet:  u={u_in:.6f} m/s, v={v_in:.6f} m/s")
    print(f"  Outlet: u={u_out:.6f} m/s, v={v_out:.6f} m/s") 
    print(f"  Area ratio: {inlet_area/outlet_area:.2f}")
    print(f"  Velocity ratio: {u_out/abs(v_in):.2f}")
    
    # Carbopol rheology parameters (same as before)
    tau_y, k, n = 30.0, 2.8, 0.65
    
    def compute_physics_loss_stable(domain_points):
        """Stable physics loss computation (same approach that worked)"""
        if len(domain_points) == 0:
            return torch.tensor(0.0, device=device)
            
        # Use reasonable subset of points
        n_points = min(1500, len(domain_points))
        indices = torch.randperm(len(domain_points))[:n_points]
        selected_points = [domain_points[i] for i in indices]
        
        xy = torch.tensor(selected_points, dtype=torch.float32, device=device, requires_grad=True)
        output = model(xy)
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]

        # Compute gradients
        u_x, u_y = torch.autograd.grad(u.sum(), xy, create_graph=True)[0].split(1, dim=1)
        v_x, v_y = torch.autograd.grad(v.sum(), xy, create_graph=True)[0].split(1, dim=1)
        p_x, p_y = torch.autograd.grad(p.sum(), xy, create_graph=True)[0].split(1, dim=1)

        # Robust rheology (same as before but with better clamping)
        shear_rate = torch.sqrt(2*((u_x)**2 + (v_y)**2) + (u_y + v_x)**2 + 1e-8)
        eta_eff = (tau_y / (shear_rate + 1e-8)) + k * torch.pow(shear_rate + 1e-8, n - 1)
        eta_eff = torch.clamp(eta_eff, min=0.01, max=500.0)  # Tighter bounds

        # Physics equations (same simplified approach)
        continuity = u_x + v_y
        
        # Simplified momentum (same as before)
        f_x = p_x - eta_eff * (u_x + u_y) + rho * g_x
        f_y = p_y - eta_eff * (v_x + v_y) + rho * g_y

        return torch.mean(continuity**2) + torch.mean(f_x**2 + f_y**2)
    
    def compute_boundary_loss_improved():
        """IMPROVED boundary condition loss with all our learnings"""
        bc_loss = torch.tensor(0.0, device=device)
        
        # IMPROVED: Inlet boundary condition
        if len(inlet_points) > 0:
            inlet = torch.tensor(inlet_points, dtype=torch.float32, device=device)
            output = model(inlet)
            u_inlet, v_inlet = output[:, 0], output[:, 1]
            
            # CORRECTED: Multiple loss terms for better enforcement
            inlet_u_mse = torch.mean((u_inlet - u_in)**2)
            inlet_v_mse = torch.mean((v_inlet - v_in)**2)
            inlet_u_mae = torch.mean(torch.abs(u_inlet - u_in))
            inlet_v_mae = torch.mean(torch.abs(v_inlet - v_in))
            
            bc_loss += 50.0 * (inlet_u_mse + inlet_v_mse) + 25.0 * (inlet_u_mae + inlet_v_mae)
        
        # IMPROVED: Outlet boundary condition
        if len(outlet_points) > 0:
            outlet = torch.tensor(outlet_points, dtype=torch.float32, device=device)
            output = model(outlet)
            u_outlet, v_outlet, p_outlet = output[:, 0], output[:, 1], output[:, 2]
            
            # CORRECTED: Multiple loss terms
            outlet_u_mse = torch.mean((u_outlet - u_out)**2)
            outlet_v_mse = torch.mean((v_outlet - v_out)**2)
            outlet_u_mae = torch.mean(torch.abs(u_outlet - u_out))
            outlet_v_mae = torch.mean(torch.abs(v_outlet - v_out))
            outlet_p_loss = torch.mean(p_outlet**2)
            
            bc_loss += 40.0 * (outlet_u_mse + outlet_v_mse) + 20.0 * (outlet_u_mae + outlet_v_mae) + 2.0 * outlet_p_loss
        
        # IMPROVED: Wall boundary condition
        if len(wall_points) > 0:
            n_wall = min(150, len(wall_points))  # Reasonable subset
            wall_indices = torch.randperm(len(wall_points))[:n_wall]
            wall_subset = [wall_points[i] for i in wall_indices]
            
            wall = torch.tensor(wall_subset, dtype=torch.float32, device=device)
            output = model(wall)
            u_wall, v_wall = output[:, 0], output[:, 1]
            
            wall_mse = torch.mean(u_wall**2 + v_wall**2)
            wall_mae = torch.mean(torch.abs(u_wall) + torch.abs(v_wall))
            
            bc_loss += 20.0 * wall_mse + 10.0 * wall_mae
        
        return bc_loss
    
    # STAGE 1: Strong BC training (1500 epochs) - same duration as before
    print(f"\nStage 1: Strong boundary condition training (1500 epochs)...")
    scheduler1 = CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=1, eta_min=1e-6)
    
    for epoch in range(1500):
        model.train()
        
        bc_loss = compute_boundary_loss_improved()
        total_loss = bc_loss  # Only boundary conditions
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Same clipping as before
        optimizer.step()
        scheduler1.step()
        
        loss_history.append(total_loss.item())
        physics_loss_history.append(0.0)
        bc_loss_history.append(bc_loss.item())
        
        if epoch % 300 == 0:
            print(f"[Stage 1: {epoch:04d}] BC Loss = {bc_loss.item():.8f}")
            
            # Detailed monitoring (improved)
            with torch.no_grad():
                if len(inlet_points) > 0:
                    inlet_sample = torch.tensor(inlet_points[:5], dtype=torch.float32, device=device)
                    pred = model(inlet_sample)
                    u_pred = pred[:, 0].mean().item()
                    v_pred = pred[:, 1].mean().item()  
                    u_error = abs(u_pred - u_in) * 1000 if u_in == 0 else abs(u_pred - u_in) / abs(u_in) * 100
                    v_error = abs(v_pred - v_in) / abs(v_in) * 100
                    print(f"  Inlet: u={u_pred:.6f} (err={u_error:.1f}), v={v_pred:.6f} (err={v_error:.1f}%)")
                
                if len(outlet_points) > 0:
                    outlet_sample = torch.tensor(outlet_points[:5], dtype=torch.float32, device=device)
                    pred = model(outlet_sample)
                    u_pred = pred[:, 0].mean().item()
                    v_pred = pred[:, 1].mean().item()
                    u_error = abs(u_pred - u_out) / u_out * 100
                    v_error = abs(v_pred - v_out) * 1000 if v_out == 0 else abs(v_pred - v_out) / abs(v_out) * 100
                    print(f"  Outlet: u={u_pred:.6f} (err={u_error:.1f}%), v={v_pred:.6f} (err={v_error:.1f})")
    
    # STAGE 2: Introduce physics (2000 epochs) - same as before
    print(f"\nStage 2: Introducing physics equations (2000 epochs)...")
    optimizer.param_groups[0]['lr'] = 5e-4  # Same learning rate as before
    scheduler2 = CosineAnnealingWarmRestarts(optimizer, T_0=400, T_mult=1, eta_min=1e-6)
    
    # Gradual physics introduction - same approach as before
    physics_weights = np.linspace(0.01, 0.2, 2000)  # More conservative than before
    
    for epoch in range(2000):
        model.train()
        
        physics_loss = compute_physics_loss_stable(domain_points)
        bc_loss = compute_boundary_loss_improved()
        
        # Gradual physics introduction
        physics_weight = physics_weights[epoch]
        total_loss = bc_loss + physics_weight * physics_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Same clipping
        optimizer.step()
        scheduler2.step()
        
        loss_history.append(total_loss.item())
        physics_loss_history.append(physics_loss.item())
        bc_loss_history.append(bc_loss.item())
        
        if epoch % 400 == 0:
            print(f"[Stage 2: {epoch:04d}] Total={total_loss.item():.6f}, BC={bc_loss.item():.6f}, Physics={physics_loss.item():.6f} (w={physics_weight:.3f})")
        
        # IMPROVED: Early intervention if physics loss starts growing
        if physics_loss.item() > 200.0:
            print(f"Physics loss high ({physics_loss.item():.2f}) - reducing physics weight")
            physics_weights[epoch:] = physics_weights[epoch] * 0.8
        
        if physics_loss.item() > 500.0:
            print(f"Physics loss too high ({physics_loss.item():.2f}) - stopping physics introduction")
            break
    
    # STAGE 3: Balanced optimization (1500 epochs) - same as before
    print(f"\nStage 3: Balanced optimization (1500 epochs)...")
    optimizer.param_groups[0]['lr'] = 2e-4  # Lower learning rate for final stage
    scheduler3 = CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=1, eta_min=1e-7)
    
    best_loss = float('inf')
    patience = 0
    max_patience = 300
    
    for epoch in range(1500):
        model.train()
        
        physics_loss = compute_physics_loss_stable(domain_points)
        bc_loss = compute_boundary_loss_improved()
        
        # Conservative balance - BC still dominates
        total_loss = bc_loss + 0.1 * physics_loss  # Even more conservative
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)  # Tighter clipping for final stage
        optimizer.step()
        scheduler3.step()
        
        loss_history.append(total_loss.item())
        physics_loss_history.append(physics_loss.item())
        bc_loss_history.append(bc_loss.item())
        
        # Early stopping with patience
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience:
            print(f"Early stopping at epoch {epoch} (converged)")
            break
        
        if epoch % 300 == 0 or epoch == 1499:
            print(f"[Stage 3: {epoch:04d}] Total={total_loss.item():.6f}, BC={bc_loss.item():.6f}, Physics={physics_loss.item():.6f}")
        
        # Safety check for Stage 3
        if physics_loss.item() > 1000.0:
            print(f"Physics loss exploding in Stage 3 - stopping")
            break
    
    # FINAL VALIDATION (improved)
    print(f"\n" + "="*60)
    print("FINAL BOUNDARY CONDITION VALIDATION")
    print("="*60)
    model.eval()
    with torch.no_grad():
        if len(inlet_points) > 0:
            inlet = torch.tensor(inlet_points, dtype=torch.float32, device=device)
            pred = model(inlet)
            u_pred = pred[:, 0].cpu().numpy()
            v_pred = pred[:, 1].cpu().numpy()
            print(f"INLET (should be downward flow):")
            print(f"  u: {u_pred.mean():.6f}±{u_pred.std():.6f} (target={u_in:.6f}) - horizontal component")
            print(f"  v: {v_pred.mean():.6f}±{v_pred.std():.6f} (target={v_in:.6f}) - vertical component")
            print(f"  u error: {abs(u_pred.mean() - u_in)*1000:.1f}‰")
            print(f"  v error: {abs(v_pred.mean() - v_in)/abs(v_in)*100:.1f}%")
        
        if len(outlet_points) > 0:
            outlet = torch.tensor(outlet_points, dtype=torch.float32, device=device)
            pred = model(outlet)
            u_pred = pred[:, 0].cpu().numpy()
            v_pred = pred[:, 1].cpu().numpy()
            p_pred = pred[:, 2].cpu().numpy()
            print(f"\nOUTLET (should be rightward flow):")
            print(f"  u: {u_pred.mean():.6f}±{u_pred.std():.6f} (target={u_out:.6f}) - horizontal component")
            print(f"  v: {v_pred.mean():.6f}±{v_pred.std():.6f} (target={v_out:.6f}) - vertical component")
            print(f"  p: {p_pred.mean():.6f}±{p_pred.std():.6f} (target=0.0000)")
            print(f"  u error: {abs(u_pred.mean() - u_out)/u_out*100:.1f}%")
            print(f"  v error: {abs(v_pred.mean() - v_out)*1000:.1f}‰")
    print("="*60)
    
    # Create loss visualization (improved)
    plt.figure(figsize=(15, 10))
    
    stage1_end = 1500
    stage2_end = stage1_end + 2000
    
    # Log scale plot
    plt.subplot(2, 1, 1)
    plt.semilogy(loss_history, 'b-', label='Total Loss', linewidth=2)
    plt.semilogy(physics_loss_history, 'g--', label='Physics Loss', linewidth=2)
    plt.semilogy(bc_loss_history, 'r-.', label='Boundary Loss', linewidth=2)
    
    plt.axvline(x=stage1_end, color='k', linestyle='--', alpha=0.7, label='Stage 1→2')
    if stage2_end < len(loss_history):
        plt.axvline(x=stage2_end, color='k', linestyle=':', alpha=0.7, label='Stage 2→3')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('IMPROVED 3-Stage Training: Return to Success + Better BCs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Linear scale plot
    plt.subplot(2, 1, 2)
    plt.plot(loss_history, 'b-', label='Total Loss', linewidth=2)
    plt.plot(physics_loss_history, 'g--', label='Physics Loss', linewidth=2)
    plt.plot(bc_loss_history, 'r-.', label='Boundary Loss', linewidth=2)
    
    plt.axvline(x=stage1_end, color='k', linestyle='--', alpha=0.7, label='Stage 1→2')
    if stage2_end < len(loss_history):
        plt.axvline(x=stage2_end, color='k', linestyle=':', alpha=0.7, label='Stage 2→3')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss (linear scale)')
    plt.title('Loss History - Improved 3-Stage Approach')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/improved_3stage_training.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nIMPROVED 3-Stage training complete!")
    print(f"Improvements over previous attempts:")
    print(f"  ✅ Better boundary condition enforcement (multiple loss terms)")
    print(f"  ✅ More conservative physics introduction (max 20% weight)")
    print(f"  ✅ Better early stopping and safety checks")
    print(f"  ✅ Experimental dimensions restored (H_right = 0.019m)")
    print(f"  ✅ Same successful approach that got 50% direction accuracy")
    print(f"Expected results: Direction accuracy >60%, Overall accuracy >30%")
    print(f"Loss plot saved to: plots/improved_3stage_training.png")
    
    return model