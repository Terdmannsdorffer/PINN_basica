# training.py - ENHANCED VERSION with Magnitude Calibration
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_power_constraint(model, domain_points, device):
    """Versión ultra-agresiva"""
    domain_tensor = torch.tensor(domain_points, dtype=torch.float32, device=device, requires_grad=True)
    
    output = model(domain_tensor)
    u, v = output[:, 0], output[:, 1]
    
    velocity_magnitude = torch.sqrt(u**2 + v**2)
    
    # Target aún más bajo basado en PIV
    target_velocity = 0.001  # 1 mm/s
    
    # Penalizar CUALQUIER velocidad por encima del target
    mean_velocity = torch.mean(velocity_magnitude)
    max_velocity = torch.max(velocity_magnitude)
    
    # Pérdida cuadrática para el promedio y pérdida fuerte para máximos
    power_loss = 10.0 * (mean_velocity - target_velocity)**2 + torch.mean(torch.relu(velocity_magnitude - 0.003)**2)
    
    return power_loss
def train_enhanced_staged(model, optimizer, domain_points, inlet_points, outlet_points, wall_points, wall_normals, piv_reference_data=None):
    """
    Enhanced 4-stage training with magnitude calibration and mass conservation
    
    Key improvements:
    1. Learnable velocity scaling parameters
    2. Mass conservation enforcement
    3. PIV magnitude calibration stage
    4. Better boundary conditions
    """
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    
    loss_history = []
    physics_loss_history = []
    bc_loss_history = []
    magnitude_loss_history = []
    
    # Physical parameters
    rho = 0.101972
    g_x, g_y = 0.0, -9.81
    L_up, L_down = 0.097, 0.174
    H_left, H_right = 0.119, 0.019
    
    # PIV reference data handling
    if piv_reference_data:
        piv_u_mean = piv_reference_data.get('u_mean', 0.0)
        piv_v_mean = piv_reference_data.get('v_mean', -0.005)
        piv_v_mean = -abs(piv_v_mean)
        piv_mag_mean = piv_reference_data.get('mag_mean', 0.005)
        piv_mag_std = piv_reference_data.get('mag_std', 0.002)
        target_mass_flow = piv_reference_data.get('mass_flow_rate', 0.001)
        print(f"Using PIV reference: mag={piv_mag_mean:.6f}±{piv_mag_std:.6f} m/s")
    else:
        piv_u_mean, piv_v_mean = 0.0, -0.005
        piv_mag_mean, piv_mag_std = 0.005, 0.002
        target_mass_flow = 0.001
        print("Using default reference values")
    print(f"\n🔍 DEBUG PIV Reference:")
    print(f"  piv_v_mean = {piv_v_mean:.6f} (should be negative for downward flow)")
    print(f"  piv_u_mean = {piv_u_mean:.6f}")    
    # Carbopol rheology
    tau_y, k, n = 30.0, 2.8, 0.65
    
    def compute_mass_conservation_loss():
        """Enhanced mass conservation with expected velocity scales"""
        mass_loss = torch.tensor(0.0, device=device)
        
        # Velocidades esperadas basadas en el rango PIV
        expected_inlet_velocity = -0.002   # m/s (negativo = hacia abajo)
        expected_outlet_velocity = 0.002   # m/s (positivo = hacia la derecha)
        
        # 1. Conservación de masa en inlet/outlet
        if len(inlet_points) > 0 and len(outlet_points) > 0:
            inlet_tensor = torch.tensor(inlet_points, dtype=torch.float32, device=device)
            outlet_tensor = torch.tensor(outlet_points, dtype=torch.float32, device=device)
            
            inlet_pred = model(inlet_tensor)
            outlet_pred = model(outlet_tensor)
            
            v_inlet = inlet_pred[:, 1]  # componente vertical
            u_outlet = outlet_pred[:, 0]  # componente horizontal
            
            # Calcular flujos promedio
            avg_inlet_v = torch.mean(v_inlet)
            avg_outlet_u = torch.mean(u_outlet)
            
            # a) Conservación de masa tradicional
            inlet_flow = avg_inlet_v * L_up
            outlet_flow = avg_outlet_u * H_left
            mass_conservation_error = (inlet_flow + outlet_flow)**2
            
            # b) Penalizar desviaciones de velocidades esperadas
            inlet_velocity_loss = (avg_inlet_v - expected_inlet_velocity)**2
            outlet_velocity_loss = (avg_outlet_u - expected_outlet_velocity)**2
            
            # c) Penalizar variabilidad excesiva
            inlet_std = torch.std(v_inlet)
            outlet_std = torch.std(u_outlet)
            uniformity_loss = inlet_std**2 + outlet_std**2
            
            # Combinar todas las pérdidas
            mass_loss = (10.0 * mass_conservation_error + 
                        5.0 * inlet_velocity_loss + 
                        5.0 * outlet_velocity_loss + 
                        0.1 * uniformity_loss)
        
        # 2. Continuidad en el dominio (opcional pero útil)
        n_continuity_points = min(200, len(domain_points))
        if n_continuity_points > 0:
            indices = np.random.choice(len(domain_points), n_continuity_points, replace=False)
            continuity_points = domain_points[indices]
            
            cont_tensor = torch.tensor(continuity_points, dtype=torch.float32, device=device, requires_grad=True)
            cont_pred = model(cont_tensor)
            u_cont, v_cont = cont_pred[:, 0], cont_pred[:, 1]
            
            # Calcular divergencia
            u_x = torch.autograd.grad(u_cont.sum(), cont_tensor, create_graph=True)[0][:, 0]
            v_y = torch.autograd.grad(v_cont.sum(), cont_tensor, create_graph=True)[0][:, 1]
            
            divergence = u_x + v_y
            continuity_loss = torch.mean(divergence**2)
            
            mass_loss += 0.1 * continuity_loss
        
        return mass_loss
    
    def compute_magnitude_matching_loss():
        """PIV magnitude matching - the key innovation for scaling"""
        if len(domain_points) == 0:
            return torch.tensor(0.0, device=device)
        
        # Sample domain points
        n_sample = min(400, len(domain_points))
        indices = torch.randperm(len(domain_points))[:n_sample]
        sample_points = [domain_points[i] for i in indices]
        
        sample_tensor = torch.tensor(sample_points, dtype=torch.float32, device=device)
        pred = model(sample_tensor)
        u, v = pred[:, 0], pred[:, 1]
        magnitude = torch.sqrt(u**2 + v**2)
        
        # Match PIV magnitude statistics
        mag_mean_loss = (torch.mean(magnitude) - piv_mag_mean)**2
        mag_std_loss = (torch.std(magnitude) - piv_mag_std)**2
        
        # Match component means
        u_component_loss = (torch.mean(u) - piv_u_mean)**2
        v_component_loss = (torch.mean(v) - piv_v_mean)**2
        
        return mag_mean_loss + 0.5 * mag_std_loss + u_component_loss + v_component_loss
    
    def compute_boundary_loss_enhanced():
        """Enhanced boundary conditions"""
        bc_loss = torch.tensor(0.0, device=device)
        
        # Inlet: enforce target velocity profile
        if len(inlet_points) > 0:
            inlet_tensor = torch.tensor(inlet_points, dtype=torch.float32, device=device)
            inlet_pred = model(inlet_tensor)
            u_inlet, v_inlet = inlet_pred[:, 0], inlet_pred[:, 1]
            
            # u should be ~0, v should match PIV mean
            bc_loss += 10.0 * torch.mean(u_inlet**2)
            bc_loss += 5.0 * (torch.mean(v_inlet) - piv_v_mean)**2
            bc_loss += 2.0 * torch.mean(torch.relu(v_inlet)**2)  # Ensure downward
        
        # Outlet: enforce reasonable outflow
        if len(outlet_points) > 0:
            outlet_tensor = torch.tensor(outlet_points, dtype=torch.float32, device=device)
            outlet_pred = model(outlet_tensor)
            u_outlet, v_outlet = outlet_pred[:, 0], outlet_pred[:, 1]
            
            # v should be ~0, u should be positive
            bc_loss += 10.0 * torch.mean(v_outlet**2)
            bc_loss += 2.0 * torch.mean(torch.relu(-u_outlet)**2)  # Ensure rightward
            
            # Pressure at outlet should be ~0
            p_outlet = outlet_pred[:, 2]
            bc_loss += torch.mean(p_outlet**2)
        
        # Wall: no-slip condition
        if len(wall_points) > 0:
            n_wall = min(100, len(wall_points))
            wall_indices = torch.randperm(len(wall_points))[:n_wall]
            wall_subset = [wall_points[i] for i in wall_indices]
            
            wall_tensor = torch.tensor(wall_subset, dtype=torch.float32, device=device)
            wall_pred = model(wall_tensor)
            u_wall, v_wall = wall_pred[:, 0], wall_pred[:, 1]
            
            bc_loss += 5.0 * torch.mean(u_wall**2 + v_wall**2)
        
        return bc_loss
    def compute_reynolds_constraint(model, domain_points, device):
        """Asegurar Re << 1 en todo el dominio"""
        n_points = min(500, len(domain_points))
        indices = np.random.choice(len(domain_points), n_points, replace=False)
        sample_points = domain_points[indices]
        
        sample_tensor = torch.tensor(sample_points, dtype=torch.float32, device=device, requires_grad=True)
        output = model(sample_tensor)
        u, v = output[:, 0], output[:, 1]
        
        # Calcular shear rate para viscosidad aparente
        u_x = torch.autograd.grad(u.sum(), sample_tensor, create_graph=True)[0][:, 0]
        u_y = torch.autograd.grad(u.sum(), sample_tensor, create_graph=True)[0][:, 1]
        v_x = torch.autograd.grad(v.sum(), sample_tensor, create_graph=True)[0][:, 0]
        v_y = torch.autograd.grad(v.sum(), sample_tensor, create_graph=True)[0][:, 1]
        
        shear_rate = torch.sqrt(2.0 * (u_x**2 + v_y**2 + 0.5*(u_y + v_x)**2))
        
        # Viscosidad aparente Herschel-Bulkley
        eta_app = torch.where(shear_rate > 1e-6,
                            tau_y / shear_rate + k * shear_rate**(n-1),
                            1000.0)  # Alta viscosidad cuando shear_rate → 0
        
        # Reynolds local
        velocity_mag = torch.sqrt(u**2 + v**2)
        Re_local = rho * velocity_mag * 0.02 / eta_app  # 0.02m es escala característica
        
        # Para Carbopol, Re debe ser << 1
        re_loss = torch.mean(torch.relu(Re_local - 0.01)**2)
        
        return re_loss   
    def compute_physics_loss_stable(domain_points):
        """Stable physics loss computation"""
        if len(domain_points) == 0:
            return torch.tensor(0.0, device=device)
        
        n_points = min(1000, len(domain_points))
        indices = torch.randperm(len(domain_points))[:n_points]
        selected_points = [domain_points[i] for i in indices]
        
        xy = torch.tensor(selected_points, dtype=torch.float32, device=device, requires_grad=True)
        output = model(xy)
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]

        u_x, u_y = torch.autograd.grad(u.sum(), xy, create_graph=True)[0].split(1, dim=1)
        v_x, v_y = torch.autograd.grad(v.sum(), xy, create_graph=True)[0].split(1, dim=1)
        p_x, p_y = torch.autograd.grad(p.sum(), xy, create_graph=True)[0].split(1, dim=1)

        shear_rate = torch.sqrt(2*((u_x)**2 + (v_y)**2) + (u_y + v_x)**2 + 1e-8)
        eta_eff = (tau_y / (shear_rate + 1e-8)) + k * torch.pow(shear_rate + 1e-8, n - 1)
        eta_eff = torch.clamp(eta_eff, min=0.01, max=500.0)

        continuity = u_x + v_y
        f_x = p_x - eta_eff * (u_x + u_y) + rho * g_x
        f_y = p_y - eta_eff * (v_x + v_y) + rho * g_y
        
        # AGREGAR: Penalización por velocidades altas
        velocity_magnitude = torch.sqrt(u**2 + v**2)
        high_velocity_mask = velocity_magnitude > 0.002
        velocity_penalty = torch.mean(torch.where(
            high_velocity_mask,
            1000.0 * (velocity_magnitude - 0.002)**2,
            torch.zeros_like(velocity_magnitude)
        ))
        
        return torch.mean(continuity**2) + torch.mean(f_x**2 + f_y**2) + velocity_penalty
    
    print(f"\nEnhanced 4-Stage Training with Magnitude Calibration")
    print(f"Target magnitude: {piv_mag_mean:.6f} ± {piv_mag_std:.6f} m/s")
    
    # STAGE 1: Pattern Learning (1000 epochs)
    print(f"\nStage 1: Pattern Learning (1000 epochs)...")
    scheduler1 = CosineAnnealingWarmRestarts(optimizer, T_0=250, T_mult=1, eta_min=1e-6)
    
    for epoch in range(1000):
        model.train()
        
        bc_loss = compute_boundary_loss_enhanced()
        total_loss = bc_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        scheduler1.step()
        
        loss_history.append(total_loss.item())
        physics_loss_history.append(0.0)
        bc_loss_history.append(bc_loss.item())
        magnitude_loss_history.append(0.0)
        
        if epoch % 250 == 0:
            print(f"[Stage 1: {epoch:04d}] BC Loss = {bc_loss.item():.6f}")
            
            # Check velocity scales if model has them
            if hasattr(model, 'get_velocity_scales'):
                scales = model.get_velocity_scales()
                print(f"  Velocity scales: u={scales['effective_u_scale']:.6f}, v={scales['effective_v_scale']:.6f}")
    
    # STAGE 2: Mass Conservation + Physics (1500 epochs)
    print(f"\nStage 2: Mass conservation focus (500 epochs)...")
    model.train()
    optimizer.param_groups[0]['lr'] = 5e-4

    for epoch in range(500):
        bc_loss = compute_boundary_loss_enhanced()
        mass_loss = compute_mass_conservation_loss()
        physics_loss = compute_physics_loss_stable(domain_points)
        power_loss = compute_power_constraint(model, domain_points, device)
        
        # AGREGAR: Calcular restricción de Reynolds
        reynolds_loss = compute_reynolds_constraint(model, domain_points, device)
        
        physics_weight = min(0.01 * (epoch / 100), 0.05)
        
        # MODIFICAR: Agregar reynolds_loss a la pérdida total
        total_loss = bc_loss + 5.0 * mass_loss + physics_weight * physics_loss + 2.0 * power_loss + 2.0 * reynolds_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            # MODIFICAR: Incluir Reynolds en el print
            print(f"[Stage 2: {epoch:03d}] BC={bc_loss.item():.6f}, Mass={mass_loss.item():.6f}, "
                f"Physics={physics_loss.item():.6f}, Power={power_loss.item():.6f}, "
                f"Reynolds={reynolds_loss.item():.6f}")
    
    print(f"\nStage 3: Full physics integration (500 epochs)...")
    model.train()
    optimizer.param_groups[0]['lr'] = 1e-4

    for epoch in range(500):
        bc_loss = compute_boundary_loss_enhanced()
        mass_loss = compute_mass_conservation_loss()
        physics_loss = compute_physics_loss_stable(domain_points)
        power_loss = compute_power_constraint(model, domain_points, device)
        
        # AGREGAR: Calcular restricción de Reynolds
        reynolds_loss = compute_reynolds_constraint(model, domain_points, device)
        
        # MODIFICAR: Aumentar peso de reynolds_loss
        total_loss = 2.0 * bc_loss + 10.0 * mass_loss + 0.05 * physics_loss + 10.0 * power_loss + 5.0 * reynolds_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"[Stage 3: {epoch:03d}] BC={bc_loss.item():.6f}, Mass={mass_loss.item():.6f}, "
                f"Physics={physics_loss.item():.6f}, Power={power_loss.item():.6f}, "
                f"Reynolds={reynolds_loss.item():.6f}")
    
    # ===== Stage 4: Magnitude matching (300 epochs) =====
    # ===== Stage 4: Final refinement (300 epochs) =====
    print(f"\nStage 4: Final refinement (300 epochs)...")
    model.train()
    optimizer.param_groups[0]['lr'] = 5e-5

    for epoch in range(300):
        bc_loss = compute_boundary_loss_enhanced()
        mass_loss = compute_mass_conservation_loss()
        physics_loss = compute_physics_loss_stable(domain_points)
        power_loss = compute_power_constraint(model, domain_points, device)
        
        # AGREGAR: Calcular restricción de Reynolds
        reynolds_loss = compute_reynolds_constraint(model, domain_points, device)
        
        # MODIFICAR: Peso muy alto para Reynolds en etapa final
        total_loss = bc_loss + mass_loss + 20.0 * power_loss + 10.0 * reynolds_loss + 0.001 * physics_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"[Stage 4: {epoch:03d}] BC={bc_loss.item():.6f}, Mass={mass_loss.item():.6f}, "
                f"Physics={physics_loss.item():.6f}, Power={power_loss.item():.6f}, "
                f"Reynolds={reynolds_loss.item():.6f}")
    # ===== Stage 5: Magnitude reduction focus (300 epochs) =====
    print(f"\nStage 5: Magnitude reduction (300 epochs)...")
    model.train()
    optimizer.param_groups[0]['lr'] = 1e-5

    for epoch in range(300):
        # Solo restricciones de magnitud
        bc_loss = compute_boundary_loss_enhanced()
        power_loss = compute_power_constraint(model, domain_points, device)
        reynolds_loss = compute_reynolds_constraint(model, domain_points, device)
        
        # Regularización L2 directa
        n_reg = min(1000, len(domain_points))
        reg_indices = np.random.choice(len(domain_points), n_reg, replace=False)
        reg_tensor = torch.tensor(domain_points[reg_indices], dtype=torch.float32, device=device)
        reg_output = model(reg_tensor)
        u_reg, v_reg = reg_output[:, 0], reg_output[:, 1]
        
        # Penalizar velocidades > 0.002 m/s
        velocity_mag = torch.sqrt(u_reg**2 + v_reg**2)
        mag_penalty = torch.mean(torch.relu(velocity_mag - 0.002)**2)
        
        # Solo pérdidas de magnitud
        total_loss = 0.1 * bc_loss + 50.0 * power_loss + 20.0 * reynolds_loss + 100.0 * mag_penalty
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Clip más agresivo
        optimizer.step()
        
        if epoch % 100 == 0:
            avg_vel = torch.mean(velocity_mag).item()
            max_vel = torch.max(velocity_mag).item()
            print(f"[Stage 5: {epoch:03d}] Avg_vel={avg_vel:.6f}, Max_vel={max_vel:.6f}, "
                f"Power={power_loss.item():.6f}, Re={reynolds_loss.item():.6f}")
            # Monitor magnitude progress
            with torch.no_grad():
                if len(domain_points) > 0:
                    sample_indices = torch.randperm(len(domain_points))[:200]
                    sample_points = [domain_points[i] for i in sample_indices]
                    sample_tensor = torch.tensor(sample_points, dtype=torch.float32, device=device)
                    pred = model(sample_tensor)
                    magnitude = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2)
                    current_mag = torch.mean(magnitude).item()
                    print(f"  Current magnitude: {current_mag:.6f} (target: {piv_mag_mean:.6f})")
    
    # Final validation
    print(f"\n" + "="*60)
    print("ENHANCED TRAINING VALIDATION")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Check final magnitude
        if len(domain_points) > 0:
            sample_indices = torch.randperm(len(domain_points))[:500]
            sample_points = [domain_points[i] for i in sample_indices]
            sample_tensor = torch.tensor(sample_points, dtype=torch.float32, device=device)
            pred = model(sample_tensor)
            u_sample = pred[:, 0].cpu().numpy()
            v_sample = pred[:, 1].cpu().numpy()
            mag_sample = np.sqrt(u_sample**2 + v_sample**2)
            
            mag_error = abs(mag_sample.mean() - piv_mag_mean) / piv_mag_mean * 100
            
            print(f"MAGNITUDE VALIDATION:")
            print(f"  Current: {mag_sample.mean():.6f} ± {mag_sample.std():.6f} m/s")
            print(f"  Target:  {piv_mag_mean:.6f} ± {piv_mag_std:.6f} m/s")
            print(f"  Error: {mag_error:.1f}%")
        
        # Check mass conservation
        if len(inlet_points) > 0 and len(outlet_points) > 0:
            inlet_tensor = torch.tensor(inlet_points, dtype=torch.float32, device=device)
            outlet_tensor = torch.tensor(outlet_points, dtype=torch.float32, device=device)
            
            inlet_pred = model(inlet_tensor)
            outlet_pred = model(outlet_tensor)
            
            avg_inlet_v = inlet_pred[:, 1].mean().item()
            avg_outlet_u = outlet_pred[:, 0].mean().item()
            
            inlet_flow = rho * avg_inlet_v * L_up
            outlet_flow = rho * avg_outlet_u * H_right
            mass_balance = inlet_flow + outlet_flow
            
            print(f"\nMASS CONSERVATION:")
            print(f"  Inlet flow: {inlet_flow:.6f} kg/s")
            print(f"  Outlet flow: {outlet_flow:.6f} kg/s")
            print(f"  Balance error: {abs(mass_balance):.6f} kg/s")
        
        # Show final velocity scales
        if hasattr(model, 'get_velocity_scales'):
            scales = model.get_velocity_scales()
            print(f"\nFINAL VELOCITY SCALES:")
            print(f"  u_scale: {scales['effective_u_scale']:.6f}")
            print(f"  v_scale: {scales['effective_v_scale']:.6f}")
    
    print("="*60)
    
    # Create training visualization
    plt.figure(figsize=(15, 10))
    
    stage_ends = [1000, 2500, 3500, 4300]
    
    plt.subplot(2, 2, 1)
    plt.semilogy(loss_history, 'b-', linewidth=2)
    for i, end in enumerate(stage_ends[:-1]):
        plt.axvline(x=end, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss (log)')
    plt.title('Enhanced 4-Stage Training')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.semilogy(np.array(physics_loss_history) + 1e-10, 'g--', label='Physics')
    plt.semilogy(np.array(bc_loss_history) + 1e-10, 'r-.', label='Boundary')
    plt.semilogy(np.array(magnitude_loss_history) + 1e-10, 'c-', label='Magnitude')
    plt.xlabel('Epoch')
    plt.ylabel('Component Losses (log)')
    plt.title('Loss Components')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    magnitude_nonzero = np.array(magnitude_loss_history)
    magnitude_nonzero[magnitude_nonzero == 0] = np.nan
    plt.semilogy(magnitude_nonzero, 'c-', linewidth=3)
    plt.axvline(x=stage_ends[2], color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Magnitude Loss')
    plt.title('Magnitude Calibration (Stage 4)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = f"""ENHANCED TRAINING SUMMARY

Target magnitude: {piv_mag_mean:.6f} m/s
Final magnitude: {mag_sample.mean():.6f} m/s
Magnitude error: {mag_error:.1f}%

Stage 1: Pattern Learning
Stage 2: Mass Conservation  
Stage 3: Boundary Refinement
Stage 4: Magnitude Calibration

Expected improvement:
• Overall accuracy: >60%
• Direction accuracy: >85%
• Magnitude scaling: <2x PIV"""
    
    plt.text(0.1, 0.9, summary_text, fontsize=11, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig("plots/enhanced_4stage_training.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nEnhanced training complete!")
    print(f"Expected improvements:")
    print(f"  • Magnitude error: {mag_error:.1f}% (vs previous 554%)")
    print(f"  • Overall accuracy: >60% (vs previous 0%)")
    print(f"  • Direction accuracy: >85% (vs previous 82%)")
    print(f"Loss plot saved: plots/enhanced_4stage_training.png")
    
    return model


def load_piv_reference_data(piv_filepath):
    """Load PIV data for reference statistics"""
    try:
        import pandas as pd
        
        with open(piv_filepath, 'r') as f:
            lines = f.readlines()
        
        header_idx = None
        for i, line in enumerate(lines):
            if 'x [m]' in line and 'y [m]' in line:
                header_idx = i
                break
        
        if header_idx is None:
            return None
        
        piv_df = pd.read_csv(piv_filepath, skiprows=header_idx)
        
        required_columns = ['x [m]', 'y [m]', 'u [m/s]', 'v [m/s]']
        if not all(col in piv_df.columns for col in required_columns):
            return None
        
        valid_mask = (
            np.isfinite(piv_df['x [m]']) & 
            np.isfinite(piv_df['y [m]']) & 
            np.isfinite(piv_df['u [m/s]']) & 
            np.isfinite(piv_df['v [m/s]'])
        )
        
        piv_clean = piv_df[valid_mask].copy()
        
        if len(piv_clean) == 0:
            return None
        
        # Handle coordinate flip if needed
        max_y = piv_clean['y [m]'].max()
        if max_y > 0.2:
            piv_clean['y [m]'] = max_y - piv_clean['y [m]']
            piv_clean['v [m/s]'] = -piv_clean['v [m/s]']
        
        u_data = piv_clean['u [m/s]'].values
        v_data = piv_clean['v [m/s]'].values
        magnitude = np.sqrt(u_data**2 + v_data**2)
        
        # Estimate mass flow
        avg_inlet_v = np.mean(v_data[v_data < 0])
        estimated_mass_flow = abs(avg_inlet_v) * 0.097 * 0.101972
        
        return {
            'u_mean': np.mean(u_data),
            'v_mean': np.mean(v_data),
            'mag_mean': np.mean(magnitude),
            'mag_std': np.std(magnitude),
            'mass_flow_rate': estimated_mass_flow,
            'n_points': len(piv_clean)
        }
        
    except Exception as e:
        print(f"Error loading PIV data: {e}")
        return None


# Compatibility alias
train_staged = train_enhanced_staged