# Fixed training.py for proper gradient tracking
import torch
import matplotlib.pyplot as plt
from boundary_conditions import compute_momentum_bc

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals):
    print("Computing loss for Carbopol simulation...")
    
    physics_loss = torch.tensor(0.0, device=device, requires_grad=True)
    bc_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Domain physics for Carbopol (if domain points available)
    if len(domain_points) > 0:
        # FIXED: Ensure requires_grad is set properly
        xy_domain = torch.tensor(domain_points, dtype=torch.float32, device=device, requires_grad=True)
        outputs = model(xy_domain)
        
        u = outputs[:, 0:1]
        v = outputs[:, 1:2]
        p = outputs[:, 2:3]
        
        # FIXED: Use correct gradient calculation approach
        u_x = torch.autograd.grad(u.sum(), xy_domain, create_graph=True)[0][:, 0:1]
        u_y = torch.autograd.grad(u.sum(), xy_domain, create_graph=True)[0][:, 1:2]
        v_x = torch.autograd.grad(v.sum(), xy_domain, create_graph=True)[0][:, 0:1] 
        v_y = torch.autograd.grad(v.sum(), xy_domain, create_graph=True)[0][:, 1:2]
        
        # FIXED: Separate calculation for pressure gradients
        p_x = torch.autograd.grad(p.sum(), xy_domain, create_graph=True)[0][:, 0:1]
        p_y = torch.autograd.grad(p.sum(), xy_domain, create_graph=True)[0][:, 1:2]
        
        # Shear rate calculation
        shear_rate = torch.sqrt(2*((u_x)**2 + (v_y)**2) + (u_y + v_x)**2 + 1e-8)
        
        # Carbopol parameters - based on the paper
        tau_y = 5.0  # Yield stress in Pa
        k = 2.5      # Consistency index
        n = 0.42     # Power law index
        
        # Herschel-Bulkley model
        eta_eff = (tau_y / (shear_rate + 1e-6)) + k * (shear_rate ** (n - 1))
        
        # Continuity equation 
        continuity = u_x + v_y
        
        # FIXED: Simplified momentum equations to avoid gradient issues
        # Instead of summing eta_eff * gradients, use element-wise operations
        f_x = p_x - (eta_eff * u_x + eta_eff * u_y)
        f_y = p_y - (eta_eff * v_x + eta_eff * v_y)
        
        physics_loss = physics_loss + 0.1 * torch.mean(continuity**2) + torch.mean(f_x**2 + f_y**2)
    
    # Inlet condition
    if len(inlet_points) > 0:
        xy_inlet = torch.tensor(inlet_points, dtype=torch.float32, device=device)
        outputs = model(xy_inlet)
        u_inlet = outputs[:, 0]
        v_inlet = outputs[:, 1]
        
        # Force stronger negative v component (downward flow) and zero u component
        inlet_loss = torch.mean(u_inlet**2) + torch.mean((v_inlet + 2)**2)
        bc_loss = bc_loss + 5.0 * inlet_loss
    
    # Outlet condition
    if len(outlet_points) > 0:
        xy_outlet = torch.tensor(outlet_points, dtype=torch.float32, device=device)
        outputs = model(xy_outlet)
        u_outlet = outputs[:, 0]
        v_outlet = outputs[:, 1]
        p_outlet = outputs[:, 2]
        
        # Force positive x-velocity (flow to the right) at outlet
        outlet_flow_loss = torch.mean((u_outlet - 0.1)**2) + torch.mean(v_outlet**2)
        outlet_pressure_loss = torch.mean(p_outlet**2)
        
        bc_loss = bc_loss + outlet_pressure_loss + 5.0 * outlet_flow_loss
    
    # Wall boundary conditions
    if len(wall_points) > 0:
        wall_loss = compute_momentum_bc(model, wall_points, wall_normals)
        bc_loss = bc_loss + 10.0 * wall_loss
    
    # Total loss
    total_loss = 0.5 * physics_loss + bc_loss
    
    return total_loss

def train_model(model, optimizer, domain_points, inlet_points, outlet_points, wall_points, wall_normals, epochs=3000):
    print(f"Starting Carbopol flow training for {epochs} epochs...")
    
    loss_history = []
    
    # FIXED: Keep original optimizer to avoid compatibility issues
    # Just add scheduler without changing the optimizer
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5, verbose=True)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        model.train()
        
        # Compute loss
        loss = compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        scheduler.step(loss)
        
        loss_history.append(loss.item())
        print(f"  Loss: {loss.item():.6f}")
        
        # FIXED: Validation check with proper error handling
        if epoch % 200 == 0 or epoch == epochs-1:
            model.eval()
            try:
                # Only check flow direction without computing full validation loss
                with torch.no_grad():
                    # Check for flow direction at key points
                    test_inlet = torch.tensor(inlet_points[:1], dtype=torch.float32, device=device)
                    test_outlet = torch.tensor(outlet_points[:1], dtype=torch.float32, device=device)
                    
                    inlet_vel = model(test_inlet)[:, :2].cpu().numpy()
                    outlet_vel = model(test_outlet)[:, :2].cpu().numpy()
                    
                    print(f"  Inlet velocity: u={inlet_vel[0,0]:.4f}, v={inlet_vel[0,1]:.4f}")
                    print(f"  Outlet velocity: u={outlet_vel[0,0]:.4f}, v={outlet_vel[0,1]:.4f}")
            except Exception as e:
                print(f"  Validation error: {e}")
        
        if epoch > 500 and loss.item() < 1e-4:
            print("Convergence reached. Stopping early.")
            break
    
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Carbopol Simulation Training Loss')
    plt.grid(True)
    plt.savefig('plots/carbopol_loss_history.png')
    plt.close()
    
    print("Carbopol simulation training completed")
    
    return model