import torch
import matplotlib.pyplot as plt
from boundary_conditions import compute_momentum_bc

# Check CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Fixed loss function with proper gradient tracking
def compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals):
    print("Computing loss...")
    
    # Initialize losses
    physics_loss = torch.tensor(0.0, device=device, requires_grad=True)
    bc_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    # Simple inlet condition (v = -0.5, u = 0)
    if len(inlet_points) > 0:
        xy_inlet = torch.tensor(inlet_points, dtype=torch.float32, device=device)
        outputs = model(xy_inlet)
        u_inlet = outputs[:, 0]
        v_inlet = outputs[:, 1]
        inlet_loss = torch.mean(u_inlet**2) + torch.mean((v_inlet + 0.5)**2)
        bc_loss = bc_loss + inlet_loss  # Use addition instead of += for gradient tracking
    
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
        bc_loss = bc_loss + 10.0 * wall_loss  # Stronger weight for wall conditions
    
    # Total loss (simplified for debugging)
    total_loss = physics_loss + bc_loss
    
    return total_loss

# Training loop with proper gradient tracking
def train_model(model, optimizer, domain_points, inlet_points, outlet_points, wall_points, wall_normals, epochs=2000):
    print(f"Starting training for {epochs} epochs...")
    
    loss_history = []
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Set model to training mode
        model.train()
        
        # Compute loss
        loss = compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss
        loss_history.append(loss.item())
        print(f"  Loss: {loss.item():.6f}")
    
    # Plot loss history
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