# training/trainer.py
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from physics.boundary_conditions import BoundaryLoss, PhysicsLoss
from domain import generate_domain_points, generate_boundary_points

class PINNTrainer:
    """Streamlined PINN trainer focused on boundary conditions"""
    
    def __init__(self, model, boundary_config, max_epochs=2500, device='cpu', output_dir=None):
        self.model = model
        self.boundary_config = boundary_config
        self.max_epochs = max_epochs
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        
        # Initialize physics and boundary losses
        self.boundary_loss = BoundaryLoss(boundary_config, device)
        self.physics_loss = PhysicsLoss(boundary_config, device)
        
        # Generate domain and boundary points
        self.domain_points = generate_domain_points()
        wall_points, wall_normals, inlet_points, outlet_points, wall_segments = generate_boundary_points()
        
        self.wall_points = wall_points
        self.inlet_points = inlet_points
        self.outlet_points = outlet_points
        
        # Training history
        self.loss_history = []
        self.bc_loss_history = []
        self.physics_loss_history = []
        
    def train(self):
        """Train the PINN model with simplified 2-stage approach"""
        print(f"Training PINN for {self.max_epochs} epochs...")
        
        # Optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=5e-4, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, eta_min=1e-6)
        
        # Stage 1: Pure boundary conditions (70% of epochs)
        stage1_epochs = int(0.7 * self.max_epochs)
        print(f"Stage 1: Boundary conditions only ({stage1_epochs} epochs)")
        
        for epoch in range(stage1_epochs):
            self.model.train()
            
            bc_loss = self.boundary_loss.compute_total_boundary_loss(
                self.model, self.inlet_points, self.outlet_points, self.wall_points
            )
            
            total_loss = bc_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            
            self.loss_history.append(total_loss.item())
            self.bc_loss_history.append(bc_loss.item())
            self.physics_loss_history.append(0.0)
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}: BC Loss = {bc_loss.item():.6f}")
        
        # Stage 2: Add physics gradually (30% of epochs)
        stage2_epochs = self.max_epochs - stage1_epochs
        print(f"Stage 2: Adding physics ({stage2_epochs} epochs)")
        
        # Reduce learning rate for stage 2
        optimizer.param_groups[0]['lr'] = 2e-4
        
        for epoch in range(stage2_epochs):
            self.model.train()
            
            bc_loss = self.boundary_loss.compute_total_boundary_loss(
                self.model, self.inlet_points, self.outlet_points, self.wall_points
            )
            
            physics_loss = self.physics_loss.compute_physics_loss(self.model, self.domain_points)
            
            # Gradually increase physics weight
            physics_weight = min(0.1, 0.01 * (epoch / stage2_epochs))
            total_loss = bc_loss + physics_weight * physics_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
            optimizer.step()
            scheduler.step()
            
            self.loss_history.append(total_loss.item())
            self.bc_loss_history.append(bc_loss.item())
            self.physics_loss_history.append(physics_loss.item())
            
            if epoch % 200 == 0:
                print(f"Epoch {stage1_epochs + epoch}: Total={total_loss.item():.6f}, "
                      f"BC={bc_loss.item():.6f}, Physics={physics_loss.item():.6f}")
        
        # Save training plot
        self._save_training_plot()
        
        print("Training completed!")
        return self.model
    
    def _save_training_plot(self):
        """Save training loss plot"""
        plt.figure(figsize=(12, 8))
        
        # Plot losses
        plt.subplot(2, 1, 1)
        plt.semilogy(self.loss_history, 'b-', label='Total Loss', linewidth=2)
        plt.semilogy(self.bc_loss_history, 'r--', label='Boundary Loss', linewidth=2)
        plt.semilogy(self.physics_loss_history, 'g:', label='Physics Loss', linewidth=2)
        
        stage1_end = int(0.7 * self.max_epochs)
        plt.axvline(x=stage1_end, color='k', linestyle='--', alpha=0.7, label='Stage 1→2')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title(f'Training History - {self.max_epochs} epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Linear scale
        plt.subplot(2, 1, 2)
        plt.plot(self.loss_history, 'b-', label='Total Loss', linewidth=2)
        plt.plot(self.bc_loss_history, 'r--', label='Boundary Loss', linewidth=2)
        plt.plot(self.physics_loss_history, 'g:', label='Physics Loss', linewidth=2)
        plt.axvline(x=stage1_end, color='k', linestyle='--', alpha=0.7, label='Stage 1→2')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss (linear scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()