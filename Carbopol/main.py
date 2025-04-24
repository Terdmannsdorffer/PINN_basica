import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

from model import SimplePINN
from domain import generate_domain_points, generate_boundary_points, inside_L
from training import train_model
from visualization import visualize_results, analyze_reflection

print("===== Starting PINN script with fixed gradients =====")
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)

# Check CUDA
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: {device}")
else:
    device = torch.device('cpu')
    print(f"Using device: {device}")

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
print("Created directories")

# Main function
def main():
    # Initialize model
    print("Creating model...")
    model = SimplePINN().to(device)
    print("Model created")
    
    # Generate domain points
    print("Generating points...")
    domain_points = generate_domain_points()
    wall_points, wall_normals, inlet_points, outlet_points, wall_segments = generate_boundary_points()
    
    # Plot the domain and points to verify
    plt.figure(figsize=(8, 6))
    plt.scatter(domain_points[:, 0], domain_points[:, 1], s=1, alpha=0.5, label='Domain')
    plt.scatter(wall_points[:, 0], wall_points[:, 1], s=5, color='black', label='Wall')
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], s=5, color='blue', label='Inlet')
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], s=5, color='red', label='Outlet')

    # Plot wall normals
    scale = 0.1
    for i, (p, n) in enumerate(zip(wall_points, wall_normals)):
        plt.arrow(p[0], p[1], scale * n[0], scale * n[1], head_width=0.03, color='green', alpha=0.5)

    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.title('L-Shaped Domain')
    plt.grid(True)
    plt.savefig('plots/domain.png')
    plt.close()
    print("Domain plot saved")
    
    print("Setting up optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting main training loop...")
    trained_model = train_model(
        model, optimizer, domain_points, inlet_points, outlet_points, 
        wall_points, wall_normals, epochs=500
    )
    
    print("Creating visualizations...")
    visualize_results(trained_model, domain_points, inside_L, wall_segments, inlet_points, outlet_points,device)
    analyze_reflection(trained_model, wall_points, wall_normals,device)
    
    print("Saving model...")
    torch.save(trained_model.state_dict(), 'models/simple_pinn.pth')
    
    print("All done!")
    return trained_model

if __name__ == "__main__":
    try:
        print("Starting main function...")
        main()
        print("Script completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()