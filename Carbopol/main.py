import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from model import DeepPINN
from training import train_model, train_staged
from domain import generate_domain_points, generate_boundary_points, inside_L
from visualization import visualize_results

print("===== Starting PINN simulation for Carbopol =====")
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("NumPy:", np.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def main():
    print("Initializing model...")
    model = DeepPINN().to(device)

    print("Generating domain and boundary points...")
    domain_points = generate_domain_points()
    wall_points, wall_normals, inlet_points, outlet_points, wall_segments = generate_boundary_points()

    plt.figure(figsize=(8, 6))
    plt.scatter(domain_points[:, 0], domain_points[:, 1], s=1, alpha=0.3, label='Domain')
    plt.scatter(wall_points[:, 0], wall_points[:, 1], s=5, c='k', label='Wall')
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], s=5, c='b', label='Inlet')
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], s=5, c='r', label='Outlet')
    for p, n in zip(wall_points, wall_normals):
        plt.arrow(p[0], p[1], 0.05 * n[0], 0.05 * n[1], head_width=0.01, color='green', alpha=0.5)
    plt.axis('equal')
    plt.title("Domain Geometry")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/domain.png")
    plt.close()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Training model...")
    trained_model = train_staged(model, optimizer, domain_points, inlet_points, outlet_points, wall_points, wall_normals)

    print("Generating visualizations...")
    visualize_results(trained_model, domain_points, inside_L, wall_segments, inlet_points, outlet_points, device)

    print("Saving model...")
    torch.save(trained_model.state_dict(), "models/carbopol_model.pth")
    print("Training complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
