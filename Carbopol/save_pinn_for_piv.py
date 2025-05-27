# save_pinn_for_piv.py
"""
Extract and save PINN velocity field data for PIV comparison
Run this AFTER training your model with main.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import DeepPINN
from domain import inside_L

def extract_pinn_velocity_field(model, device, resolution=100):
    """
    Extract velocity field from trained PINN model on a regular grid
    
    Returns:
        x_grid, y_grid: coordinate meshgrids
        u_field, v_field: velocity components  
        p_field: pressure field
        velocity_magnitude: velocity magnitude
    """
    print(f"Extracting velocity field at {resolution}x{resolution} resolution...")
    
    # L-shaped domain dimensions (same as in your domain.py)
    L_up = 0.097
    L_down = 0.157
    H_left = 0.3
    H_right = 0.1
    
    # Create coordinate grid
    x = np.linspace(0, L_down, resolution)
    y = np.linspace(0, H_left, resolution)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Flatten for model input
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()
    xy_points = np.column_stack([x_flat, y_flat])
    
    # Keep only points inside L-shaped domain
    inside_mask = np.array([inside_L(x, y) for x, y in xy_points])
    xy_inside = xy_points[inside_mask]
    
    print(f"Evaluating model at {len(xy_inside)} points inside domain...")
    
    # Get PINN predictions
    model.eval()
    with torch.no_grad():
        xy_tensor = torch.tensor(xy_inside, dtype=torch.float32, device=device)
        predictions = model(xy_tensor)
        
        # Extract velocity components and pressure
        u_inside = predictions[:, 0].cpu().numpy()  # u-velocity
        v_inside = predictions[:, 1].cpu().numpy()  # v-velocity
        p_inside = predictions[:, 2].cpu().numpy()  # pressure
    
    # Create full arrays (NaN outside domain)
    u_full = np.full(len(xy_points), np.nan)
    v_full = np.full(len(xy_points), np.nan)
    p_full = np.full(len(xy_points), np.nan)
    
    u_full[inside_mask] = u_inside
    v_full[inside_mask] = v_inside
    p_full[inside_mask] = p_inside
    
    # Reshape to grid
    u_field = u_full.reshape(x_grid.shape)
    v_field = v_full.reshape(x_grid.shape)
    p_field = p_full.reshape(x_grid.shape)
    
    # Calculate velocity magnitude
    velocity_magnitude = np.sqrt(u_field**2 + v_field**2)
    
    print("Extraction complete!")
    print(f"Max velocity: {np.nanmax(velocity_magnitude):.4f} m/s")
    print(f"Mean velocity: {np.nanmean(velocity_magnitude):.4f} m/s")
    
    return x_grid, y_grid, u_field, v_field, p_field, velocity_magnitude

def save_for_piv_comparison(x_grid, y_grid, u_field, v_field, p_field, velocity_magnitude):
    """
    Save PINN data in formats suitable for PIV comparison
    """
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Save as compressed numpy file
    filename_npz = 'data/pinn_velocity_for_piv.npz'
    np.savez_compressed(filename_npz,
                       x=x_grid, y=y_grid,
                       u=u_field, v=v_field, p=p_field,
                       velocity_magnitude=velocity_magnitude)
    print(f"Data saved as: {filename_npz}")
    
    # Also save as MATLAB format if scipy available
    try:
        import scipy.io
        filename_mat = 'data/pinn_velocity_for_piv.mat'
        scipy.io.savemat(filename_mat, {
            'x': x_grid,
            'y': y_grid, 
            'u': u_field,
            'v': v_field,
            'p': p_field,
            'velocity_magnitude': velocity_magnitude
        })
        print(f"Data also saved as MATLAB format: {filename_mat}")
    except ImportError:
        print("scipy not available - MATLAB format not saved")
    
    # Create a quick visualization
    plt.figure(figsize=(15, 5))
    
    # U-velocity
    plt.subplot(1, 3, 1)
    plt.contourf(x_grid, y_grid, u_field, levels=20, cmap='RdBu_r')
    plt.colorbar(label='U-velocity (m/s)')
    plt.title('PINN: U-velocity')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    
    # V-velocity
    plt.subplot(1, 3, 2)
    plt.contourf(x_grid, y_grid, v_field, levels=20, cmap='RdBu_r')
    plt.colorbar(label='V-velocity (m/s)')
    plt.title('PINN: V-velocity')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    
    # Velocity magnitude
    plt.subplot(1, 3, 3)
    plt.contourf(x_grid, y_grid, velocity_magnitude, levels=20, cmap='viridis')
    plt.colorbar(label='Velocity Magnitude (m/s)')
    plt.title('PINN: Velocity Magnitude')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('plots/pinn_velocity_for_piv_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Visualization saved as: plots/pinn_velocity_for_piv_comparison.png")

def main():
    """
    Main function - extracts PINN velocity field for PIV comparison
    """
    print("=== Extract PINN Data for PIV Comparison ===")
    
    # Check if trained model exists
    model_path = "models/carbopol_model.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: No trained model found at {model_path}")
        print("Please run your main.py first to train the model!")
        return
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DeepPINN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("PINN model loaded successfully!")
    
    # Extract velocity field
    print("\nExtracting velocity field...")
    x_grid, y_grid, u_field, v_field, p_field, velocity_magnitude = extract_pinn_velocity_field(
        model, device, resolution=100)
    
    # Save data
    print("\nSaving data for PIV comparison...")
    save_for_piv_comparison(x_grid, y_grid, u_field, v_field, p_field, velocity_magnitude)
    
    print("\n=== COMPLETE ===")
    print("Your PINN velocity field data is now ready for PIV comparison!")
    print("\nFiles created:")
    print("  - data/pinn_velocity_for_piv.npz (numpy format)")
    print("  - data/pinn_velocity_for_piv.mat (MATLAB format, if scipy available)")
    print("  - plots/pinn_velocity_for_piv_comparison.png (visualization)")
    print("\nYou can now load this data in MATLAB or Python to compare with your PIV results.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        traceback.print_exc()