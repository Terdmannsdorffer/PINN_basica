import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator # Make sure this import is outside the function if used elsewhere

# --- Assume previous code exists: device, model, domain_points, etc. ---
# --- Assume inside_L, wall_segments, inlet_points, outlet_points are defined ---

# Helper function to generate points inside the L-shape
def generate_points_inside(n_points, x_min, x_max, y_min, y_max, inside_func, max_tries=100):
    """Generates n_points randomly sampled points strictly inside the domain defined by inside_func."""
    points_x = []
    points_y = []
    attempts = 0
    while len(points_x) < n_points and attempts < max_tries * n_points:
        # Generate candidate points within the bounding box
        cand_x = np.random.uniform(x_min, x_max, n_points - len(points_x))
        cand_y = np.random.uniform(y_min, y_max, n_points - len(points_x))

        # Check which candidates are inside the actual domain
        is_inside = np.array([inside_func(px, py) for px, py in zip(cand_x, cand_y)])

        # Add the valid points
        points_x.extend(cand_x[is_inside])
        points_y.extend(cand_y[is_inside])
        attempts += n_points - len(points_x)

    if len(points_x) < n_points:
        print(f"Warning: Could only generate {len(points_x)} points inside the domain after many tries.")

    return np.array(points_x), np.array(points_y)


# Modified Visualization function including the fix
def visualize_results(model, domain_points, inside_L, wall_segments, inlet_points, outlet_points, device):
    print("Creating visualization...")

    # --- (Keep the existing velocity and pressure plot generation) ---
    # Get model predictions (only need this once if called within visualize_results)
    model.eval()
    with torch.no_grad():
        xy_tensor = torch.tensor(domain_points, dtype=torch.float32, device=device)
        outputs = model(xy_tensor).cpu().numpy()

    u_domain = outputs[:, 0]
    v_domain = outputs[:, 1]
    p_domain = outputs[:, 2]
    vel_mag_domain = np.sqrt(u_domain**2 + v_domain**2)

# Modified part of visualize_results function to fix viscosity calculation
# Inside the visualize_results function, add a flow direction analysis
    print("Analyzing flow direction...")
    plt.figure(figsize=(12, 9))

    # Draw domain boundaries
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    # Draw a quiver plot (vector field) to visualize direction clearly
    # Use sparser sampling for clarity
    n_samples = min(1000, len(domain_points))
    indices = np.random.choice(len(domain_points), n_samples, replace=False)
    sample_points = domain_points[indices]
    sample_u = u_domain[indices]
    sample_v = v_domain[indices]

    # Scale arrows for visibility
    arrow_scale = 40.0 
    plt.quiver(sample_points[:, 0], sample_points[:, 1], sample_u, sample_v, 
            color='blue', scale=arrow_scale, width=0.002)

    # Mark inlet and outlet clearly
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=100, label='Inlet')
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=100, label='Outlet')

    plt.title('Carbopol Flow - Direction Analysis')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/flow_direction_analysis.png')
    plt.close()
# Inside visualize_results function, replace the current shear rate and viscosity calculation with:

    print("Computing shear rate and effective viscosity for Carbopol...")
    xy_tensor.requires_grad_(True)
    outputs = model(xy_tensor)
    u = outputs[:, 0:1]
    v = outputs[:, 1:2]

    du = torch.autograd.grad(u, xy_tensor, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    dv = torch.autograd.grad(v, xy_tensor, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]

    du_dx = du[:, 0:1]
    du_dy = du[:, 1:2]
    dv_dx = dv[:, 0:1]
    dv_dy = dv[:, 1:2]

    # Calculate shear rate using proper tensor operations
    gamma_dot = torch.sqrt(2 * ((du_dx)**2 + (dv_dy)**2 + 0.5*(du_dy + dv_dx)**2) + 1e-8)

    # Carbopol parameters from rheological study
    tau_y = 5.0  # Yield stress in Pa
    k = 2.5      # Consistency index
    n = 0.42     # Power law index

    # Herschel-Bulkley model for Carbopol's effective viscosity
    eta_eff = (tau_y / (gamma_dot + 1e-6)) + k * (gamma_dot ** (n - 1))

    gamma_dot = gamma_dot.detach().cpu().numpy().flatten()
    eta_eff = eta_eff.detach().cpu().numpy().flatten()

    # Ensure the viscosity plot is correctly created
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=eta_eff, cmap='viridis', s=5, 
                        norm=plt.matplotlib.colors.LogNorm())  # Use log scale for better visualization
    plt.colorbar(scatter, label='Effective Viscosity (Pa·s)')
    plt.title('Carbopol Flow - Effective Viscosity')
    plt.xlabel('x')
    plt.ylabel('y')
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('plots/carbopol_viscosity_field.png')
    plt.close()

    # Ensure a shear rate plot is created too
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=gamma_dot, cmap='hot', s=5)
    plt.colorbar(scatter, label='Shear Rate (1/s)')
    plt.title('Carbopol Flow - Shear Rate')
    plt.xlabel('x')
    plt.ylabel('y')
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('plots/carbopol_shear_rate.png')
    plt.close()

    # Velocity magnitude plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=vel_mag_domain, cmap='viridis', s=5) # Smaller dots for clarity
    plt.colorbar(scatter, label='Velocity magnitude')
    plt.title('Flow Field - Velocity Magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    # Add walls for context
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('plots/flow_field.png')
    plt.close()

    # Pressure plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=p_domain, cmap='plasma', s=5) # Smaller dots
    plt.colorbar(scatter, label='Pressure')
    plt.title('Flow Field - Pressure')
    plt.xlabel('x')
    plt.ylabel('y')
    # Add walls for context
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('plots/pressure_field.png')
    plt.close()


    # Streamlines - Calculate grid velocity
    print("Creating streamline plot...")
    # Define bounding box slightly larger than expected domain for interpolation buffer
    x_min_bound = np.min(domain_points[:, 0]) - 0.1
    x_max_bound = np.max(domain_points[:, 0]) + 0.1
    y_min_bound = np.min(domain_points[:, 1]) - 0.1
    y_max_bound = np.max(domain_points[:, 1]) + 0.1
    nx, ny = 60, 40 # Increased resolution for better interpolation
    x_grid_coords = np.linspace(x_min_bound, x_max_bound, nx)
    y_grid_coords = np.linspace(y_min_bound, y_max_bound, ny)
    X, Y = np.meshgrid(x_grid_coords, y_grid_coords)
    grid_points_flat = np.column_stack((X.flatten(), Y.flatten()))

    # Get predictions ONLY for points inside the L-shape for accurate interpolation near boundaries
    # Use the inside_L function passed to visualize_results
    inside_mask_grid = np.array([inside_L(px, py) for px, py in grid_points_flat])
    grid_inside = grid_points_flat[inside_mask_grid]

    u_grid = np.full(X.shape, np.nan) # Initialize with NaN
    v_grid = np.full(X.shape, np.nan) # Initialize with NaN

    if len(grid_inside) > 0:
        with torch.no_grad():
            xy_grid_tensor = torch.tensor(grid_inside, dtype=torch.float32, device=device)
            grid_outputs = model(xy_grid_tensor).cpu().numpy()

        # Map results back to the grid, leaving NaNs outside the L-shape
        inside_indices_flat = np.where(inside_mask_grid)[0]
        rows, cols = np.unravel_index(inside_indices_flat, X.shape)
        u_grid[rows, cols] = grid_outputs[:, 0]
        v_grid[rows, cols] = grid_outputs[:, 1]
    else:
        print("Warning: No grid points found inside the L-shape for streamline calculation.")


    # --- (Keep the static and colored streamline plot generation) ---
    # Static streamlines
    plt.figure(figsize=(10, 8))
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    # Use nan values in u_grid/v_grid to prevent streamplot from drawing outside the domain
    plt.streamplot(X, Y, u_grid, v_grid, density=1.5, color='blue', linewidth=1, arrowsize=1.5, broken_streamlines=False)
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=30, label='Inlet', zorder=5)
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=30, label='Outlet', zorder=5)
    plt.title('Flow Field - Streamlines')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min_bound, x_max_bound)
    plt.ylim(y_min_bound, y_max_bound)
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/streamlines.png')
    plt.close()

    # Streamlines colored by velocity
    plt.figure(figsize=(12, 9))
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    vel_mag_grid = np.sqrt(u_grid**2 + v_grid**2) # NaNs will propagate
    streamlines = plt.streamplot(X, Y, u_grid, v_grid, density=1.5, color=vel_mag_grid, cmap='viridis', linewidth=1.5, arrowsize=1.5, broken_streamlines=False)
    # Avoid error if no streamlines were generated
    if hasattr(streamlines, 'lines') and streamlines.lines is not None:
         plt.colorbar(streamlines.lines, label='Velocity magnitude')
    else:
        print("Warning: No streamlines generated for colored plot.")
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=30, label='Inlet', zorder=5)
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=30, label='Outlet', zorder=5)
    plt.title('Flow Field - Streamlines Colored by Velocity Magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min_bound, x_max_bound)
    plt.ylim(y_min_bound, y_max_bound)
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/streamlines_colored.png')
    plt.close()


    print("Visualizations generation complete.")



# Reflect analysis (simplified)
def analyze_reflection(model, wall_points, wall_normals,device):
    print("Analyzing reflection at walls...")

    if len(wall_points) < 5:
        print("Not enough wall points for reflection analysis")
        return

    indices = np.linspace(0, len(wall_points)-1, 5, dtype=int)
    test_points = wall_points[indices]
    test_normals = wall_normals[indices]

    plt.figure(figsize=(15, 6))
    for i in range(len(test_points)):
        plt.subplot(1, 5, i+1)
        wall_pt = test_points[i]
        normal = test_normals[i]
        tangent = [-normal[1], normal[0]]

        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor([wall_pt], dtype=torch.float32, device=device))
            u, v = pred[0, 0].item(), pred[0, 1].item()

        plt.plot([wall_pt[0]-tangent[0]*0.2, wall_pt[0]+tangent[0]*0.2],
                 [wall_pt[1]-tangent[1]*0.2, wall_pt[1]+tangent[1]*0.2],
                 'k-', linewidth=2)

        plt.arrow(wall_pt[0], wall_pt[1], 
                  normal[0]*0.1, normal[1]*0.1, 
                  head_width=0.02, color='blue')

        plt.arrow(wall_pt[0], wall_pt[1],
                  u*0.1, v*0.1,
                  head_width=0.02, color='red')

        plt.title(f'Point {i+1}')
        plt.xlim(wall_pt[0]-0.3, wall_pt[0]+0.3)
        plt.ylim(wall_pt[1]-0.3, wall_pt[1]+0.3)
        plt.axis('equal')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('plots/reflection_analysis.png')
    plt.close()
    print("Reflection analysis saved")