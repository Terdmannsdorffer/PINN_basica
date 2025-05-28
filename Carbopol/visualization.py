import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import RegularGridInterpolator

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

# Modified Visualization function with animation
def visualize_results(model, domain_points, inside_L, wall_segments, inlet_points, outlet_points, device):
    print("Creating visualization for updated L-shape domain...")
    
    # Get L-pipe dimensions for correct scaling of visualizations
    L_up = 0.097  # Upper horizontal length
    L_down = 0.157  # Lower horizontal length
    H_left = 0.3   # Left vertical height
    H_right = 0.1  # Right vertical height

    # --- Get model predictions ---
    model.eval()
    with torch.no_grad():
        xy_tensor = torch.tensor(domain_points, dtype=torch.float32, device=device)
        outputs = model(xy_tensor).cpu().numpy()

    u_domain = outputs[:, 0]
    v_domain = outputs[:, 1]
    p_domain = outputs[:, 2]
    vel_mag_domain = np.sqrt(u_domain**2 + v_domain**2)

    # --- Flow direction analysis ---
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
    arrow_scale = 20.0  # Adjusted scale for smaller domain
    plt.quiver(sample_points[:, 0], sample_points[:, 1], sample_u, sample_v, 
            color='blue', scale=arrow_scale, width=0.002)

    # Mark inlet and outlet clearly
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=100, label='Inlet')
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=100, label='Outlet')

    plt.title('Carbopol Flow - Direction Analysis')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/flow_direction_analysis.png')
    plt.close()

    # --- Shear rate and effective viscosity calculation ---
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

    # Carbopol parameters from rheological study (updated as per training.py)
    tau_y = 35.55  # Yield stress in Pa - updated from 5.0
    k = 2.32      # Consistency index - updated from 2.5
    n = 0.74     # Power law index - updated from 0.42

    # Herschel-Bulkley model for Carbopol's effective viscosity
    eta_eff = (tau_y / (gamma_dot + 1e-6)) + k * (gamma_dot ** (n - 1))

    gamma_dot = gamma_dot.detach().cpu().numpy().flatten()
    eta_eff = eta_eff.detach().cpu().numpy().flatten()

    # --- Viscosity plot ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=eta_eff, cmap='viridis', s=5, 
                        norm=plt.matplotlib.colors.LogNorm())  # Use log scale for better visualization
    plt.colorbar(scatter, label='Effective Viscosity (Pa·s)')
    plt.title('Carbopol Flow - Effective Viscosity')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('plots/carbopol_viscosity_field.png')
    plt.close()

    # --- Shear rate plot ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=gamma_dot, cmap='hot', s=5)
    plt.colorbar(scatter, label='Shear Rate (1/s)')
    plt.title('Carbopol Flow - Shear Rate')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('plots/carbopol_shear_rate.png')
    plt.close()

    # --- Velocity magnitude plot ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=vel_mag_domain, cmap='viridis', s=5)
    plt.colorbar(scatter, label='Velocity magnitude (m/s)')
    plt.title('Flow Field - Velocity Magnitude')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    # Add walls for context
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('plots/flow_field.png')
    plt.close()

    # --- Pressure plot ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=p_domain, cmap='plasma', s=5)
    plt.colorbar(scatter, label='Pressure (Pa)')
    plt.title('Flow Field - Pressure')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    # Add walls for context
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=1)
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('plots/pressure_field.png')
    plt.close()

    # --- Streamlines setup with improved quality ---
    print("Creating high-quality streamline visualization...")
    # Define bounding box slightly larger than expected domain for interpolation buffer
    x_min_bound = -0.01
    x_max_bound = L_down + 0.01
    y_min_bound = -0.01
    y_max_bound = H_left + 0.01
    nx, ny = 300, 240  # Higher resolution for smoother streamlines
    x_grid_coords = np.linspace(x_min_bound, x_max_bound, nx)
    y_grid_coords = np.linspace(y_min_bound, y_max_bound, ny)
    X, Y = np.meshgrid(x_grid_coords, y_grid_coords)
    grid_points_flat = np.column_stack((X.flatten(), Y.flatten()))

    # Get predictions ONLY for points inside the L-shape
    inside_mask_grid = np.array([inside_L(px, py) for px, py in grid_points_flat])
    # Create a buffer zone by eroding the mask slightly (larger buffer)
    for i in range(len(inside_mask_grid)):
        x, y = grid_points_flat[i]
        if inside_mask_grid[i]:
            # Check if any of the surrounding points is outside with a larger buffer
            buffer = 0.008  # Increased buffer size for better wall separation
            for dx, dy in [(-buffer, 0), (buffer, 0), (0, -buffer), (0, buffer),
                        (-buffer, -buffer), (-buffer, buffer), (buffer, -buffer), (buffer, buffer)]:
                if not inside_L(x + dx, y + dy):
                    inside_mask_grid[i] = False
                    break
    grid_inside = grid_points_flat[inside_mask_grid]

    u_grid = np.full(X.shape, np.nan)  # Initialize with NaN
    v_grid = np.full(X.shape, np.nan)  # Initialize with NaN

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

    # Force zero velocity near boundaries to stop streamlines - more aggressive
    buffer_size = min((x_max_bound - x_min_bound)/nx, (y_max_bound - y_min_bound)/ny) * 2.5
    for j in range(ny):
        for i in range(nx):
            if not np.isnan(u_grid[j, i]):
                x, y = X[j, i], Y[j, i]
                # Check if this point is near any wall segment
                for segment in wall_segments:
                    (x1, y1), (x2, y2) = segment
                    # Calculate distance to line segment
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length > 0:
                        # Find closest point on line
                        t = max(0, min(1, ((x-x1)*(x2-x1) + (y-y1)*(y2-y1)) / (length**2)))
                        px = x1 + t * (x2 - x1)
                        py = y1 + t * (y2 - y1)
                        # Calculate distance
                        dist = np.sqrt((x-px)**2 + (y-py)**2)
                        if dist < buffer_size:
                            # Zero out velocity near walls
                            u_grid[j, i] = 0
                            v_grid[j, i] = 0
                            break

    # -- Calculate velocity magnitude grid --
    vel_mag_grid = np.sqrt(u_grid**2 + v_grid**2)  # NaNs will propagate

    # -- High quality streamlines --
    plt.figure(figsize=(12, 10))
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    # Custom colormap for velocity magnitude
    custom_cmap = plt.cm.viridis

    # Use enhanced streamplot settings
    streamlines = plt.streamplot(X, Y, u_grid, v_grid, 
                            density=2.0,  # Increased density
                            color=vel_mag_grid,  # Color by velocity
                            cmap=custom_cmap, 
                            linewidth=1.5, 
                            arrowsize=1.5, 
                            arrowstyle='->', 
                            broken_streamlines=True)  # Stop at boundaries

    # Add colorbar for velocity magnitude
    cbar = plt.colorbar(streamlines.lines, label='Velocity magnitude (m/s)')
    cbar.ax.tick_params(labelsize=10)

    # Mark inlet and outlet with more prominence
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=80, label='Inlet', zorder=5, edgecolors='black')
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=80, label='Outlet', zorder=5, edgecolors='black')

    # Improve plot aesthetics
    plt.title('Carbopol Flow - Streamlines Colored by Velocity Magnitude', fontsize=14, fontweight='bold')
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    plt.xlim(x_min_bound, x_max_bound)
    plt.ylim(y_min_bound, y_max_bound)
    plt.axis('equal')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/high_quality_streamlines.png', dpi=300)
    plt.close()

    # --- ANIMATED STREAMLINES GIF ---
    print("Creating animated streamlines GIF showing arrows moving across entire L-shape...")

    # Setup the figure for animation
    fig, ax = plt.subplots(figsize=(12, 9))

    # Draw domain boundaries
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    # Mark inlet and outlet
    ax.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=80, label='Inlet', zorder=5, edgecolors='black')
    ax.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=80, label='Outlet', zorder=5, edgecolors='black')

    # Create a static background streamplot - use a different style for better contrast with arrows
    streamplot = ax.streamplot(X, Y, u_grid, v_grid, density=1.2, 
                            color='lightgray', linewidth=0.5, arrowsize=0.5)

    # Generate arrows that will move along the streamlines
    num_arrows = 200  # More arrows for better coverage

    # Get the flattened arrays of valid points for interpolation
    valid_mask = ~np.isnan(u_grid)
    x_flat = X[valid_mask]
    y_flat = Y[valid_mask]
    u_flat = u_grid[valid_mask]
    v_flat = v_grid[valid_mask]

    # Define key regions of the L-shape to ensure coverage based on new dimensions
    regions = [
        # Vertical part (from inlet)
        {"x_min": -0.02, "x_max": 0.05, "y_min": 0.1, "y_max": H_left, "weight": 0.4},
        # Horizontal part (near outlet)
        {"x_min": L_up, "x_max": L_down, "y_min": 0, "y_max": H_right, "weight": 0.3},
        # Corner/bend area
        {"x_min": 0, "x_max": L_up, "y_min": 0, "y_max": 0.1, "weight": 0.3}
    ]

    # Initialize arrow arrays
    arrows_x = []
    arrows_y = []
    arrows_u = []
    arrows_v = []
    arrow_colors = []

    # Function to seed arrows in a specific region
    def seed_region(x_min, x_max, y_min, y_max, count):
        seeded = 0
        max_attempts = count * 5  # Limit attempts to avoid infinite loop
        attempts = 0
        
        local_x, local_y, local_u, local_v, local_colors = [], [], [], [], []
        
        while seeded < count and attempts < max_attempts:
            attempts += 1
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            
            # Only add if inside domain
            if inside_L(x, y):
                # Get velocity at this point
                distances = np.sqrt((x_flat - x)**2 + (y_flat - y)**2)
                if len(distances) > 0:
                    nearest_indices = np.argsort(distances)[:4]
                    
                    # Weighted average based on distance
                    weights = 1.0 / (distances[nearest_indices] + 1e-10)
                    weights = weights / np.sum(weights)
                    
                    u_val = np.sum(u_flat[nearest_indices] * weights)
                    v_val = np.sum(v_flat[nearest_indices] * weights)
                    
                    # Only add if velocity is non-negligible
                    vel_mag = np.sqrt(u_val**2 + v_val**2)
                    if vel_mag > 0.001:
                        local_x.append(x)
                        local_y.append(y)
                        local_u.append(u_val / vel_mag)  # Normalized direction
                        local_v.append(v_val / vel_mag)  # Normalized direction
                        local_colors.append(vel_mag)  # Color by velocity magnitude
                        seeded += 1
        
        return local_x, local_y, local_u, local_v, local_colors

    # Seed arrows in each region according to weights
    for region in regions:
        count = int(num_arrows * region["weight"])
        x_list, y_list, u_list, v_list, c_list = seed_region(
            region["x_min"], region["x_max"], 
            region["y_min"], region["y_max"], 
            count
        )
        arrows_x.extend(x_list)
        arrows_y.extend(y_list)
        arrows_u.extend(u_list)
        arrows_v.extend(v_list)
        arrow_colors.extend(c_list)

    # Create initial quiver plot for the arrows - use viridis colormap for better visibility
    quiver = ax.quiver(arrows_x, arrows_y, arrows_u, arrows_v, arrow_colors,
                    scale=25, width=0.003, cmap='viridis', pivot='mid',
                    zorder=10)

    plt.colorbar(quiver, label='Velocity Magnitude (m/s)')

    # Function to update arrow positions for animation
    def update_arrows(frame):
        # Speed factor - adjust for faster/slower movement
        speed_factor = 0.01  # Reduced for smaller domain
        
        # New positions for arrows - initialize with same size as original
        new_x = []
        new_y = []
        new_u = []
        new_v = []
        new_colors = []
        
        for i in range(len(arrows_x)):
            # Current position
            x, y = arrows_x[i], arrows_y[i]
            
            # Get velocity at current position
            if inside_L(x, y):
                # Find nearest grid points
                distances = np.sqrt((x_flat - x)**2 + (y_flat - y)**2)
                if len(distances) > 0:
                    nearest_indices = np.argsort(distances)[:4]
                    
                    # Weighted average based on distance
                    weights = 1.0 / (distances[nearest_indices] + 1e-10)
                    weights = weights / np.sum(weights)
                    
                    u_val = np.sum(u_flat[nearest_indices] * weights)
                    v_val = np.sum(v_flat[nearest_indices] * weights)
                    
                    # Update position based on velocity
                    vel_mag = np.sqrt(u_val**2 + v_val**2)
                    if vel_mag > 0.001:
                        # Movement is proportional to velocity magnitude
                        dx = u_val * speed_factor
                        dy = v_val * speed_factor
                        
                        # New position
                        new_x_pos = x + dx
                        new_y_pos = y + dy
                        
                        # Check if new position is inside domain
                        if inside_L(new_x_pos, new_y_pos):
                            new_x.append(new_x_pos)
                            new_y.append(new_y_pos)
                            new_u.append(u_val / vel_mag)  # Normalized direction
                            new_v.append(v_val / vel_mag)  # Normalized direction
                            new_colors.append(vel_mag)
                            # Update the original arrays
                            arrows_x[i] = new_x_pos
                            arrows_y[i] = new_y_pos
                            arrows_u[i] = u_val / vel_mag
                            arrows_v[i] = v_val / vel_mag
                            continue
            
            # If we reached here, either:
            # - The arrow is outside the domain
            # - The velocity is negligible
            # - There was an interpolation issue
            # So, we reset the arrow to a new position
            
            # Choose a region to reset to based on weights
            reset_region = np.random.choice(len(regions), p=[r["weight"] for r in regions])
            region = regions[reset_region]
            
            # Try to find a valid point in the chosen region
            reset_x, reset_y = x, y  # Default to current position
            for _ in range(10):  # Try up to 10 times
                reset_x = np.random.uniform(region["x_min"], region["x_max"])
                reset_y = np.random.uniform(region["y_min"], region["y_max"])
                if inside_L(reset_x, reset_y):
                    break
            
            # If inside domain, get velocity at reset position
            if inside_L(reset_x, reset_y):
                # Find nearest grid points
                distances = np.sqrt((x_flat - reset_x)**2 + (y_flat - reset_y)**2)
                if len(distances) > 0:
                    nearest_indices = np.argsort(distances)[:4]
                    
                    # Weighted average based on distance
                    weights = 1.0 / (distances[nearest_indices] + 1e-10)
                    weights = weights / np.sum(weights)
                    
                    u_reset = np.sum(u_flat[nearest_indices] * weights)
                    v_reset = np.sum(v_flat[nearest_indices] * weights)
                    
                    # Only add if velocity is non-negligible
                    vel_mag_reset = np.sqrt(u_reset**2 + v_reset**2)
                    if vel_mag_reset > 0.001:
                        new_x.append(reset_x)
                        new_y.append(reset_y)
                        new_u.append(u_reset / vel_mag_reset)  # Normalized direction
                        new_v.append(v_reset / vel_mag_reset)  # Normalized direction
                        new_colors.append(vel_mag_reset)
                        # Update the original arrays
                        arrows_x[i] = reset_x
                        arrows_y[i] = reset_y
                        arrows_u[i] = u_reset / vel_mag_reset
                        arrows_v[i] = v_reset / vel_mag_reset
                    else:
                        # Keep current position but with zero velocity
                        new_x.append(x)
                        new_y.append(y)
                        new_u.append(0.0)
                        new_v.append(0.0)
                        new_colors.append(0.0)
                        arrows_x[i] = x
                        arrows_y[i] = y
                        arrows_u[i] = 0.0
                        arrows_v[i] = 0.0
                else:
                    # Keep current position but with zero velocity
                    new_x.append(x)
                    new_y.append(y)
                    new_u.append(0.0)
                    new_v.append(0.0)
                    new_colors.append(0.0)
                    arrows_x[i] = x
                    arrows_y[i] = y
                    arrows_u[i] = 0.0
                    arrows_v[i] = 0.0
            else:
                # Keep current position but with zero velocity
                new_x.append(x)
                new_y.append(y)
                new_u.append(0.0)
                new_v.append(0.0)
                new_colors.append(0.0)
                arrows_x[i] = x
                arrows_y[i] = y
                arrows_u[i] = 0.0
                arrows_v[i] = 0.0
        
        # Ensure we have exactly the same number of elements as the original quiver
        assert len(new_x) == len(arrows_x), f"Size mismatch: {len(new_x)} != {len(arrows_x)}"
        assert len(new_u) == len(arrows_x), f"Size mismatch: {len(new_u)} != {len(arrows_x)}"
        assert len(new_v) == len(arrows_x), f"Size mismatch: {len(new_v)} != {len(arrows_x)}"
        
        # Update the quiver plot
        quiver.set_offsets(np.c_[new_x, new_y])
        quiver.set_UVC(np.array(new_u), np.array(new_v))
        
        # Update colors if we have them
        if hasattr(quiver, 'set_array') and len(new_colors) > 0:
            quiver.set_array(np.array(new_colors))
        
        return quiver,

    # Create and save the animation
    frames = 150  # Increased number of frames for smoother animation
    print(f"Creating animation with {frames} frames...")
    anim = animation.FuncAnimation(fig, update_arrows, frames=frames, interval=50, blit=True)

    # Set plot properties
    plt.title('Carbopol Flow - Animated Streamlines', fontsize=14, fontweight='bold')
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    plt.xlim(x_min_bound, x_max_bound)
    plt.ylim(y_min_bound, y_max_bound)
    plt.axis('equal')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Save animation with higher quality
    print("Saving animated streamlines GIF...")
    anim.save('plots/streamlines_animated.gif', writer='pillow', fps=20, dpi=150)
    plt.close()

    print("Visualizations generation complete.")