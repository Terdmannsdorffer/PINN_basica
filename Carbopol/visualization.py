# visualization.py - FIXED VERSION (Animation Bug Fix)
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# CONSISTENT dimensions with domain.py (RESTORED TO EXPERIMENTAL VALUES)
L_up = 0.097      # Upper horizontal length  
L_down = 0.174    # Total horizontal length
H_left = 0.119    # Left vertical height
H_right = 0.019   # RIGHT VERTICAL HEIGHT - RESTORED TO EXPERIMENTAL VALUE

def visualize_results(model, domain_points, inside_L, wall_segments, inlet_points, outlet_points, device):
    """Corrected visualization with animation bug fix"""
    print("Creating visualization for corrected L-shape domain...")
    
    # --- Get model predictions ---
    model.eval()
    with torch.no_grad():
        xy_tensor = torch.tensor(domain_points, dtype=torch.float32, device=device)
        outputs = model(xy_tensor).cpu().numpy()

    u_domain = outputs[:, 0]
    v_domain = outputs[:, 1]
    p_domain = outputs[:, 2]
    vel_mag_domain = np.sqrt(u_domain**2 + v_domain**2)

    print(f"Velocity statistics:")
    print(f"  u: [{u_domain.min():.6f}, {u_domain.max():.6f}] m/s")
    print(f"  v: [{v_domain.min():.6f}, {v_domain.max():.6f}] m/s")
    print(f"  magnitude: [{vel_mag_domain.min():.6f}, {vel_mag_domain.max():.6f}] m/s")
    print(f"  mean magnitude: {vel_mag_domain.mean():.6f} m/s")

    # --- Flow direction analysis ---
    print("Analyzing flow direction...")
    plt.figure(figsize=(12, 9))

    # Draw domain boundaries
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    # Draw a quiver plot (vector field) to visualize direction clearly
    n_samples = min(800, len(domain_points))
    indices = np.random.choice(len(domain_points), n_samples, replace=False)
    sample_points = domain_points[indices]
    sample_u = u_domain[indices]
    sample_v = v_domain[indices]

    # Scale arrows for visibility based on actual velocity scale
    max_vel = max(np.max(np.abs(sample_u)), np.max(np.abs(sample_v)))
    if max_vel > 0:
        arrow_scale = max_vel * 50  # Adjusted for the new velocity scale
    else:
        arrow_scale = 1.0
    
    plt.quiver(sample_points[:, 0], sample_points[:, 1], sample_u, sample_v, 
              color='blue', scale=arrow_scale, width=0.002, alpha=0.7)

    # Mark inlet and outlet clearly
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=100, label='Inlet', zorder=5)
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=100, label='Outlet', zorder=5)

    plt.title('Carbopol Flow - Direction Analysis (Improved 3-Stage)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim(-0.01, L_down + 0.01)
    plt.ylim(-0.01, H_left + 0.01)
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/flow_direction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Velocity magnitude plot ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=vel_mag_domain, cmap='viridis', s=5)
    plt.colorbar(scatter, label='Velocity magnitude (m/s)')
    plt.title('Flow Field - Velocity Magnitude (Improved 3-Stage)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # Add walls for context
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
        
    plt.xlim(-0.01, L_down + 0.01)
    plt.ylim(-0.01, H_left + 0.01)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/flow_field.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Pressure plot ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=p_domain, cmap='plasma', s=5)
    plt.colorbar(scatter, label='Pressure (Pa)')
    plt.title('Flow Field - Pressure (Improved 3-Stage)')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # Add walls for context
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
        
    plt.xlim(-0.01, L_down + 0.01)
    plt.ylim(-0.01, H_left + 0.01)
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/pressure_field.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Streamlines setup with corrected dimensions ---
    print("Creating high-quality streamline visualization...")
    x_min_bound = -0.005
    x_max_bound = L_down + 0.005
    y_min_bound = -0.005
    y_max_bound = H_left + 0.005
    
    nx, ny = 200, 150  # Reasonable resolution for corrected domain
    x_grid_coords = np.linspace(x_min_bound, x_max_bound, nx)
    y_grid_coords = np.linspace(y_min_bound, y_max_bound, ny)
    X, Y = np.meshgrid(x_grid_coords, y_grid_coords)
    grid_points_flat = np.column_stack((X.flatten(), Y.flatten()))

    # Get predictions ONLY for points inside the L-shape
    inside_mask_grid = np.array([inside_L(px, py) for px, py in grid_points_flat])
    
    # Create a buffer zone near walls
    for i in range(len(inside_mask_grid)):
        x, y = grid_points_flat[i]
        if inside_mask_grid[i]:
            buffer = 0.005  # Small buffer for wall separation
            for dx, dy in [(-buffer, 0), (buffer, 0), (0, -buffer), (0, buffer)]:
                if not inside_L(x + dx, y + dy):
                    inside_mask_grid[i] = False
                    break
                    
    grid_inside = grid_points_flat[inside_mask_grid]

    u_grid = np.full(X.shape, np.nan)
    v_grid = np.full(X.shape, np.nan)

    if len(grid_inside) > 0:
        with torch.no_grad():
            xy_grid_tensor = torch.tensor(grid_inside, dtype=torch.float32, device=device)
            grid_outputs = model(xy_grid_tensor).cpu().numpy()

        # Map results back to the grid
        inside_indices_flat = np.where(inside_mask_grid)[0]
        rows, cols = np.unravel_index(inside_indices_flat, X.shape)
        u_grid[rows, cols] = grid_outputs[:, 0]
        v_grid[rows, cols] = grid_outputs[:, 1]
    else:
        print("Warning: No grid points found inside the L-shape for streamline calculation.")

    # Force zero velocity near boundaries
    buffer_size = min((x_max_bound - x_min_bound)/nx, (y_max_bound - y_min_bound)/ny) * 2.0
    for j in range(ny):
        for i in range(nx):
            if not np.isnan(u_grid[j, i]):
                x, y = X[j, i], Y[j, i]
                # Check if this point is near any wall segment
                for segment in wall_segments:
                    (x1, y1), (x2, y2) = segment
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    if length > 0:
                        t = max(0, min(1, ((x-x1)*(x2-x1) + (y-y1)*(y2-y1)) / (length**2)))
                        px = x1 + t * (x2 - x1)
                        py = y1 + t * (y2 - y1)
                        dist = np.sqrt((x-px)**2 + (y-py)**2)
                        if dist < buffer_size:
                            u_grid[j, i] = 0
                            v_grid[j, i] = 0
                            break

    # Calculate velocity magnitude grid
    vel_mag_grid = np.sqrt(u_grid**2 + v_grid**2)

    # --- High quality streamlines ---
    plt.figure(figsize=(12, 10))
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    # Create streamplot
    streamlines = plt.streamplot(X, Y, u_grid, v_grid, 
                                density=1.5,
                                color=vel_mag_grid,
                                cmap='viridis', 
                                linewidth=1.0, 
                                arrowsize=1.2, 
                                arrowstyle='->', 
                                broken_streamlines=True)

    # Add colorbar for velocity magnitude
    cbar = plt.colorbar(streamlines.lines, label='Velocity magnitude (m/s)')
    cbar.ax.tick_params(labelsize=10)

    # Mark inlet and outlet
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=80, label='Inlet', zorder=5, edgecolors='black')
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=80, label='Outlet', zorder=5, edgecolors='black')

    plt.title('Carbopol Flow - Streamlines (Improved 3-Stage)', fontsize=14, fontweight='bold')
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    plt.xlim(x_min_bound, x_max_bound)
    plt.ylim(y_min_bound, y_max_bound)
    plt.axis('equal')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/high_quality_streamlines.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- SIMPLIFIED ANIMATED STREAMLINES (BUG FIX) ---
    print("Creating simplified animated streamlines...")

    fig, ax = plt.subplots(figsize=(12, 9))

    # Draw domain boundaries
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)

    # Mark inlet and outlet
    ax.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=80, label='Inlet', zorder=5, edgecolors='black')
    ax.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=80, label='Outlet', zorder=5, edgecolors='black')

    # Create background streamplot
    streamplot = ax.streamplot(X, Y, u_grid, v_grid, density=1.0, 
                              color='lightgray', linewidth=0.5, arrowsize=0.5)

    # SIMPLIFIED: Generate fewer, more stable moving arrows
    num_arrows = 50  # Reduced number for stability
    valid_mask = ~np.isnan(u_grid)
    x_flat = X[valid_mask]
    y_flat = Y[valid_mask]
    u_flat = u_grid[valid_mask]
    v_flat = v_grid[valid_mask]

    # Initialize arrows with FIXED size
    arrows_x = []
    arrows_y = []
    arrows_u = []
    arrows_v = []
    arrow_colors = []

    # Seed arrows more carefully to ensure consistent count
    for _ in range(num_arrows):
        attempts = 0
        while attempts < 50:  # Limit attempts
            # Random position in domain
            x = np.random.uniform(0, L_down)
            y = np.random.uniform(0, H_left)
            
            if inside_L(x, y):
                # Get velocity at this point
                distances = np.sqrt((x_flat - x)**2 + (y_flat - y)**2)
                if len(distances) > 0:
                    nearest_idx = np.argmin(distances)
                    u_val = u_flat[nearest_idx]
                    v_val = v_flat[nearest_idx]
                    
                    vel_mag = np.sqrt(u_val**2 + v_val**2)
                    if vel_mag > 0.001:  # Only add if velocity is significant
                        arrows_x.append(x)
                        arrows_y.append(y)
                        arrows_u.append(u_val / vel_mag)  # Normalized
                        arrows_v.append(v_val / vel_mag)  # Normalized
                        arrow_colors.append(vel_mag)
                        break
            attempts += 1
        
        # If we couldn't find a good position, add a default arrow
        if len(arrows_x) < _ + 1:
            arrows_x.append(0.05)
            arrows_y.append(0.05)
            arrows_u.append(0.0)
            arrows_v.append(0.0)
            arrow_colors.append(0.0)

    # Ensure we have exactly num_arrows
    arrows_x = arrows_x[:num_arrows]
    arrows_y = arrows_y[:num_arrows]
    arrows_u = arrows_u[:num_arrows]
    arrows_v = arrows_v[:num_arrows]
    arrow_colors = arrow_colors[:num_arrows]

    # Create initial quiver plot
    quiver = ax.quiver(arrows_x, arrows_y, arrows_u, arrows_v, arrow_colors,
                      scale=20, width=0.003, cmap='viridis', pivot='mid', zorder=10)
    plt.colorbar(quiver, label='Velocity Magnitude (m/s)')

    def update_arrows(frame):
        """FIXED: Update arrow positions with consistent array sizes"""
        speed_factor = 0.002  # Slow movement
        
        # Update positions maintaining exact same array size
        for i in range(len(arrows_x)):
            x, y = arrows_x[i], arrows_y[i]
            
            if inside_L(x, y):
                # Get velocity at current position
                distances = np.sqrt((x_flat - x)**2 + (y_flat - y)**2)
                if len(distances) > 0:
                    nearest_idx = np.argmin(distances)
                    u_val = u_flat[nearest_idx]
                    v_val = v_flat[nearest_idx]
                    
                    vel_mag = np.sqrt(u_val**2 + v_val**2)
                    if vel_mag > 0.001:
                        # Move arrow
                        new_x = x + u_val * speed_factor
                        new_y = y + v_val * speed_factor
                        
                        if inside_L(new_x, new_y):
                            arrows_x[i] = new_x
                            arrows_y[i] = new_y
                            arrows_u[i] = u_val / vel_mag
                            arrows_v[i] = v_val / vel_mag
                            arrow_colors[i] = vel_mag
                            continue
            
            # Reset arrow if it's outside or stopped
            attempts = 0
            while attempts < 10:
                reset_x = np.random.uniform(0, L_down)
                reset_y = np.random.uniform(0, H_left)
                if inside_L(reset_x, reset_y):
                    arrows_x[i] = reset_x
                    arrows_y[i] = reset_y
                    arrows_u[i] = 0.0
                    arrows_v[i] = 0.0
                    arrow_colors[i] = 0.0
                    break
                attempts += 1
        
        # Update quiver with same-sized arrays
        quiver.set_offsets(np.c_[arrows_x, arrows_y])
        quiver.set_UVC(np.array(arrows_u), np.array(arrows_v))
        quiver.set_array(np.array(arrow_colors))
        
        return quiver,

    # Create and save animation
    frames = 50  # Reduced frames for stability
    print(f"Creating animation with {frames} frames...")
    try:
        anim = animation.FuncAnimation(fig, update_arrows, frames=frames, interval=200, blit=True)

        plt.title('Carbopol Flow - Animated Streamlines (Improved 3-Stage)', fontsize=14, fontweight='bold')
        plt.xlabel('x (m)', fontsize=12)
        plt.ylabel('y (m)', fontsize=12)
        plt.xlim(x_min_bound, x_max_bound)
        plt.ylim(y_min_bound, y_max_bound)
        plt.axis('equal')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        print("Saving animated streamlines GIF...")
        anim.save('plots/streamlines_animated.gif', writer='pillow', fps=5, dpi=100)
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Animation failed: {e}")
        print("Skipping animation - static plots were created successfully")
    
    plt.close()

    print("Visualizations generation complete.")