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

    # Create plots directory if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')

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


    # Animated streamlines
    print("Creating animated streamlines...")
    # Make sure grid velocities are available
    if np.isnan(u_grid).all() or np.isnan(v_grid).all():
        print("Skipping animation: Velocity grid is empty or all NaN.")
    else:
        # Create a higher resolution grid specifically for animation
        nx_anim, ny_anim = 150, 100  # Higher resolution for better interpolation
        x_grid_anim = np.linspace(x_min_bound, x_max_bound, nx_anim)
        y_grid_anim = np.linspace(y_min_bound, y_max_bound, ny_anim)
        X_anim, Y_anim = np.meshgrid(x_grid_anim, y_grid_anim)
        grid_points_anim = np.column_stack((X_anim.flatten(), Y_anim.flatten()))
        
        # Identify points inside domain
        inside_mask_anim = np.array([inside_L(px, py) for px, py in grid_points_anim])
        grid_inside_anim = grid_points_anim[inside_mask_anim]
        
        # Get velocity field at higher resolution
        u_grid_anim = np.full(X_anim.shape, np.nan)
        v_grid_anim = np.full(X_anim.shape, np.nan)
        
        if len(grid_inside_anim) > 0:
            with torch.no_grad():
                # Process in batches to avoid memory issues
                batch_size = 10000
                for i in range(0, len(grid_inside_anim), batch_size):
                    batch = grid_inside_anim[i:i+batch_size]
                    xy_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
                    outputs = model(xy_tensor).cpu().numpy()
                    
                    # Find corresponding indices in the grid
                    batch_indices = np.where(inside_mask_anim)[0][i:i+batch_size]
                    rows, cols = np.unravel_index(batch_indices, X_anim.shape)
                    
                    u_grid_anim[rows, cols] = outputs[:, 0]
                    v_grid_anim[rows, cols] = outputs[:, 1]
        
        print("Creating improved flow animation...")

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_title('Animated Flow Field')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.axis('equal')
        ax.set_xlim(x_min_bound, x_max_bound)
        ax.set_ylim(y_min_bound, y_max_bound)
        ax.grid(True)

        # Plot domain boundaries
        for segment in wall_segments:
            (x1, y1), (x2, y2) = segment
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
        ax.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=30, label='Inlet', zorder=5)
        ax.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=30, label='Outlet', zorder=5)
        ax.legend()

        # FIXED: Determine predominant flow directions throughout the domain
        sample_points_x = np.linspace(x_min_bound+0.1, x_max_bound-0.1, 10)
        sample_points_y = np.linspace(y_min_bound+0.1, y_max_bound-0.1, 10)
        XX, YY = np.meshgrid(sample_points_x, sample_points_y)
        sample_points = np.column_stack((XX.flatten(), YY.flatten()))
        
        # Filter to points inside the domain
        sample_inside = np.array([inside_L(px, py) for px, py in sample_points])
        sample_points = sample_points[sample_inside]
        
        # Get velocities at sample points
        with torch.no_grad():
            sample_tensor = torch.tensor(sample_points, dtype=torch.float32, device=device)
            sample_outputs = model(sample_tensor).cpu().numpy()
        
        # Calculate average flow magnitude for scaling
        flow_magnitudes = np.sqrt(sample_outputs[:, 0]**2 + sample_outputs[:, 1]**2)
        avg_flow_magnitude = np.mean(flow_magnitudes)
        max_flow_magnitude = np.max(flow_magnitudes)
        print(f"Average flow magnitude: {avg_flow_magnitude:.4f}")
        print(f"Maximum flow magnitude: {max_flow_magnitude:.4f}")
        
        # FIXED: Scale time step based on domain size and flow magnitude
        domain_width = x_max_bound - x_min_bound
        dt_scale = domain_width / (50 * max_flow_magnitude)  # Aim to cross domain in ~50 steps
        dt = min(max(dt_scale, 0.01), 0.1)  # Limit between 0.01 and 0.1
        print(f"Using time step: {dt:.4f}")

        # Initialize more particles for better visualization
        particle_count = 200  # Increased particle count
        
        # FIXED: Better particle initialization strategy
        if len(inlet_points) > 0:
            # Initialize particles directly at inlet with small offsets
            inlet_indices = np.random.choice(len(inlet_points), particle_count)
            px = inlet_points[inlet_indices, 0].copy()
            py = inlet_points[inlet_indices, 1].copy()
            
            # Add very small jitter to prevent stacking
            px += np.random.uniform(-0.02, 0.02, particle_count)
            py += np.random.uniform(-0.02, 0.02, particle_count)
            
            # FIXED: Determine inlet orientation and push particles into domain
            # Test a sample of inlet points to get flow direction
            inlet_sample = np.linspace(0, len(inlet_points)-1, min(10, len(inlet_points)), dtype=int)
            inlet_test_points = inlet_points[inlet_sample]
            
            with torch.no_grad():
                inlet_tensor = torch.tensor(inlet_test_points, dtype=torch.float32, device=device)
                inlet_outputs = model(inlet_tensor).cpu().numpy()
            
            # Get average flow direction at inlet
            avg_u_inlet = np.mean(inlet_outputs[:, 0])
            avg_v_inlet = np.mean(inlet_outputs[:, 1])
            
            # Push particles slightly in flow direction
            push_scale = 0.05
            px += avg_u_inlet * push_scale / (abs(avg_u_inlet) + 1e-6)  # Avoid division by zero
            py += avg_v_inlet * push_scale / (abs(avg_v_inlet) + 1e-6)
        else:
            # Fallback to random initialization throughout the domain
            px, py = generate_points_inside(particle_count, x_min_bound, x_max_bound, y_min_bound, y_max_bound, inside_L)

        # Visualize with scatter plot for particles and lines for trails
        scatter = ax.scatter(px, py, s=20, c='blue', alpha=0.8, zorder=4)
        
        # FIXED: Store particle history for better trails
        history_length = 10  # Number of previous positions to keep
        position_history = [[(x, y)] for x, y in zip(px, py)]
        
        # Create line collection for trails
        from matplotlib.collections import LineCollection
        lines = [np.array([[x, y]]) for x, y in zip(px, py)]
        lc = LineCollection(lines, colors='blue', linewidths=1.5, alpha=0.5)
        ax.add_collection(lc)
        
        # Track particle ages
        particle_ages = np.zeros(particle_count)
        # FIXED: Allow particles to live longer to reach the outlet
        max_age = 500  # Significantly increased to allow particles to reach outlet
        
        # Keep track of particles that reached the outlet
        reached_outlet = np.zeros(particle_count, dtype=bool)
        
        # FIXED: Look-up based interpolation instead of function calls for performance
        # Create grid-based velocity lookup
        from scipy.interpolate import NearestNDInterpolator
        
        # Extract valid points and velocities from grid
        valid_mask = ~np.isnan(u_grid_anim)
        points = np.vstack((Y_anim[valid_mask], X_anim[valid_mask])).T  # (y, x) order
        u_values = u_grid_anim[valid_mask]
        v_values = v_grid_anim[valid_mask]
        
        # Create nearest neighbor interpolator for fast lookup
        u_lookup = NearestNDInterpolator(points, u_values)
        v_lookup = NearestNDInterpolator(points, v_values)
        
        # Also create a point-in-domain checker using the same interpolator approach
        # (much faster than calling inside_L repeatedly)
        domain_mask = np.zeros_like(X_anim)
        inside_indices = np.where(valid_mask)
        domain_mask[inside_indices] = 1
        
        # Make a flat version of coordinates and mask for fast lookup
        flat_Y = Y_anim.flatten()
        flat_X = X_anim.flatten()
        flat_mask = domain_mask.flatten()
        
        def fast_inside_check(x, y):
            """Fast domain check using nearest neighbor instead of exact function"""
            # Find closest grid point
            y_idx = np.argmin(np.abs(y_grid_anim - y))
            x_idx = np.argmin(np.abs(x_grid_anim - x))
            
            # If we're more than one cell away from any domain point, we're outside
            if y_idx >= len(y_grid_anim) or x_idx >= len(x_grid_anim):
                return False
                
            # Return True if the nearest grid point is inside
            return domain_mask[y_idx, x_idx] > 0
        
        # Check outlet proximity for resetting particles
        def near_outlet(x, y, threshold=0.2):
            """Check if a point is close to the outlet"""
            if len(outlet_points) == 0:
                return False
                
            # Calculate distances to all outlet points
            dists = np.sqrt(np.sum((outlet_points - np.array([x, y]))**2, axis=1))
            return np.min(dists) < threshold
        
        def update(frame):
            nonlocal reached_outlet  # Track particles that reached the outlet
            
            # Get current positions for lookup
            current_points = np.vstack((anim_state.py, anim_state.px)).T
            
            # Get velocities using fast lookup
            vx = u_lookup(current_points)
            vy = v_lookup(current_points)
            
            # Calculate velocity magnitude for coloring
            vel_mag = np.sqrt(vx**2 + vy**2)
            
            # FIXED: Calculate adaptive time step for each particle based on velocity
            # This prevents fast particles from skipping through the domain
            adaptive_dt = np.minimum(dt, 0.05 / (vel_mag + 1e-6))
            adaptive_dt = np.minimum(adaptive_dt, 0.1)  # Cap at 0.1 max
            
            # Calculate next positions
            px_next = anim_state.px + vx * adaptive_dt
            py_next = anim_state.py + vy * adaptive_dt
            
            # Check which particles are still inside the domain
            is_inside = np.array([fast_inside_check(x, y) for x, y in zip(px_next, py_next)])
            
            # Increment particle ages
            anim_state.particle_ages += 1
            
            # Check if any particles reached the outlet
            for i in range(particle_count):
                if near_outlet(px_next[i], py_next[i]):
                    reached_outlet[i] = True
            
            # FIXED: Reset particles that: 1) left domain, 2) are too old, or 3) reached outlet
            reset_mask = (~is_inside) | (anim_state.particle_ages > max_age) | reached_outlet
            num_to_reset = np.sum(reset_mask)
            
            if num_to_reset > 0 and len(inlet_points) > 0:
                # Reset particles to inlet
                inlet_indices = np.random.choice(len(inlet_points), num_to_reset)
                px_next[reset_mask] = inlet_points[inlet_indices, 0]
                py_next[reset_mask] = inlet_points[inlet_indices, 1]
                
                # Add small jitter and push in flow direction
                px_next[reset_mask] += np.random.uniform(-0.02, 0.02, num_to_reset)
                py_next[reset_mask] += np.random.uniform(-0.02, 0.02, num_to_reset)
                
                # Push in flow direction (using the pre-calculated inlet flow direction)
                px_next[reset_mask] += avg_u_inlet * push_scale / (abs(avg_u_inlet) + 1e-6)
                py_next[reset_mask] += avg_v_inlet * push_scale / (abs(avg_v_inlet) + 1e-6)
                
                # Reset particle ages and reached_outlet status
                anim_state.particle_ages[reset_mask] = 0
                reached_outlet[reset_mask] = False
                
                # Reset history
                for i in np.where(reset_mask)[0]:
                    position_history[i] = [(px_next[i], py_next[i])]
            
            # Update positions
            anim_state.px = px_next
            anim_state.py = py_next
            
            # Update position history for each particle
            for i in range(particle_count):
                # Add current position to history
                position_history[i].append((anim_state.px[i], anim_state.py[i]))
                # Limit history length
                if len(position_history[i]) > history_length:
                    position_history[i] = position_history[i][-history_length:]
            
            # Update the visualization
            scatter.set_offsets(np.column_stack([anim_state.px, anim_state.py]))
            
            # Update lines for trails
            lines = [np.array(history) for history in position_history]
            lc.set_segments(lines)
            
            # Color particles based on velocity
            scatter.set_array(vel_mag)
            
            return scatter, lc

        # Create animation state to hold variables
        class AnimationState:
            def __init__(self):
                self.px = px
                self.py = py
                self.particle_ages = particle_ages

        anim_state = AnimationState()

        # Create animation with more frames
        ani = animation.FuncAnimation(fig, update, frames=500, interval=20, blit=True)

        # Save animation
        try:
            writer = animation.PillowWriter(fps=30)
            ani.save('plots/animated_flow.gif', writer=writer)
            print("Animation saved to plots/animated_flow.gif")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Try installing required libraries with: pip install pillow")

        plt.close(fig)
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