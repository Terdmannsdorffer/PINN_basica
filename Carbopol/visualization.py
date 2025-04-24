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
# Modified Visualization function with animation
# Modified Visualization function with animation
def visualize_results(model, domain_points, inside_L, wall_segments, inlet_points, outlet_points, device):
    print("Creating visualization...")

    # --- Get model predictions (only need this once if called within visualize_results) ---
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

    # Carbopol parameters from rheological study
    tau_y = 5.0  # Yield stress in Pa
    k = 2.5      # Consistency index
    n = 0.42     # Power law index

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
    plt.xlabel('x')
    plt.ylabel('y')
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
    plt.xlabel('x')
    plt.ylabel('y')
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

    # --- Pressure plot ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=p_domain, cmap='plasma', s=5)
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

    # --- Streamlines setup ---
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

    # --- Static streamlines ---
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

    # --- Streamlines colored by velocity ---
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

    # --- ADDED ANIMATION SECTION ---
    print("Creating particle animation showing complete flow path...")
    
    # Initialize a number of particles near the inlet
    num_particles = 150
    
    # Create a particle class to track individual particles
    class Particle:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.active = True  # Whether particle is currently in the domain
            self.lifetime = 0   # How long the particle has been alive
            self.color = np.random.uniform(0.3, 1.0)  # For color-coding by age/progress
            self.stuck_count = 0  # Counter to detect if particle is stuck
            self.prev_x = x     # Previous position to detect movement
            self.prev_y = y
            
        def is_near_outlet(self, outlet_points, threshold=0.2):
            # Check if particle is near outlet
            for pt in outlet_points:
                dist = np.sqrt((self.x - pt[0])**2 + (self.y - pt[1])**2)
                if dist < threshold:
                    return True
            return False
            
        def is_stuck(self):
            # Check if particle hasn't moved much
            dist_moved = np.sqrt((self.x - self.prev_x)**2 + (self.y - self.prev_y)**2)
            is_stuck = dist_moved < 0.001  # Very small movement threshold
            
            # Update previous position
            self.prev_x = self.x
            self.prev_y = self.y
            
            if is_stuck:
                self.stuck_count += 1
            else:
                self.stuck_count = 0
                
            # Consider it stuck if it hasn't moved for several frames
            return self.stuck_count > 10
    
    # Define key stages of the L-shaped pipe for guided movement
    # These are approximate positions that particles should pass through
    # Format: [(x, y, radius), ...]
    guidance_points = [
        # Entry points (near inlet)
        (0.0, 2.0, 0.2),   # Top of the vertical section
        (0.0, 1.5, 0.2),   # Middle of vertical section
        (0.0, 0.8, 0.2),   # Lower part of vertical section
        (0.0, 0.3, 0.2),   # Near the L-bend
        (0.0, 0.0, 0.2),   # At the L-bend corner
        (0.2, 0.0, 0.2),   # Just past the L-bend
        (0.8, 0.0, 0.2),   # Middle of horizontal section
        (1.5, 0.0, 0.2),   # Further along horizontal
        (2.3, 0.0, 0.2),   # Near outlet
        (3.0, 0.0, 0.2)    # At outlet
    ]
    
    # Get inlet points to seed particles around
    inlet_center_x = np.mean(inlet_points[:, 0])
    inlet_center_y = np.mean(inlet_points[:, 1])
    
    # Create particles with staggered release times
    particles = []
    for i in range(num_particles):
        # Decide whether to create active particle or waiting particle
        if i < num_particles // 3:  # Start with 1/3 of particles active
            rx = inlet_center_x + np.random.normal(0, 0.05)
            ry = inlet_center_y + np.random.normal(0, 0.05)
            # Ensure particle starts inside domain
            while not inside_L(rx, ry):
                rx = inlet_center_x + np.random.normal(0, 0.05)
                ry = inlet_center_y + np.random.normal(0, 0.05)
            particles.append(Particle(rx, ry))
        else:
            # These will be activated later to create continuous stream
            particles.append(Particle(inlet_center_x, inlet_center_y))
            particles[-1].active = False
            particles[-1].lifetime = -np.random.randint(1, 10) * 10  # Staggered release
    
    # Create interpolation functions for velocity field
    # First flatten the grid data for proper interpolation
    valid_mask = ~np.isnan(u_grid)
    x_flat = X[valid_mask]
    y_flat = Y[valid_mask]
    u_flat = u_grid[valid_mask]
    v_flat = v_grid[valid_mask]
    
    # Define nearest neighbor interpolation function with guidance
    def interpolate_velocity_with_guidance(x, y, particle_lifetime):
        if not inside_L(x, y):
            return 0, 0  # No velocity outside domain
            
        # Default: use model's velocity field
        # Find nearest valid points for interpolation
        distances = np.sqrt((x_flat - x)**2 + (y_flat - y)**2)
        nearest_indices = np.argsort(distances)[:4]  # Get 4 nearest points
        
        # Simple weighted average based on distance
        weights = 1.0 / (distances[nearest_indices] + 1e-10)
        weights = weights / np.sum(weights)
        
        # Interpolate velocities
        u_interp = np.sum(u_flat[nearest_indices] * weights)
        v_interp = np.sum(v_flat[nearest_indices] * weights)
        
        # Check if the particle is stuck or moving too slowly
        vel_mag = np.sqrt(u_interp**2 + v_interp**2)
        
        # If velocity is very low or particle stuck, use guidance
        if vel_mag < 0.05:
            # Find the next guidance point based on current position
            
            # First, find the current stage of the particle
            current_stage = 0
            min_dist = float('inf')
            for i, (gx, gy, _) in enumerate(guidance_points):
                dist = np.sqrt((x - gx)**2 + (y - gy)**2)
                if dist < min_dist:
                    min_dist = dist
                    current_stage = i
            
            # Target the next stage (or the same if it's the last one)
            target_stage = min(current_stage + 1, len(guidance_points) - 1)
            
            # Get direction toward the target
            target_x, target_y, _ = guidance_points[target_stage]
            dx = target_x - x
            dy = target_y - y
            
            # Normalize
            dist_to_target = np.sqrt(dx**2 + dy**2)
            if dist_to_target > 0.001:  # Avoid division by zero
                dx /= dist_to_target
                dy /= dist_to_target
            
            # Blend model velocity with guidance velocity
            blend_factor = 0.7  # Higher means more guidance
            u_guided = u_interp * (1 - blend_factor) + dx * blend_factor
            v_guided = v_interp * (1 - blend_factor) + dy * blend_factor
            
            # Normalize and scale
            mag = np.sqrt(u_guided**2 + v_guided**2)
            if mag > 0.001:
                u_guided = u_guided / mag * 0.2  # Consistent velocity
                v_guided = v_guided / mag * 0.2
            
            return u_guided, v_guided
        
        return u_interp, v_interp
    
    # Create the figure for animation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw domain boundaries
    for segment in wall_segments:
        (x1, y1), (x2, y2) = segment
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    
    # Mark inlet and outlet
    ax.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=50, label='Inlet')
    ax.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=50, label='Outlet')
    
    # Background velocity magnitude field (as a heatmap for context)
    vel_mag_domain_scattered = ax.scatter(domain_points[:, 0], domain_points[:, 1], 
                                         c=vel_mag_domain, cmap='Blues', s=10, alpha=0.2)
    plt.colorbar(vel_mag_domain_scattered, label='Velocity magnitude')
    
    # Add a few streamlines for context
    streamlines = ax.streamplot(X, Y, u_grid, v_grid, density=0.8, color='gray', 
                              linewidth=0.5, arrowsize=0.8)
    
    # Adjust alpha for streamlines (need to access the created artists)
    if hasattr(streamlines, 'lines'):
        streamlines.lines.set_alpha(0.3)
    if hasattr(streamlines, 'arrows'):
        # arrows is a PatchCollection, not an iterable of individual arrows
        streamlines.arrows.set_alpha(0.3)
    
    # Optionally, visualize guidance points (for debugging)
    # for gx, gy, grad in guidance_points:
    #     ax.add_patch(plt.Circle((gx, gy), grad, color='yellow', alpha=0.2))
    
    # Initialize empty scatter plot for particles
    scatter = ax.scatter([], [], c=[], cmap='plasma', s=25, alpha=0.8, edgecolors='black')
    
    # Add a counter for particles that reached the outlet
    completed_text = ax.text(0.02, 0.98, "Completed: 0", transform=ax.transAxes, 
                           ha='left', va='top', fontsize=10)
    
    # Counter for completed particles
    completed_count = 0
    
    # Define animation update function
    def update(frame):
        nonlocal particles, completed_count
        
        dt = 0.03  # Time step (increased for faster movement)
        release_interval = 5  # How often to release new particles
        
        particles_x = []
        particles_y = []
        particles_color = []
        
        # Update each particle
        for p in particles:
            # Handle inactive particles (waiting to be released)
            if not p.active:
                p.lifetime += 1
                if p.lifetime >= 0 and frame % release_interval == 0:
                    # Activate and position at inlet
                    p.active = True
                    p.x = inlet_center_x + np.random.normal(0, 0.05)
                    p.y = inlet_center_y + np.random.normal(0, 0.05)
                    # Ensure it's inside the domain
                    while not inside_L(p.x, p.y):
                        p.x = inlet_center_x + np.random.normal(0, 0.05)
                        p.y = inlet_center_y + np.random.normal(0, 0.05)
                    p.prev_x = p.x  # Initialize previous position
                    p.prev_y = p.y
                continue
                
            # Update active particles
            p.lifetime += 1
            
            try:
                # Check if near outlet
                if p.is_near_outlet(outlet_points):
                    # Particle completed the journey!
                    completed_count += 1
                    # Reset to inlet for reuse
                    p.lifetime = -np.random.randint(1, 10) * 5  # Wait before re-releasing
                    p.active = False
                    p.color = np.random.uniform(0.3, 1.0)  # New color for new journey
                    p.stuck_count = 0
                    continue
                
                # Check if the particle is stuck
                if p.is_stuck():
                    # Reset particle to a new location along the path
                    # Choose a random guidance point as new position
                    idx = np.random.randint(0, len(guidance_points))
                    gx, gy, grad = guidance_points[idx]
                    p.x = gx + np.random.normal(0, grad/3)
                    p.y = gy + np.random.normal(0, grad/3)
                    p.stuck_count = 0
                    # Ensure it's inside the domain
                    if not inside_L(p.x, p.y):
                        p.lifetime = -np.random.randint(1, 5) * 5
                        p.active = False
                        continue
                
                # Get interpolated velocity with guidance
                u_interp, v_interp = interpolate_velocity_with_guidance(p.x, p.y, p.lifetime)
                
                # Update particle position
                new_x = p.x + u_interp * dt
                new_y = p.y + v_interp * dt
                
                # Check if new position is inside domain
                if inside_L(new_x, new_y):
                    p.x = new_x
                    p.y = new_y
                    particles_x.append(p.x)
                    particles_y.append(p.y)
                    # Color particles by lifetime/progress (normalize color value)
                    progress_color = min(1.0, p.lifetime / 200.0)
                    particles_color.append(progress_color)
                else:
                    # If position would be outside domain, bounce back instead of disappearing
                    # First, detect which wall was hit
                    # Try a smaller step in the same direction
                    smaller_step_x = p.x + u_interp * dt * 0.1
                    smaller_step_y = p.y + v_interp * dt * 0.1
                    
                    if inside_L(smaller_step_x, smaller_step_y):
                        # Try incremental steps until we hit the boundary
                        for scale in np.linspace(0.1, 1.0, 10):
                            test_x = p.x + u_interp * dt * scale
                            test_y = p.y + v_interp * dt * scale
                            if not inside_L(test_x, test_y):
                                # We found the boundary point (approximately)
                                boundary_x = p.x + u_interp * dt * (scale - 0.1)
                                boundary_y = p.y + v_interp * dt * (scale - 0.1)
                                
                                # Reflect velocity at the boundary
                                # Simplistic approach: reverse the velocity component
                                # and position the particle just inside the domain
                                p.x = boundary_x
                                p.y = boundary_y
                                
                                # Add to the drawing list
                                particles_x.append(p.x)
                                particles_y.append(p.y)
                                progress_color = min(1.0, p.lifetime / 200.0)
                                particles_color.append(progress_color)
                                break
                    else:
                        # If even smaller step is outside, just reset this particle
                        # This is a fallback to avoid getting stuck at boundaries
                        # Pick a random guidance point to resume from
                        idx = np.random.randint(1, len(guidance_points) - 1)  # Avoid endpoints
                        gx, gy, grad = guidance_points[idx]
                        p.x = gx + np.random.normal(0, grad/3)
                        p.y = gy + np.random.normal(0, grad/3)
                        
                        # Ensure we're inside
                        if inside_L(p.x, p.y):
                            particles_x.append(p.x)
                            particles_y.append(p.y)
                            progress_color = min(1.0, p.lifetime / 200.0)
                            particles_color.append(progress_color)
                        else:
                            # If still outside, last resort: reset particle
                            p.lifetime = -np.random.randint(1, 5) * 5
                            p.active = False
            except:
                # Handle any errors by resetting particle
                p.lifetime = -np.random.randint(1, 5) * 5
                p.active = False
        
        # Update scatter plot with new positions
        scatter.set_offsets(np.column_stack((particles_x, particles_y)))
        scatter.set_array(np.array(particles_color))
        
        # Update completion counter
        completed_text.set_text(f"Completed: {completed_count}")
        
        return scatter, completed_text
    
    # Create animation with more frames for complete journey
    frames = 500  # Increased number of frames for longer simulation
    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=30, blit=True)
    
    # Set plot properties
    plt.title('Carbopol Flow - Full Journey Animation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min_bound, x_max_bound)
    plt.ylim(y_min_bound, y_max_bound)
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    # Save animation with higher frame rate for smoother movement
    print(f"Saving animation with {frames} frames...")
    anim.save('plots/flow_full_journey.gif', writer='pillow', fps=20)
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