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

    # --- Streamlines setup ---
    print("Creating streamline plot...")
    # Define bounding box slightly larger than expected domain for interpolation buffer
    x_min_bound = -0.01
    x_max_bound = L_down + 0.01
    y_min_bound = -0.01
    y_max_bound = H_left + 0.01
    nx, ny = 60, 40 # Resolution for grid
    x_grid_coords = np.linspace(x_min_bound, x_max_bound, nx)
    y_grid_coords = np.linspace(y_min_bound, y_max_bound, ny)
    X, Y = np.meshgrid(x_grid_coords, y_grid_coords)
    grid_points_flat = np.column_stack((X.flatten(), Y.flatten()))

    # Get predictions ONLY for points inside the L-shape
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
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
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
         plt.colorbar(streamlines.lines, label='Velocity magnitude (m/s)')
    else:
        print("Warning: No streamlines generated for colored plot.")
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=30, label='Inlet', zorder=5)
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=30, label='Outlet', zorder=5)
    plt.title('Flow Field - Streamlines Colored by Velocity Magnitude')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
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
            
        def is_near_outlet(self, outlet_points, threshold=0.02):  # Reduced threshold for smaller domain
            # Check if particle is near outlet
            for pt in outlet_points:
                dist = np.sqrt((self.x - pt[0])**2 + (self.y - pt[1])**2)
                if dist < threshold:
                    return True
            return False
            
        def is_stuck(self):
            # Check if particle hasn't moved much
            dist_moved = np.sqrt((self.x - self.prev_x)**2 + (self.y - self.prev_y)**2)
            is_stuck = dist_moved < 0.0005  # Very small movement threshold adjusted for smaller domain
            
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
    # Updated guidance points to match new dimensions
    guidance_points = [
        # Entry points (near inlet)
        (0.0, H_left, 0.02),        # Top of the vertical section (inlet)
        (0.0, H_left*0.75, 0.02),   # Upper part of vertical section
        (0.0, H_left*0.5, 0.02),    # Middle of vertical section
        (0.0, H_left*0.25, 0.02),   # Lower part of vertical section
        (0.0, 0.02, 0.02),          # Near the L-bend
        (L_up*0.25, 0.02, 0.02),    # Just past the L-bend
        (L_up*0.5, 0.02, 0.02),     # Middle of horizontal section
        (L_up*0.75, 0.02, 0.02),    # Further along horizontal to L-corner
        (L_up, H_right*0.5, 0.02),  # Vertical section after L-corner
        (L_down*0.75, H_right*0.5, 0.02),  # Approaching outlet
        (L_down, H_right*0.5, 0.02)  # At outlet
    ]
    
    # Get inlet points to seed particles around
    inlet_center_x = np.mean(inlet_points[:, 0])
    inlet_center_y = np.mean(inlet_points[:, 1])
    
    # Create particles with staggered release times
    particles = []
    for i in range(num_particles):
        # Decide whether to create active particle or waiting particle
        if i < num_particles // 3:  # Start with 1/3 of particles active
            rx = inlet_center_x + np.random.normal(0, 0.01)  # Smaller variance for smaller domain
            ry = inlet_center_y + np.random.normal(0, 0.01)
            # Ensure particle starts inside domain
            while not inside_L(rx, ry):
                rx = inlet_center_x + np.random.normal(0, 0.01)
                ry = inlet_center_y + np.random.normal(0, 0.01)
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
        if vel_mag < 0.03:  # Adjusted threshold for smaller domain
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
                u_guided = u_guided / mag * 0.01  # Adjusted velocity scale for smaller domain
                v_guided = v_guided / mag * 0.01
            
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
    plt.colorbar(vel_mag_domain_scattered, label='Velocity magnitude (m/s)')
    
    # Add a few streamlines for context
    streamlines = ax.streamplot(X, Y, u_grid, v_grid, density=0.8, color='gray', 
                              linewidth=0.5, arrowsize=0.8)
    
    # Adjust alpha for streamlines (need to access the created artists)
    if hasattr(streamlines, 'lines'):
        streamlines.lines.set_alpha(0.3)
    if hasattr(streamlines, 'arrows'):
        # arrows is a PatchCollection, not an iterable of individual arrows
        streamlines.arrows.set_alpha(0.3)
    
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
        
        dt = 0.03  # Time step
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
                    p.x = inlet_center_x + np.random.normal(0, 0.01)  # Smaller variance for smaller domain
                    p.y = inlet_center_y + np.random.normal(0, 0.01)
                    # Ensure it's inside the domain
                    while not inside_L(p.x, p.y):
                        p.x = inlet_center_x + np.random.normal(0, 0.01)
                        p.y = inlet_center_y + np.random.normal(0, 0.01)
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
    frames = 300  # Adjusted for smaller domain
    anim = animation.FuncAnimation(
        fig, update, frames=frames, interval=30, blit=True)
    
    # Set plot properties
    plt.title('Carbopol Flow - Full Journey Animation')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim(x_min_bound, x_max_bound)
    plt.ylim(y_min_bound, y_max_bound)
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    
    # Save animation with higher frame rate for smoother movement
    print(f"Saving animation with {frames} frames...")
    anim.save('plots/flow_full_journey.gif', writer='pillow', fps=20)
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
    ax.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=50, label='Inlet', zorder=5)
    ax.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=50, label='Outlet', zorder=5)

    # Create a static background streamplot
    streamplot = ax.streamplot(X, Y, u_grid, v_grid, density=1.5, 
                            color='lightgray', linewidth=0.7, arrowsize=0.8)

    # Generate arrows that will move along the streamlines
    num_arrows = 150  # More arrows for better coverage

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

    # Create initial quiver plot for the arrows
    quiver = ax.quiver(arrows_x, arrows_y, arrows_u, arrows_v, arrow_colors,
                    scale=30, width=0.003, cmap='viridis', pivot='mid',
                    zorder=10)

    plt.colorbar(quiver, label='Velocity Magnitude (m/s)')

    # Function to update arrow positions for animation
    def update_arrows(frame):
        # Speed factor - adjust for faster/slower movement
        speed_factor = 0.01  # Reduced for smaller domain
        
        # New positions for arrows
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
        
        # Update the arrow positions and directions
        arrows_x[:] = new_x
        arrows_y[:] = new_y
        arrows_u[:] = new_u
        arrows_v[:] = new_v
        
        # Update the quiver plot
        quiver.set_offsets(np.c_[new_x, new_y])
        quiver.set_UVC(np.array(new_u), np.array(new_v))
        
        # Update colors if we have them
        if hasattr(quiver, 'set_array') and len(new_colors) > 0:
            quiver.set_array(np.array(new_colors))
        
        return quiver,

    # Create and save the animation
    frames = 100  # Number of frames
    print(f"Creating animation with {frames} frames...")
    anim = animation.FuncAnimation(fig, update_arrows, frames=frames, interval=50, blit=True)

    # Set plot properties
    plt.title('Carbopol Flow - Animated Streamlines')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.xlim(x_min_bound, x_max_bound)
    plt.ylim(y_min_bound, y_max_bound)
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    # Save animation
    print("Saving animated streamlines GIF...")
    anim.save('plots/streamlines_animated.gif', writer='pillow', fps=15)
    plt.close()
    
    print("Visualizations generation complete.")