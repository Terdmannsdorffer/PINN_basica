import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import os
from domain_utils import inside_L, W, L_vertical, L_horizontal

def visualize_basic_results(model, domain_points, 
                      inlet_points, outlet_points, wall_points, wall_normals,
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Create basic visualizations of the flow field results.
    
    Args:
        model: Trained PINN model
        domain_points: Points in the domain
        inlet_points, outlet_points, wall_points: Boundary points
        wall_normals: Normal vectors for wall points
        device: Computation device (CPU/GPU)
    """
    print("Creating basic visualizations...")
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        xy_tensor = torch.tensor(domain_points, dtype=torch.float32, device=device)
        outputs = model(xy_tensor).cpu().numpy()
    
    u = outputs[:, 0]
    v = outputs[:, 1]
    p = outputs[:, 2]
    vel_mag = np.sqrt(u**2 + v**2)
    
    # Plot 1: Velocity magnitude
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(domain_points[:, 0], domain_points[:, 1], c=vel_mag, cmap='viridis')
    plt.colorbar(scatter, label='Velocity magnitude')
    plt.title('Flow Field - Velocity Magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig('plots/flow_field.png')
    plt.close()
    
    # Plot 2: Streamlines
    # Define grid boundaries based on L-shape domain
    x_min, x_max = -0.5, L_horizontal + 0.1
    y_min, y_max = -0.5, L_vertical + 0.1
    
    # Create a regular grid
    nx, ny = 40, 30  # Number of points
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Flatten the grid for model prediction
    grid_points = np.column_stack((X.flatten(), Y.flatten()))
    
    # Filter points to only include those inside the L-shaped domain
    inside_mask = np.array([inside_L(px, py) for px, py in grid_points])
    grid_inside = grid_points[inside_mask]
    
    # Get predictions for the grid points
    with torch.no_grad():
        xy_grid = torch.tensor(grid_inside, dtype=torch.float32, device=device)
        grid_outputs = model(xy_grid).cpu().numpy()
    
    # Extract u and v components
    u_grid = np.zeros(X.shape)
    v_grid = np.zeros(X.shape)
    p_grid = np.zeros(X.shape)
    
    # Assign predicted values to the correct positions in the grid
    inside_indices = np.where(inside_mask)[0]
    for i, idx in enumerate(inside_indices):
        row, col = np.unravel_index(idx, X.shape)
        u_grid[row, col] = grid_outputs[i, 0]
        v_grid[row, col] = grid_outputs[i, 1]
        p_grid[row, col] = grid_outputs[i, 2]
    
    # Create streamline plot
    plt.figure(figsize=(10, 8))
    
    # First plot the domain outline
    segments = [
        [(-W/2, -W/2), (-W/2, L_vertical)],
        [(-W/2, L_vertical), (W/2, L_vertical)],
        [(W/2, L_vertical), (W/2, W/2)],
        [(W/2, W/2), (L_horizontal, W/2)],
        [(L_horizontal, W/2), (L_horizontal, -W/2)],
        [(L_horizontal, -W/2), (-W/2, -W/2)]
    ]
    
    for segment in segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    
    # Plot streamlines
    streamlines = plt.streamplot(X, Y, u_grid, v_grid, 
                                density=1.5, 
                                color='blue',
                                linewidth=1,
                                arrowsize=1.5)
    
    # Highlight inlet and outlet
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=30, label='Inlet')
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=30, label='Outlet')
    
    plt.title('Flow Field - Streamlines')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/streamlines.png')
    plt.close()
    
    # Plot 3: Streamlines colored by velocity magnitude
    plt.figure(figsize=(12, 9))
    
    # Plot the domain outline
    for segment in segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    
    # Calculate velocity magnitude for coloring
    vel_mag_grid = np.sqrt(u_grid**2 + v_grid**2)
    
    # Create streamlines colored by velocity magnitude
    streamlines = plt.streamplot(X, Y, u_grid, v_grid, 
                                density=1.5,
                                color=vel_mag_grid,
                                cmap='viridis',
                                linewidth=1.5,
                                arrowsize=1.5)
    
    # Add colorbar
    cbar = plt.colorbar(streamlines.lines, label='Velocity magnitude')
    
    # Highlight inlet and outlet
    plt.scatter(inlet_points[:, 0], inlet_points[:, 1], color='green', s=30, label='Inlet')
    plt.scatter(outlet_points[:, 0], outlet_points[:, 1], color='red', s=30, label='Outlet')
    
    plt.title('Flow Field - Streamlines Colored by Velocity Magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/streamlines_colored.png')
    plt.close()
    
    print("Basic visualizations saved")

def visualize_enhanced_results(model, domain_points, 
                      inlet_points, outlet_points, wall_points, wall_normals,
                      num_vis_points=5000,
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                      model_name="PINN"):
    """
    Create enhanced visualizations of the flow field results with multiple plots.
    
    Args:
        model: Trained PINN model
        domain_points: Points in the domain for training
        inlet_points, outlet_points, wall_points: Boundary points
        wall_normals: Normal vectors for wall points
        num_vis_points: Number of points to use for visualization
        device: Computation device (CPU/GPU)
        model_name: Name of the model for plot titles
    """
    print("Creating enhanced visualizations...")
    
    # Physical parameters
    u_in = 0.5  # Inlet velocity
    
    # Generate a denser set of points for visualization
    vis_points = []
    points_collected = 0
    while points_collected < num_vis_points:
        x = np.random.uniform(-0.5, L_horizontal + 0.1)
        y = np.random.uniform(-0.5, L_vertical + 0.1)
        if inside_L(x, y):
            vis_points.append([x, y])
            points_collected += 1
    
    vis_points = np.array(vis_points)
    
    # Generate boundary points for drawing the outline
    segments = [
        [(-W/2, -W/2), (-W/2, L_vertical)],
        [(-W/2, L_vertical), (W/2, L_vertical)],
        [(W/2, L_vertical), (W/2, W/2)],
        [(W/2, W/2), (L_horizontal, W/2)],
        [(L_horizontal, W/2), (L_horizontal, -W/2)],
        [(L_horizontal, -W/2), (-W/2, -W/2)]
    ]
    
    boundary_points_vis = []
    for segment in segments:
        (x1, y1), (x2, y2) = segment
        for t in np.linspace(0, 1, 50):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            boundary_points_vis.append([x, y])
    
    boundary_points_vis = np.array(boundary_points_vis)
    
    # Predict using the trained model
    model.eval()
    with torch.no_grad():
        xy_tensor_vis = torch.tensor(vis_points, dtype=torch.float32, device=device)
        outputs = model(xy_tensor_vis).cpu().numpy()
    
    u_pred = outputs[:, 0]
    v_pred = outputs[:, 1]
    p_pred = outputs[:, 2]
    vel_mag = np.sqrt(u_pred**2 + v_pred**2)
    
    # Create a larger figure with 6 subplots (3x2 layout)
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(f"L-shaped Pipe Flow - {model_name} (Water - Newtonian Fluid, Gravity Included)", fontsize=18, y=0.99)
    
    # Common arguments for scatter plots and boundary plotting
    common_scatter_args = {'s': 3, 'alpha': 0.8}
    common_plot_args = {'color': 'black', 'linewidth': 1.5}
    
    # 1. Velocity magnitude
    ax1 = plt.subplot(3, 2, 1)
    norm_vel = Normalize(vmin=0, vmax=np.percentile(vel_mag, 99.5))  # Clip outliers for better color range
    sc = ax1.scatter(vis_points[:, 0], vis_points[:, 1], c=vel_mag, cmap='viridis', norm=norm_vel, **common_scatter_args)
    plt.colorbar(sc, ax=ax1, label='Velocity magnitude (m/s)', fraction=0.046, pad=0.04)
    ax1.set_title('Velocity Magnitude')
    ax1.plot(boundary_points_vis[:, 0], boundary_points_vis[:, 1], **common_plot_args)
    
    # 2. Pressure
    ax2 = plt.subplot(3, 2, 2)
    p_min, p_max = np.percentile(p_pred, [1, 99])  # Robust range for pressure
    norm_p = Normalize(vmin=p_min, vmax=p_max)
    sc = ax2.scatter(vis_points[:, 0], vis_points[:, 1], c=p_pred, cmap='coolwarm', norm=norm_p, **common_scatter_args)
    plt.colorbar(sc, ax=ax2, label='Pressure (Pa - relative)', fraction=0.046, pad=0.04)
    ax2.set_title('Pressure Field')
    ax2.plot(boundary_points_vis[:, 0], boundary_points_vis[:, 1], **common_plot_args)
    
    # 3. U-velocity (x-direction)
    ax3 = plt.subplot(3, 2, 3)
    u_min, u_max = np.percentile(u_pred, [1, 99])
    norm_u = Normalize(vmin=min(u_min, -0.1), vmax=max(u_max, 0.1))
    sc = ax3.scatter(vis_points[:, 0], vis_points[:, 1], c=u_pred, cmap='RdBu_r', norm=norm_u, **common_scatter_args)
    plt.colorbar(sc, ax=ax3, label='U-velocity (m/s)', fraction=0.046, pad=0.04)
    ax3.set_title('U-Velocity Component (x-dir)')
    ax3.plot(boundary_points_vis[:, 0], boundary_points_vis[:, 1], **common_plot_args)
    
    # 4. V-velocity (y-direction)
    ax4 = plt.subplot(3, 2, 4)
    v_min, v_max = np.percentile(v_pred, [1, 99])
    norm_v = Normalize(vmin=min(v_min, -0.1), vmax=max(v_max, 0.1))
    sc = ax4.scatter(vis_points[:, 0], vis_points[:, 1], c=v_pred, cmap='RdBu_r', norm=norm_v, **common_scatter_args)
    plt.colorbar(sc, ax=ax4, label='V-velocity (m/s)', fraction=0.046, pad=0.04)
    ax4.set_title('V-Velocity Component (y-dir)')
    ax4.plot(boundary_points_vis[:, 0], boundary_points_vis[:, 1], **common_plot_args)
    
    # Prepare grid for streamlines and quiver plots
    x_min_grid, x_max_grid = -W/2 - 0.1, L_horizontal + 0.1
    y_min_grid, y_max_grid = -W/2 - 0.1, L_vertical + 0.1
    
    # Create a regular grid
    grid_x_lin = np.linspace(x_min_grid, x_max_grid, 100)
    grid_y_lin = np.linspace(y_min_grid, y_max_grid, 100)
    grid_x, grid_y = np.meshgrid(grid_x_lin, grid_y_lin, indexing='xy')
    grid_shape = grid_x.shape
    
    # Interpolate velocity components onto the grid
    print("Interpolating results onto grid for vector/streamline plots...")
    
    # Create interpolation based on scattered data
    u_interp = griddata((vis_points[:, 0], vis_points[:, 1]), u_pred, (grid_x, grid_y), method='linear', fill_value=0)
    v_interp = griddata((vis_points[:, 0], vis_points[:, 1]), v_pred, (grid_x, grid_y), method='linear', fill_value=0)
    vel_mag_interp = np.sqrt(u_interp**2 + v_interp**2)
    
    # Create a mask for the L-shape on the grid
    mask = np.ones(grid_shape, dtype=bool)  # Mask starts as True (masked) everywhere
    print("Creating mask for visualization grid...")
    for i in range(grid_shape[0]):  # Iterate over rows (y-dimension)
        for j in range(grid_shape[1]):  # Iterate over columns (x-dimension)
            current_x = grid_x[i, j]
            current_y = grid_y[i, j]
            if inside_L(current_x, current_y):
                mask[i, j] = False  # Unmask points inside the pipe
    
    # Apply mask to velocity fields
    u_masked = np.ma.masked_array(u_interp, mask=mask)
    v_masked = np.ma.masked_array(v_interp, mask=mask)
    vel_mag_masked = np.ma.masked_array(vel_mag_interp, mask=mask)
    
    # Apply moderate smoothing for cleaner streamlines/vectors
    print("Applying smoothing for visualization...")
    u_smooth = gaussian_filter(u_masked.filled(0), sigma=0.8)
    v_smooth = gaussian_filter(v_masked.filled(0), sigma=0.8)
    u_smooth = np.ma.masked_array(u_smooth, mask=mask)
    v_smooth = np.ma.masked_array(v_smooth, mask=mask)
    
    # 5. Vector field (Quiver Plot)
    ax5 = plt.subplot(3, 2, 5)
    step = 6  # Show roughly every 6th vector
    x_q, y_q = grid_x[::step, ::step], grid_y[::step, ::step]
    u_q, v_q = u_smooth[::step, ::step], v_smooth[::step, ::step]
    mag_q = np.sqrt(u_q**2 + v_q**2)
    
    ax5.quiver(x_q, y_q, u_q, v_q, mag_q, cmap='viridis', norm=norm_vel,
            angles='xy', scale_units='xy', scale=8, headwidth=4, headlength=5, width=0.003)
    ax5.set_title('Velocity Field (Quiver Plot)')
    ax5.plot(boundary_points_vis[:, 0], boundary_points_vis[:, 1], **common_plot_args)
    
    # 6. Streamlines
    ax6 = plt.subplot(3, 2, 6)
    
    # Define better starting points for streamlines
    num_stream_starts = 15
    start_x_inlet = np.linspace(-W/2 * 0.9, W/2 * 0.9, num_stream_starts)
    start_y_inlet = np.ones_like(start_x_inlet) * (L_vertical - 0.05)
    
    # Add additional seed points for better coverage
    start_x_vertical = np.zeros(6)  # Points along vertical section
    start_y_vertical = np.linspace(0.1, L_vertical-0.1, 6)
    
    start_x_horizontal = np.linspace(0.1, L_horizontal-0.1, 6)  # Points along horizontal section
    start_y_horizontal = np.zeros(6)
    
    # Combine all starting points
    start_points = np.vstack([
        np.column_stack([start_x_inlet, start_y_inlet]),
        np.column_stack([start_x_vertical, start_y_vertical]),
        np.column_stack([start_x_horizontal, start_y_horizontal])
    ])
    
    # Calculate the speed for coloring streamlines
    speed = np.sqrt(u_smooth**2 + v_smooth**2)
    max_speed = np.percentile(speed.compressed(), 99.5) if speed.count() > 0 else 1.0
    
    # Create line width based on speed
    lw = 2 * speed / max_speed + 0.5
    lw = np.clip(lw, 0.5, 3.0)  # Clip line widths
    
    # Convert masked arrays to regular arrays with NaN values for masked areas
    speed_data = speed.filled(np.nan) 
    lw_data = lw.filled(0.5)
    
    try:
        # Streamplot for enhanced visualization
        streamplot = ax6.streamplot(
            grid_x_lin, grid_y_lin, 
            u_smooth.filled(0), v_smooth.filled(0),
            color=speed_data, cmap='viridis', norm=norm_vel,
            linewidth=lw_data,
            density=2.0,
            arrowstyle='->', arrowsize=1.2,
            start_points=start_points,
            integration_direction='both',
            maxlength=10.0,
            minlength=0.1
        )
    except Exception as e:
        print(f"Warning: Streamplot failed - {e}. Using simplified version.")
        # Fallback to a simpler streamplot if the complex one fails
        streamplot = ax6.streamplot(
            grid_x_lin, grid_y_lin, 
            u_smooth.filled(0), v_smooth.filled(0),
            color='blue',
            linewidth=1.0,
            density=1.5,
            arrowstyle='->', arrowsize=1.0
        )
    
    ax6.set_title('Streamlines (Colored by Velocity Magnitude)')
    ax6.plot(boundary_points_vis[:, 0], boundary_points_vis[:, 1], **common_plot_args)
    ax6.set_xlim(grid_x_lin[0], grid_x_lin[-1])
    ax6.set_ylim(grid_y_lin[0], grid_y_lin[-1])
    
    # Final touches for all subplots
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(x_min_grid, x_max_grid)
        ax.set_ylim(y_min_grid, y_max_grid)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(f"plots/flow_field_summary_{model_name}.png", dpi=200)
    plt.close()
    
    # --- Velocity Profiles ---
    print("Generating velocity profile plots...")
    plt.figure(figsize=(12, 10))
    plt.suptitle(f"Velocity Profiles - {model_name} (Water - Newtonian Fluid)", fontsize=16)
    
    # Select points near cross-sections
    x_center_vert = 0  # Vertical cross-section
    y_center_horiz = 0  # Horizontal cross-section
    x_outlet_near = L_horizontal - W/2  # Cross-section near outlet
    y_inlet_near = L_vertical - W/2  # Cross-section near inlet
    
    tol = W * 0.05  # Tolerance based on pipe width
    
    # Profile 1: Vertical section (x=x_center_vert)
    idx1 = np.where(np.abs(vis_points[:, 0] - x_center_vert) < tol)[0]
    if len(idx1) > 5:
        points1 = vis_points[idx1]
        u1, v1 = u_pred[idx1], v_pred[idx1]
        sort_idx = np.argsort(points1[:, 1])  # Sort by y
        y_coords1, u1, v1 = points1[sort_idx, 1], u1[sort_idx], v1[sort_idx]
        
        ax_v1 = plt.subplot(2, 2, 1)
        ax_v1.plot(u1, y_coords1, 'b.-', label='U-velocity (x-dir)')
        ax_v1.plot(v1, y_coords1, 'g.-', label='V-velocity (y-dir)')
        ax_v1.set_title(f'Profile at x={x_center_vert:.2f}m (Vertical Section)')
        ax_v1.set_xlabel('Velocity (m/s)')
        ax_v1.set_ylabel('y (m)')
        ax_v1.legend()
        ax_v1.grid(True, linestyle='--', alpha=0.7)
        # Add lines for walls
        ax_v1.axhline(y=L_vertical, color='k', linestyle='-', lw=1.5)
        ax_v1.axhline(y=-W/2, color='k', linestyle='-', lw=1.5)
    
    # Profile 2: Horizontal section (y=y_center_horiz)
    idx2 = np.where(np.abs(vis_points[:, 1] - y_center_horiz) < tol)[0]
    if len(idx2) > 5:
        points2 = vis_points[idx2]
        u2, v2 = u_pred[idx2], v_pred[idx2]
        sort_idx = np.argsort(points2[:, 0])  # Sort by x
        x_coords2, u2, v2 = points2[sort_idx, 0], u2[sort_idx], v2[sort_idx]
        
        ax_h1 = plt.subplot(2, 2, 3)
        ax_h1.plot(x_coords2, u2, 'b.-', label='U-velocity (x-dir)')
        ax_h1.plot(x_coords2, v2, 'g.-', label='V-velocity (y-dir)')
        ax_h1.set_title(f'Profile at y={y_center_horiz:.2f}m (Horizontal Section)')
        ax_h1.set_xlabel('x (m)')
        ax_h1.set_ylabel('Velocity (m/s)')
        ax_h1.legend()
        ax_h1.grid(True, linestyle='--', alpha=0.7)
        # Add lines for walls
        ax_h1.axvline(x=-W/2, color='k', linestyle='-', lw=1.5)
        ax_h1.axvline(x=L_horizontal, color='k', linestyle='-', lw=1.5)
    
    # Profile 3: Near Inlet (y=y_inlet_near)
    idx3 = np.where(np.abs(vis_points[:, 1] - y_inlet_near) < tol)[0]
    # Also ensure we are in the vertical part for this profile
    idx3 = idx3[np.abs(vis_points[idx3, 0]) <= W/2]
    if len(idx3) > 5:
        points3 = vis_points[idx3]
        u3, v3 = u_pred[idx3], v_pred[idx3]
        sort_idx = np.argsort(points3[:, 0])  # Sort by x
        x_coords3, u3, v3 = points3[sort_idx, 0], u3[sort_idx], v3[sort_idx]
        
        ax_in = plt.subplot(2, 2, 2)
        ax_in.plot(x_coords3, u3, 'b.-', label='U-velocity (x-dir)')
        ax_in.plot(x_coords3, v3, 'g.-', label='V-velocity (y-dir)')
        # Expected inlet profile: u=0, v=-u_in
        ax_in.axhline(y=-u_in, color='r', linestyle='--', label=f'Expected v (-{u_in:.1f} m/s)')
        ax_in.set_title(f'Profile Near Inlet (y={y_inlet_near:.2f}m)')
        ax_in.set_xlabel('x (m)')
        ax_in.set_ylabel('Velocity (m/s)')
        ax_in.legend()
        ax_in.grid(True, linestyle='--', alpha=0.7)
        ax_in.set_xlim(-W/2 - 0.05, W/2 + 0.05)
    
    # Profile 4: Near Outlet (x=x_outlet_near)
    idx4 = np.where(np.abs(vis_points[:, 0] - x_outlet_near) < tol)[0]
    # Also ensure we are in the horizontal part for this profile
    idx4 = idx4[np.abs(vis_points[idx4, 1]) <= W/2]
    if len(idx4) > 5:
        points4 = vis_points[idx4]
        u4, v4 = u_pred[idx4], v_pred[idx4]
        sort_idx = np.argsort(points4[:, 1])  # Sort by y
        y_coords4, u4, v4 = points4[sort_idx, 1], u4[sort_idx], v4[sort_idx]
        
        ax_out = plt.subplot(2, 2, 4)
        ax_out.plot(u4, y_coords4, 'b.-', label='U-velocity (x-dir)')
        ax_out.plot(v4, y_coords4, 'g.-', label='V-velocity (y-dir)')
        # Expected outlet profile: u > 0, v approx 0
        ax_out.axvline(x=0, color='r', linestyle='--', label='Expected v approx 0')
        ax_out.set_title(f'Profile Near Outlet (x={x_outlet_near:.2f}m)')
        ax_out.set_xlabel('Velocity (m/s)')
        ax_out.set_ylabel('y (m)')
        ax_out.legend()
        ax_out.grid(True, linestyle='--', alpha=0.7)
        ax_out.set_ylim(-W/2 - 0.05, W/2 + 0.05)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(f"plots/velocity_profiles_{model_name}.png", dpi=200)
    plt.close()
    
    # Analyze reflection behavior
    analyze_reflection_behavior(model, wall_points, wall_normals, model_name)

def analyze_reflection_behavior(model, wall_points, wall_normals, model_name, num_points=5,
                           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Analyze the reflection behavior at the walls to verify momentum conservation.
    
    Args:
        model: Trained PINN model
        wall_points: Points on the wall boundaries
        wall_normals: Normal vectors for wall points
        model_name: Name of the model for plot titles
        num_points: Number of wall points to analyze
        device: Computation device (CPU/GPU)
    """
    try:
        print("\n--- Analyzing Reflection Behavior at Walls ---")
        
        # Parameters for momentum reflection
        restitution_coef = 1.0  # Perfect elastic reflection
        friction_coef = 0.0     # Frictionless
        
        # Select a subset of wall points for testing
        if len(wall_points) >= num_points:
            indices = np.linspace(0, len(wall_points)-1, num_points, dtype=int)
            test_points = wall_points[indices]
            test_normals = wall_normals[indices]
        else:
            test_points = wall_points
            test_normals = wall_normals
            num_points = len(wall_points)
        
        plt.figure(figsize=(15, 10))
        
        for i in range(min(num_points, 6)):  # Limit to 6 subplots
            plt.subplot(2, 3, i+1)
            
            wall_point = test_points[i]
            normal = test_normals[i]
            tangent = np.array([-normal[1], normal[0]])  # 90 degree rotation
            
            # Plot the wall section
            plt.plot([wall_point[0]-tangent[0]*0.2, wall_point[0]+tangent[0]*0.2],
                    [wall_point[1]-tangent[1]*0.2, wall_point[1]+tangent[1]*0.2],
                    'k-', linewidth=2)
            
            # Plot the normal vector
            plt.arrow(wall_point[0], wall_point[1], 
                    normal[0]*0.1, normal[1]*0.1, 
                    head_width=0.02, color='blue', label='Normal')
            
            # Test various approach trajectories
            test_distances = [0.05, 0.1, 0.15]  # Distances from wall
            test_angles = np.linspace(0, np.pi*0.75, 4)  # Approach angles
            
            for d in test_distances:
                for theta in test_angles:
                    # Create a point inside the domain, approaching the wall
                    approach_vector = -normal*np.cos(theta) + tangent*np.sin(theta)
                    approach_point = wall_point - approach_vector * d
                    
                    # Only proceed if this point is inside the domain
                    if inside_L(approach_point[0], approach_point[1]):
                        # Get velocity at this point
                        model.eval()
                        with torch.no_grad():
                            xy_test = torch.tensor([[approach_point[0], approach_point[1]]], 
                                                dtype=torch.float32, device=device)
                            output = model(xy_test).cpu().numpy()[0]
                            u_val, v_val = output[0], output[1]
                            
                        # Plot the velocity vector
                        plt.arrow(approach_point[0], approach_point[1], 
                                u_val*0.1, v_val*0.1, 
                                head_width=0.01, color='red', alpha=0.7)
                        
                        # Calculate reflection from the wall
                        vel_vector = np.array([u_val, v_val])
                        vel_normal = np.dot(vel_vector, normal) * normal
                        vel_tangent = vel_vector - vel_normal
                        
                        # Reflected velocity (based on coefficient of restitution)
                        reflected_vel = -restitution_coef * vel_normal + (1 - friction_coef) * vel_tangent
                        
                        # Calculate reflection point (mirror approach point over the wall)
                        reflection_dist = d * restitution_coef
                        reflection_point = wall_point + approach_vector * reflection_dist
                        
                        # Plot the expected reflection vector
                        plt.arrow(wall_point[0], wall_point[1], 
                                reflected_vel[0]*0.1, reflected_vel[1]*0.1, 
                                head_width=0.01, color='green', alpha=0.4)
            
            plt.title(f'Reflection at Wall Point {i+1}')
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.axis('equal')
            plt.grid(True)
            
            # Add a legend to the first subplot
            if i == 0:
                plt.legend(['Wall', 'Normal', 'Velocity', 'Expected Reflection'])
        
        plt.tight_layout()
        plt.savefig(f"plots/reflection_behavior_{model_name}.png", dpi=200)
        plt.close()
        print("Reflection behavior analysis completed.")
    except Exception as e:
        print(f"Error in reflection analysis: {e}")
        import traceback
        traceback.print_exc()

def create_animated_streamlines(model, model_name="PINN", 
                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                              frames=60, fps=20):
    """
    Create an animated streamlines visualization showing flow particles moving.
    
    Args:
        model: Trained PINN model
        model_name: Name of the model for plot titles
        device: Computation device (CPU/GPU)
        frames: Number of frames in the animation
        fps: Frames per second
    """
    print("\n--- Creating Animated Streamlines ---")
    
    # Make sure the plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Prepare grid for streamlines
    x_min, x_max = -W/2 - 0.1, L_horizontal + 0.1
    y_min, y_max = -W/2 - 0.1, L_vertical + 0.1
    
    # Create a regular grid with larger spacing
    nx, ny = 80, 80  # Reduced resolution for better stability
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)
    
    # Define domain outline
    segments = [
        [(-W/2, -W/2), (-W/2, L_vertical)],
        [(-W/2, L_vertical), (W/2, L_vertical)],
        [(W/2, L_vertical), (W/2, W/2)],
        [(W/2, W/2), (L_horizontal, W/2)],
        [(L_horizontal, W/2), (L_horizontal, -W/2)],
        [(L_horizontal, -W/2), (-W/2, -W/2)]
    ]
    
    # Create a mask for the domain
    mask = np.ones_like(X, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if inside_L(X[i, j], Y[i, j]):
                mask[i, j] = False  # Unmask points inside the domain
    
    # Get velocity field
    print("Computing velocity field...")
    u_grid = np.zeros(X.shape)
    v_grid = np.zeros(X.shape)
    
    # Predict velocities directly at grid points instead of using scattered points
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not mask[i, j]:  # Only process points inside the domain
                point = np.array([[X[i, j], Y[i, j]]])
                with torch.no_grad():
                    tensor_point = torch.tensor(point, dtype=torch.float32, device=device)
                    output = model(tensor_point).cpu().numpy()[0]
                    u_grid[i, j] = output[0]
                    v_grid[i, j] = output[1]
    
    # Apply mask to velocity fields
    u_masked = np.ma.masked_array(u_grid, mask=mask)
    v_masked = np.ma.masked_array(v_grid, mask=mask)
    
    # Apply light smoothing for cleaner streamlines
    u_smooth = gaussian_filter(u_masked.filled(0), sigma=0.5)
    v_smooth = gaussian_filter(v_masked.filled(0), sigma=0.5)
    u_smooth = np.ma.masked_array(u_smooth, mask=mask)
    v_smooth = np.ma.masked_array(v_smooth, mask=mask)
    
    # Compute velocity magnitude
    print("Computing velocity magnitude...")
    vel_mag = np.sqrt(np.maximum(0, u_smooth**2 + v_smooth**2))
    vel_max = np.percentile(vel_mag.compressed(), 99.5) if vel_mag.count() > 0 else 1.0
    
    # Generate initial particle positions
    num_particles = 150  # Reduced for better performance
    particles_x = []
    particles_y = []
    
    # Generate particles in different segments
    # Inlet (top vertical)
    for _ in range(num_particles // 4):
        x_pos = np.random.uniform(-W/2 + 0.05, W/2 - 0.05)
        y_pos = np.random.uniform(L_vertical - 0.2, L_vertical - 0.05)
        if inside_L(x_pos, y_pos):
            particles_x.append(x_pos)
            particles_y.append(y_pos)
    
    # Vertical section
    for _ in range(num_particles // 4):
        x_pos = np.random.uniform(-W/2 + 0.05, W/2 - 0.05)
        y_pos = np.random.uniform(W/2 + 0.05, L_vertical - 0.2)
        if inside_L(x_pos, y_pos):
            particles_x.append(x_pos)
            particles_y.append(y_pos)
    
    # Corner and horizontal section
    for _ in range(num_particles // 2):
        x_pos = np.random.uniform(0.05, L_horizontal - 0.1)
        y_pos = np.random.uniform(-W/2 + 0.05, W/2 - 0.05)
        if inside_L(x_pos, y_pos):
            particles_x.append(x_pos)
            particles_y.append(y_pos)
    
    particles_x = np.array(particles_x)
    particles_y = np.array(particles_y)
    
    # Ensure we have at least some particles
    if len(particles_x) < 10:
        print("Warning: Not enough particles inside domain. Adding default particles...")
        # Add some default particles in the middle of the domain
        particles_x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        particles_y = np.array([0.0, 0.5, 1.0, 0.0, 0.0])
    
    # Initialize particle ages
    particles_age = np.random.uniform(0, 1, len(particles_x))
    
    print(f"Generated {len(particles_x)} particles")
    
    # Create a simple static plot first as a fallback
    plt.figure(figsize=(10, 8))
    plt.title(f'Flow Field - {model_name}', fontsize=14)
    
    # Draw domain outline
    for segment in segments:
        (x1, y1), (x2, y2) = segment
        plt.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
    
    # Draw static streamlines
    plt.streamplot(
        X, Y, 
        u_smooth.filled(0), v_smooth.filled(0),
        color=vel_mag.filled(0), cmap='viridis',
        linewidth=1.0,
        density=1.5,
        arrowstyle='->', arrowsize=1.0
    )
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.axis('equal')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.colorbar(label='Velocity Magnitude (m/s)')
    
    # Save static plot as a fallback
    static_path = f"plots/streamlines_static_{model_name}.png"
    plt.savefig(static_path, dpi=150)
    plt.close()
    print(f"Saved static streamlines to {static_path}")
    
    # Try creating the animation
    try:
        # Animation function
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.title(f'Flow Field Animation - {model_name}', fontsize=14)
        
        # Colormap for particles
        norm = Normalize(vmin=0, vmax=vel_max)
        
        # Function to get interpolated velocity at a position
        def get_velocity_at_point(px, py):
            # Find closest grid indices
            ix = min(max(0, int((px - x_min) / (x_max - x_min) * (nx-1))), nx-2)
            iy = min(max(0, int((py - y_min) / (y_max - y_min) * (ny-1))), ny-2)
            
            # Simple bilinear interpolation
            dx = (px - X[0, ix]) / (X[0, ix+1] - X[0, ix])
            dy = (py - Y[iy, 0]) / (Y[iy+1, 0] - Y[iy, 0])
            
            # Get velocity components at surrounding points
            u00 = u_smooth[iy, ix]
            u01 = u_smooth[iy, ix+1]
            u10 = u_smooth[iy+1, ix]
            u11 = u_smooth[iy+1, ix+1]
            
            v00 = v_smooth[iy, ix]
            v01 = v_smooth[iy, ix+1]
            v10 = v_smooth[iy+1, ix]
            v11 = v_smooth[iy+1, ix+1]
            
            # Interpolate
            u_val = (1-dx)*(1-dy)*u00 + dx*(1-dy)*u01 + (1-dx)*dy*u10 + dx*dy*u11
            v_val = (1-dx)*(1-dy)*v00 + dx*(1-dy)*v01 + (1-dx)*dy*v10 + dx*dy*v11
            
            # If masked, return zero
            if isinstance(u_val, np.ma.core.MaskedConstant) or isinstance(v_val, np.ma.core.MaskedConstant):
                return 0.0, 0.0
            
            return u_val, v_val
        
        # Function to draw a single frame
        def animate(frame):
            ax.clear()
            
            # Step 1: Update particle positions using interpolated velocity field
            dt = 0.05  # Time step
            
            # New positions based on local velocities
            for i in range(len(particles_x)):
                # Get velocity at this position
                if inside_L(particles_x[i], particles_y[i]):
                    u_val, v_val = get_velocity_at_point(particles_x[i], particles_y[i])
                    
                    # Update position
                    particles_x[i] += u_val * dt
                    particles_y[i] += v_val * dt
                    
                    # Increase age
                    particles_age[i] += 0.01
                    if particles_age[i] > 1:
                        particles_age[i] = 1
                else:
                    # Particle outside domain - reset near inlet
                    particles_x[i] = np.random.uniform(-W/2 + 0.05, W/2 - 0.05)
                    particles_y[i] = np.random.uniform(L_vertical - 0.2, L_vertical - 0.05)
                    particles_age[i] = 0  # Reset age
            
            # Step 2: Draw domain outline
            for segment in segments:
                (x1, y1), (x2, y2) = segment
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2)
            
            # Step 3: Draw streamlines (background)
            strm = ax.streamplot(
                X, Y, 
                u_smooth.filled(0), v_smooth.filled(0),
                color='lightgray', linewidth=0.5,
                density=1.5,
                arrowstyle='->', arrowsize=1.0
            )
            
            # Step 4: Draw particles
            vel_at_particles = np.zeros(len(particles_x))
            for i in range(len(particles_x)):
                if inside_L(particles_x[i], particles_y[i]):
                    # Get velocity magnitude at particle position using same interpolation
                    u_val, v_val = get_velocity_at_point(particles_x[i], particles_y[i])
                    vel_at_particles[i] = np.sqrt(u_val**2 + v_val**2)
                else:
                    vel_at_particles[i] = 0
            
            # Draw particles with color based on velocity and size based on age
            scatter = ax.scatter(
                particles_x, particles_y,
                c=vel_at_particles, cmap='viridis', norm=norm,
                s=20 + 20 * particles_age,  # Size grows with age
                alpha=0.7, edgecolors='white', linewidths=0.5
            )
            
            # Set plot limits and labels
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('x (m)', fontsize=12)
            ax.set_ylabel('y (m)', fontsize=12)
            ax.set_aspect('equal')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Add colorbar on first frame
            if frame == 0:
                plt.colorbar(scatter, ax=ax, label='Velocity magnitude (m/s)')
            
            # Add time indicator
            ax.text(
                0.02, 0.96, f'Time: {frame * dt:.2f} s',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
            )
            
            return [scatter]
        
        anim = FuncAnimation(
            fig, animate, frames=frames,
            interval=1000/fps, blit=False
        )
        
        # Save the animation
        anim_path = f"plots/streamlines_animation_{model_name}.gif"
        print(f"Saving animation to {anim_path}")
        anim.save(anim_path, writer='pillow', fps=fps, dpi=100)
        plt.close()
        
        print("Animated streamlines created successfully")
        
    except Exception as e:
        print(f"Error creating animation: {e}")
        import traceback
        traceback.print_exc()
        print("\nFalling back to static streamlines visualization")
    
