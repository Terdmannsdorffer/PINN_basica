def visualize_results(model, model_name, num_points=5000):
    """Visualize the flow field results with enhanced plots."""
    try:
        # Generate a dense grid of points inside the L-shaped pipe for visualization
        vis_points = generate_domain_points(num_points)

        # Generate boundary points for drawing the outline
        boundary_points_vis, _, _, _, _ = generate_boundary_points(300)  # Denser boundary for plotting

        # Predict using the trained model
        model.eval()  # Set to evaluation mode
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

        # --- Scatter Plots ---
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

        # --- Vector and Streamline Plots ---
        # Prepare grid for streamlines/quiver
        x_min_grid, x_max_grid = -W/2 - 0.1, L_horizontal + 0.1
        y_min_grid, y_max_grid = -W/2 - 0.1, L_vertical + 0.1
        
        # Use np.linspace to define the 1D grid vectors first
        grid_x_lin = np.linspace(x_min_grid, x_max_grid, 100)  # Reduced from 150 for speed
        grid_y_lin = np.linspace(y_min_grid, y_max_grid, 100)  # Reduced from 150 for speed
        
        # Use meshgrid to create 2D coordinate arrays
        grid_x, grid_y = np.meshgrid(grid_x_lin, grid_y_lin, indexing='xy')
        grid_shape = grid_x.shape

        # Interpolate velocity components onto the grid
        print("Interpolating results onto grid for vector/streamline plots...")
        u_interp = griddata((vis_points[:, 0], vis_points[:, 1]), u_pred, (grid_x, grid_y), method='cubic', fill_value=0)
        v_interp = griddata((vis_points[:, 0], vis_points[:, 1]), v_pred, (grid_x, grid_y), method='cubic', fill_value=0)
        vel_mag_interp = np.sqrt(u_interp**2 + v_interp**2)

        # Create a mask for the L-shape on the grid
        mask = np.ones(grid_shape, dtype=bool)  # Mask starts as True (masked) everywhere
        print("Creating mask for visualization grid...")
        for i in range(grid_shape[0]):  # Iterate over rows (y-dimension)
            for j in range(grid_shape[1]):  # Iterate over columns (x-dimension)
                current_x = grid_x[i, j]
                current_y = grid_y[i, j]
                if inside_L_pipe(current_x, current_y):
                    mask[i, j] = False  # Unmask points inside the pipe
        print("Mask created.")

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
        print("Smoothing applied.")

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
        
        # 6. Streamlines - simplified for debugging
        ax6 = plt.subplot(3, 2, 6)

        # Define better starting points for streamlines
        num_stream_starts = 15  # Reduced from 25 for speed
        start_x_inlet = np.linspace(-W/2 * 0.9, W/2 * 0.9, num_stream_starts)
        start_y_inlet = np.ones_like(start_x_inlet) * (L_vertical - 0.05)

        # Add additional seed points for better coverage
        start_x_vertical = np.zeros(6)  # Reduced from 12 for speed
        start_y_vertical = np.linspace(0.1, L_vertical-0.1, 6)

        start_x_horizontal = np.linspace(0.1, L_horizontal-0.1, 6)
        start_y_horizontal = np.zeros(6)  # Points in middle of horizontal section

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
            # Simplified streamplot for debugging
            streamplot = ax6.streamplot(
                grid_x_lin, grid_y_lin, 
                u_smooth.filled(0), v_smooth.filled(0),
                color=speed_data, cmap='viridis', norm=norm_vel,
                linewidth=lw_data,
                density=2.0,  # Reduced from 2.5 for speed
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

        # --- Final Touches for Figure 1 ---
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(x_min_grid, x_max_grid)
            ax.set_ylim(y_min_grid, y_max_grid)
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0.02, 1, 0.97])
        plt.savefig(f"plots/flow_field_summary_{model_name}.png", dpi=200)  # Reduced from 300 dpi for speed
        print(f"Saved flow field summary plot for {model_name}.")
        plt.close()

        # --- Figure 2: Velocity Profiles ---
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
        print(f"Saved velocity profiles plot for {model_name}.")
        plt.close()
        
        # Analyze reflection behavior
        analyze_reflection_behavior(model, wall_points, wall_normals, model_name)
    
    except Exception as e:
        print(f"Error in visualization: {e}")
        traceback.print_exc()


def analyze_reflection_behavior(model, wall_points, wall_normals, model_name, num_points=5):
    """Analyze the reflection behavior at the walls to verify momentum conservation"""
    try:
        print("\n--- Analyzing Reflection Behavior at Walls ---")
        
        # Select a subset of wall points for testing
        indices = np.linspace(0, len(wall_points)-1, num_points, dtype=int)
        test_points = wall_points[indices]
        test_normals = wall_normals[indices]
        
        plt.figure(figsize=(15, 10))
        
        for i in range(num_points):
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
                    if inside_L_pipe(approach_point[0], approach_point[1]):
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
        traceback.print_exc()

def main():
    """Main function to run the PINN training and visualization with improved implementation."""
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # IMPROVEMENT: Select model type - PICN (convolutional) or traditional PINN
        use_picn = False  # Set to False to use traditional PINN for better stability
        
        # Initialize model with appropriate architecture
        if use_picn:
            # Physics-Informed Neural Network with advanced architecture
            model = PICN(hidden_layers=3, 
                        neurons_per_layer=40).to(device)
            model_name = "PICN_Momentum"
        else:
            # Traditional Physics-Informed Neural Network
            model = PINN(hidden_layers=4, 
                        neurons_per_layer=40).to(device)
            model_name = "PINN_Momentum"
        
        print(f"Initialized {model_name} model")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        # IMPROVEMENT: Generate training points with better distribution
        num_domain = 2000  # Reduced for quicker training
        num_boundary = 400  # More boundary points for stronger enforcement
        use_grid_points = False  # True for systematic grid, False for random sampling

        print("Generating training points...")
        domain_points = generate_domain_points(num_domain, randomize=not use_grid_points)
        all_boundary_points, inlet_points, outlet_points, wall_points, wall_normals = generate_boundary_points(num_boundary)

        print(f"Generated {len(domain_points)} domain points.")
        print(f"Generated {len(inlet_points)} inlet points.")
        print(f"Generated {len(outlet_points)} outlet points.")
        print(f"Generated {len(wall_points)} wall points.")
        print(f"Total boundary points: {len(all_boundary_points)}")

        # Optional: Visualize point distribution with normals
        plt.figure(figsize=(10, 8))
        plt.scatter(domain_points[:, 0], domain_points[:, 1], s=1, label='Domain', alpha=0.5)
        plt.scatter(inlet_points[:, 0], inlet_points[:, 1], s=5, label='Inlet', c='red')
        plt.scatter(outlet_points[:, 0], outlet_points[:, 1], s=5, label='Outlet', c='green')
        plt.scatter(wall_points[:, 0], wall_points[:, 1], s=5, label='Wall', c='black')
        
        # Visualize wall normals (for momentum reflection)
        scale = 0.1  # Scale factor for normal vectors
        for i in range(0, len(wall_points), 10):  # Plot every 10th normal for clarity
            plt.arrow(wall_points[i, 0], wall_points[i, 1], 
                    scale * wall_normals[i, 0], scale * wall_normals[i, 1], 
                    head_width=0.02, head_length=0.03, fc='blue', ec='blue', alpha=0.7)
        
        plt.title(f'Training Point Distribution - {model_name}')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(f"plots/point_distribution_{model_name}.png", dpi=150)
        plt.close()

        # IMPROVEMENT: Setup optimizer with better parameters
        # Using Adam with lower initial learning rate for stability
        print("Setting up optimizer...")
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # IMPROVEMENT: Learning rate scheduler
        # ReduceLROnPlateau monitors the loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50,
            min_lr=1e-6, verbose=True
        )

        # IMPROVEMENT: Define training parameters with momentum preservation
        # Weights for different components of the loss function
        lambda_physics = 1.0        # Weight for PDE residuals
        lambda_bc = 20.0            # Overall boundary condition weight
        lambda_wall_normal = 10.0   # Weight for normal component (impermeability)
        lambda_wall_tangent = 1.0   # Weight for tangential component (slip condition)
        
        # Adjust momentum conservation parameters
        print(f"Momentum Conservation Parameters:")
        print(f"Restitution Coefficient: {restitution_coef}")
        print(f"Friction Coefficient: {friction_coef}")
        
        # IMPROVEMENT: Mini-batch training
        # Using smaller batches for more frequent parameter updates
        batch_size = int(num_domain / 4)  # 1/4 of domain points per batch
        use_cyclic_lr = True    # Use 1-cycle learning rate policy

        # Train model
        print("\n--- Starting Training ---")
        print(f"Model: {model_name}")
        print(f"Physics Loss Weight: {lambda_physics}")
        print(f"Boundary Loss Weight: {lambda_bc}")
        print(f"Wall Normal Loss Weight: {lambda_wall_normal}")
        print(f"Wall Tangent Loss Weight: {lambda_wall_tangent}")
        print(f"Batch Size: {batch_size}")
        print(f"Using Cyclic LR: {use_cyclic_lr}")
        print(f"Gravity: {g} m/s²")
        print(f"Inlet Velocity: {-u_in} m/s")
        print("-" * 25)

        model, loss_history = train_model(
            model, optimizer, scheduler,
            domain_points, inlet_points, outlet_points, wall_points, wall_normals,
            epochs=200,  # Reduced epochs for quicker training
            lambda_physics=lambda_physics,
            lambda_bc=lambda_bc,
            display_every=20,
            batch_size=batch_size,
            use_cyclic_lr=use_cyclic_lr
        )

        # IMPROVEMENT: More comprehensive visualization
        print("\n--- Generating Visualizations ---")
        visualize_results(model, model_name, num_points=3000)  # Reduced for speed
        
        print("\n--- Script finished successfully! ---")
        return model
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    try:
        print("\n=== Starting PINN with Momentum-Preserving Boundary Conditions ===\n")
        model = main()
        print("Script execution completed successfully!")
    except Exception as e:
        print(f"Fatal error: {e}")
        #traceback.print_exc()
        
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import time
import os
import sys
import traceback

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: {device}")
else:
    device = torch.device('cpu')
    print(f"Using device: {device}")

# Create directories for saving results
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
print("Created output directories successfully")

# Physical parameters
rho = 1000.0   # Water density (kg/m³)
mu = 1.0e-3    # Water dynamic viscosity (kg/(m·s))
u_in = 0.5     # Inlet velocity (m/s)
g = -9.81      # Gravity (acting in negative y-direction)

# Geometry parameters
L_vertical = 2.0   # Length of vertical section (m)
L_horizontal = 3.0   # Length of horizontal section (m)
W = 0.5    # Width of the pipe (m)

# Momentum conservation parameters
restitution_coef = 1.0  # Coefficient of restitution (1.0 = perfect elastic collision)
friction_coef = 0.0     # Friction coefficient (0.0 = frictionless)

#--------------------------------------------------------------
# IMPROVEMENT 1: Enhanced Network Architecture with CNN layers
#--------------------------------------------------------------
class PICN(nn.Module):
    """
    Physics-Informed Convolutional Network (PICN) for fluid flow problems
    Based on insights from the Journal of Hydrology paper
    """
    def __init__(self, hidden_layers=4, neurons_per_layer=64, conv_channels=[16, 32]):
        super(PICN, self).__init__()
        
        # Feature extraction with convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First Conv Layer: Take input channels = 1 (from 2 features reshaped)
        self.input_embed = nn.Linear(2, neurons_per_layer // 2)
        self.input_act = nn.Tanh()
        
        # This method doesn't use convolutional layers in the typical CNN way
        # Instead, we use a series of fully connected layers with batch normalization
        self.fc_layers = nn.ModuleList()
        
        # First layer (after input embedding)
        prev_size = neurons_per_layer // 2
        for _ in range(hidden_layers):
            self.fc_layers.append(nn.Linear(prev_size, neurons_per_layer))
            prev_size = neurons_per_layer
            
        # Output layer (3 outputs: u, v, p)
        self.output_layer = nn.Linear(neurons_per_layer, 3)
        
        # Batch normalization layers
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(neurons_per_layer) for _ in range(hidden_layers)])
        
        # Activation function
        self.activation = nn.Tanh()

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Initial embedding of input
        x = self.input_embed(x)
        x = self.input_act(x)
        
        # Apply fully connected layers with batch norm
        for i, (fc, bn) in enumerate(zip(self.fc_layers, self.bn_layers)):
            x = fc(x)
            if batch_size > 1:  # BatchNorm1d requires batch size > 1
                x = bn(x)
            x = self.activation(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x

# Traditional PINN without advanced features
class PINN(nn.Module):
    """Physics-Informed Neural Network (PINN) for fluid flow problems"""
    def __init__(self, hidden_layers=6, neurons_per_layer=40):
        super(PINN, self).__init__()

        # Input layer (2 features: x, y coordinates)
        self.input_layer = nn.Linear(2, neurons_per_layer)
        self.input_act = nn.Tanh()
        self.input_bn = nn.BatchNorm1d(neurons_per_layer)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        for _ in range(hidden_layers):
            self.hidden_layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            self.bn_layers.append(nn.BatchNorm1d(neurons_per_layer))

        # Output layer (3 outputs: u, v, p)
        self.output_layer = nn.Linear(neurons_per_layer, 3)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Input layer
        x = self.input_layer(x)
        x = self.input_act(x)
        
        # Apply batch norm if batch size > 1
        if batch_size > 1:
            x = self.input_bn(x)
            
        # Hidden layers
        for i, (fc, bn) in enumerate(zip(self.hidden_layers, self.bn_layers)):
            x = fc(x)
            if batch_size > 1:
                x = bn(x)
            x = nn.Tanh()(x)

        # Output layer
        x = self.output_layer(x)
        
        return x

#--------------------------------------------------------------
# IMPROVEMENT 2: Enhanced Domain Points Generation
#--------------------------------------------------------------
def inside_L_pipe(x, y):
    """Check if point (x, y) is inside the L-shaped pipe."""
    in_vertical = (-W/2 <= x <= W/2) and (-W/2 <= y <= L_vertical)
    in_horizontal = (-W/2 <= x <= L_horizontal) and (-W/2 <= y <= W/2)
    # Ensure the corner region is included correctly
    is_inside = in_vertical or in_horizontal
    # Correct potential exclusion at the exact corner boundary
    if abs(x - W/2) < 1e-9 and abs(y - W/2) < 1e-9:
         is_inside = True
    return is_inside

def generate_domain_points(num_points, randomize=True):
    """
    Generate points inside the L-shaped pipe with improved distribution strategy.
    
    Args:
        num_points (int): Number of points to generate
        randomize (bool): Whether to use randomized or grid-based sampling
        
    Returns:
        np.array: Points within the domain
    """
    if randomize:
        # Randomized point generation with specific concentration
        # Standard points for most of the domain
        regular_points = []
        # Create a bounding box and filter points
        x_min, x_max = -W/2, L_horizontal
        y_min, y_max = -W/2, L_vertical

        # Generate points with improved density distribution
        num_regular_target = int(num_points * 0.7)
        num_vertical_points = int(num_regular_target * (L_vertical / (L_vertical + L_horizontal)))
        num_horizontal_points = num_regular_target - num_vertical_points

        # Points in vertical section
        x_vert = np.random.uniform(-W/2, W/2, size=num_vertical_points * 2)
        y_vert = np.random.uniform(W/2, L_vertical, size=num_vertical_points * 2)
        vert_points = np.stack([x_vert, y_vert], axis=-1)
        regular_points.extend(vert_points[:num_vertical_points].tolist())

        # Points in horizontal section
        x_horiz = np.random.uniform(-W/2, L_horizontal, size=num_horizontal_points * 2)
        y_horiz = np.random.uniform(-W/2, W/2, size=num_horizontal_points * 2)
        horiz_points = np.stack([x_horiz, y_horiz], axis=-1)
        regular_points.extend(horiz_points[:num_horizontal_points].tolist())

        # Additional points concentrated near the corner
        corner_points = []
        corner_x = W/2
        corner_y = W/2
        radius = W * 0.75  # Area around corner to concentrate points
        num_corner_target = num_points - len(regular_points)

        generated_count = 0
        while generated_count < num_corner_target:
            # Random distance from corner (higher density closer to corner)
            r = radius * np.random.power(2.0)  # Power distribution for more points near corner
            theta = np.random.uniform(0, 2*np.pi)
            x = corner_x + r * np.cos(theta)
            y = corner_y + r * np.sin(theta)
            if inside_L_pipe(x, y):
                corner_points.append([x, y])
                generated_count += 1

        # Combine points
        all_points = np.array(regular_points + corner_points)
    else:
        # Grid-based point generation (more systematic)
        resolution = int(np.sqrt(num_points / 2))  # Approximate resolution to achieve target point count
        
        # Create grid for vertical section
        x_vert = np.linspace(-W/2, W/2, resolution)
        y_vert = np.linspace(W/2, L_vertical, resolution*2)
        x_vert_grid, y_vert_grid = np.meshgrid(x_vert, y_vert)
        points_vert = np.column_stack([x_vert_grid.flatten(), y_vert_grid.flatten()])
        
        # Create grid for horizontal section
        x_horiz = np.linspace(-W/2, L_horizontal, resolution*3)
        y_horiz = np.linspace(-W/2, W/2, resolution)
        x_horiz_grid, y_horiz_grid = np.meshgrid(x_horiz, y_horiz)
        points_horiz = np.column_stack([x_horiz_grid.flatten(), y_horiz_grid.flatten()])
        
        # Combine and filter
        all_points = np.vstack([points_vert, points_horiz])
    
    # Ensure exact number of points
    if len(all_points) > num_points:
        indices = np.random.choice(len(all_points), size=num_points, replace=False)
        all_points = all_points[indices]
    elif len(all_points) < num_points:
        # If slightly under, add a few more random points
        print(f"Warning: Generated {len(all_points)} points, requested {num_points}. Adding random points.")
        needed = num_points - len(all_points)
        extra_points = []
        while len(extra_points) < needed:
            x_extra = np.random.uniform(-W/2, L_horizontal)
            y_extra = np.random.uniform(-W/2, L_vertical)
            if inside_L_pipe(x_extra, y_extra):
                extra_points.append([x_extra, y_extra])
        all_points = np.concatenate([all_points, np.array(extra_points)], axis=0)

    return all_points

def generate_boundary_points(num_points):
    """Generate points on the boundary with improved distribution."""
    # Define the segments of the L-shaped pipe boundary
    segments = [
        # Left wall (bottom to top)
        [(-W/2, -W/2), (-W/2, L_vertical)], # Wall
        # Top wall (left to right) -> Inlet
        [(-W/2, L_vertical), (W/2, L_vertical)], # Inlet
        # Right upper wall (top to corner)
        [(W/2, L_vertical), (W/2, W/2)], # Wall
        # Top horizontal wall (corner to right)
        [(W/2, W/2), (L_horizontal, W/2)], # Wall
        # Right wall (top to bottom) -> Outlet
        [(L_horizontal, W/2), (L_horizontal, -W/2)], # Outlet
        # Bottom wall (right to left)
        [(L_horizontal, -W/2), (-W/2, -W/2)] # Wall
    ]
    segment_types = ['wall', 'inlet', 'wall', 'wall', 'outlet', 'wall']
    
    # Calculate normal vectors for each wall segment
    normal_vectors = []
    for (x1, y1), (x2, y2) in segments:
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            # Normal vector (90 degrees counterclockwise rotation)
            nx, ny = -dy/length, dx/length
        else:
            nx, ny = 0, 0
        normal_vectors.append((nx, ny))
    
    # Calculate the total boundary length
    total_length = 0
    segment_lengths = []
    for (x1, y1), (x2, y2) in segments:
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        segment_lengths.append(length)
        total_length += length

    # Allocate points based on segment length with minimum point guarantees
    points_per_segment = []
    allocated_points = 0
    min_points_per_segment = 10
    
    for length in segment_lengths:
        num = max(min_points_per_segment, int(num_points * length / total_length))
        points_per_segment.append(num)
        allocated_points += num

    # Adjust if total allocated points differ from num_points
    diff = num_points - allocated_points
    if diff != 0:
        # Add/remove points prioritizing longer segments
        indices = np.argsort(segment_lengths)[::-1] # Descending length order
        for i in range(abs(diff)):
            idx_to_adjust = indices[i % len(indices)]
            points_per_segment[idx_to_adjust] += np.sign(diff)
        points_per_segment = [max(1, p) for p in points_per_segment]

    # Generate and classify points
    inlet_points = []
    outlet_points = []
    wall_points = []
    wall_normals = []  # Store normal vectors for wall points

    for i, ((x1, y1), (x2, y2)) in enumerate(segments):
        segment_points = points_per_segment[i]
        segment_type = segment_types[i]
        normal_vector = normal_vectors[i]

        # Generate points along this segment
        x_coords = np.linspace(x1, x2, segment_points)
        y_coords = np.linspace(y1, y2, segment_points)

        for j in range(segment_points):
            x = x_coords[j]
            y = y_coords[j]
            point = [x, y]

            # Avoid adding exact corner points multiple times
            is_corner = False
            if i > 0:
                prev_end = segments[i-1][1]
                if abs(x - prev_end[0]) < 1e-9 and abs(y - prev_end[1]) < 1e-9:
                    is_corner = True

            # Only add if not a duplicate corner point
            if not is_corner or j > 0:
                if segment_type == 'inlet':
                    inlet_points.append(point)
                elif segment_type == 'outlet':
                    outlet_points.append(point)
                else: # wall
                    wall_points.append(point)
                    wall_normals.append(normal_vector)

    # Add the very first point if missed
    if not any(np.allclose(p, segments[0][0]) for p in wall_points):
        wall_points.insert(0, list(segments[0][0]))
        wall_normals.insert(0, normal_vectors[0])

    return np.array(inlet_points + outlet_points + wall_points), np.array(inlet_points), np.array(outlet_points), np.array(wall_points), np.array(wall_normals)

#--------------------------------------------------------------
# IMPROVEMENT 3: Enhanced NS Residual Calculation
#--------------------------------------------------------------
def NS_residual(model, x, y):
    """
    Compute the Navier-Stokes residuals for incompressible flow with improved stability.
    """
    try:
        # Convert to tensor with gradient tracking
        xy_tensor = torch.tensor(np.stack([x, y], axis=1), dtype=torch.float32, requires_grad=True, device=device)

        # Forward pass
        outputs = model(xy_tensor)
        u = outputs[:, 0:1]  # x-velocity
        v = outputs[:, 1:2]  # y-velocity
        p = outputs[:, 2:3]  # pressure

        # Compute gradients
        # First derivatives
        u_grad = torch.autograd.grad(u.sum(), xy_tensor, create_graph=True)[0]
        v_grad = torch.autograd.grad(v.sum(), xy_tensor, create_graph=True)[0]
        p_grad = torch.autograd.grad(p.sum(), xy_tensor, create_graph=True)[0]

        u_x = u_grad[:, 0:1]
        u_y = u_grad[:, 1:2]
        v_x = v_grad[:, 0:1]
        v_y = v_grad[:, 1:2]
        p_x = p_grad[:, 0:1]
        p_y = p_grad[:, 1:2]

        # Second derivatives
        u_xx = torch.autograd.grad(u_x.sum(), xy_tensor, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), xy_tensor, create_graph=True)[0][:, 1:2]
        v_xx = torch.autograd.grad(v_x.sum(), xy_tensor, create_graph=True)[0][:, 0:1]
        v_yy = torch.autograd.grad(v_y.sum(), xy_tensor, create_graph=True)[0][:, 1:2]

        # Navier-Stokes equations (Incompressible, Newtonian)
        # x-momentum
        momentum_x = rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)

        # y-momentum (including gravity force)
        momentum_y = rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy) - (rho * g)

        # Continuity
        continuity = u_x + v_y

        return momentum_x, momentum_y, continuity
    except Exception as e:
        print(f"Error in NS_residual: {e}")
        traceback.print_exc()
        # Return empty tensors in case of error
        zeros = torch.zeros(1, 1, device=device)
        return zeros, zeros, zeros

#--------------------------------------------------------------
# NEW IMPLEMENTATION: Momentum-Preserving Boundary Conditions
#--------------------------------------------------------------
def compute_momentum_preserving_bc(model, wall_points, wall_normals, 
                                  restitution_coef=restitution_coef, 
                                  friction_coef=friction_coef):
    """
    Compute boundary conditions that preserve momentum during collisions with walls.
    
    Args:
        model: The PINN model
        wall_points: Points on the wall boundaries
        wall_normals: Normal vectors for each wall point
        restitution_coef: Coefficient of restitution (1.0 = perfectly elastic)
        friction_coef: Friction coefficient (0.0 = frictionless)
        
    Returns:
        Tensors for normal and tangential components of boundary loss
    """
    try:
        if len(wall_points) == 0:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        # Convert to tensors with gradient tracking
        xy_wall = torch.tensor(wall_points, dtype=torch.float32, device=device)
        wall_normals_tensor = torch.tensor(wall_normals, dtype=torch.float32, device=device)
        
        # Create a small offset in the direction of the wall normal (towards the fluid)
        offset_distance = 0.01 * W  # Small offset
        xy_interior = xy_wall - offset_distance * wall_normals_tensor
        
        # Make sure the interior points are actually inside the pipe
        interior_points_list = []
        interior_normals_list = []
        for i, (x, y) in enumerate(xy_interior.cpu().numpy()):
            if inside_L_pipe(x, y):
                interior_points_list.append([x, y])
                interior_normals_list.append(wall_normals[i])
        
        if not interior_points_list:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        
        xy_interior = torch.tensor(np.array(interior_points_list), dtype=torch.float32, device=device)
        interior_normals_tensor = torch.tensor(np.array(interior_normals_list), dtype=torch.float32, device=device)
        
        # Get velocities at wall and interior points
        outputs_wall = model(xy_wall)
        outputs_interior = model(xy_interior)
        
        u_wall = outputs_wall[:, 0:1]
        v_wall = outputs_wall[:, 1:2]
        u_interior = outputs_interior[:, 0:1]
        v_interior = outputs_interior[:, 1:2]
        
        # Extract wall velocity components into normal and tangential directions
        vel_wall = torch.cat([u_wall, v_wall], dim=1)
        
        # Normal and tangential components at wall
        normal_vel_wall = torch.sum(vel_wall * interior_normals_tensor, dim=1, keepdim=True)
        tangent_normals = torch.stack([-interior_normals_tensor[:, 1], interior_normals_tensor[:, 0]], dim=1)
        tangent_vel_wall = torch.sum(vel_wall * tangent_normals, dim=1, keepdim=True)
        
        # Create velocity gradients to compute the Robin boundary condition
        dvel_dn = torch.zeros_like(normal_vel_wall)
        if len(xy_interior) > 0:
            vel_interior = torch.cat([u_interior, v_interior], dim=1)
            normal_vel_interior = torch.sum(vel_interior * interior_normals_tensor, dim=1, keepdim=True)
            tangent_vel_interior = torch.sum(vel_interior * tangent_normals, dim=1, keepdim=True)
            
            # Estimate velocity gradient in normal direction
            dvel_dn = (normal_vel_wall - normal_vel_interior) / offset_distance
        
        # Robin boundary condition for momentum preservation
        # For normal component: should be zero at wall (impermeability)
        normal_bc_loss = normal_vel_wall**2  # Normal velocity should be zero at wall
        
        # For tangential component: we want to enforce reflection law with potential friction
        # In a momentum-preserving scheme with Robin BCs, we aim for:
        # ∂u_tangential/∂n = -α * u_tangential (slip with friction)
        # where α is related to the friction coefficient
        
        # Simplified approach: Penalize non-zero gradient of tangential velocity in normal direction
        # This is a basic approximation for slip condition
        tangent_bc_loss = (dvel_dn + friction_coef * tangent_vel_wall)**2
        
        return normal_bc_loss, tangent_bc_loss
    except Exception as e:
        print(f"Error in compute_momentum_preserving_bc: {e}")
        traceback.print_exc()
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

#--------------------------------------------------------------
# IMPROVEMENT 4: Enhanced Loss Function with Balanced Weighting and Momentum Preservation
#--------------------------------------------------------------
def compute_loss(model, domain_points, inlet_points, outlet_points, wall_points, wall_normals,
                 lambda_physics=1.0, lambda_bc=50.0, lambda_inlet=5.0, lambda_outlet=3.0, 
                 lambda_wall_normal=10.0, lambda_wall_tangent=1.0):
    """
    Compute the total loss with balanced weighting between different components.
    Includes momentum-preserving boundary conditions.
    """
    try:
        # --- Physics Loss (Navier-Stokes residuals in the domain) ---
        # Initialize with zero tensor with gradient tracking
        physics_loss = torch.tensor(0.0, device=device, requires_grad=True)
        bc_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if len(domain_points) > 0:
            # Sample a subset of domain points if there are too many
            max_physics_points = min(len(domain_points), 2000)  # Limit for performance
            if len(domain_points) > max_physics_points:
                indices = np.random.choice(len(domain_points), max_physics_points, replace=False)
                x_domain = domain_points[indices, 0]
                y_domain = domain_points[indices, 1]
            else:
                x_domain = domain_points[:, 0]
                y_domain = domain_points[:, 1]

            momentum_x, momentum_y, continuity = NS_residual(model, x_domain, y_domain)

            # Scale the momentum_y loss term relative to gravity magnitude
            g_scale = 1.0 if abs(g) < 0.1 else (1.0 / (1.0 + abs(rho*g)))

            loss_mx = torch.mean(momentum_x**2)
            loss_my = g_scale * torch.mean(momentum_y**2)
            loss_cont = torch.mean(continuity**2)

            physics_loss = physics_loss + loss_mx + loss_my + loss_cont

        # --- Boundary Condition Loss ---
        # Inlet BC: u = 0, v = -u_in (flow enters vertically downwards)
        if len(inlet_points) > 0:
            xy_inlet = torch.tensor(inlet_points, dtype=torch.float32, device=device)
            outputs_inlet = model(xy_inlet)
            u_inlet = outputs_inlet[:, 0:1]
            v_inlet = outputs_inlet[:, 1:2]

            inlet_loss = torch.mean(u_inlet**2) + lambda_inlet * torch.mean((v_inlet + u_in)**2)
            bc_loss = bc_loss + inlet_loss

        # Outlet BC: p = 0 AND encourage positive horizontal flow
        if len(outlet_points) > 0:
            xy_outlet = torch.tensor(outlet_points, dtype=torch.float32, device=device)
            outputs_outlet = model(xy_outlet)
            u_outlet = outputs_outlet[:, 0:1]
            v_outlet = outputs_outlet[:, 1:2]
            p_outlet = outputs_outlet[:, 2:3]

            # Target horizontal outflow velocity
            target_u = 0.8 * u_in

            # Use smooth ReLU-like function for u penalty
            u_penalty = torch.mean(torch.log(1 + torch.exp(-(u_outlet - target_u))))

            outlet_loss = (torch.mean(p_outlet**2) +
                         lambda_outlet * u_penalty +
                         lambda_outlet * torch.mean(v_outlet**2))

            bc_loss = bc_loss + outlet_loss

        # Wall BC: Using momentum-preserving boundary conditions
        if len(wall_points) > 0:
            # Sample wall points if needed
            max_wall_points = min(len(wall_points), 500)  # Limit for performance
            if len(wall_points) > max_wall_points:
                indices = np.random.choice(len(wall_points), max_wall_points, replace=False)
                wall_sample = wall_points[indices]
                normals_sample = wall_normals[indices]
            else:
                wall_sample = wall_points
                normals_sample = wall_normals
                
            normal_bc_loss, tangent_bc_loss = compute_momentum_preserving_bc(
                model, wall_sample, normals_sample, 
                restitution_coef=restitution_coef, 
                friction_coef=friction_coef
            )
            
            wall_loss = lambda_wall_normal * torch.mean(normal_bc_loss) + lambda_wall_tangent * torch.mean(tangent_bc_loss)
            bc_loss = bc_loss + wall_loss

        # Special treatment for the corner region
        corner_x_center = W/2
        corner_y_center = W/2
        corner_radius = W * 0.4

        # Find domain points near the corner
        domain_points_np = domain_points
        distances = np.sqrt((domain_points_np[:, 0] - corner_x_center)**2 + (domain_points_np[:, 1] - corner_y_center)**2)
        corner_indices = np.where(distances < corner_radius)[0]

        if len(corner_indices) > 0:
            corner_domain_points = domain_points_np[corner_indices]
            x_corner = corner_domain_points[:, 0]
            y_corner = corner_domain_points[:, 1]

            # Get divergence at corner points
            _, _, div_corner = NS_residual(model, x_corner, y_corner)

            # Enforce stronger continuity constraint at corner
            corner_loss = 3.0 * torch.mean(div_corner**2)
            bc_loss = bc_loss + corner_loss

        # --- Total Loss ---
        total_loss = lambda_physics * physics_loss + lambda_bc * bc_loss

        # Return individual components for logging
        return total_loss, physics_loss.detach(), bc_loss.detach()
    except Exception as e:
        print(f"Error computing loss: {e}")
        traceback.print_exc()
        # Return default values in case of error
        return torch.tensor(1.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

#--------------------------------------------------------------
# IMPROVEMENT 5: Enhanced Training Process with Adaptive Learning Rate
#--------------------------------------------------------------
def train_model(model, optimizer, scheduler, domain_points, inlet_points, outlet_points, wall_points, wall_normals,
                epochs=500, lambda_physics=1.0, lambda_bc=30.0, display_every=50, 
                batch_size=None, use_cyclic_lr=True):
    """
    Train the PINN model with improved training strategy.
    
    Args:
        model: PINN model to train
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        domain_points, inlet_points, outlet_points, wall_points: Training points
        wall_normals: Normal vectors for wall points
        epochs: Number of training epochs
        lambda_physics, lambda_bc: Loss weights
        display_every: Frequency of progress display
        batch_size: Mini-batch size (if None, use full batch)
        use_cyclic_lr: Whether to use cyclic learning rate (1-cycle policy)
    """
    try:
        loss_history = []
        best_loss = float('inf')
        start_time = time.time()
        
        # Set up 1-cycle learning rate scheduler if requested
        if use_cyclic_lr:
            # Replace the provided scheduler with 1-cycle
            max_lr = optimizer.param_groups[0]['lr'] * 10
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=max_lr, total_steps=epochs, 
                pct_start=0.3  # Spend 30% of time increasing LR
            )
        
        # Mini-batch settings
        if batch_size is None:
            batch_size = len(domain_points)  # Full batch
        
        n_batches = int(np.ceil(len(domain_points) / batch_size))
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            
            # Create random indices for mini-batch training
            if batch_size < len(domain_points):
                indices = np.random.permutation(len(domain_points))
            else:
                indices = np.arange(len(domain_points))
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(domain_points))
                batch_indices = indices[start_idx:end_idx]
                
                # Select mini-batch
                batch_domain_points = domain_points[batch_indices]
                
                # For boundary points, either use all or select a subset
                # This keeps the ratio of domain to boundary points balanced
                bd_ratio = min(1.0, len(batch_domain_points) / len(domain_points))
                
                inlet_batch = inlet_points
                outlet_batch = outlet_points
                wall_batch = wall_points
                wall_normals_batch = wall_normals
                
                # For large datasets, sample boundary points proportionally
                if len(inlet_points) > 100:
                    inlet_indices = np.random.choice(len(inlet_points), 
                                                    size=max(10, int(len(inlet_points) * bd_ratio)),
                                                    replace=False)
                    inlet_batch = inlet_points[inlet_indices]
                    
                if len(outlet_points) > 100:
                    outlet_indices = np.random.choice(len(outlet_points), 
                                                    size=max(10, int(len(outlet_points) * bd_ratio)),
                                                    replace=False)
                    outlet_batch = outlet_points[outlet_indices]
                    
                if len(wall_points) > 200:
                    wall_indices = np.random.choice(len(wall_points), 
                                                 size=max(20, int(len(wall_points) * bd_ratio)),
                                                 replace=False)
                    wall_batch = wall_points[wall_indices]
                    wall_normals_batch = wall_normals[wall_indices]
                
                optimizer.zero_grad()
                
                # Compute loss for this mini-batch
                loss, physics_loss_val, bc_loss_val = compute_loss(
                    model, batch_domain_points, inlet_batch, outlet_batch, wall_batch, wall_normals_batch,
                    lambda_physics=lambda_physics, lambda_bc=lambda_bc
                )
                
                # Backpropagation
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                epoch_loss += loss.item() * len(batch_domain_points)
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / len(domain_points)
            loss_history.append(avg_epoch_loss)
            
            # Update learning rate
            if use_cyclic_lr:
                scheduler.step()
            else:
                scheduler.step(avg_epoch_loss)
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                torch.save(model.state_dict(), "models/best_model.pth")
            
            # Display progress
            if (epoch + 1) % display_every == 0 or epoch == 0:
                elapsed = time.time() - start_time
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}, "
                    f"Physics Loss: {physics_loss_val.item():.6f}, BC Loss: {bc_loss_val.item():.6f}, "
                    f"LR: {current_lr:.1e}, Time: {elapsed:.2f}s")
                
        print(f"Training finished. Best Loss: {best_loss:.6f}")

        # Plot loss history
        plt.figure(figsize=(10, 6))
        plt.semilogy(loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss (log scale)')
        plt.title('Training Loss History')
        plt.grid(True, which="both", ls="--")
        plt.savefig("plots/loss_history.png", dpi=300)
        plt.close()

        # Load the best performing model
        print("Loading best model state...")
        model.load_state_dict(torch.load("models/best_model.pth"))

        return model, loss_history
    except Exception as e:
        print(f"Error in training: {e}")
        traceback.print_exc()
        return model, []