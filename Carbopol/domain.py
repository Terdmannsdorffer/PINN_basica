# domain.py - CORRECTED VERSION
import numpy as np

# CONSISTENT dimensions across all files (RESTORED TO EXPERIMENTAL VALUES)
L_up = 0.097      # Upper horizontal length  
L_down = 0.174    # Total horizontal length
H_left = 0.119    # Left vertical height
H_right = 0.019   # RIGHT VERTICAL HEIGHT - RESTORED TO EXPERIMENTAL VALUE

def inside_L(x, y, W=None, L_v=None, L_h=None):
    """
    Corrected L-shaped domain that matches PIV experimental setup
    """
    # Main rectangle minus the corner rectangle
    main_rect = (0 <= x <= L_down) and (0 <= y <= H_left)
    corner_rect = (L_up <= x <= L_down) and (H_right <= y <= H_left)
    
    return main_rect and not corner_rect

def generate_domain_points():
    """Generate internal domain points"""
    print("Generating high-resolution internal points...")
    domain_points = []
    
    # Generate uniform points
    n_uniform = 2000  # Reduced for stability
    for _ in range(n_uniform):
        x = np.random.uniform(0, L_down)
        y = np.random.uniform(0, H_left)
        if inside_L(x, y):
            domain_points.append([x, y])
    
    # Generate additional adaptive points near walls and corner
    n_corner = 500  # Reduced
    n_walls = 500   # Reduced
    
    # Corner region (L-bend)
    corner_buffer = 0.015
    for _ in range(n_corner):
        x = np.random.uniform(L_up - corner_buffer, L_up + corner_buffer)
        y = np.random.uniform(H_right - corner_buffer, H_right + corner_buffer)
        if inside_L(x, y):
            domain_points.append([x, y])
    
    # Near walls
    wall_buffer = 0.010
    wall_segments_internal = [
        [(0, 0), (0, H_left)], 
        [(0, H_left), (L_up, H_left)],
        [(L_up, H_left), (L_up, H_right)], 
        [(L_up, H_right), (L_down, H_right)],
        [(L_down, H_right), (L_down, 0)], 
        [(L_down, 0), (0, 0)]
    ]
    
    for segment in wall_segments_internal:
        (x1, y1), (x2, y2) = segment
        for _ in range(n_walls // 6):
            t = np.random.uniform(0, 1)
            x_wall = x1 + t * (x2 - x1)
            y_wall = y1 + t * (y2 - y1)
            
            # Normal direction (inward)
            nx, ny = -(y2 - y1), (x2 - x1)
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 0:
                nx, ny = nx/norm, ny/norm
            
            # Random distance from wall
            r = np.random.exponential(scale=wall_buffer)
            x = x_wall + nx * r
            y = y_wall + ny * r
            
            if inside_L(x, y):
                domain_points.append([x, y])

    domain_points = np.array(domain_points)
    print(f"Generated {len(domain_points)} internal points")
    print(f"Domain dimensions: L_up={L_up}, L_down={L_down}, H_left={H_left}, H_right={H_right}")
    print(f"Area ratio (inlet/outlet): {(L_up * 1.0) / (H_right * 1.0):.2f} (EXPERIMENTAL)")
    
    return domain_points

def generate_boundary_points():
    """Generate boundary points for corrected L-shaped domain"""
    print("Generating boundary points...")
    
    # Define wall segments for corrected L-shape
    wall_segments = [
        # Left wall (bottom to top)
        [(0, 0), (0, H_left)],
        
        # Top wall (inlet, left to right) 
        [(0, H_left), (L_up, H_left)],
        
        # Right upper wall (top to step)
        [(L_up, H_left), (L_up, H_right)], 
        
        # Step wall (left to right)
        [(L_up, H_right), (L_down, H_right)],
        
        # Right wall (outlet, step to bottom)
        [(L_down, H_right), (L_down, 0)],
        
        # Bottom wall (right to left)
        [(L_down, 0), (0, 0)]
    ]

    wall_points = []
    wall_normals = []
    inlet_points = []
    outlet_points = []

    for i, ((x1, y1), (x2, y2)) in enumerate(wall_segments):
        t_vals = np.linspace(0, 1, 40)  # Reduced for stability
        for t in t_vals:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Calculate inward normal
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            normal = (-dy/length, dx/length) if length > 0 else (0, 0)
            
            if i == 1:  # Top wall (inlet)
                inlet_points.append([x, y])
            elif i == 4:  # Right wall (outlet) 
                outlet_points.append([x, y])
            else:  # Other walls
                wall_points.append([x, y])
                wall_normals.append(normal)

    wall_points = np.array(wall_points)
    wall_normals = np.array(wall_normals)
    inlet_points = np.array(inlet_points)
    outlet_points = np.array(outlet_points)

    print(f"Generated {len(wall_points)} wall points")
    print(f"Generated {len(inlet_points)} inlet points") 
    print(f"Generated {len(outlet_points)} outlet points")
    
    return wall_points, wall_normals, inlet_points, outlet_points, wall_segments