# domain.py - Simplified domain generation
import numpy as np

# Domain dimensions
L_up = 0.097      # Upper horizontal length  
L_down = 0.174    # Total horizontal length
H_left = 0.119    # Left vertical height
H_right = 0.019   # Right vertical height

def inside_L(x, y):
    """Check if point is inside L-shaped domain"""
    main_rect = (0 <= x <= L_down) and (0 <= y <= H_left)
    corner_rect = (L_up <= x <= L_down) and (H_right <= y <= H_left)
    return main_rect and not corner_rect

def generate_domain_points(n_points=2000):
    """Generate domain points for training"""
    points = []
    
    # Uniform random sampling
    while len(points) < n_points:
        x = np.random.uniform(0, L_down)
        y = np.random.uniform(0, H_left)
        if inside_L(x, y):
            points.append([x, y])
    
    return np.array(points)

def generate_boundary_points():
    """Generate boundary points"""
    
    # Wall segments
    wall_segments = [
        [(0, 0), (0, H_left)],           # Left wall
        [(0, H_left), (L_up, H_left)],   # Top wall (inlet)
        [(L_up, H_left), (L_up, H_right)], # Step vertical
        [(L_up, H_right), (L_down, H_right)], # Step horizontal
        [(L_down, H_right), (L_down, 0)], # Right wall (outlet)
        [(L_down, 0), (0, 0)]            # Bottom wall
    ]

    wall_points = []
    wall_normals = []
    inlet_points = []
    outlet_points = []

    for i, ((x1, y1), (x2, y2)) in enumerate(wall_segments):
        t_vals = np.linspace(0, 1, 30)
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

    return (np.array(wall_points), np.array(wall_normals), 
            np.array(inlet_points), np.array(outlet_points), wall_segments)