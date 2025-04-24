import numpy as np

# Create a basic L-shaped domain
def inside_L(x, y, W=0.5, L_v=2.0, L_h=3.0):
    vertical = (-W/2 <= x <= W/2) and (-W/2 <= y <= L_v)
    horizontal = (-W/2 <= x <= L_h) and (-W/2 <= y <= W/2)
    return vertical or horizontal

# Generate a few internal points
def generate_domain_points():
    print("Generating internal points...")
    domain_points = []
    for _ in range(200):
        x = np.random.uniform(-0.5, 3.0)
        y = np.random.uniform(-0.5, 2.0)
        if inside_L(x, y):
            domain_points.append([x, y])

    domain_points = np.array(domain_points)
    print(f"Generated {len(domain_points)} internal points")
    return domain_points

# Generate boundary points
def generate_boundary_points():
    print("Defining walls...")
    W = 0.5
    L_v = 2.0
    L_h = 3.0
    wall_segments = [
        [(-W/2, -W/2), (-W/2, L_v)],      # Left wall
        [(-W/2, L_v), (W/2, L_v)],        # Top wall (inlet)
        [(W/2, L_v), (W/2, W/2)],         # Right upper wall
        [(W/2, W/2), (L_h, W/2)],         # Top horizontal wall
        [(L_h, W/2), (L_h, -W/2)],        # Right wall (outlet)
        [(L_h, -W/2), (-W/2, -W/2)]       # Bottom wall
    ]

    # Generate a few boundary points 
    print("Generating boundary points...")
    wall_points = []
    wall_normals = []
    inlet_points = []
    outlet_points = []

    for i, ((x1, y1), (x2, y2)) in enumerate(wall_segments):
        # Generate 10 points per segment
        t_vals = np.linspace(0, 1, 10)
        for t in t_vals:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Calculate normal (90° counterclockwise)
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