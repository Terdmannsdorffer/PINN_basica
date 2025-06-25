# domain.py - ENHANCED VERSION with Better Point Distribution
import numpy as np

# CONSISTENT dimensions across all files (EXPERIMENTAL VALUES)
L_up = 0.097      # Upper horizontal length  
L_down = 0.174    # Total horizontal length
H_left = 0.119    # Left vertical height
H_right = 0.019   # RIGHT VERTICAL HEIGHT - EXPERIMENTAL VALUE

def inside_L(x, y, W=None, L_v=None, L_h=None):
    """
    L-shaped domain that matches PIV experimental setup
    """
    # Main rectangle minus the corner rectangle
    main_rect = (0 <= x <= L_down) and (0 <= y <= H_left)
    corner_rect = (L_up <= x <= L_down) and (H_right <= y <= H_left)
    
    return main_rect and not corner_rect

def generate_domain_points(n_total=3000, adaptive_sampling=True):
    """
    Enhanced domain point generation with better distribution for training
    
    Args:
        n_total: Total number of domain points
        adaptive_sampling: Use adaptive sampling near boundaries and corners
    """
    print(f"Generating {n_total} domain points with enhanced distribution...")
    domain_points = []
    
    if adaptive_sampling:
        # ENHANCEMENT: Better point distribution
        n_uniform = int(0.6 * n_total)      # 60% uniform
        n_corner = int(0.2 * n_total)       # 20% near corner
        n_walls = int(0.15 * n_total)       # 15% near walls  
        n_inlet_outlet = int(0.05 * n_total) # 5% near inlet/outlet
    else:
        n_uniform = n_total
        n_corner = n_walls = n_inlet_outlet = 0
    
    # 1. Uniform points
    attempts = 0
    while len(domain_points) < n_uniform and attempts < n_uniform * 3:
        x = np.random.uniform(0, L_down)
        y = np.random.uniform(0, H_left)
        if inside_L(x, y):
            domain_points.append([x, y])
        attempts += 1
    
    if adaptive_sampling:
        # 2. Corner region (L-bend) - critical for flow physics
        corner_buffer = 0.020  # 2cm around corner
        corner_attempts = 0
        corner_added = 0
        while corner_added < n_corner and corner_attempts < n_corner * 3:
            x = np.random.uniform(L_up - corner_buffer, L_up + corner_buffer)
            y = np.random.uniform(H_right - corner_buffer, H_right + corner_buffer)
            if inside_L(x, y):
                domain_points.append([x, y])
                corner_added += 1
            corner_attempts += 1
        
        # 3. Near walls - important for boundary layer
        wall_buffer = 0.012  # 1.2cm from walls
        wall_segments_internal = [
            [(0, 0), (0, H_left)],           # Left wall
            [(0, H_left), (L_up, H_left)],   # Top wall (inlet)
            [(L_up, H_left), (L_up, H_right)],  # Internal corner wall
            [(L_up, H_right), (L_down, H_right)], # Step wall
            [(L_down, H_right), (L_down, 0)],     # Right wall (outlet)
            [(L_down, 0), (0, 0)]                 # Bottom wall
        ]
        
        wall_added = 0
        for segment in wall_segments_internal:
            (x1, y1), (x2, y2) = segment
            points_per_segment = 2 * (n_walls // len(wall_segments_internal))  # doble de resolución
            decay_scale = wall_buffer / 5  # más fino cerca de la pared
            
            for _ in range(points_per_segment):
                if wall_added >= n_walls:
                    break
                    
                # Random point along segment
                t = np.random.uniform(0, 1)
                x_wall = x1 + t * (x2 - x1)
                y_wall = y1 + t * (y2 - y1)
                
                # Normal direction (inward)
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    nx, ny = -dy/length, dx/length
                else:
                    nx, ny = 0, 0
                
                # Exponential distance from wall (more points closer to wall)
                r = np.random.exponential(scale=decay_scale)
                r = min(r, wall_buffer)
                
                x = x_wall + nx * r
                y = y_wall + ny * r
                
                if inside_L(x, y):
                    domain_points.append([x, y])
                    wall_added += 1
        
        # 4. Near inlet and outlet regions - important for boundary conditions
        inlet_outlet_added = 0
        
        # Near inlet (top boundary)
        inlet_buffer = 0.015
        for _ in range(n_inlet_outlet // 2):
            if inlet_outlet_added >= n_inlet_outlet:
                break
            x = np.random.uniform(0, L_up)
            y = np.random.uniform(H_left - inlet_buffer, H_left)
            if inside_L(x, y):
                domain_points.append([x, y])
                inlet_outlet_added += 1
        
        # Near outlet (right boundary)
        outlet_buffer = 0.015
        for _ in range(n_inlet_outlet // 2):
            if inlet_outlet_added >= n_inlet_outlet:
                break
            x = np.random.uniform(L_down - outlet_buffer, L_down)
            y = np.random.uniform(0, H_right)
            if inside_L(x, y):
                domain_points.append([x, y])
                inlet_outlet_added += 1
    # 5. Alta resolución justo en la salida (OUTLET)
    outlet_dense_x = np.linspace(L_down - 0.004, L_down - 0.0005, 12)
    outlet_dense_y = np.linspace(0.0005, H_right - 0.0005, 10)
    outlet_enhanced = []

    for x in outlet_dense_x:
        for y in outlet_dense_y:
            if inside_L(x, y):
                domain_points.append([x, y])
                outlet_enhanced.append([x, y])

    print(f"  Puntos añadidos con alta resolución en salida: {len(outlet_enhanced)}")

    domain_points = np.array(domain_points)
    
    print(f"Generated {len(domain_points)} internal points")
    print(f"Domain dimensions: L_up={L_up}, L_down={L_down}, H_left={H_left}, H_right={H_right}")
    print(f"Domain area: {calculate_domain_area():.6f} m²")
    print(f"Area ratio (inlet/outlet): {(L_up * 1.0) / (H_right * 1.0):.2f}")
    
    if adaptive_sampling:
        print(f"Point distribution:")
        print(f"  Uniform: {n_uniform} ({n_uniform/len(domain_points)*100:.1f}%)")
        print(f"  Corner region: {corner_added} ({corner_added/len(domain_points)*100:.1f}%)")
        print(f"  Near walls: {wall_added} ({wall_added/len(domain_points)*100:.1f}%)")
        print(f"  Near inlet/outlet: {inlet_outlet_added} ({inlet_outlet_added/len(domain_points)*100:.1f}%)")
    
    return domain_points

def generate_boundary_points(n_boundary_per_segment=50):
    """
    Enhanced boundary point generation with better resolution
    
    Args:
        n_boundary_per_segment: Number of points per boundary segment
    """
    print(f"Generating boundary points ({n_boundary_per_segment} per segment)...")
    
    # Define wall segments for L-shaped domain
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
        # ENHANCEMENT: Better point spacing along boundaries
        if i == 1:  # Inlet - higher resolution
            n_points = int(n_boundary_per_segment * 1.5)
        elif i == 4:  # Outlet - higher resolution  
            n_points = int(n_boundary_per_segment * 1.5)
        elif i == 2:  # Corner wall - higher resolution
            n_points = int(n_boundary_per_segment * 1.2)
        else:
            n_points = n_boundary_per_segment
        
        t_vals = np.linspace(0, 1, n_points)
        
        for t in t_vals:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Calculate inward normal
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                normal = (-dy/length, dx/length)
            else:
                normal = (0, 0)
            
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

    print(f"Generated boundary points:")
    print(f"  Wall points: {len(wall_points)}")
    print(f"  Inlet points: {len(inlet_points)} (top boundary)") 
    print(f"  Outlet points: {len(outlet_points)} (right boundary)")
    print(f"  Total boundary points: {len(wall_points) + len(inlet_points) + len(outlet_points)}")
    
    return wall_points, wall_normals, inlet_points, outlet_points, wall_segments

def calculate_domain_area():
    """Calculate the area of the L-shaped domain"""
    # Total rectangle minus corner rectangle
    total_area = L_down * H_left
    corner_area = (L_down - L_up) * (H_left - H_right)
    return total_area - corner_area

def get_domain_info():
    """Get domain information for reference"""
    return {
        'dimensions': {
            'L_up': L_up,
            'L_down': L_down, 
            'H_left': H_left,
            'H_right': H_right
        },
        'areas': {
            'total_domain': calculate_domain_area(),
            'inlet_area': L_up * 1.0,  # Inlet width × depth
            'outlet_area': H_right * 1.0,  # Outlet height × depth
            'area_ratio': L_up / H_right
        },
        'characteristic_length': np.sqrt(calculate_domain_area()),
        'aspect_ratios': {
            'overall': L_down / H_left,
            'inlet_outlet': L_up / H_right
        }
    }

def validate_point_in_domain(x, y):
    """Validate that a point is inside the domain with detailed checking"""
    if not (0 <= x <= L_down and 0 <= y <= H_left):
        return False, "Outside bounding rectangle"
    
    if L_up <= x <= L_down and H_right <= y <= H_left:
        return False, "Inside corner cutout"
    
    return True, "Valid point"

# ENHANCEMENT: Add domain visualization helper
def plot_domain_schematic(save_path=None, show_dimensions=True):
    """Create a schematic plot of the domain with dimensions"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Draw main rectangle
    main_rect = patches.Rectangle((0, 0), L_down, H_left, 
                                 linewidth=2, edgecolor='black', 
                                 facecolor='lightblue', alpha=0.5)
    ax.add_patch(main_rect)
    
    # Draw corner cutout (white)
    corner_rect = patches.Rectangle((L_up, H_right), L_down-L_up, H_left-H_right,
                                   linewidth=2, edgecolor='black',
                                   facecolor='white', alpha=1.0)
    ax.add_patch(corner_rect)
    
    # Mark inlet and outlet
    inlet_line = patches.Rectangle((0, H_left-0.002), L_up, 0.004,
                                  facecolor='green', alpha=0.8)
    ax.add_patch(inlet_line)
    
    outlet_line = patches.Rectangle((L_down-0.002, 0), 0.004, H_right,
                                   facecolor='red', alpha=0.8)
    ax.add_patch(outlet_line)
    
    if show_dimensions:
        # Add dimension annotations
        ax.annotate('', xy=(0, -0.01), xytext=(L_up, -0.01),
                   arrowprops=dict(arrowstyle='<->', color='blue'))
        ax.text(L_up/2, -0.015, f'L_up = {L_up:.3f} m', 
               ha='center', va='top', color='blue', fontweight='bold')
        
        ax.annotate('', xy=(L_up, -0.01), xytext=(L_down, -0.01),
                   arrowprops=dict(arrowstyle='<->', color='blue'))
        ax.text((L_up+L_down)/2, -0.015, f'L_down-L_up = {L_down-L_up:.3f} m', 
               ha='center', va='top', color='blue', fontweight='bold')
        
        ax.annotate('', xy=(-0.01, 0), xytext=(-0.01, H_right),
                   arrowprops=dict(arrowstyle='<->', color='blue'))
        ax.text(-0.015, H_right/2, f'H_right = {H_right:.3f} m', 
               ha='right', va='center', color='blue', fontweight='bold', rotation=90)
        
        ax.annotate('', xy=(-0.01, H_right), xytext=(-0.01, H_left),
                   arrowprops=dict(arrowstyle='<->', color='blue'))
        ax.text(-0.015, (H_right+H_left)/2, f'H_left-H_right = {H_left-H_right:.3f} m', 
               ha='right', va='center', color='blue', fontweight='bold', rotation=90)
    
    # Labels
    ax.text(L_up/2, H_left+0.005, 'INLET', ha='center', va='bottom', 
           color='green', fontweight='bold', fontsize=12)
    ax.text(L_down+0.005, H_right/2, 'OUTLET', ha='left', va='center',
           color='red', fontweight='bold', fontsize=12, rotation=90)
    
    # Domain info
    info_text = f"""Domain Information:
Area: {calculate_domain_area():.6f} m²
Area Ratio: {L_up/H_right:.2f}
Inlet Area: {L_up:.6f} m²
Outlet Area: {H_right:.6f} m²"""
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax.set_xlim(-0.03, L_down+0.03)
    ax.set_ylim(-0.03, H_left+0.03)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('L-Shaped Domain Schematic (PIV Experimental Dimensions)', 
                fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Domain schematic saved: {save_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Test the domain generation
    print("Testing enhanced domain generation...")
    
    # Generate points
    domain_pts = generate_domain_points(n_total=2000, adaptive_sampling=True)
    wall_pts, wall_norms, inlet_pts, outlet_pts, wall_segs = generate_boundary_points()
    
    # Print domain info
    info = get_domain_info()
    print(f"\nDomain Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Create schematic
    plot_domain_schematic("domain_schematic.png")