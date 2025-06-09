# velocity_scaling_analysis.py
"""
Analysis of velocity scaling mismatch and proposed fixes
"""

import numpy as np

def analyze_velocity_mismatch():
    """Analyze the velocity scaling mismatch"""
    
    print("=== VELOCITY SCALING ANALYSIS ===")
    
    # Current results
    piv_mean_velocity = 0.0234  # From your PIV data (typical magnitude)
    pinn_mean_velocity = 0.0110  # From PINN extraction
    pinn_max_velocity = 0.0442   # From PINN extraction
    
    # Boundary condition targets vs actual
    inlet_target_v = -0.015
    inlet_actual_v = -0.012107
    outlet_target_u = 0.0766
    outlet_actual_u = 0.066137
    
    print(f"Boundary Condition Analysis:")
    print(f"  Inlet v:  target={inlet_target_v:.4f}, actual={inlet_actual_v:.4f}, ratio={inlet_actual_v/inlet_target_v:.3f}")
    print(f"  Outlet u: target={outlet_target_u:.4f}, actual={outlet_actual_u:.4f}, ratio={outlet_actual_u/outlet_target_u:.3f}")
    
    # The issue: PINN is achieving ~80-86% of target velocities
    avg_bc_satisfaction = (abs(inlet_actual_v/inlet_target_v) + abs(outlet_actual_u/outlet_target_u)) / 2
    print(f"  Average BC satisfaction: {avg_bc_satisfaction:.1%}")
    
    # But PIV shows much smaller velocities
    velocity_scale_factor = pinn_mean_velocity / (piv_mean_velocity / 4.26)  # Correcting for current 4.26x error
    print(f"\nVelocity Scale Analysis:")
    print(f"  PIV mean velocity: {piv_mean_velocity:.4f} m/s")
    print(f"  PINN mean velocity: {pinn_mean_velocity:.4f} m/s") 
    print(f"  Current scale factor: {4.26:.2f}x too large")
    
    # Proposed fixes
    print(f"\n=== PROPOSED FIXES ===")
    
    # Fix 1: Reduce inlet velocity to match experimental conditions
    proposed_inlet_v = inlet_target_v * (1/4.26)  # Scale down by observed factor
    proposed_outlet_u = outlet_target_u * (1/4.26)
    
    print(f"Fix 1 - Scale Down Inlet Velocity:")
    print(f"  New inlet v: {proposed_inlet_v:.6f} m/s (was {inlet_target_v:.4f})")
    print(f"  New outlet u: {proposed_outlet_u:.6f} m/s (was {outlet_target_u:.4f})")
    
    # Fix 2: Check if Reynolds number matches
    # Re = ρVD/μ
    rho = 0.101972  # kg/m³
    characteristic_length = 0.019  # Outlet height as characteristic length
    
    # Effective viscosity for Carbopol at low shear rates
    tau_y, k, n = 30.0, 2.8, 0.65
    shear_rate_low = 0.1  # 1/s (typical for low velocity flow)
    eta_eff = (tau_y / shear_rate_low) + k * (shear_rate_low ** (n - 1))
    
    Re_pinn = rho * pinn_max_velocity * characteristic_length / eta_eff
    Re_piv = rho * (piv_mean_velocity) * characteristic_length / eta_eff
    
    print(f"\nReynolds Number Analysis:")
    print(f"  Effective viscosity: {eta_eff:.2f} Pa·s")
    print(f"  Re (PINN): {Re_pinn:.6f}")
    print(f"  Re (PIV):  {Re_piv:.6f}")
    print(f"  Ratio: {Re_pinn/Re_piv:.2f}")
    
    return proposed_inlet_v, proposed_outlet_u

def create_scaled_training_parameters():
    """Create corrected training parameters with proper velocity scaling"""
    
    proposed_inlet_v, proposed_outlet_u = analyze_velocity_mismatch()
    
    training_code = f'''
# corrected_training_parameters.py
# Use these corrected parameters in your training function

def get_corrected_flow_parameters():
    """
    Corrected flow parameters based on PIV comparison analysis
    """
    # Geometry parameters (unchanged)
    L_up, L_down = 0.097, 0.174
    H_left, H_right = 0.119, 0.019
    
    # CORRECTED: Scale down velocities to match experimental observations
    # Based on 4.26x velocity mismatch analysis
    
    # Original inlet velocity was too high by factor of ~4.26
    u_in = 0.0      # No horizontal component at inlet
    v_in = {proposed_inlet_v:.6f}   # Scaled down from -0.015 m/s
    
    # Outlet velocity from continuity equation with corrected inlet
    inlet_area = L_up    # Width of top inlet  
    outlet_area = H_right # Height of right outlet
    u_out = abs(v_in) * (inlet_area / outlet_area)  # = {proposed_outlet_u:.6f} m/s
    v_out = 0.0     # No vertical component at outlet
    
    print(f"VELOCITY-CORRECTED Flow parameters:")
    print(f"  Inlet (TOP):    u={{u_in:.6f}} m/s, v={{v_in:.6f}} m/s")
    print(f"  Outlet (RIGHT): u={{u_out:.6f}} m/s, v={{v_out:.6f}} m/s")
    print(f"  Velocity scale: {{u_out/abs(v_in):.2f}}x")
    
    return u_in, v_in, u_out, v_out

# Additional improvements for better convergence:

def improved_training_strategy():
    """
    Enhanced training strategy for better PIV matching
    """
    
    # 1. Even longer boundary condition training
    stage1_epochs = 3000  # Increased from 1500
    
    # 2. More conservative physics introduction
    max_physics_weight = 0.01  # Reduced from 0.05
    
    # 3. Better regularization
    regularization_params = {{
        'gradient_clip': 0.1,      # Tighter clipping
        'weight_decay': 1e-6,      # L2 regularization
        'dropout_rate': 0.1        # Add dropout to model
    }}
    
    # 4. Adaptive boundary condition weights
    adaptive_bc_weights = {{
        'inlet_weight': 100.0,     # Very high for critical inlet BC
        'outlet_weight': 50.0,     # High for outlet BC  
        'wall_weight': 20.0,       # Moderate for walls
        'pressure_weight': 0.1     # Low for pressure (reference)
    }}
    
    return stage1_epochs, max_physics_weight, regularization_params, adaptive_bc_weights

# Usage in your training function:
# u_in, v_in, u_out, v_out = get_corrected_flow_parameters()
'''
    
    print(f"\n=== TRAINING CODE TEMPLATE ===")
    print(training_code)
    
    return training_code

if __name__ == "__main__":
    analyze_velocity_mismatch()
    create_scaled_training_parameters()