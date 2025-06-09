# debug_piv_data.py
"""
Debug tool to understand PIV data structure and identify issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def debug_piv_file(filepath):
    """Debug a single PIV file to understand its structure"""
    print(f"=== Debugging PIV file: {filepath} ===")
    
    # Read file and find header
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    
    # Find header
    header_idx = None
    for i, line in enumerate(lines):
        if 'x [m]' in line and 'y [m]' in line:
            header_idx = i
            print(f"Header found at line {i}: {line.strip()}")
            break
    
    if header_idx is None:
        print("ERROR: Could not find header line!")
        return
    
    # Load data
    try:
        df = pd.read_csv(filepath, skiprows=header_idx)
        print(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        return
    
    # Check required columns
    required_cols = ['x [m]', 'y [m]', 'u [m/s]', 'v [m/s]']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return
    
    # Data quality analysis
    print("\n=== Data Quality Analysis ===")
    for col in required_cols:
        data = df[col]
        nan_count = data.isna().sum()
        inf_count = np.isinf(data).sum()
        finite_count = np.isfinite(data).sum()
        
        print(f"{col}:")
        print(f"  Total values: {len(data)}")
        print(f"  NaN values: {nan_count} ({nan_count/len(data)*100:.2f}%)")
        print(f"  Infinite values: {inf_count} ({inf_count/len(data)*100:.2f}%)")
        print(f"  Finite values: {finite_count} ({finite_count/len(data)*100:.2f}%)")
        
        if finite_count > 0:
            finite_data = data[np.isfinite(data)]
            print(f"  Range: [{finite_data.min():.6f}, {finite_data.max():.6f}]")
            print(f"  Mean: {finite_data.mean():.6f}")
            print(f"  Std: {finite_data.std():.6f}")
        print()
    
    # Check for completely valid rows
    valid_mask = (
        np.isfinite(df['x [m]']) & 
        np.isfinite(df['y [m]']) & 
        np.isfinite(df['u [m/s]']) & 
        np.isfinite(df['v [m/s]'])
    )
    valid_count = valid_mask.sum()
    print(f"Completely valid rows: {valid_count}/{len(df)} ({valid_count/len(df)*100:.2f}%)")
    
    if valid_count == 0:
        print("ERROR: No completely valid rows found!")
        return
    
    # Analyze valid data
    valid_df = df[valid_mask]
    
    print("\n=== Valid Data Analysis ===")
    print(f"Spatial domain:")
    print(f"  x: [{valid_df['x [m]'].min():.4f}, {valid_df['x [m]'].max():.4f}] m")
    print(f"  y: [{valid_df['y [m]'].min():.4f}, {valid_df['y [m]'].max():.4f}] m")
    
    print(f"Velocity ranges:")
    print(f"  u: [{valid_df['u [m/s]'].min():.6f}, {valid_df['u [m/s]'].max():.6f}] m/s")
    print(f"  v: [{valid_df['v [m/s]'].min():.6f}, {valid_df['v [m/s]'].max():.6f}] m/s")
    
    magnitude = np.sqrt(valid_df['u [m/s]']**2 + valid_df['v [m/s]']**2)
    print(f"  magnitude: [{magnitude.min():.6f}, {magnitude.max():.6f}] m/s")
    print(f"  mean magnitude: {magnitude.mean():.6f} m/s")
    
    # Create diagnostic plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Scatter plot of all points
    axes[0,0].scatter(df['x [m]'], df['y [m]'], c='red', s=1, alpha=0.5, label='All points')
    axes[0,0].scatter(valid_df['x [m]'], valid_df['y [m]'], c='blue', s=1, alpha=0.7, label='Valid points')
    axes[0,0].set_xlabel('x [m]')
    axes[0,0].set_ylabel('y [m]')
    axes[0,0].set_title('Data Point Distribution')
    axes[0,0].legend()
    axes[0,0].axis('equal')
    
    # U velocity distribution
    axes[0,1].hist(valid_df['u [m/s]'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('u [m/s]')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('U-velocity Distribution')
    axes[0,1].grid(True, alpha=0.3)
    
    # V velocity distribution
    axes[0,2].hist(valid_df['v [m/s]'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,2].set_xlabel('v [m/s]')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].set_title('V-velocity Distribution')
    axes[0,2].grid(True, alpha=0.3)
    
    # Magnitude distribution
    axes[1,0].hist(magnitude, bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Magnitude [m/s]')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Velocity Magnitude Distribution')
    axes[1,0].grid(True, alpha=0.3)
    
    # Vector field (subsampled)
    step = max(1, len(valid_df) // 1000)  # Subsample for visualization
    sub_df = valid_df.iloc[::step]
    axes[1,1].quiver(sub_df['x [m]'], sub_df['y [m]'], 
                    sub_df['u [m/s]'], sub_df['v [m/s]'],
                    magnitude.iloc[::step], cmap='viridis', scale=10)
    axes[1,1].set_xlabel('x [m]')
    axes[1,1].set_ylabel('y [m]')
    axes[1,1].set_title('Vector Field (Subsampled)')
    axes[1,1].axis('equal')
    
    # U vs V scatter
    axes[1,2].scatter(valid_df['u [m/s]'], valid_df['v [m/s]'], 
                     c=magnitude, cmap='viridis', s=1, alpha=0.6)
    axes[1,2].set_xlabel('u [m/s]')
    axes[1,2].set_ylabel('v [m/s]')
    axes[1,2].set_title('U vs V Velocity')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return valid_df

if __name__ == "__main__":
    # Debug your PIV file
    piv_file = "PIV/PIVs_txts/p1/PIVlab_0901.txt"  # Update this path
    
    valid_data = debug_piv_file(piv_file)
    
    if valid_data is not None:
        print(f"\n=== Summary ===")
        print(f"File can be processed with {len(valid_data)} valid data points")
        print("Ready for PINN comparison!")
    else:
        print("\n=== Summary ===")
        print("File has issues that need to be resolved before comparison")