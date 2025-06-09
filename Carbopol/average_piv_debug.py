# averaged_piv_debug.py
"""
Create averaged PIV data from a range of files, maintaining exact PIV format structure
for seamless use with existing piv_pinn_comparison.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import re

def extract_frame_number(filename):
    """Extract frame number from PIV filename"""
    patterns = [
        r'PIVlab_(\d+)\.txt',
        r'PIV_(\d+)\.txt', 
        r'frame_(\d+)\.txt',
        r'time_(\d+)\.txt',
        r'(\d+)\.txt'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    return 0

def load_and_process_single_piv(filepath):
    """Load and process a single PIV file (same as debug_piv_data.py)"""
    try:
        # Read file and find header
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find header
        header_idx = None
        for i, line in enumerate(lines):
            if 'x [m]' in line and 'y [m]' in line:
                header_idx = i
                break
        
        if header_idx is None:
            print(f"Warning: No header found in {filepath.name}")
            return None
        
        # Load data
        df = pd.read_csv(filepath, skiprows=header_idx)
        
        # Check required columns
        required_cols = ['x [m]', 'y [m]', 'u [m/s]', 'v [m/s]']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} in {filepath.name}")
            return None
        
        # Clean data - same as debug_piv_data.py
        valid_mask = (
            np.isfinite(df['x [m]']) & 
            np.isfinite(df['y [m]']) & 
            np.isfinite(df['u [m/s]']) & 
            np.isfinite(df['v [m/s]'])
        )
        
        valid_df = df[valid_mask].copy()
        
        if len(valid_df) == 0:
            print(f"Warning: No valid data in {filepath.name}")
            return None
        
        return valid_df
        
    except Exception as e:
        print(f"Error loading {filepath.name}: {str(e)}")
        return None

def create_averaged_piv_file(piv_directory, start_frame=500, end_frame=1500, 
                           output_filename="averaged_piv_steady_state.txt"):
    """
    Create averaged PIV file from range of frames, maintaining exact PIV format
    """
    
    print(f"=== Creating Averaged PIV File from frames {start_frame}-{end_frame} ===")
    
    piv_dir = Path(piv_directory)
    
    # Find all PIV files
    piv_files = list(piv_dir.glob("*.txt"))
    
    # Sort files by frame number
    file_frame_pairs = []
    for piv_file in piv_files:
        frame_num = extract_frame_number(piv_file.name)
        file_frame_pairs.append((piv_file, frame_num))
    
    # Sort by frame number
    file_frame_pairs.sort(key=lambda x: x[1])
    
    print(f"Found {len(file_frame_pairs)} PIV files")
    
    # Filter files in the specified range
    files_in_range = []
    for piv_file, frame_num in file_frame_pairs:
        if start_frame <= frame_num <= end_frame:
            files_in_range.append(piv_file)
    
    print(f"Files in range {start_frame}-{end_frame}: {len(files_in_range)}")
    
    if len(files_in_range) == 0:
        raise ValueError(f"No files found in frame range {start_frame}-{end_frame}")
    
    # Load all files and collect data
    print("Loading PIV files...")
    all_dataframes = []
    successful_files = []
    
    for i, piv_file in enumerate(files_in_range):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing file {i+1}/{len(files_in_range)}: {piv_file.name}")
        
        df = load_and_process_single_piv(piv_file)
        if df is not None:
            all_dataframes.append(df)
            successful_files.append(piv_file.name)
    
    print(f"Successfully loaded: {len(all_dataframes)} files")
    print(f"Failed files: {len(files_in_range) - len(all_dataframes)}")
    
    if len(all_dataframes) == 0:
        raise ValueError("No valid files could be loaded!")
    
    # Find common coordinates across all files
    print("Finding common coordinate grid...")
    
    # Collect all unique coordinates
    all_coords = set()
    for df in all_dataframes:
        for _, row in df.iterrows():
            # Round coordinates to avoid floating point precision issues
            x_round = round(row['x [m]'], 6)
            y_round = round(row['y [m]'], 6)
            all_coords.add((x_round, y_round))
    
    print(f"Found {len(all_coords)} unique coordinate positions")
    
    # Create coordinate arrays
    coords_list = list(all_coords)
    coords_array = np.array(coords_list)
    
    # Initialize arrays for averaging
    n_coords = len(coords_list)
    u_values = np.full((len(all_dataframes), n_coords), np.nan)
    v_values = np.full((len(all_dataframes), n_coords), np.nan)
    
    # Fill velocity values for each file
    print("Collecting velocity data at common coordinates...")
    for file_idx, df in enumerate(all_dataframes):
        if (file_idx + 1) % 50 == 0:
            print(f"  Processing file {file_idx+1}/{len(all_dataframes)}")
        
        # Create coordinate lookup for this file
        for _, row in df.iterrows():
            x_round = round(row['x [m]'], 6)
            y_round = round(row['y [m]'], 6)
            
            # Find index in coordinate list
            try:
                coord_idx = coords_list.index((x_round, y_round))
                u_values[file_idx, coord_idx] = row['u [m/s]']
                v_values[file_idx, coord_idx] = row['v [m/s]']
            except ValueError:
                # Coordinate not found (shouldn't happen)
                continue
    
    # Calculate averages (ignoring NaN values)
    print("Computing averages...")
    u_avg = np.nanmean(u_values, axis=0)
    v_avg = np.nanmean(v_values, axis=0)
    
    # Count how many files contributed to each point
    valid_count = np.sum(~np.isnan(u_values), axis=0)
    
    # Remove points with insufficient data (require at least 10% of files)
    min_files = max(1, len(all_dataframes) // 10)
    good_points_mask = valid_count >= min_files
    
    print(f"Keeping {np.sum(good_points_mask)} points with >= {min_files} contributing files")
    
    # Create final dataframe with same structure as original PIV files
    final_coords = coords_array[good_points_mask]
    final_u = u_avg[good_points_mask]
    final_v = v_avg[good_points_mask]
    final_valid_count = valid_count[good_points_mask]
    
    # Create DataFrame with exact same column structure
    averaged_df = pd.DataFrame({
        'x [m]': final_coords[:, 0],
        'y [m]': final_coords[:, 1],
        'u [m/s]': final_u,
        'v [m/s]': final_v
    })
    
    # Remove any remaining NaN values
    final_valid_mask = (
        np.isfinite(averaged_df['x [m]']) & 
        np.isfinite(averaged_df['y [m]']) & 
        np.isfinite(averaged_df['u [m/s]']) & 
        np.isfinite(averaged_df['v [m/s]'])
    )
    
    averaged_df = averaged_df[final_valid_mask].copy()
    
    print(f"Final averaged data: {len(averaged_df)} points")
    
    # Create the output file with EXACT same format as original PIV files
    output_path = Path(output_filename)
    
    # Write file exactly like original PIV files (comma-separated)
    with open(output_path, 'w') as f:
        # Write minimal header (like the original but without extra columns)
        f.write('x [m],y [m],u [m/s],v [m/s]\n')
        
        # Write data
        for _, row in averaged_df.iterrows():
            f.write(f"{row['x [m]']:.6f},{row['y [m]']:.6f},{row['u [m/s]']:.8f},{row['v [m/s]']:.8f}\n")
    
    print(f"✅ Averaged PIV file saved: {output_path}")
    print(f"   - Frames used: {start_frame} to {end_frame}")
    print(f"   - Files processed: {len(all_dataframes)}")
    print(f"   - Final data points: {len(averaged_df)}")
    
    # Run the same analysis as debug_piv_data.py on the averaged file
    print(f"\n=== Analyzing Averaged PIV File (like debug_piv_data.py) ===")
    debug_averaged_piv_file(output_path, averaged_df, len(all_dataframes), start_frame, end_frame)
    
    return str(output_path), averaged_df

def debug_averaged_piv_file(filepath, df, n_files_used, start_frame, end_frame):
    """Run same analysis as debug_piv_data.py but for averaged file"""
    
    print(f"=== Debugging Averaged PIV file: {filepath} ===")
    print(f"Created from {n_files_used} files (frames {start_frame}-{end_frame})")
    
    # Check required columns (should always be present since we created it)
    required_cols = ['x [m]', 'y [m]', 'u [m/s]', 'v [m/s]']
    print(f"Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Data quality analysis (should be all finite since we cleaned it)
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
    
    # All rows should be valid since we cleaned the data
    valid_mask = (
        np.isfinite(df['x [m]']) & 
        np.isfinite(df['y [m]']) & 
        np.isfinite(df['u [m/s]']) & 
        np.isfinite(df['v [m/s]'])
    )
    valid_count = valid_mask.sum()
    print(f"Completely valid rows: {valid_count}/{len(df)} ({valid_count/len(df)*100:.2f}%)")
    
    # Analyze the data
    print("\n=== Averaged Data Analysis ===")
    print(f"Spatial domain:")
    print(f"  x: [{df['x [m]'].min():.4f}, {df['x [m]'].max():.4f}] m")
    print(f"  y: [{df['y [m]'].min():.4f}, {df['y [m]'].max():.4f}] m")
    
    print(f"Velocity ranges:")
    print(f"  u: [{df['u [m/s]'].min():.6f}, {df['u [m/s]'].max():.6f}] m/s")
    print(f"  v: [{df['v [m/s]'].min():.6f}, {df['v [m/s]'].max():.6f}] m/s")
    
    magnitude = np.sqrt(df['u [m/s]']**2 + df['v [m/s]']**2)
    print(f"  magnitude: [{magnitude.min():.6f}, {magnitude.max():.6f}] m/s")
    print(f"  mean magnitude: {magnitude.mean():.6f} m/s")
    
    # Create diagnostic plots (same as debug_piv_data.py)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Scatter plot of all points
    axes[0,0].scatter(df['x [m]'], df['y [m]'], c='blue', s=1, alpha=0.7, label=f'Averaged points ({n_files_used} files)')
    axes[0,0].set_xlabel('x [m]')
    axes[0,0].set_ylabel('y [m]')
    axes[0,0].set_title(f'Averaged Data Point Distribution\n(Frames {start_frame}-{end_frame})')
    axes[0,0].legend()
    axes[0,0].axis('equal')
    
    # U velocity distribution
    axes[0,1].hist(df['u [m/s]'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('u [m/s]')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Averaged U-velocity Distribution')
    axes[0,1].grid(True, alpha=0.3)
    
    # V velocity distribution
    axes[0,2].hist(df['v [m/s]'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,2].set_xlabel('v [m/s]')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].set_title('Averaged V-velocity Distribution')
    axes[0,2].grid(True, alpha=0.3)
    
    # Magnitude distribution
    axes[1,0].hist(magnitude, bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Magnitude [m/s]')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Averaged Velocity Magnitude Distribution')
    axes[1,0].grid(True, alpha=0.3)
    
    # Vector field (subsampled)
    step = max(1, len(df) // 1000)  # Subsample for visualization
    sub_df = df.iloc[::step]
    q = axes[1,1].quiver(sub_df['x [m]'], sub_df['y [m]'], 
                        sub_df['u [m/s]'], sub_df['v [m/s]'],
                        magnitude.iloc[::step], cmap='viridis', scale=10)
    axes[1,1].set_xlabel('x [m]')
    axes[1,1].set_ylabel('y [m]')
    axes[1,1].set_title('Averaged Vector Field (Subsampled)')
    axes[1,1].axis('equal')
    
    # U vs V scatter
    axes[1,2].scatter(df['u [m/s]'], df['v [m/s]'], 
                     c=magnitude, cmap='viridis', s=1, alpha=0.6)
    axes[1,2].set_xlabel('u [m/s]')
    axes[1,2].set_ylabel('v [m/s]')
    axes[1,2].set_title('Averaged U vs V Velocity')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Summary ===")
    print(f"✅ Averaged PIV file created successfully!")
    print(f"📊 {len(df)} data points from {n_files_used} files")
    print(f"📁 File: {filepath}")
    print(f"🎯 Ready for PINN comparison with existing piv_pinn_comparison.py!")
    print(f"\n   Next step: python piv_pinn_comparison.py '{filepath}' 'your_pinn_file.npz'")

if __name__ == "__main__":
    # Configuration - Update these paths
    piv_directory = "PIV/PIVs_txts/p1"  # Your PIV directory
    start_frame = 900   # Skip startup transients  
    end_frame = 1000    # Skip end transients
    output_file = "averaged_piv_steady_state.txt"
    
    # Create averaged PIV file
    averaged_file_path, averaged_data = create_averaged_piv_file(
        piv_directory=piv_directory,
        start_frame=start_frame,
        end_frame=end_frame,
        output_filename=output_file
    )