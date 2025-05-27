#piv_vizualisation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from pathlib import Path
import glob
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import time

class PIVVisualizer:
    def __init__(self, data_path=None):
        """Initialize PIV Visualizer with optional data path"""
        self.data = None
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, filepath):
        """Load PIV data from file - more robust version"""
        self.filepath = filepath
        
        # Try different approaches to find the data start
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find the header line (contains column names)
        header_idx = None
        for i, line in enumerate(lines):
            if 'x [m]' in line and 'y [m]' in line:
                header_idx = i
                break
        
        if header_idx is None:
            raise ValueError(f"Could not find header in {filepath}")
        
        # Read the data starting from the header
        self.data = pd.read_csv(filepath, skiprows=header_idx)
        
        # Try to extract metadata from the first lines
        self.metadata = {}
        try:
            for i in range(min(5, header_idx)):  # Check first few lines
                line = lines[i]
                if 'FRAME:' in line:
                    self.metadata['frame'] = line.split('FRAME:')[1].split(',')[0].strip()
                if 'conversion factor xy' in line:
                    self.metadata['conversion_xy'] = float(line.split('conversion factor xy (px -> m):')[1].split(',')[0].strip())
                if 'conversion factor uv' in line:
                    self.metadata['conversion_uv'] = float(line.split('conversion factor uv (px/frame -> m/s):')[1].strip())
        except:
            # If metadata extraction fails, use defaults
            self.metadata = {
                'frame': 'Unknown',
                'conversion_xy': 1.0,
                'conversion_uv': 1.0
            }
        
        # Ensure we have the necessary columns
        required_columns = ['x [m]', 'y [m]', 'u [m/s]', 'v [m/s]']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def plot_vector_field(self, scale=1, density=1, figsize=(12, 8), save_path=None, 
                          vmin=0, vmax=0.1, invert_y=True):
        """Plot velocity vector field with consistent scaling"""
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        
        # Subsample data for clearer visualization if density < 1
        step = int(1/density) if density < 1 else 1
        
        x = self.data['x [m]'].values[::step]
        y = self.data['y [m]'].values[::step]
        u = self.data['u [m/s]'].values[::step]
        v = self.data['v [m/s]'].values[::step]
        
        # Calculate magnitude for coloring
        magnitude = np.sqrt(u**2 + v**2)
        
        # Create quiver plot with fixed color scale
        q = ax.quiver(x, y, u, v, magnitude, 
                     scale=scale, 
                     cmap='viridis', 
                     pivot='mid',
                     width=0.002,  # Also made arrows slightly thinner
                     clim=[vmin, vmax])  # Set consistent color limits
        
        # Add colorbar with fixed limits
        cbar = plt.colorbar(q, ax=ax)
        cbar.set_label('Velocity magnitude [m/s]', rotation=270, labelpad=15)
        cbar.mappable.set_clim(vmin, vmax)
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        frame_info = self.metadata.get('frame', 'Unknown')
        ax.set_title(f'PIV Vector Field - Frame {frame_info}')
        ax.set_aspect('equal')
        
        # Invert y-axis to flip the plot
        if invert_y:
            ax.invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def plot_contour(self, field='magnitude', figsize=(12, 8), save_path=None):
        """Plot contour of specified field"""
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique positions for grid
        x_unique = np.sort(self.data['x [m]'].unique())
        y_unique = np.sort(self.data['y [m]'].unique())
        
        # Create mesh grid
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # Map field to choose
        field_map = {
            'magnitude': np.sqrt(self.data['u [m/s]']**2 + self.data['v [m/s]']**2),
            'u': self.data['u [m/s]'],
            'v': self.data['v [m/s]']
        }
        
        # Add optional fields if they exist
        if 'vorticity [1/s]' in self.data.columns:
            field_map['vorticity'] = self.data['vorticity [1/s]']
        if 'divergence [1/s]' in self.data.columns:
            field_map['divergence'] = self.data['divergence [1/s]']
        
        if field not in field_map:
            raise ValueError(f"Field must be one of {list(field_map.keys())}")
        
        # Reshape data to 2D grid
        field_data = field_map[field].values
        Z = field_data.reshape(len(y_unique), len(x_unique))
        
        # Create contour plot
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
        cbar = plt.colorbar(contour, ax=ax)
        
        # Set labels
        cbar.set_label(f'{field}', rotation=270, labelpad=15)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        frame_info = self.metadata.get('frame', 'Unknown')
        ax.set_title(f'PIV {field.capitalize()} Field - Frame {frame_info}')
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def plot_streamlines(self, density=1, figsize=(12, 8), save_path=None):
        """Plot streamlines of the velocity field"""
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get unique positions for grid
        x_unique = np.sort(self.data['x [m]'].unique())
        y_unique = np.sort(self.data['y [m]'].unique())
        
        # Create mesh grid
        X, Y = np.meshgrid(x_unique, y_unique)
        
        # Reshape velocity components to 2D grid
        U = self.data['u [m/s]'].values.reshape(len(y_unique), len(x_unique))
        V = self.data['v [m/s]'].values.reshape(len(y_unique), len(x_unique))
        
        # Calculate speed for coloring
        speed = np.sqrt(U**2 + V**2)
        
        # Create streamline plot
        strm = ax.streamplot(X, Y, U, V, 
                            color=speed, 
                            cmap='viridis', 
                            density=density,
                            linewidth=1)
        
        # Add colorbar
        cbar = plt.colorbar(strm.lines, ax=ax)
        cbar.set_label('Speed [m/s]', rotation=270, labelpad=15)
        
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        frame_info = self.metadata.get('frame', 'Unknown')
        ax.set_title(f'PIV Streamlines - Frame {frame_info}')
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()

def debug_file_format(filepath, num_lines=10):
    """Debug function to check file format"""
    print(f"\nDebugging file: {filepath}")
    print("-" * 50)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    print(f"\nFirst {num_lines} lines:")
    for i, line in enumerate(lines[:num_lines]):
        print(f"Line {i}: {line.strip()}")
    
    print("\nSearching for header line...")
    for i, line in enumerate(lines):
        if 'x [m]' in line:
            print(f"Found header at line {i}: {line.strip()}")
            break
    
    return lines

def process_single_file(filepath, output_dir, plot_types=['vector'], debug=False, 
                       vmin=0, vmax=0.1, scale=0.5):
    """Process a single PIV file and create visualizations"""
    try:
        if debug:
            debug_file_format(filepath)
        
        piv = PIVVisualizer(filepath)
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get filename without extension for output naming
        filename = Path(filepath).stem
        
        # Generate requested plots
        if 'vector' in plot_types:
            piv.plot_vector_field(
                scale=scale,  # Increased from 0.05 to make arrows shorter
                density=0.5,
                save_path=os.path.join(output_dir, f'{filename}_vector.png'),
                vmin=vmin,
                vmax=vmax,
                invert_y=True
            )
        
        if 'contour' in plot_types:
            piv.plot_contour(
                field='magnitude',
                save_path=os.path.join(output_dir, f'{filename}_contour.png')
            )
        
        if 'streamlines' in plot_types:
            piv.plot_streamlines(
                density=1,
                save_path=os.path.join(output_dir, f'{filename}_streamlines.png')
            )
        
        return filename, True
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return filepath, False

def process_all_piv_files(input_dir, output_dir, plot_types=['vector'], n_processes=None, 
                          pattern='*.txt', debug_first=True, vmin=0, vmax=0.1, scale=0.5):
    """
    Process all PIV files in a directory with consistent scaling
    
    Parameters:
    -----------
    input_dir : str
        Directory containing PIV .txt files
    output_dir : str
        Directory to save output plots
    plot_types : list
        Types of plots to generate ['vector', 'contour', 'streamlines']
    n_processes : int
        Number of parallel processes (None = auto)
    pattern : str
        File pattern to match (default: '*.txt')
    debug_first : bool
        Debug the first file before processing all
    vmin : float
        Minimum velocity for color scale (default: 0)
    vmax : float
        Maximum velocity for color scale (default: 0.1)
    scale : float
        Scale for arrow length (higher = shorter arrows, default: 0.5)
    """
    
    # Get all PIV files
    piv_files = glob.glob(os.path.join(input_dir, pattern))
    print(f"Found {len(piv_files)} PIV files in {input_dir}")
    
    if len(piv_files) == 0:
        print("No files found! Check your directory path and file pattern.")
        return
    
    # Debug the first file
    if debug_first and len(piv_files) > 0:
        print("\nDebugging first file to check format...")
        debug_file_format(piv_files[0])
        
        # Try to process first file
        print("\nTrying to process first file...")
        result = process_single_file(piv_files[0], output_dir, plot_types, 
                                   debug=False, vmin=vmin, vmax=vmax, scale=scale)
        if result[1]:
            print(f"Successfully processed: {result[0]}")
        else:
            print(f"Failed to process: {result[0]}")
            print("Please check the file format and adjust the code if needed.")
            return
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process files
    if n_processes is None:
        n_processes = mp.cpu_count() - 1
    
    print(f"\nProcessing all files using {n_processes} processes...")
    
    # Create partial function with fixed arguments
    process_func = partial(process_single_file, 
                          output_dir=output_dir, 
                          plot_types=plot_types,
                          debug=False,
                          vmin=vmin,
                          vmax=vmax,
                          scale=scale)
    
    # Process files in parallel with progress bar
    start_time = time.time()
    
    with mp.Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, piv_files),
            total=len(piv_files),
            desc="Processing PIV files"
        ))
    
    # Summary
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    elapsed_time = time.time() - start_time
    
    print(f"\nProcessing complete!")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Files processed successfully: {successful}")
    print(f"Files failed: {failed}")
    print(f"Output saved to: {output_dir}")
    print(f"Velocity scale used: {vmin} - {vmax} m/s")
    print(f"Arrow scale used: {scale}")

# Example usage
if __name__ == "__main__":
    # Process all PIV files in a directory
    input_directory = ""  # Update this to your actual path
    output_directory = "piv_plots"
    
    # Process with debugging enabled and consistent velocity scale
    process_all_piv_files(
        input_dir=input_directory,
        output_dir=output_directory,
        plot_types=['vector'],
        n_processes=4,
        debug_first=True,  # This will debug the first file
        vmin=0,  # Minimum velocity for color scale
        vmax=0.1,   # Maximum velocity for color scale (0.1 m/s)
        scale=0.5  # Arrow scale (try 0.5, 1.0, or 2.0 for shorter arrows)
    )