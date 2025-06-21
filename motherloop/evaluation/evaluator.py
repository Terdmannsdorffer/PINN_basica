# evaluation/evaluator.py
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from domain import inside_L

class PINNEvaluator:
    """Evaluate PINN model against PIV data"""
    
    def __init__(self, model, device, output_dir, piv_file="averaged_piv_steady_state.txt"):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.piv_file = piv_file
        
    def evaluate_and_compare(self):
        """Evaluate model and compare with PIV data"""
        
        # Generate PINN velocity field
        pinn_data = self._extract_pinn_velocity_field()
        
        # Load and process PIV data
        try:
            piv_data = self._load_piv_data()
            
            # Compare if PIV data available
            if piv_data is not None:
                metrics = self._compare_with_piv(pinn_data, piv_data)
            else:
                # Fallback metrics without PIV
                metrics = self._basic_metrics(pinn_data)
                
        except Exception as e:
            print(f"PIV comparison failed: {e}")
            metrics = self._basic_metrics(pinn_data)
        
        # Save visualization
        self._save_velocity_visualization(pinn_data)
        
        return metrics
    
    def _extract_pinn_velocity_field(self, resolution=80):
        """Extract velocity field from PINN"""
        
        # Domain dimensions
        L_up = 0.097
        L_down = 0.174
        H_left = 0.119
        H_right = 0.019
        
        # Create grid
        x = np.linspace(0, L_down, resolution)
        y = np.linspace(0, H_left, resolution)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # Flatten and filter
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        xy_points = np.column_stack([x_flat, y_flat])
        
        inside_mask = np.array([inside_L(x, y) for x, y in xy_points])
        xy_inside = xy_points[inside_mask]
        
        # Evaluate PINN
        self.model.eval()
        with torch.no_grad():
            xy_tensor = torch.tensor(xy_inside, dtype=torch.float32, device=self.device)
            predictions = self.model(xy_tensor)
            
            u_inside = predictions[:, 0].cpu().numpy()
            v_inside = predictions[:, 1].cpu().numpy()
            p_inside = predictions[:, 2].cpu().numpy()
        
        return {
            'x': xy_inside[:, 0],
            'y': xy_inside[:, 1],
            'u': u_inside,
            'v': v_inside,
            'p': p_inside,
            'magnitude': np.sqrt(u_inside**2 + v_inside**2)
        }
    
    def _load_piv_data(self):
        """Load PIV data if available"""
        if not Path(self.piv_file).exists():
            print(f"PIV file {self.piv_file} not found")
            return None
        
        try:
            # Find header
            with open(self.piv_file, 'r') as f:
                lines = f.readlines()
            
            header_idx = None
            for i, line in enumerate(lines):
                if 'x [m]' in line and 'y [m]' in line:
                    header_idx = i
                    break
            
            if header_idx is None:
                return None
            
            # Load data
            piv_df = pd.read_csv(self.piv_file, skiprows=header_idx)
            
            # Clean data
            required_columns = ['x [m]', 'y [m]', 'u [m/s]', 'v [m/s]']
            for col in required_columns:
                if col not in piv_df.columns:
                    return None
            
            valid_mask = (
                np.isfinite(piv_df['x [m]']) & 
                np.isfinite(piv_df['y [m]']) & 
                np.isfinite(piv_df['u [m/s]']) & 
                np.isfinite(piv_df['v [m/s]'])
            )
            
            piv_df_clean = piv_df[valid_mask].copy()
            
            if len(piv_df_clean) == 0:
                return None
            
            # Coordinate transformation (flip y and v)
            max_y = piv_df_clean['y [m]'].max()
            piv_df_clean['y [m]'] = max_y - piv_df_clean['y [m]']
            
            return {
                'x': piv_df_clean['x [m]'].values,
                'y': piv_df_clean['y [m]'].values,
                'u': piv_df_clean['u [m/s]'].values,
                'v': -piv_df_clean['v [m/s]'].values,
                'magnitude': np.sqrt(piv_df_clean['u [m/s]']**2 + piv_df_clean['v [m/s]']**2)
            }
            
        except Exception as e:
            print(f"Error loading PIV data: {e}")
            return None
    
    def _compare_with_piv(self, pinn_data, piv_data):
        """Compare PINN with PIV data"""
        
        # Interpolate PINN to PIV points
        pinn_coords = np.column_stack([pinn_data['x'], pinn_data['y']])
        piv_coords = np.column_stack([piv_data['x'], piv_data['y']])
        
        pinn_u_interp = griddata(pinn_coords, pinn_data['u'], piv_coords, method='linear', fill_value=np.nan)
        pinn_v_interp = griddata(pinn_coords, pinn_data['v'], piv_coords, method='linear', fill_value=np.nan)
        
        # Valid comparison points
        valid_mask = ~(np.isnan(pinn_u_interp) | np.isnan(pinn_v_interp))
        
        if np.sum(valid_mask) == 0:
            return self._basic_metrics(pinn_data)
        
        # Extract valid data
        piv_u_valid = piv_data['u'][valid_mask]
        piv_v_valid = piv_data['v'][valid_mask]
        pinn_u_valid = pinn_u_interp[valid_mask]
        pinn_v_valid = pinn_v_interp[valid_mask]
        
        # Calculate metrics
        try:
            # R² scores
            u_r2 = r2_score(piv_u_valid, pinn_u_valid)
            v_r2 = r2_score(piv_v_valid, pinn_v_valid)
            
            # RMSE
            u_rmse = np.sqrt(mean_squared_error(piv_u_valid, pinn_u_valid))
            v_rmse = np.sqrt(mean_squared_error(piv_v_valid, pinn_v_valid))
            
            # Vector error
            vector_error = np.sqrt((piv_u_valid - pinn_u_valid)**2 + (piv_v_valid - pinn_v_valid)**2)
            mean_vector_error = np.mean(vector_error)
            
            # Direction accuracy (cosine similarity)
            piv_vectors = np.column_stack([piv_u_valid, piv_v_valid])
            pinn_vectors = np.column_stack([pinn_u_valid, pinn_v_valid])
            
            piv_norms = np.linalg.norm(piv_vectors, axis=1)
            pinn_norms = np.linalg.norm(pinn_vectors, axis=1)
            
            valid_norm_mask = (piv_norms > 1e-10) & (pinn_norms > 1e-10)
            
            if np.sum(valid_norm_mask) > 0:
                piv_normalized = piv_vectors[valid_norm_mask] / piv_norms[valid_norm_mask][:, np.newaxis]
                pinn_normalized = pinn_vectors[valid_norm_mask] / pinn_norms[valid_norm_mask][:, np.newaxis]
                
                cosine_similarities = np.sum(piv_normalized * pinn_normalized, axis=1)
                direction_accuracy = np.mean(cosine_similarities) * 100
            else:
                direction_accuracy = 0
            
            # Overall accuracy estimate
            piv_mag_mean = np.mean(np.sqrt(piv_u_valid**2 + piv_v_valid**2))
            overall_accuracy = max(0, 100 - (mean_vector_error / piv_mag_mean * 100)) if piv_mag_mean > 0 else 0
            
            return {
                'direction_accuracy': direction_accuracy,
                'overall_accuracy': overall_accuracy,
                'u_r2': u_r2,
                'v_r2': v_r2,
                'u_rmse': u_rmse,
                'v_rmse': v_rmse,
                'mean_vector_error': mean_vector_error,
                'n_comparison_points': np.sum(valid_mask)
            }
            
        except Exception as e:
            print(f"Error in metrics calculation: {e}")
            return self._basic_metrics(pinn_data)
    
    def _basic_metrics(self, pinn_data):
        """Basic metrics when PIV comparison fails"""
        vel_mag = pinn_data['magnitude']
        return {
            'direction_accuracy': 50.0,  # Neutral score
            'overall_accuracy': 25.0,   # Neutral score
            'u_r2': 0.0,
            'v_r2': 0.0,
            'u_rmse': np.std(pinn_data['u']),
            'v_rmse': np.std(pinn_data['v']),
            'mean_vector_error': np.mean(vel_mag),
            'n_comparison_points': len(pinn_data['x']),
            'pinn_only': True  # Flag to indicate no PIV comparison
        }
    
    def _save_velocity_visualization(self, pinn_data):
        """Save velocity field visualization"""
        plt.figure(figsize=(12, 8))
        
        # Velocity vectors
        plt.subplot(2, 2, 1)
        plt.quiver(pinn_data['x'], pinn_data['y'], pinn_data['u'], pinn_data['v'],
                  pinn_data['magnitude'], scale=0.1, cmap='viridis')
        plt.colorbar(label='Velocity Magnitude [m/s]')
        plt.title('PINN Velocity Field')
        plt.axis('equal')
        
        # U component
        plt.subplot(2, 2, 2)
        plt.scatter(pinn_data['x'], pinn_data['y'], c=pinn_data['u'], cmap='RdBu_r', s=20)
        plt.colorbar(label='U-velocity [m/s]')
        plt.title('U-Component')
        plt.axis('equal')
        
        # V component
        plt.subplot(2, 2, 3)
        plt.scatter(pinn_data['x'], pinn_data['y'], c=pinn_data['v'], cmap='RdBu_r', s=20)
        plt.colorbar(label='V-velocity [m/s]')
        plt.title('V-Component')
        plt.axis('equal')
        
        # Pressure
        plt.subplot(2, 2, 4)
        plt.scatter(pinn_data['x'], pinn_data['y'], c=pinn_data['p'], cmap='plasma', s=20)
        plt.colorbar(label='Pressure [Pa]')
        plt.title('Pressure Field')
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'velocity_field.png', dpi=150, bbox_inches='tight')
        plt.close()