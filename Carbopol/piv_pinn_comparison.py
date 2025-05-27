# piv_pinn_comparison.py
"""
Comprehensive tool to compare PIV experimental data with PINN predictions
Handles coordinate system differences and provides quantitative accuracy metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import os
from pathlib import Path
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

class PIVPINNComparator:
    def __init__(self):
        self.piv_data = None
        self.pinn_data = None
        self.comparison_results = {}
        
    def load_piv_data(self, piv_filepath):
        """Load PIV data from file"""
        print(f"Loading PIV data from: {piv_filepath}")
        
        with open(piv_filepath, 'r') as f:
            lines = f.readlines()
        
        # Find header
        header_idx = None
        for i, line in enumerate(lines):
            if 'x [m]' in line and 'y [m]' in line:
                header_idx = i
                break
        
        if header_idx is None:
            raise ValueError(f"Could not find header in {piv_filepath}")
        
        # Load data
        piv_df = pd.read_csv(piv_filepath, skiprows=header_idx)
        
        # Clean data - remove NaN and infinite values
        print(f"Initial PIV data points: {len(piv_df)}")
        
        # Check for required columns
        required_columns = ['x [m]', 'y [m]', 'u [m/s]', 'v [m/s]']
        for col in required_columns:
            if col not in piv_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Remove rows with NaN or infinite values
        valid_mask = (
            np.isfinite(piv_df['x [m]']) & 
            np.isfinite(piv_df['y [m]']) & 
            np.isfinite(piv_df['u [m/s]']) & 
            np.isfinite(piv_df['v [m/s]'])
        )
        
        piv_df_clean = piv_df[valid_mask].copy()
        print(f"Valid PIV data points after cleaning: {len(piv_df_clean)}")
        
        if len(piv_df_clean) == 0:
            raise ValueError("No valid PIV data points after cleaning!")
        
        # Handle coordinate flip - PIV comes flipped from MATLAB
        # Assuming your domain has max y around 0.3m based on PINN code
        max_y = piv_df_clean['y [m]'].max()
        piv_df_clean['y [m]'] = max_y - piv_df_clean['y [m]']  # Flip y-coordinates
        
        self.piv_data = {
            'x': piv_df_clean['x [m]'].values,
            'y': piv_df_clean['y [m]'].values,
            'u': piv_df_clean['u [m/s]'].values,
            'v': -piv_df_clean['v [m/s]'].values,  # Also flip v-component due to coordinate flip
            'magnitude': np.sqrt(piv_df_clean['u [m/s]']**2 + piv_df_clean['v [m/s]']**2)
        }
        
        print(f"PIV data loaded: {len(self.piv_data['x'])} points")
        print(f"PIV domain: x=[{self.piv_data['x'].min():.3f}, {self.piv_data['x'].max():.3f}]")
        print(f"PIV domain: y=[{self.piv_data['y'].min():.3f}, {self.piv_data['y'].max():.3f}]")
        print(f"PIV velocity range: u=[{self.piv_data['u'].min():.4f}, {self.piv_data['u'].max():.4f}] m/s")
        print(f"PIV velocity range: v=[{self.piv_data['v'].min():.4f}, {self.piv_data['v'].max():.4f}] m/s")
        
    def load_pinn_data(self, pinn_filepath):
        """Load PINN data from .npz file"""
        print(f"Loading PINN data from: {pinn_filepath}")
        
        data = np.load(pinn_filepath)
        
        # Extract data
        x_grid = data['x']
        y_grid = data['y']
        u_field = data['u']
        v_field = data['v']
        
        # Convert grid data to point data (flatten and remove NaNs)
        x_flat = x_grid.flatten()
        y_flat = y_grid.flatten()
        u_flat = u_field.flatten()
        v_flat = v_field.flatten()
        
        # Remove NaN points (outside domain)
        valid_mask = ~(np.isnan(u_flat) | np.isnan(v_flat))
        
        self.pinn_data = {
            'x': x_flat[valid_mask],
            'y': y_flat[valid_mask],
            'u': u_flat[valid_mask],
            'v': v_flat[valid_mask],
            'magnitude': np.sqrt(u_flat[valid_mask]**2 + v_flat[valid_mask]**2),
            'x_grid': x_grid,
            'y_grid': y_grid,
            'u_grid': u_field,
            'v_grid': v_field
        }
        
        print(f"PINN data loaded: {len(self.pinn_data['x'])} points")
        print(f"PINN domain: x=[{self.pinn_data['x'].min():.3f}, {self.pinn_data['x'].max():.3f}]")
        print(f"PINN domain: y=[{self.pinn_data['y'].min():.3f}, {self.pinn_data['y'].max():.3f}]")
        
    def interpolate_pinn_to_piv_points(self):
        """Interpolate PINN data to PIV measurement points for direct comparison"""
        if self.piv_data is None or self.pinn_data is None:
            raise ValueError("Both PIV and PINN data must be loaded first")
        
        print("Interpolating PINN data to PIV measurement points...")
        
        # Create coordinate arrays for interpolation
        pinn_coords = np.column_stack([self.pinn_data['x'], self.pinn_data['y']])
        piv_coords = np.column_stack([self.piv_data['x'], self.piv_data['y']])
        
        # Interpolate PINN velocity components to PIV points
        pinn_u_interp = griddata(pinn_coords, self.pinn_data['u'], piv_coords, 
                                method='linear', fill_value=np.nan)
        pinn_v_interp = griddata(pinn_coords, self.pinn_data['v'], piv_coords, 
                                method='linear', fill_value=np.nan)
        
        # Remove points where interpolation failed OR where PIV data has issues
        valid_mask = (
            ~(np.isnan(pinn_u_interp) | np.isnan(pinn_v_interp)) &
            ~(np.isnan(self.piv_data['u']) | np.isnan(self.piv_data['v'])) &
            np.isfinite(pinn_u_interp) & np.isfinite(pinn_v_interp) &
            np.isfinite(self.piv_data['u']) & np.isfinite(self.piv_data['v'])
        )
        
        print(f"Valid interpolation points: {np.sum(valid_mask)} out of {len(valid_mask)}")
        
        if np.sum(valid_mask) == 0:
            raise ValueError("No valid comparison points found after interpolation!")
        
        # Store interpolated comparison data
        self.comparison_data = {
            'x': self.piv_data['x'][valid_mask],
            'y': self.piv_data['y'][valid_mask],
            'piv_u': self.piv_data['u'][valid_mask],
            'piv_v': self.piv_data['v'][valid_mask],
            'pinn_u': pinn_u_interp[valid_mask],
            'pinn_v': pinn_v_interp[valid_mask],
            'piv_mag': self.piv_data['magnitude'][valid_mask],
            'pinn_mag': np.sqrt(pinn_u_interp[valid_mask]**2 + pinn_v_interp[valid_mask]**2)
        }
        
        print(f"Comparison data prepared: {len(self.comparison_data['x'])} overlapping points")
        
        # Check for any remaining NaN values (shouldn't happen, but just to be safe)
        for key, data in self.comparison_data.items():
            if isinstance(data, np.ndarray):
                nan_count = np.sum(np.isnan(data))
                if nan_count > 0:
                    print(f"WARNING: {nan_count} NaN values found in {key}")
                inf_count = np.sum(~np.isfinite(data))
                if inf_count > 0:
                    print(f"WARNING: {inf_count} infinite values found in {key}")
        
    def calculate_metrics(self):
        """Calculate quantitative comparison metrics"""
        if not hasattr(self, 'comparison_data'):
            self.interpolate_pinn_to_piv_points()
        
        print("Calculating comparison metrics...")
        
        # Final check for NaN values and clean data if necessary
        valid_indices = (
            np.isfinite(self.comparison_data['piv_u']) &
            np.isfinite(self.comparison_data['piv_v']) &
            np.isfinite(self.comparison_data['pinn_u']) &
            np.isfinite(self.comparison_data['pinn_v'])
        )
        
        n_invalid = np.sum(~valid_indices)
        if n_invalid > 0:
            print(f"Removing {n_invalid} invalid data points before metrics calculation")
            for key in self.comparison_data:
                if isinstance(self.comparison_data[key], np.ndarray):
                    self.comparison_data[key] = self.comparison_data[key][valid_indices]
        
        if len(self.comparison_data['piv_u']) == 0:
            raise ValueError("No valid data points remaining for comparison!")
        
        print(f"Computing metrics for {len(self.comparison_data['piv_u'])} valid points")
        
        # U-velocity metrics
        u_mse = mean_squared_error(self.comparison_data['piv_u'], self.comparison_data['pinn_u'])
        u_mae = mean_absolute_error(self.comparison_data['piv_u'], self.comparison_data['pinn_u'])
        u_r2 = r2_score(self.comparison_data['piv_u'], self.comparison_data['pinn_u'])
        u_rmse = np.sqrt(u_mse)
        
        # V-velocity metrics
        v_mse = mean_squared_error(self.comparison_data['piv_v'], self.comparison_data['pinn_v'])
        v_mae = mean_absolute_error(self.comparison_data['piv_v'], self.comparison_data['pinn_v'])
        v_r2 = r2_score(self.comparison_data['piv_v'], self.comparison_data['pinn_v'])
        v_rmse = np.sqrt(v_mse)
        
        # Magnitude metrics
        mag_mse = mean_squared_error(self.comparison_data['piv_mag'], self.comparison_data['pinn_mag'])
        mag_mae = mean_absolute_error(self.comparison_data['piv_mag'], self.comparison_data['pinn_mag'])
        mag_r2 = r2_score(self.comparison_data['piv_mag'], self.comparison_data['pinn_mag'])
        mag_rmse = np.sqrt(mag_mse)
        
        # Overall vector magnitude error
        vector_error = np.sqrt((self.comparison_data['piv_u'] - self.comparison_data['pinn_u'])**2 + 
                              (self.comparison_data['piv_v'] - self.comparison_data['pinn_v'])**2)
        mean_vector_error = np.mean(vector_error)
        max_vector_error = np.max(vector_error)
        
        # Percentage errors (handle division by zero)
        piv_u_mean_abs = np.mean(np.abs(self.comparison_data['piv_u']))
        piv_v_mean_abs = np.mean(np.abs(self.comparison_data['piv_v']))
        piv_mag_mean = np.mean(self.comparison_data['piv_mag'])
        
        u_mae_percent = (u_mae / piv_u_mean_abs) * 100 if piv_u_mean_abs > 0 else np.inf
        v_mae_percent = (v_mae / piv_v_mean_abs) * 100 if piv_v_mean_abs > 0 else np.inf
        mag_mae_percent = (mag_mae / piv_mag_mean) * 100 if piv_mag_mean > 0 else np.inf
        
        # Calculate overall accuracy metrics
        avg_r2 = (u_r2 + v_r2 + mag_r2) / 3
        avg_mae_percent = (u_mae_percent + v_mae_percent + mag_mae_percent) / 3
        
        # Normalized RMSE (NRMSE) - RMSE divided by range of PIV data
        piv_u_range = np.max(self.comparison_data['piv_u']) - np.min(self.comparison_data['piv_u'])
        piv_v_range = np.max(self.comparison_data['piv_v']) - np.min(self.comparison_data['piv_v'])
        piv_mag_range = np.max(self.comparison_data['piv_mag']) - np.min(self.comparison_data['piv_mag'])
        
        u_nrmse = (u_rmse / piv_u_range * 100) if piv_u_range > 0 else np.inf
        v_nrmse = (v_rmse / piv_v_range * 100) if piv_v_range > 0 else np.inf
        mag_nrmse = (mag_rmse / piv_mag_range * 100) if piv_mag_range > 0 else np.inf
        avg_nrmse = (u_nrmse + v_nrmse + mag_nrmse) / 3
        
        # Simple accuracy percentage (100% - normalized error)
        u_accuracy = max(0, 100 - (u_mae / np.mean(np.abs(self.comparison_data['piv_u'])) * 100))
        v_accuracy = max(0, 100 - (v_mae / np.mean(np.abs(self.comparison_data['piv_v'])) * 100))
        mag_accuracy = max(0, 100 - (mag_mae / piv_mag_mean * 100))
        overall_accuracy = (u_accuracy + v_accuracy + mag_accuracy) / 3
        
        # Direction accuracy (cosine similarity)
        piv_vectors = np.column_stack([self.comparison_data['piv_u'], self.comparison_data['piv_v']])
        pinn_vectors = np.column_stack([self.comparison_data['pinn_u'], self.comparison_data['pinn_v']])
        
        # Normalize vectors
        piv_norms = np.linalg.norm(piv_vectors, axis=1)
        pinn_norms = np.linalg.norm(pinn_vectors, axis=1)
        
        # Only calculate for non-zero vectors
        valid_norm_mask = (piv_norms > 1e-10) & (pinn_norms > 1e-10)
        if np.sum(valid_norm_mask) > 0:
            piv_normalized = piv_vectors[valid_norm_mask] / piv_norms[valid_norm_mask][:, np.newaxis]
            pinn_normalized = pinn_vectors[valid_norm_mask] / pinn_norms[valid_norm_mask][:, np.newaxis]
            
            # Cosine similarity (dot product of normalized vectors)
            cosine_similarities = np.sum(piv_normalized * pinn_normalized, axis=1)
            direction_accuracy = np.mean(cosine_similarities) * 100  # Convert to percentage
        else:
            direction_accuracy = 0
        
        self.comparison_results = {
            'u_component': {
                'mse': u_mse, 'mae': u_mae, 'rmse': u_rmse, 'r2': u_r2,
                'mae_percent': u_mae_percent, 'nrmse': u_nrmse, 'accuracy': u_accuracy
            },
            'v_component': {
                'mse': v_mse, 'mae': v_mae, 'rmse': v_rmse, 'r2': v_r2,
                'mae_percent': v_mae_percent, 'nrmse': v_nrmse, 'accuracy': v_accuracy
            },
            'magnitude': {
                'mse': mag_mse, 'mae': mag_mae, 'rmse': mag_rmse, 'r2': mag_r2,
                'mae_percent': mag_mae_percent, 'nrmse': mag_nrmse, 'accuracy': mag_accuracy
            },
            'vector_error': {
                'mean': mean_vector_error,
                'max': max_vector_error,
                'std': np.std(vector_error)
            },
            'overall_metrics': {
                'avg_r2': avg_r2,
                'avg_mae_percent': avg_mae_percent,
                'avg_nrmse': avg_nrmse,
                'overall_accuracy': overall_accuracy,
                'direction_accuracy': direction_accuracy,
                'mean_vector_error_normalized': mean_vector_error / piv_mag_mean if piv_mag_mean > 0 else np.inf
            },
            'n_points': len(self.comparison_data['x'])
        }
        
        return self.comparison_results
    
    def print_metrics_summary(self):
        """Print a summary of comparison metrics"""
        if not self.comparison_results:
            self.calculate_metrics()
        
        print("="*60)
        print("PIV vs PINN COMPARISON METRICS")
        print("="*60)
        print(f"Number of comparison points: {self.comparison_results['n_points']}")
        print()
        
        # Overall accuracy summary (the key metrics you're looking for!)
        overall = self.comparison_results['overall_metrics']
        print("🎯 OVERALL ACCURACY SUMMARY:")
        print(f"  Overall Accuracy:      {overall['overall_accuracy']:.2f}%")
        print(f"  Direction Accuracy:    {overall['direction_accuracy']:.2f}%") 
        print(f"  Average R² Score:      {overall['avg_r2']:.4f}")
        print(f"  Average Error:         {overall['avg_mae_percent']:.2f}%")
        print()
        
        # Component details
        u_metrics = self.comparison_results['u_component']
        print("U-VELOCITY COMPONENT:")
        print(f"  Accuracy:          {u_metrics['accuracy']:.2f}%")
        print(f"  R² Score:          {u_metrics['r2']:.4f}")
        print(f"  RMSE:              {u_metrics['rmse']:.6f} m/s")
        print(f"  Error:             {u_metrics['mae_percent']:.2f}%")
        print()
        
        v_metrics = self.comparison_results['v_component']
        print("V-VELOCITY COMPONENT:")
        print(f"  Accuracy:          {v_metrics['accuracy']:.2f}%")
        print(f"  R² Score:          {v_metrics['r2']:.4f}")
        print(f"  RMSE:              {v_metrics['rmse']:.6f} m/s")
        print(f"  Error:             {v_metrics['mae_percent']:.2f}%")
        print()
        
        mag_metrics = self.comparison_results['magnitude']
        print("VELOCITY MAGNITUDE:")
        print(f"  Accuracy:          {mag_metrics['accuracy']:.2f}%")
        print(f"  R² Score:          {mag_metrics['r2']:.4f}")
        print(f"  RMSE:              {mag_metrics['rmse']:.6f} m/s")
        print(f"  Error:             {mag_metrics['mae_percent']:.2f}%")
        print()
        
        vec_error = self.comparison_results['vector_error']
        print("VECTOR ERROR DETAILS:")
        print(f"  Mean Error:        {vec_error['mean']:.6f} m/s")
        print(f"  Max Error:         {vec_error['max']:.6f} m/s")
        print(f"  Normalized Error:  {overall['mean_vector_error_normalized']:.2f}x mean velocity")
        print()
        
        # Simple assessment
        print("📊 ASSESSMENT:")
        if overall['overall_accuracy'] > 90:
            print("  ✅ EXCELLENT accuracy (>90%)")
        elif overall['overall_accuracy'] > 70:
            print("  ✅ GOOD accuracy (70-90%)")
        elif overall['overall_accuracy'] > 50:
            print("  ⚠️  FAIR accuracy (50-70%)")
        elif overall['overall_accuracy'] > 20:
            print("  ❌ POOR accuracy (20-50%)")
        else:
            print("  ❌ VERY POOR accuracy (<20%) - fundamental mismatch")
            
        if overall['direction_accuracy'] > 80:
            print("  ✅ Flow directions match well")
        elif overall['direction_accuracy'] > 50:
            print("  ⚠️  Flow directions partially match")
        else:
            print("  ❌ Flow directions don't match")
            
        print("="*60)
    
    def plot_comparison(self, save_path=None, figsize=(20, 15)):
        """Create comprehensive comparison plots"""
        if not hasattr(self, 'comparison_data'):
            self.interpolate_pinn_to_piv_points()
        
        fig = plt.figure(figsize=figsize)
        
        # 1. Side-by-side vector fields
        plt.subplot(3, 4, 1)
        plt.quiver(self.piv_data['x'], self.piv_data['y'], 
                  self.piv_data['u'], self.piv_data['v'],
                  self.piv_data['magnitude'], cmap='viridis', scale=1)
        plt.colorbar(label='Velocity [m/s]')
        plt.title('PIV Vector Field')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        
        plt.subplot(3, 4, 2)
        plt.quiver(self.comparison_data['x'], self.comparison_data['y'],
                  self.comparison_data['pinn_u'], self.comparison_data['pinn_v'],
                  self.comparison_data['pinn_mag'], cmap='viridis', scale=1)
        plt.colorbar(label='Velocity [m/s]')
        plt.title('PINN Vector Field')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        
        # 2. Velocity magnitude comparison
        plt.subplot(3, 4, 3)
        plt.contourf(self.pinn_data['x_grid'], self.pinn_data['y_grid'],
                    np.sqrt(self.pinn_data['u_grid']**2 + self.pinn_data['v_grid']**2),
                    levels=20, cmap='viridis')
        plt.colorbar(label='|V| [m/s]')
        plt.title('PINN Velocity Magnitude')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        
        # 3. Error field
        plt.subplot(3, 4, 4)
        error_field = np.sqrt((self.comparison_data['piv_u'] - self.comparison_data['pinn_u'])**2 + 
                             (self.comparison_data['piv_v'] - self.comparison_data['pinn_v'])**2)
        scatter = plt.scatter(self.comparison_data['x'], self.comparison_data['y'], 
                            c=error_field, cmap='Reds', s=30)
        plt.colorbar(scatter, label='Vector Error [m/s]')
        plt.title('Vector Error Distribution')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        
        # 4-6. Scatter plots for components
        plt.subplot(3, 4, 5)
        plt.scatter(self.comparison_data['piv_u'], self.comparison_data['pinn_u'], 
                   alpha=0.6, s=20)
        min_val = min(self.comparison_data['piv_u'].min(), self.comparison_data['pinn_u'].min())
        max_val = max(self.comparison_data['piv_u'].max(), self.comparison_data['pinn_u'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('PIV U-velocity [m/s]')
        plt.ylabel('PINN U-velocity [m/s]')
        plt.title(f'U-component (R² = {self.comparison_results["u_component"]["r2"]:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 6)
        plt.scatter(self.comparison_data['piv_v'], self.comparison_data['pinn_v'], 
                   alpha=0.6, s=20)
        min_val = min(self.comparison_data['piv_v'].min(), self.comparison_data['pinn_v'].min())
        max_val = max(self.comparison_data['piv_v'].max(), self.comparison_data['pinn_v'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('PIV V-velocity [m/s]')
        plt.ylabel('PINN V-velocity [m/s]')
        plt.title(f'V-component (R² = {self.comparison_results["v_component"]["r2"]:.3f})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 7)
        plt.scatter(self.comparison_data['piv_mag'], self.comparison_data['pinn_mag'], 
                   alpha=0.6, s=20)
        min_val = min(self.comparison_data['piv_mag'].min(), self.comparison_data['pinn_mag'].min())
        max_val = max(self.comparison_data['piv_mag'].max(), self.comparison_data['pinn_mag'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('PIV Magnitude [m/s]')
        plt.ylabel('PINN Magnitude [m/s]')
        plt.title(f'Magnitude (R² = {self.comparison_results["magnitude"]["r2"]:.3f})')
        plt.grid(True, alpha=0.3)
        
        # 8. Error histogram
        plt.subplot(3, 4, 8)
        plt.hist(error_field, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Vector Error [m/s]')
        plt.ylabel('Frequency')
        plt.title(f'Error Distribution\n(Mean: {np.mean(error_field):.4f} m/s)')
        plt.grid(True, alpha=0.3)
        
        # 9-12. Residual plots
        plt.subplot(3, 4, 9)
        u_residuals = self.comparison_data['piv_u'] - self.comparison_data['pinn_u']
        plt.scatter(self.comparison_data['pinn_u'], u_residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('PINN U-velocity [m/s]')
        plt.ylabel('PIV - PINN (U)')
        plt.title('U-component Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 10)
        v_residuals = self.comparison_data['piv_v'] - self.comparison_data['pinn_v']
        plt.scatter(self.comparison_data['pinn_v'], v_residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('PINN V-velocity [m/s]')
        plt.ylabel('PIV - PINN (V)')
        plt.title('V-component Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 4, 11)
        mag_residuals = self.comparison_data['piv_mag'] - self.comparison_data['pinn_mag']
        plt.scatter(self.comparison_data['pinn_mag'], mag_residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        plt.xlabel('PINN Magnitude [m/s]')
        plt.ylabel('PIV - PINN (Mag)')
        plt.title('Magnitude Residuals')
        plt.grid(True, alpha=0.3)
        
        # 12. Summary metrics box
        plt.subplot(3, 4, 12)
        plt.axis('off')
        metrics_text = f"""COMPARISON METRICS
        
N points: {self.comparison_results['n_points']}

U-component:
  R² = {self.comparison_results['u_component']['r2']:.3f}
  RMSE = {self.comparison_results['u_component']['rmse']:.4f}
  
V-component:
  R² = {self.comparison_results['v_component']['r2']:.3f}
  RMSE = {self.comparison_results['v_component']['rmse']:.4f}
  
Magnitude:
  R² = {self.comparison_results['magnitude']['r2']:.3f}
  RMSE = {self.comparison_results['magnitude']['rmse']:.4f}
  
Vector Error:
  Mean = {self.comparison_results['vector_error']['mean']:.4f}
  Max = {self.comparison_results['vector_error']['max']:.4f}"""
        
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        else:
            plt.show()
            
        return fig

def compare_piv_pinn(piv_filepath, pinn_filepath, output_dir='comparison_results'):
    """
    Main function to compare PIV and PINN data
    
    Parameters:
    -----------
    piv_filepath : str
        Path to PIV .txt file
    pinn_filepath : str  
        Path to PINN .npz file
    output_dir : str
        Directory to save results
    """
    
    print("=== PIV vs PINN Comparison ===")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize comparator
    comparator = PIVPINNComparator()
    
    # Load data
    comparator.load_piv_data(piv_filepath)
    comparator.load_pinn_data(pinn_filepath)
    
    # Calculate metrics
    comparator.calculate_metrics()
    
    # Print summary
    comparator.print_metrics_summary()
    
    # Create plots
    plot_path = os.path.join(output_dir, 'piv_pinn_comparison.png')
    comparator.plot_comparison(save_path=plot_path)
    
    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'comparison_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("PIV vs PINN Comparison Metrics\n")
        f.write("="*40 + "\n\n")
        f.write(f"Number of comparison points: {comparator.comparison_results['n_points']}\n\n")
        
        for component in ['u_component', 'v_component', 'magnitude']:
            f.write(f"{component.upper()}:\n")
            metrics = comparator.comparison_results[component]
            f.write(f"  R² Score: {metrics['r2']:.6f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.6f} m/s\n")
            f.write(f"  MAE: {metrics['mae']:.6f} m/s ({metrics['mae_percent']:.2f}%)\n\n")
        
        vec_error = comparator.comparison_results['vector_error']
        f.write("VECTOR ERROR:\n")
        f.write(f"  Mean: {vec_error['mean']:.6f} m/s\n")
        f.write(f"  Max: {vec_error['max']:.6f} m/s\n")
        f.write(f"  Std Dev: {vec_error['std']:.6f} m/s\n")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Comparison plot: {plot_path}")
    print(f"  - Metrics summary: {metrics_path}")
    
    return comparator

# Example usage
if __name__ == "__main__":
    # Example file paths - update these to your actual files
    piv_file = "PIV/PIVs_txts/p1/PIVlab_0900.txt"
    pinn_file = "data/pinn_velocity_for_piv.npz"
    
    # Run comparison
    comparator = compare_piv_pinn(piv_file, pinn_file)