# hyperparameter_study.py - Main Hyperparameter Study Controller
import os
import json
import time
import torch
import numpy as np
from itertools import product
from pathlib import Path

# Import all architectures
from architectures.standard_fourier import StandardFourierPINN
from architectures.unet_skip import UNetSkipPINN
from architectures.fpn_deep import FPNDeepPINN

# Import training and evaluation
from training.trainer import PINNTrainer
from evaluation.evaluator import PINNEvaluator
from physics.boundary_conditions import BoundaryConfig

class HyperparameterStudy:
    def __init__(self, base_output_dir="hyperparameter_results", test_mode=False):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.test_mode = test_mode
        
        # Define study parameters
        self.architectures = {
            'standard_fourier': StandardFourierPINN,
            'unet_skip': UNetSkipPINN, 
            'fpn_deep': FPNDeepPINN
        }
        
        if test_mode:
            # Minimal parameters for testing
            self.activations = ['tanh']
            self.epoch_configs = [50]  # Very short for testing
            self.inlet_velocities = [-0.005]  # Just one velocity
            print("🧪 TEST MODE: Using minimal parameters")
        else:
            # Full study parameters
            self.activations = ['tanh', 'swish', 'mish']
            self.epoch_configs = [1000, 2500, 5000]  # Three different training lengths
            self.inlet_velocities = [0.0, -0.005, -0.015]  # Three starting velocities
        
        self.results = {}
        
    def run_study(self):
        """Run complete hyperparameter study"""
        print("🔬 Starting Comprehensive PINN Hyperparameter Study")
        print("="*60)
        
        total_experiments = (len(self.architectures) * len(self.activations) * 
                           len(self.epoch_configs) * len(self.inlet_velocities))
        print(f"Total experiments to run: {total_experiments}")
        
        experiment_count = 0
        
        for arch_name, arch_class in self.architectures.items():
            for activation in self.activations:
                for epochs in self.epoch_configs:
                    for inlet_vel in self.inlet_velocities:
                        experiment_count += 1
                        
                        print(f"\n📊 Experiment {experiment_count}/{total_experiments}")
                        print(f"Architecture: {arch_name}, Activation: {activation}")
                        print(f"Epochs: {epochs}, Inlet Velocity: {inlet_vel}")
                        
                        # Run single experiment
                        try:
                            results = self._run_single_experiment(
                                arch_name, arch_class, activation, epochs, inlet_vel
                            )
                            
                            # Store results
                            exp_key = f"{arch_name}_{activation}_{epochs}_{inlet_vel}"
                            self.results[exp_key] = results
                            
                            print(f"✅ Completed: Dir: {results['direction_accuracy']:.1f}%, "
                                  f"Overall: {results['overall_accuracy']:.1f}%, "
                                  f"Mag: {results['magnitude_accuracy']:.1f}%"
                                  f"{' (PIV)' if not results.get('pinn_only', False) else ' (PINN-only)'}")
                            
                        except Exception as e:
                            print(f"❌ Failed: {str(e)}")
                            # Continue with next experiment
                            continue
        
        # Save and analyze results
        self._save_results()
        self._analyze_results()
        
    def _run_single_experiment(self, arch_name, arch_class, activation, epochs, inlet_vel):
        """Run a single experiment configuration"""
        
        # Create experiment directory
        exp_name = f"{arch_name}_{activation}_{epochs}ep_{abs(inlet_vel)*1000:.0f}vel"
        exp_dir = self.base_output_dir / exp_name
        exp_dir.mkdir(exist_ok=True)
        
        # Setup boundary conditions
        boundary_config = BoundaryConfig(inlet_velocity=inlet_vel)
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = arch_class(activation=activation).to(device)
        
        # Setup trainer
        trainer = PINNTrainer(
            model=model,
            boundary_config=boundary_config,
            max_epochs=epochs,
            device=device,
            output_dir=exp_dir
        )
        
        # Train model
        trained_model = trainer.train()
        
        # Evaluate model
        evaluator = PINNEvaluator(
            model=trained_model,
            device=device,
            output_dir=exp_dir
        )
        
        # Get metrics
        metrics = evaluator.evaluate_and_compare()
        
        # Add experiment config to metrics
        metrics.update({
            'architecture': arch_name,
            'activation': activation,
            'epochs': epochs,
            'inlet_velocity': inlet_vel,
            'experiment_name': exp_name
        })
        
        return metrics
    
    def _save_results(self):
        """Save all results to JSON"""
        results_file = self.base_output_dir / "complete_results.json"
        
        # Convert numpy values to regular python types for JSON serialization
        json_results = {}
        for key, result in self.results.items():
            json_results[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                               for k, v in result.items()}
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def _analyze_results(self):
        """Analyze and rank results"""
        if not self.results:
            print("No results to analyze!")
            return
        
        print("\n" + "="*60)
        print("📈 HYPERPARAMETER STUDY ANALYSIS")
        print("="*60)
        
        # Convert to list for easier analysis
        result_list = []
        for exp_name, metrics in self.results.items():
            result_list.append({
                'name': exp_name,
                'direction_acc': metrics.get('direction_accuracy', 0),
                'overall_acc': metrics.get('overall_accuracy', 0),
                'magnitude_acc': metrics.get('magnitude_accuracy', 0),  # NEW
                'architecture': metrics.get('architecture', ''),
                'activation': metrics.get('activation', ''),
                'epochs': metrics.get('epochs', 0),
                'inlet_vel': metrics.get('inlet_velocity', 0)
            })
        
        # Sort by direction accuracy (your key metric)
        result_list.sort(key=lambda x: x['direction_acc'], reverse=True)
        
        print("\n🏆 TOP 10 RESULTS (by Direction Accuracy):")
        print("-" * 110)
        print(f"{'Rank':<4} {'Name':<35} {'Dir%':<6} {'Overall%':<8} {'Mag%':<6} {'Arch':<15} {'Act':<8}")
        print("-" * 110)
        
        for i, result in enumerate(result_list[:10], 1):
            print(f"{i:<4} {result['name']:<35} {result['direction_acc']:<6.1f} "
                  f"{result['overall_acc']:<8.1f} {result['magnitude_acc']:<6.1f} "
                  f"{result['architecture']:<15} {result['activation']:<8}")
        
        # Analysis by category
        self._analyze_by_architecture(result_list)
        self._analyze_by_activation(result_list)
        self._analyze_by_inlet_velocity(result_list)
        
    def _analyze_by_architecture(self, result_list):
        """Analyze results by architecture"""
        print(f"\n🏗️  ARCHITECTURE ANALYSIS:")
        arch_stats = {}
        for result in result_list:
            arch = result['architecture']
            if arch not in arch_stats:
                arch_stats[arch] = []
            arch_stats[arch].append(result['direction_acc'])
        
        for arch, accs in arch_stats.items():
            avg_acc = np.mean(accs)
            max_acc = np.max(accs)
            print(f"  {arch:<15}: Avg {avg_acc:.1f}%, Max {max_acc:.1f}%")
    
    def _analyze_by_activation(self, result_list):
        """Analyze results by activation"""
        print(f"\n⚡ ACTIVATION ANALYSIS:")
        act_stats = {}
        for result in result_list:
            act = result['activation']
            if act not in act_stats:
                act_stats[act] = []
            act_stats[act].append(result['direction_acc'])
        
        for act, accs in act_stats.items():
            avg_acc = np.mean(accs)
            max_acc = np.max(accs)
            print(f"  {act:<8}: Avg {avg_acc:.1f}%, Max {max_acc:.1f}%")
    
    def _analyze_by_inlet_velocity(self, result_list):
        """Analyze results by inlet velocity"""
        print(f"\n🌊 INLET VELOCITY ANALYSIS:")
        vel_stats = {}
        for result in result_list:
            vel = result['inlet_vel']
            if vel not in vel_stats:
                vel_stats[vel] = []
            vel_stats[vel].append(result['direction_acc'])
        
        for vel, accs in vel_stats.items():
            avg_acc = np.mean(accs)
            max_acc = np.max(accs)
            print(f"  {vel:>8.3f} m/s: Avg {avg_acc:.1f}%, Max {max_acc:.1f}%")


if __name__ == "__main__":
    study = HyperparameterStudy()
    study.run_study()