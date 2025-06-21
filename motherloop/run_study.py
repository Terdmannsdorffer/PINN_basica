# run_study.py - Quick execution script
"""
Quick execution script for the hyperparameter study
Run this to start the complete study
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import the study
from hyperparameter_study import HyperparameterStudy

def create_directory_structure():
    """Create necessary directories"""
    dirs = [
        'architectures',
        'physics', 
        'training',
        'evaluation',
        'hyperparameter_results'
    ]
    
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    print("✅ Directory structure created")

def main():
    print("🚀 Starting PINN Hyperparameter Study")
    print("="*50)
    
    # Create directories
    create_directory_structure()
    
    # Initialize and run study
    study = HyperparameterStudy()
    
    print("Study configuration:")
    print(f"  Architectures: {list(study.architectures.keys())}")
    print(f"  Activations: {study.activations}")
    print(f"  Epoch configs: {study.epoch_configs}")
    print(f"  Inlet velocities: {study.inlet_velocities}")
    
    total_experiments = (len(study.architectures) * len(study.activations) * 
                        len(study.epoch_configs) * len(study.inlet_velocities))
    print(f"  Total experiments: {total_experiments}")
    
    # Run the study
    study.run_study()
    
    print("\n🎉 Study completed!")
    print("Check 'hyperparameter_results' folder for detailed results")

if __name__ == "__main__":
    main()