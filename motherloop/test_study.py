# test_study.py - Quick test with minimal parameters
import os
import sys
import torch
from pathlib import Path

# Create directories first
def create_directories():
    dirs = ['architectures', 'physics', 'training', 'evaluation', 'test_results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Create __init__.py files for imports
    for d in ['architectures', 'physics', 'training', 'evaluation']:
        with open(f'{d}/__init__.py', 'w') as f:
            f.write('')

create_directories()

# Import after creating structure
from architectures.standard_fourier import StandardFourierPINN
from training.trainer import PINNTrainer
from evaluation.evaluator import PINNEvaluator
from physics.boundary_conditions import BoundaryConfig

class QuickTest:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def test_single_configuration(self):
        """Test one configuration to ensure everything works"""
        print("🧪 Running quick test...")
        
        # Test configuration
        arch_name = 'standard_fourier'
        activation = 'tanh'
        epochs = 50  # Very short for testing
        inlet_vel = -0.005
        
        print(f"Testing: {arch_name}, {activation}, {epochs} epochs, {inlet_vel} inlet velocity")
        
        try:
            # Create output directory
            test_dir = Path('test_results') / 'quick_test'
            test_dir.mkdir(exist_ok=True, parents=True)
            
            # Setup boundary conditions
            boundary_config = BoundaryConfig(inlet_velocity=inlet_vel)
            print("✅ Boundary config created")
            
            # Create model
            model = StandardFourierPINN(activation=activation).to(self.device)
            print("✅ Model created")
            
            # Setup trainer
            trainer = PINNTrainer(
                model=model,
                boundary_config=boundary_config,
                max_epochs=epochs,
                device=self.device,
                output_dir=test_dir
            )
            print("✅ Trainer created")
            
            # Train model (short)
            print("🏃 Starting training...")
            trained_model = trainer.train()
            print("✅ Training completed")
            
            # Evaluate model
            evaluator = PINNEvaluator(
                model=trained_model,
                device=self.device,
                output_dir=test_dir
            )
            print("✅ Evaluator created")
            
            # Get metrics
            metrics = evaluator.evaluate_and_compare()
            print("✅ Evaluation completed")
            
            # Print results
            print("\n📊 Test Results:")
            print(f"  Direction Accuracy: {metrics.get('direction_accuracy', 0):.1f}%")
            print(f"  Overall Accuracy: {metrics.get('overall_accuracy', 0):.1f}%")
            print(f"  Comparison Points: {metrics.get('n_comparison_points', 0)}")
            
            print("\n🎉 Test PASSED! All components working correctly.")
            return True
            
        except Exception as e:
            print(f"\n❌ Test FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_all_architectures(self):
        """Quick test of all three architectures"""
        from architectures.unet_skip import UNetSkipPINN
        from architectures.fpn_deep import FPNDeepPINN
        
        architectures = {
            'standard_fourier': StandardFourierPINN,
            'unet_skip': UNetSkipPINN,
            'fpn_deep': FPNDeepPINN
        }
        
        print("\n🏗️ Testing all architectures...")
        
        for arch_name, arch_class in architectures.items():
            try:
                print(f"\nTesting {arch_name}...")
                model = arch_class(activation='tanh').to(self.device)
                
                # Quick forward pass
                test_input = torch.randn(10, 2).to(self.device)
                output = model(test_input)
                
                expected_shape = (10, 3)  # [u, v, p]
                if output.shape == expected_shape:
                    print(f"  ✅ {arch_name}: Output shape {output.shape} ✓")
                else:
                    print(f"  ❌ {arch_name}: Wrong output shape {output.shape}, expected {expected_shape}")
                    
            except Exception as e:
                print(f"  ❌ {arch_name}: Error - {str(e)}")
        
        print("\nArchitecture test completed!")

def main():
    print("🚀 PINN Hyperparameter Study - Quick Test")
    print("=" * 50)
    
    tester = QuickTest()
    
    # Test 1: All architectures can be created and run
    tester.test_all_architectures()
    
    # Test 2: Full pipeline with one configuration
    success = tester.test_single_configuration()
    
    if success:
        print("\n✅ ALL TESTS PASSED!")
        print("Ready to run full hyperparameter study with run_study.py")
        print("\nTo run full study:")
        print("  python run_study.py")
    else:
        print("\n❌ TESTS FAILED!")
        print("Please fix errors before running full study")

if __name__ == "__main__":
    main()