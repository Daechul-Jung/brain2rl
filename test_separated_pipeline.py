#!/usr/bin/env python3
"""
Test script for the separated pipeline components

This script tests the modular components of the Brain2RL pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add core directory to path
sys.path.append('core')

from pipeline import Brain2RLPipeline, create_default_config

def create_sample_data(data_dir: str, n_subjects: int = 2, n_samples: int = 1000):
    """Create sample sensor data for testing"""
    os.makedirs(data_dir, exist_ok=True)
    
    for subject_id in range(n_subjects):
        # Generate random sensor data (6 channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        sensor_data = np.random.randn(n_samples, 6)
        
        # Add some temporal structure
        time = np.linspace(0, 10, n_samples)
        sensor_data[:, 0] += 0.5 * np.sin(2 * np.pi * time)  # Add sine wave to acc_x
        sensor_data[:, 1] += 0.3 * np.cos(2 * np.pi * time)  # Add cosine wave to acc_y
        
        # Create action labels (5 different actions)
        labels = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.3, 0.2, 0.2, 0.15, 0.15])
        
        # Add some temporal consistency to labels
        for i in range(1, len(labels)):
            if np.random.random() < 0.8:  # 80% chance to keep same label
                labels[i] = labels[i-1]
        
        # Create DataFrame
        df = pd.DataFrame(sensor_data, columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
        df['label'] = labels
        
        # Save to CSV
        file_path = os.path.join(data_dir, f'SUBJ_{subject_id:03d}.csv')
        df.to_csv(file_path, index=False)
        print(f"Created sample data: {file_path}")

def test_separated_components():
    """Test individual separated components"""
    print("Testing Separated Pipeline Components")
    print("=" * 50)
    
    # Create sample data
    data_dir = "test_data_separated"
    create_sample_data(data_dir, n_subjects=2, n_samples=500)
    
    # Test configuration
    config = create_default_config()
    config.update({
        'window_size': 50,
        'batch_size': 16,
        'classifier_epochs': 3,
        'tokenizer_epochs': 2,
        'n_tokens': 32,
        'embedding_dim': 32,
    })
    
    print(f"Configuration: {config}")
    
    try:
        # Test pipeline initialization
        print("\n1. Testing pipeline initialization...")
        pipeline = Brain2RLPipeline(config)
        print("   ✅ Pipeline initialized successfully")
        
        # Test data loading and preprocessing
        print("\n2. Testing data loading and preprocessing...")
        train_loader, val_loader, test_loader = pipeline.load_and_preprocess_data(data_dir)
        print(f"   ✅ Data loaded and preprocessed")
        print(f"   - Train: {len(train_loader.dataset)} samples")
        print(f"   - Val: {len(val_loader.dataset)} samples")
        print(f"   - Test: {len(test_loader.dataset)} samples")
        
        # Test classifier training
        print("\n3. Testing classifier training...")
        classifier_history = pipeline.train_classifier(train_loader, val_loader)
        print(f"   ✅ Classifier training completed")
        print(f"   - Final val acc: {classifier_history['val_acc'][-1]:.2f}%")
        
        # Test tokenizer training
        print("\n4. Testing tokenizer training...")
        tokenizer_history = pipeline.train_tokenizer(train_loader, val_loader)
        print(f"   ✅ Tokenizer training completed")
        print(f"   - Final val loss: {tokenizer_history['val_loss'][-1]:.4f}")
        
        # Test token generation
        print("\n5. Testing token generation...")
        token_data = pipeline.generate_tokens(test_loader)
        print(f"   ✅ Token generation completed")
        print(f"   - Tokens shape: {token_data['token_shape']}")
        
        # Test RL state creation
        print("\n6. Testing RL state creation...")
        rl_states = pipeline.create_rl_states(token_data)
        print(f"   ✅ RL states created")
        print(f"   - PPO states: {rl_states['ppo_states']['state_dim']} dimensions")
        print(f"   - SAC states: {rl_states['sac_states']['state_dim']} dimensions")
        
        print("\n✅ All separated component tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Component test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_full_separated_pipeline():
    """Test the complete separated pipeline"""
    print("\nTesting Complete Separated Pipeline")
    print("=" * 50)
    
    # Create sample data
    data_dir = "test_data_separated"
    if not os.path.exists(data_dir):
        create_sample_data(data_dir, n_subjects=2, n_samples=500)
    
    config = create_default_config()
    config.update({
        'window_size': 50,
        'batch_size': 16,
        'classifier_epochs': 3,
        'tokenizer_epochs': 2,
        'n_tokens': 32,
        'embedding_dim': 32,
    })
    
    try:
        # Initialize pipeline
        pipeline = Brain2RLPipeline(config)
        
        # Run full pipeline
        print("Running complete separated pipeline...")
        results = pipeline.run_full_pipeline(data_dir)
        
        print("\n✅ Complete separated pipeline test passed!")
        print(f"Results summary:")
        print(f"  - Data info: {results['data_info']}")
        print(f"  - Token data: {results['token_data']['token_shape']}")
        print(f"  - RL states created successfully")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Full pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Separated Pipeline Component Test Suite")
    print("=" * 60)
    
    # Test individual components first
    component_test_passed = test_separated_components()
    
    if component_test_passed:
        print("\n" + "=" * 60)
        # Test full pipeline
        full_pipeline_results = test_full_separated_pipeline()
        
        if full_pipeline_results:
            print("\n🎉 All separated pipeline tests passed!")
            print(f"Results saved to: output/")
            print(f"Models saved to: models/")
            print(f"RL states saved to: output/rl_states.npz")
        else:
            print("\n❌ Full separated pipeline test failed.")
    else:
        print("\n❌ Component tests failed. Check the errors above.")
