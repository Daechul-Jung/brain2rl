#!/usr/bin/env python3
"""
Test script for the Integrated Classification and Tokenization Pipeline

This script creates sample sensor data and tests the complete pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Add core directory to path
sys.path.append('core')

from integrated_classification_tokenization import IntegratedPipeline, create_default_config

def create_sample_data(data_dir: str, n_subjects: int = 2, n_samples: int = 1000):
    """
    Create sample sensor data for testing
    
    Args:
        data_dir: Directory to save sample data
        n_subjects: Number of subjects to create
        n_samples: Number of samples per subject
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample sensor data
    for subject_id in range(n_subjects):
        # Generate random sensor data (6 channels: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
        sensor_data = np.random.randn(n_samples, 6)
        
        # Add some temporal structure
        time = np.linspace(0, 10, n_samples)
        sensor_data[:, 0] += 0.5 * np.sin(2 * np.pi * time)  # Add sine wave to acc_x
        sensor_data[:, 1] += 0.3 * np.cos(2 * np.pi * time)  # Add cosine wave to acc_y
        
        # Create action labels (5 different actions)
        # Actions: 0=idle, 1=walk, 2=run, 3=jump, 4=sit
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

def test_pipeline():
    """Test the integrated pipeline with sample data"""
    print("Testing Integrated Classification and Tokenization Pipeline")
    print("=" * 60)
    
    # Create sample data
    data_dir = "test_data"
    create_sample_data(data_dir, n_subjects=2, n_samples=1000)
    
    # Create configuration
    config = create_default_config()
    
    # Modify config for faster testing
    config.update({
        'window_size': 50,  # Smaller window for faster processing
        'batch_size': 16,   # Smaller batch size
        'classifier_epochs': 5,  # Fewer epochs for testing
        'tokenizer_epochs': 3,   # Fewer epochs for testing
        'n_tokens': 64,     # Fewer tokens for testing
        'embedding_dim': 64, # Smaller embedding for testing
    })
    
    print(f"Configuration: {config}")
    
    # Initialize pipeline
    pipeline = IntegratedPipeline(config)
    
    try:
        # Run full pipeline
        print("\nStarting pipeline...")
        results = pipeline.run_full_pipeline(data_dir)
        
        print("\nPipeline completed successfully!")
        print(f"Data info: {results['data_info']}")
        print(f"Token data shape: {results['token_data']['token_shape']}")
        
        # Test token generation on a small batch
        print("\nTesting token generation...")
        test_tokens = results['token_data']['tokens'][:5]  # First 5 sequences
        print(f"Sample tokens shape: {test_tokens.shape}")
        print(f"Sample tokens (first sequence, first 5 time steps, first 5 token dimensions):")
        print(test_tokens[0, :5, :5])
        
        return results
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_individual_components():
    """Test individual pipeline components"""
    print("\nTesting Individual Components")
    print("=" * 40)
    
    # Create sample data
    data_dir = "test_data"
    if not os.path.exists(data_dir):
        create_sample_data(data_dir, n_subjects=1, n_samples=500)
    
    config = create_default_config()
    config.update({
        'window_size': 50,
        'batch_size': 16,
        'classifier_epochs': 3,
        'tokenizer_epochs': 2,
        'n_tokens': 32,
        'embedding_dim': 32,
    })
    
    pipeline = IntegratedPipeline(config)
    
    try:
        # Test data loading
        print("1. Testing data loading...")
        X, y = pipeline.load_sensor_data(data_dir)
        print(f"   Loaded data: {X.shape}, labels: {y.shape}")
        
        # Test preprocessing
        print("2. Testing data preprocessing...")
        X_processed, y_processed = pipeline.preprocess_data(X, y)
        print(f"   Processed data: {X_processed.shape}, encoded labels: {y_processed.shape}")
        
        # Test dataloader creation
        print("3. Testing dataloader creation...")
        train_loader, val_loader, test_loader = pipeline.create_dataloaders(X_processed, y_processed, 16)
        print(f"   Created dataloaders - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
        
        # Test classifier training
        print("4. Testing classifier training...")
        classifier_history = pipeline.train_classifier(train_loader, val_loader)
        print(f"   Classifier training completed. Final val acc: {classifier_history['val_acc'][-1]:.2f}%")
        
        # Test tokenizer training
        print("5. Testing tokenizer training...")
        tokenizer_history = pipeline.train_tokenizer(train_loader, val_loader)
        print(f"   Tokenizer training completed. Final val loss: {tokenizer_history['val_loss'][-1]:.4f}")
        
        # Test token generation
        print("6. Testing token generation...")
        token_data = pipeline.generate_tokens(test_loader)
        print(f"   Token generation completed. Tokens shape: {token_data['token_shape']}")
        
        print("\nAll component tests passed!")
        return True
        
    except Exception as e:
        print(f"Component test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Integrated Pipeline Test Suite")
    print("=" * 50)
    
    # Test individual components first
    component_test_passed = test_individual_components()
    
    if component_test_passed:
        print("\n" + "=" * 50)
        # Test full pipeline
        full_pipeline_results = test_pipeline()
        
        if full_pipeline_results:
            print("\n✅ All tests passed! Pipeline is working correctly.")
            print(f"Results saved to: output/")
            print(f"Models saved to: models/")
        else:
            print("\n❌ Full pipeline test failed.")
    else:
        print("\n❌ Component tests failed. Check the errors above.")
