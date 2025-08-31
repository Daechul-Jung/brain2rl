#!/usr/bin/env python3
"""
Script to run the Integrated Classification and Tokenization Pipeline on your actual data

This script demonstrates how to use the pipeline with your real sensor data.
"""

import os
import sys
import argparse
from pathlib import Path

# Add core directory to path
sys.path.append('core')

from integrated_classification_tokenization import IntegratedPipeline, create_default_config

def main():
    """Main function to run the pipeline on your data"""
    print("Integrated Classification and Tokenization Pipeline")
    print("=" * 60)
    print()
    
    # Configuration for your data
    config = {
        'window_size': 100,        # Adjust based on your action duration
        'batch_size': 32,          # Adjust based on your memory
        'classifier_lr': 0.001,    # Learning rate for classifier
        'classifier_epochs': 100,  # Number of training epochs
        'classifier_dropout': 0.3, # Dropout rate for classifier
        'tokenizer_lr': 0.0001,   # Learning rate for tokenizer
        'tokenizer_epochs': 50,    # Number of training epochs
        'tokenizer_dropout': 0.1,  # Dropout rate for tokenizer
        'n_tokens': 512,          # Number of tokens in vocabulary
        'embedding_dim': 128,      # Embedding dimension
        'nhead': 8,               # Number of attention heads
        'num_encoder_layers': 6    # Number of transformer layers
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Data directory - CHANGE THIS TO YOUR ACTUAL DATA PATH
    data_dir = "data"  # Replace with your actual data directory path
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        print(f"Please update the 'data_dir' variable in this script to point to your sensor data.")
        print(f"Your data should be in CSV format with columns like:")
        print(f"  acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, label")
        return
    
    print(f"📁 Data directory: {data_dir}")
    
    # Check for CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"❌ No CSV files found in '{data_dir}'")
        print(f"Please ensure your data directory contains CSV files with sensor data.")
        return
    
    print(f"📊 Found {len(csv_files)} CSV files:")
    for f in csv_files[:5]:
        print(f"  - {f}")
    if len(csv_files) > 5:
        print(f"  ... and {len(csv_files) - 5} more files")
    print()
    
    # Create output directories
    os.makedirs('output', exist_ok=True)
    os.makedirs('models/classification', exist_ok=True)
    os.makedirs('models/tokenization', exist_ok=True)
    
    try:
        # Initialize pipeline
        print("🚀 Initializing pipeline...")
        pipeline = IntegratedPipeline(config)
        
        # Run full pipeline
        print("🔄 Starting pipeline execution...")
        results = pipeline.run_full_pipeline(data_dir)
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Print results summary
        print(f"\n📈 Results Summary:")
        print(f"  - Data shape: {results['data_info']['original_shape']}")
        print(f"  - Number of classes: {results['data_info']['n_classes']}")
        print(f"  - Number of channels: {results['data_info']['n_channels']}")
        print(f"  - Generated tokens: {results['token_data']['token_shape']}")
        
        print(f"\n💾 Files saved:")
        print(f"  - Pipeline results: output/pipeline_results.pth")
        print(f"  - Generated tokens: output/generated_tokens.npz")
        print(f"  - Best classifier: models/classification/best_classifier.pth")
        print(f"  - Best tokenizer: models/tokenization/best_tokenizer.pth")
        
        # Generate training plots
        print(f"\n📊 Generating training plots...")
        if 'classifier_history' in results:
            pipeline.plot_training_history(
                results['classifier_history'], 
                'output/classifier_training.png'
            )
            print(f"  ✅ Classifier training plot: output/classifier_training.png")
        
        if 'tokenizer_history' in results:
            pipeline.plot_training_history(
                results['tokenizer_history'], 
                'output/tokenizer_training.png'
            )
            print(f"  ✅ Tokenizer training plot: output/tokenizer_training.png")
        
        print(f"\n🎯 Your tokens are ready for reinforcement learning!")
        print(f"Next steps:")
        print(f"  1. Run: python3 example_rl_integration.py")
        print(f"  2. Integrate tokens with your RL algorithm")
        print(f"  3. Train your RL agent")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        print(f"Check the log file 'pipeline.log' for detailed error information")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
