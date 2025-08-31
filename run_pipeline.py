#!/usr/bin/env python3
"""
Main script to run the Integrated Classification and Tokenization Pipeline

Usage:
    python run_pipeline.py --data-dir data --config config/pipeline_config.json
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add core directory to path
sys.path.append('core')

from core.integrated_classification_tokenization import IntegratedPipeline, create_default_config

def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str = None) -> dict:
    """Load configuration from file or create default"""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_path}")
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            print("Using default configuration")
            config = create_default_config()
    else:
        print("No config file specified, using default configuration")
        config = create_default_config()
    
    return config

def validate_data_directory(data_dir: str) -> bool:
    """Validate that the data directory contains sensor data files"""
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist")
        return False
    
    # Check for CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in '{data_dir}'")
        print("Expected format: CSV files with sensor data columns and 'label' column")
        return False
    
    print(f"Found {len(csv_files)} CSV files in data directory")
    for f in csv_files[:5]:  # Show first 5 files
        print(f"  - {f}")
    if len(csv_files) > 5:
        print(f"  ... and {len(csv_files) - 5} more files")
    
    return True

def run_pipeline(data_dir: str, config: dict, subject_ids: list = None, output_dir: str = 'output'):
    """Run the complete pipeline"""
    print(f"\n{'='*60}")
    print("INTEGRATED CLASSIFICATION AND TOKENIZATION PIPELINE")
    print(f"{'='*60}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models/classification', exist_ok=True)
    os.makedirs('models/tokenization', exist_ok=True)
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = IntegratedPipeline(config)
    
    try:
        # Run full pipeline
        print(f"\nStarting pipeline with data from: {data_dir}")
        if subject_ids:
            print(f"Processing subjects: {subject_ids}")
        
        results = pipeline.run_full_pipeline(data_dir, subject_ids)
        
        # Save configuration used
        config_save_path = os.path.join(output_dir, 'pipeline_config.json')
        with open(config_save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        
        # Print summary
        print(f"\nResults Summary:")
        print(f"  - Data shape: {results['data_info']['original_shape']}")
        print(f"  - Number of classes: {results['data_info']['n_classes']}")
        print(f"  - Number of channels: {results['data_info']['n_channels']}")
        print(f"  - Generated tokens: {results['token_data']['token_shape']}")
        
        print(f"\nFiles saved:")
        print(f"  - Pipeline results: {os.path.join(output_dir, 'pipeline_results.pth')}")
        print(f"  - Generated tokens: {os.path.join(output_dir, 'generated_tokens.npz')}")
        print(f"  - Best classifier: models/classification/best_classifier.pth")
        print(f"  - Best tokenizer: models/tokenization/best_tokenizer.pth")
        print(f"  - Configuration: {config_save_path}")
        
        # Plot training histories
        print(f"\nGenerating training plots...")
        if 'classifier_history' in results:
            pipeline.plot_training_history(
                results['classifier_history'], 
                os.path.join(output_dir, 'classifier_training.png')
            )
            print(f"  - Classifier training plot: {os.path.join(output_dir, 'classifier_training.png')}")
        
        if 'tokenizer_history' in results:
            pipeline.plot_training_history(
                results['tokenizer_history'], 
                os.path.join(output_dir, 'tokenizer_training.png')
            )
            print(f"  - Tokenizer training plot: {os.path.join(output_dir, 'tokenizer_training.png')}")
        
        return results
        
    except Exception as e:
        print(f"\n Pipeline failed: {str(e)}")
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Integrated Classification and Tokenization Pipeline for Sensor Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python run_pipeline.py --data-dir /path/to/your/data
  
  # Run with custom configuration
  python run_pipeline.py --data-dir /path/to/your/data --config config/pipeline_config.json
  
  # Run on specific subjects
  python run_pipeline.py --data-dir /path/to/your/data --subject-ids SUBJ_001 SUBJ_002
  
  # Specify output directory
  python run_pipeline.py --data-dir /path/to/your/data --output-dir my_results
        """
    )
    
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing sensor data CSV files')
    parser.add_argument('--config', type=str, default='config/pipeline_config.json',
                       help='Path to configuration file (default: config/pipeline_config.json)')
    parser.add_argument('--subject-ids', nargs='+',
                       help='List of specific subject IDs to process (if not specified, processes all CSV files)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results (default: output)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate data directory
    if not validate_data_directory(args.data_dir):
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Print configuration summary
    print(f"\nConfiguration Summary:")
    print(f"  - Window size: {config['window_size']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Classifier epochs: {config['classifier_epochs']}")
    print(f"  - Tokenizer epochs: {config['tokenizer_epochs']}")
    print(f"  - Number of tokens: {config['n_tokens']}")
    print(f"  - Embedding dimension: {config['embedding_dim']}")
    
    try:
        # Run pipeline
        results = run_pipeline(args.data_dir, config, args.subject_ids, args.output_dir)
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"Your tokens are ready for reinforcement learning!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        print("Check the log file 'pipeline.log' for detailed error information")
        sys.exit(1)

if __name__ == "__main__":
    main()

