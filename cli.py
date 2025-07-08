#!/usr/bin/env python3
"""
Brain2RL Command Line Interface
==============================

This script provides a comprehensive command-line interface for running the
Brain2RL pipeline, both in full mode and individual component modes.

Author: Brain2RL Team
Version: 1.0.0

Usage:
    python cli.py full --data-path data/ --output-dir output/
    python cli.py classification --data-path data/ --model-path models/classifier.pth
    python cli.py tokenization --classified-data results/classification.npz
    python cli.py rl-training --token-data results/tokenization.npz
    python cli.py simulation --model-path models/trained_agent.pth
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain2rl.core.main_pipeline import Brain2RLMainPipeline, create_default_config
from brain2rl.core.classification_pipeline import ClassificationPipeline
from brain2rl.core.tokenization_pipeline import TokenizationPipeline
from brain2rl.core.rl_training_pipeline import RLTrainingPipeline
from brain2rl.core.simulation_pipeline import SimulationPipeline
from brain2rl.utils.data_utils import generate_synthetic_sensor_data, save_processed_data


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        return create_default_config()


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to JSON file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def run_full_pipeline(args) -> Dict[str, Any]:
    """Run the complete Brain2RL pipeline"""
    logger = logging.getLogger('Brain2RL.CLI')
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
    
    # Override config with command line arguments
    if args.device:
        config['device'] = args.device
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pipeline
    pipeline = Brain2RLMainPipeline(config)
    
    try:
        # Run full pipeline
        logger.info("Starting full Brain2RL pipeline...")
        results = pipeline.run_full_pipeline(args.data_path)
        
        # Save results
        import torch
        results_path = os.path.join(args.output_dir, 'full_pipeline_results.pth')
        torch.save(results, results_path)
        
        # Save configuration used
        config_path = os.path.join(args.output_dir, 'pipeline_config.json')
        save_config(config, config_path)
        
        logger.info(f"Pipeline completed successfully! Results saved to {args.output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("BRAIN2RL PIPELINE SUMMARY")
        print("="*60)
        print(f"Classification Accuracy: {results['classification']['confidence'].mean():.3f}")
        print(f"Tokens Generated: {results['tokenization']['tokens'].shape[0]}")
        print(f"RL Training Improvement: {results['rl_training']['final_performance']['improvement']:.3f}")
        print(f"Simulation Success Rate: {results['simulation']['success_rate']:.3f}")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def run_classification_only(args) -> Dict[str, Any]:
    """Run only the classification pipeline"""
    logger = logging.getLogger('Brain2RL.CLI')
    
    # Create model config
    model_config = {
        'n_channels': args.n_channels,
        'n_times': args.n_times,
        'n_classes': args.n_classes,
        'dropout_rate': args.dropout_rate,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }
    
    # Initialize pipeline
    pipeline = ClassificationPipeline(
        data_dir=os.path.dirname(args.data_path),
        model_config=model_config
    )
    
    try:
        if args.mode == 'train':
            # Load and prepare data
            data, labels = pipeline.load_and_preprocess_data(args.data_path)
            pipeline.prepare_datasets(data, labels)
            
            # Train model
            logger.info("Training classification model...")
            history = pipeline.train_model()
            
            # Save model
            if args.model_path:
                pipeline.save_model(args.model_path)
            
            # Evaluate
            results = pipeline.evaluate_model()
            
            logger.info(f"Training completed. Test accuracy: {results['test_accuracy']:.2f}%")
            return results
            
        elif args.mode == 'classify':
            # Load model
            if args.model_path:
                pipeline.load_model(args.model_path)
            
            # Classify data
            logger.info("Classifying sensor data...")
            predictions, confidences = pipeline.classify_sensor_data(args.data_path)
            
            # Save results
            import numpy as np
            results_path = args.output_path or 'classification_results.npz'
            np.savez(results_path, predictions=predictions, confidences=confidences)
            
            logger.info(f"Classification completed. Results saved to {results_path}")
            return {'predictions': predictions, 'confidences': confidences}
            
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise


def run_tokenization_only(args) -> Dict[str, Any]:
    """Run only the tokenization pipeline"""
    logger = logging.getLogger('Brain2RL.CLI')
    
    # Create model config
    model_config = {
        'embedding_dim': args.embedding_dim,
        'n_tokens': args.n_tokens,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'dropout': args.dropout,
        'max_sequence_length': args.max_sequence_length,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    }
    
    # Initialize pipeline
    pipeline = TokenizationPipeline(model_config)
    
    try:
        # Load classified data
        import numpy as np
        data = np.load(args.classified_data)
        
        if isinstance(data, np.lib.npyio.NpzFile):
            classified_data = data['predictions']
            action_labels = data['confidences']
        else:
            classified_data = data
            action_labels = np.zeros(len(classified_data))
        
        if args.mode == 'train':
            # Train tokenizer
            logger.info("Training tokenization model...")
            history = pipeline.train_tokenizer(classified_data, action_labels, epochs=args.epochs)
            
            # Save model
            if args.model_path:
                pipeline.save_models(args.model_path)
            
            logger.info(f"Training completed. Final validation accuracy: {history['val_acc'][-1]:.2f}%")
            return history
            
        elif args.mode == 'tokenize':
            # Load model
            if args.model_path:
                pipeline.load_models(args.model_path)
            
            # Tokenize data
            logger.info("Tokenizing time series data...")
            token_data = pipeline.tokenize_time_series(classified_data)
            
            # Save results
            results_path = args.output_path or 'tokenization_results.npz'
            np.savez(results_path, **token_data)
            
            logger.info(f"Tokenization completed. Results saved to {results_path}")
            return token_data
            
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        raise


def run_rl_training_only(args) -> Dict[str, Any]:
    """Run only the RL training pipeline"""
    logger = logging.getLogger('Brain2RL.CLI')
    
    # Create model config
    model_config = {
        'algorithm': args.algorithm,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'gamma': args.gamma,
        'clip_range': args.clip_range,
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'token_sequence_length': args.token_sequence_length,
        'update_frequency': args.update_frequency,
        'buffer_size': args.buffer_size
    }
    
    # Initialize pipeline
    pipeline = RLTrainingPipeline(model_config)
    
    try:
        # Load token data
        import numpy as np
        token_data = np.load(args.token_data)
        
        if not isinstance(token_data, np.lib.npyio.NpzFile):
            raise ValueError("Token data must be in .npz format")
        
        token_dict = {
            'tokens': token_data['tokens'],
            'queries': token_data['queries'],
            'keys': token_data['keys'],
            'values': token_data['values']
        }
        
        # Train agent
        logger.info("Training RL agent with tokens...")
        results = pipeline.train_with_tokens(token_dict, num_episodes=args.episodes)
        
        # Save model
        if args.model_path:
            pipeline.save_models(args.model_path)
        
        # Plot results
        if args.plot_results:
            pipeline.plot_training_results(args.output_path or 'rl_training_results.png')
        
        logger.info(f"Training completed. Final improvement: {results['final_performance']['improvement']:.2f}")
        return results
        
    except Exception as e:
        logger.error(f"RL training failed: {e}")
        raise


def run_simulation_only(args) -> Dict[str, Any]:
    """Run only the simulation pipeline"""
    logger = logging.getLogger('Brain2RL.CLI')
    
    # Create simulation config
    config = {
        'robot_type': 'kuka_iiwa',
        'task': args.task,
        'use_gui': args.use_gui,
        'use_gazebo': args.use_gazebo,
        'use_ros': args.use_ros,
        'mock_mode': args.mock_mode,
        'real_time_factor': args.real_time_factor,
        'max_episode_steps': args.max_episode_steps,
        'control_frequency': args.control_frequency,
        'max_history': args.max_history
    }
    
    # Initialize pipeline
    pipeline = SimulationPipeline(config)
    
    try:
        # Load trained model
        import torch
        trained_model = torch.load(args.model_path, map_location='cpu')
        if isinstance(trained_model, dict) and 'token_guided_agent' in trained_model:
            model = trained_model['token_guided_agent']
        else:
            model = trained_model
        
        # Load token data if provided
        if args.token_data:
            import numpy as np
            token_file = np.load(args.token_data)
            token_data = {
                'tokens': token_file['tokens'],
                'queries': token_file['queries'],
                'keys': token_file['keys'],
                'values': token_file['values']
            }
            pipeline.token_data = token_data
        
        # Run simulation
        logger.info("Running KUKA robot simulation...")
        results = pipeline.run_simulation(
            trained_model=model,
            num_episodes=args.episodes,
            visualize=args.visualize
        )
        
        # Save results
        if args.save_data:
            pipeline.save_simulation_data(args.save_data)
        
        logger.info(f"Simulation completed:")
        logger.info(f"  Episodes: {results['total_episodes']}")
        logger.info(f"  Average reward: {results['average_reward']:.2f}")
        logger.info(f"  Success rate: {results['success_rate']:.2f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise
    finally:
        pipeline.cleanup()


def generate_synthetic_data(args):
    """Generate synthetic sensor data for testing"""
    logger = logging.getLogger('Brain2RL.CLI')
    
    logger.info("Generating synthetic sensor data...")
    
    data, labels = generate_synthetic_sensor_data(
        n_samples=args.n_samples,
        n_channels=args.n_channels,
        n_timesteps=args.n_timesteps,
        n_classes=args.n_classes,
        sampling_rate=args.sampling_rate
    )
    
    # Save data
    output_path = args.output_path or 'synthetic_sensor_data.npz'
    metadata = {
        'n_samples': args.n_samples,
        'n_channels': args.n_channels,
        'n_timesteps': args.n_timesteps,
        'n_classes': args.n_classes,
        'sampling_rate': args.sampling_rate
    }
    
    save_processed_data(data, labels, output_path, metadata)
    
    logger.info(f"Synthetic data generated and saved to {output_path}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Labels shape: {labels.shape}")


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='Brain2RL Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python cli.py full --data-path data/sensor_data.npz --output-dir results/

  # Train classification model
  python cli.py classification --mode train --data-path data/sensor_data.npz --model-path models/classifier.pth

  # Classify new data
  python cli.py classification --mode classify --data-path data/new_data.npz --model-path models/classifier.pth

  # Train tokenization model
  python cli.py tokenization --mode train --classified-data results/classification.npz --model-path models/tokenizer.pth

  # Tokenize data
  python cli.py tokenization --mode tokenize --classified-data results/classification.npz --model-path models/tokenizer.pth

  # Train RL agent
  python cli.py rl-training --token-data results/tokenization.npz --episodes 1000 --model-path models/rl_agent.pth

  # Run simulation
  python cli.py simulation --model-path models/rl_agent.pth --episodes 10 --visualize

  # Generate synthetic data
  python cli.py generate-data --n-samples 1000 --n-channels 32 --output-path data/synthetic_data.npz
        """
    )
    
    # Global arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for computation')
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run the complete Brain2RL pipeline')
    full_parser.add_argument('--data-path', type=str, required=True, help='Path to sensor data')
    full_parser.add_argument('--output-dir', type=str, default='output/', help='Output directory')
    full_parser.add_argument('--config', type=str, help='Path to configuration file')
    
    # Classification command
    class_parser = subparsers.add_parser('classification', help='Run classification pipeline')
    class_parser.add_argument('--mode', choices=['train', 'classify'], default='train')
    class_parser.add_argument('--data-path', type=str, required=True, help='Path to sensor data')
    class_parser.add_argument('--model-path', type=str, help='Path to save/load model')
    class_parser.add_argument('--output-path', type=str, help='Path to save results')
    class_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    class_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    class_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    class_parser.add_argument('--n-channels', type=int, default=32, help='Number of channels')
    class_parser.add_argument('--n-times', type=int, default=512, help='Number of time steps')
    class_parser.add_argument('--n-classes', type=int, default=6, help='Number of classes')
    class_parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate')
    
    # Tokenization command
    token_parser = subparsers.add_parser('tokenization', help='Run tokenization pipeline')
    token_parser.add_argument('--mode', choices=['train', 'tokenize'], default='train')
    token_parser.add_argument('--classified-data', type=str, required=True, help='Path to classified data')
    token_parser.add_argument('--model-path', type=str, help='Path to save/load model')
    token_parser.add_argument('--output-path', type=str, help='Path to save results')
    token_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    token_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    token_parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    token_parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    token_parser.add_argument('--n-tokens', type=int, default=512, help='Number of tokens')
    token_parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    token_parser.add_argument('--num-encoder-layers', type=int, default=6, help='Number of encoder layers')
    token_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    token_parser.add_argument('--max-sequence-length', type=int, default=1000, help='Max sequence length')
    
    # RL training command
    rl_parser = subparsers.add_parser('rl-training', help='Run RL training pipeline')
    rl_parser.add_argument('--token-data', type=str, required=True, help='Path to tokenized data')
    rl_parser.add_argument('--model-path', type=str, help='Path to save model')
    rl_parser.add_argument('--output-path', type=str, help='Path to save results')
    rl_parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    rl_parser.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo', help='RL algorithm')
    rl_parser.add_argument('--learning-rate', type=float, default=0.0003, help='Learning rate')
    rl_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    rl_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    rl_parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')
    rl_parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    rl_parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    rl_parser.add_argument('--token-sequence-length', type=int, default=10, help='Token sequence length')
    rl_parser.add_argument('--update-frequency', type=int, default=100, help='Update frequency')
    rl_parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    rl_parser.add_argument('--plot-results', action='store_true', help='Plot training results')
    
    # Simulation command
    sim_parser = subparsers.add_parser('simulation', help='Run simulation pipeline')
    sim_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    sim_parser.add_argument('--token-data', type=str, help='Path to token data')
    sim_parser.add_argument('--episodes', type=int, default=10, help='Number of episodes')
    sim_parser.add_argument('--task', type=str, default='reach', help='Task type')
    sim_parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    sim_parser.add_argument('--save-data', type=str, help='Path to save simulation data')
    sim_parser.add_argument('--use-gui', action='store_true', default=True, help='Use GUI')
    sim_parser.add_argument('--use-gazebo', action='store_true', default=True, help='Use Gazebo')
    sim_parser.add_argument('--use-ros', action='store_true', default=True, help='Use ROS')
    sim_parser.add_argument('--mock-mode', action='store_true', help='Use mock mode')
    sim_parser.add_argument('--real-time-factor', type=float, default=1.0, help='Real-time factor')
    sim_parser.add_argument('--max-episode-steps', type=int, default=1000, help='Max episode steps')
    sim_parser.add_argument('--control-frequency', type=int, default=100, help='Control frequency')
    sim_parser.add_argument('--max-history', type=int, default=1000, help='Max history length')
    
    # Generate data command
    gen_parser = subparsers.add_parser('generate-data', help='Generate synthetic sensor data')
    gen_parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples')
    gen_parser.add_argument('--n-channels', type=int, default=32, help='Number of channels')
    gen_parser.add_argument('--n-timesteps', type=int, default=512, help='Number of timesteps')
    gen_parser.add_argument('--n-classes', type=int, default=6, help='Number of classes')
    gen_parser.add_argument('--sampling-rate', type=float, default=250.0, help='Sampling rate')
    gen_parser.add_argument('--output-path', type=str, help='Output path')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level, args.log_file)
    
    # Handle device selection
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Auto-selected device: {device}")
    else:
        device = args.device
    
    # Route to appropriate function
    try:
        if args.command == 'full':
            run_full_pipeline(args)
        elif args.command == 'classification':
            run_classification_only(args)
        elif args.command == 'tokenization':
            run_tokenization_only(args)
        elif args.command == 'rl-training':
            run_rl_training_only(args)
        elif args.command == 'simulation':
            run_simulation_only(args)
        elif args.command == 'generate-data':
            generate_synthetic_data(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 