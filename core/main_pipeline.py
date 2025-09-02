"""
Main Pipeline Orchestrator
==========================

This module orchestrates the complete Brain2RL pipeline:
1. Classification: Sensor data -> Action classification
2. Tokenization: Classified data -> Tokens with Q/K/V matrices
3. RL Training: Token-guided KUKA robot training
4. Simulation: Real-time KUKA robot simulation

Author: Daechul Jung
Version: 1.0.0
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.tokenization_pipeline import TokenizationPipeline
from core.rl_training_pipeline import RLTrainingPipeline


class Brain2RLMainPipeline:
    """
    Main pipeline that orchestrates the complete Brain2RL workflow
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the main pipeline
        
        Args:
            config: Configuration dictionary with pipeline settings
        """
        self.config = config
        self.logger = self._setup_logger()
        
        # Initialize pipeline components
        self.classification_pipeline = None
        self.tokenization_pipeline = None
        self.rl_training_pipeline = None
        self.simulation_pipeline = None
        
        # Pipeline state
        self.is_initialized = False
        self.training_active = False
        self.simulation_active = False
        
        self.logger.info("Brain2RL Main Pipeline initialized")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('Brain2RL')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize_pipelines(self):
        """Initialize all pipeline components"""
        self.logger.info("Initializing pipeline components...")
        
        try:

            # Initialize tokenization pipeline
            self.tokenization_pipeline = TokenizationPipeline(
                model_config=self.config['tokenization']
            )
            
            # Initialize RL training pipeline
            self.rl_training_pipeline = RLTrainingPipeline(
                model_config=self.config['rl_training']
            )
            
        
            self.is_initialized = True
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipelines: {str(e)}")
            raise
    
    def run_classification_only(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run only the classification pipeline
        
        Args:
            data_path: Path to sensor data
        """
        if not self.is_initialized:
            self.initialize_pipelines()
        
        self.logger.info("Running classification pipeline...")
        
        # Load and classify sensor data
        classified_actions, confidence_scores = self.classification_pipeline.classify_sensor_data(data_path)
        
        self.logger.info(f"Classification complete. Processed {len(classified_actions)} samples")
        return classified_actions, confidence_scores
    
    def run_tokenization_only(self, classified_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run only the tokenization pipeline
        
        Args:
            classified_data: Output from classification pipeline
            
        Returns:
            Dictionary with tokens, query, key, value matrices
        """
        if not self.is_initialized:
            self.initialize_pipelines()
        
        self.logger.info("Running tokenization pipeline...")
        
        # Tokenize classified data
        token_data = self.tokenization_pipeline.tokenize_time_series(classified_data)
        
        self.logger.info(f"Tokenization complete. Generated {token_data['tokens'].shape[0]} tokens")
        return token_data
    
    def run_rl_training_only(self, token_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Run only the RL training pipeline
        
        Args:
            token_data: Output from tokenization pipeline
            
        Returns:
            Dictionary with training results and trained model
        """
        if not self.is_initialized:
            self.initialize_pipelines()
        
        self.logger.info("Running RL training pipeline...")
        
        # Train RL agent with tokens
        training_results = self.rl_training_pipeline.train_with_tokens(token_data)
        
        self.logger.info("RL training complete")
        return training_results
    
    def run_simulation_only(self, trained_model: Any) -> Dict[str, Any]:
        """
        Run only the simulation pipeline
        
        Args:
            trained_model: Trained RL model
            
        Returns:
            Dictionary with simulation results
        """
        if not self.is_initialized:
            self.initialize_pipelines()
        
        self.logger.info("Running simulation pipeline...")
        
        # Run KUKA simulation
        simulation_results = self.simulation_pipeline.run_simulation(trained_model)
        
        self.logger.info("Simulation complete")
        return simulation_results
    
    def run_full_pipeline(self, data_path: str) -> Dict[str, Any]:
        """
        Run the complete pipeline from sensor data to simulation
        
        Args:
            data_path: Path to sensor data
            
        Returns:
            Dictionary with complete pipeline results
        """
        if not self.is_initialized:
            self.initialize_pipelines()
        
        self.logger.info("Starting full Brain2RL pipeline...")
        results = {}
        
        try:
            # Step 1: Classification
            self.logger.info("Step 1/4: Running classification...")
            classified_actions, confidence_scores = self.run_classification_only(data_path)
            results['classification'] = {
                'actions': classified_actions,
                'confidence': confidence_scores
            }
            
            # Step 2: Tokenization
            self.logger.info("Step 2/4: Running tokenization...")
            token_data = self.run_tokenization_only(classified_actions)
            results['tokenization'] = token_data
            
            # Step 3: RL Training
            self.logger.info("Step 3/4: Running RL training...")
            training_results = self.run_rl_training_only(token_data)
            results['rl_training'] = training_results
            
            # Step 4: Simulation
            self.logger.info("Step 4/4: Running simulation...")
            simulation_results = self.run_simulation_only(training_results['trained_model'])
            results['simulation'] = simulation_results
            
            self.logger.info("Full pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
        
        return results
    
    def run_real_time_pipeline(self, data_stream_source: str):
        """
        Run the pipeline in real-time mode with continuous data streaming
        
        Args:
            data_stream_source: Source of real-time data stream
        """
        if not self.is_initialized:
            self.initialize_pipelines()
        
        self.logger.info("Starting real-time pipeline...")
        
        # This would be implemented for real-time streaming
        # For now, we'll create a placeholder
        self.logger.info("Real-time pipeline not yet implemented")
        raise NotImplementedError("Real-time pipeline coming soon")
    
    def save_pipeline_state(self, save_path: str):
        """Save the current pipeline state"""
        save_data = {
            'config': self.config,
            'is_initialized': self.is_initialized,
            'training_active': self.training_active,
            'simulation_active': self.simulation_active
        }
        
        torch.save(save_data, save_path)
        self.logger.info(f"Pipeline state saved to {save_path}")
    
    def load_pipeline_state(self, load_path: str):
        """Load pipeline state from file"""
        save_data = torch.load(load_path)
        
        self.config = save_data['config']
        self.is_initialized = save_data['is_initialized']
        self.training_active = save_data['training_active']
        self.simulation_active = save_data['simulation_active']
        
        self.logger.info(f"Pipeline state loaded from {load_path}")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for the pipeline"""
    return {
        'data_dir': 'data/',
        'classification': {
            'model_type': 'eeg_cnn',
            'n_channels': 32,
            'n_times': 512,
            'n_classes': 6,
            'dropout_rate': 0.5,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100
        },
        'tokenization': {
            'embedding_dim': 128,
            'n_tokens': 512,
            'nhead': 8,
            'num_encoder_layers': 6,
            'dropout': 0.1,
            'max_sequence_length': 1000
        },
        'rl_training': {
            'algorithm': 'ppo',
            'learning_rate': 0.0003,
            'batch_size': 64,
            'n_steps': 2048,
            'n_epochs': 10,
            'gamma': 0.99,
            'clip_range': 0.2
        },
        'simulation': {
            'robot_type': 'kuka_iiwa',
            'use_gui': True,
            'real_time_factor': 1.0,
            'max_episode_steps': 1000,
            'control_frequency': 100
        }
    }


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Brain2RL Pipeline')
    
    parser.add_argument('--mode', choices=['full', 'classification', 'tokenization', 'rl_training', 'simulation'], 
                       default='full', help='Pipeline mode to run')
    parser.add_argument('--data-path', type=str, required=True, help='Path to sensor data')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='output/', help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--real-time', action='store_true', help='Run in real-time mode')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    config['device'] = device
    
    # Initialize pipeline
    pipeline = Brain2RLMainPipeline(config)
    
    try:
        if args.real_time:
            pipeline.run_real_time_pipeline(args.data_path)
        elif args.mode == 'full':
            results = pipeline.run_full_pipeline(args.data_path)
            # Save results
            torch.save(results, os.path.join(args.output_dir, 'pipeline_results.pth'))
        elif args.mode == 'tokenization':
            # Load previous classification results
            classified_data = torch.load(os.path.join(args.output_dir, 'classification_results.pth'))
            results = pipeline.run_tokenization_only(classified_data[0])
            torch.save(results, os.path.join(args.output_dir, 'tokenization_results.pth'))
        elif args.mode == 'rl_training':
            # Load previous tokenization results
            token_data = torch.load(os.path.join(args.output_dir, 'tokenization_results.pth'))
            results = pipeline.run_rl_training_only(token_data)
            torch.save(results, os.path.join(args.output_dir, 'rl_training_results.pth'))

        
        print(f"Pipeline completed successfully! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main() 