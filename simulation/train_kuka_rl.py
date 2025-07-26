#!/usr/bin/env python3
"""
KUKA RL Training Script
Comprehensive training pipeline for KUKA iiwa arm reinforcement learning
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from simulation.kuka_gym_environment import KUKAGymEnvironment
from models.rl.rl_agents import KUKARLAgent

class KUKATrainingManager:
    """Manages KUKA RL training process"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.start_time = datetime.now()
        
        # Setup logging
        self._setup_logging()
        
        # Create environment
        self.env = KUKAGymEnvironment(
            task_type=config['task'],
            render_mode=config.get('render_mode', None)
        )
        
        # Get observation and action dimensions
        obs, _ = self.env.reset()
        self.obs_dim = len(obs)
        self.action_dim = 7  # KUKA iiwa has 7 joints
        
        # Create agent
        self.agent = KUKARLAgent(
            algorithm=config['algorithm'],
            observation_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=config.get('device', 'auto')
        )
        
        # Training statistics
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'evaluation_rewards': [],
            'evaluation_success_rates': [],
            'training_losses': []
        }
        
        # Create output directory
        self.output_dir = self._create_output_directory()
        
        self.logger.info(f"KUKA Training Manager initialized")
        self.logger.info(f"Task: {config['task']}, Algorithm: {config['algorithm']}")
        self.logger.info(f"Observation dim: {self.obs_dim}, Action dim: {self.action_dim}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
        self.logger = logging.getLogger('KUKATraining')
    
    def _create_output_directory(self) -> str:
        """Create output directory for training results"""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        task = self.config['task']
        algorithm = self.config['algorithm']
        
        output_dir = f"training_results/kuka_{task}_{algorithm}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training configuration
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        return output_dir
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        self.logger.info("Starting KUKA RL training...")
        
        num_episodes = self.config.get('num_episodes', 1000)
        eval_frequency = self.config.get('eval_frequency', 50)
        save_frequency = self.config.get('save_frequency', 100)
        
        best_eval_reward = -float('inf')
        recent_rewards = []
        
        try:
            for episode in range(1, num_episodes + 1):
                # Train one episode
                episode_info = self.agent.train_episode(self.env)
                
                # Log episode results
                episode_reward = episode_info['episode_reward']
                episode_steps = episode_info['episode_steps']
                success = episode_info.get('success', False)
                
                recent_rewards.append(episode_reward)
                if len(recent_rewards) > 100:
                    recent_rewards.pop(0)
                
                # Update training history
                self.training_history['episodes'].append(episode)
                self.training_history['rewards'].append(episode_reward)
                
                # Log progress
                if episode % 10 == 0:
                    avg_reward = np.mean(recent_rewards)
                    self.logger.info(
                        f"Episode {episode:4d}: "
                        f"Reward={episode_reward:7.2f}, "
                        f"Steps={episode_steps:3d}, "
                        f"Success={success}, "
                        f"Avg100={avg_reward:7.2f}"
                    )
                
                # Evaluation
                if episode % eval_frequency == 0:
                    eval_results = self.evaluate()
                    eval_reward = eval_results['mean_reward']
                    eval_success_rate = eval_results['success_rate']
                    
                    self.training_history['evaluation_rewards'].append(eval_reward)
                    self.training_history['evaluation_success_rates'].append(eval_success_rate)
                    
                    self.logger.info(
                        f"Evaluation (Ep {episode}): "
                        f"Mean Reward={eval_reward:.2f}, "
                        f"Success Rate={eval_success_rate:.2f}"
                    )
                    
                    # Save best model
                    if eval_reward > best_eval_reward:
                        best_eval_reward = eval_reward
                        best_model_path = os.path.join(self.output_dir, "best_model.pkl")
                        self.agent.save(best_model_path)
                        self.logger.info(f"New best model saved: {eval_reward:.2f}")
                
                # Periodic save
                if episode % save_frequency == 0:
                    checkpoint_path = os.path.join(self.output_dir, f"checkpoint_ep{episode}.pkl")
                    self.agent.save(checkpoint_path)
                    self._save_training_history()
                    self._plot_training_progress()
                
                # Early stopping check
                if self._should_early_stop(recent_rewards):
                    self.logger.info(f"Early stopping at episode {episode}")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        
        finally:
            # Final save
            final_model_path = os.path.join(self.output_dir, "final_model.pkl")
            self.agent.save(final_model_path)
            self._save_training_history()
            self._plot_training_progress()
            
            # Final evaluation
            final_eval = self.evaluate(num_episodes=20)
            self.logger.info(f"Final evaluation: {final_eval}")
            
            # Training summary
            training_time = datetime.now() - self.start_time
            summary = self._generate_training_summary(training_time, final_eval)
            
            self.env.close()
            
            return summary
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate current agent"""
        return self.agent.evaluate(self.env, num_episodes=num_episodes)
    
    def _should_early_stop(self, recent_rewards: List[float]) -> bool:
        """Check if training should stop early"""
        if len(recent_rewards) < 50:
            return False
        
        # Stop if no improvement in last 50 episodes
        recent_avg = np.mean(recent_rewards[-50:])
        earlier_avg = np.mean(recent_rewards[-100:-50]) if len(recent_rewards) >= 100 else 0
        
        improvement_threshold = self.config.get('early_stop_threshold', 5.0)
        
        return recent_avg - earlier_avg < improvement_threshold
    
    def _save_training_history(self):
        """Save training history to file"""
        history_path = os.path.join(self.output_dir, "training_history.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                serializable_history[key] = value
            else:
                serializable_history[key] = list(value) if hasattr(value, '__iter__') else value
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def _plot_training_progress(self):
        """Plot training progress"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'KUKA RL Training Progress - {self.config["task"].title()} Task')
            
            # Episode rewards
            axes[0, 0].plot(self.training_history['episodes'], self.training_history['rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
            
            # Moving average of rewards
            if len(self.training_history['rewards']) > 10:
                window = min(50, len(self.training_history['rewards']) // 4)
                moving_avg = np.convolve(self.training_history['rewards'], 
                                       np.ones(window)/window, mode='valid')
                axes[0, 1].plot(self.training_history['episodes'][:len(moving_avg)], moving_avg)
                axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Average Reward')
                axes[0, 1].grid(True)
            
            # Evaluation rewards
            if self.training_history['evaluation_rewards']:
                eval_episodes = np.arange(1, len(self.training_history['evaluation_rewards']) + 1) * \
                              self.config.get('eval_frequency', 50)
                axes[1, 0].plot(eval_episodes, self.training_history['evaluation_rewards'], 'ro-')
                axes[1, 0].set_title('Evaluation Rewards')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Mean Evaluation Reward')
                axes[1, 0].grid(True)
            
            # Success rates
            if self.training_history['evaluation_success_rates']:
                eval_episodes = np.arange(1, len(self.training_history['evaluation_success_rates']) + 1) * \
                              self.config.get('eval_frequency', 50)
                axes[1, 1].plot(eval_episodes, self.training_history['evaluation_success_rates'], 'go-')
                axes[1, 1].set_title('Success Rate')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Success Rate')
                axes[1, 1].set_ylim(0, 1)
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, "training_progress.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Could not create training plots: {e}")
    
    def _generate_training_summary(self, training_time, final_eval: Dict) -> Dict[str, Any]:
        """Generate training summary"""
        summary = {
            'config': self.config,
            'training_time': str(training_time),
            'total_episodes': len(self.training_history['episodes']),
            'final_evaluation': final_eval,
            'best_eval_reward': max(self.training_history['evaluation_rewards']) if self.training_history['evaluation_rewards'] else None,
            'best_episode_reward': max(self.training_history['rewards']) if self.training_history['rewards'] else None,
            'average_episode_reward': np.mean(self.training_history['rewards']) if self.training_history['rewards'] else None,
            'output_directory': self.output_dir
        }
        
        # Save summary
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary

def create_training_config(args) -> Dict[str, Any]:
    """Create training configuration from arguments"""
    return {
        'task': args.task,
        'algorithm': args.algorithm,
        'num_episodes': args.num_episodes,
        'eval_frequency': args.eval_frequency,
        'save_frequency': args.save_frequency,
        'device': args.device,
        'render_mode': args.render_mode,
        'early_stop_threshold': args.early_stop_threshold,
        'seed': args.seed
    }

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train KUKA RL Agent")
    
    # Task and algorithm
    parser.add_argument('--task', type=str, default='reach',
                       choices=['reach', 'grasp', 'manipulation', 'move'],
                       help='RL task type')
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['ppo', 'sac', 'ddpg'],
                       help='RL algorithm')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--eval_frequency', type=int, default=50,
                       help='Evaluation frequency (episodes)')
    parser.add_argument('--save_frequency', type=int, default=100,
                       help='Save frequency (episodes)')
    
    # System parameters
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Computing device')
    parser.add_argument('--render_mode', type=str, default=None,
                       choices=[None, 'human', 'rgb_array'],
                       help='Rendering mode')
    
    # Training control
    parser.add_argument('--early_stop_threshold', type=float, default=5.0,
                       help='Early stopping threshold')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Evaluation only
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to model for evaluation')
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed:
        np.random.seed(args.seed)
    
    # Create training configuration
    config = create_training_config(args)
    
    if args.evaluate_only:
        # Evaluation mode
        if not args.model_path:
            print("ERROR: --model_path required for evaluation mode")
            return
        
        print(f"Evaluating model: {args.model_path}")
        print(f"Task: {args.task}, Algorithm: {args.algorithm}")
        
        # Create environment and agent
        env = KUKAGymEnvironment(task_type=args.task, render_mode=args.render_mode)
        obs, _ = env.reset()
        obs_dim = len(obs)
        
        agent = KUKARLAgent(
            algorithm=args.algorithm,
            observation_dim=obs_dim,
            action_dim=7,
            device=args.device
        )
        
        # Load model
        agent.load(args.model_path)
        
        # Evaluate
        eval_results = agent.evaluate(env, num_episodes=20)
        
        print("\n=== Evaluation Results ===")
        print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"Success Rate: {eval_results['success_rate']:.2f}")
        print(f"Episodes: {eval_results['episodes']}")
        
        env.close()
        
    else:
        # Training mode
        print("=== KUKA RL Training ===")
        print(f"Task: {args.task}")
        print(f"Algorithm: {args.algorithm}")
        print(f"Episodes: {args.num_episodes}")
        print(f"Device: {args.device}")
        
        # Create training manager
        trainer = KUKATrainingManager(config)
        
        # Start training
        summary = trainer.train()
        
        print("\n=== Training Complete ===")
        print(f"Training time: {summary['training_time']}")
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"Final evaluation: {summary['final_evaluation']}")
        print(f"Best eval reward: {summary['best_eval_reward']}")
        print(f"Results saved to: {summary['output_directory']}")

if __name__ == "__main__":
    main() 