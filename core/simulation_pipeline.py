"""
Simulation Pipeline
==================

This module handles the simulation of KUKA robot arm with trained RL agents
in Gazebo environment, providing visualization and real-time control.

Author: Brain2RL Team
Version: 1.0.0
"""

import os
import sys
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
import threading
import queue
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from brain2rl.simulation.kuka_gym_environment import KUKAGymEnvironment
from brain2rl.simulation.kuka_ros_controller import KUKARosController
from brain2rl.simulation.kuka_gazebo_world import KUKAGazeboWorld


class SimulationMonitor:
    """
    Monitor for tracking simulation metrics and performance
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Metrics storage
        self.joint_positions = deque(maxlen=max_history)
        self.joint_velocities = deque(maxlen=max_history)
        self.joint_torques = deque(maxlen=max_history)
        self.end_effector_poses = deque(maxlen=max_history)
        self.rewards = deque(maxlen=max_history)
        self.actions = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        
        # Performance metrics
        self.success_count = 0
        self.failure_count = 0
        self.total_episodes = 0
        self.total_steps = 0
        
        # Real-time metrics
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        
        self.logger = logging.getLogger('Brain2RL.SimulationMonitor')
    
    def update(self, state: Dict[str, Any], action: np.ndarray, reward: float, info: Dict[str, Any]):
        """Update monitor with new simulation data"""
        timestamp = time.time()
        
        # Store data
        self.joint_positions.append(state.get('joint_positions', []))
        self.joint_velocities.append(state.get('joint_velocities', []))
        self.joint_torques.append(state.get('joint_torques', []))
        self.end_effector_poses.append(state.get('end_effector_pose', []))
        self.rewards.append(reward)
        self.actions.append(action)
        self.timestamps.append(timestamp)
        
        # Update episode metrics
        self.current_episode_reward += reward
        self.current_episode_steps += 1
        self.total_steps += 1
        
        # Check for episode end
        if info.get('episode_done', False):
            if info.get('success', False):
                self.success_count += 1
            else:
                self.failure_count += 1
            
            self.total_episodes += 1
            self.current_episode_reward = 0
            self.current_episode_steps = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current simulation metrics"""
        success_rate = self.success_count / max(self.total_episodes, 1)
        
        metrics = {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'success_rate': success_rate,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'current_episode_reward': self.current_episode_reward,
            'current_episode_steps': self.current_episode_steps,
            'average_reward': np.mean(list(self.rewards)) if self.rewards else 0,
            'average_episode_length': self.total_steps / max(self.total_episodes, 1)
        }
        
        return metrics
    
    def reset_episode(self):
        """Reset episode-specific metrics"""
        self.current_episode_reward = 0
        self.current_episode_steps = 0


class RealTimeVisualizer:
    """
    Real-time visualization of simulation data
    """
    
    def __init__(self, monitor: SimulationMonitor):
        self.monitor = monitor
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle('KUKA Robot Simulation Monitor')
        
        # Initialize plots
        self.joint_lines = []
        self.reward_line = None
        self.ee_trajectory = None
        
        self.setup_plots()
        
        # Animation
        self.animation = FuncAnimation(
            self.fig, self.update_plots, interval=100, blit=False
        )
        
        self.logger = logging.getLogger('Brain2RL.RealTimeVisualizer')
    
    def setup_plots(self):
        """Setup the visualization plots"""
        # Joint positions
        self.axes[0, 0].set_title('Joint Positions')
        self.axes[0, 0].set_xlabel('Time Steps')
        self.axes[0, 0].set_ylabel('Position (rad)')
        self.axes[0, 0].grid(True)
        
        # Joint velocities
        self.axes[0, 1].set_title('Joint Velocities')
        self.axes[0, 1].set_xlabel('Time Steps')
        self.axes[0, 1].set_ylabel('Velocity (rad/s)')
        self.axes[0, 1].grid(True)
        
        # Rewards
        self.axes[0, 2].set_title('Rewards')
        self.axes[0, 2].set_xlabel('Time Steps')
        self.axes[0, 2].set_ylabel('Reward')
        self.axes[0, 2].grid(True)
        
        # End effector trajectory (3D)
        self.axes[1, 0].set_title('End Effector Trajectory')
        self.axes[1, 0].set_xlabel('X (m)')
        self.axes[1, 0].set_ylabel('Y (m)')
        self.axes[1, 0].grid(True)
        
        # Actions
        self.axes[1, 1].set_title('Actions')
        self.axes[1, 1].set_xlabel('Time Steps')
        self.axes[1, 1].set_ylabel('Action Value')
        self.axes[1, 1].grid(True)
        
        # Performance metrics
        self.axes[1, 2].set_title('Performance Metrics')
        self.axes[1, 2].axis('off')
    
    def update_plots(self, frame):
        """Update all plots with current data"""
        try:
            # Clear previous plots
            for ax in self.axes.flat:
                if ax.get_title() != 'Performance Metrics':
                    ax.clear()
            
            self.setup_plots()
            
            if len(self.monitor.joint_positions) == 0:
                return
            
            # Convert data to numpy arrays
            joint_positions = np.array(list(self.monitor.joint_positions))
            joint_velocities = np.array(list(self.monitor.joint_velocities))
            rewards = np.array(list(self.monitor.rewards))
            actions = np.array(list(self.monitor.actions))
            ee_poses = np.array(list(self.monitor.end_effector_poses))
            
            # Plot joint positions
            if joint_positions.shape[0] > 0 and joint_positions.shape[1] > 0:
                for i in range(min(7, joint_positions.shape[1])):  # KUKA has 7 joints
                    self.axes[0, 0].plot(joint_positions[:, i], label=f'Joint {i+1}')
                self.axes[0, 0].legend()
            
            # Plot joint velocities
            if joint_velocities.shape[0] > 0 and joint_velocities.shape[1] > 0:
                for i in range(min(7, joint_velocities.shape[1])):
                    self.axes[0, 1].plot(joint_velocities[:, i], label=f'Joint {i+1}')
                self.axes[0, 1].legend()
            
            # Plot rewards
            if len(rewards) > 0:
                self.axes[0, 2].plot(rewards, 'b-', linewidth=2)
                
                # Add moving average
                if len(rewards) > 10:
                    window = min(50, len(rewards))
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    self.axes[0, 2].plot(range(window-1, len(rewards)), moving_avg, 'r-', linewidth=2, label='Moving Average')
                    self.axes[0, 2].legend()
            
            # Plot end effector trajectory
            if ee_poses.shape[0] > 0 and ee_poses.shape[1] >= 3:
                self.axes[1, 0].plot(ee_poses[:, 0], ee_poses[:, 1], 'b-', linewidth=2)
                self.axes[1, 0].scatter(ee_poses[-1, 0], ee_poses[-1, 1], c='red', s=50, label='Current Position')
                self.axes[1, 0].legend()
            
            # Plot actions
            if actions.shape[0] > 0 and actions.shape[1] > 0:
                for i in range(min(7, actions.shape[1])):
                    self.axes[1, 1].plot(actions[:, i], label=f'Action {i+1}')
                self.axes[1, 1].legend()
            
            # Display performance metrics
            metrics = self.monitor.get_metrics()
            metrics_text = f"""
            Episodes: {metrics['total_episodes']}
            Steps: {metrics['total_steps']}
            Success Rate: {metrics['success_rate']:.3f}
            Avg Reward: {metrics['average_reward']:.3f}
            Current Episode:
              Reward: {metrics['current_episode_reward']:.3f}
              Steps: {metrics['current_episode_steps']}
            """
            
            self.axes[1, 2].text(0.1, 0.9, metrics_text, transform=self.axes[1, 2].transAxes,
                               verticalalignment='top', fontsize=10, family='monospace')
            
        except Exception as e:
            self.logger.error(f"Error updating plots: {e}")
    
    def show(self):
        """Show the visualization"""
        plt.show()
    
    def save_plots(self, save_path: str):
        """Save current plots to file"""
        self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Plots saved to {save_path}")


class SimulationPipeline:
    """
    Pipeline for running KUKA robot simulation with trained RL agents
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the simulation pipeline
        
        Args:
            config: Configuration dictionary for simulation
        """
        self.config = config
        self.logger = logging.getLogger('Brain2RL.Simulation')
        
        # Simulation components
        self.gazebo_world = None
        self.ros_controller = None
        self.gym_environment = None
        self.trained_agent = None
        
        # Monitoring and visualization
        self.monitor = SimulationMonitor(max_history=self.config.get('max_history', 1000))
        self.visualizer = None
        
        # Simulation state
        self.is_running = False
        self.simulation_thread = None
        self.control_frequency = self.config.get('control_frequency', 100)
        
        # Token data for guided control
        self.token_data = None
        
        self.logger.info("Simulation pipeline initialized")
    
    def initialize_simulation_environment(self):
        """Initialize the simulation environment components"""
        try:
            # Initialize Gazebo world
            if self.config.get('use_gazebo', True):
                self.gazebo_world = KUKAGazeboWorld(
                    use_gui=self.config.get('use_gui', True),
                    real_time_factor=self.config.get('real_time_factor', 1.0)
                )
                self.gazebo_world.launch()
                time.sleep(2)  # Allow Gazebo to initialize
            
            # Initialize ROS controller
            if self.config.get('use_ros', True):
                self.ros_controller = KUKARosController(
                    mock_mode=self.config.get('mock_mode', False)
                )
                self.ros_controller.initialize()
                time.sleep(1)  # Allow ROS to initialize
            
            # Initialize gym environment
            self.gym_environment = KUKAGymEnvironment(
                task=self.config.get('task', 'reach'),
                render_mode=self.config.get('render_mode', 'rgb_array'),
                control_frequency=self.control_frequency,
                max_episode_steps=self.config.get('max_episode_steps', 1000)
            )
            
            self.logger.info("Simulation environment initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize simulation environment: {e}")
            raise
    
    def load_trained_agent(self, agent_model: Any, token_data: Optional[Dict[str, np.ndarray]] = None):
        """
        Load trained RL agent for simulation
        
        Args:
            agent_model: Trained RL agent model
            token_data: Optional token data for guided control
        """
        self.trained_agent = agent_model
        self.token_data = token_data
        
        # Set agent to evaluation mode
        if hasattr(self.trained_agent, 'eval'):
            self.trained_agent.eval()
        
        self.logger.info("Trained agent loaded for simulation")
    
    def get_token_sequence(self, episode_step: int, sequence_length: int = 10) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get token sequence for current simulation step
        
        Args:
            episode_step: Current step in episode
            sequence_length: Length of token sequence
            
        Returns:
            Tuple of (tokens, queries, keys, values) or None if no token data
        """
        if self.token_data is None:
            return None
        
        total_tokens = len(self.token_data['tokens'])
        
        start_idx = (episode_step * sequence_length) % total_tokens
        end_idx = min(start_idx + sequence_length, total_tokens)
        
        # Handle wrap-around
        if end_idx - start_idx < sequence_length:
            remaining = sequence_length - (end_idx - start_idx)
            tokens = np.concatenate([
                self.token_data['tokens'][start_idx:end_idx],
                self.token_data['tokens'][:remaining]
            ])
            queries = np.concatenate([
                self.token_data['queries'][start_idx:end_idx],
                self.token_data['queries'][:remaining]
            ])
            keys = np.concatenate([
                self.token_data['keys'][start_idx:end_idx],
                self.token_data['keys'][:remaining]
            ])
            values = np.concatenate([
                self.token_data['values'][start_idx:end_idx],
                self.token_data['values'][:remaining]
            ])
        else:
            tokens = self.token_data['tokens'][start_idx:end_idx]
            queries = self.token_data['queries'][start_idx:end_idx]
            keys = self.token_data['keys'][start_idx:end_idx]
            values = self.token_data['values'][start_idx:end_idx]
        
        return tokens, queries, keys, values
    
    def run_simulation(self, trained_model: Any, num_episodes: int = 10, 
                      visualize: bool = True) -> Dict[str, Any]:
        """
        Run the simulation with trained model
        
        Args:
            trained_model: Trained RL model
            num_episodes: Number of episodes to simulate
            visualize: Whether to show real-time visualization
            
        Returns:
            Dictionary with simulation results
        """
        # Initialize simulation environment
        if self.gym_environment is None:
            self.initialize_simulation_environment()
        
        # Load trained agent
        self.load_trained_agent(trained_model, self.token_data)
        
        # Setup visualization
        if visualize:
            self.visualizer = RealTimeVisualizer(self.monitor)
            # Start visualization in separate thread
            viz_thread = threading.Thread(target=self.visualizer.show)
            viz_thread.daemon = True
            viz_thread.start()
        
        # Run simulation episodes
        self.is_running = True
        episode_results = []
        
        try:
            for episode in range(num_episodes):
                self.logger.info(f"Starting simulation episode {episode + 1}/{num_episodes}")
                
                # Reset environment
                state, info = self.gym_environment.reset()
                self.monitor.reset_episode()
                
                episode_reward = 0
                episode_steps = 0
                done = False
                
                while not done and self.is_running:
                    # Get action from trained agent
                    action = self.select_action(state, episode_steps)
                    
                    # Execute action
                    next_state, reward, terminated, truncated, info = self.gym_environment.step(action)
                    done = terminated or truncated
                    
                    # Update monitoring
                    state_dict = {
                        'joint_positions': state[:7] if len(state) >= 7 else state,
                        'joint_velocities': state[7:14] if len(state) >= 14 else np.zeros(7),
                        'end_effector_pose': state[14:17] if len(state) >= 17 else np.zeros(3)
                    }
                    
                    info['episode_done'] = done
                    self.monitor.update(state_dict, action, reward, info)
                    
                    # Update state
                    state = next_state
                    episode_reward += reward
                    episode_steps += 1
                    
                    # Control simulation frequency
                    time.sleep(1.0 / self.control_frequency)
                
                # Store episode results
                episode_results.append({
                    'episode': episode,
                    'reward': episode_reward,
                    'steps': episode_steps,
                    'success': info.get('success', False)
                })
                
                self.logger.info(f"Episode {episode + 1} completed: Reward={episode_reward:.2f}, Steps={episode_steps}")
        
        except KeyboardInterrupt:
            self.logger.info("Simulation interrupted by user")
        except Exception as e:
            self.logger.error(f"Simulation error: {e}")
        finally:
            self.is_running = False
        
        # Compile results
        results = {
            'episode_results': episode_results,
            'total_episodes': len(episode_results),
            'average_reward': np.mean([r['reward'] for r in episode_results]),
            'success_rate': np.mean([r['success'] for r in episode_results]),
            'simulation_metrics': self.monitor.get_metrics()
        }
        
        self.logger.info(f"Simulation completed: {results['total_episodes']} episodes, "
                        f"Average reward: {results['average_reward']:.2f}, "
                        f"Success rate: {results['success_rate']:.2f}")
        
        return results
    
    def select_action(self, state: np.ndarray, episode_step: int) -> np.ndarray:
        """
        Select action using trained agent
        
        Args:
            state: Current state
            episode_step: Current step in episode
            
        Returns:
            Selected action
        """
        if self.trained_agent is None:
            # Random action if no agent loaded
            return np.random.uniform(-1, 1, size=7)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get token sequence if available
        token_sequence = self.get_token_sequence(episode_step)
        
        with torch.no_grad():
            if token_sequence is not None and hasattr(self.trained_agent, 'forward'):
                # Token-guided agent
                tokens, queries, keys, values = token_sequence
                tokens_tensor = torch.FloatTensor(tokens).unsqueeze(0)
                queries_tensor = torch.FloatTensor(queries).unsqueeze(0)
                keys_tensor = torch.FloatTensor(keys).unsqueeze(0)
                values_tensor = torch.FloatTensor(values).unsqueeze(0)
                
                action_logits, _ = self.trained_agent.forward(
                    state_tensor, tokens_tensor, queries_tensor, keys_tensor, values_tensor
                )
                action = action_logits.squeeze(0).cpu().numpy()
            else:
                # Standard RL agent
                if hasattr(self.trained_agent, 'select_action'):
                    action = self.trained_agent.select_action(state)
                else:
                    # Generic action selection
                    action = np.random.uniform(-1, 1, size=7)
        
        # Clip action to valid range
        action = np.clip(action, -1, 1)
        
        return action
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        self.logger.info("Simulation stop requested")
    
    def save_simulation_data(self, save_path: str):
        """Save simulation data to file"""
        data = {
            'joint_positions': list(self.monitor.joint_positions),
            'joint_velocities': list(self.monitor.joint_velocities),
            'rewards': list(self.monitor.rewards),
            'actions': list(self.monitor.actions),
            'timestamps': list(self.monitor.timestamps),
            'metrics': self.monitor.get_metrics()
        }
        
        np.savez(save_path, **data)
        self.logger.info(f"Simulation data saved to {save_path}")
    
    def cleanup(self):
        """Cleanup simulation resources"""
        try:
            self.is_running = False
            
            if self.gym_environment:
                self.gym_environment.close()
            
            if self.ros_controller:
                self.ros_controller.shutdown()
            
            if self.gazebo_world:
                self.gazebo_world.shutdown()
            
            self.logger.info("Simulation cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def main():
    """Main function for standalone simulation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulation Pipeline')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--token-data', type=str, help='Path to token data')
    parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to simulate')
    parser.add_argument('--task', type=str, default='reach', help='Task type')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--save-data', type=str, help='Path to save simulation data')
    
    args = parser.parse_args()
    
    # Create simulation config
    config = {
        'robot_type': 'kuka_iiwa',
        'task': args.task,
        'use_gui': True,
        'use_gazebo': True,
        'use_ros': True,
        'mock_mode': False,
        'real_time_factor': 1.0,
        'max_episode_steps': 1000,
        'control_frequency': 100,
        'max_history': 1000
    }
    
    # Initialize pipeline
    pipeline = SimulationPipeline(config)
    
    # Load trained model
    try:
        trained_model = torch.load(args.model_path, map_location='cpu')
        if isinstance(trained_model, dict) and 'token_guided_agent' in trained_model:
            model = trained_model['token_guided_agent']
        else:
            model = trained_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load token data if provided
    token_data = None
    if args.token_data:
        try:
            token_file = np.load(args.token_data)
            token_data = {
                'tokens': token_file['tokens'],
                'queries': token_file['queries'],
                'keys': token_file['keys'],
                'values': token_file['values']
            }
            pipeline.token_data = token_data
        except Exception as e:
            print(f"Error loading token data: {e}")
    
    try:
        # Run simulation
        results = pipeline.run_simulation(
            trained_model=model,
            num_episodes=args.episodes,
            visualize=args.visualize
        )
        
        # Save results
        if args.save_data:
            pipeline.save_simulation_data(args.save_data)
        
        print(f"Simulation completed:")
        print(f"  Episodes: {results['total_episodes']}")
        print(f"  Average reward: {results['average_reward']:.2f}")
        print(f"  Success rate: {results['success_rate']:.2f}")
        
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        pipeline.cleanup()


if __name__ == '__main__':
    main() 