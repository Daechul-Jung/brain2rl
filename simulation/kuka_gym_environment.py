#!/usr/bin/env python3
"""
KUKA Gym Environment for Reinforcement Learning
Provides OpenAI Gym interface for KUKA iiwa arm in Gazebo simulation
"""

import os
import sys
import time
import numpy as np
import threading
from typing import Dict, List, Any, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
    print("Using gymnasium (modern gym)")
except ImportError:
    try:
        import gym 
        from gym import spaces 
        GYM_AVAILABLE = True
        print("Using classic gym")
    except ImportError:
        print("ERROR: Neither gymnasium nor gym found. Please install: pip install gymnasium")
        GYM_AVAILABLE = False
        class MockEnv:
            def __init__(self): pass
        class MockSpaces:
            @staticmethod
            def Box(*args, **kwargs): return {'type': 'Box', 'args': args, 'kwargs': kwargs}
        gym = type('MockGym', (), {'Env': MockEnv})()
        spaces = MockSpaces()

try:
    from scripts.setup_ros2_environment import setup_ros2_environment
    setup_ros2_environment()
except ImportError:
    print("WARNING: Could not import ROS2 setup script")

try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from std_msgs.msg import Float64MultiArray, String, Bool  # type: ignore
    from geometry_msgs.msg import Pose, Point  # type: ignore
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 not available - using mock mode")

# Local imports
from simulation.kuka_ros_controller import KUKARosController
from simulation.kuka_gazebo_world import KUKAGazeboWorld

class KUKAGymEnvironment(gym.Env):
    """Gym Environment for KUKA iiwa arm reinforcement learning"""
    
    metadata = {
        'render_modes': ['human', 'rgb_array'], 
        'render_fps': 30
    }
    
    def __init__(self, task_type: str = "reach", render_mode: Optional[str] = None):
        """
        Initialize KUKA Gym Environment
        
        Args:
            task_type: Type of task ('reach', 'grasp', 'move', 'manipulation')
            render_mode: Rendering mode (None, 'human', 'rgb_array')
        """
        super().__init__()  # Proper parent class initialization
        
        self.task_type = task_type
        self.render_mode = render_mode
        
        # Environment configuration
        self.max_episode_steps = 200
        self.current_step = 0
        self.episode_count = 0
        
        # Initialize components
        self.world_manager = None
        self.kuka_controller = None
        self.ros_node = None
        
        # Task-specific configuration
        self._setup_task_config()
        
        # Action and observation spaces - MUST be set before calling super().__init__()
        self._setup_spaces()
        
        # Episode tracking
        self.episode_reward = 0.0
        self.episode_success = False
        self.episode_info = {}
        
        # Initialize environment
        self._initialize_environment()
        
        print(f"✓ KUKA Gym Environment initialized - Task: {task_type}")
    
    def _setup_task_config(self):
        """Setup task-specific configuration"""
        self.task_configs = {
            'reach': {
                'target_position': np.array([0.5, 0.2, 0.6]),
                'position_tolerance': 0.05,
                'success_reward': 100.0,
                'distance_reward_scale': 10.0,
                'action_penalty': 0.1
            },
            'grasp': {
                'target_position': np.array([0.6, 0.0, 0.45]),
                'grasp_tolerance': 0.03,
                'success_reward': 200.0,
                'approach_reward_scale': 15.0,
                'action_penalty': 0.15
            },
            'manipulation': {
                'object_start': np.array([0.6, 0.2, 0.45]),
                'object_goal': np.array([0.6, -0.2, 0.45]),
                'manipulation_tolerance': 0.08,
                'success_reward': 300.0,
                'manipulation_reward_scale': 20.0,
                'action_penalty': 0.2
            },
            'move': {
                'waypoints': [
                    np.array([0.4, 0.0, 0.5]),
                    np.array([0.6, 0.3, 0.7]),
                    np.array([0.5, -0.2, 0.6])
                ],
                'waypoint_tolerance': 0.06,
                'success_reward': 150.0,
                'progress_reward_scale': 12.0,
                'action_penalty': 0.12
            }
        }
        
        if self.task_type not in self.task_configs:
            raise ValueError(f"Unknown task type: {self.task_type}")
        
        self.task_config = self.task_configs[self.task_type]
        
        # Task state
        self.target_position = None
        self.current_waypoint = 0
        self.object_grasped = False
        self.task_completed = False
    
    def _setup_spaces(self):
        """Setup action and observation spaces as proper gym.Space objects"""
        # Action space: Joint position control (7 joints)
        joint_limits_lower = np.array([-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05])
        joint_limits_upper = np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])
        
        if GYM_AVAILABLE:
            self.action_space = spaces.Box(
                low=joint_limits_lower,
                high=joint_limits_upper,
                dtype=np.float32
            )
            
            # Observation space: Joint positions + velocities + end-effector pose + task info
            obs_dim = 7 + 7 + 6 + 10  # joints + velocities + ee_pose + task_features
            
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
        else:
            # Fallback for development without gym
            self.action_space = {
                'low': joint_limits_lower,
                'high': joint_limits_upper,
                'shape': (7,),
                'dtype': np.float32
            }
            self.observation_space = {
                'low': -np.inf,
                'high': np.inf,
                'shape': (30,),  # obs_dim
                'dtype': np.float32
            }
    
    def _initialize_environment(self):
        """Initialize ROS components and Gazebo world"""
        try:
            if ROS2_AVAILABLE:
                # Initialize ROS2 if not already done
                if not rclpy.ok():
                    rclpy.init()
                
                # Create world manager
                self.world_manager = KUKAGazeboWorld("kuka_env_world")
                
                # Create KUKA controller 
                self.kuka_controller = KUKARosController("kuka_env_controller")
                
                # Wait for initialization
                time.sleep(2.0)
                
                # Setup environment
                self.world_manager.setup_rl_environment()
                
                print("✓ ROS2 components initialized")
            else:
                # Mock initialization
                self.world_manager = KUKAGazeboWorld("mock_world")
                self.kuka_controller = KUKARosController("mock_controller")
                print("✓ Mock components initialized")
                
        except Exception as e:
            print(f"WARNING: Could not initialize ROS components: {e}")
            print("Using mock environment")
            self.world_manager = KUKAGazeboWorld("mock_world")
            self.kuka_controller = KUKARosController("mock_controller")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_success = False
        self.task_completed = False
        self.object_grasped = False
        self.current_waypoint = 0
        
        # Reset KUKA arm to home position
        if self.kuka_controller:
            self.kuka_controller.reset_to_home()
            time.sleep(2.0)
        
        # Reset world if needed
        if self.world_manager and hasattr(self.world_manager, 'reset_environment'):
            self.world_manager.reset_environment()
        
        # Setup task-specific targets
        self._setup_episode_targets()
        
        # Get initial observation
        observation = self._get_observation()
        
        # Episode info
        info = {
            'episode': self.episode_count,
            'task_type': self.task_type,
            'target_position': self.target_position.copy() if self.target_position is not None else None
        }
        
        self.episode_count += 1
        
        print(f"Episode {self.episode_count} started - Task: {self.task_type}")
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action and return next state"""
        self.current_step += 1
        
        # Ensure action is numpy array
        action = np.array(action, dtype=np.float32)
        
        # Apply action to KUKA arm
        if self.kuka_controller:
            success = self.kuka_controller.set_joint_positions(action, duration=0.1)
            if not success:
                print(f"WARNING: Failed to apply action: {action}")
        
        # Small delay for physics simulation
        time.sleep(0.05)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(action, observation)
        self.episode_reward += reward
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_episode_steps
        
        # Episode info
        info = {
            'step': self.current_step,
            'episode_reward': self.episode_reward,
            'task_completed': self.task_completed,
            'episode_success': self.episode_success,
            'current_waypoint': self.current_waypoint,
            'object_grasped': self.object_grasped
        }
        
        if terminated or truncated:
            info['episode_length'] = self.current_step
            info['final_reward'] = self.episode_reward
            print(f"Episode finished - Steps: {self.current_step}, Reward: {self.episode_reward:.2f}, Success: {self.episode_success}")
        
        return observation, reward, terminated, truncated, info
    
    def _setup_episode_targets(self):
        """Setup targets for current episode"""
        if self.task_type == 'reach':
            # Random target within workspace
            self.target_position = self._random_target_position()
            
        elif self.task_type == 'grasp':
            # Target at object position
            self.target_position = self.task_config['target_position'].copy()
            # Add small random offset
            self.target_position += np.random.uniform(-0.05, 0.05, 3)
            
        elif self.task_type == 'manipulation':
            # Start with object at start position
            self.target_position = self.task_config['object_start'].copy()
            self.object_grasped = False
            
        elif self.task_type == 'move':
            # Start with first waypoint
            self.current_waypoint = 0
            self.target_position = self.task_config['waypoints'][0].copy()
    
    def _random_target_position(self) -> np.ndarray:
        """Generate random target position within workspace"""
        workspace = self.kuka_controller.get_workspace_bounds() if self.kuka_controller else {
            'x': (-0.6, 0.6), 'y': (-0.6, 0.6), 'z': (0.3, 1.0)
        }
        
        x = np.random.uniform(workspace['x'][0] + 0.1, workspace['x'][1] - 0.1)
        y = np.random.uniform(workspace['y'][0] + 0.1, workspace['y'][1] - 0.1)
        z = np.random.uniform(workspace['z'][0] + 0.1, workspace['z'][1] - 0.1)
        
        return np.array([x, y, z])
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.kuka_controller:
            # Joint positions and velocities
            joint_positions = self.kuka_controller.get_joint_positions()
            joint_velocities = self.kuka_controller.get_joint_velocities()
            
            # End-effector pose
            ee_pose = self.kuka_controller.get_end_effector_pose()
            if ee_pose:
                ee_position = np.array([ee_pose.position.x, ee_pose.position.y, ee_pose.position.z])
                ee_orientation = np.array([ee_pose.orientation.x, ee_pose.orientation.y, ee_pose.orientation.z])
            else:
                ee_position = np.zeros(3)
                ee_orientation = np.zeros(3)
        else:
            # Mock observations
            joint_positions = np.random.uniform(-1, 1, 7)
            joint_velocities = np.random.uniform(-0.1, 0.1, 7)
            ee_position = np.random.uniform([0.3, -0.3, 0.5], [0.7, 0.3, 0.9])
            ee_orientation = np.zeros(3)
        
        # Task-specific features
        if self.target_position is not None:
            target_relative = self.target_position - ee_position
            distance_to_target = np.linalg.norm(target_relative)
            target_direction = target_relative / (distance_to_target + 1e-8)
        else:
            target_relative = np.zeros(3)
            distance_to_target = 0.0
            target_direction = np.zeros(3)
        
        # Task features
        task_features = np.array([
            distance_to_target,
            float(self.object_grasped),
            float(self.task_completed),
            float(self.current_waypoint) / max(1, len(self.task_config.get('waypoints', [1]))),
            self.current_step / self.max_episode_steps,
            float(self.task_type == 'reach'),
            float(self.task_type == 'grasp'),
            float(self.task_type == 'manipulation'),
            float(self.task_type == 'move'),
            0.0  # Reserved for future use
        ])
        
        # Combine all observations
        observation = np.concatenate([
            joint_positions,
            joint_velocities,
            ee_position,
            ee_orientation,
            task_features
        ]).astype(np.float32)
        
        return observation
    
    def _calculate_reward(self, action: np.ndarray, observation: np.ndarray) -> float:
        """Calculate reward for current step"""
        reward = 0.0
        
        # Extract components from observation
        ee_position = observation[14:17]  # End-effector position
        distance_to_target = observation[20]  # Distance to target
        
        # Action penalty (encourage smooth movements)
        action_penalty = np.linalg.norm(action) * self.task_config['action_penalty']
        reward -= action_penalty
        
        if self.task_type == 'reach':
            # Distance-based reward
            distance_reward = -distance_to_target * self.task_config['distance_reward_scale']
            reward += distance_reward
            
            # Success bonus
            if distance_to_target < self.task_config['position_tolerance']:
                reward += self.task_config['success_reward']
                self.task_completed = True
                self.episode_success = True
        
        elif self.task_type == 'grasp':
            # Approach reward
            approach_reward = -distance_to_target * self.task_config['approach_reward_scale']
            reward += approach_reward
            
            # Grasp success
            if distance_to_target < self.task_config['grasp_tolerance']:
                if not self.object_grasped:
                    reward += self.task_config['success_reward']
                    self.object_grasped = True
                    self.task_completed = True
                    self.episode_success = True
        
        elif self.task_type == 'manipulation':
            if not self.object_grasped:
                # Approach object
                approach_reward = -distance_to_target * self.task_config['manipulation_reward_scale']
                reward += approach_reward
                
                if distance_to_target < self.task_config['manipulation_tolerance']:
                    self.object_grasped = True
                    reward += self.task_config['success_reward'] * 0.5
                    # Update target to goal position
                    self.target_position = self.task_config['object_goal'].copy()
            else:
                # Move object to goal
                goal_distance = np.linalg.norm(ee_position - self.task_config['object_goal'])
                goal_reward = -goal_distance * self.task_config['manipulation_reward_scale']
                reward += goal_reward
                
                if goal_distance < self.task_config['manipulation_tolerance']:
                    reward += self.task_config['success_reward'] * 0.5
                    self.task_completed = True
                    self.episode_success = True
        
        elif self.task_type == 'move':
            # Waypoint navigation
            waypoint_reward = -distance_to_target * self.task_config['progress_reward_scale']
            reward += waypoint_reward
            
            if distance_to_target < self.task_config['waypoint_tolerance']:
                reward += self.task_config['success_reward'] * 0.3
                self.current_waypoint += 1
                
                if self.current_waypoint < len(self.task_config['waypoints']):
                    # Move to next waypoint
                    self.target_position = self.task_config['waypoints'][self.current_waypoint].copy()
                else:
                    # All waypoints reached
                    self.task_completed = True
                    self.episode_success = True
                    reward += self.task_config['success_reward'] * 0.7
        
        # Boundary penalty
        workspace = self.kuka_controller.get_workspace_bounds() if self.kuka_controller else {
            'x': (-0.8, 0.8), 'y': (-0.8, 0.8), 'z': (0.0, 1.2)
        }
        
        if (ee_position[0] < workspace['x'][0] or ee_position[0] > workspace['x'][1] or
            ee_position[1] < workspace['y'][0] or ee_position[1] > workspace['y'][1] or
            ee_position[2] < workspace['z'][0] or ee_position[2] > workspace['z'][1]):
            reward -= 50.0  # Strong penalty for leaving workspace
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        return self.task_completed
    
    def render(self):
        """Render environment (Gazebo handles visualization)"""
        if self.render_mode == 'human':
            # Gazebo GUI should be running for visualization
            # In a real implementation, this might open/focus Gazebo window
            pass
        elif self.render_mode == 'rgb_array':
            # Would need to capture Gazebo camera feed
            # For now, return a dummy image
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        return None
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed for reproducibility (gym.Env compatibility)"""
        if seed is not None:
            np.random.seed(seed)
            return [seed]
        return []
    
    def close(self):
        """Clean shutdown"""
        print("Closing KUKA Gym Environment...")
        
        if self.kuka_controller:
            self.kuka_controller.shutdown()
        
        if self.world_manager:
            self.world_manager.shutdown()
        
        if ROS2_AVAILABLE and rclpy.ok():
            rclpy.shutdown()
        
        print("KUKA Gym Environment closed")
    
    def get_task_info(self) -> Dict:
        """Get current task information"""
        return {
            'task_type': self.task_type,
            'target_position': self.target_position.copy() if self.target_position is not None else None,
            'current_step': self.current_step,
            'max_steps': self.max_episode_steps,
            'episode_reward': self.episode_reward,
            'task_completed': self.task_completed,
            'episode_success': self.episode_success,
            'current_waypoint': self.current_waypoint,
            'object_grasped': self.object_grasped
        }

def main():
    """Test KUKA Gym Environment"""
    if not GYM_AVAILABLE:
        print("ERROR: Gym not available. Please install: pip install gymnasium")
        return
    
    # Test different tasks
    tasks = ['reach', 'grasp', 'move', 'manipulation']
    
    for task in tasks:
        print(f"\n=== Testing {task.upper()} Task ===")
        
        env = KUKAGymEnvironment(task_type=task)
        
        try:
            # Check gym compatibility
            print(f"Action space: {env.action_space}")
            print(f"Observation space: {env.observation_space}")
            print(f"Metadata: {env.metadata}")
            
            # Run a few episodes
            for episode in range(2):
                obs, info = env.reset()
                print(f"Episode {episode + 1} - Initial obs shape: {obs.shape}")
                
                for step in range(10):
                    # Use proper gym action sampling
                    if GYM_AVAILABLE and hasattr(env.action_space, 'sample'):
                        action = env.action_space.sample()
                    else:
                        # Fallback action
                        action = np.random.uniform(-1, 1, 7)
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    print(f"  Step {step}: reward={reward:.3f}, terminated={terminated}")
                    
                    if terminated or truncated:
                        break
                
                print(f"Episode finished: {info}")
        
        except Exception as e:
            print(f"Error testing {task}: {e}")
        
        finally:
            env.close()
            time.sleep(1)
    
    print("\n Gym compatibility test completed!")

if __name__ == "__main__":
    main() 