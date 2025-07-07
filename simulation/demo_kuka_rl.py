#!/usr/bin/env python3
"""
KUKA RL Demo Script
Demonstrates the complete KUKA reinforcement learning system
"""

import os
import sys
import time
import argparse
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from simulation.kuka_gym_environment import KUKAGymEnvironment
from simulation.kuka_rl_agent import KUKARLAgent
from simulation.launch_kuka_rl import KUKARLLauncher

def demo_environment_only():
    """Demo: Test environment without RL training"""
    print("=== DEMO: Environment Testing ===")
    
    tasks = ['reach', 'grasp', 'move']
    
    for task in tasks:
        print(f"\n--- Testing {task.upper()} task ---")
        
        try:
            env = KUKAGymEnvironment(task_type=task)
            
            # Reset environment
            obs, info = env.reset()
            print(f"✓ Environment reset: obs_shape={obs.shape}")
            print(f"  Task info: {info}")
            
            # Take random actions
            for step in range(5):
                # Use proper gym action space sampling
                try:
                    action = env.action_space.sample()
                except:
                    # Fallback for mock environments
                    import numpy as np
                    action = np.random.uniform(-1, 1, 7)
                
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"  Step {step+1}: reward={reward:.3f}, terminated={terminated}")
                
                if terminated or truncated:
                    break
            
            env.close()
            print(f"✓ {task} task completed successfully")
            
        except Exception as e:
            print(f"✗ Error testing {task}: {e}")

def demo_agent_only():
    """Demo: Test RL agents without environment"""
    print("\n=== DEMO: Agent Testing ===")
    
    algorithms = ['ppo', 'sac']
    
    for algo in algorithms:
        print(f"\n--- Testing {algo.upper()} agent ---")
        
        try:
            agent = KUKARLAgent(
                algorithm=algo,
                observation_dim=30,  # Standard observation size
                action_dim=7,        # KUKA joints
                device='cpu'
            )
            
            # Test action generation
            import numpy as np
            obs = np.random.randn(30)
            
            # Training mode
            action_train, info_train = agent.get_action(obs, training=True)
            print(f"✓ Training action: shape={action_train.shape}")
            
            # Evaluation mode
            action_eval, info_eval = agent.get_action(obs, training=False)
            print(f"✓ Evaluation action: shape={action_eval.shape}")
            
            # Test save/load
            save_path = f"/tmp/demo_{algo}_agent.pkl"
            agent.save(save_path)
            agent.load(save_path)
            print(f"✓ Save/load successful")
            
        except Exception as e:
            print(f"✗ Error testing {algo}: {e}")

def demo_quick_training():
    """Demo: Quick training session"""
    print("\n=== DEMO: Quick Training ===")
    
    try:
        # Create environment
        env = KUKAGymEnvironment(task_type='reach')
        
        # Get dimensions
        obs, _ = env.reset()
        obs_dim = len(obs)
        
        # Create agent
        agent = KUKARLAgent(
            algorithm='ppo',
            observation_dim=obs_dim,
            action_dim=7,
            device='cpu'
        )
        
        print(f"Training KUKA arm for 5 episodes...")
        
        for episode in range(5):
            episode_info = agent.train_episode(env)
            
            print(f"Episode {episode+1}: "
                  f"Reward={episode_info['episode_reward']:.2f}, "
                  f"Steps={episode_info['episode_steps']}, "
                  f"Success={episode_info.get('success', False)}")
        
        # Quick evaluation
        eval_results = agent.evaluate(env, num_episodes=3)
        print(f"\nEvaluation: "
              f"Mean Reward={eval_results['mean_reward']:.2f}, "
              f"Success Rate={eval_results['success_rate']:.2f}")
        
        env.close()
        print("✓ Quick training completed successfully")
        
    except Exception as e:
        print(f"✗ Error in quick training: {e}")

def demo_system_components():
    """Demo: Test individual system components"""
    print("\n=== DEMO: System Components ===")
    
    try:
        # Test world manager
        print("--- Testing Gazebo World Manager ---")
        from simulation.kuka_gazebo_world import KUKAGazeboWorld
        
        world_manager = KUKAGazeboWorld("demo_world")
        print("✓ World manager created")
        
        # Test KUKA spawning (mock mode)
        success = world_manager.spawn_kuka_arm()
        if success:
            print("✓ KUKA arm spawn test passed")
        
        world_manager.shutdown()
        
        # Test controller
        print("\n--- Testing ROS2 Controller ---")
        from simulation.kuka_ros_controller import KUKARosController
        
        controller = KUKARosController("demo_controller")
        print("✓ Controller created")
        
        # Test joint limits
        limits = controller.get_joint_limits()
        print(f"✓ Joint limits: {len(limits['lower'])} joints")
        
        # Test workspace bounds
        workspace = controller.get_workspace_bounds()
        print(f"✓ Workspace: x={workspace['x']}, y={workspace['y']}, z={workspace['z']}")
        
        controller.shutdown()
        
        print("✓ All system components tested successfully")
        
    except Exception as e:
        print(f"✗ Error testing components: {e}")

def demo_different_tasks():
    """Demo: Show different RL tasks"""
    print("\n=== DEMO: Different RL Tasks ===")
    
    tasks_info = {
        'reach': "Move end-effector to target positions",
        'grasp': "Approach and grasp objects on table", 
        'manipulation': "Pick up object and move to goal",
        'move': "Navigate through waypoints"
    }
    
    for task, description in tasks_info.items():
        print(f"\n--- {task.upper()} Task ---")
        print(f"Description: {description}")
        
        try:
            env = KUKAGymEnvironment(task_type=task)
            
            # Get task configuration
            task_info = env.get_task_info()
            print(f"✓ Task type: {task_info['task_type']}")
            print(f"  Max steps: {task_info['max_steps']}")
            print(f"  Target: {task_info.get('target_position', 'Dynamic')}")
            
            # Test one episode
            obs, info = env.reset()
            action = [0.1, 0.0, -0.1, 0.2, 0.0, -0.1, 0.0]  # Small movement
            obs, reward, terminated, truncated, step_info = env.step(action)
            
            print(f"  Sample reward: {reward:.3f}")
            print(f"  Episode info: {step_info}")
            
            env.close()
            
        except Exception as e:
            print(f"✗ Error with {task}: {e}")

def demo_full_pipeline():
    """Demo: Complete system pipeline"""
    print("\n=== DEMO: Full System Pipeline ===")
    
    try:
        print("1. Creating launcher...")
        launcher = KUKARLLauncher()
        
        print("2. System requirements check completed")
        
        print("3. Testing environment setup...")
        env = KUKAGymEnvironment(task_type='reach')
        obs, _ = env.reset()
        print(f"✓ Environment ready: obs_dim={len(obs)}")
        
        print("4. Creating RL agent...")
        agent = KUKARLAgent(
            algorithm='ppo',
            observation_dim=len(obs),
            action_dim=7,
            device='cpu'
        )
        print("✓ Agent ready")
        
        print("5. Running mini training session...")
        for i in range(3):
            episode_info = agent.train_episode(env)
            print(f"   Episode {i+1}: reward={episode_info['episode_reward']:.2f}")
        
        print("6. Evaluation...")
        eval_results = agent.evaluate(env, num_episodes=2)
        print(f"   Results: {eval_results}")
        
        env.close()
        launcher.shutdown()
        
        print("✓ Full pipeline demo completed successfully")
        
    except Exception as e:
        print(f"✗ Error in full pipeline: {e}")

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="KUKA RL System Demo")
    parser.add_argument('--demo', type=str, default='all',
                       choices=['all', 'env', 'agent', 'train', 'components', 'tasks', 'pipeline'],
                       help='Demo type to run')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick version of demos')
    
    args = parser.parse_args()
    
    print("🤖 KUKA RL System Demo")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        if args.demo == 'all':
            print("Running all demos...")
            demo_environment_only()
            demo_agent_only()
            if not args.quick:
                demo_system_components()
                demo_different_tasks()
                demo_quick_training()
                demo_full_pipeline()
        
        elif args.demo == 'env':
            demo_environment_only()
        
        elif args.demo == 'agent':
            demo_agent_only()
        
        elif args.demo == 'train':
            demo_quick_training()
        
        elif args.demo == 'components':
            demo_system_components()
        
        elif args.demo == 'tasks':
            demo_different_tasks()
        
        elif args.demo == 'pipeline':
            demo_full_pipeline()
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    except Exception as e:
        print(f"\nDemo error: {e}")
    
    finally:
        elapsed = time.time() - start_time
        print(f"\n🎉 Demo completed in {elapsed:.1f} seconds")
        print("\nNext steps:")
        print("1. Train full model: python train_kuka_rl.py --task reach --num_episodes 500")
        print("2. Launch full system: python launch_kuka_rl.py --start_training")
        print("3. Read README.md for complete documentation")

if __name__ == "__main__":
    main() 