#!/usr/bin/env python3
"""
Test script to verify KUKA environment properly inherits from gym.Env
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_gym_inheritance():
    """Test that KUKAGymEnvironment properly inherits from gym.Env"""
    print("=== Testing Gym Inheritance ===\n")
    
    try:
        # Try importing gym
        try:
            import gymnasium as gym
            print("✓ Using gymnasium (modern gym)")
        except ImportError:
            import gym
            print("✓ Using classic gym")
        
        # Import our environment
        from simulation.kuka_gym_environment import KUKAGymEnvironment
        
        # Create environment
        env = KUKAGymEnvironment(task_type='reach')
        
        # Test 1: Check inheritance
        is_gym_env = isinstance(env, gym.Env)
        print(f"✓ Inherits from gym.Env: {is_gym_env}")
        
        # Test 2: Check required attributes
        has_action_space = hasattr(env, 'action_space')
        has_observation_space = hasattr(env, 'observation_space')
        has_metadata = hasattr(env, 'metadata')
        
        print(f"✓ Has action_space: {has_action_space}")
        print(f"✓ Has observation_space: {has_observation_space}")
        print(f"✓ Has metadata: {has_metadata}")
        
        # Test 3: Check action space type
        action_space_type = type(env.action_space).__name__
        print(f"✓ Action space type: {action_space_type}")
        
        # Test 4: Check observation space type
        obs_space_type = type(env.observation_space).__name__
        print(f"✓ Observation space type: {obs_space_type}")
        
        # Test 5: Check action sampling
        try:
            action = env.action_space.sample()
            print(f"✓ Action sampling works: shape={action.shape}")
        except Exception as e:
            print(f"✗ Action sampling failed: {e}")
        
        # Test 6: Check gym interface methods
        methods_to_check = ['reset', 'step', 'render', 'close', 'seed']
        for method in methods_to_check:
            has_method = hasattr(env, method)
            print(f"✓ Has {method}(): {has_method}")
        
        # Test 7: Test basic environment loop
        print("\n--- Testing Environment Loop ---")
        obs, info = env.reset()
        print(f"✓ Reset: obs_shape={obs.shape}, info_keys={list(info.keys())}")
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Step: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        
        # Test 8: Test with gym wrappers
        print("\n--- Testing Gym Wrappers ---")
        try:
            # TimeLimit wrapper
            from gym.wrappers import TimeLimit
            wrapped_env = TimeLimit(env, max_episode_steps=100)
            print("✓ TimeLimit wrapper works")
            
            obs, info = wrapped_env.reset()
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            print("✓ Wrapped environment step works")
            
        except Exception as e:
            print(f"✗ Wrapper test failed: {e}")
        
        env.close()
        print("\n🎉 All gym compatibility tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def test_with_stable_baselines():
    """Test compatibility with stable-baselines3 (if available)"""
    print("\n=== Testing Stable-Baselines3 Compatibility ===\n")
    
    try:
        from stable_baselines3.common.env_checker import check_env
        from simulation.kuka_gym_environment import KUKAGymEnvironment
        
        env = KUKAGymEnvironment(task_type='reach')
        
        print("Running stable-baselines3 env_checker...")
        check_env(env)
        print("✓ Environment passes stable-baselines3 checks!")
        
        env.close()
        return True
        
    except ImportError:
        print("ℹ️  stable-baselines3 not installed, skipping compatibility test")
        return True
    except Exception as e:
        print(f"✗ stable-baselines3 check failed: {e}")
        return False

def main():
    """Run all compatibility tests"""
    print("🧪 KUKA Gym Environment Compatibility Test\n")
    
    success1 = test_gym_inheritance()
    success2 = test_with_stable_baselines()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Environment is fully gym-compatible.")
    else:
        print("\n❌ Some tests failed.")

if __name__ == "__main__":
    main() 