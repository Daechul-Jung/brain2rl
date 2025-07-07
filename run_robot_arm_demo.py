#!/usr/bin/env python3
"""
Robot Arm Demo Script for Brain-Controlled Manipulation
This script demonstrates the brain-controlled robot arm system.
"""

import os
import sys
import time
import subprocess
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_setup():
    """Run the ROS2 environment setup"""
    print("Setting up ROS2 environment...")
    try:
        from scripts.setup_ros2_environment import main as setup_main
        success = setup_main()
        if not success:
            print("⚠️  ROS2 setup had issues, but continuing...")
        return True
    except Exception as e:
        print(f"Setup error: {e}")
        return False

def run_robot_arm_controller_demo():
    """Run robot arm controller in demo mode"""
    print("\n🤖 Running Robot Arm Controller Demo...")
    print("This will test the arm controller without ROS2")
    
    try:
        from simulation.robot_arm_controller import main as arm_main
        arm_main()
    except KeyboardInterrupt:
        print("Demo interrupted")
    except Exception as e:
        print(f"Demo error: {e}")

def run_object_spawner_demo():
    """Run object spawner demo"""
    print("\n📦 Running Object Spawner Demo...")
    
    try:
        from simulation.spawn_manipulation_objects import main as spawner_main
        spawner_main()
    except KeyboardInterrupt:
        print("Demo interrupted")
    except Exception as e:
        print(f"Demo error: {e}")

def run_brain_arm_interface_demo():
    """Run the full brain arm interface demo"""
    print("\n🧠 Running Brain Arm Interface Demo...")
    print("This will start the complete brain-controlled arm system")
    
    try:
        # Set up command line arguments for the demo
        sys.argv = [
            'brain_arm_interface_node.py',
            '--model', 'classification',
            '--arm_model', 'ur5', 
            '--simulation_mode',
            '--spawn_objects'
        ]
        
        from simulation.brain_arm_interface_node import main as interface_main
        interface_main()
    except KeyboardInterrupt:
        print("Demo interrupted")
    except Exception as e:
        print(f"Demo error: {e}")

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("BRAIN-CONTROLLED ROBOT ARM SIMULATION")
    print("="*60)
    print("\n🎯 What this system does:")
    print("  • Processes brain signals (EEG) to control a robot arm")
    print("  • Performs grasping and manipulation tasks")
    print("  • Supports different brain models (classification, RL, tokenization)")
    print("  • Simulates realistic manipulation scenarios")
    
    print("\n🚀 Quick Start:")
    print("  1. Setup: python run_robot_arm_demo.py --setup")
    print("  2. Test arm: python run_robot_arm_demo.py --arm-demo")
    print("  3. Full demo: python run_robot_arm_demo.py --full-demo")
    
    print("\n🔧 Manual Usage:")
    print("  # Run individual components:")
    print("  python simulation/robot_arm_controller.py")
    print("  python simulation/spawn_manipulation_objects.py")
    print("  python simulation/brain_arm_interface_node.py --simulation_mode --spawn_objects")
    
    print("\n🎮 ROS2 Commands (if ROS2 is available):")
    print("  # Send task commands:")
    print("  ros2 topic pub /brain_arm/task_command std_msgs/String \"data: 'start'\"")
    print("  ros2 topic pub /brain_arm/task_command std_msgs/String \"data: 'grasp_red_cube'\"")
    print("  ros2 topic pub /brain_arm/task_command std_msgs/String \"data: 'home'\"")
    print("  ")
    print("  # Monitor status:")
    print("  ros2 topic echo /brain_arm/task_status")
    print("  ros2 topic echo /brain_arm/current_target")
    
    print("\n📝 Available Brain Models:")
    print("  • classification: 4-class brain signal classification")
    print("  • rl: Continuous reinforcement learning control")
    print("  • tokenization: Discrete position tokenization")
    
    print("\n🤖 Supported Robot Arms:")
    print("  • ur5: Universal Robots UR5 (default)")
    print("  • ur10: Universal Robots UR10")
    print("  • kuka_iiwa: KUKA LBR iiwa")
    print("  • franka_panda: Franka Emika Panda")
    
    print("\n⚠️  Troubleshooting:")
    print("  • Import errors: Run --setup first")
    print("  • ROS2 issues: Check C:\\dev\\ros2_humble installation")
    print("  • Permission errors: Run as administrator if needed")
    
    print("\n" + "="*60)

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Robot Arm Demo Script")
    parser.add_argument('--setup', action='store_true',
                       help='Set up ROS2 environment')
    parser.add_argument('--arm-demo', action='store_true',
                       help='Run robot arm controller demo')
    parser.add_argument('--objects-demo', action='store_true',
                       help='Run object spawner demo')
    parser.add_argument('--full-demo', action='store_true',
                       help='Run complete brain arm interface demo')
    parser.add_argument('--instructions', action='store_true',
                       help='Show detailed usage instructions')
    
    args = parser.parse_args()
    
    if args.setup:
        run_setup()
    elif args.arm_demo:
        run_robot_arm_controller_demo()
    elif args.objects_demo:
        run_object_spawner_demo()
    elif args.full_demo:
        run_brain_arm_interface_demo()
    elif args.instructions:
        print_usage_instructions()
    else:
        # Default: show instructions and run quick demo
        print_usage_instructions()
        
        print("\n🎮 Running Quick Demo...")
        print("Press Enter to start arm controller demo, or Ctrl+C to exit")
        try:
            input()
            run_robot_arm_controller_demo()
        except KeyboardInterrupt:
            print("\nDemo cancelled")

if __name__ == "__main__":
    main() 