"""
ROS2 Launch File for KUKA RL System
Launches Gazebo Classic with KUKA iiwa arm and RL training components
"""

import os
import sys
import time
import subprocess
import argparse
from typing import List, Dict, Any
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.setup_ros2_environment import setup_ros2_environment
    setup_ros2_environment()
except ImportError:
    print("WARNING: Could not import ROS2 setup script")

try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from launch import LaunchDescription  # type: ignore
    from launch.actions import ExecuteProcess, DeclareLaunchArgument, LogInfo  # type: ignore
    from launch.substitutions import LaunchConfiguration  # type: ignore
    from launch_ros.actions import Node as LaunchNode  # type: ignore
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 launch system not available - using direct execution")

class KUKARLLauncher:
    """Launcher for KUKA RL system components"""
    
    def __init__(self):
        self.processes = []
        self.gazebo_process = None
        self.world_manager = None
        self.controller = None
        
        # Check system requirements
        self._check_requirements()
    
    def _check_requirements(self):
        """Check if required components are available"""
        print("Checking KUKA RL system requirements...")
        
        # Check ROS2
        if ROS2_AVAILABLE:
            print("✓ ROS2 available")
        else:
            print("✗ ROS2 not available - using mock mode")
        
        # Check Gazebo
        try:
            result = subprocess.run(['gazebo', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✓ Gazebo Classic available: {result.stdout.strip()}")
            else:
                print("✗ Gazebo Classic not found")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("✗ Gazebo Classic not found or not responding")
        
        # Check Python packages
        packages = ['torch', 'numpy', 'matplotlib']
        for package in packages:
            try:
                __import__(package)
                print(f"{package} available")
            except ImportError:
                print(f"{package} not available")
    
    def launch_gazebo(self, gui: bool = True, world_file: str = None) -> bool:
        """Launch Gazebo Classic simulation"""
        print("Launching Gazebo Classic...")
        
        # Gazebo command
        gazebo_cmd = ['gazebo']
        
        if not gui:
            gazebo_cmd.append('--headless')
        
        if world_file and os.path.exists(world_file):
            gazebo_cmd.append(world_file)
        else:
            # Use default empty world
            gazebo_cmd.extend(['--verbose', '-s', 'libgazebo_ros_factory.so'])
        
        try:
            self.gazebo_process = subprocess.Popen(
                gazebo_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for Gazebo to start
            time.sleep(5.0)
            
            if self.gazebo_process.poll() is None:
                print("✓ Gazebo Classic launched successfully")
                return True
            else:
                print("✗ Gazebo Classic failed to start")
                return False
                
        except FileNotFoundError:
            print("✗ Gazebo Classic not found. Please install Gazebo.")
            return False
        except Exception as e:
            print(f"✗ Error launching Gazebo: {e}")
            return False
    
    def launch_world_manager(self) -> bool:
        """Launch KUKA world manager"""
        print("Launching KUKA world manager...")
        
        try:
            from simulation.kuka_gazebo_world import KUKAGazeboWorld
            
            if ROS2_AVAILABLE:
                # Initialize ROS2 if not already done
                if not rclpy.ok():
                    rclpy.init()
            
            self.world_manager = KUKAGazeboWorld("kuka_rl_world")
            
            # Setup RL environment
            success = self.world_manager.setup_rl_environment()
            
            if success:
                print("✓ KUKA world manager launched successfully")
                return True
            else:
                print("✗ Failed to setup KUKA RL environment")
                return False
                
        except Exception as e:
            print(f"✗ Error launching world manager: {e}")
            return False
    
    def launch_controller(self) -> bool:
        """Launch KUKA ROS controller"""
        print("Launching KUKA ROS controller...")
        
        try:
            from simulation.kuka_ros_controller import KUKARosController
            
            self.controller = KUKARosController("kuka_rl_controller")
            
            # Wait for controller to be ready
            time.sleep(2.0)
            
            if self.controller.controller_ready:
                print("✓ KUKA ROS controller launched successfully")
                return True
            else:
                print("✗ KUKA ROS controller not ready")
                return False
                
        except Exception as e:
            print(f"✗ Error launching controller: {e}")
            return False
    
    def launch_training(self, task: str, algorithm: str, **kwargs) -> bool:
        """Launch RL training"""
        print(f"Launching RL training: {task} task with {algorithm} algorithm...")
        
        try:
            from simulation.train_kuka_rl import KUKATrainingManager
            
            # Create training configuration
            config = {
                'task': task,
                'algorithm': algorithm,
                'num_episodes': kwargs.get('num_episodes', 1000),
                'eval_frequency': kwargs.get('eval_frequency', 50),
                'save_frequency': kwargs.get('save_frequency', 100),
                'device': kwargs.get('device', 'auto'),
                'render_mode': kwargs.get('render_mode', None),
                'early_stop_threshold': kwargs.get('early_stop_threshold', 5.0),
                'seed': kwargs.get('seed', 42)
            }
            
            # Create training manager
            trainer = KUKATrainingManager(config)
            
            print("✓ RL training system ready")
            print("Starting training process...")
            
            # Start training (this will block)
            summary = trainer.train()
            
            print("✓ RL training completed")
            print(f"Results: {summary}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error launching training: {e}")
            return False
    
    def launch_complete_system(self, config: Dict[str, Any]) -> bool:
        """Launch complete KUKA RL system"""
        print("=== Launching Complete KUKA RL System ===")
        
        # Step 1: Launch Gazebo
        if not self.launch_gazebo(gui=config.get('gazebo_gui', True)):
            print("Failed to launch Gazebo")
            return False
        
        time.sleep(3.0)
        
        # Step 2: Launch world manager
        if not self.launch_world_manager():
            print("Failed to launch world manager")
            return False
        
        time.sleep(2.0)
        
        # Step 3: Launch controller
        if not self.launch_controller():
            print("Failed to launch controller")
            return False
        
        time.sleep(2.0)
        
        # Step 4: Launch training (if requested)
        if config.get('start_training', False):
            if not self.launch_training(
                task=config['task'],
                algorithm=config['algorithm'],
                **config
            ):
                print("Failed to launch training")
                return False
        
        print("✓ Complete KUKA RL system launched successfully")
        return True
    
    def shutdown(self):
        """Shutdown all components"""
        print("Shutting down KUKA RL system...")
        
        # Shutdown controller
        if self.controller:
            self.controller.shutdown()
        
        # Shutdown world manager
        if self.world_manager:
            self.world_manager.shutdown()
        
        # Shutdown Gazebo
        if self.gazebo_process:
            self.gazebo_process.terminate()
            try:
                self.gazebo_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.gazebo_process.kill()
        
        # Shutdown ROS2
        if ROS2_AVAILABLE and rclpy.ok():
            rclpy.shutdown()
        
        print("✓ KUKA RL system shut down")

def create_gazebo_world_file() -> str:
    """Create a basic Gazebo world file for KUKA RL"""
    world_content = """<?xml version="1.0"?>
<sdf version="1.6">
  <world name="kuka_rl_world">
    <!-- Physics settings -->
    <physics name="default_physics" default="0" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    
    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Gravity -->
    <gravity>0 0 -9.81</gravity>
    
    <!-- Magnetic field -->
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    
    <!-- Atmosphere -->
    <atmosphere type='adiabatic'/>
    
    <!-- Scene -->
    <scene>
      <ambient>0.4 0.4 0.4 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>1</shadows>
    </scene>
    
    <!-- Wind -->
    <wind/>
    
    <!-- Spherical coordinates -->
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>0</latitude_deg>
      <longitude_deg>0</longitude_deg>
      <elevation>0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>
  </world>
</sdf>"""
    
    world_file = "/tmp/kuka_rl_world.world"
    with open(world_file, 'w') as f:
        f.write(world_content)
    
    return world_file

def generate_launch_description():
    """Generate ROS2 launch description (if ROS2 launch available)"""
    if not ROS2_AVAILABLE:
        return None
    
    return LaunchDescription([
        # Declare arguments
        DeclareLaunchArgument('task', default_value='reach'),
        DeclareLaunchArgument('algorithm', default_value='ppo'),
        DeclareLaunchArgument('gazebo_gui', default_value='true'),
        DeclareLaunchArgument('start_training', default_value='true'),
        
        # Log info
        LogInfo(msg="Starting KUKA RL System"),
        
        # Launch Gazebo
        ExecuteProcess(
            cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
            output='screen'
        ),
        
        # KUKA world manager node
        LaunchNode(
            package='kuka_rl',
            executable='kuka_world_manager',
            name='kuka_world_manager',
            output='screen'
        ),
        
        # KUKA controller node  
        LaunchNode(
            package='kuka_rl',
            executable='kuka_controller',
            name='kuka_controller',
            output='screen'
        ),
    ])

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(description="Launch KUKA RL System")
    
    # System configuration
    parser.add_argument('--gazebo_gui', action='store_true', default=True,
                       help='Launch Gazebo with GUI')
    parser.add_argument('--no_gazebo_gui', action='store_true',
                       help='Launch Gazebo headless')
    
    # Training configuration  
    parser.add_argument('--start_training', action='store_true',
                       help='Start RL training automatically')
    parser.add_argument('--task', type=str, default='reach',
                       choices=['reach', 'grasp', 'manipulation', 'move'],
                       help='RL task type')
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['ppo', 'sac', 'ddpg'],
                       help='RL algorithm')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--device', type=str, default='auto',
                       help='Computing device (auto/cpu/cuda)')
    
    # System control
    parser.add_argument('--world_only', action='store_true',
                       help='Only launch world manager (no training)')
    parser.add_argument('--controller_only', action='store_true', 
                       help='Only launch controller (no training)')
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = KUKARLLauncher()
    
    try:
        if args.world_only:
            # Launch only world components
            print("Launching world components only...")
            if launcher.launch_gazebo(gui=not args.no_gazebo_gui):
                time.sleep(3)
                launcher.launch_world_manager()
            
        elif args.controller_only:
            # Launch only controller
            print("Launching controller only...")
            launcher.launch_controller()
            
        else:
            # Launch complete system
            config = {
                'gazebo_gui': not args.no_gazebo_gui,
                'start_training': args.start_training,
                'task': args.task,
                'algorithm': args.algorithm,
                'num_episodes': args.num_episodes,
                'device': args.device
            }
            
            launcher.launch_complete_system(config)
        
        if not args.start_training:
            # Keep system running
            print("\nKUKA RL system running. Press Ctrl+C to exit.")
            try:
                if ROS2_AVAILABLE:
                    rclpy.spin(rclpy.create_node('kuka_launcher'))
                else:
                    while True:
                        time.sleep(1.0)
            except KeyboardInterrupt:
                print("\nShutting down...")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        launcher.shutdown()

if __name__ == "__main__":
    main() 