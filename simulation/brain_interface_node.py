#!/usr/bin/env python3
"""
ROS2 Node for Brain Interface
This script creates a ROS2 node that interfaces between the brain signal processing and the robot.
"""

import os
import sys
import time
import numpy as np
import threading
import argparse

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import MultiThreadedExecutor
    from std_msgs.msg import Float32MultiArray, String, Bool
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image, JointState
    from cv_bridge import CvBridge
    
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("WARNING: ROS2 packages not found. Cannot run as ROS2 node.")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the BrainROSInterface
from simulation.ros_interface import BrainROSInterface

class BrainInterfaceNode:
    """ROS2 node wrapper for Brain Interface"""
    def __init__(self):
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Brain Interface Node")
        parser.add_argument('--data', type=str, default='eeg',
                           choices=['eeg', 'fmri'],
                           help='Data type to use')
        parser.add_argument('--model', type=str, default='classification',
                           choices=['classification', 'tokenization', 'rl'],
                           help='Model type to use')
        parser.add_argument('--mock', action='store_true',
                           help='Use mock interface even if ROS2 is available')
        
        # Parse known args and pass remaining to ROS
        args, remaining = parser.parse_known_args()
        
        # Override sys.argv with remaining args for ROS
        sys.argv = [sys.argv[0]] + remaining
        
        # Check if ROS2 is available
        if not ROS2_AVAILABLE:
            print("ERROR: ROS2 is not available. Cannot run as ROS2 node.")
            sys.exit(1)
        
        # Initialize ROS2
        rclpy.init()
        
        # Create node
        self.node = Node('brain_interface_node')
        
        # Get parameters from node
        self.node.declare_parameter('simulation_mode', True)
        self.simulation_mode = self.node.get_parameter('simulation_mode').value
        
        # Set up the brain interface
        self.interface = BrainROSInterface(
            node_name='brain_interface_inner',
            use_mock=args.mock
        )
        
        # Set up model parameters
        self.model_type = args.model
        self.data_type = args.data
        
        # Create additional publishers/subscribers specific to the node
        # These are in addition to those already in BrainROSInterface
        self.status_pub = self.node.create_publisher(
            String, '/brain/status', 10)
        
        self.emergency_stop_sub = self.node.create_subscription(
            Bool, '/brain/emergency_stop', self.emergency_stop_callback, 10)
        
        # Threading for the brain processing loop
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.brain_processing_loop)
        
        # Status tracking
        self.is_running = False
        self.iteration = 0
        
        print("Brain Interface Node initialized")
    
    def emergency_stop_callback(self, msg):
        """Handle emergency stop messages"""
        if msg.data:
            print("EMERGENCY STOP RECEIVED")
            self.stop()
            # Send zero velocity command
            self.interface.send_velocity_command(0.0, 0.0)
    
    def start(self):
        """Start the brain processing thread"""
        if not self.is_running:
            self.is_running = True
            self.stop_event.clear()
            self.thread.start()
            self.publish_status("RUNNING")
            print("Brain processing started")
    
    def stop(self):
        """Stop the brain processing thread"""
        if self.is_running:
            self.is_running = False
            self.stop_event.set()
            self.thread.join()
            self.publish_status("STOPPED")
            print("Brain processing stopped")
    
    def publish_status(self, status_str):
        """Publish status message"""
        msg = String()
        msg.data = status_str
        self.status_pub.publish(msg)
    
    def brain_processing_loop(self):
        """Main processing loop for brain signals"""
        try:
            # In a real application, we would load a model here
            # For simulation, we'll just use random data
            
            while not self.stop_event.is_set():
                try:
                    # Generate mock brain data if in simulation mode
                    if self.simulation_mode:
                        # Generate random brain signal data (64 channels, 128 timepoints)
                        brain_data = np.random.randn(64, 128)
                    else:
                        # In real mode, we would get data from a BCI device
                        # For now, just use mock data
                        brain_data = np.random.randn(64, 128)
                    
                    # Process the brain data
                    # In a real application, we would pass this to a model
                    # For simulation, we'll just use random outputs
                    
                    # Publish raw brain data
                    self.interface.publish_brain_state(brain_data)
                    
                    # Generate actions based on model type
                    if self.model_type == 'classification':
                        # Simulate classification (4 classes)
                        class_idx = np.random.randint(0, 4)
                        
                        # Map class to robot commands
                        if class_idx == 0:  # Forward
                            self.interface.send_velocity_command(0.2, 0.0)
                        elif class_idx == 1:  # Turn left
                            self.interface.send_velocity_command(0.1, 0.5)
                        elif class_idx == 2:  # Turn right
                            self.interface.send_velocity_command(0.1, -0.5)
                        else:  # Stop
                            self.interface.send_velocity_command(0.0, 0.0)
                    
                    elif self.model_type == 'rl':
                        # Simulate RL output (2D continuous action)
                        action = np.random.randn(2) * 0.5
                        # Scale to reasonable velocity commands
                        linear_x = np.clip(action[0], -0.5, 0.5)
                        angular_z = np.clip(action[1], -1.0, 1.0)
                        self.interface.send_velocity_command(linear_x, angular_z)
                    
                    elif self.model_type == 'tokenization':
                        # Simulate tokenizer output (joint positions)
                        # Assume 6 joints for a robotic arm
                        joint_positions = np.random.uniform(-1.0, 1.0, 6)
                        joint_names = [f"joint_{i}" for i in range(6)]
                        self.interface.send_joint_command(joint_positions, joint_names)
                    
                    # Update iteration counter and publish status periodically
                    self.iteration += 1
                    if self.iteration % 10 == 0:
                        status = f"RUNNING - iteration {self.iteration}"
                        self.publish_status(status)
                        print(status)
                    
                    # Sleep to maintain loop rate (10Hz)
                    time.sleep(0.1)
                
                except Exception as e:
                    print(f"Error in brain processing: {e}")
                    time.sleep(1.0)  # Sleep longer on error
        
        finally:
            # Ensure clean shutdown
            self.interface.shutdown()
    
    def run(self):
        """Run the ROS2 node"""
        try:
            # Start the brain processing
            self.start()
            
            # Spin the node
            rclpy.spin(self.node)
        
        except KeyboardInterrupt:
            print("Node interrupted by user")
        
        finally:
            # Clean shutdown
            self.stop()
            self.node.destroy_node()
            rclpy.shutdown()
            print("Node shutdown complete")

def main():
    """Main entry point"""
    try:
        # Create and run the node
        node = BrainInterfaceNode()
        node.run()
    
    except Exception as e:
        print(f"Error in main: {e}")
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main() 