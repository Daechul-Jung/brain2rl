#!/usr/bin/env python3
"""
ROS2 Interface for Brain to Agent's Actions
This module provides the interfaces to connect brain signal processing with ROS2-based robotic simulations.
"""

import os
import sys 
import numpy as np
import torch
import time
import threading
from typing import Dict, List, Any, Optional, Callable

# Import conditional to support environments without ROS2
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, String
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image, JointState
    from builtin_interfaces.msg import Time
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("WARNING: ROS2 packages not found, using mock ROS2 interface")

# Mock classes for simulation without ROS2
class MockMessage:
    """Mock message for ROS2-less environments"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class MockROS2Node:
    """Mock ROS2 node for ROS2-less environments"""
    def __init__(self, node_name):
        self.node_name = node_name
        self.publishers = {}
        self.subscriptions = {}
        self.timers = []
        print(f"Created mock ROS2 node: {node_name}")
    
    def create_publisher(self, msg_type, topic, qos_profile):
        pub = MockPublisher(topic, msg_type)
        self.publishers[topic] = pub
        return pub
    
    def create_subscription(self, msg_type, topic, callback, qos_profile):
        sub = MockSubscription(topic, msg_type, callback)
        self.subscriptions[topic] = sub
        return sub
    
    def create_timer(self, timer_period_sec, callback):
        timer = threading.Timer(timer_period_sec, callback)
        self.timers.append(timer)
        return timer
    
    def destroy_node(self):
        print(f"Destroyed mock ROS2 node: {self.node_name}")
        for timer in self.timers:
            timer.cancel()

class MockPublisher:
    """Mock publisher for ROS2-less environments"""
    def __init__(self, topic, msg_type):
        self.topic = topic
        self.msg_type = msg_type
        print(f"Created mock publisher on topic: {topic}")
    
    def publish(self, msg):
        print(f"Published mock message on topic: {self.topic}")

class MockSubscription:
    """Mock subscription for ROS2-less environments"""
    def __init__(self, topic, msg_type, callback):
        self.topic = topic
        self.msg_type = msg_type
        self.callback = callback
        print(f"Created mock subscription on topic: {self.topic}")

class BrainROSInterface:
    """Main interface between brain signals and ROS2 robotics"""
    def __init__(self, 
                 node_name: str = "brain_interface",
                 use_mock: bool = not ROS2_AVAILABLE):
        """
        Initialize the ROS interface for brain to action translation.
        
        Args:
            node_name: Name of the ROS2 node
            use_mock: Force use of mock implementation even if ROS2 is available
        """
        self.use_mock = use_mock
        self.node_name = node_name
        self.initialized = False
        self._setup_node()
    
    def _setup_node(self):
        """Set up the ROS2 node and publishers/subscribers"""
        # Initialize node
        if self.use_mock:
            self.node = MockROS2Node(self.node_name)
        else:
            rclpy.init()
            self.node = Node(self.node_name)
            
        # QoS profile for real-time control
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        ) if not self.use_mock else None
        
        # Create publishers
        self.brain_state_pub = self.node.create_publisher(
            Float32MultiArray, "/brain/state", qos)
        
        self.cmd_vel_pub = self.node.create_publisher(
            Twist, "/cmd_vel", qos)
        
        self.joint_cmd_pub = self.node.create_publisher(
            JointState, "/joint_commands", qos)
        
        # Create subscriptions
        self.sensor_sub = self.node.create_subscription(
            Image, "/camera/image_raw", self._sensor_callback, qos)
        
        self.joint_state_sub = self.node.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, qos)
        
        self.initialized = True
        print(f"{'Mock ' if self.use_mock else ''}ROS2 Brain Interface initialized")
    
    def _sensor_callback(self, msg):
        """Process incoming sensor data"""
        # In the mock version, this will never be called
        if not self.use_mock:
            # Process camera image data
            pass
    
    def _joint_state_callback(self, msg):
        """Process incoming joint state data"""
        # In the mock version, this will never be called
        if not self.use_mock:
            # Process joint state data
            pass
    
    def publish_brain_state(self, brain_state: np.ndarray):
        """
        Publish processed brain signal data to ROS2
        
        Args:
            brain_state: Numpy array of processed brain signal features
        """
        if not self.initialized:
            print("WARNING: ROS2 interface not initialized")
            return
            
        if self.use_mock:
            msg = MockMessage(data=brain_state.flatten().tolist())
        else:
            msg = Float32MultiArray()
            msg.data = brain_state.flatten().tolist()
        
        self.brain_state_pub.publish(msg)
        
    def send_velocity_command(self, linear_x: float, angular_z: float):
        """
        Send velocity command to robot
        
        Args:
            linear_x: Linear velocity (m/s)
            angular_z: Angular velocity (rad/s)
        """
        if not self.initialized:
            print("WARNING: ROS2 interface not initialized")
            return
            
        if self.use_mock:
            msg = MockMessage(
                linear=MockMessage(x=linear_x, y=0.0, z=0.0),
                angular=MockMessage(x=0.0, y=0.0, z=angular_z)
            )
        else:
            msg = Twist()
            msg.linear.x = linear_x
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = angular_z
            
        self.cmd_vel_pub.publish(msg)
    
    def send_joint_command(self, joint_positions: List[float], joint_names: List[str] = None):
        """
        Send joint position commands to robot
        
        Args:
            joint_positions: List of joint positions
            joint_names: Optional list of joint names (must match the robot's joint names)
        """
        if not self.initialized:
            print("WARNING: ROS2 interface not initialized")
            return
            
        if self.use_mock:
            msg = MockMessage(
                name=joint_names if joint_names else [],
                position=joint_positions,
                velocity=[],
                effort=[]
            )
        else:
            msg = JointState()
            if joint_names:
                msg.name = joint_names
            msg.position = joint_positions
            msg.velocity = []
            msg.effort = []
            
        self.joint_cmd_pub.publish(msg)
    
    def shutdown(self):
        """Clean shutdown of ROS2 interface"""
        if self.initialized:
            if not self.use_mock:
                rclpy.shutdown()
            else:
                self.node.destroy_node()
            self.initialized = False
            print(f"{'Mock ' if self.use_mock else ''}ROS2 Brain Interface shutdown")


def load_brain_model(model_path: str, model_type: str = 'classification'):
    """
    Load a trained brain model for inference
    
    Args:
        model_path: Path to the model file
        model_type: Type of model (classification, tokenization, rl)
    
    Returns:
        Loaded model
    """
    import torch
    
    try:
        # This is a placeholder - you would need to implement the actual model loading
        if model_type == 'classification':
            from models.classification.action_classifier import ActionClassifier
            model = ActionClassifier()
            model.load_state_dict(torch.load(model_path))
        elif model_type == 'tokenization':
            from models.tokenization.brain_tokenizer import BrainTokenizer
            model = BrainTokenizer()
            model.load_state_dict(torch.load(model_path))
        elif model_type == 'rl':
            from models.rl.brain_rl import BrainRLAgent
            model = BrainRLAgent()
            model.load_state_dict(torch.load(model_path))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.eval()  # Set to inference mode
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def run_simulation(args):
    """
    Run a ROS2-based simulation with the brain model
    
    Args:
        args: Command line arguments from brain.py
    """
    print("\n==== Starting Brain-Agent Simulation with ROS2 ====")
    
    # Check if ROS2 is available
    if not ROS2_AVAILABLE:
        print("WARNING: ROS2 not available, using mock interface")
    
    try:
        # Initialize ROS2 interface
        interface = BrainROSInterface(use_mock=not ROS2_AVAILABLE)
        
        # Load the appropriate brain model
        model_dir = os.path.join('models', args.model)
        model_path = os.path.join(model_dir, f'brain_{args.model}_model.pt')
        
        print(f"Loading brain model: {model_path}")
        model = load_brain_model(model_path, args.model)
        
        if model is None:
            print("ERROR: Failed to load brain model")
            interface.shutdown()
            return
        
        # Mock data source for simulation
        # In a real application, this would be replaced with actual brain signal data
        def mock_brain_data():
            # Generate random brain signal data for simulation
            return np.random.randn(64, 128)  # Example size: 64 channels, 128 timepoints
        
        print("Simulation running. Press Ctrl+C to exit.")
        
        # Main simulation loop
        try:
            iteration = 0
            while True:
                # Get brain data (mock or real)
                brain_data = mock_brain_data()
                
                # Process with the model
                with torch.no_grad():
                    brain_data_tensor = torch.tensor(brain_data, dtype=torch.float32)
                    # This is a placeholder - actual processing depends on model type
                    processed_output = model(brain_data_tensor)
                
                # Publish brain state to ROS2
                interface.publish_brain_state(brain_data)
                
                # Example control commands based on model output
                if args.model == 'classification':
                    # Example: Convert classification output to robot control
                    class_idx = torch.argmax(processed_output).item()
                    
                    # Map class index to robot actions (example)
                    if class_idx == 0:  # Forward
                        interface.send_velocity_command(0.2, 0.0)
                    elif class_idx == 1:  # Turn left
                        interface.send_velocity_command(0.1, 0.5)
                    elif class_idx == 2:  # Turn right
                        interface.send_velocity_command(0.1, -0.5)
                    else:  # Stop
                        interface.send_velocity_command(0.0, 0.0)
                
                elif args.model == 'rl':
                    # Example: Use RL agent output directly as continuous control
                    action = processed_output.numpy()
                    interface.send_velocity_command(action[0], action[1])
                
                elif args.model == 'tokenization':
                    # Example: Convert tokenized output to joint positions
                    joint_positions = processed_output.numpy()
                    interface.send_joint_command(joint_positions)
                
                # Sleep to maintain loop rate
                time.sleep(0.1)
                
                iteration += 1
                if iteration % 10 == 0:
                    print(f"Simulation iteration: {iteration}")
        
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        
        finally:
            # Clean shutdown
            interface.shutdown()
            print("Simulation ended")
    
    except Exception as e:
        print(f"Error in simulation: {e}") 