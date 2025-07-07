#!/usr/bin/env python3
"""
KUKA iiwa ROS2 Controller for Reinforcement Learning
This controller provides the interface between RL agents and the KUKA arm in Gazebo
"""

import os
import sys
import time
import numpy as np
import threading
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup ROS2 environment
try:
    from scripts.setup_ros2_environment import setup_ros2_environment
    setup_ros2_environment()
except ImportError:
    print("WARNING: Could not import ROS2 setup script")

try:
    import rclpy  # type: ignore
    from rclpy.node import Node  # type: ignore
    from rclpy.action import ActionClient  # type: ignore
    from std_msgs.msg import Float64MultiArray, String, Bool  # type: ignore
    from sensor_msgs.msg import JointState  # type: ignore
    from geometry_msgs.msg import Pose, Point, Quaternion  # type: ignore
    from control_msgs.action import FollowJointTrajectory  # type: ignore
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint  # type: ignore
    from builtin_interfaces.msg import Duration  # type: ignore
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 not available - using mock mode")

class KUKARosController:
    """ROS2 Controller for KUKA iiwa arm"""
    
    def __init__(self, node_name: str = "kuka_controller"):
        self.node_name = node_name
        
        # KUKA iiwa joint configuration
        self.joint_names = [
            "iiwa_joint_1", "iiwa_joint_2", "iiwa_joint_3", "iiwa_joint_4",
            "iiwa_joint_5", "iiwa_joint_6", "iiwa_joint_7"
        ]
        
        # Joint limits (radians)
        self.joint_limits = {
            'lower': np.array([-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05]),
            'upper': np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05]),
            'velocity': np.array([1.48, 1.48, 1.75, 1.31, 2.27, 2.36, 2.36])
        }
        
        # Current state
        self.current_joint_positions = np.zeros(7)
        self.current_joint_velocities = np.zeros(7)
        self.current_end_effector_pose = None
        self.controller_ready = False
        
        # Control parameters
        self.control_frequency = 50.0  # Hz
        self.position_tolerance = 0.01  # radians
        
        if ROS2_AVAILABLE:
            self._setup_ros2_node()
        else:
            self._setup_mock_node()
    
    def _setup_ros2_node(self):
        """Setup ROS2 node and interfaces"""
        rclpy.init()
        self.node = Node(self.node_name)
        
        # Publishers
        self.joint_command_pub = self.node.create_publisher(
            Float64MultiArray, '/kuka_iiwa/joint_commands', 10)
        
        self.arm_status_pub = self.node.create_publisher(
            String, '/kuka_iiwa/status', 10)
        
        self.end_effector_pose_pub = self.node.create_publisher(
            Pose, '/kuka_iiwa/end_effector_pose', 10)
        
        # Subscribers
        self.joint_state_sub = self.node.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, 10)
        
        self.rl_action_sub = self.node.create_subscription(
            Float64MultiArray, '/kuka_iiwa/rl_action', 
            self._rl_action_callback, 10)
        
        self.reset_sub = self.node.create_subscription(
            Bool, '/kuka_iiwa/reset', self._reset_callback, 10)
        
        # Action clients (for trajectory control)
        self.trajectory_client = ActionClient(
            self.node, FollowJointTrajectory, 
            '/kuka_iiwa/iiwa_arm_controller/follow_joint_trajectory')
        
        # Timer for control loop
        self.control_timer = self.node.create_timer(
            1.0/self.control_frequency, self._control_loop)
        
        # Initialize home position
        self.home_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.target_position = self.home_position.copy()
        
        print("✓ KUKA ROS2 Controller initialized")
        self.controller_ready = True
    
    def _setup_mock_node(self):
        """Setup mock node for testing"""
        print("Mock KUKA ROS2 Controller initialized")
        self.node = None
        self.controller_ready = True
        self.current_joint_positions = self.home_position = np.zeros(7)
        self.target_position = self.home_position.copy()
    
    def _joint_state_callback(self, msg: JointState):
        """Process joint state updates from Gazebo"""
        if len(msg.name) >= 7 and len(msg.position) >= 7:
            # Map joint positions by name
            joint_positions = np.zeros(7)
            joint_velocities = np.zeros(7)
            
            for i, joint_name in enumerate(self.joint_names):
                if joint_name in msg.name:
                    idx = msg.name.index(joint_name)
                    joint_positions[i] = msg.position[idx]
                    if len(msg.velocity) > idx:
                        joint_velocities[i] = msg.velocity[idx]
            
            self.current_joint_positions = joint_positions
            self.current_joint_velocities = joint_velocities
            
            # Compute end-effector pose
            self.current_end_effector_pose = self._compute_forward_kinematics(joint_positions)
            
            # Publish end-effector pose
            if self.current_end_effector_pose:
                self.end_effector_pose_pub.publish(self.current_end_effector_pose)
    
    def _rl_action_callback(self, msg: Float64MultiArray):
        """Process RL action commands"""
        if len(msg.data) == 7:
            # Direct joint position control
            action = np.array(msg.data)
            self.set_joint_positions(action)
        elif len(msg.data) == 3:
            # Cartesian position control (x, y, z)
            target_xyz = np.array(msg.data)
            self.set_end_effector_position(target_xyz)
        else:
            print(f"Invalid RL action dimension: {len(msg.data)}")
    
    def _reset_callback(self, msg: Bool):
        """Reset arm to home position"""
        if msg.data:
            self.reset_to_home()
    
    def _control_loop(self):
        """Main control loop for position control"""
        if not self.controller_ready:
            return
        
        # Simple position control
        position_error = self.target_position - self.current_joint_positions
        
        # Check if we're at target
        if np.linalg.norm(position_error) < self.position_tolerance:
            status = "REACHED_TARGET"
        else:
            status = f"MOVING_ERROR_{np.linalg.norm(position_error):.3f}"
        
        # Publish status
        if ROS2_AVAILABLE:
            status_msg = String()
            status_msg.data = status
            self.arm_status_pub.publish(status_msg)
    
    def set_joint_positions(self, positions: np.ndarray, duration: float = 2.0) -> bool:
        """Set target joint positions"""
        if len(positions) != 7:
            print(f"ERROR: Expected 7 joint positions, got {len(positions)}")
            return False
        
        # Clamp to joint limits
        clamped_positions = np.clip(positions, 
                                   self.joint_limits['lower'], 
                                   self.joint_limits['upper'])
        
        if not np.array_equal(clamped_positions, positions):
            print("WARNING: Joint positions clamped to limits")
        
        self.target_position = clamped_positions
        
        if ROS2_AVAILABLE and self.trajectory_client.server_is_ready():
            # Send trajectory command
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = self.joint_names
            
            # Create trajectory point
            point = JointTrajectoryPoint()
            point.positions = clamped_positions.tolist()
            point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))
            
            goal.trajectory.points = [point]
            
            # Send goal
            self.trajectory_client.send_goal_async(goal)
            print(f"Sent joint trajectory: {clamped_positions}")
        else:
            # Direct command publishing (for simple controllers)
            if ROS2_AVAILABLE:
                cmd_msg = Float64MultiArray()
                cmd_msg.data = clamped_positions.tolist()
                self.joint_command_pub.publish(cmd_msg)
            else:
                # Mock control
                self.current_joint_positions = clamped_positions
                print(f"Mock: Set joint positions: {clamped_positions}")
        
        return True
    
    def set_end_effector_position(self, target_xyz: np.ndarray) -> bool:
        """Set end-effector position using inverse kinematics"""
        if len(target_xyz) != 3:
            print("ERROR: Expected 3D position (x, y, z)")
            return False
        
        # Simple inverse kinematics for reaching
        joint_positions = self._inverse_kinematics(target_xyz)
        
        if joint_positions is not None:
            return self.set_joint_positions(joint_positions)
        else:
            print("ERROR: Could not solve inverse kinematics")
            return False
    
    def _inverse_kinematics(self, target_xyz: np.ndarray) -> Optional[np.ndarray]:
        """Simple inverse kinematics solver"""
        # This is a simplified IK solver for the KUKA iiwa
        # In practice, you would use a more sophisticated solver
        
        x, y, z = target_xyz
        
        # Check if target is reachable (simple reach check)
        reach = np.linalg.norm(target_xyz)
        max_reach = 0.98  # Approximate max reach of KUKA iiwa
        
        if reach > max_reach:
            print(f"Target {target_xyz} is out of reach (max: {max_reach})")
            return None
        
        # Simple geometric IK for demonstration
        # Joint 1: Base rotation
        q1 = np.arctan2(y, x)
        
        # Simplified arm positioning
        r = np.sqrt(x*x + y*y)
        
        # Joint 2: Shoulder
        q2 = np.arcsin(np.clip((z - 0.36) / 0.42, -1, 1))  # Approximate
        
        # Joint 3: Elbow
        q3 = np.pi/4  # Simple configuration
        
        # Joints 4-7: Wrist configuration for end-effector orientation
        q4 = -q2 - q3  # Keep end-effector horizontal
        q5 = 0.0
        q6 = 0.0
        q7 = 0.0
        
        joint_positions = np.array([q1, q2, q3, q4, q5, q6, q7])
        
        # Clamp to joint limits
        joint_positions = np.clip(joint_positions,
                                 self.joint_limits['lower'],
                                 self.joint_limits['upper'])
        
        return joint_positions
    
    def _compute_forward_kinematics(self, joint_positions: np.ndarray) -> Pose:
        """Compute end-effector pose from joint positions"""
        # Simplified forward kinematics for KUKA iiwa
        # This should use proper DH parameters in practice
        
        q = joint_positions
        
        # Simplified FK calculation
        # Link lengths (approximate)
        L1, L2, L3, L4 = 0.36, 0.42, 0.4, 0.22
        
        # End-effector position (simplified)
        x = (L2 * np.cos(q[1]) + L3 * np.cos(q[1] + q[2]) + L4 * np.cos(q[1] + q[2] + q[3])) * np.cos(q[0])
        y = (L2 * np.cos(q[1]) + L3 * np.cos(q[1] + q[2]) + L4 * np.cos(q[1] + q[2] + q[3])) * np.sin(q[0])
        z = L1 + L2 * np.sin(q[1]) + L3 * np.sin(q[1] + q[2]) + L4 * np.sin(q[1] + q[2] + q[3])
        
        # Create pose message
        pose = Pose()
        pose.position = Point(x=float(x), y=float(y), z=float(z))
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)  # Simplified
        
        return pose
    
    def reset_to_home(self) -> bool:
        """Reset arm to home position"""
        print("Resetting KUKA arm to home position")
        return self.set_joint_positions(self.home_position, duration=3.0)
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions"""
        return self.current_joint_positions.copy()
    
    def get_joint_velocities(self) -> np.ndarray:
        """Get current joint velocities"""
        return self.current_joint_velocities.copy()
    
    def get_end_effector_pose(self) -> Optional[Pose]:
        """Get current end-effector pose"""
        return self.current_end_effector_pose
    
    def get_joint_limits(self) -> Dict[str, np.ndarray]:
        """Get joint limits"""
        return self.joint_limits.copy()
    
    def is_motion_complete(self, tolerance: float = None) -> bool:
        """Check if arm has reached target position"""
        if tolerance is None:
            tolerance = self.position_tolerance
        
        position_error = np.linalg.norm(self.target_position - self.current_joint_positions)
        velocity_norm = np.linalg.norm(self.current_joint_velocities)
        
        return position_error < tolerance and velocity_norm < 0.1
    
    def apply_joint_velocities(self, velocities: np.ndarray) -> bool:
        """Apply joint velocities directly (for velocity control)"""
        if len(velocities) != 7:
            print(f"ERROR: Expected 7 joint velocities, got {len(velocities)}")
            return False
        
        # Clamp velocities
        clamped_velocities = np.clip(velocities,
                                   -self.joint_limits['velocity'],
                                   self.joint_limits['velocity'])
        
        if ROS2_AVAILABLE:
            # This would typically go to a velocity controller
            # For now, integrate to position
            dt = 1.0 / self.control_frequency
            new_positions = self.current_joint_positions + clamped_velocities * dt
            return self.set_joint_positions(new_positions, duration=dt)
        else:
            # Mock velocity control
            dt = 0.02  # 50Hz
            self.current_joint_positions += clamped_velocities * dt
            print(f"Mock: Applied velocities: {clamped_velocities}")
            return True
    
    def get_workspace_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get approximate workspace bounds for the arm"""
        return {
            'x': (-0.8, 0.8),
            'y': (-0.8, 0.8), 
            'z': (0.0, 1.2)
        }
    
    def shutdown(self):
        """Clean shutdown"""
        if ROS2_AVAILABLE and self.node:
            self.node.destroy_node()
            rclpy.shutdown()
        print("KUKA ROS2 Controller shut down")

def main():
    """Test KUKA controller"""
    controller = KUKARosController()
    
    try:
        # Move to home position
        controller.reset_to_home()
        time.sleep(3)
        
        # Test joint control
        test_positions = np.array([0.5, 0.3, -0.2, 0.8, 0.0, -0.5, 0.0])
        controller.set_joint_positions(test_positions)
        time.sleep(3)
        
        # Test Cartesian control
        target_xyz = np.array([0.4, 0.2, 0.6])
        controller.set_end_effector_position(target_xyz)
        time.sleep(3)
        
        # Return home
        controller.reset_to_home()
        
        if ROS2_AVAILABLE:
            print("KUKA controller running. Press Ctrl+C to exit.")
            rclpy.spin(controller.node)
        else:
            print("Mock KUKA controller test completed.")
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        controller.shutdown()

if __name__ == "__main__":
    main() 