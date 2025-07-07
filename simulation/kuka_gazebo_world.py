#!/usr/bin/env python3
"""
KUKA Gazebo World Setup for Reinforcement Learning
Sets up Gazebo Classic simulation with KUKA iiwa arm for RL training
"""

import os
import sys
import time
import subprocess
from typing import Dict, List, Any

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
    from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState  # type: ignore
    from gazebo_msgs.msg import EntityState  # type: ignore
    from geometry_msgs.msg import Pose, Point, Quaternion  # type: ignore
    from std_srvs.srv import Empty  # type: ignore
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("ROS2 not available - using mock mode")

class KUKAGazeboWorld:
    """Manages KUKA iiwa arm simulation world in Gazebo"""
    
    def __init__(self, node_name: str = "kuka_world_manager"):
        self.node_name = node_name
        self.world_objects = {}
        self.kuka_spawned = False
        
        if ROS2_AVAILABLE:
            self._setup_ros2_node()
        else:
            self._setup_mock_node()
    
    def _setup_ros2_node(self):
        """Setup ROS2 node and services"""
        rclpy.init()
        self.node = Node(self.node_name)
        
        # Gazebo service clients
        self.spawn_client = self.node.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.node.create_client(DeleteEntity, '/delete_entity')
        self.set_state_client = self.node.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.reset_world_client = self.node.create_client(Empty, '/gazebo/reset_world')
        
        # Wait for Gazebo services
        print("Waiting for Gazebo services...")
        self.spawn_client.wait_for_service(timeout_sec=10.0)
        self.delete_client.wait_for_service(timeout_sec=10.0)
        
        print("✓ KUKA Gazebo World Manager initialized")
    
    def _setup_mock_node(self):
        """Setup mock node for testing without ROS2"""
        print("Mock KUKA Gazebo World Manager initialized")
        self.node = None
    
    def get_kuka_iiwa_sdf(self) -> str:
        """Generate SDF model for KUKA iiwa arm"""
        return """
        <?xml version="1.0"?>
        <sdf version="1.6">
            <model name="kuka_iiwa">
                <static>false</static>
                
                <!-- Base Link -->
                <link name="iiwa_link_0">
                    <inertial>
                        <mass>5.0</mass>
                        <inertia>
                            <ixx>0.05</ixx><iyy>0.05</iyy><izz>0.05</izz>
                        </inertia>
                    </inertial>
                    <collision name="base_collision">
                        <geometry>
                            <cylinder><radius>0.1</radius><length>0.15</length></cylinder>
                        </geometry>
                    </collision>
                    <visual name="base_visual">
                        <geometry>
                            <cylinder><radius>0.1</radius><length>0.15</length></cylinder>
                        </geometry>
                        <material>
                            <ambient>0.2 0.2 0.2 1</ambient>
                            <diffuse>0.8 0.8 0.8 1</diffuse>
                        </material>
                    </visual>
                </link>
                
                <!-- Joint 1 -->
                <link name="iiwa_link_1">
                    <pose>0 0 0.15 0 0 0</pose>
                    <inertial>
                        <mass>3.0</mass>
                        <inertia>
                            <ixx>0.03</ixx><iyy>0.03</iyy><izz>0.03</izz>
                        </inertia>
                    </inertial>
                    <collision name="link1_collision">
                        <geometry>
                            <cylinder><radius>0.08</radius><length>0.2</length></cylinder>
                        </geometry>
                    </collision>
                    <visual name="link1_visual">
                        <geometry>
                            <cylinder><radius>0.08</radius><length>0.2</length></cylinder>
                        </geometry>
                        <material>
                            <ambient>1 0.5 0 1</ambient>
                            <diffuse>1 0.5 0 1</diffuse>
                        </material>
                    </visual>
                </link>
                
                <!-- Joint 2 -->
                <link name="iiwa_link_2">
                    <pose>0 0 0.35 0 0 0</pose>
                    <inertial>
                        <mass>3.0</mass>
                        <inertia>
                            <ixx>0.03</ixx><iyy>0.03</iyy><izz>0.03</izz>
                        </inertia>
                    </inertial>
                    <collision name="link2_collision">
                        <geometry>
                            <cylinder><radius>0.07</radius><length>0.18</length></cylinder>
                        </geometry>
                    </collision>
                    <visual name="link2_visual">
                        <geometry>
                            <cylinder><radius>0.07</radius><length>0.18</length></cylinder>
                        </geometry>
                        <material>
                            <ambient>1 0.5 0 1</ambient>
                            <diffuse>1 0.5 0 1</diffuse>
                        </material>
                    </visual>
                </link>
                
                <!-- Joint 3 -->
                <link name="iiwa_link_3">
                    <pose>0 0 0.53 0 0 0</pose>
                    <inertial>
                        <mass>2.5</mass>
                        <inertia>
                            <ixx>0.025</ixx><iyy>0.025</iyy><izz>0.025</izz>
                        </inertia>
                    </inertial>
                    <collision name="link3_collision">
                        <geometry>
                            <cylinder><radius>0.06</radius><length>0.15</length></cylinder>
                        </geometry>
                    </collision>
                    <visual name="link3_visual">
                        <geometry>
                            <cylinder><radius>0.06</radius><length>0.15</length></cylinder>
                        </geometry>
                        <material>
                            <ambient>1 0.5 0 1</ambient>
                            <diffuse>1 0.5 0 1</diffuse>
                        </material>
                    </visual>
                </link>
                
                <!-- Joint 4 -->
                <link name="iiwa_link_4">
                    <pose>0 0 0.68 0 0 0</pose>
                    <inertial>
                        <mass>2.0</mass>
                        <inertia>
                            <ixx>0.02</ixx><iyy>0.02</iyy><izz>0.02</izz>
                        </inertia>
                    </inertial>
                    <collision name="link4_collision">
                        <geometry>
                            <cylinder><radius>0.05</radius><length>0.12</length></cylinder>
                        </geometry>
                    </collision>
                    <visual name="link4_visual">
                        <geometry>
                            <cylinder><radius>0.05</radius><length>0.12</length></cylinder>
                        </geometry>
                        <material>
                            <ambient>1 0.5 0 1</ambient>
                            <diffuse>1 0.5 0 1</diffuse>
                        </material>
                    </visual>
                </link>
                
                <!-- Joint 5 -->
                <link name="iiwa_link_5">
                    <pose>0 0 0.8 0 0 0</pose>
                    <inertial>
                        <mass>1.5</mass>
                        <inertia>
                            <ixx>0.015</ixx><iyy>0.015</iyy><izz>0.015</izz>
                        </inertia>
                    </inertial>
                    <collision name="link5_collision">
                        <geometry>
                            <cylinder><radius>0.04</radius><length>0.1</length></cylinder>
                        </geometry>
                    </collision>
                    <visual name="link5_visual">
                        <geometry>
                            <cylinder><radius>0.04</radius><length>0.1</length></cylinder>
                        </geometry>
                        <material>
                            <ambient>1 0.5 0 1</ambient>
                            <diffuse>1 0.5 0 1</diffuse>
                        </material>
                    </visual>
                </link>
                
                <!-- Joint 6 -->
                <link name="iiwa_link_6">
                    <pose>0 0 0.9 0 0 0</pose>
                    <inertial>
                        <mass>1.0</mass>
                        <inertia>
                            <ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz>
                        </inertia>
                    </inertial>
                    <collision name="link6_collision">
                        <geometry>
                            <cylinder><radius>0.035</radius><length>0.08</length></cylinder>
                        </geometry>
                    </collision>
                    <visual name="link6_visual">
                        <geometry>
                            <cylinder><radius>0.035</radius><length>0.08</length></cylinder>
                        </geometry>
                        <material>
                            <ambient>1 0.5 0 1</ambient>
                            <diffuse>1 0.5 0 1</diffuse>
                        </material>
                    </visual>
                </link>
                
                <!-- Joint 7 (End-effector) -->
                <link name="iiwa_link_7">
                    <pose>0 0 0.98 0 0 0</pose>
                    <inertial>
                        <mass>0.5</mass>
                        <inertia>
                            <ixx>0.005</ixx><iyy>0.005</iyy><izz>0.005</izz>
                        </inertia>
                    </inertial>
                    <collision name="link7_collision">
                        <geometry>
                            <cylinder><radius>0.03</radius><length>0.06</length></cylinder>
                        </geometry>
                    </collision>
                    <visual name="link7_visual">
                        <geometry>
                            <cylinder><radius>0.03</radius><length>0.06</length></cylinder>
                        </geometry>
                        <material>
                            <ambient>0.2 0.8 0.2 1</ambient>
                            <diffuse>0.2 0.8 0.2 1</diffuse>
                        </material>
                    </visual>
                </link>
                
                <!-- Joints -->
                <joint name="iiwa_joint_1" type="revolute">
                    <parent>iiwa_link_0</parent>
                    <child>iiwa_link_1</child>
                    <axis><xyz>0 0 1</xyz>
                        <limit><lower>-2.96</lower><upper>2.96</upper>
                               <effort>320</effort><velocity>1.48</velocity></limit>
                    </axis>
                </joint>
                
                <joint name="iiwa_joint_2" type="revolute">
                    <parent>iiwa_link_1</parent>
                    <child>iiwa_link_2</child>
                    <axis><xyz>0 1 0</xyz>
                        <limit><lower>-2.09</lower><upper>2.09</upper>
                               <effort>320</effort><velocity>1.48</velocity></limit>
                    </axis>
                </joint>
                
                <joint name="iiwa_joint_3" type="revolute">
                    <parent>iiwa_link_2</parent>
                    <child>iiwa_link_3</child>
                    <axis><xyz>0 0 1</xyz>
                        <limit><lower>-2.96</lower><upper>2.96</upper>
                               <effort>176</effort><velocity>1.75</velocity></limit>
                    </axis>
                </joint>
                
                <joint name="iiwa_joint_4" type="revolute">
                    <parent>iiwa_link_3</parent>
                    <child>iiwa_link_4</child>
                    <axis><xyz>0 1 0</xyz>
                        <limit><lower>-2.09</lower><upper>2.09</upper>
                               <effort>176</effort><velocity>1.31</velocity></limit>
                    </axis>
                </joint>
                
                <joint name="iiwa_joint_5" type="revolute">
                    <parent>iiwa_link_4</parent>
                    <child>iiwa_link_5</child>
                    <axis><xyz>0 0 1</xyz>
                        <limit><lower>-2.96</lower><upper>2.96</upper>
                               <effort>110</effort><velocity>2.27</velocity></limit>
                    </axis>
                </joint>
                
                <joint name="iiwa_joint_6" type="revolute">
                    <parent>iiwa_link_5</parent>
                    <child>iiwa_link_6</child>
                    <axis><xyz>0 1 0</xyz>
                        <limit><lower>-2.09</lower><upper>2.09</upper>
                               <effort>40</effort><velocity>2.36</velocity></limit>
                    </axis>
                </joint>
                
                <joint name="iiwa_joint_7" type="revolute">
                    <parent>iiwa_link_6</parent>
                    <child>iiwa_link_7</child>
                    <axis><xyz>0 0 1</xyz>
                        <limit><lower>-3.05</lower><upper>3.05</upper>
                               <effort>40</effort><velocity>2.36</velocity></limit>
                    </axis>
                </joint>
                
                <!-- Gazebo plugins for control -->
                <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
                    <robotNamespace>/kuka_iiwa</robotNamespace>
                </plugin>
            </model>
        </sdf>
        """
    
    def spawn_kuka_arm(self) -> bool:
        """Spawn KUKA iiwa arm in Gazebo"""
        if self.kuka_spawned:
            print("KUKA arm already spawned")
            return True
        
        sdf_content = self.get_kuka_iiwa_sdf()
        
        # Spawn position
        pose = Pose()
        pose.position = Point(x=0.0, y=0.0, z=0.0)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        if ROS2_AVAILABLE and self.spawn_client.service_is_ready():
            request = SpawnEntity.Request()
            request.name = "kuka_iiwa"
            request.xml = sdf_content
            request.initial_pose = pose
            
            future = self.spawn_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future)
            
            if future.result().success:
                self.kuka_spawned = True
                print("✓ KUKA iiwa arm spawned successfully")
                return True
            else:
                print(f"✗ Failed to spawn KUKA arm: {future.result().status_message}")
                return False
        else:
            # Mock spawning
            self.kuka_spawned = True
            print("Mock: KUKA iiwa arm spawned")
            return True
    
    def spawn_target_object(self, name: str, position: tuple, size: float = 0.05) -> bool:
        """Spawn a target object for manipulation tasks"""
        object_sdf = f"""
        <?xml version="1.0"?>
        <sdf version="1.6">
            <model name="{name}">
                <static>false</static>
                <link name="link">
                    <collision name="collision">
                        <geometry>
                            <sphere><radius>{size}</radius></sphere>
                        </geometry>
                    </collision>
                    <visual name="visual">
                        <geometry>
                            <sphere><radius>{size}</radius></sphere>
                        </geometry>
                        <material>
                            <ambient>1 0 0 1</ambient>
                            <diffuse>1 0 0 1</diffuse>
                        </material>
                    </visual>
                    <inertial>
                        <mass>0.1</mass>
                        <inertia>
                            <ixx>0.0001</ixx><iyy>0.0001</iyy><izz>0.0001</izz>
                        </inertia>
                    </inertial>
                </link>
            </model>
        </sdf>
        """
        
        pose = Pose()
        pose.position = Point(x=position[0], y=position[1], z=position[2])
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        if ROS2_AVAILABLE and self.spawn_client.service_is_ready():
            request = SpawnEntity.Request()
            request.name = name
            request.xml = object_sdf
            request.initial_pose = pose
            
            future = self.spawn_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future)
            
            if future.result().success:
                self.world_objects[name] = position
                print(f"✓ Target object '{name}' spawned at {position}")
                return True
            else:
                print(f"✗ Failed to spawn object '{name}': {future.result().status_message}")
                return False
        else:
            # Mock spawning
            self.world_objects[name] = position
            print(f"Mock: Target object '{name}' spawned at {position}")
            return True
    
    def spawn_table(self) -> bool:
        """Spawn a table for manipulation tasks"""
        table_sdf = """
        <?xml version="1.0"?>
        <sdf version="1.6">
            <model name="table">
                <static>true</static>
                <link name="link">
                    <collision name="collision">
                        <geometry>
                            <box><size>1.0 1.0 0.05</size></box>
                        </geometry>
                    </collision>
                    <visual name="visual">
                        <geometry>
                            <box><size>1.0 1.0 0.05</size></box>
                        </geometry>
                        <material>
                            <ambient>0.8 0.6 0.4 1</ambient>
                            <diffuse>0.8 0.6 0.4 1</diffuse>
                        </material>
                    </visual>
                </link>
            </model>
        </sdf>
        """
        
        pose = Pose()
        pose.position = Point(x=0.6, y=0.0, z=0.4)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        if ROS2_AVAILABLE and self.spawn_client.service_is_ready():
            request = SpawnEntity.Request()
            request.name = "table"
            request.xml = table_sdf
            request.initial_pose = pose
            
            future = self.spawn_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future)
            
            if future.result().success:
                print("✓ Table spawned successfully")
                return True
            else:
                print(f"✗ Failed to spawn table: {future.result().status_message}")
                return False
        else:
            print("Mock: Table spawned")
            return True
    
    def setup_rl_environment(self) -> bool:
        """Setup complete RL environment with KUKA arm and objects"""
        print("Setting up KUKA RL environment...")
        
        # Spawn KUKA arm
        if not self.spawn_kuka_arm():
            return False
        
        time.sleep(2.0)
        
        # Spawn table
        if not self.spawn_table():
            return False
        
        time.sleep(1.0)
        
        # Spawn target objects at different positions
        targets = [
            ("target_1", (0.6, 0.2, 0.45)),
            ("target_2", (0.6, -0.2, 0.45)),
            ("target_3", (0.8, 0.0, 0.45)),
        ]
        
        for name, pos in targets:
            if not self.spawn_target_object(name, pos):
                return False
            time.sleep(0.5)
        
        print("✓ KUKA RL environment setup complete")
        return True
    
    def reset_environment(self) -> bool:
        """Reset the environment for new episode"""
        if ROS2_AVAILABLE and self.reset_world_client.service_is_ready():
            request = Empty.Request()
            future = self.reset_world_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future)
            print("Environment reset")
            return True
        else:
            print("Mock: Environment reset")
            return True
    
    def shutdown(self):
        """Clean shutdown"""
        if ROS2_AVAILABLE and self.node:
            self.node.destroy_node()
            rclpy.shutdown()
        print("KUKA Gazebo World Manager shut down")

def main():
    """Test KUKA world setup"""
    world_manager = KUKAGazeboWorld()
    
    try:
        # Setup RL environment
        world_manager.setup_rl_environment()
        
        if ROS2_AVAILABLE:
            print("KUKA world running. Press Ctrl+C to exit.")
            rclpy.spin(world_manager.node)
        else:
            print("Mock KUKA world setup completed.")
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        world_manager.shutdown()

if __name__ == "__main__":
    main() 