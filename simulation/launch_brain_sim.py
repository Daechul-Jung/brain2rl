#!/usr/bin/env python3
"""
ROS2 Launch file for Brain to Agent's Actions simulation environment
This launch file sets up the ROS2 environment for the brain-controlled robot simulation.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    # Launch arguments
    use_gazebo = LaunchConfiguration('use_gazebo', default='true')
    robot_model = LaunchConfiguration('robot_model', default='turtlebot3_waffle')
    world_file = LaunchConfiguration('world_file', default='empty.world')
    
    # Declare the launch arguments
    launch_args = [
        DeclareLaunchArgument(
            'use_gazebo',
            default_value='true',
            description='Whether to launch Gazebo simulation'
        ),
        DeclareLaunchArgument(
            'robot_model',
            default_value='turtlebot3_waffle',
            description='Robot model to spawn (turtlebot3_burger, turtlebot3_waffle, turtlebot3_waffle_pi)'
        ),
        DeclareLaunchArgument(
            'world_file',
            default_value='empty.world',
            description='World file for Gazebo simulation'
        )
    ]
    
    # Launch Gazebo simulation (conditional)
    gazebo = ExecuteProcess(
        condition=IfCondition(use_gazebo),
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so', world_file],
        output='screen'
    )
    
    # Launch robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_gazebo}]
    )
    
    # Spawn the robot in Gazebo (conditional)
    spawn_robot = Node(
        condition=IfCondition(use_gazebo),
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'brain_controlled_robot',
                  '-topic', 'robot_description'],
        output='screen'
    )
    
    # Brain interface node that connects to the brain.py process
    brain_interface = Node(
        package='brain_to_agent',  # This would need to be set up as a ROS2 package
        executable='brain_interface_node.py',
        name='brain_interface',
        output='screen',
        parameters=[
            {'use_sim_time': use_gazebo},
            {'simulation_mode': True}
        ]
    )
    
    # Return the LaunchDescription
    return LaunchDescription(launch_args + [
        gazebo,
        robot_state_publisher,
        spawn_robot,
        brain_interface
    ]) 