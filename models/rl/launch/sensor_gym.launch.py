from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    # Launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time')
    visualize = LaunchConfiguration('visualize')
    
    # Declare launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    visualize_arg = DeclareLaunchArgument(
        'visualize',
        default_value='true',
        description='Enable visualization'
    )
    
    # Sensor Gym Environment Node
    sensor_gym_node = Node(
        package='sensor_rl',
        executable='sensor_gym_env',
        name='sensor_gym_env',
        parameters=[{
            'use_sim_time': use_sim_time,
            'visualize': visualize,
            'window_size': 100,  # Size of the sliding window for feature extraction
            'update_rate': 10.0,  # Hz
            'max_steps': 1000,    # Maximum steps per episode
        }],
        output='screen'
    )
    
    # IMU Data Publisher Node (for testing)
    imu_publisher_node = Node(
        package='sensor_rl',
        executable='imu_publisher',
        name='imu_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
            'publish_rate': 100.0,  # Hz
            'noise_level': 0.01,    # Add some noise to the data
        }],
        condition=IfCondition(use_sim_time)
    )
    
    # Visualization Node
    visualization_node = Node(
        package='sensor_rl',
        executable='visualization_node',
        name='visualization_node',
        parameters=[{
            'use_sim_time': use_sim_time,
            'plot_rate': 1.0,  # Hz
        }],
        condition=IfCondition(visualize)
    )
    
    return LaunchDescription([
        use_sim_time_arg,
        visualize_arg,
        sensor_gym_node,
        imu_publisher_node,
        visualization_node
    ]) 