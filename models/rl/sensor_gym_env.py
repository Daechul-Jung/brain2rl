import gym
import numpy as np
import rospy
from gym import spaces
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import threading
from queue import Queue
import time

class SensorGymEnv(gym.Env):
    """
    Custom Gym Environment for sensor data that integrates with ROS2
    """
    def __init__(self, feature_processor, action_space_size=5):
        super(SensorGymEnv, self).__init__()
        
        # Initialize ROS2 node
        rclpy.init()
        self.node = Node('sensor_gym_env')
        
        # Set up QoS profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Set up publishers and subscribers
        self.imu_sub = self.node.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            qos_profile
        )
        
        self.action_pub = self.node.create_publisher(
            Float32MultiArray,
            '/sensor_gym/action',
            qos_profile
        )
        
        # Initialize feature processor
        self.feature_processor = feature_processor
        
        # Set up action and observation spaces
        self.action_space = spaces.Discrete(action_space_size)  # 5 actions: walking, running, sitting, standing, lying
        
        # Observation space will be the size of our feature vector
        feature_size = self._get_feature_size()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(feature_size,),
            dtype=np.float32
        )
        
        # Initialize data queues
        self.imu_queue = Queue(maxsize=100)
        self.feature_queue = Queue(maxsize=10)
        
        # Start ROS2 spin thread
        self.ros_thread = threading.Thread(target=self._spin_ros)
        self.ros_thread.daemon = True
        self.ros_thread.start()
        
        # Initialize state
        self.current_state = None
        self.current_features = None
        self.episode_step = 0
        self.max_steps = 1000
        
    def _get_feature_size(self):
        """Get the size of the feature vector"""
        # Create a dummy window to get feature size
        dummy_window = np.zeros((100, 6))  # 100 timesteps, 6 channels
        features = self.feature_processor.extract_cnn_features(
            dummy_window[np.newaxis, ...],
            self.feature_processor.feature_extractor,
            'cpu'
        )
        return sum(f.flatten().shape[0] for f in features.values())
    
    def _spin_ros(self):
        """Spin ROS2 node in a separate thread"""
        rclpy.spin(self.node)
    
    def imu_callback(self, msg):
        """Callback for IMU data"""
        # Extract IMU data
        acc_x = msg.linear_acceleration.x
        acc_y = msg.linear_acceleration.y
        acc_z = msg.linear_acceleration.z
        gyro_x = msg.angular_velocity.x
        gyro_y = msg.angular_velocity.y
        gyro_z = msg.angular_velocity.z
        
        # Create data point
        data_point = np.array([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        
        # Add to queue
        if not self.imu_queue.full():
            self.imu_queue.put(data_point)
    
    def _get_observation(self):
        """Get current observation from sensor data"""
        # Collect data points until we have enough for a window
        window_data = []
        while len(window_data) < self.feature_processor.window_size:
            if not self.imu_queue.empty():
                window_data.append(self.imu_queue.get())
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
        
        # Convert to numpy array
        window = np.array(window_data)
        
        # Extract features
        features = self.feature_processor.extract_cnn_features(
            window[np.newaxis, ...],
            self.feature_processor.feature_extractor,
            'cpu'
        )
        
        # Create state representation
        state = self.feature_processor.create_rl_state_representation(
            features,
            method='concat'
        )
        
        return state
    
    def step(self, action):
        """
        Execute one time step within the environment
        
        Args:
            action: The action to take (0-4 for different behaviors)
        
        Returns:
            observation: The new observation
            reward: The reward for the action
            done: Whether the episode is done
            info: Additional information
        """
        self.episode_step += 1
        
        # Publish action
        action_msg = Float32MultiArray()
        action_msg.data = [float(action)]
        self.action_pub.publish(action_msg)
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate reward (this is a placeholder - you'll need to implement your own reward function)
        reward = self._calculate_reward(action, observation)
        
        # Check if episode is done
        done = self.episode_step >= self.max_steps
        
        info = {
            'episode_step': self.episode_step,
            'action': action
        }
        
        return observation, reward, done, info
    
    def _calculate_reward(self, action, observation):
        """
        Calculate reward for the current state and action
        
        This is a placeholder - you should implement your own reward function
        based on your specific requirements
        """
        # Example reward function:
        # - Small negative reward for each step to encourage efficiency
        # - Positive reward for maintaining stable state
        # - Negative reward for sudden changes
        
        base_reward = -0.1  # Small negative reward for each step
        
        # Calculate stability reward
        if self.current_features is not None:
            stability = np.mean(np.abs(observation - self.current_features))
            stability_reward = -stability  # Negative reward for instability
        else:
            stability_reward = 0
        
        # Update current features
        self.current_features = observation
        
        return base_reward + stability_reward
    
    def reset(self):
        """Reset the environment to initial state"""
        self.episode_step = 0
        self.current_features = None
        
        # Clear queues
        while not self.imu_queue.empty():
            self.imu_queue.get()
        while not self.feature_queue.empty():
            self.feature_queue.get()
        
        # Get initial observation
        observation = self._get_observation()
        
        return observation
    
    def close(self):
        """Clean up environment"""
        self.node.destroy_node()
        rclpy.shutdown()

# Example usage
if __name__ == '__main__':
    from models.pipelines.sensor_data_pipeline import SensorDataProcessor
    
    # Initialize feature processor
    processor = SensorDataProcessor()
    
    # Create environment
    env = SensorGymEnv(processor)
    
    # Test environment
    obs = env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        if done:
            obs = env.reset()
    
    env.close() 