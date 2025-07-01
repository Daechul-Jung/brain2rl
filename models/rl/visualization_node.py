#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from queue import Queue
import time

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        
        # Parameters
        self.declare_parameter('plot_rate', 1.0)
        self.plot_rate = self.get_parameter('plot_rate').value
        
        # Subscribers
        self.create_subscription(
            Float32MultiArray,
            '/sensor_gym/action',
            self.action_callback,
            10
        )
        
        self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        
        # Data queues
        self.action_queue = Queue(maxsize=1000)
        self.imu_queue = Queue(maxsize=1000)
        self.reward_queue = Queue(maxsize=1000)
        
        # Plotting setup
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Sensor Gym Training Visualization')
        
        # Initialize plots
        self.action_plot = self.axes[0, 0].bar(range(5), [0]*5)
        self.axes[0, 0].set_title('Action Distribution')
        self.axes[0, 0].set_xlabel('Action')
        self.axes[0, 0].set_ylabel('Count')
        
        self.imu_plot, = self.axes[0, 1].plot([], [])
        self.axes[0, 1].set_title('IMU Data')
        self.axes[0, 1].set_xlabel('Time')
        self.axes[0, 1].set_ylabel('Value')
        
        self.reward_plot, = self.axes[1, 0].plot([], [])
        self.axes[1, 0].set_title('Rewards')
        self.axes[1, 0].set_xlabel('Episode')
        self.axes[1, 0].set_ylabel('Reward')
        
        self.state_plot = self.axes[1, 1].scatter([], [], c=[], cmap='viridis')
        self.axes[1, 1].set_title('State Space')
        self.axes[1, 1].set_xlabel('Dimension 1')
        self.axes[1, 1].set_ylabel('Dimension 2')
        
        # Start animation
        self.ani = FuncAnimation(
            self.fig,
            self.update_plots,
            interval=1000/self.plot_rate,
            blit=False
        )
        
        # Start ROS spin in a separate thread
        self.spin_thread = threading.Thread(target=self._spin)
        self.spin_thread.daemon = True
        self.spin_thread.start()
        
        # Show plot
        plt.show()
    
    def _spin(self):
        """Spin ROS node in a separate thread"""
        rclpy.spin(self)
    
    def action_callback(self, msg):
        """Callback for action messages"""
        self.action_queue.put(msg.data)
    
    def imu_callback(self, msg):
        """Callback for IMU messages"""
        data = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]
        self.imu_queue.put(data)
    
    def update_plots(self, frame):
        """Update all plots"""
        # Update action distribution
        if not self.action_queue.empty():
            actions = []
            while not self.action_queue.empty():
                actions.extend(self.action_queue.get())
            action_counts = np.bincount(actions, minlength=5)
            for rect, count in zip(self.action_plot, action_counts):
                rect.set_height(count)
            self.axes[0, 0].relim()
            self.axes[0, 0].autoscale_view()
        
        # Update IMU data
        if not self.imu_queue.empty():
            imu_data = []
            while not self.imu_queue.empty():
                imu_data.append(self.imu_queue.get())
            imu_data = np.array(imu_data)
            self.imu_plot.set_data(range(len(imu_data)), imu_data[:, 0])
            self.axes[0, 1].relim()
            self.axes[0, 1].autoscale_view()
        
        # Update reward plot
        if not self.reward_queue.empty():
            rewards = []
            while not self.reward_queue.empty():
                rewards.append(self.reward_queue.get())
            self.reward_plot.set_data(range(len(rewards)), rewards)
            self.axes[1, 0].relim()
            self.axes[1, 0].autoscale_view()
        
        return (
            self.action_plot,
            self.imu_plot,
            self.reward_plot,
            self.state_plot
        )

def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 