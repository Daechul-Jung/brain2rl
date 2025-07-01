"""
Simulation interface for brain-guided agents with ROS2
"""

from .ros_interface import (
    BrainROSInterface,
    run_simulation,
    # Mock classes for simulation without ROS2
    MockROS2Node,
    MockPublisher,
    MockSubscription,
    MockMessage
) 