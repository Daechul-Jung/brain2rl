"""
Brain2RL Core Module
====================

This module contains the core components for the Brain2RL pipeline:
1. Classification: Sensor data to action classification
2. Tokenization: Time series data to tokens with Q/K/V matrices
3. RL Training: Token-guided reinforcement learning
4. Simulation: KUKA robot arm simulation environment

The pipeline flow:
Sensor Data -> Classification -> Tokenization -> RL Training -> Simulation
"""

__version__ = "1.0.0"
__author__ = "Brain2RL Team" 