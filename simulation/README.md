# Brain-to-Action Simulation Guide

This directory contains the ROS2 interface for the brain-to-action simulations.

## Prerequisites

### For mock mode (no ROS2 required):
- Python 3.8+
- PyTorch
- NumPy

### For full ROS2 simulation:
1. Install ROS2 following the official guide: 
   - Windows: https://docs.ros.org/en/humble/Installation/Windows-Install-Binary.html
   - Linux: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
   - macOS: https://docs.ros.org/en/humble/Installation/macOS-Install-Binary.html

2. Install additional dependencies:
   ```
   pip install numpy torch
   ```

## Running Simulations

All simulations can be run using the main `brain.py` script in the project root with the `--mode simulate` parameter.

### Action Classification Simulation

```bash
python brain.py --mode simulate --model classification
```

This runs the simulation with a classification model that maps brain signals to discrete robot actions.

### Reinforcement Learning Simulation

```bash
python brain.py --mode simulate --model rl
```

This runs the simulation with an RL agent that generates continuous control signals for the robot.

### Brain Tokenization Simulation

```bash
python brain.py --mode simulate --model tokenization
```

This runs the simulation with a tokenizer model that maps brain signals to joint positions.

## Troubleshooting

- If you see "WARNING: ROS2 packages not found", the simulation will run in mock mode
- To use actual ROS2 functionality, make sure ROS2 is properly installed and sourced
- On Windows, you need to run `call C:\dev\ros2\humble\setup.bat` before running the simulation (adjust path as needed)
- On Linux/macOS, run `source /opt/ros/humble/setup.bash` before running the simulation 