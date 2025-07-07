# KUKA iiwa Reinforcement Learning Simulation

A comprehensive reinforcement learning system for KUKA iiwa arm control in Gazebo Classic simulation, designed for Windows OS with ROS2 Humble.

## 🎯 System Overview

This system implements a complete RL pipeline where:
- **KUKA iiwa arm** = RL Agent (learns to control 7 joints)
- **Gazebo Classic** = Environment (physics simulation)
- **ROS2** = Communication bridge between components
- **Custom Gym Environment** = Standard RL interface

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RL Agent      │◄──►│  Gym Environment │◄──►│  ROS2 Bridge    │
│   (PPO/SAC)     │    │  (Observation/   │    │                 │
│                 │    │   Reward/Action) │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Gazebo World   │◄──►│ KUKA Controller  │◄──►│   ROS2 Topics   │
│  (Physics Sim)  │    │ (Joint Control)  │    │   /Services     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 File Structure

```
brain2rl/simulation/
├── kuka_gazebo_world.py       # Gazebo world setup with KUKA arm
├── kuka_ros_controller.py     # ROS2 controller for KUKA arm
├── kuka_gym_environment.py    # Gym environment wrapper
├── kuka_rl_agent.py          # RL agents (PPO, SAC, DDPG)
├── train_kuka_rl.py          # Training script with logging
├── launch_kuka_rl.py         # System launcher
└── README.md                 # This file
```

## 🛠️ Installation

### Prerequisites

**Windows 10/11 with:**
- ROS2 Humble installed at `C:\dev\ros2_humble`
- Gazebo Classic (comes with ROS2 Humble)
- Python 3.8+

### Python Dependencies

```bash
# Core RL dependencies
pip install torch torchvision torchaudio
pip install gymnasium numpy matplotlib

# Optional: Stable-Baselines3 for advanced RL algorithms
pip install stable-baselines3

# Visualization
pip install tensorboard seaborn
```

### ROS2 Setup

The system automatically configures ROS2 environment. Ensure ROS2 Humble is installed:

```powershell
# Verify ROS2 installation
C:\dev\ros2_humble\setup.bat
ros2 --version
gazebo --version
```

## 🚀 Quick Start

### 1. Basic Training

Train KUKA arm to reach targets:

```bash
cd brain2rl/simulation
python train_kuka_rl.py --task reach --algorithm ppo --num_episodes 500
```

### 2. Complete System Launch

Launch Gazebo + KUKA + Training:

```bash
python launch_kuka_rl.py --start_training --task reach --algorithm ppo
```

### 3. Evaluation Only

Test a trained model:

```bash
python train_kuka_rl.py --evaluate_only --model_path path/to/model.pkl --task reach
```

## 🎮 Tasks Available

### 1. **Reach Task** (`reach`)
- **Goal**: Move end-effector to random target positions
- **Reward**: Distance-based reward, success bonus
- **Difficulty**: ⭐⭐☆☆☆

### 2. **Grasp Task** (`grasp`)  
- **Goal**: Approach and grasp objects on table
- **Reward**: Approach + grasp success rewards
- **Difficulty**: ⭐⭐⭐☆☆

### 3. **Manipulation Task** (`manipulation`)
- **Goal**: Pick up object and move to goal position
- **Reward**: Multi-stage rewards (approach → grasp → transport)
- **Difficulty**: ⭐⭐⭐⭐☆

### 4. **Waypoint Navigation** (`move`)
- **Goal**: Follow sequence of waypoints
- **Reward**: Progress-based rewards
- **Difficulty**: ⭐⭐⭐☆☆

## 🤖 RL Algorithms

### PPO (Proximal Policy Optimization)
```bash
python train_kuka_rl.py --algorithm ppo --task reach
```
- **Best for**: Stable training, good sample efficiency
- **Hyperparameters**: Learning rate 3e-4, clip ratio 0.2

### SAC (Soft Actor-Critic)
```bash
python train_kuka_rl.py --algorithm sac --task grasp
```
- **Best for**: Continuous control, exploration
- **Hyperparameters**: Learning rate 3e-4, temperature 0.2

### Custom Implementation
- Built-in PyTorch implementations
- Automatic device selection (CPU/CUDA)
- Save/load functionality

## 📊 Training Options

### Basic Training
```bash
python train_kuka_rl.py \
    --task reach \
    --algorithm ppo \
    --num_episodes 1000 \
    --device auto
```

### Advanced Training
```bash
python train_kuka_rl.py \
    --task manipulation \
    --algorithm sac \
    --num_episodes 2000 \
    --eval_frequency 25 \
    --save_frequency 50 \
    --device cuda \
    --early_stop_threshold 10.0
```

### Training Arguments
- `--task`: RL task type (`reach`, `grasp`, `manipulation`, `move`)
- `--algorithm`: RL algorithm (`ppo`, `sac`, `ddpg`)
- `--num_episodes`: Number of training episodes
- `--eval_frequency`: Episodes between evaluations
- `--save_frequency`: Episodes between model saves
- `--device`: Computing device (`auto`, `cpu`, `cuda`)
- `--early_stop_threshold`: Early stopping threshold

## 🎯 System Components

### 1. Gazebo World (`kuka_gazebo_world.py`)
- KUKA iiwa arm model (7-DOF)
- Physics simulation environment
- Objects and obstacles for tasks
- SDF model generation

### 2. ROS2 Controller (`kuka_ros_controller.py`)
- Joint position/velocity control
- Forward/inverse kinematics
- ROS2 topic interface
- Safety limits and bounds

### 3. Gym Environment (`kuka_gym_environment.py`)
- Standard OpenAI Gym interface
- Task-specific reward functions
- Observation space: joints + velocities + end-effector + task features
- Action space: 7 joint positions

### 4. RL Agents (`kuka_rl_agent.py`)
- PPO and SAC implementations
- Neural network architectures
- Training and evaluation loops
- Model save/load functionality

## 📈 Monitoring Training

### Training Logs
```
Episode   10: Reward= -45.23, Steps=134, Success=False, Avg100= -52.18
Episode   20: Reward= -38.91, Steps=156, Success=False, Avg100= -48.45
...
Evaluation (Ep 50): Mean Reward=-35.67, Success Rate=0.15
```

### Training Plots
Automatic generation of:
- Episode rewards over time
- Moving average rewards
- Evaluation performance
- Success rates

### Output Structure
```
training_results/kuka_reach_ppo_20241201_143022/
├── training_config.json      # Configuration used
├── training_history.json     # Episode-by-episode results
├── training_progress.png     # Training plots
├── best_model.pkl           # Best performing model
├── final_model.pkl          # Final model state
└── training_summary.json    # Final summary
```

## 🔧 System Control

### Launch Components Separately

**World Only** (Gazebo + KUKA):
```bash
python launch_kuka_rl.py --world_only
```

**Controller Only**:
```bash
python launch_kuka_rl.py --controller_only
```

**Headless Mode** (no GUI):
```bash
python launch_kuka_rl.py --no_gazebo_gui --start_training
```

### ROS2 Topics

Monitor system via ROS2:
```bash
# Joint states
ros2 topic echo /joint_states

# KUKA status
ros2 topic echo /kuka_iiwa/status

# RL actions
ros2 topic echo /kuka_iiwa/rl_action

# Reset command
ros2 topic pub /kuka_iiwa/reset std_msgs/Bool data:\ true
```

## 🐛 Troubleshooting

### Common Issues

**ROS2 Import Errors:**
```
Import 'rclpy' could not be resolved
```
- ✅ Files include `# type: ignore` comments for IDE compatibility
- System automatically sets up ROS2 environment
- Mock modes available when ROS2 unavailable

**Gazebo Launch Failures:**
```bash
# Verify Gazebo installation
gazebo --version

# Check ROS2 Gazebo plugins
ros2 pkg list | grep gazebo
```

**Training Slow Performance:**
- Use `--device cuda` if NVIDIA GPU available
- Reduce `--num_episodes` for initial testing
- Use `--no_gazebo_gui` for headless training

**System Requirements:**
- RAM: 8GB+ recommended
- GPU: Optional but recommended for training
- Storage: 2GB+ for models and logs

### Mock Mode

System supports mock mode when ROS2/Gazebo unavailable:
- Simulated environment
- Fake robot arm responses  
- Algorithm testing without full simulation

## 📚 Usage Examples

### Example 1: Quick Reach Training
```bash
python train_kuka_rl.py --task reach --num_episodes 200
```

### Example 2: Advanced Manipulation
```bash
python train_kuka_rl.py \
    --task manipulation \
    --algorithm sac \
    --num_episodes 1500 \
    --device cuda
```

### Example 3: Evaluation and Testing
```bash
# Train model
python train_kuka_rl.py --task grasp --num_episodes 800

# Find best model (in training_results/kuka_grasp_ppo_*/)
python train_kuka_rl.py \
    --evaluate_only \
    --model_path training_results/kuka_grasp_ppo_*/best_model.pkl \
    --task grasp
```

### Example 4: Complete Pipeline
```bash
# Launch full system with training
python launch_kuka_rl.py \
    --start_training \
    --task manipulation \
    --algorithm ppo \
    --num_episodes 1000
```

## 🔬 Research Applications

This system enables research in:
- **Robot Learning**: End-to-end RL for manipulation
- **Sim-to-Real Transfer**: Gazebo simulation to real KUKA arms
- **Multi-Task Learning**: Training across different manipulation tasks
- **Algorithm Comparison**: PPO vs SAC vs other algorithms
- **Human-Robot Interaction**: Brain signals to robot control

## 🤝 Contributing

To extend the system:
1. Add new tasks in `kuka_gym_environment.py`
2. Implement new RL algorithms in `kuka_rl_agent.py`
3. Create custom reward functions
4. Add new robot models in `kuka_gazebo_world.py`

## 📄 License

This project is part of the brain2rl package for brain-controlled robotics research.

---

**🎉 Ready to train your KUKA arm with reinforcement learning!**

For questions or issues, check the troubleshooting section or review the code comments in each file. 