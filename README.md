# Brain2RL Project

A comprehensive pipeline for converting brain signals to reinforcement learning control of OpenArm through tokenization and attention mechanisms.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Pipeline Components](#pipeline-components)
- [Contributing](#contributing)

## Overview

Brain2RL is an end-to-end pipeline that transforms brain signal data into robot control commands through four main stages:

1. **Classification**: EEG data → Action classification
2. **Tokenization**: Time series data → Tokens with Q/K/V matrices
3. **RL Training**: Relative Entropy Pairwise Policy Optimization reinforcement learning
4. **Simulation**: OpenArm Project and Humanoid v-5 and other experimental environment 

The pipeline enables robots to learn from human brain signals, creating a direct brain-to-robot interface for complex manipulation tasks.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Sensor Data   │───▶│  Classification  │───▶│  Tokenization   │───▶│   RL Training    │
│  (EEG/etc)      │    │                  │    │ (Transformer +  │    │ (PPO/SAC with    │
│                 │    │                  │    │  Attention)     │    │  Token Guidance) │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
                                                        │                        │
                                                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Simulation    │◀───│     Gazebo +     │◀───│   Q/K/V Token   │    │   Trained RL     │
│  Visualization  │    │   ROS2 Control   │    │    Matrices     │    │     Agent        │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
```

### Key Features

- **Multi-modal sensor support**: EEG brain signals
- **Advanced tokenization**: Transformer-based architecture with attention mechanisms
- **Token-guided RL**: Novel approach using brain signal tokens to guide robot learning
- **simulation**: OpenArm Project
- **Flexible pipeline**: Run individual components or complete end-to-end workflow


### Installation Prerequisites

- **Operating System**: Ubuntu 22.04
- **Python**: 3.8 or higher
- **CUDA**: Optional but recommended for GPU acceleration
- **ROS2 Humble**: For robot simulation (can be mocked for development)

## Pipeline Components

### 1. Classification Pipeline

Converts raw sensor data into action classifications using CNN models.

**Features:**
- Multi-channel EEG/sensor data processing
- Temporal and spatial convolution layers
- Artifact removal and signal preprocessing
- Support for multiple data formats 

**Usage:**
```bash
# Train classifier
python brain2rl/cli.py classification \
    --mode train \
    --data-path data/sensor_data.npz \
    --model-path models/classifier.pth \
    --epochs 100 \
    --batch-size 32

# Classify new data
python brain2rl/cli.py classification \
    --mode classify \
    --data-path data/new_sensor_data.npz \
    --model-path models/classifier.pth \
    --output-path results/predictions.npz
```

### 2. Tokenization Pipeline

Transforms classified time series data into tokens with Query/Key/Value matrices for attention-based control.

**Features:**
- Transformer-based architecture
- Multi-head attention mechanisms
- Query/Key/Value matrix generation
- Temporal sequence modeling

**Usage:**
```bash
# Train tokenizer
python brain2rl/cli.py tokenization \
    --mode train \
    --classified-data results/predictions.npz \
    --model-path models/tokenizer.pth \
    --embedding-dim 128 \
    --n-tokens 512

# Generate tokens
python brain2rl/cli.py tokenization \
    --mode tokenize \
    --classified-data results/predictions.npz \
    --model-path models/tokenizer.pth \
    --output-path results/tokens.npz
```

### 3. RL Training Pipeline

Trains OpenArm control using token-guided reinforcement learning.

**Features:**
- Token-guided policy networks
- PPO, REPPO, and CoMic - Actor-Critic algorithm

**Usage:**
```bash
# Train RL agent
python3 models/rl/launch/try_mj.py \
  --algo reppo \
  --sim mujoco \
  --mjcf ~/brain2rl/external/openarm_mujoco/v1/scene.xml \
  --render --steps 1000

  python3 models/rl/launch/try_gym.py \
  --algo reppo \
  --sim mujoco \
  --mjcf ~/brain2rl/external/openarm_mujoco/v1/scene.xml \
  --render --steps 1000
```

### 4. Simulation Pipeline

Runs trained agents in OpenArm simulation in ROS2 with Gazebo.

**Features:**
- Multiple manipulation tasks (reach, grasp, manipulation)
- Performance monitoring and visualization
- OpenArm 

**Usage:**
```bash
# Run simulation
 ########### Need to rewrite ############
```

## Usage Examples

### Project Structure

```
brain2rl/
├── core/                          # Main pipeline components
│   ├── main_pipeline.py          # Orchestrator
│   ├── classification_pipeline.py # Classification component
│   ├── tokenization_pipeline.py  # Tokenization component
│   ├── rl_training_pipeline.py   # RL training component
│   └── simulation_pipeline.py    # Simulation component
├── models/                        # Model architectures
│   ├── classification/           # Action classification models 
│   ├── tokenization/            # Tokenizing time series data for trajectories
│   └── rl/                      # RL model for OpenArm and general gym env
│       ├── agents               # RL Agents Collection
|       ├─- launch               # code for launching with env
│       └── utils                # Colection of Neural network and Actor-Critic network
|── scripts/
|   └─openarm/
|      └─play_policy.py  
└── README.md                     
```

### Adding New Models

1. **Classification Models**: Add to `brain2rl/models/classification/`
2. **Tokenization Models**: Add to `brain2rl/models/tokenization/`
3. **RL Algorithms**: Add to `brain2rl/models/rl/practice/agents`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Brain2RL in your research, please cite:

```bibtex
@software{brain2rl2024,
  title={Brain2RL: End-to-End Pipeline for Brain Signal to Robot Control},
  author={Daechul Jung},
  year={2025},
  url={https://github.com/Daechul-Jung/brain2rl}
}
```

## Contact

- **Email**: jungdaechul@berkeley.edu
- **GitHub**: https://github.com/Daechul-Jung/brain2rl
- **Documentation**: https://brain2rl.readthedocs.io

---

**Brain2RL** - Bridging brain signal and machines through intelligent robotics. 