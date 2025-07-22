# Brain2RL Project

A comprehensive pipeline for converting brain signals (EEG, fMRI, etc.) to reinforcement learning control of KUKA robot arms through tokenization and attention mechanisms.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Pipeline Components](#pipeline-components)
- [Contributing](#contributing)

## Overview

Brain2RL is an end-to-end pipeline that transforms brain signal data into robot control commands through four main stages:

1. **Classification**: Sensor data → Action classification
2. **Tokenization**: Time series data → Tokens with Q/K/V matrices
3. **RL Training**: Token-guided reinforcement learning
4. **Simulation**: KUKA robot arm control in Gazebo

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

- **Multi-modal sensor support**: EEG, fMRI, MEG, and other brain signals
- **Advanced tokenization**: Transformer-based architecture with attention mechanisms
- **Token-guided RL**: Novel approach using brain signal tokens to guide robot learning
- **Real-time simulation**: Full Gazebo + ROS2 integration with KUKA robot models
- **Flexible pipeline**: Run individual components or complete end-to-end workflow
- **Windows compatibility**: Designed for Windows 10/11 with WSL support

## Installation

### Prerequisites

- **Operating System**: Windows 10/11 with WSL2 or native Linux
- **Python**: 3.8 or higher
- **CUDA**: Optional but recommended for GPU acceleration
- **ROS2 Humble**: For robot simulation (can be mocked for development)
- **Gazebo Classic**: For physics simulation


## Pipeline Components

### 1. Classification Pipeline

Converts raw sensor data into action classifications using CNN models.

**Features:**
- Multi-channel EEG/sensor data processing
- Temporal and spatial convolution layers
- Artifact removal and signal preprocessing
- Support for multiple data formats (NPY, NPZ, CSV, MAT, H5)

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

Trains KUKA robot control using token-guided reinforcement learning.

**Features:**
- Token-guided policy networks
- PPO and SAC algorithm support
- Attention-based action selection
- Comparison with baseline agents

**Usage:**
```bash
# Train RL agent
python brain2rl/cli.py rl-training \
    --token-data results/tokens.npz \
    --episodes 1000 \
    --algorithm ppo \
    --model-path models/rl_agent.pth \
    --plot-results
```

### 4. Simulation Pipeline

Runs trained agents in realistic KUKA robot simulation with Gazebo and ROS2.

**Features:**
- Real-time KUKA iiwa simulation
- Multiple manipulation tasks (reach, grasp, manipulation)
- Performance monitoring and visualization
- ROS2/Gazebo integration with mock modes

**Usage:**
```bash
# Run simulation
python brain2rl/cli.py simulation \
    --model-path models/rl_agent.pth \
    --token-data results/tokens.npz \
    --episodes 10 \
    --visualize \
    --task reach
```

## Usage Examples

### Example 1: Complete Pipeline with Real EEG Data

```bash
# 1. Prepare your EEG data (see Data Formats section)
# Ensure data is in .npz format with 'data' and 'labels' arrays

# 2. Run full pipeline
python brain2rl/cli.py full \
    --data-path data/eeg_motor_imagery.npz \
    --output-dir results/eeg_experiment/ \
    --config config/eeg_config.json

# 3. Results will be saved in results/eeg_experiment/
```

### Example 2: Step-by-Step Pipeline Execution

```bash
# Step 1: Train classification model
python brain2rl/cli.py classification \
    --mode train \
    --data-path data/eeg_data.npz \
    --model-path models/eeg_classifier.pth \
    --epochs 150 \
    --n-channels 64 \
    --n-classes 4

# Step 2: Classify validation data
python brain2rl/cli.py classification \
    --mode classify \
    --data-path data/eeg_validation.npz \
    --model-path models/eeg_classifier.pth \
    --output-path results/classified_validation.npz

# Step 3: Train tokenizer
python brain2rl/cli.py tokenization \
    --mode train \
    --classified-data results/classified_validation.npz \
    --model-path models/eeg_tokenizer.pth \
    --embedding-dim 256 \
    --epochs 200

# Step 4: Generate tokens
python brain2rl/cli.py tokenization \
    --mode tokenize \
    --classified-data results/classified_validation.npz \
    --model-path models/eeg_tokenizer.pth \
    --output-path results/eeg_tokens.npz

# Step 5: Train RL agent
python brain2rl/cli.py rl-training \
    --token-data results/eeg_tokens.npz \
    --episodes 2000 \
    --algorithm ppo \
    --learning-rate 0.0003 \
    --model-path models/eeg_rl_agent.pth

# Step 6: Run simulation
python brain2rl/cli.py simulation \
    --model-path models/eeg_rl_agent.pth \
    --token-data results/eeg_tokens.npz \
    --episodes 20 \
    --task manipulation \
    --visualize \
    --save-data results/simulation_data.npz
```

### Example 3: Development and Testing

```bash
# Generate test data
python brain2rl/cli.py generate-data \
    --n-samples 500 \
    --n-channels 16 \
    --n-timesteps 256 \
    --output-path data/test_data.npz

# Quick pipeline test (reduced parameters)
python brain2rl/cli.py full \
    --data-path data/test_data.npz \
    --output-dir test_results/ \
    --device cpu

# Mock simulation (no ROS2/Gazebo required)
python brain2rl/cli.py simulation \
    --model-path test_results/full_pipeline_results.pth \
    --episodes 3 \
    --mock-mode \
    --use-gazebo false \
    --use-ros false
```

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
│   └── rl/                      # RL model for KUKA with tokens
├── simulation/                    # Robot simulation
│   ├── kuka_gym_environment.py  # KUKA Gym environment
│   ├── kuka_ros_controller.py   # ROS controller
│   └── kuka_gazebo_world.py     # Gazebo world
├── utils/                         # Utilities
│   ├── data_utils.py            # Data processing
│   └── visualization.py         # Plotting tools
├── cli.py                        # Command-line interface
└── README.md                     # This file
```

### Adding New Models

1. **Classification Models**: Add to `brain2rl/models/classification/`
2. **Tokenization Models**: Add to `brain2rl/models/tokenization/`
3. **RL Algorithms**: Add to `brain2rl/models/rl/`

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Test individual components
python brain2rl/core/classification_pipeline.py
python brain2rl/core/tokenization_pipeline.py
python brain2rl/core/rl_training_pipeline.py
python brain2rl/core/simulation_pipeline.py
```


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

- **Email**: daechul.jung@vanderbilt.edu, jungdaechul@berkeley.edu
- **GitHub**: https://github.com/Daechul-Jung/brain2rl
- **Documentation**: https://brain2rl.readthedocs.io

---

**Brain2RL** - Bridging minds and machines through intelligent robotics. 