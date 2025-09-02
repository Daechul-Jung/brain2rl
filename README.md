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
│  (EEG)          │    │                  │    │ (Transformer +  │    │ (PPO/REPPO with  │
│                 │    │                  │    │  Attention)     │    │  Token Guidance) │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
                                                        │                        │
                                                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   Simulation    │◀───│     OpenArm +    │◀───│   Q/K/V Token   │    │   Trained RL     │
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
(Trying to find out better model for classifying time series data)


**Features:**
- Multi-channel EEG/sensor data processing
- Temporal and spatial convolution layers
- Artifact removal and signal preprocessing
- Support for multiple data formats 

**Usage:**
```bash
# Train classifier (classification-only pipeline)
python3 core/classification_pipeline.py \
    --data-dir data/sensor_data/ \
    --output-dir output/ \
    --config config/classification_config.json

# Run full classification pipeline with custom parameters
python3 core/classification_pipeline.py \
    --data-dir data/sensor_data/ \
    --subject-ids subject_001 subject_002 \
    --output-dir results/classification/ \
    --config config/custom_classification.json
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
python3 core/tokenization_pipeline.py \
    --classified-data output/classification_results.pth \
    --mode train \
    --model-path models/tokenization/best_tokenizer.pth \
    --epochs 100 \
    --embedding-dim 128 \
    --n-tokens 512

# Generate tokens from classified data
python3 core/tokenization_pipeline.py \
    --classified-data output/classification_results.pth \
    --mode tokenize \
    --model-path models/tokenization/best_tokenizer.pth \
    --embedding-dim 128 \
    --n-tokens 512
```

### 3. RL Training Pipeline

Trains OpenArm control using token-guided reinforcement learning.

**Features:**
- Token-guided policy networks
- PPO, REPPO, and CoMic - Actor-Critic algorithm

**Usage:**
```bash
# Train RL agent with token guidance
python3 core/rl_training_pipeline.py \
    --token-data output/tokenization_results.npz \
    --mode train \
    --model-path models/rl/token_guided_agent.pth \
    --episodes 1000 \
    --algorithm ppo \
    --learning-rate 0.0003

# Evaluate trained RL agent
python3 core/rl_training_pipeline.py \
    --token-data output/tokenization_results.npz \
    --mode evaluate \
    --model-path models/rl/token_guided_agent.pth \
    --algorithm ppo

# Run traditional RL training (without tokens)
python3 models/rl/launch/try_mj.py \
    --algo reppo \
    --sim mujoco \
    --mjcf external/openarm_mujoco/v1/scene.xml \
    --render --steps 1000

python3 models/rl/launch/try_gym.py \
    --algo reppo \
    --sim mujoco \
    --mjcf external/openarm_mujoco/v1/scene.xml \
    --render --steps 1000
```

### 4. Main Pipeline Orchestrator

Runs the complete end-to-end pipeline from sensor data to RL training.

**Features:**
- Complete pipeline orchestration
- Individual component execution
- Real-time pipeline support (planned)
- Configuration management

**Usage:**
```bash
# Run complete end-to-end pipeline
python core/main_pipeline.py \
    --mode full \
    --data-path data/sensor_data/ \
    --output-dir output/ \
    --config config/pipeline_config.json

# Run individual pipeline components
python core/main_pipeline.py \
    --mode tokenization \
    --data-path output/classification_results.pth \
    --output-dir output/ \
    --config config/pipeline_config.json

python core/main_pipeline.py \
    --mode rl_training \
    --data-path output/tokenization_results.pth \
    --output-dir output/ \
    --config config/pipeline_config.json

# Run with custom device selection
python core/main_pipeline.py \
    --mode full \
    --data-path data/sensor_data/ \
    --device cuda \
    --output-dir output/ \
    --config config/pipeline_config.json
```

### 5. Simulation and Testing

**Features:**
- Multiple manipulation tasks (reach, grasp, manipulation)
- Performance monitoring and visualization
- OpenArm robot simulation
- MuJoCo and Gymnasium environments

**Usage:**
```bash
# Test MuJoCo environment
python3 models/rl/launch/try_mj.py \
    --algo reppo \
    --sim mujoco \
    --mjcf external/openarm_mujoco/v1/scene.xml \
    --render --steps 1000

# Test Gymnasium environment
python3 models/rl/launch/try_gym.py \
    --algo reppo \
    --sim mujoco \
    --mjcf external/openarm_mujoco/v1/scene.xml \
    --render --steps 1000

# Compare different algorithms
python models/rl/launch/compare_algo.py \
    --algos ppo reppo sac \
    --sim mujoco \
    --mjcf external/openarm_mujoco/v1/scene.xml \
    --steps 1000
```

## Usage Examples

### Complete Pipeline Workflow

Here's how to run the complete Brain2RL pipeline from start to finish:

```bash
# Step 1: Prepare your sensor data
# Place your EEG/sensor data in data/sensor_data/ directory
# Expected format: .npz files with 'X' (data) and 'y' (labels) arrays

# Step 2: Run the complete pipeline
python3 core/main_pipeline.py \
    --mode full \
    --data-path data/sensor_data/ \
    --output-dir output/ \
    --config config/pipeline_config.json

# Step 3: Check results
ls output/
# You should see:
# - classification_results.pth
# - tokenization_results.npz  
# - rl_training_results.pth
# - pipeline_results.pth
```

### Individual Component Workflow

If you prefer to run components individually:

```bash
# Step 1: Classification
python3 core/classification_pipeline.py \
    --data-dir data/sensor_data/ \
    --output-dir output/ \
    --config config/classification_config.json

# Step 2: Tokenization
python3 core/tokenization_pipeline.py \
    --classified-data output/classification_results.pth \
    --mode train \
    --model-path models/tokenization/best_tokenizer.pth \
    --epochs 100

python3 core/tokenization_pipeline.py \
    --classified-data output/classification_results.pth \
    --mode tokenize \
    --model-path models/tokenization/best_tokenizer.pth

# Step 3: RL Training
python3 core/rl_training_pipeline.py \
    --token-data output/tokenization_results.npz \
    --mode train \
    --model-path models/rl/token_guided_agent.pth \
    --episodes 1000

# Step 4: Evaluation
python3 core/rl_training_pipeline.py \
    --token-data output/tokenization_results.npz \
    --mode evaluate \
    --model-path models/rl/token_guided_agent.pth
```

### Configuration Files

Create configuration files for customizing the pipeline:

```json
// config/pipeline_config.json
{
    "classification": {
        "window_size": 100,
        "batch_size": 32,
        "classifier_lr": 0.001,
        "classifier_epochs": 100,
        "classifier_dropout": 0.3
    },
    "tokenization": {
        "embedding_dim": 128,
        "n_tokens": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "dropout": 0.1,
        "max_sequence_length": 1000
    },
    "rl_training": {
        "algorithm": "ppo",
        "learning_rate": 0.0003,
        "batch_size": 64,
        "gamma": 0.99,
        "clip_range": 0.2,
        "episodes": 1000
    }
}
```

### Project Structure

```
brain2rl/
├── core/                          # Main pipeline components
│   ├── main_pipeline.py          # Orchestrator
│   ├── classification_pipeline.py # Classification component
│   ├── tokenization_pipeline.py  # Tokenization component
│   ├── rl_training_pipeline.py   # RL training component
├── models/                        # Model architectures
│   ├── classification/           # Action classification models 
│   ├── tokenization/            # Tokenizing time series data for trajectories
│   └── rl/                      # RL model for OpenArm and general gym env
│       ├── agents               # RL Agents Collection
|       ├─- launch               # code for launching with env
│       └── utils                # Colection of Neural network and Actor-Critic network and any other utilities.
|── scripts/
|   └─openarm/
|      └─play_policy.py  
└── README.md                     
```

### Adding New Models

1. **Classification Models**: Add to `brain2rl/models/classification/`
2. **Tokenization Models**: Add to `brain2rl/models/tokenization/`
3. **RL Algorithms**: Add to `brain2rl/models/rl/agents`

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