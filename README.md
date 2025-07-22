# Brain2RL Pipeline

A comprehensive pipeline for converting brain signals (EEG, fMRI, etc.) to reinforcement learning control of KUKA robot arms through tokenization and attention mechanisms.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Pipeline Components](#pipeline-components)
- [Configuration](#configuration)
- [Data Formats](#data-formats)
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

### Step 1: Clone Repository

```bash
git clone https://github.com/Daechul-Jung/brain2rl.git
cd brain2rl
```

### Step 2: Install Python Dependencies

```bash
# Create virtual environment
python -m venv brain2rl_env
source brain2rl_env/bin/activate  # On Windows: brain2rl_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Install ROS2 Humble 

For full simulation capabilities:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ros-humble-desktop

# Source ROS2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Install Gazebo Classic

```bash
# Ubuntu/Debian
sudo apt install gazebo11 gazebo11-dev gazebo11-common
```

### Step 5: Verify Installation

```bash
python brain2rl/cli.py generate-data --n-samples 100 --output-path test_data.npz
python brain2rl/cli.py classification --mode train --data-path test_data.npz --epochs 5
```



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

## Configuration

### Default Configuration

The pipeline uses sensible defaults, but you can customize behavior with JSON configuration files:

```json
{
  "data_dir": "data/",
  "classification": {
    "model_type": "eeg_cnn",
    "n_channels": 32,
    "n_times": 512,
    "n_classes": 6,
    "dropout_rate": 0.5,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100
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
    "n_steps": 2048,
    "n_epochs": 10,
    "gamma": 0.99,
    "clip_range": 0.2
  },
  "simulation": {
    "robot_type": "kuka_iiwa",
    "use_gui": true,
    "real_time_factor": 1.0,
    "max_episode_steps": 1000,
    "control_frequency": 100
  }
}
```

### Custom Configuration

```bash
# Save custom config
cp config/default_config.json config/my_config.json
# Edit my_config.json with your parameters

# Use custom config
python brain2rl/cli.py full \
    --data-path data/my_data.npz \
    --config config/my_config.json \
    --output-dir results/
```

## Data Formats

### Input Data Format

The pipeline supports multiple input formats:

#### NPZ Format (Recommended)
```python
import numpy as np

# Save data in NPZ format
np.savez('sensor_data.npz', 
         data=sensor_array,      # Shape: (n_samples, n_channels, n_timesteps)
         labels=action_labels)   # Shape: (n_samples,) or (n_samples, n_classes)
```

#### Supported Formats
- **NPZ**: NumPy compressed arrays with 'data' and 'labels' keys
- **NPY**: NumPy arrays (labels will be auto-generated)
- **CSV**: Comma-separated values (time×channels format)
- **MAT**: MATLAB files with data variables
- **H5/HDF5**: HDF5 files with datasets

### Data Specifications

#### EEG/Sensor Data
- **Shape**: `(n_samples, n_channels, n_timesteps)`
- **Type**: `float32` or `float64`
- **Range**: Preprocessed and normalized signals
- **Channels**: 8-128 channels supported
- **Sampling Rate**: 125-1000 Hz recommended

#### Action Labels
- **Shape**: `(n_samples,)` for single labels or `(n_samples, n_classes)` for multi-labels
- **Type**: `int64` for classification
- **Classes**: 2-10 action classes supported

### Example Data Preparation

```python
import numpy as np
from brain2rl.utils.data_utils import preprocess_sensor_data, save_processed_data

# Load your raw sensor data
raw_data = np.load('raw_eeg.npy')  # Shape: (time, channels)
action_labels = np.load('actions.npy')  # Shape: (n_trials,)

# Preprocess data
processed_data = preprocess_sensor_data(
    raw_data.T,  # Convert to (channels, time)
    sampling_rate=250.0,
    apply_filters=True,
    normalize=True,
    remove_artifacts=True
)

# Create windowed samples
window_size = 512  # 2 seconds at 250 Hz
n_samples = len(processed_data) // window_size

windowed_data = np.array([
    processed_data[:, i*window_size:(i+1)*window_size] 
    for i in range(n_samples)
])

# Save in pipeline format
save_processed_data(
    windowed_data, 
    action_labels[:n_samples], 
    'processed_sensor_data.npz',
    metadata={'sampling_rate': 250.0, 'window_size': 512}
)
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

#### Memory Management
```bash
# For large datasets, reduce batch size
python brain2rl/cli.py classification --batch-size 16

# Use gradient accumulation for effective large batch training
python brain2rl/cli.py rl-training --batch-size 32 --update-frequency 4
```


```bash
# Clone repository
git clone https://github.com/your-org/brain2rl.git
cd brain2rl

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