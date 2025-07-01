# Brain to Agent's Actions

This project explores the intersection of Brain-Computer Interfaces (BCI), Deep Learning, and Reinforcement Learning (RL) to interpret brain signals and translate them into actions for robotic or computational agents.

## Project Structure

- **data/**: Contains datasets and preprocessing scripts
  - **eeg/**: EEG datasets (Grasp and Lift dataset)
  - **fmri/**: fMRI data (Human Connectome Project)
  - **preprocessed/**: Preprocessed signal data
  
- **models/**: Neural network architectures
  - **classification/**: Action classification models
  - **tokenization/**: Brain signal tokenization models
  - **rl/**: Reinforcement learning models
  
- **utils/**: Utility functions for data handling, visualization, etc.
  
- **simulation/**: ROS2 Gazebo simulation environment
  
- **notebooks/**: Jupyter notebooks for experiments and visualizations

- **scripts/**: Training and evaluation scripts

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- ROS2
- Gazebo
- NeuroPype
- NumPy, SciPy, etc.

## Setup

See `requirements.txt` for detailed dependencies.

```bash
pip install -r requirements.txt
```

## Usage

Instructions for training models, running simulations, and evaluating results will be added as the project progresses. 