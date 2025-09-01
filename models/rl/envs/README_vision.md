# OpenArm Computer Vision System

This directory contains the computer vision-enhanced OpenArm environment for reinforcement learning with visual perception.

## Overview

The system integrates computer vision capabilities with the OpenArm Mujoco simulation to:
- Detect multiple cups using color-based segmentation
- Calculate rewards based on both visual perception and physics
- Provide rich visual observations for the RL agent
- Enable more sophisticated task completion

## Files

- `openarm_scene_with_cameras.xml` - Custom scene file with wrist-mounted cameras and multiple cups
- `vision_utils.py` - Computer vision utilities for cup detection and reward calculation
- `openarm_mj_env.py` - Enhanced environment with computer vision integration
- `test_vision.py` - Test script to verify the vision system works
- `requirements_vision.txt` - Dependencies for the computer vision system

## Key Features

### 1. Multiple Cameras
- **Wrist-mounted camera**: Provides first-person view from the robot's perspective
- **Multiple cup detection**: Identifies 3 different colored cups (brown, green, blue)
- **Real-time processing**: Processes camera feed at each timestep

### 2. Computer Vision Pipeline
- **Color segmentation**: Uses HSV color space for robust cup detection
- **Contour analysis**: Identifies cup boundaries and calculates properties
- **Feature extraction**: Provides 6 vision features to the RL agent:
  - Number of detected cups (normalized)
  - Target cup visibility
  - Target cup distance from image center
  - Target cup size (normalized)
  - Target cup position (x, y normalized)

### 3. Enhanced Reward System
- **Physics-based rewards**: Distance-based rewards for arm positioning
- **Vision-based rewards**: Rewards for proper cup detection and positioning
- **Combined rewards**: Weighted combination of physics and vision rewards
- **Task completion**: Large bonus for successful cup reaching

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements_vision.txt
```

### 2. Test the Vision System
```bash
cd models/rl/envs
python test_vision.py
```

### 3. Run RL Training
```bash
cd models/rl/launch
python try_mj.py --algo reppo --sim mujoco --render
```

### 4. Customize the Environment
```python
from models.rl.envs.openarm_mj_env import OpenArmMjEnv

env = OpenArmMjEnv(
    xml_path="path/to/scene.xml",
    camera='left_wrist_cam',
    camera_size=(256, 256),
    vision_reward_weight=0.4,    # 40% vision, 60% physics
    physics_reward_weight=0.6,
    target_cup='cup1'            # Target the brown cup
)
```

## Camera Configuration

The wrist-mounted camera provides:
- **Field of view**: 60 degrees
- **Resolution**: 256x256 pixels
- **Position**: Attached to the left arm wrist
- **Orientation**: Forward-facing for cup detection

## Cup Detection

### Color Ranges (HSV)
- **Cup1 (Brown)**: H: 10-20, S: 50-255, V: 50-255
- **Cup2 (Green)**: H: 40-80, S: 50-255, V: 50-255  
- **Cup3 (Blue)**: H: 100-130, S: 50-255, V: 50-255

### Detection Parameters
- **Minimum contour area**: 100 pixels
- **Maximum contour area**: 5000 pixels
- **Morphological operations**: Closing and opening for noise reduction

## Reward Structure

### Physics Rewards (60% weight)
- **Distance penalty**: Negative reward proportional to distance to cup
- **Proximity bonus**: Additional reward when close to cup
- **Completion bonus**: Large reward (10.0) when very close (< 3cm)

### Vision Rewards (40% weight)
- **Visibility reward**: Reward for target cup being visible
- **Center alignment**: Reward for cup being in center of view
- **Size optimization**: Reward for appropriate viewing distance

## Observation Space

The observation space now includes:
- **Joint positions** (qpos)
- **Joint velocities** (qvel) 
- **End effector position** (3D)
- **Target cup position** (3D)
- **Vision features** (6D):
  - Cup count, visibility, center distance, size, x_pos, y_pos

## Future Enhancements

1. **LLM Integration**: Natural language task descriptions
2. **Multi-objective tasks**: Complex manipulation sequences
3. **Adaptive rewards**: Dynamic reward shaping based on task progress
4. **Advanced vision**: Object pose estimation, depth perception
5. **Multi-camera fusion**: Combining multiple camera views

## Troubleshooting

### Common Issues
1. **No cups detected**: Check lighting conditions and color calibration
2. **Camera not found**: Verify camera name in XML file
3. **Poor detection**: Adjust HSV color ranges in `vision_utils.py`
4. **Performance issues**: Reduce camera resolution or processing frequency

### Debug Mode
The system saves debug images:
- `camera_view_reset_*.png`: Camera feed at reset
- `detection_vis_reset_*.png`: Detection visualization at reset
- `camera_view_step_*.png`: Camera feed during training (every 50 steps)
- `detection_vis_step_*.png`: Detection visualization during training

## Performance Considerations

- **Real-time processing**: Optimized for 256x256 resolution
- **Memory usage**: Efficient image processing with minimal overhead
- **Training stability**: Balanced rewards prevent reward sparsity
- **Scalability**: Modular design allows easy addition of new vision features

