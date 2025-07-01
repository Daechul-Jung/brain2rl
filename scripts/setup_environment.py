#!/usr/bin/env python
import os
import argparse
import sys
import platform
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Setup Environment for Brain to Agent's Actions")
    parser.add_argument('--cuda', action='store_true', 
                        help='Install CUDA-compatible PyTorch (requires NVIDIA GPU)')
    parser.add_argument('--ros2', action='store_true',
                        help='Setup ROS2 environment (requires separate ROS2 installation)')
    return parser.parse_args()

def check_system():
    """Check system compatibility"""
    print("Checking system...")
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("WARNING: Python 3.8+ is recommended for this project")
    
    # Check OS
    os_name = platform.system()
    print(f"Operating System: {os_name}")
    if os_name == "Windows":
        print("NOTE: Some ROS2 features may have limited functionality on Windows")
    
    # Check for CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"CUDA devices: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not installed, can't check CUDA availability")

def create_directories():
    """Create necessary directories if they don't exist"""
    print("\nChecking and creating directories...")
    
    required_dirs = [
        os.path.join('data', 'eeg'),
        os.path.join('data', 'fmri'),
        os.path.join('data', 'preprocessed'),
        os.path.join('models', 'classification'),
        os.path.join('models', 'tokenization'),
        os.path.join('models', 'rl'),
        'utils',
        'simulation',
        'notebooks',
        'scripts',
        'runs'  # For TensorBoard logs
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory exists: {directory}")

def install_dependencies(cuda=False):
    """Install required Python packages"""
    print("\nInstalling dependencies...")
    
    try:
        # Base dependencies from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Install PyTorch with CUDA if requested
        if cuda:
            # This URL would need to be adjusted for different CUDA versions
            # This command is just an example; the actual command may vary based on PyTorch version and CUDA version
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                 "torch", "torchvision", "torchaudio", "--index-url", 
                                 "https://download.pytorch.org/whl/cu118"])
            print("Installed PyTorch with CUDA support")
        
        print("Successfully installed dependencies")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")

def setup_ros2_env(ros2=False):
    """Setup ROS2 environment if requested"""
    if not ros2:
        return
    
    print("\nSetting up ROS2 environment...")
    
    # This would normally source the ROS2 setup script
    # For Windows, this might be different
    try:
        os_name = platform.system()
        if os_name == "Linux":
            # For Linux, we'd usually source setup.bash
            print("For Linux, you should source your ROS2 installation:")
            print("source /opt/ros/<ros2-distro>/setup.bash")
        elif os_name == "Windows":
            # For Windows, the setup is different
            print("For Windows, you should call setup.bat from your ROS2 installation:")
            print("call C:\\dev\\ros2\\setup.bat")
        
        print("NOTE: ROS2 should be installed separately following the official documentation")
        print("https://docs.ros.org/en/humble/Installation.html")
    except Exception as e:
        print(f"Error setting up ROS2 environment: {e}")

def main():
    args = parse_args()
    
    print("=" * 80)
    print("Brain to Agent's Actions - Environment Setup")
    print("=" * 80)
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    print(f"Project root: {project_root}")
    
    # Check system compatibility
    check_system()
    
    # Create necessary directories
    create_directories()
    
    # Install dependencies
    install_dependencies(args.cuda)
    
    # Setup ROS2 environment if requested
    setup_ros2_env(args.ros2)
    
    print("\nSetup completed!")
    print("\nNext steps:")
    print("1. Train a classifier:      python brain.py --mode train --model classification --data eeg")
    print("2. Train a tokenizer:       python brain.py --mode train --model tokenization --data eeg")
    print("3. Train an RL agent:       python brain.py --mode train --model rl --data eeg")
    print("4. Run a simulation:        python brain.py --mode simulate --data eeg")
    print("5. Explore the Jupyter notebook: notebooks/brain_to_agent_demo.ipynb")

if __name__ == "__main__":
    main() 