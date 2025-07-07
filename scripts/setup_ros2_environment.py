#!/usr/bin/env python3
"""
Setup script for ROS2 Humble environment on Windows
This script configures the Python environment to work with ROS2 Humble installed in C:\dev\ros2_humble
"""

import os
import sys
import subprocess
import platform

def setup_ros2_environment():
    """Configure ROS2 environment variables and Python paths"""
    
    # ROS2 installation path
    ros2_root = r"C:\dev\ros2_humble"
    
    if not os.path.exists(ros2_root):
        print(f"ERROR: ROS2 installation not found at {ros2_root}")
        print("Please ensure ROS2 Humble is properly installed.")
        return False
    
    print("Setting up ROS2 Humble environment...")
    
    # Set ROS2 environment variables
    os.environ['ROS_VERSION'] = '2'
    os.environ['ROS_PYTHON_VERSION'] = '3'
    os.environ['ROS_DISTRO'] = 'humble'
    
    # Add ROS2 paths to environment
    if platform.system() == "Windows":
        # Windows-specific paths
        ros2_bin = os.path.join(ros2_root, "bin")
        ros2_lib = os.path.join(ros2_root, "lib")
        ros2_share = os.path.join(ros2_root, "share")
        
        # Add to PATH
        current_path = os.environ.get('PATH', '')
        if ros2_bin not in current_path:
            os.environ['PATH'] = ros2_bin + os.pathsep + current_path
        
        # Add to Python path
        ros2_python_path = os.path.join(ros2_root, "Lib", "site-packages")
        if os.path.exists(ros2_python_path):
            if ros2_python_path not in sys.path:
                sys.path.insert(0, ros2_python_path)
        
        # Set additional ROS2 environment variables
        os.environ['AMENT_PREFIX_PATH'] = ros2_root
        os.environ['CMAKE_PREFIX_PATH'] = ros2_root
        
        print(f"✓ Added {ros2_bin} to PATH")
        print(f"✓ Added {ros2_python_path} to Python path")
    
    # Test ROS2 import
    try:
        import rclpy
        print("✓ Successfully imported rclpy")
        return True
    except ImportError as e:
        print(f"✗ Failed to import rclpy: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure ROS2 Humble is properly installed")
        print("2. Try running this from a ROS2 command prompt")
        print("3. Check if the Python version matches ROS2's Python version")
        return False

def create_ros2_batch_script():
    """Create a batch script to source ROS2 environment"""
    batch_content = f"""@echo off
call "C:\\dev\\ros2_humble\\local_setup.bat"
set PYTHONPATH=C:\\dev\\ros2_humble\\Lib\\site-packages;%PYTHONPATH%
echo ROS2 Humble environment loaded
python %*
"""
    
    script_path = os.path.join(os.path.dirname(__file__), "run_with_ros2.bat")
    with open(script_path, 'w') as f:
        f.write(batch_content)
    
    print(f"✓ Created ROS2 batch script: {script_path}")
    print("Usage: run_with_ros2.bat your_python_script.py")

def main():
    """Main setup function"""
    print("ROS2 Humble Environment Setup")
    print("=" * 40)
    
    success = setup_ros2_environment()
    
    if success:
        print("\n✓ ROS2 environment setup completed successfully!")
        print("\nTo use ROS2 in your Python scripts:")
        print("1. Run this setup script before importing ROS2 modules")
        print("2. Or use the generated batch script for convenience")
    else:
        print("\n✗ ROS2 environment setup failed!")
        print("Please check the troubleshooting steps above.")
    
    create_ros2_batch_script()
    
    return success

if __name__ == "__main__":
    main() 