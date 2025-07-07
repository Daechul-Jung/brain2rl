@echo off
call "C:\dev\ros2_humble\local_setup.bat"
set PYTHONPATH=C:\dev\ros2_humble\Lib\site-packages;%PYTHONPATH%
echo ROS2 Humble environment loaded
python %*
