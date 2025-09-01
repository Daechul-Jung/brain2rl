#!/usr/bin/env python3
"""
Test script for the computer vision system in OpenArm environment
"""

import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.dirname(__file__))

from vision_utils import CupDetector, RewardCalculator

def create_test_image():
    """Create a synthetic test image with colored circles representing cups"""
    # Create a 256x256 RGB image
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Add background
    image[:] = [100, 100, 100]  # Gray background
    
    # Add cup1 (brown) - center
    cv2.circle(image, (128, 128), 30, [60, 30, 10], -1)  # Brown
    
    # Add cup2 (green) - top right
    cv2.circle(image, (200, 80), 25, [30, 60, 10], -1)   # Green
    
    # Add cup3 (blue) - bottom left
    cv2.circle(image, (60, 200), 35, [10, 30, 60], -1)   # Blue
    
    return image

def test_cup_detector():
    """Test the cup detection functionality"""
    print("Testing Cup Detector...")
    
    # Create test image
    test_image = create_test_image()
    
    # Initialize detector
    detector = CupDetector(camera_size=(256, 256))
    
    # Detect cups
    detected_cups = detector.detect_cups(test_image)
    
    print(f"Detected {len(detected_cups)} cups:")
    for cup_name, cup_info in detected_cups.items():
        print(f"  {cup_name}: center={cup_info['center']}, area={cup_info['area']:.1f}")
    
    # Test vision reward calculation
    reward, info = detector.calculate_vision_reward(test_image, target_cup='cup1')
    print(f"Vision reward for cup1: {reward:.3f}")
    print(f"Vision info: {info}")
    
    # Save test image
    cv2.imwrite("test_image.png", cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    
    # Save visualization
    if detected_cups:
        vis_image = detector.visualize_detection(test_image, detected_cups)
        cv2.imwrite("test_detection.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return len(detected_cups) == 3

def test_reward_calculator():
    """Test the reward calculation functionality"""
    print("\nTesting Reward Calculator...")
    
    # Create test image
    test_image = create_test_image()
    
    # Initialize calculator
    calculator = RewardCalculator(vision_weight=0.4, physics_weight=0.6)
    
    # Mock positions
    ee_pos = np.array([0.35, 0.0, 0.75])
    cup_pos = np.array([0.35, 0.0, 0.75])
    
    # Calculate total reward
    total_reward, reward_info = calculator.calculate_total_reward(
        test_image, ee_pos, cup_pos, target_cup='cup1'
    )
    
    print(f"Total reward: {total_reward:.3f}")
    print(f"Reward breakdown:")
    print(f"  Physics reward: {reward_info['physics_reward']:.3f}")
    print(f"  Vision reward: {reward_info['vision_reward']:.3f}")
    print(f"  Distance: {reward_info['distance']:.3f}")
    
    return total_reward > 0

def main():
    """Main test function"""
    print("=== OpenArm Computer Vision System Test ===\n")
    
    # Test cup detector
    detector_ok = test_cup_detector()
    
    # Test reward calculator
    calculator_ok = test_reward_calculator()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Cup Detector: {'✓ PASS' if detector_ok else '✗ FAIL'}")
    print(f"Reward Calculator: {'✓ PASS' if calculator_ok else '✗ FAIL'}")
    
    if detector_ok and calculator_ok:
        print("\n🎉 All tests passed! The computer vision system is working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
