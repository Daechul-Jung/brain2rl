import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CupDetector:
    """Computer vision-based cup detector for the OpenArm environment"""
    
    def __init__(self, camera_size: Tuple[int, int] = (256, 256)):
        self.camera_size = camera_size
        self.height, self.width = camera_size
        
        # Color ranges for different cups (HSV color space)
        self.cup_colors = {
            'cup1': {  # Brown cup
                'lower': np.array([10, 50, 50]),
                'upper': np.array([20, 255, 255])
            },
            'cup2': {  # Green cup
                'lower': np.array([40, 50, 50]),
                'upper': np.array([80, 255, 255])
            },
            'cup3': {  # Blue cup
                'lower': np.array([100, 50, 50]),
                'upper': np.array([130, 255, 255])
            }
        }
        
        # Cup detection parameters
        self.min_contour_area = 100
        self.max_contour_area = 5000
        
    def detect_cups(self, image: np.ndarray) -> Dict[str, Dict]:
        """
        Detect cups in the camera image using color-based segmentation
        
        Args:
            image: RGB image from camera (H, W, 3)
            
        Returns:
            Dictionary with detected cup information
        """
        if image is None:
            return {}
            
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        detected_cups = {}
        
        for cup_name, color_range in self.cup_colors.items():
            # Create mask for this color range
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find the largest contour (most likely the cup)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if self.min_contour_area < area < self.max_contour_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Calculate center point
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Calculate normalized position (0-1)
                    norm_x = center_x / self.width
                    norm_y = center_y / self.height
                    
                    # Calculate distance from image center
                    center_distance = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
                    
                    detected_cups[cup_name] = {
                        'center': (center_x, center_y),
                        'normalized_center': (norm_x, norm_y),
                        'area': area,
                        'distance_from_center': center_distance,
                        'bbox': (x, y, w, h)
                    }
                    
                    logger.debug(f"Detected {cup_name} at ({center_x}, {center_y}) with area {area}")
        
        return detected_cups
    
    def calculate_vision_reward(self, image: np.ndarray, target_cup: str = 'cup1') -> Tuple[float, Dict]:
        """
        Calculate reward based on computer vision analysis
        
        Args:
            image: RGB image from camera
            target_cup: Name of the target cup to focus on
            
        Returns:
            Tuple of (reward, info_dict)
        """
        detected_cups = self.detect_cups(image)
        
        if not detected_cups:
            # No cups detected - negative reward
            return -1.0, {'detected_cups': 0, 'target_visible': False}
        
        # Check if target cup is visible
        target_visible = target_cup in detected_cups
        
        if not target_visible:
            # Target cup not visible - small negative reward
            return -0.5, {'detected_cups': len(detected_cups), 'target_visible': False}
        
        # Target cup is visible - calculate reward based on position
        target_info = detected_cups[target_cup]
        
        # Reward for being in center of view (closer to center = higher reward)
        center_reward = 1.0 - target_info['distance_from_center']
        
        # Reward for appropriate size (not too close, not too far)
        area = target_info['area']
        size_reward = 0.0
        if 200 < area < 3000:  # Good viewing distance
            size_reward = 1.0
        elif area < 200:  # Too far
            size_reward = 0.5
        else:  # Too close
            size_reward = 0.3
        
        # Combine rewards
        total_reward = 0.6 * center_reward + 0.4 * size_reward
        
        info = {
            'detected_cups': len(detected_cups),
            'target_visible': True,
            'target_center_distance': target_info['distance_from_center'],
            'target_area': area,
            'center_reward': center_reward,
            'size_reward': size_reward,
            'total_reward': total_reward
        }
        
        return total_reward, info
    
    def visualize_detection(self, image: np.ndarray, detected_cups: Dict) -> np.ndarray:
        """
        Draw detection results on the image for debugging
        
        Args:
            image: Original RGB image
            detected_cups: Dictionary of detected cups
            
        Returns:
            Image with detection visualization
        """
        vis_image = image.copy()
        
        for cup_name, cup_info in detected_cups.items():
            x, y, w, h = cup_info['bbox']
            center_x, center_y = cup_info['center']
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_image, (center_x, center_y), 3, (255, 0, 0), -1)
            
            # Draw label
            cv2.putText(vis_image, cup_name, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image


class RewardCalculator:
    """Calculate comprehensive rewards combining vision and physics"""
    
    def __init__(self, vision_weight: float = 0.4, physics_weight: float = 0.6):
        self.vision_weight = vision_weight
        self.physics_weight = physics_weight
        self.cup_detector = CupDetector()
        
    def calculate_total_reward(self, 
                             image: np.ndarray, 
                             ee_pos: np.ndarray, 
                             cup_pos: np.ndarray,
                             target_cup: str = 'cup1') -> Tuple[float, Dict]:
        """
        Calculate total reward combining computer vision and physics-based rewards
        
        Args:
            image: Camera image
            ee_pos: End effector position
            cup_pos: Cup position
            target_cup: Target cup name
            
        Returns:
            Tuple of (total_reward, info_dict)
        """
        # Physics-based reward (distance to cup)
        dist = np.linalg.norm(ee_pos - cup_pos)
        physics_reward = -dist  # Closer is better
        
        # Add shaping reward for being close
        if dist < 0.2:
            physics_reward += 0.1 * (0.2 - dist)
        
        # Vision-based reward
        vision_reward, vision_info = self.cup_detector.calculate_vision_reward(image, target_cup)
        
        # Combine rewards
        total_reward = (self.physics_weight * physics_reward + 
                       self.vision_weight * vision_reward)
        
        # Additional reward for successful task completion
        if dist < 0.03:  # Very close to cup
            total_reward += 10.0  # Large completion bonus
        
        info = {
            'physics_reward': physics_reward,
            'vision_reward': vision_reward,
            'total_reward': total_reward,
            'distance': dist,
            'vision_info': vision_info
        }
        
        return total_reward, info
