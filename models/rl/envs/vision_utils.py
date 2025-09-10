import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CupDetector:
    """Computer vision-based cup detector for the OpenArm environment"""
    
    def __init__(self, camera_size: Tuple[int, int] = (256, 256),
                 cup_geom_ids: Optional[np.ndarray] = None):
        self.camera_size = camera_size
        self.height, self.width = camera_size
        self.cup_geom_ids = (None if cup_geom_ids is None
                             else np.asarray(cup_geom_ids, dtype=np.int32))
        # Color ranges for different cups (HSV color space)
        self.cup_colors = {
            'cup1': {  # Brown cup
                'lower': np.array([10, 50, 50]),
                'upper': np.array([20, 255, 255])
            }
        }
        
        # Cup detection parameters
        self.min_contour_area = 100
        self.max_contour_area = 5000
        
    def detect_cups(self, image: np.ndarray, seg: np.ndarray | None = None) -> Dict[str, Dict]:
        """
        Detect cups in the camera image using color-based segmentation
        
        Args:
            image: RGB image from camera (H, W, 3)
            
        Returns:
            Dictionary with detected cup information
        """
        if image is None:
            return {}
        
        if seg is not None and self.cup_geom_ids is not None and self.cup_geom_ids.size > 0:
            if seg.ndim == 3:  # sometimes HxWx1
                seg = seg[..., 0]
            mask = np.isin(seg, self.cup_geom_ids)
            if not np.any(mask):
                return {}
            mask_u8 = (mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            det = {}
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < self.min_contour_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2
                nx, ny = cx / self.width, cy / self.height
                det[f"cup{i+1}"] = {
                    "center": (int(cx), int(cy)),
                    "normalized_center": (float(nx), float(ny)),
                    "area": float(area),
                    "distance_from_center": float(np.hypot(nx - 0.5, ny - 0.5)),
                    "bbox": (int(x), int(y), int(w), int(h)),
                }
            return det
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        detected = {}
        ranges = {
            'cup1': {'lower': np.array([  5,  30,  20]), 'upper': np.array([30, 255, 255])},  # brownish
            'cup2': {'lower': np.array([ 35,  30,  20]), 'upper': np.array([85, 255, 255])},  # green
            'cup3': {'lower': np.array([ 95,  30,  20]), 'upper': np.array([135,255, 255])},  # blue
        }
        
        for name, cr in ranges.items():
            mask = cv2.inRange(hsv, cr['lower'], cr['upper'])
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5,5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if not (self.min_contour_area <= area <= self.max_contour_area):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            nx, ny = cx / self.width, cy / self.height
            detected[name] = {
                "center": (int(cx), int(cy)),
                "normalized_center": (float(nx), float(ny)),
                "area": float(area),
                "distance_from_center": float(np.hypot(nx - 0.5, ny - 0.5)),
                "bbox": (int(x), int(y), int(w), int(h)),
            }
        return detected
    
    def calculate_vision_reward(self, image: np.ndarray, target_cup: str = 'cup1', seg: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
        detected_cups = self.detect_cups(image, seg=seg)
        
        if not detected_cups:
            # No cups detected - negative reward
            return -1.0, {'detected_cups': 0, 'target_visible': False}
        
        # Check if target cup is visible
        target_visible = target_cup in detected_cups
        
        if not target_visible:
            # Target cup not visible - small negative reward
            best_name = min(detected_cups, key=lambda k: detected_cups[k]['distance_from_center'])
            target_cup = best_name
            target_visible = True
        
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
    
    def __init__(self, vision_weight: float = 0.4, physics_weight: float = 0.6,
                 cup_geom_ids: Optional[np.ndarray] = None, camera_size: Tuple[int,int]=(256,256)):
        self.vision_weight = vision_weight
        self.physics_weight = physics_weight
        self.cup_detector = CupDetector(camera_size=camera_size, cup_geom_ids=cup_geom_ids)
        
    def calculate_total_reward(self, image: np.ndarray, ee_pos: np.ndarray, cup_pos: np.ndarray,
                               target_cup: str = 'cup1', seg: Optional[np.ndarray] = None) -> Tuple[float, Dict]:
        dist = float(np.linalg.norm(ee_pos - cup_pos))
        physics_reward = -dist + (0.1 * (0.2 - dist) if dist < 0.2 else 0.0)
        vision_reward, vision_info = self.cup_detector.calculate_vision_reward(image, target_cup, seg=seg)
        total_reward = self.physics_weight * physics_reward + self.vision_weight * vision_reward
        if dist < 0.03:
            total_reward += 10.0
        info = {"physics_reward": physics_reward, "vision_reward": vision_reward,
                "total_reward": total_reward, "distance": dist, "vision_info": vision_info}
        return total_reward, info
