# rewards_util.py
import numpy as np
from typing import Optional

def cal_distance_cup_ee(ee: np.ndarray, cup: np.ndarray) -> float:
    return float(np.linalg.norm(ee - cup))

def cal_distance_ee2goal(ee: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(ee - goal))

def cal_distance_cup2goal(cup: np.ndarray, goal: np.ndarray) -> float:
    return float(np.linalg.norm(cup - goal))

def is_coming(prev_dist: float, curr_dist: float, tol: float = 1e-6) -> bool:
    """True if distance decreased (moving toward)."""
    return (prev_dist - curr_dist) > tol

def is_moving(prev_pos: np.ndarray, curr_pos: np.ndarray, min_speed: float = 1e-4) -> bool:
    """True if object moved more than a tiny threshold."""
    return float(np.linalg.norm(curr_pos - prev_pos)) > min_speed

def is_holding(grasping_flag: bool) -> bool:
    """Alias to reflect your pipeline semantics."""
    return bool(grasping_flag)

def is_grab(num_contacts_with_object: int, aperture: float, min_close: float = 0.002, max_close: float = 0.020) -> bool:
    """Simple grasp heuristic: at least two contacts and aperture in a closing band."""
    return (num_contacts_with_object >= 2) and (min_close < aperture < max_close)
