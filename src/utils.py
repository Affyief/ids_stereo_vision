"""
Utility functions for stereo vision system.
"""

import yaml
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
import os

logger = logging.getLogger(__name__)


def load_yaml_config(config_path: str) -> Optional[Dict[str, Any]]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML file
        
    Returns:
        Dictionary with configuration, or None if loading failed
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return None


def save_yaml_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Saved configuration to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        return False


def convert_numpy_to_list(obj):
    """
    Convert numpy arrays to lists for YAML serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Converted object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj


def load_stereo_calibration(calib_path: str) -> Optional[Dict[str, Any]]:
    """
    Load stereo calibration parameters.
    
    Args:
        calib_path: Path to calibration YAML file
        
    Returns:
        Dictionary with calibration parameters, or None if loading failed
    """
    config = load_yaml_config(calib_path)
    if config is None:
        return None
    
    # Convert lists back to numpy arrays
    if config.get('left_camera', {}).get('camera_matrix') is not None:
        config['left_camera']['camera_matrix'] = np.array(
            config['left_camera']['camera_matrix']
        )
        config['left_camera']['distortion_coefficients'] = np.array(
            config['left_camera']['distortion_coefficients']
        )
    
    if config.get('right_camera', {}).get('camera_matrix') is not None:
        config['right_camera']['camera_matrix'] = np.array(
            config['right_camera']['camera_matrix']
        )
        config['right_camera']['distortion_coefficients'] = np.array(
            config['right_camera']['distortion_coefficients']
        )
    
    stereo = config.get('stereo', {})
    if stereo.get('rotation_matrix') is not None:
        stereo['rotation_matrix'] = np.array(stereo['rotation_matrix'])
        stereo['translation_vector'] = np.array(stereo['translation_vector'])
        stereo['essential_matrix'] = np.array(stereo['essential_matrix'])
        stereo['fundamental_matrix'] = np.array(stereo['fundamental_matrix'])
        stereo['rectification_left'] = np.array(stereo['rectification_left'])
        stereo['rectification_right'] = np.array(stereo['rectification_right'])
        stereo['projection_left'] = np.array(stereo['projection_left'])
        stereo['projection_right'] = np.array(stereo['projection_right'])
        stereo['disparity_to_depth_matrix'] = np.array(stereo['disparity_to_depth_matrix'])
    
    return config


def save_stereo_calibration(calib_params: Dict[str, Any], calib_path: str) -> bool:
    """
    Save stereo calibration parameters.
    
    Args:
        calib_params: Calibration parameters dictionary
        calib_path: Path to save calibration YAML file
        
    Returns:
        True if successful, False otherwise
    """
    # Convert numpy arrays to lists
    calib_to_save = convert_numpy_to_list(calib_params)
    return save_yaml_config(calib_to_save, calib_path)


def ensure_directory(path: str) -> bool:
    """
    Ensure directory exists, create if not.
    
    Args:
        path: Directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


def calculate_depth_from_disparity(
    disparity: np.ndarray,
    baseline: float,
    focal_length: float
) -> np.ndarray:
    """
    Calculate depth map from disparity map.
    
    Args:
        disparity: Disparity map in pixels
        baseline: Distance between cameras in mm
        focal_length: Focal length in pixels
        
    Returns:
        Depth map in mm
    """
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (baseline * focal_length) / disparity
        depth[disparity <= 0] = 0  # Invalid disparities
        depth[~np.isfinite(depth)] = 0  # Handle inf and nan
    
    return depth


def create_depth_colormap(
    depth: np.ndarray,
    min_depth: float = 100,
    max_depth: float = 3000,
    colormap: int = None
) -> np.ndarray:
    """
    Create colored depth map for visualization.
    
    Args:
        depth: Depth map in mm
        min_depth: Minimum depth for color scaling
        max_depth: Maximum depth for color scaling
        colormap: OpenCV colormap (default: cv2.COLORMAP_JET)
        
    Returns:
        Colored depth map (BGR)
    """
    import cv2
    
    if colormap is None:
        colormap = cv2.COLORMAP_JET
    
    # Normalize depth to 0-255 range
    depth_normalized = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_uint8, colormap)
    
    # Set invalid depths (0) to black
    mask = depth <= 0
    depth_colored[mask] = [0, 0, 0]
    
    return depth_colored


def compute_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvecs: list,
    tvecs: list
) -> float:
    """
    Compute mean reprojection error for calibration.
    
    Args:
        object_points: 3D points in world coordinates
        image_points: 2D points in image coordinates
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
        
    Returns:
        Mean reprojection error in pixels
    """
    import cv2
    
    total_error = 0
    total_points = 0
    
    for i in range(len(object_points)):
        projected_points, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )
        error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2)
        total_error += error
        total_points += len(object_points[i])
    
    mean_error = total_error / total_points
    return mean_error


def draw_distance_overlay(
    image: np.ndarray,
    depth: np.ndarray,
    num_points: int = 9
) -> np.ndarray:
    """
    Draw distance measurements overlay on image.
    
    Args:
        image: Input image (BGR)
        depth: Depth map in mm
        num_points: Number of measurement points (default: 9 for 3x3 grid)
        
    Returns:
        Image with distance overlay
    """
    import cv2
    
    overlay = image.copy()
    h, w = depth.shape[:2]
    
    # Calculate grid points
    grid_size = int(np.sqrt(num_points))
    step_y = h // (grid_size + 1)
    step_x = w // (grid_size + 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    for i in range(1, grid_size + 1):
        for j in range(1, grid_size + 1):
            y = i * step_y
            x = j * step_x
            
            # Get depth value at this point
            if 0 <= y < h and 0 <= x < w:
                depth_value = depth[y, x]
                
                if depth_value > 0:
                    # Draw circle
                    cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
                    
                    # Draw distance text
                    if depth_value < 1000:
                        text = f"{depth_value:.0f}mm"
                    else:
                        text = f"{depth_value/1000:.2f}m"
                    
                    # Background for text
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    cv2.rectangle(
                        overlay,
                        (x - text_size[0]//2 - 5, y - text_size[1] - 10),
                        (x + text_size[0]//2 + 5, y - 5),
                        (0, 0, 0),
                        -1
                    )
                    
                    # Draw text
                    cv2.putText(
                        overlay,
                        text,
                        (x - text_size[0]//2, y - 8),
                        font,
                        font_scale,
                        (0, 255, 0),
                        thickness
                    )
    
    return overlay


def get_distance_at_point(depth: np.ndarray, x: int, y: int, window_size: int = 5) -> float:
    """
    Get distance at a specific point with averaging.
    
    Args:
        depth: Depth map in mm
        x: X coordinate
        y: Y coordinate
        window_size: Size of averaging window
        
    Returns:
        Average distance in mm, or 0 if invalid
    """
    h, w = depth.shape[:2]
    
    # Define window bounds
    y_min = max(0, y - window_size // 2)
    y_max = min(h, y + window_size // 2 + 1)
    x_min = max(0, x - window_size // 2)
    x_max = min(w, x + window_size // 2 + 1)
    
    # Get window
    window = depth[y_min:y_max, x_min:x_max]
    
    # Filter out invalid depths
    valid_depths = window[window > 0]
    
    if len(valid_depths) > 0:
        return np.median(valid_depths)
    else:
        return 0.0
