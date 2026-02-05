"""
Utility functions for the IDS Stereo Vision System.

This module provides configuration loading, logging setup, and helper functions
for the stereo vision system.
"""

import os
import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import yaml
import numpy as np
import cv2


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("ids_stereo_vision")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    # Assuming this file is in src/utils.py
    return Path(__file__).parent.parent


def load_calibration_data(
    calibration_path: str
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load calibration data from .npz file.
    
    Args:
        calibration_path: Path to calibration .npz file
        
    Returns:
        Dictionary containing calibration parameters or None if file doesn't exist
    """
    if not os.path.exists(calibration_path):
        return None
    
    data = np.load(calibration_path)
    return {key: data[key] for key in data.files}


def save_calibration_data(
    calibration_path: str,
    **kwargs: np.ndarray
) -> None:
    """
    Save calibration data to .npz file.
    
    Args:
        calibration_path: Path to save calibration data
        **kwargs: Calibration parameters to save
    """
    os.makedirs(os.path.dirname(calibration_path), exist_ok=True)
    np.savez(calibration_path, **kwargs)


def calculate_reprojection_error(
    obj_points: np.ndarray,
    img_points: np.ndarray,
    rvecs: np.ndarray,
    tvecs: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> float:
    """
    Calculate mean reprojection error for calibration validation.
    
    Args:
        obj_points: 3D object points
        img_points: 2D image points
        rvecs: Rotation vectors
        tvecs: Translation vectors
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        
    Returns:
        Mean reprojection error in pixels
    """
    total_error = 0
    total_points = 0
    
    for i in range(len(obj_points)):
        img_points_projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(img_points[i], img_points_projected, cv2.NORM_L2)
        total_error += error
        total_points += len(obj_points[i])
    
    return total_error / total_points


def estimate_camera_matrix(
    image_width: int,
    image_height: int,
    focal_length_mm: float = 8.0,
    sensor_width_mm: float = 5.702,
    sensor_height_mm: float = 4.277
) -> np.ndarray:
    """
    Estimate camera intrinsic matrix based on focal length.
    
    For IDS U3-3680XCP-C:
    - Sensor size: 5.702 mm x 4.277 mm (1/2.5" format)
    - Resolution: 2592 x 1944 pixels
    
    Args:
        image_width: Image width in pixels
        image_height: Image height in pixels
        focal_length_mm: Lens focal length in millimeters
        sensor_width_mm: Sensor width in millimeters
        sensor_height_mm: Sensor height in millimeters
        
    Returns:
        Estimated 3x3 camera matrix
    """
    # Calculate focal length in pixels
    fx = focal_length_mm * image_width / sensor_width_mm
    fy = focal_length_mm * image_height / sensor_height_mm
    
    # Principal point (image center)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    return camera_matrix


def estimate_camera_matrix_from_lens(focal_length_mm, sensor_width_mm, sensor_height_mm, 
                                     image_width_px, image_height_px, pixel_size_um=2.2):
    """
    Estimate camera intrinsic matrix from lens and sensor specifications.
    
    Args:
        focal_length_mm: Lens focal length in millimeters
        sensor_width_mm: Sensor width in millimeters
        sensor_height_mm: Sensor height in millimeters
        image_width_px: Image width in pixels
        image_height_px: Image height in pixels
        pixel_size_um: Pixel size in micrometers (default: 2.2 for AR0521)
    
    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
        fov: dict with horizontal, vertical, and diagonal FOV in degrees
    
    Example:
        For IDS U3-3680XCP-C with 6mm lens:
        >>> matrix, fov = estimate_camera_matrix_from_lens(
        ...     focal_length_mm=6.0,
        ...     sensor_width_mm=5.702,
        ...     sensor_height_mm=4.277,
        ...     image_width_px=2592,
        ...     image_height_px=1944,
        ...     pixel_size_um=2.2
        ... )
        >>> print(f"fx={matrix[0,0]:.1f}, fy={matrix[1,1]:.1f}")
        fx=2727.3, fy=2727.3
        >>> print(f"Horizontal FOV: {fov['horizontal']:.1f}°")
        Horizontal FOV: 50.7°
    """
    # Convert pixel size to mm
    pixel_size_mm = pixel_size_um / 1000.0
    
    # Calculate focal length in pixels
    # fx = focal_length_mm / pixel_size_mm
    fx = focal_length_mm / pixel_size_mm
    fy = focal_length_mm / pixel_size_mm
    
    # Principal point (image center)
    cx = image_width_px / 2.0
    cy = image_height_px / 2.0
    
    # Build camera matrix
    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float64)
    
    # Calculate field of view
    # FOV = 2 × arctan(sensor_size / (2 × focal_length))
    fov_h_rad = 2 * np.arctan(sensor_width_mm / (2 * focal_length_mm))
    fov_v_rad = 2 * np.arctan(sensor_height_mm / (2 * focal_length_mm))
    fov_d_rad = 2 * np.arctan(np.sqrt(sensor_width_mm**2 + sensor_height_mm**2) / (2 * focal_length_mm))
    
    fov = {
        'horizontal': np.degrees(fov_h_rad),
        'vertical': np.degrees(fov_v_rad),
        'diagonal': np.degrees(fov_d_rad)
    }
    
    return camera_matrix, fov


def load_lens_config(config_path='config/camera_config.yaml'):
    """
    Load lens configuration and compute optical parameters.
    
    Returns:
        dict with lens specs and computed optical parameters
    """
    config = load_config(config_path)
    
    lens_config = config.get('cameras', {}).get('lens', {})
    resolution = config.get('cameras', {}).get('resolution', {})
    
    if not lens_config:
        return None
    
    focal_length_mm = lens_config.get('focal_length_mm', 6.0)
    width_px = resolution.get('width', 2592)
    height_px = resolution.get('height', 1944)
    
    # AR0521 sensor specs
    sensor_width_mm = 5.702
    sensor_height_mm = 4.277
    pixel_size_um = 2.2
    
    camera_matrix, fov = estimate_camera_matrix_from_lens(
        focal_length_mm=focal_length_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
        image_width_px=width_px,
        image_height_px=height_px,
        pixel_size_um=pixel_size_um
    )
    
    return {
        'lens': lens_config,
        'camera_matrix': camera_matrix,
        'fov': fov,
        'fx': camera_matrix[0, 0],
        'fy': camera_matrix[1, 1],
        'cx': camera_matrix[0, 2],
        'cy': camera_matrix[1, 2]
    }


def calculate_depth_accuracy(baseline_mm, focal_length_pixels, disparity_pixels, depth_m):
    """
    Calculate depth measurement accuracy based on stereo geometry.
    
    Args:
        baseline_mm: Distance between cameras in millimeters
        focal_length_pixels: Focal length in pixels (fx or fy)
        disparity_pixels: Disparity resolution (typically 1/16 pixel for OpenCV)
        depth_m: Depth distance in meters
    
    Returns:
        Depth accuracy in millimeters
    
    Formula:
        ΔZ = (Z² × Δd) / (B × f)
        where:
        - Z = depth
        - Δd = disparity error
        - B = baseline
        - f = focal length
    """
    depth_mm = depth_m * 1000
    baseline_m = baseline_mm / 1000
    
    # OpenCV stereo has ~1/16 pixel disparity resolution
    disparity_error = disparity_pixels
    
    # Depth error
    depth_error_mm = (depth_mm ** 2 * disparity_error) / (baseline_mm * focal_length_pixels)
    
    return depth_error_mm


def disparity_to_depth(
    disparity: np.ndarray,
    focal_length_pixels: float,
    baseline_mm: float
) -> np.ndarray:
    """
    Convert disparity map to depth map.
    
    Uses the formula: Z = (f × B) / d
    Where:
    - Z = distance to object (depth) in mm
    - f = focal length in pixels
    - B = baseline (distance between cameras) in mm
    - d = disparity in pixels
    
    Args:
        disparity: Disparity map in pixels
        focal_length_pixels: Focal length in pixels
        baseline_mm: Baseline distance in millimeters
        
    Returns:
        Depth map in millimeters
    """
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = (focal_length_pixels * baseline_mm) / disparity
        depth[disparity <= 0] = 0  # Invalid disparities
        depth[~np.isfinite(depth)] = 0  # Infinite or NaN values
    
    return depth


def format_distance(distance_mm: float, max_distance_m: float = 10.0) -> str:
    """
    Format distance for display.
    
    Args:
        distance_mm: Distance in millimeters
        max_distance_m: Maximum valid distance in meters
        
    Returns:
        Formatted distance string
    """
    if distance_mm <= 0 or distance_mm > max_distance_m * 1000:
        return "N/A"
    
    if distance_mm < 1000:
        return f"{distance_mm:.1f} mm"
    else:
        return f"{distance_mm / 1000:.2f} m"


class PerformanceTimer:
    """Simple performance timer for profiling."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
    
    def get_fps(self) -> float:
        """Get frames per second based on elapsed time."""
        if self.elapsed > 0:
            return 1.0 / self.elapsed
        return 0.0


class FPSCounter:
    """Moving average FPS counter."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = []
        self.last_time = time.time()
    
    def update(self) -> float:
        """
        Update FPS counter with new frame.
        
        Returns:
            Current FPS estimate
        """
        current_time = time.time()
        frame_time = current_time - self.last_time
        self.last_time = current_time
        
        self.frame_times.append(frame_time)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            if avg_frame_time > 0:
                return 1.0 / avg_frame_time
        
        return 0.0


def save_image_with_timestamp(
    image: np.ndarray,
    prefix: str = "image",
    output_dir: str = "."
) -> str:
    """
    Save image with timestamp in filename.
    
    Args:
        image: Image to save
        prefix: Filename prefix
        output_dir: Output directory
        
    Returns:
        Path to saved image
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, image)
    return filepath


def print_camera_info(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: Tuple[int, int]
) -> None:
    """
    Print camera calibration information.
    
    Args:
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        image_size: Image size (width, height)
    """
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    print(f"Image size: {image_size[0]} x {image_size[1]}")
    print(f"Focal length: fx={fx:.2f}, fy={fy:.2f} pixels")
    print(f"Principal point: cx={cx:.2f}, cy={cy:.2f} pixels")
    print(f"Distortion coefficients: {dist_coeffs.ravel()}")
