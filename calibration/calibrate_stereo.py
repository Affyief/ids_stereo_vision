#!/usr/bin/env python3
"""
Stereo camera calibration script.

This script performs stereo calibration using pre-calibrated left and right cameras.
It computes the extrinsic parameters (rotation and translation) between cameras
and generates rectification parameters.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    setup_logging,
    load_config,
    load_calibration_data,
    save_calibration_data,
    get_project_root
)


def find_stereo_checkerboard_corners(
    left_images_dir: str,
    right_images_dir: str,
    checkerboard_size: tuple,
    square_size_mm: float
) -> tuple:
    """
    Find checkerboard corners in stereo image pairs.
    
    Args:
        left_images_dir: Directory with left camera images
        right_images_dir: Directory with right camera images
        checkerboard_size: Checkerboard size (rows, cols)
        square_size_mm: Size of checkerboard square in mm
        
    Returns:
        Tuple of (object_points, left_img_points, right_img_points, image_size)
    """
    logger = logging.getLogger(__name__)
    
    # Prepare object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm
    
    # Arrays to store points
    obj_points = []
    left_img_points = []
    right_img_points = []
    image_size = None
    
    # Get list of images (must have matching pairs)
    left_files = sorted([
        f for f in os.listdir(left_images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    right_files = sorted([
        f for f in os.listdir(right_images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if len(left_files) != len(right_files):
        logger.error("Number of left and right images don't match!")
        return None, None, None, None
    
    logger.info(f"Processing {len(left_files)} stereo pairs...")
    
    successful_pairs = 0
    
    for left_file, right_file in tqdm(zip(left_files, right_files), total=len(left_files)):
        # Load images
        left_img = cv2.imread(os.path.join(left_images_dir, left_file))
        right_img = cv2.imread(os.path.join(right_images_dir, right_file))
        
        if left_img is None or right_img is None:
            logger.warning(f"Failed to load pair: {left_file}, {right_file}")
            continue
        
        # Store image size
        if image_size is None:
            image_size = (left_img.shape[1], left_img.shape[0])
        
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Find corners in both images
        ret_left, corners_left = cv2.findChessboardCorners(
            left_gray,
            checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        ret_right, corners_right = cv2.findChessboardCorners(
            right_gray,
            checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # Only use pairs where both corners are found
        if ret_left and ret_right:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
            
            obj_points.append(objp)
            left_img_points.append(corners_left)
            right_img_points.append(corners_right)
            successful_pairs += 1
        else:
            logger.warning(f"Checkerboard not found in pair: {left_file}, {right_file}")
    
    logger.info(f"Successfully processed {successful_pairs}/{len(left_files)} stereo pairs")
    
    if successful_pairs < 10:
        logger.error("Not enough successful pairs for stereo calibration (minimum 10 recommended)")
        return None, None, None, None
    
    return obj_points, left_img_points, right_img_points, image_size


def stereo_calibrate(
    obj_points: list,
    left_img_points: list,
    right_img_points: list,
    left_camera_matrix: np.ndarray,
    left_dist_coeffs: np.ndarray,
    right_camera_matrix: np.ndarray,
    right_dist_coeffs: np.ndarray,
    image_size: tuple
) -> tuple:
    """
    Perform stereo calibration.
    
    Args:
        obj_points: 3D object points
        left_img_points: Left camera 2D points
        right_img_points: Right camera 2D points
        left_camera_matrix: Left camera matrix
        left_dist_coeffs: Left distortion coefficients
        right_camera_matrix: Right camera matrix
        right_dist_coeffs: Right distortion coefficients
        image_size: Image size (width, height)
        
    Returns:
        Tuple of stereo calibration parameters
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting stereo calibration...")
    
    # Stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        left_img_points,
        right_img_points,
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        image_size,
        criteria=criteria,
        flags=flags
    )
    
    logger.info(f"Stereo calibration RMS error: {ret:.4f} pixels")
    
    # Calculate baseline (distance between cameras)
    baseline_mm = np.linalg.norm(T)
    logger.info(f"Baseline distance: {baseline_mm:.2f} mm = {baseline_mm/10:.2f} cm")
    
    return ret, R, T, E, F


def stereo_rectify(
    left_camera_matrix: np.ndarray,
    left_dist_coeffs: np.ndarray,
    right_camera_matrix: np.ndarray,
    right_dist_coeffs: np.ndarray,
    image_size: tuple,
    R: np.ndarray,
    T: np.ndarray
) -> tuple:
    """
    Compute stereo rectification parameters.
    
    Args:
        left_camera_matrix: Left camera matrix
        left_dist_coeffs: Left distortion coefficients
        right_camera_matrix: Right camera matrix
        right_dist_coeffs: Right distortion coefficients
        image_size: Image size (width, height)
        R: Rotation matrix between cameras
        T: Translation vector between cameras
        
    Returns:
        Tuple of (R1, R2, P1, P2, Q, roi_left, roi_right)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Computing rectification parameters...")
    
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )
    
    logger.info("Rectification computed successfully")
    
    return R1, R2, P1, P2, Q, roi_left, roi_right


def main():
    parser = argparse.ArgumentParser(description='Calibrate stereo camera system')
    parser.add_argument(
        '--left-images',
        type=str,
        help='Directory with left camera calibration images'
    )
    parser.add_argument(
        '--right-images',
        type=str,
        help='Directory with right camera calibration images'
    )
    parser.add_argument(
        '--left-calibration',
        type=str,
        help='Left camera calibration file'
    )
    parser.add_argument(
        '--right-calibration',
        type=str,
        help='Right camera calibration file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/camera_config.yaml',
        help='Path to camera configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output stereo calibration file path'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Get project root
    project_root = get_project_root()
    
    # Load configuration
    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)
    
    # Get checkerboard parameters
    checkerboard_size = (
        config['calibration']['checkerboard']['rows'],
        config['calibration']['checkerboard']['cols']
    )
    square_size_mm = config['calibration']['checkerboard'].get('square_size_mm') or config['calibration']['checkerboard'].get('square_size', 25.0)
    
    # Determine paths
    left_images_dir = args.left_images or os.path.join(
        project_root, 'calibration', 'calibration_images', 'left'
    )
    right_images_dir = args.right_images or os.path.join(
        project_root, 'calibration', 'calibration_images', 'right'
    )
    
    left_cal_path = args.left_calibration or os.path.join(
        project_root, 'calibration_data', 'left_camera_calibration.npz'
    )
    right_cal_path = args.right_calibration or os.path.join(
        project_root, 'calibration_data', 'right_camera_calibration.npz'
    )
    
    output_path = args.output or os.path.join(
        project_root, 'calibration_data', 'stereo_calibration.npz'
    )
    
    # Check if calibration files exist
    if not os.path.exists(left_cal_path):
        logger.error(f"Left camera calibration not found: {left_cal_path}")
        logger.info("Please run calibrate_single_camera.py first for the left camera")
        return 1
    
    if not os.path.exists(right_cal_path):
        logger.error(f"Right camera calibration not found: {right_cal_path}")
        logger.info("Please run calibrate_single_camera.py first for the right camera")
        return 1
    
    # Load individual camera calibrations
    logger.info("Loading camera calibrations...")
    left_cal = load_calibration_data(left_cal_path)
    right_cal = load_calibration_data(right_cal_path)
    
    logger.info("Performing stereo calibration...")
    logger.info(f"Left images: {left_images_dir}")
    logger.info(f"Right images: {right_images_dir}")
    
    # Find checkerboard corners in stereo pairs
    obj_points, left_img_points, right_img_points, image_size = find_stereo_checkerboard_corners(
        left_images_dir,
        right_images_dir,
        checkerboard_size,
        square_size_mm
    )
    
    if obj_points is None:
        logger.error("Failed to find checkerboard corners in stereo pairs")
        return 1
    
    # Perform stereo calibration
    ret, R, T, E, F = stereo_calibrate(
        obj_points,
        left_img_points,
        right_img_points,
        left_cal['camera_matrix'],
        left_cal['dist_coeffs'],
        right_cal['camera_matrix'],
        right_cal['dist_coeffs'],
        image_size
    )
    
    # Compute rectification
    R1, R2, P1, P2, Q, roi_left, roi_right = stereo_rectify(
        left_cal['camera_matrix'],
        left_cal['dist_coeffs'],
        right_cal['camera_matrix'],
        right_cal['dist_coeffs'],
        image_size,
        R,
        T
    )
    
    # Calculate baseline
    baseline_mm = np.linalg.norm(T)
    
    # Print results
    print("\n" + "="*60)
    print("Stereo Calibration Results:")
    print("="*60)
    print(f"RMS error: {ret:.4f} pixels")
    print(f"Baseline: {baseline_mm:.2f} mm ({baseline_mm/10:.2f} cm)")
    print(f"\nRotation matrix R:\n{R}")
    print(f"\nTranslation vector T:\n{T.ravel()}")
    print("="*60 + "\n")
    
    # Save stereo calibration
    save_calibration_data(
        output_path,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        image_width=image_size[0],
        image_height=image_size[1],
        baseline_mm=baseline_mm,
        rms_error=ret
    )
    
    logger.info(f"Stereo calibration saved to: {output_path}")
    
    # Recommendation
    if ret < 1.0:
        logger.info("✓ Excellent stereo calibration! (error < 1.0 pixels)")
    elif ret < 2.0:
        logger.info("✓ Good stereo calibration (error < 2.0 pixels)")
    else:
        logger.warning("⚠ High stereo calibration error. Consider recapturing images.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
