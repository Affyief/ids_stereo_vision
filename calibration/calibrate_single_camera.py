#!/usr/bin/env python3
"""
Single camera calibration script.

This script calibrates a single camera using a checkerboard pattern.
It computes the camera's intrinsic parameters (camera matrix and distortion coefficients).
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
    save_calibration_data,
    calculate_reprojection_error,
    print_camera_info,
    get_project_root
)


def find_checkerboard_corners(
    images_dir: str,
    checkerboard_size: tuple,
    square_size_mm: float,
    show_corners: bool = False
) -> tuple:
    """
    Find checkerboard corners in calibration images.
    
    Args:
        images_dir: Directory containing calibration images
        checkerboard_size: Checkerboard size (rows, cols)
        square_size_mm: Size of checkerboard square in mm
        show_corners: Show detected corners for verification
        
    Returns:
        Tuple of (object_points, image_points, image_size)
    """
    logger = logging.getLogger(__name__)
    
    # Prepare object points (3D coordinates of checkerboard corners)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm
    
    # Arrays to store object points and image points
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane
    image_size = None
    
    # Get list of images
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    if not image_files:
        logger.error(f"No images found in {images_dir}")
        return None, None, None
    
    logger.info(f"Processing {len(image_files)} images...")
    
    successful_images = 0
    
    for filename in tqdm(image_files):
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            logger.warning(f"Failed to load image: {filename}")
            continue
        
        # Store image size
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            checkerboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            obj_points.append(objp)
            img_points.append(corners_refined)
            successful_images += 1
            
            # Optionally show corners
            if show_corners:
                img_with_corners = cv2.drawChessboardCorners(
                    img.copy(),
                    checkerboard_size,
                    corners_refined,
                    ret
                )
                cv2.imshow('Checkerboard Corners', img_with_corners)
                cv2.waitKey(500)
        else:
            logger.warning(f"Checkerboard not found in: {filename}")
    
    if show_corners:
        cv2.destroyAllWindows()
    
    logger.info(f"Successfully processed {successful_images}/{len(image_files)} images")
    
    if successful_images < 10:
        logger.error("Not enough successful images for calibration (minimum 10 recommended)")
        return None, None, None
    
    return obj_points, img_points, image_size


def calibrate_camera(
    obj_points: list,
    img_points: list,
    image_size: tuple,
    use_lens_init: bool = True
) -> tuple:
    """
    Calibrate camera using detected corners.
    
    Args:
        obj_points: 3D object points
        img_points: 2D image points
        image_size: Image size (width, height)
        use_lens_init: Use lens-based initial camera matrix estimate
        
    Returns:
        Tuple of (ret, camera_matrix, dist_coeffs, rvecs, tvecs)
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting camera calibration...")
    
    # Get initial camera matrix from lens specs if available
    camera_matrix_init = None
    flags = 0
    
    if use_lens_init:
        try:
            from src.utils import load_lens_config
            lens_config = load_lens_config()
            
            if lens_config:
                camera_matrix_init = lens_config['camera_matrix']
                flags = cv2.CALIB_USE_INTRINSIC_GUESS
                logger.info(f"Using lens-based initial camera matrix:")
                logger.info(f"  fx = {lens_config['fx']:.1f} pixels")
                logger.info(f"  fy = {lens_config['fy']:.1f} pixels")
                logger.info(f"  Field of View: {lens_config['fov']['horizontal']:.1f}° × {lens_config['fov']['vertical']:.1f}°")
        except Exception as e:
            logger.warning(f"Could not load lens config for initial guess: {e}")
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        camera_matrix_init,
        None,
        flags=flags
    )
    
    logger.info(f"Calibration RMS error: {ret:.4f} pixels")
    
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def main():
    parser = argparse.ArgumentParser(description='Calibrate a single camera')
    parser.add_argument(
        '--camera',
        choices=['left', 'right'],
        required=True,
        help='Which camera to calibrate'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        help='Directory containing calibration images (default: calibration/calibration_images/{camera})'
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
        help='Output calibration file path (default: calibration_data/{camera}_camera_calibration.npz)'
    )
    parser.add_argument(
        '--show-corners',
        action='store_true',
        help='Show detected corners for verification'
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
    square_size_mm = config['calibration']['checkerboard']['square_size']
    
    # Determine images directory
    if args.images_dir:
        images_dir = args.images_dir
    else:
        images_dir = os.path.join(
            project_root,
            'calibration',
            'calibration_images',
            args.camera
        )
    
    if not os.path.exists(images_dir):
        logger.error(f"Images directory not found: {images_dir}")
        logger.info("Please capture calibration images first using capture_calibration_images.py")
        return 1
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(
            project_root,
            'calibration_data',
            f'{args.camera}_camera_calibration.npz'
        )
    
    logger.info(f"Calibrating {args.camera} camera...")
    logger.info(f"Checkerboard size: {checkerboard_size}")
    logger.info(f"Square size: {square_size_mm} mm")
    logger.info(f"Images directory: {images_dir}")
    
    # Find checkerboard corners
    obj_points, img_points, image_size = find_checkerboard_corners(
        images_dir,
        checkerboard_size,
        square_size_mm,
        args.show_corners
    )
    
    if obj_points is None:
        logger.error("Failed to find checkerboard corners")
        return 1
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
        obj_points,
        img_points,
        image_size
    )
    
    # Calculate reprojection error
    mean_error = calculate_reprojection_error(
        obj_points,
        img_points,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs
    )
    
    logger.info(f"Mean reprojection error: {mean_error:.4f} pixels")
    
    # Print calibration results
    print("\n" + "="*60)
    print(f"Calibration results for {args.camera.upper()} camera:")
    print("="*60)
    print_camera_info(camera_matrix, dist_coeffs, image_size)
    print(f"RMS error: {ret:.4f} pixels")
    print(f"Mean reprojection error: {mean_error:.4f} pixels")
    print("="*60 + "\n")
    
    # Save calibration data
    save_calibration_data(
        output_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        image_width=image_size[0],
        image_height=image_size[1],
        rms_error=ret,
        mean_reprojection_error=mean_error
    )
    
    logger.info(f"Calibration saved to: {output_path}")
    
    # Recommendation based on error
    if mean_error < 0.5:
        logger.info("✓ Excellent calibration! (error < 0.5 pixels)")
    elif mean_error < 1.0:
        logger.info("✓ Good calibration (error < 1.0 pixels)")
    else:
        logger.warning("⚠ High calibration error. Consider recapturing images with better coverage.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
