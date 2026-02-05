#!/usr/bin/env python3
"""
Run Calibration Script
Performs stereo camera calibration using captured images.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging
from calibration.calibrate_cameras import StereoCalibrator
from src.utils import load_yaml_config, save_stereo_calibration
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Calibrate stereo camera system")
    parser.add_argument(
        '--config',
        type=str,
        default='config/stereo_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='calibration/chessboard_images',
        help='Directory containing calibration images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='config/camera_params.yaml',
        help='Output path for calibration results'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize detected corners'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_yaml_config(args.config)
    if config is None:
        logger.error("Failed to load configuration")
        return 1
    
    # Get calibration pattern settings
    pattern_config = config['system']['calibration_pattern']
    pattern_size = (pattern_config['width'], pattern_config['height'])
    square_size = pattern_config['square_size']
    
    logger.info(f"Calibration pattern: {pattern_size[0]}x{pattern_size[1]}")
    logger.info(f"Square size: {square_size} mm")
    
    # Initialize calibrator
    calibrator = StereoCalibrator(pattern_size, square_size)
    
    # Load images
    left_path = os.path.join(args.images, 'left')
    right_path = os.path.join(args.images, 'right')
    
    logger.info(f"Loading images from {args.images}...")
    left_images, right_images = calibrator.load_image_pairs(left_path, right_path)
    
    if len(left_images) == 0 or len(right_images) == 0:
        logger.error("No valid image pairs found")
        return 1
    
    logger.info(f"Loaded {len(left_images)} stereo pairs")
    
    # Visualize corner detection if requested
    if args.visualize:
        logger.info("Visualizing corner detection (press any key to continue)...")
        for idx, (left_img, right_img) in enumerate(zip(left_images[:5], right_images[:5])):
            corners_left = calibrator.find_chessboard_corners(left_img)
            corners_right = calibrator.find_chessboard_corners(right_img)
            
            if corners_left is not None:
                vis_left = calibrator.draw_chessboard_corners(left_img, corners_left, True)
            else:
                vis_left = left_img
            
            if corners_right is not None:
                vis_right = calibrator.draw_chessboard_corners(right_img, corners_right, True)
            else:
                vis_right = right_img
            
            combined = cv2.hconcat([vis_left, vis_right])
            h, w = combined.shape[:2]
            if w > 1920:
                scale = 1920 / w
                combined = cv2.resize(combined, (int(w*scale), int(h*scale)))
            
            cv2.imshow(f"Corner Detection - Pair {idx+1}", combined)
            cv2.waitKey(500)
        
        cv2.destroyAllWindows()
    
    # Get image size
    image_size = (left_images[0].shape[1], left_images[0].shape[0])
    logger.info(f"Image size: {image_size}")
    
    # Perform stereo calibration
    logger.info("Starting stereo calibration...")
    calibration_results = calibrator.calibrate_stereo(
        left_images,
        right_images,
        image_size
    )
    
    if calibration_results is None:
        logger.error("Calibration failed")
        return 1
    
    # Save calibration results
    logger.info(f"Saving calibration results to {args.output}...")
    if save_stereo_calibration(calibration_results, args.output):
        logger.info("Calibration complete and saved successfully")
        
        # Print summary
        stereo = calibration_results['stereo']
        logger.info("\n" + "="*50)
        logger.info("CALIBRATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Baseline: {stereo['baseline']:.2f} mm")
        logger.info(f"Calibration error: {stereo['calibration_error']:.4f} pixels")
        logger.info(f"Left camera focal length: fx={calibration_results['left_camera']['camera_matrix'][0,0]:.2f}, "
                   f"fy={calibration_results['left_camera']['camera_matrix'][1,1]:.2f}")
        logger.info(f"Right camera focal length: fx={calibration_results['right_camera']['camera_matrix'][0,0]:.2f}, "
                   f"fy={calibration_results['right_camera']['camera_matrix'][1,1]:.2f}")
        logger.info("="*50)
        
        return 0
    else:
        logger.error("Failed to save calibration results")
        return 1


if __name__ == "__main__":
    sys.exit(main())
