#!/usr/bin/env python3
"""
Script to capture calibration images from stereo cameras.

This interactive script helps capture checkerboard images from both cameras
simultaneously for calibration purposes.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config, get_project_root
from src.camera_interface import create_stereo_camera


def main():
    parser = argparse.ArgumentParser(
        description='Capture calibration images from stereo cameras'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/camera_config.yaml',
        help='Path to camera configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for calibration images'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        help='Number of images to capture (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Get project root
    project_root = get_project_root()
    
    # Load configuration
    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(project_root, 'calibration', 'calibration_images')
    
    left_output = os.path.join(output_dir, 'left')
    right_output = os.path.join(output_dir, 'right')
    
    # Create output directories
    os.makedirs(left_output, exist_ok=True)
    os.makedirs(right_output, exist_ok=True)
    
    # Get number of images to capture
    num_images = args.num_images or config['calibration']['capture']['num_images']
    delay = config['calibration']['capture']['delay']
    
    logger.info("=" * 60)
    logger.info("Stereo Camera Calibration Image Capture")
    logger.info("=" * 60)
    logger.info(f"Target images: {num_images}")
    logger.info(f"Delay between captures: {delay}s")
    logger.info(f"Output directories:")
    logger.info(f"  Left:  {left_output}")
    logger.info(f"  Right: {right_output}")
    logger.info("=" * 60)
    logger.info("\nInstructions:")
    logger.info("1. Position the checkerboard in front of both cameras")
    logger.info("2. Press SPACE to capture an image pair")
    logger.info("3. Move the checkerboard to different positions and angles")
    logger.info("4. Capture images from various distances and orientations")
    logger.info("5. Press 'q' to quit early")
    logger.info("\nTips for good calibration:")
    logger.info("- Cover the entire field of view")
    logger.info("- Include images at different depths")
    logger.info("- Tilt the checkerboard at various angles")
    logger.info("- Ensure checkerboard is visible in BOTH cameras")
    logger.info("=" * 60 + "\n")
    
    # Initialize stereo camera
    logger.info("Initializing cameras...")
    stereo_camera = create_stereo_camera(config)
    
    if not stereo_camera.open():
        logger.error("Failed to open cameras")
        return 1
    
    logger.info("Cameras initialized successfully\n")
    
    # Create windows
    cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
    
    captured_count = 0
    last_capture_time = 0
    
    try:
        while captured_count < num_images:
            # Capture frames
            left_frame, right_frame = stereo_camera.capture_frames()
            
            if left_frame is None or right_frame is None:
                logger.error("Failed to capture frames")
                continue
            
            # Create display frames with overlays
            left_display = left_frame.copy()
            right_display = right_frame.copy()
            
            # Add status text
            status_text = f"Captured: {captured_count}/{num_images} - Press SPACE to capture, 'q' to quit"
            cv2.putText(
                left_display,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                right_display,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Show frames
            cv2.imshow('Left Camera', left_display)
            cv2.imshow('Right Camera', right_display)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space bar to capture
                current_time = time.time()
                
                # Enforce delay between captures
                if current_time - last_capture_time < delay:
                    remaining = delay - (current_time - last_capture_time)
                    logger.info(f"Please wait {remaining:.1f}s before next capture")
                    continue
                
                # Save images
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                left_filename = f"left_{captured_count:03d}_{timestamp}.png"
                right_filename = f"right_{captured_count:03d}_{timestamp}.png"
                
                left_path = os.path.join(left_output, left_filename)
                right_path = os.path.join(right_output, right_filename)
                
                cv2.imwrite(left_path, left_frame)
                cv2.imwrite(right_path, right_frame)
                
                captured_count += 1
                last_capture_time = current_time
                
                logger.info(f"✓ Captured pair {captured_count}/{num_images}")
                
                # Visual feedback
                for _ in range(2):
                    cv2.imshow('Left Camera', left_frame * 0.5)
                    cv2.imshow('Right Camera', right_frame * 0.5)
                    cv2.waitKey(100)
            
            elif key == ord('q'):  # Quit
                logger.info("Capture cancelled by user")
                break
    
    finally:
        # Cleanup
        stereo_camera.close()
        cv2.destroyAllWindows()
    
    logger.info("\n" + "=" * 60)
    logger.info(f"Capture complete! {captured_count} image pairs saved.")
    logger.info("=" * 60)
    
    if captured_count >= 10:
        logger.info("\n✓ You have enough images for calibration!")
        logger.info("Next steps:")
        logger.info("1. Run: python calibration/calibrate_single_camera.py --camera left")
        logger.info("2. Run: python calibration/calibrate_single_camera.py --camera right")
        logger.info("3. Run: python calibration/calibrate_stereo.py")
    else:
        logger.warning(f"\n⚠ Only {captured_count} pairs captured. Minimum 10 recommended.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
