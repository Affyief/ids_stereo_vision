#!/usr/bin/env python3
"""
Capture Calibration Images Script
Captures stereo image pairs for camera calibration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import argparse
import logging
from src.camera_interface import StereoCameraSystem
from src.utils import load_yaml_config, ensure_directory
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Capture stereo calibration images")
    parser.add_argument(
        '--config',
        type=str,
        default='config/stereo_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='calibration/chessboard_images',
        help='Output directory for calibration images'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=30,
        help='Number of image pairs to capture'
    )
    parser.add_argument(
        '--delay',
        type=int,
        default=2,
        help='Delay between captures in seconds'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_yaml_config(args.config)
    if config is None:
        logger.error("Failed to load configuration")
        return 1
    
    # Create output directories
    left_dir = os.path.join(args.output, 'left')
    right_dir = os.path.join(args.output, 'right')
    ensure_directory(left_dir)
    ensure_directory(right_dir)
    
    # Get camera settings
    cam_config = config.get('cameras', {})
    left_id = cam_config.get('left', {}).get('device_id', 0)
    right_id = cam_config.get('right', {}).get('device_id', 1)
    exposure = cam_config.get('left', {}).get('exposure', 10)
    gain = int(cam_config.get('left', {}).get('gain', 1))
    
    logger.info("Initializing stereo camera system...")
    stereo_system = StereoCameraSystem(left_id, right_id)
    
    if not stereo_system.initialize(exposure=exposure, gain=gain):
        logger.error("Failed to initialize cameras")
        return 1
    
    logger.info(f"Cameras initialized successfully")
    logger.info(f"Will capture {args.count} image pairs")
    logger.info(f"Images will be saved to: {args.output}")
    logger.info("Press SPACE to capture, 'q' to quit, 's' to skip")
    
    captured = 0
    
    try:
        while captured < args.count:
            # Capture frames
            left_frame, right_frame = stereo_system.capture_stereo_pair()
            
            if left_frame is None or right_frame is None:
                logger.error("Failed to capture frames")
                time.sleep(0.1)
                continue
            
            # Display
            display = cv2.hconcat([left_frame, right_frame])
            h, w = display.shape[:2]
            if w > 1920:
                scale = 1920 / w
                display = cv2.resize(display, (int(w*scale), int(h*scale)))
            
            # Add text
            text = f"Captured: {captured}/{args.count} - Press SPACE to capture, Q to quit"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            cv2.imshow("Calibration Capture", display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quit requested")
                break
            elif key == ord(' '):
                # Capture image pair
                filename = f"calib_{captured:03d}.png"
                left_path = os.path.join(left_dir, filename)
                right_path = os.path.join(right_dir, filename)
                
                cv2.imwrite(left_path, left_frame)
                cv2.imwrite(right_path, right_frame)
                
                captured += 1
                logger.info(f"Captured pair {captured}/{args.count}")
                
                # Show captured feedback
                display_captured = display.copy()
                cv2.putText(display_captured, "CAPTURED!", 
                           (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 
                           2, (0, 255, 0), 3)
                cv2.imshow("Calibration Capture", display_captured)
                cv2.waitKey(500)
                
                # Wait before next capture
                time.sleep(args.delay)
            elif key == ord('s'):
                # Skip and increment counter without saving
                captured += 1
                logger.info(f"Skipped pair {captured}/{args.count}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        stereo_system.release()
        cv2.destroyAllWindows()
    
    logger.info(f"Capture complete. Saved {captured} pairs to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
