#!/usr/bin/env python3
"""
Test script to verify camera connections and functionality.

This script checks if both cameras are accessible and can capture frames.
"""

import os
import sys
import argparse
from pathlib import Path
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config, get_project_root, FPSCounter
from src.camera_interface import create_stereo_camera


def main():
    parser = argparse.ArgumentParser(description='Test stereo camera system')
    parser.add_argument(
        '--config',
        type=str,
        default='config/camera_config.yaml',
        help='Path to camera configuration file'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='Test duration in seconds (0 for unlimited)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Get project root
    project_root = get_project_root()
    
    # Load configuration
    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)
    
    logger.info("=" * 60)
    logger.info("Stereo Camera System Test")
    logger.info("=" * 60)
    logger.info("\nCamera Configuration:")
    logger.info(f"  Resolution: {config['cameras']['resolution']['width']}x{config['cameras']['resolution']['height']}")
    logger.info(f"  Frame rate: {config['cameras']['framerate']} fps")
    logger.info(f"  Left camera ID: {config['cameras']['left_camera']['device_id']}")
    logger.info(f"  Right camera ID: {config['cameras']['right_camera']['device_id']}")
    logger.info("=" * 60 + "\n")
    
    # Initialize cameras
    logger.info("Initializing cameras...")
    stereo_camera = create_stereo_camera(config)
    
    if not stereo_camera.open():
        logger.error("Failed to open cameras!")
        logger.info("\nTroubleshooting:")
        logger.info("1. Check that both cameras are connected")
        logger.info("2. Verify camera device IDs in config/camera_config.yaml")
        logger.info("3. Try listing available cameras with: ls /dev/video*")
        logger.info("4. Ensure no other application is using the cameras")
        return 1
    
    logger.info("✓ Cameras opened successfully!\n")
    
    # Test frame capture
    logger.info("Testing frame capture...")
    left_frame, right_frame = stereo_camera.capture_frames()
    
    if left_frame is None:
        logger.error("✗ Failed to capture frame from left camera")
        stereo_camera.close()
        return 1
    else:
        logger.info(f"✓ Left camera: {left_frame.shape}")
    
    if right_frame is None:
        logger.error("✗ Failed to capture frame from right camera")
        stereo_camera.close()
        return 1
    else:
        logger.info(f"✓ Right camera: {right_frame.shape}\n")
    
    # Display live feed
    logger.info("Displaying live feed...")
    logger.info("Press 'q' to quit, 's' to save test images\n")
    
    cv2.namedWindow('Left Camera', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Right Camera', cv2.WINDOW_NORMAL)
    
    fps_counter = FPSCounter()
    frame_count = 0
    import time
    start_time = time.time()
    
    try:
        while True:
            # Capture frames
            left_frame, right_frame = stereo_camera.capture_frames()
            
            if left_frame is None or right_frame is None:
                logger.warning("Failed to capture frames")
                continue
            
            # Update FPS
            fps = fps_counter.update()
            frame_count += 1
            
            # Add FPS overlay
            cv2.putText(
                left_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.putText(
                right_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Display frames
            cv2.imshow('Left Camera', left_frame)
            cv2.imshow('Right Camera', right_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Test stopped by user")
                break
            elif key == ord('s'):
                # Save test images
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                left_path = f"test_left_{timestamp}.png"
                right_path = f"test_right_{timestamp}.png"
                cv2.imwrite(left_path, left_frame)
                cv2.imwrite(right_path, right_frame)
                logger.info(f"✓ Saved test images: {left_path}, {right_path}")
            
            # Check duration
            if args.duration > 0 and (time.time() - start_time) > args.duration:
                logger.info(f"Test duration ({args.duration}s) completed")
                break
    
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    
    finally:
        # Cleanup
        stereo_camera.close()
        cv2.destroyAllWindows()
    
    # Print summary
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    logger.info("=" * 60)
    logger.info(f"Frames captured: {frame_count}")
    logger.info(f"Duration: {elapsed_time:.1f} seconds")
    logger.info(f"Average FPS: {avg_fps:.1f}")
    logger.info("=" * 60 + "\n")
    
    logger.info("✓ Camera test completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. If cameras work well, proceed to calibration:")
    logger.info("   python scripts/capture_calibration_images.py")
    logger.info("2. After capturing images, run calibration scripts")
    logger.info("3. Then run the stereo vision system:")
    logger.info("   python scripts/run_stereo_vision.py")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
