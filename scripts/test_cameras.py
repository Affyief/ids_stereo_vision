#!/usr/bin/env python3
"""
Test script to verify IDS Peak camera connections and functionality.

This script checks if both cameras are accessible and can capture frames.
Uses IDS Peak SDK (modern GenICam interface).
"""

import os
import sys
import argparse
import time
from pathlib import Path
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, load_config, get_project_root, FPSCounter
from src.camera_interface_peak import (
    create_stereo_camera_from_config,
    list_ids_peak_cameras,
    IDS_PEAK_AVAILABLE
)


def main():
    parser = argparse.ArgumentParser(description='Test IDS Peak stereo camera system')
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
    
    logger.info("=" * 70)
    logger.info("IDS Peak Stereo Camera System Test")
    logger.info("=" * 70)
    
    # Check IDS Peak availability
    if not IDS_PEAK_AVAILABLE:
        logger.error("❌ IDS Peak SDK not available!")
        logger.error("")
        logger.error("Please install IDS Peak:")
        logger.error("Linux:")
        logger.error("  1. Download from: https://en.ids-imaging.com/downloads.html")
        logger.error("  2. Install: sudo dpkg -i ids-peak_*.deb")
        logger.error("  3. Install Python bindings:")
        logger.error("     pip install /opt/ids/peak/lib/python*/ids_peak-*.whl")
        logger.error("     pip install /opt/ids/peak/lib/python*/ids_peak_ipl-*.whl")
        logger.error("")
        logger.error("Windows:")
        logger.error("  Download and run installer from IDS website")
        return 1
    
    logger.info("\n✓ IDS Peak SDK detected")
    
    # List available cameras
    logger.info("\nScanning for cameras...")
    cameras = list_ids_peak_cameras()
    
    if len(cameras) == 0:
        logger.error("❌ No cameras detected!")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check USB 3.0 connections")
        logger.error("  2. Verify IDS Peak Cockpit can see cameras")
        logger.error("  3. Linux: Check permissions (sudo usermod -a -G video $USER)")
        logger.error("  4. Run: python scripts/list_cameras.py")
        return 1
    
    logger.info(f"✓ Found {len(cameras)} camera(s):")
    for cam in cameras:
        logger.info(f"  [{cam['index']}] {cam['model']} (S/N: {cam['serial']})")
    
    if len(cameras) < 2:
        logger.warning("\n⚠ Warning: Only 1 camera detected. Stereo requires 2 cameras.")
    
    # Get project root
    project_root = get_project_root()
    
    # Load configuration
    config_path = os.path.join(project_root, args.config)
    config = load_config(config_path)
    
    logger.info("\n" + "=" * 70)
    logger.info("Camera Configuration:")
    logger.info("=" * 70)
    logger.info(f"  Resolution: {config['cameras']['resolution']['width']}x{config['cameras']['resolution']['height']}")
    logger.info(f"  Frame rate: {config['cameras']['framerate']} fps")
    logger.info(f"  Exposure: {config['cameras'].get('exposure_us', 'auto')} µs")
    logger.info(f"  Gain: {config['cameras'].get('gain', 1.0)}")
    logger.info(f"  Pixel format: {config['cameras'].get('pixel_format', 'BGR8')}")
    
    if config['cameras'].get('use_serial_numbers', False):
        logger.info(f"  Left S/N: {config['cameras']['left_camera'].get('serial_number', 'Not set')}")
        logger.info(f"  Right S/N: {config['cameras']['right_camera'].get('serial_number', 'Not set')}")
    else:
        logger.info(f"  Left index: {config['cameras']['left_camera'].get('device_index', 0)}")
        logger.info(f"  Right index: {config['cameras']['right_camera'].get('device_index', 1)}")
    
    logger.info("=" * 70 + "\n")
    
    # Initialize cameras
    logger.info("Initializing stereo camera system...")
    stereo_camera = create_stereo_camera_from_config(config)
    
    # Get parameters from config
    width = config['cameras']['resolution']['width']
    height = config['cameras']['resolution']['height']
    exposure_us = config['cameras'].get('exposure_us')
    gain = config['cameras'].get('gain')
    pixel_format = config['cameras'].get('pixel_format', 'BGR8')
    
    if not stereo_camera.initialize(
        width=width,
        height=height,
        exposure_us=exposure_us,
        gain=gain,
        pixel_format=pixel_format
    ):
        logger.error("❌ Failed to initialize cameras!")
        logger.error("\nTroubleshooting:")
        logger.error("1. Verify serial numbers in config match detected cameras")
        logger.error("2. Run: python scripts/list_cameras.py")
        logger.error("3. Check that cameras are not in use by another application")
        logger.error("4. Try with device_index instead of serial_number")
        return 1
    
    logger.info("✓ Cameras initialized successfully!\n")
    
    # Test frame capture
    logger.info("Testing frame capture...")
    left_frame, right_frame = stereo_camera.capture_stereo_pair()
    
    if left_frame is None:
        logger.error("✗ Failed to capture frame from left camera")
        stereo_camera.release()
        return 1
    else:
        logger.info(f"✓ Left camera: {left_frame.shape}")
    
    if right_frame is None:
        logger.error("✗ Failed to capture frame from right camera")
        stereo_camera.release()
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
    start_time = time.time()
    
    try:
        while True:
            # Capture frames
            left_frame, right_frame = stereo_camera.capture_stereo_pair()
            
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
        stereo_camera.release()
        cv2.destroyAllWindows()
    
    # Print summary
    elapsed_time = time.time() - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    logger.info("\n" + "=" * 70)
    logger.info("Test Summary:")
    logger.info("=" * 70)
    logger.info(f"Frames captured: {frame_count}")
    logger.info(f"Duration: {elapsed_time:.1f} seconds")
    logger.info(f"Average FPS: {avg_fps:.1f}")
    logger.info("=" * 70 + "\n")
    
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
