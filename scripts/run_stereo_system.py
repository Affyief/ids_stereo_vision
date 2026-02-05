#!/usr/bin/env python3
"""
Run Stereo System - Main Application
Real-time stereo vision system with depth estimation and visualization.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import argparse
import logging
import time
import numpy as np
from src.camera_interface import StereoCameraSystem
from src.stereo_processor import StereoProcessor
from src.visualization import StereoVisualizer
from src.utils import load_yaml_config, load_stereo_calibration, ensure_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run stereo vision system")
    parser.add_argument(
        '--config',
        type=str,
        default='config/stereo_config.yaml',
        help='Path to system configuration file'
    )
    parser.add_argument(
        '--calib',
        type=str,
        default='config/camera_params.yaml',
        help='Path to calibration parameters file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for saved data'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without display (for testing)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_yaml_config(args.config)
    if config is None:
        logger.error("Failed to load configuration")
        return 1
    
    # Load calibration
    logger.info("Loading calibration parameters...")
    calibration = load_stereo_calibration(args.calib)
    if calibration is None:
        logger.error("Failed to load calibration parameters")
        logger.error("Please run calibration first using scripts/run_calibration.py")
        return 1
    
    # Check if calibration is valid
    if calibration['left_camera']['camera_matrix'] is None:
        logger.error("Calibration parameters are empty. Please calibrate cameras first.")
        return 1
    
    # Create output directory
    ensure_directory(args.output)
    
    # Get camera settings
    cam_config = config.get('cameras', {})
    left_id = cam_config.get('left', {}).get('device_id', 0)
    right_id = cam_config.get('right', {}).get('device_id', 1)
    exposure = cam_config.get('left', {}).get('exposure', 10)
    gain = int(cam_config.get('left', {}).get('gain', 1))
    
    # Initialize cameras
    logger.info("Initializing stereo camera system...")
    stereo_system = StereoCameraSystem(left_id, right_id)
    
    if not stereo_system.initialize(exposure=exposure, gain=gain):
        logger.error("Failed to initialize cameras")
        return 1
    
    logger.info("Cameras initialized successfully")
    
    # Initialize stereo processor
    logger.info("Initializing stereo processor...")
    try:
        processor = StereoProcessor(calibration, config)
        logger.info("Stereo processor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize stereo processor: {e}")
        stereo_system.release()
        return 1
    
    # Initialize visualizer
    if not args.no_display:
        visualizer = StereoVisualizer(config)
        visualizer.setup_mouse_callback()
        logger.info("Visualizer initialized")
    
    # Print instructions
    logger.info("\n" + "="*60)
    logger.info("STEREO VISION SYSTEM - CONTROLS")
    logger.info("="*60)
    logger.info("q - Quit application")
    logger.info("s - Save current frame and depth map")
    logger.info("m - Toggle measurement mode")
    logger.info("p - Save 3D point cloud")
    logger.info("d - Toggle disparity display")
    logger.info("r - Toggle raw view (no overlays)")
    logger.info("="*60 + "\n")
    
    # Main loop variables
    frame_count = 0
    save_count = 0
    show_disparity = True
    show_overlays = True
    
    try:
        while True:
            # Capture frames
            left_frame, right_frame = stereo_system.capture_stereo_pair()
            
            if left_frame is None or right_frame is None:
                logger.warning("Failed to capture frames")
                time.sleep(0.1)
                continue
            
            # Process stereo pair
            try:
                results = processor.process_stereo_pair(left_frame, right_frame)
                rectified_left = results['rectified_left']
                rectified_right = results['rectified_right']
                disparity = results['disparity']
                depth = results['depth']
            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)
                continue
            
            # Display results
            if not args.no_display:
                key = visualizer.display_results(
                    rectified_left,
                    depth,
                    disparity if show_disparity else None,
                    show_measurements=show_overlays
                )
                
                # Handle key presses
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    # Save frame and depth
                    timestamp = int(time.time() * 1000)
                    left_path = os.path.join(args.output, f"left_{timestamp}.png")
                    depth_path = os.path.join(args.output, f"depth_{timestamp}.npy")
                    depth_vis_path = os.path.join(args.output, f"depth_vis_{timestamp}.png")
                    
                    cv2.imwrite(left_path, rectified_left)
                    np.save(depth_path, depth)
                    
                    # Save colored depth map
                    depth_colored = visualizer.create_depth_colormap(depth)
                    cv2.imwrite(depth_vis_path, depth_colored)
                    
                    save_count += 1
                    logger.info(f"Saved frame and depth map #{save_count} to {args.output}")
                    
                elif key == ord('m'):
                    # Toggle measurement mode
                    visualizer.toggle_measurement_mode()
                    
                elif key == ord('p'):
                    # Save point cloud
                    try:
                        points, colors = processor.compute_point_cloud(disparity, rectified_left)
                        
                        # Save as PLY file
                        timestamp = int(time.time() * 1000)
                        ply_path = os.path.join(args.output, f"pointcloud_{timestamp}.ply")
                        
                        with open(ply_path, 'w') as f:
                            # Write PLY header
                            f.write("ply\n")
                            f.write("format ascii 1.0\n")
                            f.write(f"element vertex {len(points)}\n")
                            f.write("property float x\n")
                            f.write("property float y\n")
                            f.write("property float z\n")
                            f.write("property uchar red\n")
                            f.write("property uchar green\n")
                            f.write("property uchar blue\n")
                            f.write("end_header\n")
                            
                            # Write points
                            for point, color in zip(points, colors):
                                f.write(f"{point[0]} {point[1]} {point[2]} ")
                                f.write(f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
                        
                        logger.info(f"Saved point cloud to {ply_path}")
                    except Exception as e:
                        logger.error(f"Failed to save point cloud: {e}")
                
                elif key == ord('d'):
                    # Toggle disparity display
                    show_disparity = not show_disparity
                    logger.info(f"Disparity display: {'ON' if show_disparity else 'OFF'}")
                
                elif key == ord('r'):
                    # Toggle overlays
                    show_overlays = not show_overlays
                    logger.info(f"Overlays: {'ON' if show_overlays else 'OFF'}")
            
            frame_count += 1
            
            # Periodic status update
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        stereo_system.release()
        if not args.no_display:
            visualizer.cleanup()
        logger.info("Shutdown complete")
    
    logger.info(f"Total frames processed: {frame_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
