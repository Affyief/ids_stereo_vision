#!/usr/bin/env python3
"""
Main stereo vision application.

This script runs the complete stereo vision system, capturing live frames,
computing depth maps, and displaying results with distance measurements.
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    setup_logging,
    load_config,
    get_project_root,
    FPSCounter,
    save_image_with_timestamp
)
from src.camera_interface_peak import create_stereo_camera_from_config, IDS_PEAK_AVAILABLE
from src.stereo_processor import create_stereo_processor
from src.depth_visualizer import DepthVisualizer


def print_controls():
    """Print keyboard controls."""
    print("\n" + "=" * 60)
    print("Keyboard Controls:")
    print("=" * 60)
    print("  q       - Quit application")
    print("  s       - Save current frame and depth map")
    print("  c       - Cycle through colormaps")
    print("  f       - Toggle WLS filtering")
    print("  m       - Toggle measurement grid")
    print("  x       - Toggle crosshair")
    print("  +/-     - Adjust number of disparities")
    print("  [/]     - Adjust block size")
    print("  r       - Start/stop recording (NOT IMPLEMENTED)")
    print("  p       - Pause/resume")
    print("  h       - Show this help")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run stereo vision system with live depth mapping'
    )
    parser.add_argument(
        '--camera-config',
        type=str,
        default='config/camera_config.yaml',
        help='Path to camera configuration file'
    )
    parser.add_argument(
        '--stereo-config',
        type=str,
        default='config/stereo_config.yaml',
        help='Path to stereo configuration file'
    )
    parser.add_argument(
        '--calibration-dir',
        type=str,
        default='calibration_data',
        help='Directory containing calibration files'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without display (for testing)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    
    # Get project root
    project_root = get_project_root()
    
    logger.info("=" * 60)
    logger.info("IDS Peak Stereo Vision System")
    logger.info("=" * 60 + "\n")
    
    # Check IDS Peak availability
    if not IDS_PEAK_AVAILABLE:
        logger.error("❌ IDS Peak SDK not available!")
        logger.error("Please install from: https://en.ids-imaging.com/downloads.html")
        return 1
    
    # Load configurations
    logger.info("Loading configurations...")
    camera_config_path = os.path.join(project_root, args.camera_config)
    stereo_config_path = os.path.join(project_root, args.stereo_config)
    
    camera_config = load_config(camera_config_path)
    stereo_config = load_config(stereo_config_path)
    
    logger.info("✓ Configurations loaded\n")
    
    # Load calibration and create stereo processor
    logger.info("Loading calibration data...")
    calibration_dir = os.path.join(project_root, args.calibration_dir)
    
    stereo_processor = create_stereo_processor(calibration_dir, stereo_config)
    
    if stereo_processor is None:
        logger.error("Failed to load calibration data!")
        logger.info("\nPlease complete calibration first:")
        logger.info("1. python scripts/capture_calibration_images.py")
        logger.info("2. python calibration/calibrate_single_camera.py --camera left")
        logger.info("3. python calibration/calibrate_single_camera.py --camera right")
        logger.info("4. python calibration/calibrate_stereo.py")
        return 1
    
    logger.info("✓ Calibration data loaded\n")
    
    # Initialize cameras
    logger.info("Initializing cameras...")
    stereo_camera = create_stereo_camera_from_config(camera_config)
    
    # Get parameters from config
    width = camera_config['cameras']['resolution']['width']
    height = camera_config['cameras']['resolution']['height']
    exposure_us = camera_config['cameras'].get('exposure_us')
    gain = camera_config['cameras'].get('gain')
    pixel_format = camera_config['cameras'].get('pixel_format', 'BGR8')
    
    if not stereo_camera.initialize(
        width=width,
        height=height,
        exposure_us=exposure_us,
        gain=gain,
        pixel_format=pixel_format
    ):
        logger.error("Failed to initialize cameras!")
        return 1
    
    logger.info("✓ Cameras initialized\n")
    
    # Create depth visualizer
    visualizer = DepthVisualizer(stereo_config)
    
    # Create window and set mouse callback
    if not args.no_display:
        window_name = 'IDS Stereo Vision System'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, visualizer.mouse_callback)
        print_controls()
    
    # State variables
    fps_counter = FPSCounter()
    paused = False
    use_wls = stereo_config['stereo']['post_process']['use_wls_filter']
    save_counter = 0
    
    # Colormap cycling
    colormaps = ['TURBO', 'JET', 'HSV', 'VIRIDIS', 'PLASMA', 'INFERNO', 'MAGMA', 'HOT', 'COOL']
    current_colormap_idx = 0
    
    logger.info("Starting stereo vision processing...")
    logger.info("Press 'h' for help\n")
    
    try:
        while True:
            if not paused:
                # Capture frames
                left_frame, right_frame = stereo_camera.capture_stereo_pair()
                
                if left_frame is None or right_frame is None:
                    logger.warning("Failed to capture frames")
                    continue
                
                # Process stereo pair
                left_rect, right_rect, disparity, depth = stereo_processor.process_stereo_pair(
                    left_frame,
                    right_frame
                )
                
                # Update FPS
                fps = fps_counter.update()
                
                # Create visualization
                if not args.no_display:
                    visualization = visualizer.create_visualization(
                        left_rect,
                        right_rect,
                        depth,
                        fps
                    )
            
            # Display
            if not args.no_display:
                cv2.imshow(window_name, visualization)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("Quit requested by user")
                break
            
            elif key == ord('s'):
                # Save current frame and depth map
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                
                # Save images
                cv2.imwrite(f"captured_left_{timestamp}.png", left_rect)
                cv2.imwrite(f"captured_right_{timestamp}.png", right_rect)
                cv2.imwrite(f"captured_depth_{timestamp}.png", visualizer.create_depth_colormap(depth))
                
                # Save depth data
                np.save(f"captured_depth_{timestamp}.npy", depth)
                
                save_counter += 1
                logger.info(f"✓ Saved images and depth map ({save_counter})")
            
            elif key == ord('c'):
                # Cycle colormap
                current_colormap_idx = (current_colormap_idx + 1) % len(colormaps)
                new_colormap = colormaps[current_colormap_idx]
                visualizer.set_colormap(new_colormap)
                logger.info(f"Colormap: {new_colormap}")
            
            elif key == ord('f'):
                # Toggle WLS filter
                use_wls = not use_wls
                stereo_config['stereo']['post_process']['use_wls_filter'] = use_wls
                logger.info(f"WLS filter: {'ON' if use_wls else 'OFF'}")
            
            elif key == ord('m'):
                # Toggle measurement grid
                visualizer.toggle_measurements()
            
            elif key == ord('x'):
                # Toggle crosshair
                visualizer.toggle_crosshair()
            
            elif key == ord('+') or key == ord('='):
                # Increase num_disparities
                current_num_disp = stereo_config['stereo']['sgbm']['num_disparities']
                new_num_disp = min(256, current_num_disp + 16)
                stereo_processor.update_stereo_params(num_disparities=new_num_disp)
                stereo_config['stereo']['sgbm']['num_disparities'] = new_num_disp
                logger.info(f"Number of disparities: {new_num_disp}")
            
            elif key == ord('-') or key == ord('_'):
                # Decrease num_disparities
                current_num_disp = stereo_config['stereo']['sgbm']['num_disparities']
                new_num_disp = max(16, current_num_disp - 16)
                stereo_processor.update_stereo_params(num_disparities=new_num_disp)
                stereo_config['stereo']['sgbm']['num_disparities'] = new_num_disp
                logger.info(f"Number of disparities: {new_num_disp}")
            
            elif key == ord('['):
                # Decrease block size
                current_block_size = stereo_config['stereo']['sgbm']['block_size']
                new_block_size = max(5, current_block_size - 2)
                stereo_processor.update_stereo_params(block_size=new_block_size)
                stereo_config['stereo']['sgbm']['block_size'] = new_block_size
                logger.info(f"Block size: {new_block_size}")
            
            elif key == ord(']'):
                # Increase block size
                current_block_size = stereo_config['stereo']['sgbm']['block_size']
                new_block_size = min(21, current_block_size + 2)
                stereo_processor.update_stereo_params(block_size=new_block_size)
                stereo_config['stereo']['sgbm']['block_size'] = new_block_size
                logger.info(f"Block size: {new_block_size}")
            
            elif key == ord('p'):
                # Pause/resume
                paused = not paused
                logger.info("PAUSED" if paused else "RESUMED")
            
            elif key == ord('h'):
                # Show help
                print_controls()
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        logger.info("\nCleaning up...")
        stereo_camera.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.info("✓ Cleanup complete")
    
    logger.info("\n" + "=" * 60)
    logger.info("Stereo vision system stopped")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
