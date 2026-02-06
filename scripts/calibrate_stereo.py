#!/usr/bin/env python3
"""
Stereo camera calibration tool for IDS stereo vision system.

This script captures synchronized stereo image pairs from two cameras,
detects checkerboard corners, computes intrinsic and extrinsic calibration,
and saves the results to config/stereo_calibration.yaml.

Supports the Robotiq U3 calibration pattern:
- 16×22 inner corners (17×23 squares)
- 8mm square size
"""

import sys
import os
import cv2
import numpy as np
import yaml
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera_interface import list_ids_peak_cameras, StereoCameraSystem
from src.utils import load_config
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('stereo_calibration')


def capture_calibration_images(stereo_system, pattern_size, num_images=25):
    """
    Capture stereo image pairs for calibration with live preview.
    
    Args:
        stereo_system: Initialized StereoCameraSystem
        pattern_size: Tuple of (rows, cols) for inner corners
        num_images: Target number of image pairs to capture
        
    Returns:
        Lists of (left_images, right_images, object_points, left_corners, right_corners)
    """
    left_images = []
    right_images = []
    object_points_list = []
    left_corners_list = []
    right_corners_list = []
    
    # Prepare 3D object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= 8.0  # 8mm square size for Robotiq U3 pattern
    
    print("\n" + "=" * 70)
    print("STEREO CALIBRATION - Image Capture Mode")
    print("=" * 70)
    print(f"Pattern: {pattern_size[0]}×{pattern_size[1]} inner corners (Robotiq U3)")
    print(f"Target: {num_images} image pairs")
    print("\nInstructions:")
    print("  - Position checkerboard so BOTH cameras can see it clearly")
    print("  - Press SPACE when corners are detected (green overlay)")
    print("  - Capture from different angles and positions")
    print("  - Cover entire field of view")
    print("  - Press 'q' to finish early (minimum 20 pairs recommended)")
    print("=" * 70 + "\n")
    
    captured_count = 0
    
    try:
        while captured_count < num_images:
            # Capture stereo pair
            left_frame, right_frame = stereo_system.capture_stereo_pair()
            
            if left_frame is None or right_frame is None:
                logger.error("Failed to capture frames")
                break
            
            # Convert to grayscale for corner detection
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret_left, corners_left = cv2.findChessboardCorners(
                left_gray, pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            ret_right, corners_right = cv2.findChessboardCorners(
                right_gray, pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            # Create display frames
            left_display = left_frame.copy()
            right_display = right_frame.copy()
            
            # Draw corners if found
            both_found = ret_left and ret_right
            if ret_left:
                cv2.drawChessboardCorners(left_display, pattern_size, corners_left, ret_left)
            if ret_right:
                cv2.drawChessboardCorners(right_display, pattern_size, corners_right, ret_right)
            
            # Add status text
            font = cv2.FONT_HERSHEY_SIMPLEX
            status_color = (0, 255, 0) if both_found else (0, 0, 255)
            status_text = f"Captured: {captured_count}/{num_images}"
            
            cv2.putText(left_display, status_text, (10, 30),
                       font, 1, status_color, 2)
            
            if both_found:
                cv2.putText(left_display, "READY - Press SPACE", (10, 70),
                           font, 0.8, (0, 255, 0), 2)
                cv2.putText(right_display, "READY - Press SPACE", (10, 70),
                           font, 0.8, (0, 255, 0), 2)
            else:
                left_status = "OK" if ret_left else "NOT FOUND"
                right_status = "OK" if ret_right else "NOT FOUND"
                cv2.putText(left_display, f"Left: {left_status}", (10, 70),
                           font, 0.7, status_color, 2)
                cv2.putText(right_display, f"Right: {right_status}", (10, 70),
                           font, 0.7, status_color, 2)
            
            # Resize for display
            scale = 0.6
            h, w = left_display.shape[:2]
            display_size = (int(w * scale), int(h * scale))
            left_small = cv2.resize(left_display, display_size)
            right_small = cv2.resize(right_display, display_size)
            
            # Concatenate horizontally
            combined = np.hstack([left_small, right_small])
            cv2.imshow("Stereo Calibration - Left | Right", combined)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and both_found:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left_refined = cv2.cornerSubPix(
                    left_gray, corners_left, (11, 11), (-1, -1), criteria
                )
                corners_right_refined = cv2.cornerSubPix(
                    right_gray, corners_right, (11, 11), (-1, -1), criteria
                )
                
                # Store data
                left_images.append(left_frame)
                right_images.append(right_frame)
                object_points_list.append(objp)
                left_corners_list.append(corners_left_refined)
                right_corners_list.append(corners_right_refined)
                
                captured_count += 1
                logger.info(f"Captured pair {captured_count}/{num_images}")
                
                # Brief pause to show feedback
                time.sleep(0.3)
                
            elif key == ord('q'):
                if captured_count >= 20:
                    print(f"\nFinishing early with {captured_count} pairs")
                    break
                else:
                    print(f"\nNeed at least 20 pairs (have {captured_count}). Continue capturing or press 'q' again to abort.")
                    key2 = cv2.waitKey(2000) & 0xFF
                    if key2 == ord('q'):
                        print("Calibration aborted.")
                        return None
    
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
        if captured_count >= 20:
            print(f"Continuing with {captured_count} captured pairs")
        else:
            return None
    
    finally:
        cv2.destroyAllWindows()
    
    print(f"\n✓ Captured {captured_count} stereo pairs")
    return (left_images, right_images, object_points_list, left_corners_list, right_corners_list)


def calibrate_cameras(object_points, left_corners, right_corners, image_size):
    """
    Perform stereo calibration.
    
    Args:
        object_points: List of 3D object points
        left_corners: List of left image corner points
        right_corners: List of right image corner points
        image_size: Image size (width, height)
        
    Returns:
        Dictionary with calibration results
    """
    logger.info("Performing stereo calibration...")
    
    # Initial guess for camera matrix
    focal_length = image_size[0]  # Rough estimate
    camera_matrix_init = np.array([
        [focal_length, 0, image_size[0] / 2],
        [0, focal_length, image_size[1] / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs_init = np.zeros(5, dtype=np.float64)
    
    # Calibration flags
    flags = (cv2.CALIB_FIX_ASPECT_RATIO | 
             cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_PRINCIPAL_POINT)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    
    # Stereo calibration
    ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        left_corners,
        right_corners,
        camera_matrix_init.copy(),
        dist_coeffs_init.copy(),
        camera_matrix_init.copy(),
        dist_coeffs_init.copy(),
        image_size,
        criteria=criteria,
        flags=flags
    )
    
    logger.info(f"Stereo calibration RMS error: {ret:.4f} pixels")
    
    # Compute stereo rectification
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        M1, d1, M2, d2,
        image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0
    )
    
    # Calculate baseline
    baseline_mm = np.linalg.norm(T)
    
    # Compute rectification maps
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        M1, d1, R1, P1, image_size, cv2.CV_32FC1
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        M2, d2, R2, P2, image_size, cv2.CV_32FC1
    )
    
    return {
        'rms_error': float(ret),
        'left_camera_matrix': M1,
        'left_dist_coeffs': d1,
        'right_camera_matrix': M2,
        'right_dist_coeffs': d2,
        'rotation_matrix': R,
        'translation_vector': T,
        'essential_matrix': E,
        'fundamental_matrix': F,
        'rectification_left': R1,
        'rectification_right': R2,
        'projection_left': P1,
        'projection_right': P2,
        'disparity_to_depth_matrix': Q,
        'baseline_mm': float(baseline_mm),
        'image_size': image_size,
        'roi_left': roi_left,
        'roi_right': roi_right,
        'map_left_x': map_left_x,
        'map_left_y': map_left_y,
        'map_right_x': map_right_x,
        'map_right_y': map_right_y
    }


def save_calibration(calibration_data, output_path):
    """
    Save calibration data to YAML file.
    
    Args:
        calibration_data: Dictionary with calibration results
        output_path: Path to output YAML file
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy arrays to lists for YAML serialization
    yaml_data = {
        'stereo_calibration': {
            'rms_error': calibration_data['rms_error'],
            'baseline_mm': calibration_data['baseline_mm'],
            'image_width': int(calibration_data['image_size'][0]),
            'image_height': int(calibration_data['image_size'][1]),
            'left_camera': {
                'camera_matrix': calibration_data['left_camera_matrix'].tolist(),
                'distortion_coeffs': calibration_data['left_dist_coeffs'].flatten().tolist(),
                'rectification_matrix': calibration_data['rectification_left'].tolist(),
                'projection_matrix': calibration_data['projection_left'].tolist(),
                'roi': list(calibration_data['roi_left'])
            },
            'right_camera': {
                'camera_matrix': calibration_data['right_camera_matrix'].tolist(),
                'distortion_coeffs': calibration_data['right_dist_coeffs'].flatten().tolist(),
                'rectification_matrix': calibration_data['rectification_right'].tolist(),
                'projection_matrix': calibration_data['projection_right'].tolist(),
                'roi': list(calibration_data['roi_right'])
            },
            'stereo': {
                'rotation_matrix': calibration_data['rotation_matrix'].tolist(),
                'translation_vector': calibration_data['translation_vector'].flatten().tolist(),
                'essential_matrix': calibration_data['essential_matrix'].tolist(),
                'fundamental_matrix': calibration_data['fundamental_matrix'].tolist(),
                'Q_matrix': calibration_data['disparity_to_depth_matrix'].tolist()
            }
        }
    }
    
    # Save YAML
    with open(output_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    
    # Also save numpy arrays in .npz format for faster loading
    npz_path = output_path.replace('.yaml', '.npz')
    np.savez(
        npz_path,
        left_camera_matrix=calibration_data['left_camera_matrix'],
        left_dist_coeffs=calibration_data['left_dist_coeffs'],
        right_camera_matrix=calibration_data['right_camera_matrix'],
        right_dist_coeffs=calibration_data['right_dist_coeffs'],
        R=calibration_data['rotation_matrix'],
        T=calibration_data['translation_vector'],
        E=calibration_data['essential_matrix'],
        F=calibration_data['fundamental_matrix'],
        R1=calibration_data['rectification_left'],
        R2=calibration_data['rectification_right'],
        P1=calibration_data['projection_left'],
        P2=calibration_data['projection_right'],
        Q=calibration_data['disparity_to_depth_matrix'],
        map_left_x=calibration_data['map_left_x'],
        map_left_y=calibration_data['map_left_y'],
        map_right_x=calibration_data['map_right_x'],
        map_right_y=calibration_data['map_right_y'],
        baseline_mm=np.array(calibration_data['baseline_mm']),
        rms_error=np.array(calibration_data['rms_error'])
    )
    
    logger.info(f"Calibration saved to {output_path}")
    logger.info(f"Numpy data saved to {npz_path}")


def main():
    print("=" * 70)
    print("IDS STEREO VISION - Stereo Camera Calibration")
    print("=" * 70)
    print("Pattern: Robotiq U3 (16×22 inner corners, 8mm squares)")
    print("=" * 70 + "\n")
    
    # Pattern size for Robotiq U3
    pattern_size = (16, 22)  # Inner corners
    
    # List available cameras
    print("1. Scanning for IDS Peak cameras...")
    cameras = list_ids_peak_cameras()
    
    if len(cameras) < 2:
        print(f"✗ Need 2 cameras, found {len(cameras)}")
        return 1
    
    print(f"✓ Found {len(cameras)} cameras")
    for cam in cameras:
        print(f"  [{cam['index']}] {cam['model']} (S/N: {cam['serial']})")
    
    # Load configuration
    print("\n2. Loading configuration...")
    try:
        config = load_config('config/camera_config.yaml')
        camera_config = config.get('cameras', {})
        
        # Get camera IDs
        left_id = camera_config.get('left_camera', {}).get('serial_number')
        if not left_id:
            left_id = camera_config.get('left_camera', {}).get('device_id', 0)
        
        right_id = camera_config.get('right_camera', {}).get('serial_number')
        if not right_id:
            right_id = camera_config.get('right_camera', {}).get('device_id', 1)
        
        width = camera_config.get('resolution', {}).get('width', 1296)
        height = camera_config.get('resolution', {}).get('height', 972)
        framerate = camera_config.get('framerate', 30)
        exposure = camera_config.get('exposure_us', 10000)
        gain = camera_config.get('gain_db', 0.0)
        pixel_format = camera_config.get('pixel_format', 'BGR8')
        
        print(f"  Resolution: {width}×{height}")
        print(f"  Left camera: {left_id}")
        print(f"  Right camera: {right_id}")
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    # Initialize stereo system
    print("\n3. Initializing stereo camera system...")
    stereo = StereoCameraSystem(left_id=left_id, right_id=right_id)
    
    if not stereo.initialize(width, height, exposure, gain, framerate, pixel_format):
        print("✗ Failed to initialize cameras!")
        return 1
    
    print("✓ Cameras initialized")
    
    try:
        # Capture calibration images
        print("\n4. Capturing calibration images...")
        result = capture_calibration_images(stereo, pattern_size, num_images=25)
        
        if result is None:
            print("✗ Insufficient calibration images")
            return 1
        
        left_images, right_images, object_points, left_corners, right_corners = result
        
        if len(left_images) < 20:
            print("✗ Insufficient calibration images")
            return 1
        
        # Get image size
        image_size = (left_images[0].shape[1], left_images[0].shape[0])
        
        # Perform calibration
        print("\n5. Computing stereo calibration...")
        calibration_data = calibrate_cameras(
            object_points, left_corners, right_corners, image_size
        )
        
        # Save results
        print("\n6. Saving calibration...")
        output_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'stereo_calibration.yaml'
        )
        save_calibration(calibration_data, output_path)
        
        # Print results
        print("\n" + "=" * 70)
        print("CALIBRATION RESULTS")
        print("=" * 70)
        print(f"RMS Error:    {calibration_data['rms_error']:.4f} pixels")
        print(f"Baseline:     {calibration_data['baseline_mm']:.2f} mm "
              f"({calibration_data['baseline_mm']/10:.2f} cm)")
        print(f"Image Size:   {image_size[0]}×{image_size[1]}")
        print(f"Focal Length: {calibration_data['left_camera_matrix'][0,0]:.1f} pixels")
        
        if calibration_data['rms_error'] < 0.5:
            print("\n✓✓✓ EXCELLENT calibration!")
        elif calibration_data['rms_error'] < 1.0:
            print("\n✓✓ GOOD calibration")
        elif calibration_data['rms_error'] < 2.0:
            print("\n✓ Acceptable calibration")
        else:
            print("\n⚠ High error - consider recapturing images")
        
        print("=" * 70 + "\n")
        
        return 0
        
    finally:
        stereo.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main())
