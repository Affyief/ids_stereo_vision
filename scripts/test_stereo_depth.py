#!/usr/bin/env python3
"""
Real-time stereo depth measurement tool.

Loads stereo calibration data, computes disparity maps using StereoSGBM,
converts disparity to depth/distance, and provides interactive distance
measurement by clicking on the image.

Features:
- Real-time depth map visualization
- Click to measure distance at any point
- Toggle between rectified image and disparity map
- Save images and depth maps
- Adjustable parameters
"""

import sys
import os
import cv2
import numpy as np
import yaml
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
logger = logging.getLogger('stereo_depth')


class StereoDepthMeasurement:
    """Real-time stereo depth measurement system."""
    
    def __init__(self, stereo_system, calibration_data):
        """
        Initialize the depth measurement system.
        
        Args:
            stereo_system: Initialized StereoCameraSystem
            calibration_data: Dictionary with stereo calibration
        """
        self.stereo = stereo_system
        self.calib = calibration_data
        
        # Current mouse position
        self.mouse_x = -1
        self.mouse_y = -1
        
        # Display mode: 'disparity' or 'rectified'
        self.display_mode = 'disparity'
        
        # Create stereo matcher (StereoSGBM with 3-way mode)
        self.create_stereo_matcher()
        
        # Saved image counter
        self.save_counter = 0
        
        logger.info("Stereo depth measurement system initialized")
    
    def create_stereo_matcher(self):
        """Create StereoSGBM matcher with optimized parameters."""
        # Parameters for StereoSGBM
        self.num_disparities = 128  # Must be divisible by 16
        self.block_size = 5  # Odd number, 5-21
        
        # Create StereoSGBM with MODE_SGBM_3WAY (highest quality)
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,  # Recommended formula
            P2=32 * 3 * self.block_size ** 2,  # Recommended formula
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # 3-way mode for best quality
        )
        
        logger.info(f"StereoSGBM created: {self.num_disparities} disparities, "
                   f"block size {self.block_size}, 3-way mode")
    
    def compute_disparity(self, left_rect, right_rect):
        """
        Compute disparity map from rectified stereo pair.
        
        Args:
            left_rect: Rectified left image
            right_rect: Rectified right image
            
        Returns:
            Disparity map (float32)
        """
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = self.stereo_matcher.compute(left_gray, right_gray)
        
        # Convert to float32 and scale
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def disparity_to_depth(self, disparity):
        """
        Convert disparity to depth in millimeters.
        
        Args:
            disparity: Disparity map
            
        Returns:
            Depth map in millimeters
        """
        # Get focal length and baseline from calibration
        focal_length = self.calib['left_camera']['camera_matrix'][0][0]
        baseline_mm = self.calib['stereo_calibration']['baseline_mm']
        
        # Avoid division by zero
        disparity_safe = np.where(disparity > 0, disparity, 0.01)
        
        # Compute depth: Z = (f * B) / d
        depth_mm = (focal_length * baseline_mm) / disparity_safe
        
        # Filter invalid depths
        depth_mm = np.where(disparity > 0, depth_mm, 0)
        
        return depth_mm
    
    def colorize_disparity(self, disparity):
        """
        Colorize disparity map for visualization.
        
        Args:
            disparity: Disparity map
            
        Returns:
            Color-coded disparity image
        """
        # Normalize disparity to 0-255
        disparity_normalized = cv2.normalize(
            disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        # Apply colormap (TURBO for better visualization)
        disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_TURBO)
        
        # Make invalid disparities black
        mask = disparity <= 0
        disparity_color[mask] = [0, 0, 0]
        
        return disparity_color
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for distance measurement."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
    
    def run(self):
        """Run the real-time depth measurement loop."""
        print("\n" + "=" * 70)
        print("STEREO DEPTH MEASUREMENT")
        print("=" * 70)
        print("Controls:")
        print("  MOVE MOUSE   - Measure distance at point")
        print("  'd'          - Toggle disparity/rectified view")
        print("  's'          - Save current images")
        print("  '+'          - Increase disparities")
        print("  '-'          - Decrease disparities")
        print("  'q' or ESC   - Quit")
        print("=" * 70 + "\n")
        
        # Create window and set mouse callback
        window_name = "Stereo Depth Measurement"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        # Get rectification maps
        map_left_x = self.calib['map_left_x']
        map_left_y = self.calib['map_left_y']
        map_right_x = self.calib['map_right_x']
        map_right_y = self.calib['map_right_y']
        
        try:
            while True:
                # Capture stereo pair
                left_frame, right_frame = self.stereo.capture_stereo_pair()
                
                if left_frame is None or right_frame is None:
                    logger.error("Failed to capture frames")
                    break
                
                # Rectify images
                left_rect = cv2.remap(
                    left_frame, map_left_x, map_left_y,
                    cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
                )
                right_rect = cv2.remap(
                    right_frame, map_right_x, map_right_y,
                    cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
                )
                
                # Compute disparity
                disparity = self.compute_disparity(left_rect, right_rect)
                
                # Convert to depth
                depth_mm = self.disparity_to_depth(disparity)
                
                # Create visualization
                if self.display_mode == 'disparity':
                    display = self.colorize_disparity(disparity)
                else:
                    display = left_rect.copy()
                
                # Add crosshair and distance at mouse position
                if 0 <= self.mouse_x < display.shape[1] and 0 <= self.mouse_y < display.shape[0]:
                    # Draw crosshair
                    cv2.drawMarker(
                        display, (self.mouse_x, self.mouse_y),
                        (0, 255, 0), cv2.MARKER_CROSS, 20, 2
                    )
                    
                    # Get distance at this point
                    distance_mm = depth_mm[self.mouse_y, self.mouse_x]
                    disparity_value = disparity[self.mouse_y, self.mouse_x]
                    
                    if 0 < distance_mm < 10000:  # Valid range
                        distance_m = distance_mm / 1000.0
                        text = f"Distance: {distance_m:.2f} m ({distance_mm:.0f} mm)"
                        disp_text = f"Disparity: {disparity_value:.1f} px"
                    else:
                        text = "Distance: INVALID"
                        disp_text = f"Disparity: {disparity_value:.1f} px"
                    
                    # Draw text with background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    
                    # Get text size for background
                    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    (disp_w, disp_h), _ = cv2.getTextSize(disp_text, font, font_scale, thickness)
                    
                    # Draw background rectangles
                    cv2.rectangle(display, (self.mouse_x + 15, self.mouse_y - 40),
                                 (self.mouse_x + 15 + text_w + 10, self.mouse_y - 40 + text_h + 10),
                                 (0, 0, 0), -1)
                    cv2.rectangle(display, (self.mouse_x + 15, self.mouse_y - 20),
                                 (self.mouse_x + 15 + disp_w + 10, self.mouse_y - 20 + disp_h + 10),
                                 (0, 0, 0), -1)
                    
                    # Draw text
                    cv2.putText(display, text, (self.mouse_x + 20, self.mouse_y - 25),
                               font, font_scale, (0, 255, 0), thickness)
                    cv2.putText(display, disp_text, (self.mouse_x + 20, self.mouse_y - 5),
                               font, font_scale, (0, 255, 0), thickness)
                
                # Add info overlay
                info_y = 30
                font = cv2.FONT_HERSHEY_SIMPLEX
                info_color = (0, 255, 0)
                
                cv2.putText(display, f"Mode: {self.display_mode.upper()}", (10, info_y),
                           font, 0.7, info_color, 2)
                cv2.putText(display, f"Disparities: {self.num_disparities}", (10, info_y + 30),
                           font, 0.7, info_color, 2)
                cv2.putText(display, f"Block Size: {self.block_size}", (10, info_y + 60),
                           font, 0.7, info_color, 2)
                
                # Show display
                cv2.imshow(window_name, display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('d'):
                    # Toggle display mode
                    self.display_mode = 'rectified' if self.display_mode == 'disparity' else 'disparity'
                    logger.info(f"Display mode: {self.display_mode}")
                elif key == ord('s'):
                    # Save images
                    self.save_counter += 1
                    cv2.imwrite(f'depth_left_{self.save_counter:03d}.png', left_rect)
                    cv2.imwrite(f'depth_right_{self.save_counter:03d}.png', right_rect)
                    cv2.imwrite(f'depth_disparity_{self.save_counter:03d}.png',
                               self.colorize_disparity(disparity))
                    
                    # Save depth as numpy array
                    np.save(f'depth_map_{self.save_counter:03d}.npy', depth_mm)
                    
                    logger.info(f"Saved images {self.save_counter:03d}")
                elif key == ord('+') or key == ord('='):
                    # Increase disparities
                    self.num_disparities = min(256, self.num_disparities + 16)
                    self.create_stereo_matcher()
                    logger.info(f"Disparities: {self.num_disparities}")
                elif key == ord('-') or key == ord('_'):
                    # Decrease disparities
                    self.num_disparities = max(16, self.num_disparities - 16)
                    self.create_stereo_matcher()
                    logger.info(f"Disparities: {self.num_disparities}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print("✓ Depth measurement complete!")
        print("=" * 70)


def load_stereo_calibration(yaml_path, npz_path):
    """
    Load stereo calibration data.
    
    Args:
        yaml_path: Path to YAML calibration file
        npz_path: Path to NPZ calibration file
        
    Returns:
        Dictionary with calibration data
    """
    # Try to load from NPZ (faster)
    if os.path.exists(npz_path):
        logger.info(f"Loading calibration from {npz_path}")
        npz_data = np.load(npz_path)
        
        # Also load YAML for metadata
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        return {
            'stereo_calibration': yaml_data['stereo_calibration'],
            'left_camera': yaml_data['stereo_calibration']['left_camera'],
            'right_camera': yaml_data['stereo_calibration']['right_camera'],
            'map_left_x': npz_data['map_left_x'],
            'map_left_y': npz_data['map_left_y'],
            'map_right_x': npz_data['map_right_x'],
            'map_right_y': npz_data['map_right_y']
        }
    else:
        logger.error(f"Calibration file not found: {npz_path}")
        logger.info("Please run scripts/calibrate_stereo.py first")
        return None


def main():
    print("=" * 70)
    print("IDS STEREO VISION - Real-Time Depth Measurement")
    print("=" * 70 + "\n")
    
    # Check for calibration file
    calib_yaml = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'stereo_calibration.yaml'
    )
    calib_npz = os.path.join(
        os.path.dirname(__file__), '..', 'config', 'stereo_calibration.npz'
    )
    
    if not os.path.exists(calib_yaml):
        print(f"✗ Calibration file not found: {calib_yaml}")
        print("\nPlease run: python scripts/calibrate_stereo.py")
        return 1
    
    # Load calibration
    print("1. Loading stereo calibration...")
    calibration = load_stereo_calibration(calib_yaml, calib_npz)
    
    if calibration is None:
        return 1
    
    baseline = calibration['stereo_calibration']['baseline_mm']
    rms_error = calibration['stereo_calibration']['rms_error']
    print(f"  ✓ Baseline: {baseline:.1f} mm")
    print(f"  ✓ RMS error: {rms_error:.4f} pixels")
    
    # Load camera configuration
    print("\n2. Loading camera configuration...")
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
        
        print(f"  ✓ Resolution: {width}×{height}")
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    # List available cameras
    print("\n3. Scanning for cameras...")
    cameras = list_ids_peak_cameras()
    
    if len(cameras) < 2:
        print(f"✗ Need 2 cameras, found {len(cameras)}")
        return 1
    
    print(f"  ✓ Found {len(cameras)} cameras")
    
    # Initialize stereo system
    print("\n4. Initializing stereo camera system...")
    stereo = StereoCameraSystem(left_id=left_id, right_id=right_id)
    
    if not stereo.initialize(width, height, exposure, gain, framerate, pixel_format):
        print("✗ Failed to initialize cameras!")
        return 1
    
    print("  ✓ Cameras initialized")
    
    try:
        # Create and run depth measurement system
        print("\n5. Starting depth measurement...\n")
        depth_system = StereoDepthMeasurement(stereo, calibration)
        depth_system.run()
        
        return 0
        
    finally:
        stereo.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main())
