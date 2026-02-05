#!/usr/bin/env python3
"""
Test IDS Peak cameras
Lists available cameras and tests capture
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera_interface import list_ids_peak_cameras, StereoCameraSystem
from src.utils import load_config
import logging
import cv2
import numpy as np
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ids_stereo_vision')


def main():
    print("=" * 60)
    print("IDS Peak Stereo Camera System Test")
    print("=" * 60)
    
    # List available cameras
    print("\n1. Scanning for IDS Peak cameras...")
    cameras = list_ids_peak_cameras()
    
    if not cameras:
        print("✗ No IDS Peak cameras found!")
        print("\nTroubleshooting:")
        print("1. Check cameras are connected to USB 3.0 ports")
        print("2. Verify IDS Peak Cockpit can see the cameras")
        print("3. Check permissions: sudo usermod -a -G video $USER (then log out/in)")
        print("4. Try: sudo chmod 666 /dev/bus/usb/*/*")
        return 1
    
    print(f"\n✓ Found {len(cameras)} camera(s):")
    for cam in cameras:
        print(f"  [{cam['index']}] {cam['model']} (S/N: {cam['serial']})")
        print(f"      Interface: {cam['interface']}")
    
    # Load config
    print("\n2. Loading configuration...")
    try:
        config = load_config()
        camera_config = config.get('cameras', {})
        
        # Display lens configuration if available
        lens_config = camera_config.get('lens', {})
        if lens_config:
            print(f"\nLens Configuration:")
            print(f"  Model: {lens_config.get('model', 'Unknown')}")
            print(f"  Focal length: {lens_config.get('focal_length_mm', 'Unknown')}mm")
            print(f"  F-number: f/{lens_config.get('f_number', 'Unknown')}")
            
            # Calculate and display FOV
            try:
                from src.utils import load_lens_config
                optical = load_lens_config()
                if optical:
                    fov = optical['fov']
                    print(f"  Field of View: {fov['horizontal']:.1f}° × {fov['vertical']:.1f}° (H×V)")
                    print(f"  Estimated fx/fy: {optical['fx']:.1f} pixels")
            except Exception as e:
                logger.warning(f"Could not compute optical parameters: {e}")
        
        # Determine camera IDs
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
        
        print(f"  Resolution: {width}x{height}")
        print(f"  Frame rate: {framerate} fps")
        print(f"  Pixel format: {pixel_format}")
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
    
    # Get camera info
    info = stereo.get_camera_info()
    print("\n✓ Cameras initialized:")
    print(f"  Left:  {info['left'].get('model')} (S/N: {info['left'].get('serial')})")
    print(f"  Right: {info['right'].get('model')} (S/N: {info['right'].get('serial')})")
    
    # Test capture
    print("\n4. Testing frame capture...")
    print("   Press 'Q' or 'ESC' to quit, 'S' to save test images")
    
    frame_count = 0
    start_time = time.time()
    fps = 0
    
    # Flag to track if we've already shown color warning
    color_warning_shown = False
    
    try:
        while True:
            left_frame, right_frame = stereo.capture_stereo_pair()
            
            if left_frame is None or right_frame is None:
                print("✗ Failed to capture frames")
                break
            
            # Check for color mode on first frame
            if frame_count == 0 and not color_warning_shown:
                channels = left_frame.shape[2] if len(left_frame.shape) == 3 else 1
                print(f"\n✓ Color Detection:")
                print(f"  Camera format: {pixel_format}")
                if pixel_format.startswith('Bayer'):
                    print(f"  Output format: BGR8 (demosaiced)")
                else:
                    print(f"  Output format: {pixel_format}")
                print(f"  Frame shape: {left_frame.shape}")
                print(f"  Channels: {channels}")
                if channels == 3:
                    print(f"  Status: ✓ FULL RGB COLOR")
                elif channels == 1:
                    print(f"  Status: ✗ MONOCHROME")
                    print("\n⚠⚠⚠ WARNING: CAMERAS ARE IN MONOCHROME MODE! ⚠⚠⚠")
                    print(f"Frame shape: {left_frame.shape} (should be 3D with 3 channels)")
                    print("Run: python scripts/diagnose_camera_format.py")
                    print("⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠\n")
                color_warning_shown = True
            
            frame_count += 1
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
            
            # Validate frame dimensions
            if left_frame.shape[1] == 0 or left_frame.shape[0] == 0:
                print("✗ Invalid frame dimensions")
                break
            
            # Resize frames for display (800px width each)
            display_width = 800
            scale = display_width / left_frame.shape[1]
            display_height = int(left_frame.shape[0] * scale)
            
            left_display = cv2.resize(left_frame, (display_width, display_height))
            right_display = cv2.resize(right_frame, (display_width, display_height))
            
            # Add text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_color = (0, 255, 0)  # Green
            
            # Left camera labels
            cv2.putText(left_display, "Left Camera", (10, 30), 
                       font, font_scale, text_color, font_thickness)
            cv2.putText(left_display, f"S/N: {info['left'].get('serial', 'N/A')}", (10, 60), 
                       font, font_scale * 0.6, text_color, font_thickness - 1)
            
            # Right camera labels
            cv2.putText(right_display, "Right Camera", (10, 30), 
                       font, font_scale, text_color, font_thickness)
            cv2.putText(right_display, f"S/N: {info['right'].get('serial', 'N/A')}", (10, 60), 
                       font, font_scale * 0.6, text_color, font_thickness - 1)
            
            # Add FPS counter and instructions at bottom
            fps_text = f"FPS: {fps:.1f}"
            instructions = "Press 'Q' or 'ESC' to quit | 'S' to save"
            
            cv2.putText(left_display, instructions, (10, display_height - 20), 
                       font, 0.5, text_color, 1)
            cv2.putText(right_display, fps_text, (display_width - 120, display_height - 20), 
                       font, 0.5, text_color, 1)
            
            # Concatenate horizontally
            stereo_view = np.hstack([left_display, right_display])
            
            # Add separator line between cameras
            separator_x = display_width
            cv2.line(stereo_view, (separator_x, 0), (separator_x, display_height), 
                    (255, 255, 255), 2)
            
            # Display in single window
            cv2.imshow("Stereo Camera View - Left | Right", stereo_view)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                print(f"\n✓ Captured {frame_count} frames successfully")
                break
            elif key == ord('s'):
                cv2.imwrite('test_left.png', left_frame)
                cv2.imwrite('test_right.png', right_frame)
                print(f"  Saved test_left.png and test_right.png")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        stereo.release()
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✓ Test complete!")
    print("=" * 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
