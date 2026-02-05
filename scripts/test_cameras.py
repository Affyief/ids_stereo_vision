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
        
        print(f"  Resolution: {width}x{height}")
        print(f"  Frame rate: {framerate} fps")
        print(f"  Left camera: {left_id}")
        print(f"  Right camera: {right_id}")
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1
    
    # Initialize stereo system
    print("\n3. Initializing stereo camera system...")
    stereo = StereoCameraSystem(left_id=left_id, right_id=right_id)
    
    if not stereo.initialize(width, height, exposure, gain, framerate):
        print("✗ Failed to initialize cameras!")
        return 1
    
    # Get camera info
    info = stereo.get_camera_info()
    print("\n✓ Cameras initialized:")
    print(f"  Left:  {info['left'].get('model')} (S/N: {info['left'].get('serial')})")
    print(f"  Right: {info['right'].get('model')} (S/N: {info['right'].get('serial')})")
    
    # Test capture
    print("\n4. Testing frame capture...")
    print("   Press 'Q' to quit, 'S' to save test images")
    
    frame_count = 0
    try:
        while True:
            left_frame, right_frame = stereo.capture_stereo_pair()
            
            if left_frame is None or right_frame is None:
                print("✗ Failed to capture frames")
                break
            
            frame_count += 1
            
            # Display
            cv2.imshow("Left Camera", left_frame)
            cv2.imshow("Right Camera", right_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
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
