#!/usr/bin/env python3
"""
List all detected IDS Peak cameras with detailed information.

This utility helps identify cameras by serial number for configuration.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera_interface_peak import list_ids_peak_cameras, IDS_PEAK_AVAILABLE


def main():
    """Main function to list all IDS Peak cameras."""
    print("=" * 70)
    print("IDS Peak Camera Detection Utility")
    print("=" * 70)
    print()
    
    # Check if IDS Peak is available
    if not IDS_PEAK_AVAILABLE:
        print("❌ IDS Peak SDK not detected!")
        print()
        print("Please install IDS Peak:")
        print()
        print("Linux:")
        print("  1. Download from: https://en.ids-imaging.com/downloads.html")
        print("  2. Select: IDS peak -> Linux -> IDS peak 2.x")
        print("  3. Install: sudo dpkg -i ids-peak_*.deb")
        print("  4. Install Python bindings:")
        print("     pip install /opt/ids/peak/lib/python*/ids_peak-*.whl")
        print("     pip install /opt/ids/peak/lib/python*/ids_peak_ipl-*.whl")
        print()
        print("Windows:")
        print("  1. Download and run installer from IDS website")
        print("  2. Python bindings usually installed automatically")
        print()
        return 1
    
    # List cameras
    print("Scanning for IDS Peak cameras...")
    print()
    
    cameras = list_ids_peak_cameras()
    
    if len(cameras) == 0:
        print("❌ No IDS Peak cameras found!")
        print()
        print("Troubleshooting:")
        print("  1. Check USB 3.0 connection")
        print("  2. Verify camera is powered")
        print("  3. Check IDS Peak Cockpit can detect camera")
        print("  4. Linux: Check permissions (sudo usermod -a -G video $USER)")
        print("  5. Try different USB port/controller")
        print()
        return 1
    
    print(f"✓ Found {len(cameras)} IDS Peak camera(s):")
    print()
    
    # Display camera information
    for cam in cameras:
        print(f"Camera {cam['index']}:")
        print(f"  Model:      {cam['model']}")
        print(f"  Serial:     {cam['serial']}")
        print(f"  Interface:  {cam['interface']}")
        print(f"  Status:     {'✓ Available' if cam['accessible'] else '✗ In use'}")
        
        # Test capture if accessible
        if cam['accessible']:
            test_result = test_camera_capture(cam['serial'], cam['index'])
            if test_result:
                print(f"  Test:       ✓ Capture successful")
            else:
                print(f"  Test:       ✗ Capture failed")
        
        print()
    
    # Generate configuration suggestion
    if len(cameras) >= 2:
        print("=" * 70)
        print("Configuration Suggestion for config/camera_config.yaml:")
        print("=" * 70)
        print()
        print("cameras:")
        print("  use_serial_numbers: true")
        print()
        print("  left_camera:")
        print(f"    serial_number: \"{cameras[0]['serial']}\"")
        print(f"    device_index: {cameras[0]['index']}")
        print(f"    name: \"Left Camera - {cameras[0]['model']}\"")
        print()
        print("  right_camera:")
        print(f"    serial_number: \"{cameras[1]['serial']}\"")
        print(f"    device_index: {cameras[1]['index']}")
        print(f"    name: \"Right Camera - {cameras[1]['model']}\"")
        print()
    elif len(cameras) == 1:
        print("⚠ Warning: Only 1 camera detected. Stereo vision requires 2 cameras.")
        print()
        print("Configuration for single camera:")
        print()
        print("cameras:")
        print("  left_camera:")
        print(f"    serial_number: \"{cameras[0]['serial']}\"")
        print(f"    device_index: {cameras[0]['index']}")
        print()
    
    print("=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Copy the configuration above to config/camera_config.yaml")
    print("2. Mount cameras in stereo rig")
    print("3. Measure baseline distance between camera centers")
    print("4. Run: python scripts/test_cameras.py")
    print("5. Begin calibration: python scripts/capture_calibration_images.py")
    print()
    
    return 0


def test_camera_capture(serial: str, index: int) -> bool:
    """
    Test basic image capture from a camera.
    
    Args:
        serial: Camera serial number
        index: Camera device index
        
    Returns:
        True if capture successful
    """
    try:
        from src.camera_interface_peak import IDSPeakCamera
        
        # Create camera
        camera = IDSPeakCamera(
            serial_number=serial,
            device_index=index,
            name="Test"
        )
        
        # Initialize with minimal settings
        if not camera.initialize(width=640, height=480):
            camera.release()
            return False
        
        # Start acquisition
        if not camera.start_acquisition():
            camera.release()
            return False
        
        # Capture one frame
        frame = camera.capture_frame(timeout_ms=2000)
        
        # Cleanup
        camera.release()
        
        return frame is not None
        
    except Exception as e:
        print(f"    Error: {e}")
        return False


if __name__ == '__main__':
    sys.exit(main())
