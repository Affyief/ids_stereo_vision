#!/usr/bin/env python3
"""
Diagnostic script to check camera pixel format support
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera_interface import list_ids_peak_cameras
import logging

try:
    from ids_peak import ids_peak as peak
    IDS_PEAK_AVAILABLE = True
except ImportError:
    IDS_PEAK_AVAILABLE = False

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def diagnose_cameras():
    """Check what pixel formats cameras actually support"""
    
    if not IDS_PEAK_AVAILABLE:
        print("IDS Peak SDK not available!")
        return
    
    # Initialize Peak
    peak.Library.Initialize()
    
    # Get devices
    device_manager = peak.DeviceManager.Instance()
    device_manager.Update()
    devices = device_manager.Devices()
    
    print(f"\n{'='*60}")
    print(f"IDS Peak Camera Format Diagnostic")
    print(f"{'='*60}\n")
    
    if not devices:
        print("✗ No IDS Peak cameras found!")
        print("\nTroubleshooting:")
        print("1. Check cameras are connected to USB 3.0 ports")
        print("2. Verify IDS Peak Cockpit can see the cameras")
        print("3. Check permissions: sudo usermod -a -G video $USER (then log out/in)")
        print("4. Try: sudo chmod 666 /dev/bus/usb/*/*")
        peak.Library.Close()
        return
    
    for idx, device_descriptor in enumerate(devices):
        print(f"\nCamera {idx}: {device_descriptor.ModelName()}")
        print(f"Serial: {device_descriptor.SerialNumber()}")
        
        try:
            # Open device
            device = device_descriptor.OpenDevice(peak.DeviceAccessType_Control)
            nodemap = device.RemoteDevice().NodeMaps()[0]
            
            # Get pixel format node
            pixel_format_node = nodemap.FindNode("PixelFormat")
            
            if pixel_format_node:
                # Get all available formats
                entries = pixel_format_node.Entries()
                
                print(f"\nAvailable Pixel Formats:")
                for entry in entries:
                    if entry.IsAvailable():
                        format_name = entry.SymbolicValue()
                        is_current = (format_name == pixel_format_node.CurrentEntry().SymbolicValue())
                        marker = "← CURRENT" if is_current else ""
                        print(f"  - {format_name} {marker}")
                
                # Try to set color formats in order of preference
                color_formats = ["BGR8", "RGB8", "BayerRG8"]
                format_set_successfully = False
                
                for format_name in color_formats:
                    print(f"\nAttempting to set {format_name}...")
                    try:
                        for entry in entries:
                            if entry.SymbolicValue() == format_name:
                                pixel_format_node.SetCurrentEntry(entry)
                                current = pixel_format_node.CurrentEntry().SymbolicValue()
                                print(f"✓ Success! Current format: {current}")
                                if "Bayer" in format_name:
                                    print("Note: Bayer format will be demosaiced to BGR in capture")
                                format_set_successfully = True
                                break
                    except Exception as e:
                        print(f"✗ Failed: {e}")
                    
                    if format_set_successfully:
                        break
                
                if not format_set_successfully:
                    print("\n✗ Could not set any color format!")
            else:
                print("✗ PixelFormat node not found!")
            
            device.Close()
            
        except Exception as e:
            print(f"Error opening camera: {e}")
    
    peak.Library.Close()
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    diagnose_cameras()
