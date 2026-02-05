#!/usr/bin/env python3
"""
Diagnostic script to check camera pixel format support
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera_interface import list_ids_peak_cameras
import logging
import cv2
import numpy as np

try:
    from ids_peak import ids_peak as peak
    from ids_peak_ipl import ids_peak_ipl as ipl
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
                
                # Try to set Bayer formats and test conversion
                print(f"\nTesting Bayer to BGR conversion...")
                bayer_format_set = False
                
                # Try to set BayerGR8
                if any(entry.SymbolicValue() == "BayerGR8" for entry in entries if entry.IsAvailable()):
                    try:
                        for entry in entries:
                            if entry.SymbolicValue() == "BayerGR8":
                                pixel_format_node.SetCurrentEntry(entry)
                                current = pixel_format_node.CurrentEntry().SymbolicValue()
                                print(f"✓ Set format to: {current}")
                                bayer_format_set = True
                                break
                    except Exception as e:
                        print(f"✗ Failed to set BayerGR8: {e}")
                
                # Test Bayer capture and conversion
                if bayer_format_set:
                    try:
                        # Setup minimal data stream for test
                        datastreams = device.DataStreams()
                        if datastreams:
                            datastream = datastreams[0].OpenDataStream()
                            
                            # Allocate buffer
                            payload_size = nodemap.FindNode("PayloadSize").Value()
                            buffer = datastream.AllocAndAnnounceBuffer(payload_size)
                            datastream.QueueBuffer(buffer)
                            
                            # Start acquisition
                            datastream.StartAcquisition()
                            nodemap.FindNode("TLParamsLocked").SetValue(1)
                            nodemap.FindNode("AcquisitionStart").Execute()
                            nodemap.FindNode("AcquisitionStart").WaitUntilDone()
                            
                            # Capture a frame
                            buffer = datastream.WaitForFinishedBuffer(5000)
                            
                            # Convert to IPL image
                            ipl_image = ipl.Image.CreateFromSizeAndBuffer(
                                buffer.PixelFormat(),
                                buffer.BasePtr(),
                                buffer.Size(),
                                buffer.Width(),
                                buffer.Height()
                            )
                            
                            # Get as 2D numpy array (raw Bayer)
                            numpy_image = ipl_image.get_numpy_2D()
                            
                            # Demosaic using OpenCV
                            bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_BayerGR2BGR)
                            
                            print(f"✓ Bayer conversion successful!")
                            print(f"  Input: {numpy_image.shape} (Bayer raw)")
                            print(f"  Output: {bgr_image.shape} (BGR color)")
                            print(f"  Channels: {bgr_image.shape[2]}")
                            
                            # Stop acquisition
                            nodemap.FindNode("AcquisitionStop").Execute()
                            datastream.KillWait()
                            datastream.StopAcquisition(peak.AcquisitionStopMode_Default)
                            datastream.Flush(peak.DataStreamFlushMode_DiscardAll)
                            datastream.RevokeBuffer(buffer)
                            
                    except Exception as e:
                        print(f"✗ Bayer conversion test failed: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Also try to set BGR8 if available
                print(f"\nAttempting to set BGR8...")
                try:
                    bgr8_available = any(entry.SymbolicValue() == "BGR8" for entry in entries if entry.IsAvailable())
                    if bgr8_available:
                        for entry in entries:
                            if entry.SymbolicValue() == "BGR8":
                                pixel_format_node.SetCurrentEntry(entry)
                                current = pixel_format_node.CurrentEntry().SymbolicValue()
                                print(f"✓ Success! Current format: {current}")
                                break
                    else:
                        print(f"✗ BGR8 not available")
                        print(f"Note: Use Bayer format with automatic demosaicing")
                except Exception as e:
                    print(f"✗ Failed: {e}")
            else:
                print("✗ PixelFormat node not found!")
            
            device.Close()
            
        except Exception as e:
            print(f"Error opening camera: {e}")
            import traceback
            traceback.print_exc()
    
    peak.Library.Close()
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    diagnose_cameras()
