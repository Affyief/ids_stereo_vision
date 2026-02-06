#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.camera_interface import StereoCameraSystem
from ids_peak_ipl import ids_peak_ipl as ipl
import cv2
import numpy as np

print("Debugging Bayer capture with ImageConverter...")

stereo = StereoCameraSystem(left_id=0, right_id=1)
stereo.initialize(1296, 972, 10000, 1.0, 30, 'BGR8')

# Capture ONE frame
left_frame, right_frame = stereo.capture_stereo_pair()

print(f"\n=== Frame Info ===")
print(f"Shape: {left_frame.shape}")
print(f"Dtype: {left_frame.dtype}")

if len(left_frame.shape) == 3:
    print(f"Channels: {left_frame.shape[2]}")
    
    # Check if channels differ (real color vs grayscale-as-BGR)
    b_channel = left_frame[:, :, 0]
    g_channel = left_frame[:, :, 1]
    r_channel = left_frame[:, :, 2]
    
    print(f"\nChannel Statistics:")
    print(f"  Blue  mean: {b_channel.mean():.2f}, std: {b_channel.std():.2f}")
    print(f"  Green mean: {g_channel.mean():.2f}, std: {g_channel.std():.2f}")
    print(f"  Red   mean: {r_channel.mean():.2f}, std: {r_channel.std():.2f}")
    
    # Check if identical
    all_identical = (np.array_equal(b_channel, g_channel) and 
                     np.array_equal(g_channel, r_channel))
    
    if all_identical:
        print("\n✗ ALL CHANNELS IDENTICAL = Grayscale!")
    else:
        print("\n✓ CHANNELS DIFFER = Full RGB Color!")
        
        # Sample pixels to show color variation
        print(f"\nSample pixels (BGR):")
        print(f"  Pixel [100, 100]: {left_frame[100, 100, :]}")
        print(f"  Pixel [500, 500]: {left_frame[500, 500, :]}")
        
elif len(left_frame.shape) == 2:
    print(f"Channels: 1 (grayscale or raw Bayer)")
    print("\n✗ FRAME IS 2D!")
    print("ImageConverter did NOT run or failed!")
    print("Check that ids_peak_ipl is properly installed.")

# Save and verify file type
cv2.imwrite('test_color_frame.png', left_frame)
print(f"\nSaved: test_color_frame.png")

import subprocess
result = subprocess.run(['file', 'test_color_frame.png'], 
                       capture_output=True, text=True)
print(f"File type: {result.stdout.strip()}")

if 'RGB' in result.stdout or 'color' in result.stdout:
    print("✓✓✓ File contains RGB COLOR data!")
elif 'grayscale' in result.stdout:
    print("✗✗✗ File is GRAYSCALE - conversion failed!")

stereo.release()
print("\nDone!")
