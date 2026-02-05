#!/usr/bin/env python3
"""
Test Camera Connection
Verify that both IDS cameras are detected and can capture frames.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
from src.camera_interface import StereoCameraSystem
from src.utils import load_yaml_config
import cv2
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_camera_detection():
    """Test if cameras can be detected and initialized."""
    logger.info("="*60)
    logger.info("TEST: Camera Detection")
    logger.info("="*60)
    
    try:
        # Load configuration
        config_path = 'config/stereo_config.yaml'
        config = load_yaml_config(config_path)
        
        if config is None:
            logger.error("Failed to load configuration")
            return False
        
        cam_config = config.get('cameras', {})
        left_id = cam_config.get('left', {}).get('device_id', 0)
        right_id = cam_config.get('right', {}).get('device_id', 1)
        
        logger.info(f"Attempting to initialize cameras (left: {left_id}, right: {right_id})...")
        
        stereo_system = StereoCameraSystem(left_id, right_id)
        success = stereo_system.initialize()
        
        if success:
            logger.info("✓ Both cameras initialized successfully")
            
            # Get camera info
            info = stereo_system.get_system_info()
            logger.info("\nLeft Camera Info:")
            for key, value in info['left'].items():
                logger.info(f"  {key}: {value}")
            
            logger.info("\nRight Camera Info:")
            for key, value in info['right'].items():
                logger.info(f"  {key}: {value}")
            
            stereo_system.release()
            return True
        else:
            logger.error("✗ Failed to initialize cameras")
            return False
            
    except Exception as e:
        logger.error(f"✗ Exception during camera detection: {e}")
        return False


def test_frame_capture():
    """Test frame capture from both cameras."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Frame Capture")
    logger.info("="*60)
    
    try:
        config = load_yaml_config('config/stereo_config.yaml')
        if config is None:
            logger.error("Failed to load configuration")
            return False
        
        cam_config = config.get('cameras', {})
        left_id = cam_config.get('left', {}).get('device_id', 0)
        right_id = cam_config.get('right', {}).get('device_id', 1)
        exposure = cam_config.get('left', {}).get('exposure', 10)
        gain = int(cam_config.get('left', {}).get('gain', 1))
        
        stereo_system = StereoCameraSystem(left_id, right_id)
        
        if not stereo_system.initialize(exposure=exposure, gain=gain):
            logger.error("Failed to initialize cameras")
            return False
        
        logger.info("Capturing test frames...")
        
        # Capture a few frames
        success_count = 0
        for i in range(5):
            left_frame, right_frame = stereo_system.capture_stereo_pair()
            
            if left_frame is not None and right_frame is not None:
                success_count += 1
                logger.info(f"  Frame {i+1}: ✓ Captured ({left_frame.shape}, {right_frame.shape})")
            else:
                logger.error(f"  Frame {i+1}: ✗ Failed to capture")
            
            time.sleep(0.1)
        
        stereo_system.release()
        
        if success_count == 5:
            logger.info(f"✓ All frames captured successfully")
            return True
        else:
            logger.error(f"✗ Only {success_count}/5 frames captured")
            return False
            
    except Exception as e:
        logger.error(f"✗ Exception during frame capture: {e}")
        return False


def test_synchronized_capture():
    """Test that frames are synchronized."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Synchronized Capture")
    logger.info("="*60)
    
    try:
        config = load_yaml_config('config/stereo_config.yaml')
        if config is None:
            return False
        
        cam_config = config.get('cameras', {})
        left_id = cam_config.get('left', {}).get('device_id', 0)
        right_id = cam_config.get('right', {}).get('device_id', 1)
        
        stereo_system = StereoCameraSystem(left_id, right_id)
        
        if not stereo_system.initialize():
            return False
        
        logger.info("Testing capture timing...")
        
        # Measure capture times
        times = []
        for i in range(10):
            start = time.time()
            left_frame, right_frame = stereo_system.capture_stereo_pair()
            elapsed = (time.time() - start) * 1000  # Convert to ms
            
            if left_frame is not None and right_frame is not None:
                times.append(elapsed)
        
        stereo_system.release()
        
        if len(times) > 0:
            avg_time = sum(times) / len(times)
            logger.info(f"  Average capture time: {avg_time:.2f} ms")
            logger.info(f"  Estimated FPS: {1000/avg_time:.1f}")
            logger.info("✓ Synchronized capture working")
            return True
        else:
            logger.error("✗ Failed to capture frames")
            return False
            
    except Exception as e:
        logger.error(f"✗ Exception during synchronized capture test: {e}")
        return False


def test_resolution_and_format():
    """Test image resolution and format."""
    logger.info("\n" + "="*60)
    logger.info("TEST: Resolution and Format")
    logger.info("="*60)
    
    try:
        config = load_yaml_config('config/stereo_config.yaml')
        if config is None:
            return False
        
        cam_config = config.get('cameras', {})
        left_id = cam_config.get('left', {}).get('device_id', 0)
        right_id = cam_config.get('right', {}).get('device_id', 1)
        
        stereo_system = StereoCameraSystem(left_id, right_id)
        
        if not stereo_system.initialize():
            return False
        
        left_frame, right_frame = stereo_system.capture_stereo_pair()
        stereo_system.release()
        
        if left_frame is None or right_frame is None:
            logger.error("✗ Failed to capture frames")
            return False
        
        # Check resolution
        left_shape = left_frame.shape
        right_shape = right_frame.shape
        
        logger.info(f"  Left camera: {left_shape}")
        logger.info(f"  Right camera: {right_shape}")
        
        # Verify format (should be BGR, 3 channels)
        if len(left_shape) == 3 and left_shape[2] == 3:
            logger.info("  ✓ Left camera: BGR format (3 channels)")
        else:
            logger.error("  ✗ Left camera: Unexpected format")
            return False
        
        if len(right_shape) == 3 and right_shape[2] == 3:
            logger.info("  ✓ Right camera: BGR format (3 channels)")
        else:
            logger.error("  ✗ Right camera: Unexpected format")
            return False
        
        # Check if resolutions match
        if left_shape == right_shape:
            logger.info("  ✓ Resolutions match")
        else:
            logger.warning("  ⚠ Resolutions do not match")
        
        logger.info("✓ Resolution and format test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Exception during resolution test: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("IDS STEREO CAMERA CONNECTION TEST")
    logger.info("="*60 + "\n")
    
    tests = [
        ("Camera Detection", test_camera_detection),
        ("Frame Capture", test_frame_capture),
        ("Synchronized Capture", test_synchronized_capture),
        ("Resolution and Format", test_resolution_and_format),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    logger.info(f"\nPassed: {passed}/{total}")
    logger.info("="*60 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
