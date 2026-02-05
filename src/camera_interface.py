"""
Camera interface for IDS U3-3680XCP-C cameras.

This module provides a unified interface for IDS cameras with fallback to OpenCV.
Supports both PyuEye SDK and standard USB3 Vision interface.
"""

import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2

# Try to import PyuEye SDK
try:
    from pyueye import ueye
    PYUEYE_AVAILABLE = True
except ImportError:
    PYUEYE_AVAILABLE = False
    logging.warning("PyuEye SDK not available. Will use OpenCV fallback.")


class CameraInterface:
    """Base camera interface class."""
    
    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (1296, 972),
        framerate: int = 30
    ):
        """
        Initialize camera interface.
        
        Args:
            camera_id: Camera device ID or index
            resolution: Desired resolution (width, height)
            framerate: Desired frame rate
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.framerate = framerate
        self.is_open = False
        self.logger = logging.getLogger(__name__)
    
    def open(self) -> bool:
        """
        Open camera connection.
        
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError
    
    def close(self) -> None:
        """Close camera connection."""
        raise NotImplementedError
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Captured frame as numpy array or None if failed
        """
        raise NotImplementedError
    
    def set_exposure(self, exposure_ms: float) -> bool:
        """
        Set camera exposure time.
        
        Args:
            exposure_ms: Exposure time in milliseconds
            
        Returns:
            True if successful
        """
        raise NotImplementedError
    
    def set_gain(self, gain: float) -> bool:
        """
        Set camera gain.
        
        Args:
            gain: Gain value
            
        Returns:
            True if successful
        """
        raise NotImplementedError


class IDSCamera(CameraInterface):
    """IDS camera interface using PyuEye SDK."""
    
    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (1296, 972),
        framerate: int = 30
    ):
        """
        Initialize IDS camera.
        
        Args:
            camera_id: Camera device ID
            resolution: Desired resolution (width, height)
            framerate: Desired frame rate
        """
        super().__init__(camera_id, resolution, framerate)
        self.cam_handle = ueye.HIDS(camera_id)
        self.img_buffer = None
        self.mem_id = ueye.INT()
    
    def open(self) -> bool:
        """Open IDS camera connection."""
        try:
            # Initialize camera
            ret = ueye.is_InitCamera(self.cam_handle, None)
            if ret != ueye.IS_SUCCESS:
                self.logger.error(f"Failed to initialize IDS camera {self.camera_id}")
                return False
            
            # Set color mode
            ret = ueye.is_SetColorMode(self.cam_handle, ueye.IS_CM_BGR8_PACKED)
            
            # Set AOI (Area of Interest) - resolution
            rect_aoi = ueye.IS_RECT()
            rect_aoi.s32X = ueye.INT(0)
            rect_aoi.s32Y = ueye.INT(0)
            rect_aoi.s32Width = ueye.INT(self.resolution[0])
            rect_aoi.s32Height = ueye.INT(self.resolution[1])
            ueye.is_AOI(self.cam_handle, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
            
            # Allocate memory
            width = self.resolution[0]
            height = self.resolution[1]
            bitspixel = 24  # for BGR8
            
            self.img_buffer = ueye.c_mem_p()
            ueye.is_AllocImageMem(
                self.cam_handle,
                width, height, bitspixel,
                self.img_buffer, self.mem_id
            )
            
            # Set active memory
            ueye.is_SetImageMem(self.cam_handle, self.img_buffer, self.mem_id)
            
            # Enable capturing
            ueye.is_CaptureVideo(self.cam_handle, ueye.IS_DONT_WAIT)
            
            self.is_open = True
            self.logger.info(f"IDS camera {self.camera_id} opened successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open IDS camera: {e}")
            return False
    
    def close(self) -> None:
        """Close IDS camera connection."""
        if self.is_open:
            try:
                # Stop capture
                ueye.is_StopLiveVideo(self.cam_handle, ueye.IS_FORCE_VIDEO_STOP)
                
                # Free memory
                if self.img_buffer:
                    ueye.is_FreeImageMem(self.cam_handle, self.img_buffer, self.mem_id)
                
                # Close camera
                ueye.is_ExitCamera(self.cam_handle)
                
                self.is_open = False
                self.logger.info(f"IDS camera {self.camera_id} closed")
            except Exception as e:
                self.logger.error(f"Error closing IDS camera: {e}")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame from IDS camera."""
        if not self.is_open:
            return None
        
        try:
            # Wait for image
            ueye.is_WaitForNextImage(
                self.cam_handle,
                1000,  # timeout in ms
                self.img_buffer,
                self.mem_id
            )
            
            # Create numpy array from buffer
            array = ueye.get_data(
                self.img_buffer,
                self.resolution[0],
                self.resolution[1],
                24,  # bits per pixel
                3,   # bytes per pixel
                copy=True
            )
            
            # Reshape to image
            frame = np.reshape(array, (self.resolution[1], self.resolution[0], 3))
            
            # Unlock buffer
            ueye.is_UnlockSeqBuf(self.cam_handle, self.mem_id, self.img_buffer)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Failed to capture frame: {e}")
            return None
    
    def set_exposure(self, exposure_ms: float) -> bool:
        """Set exposure time for IDS camera."""
        if not self.is_open:
            return False
        
        try:
            exposure = ueye.DOUBLE(exposure_ms)
            ret = ueye.is_Exposure(
                self.cam_handle,
                ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
                exposure,
                ueye.sizeof(exposure)
            )
            return ret == ueye.IS_SUCCESS
        except Exception as e:
            self.logger.error(f"Failed to set exposure: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set gain for IDS camera."""
        if not self.is_open:
            return False
        
        try:
            # IDS gain is usually in percentage (0-100)
            gain_value = ueye.INT(int(gain * 100))
            ret = ueye.is_SetHardwareGain(
                self.cam_handle,
                gain_value,
                ueye.IS_IGNORE_PARAMETER,
                ueye.IS_IGNORE_PARAMETER,
                ueye.IS_IGNORE_PARAMETER
            )
            return ret == ueye.IS_SUCCESS
        except Exception as e:
            self.logger.error(f"Failed to set gain: {e}")
            return False


class OpenCVCamera(CameraInterface):
    """OpenCV camera interface (fallback)."""
    
    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (1296, 972),
        framerate: int = 30
    ):
        """
        Initialize OpenCV camera.
        
        Args:
            camera_id: Camera device ID
            resolution: Desired resolution (width, height)
            framerate: Desired frame rate
        """
        super().__init__(camera_id, resolution, framerate)
        self.cap = None
    
    def open(self) -> bool:
        """Open camera using OpenCV."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Set framerate
            self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
            
            self.is_open = True
            self.logger.info(f"OpenCV camera {self.camera_id} opened successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open OpenCV camera: {e}")
            return False
    
    def close(self) -> None:
        """Close OpenCV camera."""
        if self.cap is not None:
            self.cap.release()
            self.is_open = False
            self.logger.info(f"OpenCV camera {self.camera_id} closed")
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture frame from OpenCV camera."""
        if not self.is_open or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def set_exposure(self, exposure_ms: float) -> bool:
        """Set exposure for OpenCV camera."""
        if not self.is_open or self.cap is None:
            return False
        
        # Convert milliseconds to camera-specific value
        # This is camera-dependent and may not work on all cameras
        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_ms)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to set exposure: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set gain for OpenCV camera."""
        if not self.is_open or self.cap is None:
            return False
        
        try:
            self.cap.set(cv2.CAP_PROP_GAIN, gain)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to set gain: {e}")
            return False


class StereoCamera:
    """Stereo camera system with two synchronized cameras."""
    
    def __init__(
        self,
        left_id: int = 0,
        right_id: int = 1,
        resolution: Tuple[int, int] = (1296, 972),
        framerate: int = 30,
        use_ids_sdk: bool = True
    ):
        """
        Initialize stereo camera system.
        
        Args:
            left_id: Left camera device ID
            right_id: Right camera device ID
            resolution: Resolution for both cameras
            framerate: Frame rate for both cameras
            use_ids_sdk: Try to use IDS SDK if available
        """
        self.logger = logging.getLogger(__name__)
        
        # Choose camera interface
        if use_ids_sdk and PYUEYE_AVAILABLE:
            self.left_camera = IDSCamera(left_id, resolution, framerate)
            self.right_camera = IDSCamera(right_id, resolution, framerate)
            self.logger.info("Using IDS SDK for cameras")
        else:
            self.left_camera = OpenCVCamera(left_id, resolution, framerate)
            self.right_camera = OpenCVCamera(right_id, resolution, framerate)
            self.logger.info("Using OpenCV for cameras")
    
    def open(self) -> bool:
        """
        Open both cameras.
        
        Returns:
            True if both cameras opened successfully
        """
        left_ok = self.left_camera.open()
        right_ok = self.right_camera.open()
        
        if not left_ok:
            self.logger.error("Failed to open left camera")
        if not right_ok:
            self.logger.error("Failed to open right camera")
        
        return left_ok and right_ok
    
    def close(self) -> None:
        """Close both cameras."""
        self.left_camera.close()
        self.right_camera.close()
    
    def capture_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture synchronized frames from both cameras.
        
        Returns:
            Tuple of (left_frame, right_frame), or (None, None) if failed
        """
        left_frame = self.left_camera.capture_frame()
        right_frame = self.right_camera.capture_frame()
        
        return left_frame, right_frame
    
    def set_exposure(self, exposure_ms: float) -> bool:
        """
        Set exposure for both cameras.
        
        Args:
            exposure_ms: Exposure time in milliseconds
            
        Returns:
            True if both cameras set successfully
        """
        left_ok = self.left_camera.set_exposure(exposure_ms)
        right_ok = self.right_camera.set_exposure(exposure_ms)
        return left_ok and right_ok
    
    def set_gain(self, gain: float) -> bool:
        """
        Set gain for both cameras.
        
        Args:
            gain: Gain value
            
        Returns:
            True if both cameras set successfully
        """
        left_ok = self.left_camera.set_gain(gain)
        right_ok = self.right_camera.set_gain(gain)
        return left_ok and right_ok


def create_stereo_camera(config: Dict[str, Any]) -> StereoCamera:
    """
    Create stereo camera system from configuration.
    
    Args:
        config: Camera configuration dictionary
        
    Returns:
        Configured StereoCamera instance
    """
    resolution = (
        config['cameras']['resolution']['width'],
        config['cameras']['resolution']['height']
    )
    framerate = config['cameras']['framerate']
    
    left_id = config['cameras']['left_camera']['device_id']
    right_id = config['cameras']['right_camera']['device_id']
    
    return StereoCamera(
        left_id=left_id,
        right_id=right_id,
        resolution=resolution,
        framerate=framerate,
        use_ids_sdk=PYUEYE_AVAILABLE
    )
