"""
Camera Interface Module for IDS uEye Cameras
Provides a wrapper for the IDS uEye SDK to interface with IDS u3-3680xcp-c cameras.
"""

import numpy as np
from pyueye import ueye
import logging
from typing import Tuple, Optional, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDSCamera:
    """
    Wrapper class for IDS uEye camera interface.
    
    Supports IDS u3-3680xcp-c cameras via the PyuEye SDK.
    """
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize IDS camera.
        
        Args:
            camera_id: Camera device ID (0 for first camera, 1 for second, etc.)
        """
        self.camera_id = camera_id
        self.h_cam = ueye.HIDS(camera_id)
        self.img_buffer = None
        self.mem_id = ueye.int()
        self.pitch = ueye.INT()
        self.width = 0
        self.height = 0
        self.bits_per_pixel = 24  # RGB8
        self.is_initialized = False
        
    def initialize(self, width: int = 2592, height: int = 1944) -> bool:
        """
        Initialize the camera and allocate memory.
        
        Args:
            width: Image width in pixels (default: 2592)
            height: Image height in pixels (default: 1944)
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize camera
            ret = ueye.is_InitCamera(self.h_cam, None)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"Failed to initialize camera {self.camera_id}: {ret}")
                return False
            
            # Set color mode
            ret = ueye.is_SetColorMode(self.h_cam, ueye.IS_CM_BGR8_PACKED)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"Failed to set color mode: {ret}")
                return False
            
            # Set AOI (Area of Interest)
            rect_aoi = ueye.IS_RECT()
            rect_aoi.s32X = ueye.int(0)
            rect_aoi.s32Y = ueye.int(0)
            rect_aoi.s32Width = ueye.int(width)
            rect_aoi.s32Height = ueye.int(height)
            
            ret = ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
            if ret != ueye.IS_SUCCESS:
                logger.warning(f"Failed to set AOI, using default: {ret}")
            
            # Get actual image size
            rect_aoi_get = ueye.IS_RECT()
            ueye.is_AOI(self.h_cam, ueye.IS_AOI_IMAGE_GET_AOI, rect_aoi_get, ueye.sizeof(rect_aoi_get))
            self.width = rect_aoi_get.s32Width.value
            self.height = rect_aoi_get.s32Height.value
            
            logger.info(f"Camera {self.camera_id} initialized: {self.width}x{self.height}")
            
            # Allocate image memory
            self.img_buffer = ueye.c_mem_p()
            ret = ueye.is_AllocImageMem(
                self.h_cam,
                self.width,
                self.height,
                self.bits_per_pixel,
                self.img_buffer,
                self.mem_id
            )
            if ret != ueye.IS_SUCCESS:
                logger.error(f"Failed to allocate image memory: {ret}")
                return False
            
            # Set active memory
            ret = ueye.is_SetImageMem(self.h_cam, self.img_buffer, self.mem_id)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"Failed to set active memory: {ret}")
                return False
            
            # Enable capturing
            ret = ueye.is_CaptureVideo(self.h_cam, ueye.IS_DONT_WAIT)
            if ret != ueye.IS_SUCCESS:
                logger.error(f"Failed to start video capture: {ret}")
                return False
            
            # Get pitch
            ueye.is_InquireImageMem(
                self.h_cam,
                self.img_buffer,
                self.mem_id,
                None, None,
                None,
                self.pitch
            )
            
            self.is_initialized = True
            logger.info(f"Camera {self.camera_id} ready to capture")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera {self.camera_id}: {e}")
            return False
    
    def set_exposure(self, exposure_ms: float) -> bool:
        """
        Set camera exposure time.
        
        Args:
            exposure_ms: Exposure time in milliseconds
            
        Returns:
            True if successful, False otherwise
        """
        exposure = ueye.double(exposure_ms)
        ret = ueye.is_Exposure(
            self.h_cam,
            ueye.IS_EXPOSURE_CMD_SET_EXPOSURE,
            exposure,
            ueye.sizeof(exposure)
        )
        if ret == ueye.IS_SUCCESS:
            logger.info(f"Camera {self.camera_id} exposure set to {exposure_ms}ms")
            return True
        else:
            logger.error(f"Failed to set exposure: {ret}")
            return False
    
    def set_gain(self, gain: int) -> bool:
        """
        Set camera hardware gain.
        
        Args:
            gain: Gain value (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        ret = ueye.is_SetHardwareGain(
            self.h_cam,
            int(gain),
            ueye.IS_IGNORE_PARAMETER,
            ueye.IS_IGNORE_PARAMETER,
            ueye.IS_IGNORE_PARAMETER
        )
        if ret == ueye.IS_SUCCESS:
            logger.info(f"Camera {self.camera_id} gain set to {gain}")
            return True
        else:
            logger.error(f"Failed to set gain: {ret}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Numpy array containing the image (BGR format), or None if capture failed
        """
        if not self.is_initialized:
            logger.error("Camera not initialized")
            return None
        
        try:
            # Create numpy array from image buffer
            array = ueye.get_data(
                self.img_buffer,
                self.width,
                self.height,
                self.bits_per_pixel,
                self.pitch,
                copy=True
            )
            
            # Reshape to image
            frame = np.reshape(array, (self.height.value, self.width.value, 3))
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing frame from camera {self.camera_id}: {e}")
            return None
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera information.
        
        Returns:
            Dictionary containing camera information
        """
        if not self.is_initialized:
            return {}
        
        cam_info = ueye.CAMINFO()
        ret = ueye.is_GetCameraInfo(self.h_cam, cam_info)
        
        sensor_info = ueye.SENSORINFO()
        ret2 = ueye.is_GetSensorInfo(self.h_cam, sensor_info)
        
        info = {
            'camera_id': self.camera_id,
            'width': self.width,
            'height': self.height,
            'bits_per_pixel': self.bits_per_pixel,
            'is_initialized': self.is_initialized
        }
        
        if ret == ueye.IS_SUCCESS:
            info['serial'] = cam_info.SerNo.decode('utf-8')
            info['device_id'] = cam_info.DeviceID.decode('utf-8')
        
        if ret2 == ueye.IS_SUCCESS:
            info['sensor_name'] = sensor_info.strSensorName.decode('utf-8')
            info['max_width'] = sensor_info.nMaxWidth
            info['max_height'] = sensor_info.nMaxHeight
        
        return info
    
    def release(self):
        """
        Release camera resources and clean up.
        """
        if self.is_initialized:
            try:
                # Stop video capture
                ueye.is_StopLiveVideo(self.h_cam, ueye.IS_WAIT)
                
                # Free image memory
                if self.img_buffer is not None:
                    ueye.is_FreeImageMem(self.h_cam, self.img_buffer, self.mem_id)
                
                # Exit camera
                ueye.is_ExitCamera(self.h_cam)
                
                self.is_initialized = False
                logger.info(f"Camera {self.camera_id} released")
                
            except Exception as e:
                logger.error(f"Error releasing camera {self.camera_id}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False


class StereoCameraSystem:
    """
    Manages two IDS cameras for stereo vision.
    """
    
    def __init__(self, left_id: int = 0, right_id: int = 1):
        """
        Initialize stereo camera system.
        
        Args:
            left_id: Device ID for left camera
            right_id: Device ID for right camera
        """
        self.left_camera = IDSCamera(left_id)
        self.right_camera = IDSCamera(right_id)
        self.is_initialized = False
    
    def initialize(self, width: int = 2592, height: int = 1944,
                   exposure: float = 10.0, gain: int = 1) -> bool:
        """
        Initialize both cameras with synchronized settings.
        
        Args:
            width: Image width
            height: Image height
            exposure: Exposure time in milliseconds
            gain: Hardware gain value
            
        Returns:
            True if both cameras initialized successfully
        """
        # Initialize left camera
        if not self.left_camera.initialize(width, height):
            logger.error("Failed to initialize left camera")
            return False
        
        # Initialize right camera
        if not self.right_camera.initialize(width, height):
            logger.error("Failed to initialize right camera")
            self.left_camera.release()
            return False
        
        # Set exposure for both cameras
        self.left_camera.set_exposure(exposure)
        self.right_camera.set_exposure(exposure)
        
        # Set gain for both cameras
        self.left_camera.set_gain(gain)
        self.right_camera.set_gain(gain)
        
        self.is_initialized = True
        logger.info("Stereo camera system initialized successfully")
        return True
    
    def capture_stereo_pair(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture synchronized frames from both cameras.
        
        Returns:
            Tuple of (left_frame, right_frame), or (None, None) if capture failed
        """
        if not self.is_initialized:
            logger.error("Stereo system not initialized")
            return None, None
        
        left_frame = self.left_camera.capture_frame()
        right_frame = self.right_camera.capture_frame()
        
        return left_frame, right_frame
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about both cameras.
        
        Returns:
            Dictionary with left and right camera info
        """
        return {
            'left': self.left_camera.get_camera_info(),
            'right': self.right_camera.get_camera_info()
        }
    
    def release(self):
        """Release both cameras."""
        self.left_camera.release()
        self.right_camera.release()
        self.is_initialized = False
        logger.info("Stereo camera system released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
