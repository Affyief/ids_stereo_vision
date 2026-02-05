"""
IDS Peak Camera Interface (Modern GenICam/GenTL Interface)

This module provides a complete interface for IDS cameras using the IDS Peak SDK.
Uses GenICam node map for parameter configuration and supports serial number-based
camera identification.

Requirements:
    - IDS Peak SDK installed
    - ids_peak Python module
    - ids_peak_ipl for image processing
"""

import logging
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

# Try to import IDS Peak SDK
try:
    from ids_peak import ids_peak
    from ids_peak import ids_peak_ipl_extension
    IDS_PEAK_AVAILABLE = True
except ImportError:
    IDS_PEAK_AVAILABLE = False
    logging.warning("IDS Peak SDK not available. Please install from https://en.ids-imaging.com/downloads.html")


class IDSPeakCamera:
    """
    IDS Peak camera interface using GenICam node map.
    
    Supports:
        - Serial number-based identification
        - Device index fallback
        - GenICam parameter configuration
        - Buffer management
        - Proper resource cleanup
    """
    
    def __init__(
        self,
        serial_number: Optional[str] = None,
        device_index: int = 0,
        name: str = "Camera"
    ):
        """
        Initialize IDS Peak camera.
        
        Args:
            serial_number: Camera serial number (preferred identification method)
            device_index: Device index (fallback if serial not found)
            name: Human-readable camera name for logging
        """
        self.serial_number = serial_number
        self.device_index = device_index
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # IDS Peak objects
        self.device = None
        self.datastream = None
        self.nodemap = None
        self.is_acquiring = False
        
        # Camera info
        self.camera_info = {}
        
    def initialize(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        exposure_us: Optional[float] = None,
        gain: Optional[float] = None,
        pixel_format: str = "BGR8"
    ) -> bool:
        """
        Initialize camera with specified parameters.
        
        Args:
            width: Image width (None to use maximum)
            height: Image height (None to use maximum)
            exposure_us: Exposure time in microseconds
            gain: Gain value
            pixel_format: Pixel format (BGR8, RGB8, Mono8, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        if not IDS_PEAK_AVAILABLE:
            self.logger.error("IDS Peak SDK not available!")
            self.logger.error("Install from: https://en.ids-imaging.com/downloads.html")
            self.logger.error("Then: pip install /opt/ids/peak/lib/python*/ids_peak-*.whl")
            return False
        
        try:
            # Initialize IDS Peak library
            ids_peak.Library.Initialize()
            self.logger.info("IDS Peak library initialized")
            
            # Get device manager
            device_manager = ids_peak.DeviceManager.Instance()
            device_manager.Update()
            
            # Find and open device
            if not self._open_device(device_manager):
                return False
            
            # Get node map for parameter configuration
            self.nodemap = self.device.RemoteDevice().NodeMaps()[0]
            
            # Get camera info
            self._get_camera_info()
            
            # Configure camera parameters
            self._configure_resolution(width, height)
            
            if pixel_format:
                self._set_pixel_format(pixel_format)
            
            if exposure_us is not None:
                self.set_exposure(exposure_us)
            
            if gain is not None:
                self.set_gain(gain)
            
            # Setup data stream
            if not self._setup_datastream():
                return False
            
            self.logger.info(f"{self.name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            self.release()
            return False
    
    def _open_device(self, device_manager) -> bool:
        """
        Open device by serial number or device index.
        
        Args:
            device_manager: IDS Peak DeviceManager instance
            
        Returns:
            True if device opened successfully
        """
        try:
            # Get list of devices
            devices = device_manager.Devices()
            
            if len(devices) == 0:
                self.logger.error("No IDS Peak cameras detected!")
                self.logger.error("Check:")
                self.logger.error("  1. Camera is connected via USB 3.0")
                self.logger.error("  2. IDS Peak Cockpit can detect the camera")
                self.logger.error("  3. Camera permissions (Linux: sudo usermod -a -G video $USER)")
                return False
            
            # Find device by serial number or use index
            device_descriptor = None
            
            if self.serial_number:
                # Try to find by serial number
                for desc in devices:
                    if desc.SerialNumber() == self.serial_number:
                        device_descriptor = desc
                        self.logger.info(f"Found camera with serial {self.serial_number}")
                        break
                
                if device_descriptor is None:
                    self.logger.warning(f"Camera with serial {self.serial_number} not found")
                    self.logger.warning(f"Falling back to device index {self.device_index}")
            
            # Fallback to device index
            if device_descriptor is None:
                if self.device_index >= len(devices):
                    self.logger.error(f"Device index {self.device_index} out of range")
                    self.logger.error(f"Found {len(devices)} camera(s)")
                    return False
                
                device_descriptor = devices[self.device_index]
                self.logger.info(f"Using device index {self.device_index}")
            
            # Open device with control access
            self.device = device_descriptor.OpenDevice(ids_peak.DeviceAccessType_Control)
            self.logger.info(f"Opened device: {device_descriptor.ModelName()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open device: {e}")
            return False
    
    def _get_camera_info(self) -> None:
        """Get and store camera information."""
        try:
            if self.nodemap is None:
                return
            
            self.camera_info = {
                'model': self._get_node_value('DeviceModelName', 'Unknown'),
                'serial': self._get_node_value('DeviceSerialNumber', 'Unknown'),
                'vendor': self._get_node_value('DeviceVendorName', 'Unknown'),
                'firmware': self._get_node_value('DeviceFirmwareVersion', 'Unknown'),
            }
            
            self.logger.info(f"Camera Info: {self.camera_info}")
            
        except Exception as e:
            self.logger.warning(f"Could not get camera info: {e}")
    
    def _get_node_value(self, node_name: str, default: Any = None) -> Any:
        """
        Get value from GenICam node.
        
        Args:
            node_name: Name of the node
            default: Default value if node not found
            
        Returns:
            Node value or default
        """
        try:
            node = self.nodemap.FindNode(node_name)
            if node is not None:
                return node.Value()
        except:
            pass
        return default
    
    def _configure_resolution(self, width: Optional[int], height: Optional[int]) -> None:
        """
        Configure camera resolution.
        
        Args:
            width: Desired width (None for maximum)
            height: Desired height (None for maximum)
        """
        try:
            # Get maximum resolution if not specified
            if width is None:
                width_node = self.nodemap.FindNode("Width")
                width = width_node.Maximum()
            
            if height is None:
                height_node = self.nodemap.FindNode("Height")
                height = height_node.Maximum()
            
            # Set offset to 0 (start from top-left)
            offset_x_node = self.nodemap.FindNode("OffsetX")
            if offset_x_node is not None:
                offset_x_node.SetValue(0)
            
            offset_y_node = self.nodemap.FindNode("OffsetY")
            if offset_y_node is not None:
                offset_y_node.SetValue(0)
            
            # Set width and height
            width_node = self.nodemap.FindNode("Width")
            width_node.SetValue(width)
            
            height_node = self.nodemap.FindNode("Height")
            height_node.SetValue(height)
            
            self.logger.info(f"Resolution set to {width}x{height}")
            
        except Exception as e:
            self.logger.error(f"Failed to set resolution: {e}")
    
    def _set_pixel_format(self, pixel_format: str) -> None:
        """
        Set pixel format.
        
        Args:
            pixel_format: Pixel format string (e.g., "BGR8", "RGB8", "Mono8")
        """
        try:
            pixel_format_node = self.nodemap.FindNode("PixelFormat")
            if pixel_format_node is not None:
                # Get available formats
                available_formats = [entry.SymbolicValue() 
                                    for entry in pixel_format_node.Entries()]
                
                if pixel_format in available_formats:
                    pixel_format_node.SetCurrentEntry(pixel_format)
                    self.logger.info(f"Pixel format set to {pixel_format}")
                else:
                    self.logger.warning(f"Pixel format {pixel_format} not available")
                    self.logger.warning(f"Available formats: {available_formats}")
        except Exception as e:
            self.logger.error(f"Failed to set pixel format: {e}")
    
    def _setup_datastream(self) -> bool:
        """
        Setup data stream and allocate buffers.
        
        Returns:
            True if successful
        """
        try:
            # Get datastream
            datastreams = self.device.DataStreams()
            if len(datastreams) == 0:
                self.logger.error("No datastream available")
                return False
            
            self.datastream = datastreams[0].OpenDataStream()
            
            # Allocate and announce buffers
            # Use minimum required buffers for low latency
            num_buffers = 3
            
            for i in range(num_buffers):
                buffer = self.datastream.AllocAndAnnounceBuffer()
                self.datastream.QueueBuffer(buffer)
            
            self.logger.info(f"Allocated {num_buffers} buffers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup datastream: {e}")
            return False
    
    def start_acquisition(self) -> bool:
        """
        Start image acquisition.
        
        Returns:
            True if successful
        """
        try:
            if self.is_acquiring:
                return True
            
            # Start datastream
            self.datastream.StartAcquisition()
            
            # Start acquisition on camera
            acquisition_start = self.nodemap.FindNode("AcquisitionStart")
            acquisition_start.Execute()
            acquisition_start.WaitUntilDone()
            
            self.is_acquiring = True
            self.logger.info("Acquisition started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start acquisition: {e}")
            return False
    
    def stop_acquisition(self) -> None:
        """Stop image acquisition."""
        try:
            if not self.is_acquiring:
                return
            
            # Stop acquisition on camera
            try:
                acquisition_stop = self.nodemap.FindNode("AcquisitionStop")
                acquisition_stop.Execute()
            except:
                pass
            
            # Stop datastream
            if self.datastream is not None:
                try:
                    self.datastream.KillWait()
                    self.datastream.StopAcquisition()
                except:
                    pass
            
            self.is_acquiring = False
            self.logger.info("Acquisition stopped")
            
        except Exception as e:
            self.logger.warning(f"Error stopping acquisition: {e}")
    
    def capture_frame(self, timeout_ms: int = 5000) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            Frame as numpy array (BGR8/RGB8/Mono8) or None if failed
        """
        if not self.is_acquiring:
            self.logger.error("Acquisition not started")
            return None
        
        try:
            # Wait for buffer
            buffer = self.datastream.WaitForFinishedBuffer(timeout_ms)
            
            # Convert to IDS Peak IPL image
            ipl_image = ids_peak_ipl_extension.BufferToImage(buffer)
            
            # Convert to numpy array
            # The buffer contains raw pixel data
            numpy_array = ipl_image.get_numpy_3D()
            
            # Requeue buffer for next capture
            self.datastream.QueueBuffer(buffer)
            
            return numpy_array
            
        except Exception as e:
            self.logger.error(f"Failed to capture frame: {e}")
            return None
    
    def set_exposure(self, exposure_us: float) -> bool:
        """
        Set exposure time.
        
        Args:
            exposure_us: Exposure time in microseconds (NOTE: Peak uses microseconds!)
            
        Returns:
            True if successful
        """
        try:
            exposure_node = self.nodemap.FindNode("ExposureTime")
            if exposure_node is not None:
                # Clamp to valid range
                min_exp = exposure_node.Minimum()
                max_exp = exposure_node.Maximum()
                exposure_us = max(min_exp, min(max_exp, exposure_us))
                
                exposure_node.SetValue(exposure_us)
                self.logger.debug(f"Exposure set to {exposure_us} µs")
                return True
            else:
                self.logger.warning("ExposureTime node not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to set exposure: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """
        Set camera gain.
        
        Args:
            gain: Gain value
            
        Returns:
            True if successful
        """
        try:
            gain_node = self.nodemap.FindNode("Gain")
            if gain_node is not None:
                # Clamp to valid range
                min_gain = gain_node.Minimum()
                max_gain = gain_node.Maximum()
                gain = max(min_gain, min(max_gain, gain))
                
                gain_node.SetValue(gain)
                self.logger.debug(f"Gain set to {gain}")
                return True
            else:
                self.logger.warning("Gain node not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to set gain: {e}")
            return False
    
    def get_camera_info(self) -> Dict[str, Any]:
        """
        Get camera information.
        
        Returns:
            Dictionary with camera information
        """
        return self.camera_info.copy()
    
    def release(self) -> None:
        """Release camera resources and cleanup."""
        try:
            # Stop acquisition
            self.stop_acquisition()
            
            # Release datastream
            if self.datastream is not None:
                try:
                    # Flush and revoke buffers
                    self.datastream.Flush(ids_peak.DataStreamFlushMode_DiscardAll)
                    
                    # Revoke buffers
                    for buffer in self.datastream.AnnouncedBuffers():
                        self.datastream.RevokeBuffer(buffer)
                except:
                    pass
            
            # Close device
            if self.device is not None:
                try:
                    self.device = None
                except:
                    pass
            
            # Close library
            try:
                ids_peak.Library.Close()
            except:
                pass
            
            self.logger.info(f"{self.name} released")
            
        except Exception as e:
            self.logger.warning(f"Error during release: {e}")


class IDSPeakStereoSystem:
    """
    Stereo camera system using two IDS Peak cameras.
    
    Manages synchronized capture from both cameras and provides
    a unified interface for stereo vision applications.
    """
    
    def __init__(
        self,
        left_serial: Optional[str] = None,
        right_serial: Optional[str] = None,
        left_index: int = 0,
        right_index: int = 1
    ):
        """
        Initialize stereo camera system.
        
        Args:
            left_serial: Left camera serial number
            right_serial: Right camera serial number
            left_index: Left camera device index (fallback)
            right_index: Right camera device index (fallback)
        """
        self.logger = logging.getLogger(__name__)
        
        self.left_camera = IDSPeakCamera(
            serial_number=left_serial,
            device_index=left_index,
            name="Left"
        )
        
        self.right_camera = IDSPeakCamera(
            serial_number=right_serial,
            device_index=right_index,
            name="Right"
        )
    
    def initialize(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        exposure_us: Optional[float] = None,
        gain: Optional[float] = None,
        pixel_format: str = "BGR8"
    ) -> bool:
        """
        Initialize both cameras with same parameters.
        
        Args:
            width: Image width
            height: Image height
            exposure_us: Exposure time in microseconds
            gain: Gain value
            pixel_format: Pixel format
            
        Returns:
            True if both cameras initialized successfully
        """
        self.logger.info("Initializing stereo camera system...")
        
        # Initialize left camera
        if not self.left_camera.initialize(width, height, exposure_us, gain, pixel_format):
            self.logger.error("Failed to initialize left camera")
            return False
        
        # Initialize right camera
        if not self.right_camera.initialize(width, height, exposure_us, gain, pixel_format):
            self.logger.error("Failed to initialize right camera")
            self.left_camera.release()
            return False
        
        # Start acquisition on both cameras
        if not self.left_camera.start_acquisition():
            self.logger.error("Failed to start left camera acquisition")
            self.release()
            return False
        
        if not self.right_camera.start_acquisition():
            self.logger.error("Failed to start right camera acquisition")
            self.release()
            return False
        
        self.logger.info("Stereo camera system initialized successfully")
        return True
    
    def capture_stereo_pair(
        self,
        timeout_ms: int = 5000
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture synchronized frames from both cameras.
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            Tuple of (left_frame, right_frame)
        """
        left_frame = self.left_camera.capture_frame(timeout_ms)
        right_frame = self.right_camera.capture_frame(timeout_ms)
        
        return left_frame, right_frame
    
    def set_exposure(self, exposure_us: float) -> bool:
        """Set exposure for both cameras."""
        left_ok = self.left_camera.set_exposure(exposure_us)
        right_ok = self.right_camera.set_exposure(exposure_us)
        return left_ok and right_ok
    
    def set_gain(self, gain: float) -> bool:
        """Set gain for both cameras."""
        left_ok = self.left_camera.set_gain(gain)
        right_ok = self.right_camera.set_gain(gain)
        return left_ok and right_ok
    
    def release(self) -> None:
        """Release both cameras."""
        self.left_camera.release()
        self.right_camera.release()
        self.logger.info("Stereo camera system released")


def list_ids_peak_cameras() -> List[Dict[str, str]]:
    """
    List all available IDS Peak cameras.
    
    Returns:
        List of dictionaries with camera information
    """
    if not IDS_PEAK_AVAILABLE:
        return []
    
    cameras = []
    
    try:
        # Initialize library
        ids_peak.Library.Initialize()
        
        # Get device manager
        device_manager = ids_peak.DeviceManager.Instance()
        device_manager.Update()
        
        # Get devices
        devices = device_manager.Devices()
        
        for i, device_desc in enumerate(devices):
            camera_info = {
                'index': i,
                'model': device_desc.ModelName(),
                'serial': device_desc.SerialNumber(),
                'interface': device_desc.ParentInterface().DisplayName(),
                'accessible': device_desc.IsOpenable()
            }
            cameras.append(camera_info)
        
        # Close library
        ids_peak.Library.Close()
        
    except Exception as e:
        logging.error(f"Error listing cameras: {e}")
    
    return cameras


def create_stereo_camera_from_config(config: Dict[str, Any]) -> IDSPeakStereoSystem:
    """
    Create stereo camera system from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured IDSPeakStereoSystem instance
    """
    cameras_config = config['cameras']
    
    # Get serial numbers or device indices
    use_serial = cameras_config.get('use_serial_numbers', False)
    
    left_serial = None
    right_serial = None
    left_index = 0
    right_index = 1
    
    if use_serial:
        left_serial = cameras_config['left_camera'].get('serial_number')
        right_serial = cameras_config['right_camera'].get('serial_number')
    
    left_index = cameras_config['left_camera'].get('device_index', 0)
    right_index = cameras_config['right_camera'].get('device_index', 1)
    
    return IDSPeakStereoSystem(
        left_serial=left_serial,
        right_serial=right_serial,
        left_index=left_index,
        right_index=right_index
    )
