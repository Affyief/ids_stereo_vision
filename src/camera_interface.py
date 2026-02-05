"""
IDS Peak Camera Interface for Stereo Vision
Uses modern IDS Peak SDK with GenICam/GenTL
"""

import logging
from typing import Tuple, Optional, List
import numpy as np

try:
    from ids_peak import ids_peak as peak
    from ids_peak_ipl import ids_peak_ipl as ipl
    IDS_PEAK_AVAILABLE = True
except ImportError:
    IDS_PEAK_AVAILABLE = False
    logging.warning("IDS Peak SDK not available. Install from: https://en.ids-imaging.com/downloads.html")

logger = logging.getLogger(__name__)


class IDSPeakCamera:
    """
    Modern IDS Peak camera interface
    Supports USB3 Vision cameras with GenICam
    """
    
    def __init__(self, serial_number: Optional[str] = None, device_index: int = 0):
        self.serial_number = serial_number
        self.device_index = device_index
        self.device = None
        self.datastream = None
        self.nodemap_remote_device = None
        self.buffers = []
        
    def initialize(self, width: int = 2592, height: int = 1944, 
                   exposure_us: float = 10000, gain_db: float = 0.0,
                   framerate: float = 30.0, pixel_format: str = "BGR8") -> bool:
        """
        Initialize IDS Peak camera with GenICam parameters
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            exposure_us: Exposure time in microseconds
            gain_db: Gain in dB
            framerate: Target frame rate in fps
            pixel_format: Pixel format ("BGR8", "RGB8", "BayerRG8", or "Mono8")
        """
        
        if not IDS_PEAK_AVAILABLE:
            logger.error("IDS Peak SDK not installed!")
            return False
        
        try:
            # Initialize Peak library
            peak.Library.Initialize()
            
            # Create device manager
            device_manager = peak.DeviceManager.Instance()
            device_manager.Update()
            
            # List available devices
            devices = device_manager.Devices()
            
            if not devices:
                logger.error("No IDS Peak cameras found!")
                logger.info("Check: 1) Cameras connected 2) IDS Peak Cockpit can see them 3) Permissions")
                return False
            
            # Find and open device
            target_device = None
            
            if self.serial_number:
                # Find by serial number
                for dev in devices:
                    if dev.SerialNumber() == self.serial_number:
                        target_device = dev
                        logger.info(f"Found camera by serial: {self.serial_number}")
                        break
                
                if not target_device:
                    logger.error(f"Camera with serial {self.serial_number} not found")
                    logger.info(f"Available cameras: {[d.SerialNumber() for d in devices]}")
                    return False
            else:
                # Use device index
                if self.device_index >= len(devices):
                    logger.error(f"Device index {self.device_index} out of range (found {len(devices)} cameras)")
                    return False
                
                target_device = devices[self.device_index]
                logger.info(f"Using camera index {self.device_index}: {target_device.SerialNumber()}")
            
            # Open device
            self.device = target_device.OpenDevice(peak.DeviceAccessType_Control)
            
            # Get remote device nodemap (camera features)
            self.nodemap_remote_device = self.device.RemoteDevice().NodeMaps()[0]
            
            # Configure camera
            self._configure_camera(width, height, exposure_us, gain_db, framerate, pixel_format)
            
            # Setup data stream
            self._setup_datastream()
            
            # Start acquisition
            self._start_acquisition()
            
            logger.info(f"✓ IDS Peak camera initialized: {target_device.ModelName()} (S/N: {target_device.SerialNumber()})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize IDS Peak camera: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _configure_camera(self, width: int, height: int, exposure_us: float, 
                          gain_db: float, framerate: float, pixel_format: str = "BGR8"):
        """Configure camera parameters via GenICam nodemap"""
        
        nm = self.nodemap_remote_device
        
        # Set pixel format
        try:
            pixel_format_node = nm.FindNode("PixelFormat")
            pixel_format_entries = pixel_format_node.Entries()
            
            # Build a set of available formats for O(1) lookup
            available_formats = {entry.SymbolicValue() for entry in pixel_format_entries}
            
            # Try requested format first
            format_found = False
            if pixel_format in available_formats:
                for entry in pixel_format_entries:
                    if entry.SymbolicValue() == pixel_format:
                        pixel_format_node.SetCurrentEntry(entry)
                        logger.info(f"Set pixel format: {pixel_format}")
                        format_found = True
                        break
            
            # If requested format not found, try fallback options (excluding already tried format)
            if not format_found:
                logger.warning(f"Requested format '{pixel_format}' not available, trying fallbacks...")
                fallback_formats = ["BGR8", "BayerRG8", "BayerBG8", "Mono8"]
                
                for fallback in fallback_formats:
                    if fallback != pixel_format and fallback in available_formats:
                        for entry in pixel_format_entries:
                            if entry.SymbolicValue() == fallback:
                                pixel_format_node.SetCurrentEntry(entry)
                                logger.info(f"Set pixel format (fallback): {fallback}")
                                format_found = True
                                break
                        if format_found:
                            break
        except Exception as e:
            logger.warning(f"Could not set pixel format: {e}")
        
        # Set width and height
        try:
            nm.FindNode("Width").SetValue(width)
            nm.FindNode("Height").SetValue(height)
            logger.info(f"Set resolution: {width}x{height}")
        except Exception as e:
            logger.warning(f"Could not set resolution: {e}")
        
        # Set exposure time
        try:
            # Set exposure mode to manual
            exposure_auto = nm.FindNode("ExposureAuto")
            if exposure_auto:
                exposure_auto.SetCurrentEntry("Off")
            
            exposure_time_node = nm.FindNode("ExposureTime")
            exposure_time_node.SetValue(exposure_us)
            logger.info(f"Set exposure: {exposure_us} µs")
        except Exception as e:
            logger.warning(f"Could not set exposure: {e}")
        
        # Set gain
        try:
            # Set gain mode to manual
            gain_auto = nm.FindNode("GainAuto")
            if gain_auto:
                gain_auto.SetCurrentEntry("Off")
            
            gain_node = nm.FindNode("Gain")
            gain_node.SetValue(gain_db)
            logger.info(f"Set gain: {gain_db} dB")
        except Exception as e:
            logger.warning(f"Could not set gain: {e}")
        
        # Set frame rate
        try:
            # Enable frame rate control
            fps_enable = nm.FindNode("AcquisitionFrameRateEnable")
            if fps_enable:
                fps_enable.SetValue(True)
            
            fps_node = nm.FindNode("AcquisitionFrameRate")
            if fps_node:
                fps_node.SetValue(framerate)
                logger.info(f"Set frame rate: {framerate} fps")
        except Exception as e:
            logger.warning(f"Could not set frame rate: {e}")
    
    def _setup_datastream(self):
        """Setup data stream and allocate buffers"""
        
        # Open data stream
        datastreams = self.device.DataStreams()
        if not datastreams:
            raise RuntimeError("No data streams available")
        
        self.datastream = datastreams[0].OpenDataStream()
        
        # Get payload size
        payload_size = self.nodemap_remote_device.FindNode("PayloadSize").Value()
        
        # Allocate and announce buffers (use 10 buffers for smooth streaming)
        for i in range(10):
            buffer = self.datastream.AllocAndAnnounceBuffer(payload_size)
            self.buffers.append(buffer)
            self.datastream.QueueBuffer(buffer)
        
        logger.info(f"Allocated {len(self.buffers)} buffers ({payload_size} bytes each)")
    
    def _start_acquisition(self):
        """Start camera acquisition"""
        
        # Start data stream
        self.datastream.StartAcquisition()
        
        # Lock parameters that shouldn't change during acquisition
        self.nodemap_remote_device.FindNode("TLParamsLocked").SetValue(1)
        
        # Start acquisition on camera
        self.nodemap_remote_device.FindNode("AcquisitionStart").Execute()
        self.nodemap_remote_device.FindNode("AcquisitionStart").WaitUntilDone()
        
        logger.info("✓ Acquisition started")
    
    def capture_frame(self, timeout_ms: int = 5000) -> Optional[np.ndarray]:
        """
        Capture a single frame
        
        Args:
            timeout_ms: Timeout in milliseconds
            
        Returns:
            numpy array with image data (BGR8 or Mono8) or None if failed
        """
        
        if not self.datastream:
            logger.error("Data stream not initialized")
            return None
        
        try:
            # Wait for filled buffer
            buffer = self.datastream.WaitForFinishedBuffer(timeout_ms)
            
            # Convert to IDS Peak IPL image
            ipl_image = ipl.Image.CreateFromSizeAndBuffer(
                buffer.PixelFormat(),
                buffer.BasePtr(),
                buffer.Size(),
                buffer.Width(),
                buffer.Height()
            )
            
            # Convert to numpy array
            # Check if color or mono
            if ipl_image.PixelFormat().NumChannels() == 3:
                # Color image (BGR)
                numpy_image = ipl_image.get_numpy_3D()
            else:
                # Mono image
                numpy_image = ipl_image.get_numpy_1D().reshape(
                    (ipl_image.Height(), ipl_image.Width())
                )
            
            # Requeue buffer for next capture
            self.datastream.QueueBuffer(buffer)
            
            return numpy_image
            
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    def get_camera_info(self) -> dict:
        """Get camera information"""
        
        if not self.device:
            return {}
        
        try:
            return {
                'model': self.device.ModelName(),
                'serial': self.device.SerialNumber(),
                'vendor': self.device.VendorName(),
                'width': self.nodemap_remote_device.FindNode("Width").Value(),
                'height': self.nodemap_remote_device.FindNode("Height").Value(),
                'pixel_format': self.nodemap_remote_device.FindNode("PixelFormat").CurrentEntry().SymbolicValue(),
                'exposure': self.nodemap_remote_device.FindNode("ExposureTime").Value(),
                'gain': self.nodemap_remote_device.FindNode("Gain").Value(),
            }
        except Exception as e:
            logger.warning(f"Could not get camera info: {e}")
            return {}
    
    def release(self):
        """Stop acquisition and release camera resources"""
        
        try:
            if self.datastream:
                # Stop acquisition
                self.nodemap_remote_device.FindNode("AcquisitionStop").Execute()
                
                # Stop data stream
                self.datastream.KillWait()
                self.datastream.StopAcquisition(peak.AcquisitionStopMode_Default)
                
                # Flush and revoke buffers
                self.datastream.Flush(peak.DataStreamFlushMode_DiscardAll)
                
                for buffer in self.buffers:
                    self.datastream.RevokeBuffer(buffer)
                
                self.buffers.clear()
            
            self.datastream = None
            self.device = None
            
            logger.info("✓ Camera released")
            
        except Exception as e:
            logger.warning(f"Error releasing camera: {e}")


class StereoCameraSystem:
    """
    Stereo camera system using two IDS Peak cameras
    """
    
    def __init__(self, left_id, right_id):
        """
        Initialize stereo camera system
        
        Args:
            left_id: Serial number (str) or device index (int) for left camera
            right_id: Serial number (str) or device index (int) for right camera
        """
        
        # Create camera instances
        if isinstance(left_id, str):
            self.left_camera = IDSPeakCamera(serial_number=left_id)
        else:
            self.left_camera = IDSPeakCamera(device_index=left_id)
        
        if isinstance(right_id, str):
            self.right_camera = IDSPeakCamera(serial_number=right_id)
        else:
            self.right_camera = IDSPeakCamera(device_index=right_id)
    
    def initialize(self, width: int = 2592, height: int = 1944,
                   exposure_us: float = 10000, gain_db: float = 0.0,
                   framerate: float = 30.0, pixel_format: str = "BGR8") -> bool:
        """
        Initialize both cameras with identical settings
        
        Args:
            width: Image width
            height: Image height
            exposure_us: Exposure time in microseconds
            gain_db: Gain in dB
            framerate: Target frame rate
            pixel_format: Pixel format ("BGR8", "RGB8", "BayerRG8", or "Mono8")
        """
        
        logger.info("Initializing left camera...")
        left_ok = self.left_camera.initialize(width, height, exposure_us, gain_db, framerate, pixel_format)
        
        if not left_ok:
            logger.error("Failed to initialize left camera")
            return False
        
        logger.info("Initializing right camera...")
        right_ok = self.right_camera.initialize(width, height, exposure_us, gain_db, framerate, pixel_format)
        
        if not right_ok:
            logger.error("Failed to initialize right camera")
            self.left_camera.release()
            return False
        
        logger.info("✓ Both cameras initialized successfully")
        return True
    
    def capture_stereo_pair(self, timeout_ms: int = 5000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Capture synchronized pair of frames
        
        Returns:
            Tuple of (left_frame, right_frame) as numpy arrays
        """
        
        left_frame = self.left_camera.capture_frame(timeout_ms)
        right_frame = self.right_camera.capture_frame(timeout_ms)
        
        return left_frame, right_frame
    
    def get_camera_info(self) -> dict:
        """Get information about both cameras"""
        return {
            'left': self.left_camera.get_camera_info(),
            'right': self.right_camera.get_camera_info()
        }
    
    def release(self):
        """Release both cameras"""
        logger.info("Releasing cameras...")
        self.left_camera.release()
        self.right_camera.release()
        
        # Close Peak library
        try:
            peak.Library.Close()
        except Exception as e:
            logger.warning(f"Error closing Peak library: {e}")


def list_ids_peak_cameras() -> List[dict]:
    """
    List all available IDS Peak cameras
    
    Returns:
        List of camera info dictionaries
    """
    
    if not IDS_PEAK_AVAILABLE:
        logger.error("IDS Peak SDK not available")
        return []
    
    try:
        peak.Library.Initialize()
        
        device_manager = peak.DeviceManager.Instance()
        device_manager.Update()
        
        devices = device_manager.Devices()
        
        camera_list = []
        for i, dev in enumerate(devices):
            camera_list.append({
                'index': i,
                'model': dev.ModelName(),
                'serial': dev.SerialNumber(),
                'vendor': dev.VendorName(),
                'interface': dev.ParentInterface().DisplayName()
            })
        
        peak.Library.Close()
        
        return camera_list
        
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        return []


# Legacy compatibility functions for existing code
def create_stereo_camera(config: dict):
    """
    Create stereo camera system from configuration (compatibility wrapper)
    
    Args:
        config: Camera configuration dictionary
        
    Returns:
        Configured StereoCameraSystem instance wrapped with compatibility layer
    """
    
    resolution = config['cameras']['resolution']
    width = resolution['width']
    height = resolution['height']
    framerate = config['cameras'].get('framerate', 30)
    
    # Get exposure and gain with fallback to defaults
    exposure_us = config['cameras'].get('exposure_us', 10000)
    if config['cameras'].get('exposure') == 'auto':
        exposure_us = 10000  # Default value for auto
    
    gain_db = config['cameras'].get('gain_db', 0.0)
    
    # Get pixel format with fallback to BGR8 (color)
    pixel_format = config['cameras'].get('pixel_format', 'BGR8')
    
    # Get camera IDs (serial or device_id)
    left_config = config['cameras']['left_camera']
    right_config = config['cameras']['right_camera']
    
    left_id = left_config.get('serial_number', left_config.get('device_id', 0))
    right_id = right_config.get('serial_number', right_config.get('device_id', 1))
    
    # Create stereo system
    stereo = StereoCameraSystem(left_id, right_id)
    
    # Wrap with compatibility layer
    class CompatibilityStereoCamera:
        def __init__(self, stereo_system, init_params):
            self.stereo = stereo_system
            self.init_params = init_params
            self.is_open = False
            
        def open(self) -> bool:
            """Open cameras (compatibility method)"""
            self.is_open = self.stereo.initialize(**self.init_params)
            return self.is_open
        
        def close(self) -> None:
            """Close cameras (compatibility method)"""
            self.stereo.release()
            self.is_open = False
        
        def capture_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
            """Capture frames (compatibility method)"""
            return self.stereo.capture_stereo_pair()
    
    init_params = {
        'width': width,
        'height': height,
        'exposure_us': exposure_us,
        'gain_db': gain_db,
        'framerate': framerate,
        'pixel_format': pixel_format
    }
    
    return CompatibilityStereoCamera(stereo, init_params)
