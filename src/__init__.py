"""IDS Stereo Vision System - Core Package"""

__version__ = "1.0.0"
__author__ = "IDS Stereo Vision Team"

from . import utils
from . import camera_interface
from . import stereo_processor
from . import depth_visualizer

__all__ = [
    'utils',
    'camera_interface',
    'stereo_processor',
    'depth_visualizer',
]
