"""
Stereo processing module for depth computation.

This module handles stereo image rectification, disparity computation,
and depth map generation.
"""

import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
import cv2


class StereoProcessor:
    """
    Stereo vision processor for computing depth from stereo image pairs.
    """
    
    def __init__(
        self,
        left_calibration: Dict[str, np.ndarray],
        right_calibration: Dict[str, np.ndarray],
        stereo_calibration: Dict[str, np.ndarray],
        stereo_config: Dict[str, Any]
    ):
        """
        Initialize stereo processor.
        
        Args:
            left_calibration: Left camera calibration data
            right_calibration: Right camera calibration data
            stereo_calibration: Stereo calibration data
            stereo_config: Stereo processing configuration
        """
        self.logger = logging.getLogger(__name__)
        
        # Camera calibration parameters
        self.left_camera_matrix = left_calibration['camera_matrix']
        self.left_dist_coeffs = left_calibration['dist_coeffs']
        self.right_camera_matrix = right_calibration['camera_matrix']
        self.right_dist_coeffs = right_calibration['dist_coeffs']
        
        # Stereo calibration parameters
        self.R = stereo_calibration['R']
        self.T = stereo_calibration['T']
        self.E = stereo_calibration.get('E')
        self.F = stereo_calibration.get('F')
        
        # Rectification parameters
        self.R1 = stereo_calibration['R1']
        self.R2 = stereo_calibration['R2']
        self.P1 = stereo_calibration['P1']
        self.P2 = stereo_calibration['P2']
        self.Q = stereo_calibration['Q']
        
        # Image size
        self.image_size = (
            int(stereo_calibration['image_width']),
            int(stereo_calibration['image_height'])
        )
        
        # Stereo configuration
        self.config = stereo_config
        self.baseline_mm = stereo_config['stereo']['baseline_mm']
        self.algorithm = stereo_config['stereo']['algorithm']
        
        # Initialize stereo matcher
        self._init_stereo_matcher()
        
        # Initialize rectification maps
        self._init_rectification_maps()
        
        # Initialize post-processing
        if self.config['stereo']['post_process']['use_wls_filter']:
            self._init_wls_filter()
        else:
            self.wls_filter = None
    
    def _init_stereo_matcher(self) -> None:
        """Initialize stereo matching algorithm."""
        if self.algorithm == "SGBM":
            params = self.config['stereo']['sgbm']
            
            # Convert mode string to OpenCV constant
            mode_map = {
                "SGBM": cv2.STEREO_SGBM_MODE_SGBM,
                "HH": cv2.STEREO_SGBM_MODE_HH,
                "SGBM_3WAY": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                "HH4": cv2.STEREO_SGBM_MODE_HH4
            }
            mode = mode_map.get(params['mode'], cv2.STEREO_SGBM_MODE_HH)
            
            self.left_matcher = cv2.StereoSGBM_create(
                minDisparity=params['min_disparity'],
                numDisparities=params['num_disparities'],
                blockSize=params['block_size'],
                P1=params['P1'],
                P2=params['P2'],
                disp12MaxDiff=params['disp12_max_diff'],
                uniquenessRatio=params['uniqueness_ratio'],
                speckleWindowSize=params['speckle_window_size'],
                speckleRange=params['speckle_range'],
                mode=mode
            )
            
        elif self.algorithm == "BM":
            params = self.config['stereo']['bm']
            
            self.left_matcher = cv2.StereoBM_create(
                numDisparities=params['num_disparities'],
                blockSize=params['block_size']
            )
            self.left_matcher.setMinDisparity(params['min_disparity'])
        
        else:
            raise ValueError(f"Unknown stereo algorithm: {self.algorithm}")
        
        self.logger.info(f"Initialized {self.algorithm} stereo matcher")
    
    def _init_rectification_maps(self) -> None:
        """Initialize rectification maps for both cameras."""
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.left_camera_matrix,
            self.left_dist_coeffs,
            self.R1,
            self.P1,
            self.image_size,
            cv2.CV_16SC2
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.right_camera_matrix,
            self.right_dist_coeffs,
            self.R2,
            self.P2,
            self.image_size,
            cv2.CV_16SC2
        )
        
        self.logger.info("Initialized rectification maps")
    
    def _init_wls_filter(self) -> None:
        """Initialize WLS (Weighted Least Squares) filter for disparity refinement."""
        try:
            # Create right matcher for WLS filter
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            
            # Create WLS filter
            wls_lambda = self.config['stereo']['post_process']['wls_lambda']
            wls_sigma = self.config['stereo']['post_process']['wls_sigma']
            
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.wls_filter.setLambda(wls_lambda)
            self.wls_filter.setSigmaColor(wls_sigma)
            
            self.logger.info("Initialized WLS filter")
        except AttributeError:
            self.logger.warning("WLS filter not available in this OpenCV build")
            self.wls_filter = None
    
    def rectify_images(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            Tuple of (rectified_left, rectified_right)
        """
        left_rectified = cv2.remap(
            left_image,
            self.map1_left,
            self.map2_left,
            cv2.INTER_LINEAR
        )
        
        right_rectified = cv2.remap(
            right_image,
            self.map1_right,
            self.map2_right,
            cv2.INTER_LINEAR
        )
        
        return left_rectified, right_rectified
    
    def compute_disparity(
        self,
        left_rectified: np.ndarray,
        right_rectified: np.ndarray,
        use_wls: bool = True
    ) -> np.ndarray:
        """
        Compute disparity map from rectified stereo images.
        
        Args:
            left_rectified: Rectified left image
            right_rectified: Rectified right image
            use_wls: Use WLS filtering if available
            
        Returns:
            Disparity map
        """
        # Convert to grayscale if needed
        if len(left_rectified.shape) == 3:
            left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_rectified
            right_gray = right_rectified
        
        # Compute disparity
        disparity_left = self.left_matcher.compute(left_gray, right_gray)
        
        # Apply WLS filter if available and requested
        if use_wls and self.wls_filter is not None and hasattr(self, 'right_matcher'):
            disparity_right = self.right_matcher.compute(right_gray, left_gray)
            disparity = self.wls_filter.filter(
                disparity_left,
                left_gray,
                disparity_map_right=disparity_right
            )
        else:
            disparity = disparity_left
        
        # Convert to float and normalize
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def post_process_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to disparity map.
        
        Args:
            disparity: Raw disparity map
            
        Returns:
            Post-processed disparity map
        """
        # Apply median blur if configured
        median_kernel = self.config['stereo']['post_process']['median_blur']
        if median_kernel > 0:
            # Ensure kernel size is odd
            if median_kernel % 2 == 0:
                median_kernel += 1
            disparity = cv2.medianBlur(disparity.astype(np.uint8), median_kernel).astype(np.float32)
        
        return disparity
    
    def disparity_to_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Convert disparity map to depth map.
        
        Args:
            disparity: Disparity map in pixels
            
        Returns:
            Depth map in millimeters
        """
        # Get focal length from projection matrix
        focal_length = self.P1[0, 0]  # fx in pixels
        
        # Calculate depth using: Z = (f × B) / d
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (focal_length * self.baseline_mm) / disparity
            
            # Filter invalid depths
            depth[disparity <= 0] = 0
            depth[~np.isfinite(depth)] = 0
            
            # Apply distance limits
            min_dist = self.config['stereo']['visualization']['min_distance_m'] * 1000
            max_dist = self.config['stereo']['visualization']['max_distance_m'] * 1000
            depth[depth < min_dist] = 0
            depth[depth > max_dist] = 0
        
        return depth
    
    def process_stereo_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete stereo processing pipeline.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            Tuple of (left_rectified, right_rectified, disparity, depth)
        """
        # Rectify images
        left_rectified, right_rectified = self.rectify_images(left_image, right_image)
        
        # Compute disparity
        use_wls = self.config['stereo']['post_process']['use_wls_filter']
        disparity = self.compute_disparity(left_rectified, right_rectified, use_wls)
        
        # Post-process disparity
        disparity = self.post_process_disparity(disparity)
        
        # Convert to depth
        depth = self.disparity_to_depth(disparity)
        
        return left_rectified, right_rectified, disparity, depth
    
    def get_depth_at_point(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """
        Get depth value at a specific point.
        
        Args:
            depth_map: Depth map in millimeters
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Depth value in millimeters, or 0 if invalid
        """
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return float(depth_map[y, x])
        return 0.0
    
    def update_stereo_params(self, **kwargs) -> None:
        """
        Update stereo matching parameters dynamically.
        
        Args:
            **kwargs: Parameters to update (e.g., num_disparities, block_size)
        """
        if 'num_disparities' in kwargs:
            num_disp = kwargs['num_disparities']
            # Ensure it's divisible by 16
            num_disp = max(16, (num_disp // 16) * 16)
            self.left_matcher.setNumDisparities(num_disp)
            self.logger.info(f"Updated numDisparities to {num_disp}")
        
        if 'block_size' in kwargs:
            block_size = kwargs['block_size']
            # Ensure it's odd
            if block_size % 2 == 0:
                block_size += 1
            block_size = max(5, min(21, block_size))
            self.left_matcher.setBlockSize(block_size)
            self.logger.info(f"Updated blockSize to {block_size}")
        
        if 'uniqueness_ratio' in kwargs and hasattr(self.left_matcher, 'setUniquenessRatio'):
            ratio = kwargs['uniqueness_ratio']
            self.left_matcher.setUniquenessRatio(ratio)
            self.logger.info(f"Updated uniquenessRatio to {ratio}")


def create_stereo_processor(
    calibration_dir: str,
    stereo_config: Dict[str, Any]
) -> Optional[StereoProcessor]:
    """
    Create stereo processor from calibration files.
    
    Args:
        calibration_dir: Directory containing calibration files
        stereo_config: Stereo configuration
        
    Returns:
        StereoProcessor instance or None if calibration files not found
    """
    import os
    from .utils import load_calibration_data
    
    # Load calibration data
    left_cal_path = os.path.join(calibration_dir, 'left_camera_calibration.npz')
    right_cal_path = os.path.join(calibration_dir, 'right_camera_calibration.npz')
    stereo_cal_path = os.path.join(calibration_dir, 'stereo_calibration.npz')
    
    left_cal = load_calibration_data(left_cal_path)
    right_cal = load_calibration_data(right_cal_path)
    stereo_cal = load_calibration_data(stereo_cal_path)
    
    if left_cal is None or right_cal is None or stereo_cal is None:
        logging.error("Calibration files not found. Please run calibration first.")
        return None
    
    return StereoProcessor(left_cal, right_cal, stereo_cal, stereo_config)
