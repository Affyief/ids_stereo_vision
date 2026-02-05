"""
Stereo Processing Module
Handles stereo rectification, matching, and depth computation.
"""

import numpy as np
import cv2
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class StereoProcessor:
    """
    Handles stereo image processing including rectification and depth estimation.
    """
    
    def __init__(self, calibration_params: Dict[str, Any], config: Dict[str, Any]):
        """
        Initialize stereo processor with calibration parameters.
        
        Args:
            calibration_params: Dictionary containing calibration data
            config: System configuration dictionary
        """
        self.calib = calibration_params
        self.config = config
        
        # Extract calibration parameters
        self.left_camera_matrix = calibration_params['left_camera']['camera_matrix']
        self.left_dist_coeffs = calibration_params['left_camera']['distortion_coefficients']
        self.right_camera_matrix = calibration_params['right_camera']['camera_matrix']
        self.right_dist_coeffs = calibration_params['right_camera']['distortion_coefficients']
        
        stereo = calibration_params['stereo']
        self.R1 = stereo['rectification_left']
        self.R2 = stereo['rectification_right']
        self.P1 = stereo['projection_left']
        self.P2 = stereo['projection_right']
        self.Q = stereo['disparity_to_depth_matrix']
        self.baseline = stereo['baseline']
        
        # Image size
        img_size = tuple(calibration_params['left_camera']['image_size'])
        
        # Compute rectification maps
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.left_camera_matrix,
            self.left_dist_coeffs,
            self.R1,
            self.P1,
            img_size,
            cv2.CV_16SC2
        )
        
        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.right_camera_matrix,
            self.right_dist_coeffs,
            self.R2,
            self.P2,
            img_size,
            cv2.CV_16SC2
        )
        
        # Initialize stereo matcher
        self.stereo_matcher = self._create_stereo_matcher()
        
        logger.info("Stereo processor initialized")
    
    def _create_stereo_matcher(self):
        """
        Create stereo matcher based on configuration.
        
        Returns:
            OpenCV stereo matcher object
        """
        stereo_config = self.config['stereo_matching']
        algorithm = stereo_config.get('algorithm', 'SGBM')
        
        min_disp = stereo_config.get('min_disparity', 0)
        num_disp = stereo_config.get('num_disparities', 128)
        block_size = stereo_config.get('block_size', 11)
        
        if algorithm == 'BM':
            matcher = cv2.StereoBM_create(
                numDisparities=num_disp,
                blockSize=block_size
            )
            logger.info("Using StereoBM matcher")
        else:
            # SGBM (Semi-Global Block Matching)
            matcher = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=block_size,
                P1=8 * 3 * block_size ** 2,
                P2=32 * 3 * block_size ** 2,
                disp12MaxDiff=stereo_config.get('disp12_max_diff', 1),
                uniquenessRatio=stereo_config.get('uniqueness_ratio', 10),
                speckleWindowSize=stereo_config.get('speckle_window_size', 100),
                speckleRange=stereo_config.get('speckle_range', 32),
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
            logger.info("Using StereoSGBM matcher")
        
        return matcher
    
    def rectify_stereo_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair.
        
        Args:
            left_image: Left camera image (BGR)
            right_image: Right camera image (BGR)
            
        Returns:
            Tuple of (rectified_left, rectified_right)
        """
        rectified_left = cv2.remap(
            left_image,
            self.map1_left,
            self.map2_left,
            cv2.INTER_LINEAR
        )
        
        rectified_right = cv2.remap(
            right_image,
            self.map1_right,
            self.map2_right,
            cv2.INTER_LINEAR
        )
        
        return rectified_left, rectified_right
    
    def compute_disparity(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray
    ) -> np.ndarray:
        """
        Compute disparity map from rectified stereo pair.
        
        Args:
            left_image: Rectified left image (BGR or grayscale)
            right_image: Rectified right image (BGR or grayscale)
            
        Returns:
            Disparity map (float32)
        """
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
        
        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_image
        
        # Compute disparity
        disparity = self.stereo_matcher.compute(left_gray, right_gray)
        
        # Convert to float and scale
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def compute_depth(self, disparity: np.ndarray) -> np.ndarray:
        """
        Compute depth map from disparity map.
        
        Args:
            disparity: Disparity map (float32)
            
        Returns:
            Depth map in mm
        """
        # Get focal length from projection matrix
        fx = self.P1[0, 0]
        
        # Compute depth using baseline and focal length
        # depth = (baseline * fx) / disparity
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = (self.baseline * fx) / disparity
            depth[disparity <= 0] = 0
            depth[~np.isfinite(depth)] = 0
        
        return depth
    
    def filter_disparity(self, disparity: np.ndarray) -> np.ndarray:
        """
        Apply filtering to disparity map to reduce noise.
        
        Args:
            disparity: Raw disparity map
            
        Returns:
            Filtered disparity map
        """
        # Create WLS filter for disparity refinement
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.stereo_matcher)
        wls_filter.setLambda(8000)
        wls_filter.setSigmaColor(1.5)
        
        # For WLS filter, we need right disparity too
        # For simplicity, we'll use median filter
        filtered = cv2.medianBlur(disparity.astype(np.float32), 5)
        
        return filtered
    
    def compute_point_cloud(
        self,
        disparity: np.ndarray,
        left_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 3D point cloud from disparity map.
        
        Args:
            disparity: Disparity map
            left_image: Left rectified image for colors
            
        Returns:
            Tuple of (points_3d, colors) where points_3d is Nx3 and colors is Nx3
        """
        # Reproject to 3D
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        
        # Filter out invalid points
        mask = disparity > 0
        points = points_3d[mask]
        
        # Get colors
        if len(left_image.shape) == 3:
            colors = left_image[mask]
            # Convert BGR to RGB
            colors = colors[:, [2, 1, 0]]
        else:
            colors = np.stack([left_image[mask]] * 3, axis=1)
        
        return points, colors
    
    def process_stereo_pair(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Process complete stereo pair: rectify, compute disparity and depth.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            Dictionary with 'rectified_left', 'rectified_right', 'disparity', 'depth'
        """
        # Rectify
        rect_left, rect_right = self.rectify_stereo_pair(left_image, right_image)
        
        # Compute disparity
        disparity = self.compute_disparity(rect_left, rect_right)
        
        # Filter disparity
        disparity_filtered = self.filter_disparity(disparity)
        
        # Compute depth
        depth = self.compute_depth(disparity_filtered)
        
        return {
            'rectified_left': rect_left,
            'rectified_right': rect_right,
            'disparity': disparity_filtered,
            'depth': depth
        }
    
    def visualize_epipolar_lines(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        num_lines: int = 10
    ) -> np.ndarray:
        """
        Visualize epipolar lines on rectified stereo pair.
        
        Args:
            left_image: Rectified left image
            right_image: Rectified right image
            num_lines: Number of horizontal lines to draw
            
        Returns:
            Combined image with epipolar lines
        """
        # Combine images side by side
        combined = np.hstack((left_image, right_image))
        h, w = left_image.shape[:2]
        
        # Draw horizontal lines
        step = h // (num_lines + 1)
        for i in range(1, num_lines + 1):
            y = i * step
            cv2.line(combined, (0, y), (w * 2, y), (0, 255, 0), 1)
        
        # Draw vertical separator
        cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
        
        return combined
    
    def get_focal_length(self) -> float:
        """
        Get focal length in pixels.
        
        Returns:
            Focal length (fx)
        """
        return self.P1[0, 0]
    
    def get_baseline(self) -> float:
        """
        Get baseline distance in mm.
        
        Returns:
            Baseline distance
        """
        return self.baseline
