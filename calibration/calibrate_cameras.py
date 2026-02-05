"""
Camera Calibration Module
Handles intrinsic and extrinsic calibration for stereo camera system.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
import glob
import os

logger = logging.getLogger(__name__)


class StereoCalibrator:
    """
    Handles stereo camera calibration including intrinsic and extrinsic parameters.
    """
    
    def __init__(self, pattern_size: Tuple[int, int], square_size: float):
        """
        Initialize calibrator.
        
        Args:
            pattern_size: Chessboard pattern size (width, height) in internal corners
            square_size: Size of chessboard square in mm
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        logger.info(f"Calibrator initialized with pattern {pattern_size} and square size {square_size}mm")
    
    def find_chessboard_corners(
        self,
        image: np.ndarray,
        refine: bool = True
    ) -> Optional[np.ndarray]:
        """
        Find chessboard corners in image.
        
        Args:
            image: Input image (BGR or grayscale)
            refine: Whether to refine corner positions
            
        Returns:
            Corner positions, or None if not found
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if ret and refine:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return corners if ret else None
    
    def draw_chessboard_corners(
        self,
        image: np.ndarray,
        corners: np.ndarray,
        found: bool = True
    ) -> np.ndarray:
        """
        Draw detected chessboard corners on image.
        
        Args:
            image: Input image
            corners: Corner positions
            found: Whether corners were found successfully
            
        Returns:
            Image with drawn corners
        """
        img_corners = image.copy()
        cv2.drawChessboardCorners(img_corners, self.pattern_size, corners, found)
        return img_corners
    
    def calibrate_single_camera(
        self,
        images: List[np.ndarray],
        image_size: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List], Optional[List]]:
        """
        Calibrate a single camera.
        
        Args:
            images: List of calibration images
            image_size: Image size (width, height)
            
        Returns:
            Tuple of (camera_matrix, dist_coeffs, rvecs, tvecs) or (None, None, None, None) if failed
        """
        # Arrays to store object points and image points
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane
        
        logger.info(f"Processing {len(images)} images for calibration...")
        
        for idx, image in enumerate(images):
            corners = self.find_chessboard_corners(image)
            
            if corners is not None:
                obj_points.append(self.objp)
                img_points.append(corners)
                logger.debug(f"Image {idx+1}/{len(images)}: Corners found")
            else:
                logger.warning(f"Image {idx+1}/{len(images)}: Corners not found")
        
        if len(obj_points) < 10:
            logger.error(f"Not enough valid images for calibration (found {len(obj_points)}, need at least 10)")
            return None, None, None, None
        
        logger.info(f"Calibrating camera with {len(obj_points)} valid images...")
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            image_size,
            None,
            None
        )
        
        if ret:
            logger.info(f"Camera calibration successful. RMS error: {ret:.4f}")
            logger.info(f"Camera matrix:\n{camera_matrix}")
            logger.info(f"Distortion coefficients: {dist_coeffs.ravel()}")
            return camera_matrix, dist_coeffs, rvecs, tvecs
        else:
            logger.error("Camera calibration failed")
            return None, None, None, None
    
    def calibrate_stereo(
        self,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
        image_size: Tuple[int, int],
        camera_matrix_left: Optional[np.ndarray] = None,
        dist_coeffs_left: Optional[np.ndarray] = None,
        camera_matrix_right: Optional[np.ndarray] = None,
        dist_coeffs_right: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Perform stereo calibration.
        
        Args:
            left_images: List of left camera images
            right_images: List of right camera images
            image_size: Image size (width, height)
            camera_matrix_left: Pre-calibrated left camera matrix (optional)
            dist_coeffs_left: Pre-calibrated left distortion coefficients (optional)
            camera_matrix_right: Pre-calibrated right camera matrix (optional)
            dist_coeffs_right: Pre-calibrated right distortion coefficients (optional)
            
        Returns:
            Dictionary with stereo calibration parameters, or None if failed
        """
        if len(left_images) != len(right_images):
            logger.error("Number of left and right images must match")
            return None
        
        # Arrays to store object points and image points
        obj_points = []
        img_points_left = []
        img_points_right = []
        
        logger.info(f"Processing {len(left_images)} stereo pairs for calibration...")
        
        for idx, (left_img, right_img) in enumerate(zip(left_images, right_images)):
            # Find corners in both images
            corners_left = self.find_chessboard_corners(left_img)
            corners_right = self.find_chessboard_corners(right_img)
            
            if corners_left is not None and corners_right is not None:
                obj_points.append(self.objp)
                img_points_left.append(corners_left)
                img_points_right.append(corners_right)
                logger.debug(f"Pair {idx+1}/{len(left_images)}: Corners found in both images")
            else:
                logger.warning(f"Pair {idx+1}/{len(left_images)}: Corners not found in one or both images")
        
        if len(obj_points) < 10:
            logger.error(f"Not enough valid stereo pairs for calibration (found {len(obj_points)}, need at least 10)")
            return None
        
        logger.info(f"Calibrating stereo system with {len(obj_points)} valid pairs...")
        
        # If intrinsic parameters not provided, calibrate individual cameras first
        if camera_matrix_left is None or dist_coeffs_left is None:
            logger.info("Calibrating left camera...")
            camera_matrix_left, dist_coeffs_left, _, _ = self.calibrate_single_camera(
                left_images, image_size
            )
            if camera_matrix_left is None:
                return None
        
        if camera_matrix_right is None or dist_coeffs_right is None:
            logger.info("Calibrating right camera...")
            camera_matrix_right, dist_coeffs_right, _, _ = self.calibrate_single_camera(
                right_images, image_size
            )
            if camera_matrix_right is None:
                return None
        
        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points_left,
            img_points_right,
            camera_matrix_left,
            dist_coeffs_left,
            camera_matrix_right,
            dist_coeffs_right,
            image_size,
            criteria=criteria,
            flags=flags
        )
        
        if not ret:
            logger.error("Stereo calibration failed")
            return None
        
        logger.info(f"Stereo calibration successful. RMS error: {ret:.4f}")
        logger.info(f"Rotation matrix:\n{R}")
        logger.info(f"Translation vector: {T.ravel()}")
        
        # Compute rectification transforms
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            K1, D1, K2, D2,
            image_size,
            R, T,
            alpha=0,
            newImageSize=image_size
        )
        
        # Calculate baseline (distance between cameras)
        baseline = np.linalg.norm(T)
        
        logger.info(f"Baseline: {baseline:.2f} mm")
        logger.info(f"Rectification completed")
        
        # Compile results
        calibration_results = {
            'left_camera': {
                'camera_matrix': K1,
                'distortion_coefficients': D1,
                'image_size': list(image_size)
            },
            'right_camera': {
                'camera_matrix': K2,
                'distortion_coefficients': D2,
                'image_size': list(image_size)
            },
            'stereo': {
                'rotation_matrix': R,
                'translation_vector': T,
                'essential_matrix': E,
                'fundamental_matrix': F,
                'rectification_left': R1,
                'rectification_right': R2,
                'projection_left': P1,
                'projection_right': P2,
                'disparity_to_depth_matrix': Q,
                'baseline': float(baseline),
                'calibration_error': float(ret)
            }
        }
        
        return calibration_results
    
    def load_image_pairs(
        self,
        left_path: str,
        right_path: str,
        pattern: str = "*.png"
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load stereo image pairs from directories.
        
        Args:
            left_path: Path to left images directory
            right_path: Path to right images directory
            pattern: File pattern to match
            
        Returns:
            Tuple of (left_images, right_images) lists
        """
        left_files = sorted(glob.glob(os.path.join(left_path, pattern)))
        right_files = sorted(glob.glob(os.path.join(right_path, pattern)))
        
        if len(left_files) == 0 or len(right_files) == 0:
            logger.error("No images found in one or both directories")
            return [], []
        
        if len(left_files) != len(right_files):
            logger.warning(f"Mismatch in number of images: {len(left_files)} left, {len(right_files)} right")
        
        left_images = []
        right_images = []
        
        for left_file, right_file in zip(left_files, right_files):
            left_img = cv2.imread(left_file)
            right_img = cv2.imread(right_file)
            
            if left_img is not None and right_img is not None:
                left_images.append(left_img)
                right_images.append(right_img)
            else:
                logger.warning(f"Failed to load pair: {left_file}, {right_file}")
        
        logger.info(f"Loaded {len(left_images)} stereo pairs")
        return left_images, right_images
    
    def validate_calibration(
        self,
        calibration_params: Dict[str, Any],
        test_images_left: List[np.ndarray],
        test_images_right: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Validate calibration quality using test images.
        
        Args:
            calibration_params: Calibration parameters
            test_images_left: Left test images
            test_images_right: Right test images
            
        Returns:
            Validation results dictionary
        """
        results = {
            'epipolar_errors': [],
            'rectification_quality': []
        }
        
        K1 = calibration_params['left_camera']['camera_matrix']
        D1 = calibration_params['left_camera']['distortion_coefficients']
        K2 = calibration_params['right_camera']['camera_matrix']
        D2 = calibration_params['right_camera']['distortion_coefficients']
        R1 = calibration_params['stereo']['rectification_left']
        R2 = calibration_params['stereo']['rectification_right']
        P1 = calibration_params['stereo']['projection_left']
        P2 = calibration_params['stereo']['projection_right']
        
        img_size = tuple(calibration_params['left_camera']['image_size'])
        
        # Create rectification maps
        map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_16SC2)
        map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_16SC2)
        
        for left_img, right_img in zip(test_images_left, test_images_right):
            # Rectify images
            rect_left = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
            rect_right = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)
            
            # Find corners in rectified images
            corners_left = self.find_chessboard_corners(rect_left)
            corners_right = self.find_chessboard_corners(rect_right)
            
            if corners_left is not None and corners_right is not None:
                # Calculate epipolar error (y-coordinate difference)
                y_diff = np.abs(corners_left[:, 0, 1] - corners_right[:, 0, 1])
                avg_error = np.mean(y_diff)
                max_error = np.max(y_diff)
                
                results['epipolar_errors'].append({
                    'mean': float(avg_error),
                    'max': float(max_error)
                })
                
                logger.debug(f"Epipolar error - Mean: {avg_error:.2f}, Max: {max_error:.2f} pixels")
        
        if results['epipolar_errors']:
            avg_mean = np.mean([e['mean'] for e in results['epipolar_errors']])
            avg_max = np.mean([e['max'] for e in results['epipolar_errors']])
            logger.info(f"Average epipolar error - Mean: {avg_mean:.2f}, Max: {avg_max:.2f} pixels")
            results['overall_mean_error'] = float(avg_mean)
            results['overall_max_error'] = float(avg_max)
        
        return results
