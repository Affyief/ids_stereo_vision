"""
Visualization Module
Handles display of stereo images, depth maps, and distance measurements.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Callable
import time

logger = logging.getLogger(__name__)


class StereoVisualizer:
    """
    Handles visualization of stereo vision system output.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize visualizer with configuration.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        self.display_config = config.get('display', {})
        
        self.window_width = self.display_config.get('window_width', 1920)
        self.window_height = self.display_config.get('window_height', 1080)
        self.fps_target = self.display_config.get('fps_target', 30)
        self.min_distance = self.display_config.get('min_distance', 100)
        self.max_distance = self.display_config.get('max_distance', 3000)
        
        # Get colormap
        colormap_name = self.display_config.get('depth_color_map', 'JET')
        self.colormap = getattr(cv2, f'COLORMAP_{colormap_name}', cv2.COLORMAP_JET)
        
        # FPS tracking
        self.fps_history = []
        self.frame_times = []
        self.last_frame_time = time.time()
        
        # Mouse interaction
        self.mouse_x = -1
        self.mouse_y = -1
        self.measurement_mode = False
        
        logger.info("Stereo visualizer initialized")
    
    def create_depth_colormap(self, depth: np.ndarray) -> np.ndarray:
        """
        Create colored depth map for visualization.
        
        Args:
            depth: Depth map in mm
            
        Returns:
            Colored depth map (BGR)
        """
        # Normalize depth to 0-255 range
        depth_normalized = np.clip(
            (depth - self.min_distance) / (self.max_distance - self.min_distance),
            0, 1
        )
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_uint8, self.colormap)
        
        # Set invalid depths (0) to black
        mask = depth <= 0
        depth_colored[mask] = [0, 0, 0]
        
        return depth_colored
    
    def draw_distance_overlay(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        num_points: int = 9
    ) -> np.ndarray:
        """
        Draw distance measurements overlay on image.
        
        Args:
            image: Input image (BGR)
            depth: Depth map in mm
            num_points: Number of measurement points
            
        Returns:
            Image with distance overlay
        """
        overlay = image.copy()
        h, w = depth.shape[:2]
        
        # Calculate grid points
        grid_size = int(np.sqrt(num_points))
        step_y = h // (grid_size + 1)
        step_x = w // (grid_size + 1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                y = i * step_y
                x = j * step_x
                
                # Get depth value at this point
                if 0 <= y < h and 0 <= x < w:
                    depth_value = self._get_distance_at_point(depth, x, y)
                    
                    if depth_value > 0:
                        # Draw circle
                        cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
                        
                        # Format distance text
                        if depth_value < 1000:
                            text = f"{depth_value:.0f}mm"
                        else:
                            text = f"{depth_value/1000:.2f}m"
                        
                        # Background for text
                        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                        cv2.rectangle(
                            overlay,
                            (x - text_size[0]//2 - 5, y - text_size[1] - 10),
                            (x + text_size[0]//2 + 5, y - 5),
                            (0, 0, 0),
                            -1
                        )
                        
                        # Draw text
                        cv2.putText(
                            overlay,
                            text,
                            (x - text_size[0]//2, y - 8),
                            font,
                            font_scale,
                            (0, 255, 0),
                            thickness
                        )
        
        return overlay
    
    def _get_distance_at_point(
        self,
        depth: np.ndarray,
        x: int,
        y: int,
        window_size: int = 5
    ) -> float:
        """
        Get distance at a specific point with averaging.
        
        Args:
            depth: Depth map in mm
            x: X coordinate
            y: Y coordinate
            window_size: Size of averaging window
            
        Returns:
            Average distance in mm
        """
        h, w = depth.shape[:2]
        
        # Define window bounds
        y_min = max(0, y - window_size // 2)
        y_max = min(h, y + window_size // 2 + 1)
        x_min = max(0, x - window_size // 2)
        x_max = min(w, x + window_size // 2 + 1)
        
        # Get window
        window = depth[y_min:y_max, x_min:x_max]
        
        # Filter out invalid depths
        valid_depths = window[window > 0]
        
        if len(valid_depths) > 0:
            return np.median(valid_depths)
        else:
            return 0.0
    
    def draw_mouse_measurement(
        self,
        image: np.ndarray,
        depth: np.ndarray
    ) -> np.ndarray:
        """
        Draw measurement at mouse position.
        
        Args:
            image: Input image
            depth: Depth map in mm
            
        Returns:
            Image with mouse measurement
        """
        if self.mouse_x < 0 or self.mouse_y < 0:
            return image
        
        h, w = depth.shape[:2]
        if self.mouse_x >= w or self.mouse_y >= h:
            return image
        
        overlay = image.copy()
        
        # Get distance at mouse position
        distance = self._get_distance_at_point(depth, self.mouse_x, self.mouse_y)
        
        if distance > 0:
            # Draw crosshair
            cv2.drawMarker(
                overlay,
                (self.mouse_x, self.mouse_y),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                20,
                2
            )
            
            # Draw distance text
            if distance < 1000:
                text = f"Distance: {distance:.0f}mm"
            else:
                text = f"Distance: {distance/1000:.2f}m"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Position text near cursor
            text_x = self.mouse_x + 15
            text_y = self.mouse_y - 15
            
            # Keep text in bounds
            if text_x + text_size[0] > w:
                text_x = self.mouse_x - text_size[0] - 15
            if text_y < text_size[1]:
                text_y = self.mouse_y + text_size[1] + 15
            
            # Background
            cv2.rectangle(
                overlay,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                (0, 0, 0),
                -1
            )
            
            # Text
            cv2.putText(
                overlay,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 255, 255),
                thickness
            )
        
        return overlay
    
    def draw_fov_boundaries(
        self,
        image: np.ndarray,
        depth: np.ndarray
    ) -> np.ndarray:
        """
        Draw field of view boundaries with distance markers.
        
        Args:
            image: Input image
            depth: Depth map in mm
            
        Returns:
            Image with FOV boundaries
        """
        overlay = image.copy()
        h, w = depth.shape[:2]
        
        # Draw border
        cv2.rectangle(overlay, (10, 10), (w-10, h-10), (255, 255, 255), 2)
        
        # Get distances at corners and center
        points = [
            (w//4, h//4, "NW"),
            (3*w//4, h//4, "NE"),
            (w//2, h//2, "C"),
            (w//4, 3*h//4, "SW"),
            (3*w//4, 3*h//4, "SE")
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        for x, y, label in points:
            distance = self._get_distance_at_point(depth, x, y)
            if distance > 0:
                if distance < 1000:
                    text = f"{label}: {distance:.0f}mm"
                else:
                    text = f"{label}: {distance/1000:.2f}m"
                
                cv2.putText(
                    overlay,
                    text,
                    (15, h - 20 - len([p for p in points if p[2] <= label]) * 20),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness
                )
        
        return overlay
    
    def draw_fps(self, image: np.ndarray) -> np.ndarray:
        """
        Draw FPS counter on image.
        
        Args:
            image: Input image
            
        Returns:
            Image with FPS counter
        """
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time > 0 else 0
        self.last_frame_time = current_time
        
        # Update FPS history
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        
        # Draw FPS
        text = f"FPS: {avg_fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Top right corner
        x = image.shape[1] - text_size[0] - 10
        y = text_size[1] + 10
        
        # Background
        cv2.rectangle(
            image,
            (x - 5, y - text_size[1] - 5),
            (x + text_size[0] + 5, y + 5),
            (0, 0, 0),
            -1
        )
        
        # Text
        color = (0, 255, 0) if avg_fps >= self.fps_target * 0.8 else (0, 165, 255)
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
        
        return image
    
    def draw_info_panel(
        self,
        image: np.ndarray,
        info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw information panel on image.
        
        Args:
            image: Input image
            info: Information dictionary
            
        Returns:
            Image with info panel
        """
        overlay = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        line_height = 20
        x = 10
        y = 30
        
        # Draw background
        panel_height = len(info) * line_height + 20
        cv2.rectangle(overlay, (5, 5), (300, panel_height), (0, 0, 0), -1)
        
        # Draw info lines
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness)
            y += line_height
        
        return overlay
    
    def display_stereo_view(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        window_name: str = "Stereo View"
    ) -> int:
        """
        Display stereo image pair side by side.
        
        Args:
            left_image: Left image
            right_image: Right image
            window_name: Window name
            
        Returns:
            Key pressed by user
        """
        combined = np.hstack((left_image, right_image))
        
        # Resize if needed
        h, w = combined.shape[:2]
        if w > self.window_width:
            scale = self.window_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            combined = cv2.resize(combined, (new_w, new_h))
        
        cv2.imshow(window_name, combined)
        return cv2.waitKey(1) & 0xFF
    
    def display_results(
        self,
        rectified_left: np.ndarray,
        depth: np.ndarray,
        disparity: Optional[np.ndarray] = None,
        show_measurements: bool = True
    ) -> int:
        """
        Display processing results with overlays.
        
        Args:
            rectified_left: Rectified left image
            depth: Depth map
            disparity: Optional disparity map
            show_measurements: Whether to show distance measurements
            
        Returns:
            Key pressed by user
        """
        # Create depth colormap
        depth_colored = self.create_depth_colormap(depth)
        
        # Create overlay image
        if show_measurements:
            overlay = self.draw_distance_overlay(rectified_left, depth)
            if self.measurement_mode:
                overlay = self.draw_mouse_measurement(overlay, depth)
            overlay = self.draw_fov_boundaries(overlay, depth)
        else:
            overlay = rectified_left.copy()
        
        # Add FPS counter
        overlay = self.draw_fps(overlay)
        depth_colored = self.draw_fps(depth_colored)
        
        # Combine views
        if disparity is not None:
            # Normalize disparity for display
            disp_vis = cv2.normalize(
                disparity,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
                cv2.CV_8U
            )
            disp_colored = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            
            # Create 2x2 grid
            top_row = np.hstack((overlay, depth_colored))
            bottom_row = np.hstack((rectified_left, disp_colored))
            combined = np.vstack((top_row, bottom_row))
        else:
            # Side by side
            combined = np.hstack((overlay, depth_colored))
        
        # Resize if needed
        h, w = combined.shape[:2]
        if w > self.window_width or h > self.window_height:
            scale = min(self.window_width / w, self.window_height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            combined = cv2.resize(combined, (new_w, new_h))
        
        cv2.imshow("Stereo Vision System", combined)
        return cv2.waitKey(1) & 0xFF
    
    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback for interactive measurement.
        
        Args:
            event: Mouse event
            x: X coordinate
            y: Y coordinate
            flags: Event flags
            param: User data
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
    
    def setup_mouse_callback(self, window_name: str = "Stereo Vision System"):
        """
        Setup mouse callback for measurement mode.
        
        Args:
            window_name: Name of window to attach callback
        """
        cv2.setMouseCallback(window_name, self.mouse_callback)
    
    def toggle_measurement_mode(self):
        """Toggle interactive measurement mode."""
        self.measurement_mode = not self.measurement_mode
        logger.info(f"Measurement mode: {'ON' if self.measurement_mode else 'OFF'}")
    
    def cleanup(self):
        """Cleanup visualization resources."""
        cv2.destroyAllWindows()
        logger.info("Visualizer cleaned up")
