"""
Depth visualization module for displaying stereo vision results.

This module provides real-time visualization of stereo images, depth maps,
and distance measurements.
"""

import logging
from typing import Dict, Tuple, Optional, Any
import numpy as np
import cv2


class DepthVisualizer:
    """
    Visualizer for stereo vision depth maps and distance measurements.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize depth visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Visualization parameters
        vis_config = config['stereo']['visualization']
        self.colormap_name = vis_config['colormap']
        self.min_distance_m = vis_config['min_distance_m']
        self.max_distance_m = vis_config['max_distance_m']
        self.num_measurement_points = vis_config['measurement_points']
        
        # Get colormap
        self.colormap = self._get_colormap(self.colormap_name)
        
        # Mouse interaction state
        self.mouse_x = -1
        self.mouse_y = -1
        self.click_x = -1
        self.click_y = -1
        
        # Display state
        self.show_measurements = True
        self.show_crosshair = True
    
    def _get_colormap(self, name: str) -> int:
        """
        Get OpenCV colormap constant from name.
        
        Args:
            name: Colormap name
            
        Returns:
            OpenCV colormap constant
        """
        colormap_dict = {
            'JET': cv2.COLORMAP_JET,
            'TURBO': cv2.COLORMAP_TURBO,
            'HSV': cv2.COLORMAP_HSV,
            'HOT': cv2.COLORMAP_HOT,
            'COOL': cv2.COLORMAP_COOL,
            'RAINBOW': cv2.COLORMAP_RAINBOW,
            'VIRIDIS': cv2.COLORMAP_VIRIDIS,
            'PLASMA': cv2.COLORMAP_PLASMA,
            'INFERNO': cv2.COLORMAP_INFERNO,
            'MAGMA': cv2.COLORMAP_MAGMA,
        }
        return colormap_dict.get(name, cv2.COLORMAP_TURBO)
    
    def set_colormap(self, name: str) -> None:
        """
        Change colormap for depth visualization.
        
        Args:
            name: Colormap name
        """
        self.colormap_name = name
        self.colormap = self._get_colormap(name)
        self.logger.info(f"Changed colormap to {name}")
    
    def create_depth_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create colored depth map for visualization.
        
        Args:
            depth_map: Depth map in millimeters
            
        Returns:
            Colored depth map (BGR)
        """
        # Normalize depth to 0-255 range
        min_depth = self.min_distance_m * 1000
        max_depth = self.max_distance_m * 1000
        
        # Create mask for valid depths
        valid_mask = (depth_map > min_depth) & (depth_map < max_depth)
        
        # Normalize
        depth_normalized = np.zeros_like(depth_map, dtype=np.uint8)
        if valid_mask.any():
            depth_normalized[valid_mask] = np.clip(
                255 * (depth_map[valid_mask] - min_depth) / (max_depth - min_depth),
                0, 255
            ).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, self.colormap)
        
        # Set invalid depths to black
        depth_colored[~valid_mask] = [0, 0, 0]
        
        return depth_colored
    
    def get_distance_color(self, distance_mm: float) -> Tuple[int, int, int]:
        """
        Get color for distance value (near=red, medium=yellow, far=green).
        
        Args:
            distance_mm: Distance in millimeters
            
        Returns:
            BGR color tuple
        """
        if distance_mm <= 0:
            return (128, 128, 128)  # Gray for invalid
        
        distance_m = distance_mm / 1000.0
        
        # Near distances (< 1m): Red
        if distance_m < 1.0:
            return (0, 0, 255)
        # Medium distances (1-3m): Yellow
        elif distance_m < 3.0:
            return (0, 255, 255)
        # Far distances (>3m): Green
        else:
            return (0, 255, 0)
    
    def format_distance(self, distance_mm: float) -> str:
        """
        Format distance for display.
        
        Args:
            distance_mm: Distance in millimeters
            
        Returns:
            Formatted distance string
        """
        if distance_mm <= 0 or distance_mm > self.max_distance_m * 1000:
            return "N/A"
        
        if distance_mm < 1000:
            return f"{distance_mm:.0f}mm"
        else:
            return f"{distance_mm / 1000:.2f}m"
    
    def draw_crosshair(
        self,
        image: np.ndarray,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """
        Draw crosshair at image center with distance.
        
        Args:
            image: Image to draw on
            depth_map: Depth map for distance lookup
            
        Returns:
            Image with crosshair
        """
        if not self.show_crosshair:
            return image
        
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2
        
        # Get depth at center
        depth = depth_map[cy, cx] if 0 <= cy < h and 0 <= cx < w else 0
        
        # Draw crosshair
        color = self.get_distance_color(depth)
        cv2.line(image, (cx - 20, cy), (cx + 20, cy), color, 2)
        cv2.line(image, (cx, cy - 20), (cx, cy + 20), color, 2)
        cv2.circle(image, (cx, cy), 5, color, -1)
        
        # Draw distance text
        dist_text = self.format_distance(depth)
        cv2.putText(
            image,
            f"Center: {dist_text}",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
        
        return image
    
    def draw_measurement_grid(
        self,
        image: np.ndarray,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """
        Draw grid of distance measurements.
        
        Args:
            image: Image to draw on
            depth_map: Depth map for distance lookup
            
        Returns:
            Image with measurement grid
        """
        if not self.show_measurements:
            return image
        
        h, w = image.shape[:2]
        
        # Calculate grid spacing
        grid_size = int(np.sqrt(self.num_measurement_points))
        step_x = w // (grid_size + 1)
        step_y = h // (grid_size + 1)
        
        # Draw measurements at grid points
        for i in range(1, grid_size + 1):
            for j in range(1, grid_size + 1):
                x = j * step_x
                y = i * step_y
                
                if 0 <= y < h and 0 <= x < w:
                    depth = depth_map[y, x]
                    color = self.get_distance_color(depth)
                    
                    # Draw point
                    cv2.circle(image, (x, y), 3, color, -1)
                    
                    # Draw distance text
                    dist_text = self.format_distance(depth)
                    cv2.putText(
                        image,
                        dist_text,
                        (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1
                    )
        
        return image
    
    def draw_mouse_measurement(
        self,
        image: np.ndarray,
        depth_map: np.ndarray
    ) -> np.ndarray:
        """
        Draw distance measurement at mouse position.
        
        Args:
            image: Image to draw on
            depth_map: Depth map for distance lookup
            
        Returns:
            Image with mouse measurement
        """
        h, w = image.shape[:2]
        
        # Draw at mouse hover position
        if 0 <= self.mouse_x < w and 0 <= self.mouse_y < h:
            depth = depth_map[self.mouse_y, self.mouse_x]
            color = self.get_distance_color(depth)
            
            # Draw circle at mouse position
            cv2.circle(image, (self.mouse_x, self.mouse_y), 5, color, 2)
            
            # Draw distance text
            dist_text = self.format_distance(depth)
            cv2.putText(
                image,
                f"Hover: {dist_text}",
                (self.mouse_x + 10, self.mouse_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Draw at clicked position
        if 0 <= self.click_x < w and 0 <= self.click_y < h:
            depth = depth_map[self.click_y, self.click_x]
            color = self.get_distance_color(depth)
            
            # Draw crosshair at clicked position
            cv2.drawMarker(
                image,
                (self.click_x, self.click_y),
                color,
                cv2.MARKER_CROSS,
                20,
                2
            )
            
            # Draw distance text
            dist_text = self.format_distance(depth)
            cv2.putText(
                image,
                f"Click: {dist_text}",
                (self.click_x + 15, self.click_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        return image
    
    def draw_statistics(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Draw statistics overlay (min/max distance, FPS).
        
        Args:
            image: Image to draw on
            depth_map: Depth map for statistics
            fps: Current FPS
            
        Returns:
            Image with statistics
        """
        # Calculate statistics
        valid_depths = depth_map[depth_map > 0]
        
        if len(valid_depths) > 0:
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            avg_depth = np.mean(valid_depths)
        else:
            min_depth = 0
            max_depth = 0
            avg_depth = 0
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (250, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # Draw text
        y_offset = 30
        cv2.putText(
            image,
            f"FPS: {fps:.1f}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        
        y_offset += 25
        cv2.putText(
            image,
            f"Min: {self.format_distance(min_depth)}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        y_offset += 25
        cv2.putText(
            image,
            f"Max: {self.format_distance(max_depth)}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        y_offset += 25
        cv2.putText(
            image,
            f"Avg: {self.format_distance(avg_depth)}",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return image
    
    def create_visualization(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        depth_map: np.ndarray,
        fps: float = 0.0
    ) -> np.ndarray:
        """
        Create complete visualization with all overlays.
        
        Args:
            left_image: Left camera image (rectified)
            right_image: Right camera image (rectified)
            depth_map: Depth map in millimeters
            fps: Current FPS
            
        Returns:
            Combined visualization image
        """
        # Create depth colormap
        depth_colored = self.create_depth_colormap(depth_map)
        
        # Create copy of left image for overlays
        left_with_overlays = left_image.copy()
        
        # Draw all overlays
        left_with_overlays = self.draw_crosshair(left_with_overlays, depth_map)
        left_with_overlays = self.draw_measurement_grid(left_with_overlays, depth_map)
        left_with_overlays = self.draw_mouse_measurement(left_with_overlays, depth_map)
        left_with_overlays = self.draw_statistics(left_with_overlays, depth_map, fps)
        
        # Combine images into a grid
        # Top row: left image with overlays, right image
        top_row = np.hstack([left_with_overlays, right_image])
        
        # Bottom row: depth colormap (stretched to match top row width)
        depth_stretched = cv2.resize(depth_colored, (top_row.shape[1], depth_colored.shape[0]))
        
        # Add labels
        cv2.putText(
            top_row,
            "Left Camera",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        cv2.putText(
            top_row,
            "Right Camera",
            (left_image.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        cv2.putText(
            depth_stretched,
            f"Depth Map ({self.colormap_name})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        # Stack vertically
        combined = np.vstack([top_row, depth_stretched])
        
        return combined
    
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """
        Mouse callback for interactive distance measurement.
        
        Args:
            event: Mouse event type
            x: Mouse x coordinate
            y: Mouse y coordinate
            flags: Event flags
            param: Additional parameters
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.click_x = x
            self.click_y = y
    
    def toggle_measurements(self) -> None:
        """Toggle measurement grid display."""
        self.show_measurements = not self.show_measurements
        self.logger.info(f"Measurement grid: {'ON' if self.show_measurements else 'OFF'}")
    
    def toggle_crosshair(self) -> None:
        """Toggle crosshair display."""
        self.show_crosshair = not self.show_crosshair
        self.logger.info(f"Crosshair: {'ON' if self.show_crosshair else 'OFF'}")
