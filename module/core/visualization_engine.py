"""
Unified visualization engine for object detection results.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt


class VisualizationEngine:
    """
    Unified visualization engine that can handle results from different detection strategies.
    
    Provides common visualization functions for:
    - Displaying detected objects with bounding boxes
    - Showing classification results
    - Creating overlays and annotations
    - Performance metrics display
    """
    
    def __init__(self, 
                 show_bounding_boxes: bool = True,
                 show_class_names: bool = True,
                 show_confidence: bool = True,
                 show_performance_info: bool = True):
        """
        Initialize the visualization engine.
        
        Args:
            show_bounding_boxes: Whether to draw bounding boxes around detected objects
            show_class_names: Whether to display object class names
            show_confidence: Whether to display classification confidence scores
            show_performance_info: Whether to display performance statistics
        """
        self.show_bounding_boxes = show_bounding_boxes
        self.show_class_names = show_class_names
        self.show_confidence = show_confidence
        self.show_performance_info = show_performance_info
        
        # Color palette for different object classes
        self.class_colors = {}
        self.default_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (255, 192, 203), # Pink
            (165, 42, 42),  # Brown
        ]
    
    def get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """
        Get a consistent color for a given class name.
        
        Args:
            class_name: Name of the object class
            
        Returns:
            RGB color tuple
        """
        if class_name not in self.class_colors:
            color_index = len(self.class_colors) % len(self.default_colors)
            self.class_colors[class_name] = self.default_colors[color_index]
        
        return self.class_colors[class_name]
    
    def draw_object_info_on_image(self, 
                                 image: np.ndarray,
                                 detected_objects: List[Dict[str, Any]],
                                 coordinate_transform_func: Optional[callable] = None) -> np.ndarray:
        """
        Draw object detection information on an image.
        
        Args:
            image: Input image to draw on
            detected_objects: List of detected object dictionaries
            coordinate_transform_func: Optional function to transform 3D coordinates to 2D image coordinates
            
        Returns:
            Image with detection overlay
        """
        result_image = image.copy()
        
        for obj in detected_objects:
            class_name = obj.get('class_name', 'Unknown')
            confidence = obj.get('confidence', 0.0)
            center_3d = obj.get('center', [0, 0, 0])
            
            # Get color for this class
            color = self.get_color_for_class(class_name)
            
            # If we have a coordinate transform function, use it to get 2D position
            if coordinate_transform_func:
                try:
                    center_2d = coordinate_transform_func(center_3d)
                    
                    # Draw a circle at the object center
                    cv2.circle(result_image, tuple(map(int, center_2d)), 8, color, -1)
                    cv2.circle(result_image, tuple(map(int, center_2d)), 10, (255, 255, 255), 2)
                    
                    # Draw text with class name and confidence
                    if self.show_class_names or self.show_confidence:
                        text_parts = []
                        if self.show_class_names:
                            text_parts.append(class_name)
                        if self.show_confidence and confidence > 0:
                            text_parts.append(f"{confidence:.2f}")
                        
                        text = " ".join(text_parts)
                        text_pos = (int(center_2d[0]) + 15, int(center_2d[1]) - 10)
                        
                        # Draw text background
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(result_image, 
                                    (text_pos[0] - 2, text_pos[1] - text_size[1] - 2),
                                    (text_pos[0] + text_size[0] + 2, text_pos[1] + 5),
                                    (0, 0, 0), -1)
                        
                        # Draw text
                        cv2.putText(result_image, text, text_pos,
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                except Exception as e:
                    # If coordinate transform fails, skip this object
                    continue
        
        return result_image
    
    def add_performance_overlay(self, 
                               image: np.ndarray,
                               timing_info: Dict[str, float],
                               strategy_info: Dict[str, str],
                               frame_count: int = 0) -> np.ndarray:
        """
        Add performance information overlay to an image.
        
        Args:
            image: Input image
            timing_info: Dictionary with timing information
            strategy_info: Dictionary with strategy information
            frame_count: Current frame number
            
        Returns:
            Image with performance overlay
        """
        if not self.show_performance_info:
            return image
        
        result_image = image.copy()
        
        # Performance text
        perf_text = [
            f"Frame: {frame_count}",
            f"Objects: {timing_info.get('num_objects', 0)}",
            f"Total: {timing_info.get('total_time', 0)*1000:.1f}ms",
            f"Clustering: {timing_info.get('clustering_time', 0)*1000:.1f}ms",
            f"Classification: {timing_info.get('classification_time', 0)*1000:.1f}ms",
            f"Strategy: {strategy_info.get('clustering_strategy', 'Unknown')}",
            f"Classifier: {strategy_info.get('classification_strategy', 'None')}"
        ]
        
        # Draw performance info
        y_offset = 30
        for i, text in enumerate(perf_text):
            y_pos = y_offset + i * 25
            
            # Text background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_image, 
                        (10, y_pos - text_size[1] - 5),
                        (15 + text_size[0], y_pos + 5),
                        (0, 0, 0), -1)
            
            # Text
            cv2.putText(result_image, text, (15, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def toggle_bounding_boxes(self):
        """Toggle bounding box display."""
        self.show_bounding_boxes = not self.show_bounding_boxes
    
    def toggle_class_names(self):
        """Toggle class name display."""
        self.show_class_names = not self.show_class_names
    
    def toggle_confidence(self):
        """Toggle confidence score display."""
        self.show_confidence = not self.show_confidence
    
    def toggle_performance_info(self):
        """Toggle performance information display."""
        self.show_performance_info = not self.show_performance_info
    
    def add_status_overlay(self, image: np.ndarray, status_message: str,
                          progress_info: Optional[Dict[str, Any]] = None,
                          position: Tuple[int, int] = (10, 30),
                          font_scale: float = 0.8,
                          color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Add status overlay to image with optional progress information.
        
        Args:
            image: Input image to add overlay to
            status_message: Main status message to display
            progress_info: Optional dictionary with additional progress information
            position: Text position as (x, y)
            font_scale: Font scale for main message
            color: Text color as (B, G, R)
            
        Returns:
            Image with status overlay added
        """
        result_image = image.copy()
        
        # Add main status message
        cv2.putText(result_image, status_message.upper(), position,
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        # Add progress information if provided
        if progress_info:
            y_offset = position[1] + 40
            
            # Add remaining time if specified
            if progress_info.get('remaining_time') is not None:
                remaining = progress_info['remaining_time']
                cv2.putText(result_image, f"Remaining: {remaining:.1f}s", 
                           (position[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            # Add position information if specified
            if progress_info.get('position') is not None:
                pos_text = f"Position: {progress_info['position']}"
                cv2.putText(result_image, pos_text, (position[0], y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            # Add custom lines if specified
            if progress_info.get('custom_lines'):
                for line in progress_info['custom_lines']:
                    cv2.putText(result_image, line, (position[0], y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    y_offset += 25
        
        return result_image
    
    def add_countdown_overlay(self, image: np.ndarray, remaining_time: float,
                             message: str = "STARTING IN",
                             show_large_number: bool = True) -> np.ndarray:
        """
        Add countdown overlay with large countdown number.
        
        Args:
            image: Input image
            remaining_time: Remaining time in seconds
            message: Countdown message
            show_large_number: Whether to show large countdown number
            
        Returns:
            Image with countdown overlay
        """
        result_image = image.copy()
        
        # Add countdown message
        cv2.putText(result_image, message, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add large countdown number if enabled
        if show_large_number:
            countdown_num = int(remaining_time) + 1
            text_size = cv2.getTextSize(str(countdown_num), cv2.FONT_HERSHEY_SIMPLEX, 4.0, 8)[0]
            
            # Center the countdown number
            x = (result_image.shape[1] - text_size[0]) // 2
            y = (result_image.shape[0] + text_size[1]) // 2
            
            cv2.putText(result_image, str(countdown_num), (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 8)
        
        # Add remaining time in text
        cv2.putText(result_image, f"Remaining: {remaining_time:.1f}s", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image
    
    def create_multi_panel_display(self, panels: List[np.ndarray], 
                                  panel_height: int = 400,
                                  spacing: int = 5,
                                  background_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        Create a multi-panel horizontal display from a list of images.
        
        Args:
            panels: List of images to display horizontally
            panel_height: Target height for all panels
            spacing: Spacing between panels in pixels
            background_color: Background color for spacing
            
        Returns:
            Combined multi-panel image
        """
        if not panels:
            return np.zeros((panel_height, 640, 3), dtype=np.uint8) + background_color
        
        # Resize all panels to target height while maintaining aspect ratio
        resized_panels = []
        for panel in panels:
            if len(panel.shape) == 2:  # Grayscale to color
                panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
            
            aspect_ratio = panel.shape[1] / panel.shape[0]
            panel_width = int(panel_height * aspect_ratio)
            resized_panel = cv2.resize(panel, (panel_width, panel_height))
            resized_panels.append(resized_panel)
        
        # Calculate total width
        total_width = sum(panel.shape[1] for panel in resized_panels) + spacing * (len(resized_panels) - 1)
        
        # Create combined image
        combined = np.zeros((panel_height, total_width, 3), dtype=np.uint8) + background_color
        
        # Place panels
        x_offset = 0
        for panel in resized_panels:
            combined[:, x_offset:x_offset + panel.shape[1]] = panel
            x_offset += panel.shape[1] + spacing
        
        return combined
    
    def add_info_overlay(self, image: np.ndarray, info_lines: List[str],
                        position: Tuple[int, int] = (10, 30),
                        colors: Optional[List[Tuple[int, int, int]]] = None,
                        font_scale: float = 0.6,
                        line_spacing: int = 25) -> np.ndarray:
        """
        Add multiple lines of information as overlay.
        
        Args:
            image: Input image
            info_lines: List of text lines to display
            position: Starting position for first line
            colors: Optional list of colors for each line (uses white if not specified)
            font_scale: Font scale for text
            line_spacing: Vertical spacing between lines
            
        Returns:
            Image with info overlay added
        """
        result_image = image.copy()
        
        for i, line in enumerate(info_lines):
            y_pos = position[1] + i * line_spacing
            
            # Use specified color or default to white
            if colors and i < len(colors):
                color = colors[i]
            else:
                color = (255, 255, 255)
            
            cv2.putText(result_image, line, (position[0], y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        return result_image