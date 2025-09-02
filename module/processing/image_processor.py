"""
Image processing utilities for BEV creation and manipulation.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional


class ImageProcessor:
    """
    Image processing utilities for Bird's Eye View (BEV) operations.
    
    Provides utilities for:
    - BEV image creation from point clouds
    - Morphological operations
    - Connected components analysis
    - Color filtering and thresholding
    """
    
    def __init__(self, default_image_size: int = 400):
        """
        Initialize image processor.
        
        Args:
            default_image_size: Default size for generated images
        """
        self.default_image_size = default_image_size
    
    def create_bev_image(self, vertices: np.ndarray,
                        x_range: Tuple[float, float],
                        z_range: Tuple[float, float],
                        height_colormap: bool = True,
                        height_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Create Bird's Eye View image from 3D point cloud.
        
        Args:
            vertices: Point cloud vertices (N, 3)
            x_range: X-axis range (min, max) in meters
            z_range: Z-axis range (min, max) in meters  
            height_colormap: Whether to color points by height
            height_range: Optional height range for coloring
            
        Returns:
            BEV image as numpy array
        """
        if vertices is None or len(vertices) == 0:
            return np.zeros((self.default_image_size, self.default_image_size, 3), dtype=np.uint8)
        
        # Filter out invalid points
        valid_mask = vertices[:, 2] > 0
        valid_points = vertices[valid_mask]
        
        if len(valid_points) == 0:
            return np.zeros((self.default_image_size, self.default_image_size, 3), dtype=np.uint8)
        
        # Use aligned coordinates for top-down view
        x_coords = valid_points[:, 0]  # Left-right
        z_coords = valid_points[:, 2]  # Forward-backward  
        y_coords = valid_points[:, 1]  # Height
        
        # Calculate aspect ratio and image dimensions
        x_span = x_range[1] - x_range[0]
        z_span = z_range[1] - z_range[0]
        
        if x_span == 0 or z_span == 0:
            return np.zeros((self.default_image_size, self.default_image_size, 3), dtype=np.uint8)
        
        # Maintain aspect ratio
        aspect_ratio = x_span / z_span
        
        if aspect_ratio > 1:  # Wider than tall
            img_width = self.default_image_size
            img_height = int(self.default_image_size / aspect_ratio)
        else:  # Taller than wide
            img_height = self.default_image_size
            img_width = int(self.default_image_size * aspect_ratio)
        
        # Ensure minimum size
        img_width = max(img_width, 100)
        img_height = max(img_height, 100)
        
        # Convert to image coordinates
        x_img = ((x_coords - x_range[0]) / x_span * img_width).astype(int)
        z_img = ((z_coords - z_range[0]) / z_span * img_height).astype(int)
        
        # Filter points within image bounds
        valid_img_mask = (x_img >= 0) & (x_img < img_width) & (z_img >= 0) & (z_img < img_height)
        x_img = x_img[valid_img_mask]
        z_img = z_img[valid_img_mask]
        y_valid = y_coords[valid_img_mask]
        
        # Create top-down image
        bev_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Color points
        if len(y_valid) > 0:
            if height_colormap:
                # Color by height
                if height_range is not None:
                    y_min, y_max = height_range
                else:
                    y_min, y_max = np.min(y_valid), np.max(y_valid)
                
                if y_max > y_min:
                    y_normalized = np.clip((y_valid - y_min) / (y_max - y_min), 0, 1)
                else:
                    y_normalized = np.zeros_like(y_valid)
                
                # Apply colormap (red for high, blue for low)
                colors = plt.cm.jet(y_normalized)[:, :3] * 255
            else:
                # Single color
                colors = np.full((len(y_valid), 3), [255, 255, 255])
            
            # Draw points
            point_size = max(1, min(3, self.default_image_size // 200))
            
            for i in range(len(x_img)):
                cv2.circle(bev_image, (x_img[i], img_height - z_img[i] - 1), point_size, 
                          (int(colors[i][2]), int(colors[i][1]), int(colors[i][0])), -1)
        
        return bev_image
    
    def apply_morphological_operations(self, image: np.ndarray,
                                     kernel_size: int = 5,
                                     operations: list = None) -> np.ndarray:
        """
        Apply morphological operations for noise reduction.
        
        Args:
            image: Input image
            kernel_size: Size of morphological kernel
            operations: List of operations ('open', 'close', 'erode', 'dilate')
            
        Returns:
            Processed image
        """
        if operations is None:
            operations = ['open', 'close']
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply binary threshold
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply operations
        result = binary
        for op in operations:
            if op == 'open':
                result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            elif op == 'close':
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            elif op == 'erode':
                result = cv2.erode(result, kernel, iterations=1)
            elif op == 'dilate':
                result = cv2.dilate(result, kernel, iterations=1)
        
        # Convert back to BGR if input was color
        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def find_connected_components(self, binary_image: np.ndarray,
                                min_size: int = 30,
                                max_size: int = 150) -> Tuple[int, np.ndarray, list]:
        """
        Find connected components in binary image.
        
        Args:
            binary_image: Binary input image
            min_size: Minimum component size to keep
            max_size: Maximum component size to keep
            
        Returns:
            Tuple of (num_components, labels_image, valid_components)
        """
        # Convert to grayscale if needed
        if len(binary_image.shape) == 3:
            gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary_image
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(gray)
        
        # Filter components by size
        valid_components = []
        for label in range(1, num_labels):  # Skip background (0)
            mask = labels == label
            component_size = np.sum(mask)
            
            if min_size <= component_size <= max_size:
                valid_components.append({
                    'label': label,
                    'mask': mask,
                    'size': component_size
                })
        
        return num_labels, labels, valid_components
    
    def apply_color_filter(self, image: np.ndarray,
                          color_ranges: dict) -> np.ndarray:
        """
        Apply RGB color filtering to image.
        
        Args:
            image: Input color image
            color_ranges: Dictionary with color filter parameters:
                - 'r_min', 'r_max': Red channel range (0-1)
                - 'g_min', 'g_max': Green channel range (0-1)  
                - 'b_min', 'b_max': Blue channel range (0-1)
                
        Returns:
            Filtered binary mask
        """
        if len(image.shape) != 3:
            return image
        
        # Normalize to 0-1 range
        image_norm = image.astype(np.float32) / 255.0
        
        # Extract ranges
        r_min, r_max = color_ranges.get('r_min', 0), color_ranges.get('r_max', 1)
        g_min, g_max = color_ranges.get('g_min', 0), color_ranges.get('g_max', 1)
        b_min, b_max = color_ranges.get('b_min', 0), color_ranges.get('b_max', 1)
        
        # Create mask (BGR format)
        mask = ((image_norm[:, :, 2] >= r_min) & (image_norm[:, :, 2] <= r_max) &  # R channel
                (image_norm[:, :, 1] >= g_min) & (image_norm[:, :, 1] <= g_max) &  # G channel
                (image_norm[:, :, 0] >= b_min) & (image_norm[:, :, 0] <= b_max))   # B channel
        
        # Convert to uint8
        result = mask.astype(np.uint8) * 255
        
        return result
    
    def create_visualization_overlay(self, base_image: np.ndarray,
                                   components: list,
                                   x_range: Tuple[float, float],
                                   z_range: Tuple[float, float],
                                   show_bboxes: bool = True,
                                   show_labels: bool = True) -> np.ndarray:
        """
        Create visualization overlay with bounding boxes and labels.
        
        Args:
            base_image: Base image to draw on
            components: List of connected components
            x_range: X-axis range for coordinate conversion
            z_range: Z-axis range for coordinate conversion
            show_bboxes: Whether to draw bounding boxes
            show_labels: Whether to draw component labels
            
        Returns:
            Image with overlay
        """
        result = base_image.copy()
        height, width = result.shape[:2]
        
        # Generate colors for components
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255), (255, 165, 0), (128, 0, 128)]
        
        for i, comp in enumerate(components):
            color = colors[i % len(colors)]
            mask = comp['mask']
            
            if show_bboxes:
                # Find bounding box of component
                coords = np.column_stack(np.where(mask))
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    
                    # Draw bounding box
                    cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, 2)
            
            if show_labels:
                # Find center of component
                coords = np.column_stack(np.where(mask))
                if len(coords) > 0:
                    center_y, center_x = coords.mean(axis=0).astype(int)
                    
                    # Draw label
                    label = f"C{comp['label']}"
                    cv2.putText(result, label, (center_x + 5, center_y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result
    
    def resize_maintaining_aspect_ratio(self, image: np.ndarray, 
                                      target_size: int) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            target_size: Target size for the larger dimension
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        if width > height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
        
        return cv2.resize(image, (new_width, new_height))
    
    def add_coordinate_grid(self, image: np.ndarray,
                           x_range: Tuple[float, float],
                           z_range: Tuple[float, float],
                           grid_spacing: float = 0.1) -> np.ndarray:
        """
        Add coordinate grid to BEV image.
        
        Args:
            image: Input BEV image
            x_range: X-axis range in meters
            z_range: Z-axis range in meters
            grid_spacing: Grid spacing in meters
            
        Returns:
            Image with coordinate grid
        """
        result = image.copy()
        height, width = result.shape[:2]
        
        x_span = x_range[1] - x_range[0]
        z_span = z_range[1] - z_range[0]
        
        # Draw vertical lines (constant X)
        x_start = np.ceil(x_range[0] / grid_spacing) * grid_spacing
        x = x_start
        while x <= x_range[1]:
            x_img = int((x - x_range[0]) / x_span * width)
            if 0 <= x_img < width:
                cv2.line(result, (x_img, 0), (x_img, height-1), (128, 128, 128), 1)
            x += grid_spacing
        
        # Draw horizontal lines (constant Z)
        z_start = np.ceil(z_range[0] / grid_spacing) * grid_spacing
        z = z_start
        while z <= z_range[1]:
            z_img = int((z - z_range[0]) / z_span * height)
            z_img = height - z_img - 1  # Flip for display
            if 0 <= z_img < height:
                cv2.line(result, (0, z_img), (width-1, z_img), (128, 128, 128), 1)
            z += grid_spacing
        
        return result