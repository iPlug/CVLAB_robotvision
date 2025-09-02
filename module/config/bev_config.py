"""
Configuration for BEV clustering strategy.
"""

from typing import Dict, Any, Optional
from .detection_config import DetectionConfig


class BEVConfig(DetectionConfig):
    """
    Configuration class for BEV (Bird's Eye View) clustering strategy.
    
    Contains all parameters needed for BEV-based object detection:
    - ROI parameters for point filtering
    - Display parameters for visualization
    - Clustering parameters
    - Temporal filtering settings
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize BEV configuration with defaults.
        
        Args:
            config_dict: Optional dictionary with custom parameters
        """
        # Initialize base configuration
        super().__init__()
        
        # BEV-specific default configuration
        bev_defaults = {
            # ROI parameters (in meters, after rotation) - for filtering points
            'roi_params': {
                'x_min': -1.0,
                'x_max': 1.0,
                'y_min': 0.08,   # Height range
                'y_max': 0.135,
                'z_min': 0,      # Depth range
                'z_max': 0.5
            },
            
            # Display ROI parameters (in meters) - for limiting BEV display area
            'display_roi_params': {
                'x_min': -1.5,
                'x_max': 1.5,
                'y_min': -0.8,   # Height range
                'y_max': 0.5,
                'z_min': 0,      # Depth range
                'z_max': 1.0
            },
            
            # BEV clustering parameters
            'clustering_params': {
                'min_cluster_size': 30,   # Minimum pixels for a valid cluster
                'max_cluster_size': 150,  # Maximum pixels for a valid cluster
                'morphology_kernel_size': 5,  # Kernel size for morphological operations
                'window_size': 5,         # Number of frames to average for noise reduction
            },
            
            # BEV image creation parameters
            'bev_params': {
                'image_size': 400,        # Default BEV image size
                'height_colormap': True,  # Whether to color points by height
                'point_size': 2,          # Point size for drawing
                'use_aspect_ratio': True  # Maintain aspect ratio
            },
            
            # Temporal filtering parameters
            'temporal_params': {
                'enabled': True,
                'persistence_threshold': 0.7,  # Minimum ratio of frames for pixel persistence
                'min_frames_for_filtering': 3   # Minimum frames needed for temporal filtering
            },
            
            # Morphological operations parameters
            'morphology_params': {
                'operations': ['open', 'close'],  # Sequence of operations to apply
                'kernel_shape': 'ellipse',        # 'ellipse', 'rect', 'cross'
                'open_kernel_size': 5,
                'close_kernel_size': 5
            },
            
            # Visualization parameters
            'visualization_params': {
                'show_bounding_boxes': True,
                'show_cluster_ids': True,
                'show_coordinate_grid': False,
                'grid_spacing': 0.1,
                'bbox_color': [0, 255, 0],      # Green bounding boxes
                'text_color': [255, 255, 255]   # White text
            },
            
            # Performance parameters
            'performance_params': {
                'enable_gpu_acceleration': False,
                'max_points_per_frame': 100000,
                'downsample_for_bev': True,
                'downsample_voxel_size': 0.01
            }
        }
        
        # Update with defaults
        self.update_config(bev_defaults)
        
        # Update with provided configuration
        if config_dict:
            self.update_config(config_dict)
    
    def validate_config(self) -> bool:
        """
        Validate BEV-specific configuration parameters.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Validate ROI parameters
            roi = self.get('roi_params', {})
            if roi['x_min'] >= roi['x_max']:
                print("Error: ROI x_min must be less than x_max")
                return False
            if roi['y_min'] >= roi['y_max']:
                print("Error: ROI y_min must be less than y_max")
                return False
            if roi['z_min'] >= roi['z_max']:
                print("Error: ROI z_min must be less than z_max")
                return False
            
            # Validate clustering parameters
            cluster = self.get('clustering_params', {})
            if cluster['min_cluster_size'] <= 0:
                print("Error: min_cluster_size must be positive")
                return False
            if cluster['max_cluster_size'] <= cluster['min_cluster_size']:
                print("Error: max_cluster_size must be greater than min_cluster_size")
                return False
            if cluster['window_size'] <= 0:
                print("Error: window_size must be positive")
                return False
            
            # Validate BEV parameters
            bev = self.get('bev_params', {})
            if bev['image_size'] <= 0:
                print("Error: image_size must be positive")
                return False
            
            # Validate temporal parameters
            temporal = self.get('temporal_params', {})
            persistence = temporal.get('persistence_threshold', 0.7)
            if not (0.0 <= persistence <= 1.0):
                print("Error: persistence_threshold must be between 0 and 1")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating BEV configuration: {e}")
            return False
    
    def reset_to_defaults(self):
        """Reset configuration to BEV defaults."""
        self.__init__()
    
    def get_roi_bounds(self) -> Dict[str, float]:
        """
        Get ROI bounds in a convenient format.
        
        Returns:
            Dictionary with ROI bounds
        """
        roi = self.get('roi_params', {})
        return {
            'x_range': (roi['x_min'], roi['x_max']),
            'y_range': (roi['y_min'], roi['y_max']),
            'z_range': (roi['z_min'], roi['z_max'])
        }
    
    def get_display_bounds(self) -> Dict[str, float]:
        """
        Get display ROI bounds in a convenient format.
        
        Returns:
            Dictionary with display bounds
        """
        display_roi = self.get('display_roi_params', {})
        return {
            'x_range': (display_roi['x_min'], display_roi['x_max']),
            'y_range': (display_roi['y_min'], display_roi['y_max']),
            'z_range': (display_roi['z_min'], display_roi['z_max'])
        }
    
    def set_roi_bounds(self, x_range: tuple, y_range: tuple, z_range: tuple):
        """
        Set ROI bounds from range tuples.
        
        Args:
            x_range: (x_min, x_max) tuple
            y_range: (y_min, y_max) tuple
            z_range: (z_min, z_max) tuple
        """
        self.set('roi_params.x_min', x_range[0])
        self.set('roi_params.x_max', x_range[1])
        self.set('roi_params.y_min', y_range[0])
        self.set('roi_params.y_max', y_range[1])
        self.set('roi_params.z_min', z_range[0])
        self.set('roi_params.z_max', z_range[1])
    
    def set_display_bounds(self, x_range: tuple, y_range: tuple, z_range: tuple):
        """
        Set display ROI bounds from range tuples.
        
        Args:
            x_range: (x_min, x_max) tuple
            y_range: (y_min, y_max) tuple
            z_range: (z_min, z_max) tuple
        """
        self.set('display_roi_params.x_min', x_range[0])
        self.set('display_roi_params.x_max', x_range[1])
        self.set('display_roi_params.y_min', y_range[0])
        self.set('display_roi_params.y_max', y_range[1])
        self.set('display_roi_params.z_min', z_range[0])
        self.set('display_roi_params.z_max', z_range[1])
    
    def get_clustering_settings(self) -> Dict[str, Any]:
        """
        Get clustering settings in a convenient format.
        
        Returns:
            Dictionary with clustering settings
        """
        return self.get('clustering_params', {})
    
    def create_optimized_config(self) -> 'BEVConfig':
        """
        Create an optimized configuration for better performance.
        
        Returns:
            New BEVConfig instance with optimized settings
        """
        optimized = self.copy()
        
        # Optimize for performance
        optimized.set('clustering_params.window_size', 3)  # Smaller temporal window
        optimized.set('bev_params.image_size', 300)        # Smaller image size
        optimized.set('morphology_params.open_kernel_size', 3)
        optimized.set('morphology_params.close_kernel_size', 3)
        optimized.set('performance_params.enable_gpu_acceleration', True)
        optimized.set('performance_params.downsample_for_bev', True)
        optimized.set('temporal_params.min_frames_for_filtering', 2)
        
        return optimized
    
    def create_high_quality_config(self) -> 'BEVConfig':
        """
        Create a high-quality configuration for better accuracy.
        
        Returns:
            New BEVConfig instance with high-quality settings
        """
        high_quality = self.copy()
        
        # Optimize for quality
        high_quality.set('clustering_params.window_size', 7)     # Larger temporal window
        high_quality.set('bev_params.image_size', 600)           # Larger image size
        high_quality.set('clustering_params.min_cluster_size', 20)  # More sensitive clustering
        high_quality.set('temporal_params.persistence_threshold', 0.8)  # Stricter persistence
        high_quality.set('temporal_params.min_frames_for_filtering', 5)
        high_quality.set('performance_params.downsample_for_bev', False)  # No downsampling
        
        return high_quality