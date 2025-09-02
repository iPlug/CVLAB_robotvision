"""
Abstract base class for sensor management.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any
import numpy as np


class SensorManager(ABC):
    """
    Abstract base class for managing different sensor inputs.
    
    Provides a unified interface for:
    - Sensor initialization and configuration
    - Frame acquisition and processing
    - Sensor parameter management
    - Error handling and cleanup
    """
    
    def __init__(self):
        """Initialize sensor manager."""
        self.is_initialized = False
        self.is_running = False
        self.frame_count = 0
        
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the sensor.
        
        Args:
            **kwargs: Sensor-specific initialization parameters
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """
        Start sensor data acquisition.
        
        Returns:
            True if started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def stop(self):
        """Stop sensor data acquisition."""
        pass
    
    @abstractmethod
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Get the next frame from the sensor.
        
        Returns:
            Tuple of (vertices, color_image, metadata) where:
            - vertices: 3D point cloud vertices (N, 3) or None
            - color_image: RGB/BGR color image or None
            - metadata: Dictionary with frame metadata (timestamp, frame_id, etc.)
        """
        pass
    
    @abstractmethod
    def get_sensor_info(self) -> Dict[str, Any]:
        """
        Get sensor information and capabilities.
        
        Returns:
            Dictionary with sensor information
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if sensor is available and working.
        
        Returns:
            True if sensor is available, False otherwise
        """
        pass
    
    def cleanup(self):
        """
        Cleanup sensor resources.
        Default implementation stops the sensor.
        """
        if self.is_running:
            self.stop()
        self.is_initialized = False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current sensor status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'initialized': self.is_initialized,
            'running': self.is_running,
            'frame_count': self.frame_count,
            'available': self.is_available()
        }
    
    def reset_frame_count(self):
        """Reset the frame counter."""
        self.frame_count = 0
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()