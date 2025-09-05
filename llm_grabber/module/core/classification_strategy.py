"""
Abstract base class for classification strategies.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Any
import numpy as np


class ClassificationStrategy(ABC):
    """
    Abstract base class for object classification strategies.
    
    Classification strategies take individual point cloud clusters and classify
    them into object categories, returning a class name and confidence score.
    """
    
    @abstractmethod
    def classify_object(self, point_cloud: Any) -> Tuple[Optional[str], float]:
        """
        Classify a single object point cloud.
        
        Args:
            point_cloud: Point cloud data for a single object cluster.
                        Can be numpy array, Open3D PointCloud, or other format
                        depending on the specific strategy implementation.
            
        Returns:
            Tuple of (class_name, confidence) where:
            - class_name: String name of the predicted class, or None if classification failed
            - confidence: Float confidence score between 0 and 1
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of this classification strategy.
        
        Returns:
            String name of the strategy (e.g., "PointNet Subspace", "CNN Classifier")
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Check if the classification model is properly loaded and ready for inference.
        
        Returns:
            True if the strategy is ready for classification, False otherwise
        """
        pass
    
    def preprocess_cluster(self, point_cloud: Any) -> Any:
        """
        Optional preprocessing step for individual clusters before classification.
        Default implementation returns the point cloud unchanged.
        
        Args:
            point_cloud: Raw point cloud cluster
            
        Returns:
            Preprocessed point cloud cluster
        """
        return point_cloud
    
    def get_supported_classes(self) -> Optional[list]:
        """
        Get list of classes that this strategy can classify.
        Default implementation returns None (unknown classes).
        
        Returns:
            List of class names, or None if not applicable
        """
        return None
    
    def get_model_info(self) -> dict:
        """
        Get information about the underlying model.
        Default implementation returns basic strategy name.
        
        Returns:
            Dictionary with model information
        """
        return {
            "strategy_name": self.get_strategy_name(),
            "loaded": self.is_loaded()
        }