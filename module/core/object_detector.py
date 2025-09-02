"""
Main object detector class that orchestrates clustering and classification strategies.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import time

from .clustering_strategy import ClusteringStrategy
from .classification_strategy import ClassificationStrategy


class ObjectDetector:
    """
    Main object detection system that combines clustering and classification strategies.
    
    This class provides a unified interface for object detection by:
    1. Taking point cloud input
    2. Using a clustering strategy to find objects
    3. Using a classification strategy to identify each object
    4. Returning enriched detection results
    """
    
    def __init__(self, 
                 clustering_strategy: ClusteringStrategy,
                 classification_strategy: Optional[ClassificationStrategy] = None,
                 enable_timing: bool = True):
        """
        Initialize the object detector.
        
        Args:
            clustering_strategy: Strategy for finding objects in point clouds
            classification_strategy: Optional strategy for classifying detected objects
            enable_timing: Whether to track processing times
        """
        self.clustering_strategy = clustering_strategy
        self.classification_strategy = classification_strategy
        self.enable_timing = enable_timing
        
        # Statistics tracking
        self.stats = {
            "total_detections": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "clustering_time": 0.0,
            "classification_time": 0.0
        }
    
    def detect_objects(self, 
                      vertices: np.ndarray, 
                      color_image: Optional[np.ndarray] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Detect and classify objects in a point cloud.
        
        Args:
            vertices: Point cloud vertices as numpy array of shape (N, 3)
            color_image: Optional color image for strategies that use color information
            
        Returns:
            Tuple of (detected_objects, timing_info) where:
            - detected_objects: List of object dictionaries with cluster and classification info
            - timing_info: Dictionary with processing time statistics
        """
        start_time = time.time() if self.enable_timing else 0
        
        # Step 1: Cluster objects
        clustering_start = time.time() if self.enable_timing else 0
        clusters = self.clustering_strategy.cluster_objects(vertices, color_image)
        clustering_time = time.time() - clustering_start if self.enable_timing else 0
        
        # Step 2: Classify objects (if classification strategy is provided)
        classification_start = time.time() if self.enable_timing else 0
        detected_objects = []
        
        for cluster in clusters:
            # Start with cluster information
            obj_info = cluster.copy()
            
            # Add classification if strategy is available
            if self.classification_strategy and self.classification_strategy.is_loaded():
                try:
                    # Create point cloud object for classification
                    # The classification strategy will handle the specific format it needs
                    class_name, confidence = self.classification_strategy.classify_object(cluster)
                    
                    obj_info['class_name'] = class_name
                    obj_info['confidence'] = confidence
                    obj_info['classified'] = True
                    
                except Exception as e:
                    # Fallback if classification fails
                    obj_info['class_name'] = f"Obj_{cluster['id']:02d}"
                    obj_info['confidence'] = 0.0
                    obj_info['classified'] = False
                    obj_info['classification_error'] = str(e)
            else:
                # No classification available
                obj_info['class_name'] = f"Obj_{cluster['id']:02d}"
                obj_info['confidence'] = 0.0
                obj_info['classified'] = False
            
            detected_objects.append(obj_info)
        
        classification_time = time.time() - classification_start if self.enable_timing else 0
        total_time = time.time() - start_time if self.enable_timing else 0
        
        # Update statistics
        if self.enable_timing:
            self.stats["total_detections"] += 1
            self.stats["total_processing_time"] += total_time
            self.stats["average_processing_time"] = self.stats["total_processing_time"] / self.stats["total_detections"]
            self.stats["clustering_time"] += clustering_time
            self.stats["classification_time"] += classification_time
        
        # Timing information for this detection
        timing_info = {
            "total_time": total_time,
            "clustering_time": clustering_time,
            "classification_time": classification_time,
            "num_objects": len(detected_objects)
        }
        
        return detected_objects, timing_info
    
    def set_clustering_strategy(self, strategy: ClusteringStrategy):
        """
        Change the clustering strategy at runtime.
        
        Args:
            strategy: New clustering strategy to use
        """
        self.clustering_strategy = strategy
    
    def set_classification_strategy(self, strategy: Optional[ClassificationStrategy]):
        """
        Change the classification strategy at runtime.
        
        Args:
            strategy: New classification strategy to use, or None to disable classification
        """
        self.classification_strategy = strategy
    
    def get_strategy_info(self) -> Dict[str, str]:
        """
        Get information about current strategies.
        
        Returns:
            Dictionary with strategy information
        """
        info = {
            "clustering_strategy": self.clustering_strategy.get_strategy_name()
        }
        
        if self.classification_strategy:
            info["classification_strategy"] = self.classification_strategy.get_strategy_name()
            info["classification_loaded"] = str(self.classification_strategy.is_loaded())
        else:
            info["classification_strategy"] = "None"
            info["classification_loaded"] = "False"
        
        return info
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset all processing statistics."""
        self.stats = {
            "total_detections": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "clustering_time": 0.0,
            "classification_time": 0.0
        }