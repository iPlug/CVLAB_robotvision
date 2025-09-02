"""
Abstract base class for clustering strategies.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class ClusteringStrategy(ABC):
    """
    Abstract base class for object clustering strategies.
    
    Clustering strategies take point cloud data and identify distinct objects/clusters.
    They return a list of cluster information that can be used for further processing
    like classification or visualization.
    """
    
    @abstractmethod
    def cluster_objects(self, vertices: np.ndarray, color_image: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Cluster objects from point cloud data.
        
        Args:
            vertices: Point cloud vertices as numpy array of shape (N, 3) where each row is [x, y, z]
            color_image: Optional color image for color-based filtering
            
        Returns:
            List of cluster dictionaries, each containing:
            - 'id': Unique cluster identifier
            - 'points': Point cloud data for this cluster (numpy array)
            - 'center': 3D center point of the cluster [x, y, z]
            - 'min_coords': Minimum coordinates of bounding box [x_min, y_min, z_min]
            - 'max_coords': Maximum coordinates of bounding box [x_max, y_max, z_max]
            - 'dimensions': Cluster dimensions [width, height, depth]
            - 'num_points': Number of points in the cluster
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get the name of this clustering strategy.
        
        Returns:
            String name of the strategy (e.g., "BEV Clustering", "DBSCAN Clustering")
        """
        pass
    
    def preprocess_points(self, vertices: np.ndarray) -> np.ndarray:
        """
        Optional preprocessing step for point cloud data.
        Default implementation returns points unchanged.
        
        Args:
            vertices: Raw point cloud vertices
            
        Returns:
            Preprocessed point cloud vertices
        """
        return vertices
    
    def postprocess_clusters(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optional postprocessing step for cluster data.
        Default implementation returns clusters unchanged.
        
        Args:
            clusters: Raw cluster data
            
        Returns:
            Postprocessed cluster data
        """
        return clusters