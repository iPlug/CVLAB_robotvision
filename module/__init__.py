"""
Modular object detection and clustering system for computer vision applications.

This module provides pluggable strategies for:
- Object clustering (BEV-based, DBSCAN-based)
- Object classification (PointNet + Subspace methods)
- Sensor management (RealSense, file-based)
- Unified applications for real-time and batch processing
"""

__version__ = "1.0.0"
__author__ = "CVLAB Robotvision"

# Core interfaces
from .core.object_detector import ObjectDetector
from .core.clustering_strategy import ClusteringStrategy
from .core.classification_strategy import ClassificationStrategy

# Strategy implementations
from .strategies.bev_clustering_strategy import BEVClusteringStrategy
from .strategies.dbscan_clustering_strategy import DBSCANClusteringStrategy
from .strategies.pointnet_subspace_classifier_strategy import PointNetSubspaceClassifierStrategy

__all__ = [
    'ObjectDetector',
    'ClusteringStrategy', 
    'ClassificationStrategy',
    'BEVClusteringStrategy',
    'DBSCANClusteringStrategy',
    'PointNetSubspaceClassifierStrategy'
]