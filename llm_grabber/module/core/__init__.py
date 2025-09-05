"""
Core abstract base classes and interfaces for the object detection system.
"""

from .object_detector import ObjectDetector
from .clustering_strategy import ClusteringStrategy
from .classification_strategy import ClassificationStrategy
from .visualization_engine import VisualizationEngine

__all__ = [
    'ObjectDetector',
    'ClusteringStrategy',
    'ClassificationStrategy', 
    'VisualizationEngine'
]