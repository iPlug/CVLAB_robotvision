"""
Application classes for different use cases.
"""

from .unified_detection_app import UnifiedDetectionApp
from .realtime_detection_app import RealTimeDetectionApp
from .batch_processing_app import BatchProcessingApp

__all__ = [
    'UnifiedDetectionApp',
    'RealTimeDetectionApp',
    'BatchProcessingApp'
]