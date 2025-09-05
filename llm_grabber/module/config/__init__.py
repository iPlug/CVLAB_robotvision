"""
Configuration management for different detection strategies.
"""

from .detection_config import DetectionConfig
from .bev_config import BEVConfig
from .pointnet_config import PointNetConfig
from .config_factory import ConfigFactory

__all__ = [
    'DetectionConfig',
    'BEVConfig',
    'PointNetConfig',
    'ConfigFactory'
]