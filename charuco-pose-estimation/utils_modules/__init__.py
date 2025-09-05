#!/usr/bin/env python3
"""
Modular utilities package for CharUco pose estimation applications.

This package splits the original monolithic utils.py into focused modules:
- robot_utils.py: Robot connection and movement management
- camera_utils.py: Camera setup and intrinsics management  
- calibration_utils.py: Eye-in-hand calibration workflow and validation
- coordinate_utils.py: Coordinate transformations and pose handling
- detection_utils.py: Common detection utilities and result saving

Usage:
    from utils_modules import RobotUtils, CameraUtils, CalibrationUtils, CoordinateUtils, DetectionUtils
    
    # Or import individual classes
    from utils_modules.robot_utils import RobotUtils
    from utils_modules.camera_utils import CameraUtils
    # etc.
"""

from .robot_utils import RobotUtils
from .camera_utils import CameraUtils  
from .calibration_utils import CalibrationUtils
from .coordinate_utils import CoordinateUtils
from .detection_utils import DetectionUtils
from .point_cloud_utils import PointCloudUtils
from .visualization_utils import VisualizationUtils

__all__ = [
    'RobotUtils',
    'CameraUtils', 
    'CalibrationUtils',
    'CoordinateUtils',
    'DetectionUtils',
    'PointCloudUtils',
    'VisualizationUtils'
]