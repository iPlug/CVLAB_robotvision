"""
Robot control module for object tracking applications.

This module provides abstractions for robot control with specific implementations
for different robot types (myCobot, etc.).
"""

__version__ = "1.0.0"

# Core interfaces
from .robot_controller import RobotController

# Implementations  
from .mycobot_controller import MyCobotController

__all__ = [
    'RobotController',
    'MyCobotController'
]