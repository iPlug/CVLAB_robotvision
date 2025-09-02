"""
Utility modules for common functionality across applications.

This package provides reusable utilities that are shared across multiple
detection and tracking applications to reduce code duplication and ensure
consistent behavior.

Modules:
- object_utils: Object position extraction and selection utilities
- robot_utils: Robot setup, movement, and initialization utilities  
- wait_utils: Non-blocking wait functions with live visualization
"""

from .object_utils import (
    extract_object_position,
    find_closest_object,
    find_largest_object,
    validate_object_for_calibration,
    calculate_object_distance
)

from .robot_utils import (
    setup_robot_with_stabilization,
    move_to_standard_pose,
    initialize_robot_for_application
)

from .wait_utils import (
    wait_with_live_visualization,
    wait_for_command_completion
)

__all__ = [
    # Object utilities
    'extract_object_position',
    'find_closest_object', 
    'find_largest_object',
    'validate_object_for_calibration',
    'calculate_object_distance',
    
    # Robot utilities
    'setup_robot_with_stabilization',
    'move_to_standard_pose',
    'initialize_robot_for_application',
    
    # Wait utilities
    'wait_with_live_visualization',
    'wait_for_command_completion'
]