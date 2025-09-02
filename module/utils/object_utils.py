"""
Object processing utilities for position extraction and object selection.

This module provides common utilities for working with detected objects,
including position extraction, object selection, and validation functions
that are shared across multiple applications.
"""

import numpy as np
from typing import List, Dict, Any, Optional


def extract_object_position(obj: Dict[str, Any]) -> Optional[List[float]]:
    """
    Extract 3D position from object in camera coordinates.
    
    Handles both 'center_mm' and 'center' formats consistently across all applications.
    
    Args:
        obj: Detected object dictionary containing position information
        
    Returns:
        Position as [forward, side, height] in mm, or None if extraction fails
        
    Note:
        - center_mm format: [x, y, z] = [side, height, forward] in mm
        - center format: [x, y, z] = [side, height, forward] in meters  
        - Returns: [forward, side, height] for consistency with robot coordinates
    """
    try:
        if 'center_mm' in obj:
            # center_mm format: [x, y, z] = [side, height, forward]
            center_mm = obj['center_mm']
            return [center_mm[2], center_mm[0], center_mm[1]]  # [forward, side, height]
        elif 'center' in obj:
            # center format: [x, y, z] = [side, height, forward] in meters
            center = obj['center']
            return [center[2]*1000, center[0]*1000, center[1]*1000]  # Convert to mm
        else:
            return None
    except (KeyError, IndexError, TypeError):
        return None


def find_closest_object(detected_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find the closest detected object by distance from camera origin.
    
    Args:
        detected_objects: List of detected object dictionaries
        
    Returns:
        Closest object dictionary, or None if no valid objects found
    """
    if not detected_objects:
        return None
    
    closest_object = None
    min_distance = float('inf')
    
    for obj in detected_objects:
        position = extract_object_position(obj)
        if position is None:
            continue
        
        # Calculate distance from camera origin
        distance = calculate_object_distance(obj)
        if distance is not None and distance < min_distance:
            min_distance = distance
            closest_object = obj
    
    return closest_object


def find_largest_object(detected_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find the largest detected object by volume or area.
    
    Args:
        detected_objects: List of detected object dictionaries
        
    Returns:
        Largest object dictionary, or None if no valid objects found
    """
    if not detected_objects:
        return None
    
    largest_object = None
    max_size = 0.0
    
    for obj in detected_objects:
        # Try different size metrics in order of preference
        size = None
        
        # Method 1: Volume (best if available)
        if 'volume' in obj:
            size = obj['volume']
        # Method 2: Bounding box volume
        elif 'bbox' in obj and len(obj['bbox']) >= 6:
            bbox = obj['bbox']
            size = abs((bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (bbox[5] - bbox[2]))
        # Method 3: Point count as proxy for size
        elif 'point_count' in obj:
            size = obj['point_count']
        # Method 4: Area (2D approximation)
        elif 'area' in obj:
            size = obj['area']
        
        if size is not None and size > max_size:
            max_size = size
            largest_object = obj
    
    return largest_object


def calculate_object_distance(obj: Dict[str, Any]) -> Optional[float]:
    """
    Calculate distance from camera origin to object.
    
    Args:
        obj: Detected object dictionary
        
    Returns:
        Distance in mm, or None if calculation fails
    """
    position = extract_object_position(obj)
    if position is None:
        return None
    
    try:
        # Calculate Euclidean distance from origin
        distance = np.sqrt(sum(p**2 for p in position))
        return float(distance)
    except (TypeError, ValueError):
        return None


def validate_object_for_calibration(obj: Dict[str, Any], 
                                  min_size: Optional[float] = None,
                                  max_distance: Optional[float] = None,
                                  min_distance: Optional[float] = None) -> bool:
    """
    Validate if an object is suitable for calibration purposes.
    
    Args:
        obj: Detected object dictionary
        min_size: Minimum object size (volume, area, or point count)
        max_distance: Maximum distance from camera in mm
        min_distance: Minimum distance from camera in mm
        
    Returns:
        True if object passes validation criteria
    """
    # Check if we can extract position
    position = extract_object_position(obj)
    if position is None:
        return False
    
    # Check distance constraints
    if max_distance is not None or min_distance is not None:
        distance = calculate_object_distance(obj)
        if distance is None:
            return False
        
        if max_distance is not None and distance > max_distance:
            return False
        
        if min_distance is not None and distance < min_distance:
            return False
    
    # Check size constraints
    if min_size is not None:
        # Try different size metrics
        size = None
        
        if 'volume' in obj:
            size = obj['volume']
        elif 'point_count' in obj:
            size = obj['point_count']
        elif 'area' in obj:
            size = obj['area']
        elif 'bbox' in obj and len(obj['bbox']) >= 6:
            bbox = obj['bbox']
            size = abs((bbox[3] - bbox[0]) * (bbox[4] - bbox[1]) * (bbox[5] - bbox[2]))
        
        if size is None or size < min_size:
            return False
    
    return True


def validate_object_for_interaction(obj: Dict[str, Any],
                                  min_size: Optional[float] = None,
                                  max_distance: Optional[float] = None,
                                  reachable_workspace: Optional[Dict[str, tuple]] = None) -> bool:
    """
    Validate if an object is suitable for robot interaction.
    
    Args:
        obj: Detected object dictionary
        min_size: Minimum object size for reliable interaction
        max_distance: Maximum distance for robot reach
        reachable_workspace: Workspace bounds as {'x': (min, max), 'y': (min, max), 'z': (min, max)}
        
    Returns:
        True if object is suitable for robot interaction
    """
    position = extract_object_position(obj)
    if position is None:
        return False
    
    # Check distance constraint
    if max_distance is not None:
        distance = calculate_object_distance(obj)
        if distance is None or distance > max_distance:
            return False
    
    # Check workspace bounds
    if reachable_workspace is not None:
        forward, side, height = position
        
        if 'x' in reachable_workspace:
            x_min, x_max = reachable_workspace['x']
            if not (x_min <= forward <= x_max):
                return False
        
        if 'y' in reachable_workspace:
            y_min, y_max = reachable_workspace['y']  
            if not (y_min <= side <= y_max):
                return False
        
        if 'z' in reachable_workspace:
            z_min, z_max = reachable_workspace['z']
            if not (z_min <= height <= z_max):
                return False
    
    # Check size constraint
    if min_size is not None:
        if not validate_object_for_calibration(obj, min_size=min_size):
            return False
    
    return True


def get_object_info_string(obj: Dict[str, Any], include_position: bool = True, 
                          include_size: bool = True) -> str:
    """
    Generate a human-readable info string for an object.
    
    Args:
        obj: Detected object dictionary
        include_position: Whether to include position information
        include_size: Whether to include size information
        
    Returns:
        Formatted info string
    """
    info_parts = []
    
    # Add position info
    if include_position:
        position = extract_object_position(obj)
        if position is not None:
            info_parts.append(f"Pos: ({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f})mm")
        
        distance = calculate_object_distance(obj)
        if distance is not None:
            info_parts.append(f"Dist: {distance:.0f}mm")
    
    # Add size info
    if include_size:
        if 'volume' in obj:
            info_parts.append(f"Vol: {obj['volume']:.1f}")
        elif 'point_count' in obj:
            info_parts.append(f"Points: {obj['point_count']}")
        elif 'area' in obj:
            info_parts.append(f"Area: {obj['area']:.1f}")
    
    # Add classification info if available
    if 'class_name' in obj:
        class_info = obj['class_name']
        if 'confidence' in obj:
            class_info += f" ({obj['confidence']:.2f})"
        info_parts.append(f"Class: {class_info}")
    
    return " | ".join(info_parts) if info_parts else "No info available"