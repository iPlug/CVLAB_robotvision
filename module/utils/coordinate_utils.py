"""
Coordinate system utilities for myCobot robot operations.

Provides functions for coordinate transformations and orientation calculations
that are compatible with myCobot's specific coordinate system and angle conventions.
"""

import math
from typing import List, Tuple, Optional
import numpy as np


def calculate_mycobot_pointing_orientation(robot_pos: List[float], target_pos: List[float]) -> List[float]:
    """
    Calculate orientation angles for myCobot to point from robot position toward target position.
    
    Uses myCobot-specific coordinate system [forward, side, height] and orientation conventions
    where [0, 180, 0] represents pointing straight down.
    
    Args:
        robot_pos: Robot position [forward, side, height] in mm
        target_pos: Target position [forward, side, height] in mm
        
    Returns:
        Orientation angles [rx, ry, rz] in myCobot conventions (degrees)
    """
    # Calculate direction vector from robot to target
    # myCobot coordinates: [forward, side, height]
    d_forward = target_pos[0] - robot_pos[0]  # Forward direction
    d_side = target_pos[1] - robot_pos[1]     # Side direction  
    d_height = target_pos[2] - robot_pos[2]   # Height direction
    
    # Calculate horizontal distance (forward + side)
    horizontal_dist = math.sqrt(d_forward**2 + d_side**2)
    
    if horizontal_dist > 0.1:  # Avoid division by zero
        # Calculate the angle from horizontal to target
        # Positive d_height = target above robot, negative = target below robot
        elevation_angle = math.degrees(math.atan2(d_height, horizontal_dist))
        
        # Calculate horizontal direction (yaw) - angle from forward axis to target
        # Positive d_side = target to the left, negative = target to the right
        yaw_angle = math.degrees(math.atan2(d_side, d_forward))
        
        # myCobot orientation mapping:
        # Base pointing down orientation is [0, 180, 0]
        # Adjust ry (pitch) based on elevation to target
        # 180° = pointing down, smaller values = pointing more forward/up
        
        # Convert elevation angle to myCobot ry convention
        # For horizontal targets (dz ≈ 0), we want to point forward (ry ≈ 90-100)
        # For targets below (negative dz), we want to point down-forward
        # For targets above (positive dz), we want to point up-forward
        
        if abs(elevation_angle) < 5:  # Nearly horizontal target
            pointing_ry = 90  # Point forward horizontally
        elif elevation_angle > 0:  # Target above robot
            pointing_ry = max(0, 90 - elevation_angle)  # Point upward
        else:  # Target below robot  
            pointing_ry = min(180, 90 - elevation_angle)  # Point downward-forward
        
        # Ensure reasonable range
        pointing_ry = max(-180, min(180, pointing_ry))
        
        # For rx and rz, start from base and adjust for horizontal direction
        # rx typically stays near 0 for simple pointing
        pointing_rx = 0
        
        # rz adjusts for horizontal direction
        pointing_rz = yaw_angle
        
        return [pointing_rx, pointing_ry, pointing_rz]
    else:
        # Target is directly above/below robot (no horizontal offset)
        if d_height > 0:
            # Point straight up - adjust ry to be much less than 180
            return [0, 0, 0]  # Point straight up
        else:
            # Point straight down - keep base pointing down orientation
            return [0, 180, 0]  # Point straight down


def calculate_eye_in_hand_orientation(camera_pos: List[float], target_pos: List[float], 
                                    camera_offset: Optional[List[float]] = None) -> List[float]:
    """
    Calculate orientation for eye-in-hand setup where camera is mounted on robot end-effector.
    
    For eye-in-hand configurations, the camera is typically mounted such that it points
    in the -Z direction of the end-effector frame. This function calculates the required
    end-effector orientation to point the camera at the target.
    
    Args:
        camera_pos: Camera position [forward, side, height] in mm
        target_pos: Target position [forward, side, height] in mm  
        camera_offset: Optional camera offset from end-effector [dx, dy, dz] in mm
        
    Returns:
        End-effector orientation [rx, ry, rz] for camera to look at target (degrees)
    """
    # If camera has an offset from end-effector, adjust the calculation
    if camera_offset:
        # Calculate where end-effector should be for camera to be at camera_pos
        effector_pos = [
            camera_pos[0] - camera_offset[0],
            camera_pos[1] - camera_offset[1], 
            camera_pos[2] - camera_offset[2]
        ]
    else:
        effector_pos = camera_pos
    
    # Calculate direction vector from camera to target
    d_forward = target_pos[0] - camera_pos[0]
    d_side = target_pos[1] - camera_pos[1]
    d_height = target_pos[2] - camera_pos[2]
    
    horizontal_dist = math.sqrt(d_forward**2 + d_side**2)
    
    if horizontal_dist > 0.1:
        # Calculate angles for camera to point at target
        elevation_angle = math.degrees(math.atan2(d_height, horizontal_dist))
        yaw_angle = math.degrees(math.atan2(d_side, d_forward))
        
        # For eye-in-hand, camera typically points in -Z direction of end-effector
        # So we need to calculate end-effector orientation that achieves desired camera pointing
        
        # Base camera pointing down orientation in end-effector frame
        # Camera -Z points toward target, so end-effector needs specific orientation
        
        # Convert camera pointing direction to end-effector orientation
        # This is simplified - real eye-in-hand may need calibrated camera-to-effector transform
        
        # For myCobot with camera pointing in -Z direction of end-effector:
        # When camera needs to point down, end-effector ry ≈ 180
        # When camera needs to point forward, end-effector ry ≈ 90
        # When camera needs to point up, end-effector ry ≈ 0
        
        if abs(elevation_angle) < 5:  # Nearly horizontal target
            effector_ry = 90  # End-effector orientation for camera to point horizontally
        elif elevation_angle > 0:  # Target above camera
            effector_ry = max(0, 90 - elevation_angle)  # Point camera upward
        else:  # Target below camera
            effector_ry = min(180, 90 - elevation_angle)  # Point camera downward
        
        # Clamp to myCobot ranges
        effector_ry = max(-180, min(180, effector_ry))
        
        # End-effector rx and rz for camera pointing
        effector_rx = 0  # Typically no roll for simple pointing
        effector_rz = yaw_angle  # Rotate end-effector to point camera toward target horizontally
        
        return [effector_rx, effector_ry, effector_rz]
    else:
        # Target directly above/below camera
        if d_height > 0:
            return [0, 0, 0]  # Point camera straight up
        else:
            return [0, 180, 0]  # Point camera straight down


def validate_mycobot_position(position: List[float], workspace_bounds: dict = None) -> Tuple[bool, str]:
    """
    Validate if position is within myCobot workspace and safety constraints.
    
    Args:
        position: Position to validate [forward, side, height, rx, ry, rz]
        workspace_bounds: Optional workspace bounds dict with 'forward', 'side', 'height' keys
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if len(position) < 3:
        return False, "Position must have at least [forward, side, height]"
    
    forward, side, height = position[:3]
    
    # Default myCobot 280 workspace bounds if not specified
    if workspace_bounds is None:
        workspace_bounds = {
            'forward': (50, 350),   # 50-350mm forward range
            'side': (-280, 280),    # ±280mm side range  
            'height': (50, 420)     # 50-420mm height range
        }
    
    # Check workspace bounds
    if 'forward' in workspace_bounds:
        f_min, f_max = workspace_bounds['forward']
        if not (f_min <= forward <= f_max):
            return False, f"Forward position {forward:.1f} outside bounds [{f_min:.1f}, {f_max:.1f}]"
    
    if 'side' in workspace_bounds:
        s_min, s_max = workspace_bounds['side']
        if not (s_min <= side <= s_max):
            return False, f"Side position {side:.1f} outside bounds [{s_min:.1f}, {s_max:.1f}]"
    
    if 'height' in workspace_bounds:
        h_min, h_max = workspace_bounds['height']
        if not (h_min <= height <= h_max):
            return False, f"Height position {height:.1f} outside bounds [{h_min:.1f}, {h_max:.1f}]"
    
    # Check myCobot 280 reach constraint from J1 (base joint at 131.56mm height)
    J1_HEIGHT = 131.56  # Height of J1 above physical base
    
    # Calculate distance from J1 (not physical base)
    distance_from_j1 = math.sqrt(forward**2 + side**2 + (height - J1_HEIGHT)**2)
    MAX_REACH = 280  # 280mm effective radius from J1
    
    if distance_from_j1 > MAX_REACH:
        return False, f"Position distance from J1: {distance_from_j1:.1f} exceeds max reach {MAX_REACH:.1f}"
    
    # Check minimum distance from J1 (avoid singularities)
    if distance_from_j1 < 50:
        return False, f"Position too close to J1: {distance_from_j1:.1f} < 50mm (singularity risk)"
    
    return True, "Position is valid"


def convert_standard_to_mycobot_coords(standard_coords: List[float]) -> List[float]:
    """
    Convert standard [x, y, z] coordinates to myCobot [forward, side, height] format.
    
    Args:
        standard_coords: Standard cartesian coordinates [x, y, z, ...]
        
    Returns:
        MyCobot coordinates [forward, side, height, ...]
    """
    if len(standard_coords) < 3:
        return standard_coords
    
    # Assuming standard x=forward, y=side, z=height mapping
    # Adjust this mapping based on your coordinate system setup
    mycobot_coords = [
        standard_coords[0],  # x -> forward
        standard_coords[1],  # y -> side  
        standard_coords[2]   # z -> height
    ]
    
    # Preserve any additional elements (orientations)
    if len(standard_coords) > 3:
        mycobot_coords.extend(standard_coords[3:])
    
    return mycobot_coords


def convert_mycobot_to_standard_coords(mycobot_coords: List[float]) -> List[float]:
    """
    Convert myCobot [forward, side, height] coordinates to standard [x, y, z] format.
    
    Args:
        mycobot_coords: MyCobot coordinates [forward, side, height, ...]
        
    Returns:
        Standard cartesian coordinates [x, y, z, ...]
    """
    if len(mycobot_coords) < 3:
        return mycobot_coords
    
    # Assuming myCobot forward=x, side=y, height=z mapping  
    # Adjust this mapping based on your coordinate system setup
    standard_coords = [
        mycobot_coords[0],  # forward -> x
        mycobot_coords[1],  # side -> y
        mycobot_coords[2]   # height -> z
    ]
    
    # Preserve any additional elements (orientations)
    if len(mycobot_coords) > 3:
        standard_coords.extend(mycobot_coords[3:])
    
    return standard_coords


def calculate_reachable_position(target_coords: List[float], max_reach: float = 280.0) -> Tuple[List[float], bool]:
    """
    Calculate reachable position for target coordinates using myCobot J1-based calculation.
    
    If the coordinate is reachable, returns it unchanged.
    If it's unreachable, scales it to the maximum reach boundary in the direction from J1 to the target.
    
    Args:
        target_coords: Target coordinates [forward, side, height, rx, ry, rz]
        max_reach: Maximum reach distance from J1 in mm
        
    Returns:
        Tuple of (adjusted_coords, was_adjusted)
    """
    if len(target_coords) < 3:
        return target_coords, False
    
    J1_HEIGHT = 131.56  # Height of J1 above physical base
    
    # Extract the position and orientation
    forward, side, height = target_coords[:3]
    target_orientation = target_coords[3:] if len(target_coords) > 3 else []
    
    # Calculate the distance from J1 (not physical base)
    distance_from_j1 = math.sqrt(forward**2 + side**2 + (height - J1_HEIGHT)**2)
    
    if distance_from_j1 > max_reach:
        # Create vector from J1 to target
        target_from_j1 = np.array([forward, side, height - J1_HEIGHT])
        
        # Calculate the unit vector (direction from J1)
        direction_vector = target_from_j1 / distance_from_j1
        
        # Scale the vector to the maximum reach (with 90% safety margin)
        scaled_from_j1 = direction_vector * (max_reach * 0.9)
        
        # Convert back to absolute coordinates (add J1 height back)
        scaled_position = [
            scaled_from_j1[0], 
            scaled_from_j1[1], 
            scaled_from_j1[2] + J1_HEIGHT
        ]
        
        # Combine the new position with the original orientation
        new_coords = scaled_position + target_orientation
        
        return new_coords, True
    else:
        return target_coords, False