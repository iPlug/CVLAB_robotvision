#!/usr/bin/env python3
"""
Coordinate System Mapping Test

This script tests coordinate transformations to understand the mapping between
camera coordinates and robot coordinates. Helps debug why LiDAR coordinates
don't match expected robot positions.
"""

import numpy as np
from pathlib import Path

def load_transformation_matrix(file_path):
    """Load transformation matrix from file."""
    try:
        matrix = np.load(file_path)
        print(f"Loaded transformation matrix from: {file_path}")
        print("Matrix:")
        print(matrix)
        print(f"Translation (meters): {matrix[:3, 3]}")
        print(f"Translation (mm): {matrix[:3, 3] * 1000}")
        return matrix
    except Exception as e:
        print(f"Failed to load matrix: {e}")
        return None

def transform_camera_to_robot(camera_position, transform_matrix):
    """Transform camera position to robot coordinates."""
    # Convert to homogeneous coordinates
    camera_homo = np.append(camera_position, 1.0)
    
    # Transform using matrix
    robot_homo = transform_matrix @ camera_homo
    robot_position = robot_homo[:3]
    
    # Convert to millimeters
    robot_position_mm = robot_position * 1000.0
    
    return robot_position, robot_position_mm

def main():
    print("=== COORDINATE SYSTEM MAPPING TEST ===")
    print()
    
    # Load transformation matrix
    matrix_path = "charuco-pose-estimation/calibration_result_1754622347/eye_in_hand_transform.npy"
    transform_matrix = load_transformation_matrix(matrix_path)
    
    if transform_matrix is None:
        return
    
    print()
    print("=== TEST COORDINATE TRANSFORMATIONS ===")
    
    # Test the actual LiDAR detected position
    print("\n1. ACTUAL LIDAR DETECTED POSITION:")
    lidar_camera_pos = np.array([-0.2, 0.015, 0.67])  # From debug output
    robot_pos_m, robot_pos_mm = transform_camera_to_robot(lidar_camera_pos, transform_matrix)
    print(f"Camera position (m): {lidar_camera_pos}")
    print(f"Robot position (m): {robot_pos_m}")
    print(f"Robot position (mm): {robot_pos_mm}")
    
    # Test what camera position would give [500, 0, 0] robot coordinates
    print("\n2. REVERSE CALCULATION - What camera position gives [500, 0, 0]mm robot?")
    target_robot_mm = np.array([500.0, 0.0, 0.0])  # User's expected position
    target_robot_m = target_robot_mm / 1000.0  # Convert to meters
    
    # Use inverse transformation: camera = inv(transform_matrix) @ robot
    try:
        inv_transform = np.linalg.inv(transform_matrix)
        target_robot_homo = np.append(target_robot_m, 1.0)
        required_camera_homo = inv_transform @ target_robot_homo
        required_camera_pos = required_camera_homo[:3]
        
        print(f"Required camera position (m): {required_camera_pos}")
        print(f"Required camera position (mm): {required_camera_pos * 1000}")
        
        # Verify by transforming back
        verify_robot_m, verify_robot_mm = transform_camera_to_robot(required_camera_pos, transform_matrix)
        print(f"Verification - transforms back to (mm): {verify_robot_mm}")
        
    except Exception as e:
        print(f"Inverse calculation failed: {e}")
    
    # Test some coordinate system hypotheses
    print("\n3. COORDINATE SYSTEM MAPPING TESTS:")
    
    # Test different coordinate mappings
    test_camera_positions = [
        ("Straight ahead", np.array([0.0, 0.0, 0.5])),
        ("Right side", np.array([0.1, 0.0, 0.5])),
        ("Left side", np.array([-0.1, 0.0, 0.5])),
        ("Up", np.array([0.0, 0.1, 0.5])),
        ("Down", np.array([0.0, -0.1, 0.5])),
    ]
    
    for name, cam_pos in test_camera_positions:
        robot_m, robot_mm = transform_camera_to_robot(cam_pos, transform_matrix)
        print(f"{name:12s}: Camera {cam_pos} → Robot {robot_mm.astype(int)}mm")
    
    print("\n4. COORDINATE SYSTEM ANALYSIS:")
    print("Camera coordinate system: X=right, Y=down, Z=forward")
    print("Robot coordinate system: X=forward, Y=left, Z=up")
    print()
    print("From the transformation matrix rotation part:")
    rotation = transform_matrix[:3, :3]
    print("Rotation matrix:")
    print(rotation)
    print()
    print("Camera X-axis (right) maps to robot direction:", rotation[:, 0])
    print("Camera Y-axis (down) maps to robot direction:", rotation[:, 1])
    print("Camera Z-axis (forward) maps to robot direction:", rotation[:, 2])
    
    print("\n5. PHYSICAL OBJECT POSITION ANALYSIS:")
    print("According to CLAUDE.md, object should be at [500, 0, 0]mm robot coordinates:")
    print("- 500mm FORWARD from robot base (positive X)")
    print("- 0mm left/right (Y = 0)")
    print("- 0mm above table level (Z = 0)")
    print()
    print("But LiDAR detects object at [-181, +83, +708]mm robot coordinates:")
    print("- 181mm BACKWARD from robot base (negative X)")
    print("- 83mm to the LEFT (positive Y)")  
    print("- 708mm ABOVE table level (positive Z)")
    print()
    print("CONCLUSION: The object detected by LiDAR is NOT at [500, 0, 0]mm.")
    print("Either:")
    print("1. The object is physically located elsewhere")
    print("2. There's a sign convention error in the coordinate system")
    print("3. The calibration matrix has a systematic error")
    
    # Test specific camera position that should give [500, 0, 0]
    print("\n6. CAMERA POSITION NEEDED FOR [500, 0, 0]mm ROBOT POSITION:")
    required_cam_pos = np.array([0.37722647, 0.38294157, -0.03972369])
    print(f"Required camera position: {required_cam_pos * 1000} mm")
    print("This means the object should appear:")
    print(f"- {required_cam_pos[0]*1000:.0f}mm to the RIGHT of camera center")
    print(f"- {required_cam_pos[1]*1000:.0f}mm BELOW camera center") 
    print(f"- {abs(required_cam_pos[2])*1000:.0f}mm BEHIND camera center")
    print()
    print("Current LiDAR detection shows object:")
    lidar_cam_mm = lidar_camera_pos * 1000
    print(f"- {abs(lidar_cam_mm[0]):.0f}mm to the LEFT of camera center")
    print(f"- {lidar_cam_mm[1]:.0f}mm ABOVE camera center")
    print(f"- {lidar_cam_mm[2]:.0f}mm in FRONT of camera center")

if __name__ == "__main__":
    main()