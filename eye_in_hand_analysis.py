#!/usr/bin/env python3
"""
Eye-in-Hand Coordinate Analysis

Comprehensive analysis of the eye-in-hand setup coordinate transformations
considering the robot end effector position.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.robot.mycobot_controller import MyCobotController

def load_transformation_matrix(file_path):
    """Load transformation matrix from file."""
    try:
        matrix = np.load(file_path)
        return matrix
    except Exception as e:
        print(f"Failed to load matrix: {e}")
        return None

def main():
    print("=== EYE-IN-HAND COORDINATE ANALYSIS ===")
    print()
    
    # 1. Load transformation matrix
    matrix_path = "charuco-pose-estimation/calibration_result_1754622347/eye_in_hand_transform.npy"
    T_robot_to_camera = load_transformation_matrix(matrix_path)
    
    if T_robot_to_camera is None:
        return
    
    print("Eye-in-hand transformation matrix loaded:")
    print("T_robot_to_camera (robot base → camera):")
    print(T_robot_to_camera)
    print(f"Translation: {T_robot_to_camera[:3, 3] * 1000}mm")
    print()
    
    # 2. Get current robot end effector position
    try:
        print("Getting current robot end effector position...")
        robot = MyCobotController()
        
        if robot.connect():
            current_coords = robot.get_current_position()
            robot.disconnect()
            
            if current_coords:
                robot_pos = np.array(current_coords[:3]) / 1000.0  # Convert to meters
                robot_pos_mm = np.array(current_coords[:3])  # Keep in mm
                robot_orient = current_coords[3:]  # Orientation in degrees
                
                print(f"Robot end effector position: {robot_pos_mm}mm")
                print(f"Robot end effector orientation: {robot_orient}°")
                print()
            else:
                print("Could not get robot position, using last known position")
                robot_pos_mm = np.array([81.2, -63.1, 385.5])
                robot_pos = robot_pos_mm / 1000.0
        else:
            print("Could not connect to robot, using last known position")
            robot_pos_mm = np.array([81.2, -63.1, 385.5])
            robot_pos = robot_pos_mm / 1000.0
            
    except Exception as e:
        print(f"Robot error: {e}, using last known position")
        robot_pos_mm = np.array([81.2, -63.1, 385.5])
        robot_pos = robot_pos_mm / 1000.0
    
    # 3. LiDAR detected object position in camera coordinates
    camera_object_pos = np.array([-0.2, 0.015, 0.67])  # meters
    print(f"Object position in camera coordinates: {camera_object_pos}m")
    print(f"Object position in camera coordinates: {camera_object_pos * 1000}mm")
    print()
    
    # 4. Transform object position to robot base coordinates
    camera_object_homo = np.append(camera_object_pos, 1.0)
    robot_object_homo = T_robot_to_camera @ camera_object_homo
    robot_object_pos = robot_object_homo[:3]  # meters
    robot_object_pos_mm = robot_object_pos * 1000.0  # mm
    
    print("=== COORDINATE TRANSFORMATION RESULTS ===")
    print(f"Object position in robot base coordinates: {robot_object_pos_mm}mm")
    print()
    
    # 5. Analysis of the eye-in-hand setup
    print("=== EYE-IN-HAND SETUP ANALYSIS ===")
    print("Understanding the coordinate transformation chain:")
    print()
    print("1. CAMERA MOUNT ON ROBOT:")
    print(f"   Robot end effector is at: {robot_pos_mm}mm")
    print(f"   Camera is mounted on the end effector with offset from calibration")
    print()
    print("2. OBJECT DETECTION:")
    print(f"   Camera sees object at: {camera_object_pos * 1000}mm")
    print(f"   This means object is:")
    print(f"   - {abs(camera_object_pos[0]) * 1000:.0f}mm to the {'LEFT' if camera_object_pos[0] < 0 else 'RIGHT'} of camera")
    print(f"   - {abs(camera_object_pos[1]) * 1000:.0f}mm {'ABOVE' if camera_object_pos[1] > 0 else 'BELOW'} camera")
    print(f"   - {camera_object_pos[2] * 1000:.0f}mm in FRONT of camera")
    print()
    print("3. ROBOT BASE COORDINATES:")
    print(f"   After transformation: {robot_object_pos_mm}mm")
    print(f"   Object is:")
    print(f"   - {abs(robot_object_pos_mm[0]):.0f}mm {'FORWARD' if robot_object_pos_mm[0] > 0 else 'BACKWARD'} from robot base")
    print(f"   - {abs(robot_object_pos_mm[1]):.0f}mm {'LEFT' if robot_object_pos_mm[1] > 0 else 'RIGHT'} from robot base")
    print(f"   - {robot_object_pos_mm[2]:.0f}mm {'ABOVE' if robot_object_pos_mm[2] > 0 else 'BELOW'} table level")
    print()
    
    # 6. Compare with expected position
    expected_pos = np.array([500.0, 0.0, 0.0])  # mm
    difference = robot_object_pos_mm - expected_pos
    distance_error = np.linalg.norm(difference)
    
    print("4. POSITION COMPARISON:")
    print(f"   Expected object position: {expected_pos}mm")
    print(f"   Detected object position: {robot_object_pos_mm}mm")
    print(f"   Difference: {difference}mm")
    print(f"   Distance error: {distance_error:.1f}mm")
    print()
    
    # 7. Verify if the object is actually at [500, 0, 0]
    print("5. POSITION VERIFICATION:")
    if distance_error < 100:  # Within 10cm
        print("   ✓ Object appears to be close to expected position [500, 0, 0]mm")
        print("   The small difference might be due to:")
        print("   - Measurement accuracy (~20-50mm typical)")
        print("   - Object placement precision")
        print("   - Calibration residual errors")
    elif distance_error < 200:  # Within 20cm
        print("   ⚠ Object is somewhat close to expected position but with significant error")
        print("   Possible causes:")
        print("   - Object not precisely placed at [500, 0, 0]mm")
        print("   - Robot end effector moved since calibration")
        print("   - Need to recalibrate or verify object placement")
    else:
        print("   ✗ Object is far from expected position [500, 0, 0]mm")
        print("   Likely causes:")
        print("   - Object is physically located elsewhere")
        print("   - Major calibration error")
        print("   - Wrong coordinate system interpretation")
    
    print()
    print("=== RECOMMENDATIONS ===")
    if distance_error > 100:
        print("1. Physically verify object is at [500, 0, 0]mm robot coordinates")
        print("2. Use robot to move to [500, 0, 0]mm and check if object is there")
        print("3. Consider if robot has moved significantly since calibration")
    else:
        print("1. Coordinate transformation appears to be working correctly")
        print("2. Small position error is within typical measurement accuracy")

if __name__ == "__main__":
    main()