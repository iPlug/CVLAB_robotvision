#!/usr/bin/env python3
"""
Deep Analysis of the 90-Degree Rotation Error

Since all conventional fixes failed, this script analyzes the fundamental
assumptions in the validation and looks for deeper issues.
"""

import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation

def load_calibration_data():
    """Load the existing calibration data."""
    calibration_path = "charuco-pose-estimation/calibration_result_1754622347/calibration_data.json"
    with open(calibration_path, 'r') as f:
        data = json.load(f)
    return data

def analyze_charuco_poses(data):
    """Analyze the actual CharUco poses detected during calibration."""
    print("ANALYZING CHARUCO POSES")
    print("="*60)
    
    charuco_positions = []
    charuco_rotations = []
    
    for i, point in enumerate(data['points']):
        charuco_transform = np.array(point['charuco_transform'])
        position = charuco_transform[:3, 3]
        rotation_matrix = charuco_transform[:3, :3]
        
        # Convert rotation matrix to Euler angles
        rotation = Rotation.from_matrix(rotation_matrix)
        euler = rotation.as_euler('xyz', degrees=True)
        
        print(f"Point {i+1}:")
        print(f"  Position: [{position[0]:7.3f}, {position[1]:7.3f}, {position[2]:7.3f}] m")
        print(f"  Rotation: [{euler[0]:7.1f}, {euler[1]:7.1f}, {euler[2]:7.1f}] deg")
        
        charuco_positions.append(position)
        charuco_rotations.append(euler)
    
    charuco_positions = np.array(charuco_positions)
    charuco_rotations = np.array(charuco_rotations)
    
    print(f"\nPosition statistics (meters):")
    print(f"  X: {charuco_positions[:, 0].min():.3f} to {charuco_positions[:, 0].max():.3f}")
    print(f"  Y: {charuco_positions[:, 1].min():.3f} to {charuco_positions[:, 1].max():.3f}")  
    print(f"  Z: {charuco_positions[:, 2].min():.3f} to {charuco_positions[:, 2].max():.3f}")
    
    print(f"\nRotation statistics (degrees):")
    print(f"  X: {charuco_rotations[:, 0].min():.1f} to {charuco_rotations[:, 0].max():.1f}")
    print(f"  Y: {charuco_rotations[:, 1].min():.1f} to {charuco_rotations[:, 1].max():.1f}")
    print(f"  Z: {charuco_rotations[:, 2].min():.1f} to {charuco_rotations[:, 2].max():.1f}")
    
    return charuco_positions, charuco_rotations

def analyze_robot_poses(data):
    """Analyze the robot poses during calibration."""
    print("\nANALYZING ROBOT POSES")
    print("="*60)
    
    robot_positions = []
    robot_rotations = []
    
    for i, point in enumerate(data['points']):
        robot_pose = point['robot_pose']
        position = robot_pose[:3]  # mm
        rotation = robot_pose[3:]  # degrees
        
        print(f"Point {i+1}:")
        print(f"  Position: [{position[0]:7.1f}, {position[1]:7.1f}, {position[2]:7.1f}] mm")
        print(f"  Rotation: [{rotation[0]:7.1f}, {rotation[1]:7.1f}, {rotation[2]:7.1f}] deg")
        
        robot_positions.append(position)
        robot_rotations.append(rotation)
    
    robot_positions = np.array(robot_positions)
    robot_rotations = np.array(robot_rotations)
    
    print(f"\nPosition statistics (mm):")
    print(f"  X: {robot_positions[:, 0].min():.1f} to {robot_positions[:, 0].max():.1f}")
    print(f"  Y: {robot_positions[:, 1].min():.1f} to {robot_positions[:, 1].max():.1f}")  
    print(f"  Z: {robot_positions[:, 2].min():.1f} to {robot_positions[:, 2].max():.1f}")
    
    return robot_positions, robot_rotations

def analyze_transformation_consistency(data):
    """Analyze if the transformations are internally consistent."""
    print("\nANALYZING TRANSFORMATION CONSISTENCY") 
    print("="*60)
    
    eye_in_hand_transform = np.array(data['eye_in_hand_transform'])
    
    print("Eye-in-hand transformation matrix:")
    print(eye_in_hand_transform)
    
    # Extract rotation and analyze
    rotation_matrix = eye_in_hand_transform[:3, :3]
    translation = eye_in_hand_transform[:3, 3]
    
    rotation = Rotation.from_matrix(rotation_matrix)
    euler = rotation.as_euler('xyz', degrees=True)
    
    print(f"\nEye-in-hand transformation:")
    print(f"  Translation: [{translation[0]:7.3f}, {translation[1]:7.3f}, {translation[2]:7.3f}] m")
    print(f"  Rotation: [{euler[0]:7.1f}, {euler[1]:7.1f}, {euler[2]:7.1f}] deg")
    
    return eye_in_hand_transform

def test_validation_assumptions(data):
    """Test if the validation assumptions are correct."""
    print("\nTESTING VALIDATION ASSUMPTIONS")
    print("="*60)
    
    # Test different pattern positions
    pattern_positions = [
        ([0.3, 0.1, 0.0], "Original assumption [300, 100, 0]mm"),
        ([0.0, 0.0, 0.0], "Origin [0, 0, 0]mm"), 
        ([0.1, 0.3, 0.0], "Swapped XY [100, 300, 0]mm"),
        ([0.3, -0.1, 0.0], "Negative Y [300, -100, 0]mm"),
        ([-0.3, 0.1, 0.0], "Negative X [-300, 100, 0]mm")
    ]
    
    eye_in_hand_transform = np.array(data['eye_in_hand_transform'])
    
    print("Testing different pattern position assumptions:")
    print("-" * 60)
    
    for pattern_pos, description in pattern_positions:
        pattern_position = np.array(pattern_pos)
        
        translation_errors = []
        predicted_positions = []
        
        for point in data['points']:
            robot_transform = np.array(point['robot_transform'])
            charuco_transform = np.array(point['charuco_transform'])
            
            # Predict pattern position
            pattern_in_base = robot_transform @ eye_in_hand_transform @ charuco_transform
            predicted_position = pattern_in_base[:3, 3]
            predicted_positions.append(predicted_position)
            
            # Calculate error
            error = np.linalg.norm(predicted_position - pattern_position)
            translation_errors.append(error * 1000)  # Convert to mm
        
        mean_error = np.mean(translation_errors)
        std_error = np.std(translation_errors)
        
        print(f"{description:<35} Mean: {mean_error:6.2f}mm, Std: {std_error:6.2f}mm")
        
        if mean_error < 30:  # Good result
            print(f"  -> This looks promising! Predicted positions:")
            for i, pred_pos in enumerate(predicted_positions[:3]):  # Show first 3
                print(f"     Point {i+1}: [{pred_pos[0]:7.3f}, {pred_pos[1]:7.3f}, {pred_pos[2]:7.3f}] m")

def analyze_coordinate_system_mismatch(data):
    """Look for fundamental coordinate system mismatches."""
    print("\nANALYZING COORDINATE SYSTEM MISMATCHES")
    print("="*60)
    
    # Check if CharUco Z-axis is consistently pointing in same direction
    z_directions = []
    
    for point in data['points']:
        charuco_transform = np.array(point['charuco_transform'])
        z_axis = charuco_transform[:3, 2]  # Z-axis direction
        z_directions.append(z_axis)
        
    z_directions = np.array(z_directions)
    
    print("CharUco Z-axis directions (should be consistent for horizontal pattern):")
    for i, z_axis in enumerate(z_directions):
        print(f"Point {i+1}: [{z_axis[0]:6.3f}, {z_axis[1]:6.3f}, {z_axis[2]:6.3f}]")
    
    # Calculate consistency
    mean_z = np.mean(z_directions, axis=0)
    print(f"\nMean Z-axis: [{mean_z[0]:6.3f}, {mean_z[1]:6.3f}, {mean_z[2]:6.3f}]")
    
    # Check if Z is pointing consistently up/down (should be ~[0, 0, ±1] for horizontal pattern)
    if abs(mean_z[2]) > 0.8:
        if mean_z[2] > 0:
            print("✓ Z-axis pointing UP (away from table) - this is correct")
        else:
            print("✓ Z-axis pointing DOWN (into table) - this might be correct depending on OpenCV version")
    else:
        print("⚠ Z-axis not pointing up/down - pattern may not be horizontal or coordinate system issue")
        
    # Check Z-axis stability
    z_variations = np.std(z_directions, axis=0)
    print(f"Z-axis variation: [{z_variations[0]:6.3f}, {z_variations[1]:6.3f}, {z_variations[2]:6.3f}]")
    
    if np.max(z_variations) > 0.2:
        print("⚠ High Z-axis variation - pattern orientation inconsistent or detection noise")
    else:
        print("✓ Z-axis relatively stable")

def main():
    """Main analysis function."""
    print("DEEP ANALYSIS OF 90-DEGREE ROTATION ERROR")
    print("="*80)
    
    data = load_calibration_data()
    
    # Analyze the raw data
    charuco_positions, charuco_rotations = analyze_charuco_poses(data)
    robot_positions, robot_rotations = analyze_robot_poses(data)
    
    # Analyze transformation
    eye_in_hand = analyze_transformation_consistency(data)
    
    # Test validation assumptions
    test_validation_assumptions(data)
    
    # Look for coordinate system issues
    analyze_coordinate_system_mismatch(data)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("Check the output above for potential sources of the 90° rotation error.")

if __name__ == "__main__":
    main()