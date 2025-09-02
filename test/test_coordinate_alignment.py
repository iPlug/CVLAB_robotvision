#!/usr/bin/env python3
"""
Test Coordinate System Alignment Based on User Observation

The user observed that CharUco detection shows:
- GREEN axis → Robot X-axis (forward)
- RED axis → Robot Y-axis (left) 
- BLUE axis → Robot Z-axis (up)

This means the expected rotation should be identity matrix.
The 95° error indicates a real coordinate system problem.
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

def analyze_predicted_vs_expected_rotation(data):
    """
    Test the rotation validation with identity matrix as expected rotation.
    This should reveal the source of the 95° systematic error.
    """
    print("TESTING ROTATION WITH IDENTITY MATRIX EXPECTATION")
    print("="*70)
    
    eye_in_hand_transform = np.array(data['eye_in_hand_transform'])
    
    # Expected rotation: Identity (based on user's observation of axis alignment)
    expected_rotation = np.eye(3)
    
    print("Expected rotation matrix (identity - axes aligned):")
    print(expected_rotation)
    print()
    
    rotation_errors = []
    rotation_matrices = []
    
    for i, point in enumerate(data['points']):
        robot_transform = np.array(point['robot_transform'])
        charuco_transform = np.array(point['charuco_transform'])
        
        # Predict pattern in robot base coordinates
        pattern_in_base = robot_transform @ eye_in_hand_transform @ charuco_transform
        predicted_rotation = pattern_in_base[:3, :3]
        rotation_matrices.append(predicted_rotation)
        
        # Calculate rotation error from identity
        rotation_error_matrix = predicted_rotation @ expected_rotation.T
        rotation_error_trace = np.trace(rotation_error_matrix)
        rotation_error_angle = np.arccos(np.clip((rotation_error_trace - 1) / 2, -1, 1))
        rotation_error_degrees = np.degrees(rotation_error_angle)
        
        rotation_errors.append(rotation_error_degrees)
        
        print(f"Point {i+1}:")
        print(f"  Predicted rotation matrix:")
        for row in predicted_rotation:
            print(f"    [{row[0]:7.4f}, {row[1]:7.4f}, {row[2]:7.4f}]")
        print(f"  Rotation error from identity: {rotation_error_degrees:.2f}°")
        print()
    
    mean_rotation_error = np.mean(rotation_errors)
    print(f"Mean rotation error from identity: {mean_rotation_error:.2f}°")
    print(f"This confirms the ~95° systematic error exists")
    
    return rotation_matrices, rotation_errors, mean_rotation_error

def analyze_rotation_pattern(rotation_matrices):
    """Analyze the pattern in the rotation matrices to find the systematic error."""
    print("\nANALYZING ROTATION PATTERN")
    print("="*50)
    
    # Convert rotation matrices to Euler angles
    euler_angles = []
    axis_directions = {'x': [], 'y': [], 'z': []}
    
    for i, rot_matrix in enumerate(rotation_matrices):
        # Convert to Euler angles
        rotation = Rotation.from_matrix(rot_matrix)
        euler = rotation.as_euler('xyz', degrees=True)
        euler_angles.append(euler)
        
        # Extract axis directions
        x_axis = rot_matrix[:, 0]  # First column = X-axis direction
        y_axis = rot_matrix[:, 1]  # Second column = Y-axis direction  
        z_axis = rot_matrix[:, 2]  # Third column = Z-axis direction
        
        axis_directions['x'].append(x_axis)
        axis_directions['y'].append(y_axis)
        axis_directions['z'].append(z_axis)
        
        print(f"Point {i+1} - Euler angles: [{euler[0]:6.1f}, {euler[1]:6.1f}, {euler[2]:6.1f}]°")
    
    # Calculate average Euler angles
    euler_angles = np.array(euler_angles)
    mean_euler = np.mean(euler_angles, axis=0)
    std_euler = np.std(euler_angles, axis=0)
    
    print(f"\nEuler angle statistics:")
    print(f"  Mean: [{mean_euler[0]:6.1f}, {mean_euler[1]:6.1f}, {mean_euler[2]:6.1f}]°")
    print(f"  Std:  [{std_euler[0]:6.1f}, {std_euler[1]:6.1f}, {std_euler[2]:6.1f}]°")
    
    # Calculate average axis directions
    mean_axes = {}
    for axis in ['x', 'y', 'z']:
        mean_axes[axis] = np.mean(axis_directions[axis], axis=0)
        
    print(f"\nAverage axis directions in robot coordinates:")
    print(f"  X-axis (should be [1,0,0]): [{mean_axes['x'][0]:6.3f}, {mean_axes['x'][1]:6.3f}, {mean_axes['x'][2]:6.3f}]")
    print(f"  Y-axis (should be [0,1,0]): [{mean_axes['y'][0]:6.3f}, {mean_axes['y'][1]:6.3f}, {mean_axes['y'][2]:6.3f}]") 
    print(f"  Z-axis (should be [0,0,1]): [{mean_axes['z'][0]:6.3f}, {mean_axes['z'][1]:6.3f}, {mean_axes['z'][2]:6.3f}]")
    
    return mean_euler, mean_axes

def test_coordinate_corrections(data, mean_euler):
    """Test different coordinate frame corrections based on the systematic error."""
    print("\nTESTING COORDINATE CORRECTIONS")
    print("="*50)
    
    # Based on the mean Euler angles, test corrective transformations
    test_corrections = [
        ("No correction", np.eye(3)),
        ("90° around X", Rotation.from_euler('x', 90, degrees=True).as_matrix()),
        ("90° around Y", Rotation.from_euler('y', 90, degrees=True).as_matrix()),
        ("90° around Z", Rotation.from_euler('z', 90, degrees=True).as_matrix()),
        ("-90° around X", Rotation.from_euler('x', -90, degrees=True).as_matrix()),
        ("-90° around Y", Rotation.from_euler('y', -90, degrees=True).as_matrix()),
        ("-90° around Z", Rotation.from_euler('z', -90, degrees=True).as_matrix()),
        ("180° around X", Rotation.from_euler('x', 180, degrees=True).as_matrix()),
        ("180° around Y", Rotation.from_euler('y', 180, degrees=True).as_matrix()),
        ("180° around Z", Rotation.from_euler('z', 180, degrees=True).as_matrix()),
    ]
    
    eye_in_hand_transform = np.array(data['eye_in_hand_transform'])
    
    print("Testing corrections to minimize rotation error from identity:")
    print("-" * 70)
    
    best_correction = None
    best_error = float('inf')
    
    for correction_name, correction_matrix in test_corrections:
        rotation_errors = []
        
        for point in data['points']:
            robot_transform = np.array(point['robot_transform'])
            charuco_transform = np.array(point['charuco_transform'])
            
            # Apply correction to CharUco transform
            corrected_charuco = charuco_transform.copy()
            corrected_charuco[:3, :3] = charuco_transform[:3, :3] @ correction_matrix
            
            # Predict pattern in robot base coordinates
            pattern_in_base = robot_transform @ eye_in_hand_transform @ corrected_charuco
            predicted_rotation = pattern_in_base[:3, :3]
            
            # Calculate rotation error from identity
            rotation_error_matrix = predicted_rotation @ np.eye(3).T
            rotation_error_trace = np.trace(rotation_error_matrix)
            rotation_error_angle = np.arccos(np.clip((rotation_error_trace - 1) / 2, -1, 1))
            rotation_error_degrees = np.degrees(rotation_error_angle)
            
            rotation_errors.append(rotation_error_degrees)
        
        mean_error = np.mean(rotation_errors)
        print(f"{correction_name:<15}: {mean_error:6.2f}° mean error")
        
        if mean_error < best_error:
            best_error = mean_error
            best_correction = (correction_name, correction_matrix)
    
    print(f"\nBest correction: {best_correction[0]} with {best_error:.2f}° error")
    
    if best_error < 10:  # Significant improvement
        print("🎉 FOUND COORDINATE SYSTEM FIX!")
        print(f"Apply this correction matrix to CharUco detections:")
        print(best_correction[1])
    else:
        print("⚠ No simple rotation fixes the coordinate system")
        print("The problem may be in a different part of the transformation chain")
    
    return best_correction, best_error

def main():
    """Main analysis function."""
    print("COORDINATE ALIGNMENT ANALYSIS")
    print("="*80)
    print("Based on user observation that CharUco axes align with robot axes")
    print("Expected: CharUco GREEN→Robot X, RED→Robot Y, BLUE→Robot Z")
    print("="*80)
    
    data = load_calibration_data()
    
    # Test with identity matrix as expected rotation
    rotation_matrices, rotation_errors, mean_error = analyze_predicted_vs_expected_rotation(data)
    
    # Analyze the rotation pattern
    mean_euler, mean_axes = analyze_rotation_pattern(rotation_matrices)
    
    # Test coordinate corrections
    best_correction, best_error = test_coordinate_corrections(data, mean_euler)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Current rotation error from identity: {mean_error:.2f}°")
    print(f"Best correction found: {best_correction[0]} → {best_error:.2f}°")
    
    if best_error < 5:
        print("✅ Coordinate system can be fixed with simple correction")
    elif best_error < 20:
        print("⚠ Partial improvement possible, may need complex correction") 
    else:
        print("❌ No simple correction works, deeper investigation needed")

if __name__ == "__main__":
    main()