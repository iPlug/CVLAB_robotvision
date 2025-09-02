#!/usr/bin/env python3
"""
Test the Fixed Calibration with 90° Z-Rotation Correction

This script tests the corrected coordinate system using existing calibration data
to verify that the 90° rotation fix resolves the systematic error.
"""

import numpy as np
import json
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation

def load_calibration_data():
    """Load the existing calibration data."""
    calibration_path = "charuco-pose-estimation/calibration_result_1754622347/calibration_data.json"
    with open(calibration_path, 'r') as f:
        data = json.load(f)
    return data

def charuco_pose_to_transform_matrix_corrected(rvec, tvec):
    """Convert CharUco rvec, tvec to 4x4 transformation matrix with coordinate correction."""
    # Convert rotation vector to rotation matrix
    rot_matrix, _ = cv2.Rodrigues(np.array(rvec))
    
    # Apply coordinate system correction: 90° rotation around Z-axis
    correction_matrix = np.array([
        [0, -1, 0],  # 90° rotation around Z
        [1,  0, 0], 
        [0,  0, 1]
    ])
    
    # Apply correction to rotation matrix
    corrected_rot_matrix = rot_matrix @ correction_matrix
    
    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = corrected_rot_matrix
    transform[:3, 3] = np.array(tvec).flatten()
    
    return transform

def test_corrected_validation(data):
    """Test validation with the corrected coordinate system."""
    print("TESTING CORRECTED CALIBRATION VALIDATION")
    print("="*60)
    
    eye_in_hand_transform = np.array(data['eye_in_hand_transform'])
    pattern_position_robot = np.array([0.3, 0.1, 0.0])  # Known position
    
    translation_errors = []
    rotation_errors = []
    
    print("Applying 90° Z-rotation correction to CharUco coordinates...")
    print()
    
    for i, point in enumerate(data['points']):
        robot_transform = np.array(point['robot_transform'])
        
        # Apply correction to CharUco transform
        charuco_rvec = point['charuco_rvec']
        charuco_tvec = point['charuco_tvec']
        corrected_charuco_transform = charuco_pose_to_transform_matrix_corrected(charuco_rvec, charuco_tvec)
        
        # Predict pattern position and rotation
        pattern_in_base = robot_transform @ eye_in_hand_transform @ corrected_charuco_transform
        predicted_position = pattern_in_base[:3, 3]
        predicted_rotation = pattern_in_base[:3, :3]
        
        # Calculate position error
        position_error = np.linalg.norm(predicted_position - pattern_position_robot)
        translation_errors.append(position_error * 1000)  # mm
        
        # Calculate rotation error from identity
        expected_rotation = np.eye(3)
        rotation_error_matrix = predicted_rotation @ expected_rotation.T
        rotation_error_trace = np.trace(rotation_error_matrix)
        rotation_error_angle = np.arccos(np.clip((rotation_error_trace - 1) / 2, -1, 1))
        rotation_error_degrees = np.degrees(rotation_error_angle)
        rotation_errors.append(rotation_error_degrees)
        
        print(f"Point {i+1}:")
        print(f"  Position error: {position_error*1000:.2f} mm")
        print(f"  Rotation error: {rotation_error_degrees:.2f}° (was ~95°)")
        print(f"  Predicted position: [{predicted_position[0]:.3f}, {predicted_position[1]:.3f}, {predicted_position[2]:.3f}] m")
    
    mean_translation_error = np.mean(translation_errors)
    mean_rotation_error = np.mean(rotation_errors)
    
    print("\n" + "="*60)
    print("CORRECTED VALIDATION RESULTS")
    print("="*60)
    print(f"Mean translation error: {mean_translation_error:.2f} mm (was 22.33 mm)")
    print(f"Mean rotation error: {mean_rotation_error:.2f}° (was 95.08°)")
    
    # Quality assessment
    if mean_rotation_error < 5:
        rotation_quality = "EXCELLENT"
    elif mean_rotation_error < 15:
        rotation_quality = "GOOD"
    else:
        rotation_quality = "NEEDS WORK"
    
    if mean_translation_error < 10:
        translation_quality = "EXCELLENT"
    elif mean_translation_error < 30:
        translation_quality = "GOOD"
    else:
        translation_quality = "NEEDS WORK"
    
    print(f"\nQuality Assessment:")
    print(f"  Translation: {translation_quality}")
    print(f"  Rotation: {rotation_quality}")
    
    improvement = 95.08 - mean_rotation_error
    print(f"\nRotation Error Improvement: {improvement:.2f}° reduction!")
    
    if improvement > 80:
        print("🎉 MASSIVE IMPROVEMENT! The 90° Z-rotation fix works!")
    elif improvement > 30:
        print("✅ SIGNIFICANT IMPROVEMENT! Coordinate fix is working.")
    else:
        print("⚠ LIMITED IMPROVEMENT - may need additional corrections.")
    
    return mean_translation_error, mean_rotation_error

def test_axis_alignment(data):
    """Test if the corrected axes now align with robot coordinates."""
    print("\nTESTING AXIS ALIGNMENT WITH CORRECTION")
    print("="*50)
    
    eye_in_hand_transform = np.array(data['eye_in_hand_transform'])
    axis_directions = {'x': [], 'y': [], 'z': []}
    
    for point in data['points']:
        robot_transform = np.array(point['robot_transform'])
        
        # Apply correction to CharUco transform
        charuco_rvec = point['charuco_rvec']
        charuco_tvec = point['charuco_tvec']
        corrected_charuco_transform = charuco_pose_to_transform_matrix_corrected(charuco_rvec, charuco_tvec)
        
        # Get pattern in robot base coordinates
        pattern_in_base = robot_transform @ eye_in_hand_transform @ corrected_charuco_transform
        
        # Extract axis directions
        x_axis = pattern_in_base[:3, 0]  # Should be [1,0,0]
        y_axis = pattern_in_base[:3, 1]  # Should be [0,1,0]
        z_axis = pattern_in_base[:3, 2]  # Should be [0,0,1]
        
        axis_directions['x'].append(x_axis)
        axis_directions['y'].append(y_axis)
        axis_directions['z'].append(z_axis)
    
    # Calculate average axis directions
    mean_axes = {}
    for axis in ['x', 'y', 'z']:
        mean_axes[axis] = np.mean(axis_directions[axis], axis=0)
    
    print("Average corrected axis directions in robot coordinates:")
    print(f"  X-axis: [{mean_axes['x'][0]:7.3f}, {mean_axes['x'][1]:7.3f}, {mean_axes['x'][2]:7.3f}] (should be [1,0,0])")
    print(f"  Y-axis: [{mean_axes['y'][0]:7.3f}, {mean_axes['y'][1]:7.3f}, {mean_axes['y'][2]:7.3f}] (should be [0,1,0])")
    print(f"  Z-axis: [{mean_axes['z'][0]:7.3f}, {mean_axes['z'][1]:7.3f}, {mean_axes['z'][2]:7.3f}] (should be [0,0,1])")
    
    # Check alignment quality
    x_error = np.linalg.norm(mean_axes['x'] - np.array([1, 0, 0]))
    y_error = np.linalg.norm(mean_axes['y'] - np.array([0, 1, 0]))
    z_error = np.linalg.norm(mean_axes['z'] - np.array([0, 0, 1]))
    
    print(f"\nAxis alignment errors:")
    print(f"  X-axis error: {x_error:.4f} (was ~1.4)")
    print(f"  Y-axis error: {y_error:.4f} (was ~1.4)")
    print(f"  Z-axis error: {z_error:.4f} (was ~0.1)")
    
    total_alignment_error = x_error + y_error + z_error
    print(f"  Total alignment error: {total_alignment_error:.4f}")
    
    if total_alignment_error < 0.1:
        print("🎯 PERFECT ALIGNMENT! Axes match robot coordinates.")
    elif total_alignment_error < 0.3:
        print("✅ EXCELLENT ALIGNMENT! Very close to robot coordinates.")
    else:
        print("⚠ Axes still not perfectly aligned - may need fine-tuning.")

def main():
    """Main test function."""
    print("TESTING 90° Z-ROTATION COORDINATE CORRECTION")
    print("="*80)
    
    data = load_calibration_data()
    
    # Test corrected validation
    trans_error, rot_error = test_corrected_validation(data)
    
    # Test axis alignment
    test_axis_alignment(data)
    
    print("\n" + "="*80)
    print("FINAL ASSESSMENT")
    print("="*80)
    
    if rot_error < 10 and trans_error < 30:
        print("🎉 SUCCESS! The 90° Z-rotation correction fixes the calibration!")
        print("✅ Both rotation and translation errors are now within acceptable ranges.")
        print("✅ The CharUco axes should now align with robot axes as observed.")
        print("\nRecommendation: Use the corrected calibration files for accurate results!")
    else:
        print("⚠ PARTIAL SUCCESS - improvement achieved but may need additional corrections.")
        print(f"Rotation error: {rot_error:.2f}° (target: <10°)")
        print(f"Translation error: {trans_error:.2f}mm (target: <30mm)")

if __name__ == "__main__":
    main()