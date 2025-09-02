#!/usr/bin/env python3
"""
Test script to verify the loaded transformation matrix and demonstrate coordinate transformations.
"""

import numpy as np
from scipy.spatial.transform import Rotation
import sys

def load_and_analyze_transformation_matrix(matrix_file):
    """Load and analyze the transformation matrix."""
    print("="*60)
    print("TRANSFORMATION MATRIX ANALYSIS")
    print("="*60)
    
    try:
        transform = np.load(matrix_file)
        print(f"Successfully loaded: {matrix_file}")
        print("\nTransformation Matrix:")
        print(transform)
        
        # Extract translation and rotation
        translation = transform[:3, 3] * 1000  # Convert to mm
        rotation_matrix = transform[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        
        print(f"\nCamera position relative to robot hand:")
        print(f"  Translation: X={translation[0]:.1f}mm, Y={translation[1]:.1f}mm, Z={translation[2]:.1f}mm")
        print(f"  Rotation: RX={euler_angles[0]:.1f}°, RY={euler_angles[1]:.1f}°, RZ={euler_angles[2]:.1f}°")
        
        return transform
        
    except Exception as e:
        print(f"Error loading matrix: {e}")
        return None

def simulate_charuco_detection(eye_in_hand_transform):
    """Simulate CharUco detection and show coordinate transformation."""
    print("\n" + "="*60)
    print("SIMULATED CHARUCO POSITION PREDICTION")
    print("="*60)
    
    # Simulate CharUco pattern detected by camera at different positions
    test_positions = [
        ([0.050, 0.000, 0.200], "50mm right, 200mm forward"),
        ([0.000, 0.000, 0.150], "Centered, 150mm forward"),
        ([-0.030, 0.020, 0.250], "30mm left, 20mm down, 250mm forward"),
        ([0.020, -0.040, 0.180], "20mm right, 40mm up, 180mm forward")
    ]
    
    print("Simulating CharUco detections in camera coordinates:")
    print("(Camera coordinates: X=right, Y=down, Z=forward)")
    print()
    
    for i, (pos_camera, description) in enumerate(test_positions, 1):
        # Create camera-to-CharUco transformation matrix
        camera_to_charuco = np.eye(4)
        camera_to_charuco[:3, 3] = pos_camera
        
        # Transform to robot hand coordinates
        # Hand_to_CharUco = Eye_in_hand^(-1) @ Camera_to_CharUco
        hand_to_charuco = np.linalg.inv(eye_in_hand_transform) @ camera_to_charuco
        
        # Extract position relative to hand
        position_in_hand = hand_to_charuco[:3, 3] * 1000.0  # Convert to millimeters
        
        print(f"Test {i}: {description}")
        print(f"  Camera coordinates: X={pos_camera[0]*1000:.0f}mm, Y={pos_camera[1]*1000:.0f}mm, Z={pos_camera[2]*1000:.0f}mm")
        print(f"  Hand coordinates:   X={position_in_hand[0]:.1f}mm, Y={position_in_hand[1]:.1f}mm, Z={position_in_hand[2]:.1f}mm")
        print(f"    (Robot coordinates: X=forward, Y=left, Z=up)")
        print()

def main():
    """Main function."""
    print("CHARUCO TRANSFORMATION MATRIX TEST")
    print("="*60)
    print("This script tests the loaded transformation matrix and demonstrates")
    print("how CharUco positions are transformed from camera to robot hand coordinates.")
    print()
    
    # Test different matrix files
    matrix_files = [
        'charuco_eye_in_hand_transform.npy',
        'eye_in_hand_transform.npy',
        'eye_in_hand_fixed_transform.npy'
    ]
    
    for matrix_file in matrix_files:
        try:
            transform = load_and_analyze_transformation_matrix(matrix_file)
            if transform is not None:
                simulate_charuco_detection(transform)
                break  # Use the first successfully loaded matrix
        except FileNotFoundError:
            print(f"File not found: {matrix_file}")
            continue
    else:
        print("No transformation matrix files found!")
        return 1
    
    print("="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Connect RealSense camera to test real-time detection")
    print("2. Place 8x11 CharUco pattern (25mm squares, 20mm markers) in camera view")
    print("3. Run: python charuco_realtime_hand_position.py")
    print("4. The system will show real-time position relative to robot hand")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())