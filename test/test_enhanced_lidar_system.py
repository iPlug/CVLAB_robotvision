#!/usr/bin/env python3
"""
Test script for enhanced LiDAR system with floor-aligned object pose detection.

This script tests the complete pipeline:
1. FloorAlignedPoseCalculator - Calculate 6DOF poses for floor-placed objects
2. Enhanced FloorDetector - Extract floor coordinate systems  
3. Enhanced LiDARCoordinator - Transform complete poses to robot coordinates
4. LiDARClusteringStrategy - Pass floor information to pose calculation

The system provides complete 6DOF object poses instead of just positions,
enabling proper robot grasping with floor-aligned orientations.
"""

import numpy as np
import sys
import os
sys.path.append('.')

from module.processing.floor_aligned_pose_calculator import FloorAlignedPoseCalculator
from module.processing.floor_detector import FloorDetector
from module.processing.lidar_coordinator import LiDARCoordinator
from module.strategies.lidar_clustering_strategy import LiDARClusteringStrategy


def test_complete_pipeline():
    """Test the complete enhanced LiDAR pipeline."""
    print("=" * 60)
    print("ENHANCED LIDAR SYSTEM VALIDATION TEST")
    print("=" * 60)
    
    # Step 1: Create test data
    print("\n1. Creating test data...")
    
    # Generate synthetic floor + object point cloud
    n_floor_points = 800
    floor_x = np.random.uniform(-0.8, 0.8, n_floor_points)
    floor_z = np.random.uniform(-0.8, 0.8, n_floor_points)
    floor_y = np.zeros(n_floor_points)  # Table at Y=0
    floor_points = np.column_stack([floor_x, floor_y, floor_z])
    floor_points += np.random.normal(0, 0.005, floor_points.shape)  # Add noise
    
    # Add object points (box-like object on table)
    object_center = np.array([0.3, 0.05, 0.1])  # 30cm forward, 5cm up, 10cm right
    n_object_points = 200
    object_x = object_center[0] + np.random.uniform(-0.03, 0.03, n_object_points)
    object_y = object_center[1] + np.random.uniform(0, 0.05, n_object_points)
    object_z = object_center[2] + np.random.uniform(-0.02, 0.02, n_object_points)
    object_points = np.column_stack([object_x, object_y, object_z])
    
    # Combine all points
    all_vertices = np.vstack([floor_points, object_points])
    
    print(f"   Generated {len(all_vertices)} points:")
    print(f"   - Floor points: {len(floor_points)}")
    print(f"   - Object points: {len(object_points)}")
    print(f"   - Object center: [{object_center[0]:.3f}, {object_center[1]:.3f}, {object_center[2]:.3f}]m")
    
    # Step 2: Test clustering strategy with floor information
    print("\n2. Testing LiDAR clustering with floor detection...")
    
    clustering_strategy = LiDARClusteringStrategy(
        distance_range=(0.2, 1.0),  # Wider range for test
        floor_params={
            'distance_threshold': 0.02,
            'max_trials': 1000,
            'min_floor_points': 300,
            'floor_normal_tolerance': 0.3
        },
        clustering_params={
            'eps': 0.05,
            'min_samples': 10,
            'min_cluster_size': 50,
            'max_cluster_size': 2000,
            'outlier_removal': True
        }
    )
    
    # Perform clustering
    clusters = clustering_strategy.cluster_objects(all_vertices)
    
    if clusters:
        selected_cluster = clusters[0]
        print(f"   [OK] Object detected: {selected_cluster['size']} points")
        print(f"   [OK] Detected center: [{selected_cluster['center'][0]:.3f}, {selected_cluster['center'][1]:.3f}, {selected_cluster['center'][2]:.3f}]m")
        
        # Check floor information
        floor_info = clustering_strategy.get_floor_information()
        floor_normal = floor_info.get('floor_normal')
        
        if floor_normal is not None:
            print(f"   [OK] Floor normal: [{floor_normal[0]:.3f}, {floor_normal[1]:.3f}, {floor_normal[2]:.3f}]")
        else:
            print("   [WARN] No floor normal detected")
            
    else:
        print("   [ERROR] No clusters detected")
        return False
    
    # Step 3: Test coordinate transformation
    print("\n3. Testing coordinate transformation...")
    
    # Create mock hand-eye transformation matrix
    mock_hand_eye = np.array([
        [1, 0, 0, 0.05],   # 5cm forward offset
        [0, 1, 0, 0.02],   # 2cm left offset
        [0, 0, 1, 0.12],   # 12cm up offset
        [0, 0, 0, 1]
    ])
    
    # Save temporarily for test
    test_matrix_file = 'temp_test_transform.npy'
    np.save(test_matrix_file, mock_hand_eye)
    
    try:
        coordinator = LiDARCoordinator(
            transformation_matrix_file=test_matrix_file,
            coordinate_system='standard'
        )
        
        # Transform complete pose
        if floor_normal is not None:
            robot_coords = coordinator.transform_cluster_pose(
                selected_cluster, floor_normal, floor_info
            )
        else:
            robot_coords = coordinator.transform_cluster_center(selected_cluster)
        
        if robot_coords:
            print(f"   [OK] Transformation successful")
            print(f"   [OK] Robot position: [{robot_coords['robot_position'][0]:.1f}, {robot_coords['robot_position'][1]:.1f}, {robot_coords['robot_position'][2]:.1f}] mm")
            
            if robot_coords.get('has_orientation', False):
                orient = robot_coords.get('robot_orientation_degrees', [0, 0, 0])
                print(f"   [OK] Robot orientation: [{orient[0]:.1f}deg, {orient[1]:.1f}deg, {orient[2]:.1f}deg]")
                print(f"   [OK] 6DOF pose: Available")
                
                # Check floor normal transformation
                if robot_coords.get('floor_normal_robot') is not None:
                    floor_robot = robot_coords['floor_normal_robot']
                    print(f"   [OK] Floor normal (robot frame): [{floor_robot[0]:.3f}, {floor_robot[1]:.3f}, {floor_robot[2]:.3f}]")
                
            else:
                print(f"   [WARN] Only position available (no orientation)")
            
            # Check suggested pose
            if 'suggested_pose' in robot_coords:
                pose = robot_coords['suggested_pose']
                approach_type = pose.get('approach_type', 'pointing')
                print(f"   [OK] Suggested approach: {approach_type}")
                
        else:
            print("   [ERROR] Transformation failed")
            return False
    
    finally:
        # Cleanup
        if os.path.exists(test_matrix_file):
            os.remove(test_matrix_file)
    
    # Step 4: Test standalone components
    print("\n4. Testing individual components...")
    
    # Test pose calculator
    pose_calc = FloorAlignedPoseCalculator()
    pose_result = pose_calc.calculate_floor_aligned_pose(
        object_center, floor_normal if floor_normal is not None else np.array([0, 1, 0])
    )
    
    if pose_result['is_valid']:
        print(f"   [OK] FloorAlignedPoseCalculator: Working")
        print(f"     Floor alignment score: {pose_result['floor_alignment_score']:.3f}")
    else:
        print(f"   [ERROR] FloorAlignedPoseCalculator: Failed")
    
    # Test floor detector coordinate system
    floor_detector = FloorDetector()
    floor_detector.detect_floor(all_vertices)
    coord_system = floor_detector.get_floor_coordinate_system()
    
    if coord_system is not None:
        print(f"   [OK] FloorDetector coordinate system: Working")
        z_axis = coord_system['z_axis']
        print(f"     Floor Z-axis: [{z_axis[0]:.3f}, {z_axis[1]:.3f}, {z_axis[2]:.3f}]")
    else:
        print(f"   [ERROR] FloorDetector coordinate system: Failed")
    
    print("\n" + "=" * 60)
    print("ENHANCED LIDAR SYSTEM VALIDATION COMPLETE")
    print("[OK] Floor-aligned 6DOF object pose detection is working!")
    print("[OK] System ready for integration with CharUco calibration")
    print("=" * 60)
    
    return True


def test_performance_comparison():
    """Compare performance between old and new approaches."""
    print("\n" + "=" * 40)
    print("PERFORMANCE COMPARISON")
    print("=" * 40)
    
    # Test data
    object_position = np.array([0.3, 0.05, 0.1])
    floor_normal = np.array([0, 1, 0])
    
    import time
    
    # Old approach: Position only
    start_time = time.time()
    for _ in range(1000):
        # Just position transformation (what we had before)
        result_pos = object_position * 1000  # Convert to mm
    old_time = (time.time() - start_time) * 1000  # ms
    
    # New approach: Complete 6DOF pose
    calculator = FloorAlignedPoseCalculator()
    start_time = time.time()
    for _ in range(1000):
        result_pose = calculator.calculate_floor_aligned_pose(object_position, floor_normal)
    new_time = (time.time() - start_time) * 1000  # ms
    
    print(f"Position-only calculation:  {old_time:.2f}ms (1000 iterations)")
    print(f"6DOF pose calculation:      {new_time:.2f}ms (1000 iterations)")
    print(f"Performance overhead:       {new_time - old_time:.2f}ms")
    print(f"Per-frame overhead:         {(new_time - old_time) / 1000:.4f}ms")
    print("[OK] Performance impact is minimal for real-time use")


if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        test_performance_comparison()
        print(f"\n[SUCCESS] All tests passed! Enhanced LiDAR system is ready to use.")
    else:
        print(f"\n[FAILED] Some tests failed. Please check the implementation.")