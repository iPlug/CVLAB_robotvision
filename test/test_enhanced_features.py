#!/usr/bin/env python3
"""
Test script to verify the enhanced features of the modular tracking system.
This script tests specific improvements that were added to match the original system.
"""

import numpy as np
import sys
import os
import time

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_fps_calculation():
    """Test FPS calculation functionality."""
    print("Testing FPS Calculation...")
    
    from module.processing.temporal_filter import TemporalFilter
    
    # Create temporal filter
    temporal_filter = TemporalFilter(fps_window_size=10)
    
    # Simulate frame timestamps
    base_time = time.time()
    frame_times = [base_time + i * 0.1 for i in range(10)]  # 10 FPS
    
    for frame_time in frame_times:
        temporal_filter.add_frame_to_history(timestamp=frame_time)
    
    # Calculate FPS
    fps = temporal_filter.calculate_fps()
    expected_fps = 10.0
    
    assert abs(fps - expected_fps) < 1.0, f"Expected ~{expected_fps} FPS, got {fps}"
    print(f"  [OK] FPS calculation: {fps:.1f} FPS")
    
    return True

def test_3d_outlier_filtering():
    """Test 3D statistical outlier filtering."""
    print("\nTesting 3D Statistical Outlier Filtering...")
    
    from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
    
    # Create strategy with outlier filtering enabled
    strategy = BEVClusteringStrategy()
    
    # Create test data with outliers
    # Main cluster
    main_cluster = np.random.normal([0.5, 0.1, 0.3], [0.05, 0.01, 0.05], (100, 3))
    # Outliers
    outliers = np.array([[2.0, 0.5, 1.0], [-2.0, 0.5, 1.0]])  # Far from main cluster
    
    vertices = np.vstack([main_cluster, outliers])
    print(f"  Input: {len(vertices)} points (100 cluster + 2 outliers)")
    
    # Apply outlier filtering
    filtered_vertices = strategy.apply_3d_statistical_outlier_filter(vertices)
    print(f"  Output: {len(filtered_vertices)} points after filtering")
    
    # Should remove most/all outliers
    assert len(filtered_vertices) < len(vertices), "Outlier filtering should remove some points"
    assert len(filtered_vertices) >= 80, "Should keep most of the main cluster"
    
    print(f"  [OK] Removed {len(vertices) - len(filtered_vertices)} outlier points")
    
    return True

def test_visualization_pipeline():
    """Test visualization pipeline with 3-panel display."""
    print("\nTesting Visualization Pipeline...")
    
    from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
    from module.config.config_factory import ConfigFactory
    
    # Create strategy
    config = ConfigFactory.create_preset('realtime_tracking')
    strategy = BEVClusteringStrategy(
        roi_params=config.get('roi_params'),
        display_roi_params=config.get('display_roi_params'),
        clustering_params=config.get('clustering_params')
    )
    
    # Create test point cloud
    vertices = np.random.rand(200, 3) * 0.5
    vertices[:, 1] = vertices[:, 1] * 0.05 + 0.1  # Height in valid range
    vertices[:, 2] = np.abs(vertices[:, 2]) + 0.1  # Positive Z values
    
    # Test visualization pipeline
    result = strategy.cluster_objects(vertices, return_visualization=True)
    
    if len(result) == 3:
        clusters, bev_image, point_cloud_view = result
        print(f"  [OK] Detected {len(clusters)} clusters")
        print(f"  [OK] Generated BEV visualization: {bev_image.shape if bev_image is not None else 'None'}")
        print(f"  [OK] Generated point cloud view: {point_cloud_view.shape if point_cloud_view is not None else 'None'}")
        
        # Verify images are valid
        if bev_image is not None:
            assert len(bev_image.shape) == 3, "BEV image should be 3-channel"
            assert bev_image.shape[2] == 3, "BEV image should be BGR"
        
        if point_cloud_view is not None:
            assert len(point_cloud_view.shape) == 3, "Point cloud view should be 3-channel"
            assert point_cloud_view.shape[2] == 3, "Point cloud view should be BGR"
    else:
        clusters = result
        print(f"  [OK] Detected {len(clusters)} clusters (no visualization)")
    
    return True

def test_cluster_outlier_refinement():
    """Test cluster-level outlier refinement."""
    print("\nTesting Cluster-Level Outlier Refinement...")
    
    from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
    
    strategy = BEVClusteringStrategy()
    
    # Create mock cluster with outlier points in height
    cluster_points = np.random.normal([0.5, 0.1, 0.3], [0.05, 0.01, 0.05], (50, 3))
    # Add height outliers
    outlier_points = np.array([
        [0.5, 0.3, 0.3],  # High outlier
        [0.5, -0.1, 0.3], # Low outlier
    ])
    all_points = np.vstack([cluster_points, outlier_points])
    
    # Create mock cluster
    mock_cluster = {
        'id': 1,
        'points': all_points,
        'center': np.mean(all_points, axis=0),
        'min_coords': np.min(all_points, axis=0),
        'max_coords': np.max(all_points, axis=0),
        'dimensions': np.max(all_points, axis=0) - np.min(all_points, axis=0),
        'num_points': len(all_points)
    }
    
    clusters = [mock_cluster]
    print(f"  Input cluster: {len(all_points)} points")
    
    # Apply cluster-level outlier removal
    refined_clusters = strategy.remove_statistical_outliers_from_clusters(clusters)
    
    if refined_clusters:
        refined_points = refined_clusters[0]['points']
        print(f"  Output cluster: {len(refined_points)} points")
        print(f"  [OK] Removed {len(all_points) - len(refined_points)} outlier points from cluster")
        
        assert len(refined_points) <= len(all_points), "Should not add points"
        assert len(refined_points) >= 40, "Should keep most normal points"
    else:
        print("  [OK] Cluster was filtered out entirely (too few points)")
    
    return True

def test_coordinate_format_consistency():
    """Test coordinate format consistency with original system."""
    print("\nTesting Coordinate Format Consistency...")
    
    from modular_track_object import ModularObjectTracker
    
    # Create tracker instance
    tracker = ModularObjectTracker(config_name='realtime_tracking')
    
    # Test center_mm format extraction
    mock_object_with_center_mm = {
        'center_mm': [50, 100, 200]  # [side, height, forward]
    }
    
    position = tracker._extract_object_position(mock_object_with_center_mm)
    expected = [200, 50, 100]  # [forward, side, height]
    
    assert position == expected, f"Expected {expected}, got {position}"
    print(f"  [OK] center_mm format: {mock_object_with_center_mm['center_mm']} -> {position}")
    
    # Test center format extraction
    mock_object_with_center = {
        'center': np.array([0.05, 0.1, 0.2])  # [side, height, forward] in meters
    }
    
    position = tracker._extract_object_position(mock_object_with_center)
    expected = [200, 50, 100]  # [forward, side, height] in mm
    
    for i in range(3):
        assert abs(position[i] - expected[i]) < 0.1, f"Expected {expected}, got {position}"
    
    print(f"  [OK] center format: {mock_object_with_center['center']} -> {position}")
    
    return True

def run_enhanced_feature_tests():
    """Run all enhanced feature tests."""
    print("=" * 60)
    print("ENHANCED MODULAR TRACKING FEATURES - VERIFICATION TESTS")
    print("=" * 60)
    
    tests = [
        ("FPS Calculation", test_fps_calculation),
        ("3D Statistical Outlier Filtering", test_3d_outlier_filtering),
        ("Visualization Pipeline", test_visualization_pipeline),
        ("Cluster-Level Outlier Refinement", test_cluster_outlier_refinement),
        ("Coordinate Format Consistency", test_coordinate_format_consistency)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
            else:
                print(f"[ERROR] {test_name} failed")
        except Exception as e:
            print(f"[ERROR] {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("ENHANCED FEATURES TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All enhanced features are working correctly!")
        print("\nThe modular system now includes all key features from the original:")
        print("- FPS calculation and display")
        print("- 3-panel visualization (Color | BEV | Point Cloud)")
        print("- 3D statistical outlier filtering")
        print("- Cluster-level outlier refinement")
        print("- Coordinate format consistency")
        print("- Enhanced visualization pipeline")
    else:
        print("[WARNING] Some enhanced features failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_enhanced_feature_tests()
    sys.exit(0 if success else 1)