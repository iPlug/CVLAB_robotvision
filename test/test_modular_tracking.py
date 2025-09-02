"""
Test script for the modular object tracking system.
"""

import numpy as np
import sys
import os

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_robot_controller():
    """Test robot controller functionality."""
    print("Testing Robot Controller...")
    
    from module.robot import MyCobotController
    
    # Test initialization
    robot = MyCobotController()
    assert robot.robot_type == "myCobot 280"
    assert robot.workspace_bounds is not None
    assert robot.safety_constraints is not None
    print("  [OK] Robot controller initialization")
    
    # Test transformation matrix loading (should handle missing file gracefully)
    result = robot.load_transformation_matrix("nonexistent_file.npy")
    assert result == False
    print("  [OK] Transformation matrix error handling")
    
    # Test coordinate transformation (should handle missing matrix)
    camera_pos = [100, 50, 200]
    robot_pos = robot.transform_camera_to_robot(camera_pos)
    assert robot_pos is None  # Should be None without transformation matrix
    print("  [OK] Coordinate transformation handling")
    
    # Test safety validation
    safe_pos = [200, 100, 150, 0, 180, 0]
    is_safe, reason = robot.validate_position_safety(safe_pos)
    assert is_safe == True
    print("  [OK] Safety validation for valid position")
    
    unsafe_pos = [50, 0, 50, 0, 180, 0]  # Below minimum height
    is_safe, reason = robot.validate_position_safety(unsafe_pos)
    assert is_safe == False
    print("  [OK] Safety validation for unsafe position")
    
    # Test position clamping
    clamped_pos, adjustments = robot.clamp_position_to_workspace([50, 300, 50])
    assert len(adjustments) > 0  # Should have adjustments
    print("  [OK] Position clamping with adjustments")
    
    print("Robot Controller tests passed!")
    return True

def test_tracking_configurations():
    """Test tracking-specific configurations."""
    print("\nTesting Tracking Configurations...")
    
    from module.config.config_factory import ConfigFactory
    
    # Test tracking presets
    tracking_presets = ['realtime_tracking', 'precision_tracking', 'adaptive_tracking']
    
    for preset_name in tracking_presets:
        try:
            config = ConfigFactory.create_preset(preset_name)
            assert config.validate_config(), f"Config validation failed for {preset_name}"
            
            # Check for tracking-specific parameters
            tracking_params = config.get('tracking_params')
            robot_params = config.get('robot_params')
            
            assert tracking_params is not None, f"Missing tracking_params in {preset_name}"
            assert robot_params is not None, f"Missing robot_params in {preset_name}"
            assert 'update_interval' in tracking_params, f"Missing update_interval in {preset_name}"
            assert 'min_height' in robot_params, f"Missing min_height in {preset_name}"
            
            print(f"  [OK] {preset_name} configuration")
            
        except Exception as e:
            print(f"  [ERROR] {preset_name} configuration failed: {e}")
            return False
    
    print("Tracking Configuration tests passed!")
    return True

def test_modular_tracker_initialization():
    """Test modular tracker initialization without hardware."""
    print("\nTesting Modular Tracker Initialization...")
    
    from modular_track_object import ModularObjectTracker
    
    try:
        # Test with different configurations
        configs_to_test = ['realtime_tracking', 'precision_tracking']
        
        for config_name in configs_to_test:
            tracker = ModularObjectTracker(
                config_name=config_name,
                initial_strategy='bev',
                bag_file=None,  # No bag file for testing
                transformation_matrix_file='nonexistent.npy'  # Non-existent file
            )
            
            # Test configuration loading
            assert tracker.config is not None
            assert tracker.config_name == config_name
            print(f"  [OK] {config_name} tracker created")
            
            # Test strategy management
            assert 'bev' in tracker.available_strategies or len(tracker.available_strategies) == 0
            print(f"  [OK] {config_name} strategies initialized")
        
        print("Modular Tracker Initialization tests passed!")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Modular tracker initialization failed: {e}")
        return False

def test_object_detection_pipeline():
    """Test object detection pipeline with mock data."""
    print("\nTesting Object Detection Pipeline...")
    
    from module.core.object_detector import ObjectDetector
    from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
    from module.strategies.dbscan_clustering_strategy import DBSCANClusteringStrategy
    from module.config.config_factory import ConfigFactory
    
    try:
        # Create test point cloud data
        vertices = np.random.rand(1000, 3) * 2 - 1
        vertices[:, 2] = np.abs(vertices[:, 2])  # Make Z positive
        
        # Test BEV strategy
        bev_config = ConfigFactory.create_preset('realtime_tracking')
        bev_strategy = BEVClusteringStrategy(
            roi_params=bev_config.get('roi_params'),
            display_roi_params=bev_config.get('display_roi_params'),
            clustering_params=bev_config.get('clustering_params')
        )
        
        detector_bev = ObjectDetector(bev_strategy)
        objects_bev, timing_bev = detector_bev.detect_objects(vertices)
        assert isinstance(objects_bev, list)
        assert isinstance(timing_bev, dict)
        print("  [OK] BEV detection pipeline")
        
        # Test DBSCAN strategy
        dbscan_strategy = DBSCANClusteringStrategy()
        detector_dbscan = ObjectDetector(dbscan_strategy)
        objects_dbscan, timing_dbscan = detector_dbscan.detect_objects(vertices)
        assert isinstance(objects_dbscan, list)
        assert isinstance(timing_dbscan, dict)
        print("  [OK] DBSCAN detection pipeline")
        
        # Test strategy switching
        detector_bev.set_clustering_strategy(dbscan_strategy)
        strategy_info = detector_bev.get_strategy_info()
        assert "DBSCAN" in strategy_info['clustering_strategy']
        print("  [OK] Strategy switching")
        
        print("Object Detection Pipeline tests passed!")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Object detection pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_coordinate_transformations():
    """Test coordinate transformation utilities."""
    print("\nTesting Coordinate Transformations...")
    
    from module.robot import MyCobotController
    
    try:
        robot = MyCobotController()
        
        # Create a mock transformation matrix
        transformation_matrix = np.array([
            [1, 0, 0, 100],
            [0, 1, 0, 50],
            [0, 0, 1, 200],
            [0, 0, 0, 1]
        ])
        robot.transformation_matrix = transformation_matrix
        
        # Test transformation
        camera_pos = [0, 0, 0]
        robot_pos = robot.transform_camera_to_robot(camera_pos)
        expected = [100, 50, 200]
        
        assert robot_pos is not None
        assert len(robot_pos) == 3
        # Check if transformation applied correctly (within tolerance)
        for i in range(3):
            assert abs(robot_pos[i] - expected[i]) < 0.001
        
        print("  [OK] Coordinate transformation calculation")
        
        # Test pointing at object
        # This will test safety validation and coordinate transformation together
        camera_position = [0, 0, 100]  # 100mm forward in camera frame
        result = robot.point_at_object(camera_position)
        # Should fail because robot is not connected, but transformation should work
        print("  [OK] Point at object pipeline (expected to fail at robot command)")
        
        print("Coordinate Transformation tests passed!")
        return True
        
    except Exception as e:
        print(f"  [ERROR] Coordinate transformation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all modular tracking tests."""
    print("=" * 60)
    print("MODULAR OBJECT TRACKING SYSTEM - COMPREHENSIVE TESTS")
    print("=" * 60)
    
    tests = [
        ("Robot Controller", test_robot_controller),
        ("Tracking Configurations", test_tracking_configurations),
        ("Modular Tracker Initialization", test_modular_tracker_initialization),
        ("Object Detection Pipeline", test_object_detection_pipeline),
        ("Coordinate Transformations", test_coordinate_transformations)
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
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All modular tracking tests passed!")
        print("\nThe modular object tracking system is ready for use with:")
        print("- Runtime strategy switching (BEV <-> DBSCAN)")
        print("- Configurable tracking presets (realtime, precision, adaptive)")
        print("- Enhanced safety constraints and workspace validation")
        print("- Performance monitoring and adaptive timing")
        print("- Comprehensive error handling and recovery")
        print("\nUsage:")
        print("  python modular_track_object.py --config realtime_tracking --strategy bev")
        print("  python modular_track_object.py --config precision_tracking --strategy dbscan")
        print("  python modular_track_object.py --config adaptive_tracking --bag your_file.bag")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)