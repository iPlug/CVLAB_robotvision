"""
Test script to verify module functionality and strategy combinations.
"""

import numpy as np
import sys
import os

# Add module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        # Core imports
        from module.core.object_detector import ObjectDetector
        from module.core.clustering_strategy import ClusteringStrategy
        from module.core.classification_strategy import ClassificationStrategy
        from module.core.visualization_engine import VisualizationEngine
        print("[OK] Core modules imported successfully")
        
        # Strategy imports
        from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
        from module.strategies.dbscan_clustering_strategy import DBSCANClusteringStrategy
        from module.strategies.pointnet_subspace_classifier_strategy import PointNetSubspaceClassifierStrategy
        print("[OK] Strategy modules imported successfully")
        
        # Processing imports
        from module.processing.point_cloud_processor import PointCloudProcessor
        from module.processing.image_processor import ImageProcessor
        from module.processing.temporal_filter import TemporalFilter
        print("[OK] Processing modules imported successfully")
        
        # Sensor imports
        from module.sensors.sensor_manager import SensorManager
        from module.sensors.realsense_manager import RealSenseManager
        from module.sensors.file_based_manager import FileBasedManager
        print("[OK] Sensor modules imported successfully")
        
        # Config imports
        from module.config.detection_config import DetectionConfig
        from module.config.bev_config import BEVConfig
        from module.config.pointnet_config import PointNetConfig
        from module.config.config_factory import ConfigFactory
        print("[OK] Configuration modules imported successfully")
        
        # App imports
        from module.apps.unified_detection_app import UnifiedDetectionApp
        from module.apps.realtime_detection_app import RealTimeDetectionApp
        from module.apps.batch_processing_app import BatchProcessingApp
        print("[OK] Application modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False

def test_configurations():
    """Test configuration creation and validation."""
    print("\nTesting configurations...")
    
    try:
        from module.config.config_factory import ConfigFactory
        
        # Test BEV config
        bev_config = ConfigFactory.create_config('bev')
        assert bev_config.validate_config(), "BEV config validation failed"
        print("[OK] BEV configuration created and validated")
        
        # Test PointNet config
        pointnet_config = ConfigFactory.create_config('pointnet')
        assert pointnet_config.validate_config(), "PointNet config validation failed"
        print("[OK] PointNet configuration created and validated")
        
        # Test presets
        realtime_preset = ConfigFactory.create_preset('realtime_bev')
        assert realtime_preset.validate_config(), "Realtime BEV preset validation failed"
        print("[OK] Realtime BEV preset created and validated")
        
        offline_preset = ConfigFactory.create_preset('offline_pointnet')
        assert offline_preset.validate_config(), "Offline PointNet preset validation failed"
        print("[OK] Offline PointNet preset created and validated")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Configuration test error: {e}")
        return False

def test_clustering_strategies():
    """Test clustering strategy creation and basic functionality."""
    print("\nTesting clustering strategies...")
    
    try:
        from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
        from module.strategies.dbscan_clustering_strategy import DBSCANClusteringStrategy
        from module.config.config_factory import ConfigFactory
        
        # Create sample point cloud data
        vertices = np.random.rand(1000, 3) * 2 - 1  # Random points in [-1, 1] cube
        vertices[:, 2] = np.abs(vertices[:, 2])  # Make Z positive (depth)
        
        # Test BEV clustering strategy
        bev_config = ConfigFactory.create_config('bev')
        bev_strategy = BEVClusteringStrategy(
            roi_params=bev_config.get('roi_params'),
            display_roi_params=bev_config.get('display_roi_params'),
            clustering_params=bev_config.get('clustering_params')
        )
        
        bev_clusters = bev_strategy.cluster_objects(vertices)
        assert isinstance(bev_clusters, list), "BEV strategy should return list"
        assert bev_strategy.get_strategy_name() == "BEV Image Clustering"
        print("[OK] BEV clustering strategy works")
        
        # Test DBSCAN clustering strategy
        dbscan_strategy = DBSCANClusteringStrategy()
        dbscan_clusters = dbscan_strategy.cluster_objects(vertices)
        assert isinstance(dbscan_clusters, list), "DBSCAN strategy should return list"
        assert dbscan_strategy.get_strategy_name() == "DBSCAN 3D Clustering"
        print("[OK] DBSCAN clustering strategy works")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Clustering strategy test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_object_detector():
    """Test ObjectDetector with different strategy combinations."""
    print("\nTesting ObjectDetector...")
    
    try:
        from module.core.object_detector import ObjectDetector
        from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
        from module.strategies.dbscan_clustering_strategy import DBSCANClusteringStrategy
        from module.config.config_factory import ConfigFactory
        
        # Create sample data
        vertices = np.random.rand(500, 3) * 2 - 1
        vertices[:, 2] = np.abs(vertices[:, 2])
        
        # Test with BEV clustering only
        bev_config = ConfigFactory.create_preset('realtime_bev')
        bev_strategy = BEVClusteringStrategy(
            roi_params=bev_config.get('roi_params'),
            display_roi_params=bev_config.get('display_roi_params'),
            clustering_params=bev_config.get('clustering_params')
        )
        
        detector_bev = ObjectDetector(bev_strategy)
        objects_bev, timing_bev = detector_bev.detect_objects(vertices)
        assert isinstance(objects_bev, list), "Should return list of objects"
        assert isinstance(timing_bev, dict), "Should return timing info"
        print("[OK] ObjectDetector with BEV clustering works")
        
        # Test with DBSCAN clustering only
        dbscan_strategy = DBSCANClusteringStrategy()
        detector_dbscan = ObjectDetector(dbscan_strategy)
        objects_dbscan, timing_dbscan = detector_dbscan.detect_objects(vertices)
        assert isinstance(objects_dbscan, list), "Should return list of objects"
        print("[OK] ObjectDetector with DBSCAN clustering works")
        
        # Test strategy switching
        detector_bev.set_clustering_strategy(dbscan_strategy)
        strategy_info = detector_bev.get_strategy_info()
        assert "DBSCAN" in strategy_info['clustering_strategy']
        print("[OK] Strategy switching works")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] ObjectDetector test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_processing_utilities():
    """Test processing utility classes."""
    print("\nTesting processing utilities...")
    
    try:
        from module.processing.point_cloud_processor import PointCloudProcessor
        from module.processing.image_processor import ImageProcessor
        from module.processing.temporal_filter import TemporalFilter
        
        # Test PointCloudProcessor
        processor = PointCloudProcessor()
        vertices = np.random.rand(1000, 3) * 2 - 1
        
        # Test ROI filtering
        roi_params = {'x_min': -0.5, 'x_max': 0.5, 'y_min': -0.5, 'y_max': 0.5, 'z_min': 0, 'z_max': 1}
        filtered = processor.apply_roi_filter(vertices, roi_params)
        assert len(filtered) <= len(vertices), "ROI filter should reduce or maintain point count"
        print("[OK] PointCloudProcessor works")
        
        # Test ImageProcessor
        img_processor = ImageProcessor()
        bev_image = img_processor.create_bev_image(vertices, (-1, 1), (0, 2))
        assert bev_image.shape[2] == 3, "BEV image should be RGB"
        print("[OK] ImageProcessor works")
        
        # Test TemporalFilter
        temporal = TemporalFilter(window_size=3)
        test_objects = [{'id': 1, 'center': np.array([0, 0, 1])}]
        stabilized = temporal.stabilize_object_positions(test_objects)
        assert len(stabilized) == len(test_objects), "Should return same number of objects"
        print("[OK] TemporalFilter works")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Processing utilities test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sensor_managers():
    """Test sensor manager classes (basic instantiation)."""
    print("\nTesting sensor managers...")
    
    try:
        from module.sensors.file_based_manager import FileBasedManager
        
        # Test FileBasedManager (safe to test without actual files)
        file_manager = FileBasedManager("test_folder")
        assert file_manager.get_sensor_info()['sensor_type'] == 'File-based'
        print("[OK] FileBasedManager instantiation works")
        
        # Note: RealSenseManager requires actual hardware/drivers, so we skip testing it
        print("[OK] Sensor manager tests completed (RealSense skipped - requires hardware)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Sensor manager test error: {e}")
        return False

def test_visualization_engine():
    """Test visualization engine."""
    print("\nTesting visualization engine...")
    
    try:
        from module.core.visualization_engine import VisualizationEngine
        
        visualizer = VisualizationEngine()
        
        # Test color assignment
        color1 = visualizer.get_color_for_class("bottle")
        color2 = visualizer.get_color_for_class("box")
        assert color1 != color2, "Different classes should have different colors"
        
        # Test with sample image
        import cv2
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test performance overlay
        timing_info = {'total_time': 0.1, 'clustering_time': 0.05, 'classification_time': 0.03, 'num_objects': 2}
        strategy_info = {'clustering_strategy': 'BEV', 'classification_strategy': 'None'}
        
        result_image = visualizer.add_performance_overlay(test_image, timing_info, strategy_info, 100)
        assert result_image.shape == test_image.shape, "Output image should have same shape"
        print("[OK] VisualizationEngine works")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] VisualizationEngine test error: {e}")
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("MODULAR OBJECT DETECTION SYSTEM - FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configurations),
        ("Clustering Strategy Tests", test_clustering_strategies),
        ("ObjectDetector Tests", test_object_detector),
        ("Processing Utility Tests", test_processing_utilities),
        ("Sensor Manager Tests", test_sensor_managers),
        ("Visualization Engine Tests", test_visualization_engine)
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
        print("[SUCCESS] All tests passed! The modular system is working correctly.")
        print("\nYou can now use the system with:")
        print("- BEV clustering for real-time applications")
        print("- DBSCAN clustering for high-quality offline processing")
        print("- PointNet+Subspace classification (when models are available)")
        print("- Runtime strategy switching in UnifiedDetectionApp")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)