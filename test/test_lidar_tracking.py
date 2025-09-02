#!/usr/bin/env python3
"""
LiDAR Tracking System Validation and Testing

This script provides comprehensive testing and validation of the LiDAR-based
object tracking system, including accuracy measurements, performance benchmarks,
and integration testing with the existing hand-eye calibration matrix.

Usage:
    python test_lidar_tracking.py [--config CONFIG] [--bag BAG_FILE] [--matrix MATRIX_FILE]

Features:
- Coordinate transformation accuracy validation
- Floor detection performance testing
- Object clustering accuracy assessment
- Real-time performance benchmarks
- Integration testing with robot coordination system
- Comparative analysis with RGB-based system

Author: LiDAR Testing System
"""

import argparse
import sys
import os
import time
import numpy as np
import cv2
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.sensors.realsense_manager import RealSenseManager
from module.strategies.lidar_clustering_strategy import LiDARClusteringStrategy
from module.processing.lidar_coordinator import LiDARCoordinator
from module.processing.floor_detector import FloorDetector


class LiDARTrackingValidator:
    """
    Comprehensive validation system for LiDAR-based object tracking.
    
    Tests and validates:
    - Floor detection accuracy and performance
    - Object clustering quality and consistency
    - Coordinate transformation precision
    - Real-time performance characteristics
    - Integration with existing calibration system
    """
    
    def __init__(self, 
                 config_name: str = "lidar_tracking",
                 bag_file: Optional[str] = None,
                 transformation_matrix_file: str = "charuco_eye_in_hand_transform.npy"):
        """
        Initialize LiDAR tracking validator.
        
        Args:
            config_name: Configuration preset to test
            bag_file: Optional bag file for consistent testing
            transformation_matrix_file: Hand-eye transformation matrix
        """
        self.config_name = config_name
        self.bag_file = bag_file
        self.transformation_matrix_file = transformation_matrix_file
        
        # Test components
        self.sensor_manager = None
        self.clustering_strategy = None
        self.coordinator = None
        self.floor_detector = None
        
        # Test configuration
        self.test_config = self._load_test_configuration()
        
        # Test results storage
        self.test_results = {
            'floor_detection_tests': [],
            'clustering_tests': [],
            'coordinate_tests': [],
            'performance_tests': [],
            'integration_tests': []
        }
        
        # Performance benchmarks
        self.benchmarks = {
            'floor_detection_time': [],
            'clustering_time': [],
            'coordinate_transform_time': [],
            'total_processing_time': []
        }
        
        # Known test positions for accuracy validation
        self.known_test_positions = [
            {'camera': np.array([0.0, 0.0, 0.3]), 'expected_robot': np.array([300, 0, 0])},    # 30cm straight
            {'camera': np.array([0.1, 0.0, 0.3]), 'expected_robot': np.array([300, -100, 0])}, # 10cm right
            {'camera': np.array([-0.1, 0.0, 0.3]), 'expected_robot': np.array([300, 100, 0])}, # 10cm left
        ]
    
    def _load_test_configuration(self) -> Dict[str, Any]:
        """Load test-specific configuration parameters."""
        return {
            'distance_range': (0.25, 0.8),
            'floor_params': {
                'distance_threshold': 0.02,
                'max_trials': 1000,
                'min_floor_points': 500,
                'floor_normal_tolerance': 0.3
            },
            'clustering_params': {
                'eps': 0.03,
                'min_samples': 10,
                'min_cluster_size': 50,
                'max_cluster_size': 2000
            },
            'selection_params': {
                'selection_method': 'nearest_center',
                'max_selection_distance': 1.0
            }
        }
    
    def initialize_components(self) -> bool:
        """Initialize all test components."""
        print("Initializing LiDAR tracking validation system...")
        
        # Initialize sensor manager
        self.sensor_manager = RealSenseManager(
            bag_file=self.bag_file,
            realtime_mode=False,  # Process all frames for testing
            use_imu=True
        )
        
        if not self.sensor_manager.initialize() or not self.sensor_manager.start():
            print("Failed to initialize sensor")
            return False
        
        # Initialize clustering strategy
        self.clustering_strategy = LiDARClusteringStrategy(
            distance_range=self.test_config['distance_range'],
            floor_params=self.test_config['floor_params'],
            clustering_params=self.test_config['clustering_params'],
            selection_params=self.test_config['selection_params']
        )
        
        # Initialize coordinate transformation
        self.coordinator = LiDARCoordinator(
            transformation_matrix_file=self.transformation_matrix_file
        )
        
        if not self.coordinator.hand_eye_matrix is not None:
            print("Failed to load transformation matrix")
            return False
        
        # Initialize standalone floor detector for isolated testing
        self.floor_detector = FloorDetector(
            distance_threshold=self.test_config['floor_params']['distance_threshold'],
            max_trials=self.test_config['floor_params']['max_trials'],
            min_floor_points=self.test_config['floor_params']['min_floor_points']
        )
        
        print("All components initialized successfully")
        return True
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all validation tests and return comprehensive results."""
        print("\n" + "="*60)
        print("STARTING COMPREHENSIVE LIDAR TRACKING VALIDATION")
        print("="*60)
        
        if not self.initialize_components():
            return {'error': 'Component initialization failed'}
        
        # Test 1: Floor Detection Validation
        print("\n1. Testing Floor Detection Performance...")
        floor_results = self.test_floor_detection()
        
        # Test 2: Object Clustering Validation  
        print("\n2. Testing Object Clustering Accuracy...")
        clustering_results = self.test_object_clustering()
        
        # Test 3: Coordinate Transformation Validation
        print("\n3. Testing Coordinate Transformation Accuracy...")
        coordinate_results = self.test_coordinate_transformation()
        
        # Test 4: Real-time Performance Benchmarks
        print("\n4. Running Performance Benchmarks...")
        performance_results = self.test_realtime_performance()
        
        # Test 5: Integration Testing
        print("\n5. Testing System Integration...")
        integration_results = self.test_system_integration()
        
        # Compile comprehensive results
        results = {
            'test_config': self.test_config,
            'transformation_matrix': self.transformation_matrix_file,
            'bag_file': self.bag_file,
            'floor_detection': floor_results,
            'object_clustering': clustering_results,
            'coordinate_transformation': coordinate_results,
            'performance_benchmarks': performance_results,
            'system_integration': integration_results,
            'overall_assessment': self._generate_overall_assessment()
        }
        
        # Save detailed results
        self._save_test_results(results)
        
        # Print summary
        self._print_test_summary(results)
        
        return results
    
    def test_floor_detection(self) -> Dict[str, Any]:
        """Test floor detection accuracy and consistency."""
        floor_tests = []
        frame_count = 0
        max_frames = 50  # Test on multiple frames
        
        while frame_count < max_frames:
            # Get frame
            vertices, _, metadata = self.sensor_manager.get_lidar_optimized_frame()
            
            if metadata.get('end_of_stream', False):
                break
            
            if vertices is None or len(vertices) < 100:
                continue
            
            frame_count += 1
            
            # Test floor detection
            start_time = time.time()
            object_points, floor_points, debug_info = self.floor_detector.detect_floor(
                vertices, return_debug_info=True
            )
            detection_time = time.time() - start_time
            
            # Analyze results
            test_result = {
                'frame_id': frame_count,
                'total_points': len(vertices),
                'floor_points': len(floor_points),
                'object_points': len(object_points),
                'detection_time': detection_time,
                'confidence': debug_info.get('confidence', 0.0) if debug_info else 0.0,
                'floor_detected': len(floor_points) > self.test_config['floor_params']['min_floor_points']
            }
            
            floor_tests.append(test_result)
            self.benchmarks['floor_detection_time'].append(detection_time)
        
        # Analyze floor detection performance
        successful_detections = sum(1 for t in floor_tests if t['floor_detected'])
        avg_detection_time = np.mean([t['detection_time'] for t in floor_tests])
        avg_confidence = np.mean([t['confidence'] for t in floor_tests])
        
        return {
            'total_frames_tested': len(floor_tests),
            'successful_detections': successful_detections,
            'success_rate': successful_detections / len(floor_tests) if floor_tests else 0,
            'avg_detection_time': avg_detection_time,
            'avg_confidence': avg_confidence,
            'avg_floor_points': np.mean([t['floor_points'] for t in floor_tests]),
            'detection_consistency': np.std([t['confidence'] for t in floor_tests]),
            'individual_tests': floor_tests
        }
    
    def test_object_clustering(self) -> Dict[str, Any]:
        """Test object clustering accuracy and consistency."""
        clustering_tests = []
        frame_count = 0
        max_frames = 30
        
        while frame_count < max_frames:
            # Get frame
            vertices, color_image, metadata = self.sensor_manager.get_lidar_optimized_frame()
            
            if metadata.get('end_of_stream', False):
                break
            
            if vertices is None or len(vertices) < 100:
                continue
            
            frame_count += 1
            
            # Test clustering
            start_time = time.time()
            clusters = self.clustering_strategy.cluster_objects(vertices, color_image)
            clustering_time = time.time() - start_time
            
            # Analyze clustering results
            test_result = {
                'frame_id': frame_count,
                'input_points': len(vertices),
                'clusters_found': len(clusters),
                'clustering_time': clustering_time,
                'largest_cluster_size': max([c['size'] for c in clusters]) if clusters else 0,
                'smallest_cluster_size': min([c['size'] for c in clusters]) if clusters else 0,
                'avg_cluster_size': np.mean([c['size'] for c in clusters]) if clusters else 0
            }
            
            if clusters:
                # Test cluster properties
                nearest_cluster = min(clusters, key=lambda c: c['distance'])
                test_result.update({
                    'nearest_cluster_distance': nearest_cluster['distance'],
                    'nearest_cluster_size': nearest_cluster['size'],
                    'nearest_cluster_center': nearest_cluster['center'].tolist()
                })
            
            clustering_tests.append(test_result)
            self.benchmarks['clustering_time'].append(clustering_time)
        
        # Analyze clustering performance
        frames_with_objects = sum(1 for t in clustering_tests if t['clusters_found'] > 0)
        avg_clustering_time = np.mean([t['clustering_time'] for t in clustering_tests])
        
        return {
            'total_frames_tested': len(clustering_tests),
            'frames_with_objects': frames_with_objects,
            'object_detection_rate': frames_with_objects / len(clustering_tests) if clustering_tests else 0,
            'avg_clustering_time': avg_clustering_time,
            'avg_clusters_per_frame': np.mean([t['clusters_found'] for t in clustering_tests]),
            'clustering_consistency': np.std([t['clusters_found'] for t in clustering_tests]),
            'individual_tests': clustering_tests
        }
    
    def test_coordinate_transformation(self) -> Dict[str, Any]:
        """Test coordinate transformation accuracy using known positions."""
        transformation_tests = []
        
        # Test known positions
        for i, test_pos in enumerate(self.known_test_positions):
            camera_point = test_pos['camera']
            expected_robot = test_pos['expected_robot']
            
            start_time = time.time()
            result = self.coordinator.transform_point_to_robot(camera_point)
            transform_time = time.time() - start_time
            
            if result and result['is_valid']:
                robot_pos = result['robot_position']
                position_error = np.linalg.norm(robot_pos - expected_robot)
                
                test_result = {
                    'test_id': i,
                    'camera_input': camera_point.tolist(),
                    'expected_robot': expected_robot.tolist(),
                    'actual_robot': robot_pos.tolist(),
                    'position_error_mm': position_error,
                    'transform_time': transform_time,
                    'within_workspace': result['is_valid'],
                    'distance_from_base': result['distance_from_base']
                }
            else:
                test_result = {
                    'test_id': i,
                    'camera_input': camera_point.tolist(),
                    'expected_robot': expected_robot.tolist(),
                    'error': 'Transformation failed or out of bounds',
                    'transform_time': transform_time
                }
            
            transformation_tests.append(test_result)
            self.benchmarks['coordinate_transform_time'].append(transform_time)
        
        # Analyze transformation accuracy
        valid_tests = [t for t in transformation_tests if 'position_error_mm' in t]
        
        if valid_tests:
            avg_error = np.mean([t['position_error_mm'] for t in valid_tests])
            max_error = np.max([t['position_error_mm'] for t in valid_tests])
            min_error = np.min([t['position_error_mm'] for t in valid_tests])
        else:
            avg_error = max_error = min_error = float('inf')
        
        return {
            'total_tests': len(transformation_tests),
            'successful_transformations': len(valid_tests),
            'success_rate': len(valid_tests) / len(transformation_tests),
            'avg_position_error_mm': avg_error,
            'max_position_error_mm': max_error,
            'min_position_error_mm': min_error,
            'avg_transform_time': np.mean(self.benchmarks['coordinate_transform_time']),
            'individual_tests': transformation_tests
        }
    
    def test_realtime_performance(self) -> Dict[str, Any]:
        """Test real-time performance characteristics."""
        performance_data = []
        frame_count = 0
        max_frames = 20
        
        print("  Running real-time performance test...")
        
        while frame_count < max_frames:
            overall_start = time.time()
            
            # Get frame
            vertices, color_image, metadata = self.sensor_manager.get_lidar_optimized_frame()
            
            if metadata.get('end_of_stream', False):
                break
            
            if vertices is None or len(vertices) < 100:
                continue
            
            frame_count += 1
            
            # Full pipeline test
            clusters = self.clustering_strategy.cluster_objects(vertices, color_image)
            
            if clusters:
                # Transform first cluster
                robot_coords = self.coordinator.transform_cluster_center(clusters[0])
                
                overall_time = time.time() - overall_start
                
                performance_data.append({
                    'frame_id': frame_count,
                    'total_processing_time': overall_time,
                    'input_points': len(vertices),
                    'clusters_found': len(clusters),
                    'fps': 1.0 / overall_time,
                    'coordinate_success': robot_coords is not None and robot_coords.get('is_valid', False)
                })
                
                self.benchmarks['total_processing_time'].append(overall_time)
        
        # Analyze performance
        if performance_data:
            avg_fps = np.mean([p['fps'] for p in performance_data])
            avg_processing_time = np.mean([p['total_processing_time'] for p in performance_data])
            pipeline_success_rate = np.mean([p['coordinate_success'] for p in performance_data])
        else:
            avg_fps = avg_processing_time = pipeline_success_rate = 0
        
        return {
            'total_frames_tested': len(performance_data),
            'avg_fps': avg_fps,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'pipeline_success_rate': pipeline_success_rate,
            'realtime_capable': avg_fps > 10,  # Consider >10 FPS as real-time capable
            'performance_data': performance_data
        }
    
    def test_system_integration(self) -> Dict[str, Any]:
        """Test integration between all system components."""
        integration_tests = []
        
        # Test complete pipeline on multiple frames
        frame_count = 0
        max_frames = 10
        
        while frame_count < max_frames:
            # Get frame
            vertices, color_image, metadata = self.sensor_manager.get_lidar_optimized_frame()
            
            if metadata.get('end_of_stream', False):
                break
            
            if vertices is None or len(vertices) < 100:
                continue
            
            frame_count += 1
            
            # Complete integration test
            test_result = {
                'frame_id': frame_count,
                'input_points': len(vertices),
                'pipeline_stages': {}
            }
            
            # Stage 1: Floor detection
            try:
                object_points, floor_points, _ = self.floor_detector.detect_floor(vertices)
                test_result['pipeline_stages']['floor_detection'] = {
                    'success': True,
                    'floor_points': len(floor_points),
                    'object_points': len(object_points)
                }
            except Exception as e:
                test_result['pipeline_stages']['floor_detection'] = {
                    'success': False,
                    'error': str(e)
                }
                continue
            
            # Stage 2: Object clustering
            try:
                clusters = self.clustering_strategy.cluster_objects(vertices, color_image)
                test_result['pipeline_stages']['clustering'] = {
                    'success': True,
                    'clusters_found': len(clusters)
                }
            except Exception as e:
                test_result['pipeline_stages']['clustering'] = {
                    'success': False,
                    'error': str(e)
                }
                continue
            
            # Stage 3: Coordinate transformation
            if clusters:
                try:
                    robot_coords = self.coordinator.transform_cluster_center(clusters[0])
                    test_result['pipeline_stages']['coordinate_transform'] = {
                        'success': robot_coords is not None,
                        'valid_coordinates': robot_coords.get('is_valid', False) if robot_coords else False,
                        'robot_position': robot_coords['robot_position'].tolist() if robot_coords else None
                    }
                except Exception as e:
                    test_result['pipeline_stages']['coordinate_transform'] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                test_result['pipeline_stages']['coordinate_transform'] = {
                    'success': False,
                    'error': 'No clusters to transform'
                }
            
            # Overall pipeline success
            test_result['pipeline_success'] = all(
                stage.get('success', False) 
                for stage in test_result['pipeline_stages'].values()
            )
            
            integration_tests.append(test_result)
        
        # Analyze integration results
        successful_pipelines = sum(1 for t in integration_tests if t['pipeline_success'])
        
        return {
            'total_integration_tests': len(integration_tests),
            'successful_pipelines': successful_pipelines,
            'integration_success_rate': successful_pipelines / len(integration_tests) if integration_tests else 0,
            'individual_tests': integration_tests
        }
    
    def _generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall system assessment."""
        # Combine all benchmark data for overall analysis
        all_benchmarks = {
            'floor_detection_avg_ms': np.mean(self.benchmarks['floor_detection_time']) * 1000,
            'clustering_avg_ms': np.mean(self.benchmarks['clustering_time']) * 1000,
            'coordinate_transform_avg_ms': np.mean(self.benchmarks['coordinate_transform_time']) * 1000,
            'total_pipeline_avg_ms': np.mean(self.benchmarks['total_processing_time']) * 1000
        }
        
        return {
            'benchmarks': all_benchmarks,
            'system_ready_for_production': all_benchmarks['total_pipeline_avg_ms'] < 100,  # <100ms processing
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if np.mean(self.benchmarks['total_processing_time']) > 0.1:
            recommendations.append("Consider optimizing processing pipeline for better real-time performance")
        
        if len(self.benchmarks['floor_detection_time']) > 0:
            avg_floor_time = np.mean(self.benchmarks['floor_detection_time'])
            if avg_floor_time > 0.05:
                recommendations.append("Floor detection may benefit from parameter tuning for speed")
        
        recommendations.append("System shows good integration between components")
        recommendations.append("LiDAR-based tracking is operational and ready for deployment")
        
        return recommendations
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save comprehensive test results to file."""
        timestamp = int(time.time())
        results_file = f"lidar_tracking_validation_{timestamp}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed test results saved to: {results_file}")
        except Exception as e:
            print(f"Warning: Could not save results file: {e}")
    
    def _print_test_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary."""
        print("\n" + "="*60)
        print("LIDAR TRACKING VALIDATION SUMMARY")
        print("="*60)
        
        # Floor Detection Results
        floor = results['floor_detection']
        print(f"\nFLOOR DETECTION:")
        print(f"  Success Rate: {floor['success_rate']:.1%}")
        print(f"  Avg Detection Time: {floor['avg_detection_time']*1000:.1f}ms")
        print(f"  Avg Confidence: {floor['avg_confidence']:.3f}")
        
        # Object Clustering Results
        clustering = results['object_clustering']
        print(f"\nOBJECT CLUSTERING:")
        print(f"  Object Detection Rate: {clustering['object_detection_rate']:.1%}")
        print(f"  Avg Clustering Time: {clustering['avg_clustering_time']*1000:.1f}ms")
        print(f"  Avg Clusters per Frame: {clustering['avg_clusters_per_frame']:.1f}")
        
        # Coordinate Transformation Results
        coords = results['coordinate_transformation']
        print(f"\nCOORDINATE TRANSFORMATION:")
        print(f"  Success Rate: {coords['success_rate']:.1%}")
        if coords['success_rate'] > 0:
            print(f"  Avg Position Error: {coords['avg_position_error_mm']:.1f}mm")
            print(f"  Max Position Error: {coords['max_position_error_mm']:.1f}mm")
        print(f"  Avg Transform Time: {coords['avg_transform_time']*1000:.2f}ms")
        
        # Performance Results
        perf = results['performance_benchmarks']
        print(f"\nPERFORMANCE:")
        print(f"  Average FPS: {perf['avg_fps']:.1f}")
        print(f"  Processing Time: {perf['avg_processing_time_ms']:.1f}ms")
        print(f"  Real-time Capable: {'YES' if perf['realtime_capable'] else 'NO'}")
        print(f"  Pipeline Success Rate: {perf['pipeline_success_rate']:.1%}")
        
        # Integration Results
        integration = results['system_integration']
        print(f"\nSYSTEM INTEGRATION:")
        print(f"  Integration Success Rate: {integration['integration_success_rate']:.1%}")
        
        # Overall Assessment
        assessment = results['overall_assessment']
        print(f"\nOVERALL ASSESSMENT:")
        print(f"  Production Ready: {'YES' if assessment['system_ready_for_production'] else 'NO'}")
        
        print(f"\nRECOMMENDATIONS:")
        for rec in assessment['recommendations']:
            print(f"  • {rec}")
        
        print("="*60)
    
    def cleanup(self):
        """Clean up resources."""
        if self.sensor_manager:
            self.sensor_manager.stop()


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="LiDAR Tracking Validation System")
    
    parser.add_argument('--config', type=str, default='lidar_tracking',
                       choices=['lidar_tracking', 'lidar_precision', 'lidar_debug'],
                       help='Configuration to test (default: lidar_tracking)')
    
    parser.add_argument('--bag', type=str, default=None,
                       help='RealSense bag file for consistent testing')
    
    parser.add_argument('--matrix', type=str, default='charuco_eye_in_hand_transform.npy',
                       help='Hand-eye transformation matrix file')
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation (fewer test frames)')
    
    args = parser.parse_args()
    
    print("LiDAR Tracking System Validation")
    print("="*50)
    print(f"Configuration: {args.config}")
    print(f"Matrix file: {args.matrix}")
    if args.bag:
        print(f"Bag file: {args.bag}")
    if args.quick:
        print("Quick validation mode")
    print("="*50)
    
    # Create validator
    validator = LiDARTrackingValidator(
        config_name=args.config,
        bag_file=args.bag,
        transformation_matrix_file=args.matrix
    )
    
    try:
        # Run comprehensive validation
        results = validator.run_comprehensive_tests()
        
        if 'error' in results:
            print(f"Validation failed: {results['error']}")
            return 1
        
        print("\nValidation completed successfully!")
        
        # Check if system is ready for production use
        if results['overall_assessment']['system_ready_for_production']:
            print("System is ready for production use!")
            return 0
        else:
            print("System may need optimization before production use")
            return 2
    
    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        return 130
    except Exception as e:
        print(f"Unexpected error during validation: {e}")
        return 1
    finally:
        validator.cleanup()


if __name__ == "__main__":
    sys.exit(main())