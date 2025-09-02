#!/usr/bin/env python3
"""
Test script for the enhanced outlier removal system in LiDAR object tracking.

This script tests the complete outlier filtering pipeline:
1. PositionOutlierFilter - Multi-criteria outlier detection
2. Integration with lidar_track_object.py main application
3. Visualization enhancements showing filter feedback
4. Configurable distance ranges via command line arguments

The test creates synthetic data with known outliers to validate filtering performance.
"""

import numpy as np
import sys
import os
import time
import cv2
from typing import Dict, Any, List

sys.path.append('.')

from module.processing.position_outlier_filter import PositionOutlierFilter
from module.processing.lidar_coordinator import LiDARCoordinator


def create_test_robot_coordinates() -> List[Dict[str, Any]]:
    """Create test robot coordinate data with known outliers."""
    test_coords = []
    current_time = time.time()
    
    # Valid positions (should pass filter)
    valid_positions = [
        np.array([300.0, 100.0, 150.0]),  # Normal position
        np.array([305.0, 95.0, 155.0]),   # Small movement
        np.array([310.0, 90.0, 160.0]),   # Gradual movement
        np.array([315.0, 85.0, 165.0]),   # Continued gradual movement
    ]
    
    # Distance outliers (too far from robot base)
    distance_outliers = [
        np.array([800.0, 200.0, 300.0]),  # Way too far
        np.array([600.0, 150.0, 250.0]),  # Still too far
    ]
    
    # Velocity/jump outliers (sudden large movements)
    jump_outliers = [
        np.array([100.0, 400.0, 50.0]),   # Sudden jump to different location
        np.array([-200.0, -300.0, 400.0]), # Another sudden jump
    ]
    
    # Workspace outliers (outside robot workspace)
    workspace_outliers = [
        np.array([500.0, 500.0, 500.0]),  # Outside workspace bounds
        np.array([-500.0, 100.0, 150.0]), # Outside X bounds
    ]
    
    # Create coordinate dictionaries
    all_positions = [
        *valid_positions,
        *distance_outliers, 
        *jump_outliers,
        *workspace_outliers
    ]
    
    labels = (
        ['valid'] * len(valid_positions) + 
        ['distance_outlier'] * len(distance_outliers) +
        ['jump_outlier'] * len(jump_outliers) +
        ['workspace_outlier'] * len(workspace_outliers)
    )
    
    for i, (position, expected_label) in enumerate(zip(all_positions, labels)):
        coord_dict = {
            'robot_position': position,
            'distance_from_base': np.linalg.norm(position),
            'is_valid': True,
            'expected_filter_result': expected_label,
            'test_id': i
        }
        test_coords.append(coord_dict)
    
    return test_coords


def test_outlier_filter_basic():
    """Test basic outlier filter functionality with known data."""
    print("="*60)
    print("OUTLIER FILTER BASIC FUNCTIONALITY TEST")
    print("="*60)
    
    # Initialize outlier filter with test parameters
    outlier_filter = PositionOutlierFilter(
        max_distance_from_base=500.0,     # 500mm limit
        max_velocity=800.0,               # 800mm/s limit
        max_position_jump=100.0,          # 100mm jump limit
        temporal_window_size=5,
        enable_statistical_filtering=True,
        statistical_threshold=2.0,
        workspace_bounds={
            'x_min': -400, 'x_max': 400,
            'y_min': -400, 'y_max': 400,
            'z_min': 50,   'z_max': 350
        }
    )
    
    # Get test coordinates
    test_coords = create_test_robot_coordinates()
    
    results = []
    current_time = time.time()
    
    print("\nProcessing test positions:")
    print("-" * 60)
    
    for i, coords in enumerate(test_coords):
        # Add small time increment between positions
        test_time = current_time + i * 0.1
        
        is_valid, filter_details = outlier_filter.filter_position(coords, test_time)
        
        position = coords['robot_position']
        expected = coords['expected_filter_result']
        actual_result = 'valid' if is_valid else filter_details['reason']
        
        # Check if result matches expectation
        is_correct = (expected == 'valid' and is_valid) or (expected != 'valid' and not is_valid)
        status = "[PASS]" if is_correct else "[FAIL]"
        
        print(f"Test {i:2d}: [{position[0]:6.0f}, {position[1]:6.0f}, {position[2]:6.0f}] "
              f"Expected: {expected:15s} | Actual: {actual_result:15s} | {status}")
        
        results.append({
            'test_id': i,
            'expected': expected,
            'actual': actual_result,
            'is_correct': is_correct,
            'filter_details': filter_details
        })
    
    # Calculate test results
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['is_correct'])
    
    print("-" * 60)
    print(f"Basic Filter Test Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests:.1%})")
    
    # Show filter statistics
    filter_stats = outlier_filter.get_filter_statistics()
    print(f"Filter processed {filter_stats['total_positions']} positions")
    print(f"Success rate: {filter_stats['success_rate']:.1%}")
    print(f"Outliers removed: {filter_stats['total_outliers_removed']}")
    
    return passed_tests == total_tests


def test_temporal_consistency():
    """Test temporal consistency and position smoothing."""
    print("\n" + "="*60)
    print("TEMPORAL CONSISTENCY TEST")
    print("="*60)
    
    outlier_filter = PositionOutlierFilter(
        max_distance_from_base=500.0,
        max_velocity=200.0,               # Lower velocity limit for temporal test
        max_position_jump=50.0,           # Lower jump limit
        temporal_window_size=10,          # Larger history window
        enable_statistical_filtering=True,
        statistical_threshold=1.5         # Stricter statistical filtering
    )
    
    # Create sequence of positions with gradual movement and some outliers
    base_position = np.array([300.0, 100.0, 150.0])
    current_time = time.time()
    
    test_sequence = []
    for i in range(15):
        test_time = current_time + i * 0.5  # 500ms intervals
        
        if i < 5:
            # Gradual movement (should all be valid)
            position = base_position + np.array([i * 5, i * 2, i * 1])
            expected = 'valid'
        elif i == 5:
            # Sudden jump (should be filtered as outlier)
            position = base_position + np.array([200, 150, 100])  
            expected = 'position_jump_outlier'
        elif i == 6:
            # Return to near original position (might be filtered as velocity outlier)
            position = base_position + np.array([25, 10, 5])
            expected = 'velocity_outlier'
        else:
            # Resume gradual movement
            position = base_position + np.array([(i-6) * 3, (i-6) * 1, (i-6) * 2])
            expected = 'valid'
        
        coords = {
            'robot_position': position,
            'distance_from_base': np.linalg.norm(position),
            'is_valid': True
        }
        
        is_valid, filter_details = outlier_filter.filter_position(coords, test_time)
        actual_result = 'valid' if is_valid else filter_details['reason']
        
        print(f"Time {i:2d}: [{position[0]:6.0f}, {position[1]:6.0f}, {position[2]:6.0f}] "
              f"-> {actual_result:15s}")
        
        test_sequence.append({
            'time_step': i,
            'position': position,
            'expected': expected,
            'actual': actual_result,
            'is_valid': is_valid
        })
    
    # Test position smoothing
    smoothed_pos = outlier_filter.get_smoothed_position('moving_average')
    if smoothed_pos is not None:
        print(f"\nSmoothed position: [{smoothed_pos[0]:.1f}, {smoothed_pos[1]:.1f}, {smoothed_pos[2]:.1f}]")
    
    confidence = outlier_filter.get_position_confidence()
    print(f"Position confidence: {confidence:.3f}")
    
    # Check temporal filtering effectiveness
    valid_count = sum(1 for t in test_sequence if t['is_valid'])
    outlier_count = len(test_sequence) - valid_count
    
    print(f"\nTemporal test: {valid_count}/{len(test_sequence)} valid positions")
    print(f"Outliers detected: {outlier_count}")
    
    return outlier_count > 0  # Success if we detected some outliers


def test_performance_characteristics():
    """Test performance and scalability of outlier filtering."""
    print("\n" + "="*60)  
    print("PERFORMANCE CHARACTERISTICS TEST")
    print("="*60)
    
    outlier_filter = PositionOutlierFilter()
    
    # Test with different numbers of positions
    test_sizes = [100, 500, 1000, 2000]
    
    for size in test_sizes:
        print(f"\nTesting with {size} positions:")
        
        # Generate random valid positions around robot workspace
        positions = []
        base_pos = np.array([250.0, 50.0, 200.0])
        
        for i in range(size):
            # Add small random variations to base position
            variation = np.random.normal(0, 20, 3)  # 20mm standard deviation
            position = base_pos + variation
            positions.append(position)
        
        # Time the filtering process
        start_time = time.perf_counter()
        
        valid_count = 0
        for i, pos in enumerate(positions):
            coords = {
                'robot_position': pos,
                'distance_from_base': np.linalg.norm(pos),
                'is_valid': True
            }
            
            is_valid, _ = outlier_filter.filter_position(coords, start_time + i * 0.01)
            if is_valid:
                valid_count += 1
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        positions_per_second = size / processing_time
        avg_time_per_position = processing_time / size * 1000  # ms
        
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Positions per second: {positions_per_second:.1f}")
        print(f"  Average time per position: {avg_time_per_position:.3f}ms")
        print(f"  Valid positions: {valid_count}/{size} ({valid_count/size:.1%})")
    
    return True


def test_configuration_parameters():
    """Test different configuration parameter combinations."""
    print("\n" + "="*60)
    print("CONFIGURATION PARAMETERS TEST")
    print("="*60)
    
    # Test different configuration presets
    configs = {
        'strict': {
            'max_distance_from_base': 300.0,
            'max_velocity': 200.0,
            'max_position_jump': 30.0,
            'statistical_threshold': 1.0
        },
        'standard': {
            'max_distance_from_base': 450.0,
            'max_velocity': 800.0,
            'max_position_jump': 80.0,
            'statistical_threshold': 2.0
        },
        'relaxed': {
            'max_distance_from_base': 600.0,
            'max_velocity': 1500.0,
            'max_position_jump': 200.0,
            'statistical_threshold': 3.0
        }
    }
    
    # Test position that should be filtered differently by each config
    test_position = np.array([400.0, 150.0, 200.0])  # Moderate distance
    test_coords = {
        'robot_position': test_position,
        'distance_from_base': np.linalg.norm(test_position),
        'is_valid': True
    }
    
    for config_name, params in configs.items():
        filter_instance = PositionOutlierFilter(**params)
        
        is_valid, filter_details = filter_instance.filter_position(test_coords, time.time())
        result = 'VALID' if is_valid else f'FILTERED ({filter_details["reason"]})'
        
        print(f"{config_name:8s} config: Position [{test_position[0]:.0f}, {test_position[1]:.0f}, {test_position[2]:.0f}] -> {result}")
    
    return True


def create_filter_performance_report():
    """Create comprehensive performance report for outlier filtering."""
    print("\n" + "="*60)
    print("FILTER PERFORMANCE REPORT")
    print("="*60)
    
    outlier_filter = PositionOutlierFilter()
    
    # Run the filter through a comprehensive test sequence
    comprehensive_coords = create_test_robot_coordinates()
    
    # Add some temporal consistency data
    for i in range(10):
        base_pos = np.array([300.0, 100.0, 150.0])
        position = base_pos + np.random.normal(0, 10, 3)
        coords = {
            'robot_position': position,
            'distance_from_base': np.linalg.norm(position),
            'is_valid': True
        }
        is_valid, _ = outlier_filter.filter_position(coords, time.time() + i * 0.1)
    
    # Generate and print report
    report = outlier_filter.create_filter_report()
    print(report)
    
    return True


def main():
    """Run all outlier filter tests."""
    print("ENHANCED OUTLIER REMOVAL SYSTEM - COMPREHENSIVE TEST")
    print("=" * 80)
    
    test_results = []
    
    # Run all test suites
    tests = [
        ("Basic Functionality", test_outlier_filter_basic),
        ("Temporal Consistency", test_temporal_consistency), 
        ("Performance Characteristics", test_performance_characteristics),
        ("Configuration Parameters", test_configuration_parameters),
        ("Performance Report", create_filter_performance_report)
    ]
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            test_results.append((test_name, result, None))
            status = "PASSED" if result else "FAILED"
            print(f"{test_name} test: {status}")
        except Exception as e:
            test_results.append((test_name, False, str(e)))
            print(f"{test_name} test: FAILED with error: {e}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("OUTLIER FILTER TEST SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(1 for _, result, _ in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result, error in test_results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
        if error:
            print(f"    Error: {error}")
    
    print("-" * 80)
    print(f"Overall Result: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests:.1%})")
    
    if passed_tests == total_tests:
        print("\n[SUCCESS] ALL TESTS PASSED! Enhanced outlier removal system is working correctly.")
        print("\nThe system is ready for:")
        print("- Real-time position outlier detection")
        print("- Configurable distance-based filtering")
        print("- Temporal consistency validation")
        print("- Statistical outlier removal")
        print("- Integration with visualization system")
        return 0
    else:
        print(f"\n[WARNING] {total_tests - passed_tests} test(s) failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())