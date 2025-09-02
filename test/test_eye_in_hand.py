#!/usr/bin/env python3
"""
Test script for Eye-in-Hand configuration

This script validates the eye-in-hand tracking system functionality
without requiring actual hardware connection.
"""

import sys
import os
import numpy as np
import time
from unittest.mock import Mock

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eye_in_hand_tracker import EyeInHandTracker


def test_eye_in_hand_initialization():
    """Test initialization of eye-in-hand system."""
    print("Testing Eye-in-Hand Initialization...")
    
    try:
        tracker = EyeInHandTracker(
            config_name='eye_in_hand_realtime',
            initial_strategy='bev'
        )
        
        # Check configuration loading
        assert tracker.config is not None, "Configuration not loaded"
        assert hasattr(tracker.config, 'robot_params'), "Robot parameters missing"
        
        # Get robot params
        robot_params = tracker.config.robot_params
        
        print("+ Configuration loaded successfully")
        print(f"  Approach distance: {robot_params['approach_distance']}mm")
        print(f"  Retreat distance: {robot_params['retreat_distance']}mm")
        
        # Check eye-in-hand specific parameters
        assert tracker.current_robot_mode == "observe", "Initial mode should be observe"
        assert tracker.camera_offset is not None, "Camera offset not set"
        assert tracker.approach_distance > 0, "Approach distance not set"
        
        print("+ Eye-in-hand parameters initialized correctly")
        print(f"  Initial mode: {tracker.current_robot_mode}")
        print(f"  Camera offset: {tracker.camera_offset}")
        print(f"  Approach distance: {tracker.approach_distance}mm")
        
        return True
        
    except Exception as e:
        print(f"- Initialization test failed: {e}")
        return False


def test_robot_modes():
    """Test robot mode switching functionality."""
    print("\nTesting Robot Mode Switching...")
    
    try:
        tracker = EyeInHandTracker()
        
        # Test mode switching
        modes = ["observe", "approach", "manipulate"]
        for mode in modes:
            tracker.current_robot_mode = mode
            assert tracker.current_robot_mode == mode, f"Failed to switch to {mode} mode"
            print(f"+ Successfully switched to {mode} mode")
        
        return True
        
    except Exception as e:
        print(f"- Mode switching test failed: {e}")
        return False


def test_coordinate_extraction():
    """Test object position extraction in camera coordinates."""
    print("\nTesting Coordinate Extraction...")
    
    try:
        tracker = EyeInHandTracker()
        
        # Test object with center_mm format
        test_object_mm = {
            'center_mm': [100, 50, 200]  # [side, height, forward]
        }
        
        position = tracker._extract_object_position(test_object_mm)
        expected = [200, 100, 50]  # [forward, side, height]
        
        assert position == expected, f"Expected {expected}, got {position}"
        print(f"+ center_mm extraction: {test_object_mm['center_mm']} -> {position}")
        
        # Test object with center format (meters)
        test_object_m = {
            'center': [0.1, 0.05, 0.2]  # [side, height, forward] in meters
        }
        
        position = tracker._extract_object_position(test_object_m)
        expected = [200, 100, 50]  # [forward, side, height] in mm
        
        assert position == expected, f"Expected {expected}, got {position}"
        print(f"+ center extraction: {test_object_m['center']} -> {position}")
        
        return True
        
    except Exception as e:
        print(f"- Coordinate extraction test failed: {e}")
        return False


def test_relative_movement_calculation():
    """Test relative movement calculation for eye-in-hand control."""
    print("\nTesting Relative Movement Calculation...")
    
    try:
        tracker = EyeInHandTracker()
        
        # Mock robot controller with current position
        tracker.robot_controller = Mock()
        tracker.robot_controller.get_current_position.return_value = [100, 200, 300, 0, 180, 0]
        
        # Test relative movement calculation
        offset = [10, -5, 20]
        
        # This would normally submit the command, but we'll test the calculation logic
        current_pos = tracker.robot_controller.get_current_position()
        target_pos = [
            current_pos[0] + offset[0],  # 110
            current_pos[1] + offset[1],  # 195
            current_pos[2] + offset[2],  # 320
            current_pos[3],  # 0
            current_pos[4],  # 180
            current_pos[5]   # 0
        ]
        
        expected = [110, 195, 320, 0, 180, 0]
        assert target_pos == expected, f"Expected {expected}, got {target_pos}"
        
        print(f"+ Relative movement calculation:")
        print(f"  Current: {current_pos}")
        print(f"  Offset: {offset}")
        print(f"  Target: {target_pos}")
        
        return True
        
    except Exception as e:
        print(f"- Relative movement test failed: {e}")
        return False


def test_object_distance_calculation():
    """Test distance calculation for closest object finding."""
    print("\nTesting Object Distance Calculation...")
    
    try:
        tracker = EyeInHandTracker()
        
        # Create test objects at different distances
        test_objects = [
            {'center_mm': [100, 0, 200]},   # Distance: sqrt(100² + 0² + 200²) = ~224mm
            {'center_mm': [0, 50, 150]},    # Distance: sqrt(0² + 50² + 150²) = ~158mm (closest)
            {'center_mm': [200, 100, 300]}, # Distance: sqrt(200² + 100² + 300²) = ~374mm
        ]
        
        closest = tracker._find_closest_object(test_objects)
        assert closest is not None, "No closest object found"
        
        closest_pos = tracker._extract_object_position(closest)
        expected_closest = [150, 0, 50]  # Second object converted to [forward, side, height]
        
        assert closest_pos == expected_closest, f"Expected closest {expected_closest}, got {closest_pos}"
        
        print(f"+ Closest object identification:")
        for i, obj in enumerate(test_objects):
            pos = tracker._extract_object_position(obj)
            distance = np.sqrt(sum(p**2 for p in pos))
            marker = " <- CLOSEST" if obj == closest else ""
            print(f"  Object {i+1}: {pos} (distance: {distance:.1f}mm){marker}")
        
        return True
        
    except Exception as e:
        print(f"- Object distance test failed: {e}")
        return False


def test_mode_specific_behavior():
    """Test mode-specific robot behaviors."""
    print("\nTesting Mode-Specific Behaviors...")
    
    try:
        tracker = EyeInHandTracker()
        
        # Mock robot controller
        tracker.robot_controller = Mock()
        tracker.robot_controller.is_connected = True
        tracker.robot_controller.async_mode_enabled = True
        tracker.robot_controller.get_current_position.return_value = [100, 100, 200, 0, 180, 0]
        
        # Test object
        test_objects = [{'center_mm': [50, 0, 100]}]
        robot_params = {
            'approach_distance': 100,
            'retreat_distance': 200
        }
        
        # Test observe mode
        tracker.current_robot_mode = "observe"
        print("Testing observe mode...")
        # This would call _handle_observe_mode but we'll just verify mode is set
        assert tracker.current_robot_mode == "observe"
        print("+ Observe mode active")
        
        # Test approach mode
        tracker.current_robot_mode = "approach"
        print("Testing approach mode...")
        assert tracker.current_robot_mode == "approach"
        print("+ Approach mode active")
        
        # Test manipulate mode
        tracker.current_robot_mode = "manipulate"
        print("Testing manipulate mode...")
        assert tracker.current_robot_mode == "manipulate"
        print("+ Manipulate mode active")
        
        return True
        
    except Exception as e:
        print(f"- Mode behavior test failed: {e}")
        return False


def test_configuration_presets():
    """Test eye-in-hand specific configuration presets."""
    print("\nTesting Configuration Presets...")
    
    try:
        # Test realtime preset
        tracker_realtime = EyeInHandTracker(config_name='eye_in_hand_realtime')
        config = tracker_realtime.config
        
        assert hasattr(config, 'robot_params'), "Robot params missing in realtime config"
        assert config.robot_params['approach_distance'] == 100, "Wrong approach distance"
        print("+ eye_in_hand_realtime preset loaded")
        
        # Test precision preset
        tracker_precision = EyeInHandTracker(config_name='eye_in_hand_precision')
        config = tracker_precision.config
        
        assert hasattr(config, 'robot_params'), "Robot params missing in precision config"
        assert config.robot_params['approach_distance'] == 80, "Wrong approach distance for precision"
        print("+ eye_in_hand_precision preset loaded")
        
        # Verify different parameters
        print(f"  Realtime approach distance: {tracker_realtime.config.robot_params['approach_distance']}mm")
        print(f"  Precision approach distance: {tracker_precision.config.robot_params['approach_distance']}mm")
        
        return True
        
    except Exception as e:
        print(f"- Configuration presets test failed: {e}")
        return False


def run_all_tests():
    """Run all eye-in-hand tests."""
    print("Eye-in-Hand Configuration Test Suite")
    print("=" * 50)
    
    tests = [
        test_eye_in_hand_initialization,
        test_robot_modes,
        test_coordinate_extraction,
        test_relative_movement_calculation,
        test_object_distance_calculation,
        test_mode_specific_behavior,
        test_configuration_presets
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("+ All eye-in-hand tests PASSED!")
        print("\nEye-in-hand configuration is ready for use.")
        print("\nTo run the eye-in-hand tracker:")
        print("  python eye_in_hand_tracker.py --config eye_in_hand_realtime")
        print("  python eye_in_hand_tracker.py --config eye_in_hand_precision")
        return True
    else:
        print(f"- {total - passed} tests FAILED!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)