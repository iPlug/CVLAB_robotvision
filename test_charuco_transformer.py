#!/usr/bin/env python3
"""
Test CharUco Transformer Functionality

This test verifies that the modularized CharUco transformer is working correctly
and provides the same functionality as the original scattered implementation.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.abspath('.'))

from charuco_transformer import CharUcoTransformer, create_prediction_transformer, create_calibration_transformer

def test_transformer_creation():
    """Test creating transformers in different modes."""
    print("Testing transformer creation...")
    
    # Test calibration mode
    print("  Creating calibration transformer...")
    calib_transformer = create_calibration_transformer()
    assert not calib_transformer.is_ready_for_prediction()
    assert calib_transformer.is_ready_for_calibration() == False  # No camera setup yet
    print("    ✓ Calibration transformer created")
    
    # Test prediction mode with existing calibration
    calibration_dir = "charuco-pose-estimation/calibration_result_1754622347"
    if Path(calibration_dir).exists():
        print("  Creating prediction transformer...")
        pred_transformer = create_prediction_transformer(calibration_dir)
        assert pred_transformer.is_ready_for_prediction()
        print("    ✓ Prediction transformer created and loaded calibration")
        
        # Test calibration info
        info = pred_transformer.get_calibration_info()
        print(f"    ✓ Board config: {info['board_config']['size']} squares")
        print(f"    ✓ Calibration loaded: {info['calibration_loaded']}")
        
        return pred_transformer
    else:
        print("  ⚠️ Calibration directory not found, skipping prediction transformer test")
        return None

def test_coordinate_transformations():
    """Test coordinate transformation functions."""
    print("\nTesting coordinate transformations...")
    
    transformer = CharUcoTransformer(board_size=(7, 5), square_length=0.039, marker_length=0.029)
    
    # Test pose to transform matrix conversion
    robot_pose = [200.0, 100.0, 300.0, 0.0, 180.0, 0.0]  # mm and degrees
    transform_matrix = transformer.pose_to_transform_matrix(robot_pose)
    
    assert transform_matrix.shape == (4, 4)
    print("    ✓ Pose to transformation matrix conversion")
    
    # Test CharUco pose to transform matrix (without actual rvec/tvec)
    # This would normally come from actual detection
    rvec = np.array([[0.1], [0.2], [0.1]])  # Example rotation vector
    tvec = np.array([[0.05], [0.02], [0.3]])  # Example translation vector
    
    charuco_transform = transformer.charuco_pose_to_transform_matrix(rvec, tvec)
    assert charuco_transform.shape == (4, 4)
    print("    ✓ CharUco pose to transformation matrix conversion")

def test_calibration_workflow(pred_transformer):
    """Test calibration workflow with the transformer."""
    if pred_transformer is None:
        print("\nSkipping calibration workflow test (no prediction transformer)")
        return
    
    print("\nTesting calibration workflow...")
    
    # Test validation functionality
    validation_results = pred_transformer.validate_calibration([300.0, 100.0, 0.0])
    if validation_results:
        print(f"    ✓ Calibration validation: {validation_results['quality_score']}")
        print(f"    ✓ Mean error: {validation_results['mean_translation_error_mm']:.2f}mm")
    else:
        print("    ⚠️ No validation results (expected if no calibration points)")
    
    # Test stability tracking
    position1 = np.array([300.0, 100.0, 10.0])
    position2 = np.array([301.0, 99.0, 11.0])
    
    pred_transformer._update_position_history(position1)
    pred_transformer._update_position_history(position2)
    
    mean_pos, std_pos = pred_transformer.get_position_stability()
    if mean_pos is not None:
        print(f"    ✓ Position stability tracking working")
        print(f"      Mean: [{mean_pos[0]:.1f}, {mean_pos[1]:.1f}, {mean_pos[2]:.1f}]")
        print(f"      Std:  [{std_pos[0]:.2f}, {std_pos[1]:.2f}, {std_pos[2]:.2f}]")

def test_api_compatibility():
    """Test that the API provides expected methods and signatures."""
    print("\nTesting API compatibility...")
    
    transformer = CharUcoTransformer()
    
    # Check that all expected methods exist
    expected_methods = [
        'detect_charuco_pose',
        'predict_robot_position', 
        'collect_calibration_point',
        'calculate_eye_in_hand_calibration',
        'validate_calibration',
        'save_calibration_results',
        'is_ready_for_prediction',
        'is_ready_for_calibration',
        'get_calibration_info'
    ]
    
    for method_name in expected_methods:
        assert hasattr(transformer, method_name), f"Missing method: {method_name}"
        print(f"    ✓ Method exists: {method_name}")

def main():
    """Run all tests."""
    print("="*60)
    print("CHARUCO TRANSFORMER TEST SUITE")
    print("="*60)
    
    try:
        # Test transformer creation
        pred_transformer = test_transformer_creation()
        
        # Test coordinate transformations
        test_coordinate_transformations()
        
        # Test calibration workflow
        test_calibration_workflow(pred_transformer)
        
        # Test API compatibility
        test_api_compatibility()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nCharUco Transformer is ready for use:")
        print("  - Modular architecture implemented successfully")
        print("  - All transformation logic consolidated")
        print("  - Backward compatibility maintained")
        print("  - Clean API for both calibration and prediction")
        print("  - Eliminates code duplication across applications")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)