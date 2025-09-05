#!/usr/bin/env python3
"""
Calibration Verification Script

Verifies that the calibrated transformation matrix and related files
are correctly copied and accessible for the LLM grabber system.
"""

import os
import sys
import numpy as np
import json
from pathlib import Path

def verify_calibration_files():
    """Verify all calibration files are present and valid."""
    
    print("🔍 Verifying Calibration Files")
    print("="*50)
    
    # Expected calibration directory (as referenced in camera_utils.py)
    cal_dir = Path("calibration_result_1754622347")
    
    if not cal_dir.exists():
        print("❌ Calibration directory not found!")
        return False
    
    print(f"✓ Calibration directory found: {cal_dir}")
    
    # Check essential files
    essential_files = [
        "eye_in_hand_transform.npy",
        "camera_intrinsics.txt", 
        "calibration_data.json",
        "README.txt"
    ]
    
    missing_files = []
    for file_name in essential_files:
        file_path = cal_dir / file_name
        if file_path.exists():
            print(f"✓ {file_name}")
        else:
            print(f"❌ {file_name} - MISSING")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    return True

def verify_transformation_matrix():
    """Verify the transformation matrix can be loaded and is valid."""
    
    print("\n🎯 Verifying Transformation Matrix")
    print("="*50)
    
    try:
        # Load the transformation matrix
        matrix_path = Path("calibration_result_1754622347/eye_in_hand_transform.npy")
        transform_matrix = np.load(matrix_path)
        
        print(f"✓ Matrix loaded successfully from: {matrix_path}")
        print(f"✓ Matrix shape: {transform_matrix.shape}")
        
        # Verify it's a 4x4 homogeneous transformation matrix
        if transform_matrix.shape != (4, 4):
            print(f"❌ Invalid matrix shape: {transform_matrix.shape} (expected: (4, 4))")
            return False
        
        # Check if bottom row is [0, 0, 0, 1] (homogeneous transform property)
        expected_bottom_row = np.array([0, 0, 0, 1])
        actual_bottom_row = transform_matrix[3, :]
        
        if not np.allclose(actual_bottom_row, expected_bottom_row, atol=1e-10):
            print(f"❌ Invalid bottom row: {actual_bottom_row} (expected: {expected_bottom_row})")
            return False
        
        print("✓ Valid homogeneous transformation matrix")
        
        # Display the matrix
        print("\n📐 Transformation Matrix:")
        print(transform_matrix)
        
        # Check rotation part (3x3 upper-left) is approximately orthogonal
        rotation_part = transform_matrix[:3, :3]
        should_be_identity = rotation_part @ rotation_part.T
        identity = np.eye(3)
        
        if np.allclose(should_be_identity, identity, atol=1e-6):
            print("✓ Rotation matrix is orthogonal (valid rotation)")
        else:
            print("⚠️  Rotation matrix may not be perfectly orthogonal (but could still work)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading transformation matrix: {e}")
        return False

def verify_camera_intrinsics():
    """Verify camera intrinsics file is present and readable."""
    
    print("\n📷 Verifying Camera Intrinsics") 
    print("="*50)
    
    try:
        intrinsics_path = Path("calibration_result_1754622347/camera_intrinsics.txt")
        
        with open(intrinsics_path, 'r') as f:
            content = f.read().strip()
        
        print(f"✓ Camera intrinsics loaded from: {intrinsics_path}")
        print(f"✓ Content length: {len(content)} characters")
        
        # Try to parse as numpy array
        lines = content.split('\n')
        if len(lines) >= 3:
            print("✓ Contains multiple lines (likely camera matrix format)")
        
        print("\n📐 Camera Intrinsics Content:")
        print(content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading camera intrinsics: {e}")
        return False

def verify_calibration_metadata():
    """Verify calibration metadata JSON file."""
    
    print("\n📊 Verifying Calibration Metadata")
    print("="*50)
    
    try:
        metadata_path = Path("calibration_result_1754622347/calibration_data.json")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Metadata loaded from: {metadata_path}")
        print(f"✓ Contains {len(metadata)} top-level keys")
        
        # Show key information
        key_info = {
            'camera_matrix': 'Camera intrinsics matrix',
            'distortion_coefficients': 'Camera distortion parameters', 
            'calibration_points': 'Number of calibration data points',
            'rms_error': 'Calibration accuracy metric',
            'board_size': 'CharUco board dimensions'
        }
        
        print("\n📋 Available Metadata:")
        for key, description in key_info.items():
            if key in metadata:
                value = metadata[key]
                if isinstance(value, (list, np.ndarray)) and len(value) > 5:
                    print(f"  ✓ {key}: [array with {len(value)} elements]")
                else:
                    print(f"  ✓ {key}: {value}")
            else:
                print(f"  - {key}: [not available]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading calibration metadata: {e}")
        return False

def main():
    """Main verification function."""
    
    print("🚀 LLM GRABBER - CALIBRATION VERIFICATION")
    print("="*60)
    print("Verifying that calibrated transformation matrix and")
    print("related files are correctly copied and accessible.\n")
    
    # Change to llm_grabber directory if not already there
    if not Path("robot_motions.py").exists():
        if Path("llm_grabber/robot_motions.py").exists():
            os.chdir("llm_grabber")
            print("📁 Changed to llm_grabber directory\n")
        else:
            print("❌ Could not find llm_grabber directory or robot_motions.py")
            return False
    
    # Run verification steps
    steps = [
        ("Calibration Files", verify_calibration_files),
        ("Transformation Matrix", verify_transformation_matrix), 
        ("Camera Intrinsics", verify_camera_intrinsics),
        ("Calibration Metadata", verify_calibration_metadata)
    ]
    
    results = []
    for step_name, step_func in steps:
        try:
            result = step_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error in {step_name}: {e}")
            results.append(False)
    
    # Final summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (step_name, _) in enumerate(steps):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} - {step_name}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL CALIBRATION FILES VERIFIED!")
        print("✅ The LLM grabber system has access to all calibrated")
        print("   transformation matrices and camera parameters.")
        print("🚀 Ready for robot control with accurate coordinate transformations!")
        return True
    else:
        print(f"\n❌ {total - passed} verification checks failed.")
        print("🔧 Please check the errors above and ensure all calibration")
        print("   files are properly copied to the llm_grabber directory.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)