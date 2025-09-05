#!/usr/bin/env python3
"""
Comprehensive Standalone Verification for LLM Grabber

This script verifies that the llm_grabber folder is completely isolated
and standalone, with no external dependencies on parent directories.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_sys_path_references():
    """Check for problematic sys.path references to parent directories."""
    
    print("🔍 Checking for external sys.path references...")
    
    py_files = list(Path('.').rglob('*.py'))
    issues = []
    
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            for i, line in enumerate(lines, 1):
                if 'sys.path' in line and ('..', '..' , '...') in line:
                    # Check if it's going outside llm_grabber
                    if 'dirname' in line and line.count('dirname') > 2:
                        issues.append(f"{py_file}:{i} - {line.strip()}")
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
    
    if issues:
        print("❌ Found external sys.path references:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ No problematic sys.path references found")
        return True

def check_hardcoded_paths():
    """Check for hardcoded paths to parent directories."""
    
    print("\n🔍 Checking for hardcoded parent directory paths...")
    
    py_files = list(Path('.').rglob('*.py'))
    issues = []
    
    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            for i, line in enumerate(lines, 1):
                if 'charuco-pose-estimation' in line and py_file.name != 'standalone_verification.py':
                    issues.append(f"{py_file}:{i} - {line.strip()}")
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
    
    if issues:
        print("❌ Found hardcoded parent directory paths:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print("✅ No hardcoded parent directory paths found")
        return True

def check_required_files():
    """Check that all required files are present in llm_grabber."""
    
    print("\n📁 Checking for required files...")
    
    required_files = [
        'main.py',
        'robot_motions.py',
        'requirements.txt',
        'README.md',
        'utils_modules/__init__.py',
        'module/robot/mycobot_controller.py',
        'myCobot',  # Directory
        'calibration_result_1754622347/eye_in_hand_transform.npy',
        'recorded_coords_280.json',
        'intrinsic_l515.txt'
    ]
    
    missing_files = []
    for required_file in required_files:
        path = Path(required_file)
        if not path.exists():
            missing_files.append(required_file)
        else:
            print(f"✅ {required_file}")
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        return False
    else:
        print("\n✅ All required files present")
        return True

def test_imports():
    """Test that all main imports work without external dependencies."""
    
    print("\n🧪 Testing standalone imports...")
    
    import_tests = [
        "from robot_motions import RobotMotionController",
        "from utils_modules import RobotUtils, CameraUtils, CoordinateUtils, DetectionUtils",
        "from module.robot.mycobot_controller import MyCobotController",
        "from main import EnhancedLidarPredictor"
    ]
    
    failed_imports = []
    for import_test in import_tests:
        try:
            exec(import_test)
            print(f"✅ {import_test}")
        except Exception as e:
            print(f"❌ {import_test} - {e}")
            failed_imports.append(import_test)
    
    if failed_imports:
        print(f"\n❌ Failed imports: {len(failed_imports)}")
        return False
    else:
        print("\n✅ All imports successful")
        return True

def test_functionality():
    """Test basic functionality without camera hardware."""
    
    print("\n⚡ Testing basic functionality...")
    
    try:
        # Test robot motion controller
        from robot_motions import RobotMotionController
        robot_controller = RobotMotionController()
        status = robot_controller.get_robot_status()
        
        if 'available' in status:
            print("✅ RobotMotionController functional")
        else:
            print("❌ RobotMotionController status invalid")
            return False
        
        # Test main class import (but don't instantiate without camera)
        from main import EnhancedLidarPredictor
        print("✅ EnhancedLidarPredictor importable")
        
        # Test calibration loading
        import numpy as np
        transform_matrix = np.load('calibration_result_1754622347/eye_in_hand_transform.npy')
        if transform_matrix.shape == (4, 4):
            print("✅ Calibration matrix loadable")
        else:
            print("❌ Invalid calibration matrix")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run comprehensive standalone verification."""
    
    print("🚀 LLM GRABBER STANDALONE VERIFICATION")
    print("="*60)
    print("Verifying that llm_grabber is completely isolated")
    print("and contains all necessary files and dependencies.\n")
    
    # Change to llm_grabber directory
    if Path('main.py').exists():
        print("📁 Already in llm_grabber directory\n")
    else:
        print("❌ Not in llm_grabber directory!")
        print("Please run this script from within the llm_grabber folder.")
        return False
    
    # Run verification tests
    tests = [
        ("External References Check", check_sys_path_references),
        ("Hardcoded Paths Check", check_hardcoded_paths), 
        ("Required Files Check", check_required_files),
        ("Import Tests", test_imports),
        ("Functionality Tests", test_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {e}")
            results.append(False)
    
    # Final summary
    print("\n" + "="*60)
    print("📊 STANDALONE VERIFICATION SUMMARY")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 STANDALONE VERIFICATION SUCCESSFUL!")
        print("✅ The llm_grabber folder is completely isolated and self-contained")
        print("✅ No external dependencies on parent directories")
        print("✅ All required files and configurations present")
        print("✅ Robot motion controller ready for LLM integration")
        print("✅ Calibration files properly accessible")
        print("\n🚀 READY FOR DEPLOYMENT AS STANDALONE SYSTEM!")
        return True
    else:
        print(f"\n❌ {total - passed} verification tests failed.")
        print("🔧 Please check the errors above and fix issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)