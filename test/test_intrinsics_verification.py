#!/usr/bin/env python3
"""
Verify which camera intrinsics are being used for coordinate transformations.
This test compares RealSense factory intrinsics vs CharUco calibrated intrinsics.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from virtual_fixed_tracking_charuco import CharucoVirtualFixedCameraTracker
from module.sensors.realsense_manager import RealSenseManager

def test_intrinsics_usage():
    """Test and compare intrinsics being used by the system."""
    
    print("Camera Intrinsics Verification Test")
    print("=" * 50)
    
    # Create CharUco tracker to load calibration data
    print("1. Loading CharUco calibration data...")
    tracker = CharucoVirtualFixedCameraTracker()
    charuco_loaded = tracker._load_charuco_transformation()
    
    if charuco_loaded and tracker.charuco_camera_matrix is not None:
        print("   CharUco intrinsics loaded:")
        charuco_fx = tracker.charuco_camera_matrix[0,0]
        charuco_fy = tracker.charuco_camera_matrix[1,1]
        charuco_cx = tracker.charuco_camera_matrix[0,2]
        charuco_cy = tracker.charuco_camera_matrix[1,2]
        print(f"     fx={charuco_fx:.2f}, fy={charuco_fy:.2f}")
        print(f"     cx={charuco_cx:.2f}, cy={charuco_cy:.2f}")
    else:
        print("   CharUco intrinsics: NOT AVAILABLE")
        return
    
    # Create RealSense manager to get factory intrinsics
    print("\n2. Getting RealSense factory intrinsics...")
    rs_manager = RealSenseManager(None, realtime_mode=True, use_imu=True)
    
    if rs_manager.initialize() and rs_manager.start():
        print("   RealSense camera initialized successfully")
        
        # Get factory intrinsics
        factory_intrinsics = rs_manager.get_camera_intrinsics()
        
        if factory_intrinsics:
            print("   RealSense factory intrinsics:")
            print(f"     fx={factory_intrinsics.fx:.2f}, fy={factory_intrinsics.fy:.2f}")
            print(f"     cx={factory_intrinsics.ppx:.2f}, cy={factory_intrinsics.ppy:.2f}")
            
            # Compare intrinsics
            print("\n3. Comparison Analysis:")
            fx_diff = abs(charuco_fx - factory_intrinsics.fx)
            fy_diff = abs(charuco_fy - factory_intrinsics.fy)
            cx_diff = abs(charuco_cx - factory_intrinsics.ppx)
            cy_diff = abs(charuco_cy - factory_intrinsics.ppy)
            
            print(f"   fx difference: {fx_diff:.2f}")
            print(f"   fy difference: {fy_diff:.2f}")
            print(f"   cx difference: {cx_diff:.2f}")
            print(f"   cy difference: {cy_diff:.2f}")
            
            # Determine if they are significantly different
            threshold = 5.0  # pixels
            if any([fx_diff > threshold, fy_diff > threshold, cx_diff > threshold, cy_diff > threshold]):
                print(f"   Status: INTRINSICS ARE SIGNIFICANTLY DIFFERENT (>{threshold} pixels)")
                print("   CRITICAL: The system may be using wrong intrinsics!")
                
                # Check which ones the system is actually using
                print("\n4. Current Usage Analysis:")
                print("   Point cloud generation: Uses RealSense factory intrinsics")
                print("   CharUco transformation: Uses eye-in-hand matrix with 3D coordinates")
                print("   ISSUE: 3D coordinates from factory intrinsics may not match CharUco calibration!")
                
            else:
                print(f"   Status: INTRINSICS ARE SIMILAR (<{threshold} pixels)")
                print("   OK: Factory and CharUco intrinsics are close enough")
                
        else:
            print("   Could not get RealSense factory intrinsics")
            
        rs_manager.stop()
    else:
        print("   Could not initialize RealSense camera")
        print("   (This is expected if no camera is connected)")
    
    # Analysis of the transformation chain
    print("\n5. Transformation Chain Analysis:")
    print("   Current implementation:")
    print("     1. RealSense depth → 3D points (using FACTORY intrinsics)")
    print("     2. 3D points → robot coordinates (using CharUco eye-in-hand matrix)")
    print("     3. This creates a mismatch if intrinsics differ significantly!")
    
    print("\n   Correct implementation should be:")
    print("     1. RealSense depth → 3D points (using CHARUCO intrinsics)")
    print("     2. 3D points → robot coordinates (using CharUco eye-in-hand matrix)")
    print("     3. This ensures consistent calibration throughout the chain")
    
    if charuco_loaded:
        print("\n6. Recommendation:")
        if hasattr(tracker, 'charuco_camera_matrix') and tracker.charuco_camera_matrix is not None:
            print("   MODIFY RealSenseManager to accept external intrinsics")
            print("   OR override point cloud generation with CharUco intrinsics")
            print("   This will ensure the entire calibration chain uses consistent parameters")
        else:
            print("   CharUco intrinsics not loaded - system may work with factory intrinsics")

if __name__ == "__main__":
    try:
        test_intrinsics_usage()
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()