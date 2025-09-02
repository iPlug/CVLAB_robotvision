#!/usr/bin/env python3
"""
Test to verify L515 LiDAR intrinsics usage vs true direct distance measurement.
"""

import sys
import os
import numpy as np

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.sensors.realsense_manager import RealSenseManager

def test_l515_intrinsics():
    """Test L515 LiDAR intrinsics and point cloud generation."""
    
    print("Intel RealSense L515 LiDAR Intrinsics Test")
    print("=" * 50)
    
    # Test 1: Check if L515 provides intrinsics
    print("1. Testing L515 intrinsics availability...")
    rs_manager = RealSenseManager(None, realtime_mode=True, use_imu=True)
    
    if rs_manager.initialize() and rs_manager.start():
        print("   L515 initialized successfully")
        
        # Get intrinsics
        intrinsics = rs_manager.get_camera_intrinsics()
        
        if intrinsics:
            print("   L515 DOES provide intrinsics:")
            print(f"     fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
            print(f"     cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
            print(f"     Resolution: {intrinsics.width}x{intrinsics.height}")
            
            # Test 2: Check if they're used in point cloud generation
            print("\n2. Testing point cloud generation method...")
            
            # Get a frame to see which method is used
            try:
                vertices, color_image, metadata = rs_manager.get_frame()
                
                if vertices is not None:
                    print(f"   Generated {len(vertices)} vertices")
                    
                    if rs_manager.external_intrinsics is not None:
                        print("   Method: Custom external intrinsics (CharUco)")
                    else:
                        print("   Method: L515 hardware intrinsics (SDK default)")
                        print("   Question: Are hardware intrinsics actually needed for L515?")
                        
                    # Analysis of point cloud properties
                    if len(vertices) > 0:
                        distances = np.linalg.norm(vertices, axis=1)
                        print(f"   Distance range: {np.min(distances):.3f}m to {np.max(distances):.3f}m")
                        print(f"   Mean distance: {np.mean(distances):.3f}m")
                        
                else:
                    print("   No point cloud data received")
                    
            except Exception as e:
                print(f"   Error getting frame: {e}")
                
        else:
            print("   L515 does NOT provide intrinsics")
            print("   This would confirm true LiDAR behavior!")
            
        rs_manager.stop()
    else:
        print("   Could not initialize L515 camera")
        print("   (Expected if no L515 connected)")
    
    # Test 3: Theoretical analysis
    print("\n3. Theoretical Analysis:")
    print("   LiDAR principle: Laser beam → distance measurement")
    print("   Each point should have direct (x, y, z) coordinates")
    print("   Intrinsics might be used for:")
    print("     - RGB-to-depth alignment (L515 has RGB camera)")
    print("     - SDK compatibility (same API as stereo cameras)")
    print("     - Standard output format conversion")
    
    print("\n4. Key Question for Your Use Case:")
    print("   If L515 provides direct 3D coordinates, why use CharUco calibration?")
    print("   Possible reasons:")
    print("     a) L515 still uses camera-like coordinate system")
    print("     b) CharUco improves accuracy even for LiDAR")
    print("     c) RGB-depth alignment needs intrinsics")
    print("     d) Your robot calibration was done with CharUco pattern")
    
    print("\n" + "=" * 50)
    print("Recommendation:")
    print("Run this test with your L515 connected to see actual behavior!")
    
if __name__ == "__main__":
    try:
        test_l515_intrinsics()
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()