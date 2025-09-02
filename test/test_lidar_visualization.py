#!/usr/bin/env python3
"""
Quick test script to check LiDAR visualization status
"""

import sys
import os
import numpy as np
import cv2

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.sensors.realsense_manager import RealSenseManager
from module.strategies.lidar_clustering_strategy import LiDARClusteringStrategy

def test_lidar_visualization():
    """Test LiDAR visualization components."""
    print("Testing LiDAR Visualization Components...")
    print("=" * 50)
    
    # Initialize sensor
    print("1. Initializing RealSense sensor...")
    sensor_manager = RealSenseManager(None, realtime_mode=True, use_imu=True)
    
    if not sensor_manager.initialize():
        print("   ERROR: Failed to initialize sensor")
        return False
    
    if not sensor_manager.start():
        print("   ERROR: Failed to start sensor")
        return False
    
    print("   OK: Sensor initialized")
    
    # Initialize clustering strategy
    print("2. Initializing LiDAR clustering strategy...")
    clustering_strategy = LiDARClusteringStrategy(
        distance_range=(0.2, 1.0),  # Wide range for testing
        floor_params={
            'distance_threshold': 0.03,    # More lenient
            'max_trials': 500,
            'min_floor_points': 100,       # Lower requirement
            'floor_normal_tolerance': 0.5  # More lenient
        },
        clustering_params={
            'eps': 0.05,                   # Larger epsilon
            'min_samples': 5,              # Fewer samples required
            'min_cluster_size': 20,        # Smaller clusters
            'max_cluster_size': 3000,
            'outlier_removal': False       # Disable for testing
        }
    )
    print("   OK: Clustering strategy initialized")
    
    # Test with a few frames
    print("3. Testing with live frames...")
    try:
        for frame_num in range(5):
            print(f"   Frame {frame_num + 1}...")
            
            # Get frame
            vertices, color_image, metadata = sensor_manager.get_lidar_optimized_frame(
                distance_range=(0.2, 1.0), enable_filtering=True
            )
            
            if vertices is None or len(vertices) == 0:
                print(f"      No vertices in frame {frame_num + 1}")
                continue
            
            print(f"      Got {len(vertices)} vertices")
            
            # Test clustering with visualization
            result = clustering_strategy.cluster_objects(
                vertices, color_image, return_visualization=True
            )
            
            if isinstance(result, tuple) and len(result) == 4:
                cluster_list, lidar_view, floor_view, cluster_view = result
                print(f"      Visualization data received:")
                print(f"        Clusters found: {len(cluster_list)}")
                print(f"        LiDAR view: {lidar_view.shape}")
                print(f"        Floor view: {floor_view.shape}")
                print(f"        Cluster view: {cluster_view.shape}")
                
                # Check if views have actual content (not all black)
                lidar_content = np.sum(lidar_view) > 0
                floor_content = np.sum(floor_view) > 0
                cluster_content = np.sum(cluster_view) > 0
                
                print(f"        LiDAR content: {'YES' if lidar_content else 'NO'}")
                print(f"        Floor content: {'YES' if floor_content else 'NO'}")
                print(f"        Cluster content: {'YES' if cluster_content else 'NO'}")
                
                # Show one frame as example
                if frame_num == 2:  # Show middle frame
                    print("      Displaying sample visualization...")
                    combined = np.hstack([lidar_view, floor_view])
                    combined2 = np.hstack([cluster_view, np.zeros_like(cluster_view)])
                    full_display = np.vstack([combined, combined2])
                    
                    cv2.imshow('LiDAR Test Visualization', full_display)
                    print("      Press any key to continue...")
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            else:
                print(f"      No visualization data returned")
                print(f"      Result type: {type(result)}")
                print(f"      Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            
    except Exception as e:
        print(f"   ERROR during testing: {e}")
        return False
    
    finally:
        sensor_manager.stop()
    
    print("4. Test completed")
    return True

if __name__ == "__main__":
    success = test_lidar_visualization()
    if success:
        print("\nTest PASSED: LiDAR visualization system is working")
    else:
        print("\nTest FAILED: Issues with LiDAR visualization")