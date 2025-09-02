#!/usr/bin/env python3
"""
Test script to isolate checkerboard detection issues.
"""

import cv2
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.strategies.checkerboard_detection_strategy import CheckerboardDetectionStrategy
from module.sensors.realsense_manager import RealSenseManager

def test_checkerboard_detection():
    """Test checkerboard detection with real camera data."""
    print("Testing checkerboard detection...")
    
    # Initialize RealSense camera
    sensor_manager = RealSenseManager(realtime_mode=True, use_imu=False)
    
    if not sensor_manager.initialize():
        print("ERROR: Could not initialize RealSense camera")
        return False
    
    if not sensor_manager.start():
        print("ERROR: Could not start RealSense camera")
        return False
    
    print("Camera started successfully")
    
    # Create checkerboard detection strategy
    strategy = CheckerboardDetectionStrategy(
        pattern_sizes=[(7, 5)],  # Test with 7x5 pattern
        square_size=15.0,  # 15mm squares from generated pattern
        min_pattern_coverage=0.05,  # Lower threshold for testing
        max_pattern_coverage=0.9,
        quality_threshold=0.3  # Lower quality threshold for testing
    )
    
    print("Strategy initialized with patterns:", strategy.pattern_sizes)
    print("Square size:", strategy.square_size, "mm")
    print("Quality threshold:", strategy.quality_threshold)
    
    try:
        print("\nCapturing image for detection test...")
        
        # Get camera frame
        vertices, color_image, metadata = sensor_manager.get_frame()
        
        if color_image is None:
            print("ERROR: Could not capture color image")
            return False
        
        print(f"Captured image shape: {color_image.shape}")
        print(f"Image dtype: {color_image.dtype}")
        print(f"Image value range: {color_image.min()} - {color_image.max()}")
        
        # Convert to grayscale for inspection
        if len(color_image.shape) == 3:
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = color_image
        
        print(f"Grayscale image shape: {gray_image.shape}")
        print(f"Grayscale dtype: {gray_image.dtype}")
        print(f"Grayscale value range: {gray_image.min()} - {gray_image.max()}")
        
        # Test direct OpenCV detection first
        print("\n--- Direct OpenCV Test ---")
        for pattern_size in [(7, 5), (9, 6), (8, 6)]:
            print(f"Testing pattern size {pattern_size} with OpenCV directly...")
            
            try:
                ret, corners = cv2.findChessboardCorners(
                    gray_image, 
                    pattern_size, 
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
                )
                
                print(f"  Pattern {pattern_size}: ret={ret}, corners={'found' if corners is not None else 'None'}")
                
                if ret and corners is not None:
                    print(f"  Found {len(corners)} corners")
                    print(f"  Corner shape: {corners.shape}")
                    break
                    
            except Exception as e:
                print(f"  OpenCV error for {pattern_size}: {e}")
        
        # Test with our strategy
        print("\n--- Strategy Test ---")
        detected_patterns = strategy.cluster_objects(vertices, color_image, return_visualization=False)
        
        print(f"Strategy detected {len(detected_patterns)} patterns")
        
        for i, pattern in enumerate(detected_patterns):
            print(f"Pattern {i+1}:")
            print(f"  ID: {pattern.get('id', 'Unknown')}")
            print(f"  Pattern size: {pattern.get('pattern_size', 'Unknown')}")
            print(f"  Quality score: {pattern.get('quality_score', 0):.3f}")
            print(f"  Coverage: {pattern.get('coverage', 0):.3f}")
            print(f"  Corners count: {pattern.get('corners_count', 0)}")
        
        # Get detection statistics
        stats = strategy.get_detection_statistics()
        print(f"\nDetection Statistics:")
        print(f"  Total attempts: {stats['total_attempts']}")
        print(f"  Successful detections: {stats['successful_detections']}")
        print(f"  Success rate: {stats['success_rate']:.2%}")
        print(f"  Average detection time: {stats['avg_detection_time_ms']:.1f}ms")
        
        return len(detected_patterns) > 0
        
    except Exception as e:
        print(f"ERROR during detection test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        sensor_manager.cleanup()

if __name__ == "__main__":
    success = test_checkerboard_detection()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)