#!/usr/bin/env python3
"""
Simple Real-Time CharUco Position Steps

Shows the key transformation steps from CharUco detection to robot position
in a clean, easy-to-follow format.

Usage:
    python charuco_predict_steps_simple.py
"""

import sys
import os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.sensors.realsense_manager import RealSenseManager
from module.strategies.charuco_detection_strategy import CharucoDetectionStrategy
from module.config.config_factory import ConfigFactory

def rvec_tvec_to_matrix(rvec, tvec):
    """Convert rvec and tvec to transformation matrix."""
    if rvec is None or tvec is None:
        return None
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 0:3] = rotation_matrix
    transform_matrix[0:3, 3] = tvec.flatten() / 1000.0  # Convert mm to meters
    return transform_matrix

def main():
    """Main function with simple step output."""
    print("CharUco Real-Time Position Pipeline")
    print("=" * 50)

    # Load transformation matrix
    try:
        matrix_file = 'charuco_eye_in_hand_transform.npy'
        T_robot_to_depth = np.load(matrix_file)
        print(f"✅ Loaded eye-in-hand calibration: {matrix_file}")
    except FileNotFoundError:
        print(f"❌ Could not find: {matrix_file}")
        print("Please run CharUco calibration first.")
        return 1

    # Initialize camera
    try:
        camera_manager = RealSenseManager()
        if not camera_manager.initialize() or not camera_manager.start():
            raise RuntimeError("Camera initialization failed")
        print("✅ Camera initialized")

        # Get camera transformations
        extrinsics_data = camera_manager.get_factory_extrinsics()
        T_depth_to_color = extrinsics_data['transformation_matrix']
        T_color_to_depth = np.linalg.inv(T_depth_to_color)
        T_robot_to_color = T_robot_to_depth @ np.linalg.inv(T_color_to_depth)
        print("✅ Camera transformations calculated")

        # Initialize CharUco detector
        config = ConfigFactory.create_preset('charuco_standard')
        pattern_params = config.config.get('pattern_params', {})
        
        charuco_strategy = CharucoDetectionStrategy(
            pattern_size=tuple(pattern_params.get('pattern_size', [8, 11])),
            square_size_mm=pattern_params.get('square_size_mm', 25.0),
            marker_size_mm=pattern_params.get('marker_size_mm', 20.0),
            aruco_dict_type=cv2.aruco.DICT_6X6_250,
            bypass_filtering=True
        )
        
        intrinsics = camera_manager.get_camera_intrinsics()
        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(intrinsics.coeffs)
        charuco_strategy.set_camera_calibration(camera_matrix, dist_coeffs)
        print("✅ CharUco detector ready")

        print("\n🚀 Starting real-time prediction...")
        print("Press 'q' to quit\n")
        
        frame_count = 0
        
        while True:
            vertices, color_frame, metadata = camera_manager.get_frame()
            if color_frame is None:
                continue

            frame_count += 1
            
            # Step 1: Detect CharUco pattern
            detected_patterns, visualization_data, _ = charuco_strategy.cluster_objects(
                None, color_frame, return_visualization=True
            )
            
            if detected_patterns and detected_patterns[0].get('pose_data'):
                detection_result = detected_patterns[0]
                rvec = detection_result['pose_data']['rvec']
                tvec = detection_result['pose_data']['tvec']
                
                print(f"\n📋 FRAME {frame_count}")
                print("-" * 30)
                
                # Step 2: Show raw detection
                tvec_mm = tvec.flatten()
                print(f"Step 1: CharUco detected at camera position")
                print(f"        tvec (mm): [{tvec_mm[0]:6.1f}, {tvec_mm[1]:6.1f}, {tvec_mm[2]:6.1f}]")
                
                # Step 3: Convert to transformation matrix
                T_color_to_pattern = rvec_tvec_to_matrix(rvec, tvec)
                if T_color_to_pattern is not None:
                    pattern_pos_camera_m = T_color_to_pattern[:3, 3]
                    print(f"Step 2: Convert to camera coords (meters)")
                    print(f"        position: [{pattern_pos_camera_m[0]:6.3f}, {pattern_pos_camera_m[1]:6.3f}, {pattern_pos_camera_m[2]:6.3f}]")
                    
                    # Step 4: Transform to robot base
                    T_robot_to_pattern = T_robot_to_color @ T_color_to_pattern
                    position_m = T_robot_to_pattern[:3, 3]
                    position_mm = position_m * 1000
                    
                    print(f"Step 3: Transform to robot base coords")
                    print(f"        position (m): [{position_m[0]:6.3f}, {position_m[1]:6.3f}, {position_m[2]:6.3f}]")
                    
                    # Step 5: Final result
                    print(f"Step 4: FINAL ROBOT POSITION (mm)")
                    print(f"        X (forward): {position_mm[0]:7.1f}mm")
                    print(f"        Y (left):    {position_mm[1]:7.1f}mm") 
                    print(f"        Z (up):      {position_mm[2]:7.1f}mm")
                    
                    # Show on image
                    if visualization_data is not None:
                        color_frame = visualization_data
                    
                    # Add text overlay
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.rectangle(color_frame, (10, 10), (400, 120), (0, 0, 0), -1)
                    cv2.rectangle(color_frame, (10, 10), (400, 120), (255, 255, 255), 2)
                    
                    cv2.putText(color_frame, f"Frame {frame_count}", (20, 35), font, 0.6, (255, 255, 255), 2)
                    cv2.putText(color_frame, f"X: {position_mm[0]:6.1f}mm", (20, 60), font, 0.5, (0, 0, 255), 2)
                    cv2.putText(color_frame, f"Y: {position_mm[1]:6.1f}mm", (20, 80), font, 0.5, (0, 255, 0), 2)
                    cv2.putText(color_frame, f"Z: {position_mm[2]:6.1f}mm", (20, 100), font, 0.5, (255, 0, 0), 2)
                    
                    # Draw coordinate axes
                    cv2.drawFrameAxes(color_frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

            else:
                print(f"Frame {frame_count}: No CharUco pattern detected", end='\r')
                
                # Add text to image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(color_frame, (10, 10), (300, 50), (0, 0, 0), -1)
                cv2.putText(color_frame, f"Frame {frame_count}: Searching...", (20, 35), font, 0.6, (0, 0, 255), 2)

            # Display
            cv2.imshow("CharUco Position Steps", color_frame)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'camera_manager' in locals():
            camera_manager.stop()
        cv2.destroyAllWindows()
        print("\nCleanup complete.")

if __name__ == "__main__":
    main()