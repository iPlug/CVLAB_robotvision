#!/usr/bin/env python3
"""
LiDAR-Based Real-Time Position Prediction

This script adapts the CharUco position prediction approach for LiDAR sensor data.
It uses the same eye-in-hand calibration matrix and coordinate transformation
approach as charuco_predict_position.py, but replaces CharUco pattern detection
with LiDAR object detection.

Key Steps (adapted from CharUco script):
1. Load robot-to-depth transformation matrix from calibration
2. Initialize RealSense camera and get factory extrinsics
3. Calculate T_robot_to_color transformation  
4. In real-time loop:
   a. Get depth and color frames
   b. Detect object using LiDAR point cloud processing
   c. Get object position relative to color camera frame
   d. Transform to robot base coordinates
   e. Display predicted position
"""

import sys
import os
import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.sensors.realsense_manager import RealSenseManager
from module.strategies.enhanced_lidar_strategy import EnhancedLiDARStrategy
from module.config.config_factory import ConfigFactory

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LiDAR-based real-time position prediction (adapted from CharUco approach).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--matrix',
        type=str,
        default='charuco-pose-estimation/calibration_result_1754622347/eye_in_hand_transform.npy',
        help='Path to the eye-in-hand transformation matrix file'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='enhanced_lidar_precise',
        help='Configuration preset for LiDAR detection'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()

def create_position_transformation_matrix(position):
    """Create a 4x4 transformation matrix from position (adapted from rvec_tvec_to_matrix)."""
    if position is None:
        return None
    try:
        # Create 4x4 transformation matrix with identity rotation
        transform_matrix = np.eye(4)
        # LiDAR position is already in meters (same as CharUco after conversion)
        transform_matrix[0:3, 3] = position.flatten()
        return transform_matrix
    except Exception as e:
        print(f"Error creating position transformation matrix: {e}")
        return None

def main():
    """Main prediction function (adapted from CharUco script)."""
    args = parse_arguments()

    print("--- LiDAR Real-Time Position Prediction (CharUco Method) ---")

    # --- 1. Load Transformation Matrix (same as CharUco) ---
    try:
        T_robot_to_depth = np.load(args.matrix)
        print(f"Successfully loaded transformation matrix from: {args.matrix}")
        if args.verbose:
            print("T_robot_to_depth:", T_robot_to_depth)
    except FileNotFoundError:
        print(f"ERROR: Transformation matrix file not found at '{args.matrix}'")
        print("Please ensure the eye-in-hand calibration file exists.")
        return 1

    # --- 2. Initialize Camera and Dependencies (same as CharUco) ---
    camera_manager = None
    try:
        print("Initializing RealSense camera...")
        camera_manager = RealSenseManager()
        if not camera_manager.initialize():
            raise RuntimeError("Failed to initialize RealSense camera.")
        if not camera_manager.start():
            raise RuntimeError("Failed to start RealSense camera.")
        print("RealSense camera initialized and started.")

        # Get factory extrinsics (T_depth_to_color) and invert it (same as CharUco)
        extrinsics_data = camera_manager.get_factory_extrinsics()
        if extrinsics_data is None:
            raise RuntimeError("Failed to get factory extrinsics from RealSense camera.")
        T_depth_to_color = extrinsics_data['transformation_matrix']
        T_color_to_depth = np.linalg.inv(T_depth_to_color)
        
        if args.verbose:
            print("T_color_to_depth:", T_color_to_depth)

        # --- 3. Calculate Required Transform (EXACTLY same as CharUco) ---
        # We have T_robot_to_depth and T_color_to_depth.
        # We need T_robot_to_color to transform from color camera frame to robot base.
        # T_robot_to_depth = T_robot_to_color @ T_color_to_depth
        # T_robot_to_color = T_robot_to_depth @ inv(T_color_to_depth)
        T_robot_to_color = T_robot_to_depth @ np.linalg.inv(T_color_to_depth)
        print("Calculated T_robot_to_color transform.")
        if args.verbose:
            print("T_robot_to_color:", T_robot_to_color)

        # --- 4. Initialize LiDAR Detection Strategy ---
        config = ConfigFactory.create_preset(args.config)
        lidar_strategy = EnhancedLiDARStrategy(
            spatial_filter_params=config.config.get('spatial_filter_params', {}),
            gravity_params=config.config.get('gravity_params', {}),
            floor_params=config.config.get('floor_params', {}),
            object_height_params=config.config.get('object_height_params', {}),
            clustering_params=config.config.get('clustering_params', {}),
            visualization_params=config.config.get('visualization_params', {})
        )
        
        print("LiDAR detection strategy initialized.")

        # --- 5. Real-time Prediction Loop (adapted from CharUco) ---
        print("\nStarting real-time LiDAR position prediction. Press 'q' to exit.")
        print("Following CharUco coordinate transformation method...")
        
        while True:
            vertices, color_frame, metadata = camera_manager.get_frame()
            if color_frame is None:
                continue

            # Detect objects using LiDAR (instead of CharUco pattern)
            result = lidar_strategy.cluster_objects(
                vertices, color_frame, return_visualization=True
            )
            
            # Handle variable return format
            if isinstance(result, tuple):
                detected_objects = result[0]
                visualization_data = result[1] if len(result) > 1 else None
            else:
                detected_objects = result
                visualization_data = None
            
            if detected_objects and len(detected_objects) > 0:
                # Get the first detected object (similar to CharUco single pattern)
                detection_result = detected_objects[0]
                
                # Get object position relative to the camera (depth camera coordinates)
                if 'center' in detection_result:
                    object_position_depth = detection_result['center']  # Already in meters
                    
                    # Transform depth camera position to color camera coordinates
                    # (CharUco works in color camera frame)
                    depth_position_homo = np.append(object_position_depth, 1.0)
                    color_position_homo = T_depth_to_color @ depth_position_homo
                    object_position_color = color_position_homo[:3]
                    
                    if args.verbose:
                        print(f"\nDEBUG:")
                        print(f"Object in depth frame (m): {object_position_depth}")
                        print(f"Object in color frame (m): {object_position_color}")
                    
                    # Create transformation matrix (adapted from rvec_tvec_to_matrix)
                    T_color_to_object = create_position_transformation_matrix(object_position_color)

                    if T_color_to_object is not None:
                        # Transform to robot base frame (EXACTLY same as CharUco)
                        T_robot_to_object = T_robot_to_color @ T_color_to_object
                        
                        # Extract position (same as CharUco)
                        position = T_robot_to_object[:3, 3]  # This is in meters
                        
                        if args.verbose:
                            print(f"T_color_to_object translation (m): {T_color_to_object[:3, 3]}")
                            print(f"T_robot_to_object translation (m): {position}")
                            print(f"Position in mm: {position * 1000}")

                        # Display results on console (same format as CharUco)
                        print(f"Predicted Position (mm): X={position[0]*1000:.2f}, Y={position[1]*1000:.2f}, Z={position[2]*1000:.2f}", end='\r')
                        
                        # Draw on image using visualization_data
                        if visualization_data is not None:
                            color_frame = visualization_data

                        # Draw predicted position on the OpenCV window (same as CharUco)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        font_thickness = 2
                        text_color = (0, 255, 0) # Green color
                        bg_color = (0, 0, 0) # Black background

                        # Draw background rectangles for better visibility
                        cv2.rectangle(color_frame, (5, 5), (350, 120), bg_color, -1)
                        cv2.rectangle(color_frame, (5, 5), (350, 120), (255, 255, 255), 1)

                        # Draw text with proper spacing
                        cv2.putText(color_frame, f"Robot Position (LiDAR):", (10, 25), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                        cv2.putText(color_frame, f"X: {position[0]*1000:.1f} mm", (10, 50), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                        cv2.putText(color_frame, f"Y: {position[1]*1000:.1f} mm", (10, 75), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                        cv2.putText(color_frame, f"Z: {position[2]*1000:.1f} mm", (10, 100), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            else:
                # If no object detected (same as CharUco)
                print("No LiDAR object detected.                              ", end='\r')
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                text_color = (0, 0, 255) # Red color
                bg_color = (0, 0, 0) # Black background

                # Draw background rectangle for better visibility
                cv2.rectangle(color_frame, (5, 5), (350, 50), bg_color, -1)
                cv2.rectangle(color_frame, (5, 5), (350, 50), (255, 255, 255), 1)
                cv2.putText(color_frame, "No LiDAR object detected.", (10, 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            cv2.imshow("LiDAR Real-time Prediction", color_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nPrediction stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if camera_manager:
            camera_manager.stop()
        cv2.destroyAllWindows()
        print("\nCleanup complete. Exiting.")
        return 0

if __name__ == "__main__":
    sys.exit(main())