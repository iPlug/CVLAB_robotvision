#!/usr/bin/env python3
"""
CharUco-Based Real-Time Position Prediction

This script uses a pre-calculated eye-in-hand calibration matrix to predict the
real-world position of a CharUco pattern relative to the robot's base.

The script performs the following steps:
1.  Loads the robot-to-camera transformation matrix from the calibration process.
2.  Initializes a RealSense camera and its factory intrinsics/extrinsics.
3.  Calculates the required `T_robot_to_color` transformation.
4.  In a real-time loop:
    a. Captures images from the camera.
    b. Detects the CharUco pattern to get its pose relative to the camera.
    c. Transforms the camera-relative pose into the robot's coordinate frame.
    d. Displays the predicted position and the annotated video feed.

Usage:
    python charuco_predict_position.py [options]

Example:
    # Predict position using the default transformation matrix
    python charuco_predict_position.py

    # Specify a custom transformation matrix
    python charuco_predict_position.py --matrix my_transform.npy
"""

import sys
import os
import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.sensors.realsense_manager import RealSenseManager
from module.strategies.charuco_detection_strategy import CharucoDetectionStrategy
from module.config.config_factory import ConfigFactory

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CharUco-based real-time position prediction.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--matrix',
        type=str,
        default='charuco_eye_in_hand_transform.npy',
        help='Path to the eye-in-hand transformation matrix file (default: charuco_eye_in_hand_transform.npy)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='charuco_standard',
        help='Configuration preset for CharUco pattern (default: charuco_standard)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--no-filtering',
        action='store_true',
        help='Disable quality filtering for more permissive detection'
    )
    return parser.parse_args()

def rvec_tvec_to_matrix(rvec, tvec):
    """Convert rvec and tvec to a 4x4 transformation matrix."""
    if rvec is None or tvec is None:
        return None
    try:
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # Create 4x4 transformation matrix
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = rotation_matrix
        # CRITICAL UNIT CONVERSION FIX:
        # CharUco detection returns tvec in millimeters, but transformation pipeline expects meters.
        # Without this conversion, robot coordinates are magnified 1000x causing km-scale errors.
        # This ensures proper robot coordinate output in mm relative to robot base.
        transform_matrix[0:3, 3] = tvec.flatten() / 1000.0  # Convert mm to meters
        return transform_matrix
    except Exception as e:
        print(f"Error converting rvec/tvec to matrix: {e}")
        return None

def main():
    """Main prediction function."""
    args = parse_arguments()

    print("--- CharUco Real-Time Position Prediction ---")

    # --- 1. Load Transformation Matrix ---
    try:
        T_robot_to_depth = np.load(args.matrix)
        print(f"Successfully loaded transformation matrix from: {args.matrix}")
        if args.verbose:
            print("T_robot_to_depth:", T_robot_to_depth)
    except FileNotFoundError:
        print(f"ERROR: Transformation matrix file not found at '{args.matrix}'")
        print("Please run 'charuco_eye_in_hand.py' first to generate the calibration file.")
        return 1

    # --- 2. Initialize Camera and Dependencies ---
    camera_manager = None
    try:
        print("Initializing RealSense camera...")
        camera_manager = RealSenseManager()
        if not camera_manager.initialize():
            raise RuntimeError("Failed to initialize RealSense camera.")
        if not camera_manager.start():
            raise RuntimeError("Failed to start RealSense camera.")
        print("RealSense camera initialized and started.")

        # Get factory extrinsics (T_depth_to_color) and invert it
        extrinsics_data = camera_manager.get_factory_extrinsics()
        if extrinsics_data is None:
            raise RuntimeError("Failed to get factory extrinsics from RealSense camera.")
        T_depth_to_color = extrinsics_data['transformation_matrix']
        # The extrinsics are from color to depth, so we need to invert it to get depth to color
        T_color_to_depth = np.linalg.inv(T_depth_to_color)
        
        if args.verbose:
            print("T_color_to_depth:", T_color_to_depth)

        # --- 3. Calculate Required Transform ---
        # We have T_robot_to_depth and T_color_to_depth.
        # We need T_robot_to_color to combine with the pattern pose T_color_to_pattern.
        # T_robot_to_depth = T_robot_to_color @ T_color_to_depth
        # T_robot_to_color = T_robot_to_depth @ inv(T_color_to_depth)
        T_robot_to_color = T_robot_to_depth @ np.linalg.inv(T_color_to_depth)
        print("Calculated T_robot_to_color transform.")
        if args.verbose:
            print("T_robot_to_color:", T_robot_to_color)

        # --- 4. Initialize CharUco Detector ---
        config = ConfigFactory.create_preset(args.config)
        # Extract CharUco pattern parameters from the config
        pattern_params = config.config.get('pattern_params', {})
        aruco_dict_map = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
            'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
            'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
            'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
            'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
            'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
            'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
            'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
            'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
            'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
            'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
            'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
            'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
            'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
            'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
        }

        # Get aruco_dict_type from config, defaulting to integer if not found or invalid
        aruco_dict_from_config = pattern_params.get('aruco_dict_type', cv2.aruco.DICT_6X6_250)
        if isinstance(aruco_dict_from_config, str):
            aruco_dict_type_val = aruco_dict_map.get(aruco_dict_from_config, cv2.aruco.DICT_6X6_250)
        else:
            aruco_dict_type_val = aruco_dict_from_config

        charuco_strategy = CharucoDetectionStrategy(
            pattern_size=tuple(pattern_params.get('pattern_size', [8, 11])),
            square_size_mm=pattern_params.get('square_size_mm', 25.0),
            marker_size_mm=pattern_params.get('marker_size_mm', 20.0),
            aruco_dict_type=aruco_dict_type_val, # Use the resolved integer value
            bypass_filtering=args.no_filtering # Use command line argument for filtering control
        )
        
        # Use factory intrinsics for detection
        intrinsics = camera_manager.get_camera_intrinsics()
        # Construct camera matrix and distortion coefficients from intrinsics
        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(intrinsics.coeffs)

        charuco_strategy.set_camera_calibration(
            camera_matrix, dist_coeffs
        )
        print("CharUco detector initialized with factory camera intrinsics.")

        # --- 5. Real-time Prediction Loop ---
        print("\nStarting real-time prediction. Press 'q' to exit.")
        while True:
            vertices, color_frame, metadata = camera_manager.get_frame()
            if color_frame is None:
                continue

            # Detect pattern
            detected_patterns, visualization_data, _ = charuco_strategy.cluster_objects(None, color_frame, return_visualization=True)
            
            if detected_patterns and detected_patterns[0].get('pose_data'):
                # Assuming only one pattern is being tracked
                detection_result = detected_patterns[0]
                
                # Get pattern pose relative to the color camera
                rvec = detection_result['pose_data']['rvec']
                tvec = detection_result['pose_data']['tvec']
                T_color_to_pattern = rvec_tvec_to_matrix(rvec, tvec)

                if T_color_to_pattern is not None:
                    # NOTE: CharUco detection handles unit conversion (mm to meters)
                    # tvec is in meters, no additional coordinate system correction applied
                    
                    # Transform to robot base frame
                    T_robot_to_pattern = T_robot_to_color @ T_color_to_pattern
                    
                    # Extract position and orientation
                    position = T_robot_to_pattern[:3, 3]  # This is now correctly in meters
                    orientation_rad = Rotation.from_matrix(T_robot_to_pattern[:3, :3]).as_euler('zyx', degrees=False)
                    orientation_deg = np.degrees(orientation_rad)
                    
                    # Debug output
                    if args.verbose:
                        print(f"\nDEBUG:")
                        print(f"tvec from CharUco (m): {tvec.flatten()}")
                        print(f"T_color_to_pattern translation (m): {T_color_to_pattern[:3, 3]}")
                        print(f"T_robot_to_pattern translation (m): {position}")
                        print(f"Position in mm: {position * 1000}")

                    # Display results on console
                    print(f"Predicted Position (mm): X={position[0]*1000:.2f}, Y={position[1]*1000:.2f}, Z={position[2]*1000:.2f}", end='\r')
                    
                    # Draw on image using visualization_data
                    if visualization_data is not None:
                        color_frame = visualization_data

                    # Draw predicted position on the OpenCV window with background
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2
                    text_color = (0, 255, 0) # Green color
                    bg_color = (0, 0, 0) # Black background

                    # Draw background rectangles for better visibility
                    cv2.rectangle(color_frame, (5, 5), (350, 120), bg_color, -1)
                    cv2.rectangle(color_frame, (5, 5), (350, 120), (255, 255, 255), 1)

                    # Draw text with proper spacing
                    cv2.putText(color_frame, f"Robot Position:", (10, 25), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    cv2.putText(color_frame, f"X: {position[0]*1000:.1f} mm", (10, 50), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    cv2.putText(color_frame, f"Y: {position[1]*1000:.1f} mm", (10, 75), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
                    cv2.putText(color_frame, f"Z: {position[2]*1000:.1f} mm", (10, 100), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

                    # Use cv2.drawFrameAxes
                    cv2.drawFrameAxes(color_frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            else:
                # If no pattern detected, clear the console line and display message on window
                print("No CharUco pattern detected.                              ", end='\r')
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                text_color = (0, 0, 255) # Red color
                bg_color = (0, 0, 0) # Black background

                # Draw background rectangle for better visibility
                cv2.rectangle(color_frame, (5, 5), (350, 50), bg_color, -1)
                cv2.rectangle(color_frame, (5, 5), (350, 50), (255, 255, 255), 1)
                cv2.putText(color_frame, "No CharUco pattern detected.", (10, 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)


            cv2.imshow("Real-time Prediction", color_frame)
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