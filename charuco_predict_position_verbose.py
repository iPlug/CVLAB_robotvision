#!/usr/bin/env python3
"""
CharUco-Based Real-Time Position Prediction with Step-by-Step Output

This script shows detailed step-by-step transformation pipeline from CharUco detection
to robot base coordinates in real-time.

Usage:
    python charuco_predict_position_verbose.py [options]

Example:
    # Run with detailed step-by-step output
    python charuco_predict_position_verbose.py --verbose
"""

import sys
import os
import argparse
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.sensors.realsense_manager import RealSenseManager
from module.strategies.charuco_detection_strategy import CharucoDetectionStrategy
from module.config.config_factory import ConfigFactory

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CharUco-based real-time position prediction with step-by-step output.",
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
        help='Enable detailed step-by-step output'
    )
    parser.add_argument(
        '--step-delay',
        type=float,
        default=0.1,
        help='Delay between step outputs in seconds (default: 0.1)'
    )
    return parser.parse_args()

def print_step(step_num, title, content, delay=0):
    """Print a numbered step with formatting."""
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*60}")
    print(content)
    if delay > 0:
        time.sleep(delay)

def print_matrix(matrix, name, units=""):
    """Print a transformation matrix with formatting."""
    print(f"{name} {units}:")
    for i, row in enumerate(matrix):
        if i < 3:
            print(f"  [{row[0]:8.5f} {row[1]:8.5f} {row[2]:8.5f} {row[3]:8.5f}]")
        else:
            print(f"  [{row[0]:8.0f} {row[1]:8.0f} {row[2]:8.0f} {row[3]:8.0f}]")

def print_position(pos, name, units="mm"):
    """Print position with formatting."""
    print(f"{name}: X={pos[0]:.2f}{units}, Y={pos[1]:.2f}{units}, Z={pos[2]:.2f}{units}")

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
        # CRITICAL UNIT CONVERSION: Convert mm to meters
        transform_matrix[0:3, 3] = tvec.flatten() / 1000.0
        return transform_matrix
    except Exception as e:
        print(f"Error converting rvec/tvec to matrix: {e}")
        return None

def main():
    """Main prediction function with step-by-step output."""
    args = parse_arguments()

    print("CharUco Real-Time Position Prediction")
    print("Step-by-Step Transformation Pipeline")
    print("=" * 60)

    # --- Load Transformation Matrix ---
    try:
        T_robot_to_depth = np.load(args.matrix)
        print(f"✅ Loaded transformation matrix from: {args.matrix}")
        if args.verbose:
            print_matrix(T_robot_to_depth, "T_robot_to_depth")
    except FileNotFoundError:
        print(f"❌ ERROR: Transformation matrix file not found at '{args.matrix}'")
        return 1

    # --- Initialize Camera ---
    camera_manager = None
    try:
        print("\n🔄 Initializing RealSense camera...")
        camera_manager = RealSenseManager()
        if not camera_manager.initialize() or not camera_manager.start():
            raise RuntimeError("Failed to initialize RealSense camera.")
        print("✅ RealSense camera initialized")

        # Get factory extrinsics
        extrinsics_data = camera_manager.get_factory_extrinsics()
        if extrinsics_data is None:
            raise RuntimeError("Failed to get factory extrinsics.")
        T_depth_to_color = extrinsics_data['transformation_matrix']
        T_color_to_depth = np.linalg.inv(T_depth_to_color)
        
        # Calculate robot to color transformation
        T_robot_to_color = T_robot_to_depth @ np.linalg.inv(T_color_to_depth)
        print("✅ Calculated T_robot_to_color transform")

        # Initialize CharUco detector
        config = ConfigFactory.create_preset(args.config)
        pattern_params = config.config.get('pattern_params', {})
        aruco_dict_map = {
            'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
            'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
        }

        aruco_dict_from_config = pattern_params.get('aruco_dict_type', cv2.aruco.DICT_6X6_250)
        if isinstance(aruco_dict_from_config, str):
            aruco_dict_type_val = aruco_dict_map.get(aruco_dict_from_config, cv2.aruco.DICT_6X6_250)
        else:
            aruco_dict_type_val = aruco_dict_from_config

        charuco_strategy = CharucoDetectionStrategy(
            pattern_size=tuple(pattern_params.get('pattern_size', [8, 11])),
            square_size_mm=pattern_params.get('square_size_mm', 25.0),
            marker_size_mm=pattern_params.get('marker_size_mm', 20.0),
            aruco_dict_type=aruco_dict_type_val,
            bypass_filtering=True
        )
        
        # Use factory intrinsics
        intrinsics = camera_manager.get_camera_intrinsics()
        camera_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(intrinsics.coeffs)

        charuco_strategy.set_camera_calibration(camera_matrix, dist_coeffs)
        print("✅ CharUco detector initialized")

        # Real-time loop with step-by-step output
        print("\n🚀 Starting real-time prediction with step-by-step output")
        print("Press 'q' to exit, 'v' to toggle verbose mode")
        
        verbose_mode = args.verbose
        frame_count = 0
        
        while True:
            vertices, color_frame, metadata = camera_manager.get_frame()
            if color_frame is None:
                continue

            frame_count += 1
            
            # Detect pattern
            detected_patterns, visualization_data, _ = charuco_strategy.cluster_objects(
                None, color_frame, return_visualization=True
            )
            
            if detected_patterns and detected_patterns[0].get('pose_data'):
                detection_result = detected_patterns[0]
                rvec = detection_result['pose_data']['rvec']
                tvec = detection_result['pose_data']['tvec']
                
                if verbose_mode:
                    print(f"\n🎯 FRAME {frame_count} - CharUco Pattern Detected!")
                    
                    # Step 1: Pattern Detection
                    print_step(1, "CHARUCO PATTERN DETECTION", 
                              f"✅ CharUco pattern detected successfully\n"
                              f"📏 Raw tvec (camera coords, mm): {tvec.flatten()}\n"
                              f"🔄 Raw rvec (rotation vector): {rvec.flatten()}", 
                              args.step_delay)
                    
                    # Step 2: Camera to CharUco transformation
                    T_color_to_pattern = rvec_tvec_to_matrix(rvec, tvec)
                    if T_color_to_pattern is not None:
                        print_step(2, "CAMERA-TO-CHARUCO TRANSFORMATION",
                                  f"🔄 Converting rvec/tvec to transformation matrix\n"
                                  f"⚠️  CRITICAL: Converting tvec from mm to meters (/1000)\n"
                                  f"📍 Pattern position in camera coords (m): {T_color_to_pattern[:3, 3]}",
                                  args.step_delay)
                        
                        if verbose_mode:
                            print_matrix(T_color_to_pattern, "T_color_to_pattern", "(meters)")
                        
                        # Step 3: Coordinate frame transformations
                        print_step(3, "COORDINATE FRAME TRANSFORMATIONS",
                                  f"🔗 Applying transformation chain:\n"
                                  f"   Robot Base → Depth Camera → Color Camera → CharUco Pattern\n"
                                  f"🧮 T_robot_to_pattern = T_robot_to_color @ T_color_to_pattern",
                                  args.step_delay)
                        
                        # Step 4: Transform to robot base
                        T_robot_to_pattern = T_robot_to_color @ T_color_to_pattern
                        
                        print_step(4, "ROBOT BASE COORDINATE TRANSFORMATION",
                                  f"✅ Pattern transformed to robot base coordinates\n"
                                  f"📍 Position in robot base (meters): {T_robot_to_pattern[:3, 3]}",
                                  args.step_delay)
                        
                        if verbose_mode:
                            print_matrix(T_robot_to_pattern, "T_robot_to_pattern", "(meters)")
                        
                        # Step 5: Extract final position
                        position_m = T_robot_to_pattern[:3, 3]
                        position_mm = position_m * 1000
                        
                        print_step(5, "FINAL POSITION EXTRACTION",
                                  f"📐 Extracting translation component from transformation matrix\n"
                                  f"🔄 Converting from meters to millimeters (*1000)\n"
                                  f"🎯 FINAL POSITION (Robot Base Coordinates):",
                                  args.step_delay)
                        
                        print_position(position_mm, "   CharUco Pattern Position", "mm")
                        
                        print(f"\n📊 COORDINATE SYSTEM: Robot Base")
                        print(f"   🔴 X-axis: Forward  (+{position_mm[0]:.1f}mm)")
                        print(f"   🟢 Y-axis: Left     (+{position_mm[1]:.1f}mm)")
                        print(f"   🔵 Z-axis: Up       (+{position_mm[2]:.1f}mm)")
                        
                        # Calculate orientation
                        orientation_rad = Rotation.from_matrix(T_robot_to_pattern[:3, :3]).as_euler('zyx', degrees=False)
                        orientation_deg = np.degrees(orientation_rad)
                        
                        print(f"\n🔄 ORIENTATION (ZYX Euler angles):")
                        print(f"   🔄 Z-rotation: {orientation_deg[0]:.1f}°")
                        print(f"   🔄 Y-rotation: {orientation_deg[1]:.1f}°")
                        print(f"   🔄 X-rotation: {orientation_deg[2]:.1f}°")
                        
                else:
                    # Simple output mode
                    T_color_to_pattern = rvec_tvec_to_matrix(rvec, tvec)
                    if T_color_to_pattern is not None:
                        T_robot_to_pattern = T_robot_to_color @ T_color_to_pattern
                        position_mm = T_robot_to_pattern[:3, 3] * 1000
                        print(f"Frame {frame_count:4d}: X={position_mm[0]:7.2f}mm, Y={position_mm[1]:7.2f}mm, Z={position_mm[2]:7.2f}mm", end='\r')

                # Draw visualization
                if visualization_data is not None:
                    color_frame = visualization_data

                # Add status text to image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(color_frame, (5, 5), (400, 80), (0, 0, 0), -1)
                cv2.rectangle(color_frame, (5, 5), (400, 80), (255, 255, 255), 1)
                cv2.putText(color_frame, f"Frame: {frame_count}", (10, 25), font, 0.5, (255, 255, 255), 1)
                if T_color_to_pattern is not None:
                    cv2.putText(color_frame, f"X: {position_mm[0]:.1f}mm", (10, 45), font, 0.5, (0, 255, 0), 1)
                    cv2.putText(color_frame, f"Y: {position_mm[1]:.1f}mm", (150, 45), font, 0.5, (0, 255, 0), 1)
                    cv2.putText(color_frame, f"Z: {position_mm[2]:.1f}mm", (290, 45), font, 0.5, (0, 255, 0), 1)
                cv2.putText(color_frame, f"Verbose: {'ON' if verbose_mode else 'OFF'} (press 'v')", (10, 65), font, 0.4, (255, 255, 0), 1)

                cv2.drawFrameAxes(color_frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

            else:
                if not verbose_mode:
                    print(f"Frame {frame_count:4d}: No CharUco pattern detected", end='\r')
                
                # No pattern detected - add status to image
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.rectangle(color_frame, (5, 5), (350, 50), (0, 0, 0), -1)
                cv2.rectangle(color_frame, (5, 5), (350, 50), (255, 255, 255), 1)
                cv2.putText(color_frame, f"Frame {frame_count}: Searching...", (10, 30), font, 0.5, (0, 0, 255), 1)

            # Display frame
            cv2.imshow("CharUco Step-by-Step Prediction", color_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                verbose_mode = not verbose_mode
                print(f"\n🔄 Verbose mode: {'ON' if verbose_mode else 'OFF'}")

    except KeyboardInterrupt:
        print("\n\n⛔ Prediction stopped by user.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if camera_manager:
            camera_manager.stop()
        cv2.destroyAllWindows()
        print("\n🧹 Cleanup complete. Exiting.")
        return 0

if __name__ == "__main__":
    sys.exit(main())