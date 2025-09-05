#!/usr/bin/env python3
"""
Real-time CharUco Pose Estimation

Converts the static notebook implementation to real-time camera-based pose estimation.
Based on the original notebook but with real-time camera feed and visualization.

Features:
- Real-time RealSense camera input
- Live CharUco detection and pose estimation
- Real-time visualization with pose axes
- Keyboard controls for interaction
- Compatible with both 7x5 and 8x11 CharUco patterns
- Configurable coordinate system origin (bottom-left corner or center of board)

Usage:
    python charuco_realtime_pose.py [options]
    
    # Use default bottom-left origin
    python charuco_realtime_pose.py
    
    # Use center of board as origin
    python charuco_realtime_pose.py --center-origin

Controls:
    's' - Save current frame and results
    'c' - Move robot to next calibration position  
    'q' - Quit and calculate eye-in-hand calibration
    ESC - Exit without calibration

Coordinate Systems:
    Bottom-left origin (default): Origin at bottom-left corner of CharUco board
    Center origin (--center-origin): Origin at center of the center marker of CharUco board
    Both use: X=right, Y=up, Z=outward from board
"""

import sys
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
import time
import json
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv2 import aruco

# Import utilities and new modular transformer
from utils_modules import RobotUtils, CameraUtils, DetectionUtils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from charuco_transformer import CharUcoTransformer


class RealtimeCharucoPose:
    """Real-time CharUco pose estimation system."""
    
    def __init__(self, config_path="charuco-pose-estimation", use_center_origin=False):
        """Initialize the real-time pose estimation system.
        
        Args:
            config_path: Path to configuration directory
            use_center_origin: If True, use center of board as origin. If False, use bottom-left corner.
        """
        self.config_path = Path(config_path)
        
        # Initialize the modular CharUco transformer for calibration mode
        self.charuco_transformer = CharUcoTransformer(
            calibration_path=None,  # Calibration mode
            board_size=(7, 5),
            square_length=0.039,  # 39mm squares
            marker_length=0.029,  # 29mm markers
            use_center_origin=use_center_origin
        )
        
        # Initialize remaining utility classes
        self.robot_utils = RobotUtils()
        self.camera_utils = CameraUtils(config_path)
        self.detection_utils = DetectionUtils(config_path)
        
        # Load robot poses and camera intrinsics
        self.robot_utils.load_robot_poses()
        self.camera_utils.load_camera_intrinsics()
        
        # Setup CharUco transformer with camera parameters
        if self.camera_utils.camera_matrix is not None:
            self.charuco_transformer.setup_camera_and_detection(
                self.camera_utils.camera_matrix,
                self.camera_utils.dist_coeffs
            )
        
        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Frame counter for saving
        self.frame_count = 0
        
        # Print coordinate system information
        coord_info = self.charuco_transformer.get_calibration_info()
        print(f"CharUco Transformer initialized for calibration mode")
        board_config = coord_info['board_config']
        print(f"Board: {board_config['size']} squares, {board_config['square_length_mm']:.1f}mm squares")
        print(f"Origin: {'Center' if board_config['use_center_origin'] else 'Bottom-left'}")
        print(f"Ready for calibration: {self.charuco_transformer.is_ready_for_calibration()}")
        
    def setup_detectors(self):
        """Setup detectors using transformer - deprecated, kept for compatibility."""
        # Detectors are now handled by the CharUco transformer
        pass
    
    def setup_camera(self):
        """Setup RealSense camera with optimized settings."""
        success = self.camera_utils.setup_camera(self.pipeline, self.config)
        if success:
            # Update CharUco transformer with new camera parameters
            self.charuco_transformer.setup_camera_and_detection(
                self.camera_utils.camera_matrix,
                self.camera_utils.dist_coeffs
            )
        return success
    
    def check_subpixel(self, corners):
        """Check if corners are subpixel accurate."""
        return self.detection_utils.check_subpixel(corners)
    
    def get_robot_pose(self):
        """Get current robot end-effector pose."""
        return self.robot_utils.get_robot_pose()
    
    def collect_calibration_point(self, image):
        """Collect robot and CharUco poses for calibration using new transformer."""
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return False
        return self.charuco_transformer.collect_calibration_point(image, robot_pose)
    
    def calculate_eye_in_hand_calibration(self):
        """Calculate eye-in-hand calibration using collected data points."""
        return self.charuco_transformer.calculate_eye_in_hand_calibration()
    
    def save_calibration_results(self, eye_in_hand_transform=None):
        """Save calibration results to file."""
        return self.charuco_transformer.save_calibration_results()
    
    def validate_calibration(self, eye_in_hand_transform=None):
        """Validate calibration quality using transformer."""
        return self.charuco_transformer.validate_calibration([300.0, 100.0, 0.0])
    
    def validate_calibration_center_origin(self, eye_in_hand_transform):
        """Validate calibration quality using center origin coordinate system."""
        if len(self.calibration_utils.calibration_pairs) < 3:
            return None
        
        coord_system = "center"
        print(f"\nValidating calibration quality using {coord_system} coordinate system...")
        
        validation_results = {
            'reprojection_errors': [],
            'pose_errors': [],
            'mean_reprojection_error': 0.0,
            'max_reprojection_error': 0.0,
            'mean_translation_error': 0.0,
            'mean_rotation_error': 0.0,
            'quality_score': 'Unknown',
            'coordinate_system': coord_system
        }
        
        # Calculate center offset for validation
        center_offset_x = (self.board_size[1] - 1) * self.square_length / 2 + self.square_length / 2  # Height: X offset
        center_offset_y = (self.board_size[0] - 1) * self.square_length / 2 + self.square_length / 2  # Width: Y offset
        target_position = np.array([0.3 + center_offset_x, 0.1 - center_offset_y, 0.0])  # X+, Y- for center
        
        translation_errors = []
        rotation_errors = []
        
        for i, pair in enumerate(self.calibration_utils.calibration_pairs):
            robot_transform = pair['robot_transform']
            charuco_transform = self.charuco_pose_to_transform_matrix_center(pair['charuco_rvec'], pair['charuco_tvec'])
            
            expected_pattern = robot_transform @ eye_in_hand_transform @ charuco_transform
            expected_position = expected_pattern[:3, 3]
            
            # Calculate translation error
            translation_error = np.linalg.norm(expected_position - target_position)
            translation_errors.append(translation_error * 1000)  # Convert to mm
            
            # Calculate rotation error
            expected_rotation = np.eye(3)
            predicted_rotation = expected_pattern[:3, :3]
            
            rotation_error_matrix = predicted_rotation @ expected_rotation.T
            rotation_error_trace = np.trace(rotation_error_matrix)
            rotation_error_angle = np.arccos(np.clip((rotation_error_trace - 1) / 2, -1, 1))
            rotation_error_degrees = np.degrees(rotation_error_angle)
            rotation_errors.append(rotation_error_degrees)
        
        # Calculate summary statistics
        validation_results['mean_translation_error'] = np.mean(translation_errors)
        validation_results['max_reprojection_error'] = np.max(translation_errors)
        validation_results['mean_rotation_error'] = np.mean(rotation_errors)
        validation_results['reprojection_errors'] = translation_errors
        
        # Quality assessment
        mean_error_mm = validation_results['mean_translation_error']
        
        if mean_error_mm < 10.0:
            validation_results['quality_score'] = 'Excellent'
        elif mean_error_mm < 25.0:
            validation_results['quality_score'] = 'Good'  
        elif mean_error_mm < 50.0:
            validation_results['quality_score'] = 'Fair'
        else:
            validation_results['quality_score'] = 'Poor'
        
        return validation_results
    
    def charuco_pose_to_transform_matrix_center(self, rvec, tvec):
        """Convert CharUco rvec, tvec to 4x4 transformation matrix with center origin."""
        # Convert rotation vector to rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)
        
        # Apply coordinate system correction: 90° rotation around Z-axis
        correction_matrix = np.array([
            [0, -1, 0],  # 90° rotation around Z
            [1,  0, 0], 
            [0,  0, 1]
        ])
        
        # Apply correction to rotation matrix
        corrected_rot_matrix = rot_matrix @ correction_matrix
        
        # For center origin, we need to adjust the translation as well
        center_offset = np.array([
            (self.board_size[0] - 1) * self.square_length / 2 + self.square_length / 2,
            (self.board_size[1] - 1) * self.square_length / 2 + self.square_length / 2,
            0.0
        ])
        
        # Transform the translation to center origin
        adjusted_tvec = tvec.flatten() - center_offset
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = corrected_rot_matrix
        transform[:3, 3] = adjusted_tvec
        
        return transform
    
    def print_validation_report(self, validation_results):
        """Print a detailed validation report."""
        return self.calibration_utils.print_validation_report(validation_results)
    
    def interactive_validation_review(self, validation_results):
        """Interactive review of calibration quality."""
        return self.calibration_utils.interactive_validation_review(validation_results)
    
    def advanced_calibration_metrics(self, eye_in_hand_transform):
        """Calculate advanced calibration metrics."""
        return self.calibration_utils.advanced_calibration_metrics(eye_in_hand_transform)
    
    def load_robot_poses(self):
        """Load predefined robot poses from JSON file."""
        return self.robot_utils.load_robot_poses()
    
    def move_robot_to_next_pose(self):
        """Move robot to the next preset pose."""
        return self.robot_utils.move_robot_to_next_pose()
    
    def get_robot_movement_status(self):
        """Get current robot movement status for display."""
        return self.robot_utils.get_robot_movement_status()
    
    def detect_and_estimate_pose(self, image):
        """Detect CharUco and estimate pose using transformer."""
        # Use the modular CharUco transformer for detection and pose estimation
        results = self.charuco_transformer.detect_charuco_pose(image)
        
        # Convert to expected format for backward compatibility
        formatted_results = {
            'marker_corners': None,  # Not provided by transformer
            'marker_ids': None,      # Not provided by transformer  
            'charuco_corners': results.get('charuco_corners'),
            'charuco_ids': results.get('charuco_ids'),
            'rvec': results.get('rvec'),
            'tvec': results.get('tvec'),
            'valid_pose': results.get('valid_pose', False),
            'num_corners': results.get('num_corners', 0),
            'subpixel_corners': results.get('subpixel_corners', False)
        }
        
        return formatted_results
    
    def draw_results(self, image, results):
        """Draw detection and pose estimation results."""
        vis_image = image.copy()
        
        # Draw ArUco markers
        if results['marker_ids'] is not None:
            vis_image = aruco.drawDetectedMarkers(vis_image, results['marker_corners'], results['marker_ids'])
        
        # Draw CharUco corners
        if results['charuco_corners'] is not None:
            vis_image = aruco.drawDetectedCornersCharuco(vis_image, results['charuco_corners'], results['charuco_ids'])
        
        # Draw pose axes
        if results['valid_pose']:
            cv2.drawFrameAxes(
                vis_image, self.charuco_transformer.camera_matrix, self.charuco_transformer.dist_coeffs,
                results['rvec'], results['tvec'], self.charuco_transformer.square_length
            )
            
            # Add pose information text
            rotation_matrix = cv2.Rodrigues(results['rvec'])[0]
            translation = results['tvec'].flatten()
            
            # Display translation
            cv2.putText(vis_image, f"T: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display rotation (Euler angles)
            sy = math.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] +  rotation_matrix[1,0] * rotation_matrix[1,0])
            singular = sy < 1e-6
            
            if not singular:
                x = math.atan2(rotation_matrix[2,1], rotation_matrix[2,2])
                y = math.atan2(-rotation_matrix[2,0], sy)
                z = math.atan2(rotation_matrix[1,0], rotation_matrix[0,0])
            else:
                x = math.atan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                y = math.atan2(-rotation_matrix[2,0], sy)
                z = 0
            
            euler_angles = [math.degrees(x), math.degrees(y), math.degrees(z)]
            cv2.putText(vis_image, f"R: [{euler_angles[0]:.1f}, {euler_angles[1]:.1f}, {euler_angles[2]:.1f}]",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add detection status
        if results['valid_pose']:
            status_text = "CharUco pose estimated"
            color = (0, 255, 0)
        elif results['charuco_corners'] is not None:
            num_corners = len(results['charuco_corners'])
            status_text = f"CharUco corners: {num_corners} (need >=4 for pose)"
            color = (0, 255, 255) if num_corners >= 4 else (0, 165, 255)
        elif results['marker_ids'] is not None:
            num_markers = len(results['marker_ids'])
            status_text = f"ArUco markers: {num_markers} (searching CharUco...)"
            color = (255, 165, 0)
        else:
            status_text = "Searching for CharUco pattern..."
            color = (0, 0, 255)
            
        cv2.putText(vis_image, status_text, (10, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add calibration info
        calibration_info = self.charuco_transformer.get_calibration_info()
        calib_text = f"Calibration points: {calibration_info['calibration_points_collected']}"
        cv2.putText(vis_image, calib_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add robot movement status
        robot_status = self.get_robot_movement_status()
        cv2.putText(vis_image, robot_status, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Add coordinate system info
        coord_info = self.get_coordinate_transform_info()
        coord_text = f"Origin: {coord_info['origin']}"
        cv2.putText(vis_image, coord_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        # Add instructions
        cv2.putText(vis_image, "Controls: 's'=collect, 'c'=move robot, 'q'=calibrate", (10, image.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image
    
    def save_results(self, image, results):
        """Save current frame and results."""
        # Save using utility with custom output directory name
        output_dir = self.detection_utils.save_results(image, results, "realtime_output")
        
        # Save additional CharUco-specific data if pose is valid
        if results['valid_pose']:
            # Save corner points (using selected coordinate system)
            if results['charuco_corners'] is not None:
                if self.use_center_origin:
                    obj_points, img_points = self.match_image_points_center_origin(results['charuco_corners'], results['charuco_ids'])
                else:
                    obj_points, img_points = self.match_image_points_corrected(results['charuco_corners'], results['charuco_ids'])
                np.savetxt(output_dir / 'objPoints.txt', obj_points.reshape(-1, 3))
                np.savetxt(output_dir / 'imgPoints.txt', img_points.reshape(-1, 2))
                
                # Also save original OpenCV coordinates for comparison
                obj_points_orig, img_points_orig = self.board.matchImagePoints(results['charuco_corners'], results['charuco_ids'])
                np.savetxt(output_dir / 'objPoints_opencv_original.txt', obj_points_orig.reshape(-1, 3))
                np.savetxt(output_dir / 'imgPoints_opencv_original.txt', img_points_orig.reshape(-1, 2))
    
    def run(self):
        """Run the real-time pose estimation."""
        if not self.setup_camera():
            print("Failed to setup camera. Exiting.")
            return
        
        print("\nStarting real-time CharUco calibration...")
        print("Controls:")
        print("  's' - Collect calibration point (robot + CharUco poses)")
        print("  'c' - Move robot to next preset position")
        print("  'q' - Quit and calculate eye-in-hand calibration")
        print("  ESC - Exit without calibration")
        print("\nUse 'c' to move robot automatically, then 's' to collect calibration points.")
        print("Need at least 3 points for calibration.")
        print()
        
        try:
            while True:
                # Get frame from camera
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                # Convert to numpy array
                image = np.asanyarray(color_frame.get_data())
                
                # Detect and estimate pose
                results = self.detect_and_estimate_pose(image)
                
                # Draw results
                vis_image = self.draw_results(image, results)
                
                # Display
                cv2.imshow('CharUco Real-time Pose Estimation', vis_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # 'q' - quit and calibrate
                    break
                elif key == 27:  # ESC - exit without calibration
                    print("Exiting without calibration")
                    self.pipeline.stop()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('s'):  # 's' - collect calibration point
                    success = self.collect_calibration_point(color_image)
                    if success:
                        self.frame_count += 1
                elif key == ord('c'):  # 'c' - move robot to next pose
                    print("\nMoving robot to next position...")
                    self.move_robot_to_next_pose()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
            # Perform calibration if we have enough points
            calibration_info = self.charuco_transformer.get_calibration_info()
            if calibration_info['calibration_points_collected'] >= 3:
                while True:  # Allow interactive validation loop
                    print(f"\nPerforming eye-in-hand calibration with {calibration_info['calibration_points_collected']} points...")
                    
                    eye_in_hand_transform = self.calculate_eye_in_hand_calibration()
                    
                    if eye_in_hand_transform is not None:
                        # Validate calibration quality using transformer
                        validation_results = self.validate_calibration()
                        
                        # Skip advanced metrics - handled by transformer validation
                        
                        # Interactive validation review
                        decision = self.interactive_validation_review(validation_results)
                        
                        if decision is True:  # Accept calibration
                            # Save calibration results using transformer
                            output_dir = self.save_calibration_results()
                            
                            # Save validation results too
                            if validation_results:
                                validation_file = output_dir / 'validation_report.json'
                                with open(validation_file, 'w') as f:
                                    json.dump(validation_results, f, indent=2)
                            
                            print("\n" + "="*60)
                            print("CALIBRATION COMPLETE!")
                            print("="*60)
                            print(f"Eye-in-hand transformation matrix:")
                            print(eye_in_hand_transform)
                            print(f"Quality: {validation_results['quality_score'] if validation_results else 'Unknown'}")
                            print(f"\nResults saved to: {output_dir}")
                            print(f"Use this file for robot-camera coordination:")
                            print(f"  {output_dir / 'eye_in_hand_transform.npy'}")
                            print("="*60)
                            break
                            
                        elif decision is False:  # Continue collecting or reset
                            calibration_info = self.charuco_transformer.get_calibration_info()
                            if calibration_info['calibration_points_collected'] == 0:  # Reset was chosen
                                print("Calibration data cleared. Returning to data collection...")
                                break
                            else:  # Continue collecting chosen
                                print("Returning to camera view for more data collection...")
                                # Return to camera loop for more points
                                return self.run()  # Recursive call to restart collection
                                
                        elif decision is None:  # Quit without saving
                            print("Exiting without saving calibration...")
                            break
                    else:
                        print("Calibration failed - check your data points")
                        break
            else:
                calibration_info = self.charuco_transformer.get_calibration_info()
                print(f"Not enough calibration points ({calibration_info['calibration_points_collected']}/3) - no calibration performed")
            
            print("Real-time calibration system stopped")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time CharUco pose estimation")
    parser.add_argument('--config-path', type=str, default='charuco-pose-estimation',
                       help='Path to configuration directory')
    parser.add_argument('--square-size', type=float, default=0.039,
                       help='CharUco square size in meters (default: 0.039)')
    parser.add_argument('--marker-size', type=float, default=0.029,
                       help='ArUco marker size in meters (default: 0.029)')
    parser.add_argument('--board-size', nargs=2, type=int, default=[7, 5],
                       help='CharUco board dimensions (default: 7 5)')
    parser.add_argument('--center-origin', action='store_true',
                       help='Use center of board as coordinate origin instead of bottom-left corner')
    
    args = parser.parse_args()
    
    # Create pose estimation system
    pose_estimator = RealtimeCharucoPose(args.config_path, args.center_origin)
    
    # Update parameters from command line
    pose_estimator.square_length = args.square_size
    pose_estimator.marker_length = args.marker_size
    pose_estimator.board_size = tuple(args.board_size)
    
    # Recreate board with new parameters
    pose_estimator.board = aruco.CharucoBoard(
        pose_estimator.board_size, 
        pose_estimator.square_length, 
        pose_estimator.marker_length, 
        pose_estimator.dictionary
    )
    
    # Run the system
    pose_estimator.run()


if __name__ == "__main__":
    main()