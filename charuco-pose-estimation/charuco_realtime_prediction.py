#!/usr/bin/env python3
"""
Real-time CharUco Position Prediction

Real-time prediction of CharUco pattern position relative to robot arm in robot coordinates.
Uses pre-calibrated eye-in-hand transformation for accurate position prediction.

Features:
- Real-time CharUco position prediction relative to robot base
- Robot movement presets for different viewpoints
- Live visualization with position display
- Robot coordinate system (X=forward, Y=left, Z=up)

Usage:
    python charuco_realtime_prediction.py [options]

Controls:
    'c' - Move robot to next preset position
    'q' - Quit application
    ESC - Exit
"""

import sys
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv2 import aruco

# Import utilities and new modular transformer
from utils import RobotUtils, CameraUtils, DetectionUtils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from charuco_transformer import CharUcoTransformer


class RealtimeCharucoPredictor:
    """Real-time CharUco position prediction system."""
    
    def __init__(self, config_path="charuco-pose-estimation", use_center_origin=False):
        """Initialize the real-time pose estimation system.
        
        Args:
            config_path: Path to configuration directory
            use_center_origin: If True, use center of board as origin. If False, use bottom-left corner.
        """
        self.config_path = Path(config_path)
        
        # Find latest calibration directory
        calibration_dir = self._find_latest_calibration_dir(config_path)
        
        # Initialize the modular CharUco transformer for prediction mode
        self.charuco_transformer = CharUcoTransformer(
            calibration_path=calibration_dir,
            board_size=(7, 5),
            square_length=0.039,  # 39mm squares
            marker_length=0.029,  # 29mm markers
            use_center_origin=use_center_origin
        )
        
        # Initialize remaining utility classes
        self.robot_utils = RobotUtils()
        self.camera_utils = CameraUtils(config_path)
        self.detection_utils = DetectionUtils(config_path)
        
        # Load robot poses
        self.robot_utils.load_robot_poses()
        
        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Frame counter for saving
        self.frame_count = 0
        
        # Print initialization status
        if self.charuco_transformer.is_ready_for_prediction():
            print("[OK] CharUco Transformer initialized for real-time prediction")
            calibration_info = self.charuco_transformer.get_calibration_info()
            board_config = calibration_info['board_config']
            print(f"     Board: {board_config['size']} squares, {board_config['square_length_mm']:.1f}mm squares")
            print(f"     Origin: {'Center' if board_config['use_center_origin'] else 'Bottom-left'}")
            if 'transformation_info' in calibration_info:
                print(f"     Eye-in-hand transformation loaded")
        else:
            print("[ERROR] CharUco Transformer not ready for prediction - check calibration files")
    
    def _find_latest_calibration_dir(self, config_path):
        """Find the latest calibration directory."""
        config_dir = Path(config_path)
        calibration_dirs = list(config_dir.glob('calibration_result_*'))
        if calibration_dirs:
            # Sort by timestamp in directory name
            calibration_dirs.sort(key=lambda x: int(x.name.split('_')[-1]))
            latest_dir = calibration_dirs[-1]
            print(f"[INFO] Using calibration: {latest_dir}")
            return latest_dir
        else:
            # Try default directory
            default_dir = config_dir / 'calibration_result_1754622347'
            if default_dir.exists():
                print(f"[INFO] Using default calibration: {default_dir}")
                return default_dir
            else:
                print("[ERROR] No calibration directory found")
                return None
    
    def setup_detectors(self):
        """Setup detectors - deprecated, now handled by transformer."""
        # Detectors are now handled by the CharUco transformer
        pass
    
    def predict_charuco_position(self, image, robot_pose):
        """
        Predict CharUco position in robot base coordinates using transformer.
        
        Args:
            image: Input BGR image
            robot_pose: Current robot pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            
        Returns:
            np.array: CharUco position in robot base coordinates [x, y, z] in mm or None if not detected
        """
        return self.charuco_transformer.predict_robot_position(image, robot_pose, output_units='mm')
    
    def update_position_history(self, position):
        """Update position history for stability analysis."""
        return self.charuco_transformer._update_position_history(position)
    
    def get_position_stability(self):
        """Get position stability metrics."""
        return self.charuco_transformer.get_position_stability()
    
    def draw_prediction_results(self, image, results, robot_pose, position_in_base):
        """
        Draw real-time prediction results on the image.
        
        Args:
            image: Input image
            results: Detection results from detect_and_estimate_pose
            robot_pose: Current robot pose information
            position_in_base: Predicted CharUco position in robot base coordinates (mm)
            
        Returns:
            np.array: Image with prediction results overlaid
        """
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        # Draw CharUco detection results
        if results['valid_pose']:
            # Draw detected corners and axes
            if 'charuco_corners' in results and results['charuco_corners'] is not None:
                aruco.drawDetectedCornersCharuco(vis_image, results['charuco_corners'], 
                                                results['charuco_ids'], (0, 255, 0))
            
            # Draw coordinate axes on the CharUco pattern
            if results['rvec'] is not None and results['tvec'] is not None:
                cv2.drawFrameAxes(vis_image, self.charuco_transformer.camera_matrix, self.charuco_transformer.dist_coeffs,
                                results['rvec'], results['tvec'], self.charuco_transformer.square_length, 3)
        
        # Create overlay for text information
        overlay = vis_image.copy()
        alpha = 0.8
        
        # Main info panel
        panel_width = 500
        panel_height = 200
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (255, 255, 255), 2)
        
        # Blend overlay
        vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)
        
        # Title
        cv2.putText(vis_image, "Real-time CharUco Position Prediction", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Detection status
        y = 60
        if results['valid_pose']:
            cv2.putText(vis_image, "Status: CharUco DETECTED", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show position prediction if available
            if position_in_base is not None:
                y += 25
                cv2.putText(vis_image, f"Position in Robot Base (mm):", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 20
                cv2.putText(vis_image, f"  X(forward): {position_in_base[0]:7.1f}", (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                y += 18
                cv2.putText(vis_image, f"  Y(left):    {position_in_base[1]:7.1f}", (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                y += 18
                cv2.putText(vis_image, f"  Z(up):      {position_in_base[2]:7.1f}", (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
                
                # Position stability
                mean_pos, std_pos = self.get_position_stability()
                if mean_pos is not None:
                    y += 22
                    cv2.putText(vis_image, f"Stability (std): {std_pos[0]:.1f}, {std_pos[1]:.1f}, {std_pos[2]:.1f}",
                               (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            else:
                y += 20
                cv2.putText(vis_image, "Robot not available for prediction", (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(vis_image, "Status: Searching for CharUco...", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Robot status
        y += 30
        if robot_pose is not None:
            cv2.putText(vis_image, f"Robot Connected: Position available", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(vis_image, f"Robot: Not connected", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        
        # Controls info
        cv2.putText(vis_image, "Controls: 'c'=Move robot, 'q'=Quit", 
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_image, "Using pre-calibrated eye-in-hand transformation", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image
    
    def setup_camera(self):
        """Setup RealSense camera with optimized settings."""
        success = self.camera_utils.setup_camera(self.pipeline, self.config)
        if success:
            # Update detectors with new camera matrix
            self.setup_detectors()
        return success
    
    def check_subpixel(self, corners):
        """Check if corners are subpixel accurate."""
        return self.detection_utils.check_subpixel(corners)
    
    def match_image_points_corrected(self, charuco_corners, charuco_ids):
        """Match image points with corrected coordinate system."""
        return self.coordinate_utils.match_image_points_corrected(charuco_corners, charuco_ids)
    
    def convert_to_center_origin_simple(self, tvec, rvec):
        """Convert bottom-left origin tvec to center origin using simple offset approach."""
        return self.coordinate_utils.convert_to_center_origin_simple(tvec, rvec)
    
    def get_robot_pose(self):
        """Get current robot end-effector pose."""
        return self.robot_utils.get_robot_pose()
    
    def pose_to_transform_matrix(self, pose):
        """Convert [x, y, z, rx, ry, rz] pose to 4x4 transformation matrix."""
        return self.coordinate_utils.pose_to_transform_matrix(pose)
    
    def charuco_pose_to_transform_matrix(self, rvec, tvec):
        """Convert CharUco rvec, tvec to 4x4 transformation matrix with coordinate correction."""
        return self.coordinate_utils.charuco_pose_to_transform_matrix(rvec, tvec)
    
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
        """Detect CharUco and estimate pose."""
        # Detect ArUco markers
        marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(image)
        
        results = {
            'marker_corners': marker_corners,
            'marker_ids': marker_ids,
            'charuco_corners': None,
            'charuco_ids': None,
            'rvec': None,
            'tvec': None,
            'valid_pose': False
        }
        
        if marker_ids is not None and len(marker_ids) > 0:
            # Detect CharUco corners
            charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(
                image, None, None, marker_corners, marker_ids
            )
            
            results['charuco_corners'] = charuco_corners
            results['charuco_ids'] = charuco_ids
            
            # Estimate pose if we have enough corners
            if charuco_corners is not None and len(charuco_corners) >= 4:
                # Always use bottom-left origin for detection (simpler and consistent)
                # Center origin mode applies simple offset after detection
                obj_points, img_points = self.match_image_points_corrected(charuco_corners, charuco_ids)
                
                # Try different PnP algorithms based on number of points
                valid = False
                rvec = None
                tvec = None
                
                if len(obj_points) >= 6:
                    # Use default DLT algorithm for 6+ points
                    valid, rvec, tvec = cv2.solvePnP(
                        obj_points, img_points, self.camera_utils.camera_matrix, self.camera_utils.dist_coeffs
                    )
                elif len(obj_points) >= 4:
                    # Use P3P algorithm for 4-5 points
                    try:
                        valid, rvec, tvec = cv2.solvePnP(
                            obj_points, img_points, self.camera_utils.camera_matrix, self.camera_utils.dist_coeffs,
                            flags=cv2.SOLVEPNP_P3P
                        )
                    except cv2.error:
                        # Fallback to EPNP if P3P fails
                        try:
                            valid, rvec, tvec = cv2.solvePnP(
                                obj_points, img_points, self.camera_utils.camera_matrix, self.camera_utils.dist_coeffs,
                                flags=cv2.SOLVEPNP_EPNP
                            )
                        except cv2.error:
                            valid = False
                
                if valid and rvec is not None and tvec is not None:
                    # Apply simple center origin conversion if requested
                    if self.use_center_origin:
                        tvec = self.convert_to_center_origin_simple(tvec, rvec)
                    
                    results['rvec'] = rvec
                    results['tvec'] = tvec
                    results['valid_pose'] = True
        
        return results
    
    def draw_results(self, image, results):
        """Draw detection and pose estimation results."""
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        # Draw ArUco markers
        if results['marker_ids'] is not None:
            vis_image = aruco.drawDetectedMarkers(vis_image, results['marker_corners'], results['marker_ids'])
        
        # Draw CharUco corners
        if results['charuco_corners'] is not None:
            vis_image = aruco.drawDetectedCornersCharuco(vis_image, results['charuco_corners'], results['charuco_ids'])
        
        # Draw pose axes
        if results['valid_pose']:
            cv2.drawFrameAxes(
                vis_image, self.camera_utils.camera_matrix, self.camera_utils.dist_coeffs,
                results['rvec'], results['tvec'], self.square_length
            )
            
            # Add pose information text
            rotation_matrix = cv2.Rodrigues(results['rvec'])[0]
            translation = results['tvec'].flatten()
            
            # Display translation
            cv2.putText(vis_image, f"T: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display rotation (Euler angles)
            import math
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
            
        cv2.putText(vis_image, status_text, (10, image.shape[0] - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add robot movement status
        robot_status = self.get_robot_movement_status()
        cv2.putText(vis_image, robot_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add origin mode info
        origin_mode = "Center Origin" if self.use_center_origin else "Bottom-Left Origin"
        cv2.putText(vis_image, f"Origin Mode: {origin_mode}", (10, h - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls info
        cv2.putText(vis_image, "Controls: 'c'=Move robot, 'q'=Quit", 
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_image, "Using pre-calibrated eye-in-hand transformation", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image
    
    def save_results(self, image, results):
        """Save current frame and results."""
        timestamp = int(time.time())
        output_dir = self.config_path / f"realtime_output_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        # Save original image
        cv2.imwrite(str(output_dir / "image.png"), image)
        
        # Save visualization
        vis_image = self.draw_results(image, results)
        cv2.imwrite(str(output_dir / "detection.png"), vis_image)
        
        # Save pose data if available
        if results['valid_pose']:
            # Save transformation matrices
            rotation_matrix = cv2.Rodrigues(results['rvec'])[0]
            tvec = results['tvec']
            
            # CharUco transformation matrix
            charuco_tf = np.concatenate((
                np.concatenate((rotation_matrix, tvec), axis=1),
                np.array([[0, 0, 0, 1]])
            ))
            
            # Camera transformation matrix
            camera_position = -np.matrix(rotation_matrix).T * np.matrix(tvec)
            camera_tf = np.concatenate((
                np.concatenate((rotation_matrix.T, camera_position), axis=1),
                np.array([[0, 0, 0, 1]])
            ))
            
            np.savetxt(output_dir / 'charuco_tf.txt', charuco_tf)
            np.savetxt(output_dir / 'camera_tf.txt', camera_tf)
            
            # Save corner points (using corrected coordinate system)
            if results['charuco_corners'] is not None:
                obj_points, img_points = self.match_image_points_corrected(results['charuco_corners'], results['charuco_ids'])
                np.savetxt(output_dir / 'objPoints.txt', obj_points.reshape(-1, 3))
                np.savetxt(output_dir / 'imgPoints.txt', img_points.reshape(-1, 2))
                
                # Also save original OpenCV coordinates for comparison
                obj_points_orig, img_points_orig = self.board.matchImagePoints(results['charuco_corners'], results['charuco_ids'])
                np.savetxt(output_dir / 'objPoints_opencv_original.txt', obj_points_orig.reshape(-1, 3))
                np.savetxt(output_dir / 'imgPoints_opencv_original.txt', img_points_orig.reshape(-1, 2))
            
            print(f"Results saved to {output_dir}")
        else:
            print(f"Image saved to {output_dir} (no pose detected)")
    
    def run(self):
        """Run the real-time position prediction."""
        if not self.setup_camera():
            print("Failed to setup camera. Exiting.")
            return
        
        print("\nStarting real-time CharUco position prediction...")
        print("Controls:")
        print("  'c' - Move robot to next preset position")
        print("  'q' - Quit application")
        print("  ESC - Exit")
        print("\nReal-time prediction active - CharUco position will be shown in robot coordinates.")
        print("Robot coordinates: X=forward(+), Y=left(+), Z=up(+), units in mm")
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
                
                # Get current robot position for prediction
                robot_pose = self.get_robot_pose()
                
                # Predict CharUco position if detected
                position_in_base = None
                if results['valid_pose'] and robot_pose is not None:
                    # Create camera-to-charuco transformation matrix from rvec and tvec
                    # tvec is already converted to center origin if needed, so always use standard transformation
                    charuco_transform = self.charuco_pose_to_transform_matrix(results['rvec'], results['tvec'])
                    # Convert robot pose to transformation matrix
                    robot_transform = self.pose_to_transform_matrix(robot_pose)
                    
                    # Predict position in robot base coordinates
                    position_in_base = self.predict_charuco_position(charuco_transform, robot_transform)
                    self.update_position_history(position_in_base)
                
                # Draw results with position prediction
                vis_image = self.draw_prediction_results(image, results, robot_pose, position_in_base)
                
                # Display
                cv2.imshow('CharUco Real-time Position Prediction', vis_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # 'q' - quit
                    break
                elif key == 27:  # ESC - exit
                    print("Exiting...")
                    self.pipeline.stop()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('c'):  # 'c' - move robot to next pose
                    print("\nMoving robot to next position...")
                    self.move_robot_to_next_pose()
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
            print("Real-time position prediction stopped")


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
    
    # Create prediction system
    pose_estimator = RealtimeCharucoPredictor(args.config_path, args.center_origin)
    
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