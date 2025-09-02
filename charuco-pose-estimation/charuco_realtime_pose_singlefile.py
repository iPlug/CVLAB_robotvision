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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv2 import aruco

# Import utilities
from utils import RobotUtils, CameraUtils, CalibrationUtils, CoordinateUtils, DetectionUtils


class RealtimeCharucoPose:
    """Real-time CharUco pose estimation system."""
    
    def __init__(self, config_path="charuco-pose-estimation", use_center_origin=False):
        """Initialize the real-time pose estimation system.
        
        Args:
            config_path: Path to configuration directory
            use_center_origin: If True, use center of board as origin. If False, use bottom-left corner.
        """
        self.config_path = Path(config_path)
        
        # CharUco board parameters (measured values)
        self.square_length = 0.039  # 39mm squares
        self.marker_length = 0.029  # 29mm markers
        self.board_size = (7, 5)   # 7x5 grid
        
        # Coordinate system option
        self.use_center_origin = use_center_origin
        
        # ArUco/CharUco setup
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.board = aruco.CharucoBoard(self.board_size, self.square_length, self.marker_length, self.dictionary)
        
        # Initialize utility classes
        self.robot_utils = RobotUtils()
        self.camera_utils = CameraUtils(config_path)
        self.calibration_utils = CalibrationUtils(config_path)
        self.coordinate_utils = CoordinateUtils(self.board_size, self.square_length, self.marker_length)
        self.detection_utils = DetectionUtils(config_path)
        
        # Load robot poses
        self.robot_utils.load_robot_poses()
        
        # Load camera intrinsics
        self.camera_utils.load_camera_intrinsics()
        
        # Setup detectors with camera matrix
        self.setup_detectors()
        
        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Frame counter for saving
        self.frame_count = 0
        
        # Print coordinate system information
        coord_info = self.get_coordinate_transform_info()
        print(f"Coordinate system: {coord_info['origin']} origin")
        print(f"Description: {coord_info['description']}")
        print(f"Axes: {coord_info['axes']}")
        
    def setup_detectors(self):
        """Setup ArUco and CharUco detectors."""
        # ArUco detector parameters
        detector_params = aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        self.aruco_detector = aruco.ArucoDetector(self.dictionary, detector_params)
        
        # CharUco detector parameters
        charuco_params = aruco.CharucoParameters()
        if self.camera_utils.camera_matrix is not None:
            charuco_params.cameraMatrix = self.camera_utils.camera_matrix
        self.charuco_detector = aruco.CharucoDetector(self.board, charuco_params)
    
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
        """
        Match image points with corrected coordinate system (bottom-left origin, outward Z).
        This matches the notebook's coordinate system and standard robotics convention.
        """
        return self.coordinate_utils.match_image_points_corrected(charuco_corners, charuco_ids)
    
    def match_image_points_center_origin(self, charuco_corners, charuco_ids):
        """
        Match image points with center-based coordinate system (center origin, outward Z).
        The origin is at the center of the center marker of the CharUco board.
        """
        objPoints = []
        imgPoints = []
        
        # Calculate the true geometric center of the board
        # For a 7x5 board, we need to find the center of the center square
        center_x = (self.board_size[0] - 1) * self.square_length / 2
        center_y = (self.board_size[1] - 1) * self.square_length / 2
        
        # Add half square length to move from corner to CENTER of the center square
        center_x += self.square_length / 2
        center_y += self.square_length / 2
        
        center_offset = np.array([[center_x, center_y, 0]])
        
        for i in range(len(charuco_ids)):
            index = charuco_ids[i]
            # Get the corner position from the board
            corner_pos = self.board.getChessboardCorners()[index]
            
            # Apply coordinate system correction (bottom-left origin first)
            # Flip Y-axis to make bottom-left the origin (instead of top-left)
            corrected_pos = corner_pos.copy()
            corrected_pos[0][1] = self.board.getRightBottomCorner()[1] - corrected_pos[0][1]
            
            # Then translate to center origin (center of center marker)
            centered_pos = corrected_pos - center_offset
            
            objPoints.append(centered_pos)
            imgPoints.append(charuco_corners[i])
        
        return np.array(objPoints), np.array(imgPoints)
    
    def get_coordinate_transform_info(self):
        """Get information about the current coordinate system."""
        if self.use_center_origin:
            center_x = (self.board_size[0] - 1) * self.square_length / 2 + self.square_length / 2
            center_y = (self.board_size[1] - 1) * self.square_length / 2 + self.square_length / 2
            return {
                'origin': 'center',
                'description': f'Center of center marker (offset: {center_x:.3f}m, {center_y:.3f}m)',
                'axes': 'X=right, Y=up, Z=outward'
            }
        else:
            return {
                'origin': 'bottom-left',
                'description': 'Bottom-left corner of CharUco board',
                'axes': 'X=right, Y=up, Z=outward'
            }
    
    def get_robot_pose(self):
        """Get current robot end-effector pose."""
        if self.robot is not None:
            # Check robot connection status
            if hasattr(self.robot, 'is_connected') and not self.robot.is_connected:
                print("Debug: Robot controller exists but is not connected")
                print("Tip: Make sure myCobot is powered on and connected via USB")
            else:
                try:
                    # Get robot pose [x, y, z, rx, ry, rz]
                    pose = self.robot.get_current_position()
                    print(f"Debug: Robot returned pose: {pose}")
                    if pose and len(pose) >= 6:
                        print(f"Using robot pose: {pose[:6]}")
                        return pose[:6]  # [x, y, z, rx, ry, rz]
                    else:
                        print(f"Debug: Invalid pose data - pose={pose}, length={len(pose) if pose else 0}")
                        print("Tip: Robot may need to be powered on or moved manually first")
                except Exception as e:
                    print(f"Failed to get robot pose: {e}")
        else:
            print("Debug: No robot controller available")
        
        # Fallback: manual input
        print("\nRobot pose not available. Enter robot end-effector pose manually:")
        try:
            x = float(input("X (mm): "))
            y = float(input("Y (mm): "))
            z = float(input("Z (mm): "))
            rx = float(input("RX (degrees): "))
            ry = float(input("RY (degrees): "))
            rz = float(input("RZ (degrees): "))
            return [x, y, z, rx, ry, rz]
        except (ValueError, KeyboardInterrupt):
            print("Input cancelled, skipping pose collection")
            return None
    
    def pose_to_transform_matrix(self, pose):
        """Convert [x, y, z, rx, ry, rz] pose to 4x4 transformation matrix."""
        x, y, z, rx, ry, rz = pose
        
        # Convert to meters and radians
        x_m, y_m, z_m = x/1000.0, y/1000.0, z/1000.0
        rx_rad, ry_rad, rz_rad = np.radians([rx, ry, rz])
        
        # Create rotation matrix from Euler angles (XYZ intrinsic order - test original)
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_euler('xyz', [rx_rad, ry_rad, rz_rad])
        rot_matrix = rotation.as_matrix()
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = [x_m, y_m, z_m]
        
        return transform
    
    def charuco_pose_to_transform_matrix(self, rvec, tvec):
        """Convert CharUco rvec, tvec to 4x4 transformation matrix with coordinate correction."""
        # Convert rotation vector to rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)
        
        # Apply coordinate system correction: 90° rotation around Z-axis
        # This fixes the systematic 95° rotation error by aligning CharUco axes with robot axes
        # Based on analysis: CharUco X/Y axes need 90° Z-rotation to match robot coordinates
        correction_matrix = np.array([
            [0, -1, 0],  # 90° rotation around Z
            [1,  0, 0], 
            [0,  0, 1]
        ])
        
        # Apply correction to rotation matrix
        corrected_rot_matrix = rot_matrix @ correction_matrix
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = corrected_rot_matrix
        transform[:3, 3] = tvec.flatten()
        
        return transform
    
    def collect_calibration_point(self, results):
        """Collect robot and CharUco poses for calibration."""
        if not results['valid_pose']:
            print("No CharUco pose detected - cannot collect calibration point")
            return False
        
        print(f"\nCollecting calibration point {len(self.calibration_pairs) + 1}...")
        
        # Get robot pose
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return False
        
        # Get CharUco pose
        charuco_rvec = results['rvec']
        charuco_tvec = results['tvec']
        
        # Convert to transformation matrices
        robot_transform = self.pose_to_transform_matrix(robot_pose)
        charuco_transform = self.charuco_pose_to_transform_matrix(charuco_rvec, charuco_tvec)
        
        # Store calibration pair
        self.calibration_pairs.append({
            'robot_pose': robot_pose,
            'robot_transform': robot_transform,
            'charuco_rvec': charuco_rvec,
            'charuco_tvec': charuco_tvec,
            'charuco_transform': charuco_transform,
            'timestamp': time.time()
        })
        
        print(f"Calibration point collected: Robot=[{robot_pose[0]:.1f}, {robot_pose[1]:.1f}, {robot_pose[2]:.1f}]")
        print(f"Total points: {len(self.calibration_pairs)}")
        
        return True
    
    def calculate_eye_in_hand_calibration(self):
        """Calculate eye-in-hand calibration using collected data points."""
        if len(self.calibration_pairs) < 3:
            print(f"Need at least 3 calibration points, have {len(self.calibration_pairs)}")
            return None
        
        print(f"\nCalculating eye-in-hand calibration from {len(self.calibration_pairs)} points...")
        
        # Prepare data for OpenCV calibrateHandEye
        robot_transforms = []
        charuco_transforms = []
        
        for pair in self.calibration_pairs:
            robot_transforms.append(pair['robot_transform'])
            charuco_transforms.append(pair['charuco_transform'])
        
        # Convert to OpenCV format (rotation matrices and translation vectors)
        R_gripper2base = [T[:3, :3] for T in robot_transforms]
        t_gripper2base = [T[:3, 3].reshape(3, 1) for T in robot_transforms]
        R_target2cam = [T[:3, :3] for T in charuco_transforms]
        t_target2cam = [T[:3, 3].reshape(3, 1) for T in charuco_transforms]
        
        # Calculate hand-eye calibration
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # Create 4x4 transformation matrix
            eye_in_hand_transform = np.eye(4)
            eye_in_hand_transform[:3, :3] = R_cam2gripper
            eye_in_hand_transform[:3, 3] = t_cam2gripper.flatten()
            
            print("Eye-in-hand calibration successful!")
            print(f"Transformation matrix:\n{eye_in_hand_transform}")
            
            return eye_in_hand_transform
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return None
    
    def save_calibration_results(self, eye_in_hand_transform):
        """Save calibration results to file."""
        timestamp = int(time.time())
        output_dir = self.config_path / f"calibration_result_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        # Save transformation matrix
        np.savetxt(output_dir / 'eye_in_hand_transform.txt', eye_in_hand_transform)
        np.save(output_dir / 'eye_in_hand_transform.npy', eye_in_hand_transform)
        
        # Save camera intrinsics used for this calibration
        np.savetxt(output_dir / 'camera_intrinsics.txt', self.camera_matrix)
        
        # Create a README file explaining the intrinsics requirement
        readme_content = f"""CALIBRATION RESULT - {timestamp}

IMPORTANT: This calibration result MUST be used with the specific camera intrinsics.

Files:
- eye_in_hand_transform.npy: Main transformation matrix
- camera_intrinsics.txt: Camera intrinsics used during calibration
- calibration_data.json: Complete calibration data including intrinsics

USAGE:
When using this transformation matrix for prediction, you MUST load and use 
the camera intrinsics from camera_intrinsics.txt, NOT from intrinsic.txt
or RealSense SDK intrinsics.

Camera Matrix Used:
{self.camera_matrix}

Distortion Coefficients Used:
{self.dist_coeffs}

CharUco Board Parameters:
- Size: {self.board_size}
- Square Length: {self.square_length}m
- Marker Length: {self.marker_length}m

Coordinate System:
- Origin: {self.get_coordinate_transform_info()['origin']}
- Description: {self.get_coordinate_transform_info()['description']}
- Axes: {self.get_coordinate_transform_info()['axes']}

Number of calibration points: {len(self.calibration_pairs)}
"""
        
        with open(output_dir / 'README.txt', 'w') as f:
            f.write(readme_content)
        
        # Save calibration data including camera intrinsics
        calibration_data = {
            'timestamp': timestamp,
            'num_points': len(self.calibration_pairs),
            'camera_intrinsics': {
                'camera_matrix': self.camera_matrix.tolist(),
                'dist_coeffs': self.dist_coeffs.tolist(),
                'source': 'intrinsic.txt',
                'note': 'These intrinsics were used during calibration and MUST be used with this transformation matrix'
            },
            'charuco_board': {
                'board_size': self.board_size,
                'square_length': self.square_length,
                'marker_length': self.marker_length
            },
            'coordinate_system': self.get_coordinate_transform_info(),
            'eye_in_hand_transform': eye_in_hand_transform.tolist(),
            'points': []
        }
        
        for i, pair in enumerate(self.calibration_pairs):
            point_data = {
                'id': i + 1,
                'robot_pose': pair['robot_pose'],
                'robot_transform': pair['robot_transform'].tolist(),
                'charuco_rvec': pair['charuco_rvec'].tolist(),
                'charuco_tvec': pair['charuco_tvec'].tolist(),
                'charuco_transform': pair['charuco_transform'].tolist(),
                'timestamp': pair['timestamp']
            }
            calibration_data['points'].append(point_data)
        
        # Save to JSON
        with open(output_dir / 'calibration_data.json', 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        # Save transformation matrices separately
        for i, pair in enumerate(self.calibration_pairs):
            np.savetxt(output_dir / f'robot_transform_{i+1}.txt', pair['robot_transform'])
            np.savetxt(output_dir / f'charuco_transform_{i+1}.txt', pair['charuco_transform'])
        
        print(f"Calibration results saved to: {output_dir}")
        print(f"Main transformation matrix: {output_dir / 'eye_in_hand_transform.npy'}")
        
        return output_dir
    
    def validate_calibration(self, eye_in_hand_transform):
        """Validate calibration quality using selected coordinate system."""
        if len(self.calibration_pairs) < 3:
            return None
        
        coord_system = "center" if self.use_center_origin else "bottom-left"
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
        
        # Calculate the correct transformation matrix for the selected coordinate system
        if self.use_center_origin:
            print("Calculating center origin transformation matrix...")
            correct_eye_in_hand_transform = self.calculate_eye_in_hand_for_coordinate_system(use_center=True)
            # Calculate center position based on board parameters
            center_offset_x = (self.board_size[1] - 1) * self.square_length / 2 + self.square_length / 2  # Height: X offset
            center_offset_y = (self.board_size[0] - 1) * self.square_length / 2 + self.square_length / 2  # Width: Y offset
            target_position = np.array([0.3 + center_offset_x, 0.1 - center_offset_y, 0.0])  # X+, Y- for center
            print(f"DEBUG: Center target position: [{target_position[0]*1000:.1f}, {target_position[1]*1000:.1f}, {target_position[2]*1000:.1f}]mm")
            print(f"DEBUG: Center offset: [{center_offset_x*1000:.1f}, {center_offset_y*1000:.1f}, 0.0]mm")
        else:
            print("Using bottom-left origin transformation matrix...")
            correct_eye_in_hand_transform = eye_in_hand_transform  # This should already be bottom-left
            target_position = np.array([0.3, 0.1, 0.0])  # [300mm forward, 100mm left, 0mm height] - bottom-left corner
            print(f"DEBUG: Bottom-left target position: [{target_position[0]*1000:.1f}, {target_position[1]*1000:.1f}, {target_position[2]*1000:.1f}]mm")
        
        # Track validation errors
        translation_errors = []
        rotation_errors = []
        
        for i, pair in enumerate(self.calibration_pairs):
            # Get transforms
            robot_transform = pair['robot_transform']  # Robot base to gripper
            
            # Calculate expected position using selected coordinate system
            if self.use_center_origin:
                charuco_transform = self.charuco_pose_to_transform_matrix_center(pair['charuco_rvec'], pair['charuco_tvec'])
            else:
                charuco_transform = self.charuco_pose_to_transform_matrix(pair['charuco_rvec'], pair['charuco_tvec'])
            
            expected_pattern = robot_transform @ correct_eye_in_hand_transform @ charuco_transform
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
            
            print(f"  Point {i+1}: expected [{expected_position[0]*1000:.1f}, {expected_position[1]*1000:.1f}, {expected_position[2]*1000:.1f}]mm vs target [{target_position[0]*1000:.1f}, {target_position[1]*1000:.1f}, {target_position[2]*1000:.1f}]mm -> Error: {translation_error*1000:.1f}mm")
        
        # Calculate summary statistics
        validation_results['mean_translation_error'] = np.mean(translation_errors)
        validation_results['max_reprojection_error'] = np.max(translation_errors)
        validation_results['mean_rotation_error'] = np.mean(rotation_errors)
        validation_results['reprojection_errors'] = translation_errors
        
        # Quality assessment based on pattern position accuracy
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
    
    def calculate_eye_in_hand_for_coordinate_system(self, use_center=False):
        """Calculate eye-in-hand calibration using specified coordinate system."""
        if len(self.calibration_pairs) < 3:
            return None
        
        # Prepare data for OpenCV calibrateHandEye using specified coordinate system
        robot_transforms = []
        charuco_transforms = []
        
        for pair in self.calibration_pairs:
            robot_transforms.append(pair['robot_transform'])
            
            # Recalculate CharUco transform using specified coordinate system
            charuco_rvec = pair['charuco_rvec']
            charuco_tvec = pair['charuco_tvec']
            
            # Get the stored charuco corners and IDs to recalculate pose
            # Note: We need to store the original corner data for this to work
            # For now, we'll use the existing approach but this is a limitation
            
            if use_center:
                # Apply center origin transformation
                charuco_transform = self.charuco_pose_to_transform_matrix_center(charuco_rvec, charuco_tvec)
            else:
                # Apply bottom-left origin transformation
                charuco_transform = self.charuco_pose_to_transform_matrix(charuco_rvec, charuco_tvec)
            
            charuco_transforms.append(charuco_transform)
        
        # Convert to OpenCV format (rotation matrices and translation vectors)
        R_gripper2base = [T[:3, :3] for T in robot_transforms]
        t_gripper2base = [T[:3, 3].reshape(3, 1) for T in robot_transforms]
        R_target2cam = [T[:3, :3] for T in charuco_transforms]
        t_target2cam = [T[:3, 3].reshape(3, 1) for T in charuco_transforms]
        
        # Calculate hand-eye calibration
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # Create 4x4 transformation matrix
            transform = np.eye(4)
            transform[:3, :3] = R_cam2gripper
            transform[:3, 3] = t_cam2gripper.flatten()
            
            return transform
            
        except Exception as e:
            print(f"Calibration failed for coordinate system: {e}")
            return None
    
    def charuco_pose_to_transform_matrix_center(self, rvec, tvec):
        """Convert CharUco rvec, tvec to 4x4 transformation matrix with center origin coordinate system."""
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
        # The tvec from OpenCV is relative to bottom-left, we need it relative to center
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
        if validation_results is None:
            print("No validation data available")
            return
        
        coord_system = validation_results.get('coordinate_system', 'unknown')
        print("\n" + "="*60)
        print(f"CALIBRATION QUALITY REPORT ({coord_system.upper()} ORIGIN)")
        print("="*60)
        print(f"Number of calibration points: {len(self.calibration_pairs)}")
        print(f"Coordinate system: {coord_system}")
        print(f"Mean translation error: {validation_results['mean_translation_error']:.2f} mm")
        print(f"Max translation error: {validation_results['max_reprojection_error']:.2f} mm")
        print(f"Mean rotation error: {validation_results['mean_rotation_error']:.2f}°")
        print(f"Quality assessment: {validation_results['quality_score']}")
        
        print("\nPer-point errors:")
        for i, error in enumerate(validation_results['reprojection_errors']):
            print(f"  Point {i+1}: {error:.2f} mm")
        
        print("\nQuality Guidelines:")
        print("  Excellent: <10mm translation, <5° rotation")
        print("  Good:      <25mm translation, <10° rotation")
        print("  Fair:      <50mm translation, <15° rotation")
        print("  Poor:      >50mm translation or >15° rotation")
        
        if validation_results['quality_score'] in ['Poor', 'Fair']:
            print("\nSuggestions for improvement:")
            print("  - Collect more calibration points")
            print("  - Use more diverse robot poses (different angles/distances)")
            print("  - Ensure CharUco pattern is clearly visible in all poses")
            print("  - Check for camera calibration accuracy")
            print("  - Verify robot pose accuracy")
        
        print("="*60)
    
    def interactive_validation_review(self, validation_results):
        """Interactive review of calibration quality with options to continue or recalibrate."""
        if validation_results is None:
            return True
        
        self.print_validation_report(validation_results)
        
        while True:
            print(f"\nCalibration quality: {validation_results['quality_score']}")
            print("Options:")
            print("  'a' - Accept calibration and save results")
            print("  'c' - Continue collecting more points")
            print("  'r' - Reset and start over")
            print("  'q' - Quit without saving")
            
            try:
                choice = input("Your choice (a/c/r/q): ").lower().strip()
            except KeyboardInterrupt:
                return False
            
            if choice == 'a':
                return True  # Accept calibration
            elif choice == 'c':
                print("Continue collecting calibration points...")
                return False  # Continue calibration
            elif choice == 'r':
                print("Resetting calibration data...")
                self.calibration_pairs.clear()
                return False  # Reset and continue
            elif choice == 'q':
                print("Quitting without saving...")
                return None  # Quit without saving
            else:
                print("Invalid choice. Please enter 'a', 'c', 'r', or 'q'")
    
    def advanced_calibration_metrics(self, eye_in_hand_transform):
        """Calculate advanced calibration metrics."""
        if len(self.calibration_pairs) < 4:
            return None
            
        print("\nCalculating advanced calibration metrics...")
        
        # Calculate condition number of the calibration system
        # This indicates how well-conditioned the calibration problem is
        robot_transforms = [pair['robot_transform'] for pair in self.calibration_pairs]
        
        # Check pose diversity
        positions = np.array([T[:3, 3] for T in robot_transforms])
        position_span = np.max(positions, axis=0) - np.min(positions, axis=0)
        position_diversity = np.linalg.norm(position_span)
        
        # Check rotation diversity  
        from scipy.spatial.transform import Rotation
        rotations = [Rotation.from_matrix(T[:3, :3]) for T in robot_transforms]
        
        # Calculate angular span
        angular_differences = []
        for i in range(len(rotations)):
            for j in range(i+1, len(rotations)):
                angular_diff = (rotations[i] * rotations[j].inv()).magnitude()
                angular_differences.append(np.degrees(angular_diff))
        
        max_angular_span = max(angular_differences) if angular_differences else 0
        
        metrics = {
            'position_diversity_m': position_diversity,
            'max_angular_span_deg': max_angular_span,
            'position_span_xyz': position_span * 1000,  # Convert to mm
            'num_points': len(self.calibration_pairs)
        }
        
        print(f"  Position diversity: {position_diversity*1000:.1f} mm")
        print(f"  Position span: X={position_span[0]*1000:.1f}mm, Y={position_span[1]*1000:.1f}mm, Z={position_span[2]*1000:.1f}mm") 
        print(f"  Max angular span: {max_angular_span:.1f}°")
        
        # Recommendations
        if position_diversity < 0.1:  # Less than 100mm span
            print("  Warning: Low position diversity - consider more spread-out poses")
        if max_angular_span < 30:  # Less than 30 degrees
            print("  Warning: Low rotation diversity - consider more varied orientations")
            
        return metrics
    
    def load_robot_poses(self):
        """Load predefined robot poses from JSON file."""
        poses_file = Path("recorded_coords_280.json")
        try:
            if poses_file.exists():
                with open(poses_file, 'r') as f:
                    self.preset_poses = json.load(f)
                print(f"Loaded {len(self.preset_poses)} preset robot poses")
            else:
                # Default calibration poses if file doesn't exist
                self.preset_poses = [
                    [200, 0, 300, 0, 180, 0],      # Center, looking down
                    [150, -100, 280, -20, 160, -30],  # Left side, angled
                    [150, 100, 280, 20, 160, 30],     # Right side, angled
                    [250, 0, 250, 0, 145, 0],         # Forward, tilted
                    [180, -80, 320, -15, 175, -20],   # Left high
                    [180, 80, 320, 15, 175, 20],      # Right high
                ]
                print("Using default calibration poses (recorded_coords_280.json not found)")
        except Exception as e:
            print(f"Error loading poses: {e}")
            # Fallback poses
            self.preset_poses = [
                [200, 0, 300, 0, 180, 0],
                [150, -50, 280, 0, 160, -30],
                [150, 50, 280, 0, 160, 30],
            ]
    
    def move_robot_to_next_pose(self):
        """Move robot to the next preset pose."""
        if not self.robot or not hasattr(self.robot, 'is_connected') or not self.robot.is_connected:
            print("Robot not connected - cannot move automatically")
            return False
        
        if not self.preset_poses:
            print("No preset poses available")
            return False
        
        # Move to next pose
        pose = self.preset_poses[self.current_pose_index]
        self.current_pose_index = (self.current_pose_index + 1) % len(self.preset_poses)
        
        print(f"Moving robot to pose {self.current_pose_index}/{len(self.preset_poses)}: {pose}")
        
        try:
            success = self.robot.move_to_position(pose)
            if success:
                print(f"Robot moved successfully to position {self.current_pose_index}")
                time.sleep(1)  # Give robot time to settle
                return True
            else:
                print("Robot movement failed")
                return False
        except Exception as e:
            print(f"Error moving robot: {e}")
            return False
    
    def get_robot_movement_status(self):
        """Get current robot movement status for display."""
        if not self.preset_poses:
            return "No preset poses"
        
        total_poses = len(self.preset_poses)
        current = self.current_pose_index
        next_pose = self.preset_poses[current]
        
        return f"Next pose {current+1}/{total_poses}: [{next_pose[0]:.0f}, {next_pose[1]:.0f}, {next_pose[2]:.0f}]"
    
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
                # Get object and image points with selected coordinate system
                if self.use_center_origin:
                    obj_points, img_points = self.match_image_points_center_origin(charuco_corners, charuco_ids)
                else:
                    obj_points, img_points = self.match_image_points_corrected(charuco_corners, charuco_ids)
                
                # Try different PnP algorithms based on number of points
                valid = False
                rvec = None
                tvec = None
                
                if len(obj_points) >= 6:
                    # Use default DLT algorithm for 6+ points
                    valid, rvec, tvec = cv2.solvePnP(
                        obj_points, img_points, self.camera_matrix, self.dist_coeffs
                    )
                elif len(obj_points) >= 4:
                    # Use P3P algorithm for 4-5 points
                    try:
                        valid, rvec, tvec = cv2.solvePnP(
                            obj_points, img_points, self.camera_matrix, self.dist_coeffs,
                            flags=cv2.SOLVEPNP_P3P
                        )
                    except cv2.error:
                        # Fallback to EPNP if P3P fails
                        try:
                            valid, rvec, tvec = cv2.solvePnP(
                                obj_points, img_points, self.camera_matrix, self.dist_coeffs,
                                flags=cv2.SOLVEPNP_EPNP
                            )
                        except cv2.error:
                            valid = False
                
                if valid and rvec is not None and tvec is not None:
                    results['rvec'] = rvec
                    results['tvec'] = tvec
                    results['valid_pose'] = True
        
        return results
    
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
                vis_image, self.camera_matrix, self.dist_coeffs,
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
            
        cv2.putText(vis_image, status_text, (10, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add calibration info
        calib_text = f"Calibration points: {len(self.calibration_pairs)}"
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
            
            print(f"Results saved to {output_dir}")
        else:
            print(f"Image saved to {output_dir} (no pose detected)")
    
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
                    success = self.collect_calibration_point(results)
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
            if len(self.calibration_pairs) >= 3:
                while True:  # Allow interactive validation loop
                    print(f"\nPerforming eye-in-hand calibration with {len(self.calibration_pairs)} points...")
                    
                    eye_in_hand_transform = self.calculate_eye_in_hand_calibration()
                    
                    if eye_in_hand_transform is not None:
                        # Validate calibration quality
                        validation_results = self.validate_calibration(eye_in_hand_transform)
                        
                        # Calculate advanced metrics
                        if len(self.calibration_pairs) >= 4:
                            self.advanced_calibration_metrics(eye_in_hand_transform)
                        
                        # Interactive validation review
                        decision = self.interactive_validation_review(validation_results)
                        
                        if decision is True:  # Accept calibration
                            # Save calibration results
                            output_dir = self.save_calibration_results(eye_in_hand_transform)
                            
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
                            if len(self.calibration_pairs) == 0:  # Reset was chosen
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
                print(f"Not enough calibration points ({len(self.calibration_pairs)}/3) - no calibration performed")
            
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