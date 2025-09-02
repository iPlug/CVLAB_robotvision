#!/usr/bin/env python3
"""
CharUco Transformer - Modularized CharUco Pattern Detection and Robot Coordination

A comprehensive, plug-and-play module for CharUco pattern detection, pose estimation,
and robot coordinate transformations. This module consolidates all CharUco-related
transformation logic from the charuco-pose-estimation folder into a single, reusable component.

Features:
- Real-time CharUco pattern detection and pose estimation
- Eye-in-hand calibration calculation and management
- Robot coordinate system transformations
- Coordinate system origin conversion (bottom-left vs center)
- Position prediction and validation
- Calibration quality assessment
- Clean, type-agnostic API similar to CameraToRobotTransformer

Usage:
    # For calibration
    transformer = CharUcoTransformer()
    transformer.setup_camera_and_detection(camera_matrix, dist_coeffs)
    transformer.collect_calibration_point(image, robot_pose)
    eye_in_hand = transformer.calculate_calibration()
    
    # For real-time prediction
    transformer = CharUcoTransformer('calibration_result_dir')
    robot_position = transformer.predict_robot_position(image, robot_pose)
"""

import sys
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
import time
import json
from scipy.spatial.transform import Rotation
from typing import Union, List, Tuple, Optional, Dict, Any
from cv2 import aruco


class CharUcoTransformer:
    """
    Universal CharUco pattern transformer for robot coordination applications.
    
    This class provides a unified interface for CharUco pattern detection, pose estimation,
    eye-in-hand calibration, and robot coordinate transformations. It consolidates all
    transformation logic previously scattered across multiple files in charuco-pose-estimation/.
    
    Coordinate Systems:
    - CharUco: Bottom-left origin (default) or center origin, X=right, Y=up, Z=outward (meters)
    - Camera: X=right, Y=down, Z=forward (meters) 
    - Robot: X=forward, Y=left, Z=up (millimeters)
    """
    
    def __init__(self, 
                 calibration_path: Union[str, Path] = None,
                 board_size: Tuple[int, int] = (7, 5),
                 square_length: float = 0.039,
                 marker_length: float = 0.029,
                 use_center_origin: bool = False,
                 prediction_offset: Union[List[float], np.ndarray] = None):
        """
        Initialize the CharUco transformer.
        
        Args:
            calibration_path: Path to calibration directory or None for calibration mode
            board_size: CharUco board dimensions (width, height) 
            square_length: Square size in meters (39mm default)
            marker_length: Marker size in meters (29mm default)
            use_center_origin: If True, use center origin; if False, use bottom-left origin
            prediction_offset: 3D offset [x, y, z] in mm to add to all predictions (default: [0, 0, 0])
                             Useful for fine-tuning predictions or accounting for mounting offsets
        """
        # CharUco board configuration
        self.board_size = board_size
        self.square_length = square_length
        self.marker_length = marker_length
        self.use_center_origin = use_center_origin
        
        # Prediction offset configuration
        if prediction_offset is None:
            self.prediction_offset = np.array([0.0, 0.0, 0.0])
        else:
            self.prediction_offset = np.array(prediction_offset, dtype=float)
            if self.prediction_offset.shape != (3,):
                raise ValueError(f"prediction_offset must be [x, y, z], got shape {self.prediction_offset.shape}")
        
        # ArUco/CharUco setup
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.board = aruco.CharucoBoard(board_size, square_length, marker_length, self.dictionary)
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = np.zeros(5)
        
        # Detection parameters
        self.aruco_detector = None
        self.charuco_detector = None
        
        # Calibration data
        self.eye_in_hand_transform = None
        self.calibration_pairs = []
        self.calibration_info = {}
        
        # Position tracking
        self.recent_positions = []
        self.position_history_size = 10
        
        # Load calibration if path provided
        if calibration_path:
            self.load_calibration(calibration_path)
    
    def setup_camera_and_detection(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray = None):
        """
        Setup camera parameters and initialize detectors.
        
        Args:
            camera_matrix: 3x3 camera calibration matrix
            dist_coeffs: Distortion coefficients (optional, defaults to zeros)
        """
        self.camera_matrix = camera_matrix.copy()
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs.copy()
        else:
            self.dist_coeffs = np.zeros(5)
        
        # Setup ArUco detector
        detector_params = aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        self.aruco_detector = aruco.ArucoDetector(self.dictionary, detector_params)
        
        # Setup CharUco detector  
        charuco_params = aruco.CharucoParameters()
        charuco_params.cameraMatrix = self.camera_matrix
        charuco_params.distCoeffs = self.dist_coeffs
        self.charuco_detector = aruco.CharucoDetector(self.board, charuco_params)
        
        print(f"[OK] CharUco detector initialized")
        print(f"     Board: {self.board_size[0]}x{self.board_size[1]} squares")
        print(f"     Square: {self.square_length*1000:.1f}mm, Marker: {self.marker_length*1000:.1f}mm")
        print(f"     Origin: {'Center' if self.use_center_origin else 'Bottom-left'}")
    
    def load_calibration(self, calibration_path: Union[str, Path]) -> bool:
        """
        Load pre-calculated eye-in-hand calibration from directory.
        
        Args:
            calibration_path: Path to calibration directory
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        calib_dir = Path(calibration_path)
        
        try:
            # Load transformation matrix
            transform_file = calib_dir / 'eye_in_hand_transform.npy'
            if not transform_file.exists():
                raise FileNotFoundError(f"No eye_in_hand_transform.npy found in {calib_dir}")
            
            self.eye_in_hand_transform = np.load(transform_file)
            print(f"[OK] Loaded eye-in-hand transformation from: {transform_file}")
            
            # Load camera intrinsics used during calibration
            intrinsics_file = calib_dir / 'camera_intrinsics.txt'
            if intrinsics_file.exists():
                calibration_intrinsics = np.loadtxt(intrinsics_file)
                self.camera_matrix = calibration_intrinsics
                self.dist_coeffs = np.zeros(5)
                print(f"[OK] Using calibration camera intrinsics")
                print(f"     fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
                print(f"     cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}")
            
            # Load calibration metadata
            self._load_calibration_metadata(calib_dir)
            
            # Setup detectors with loaded intrinsics
            if self.camera_matrix is not None:
                self.setup_camera_and_detection(self.camera_matrix, self.dist_coeffs)
            
            self._print_transformation_info()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load calibration: {e}")
            return False
    
    def _load_calibration_metadata(self, calibration_dir: Path):
        """Load calibration metadata if available."""
        metadata_file = calibration_dir / 'calibration_data.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.calibration_info = json.load(f)
                print(f"[INFO] Loaded calibration metadata")
            except Exception as e:
                print(f"[WARN] Could not load calibration metadata: {e}")
    
    def _print_transformation_info(self):
        """Print transformation matrix information."""
        if self.eye_in_hand_transform is None:
            return
            
        print("Eye-in-Hand Transformation Matrix:")
        for i, row in enumerate(self.eye_in_hand_transform):
            print(f"  [{row[0]:8.5f} {row[1]:8.5f} {row[2]:8.5f} {row[3]:8.5f}]")
        
        translation = self.eye_in_hand_transform[:3, 3]
        print(f"Translation (camera to gripper): [{translation[0]*1000:.1f}, {translation[1]*1000:.1f}, {translation[2]*1000:.1f}] mm")
        
        if self.calibration_info:
            num_points = self.calibration_info.get('num_points', 'Unknown')
            print(f"Calibration points used: {num_points}")
    
    def detect_charuco_pose(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect CharUco pattern and estimate its pose in camera coordinates.
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary with detection results:
            {
                'valid_pose': bool,
                'charuco_corners': array or None,
                'charuco_ids': array or None, 
                'rvec': array or None,
                'tvec': array or None,
                'num_corners': int,
                'subpixel_corners': bool
            }
        """
        if self.charuco_detector is None or self.camera_matrix is None:
            raise RuntimeError("Camera and detection not setup. Call setup_camera_and_detection() first.")
        
        results = {
            'valid_pose': False,
            'charuco_corners': None,
            'charuco_ids': None,
            'rvec': None,
            'tvec': None,
            'num_corners': 0,
            'subpixel_corners': False
        }
        
        try:
            # Detect CharUco pattern
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.charuco_detector.detectBoard(image)
            
            if charuco_ids is not None and len(charuco_ids) >= 4:
                results['charuco_corners'] = charuco_corners
                results['charuco_ids'] = charuco_ids
                results['num_corners'] = len(charuco_ids)
                results['subpixel_corners'] = self._check_subpixel_corners(charuco_corners)
                
                # Estimate pose using corrected coordinate matching
                if self.use_center_origin:
                    # Use center origin approach
                    success, rvec, tvec = cv2.solvePnP(
                        self._get_center_origin_object_points(charuco_ids),
                        charuco_corners,
                        self.camera_matrix,
                        self.dist_coeffs
                    )
                else:
                    # Use corrected bottom-left origin approach
                    obj_points, img_points = self._match_image_points_corrected(charuco_corners, charuco_ids)
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        img_points,
                        self.camera_matrix,
                        self.dist_coeffs
                    )
                
                if success:
                    results['valid_pose'] = True
                    results['rvec'] = rvec
                    results['tvec'] = tvec
                    
        except Exception as e:
            print(f"[WARN] CharUco detection failed: {e}")
        
        return results
    
    def _check_subpixel_corners(self, corners: np.ndarray) -> bool:
        """Check if corners are subpixel accurate."""
        if corners is None:
            return False
        for corner in corners.flatten():
            if not float(corner).is_integer():
                return True
        return False
    
    def _match_image_points_corrected(self, charuco_corners: np.ndarray, charuco_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match image points with corrected coordinate system (bottom-left origin).
        This implements the coordinate system correction from utils.py
        """
        obj_points = []
        img_points = []
        
        for i in range(len(charuco_ids)):
            corner_id = charuco_ids[i][0]
            # Get the corner position from the board
            corner_pos = self.board.getChessboardCorners()[corner_id]
            
            # Apply coordinate system correction (bottom-left origin)
            # Flip Y-axis to make bottom-left the origin (instead of top-left)
            corrected_pos = corner_pos.copy()
            corrected_pos[0][1] = self.board.getRightBottomCorner()[1] - corrected_pos[0][1]
            
            obj_points.append(corrected_pos)
            img_points.append(charuco_corners[i])
        
        return np.array(obj_points), np.array(img_points)
    
    def _get_center_origin_object_points(self, charuco_ids: np.ndarray) -> np.ndarray:
        """Get object points for center origin coordinate system."""
        obj_points = []
        
        # Calculate center offset
        center_x = (self.board_size[0] - 1) * self.square_length / 2 + self.square_length / 2  
        center_y = (self.board_size[1] - 1) * self.square_length / 2 + self.square_length / 2
        
        for corner_id in charuco_ids.flatten():
            # Get standard corner position
            corner_pos = self.board.getChessboardCorners()[corner_id][0]
            
            # Apply center offset
            centered_pos = corner_pos.copy()
            centered_pos[0] -= center_x
            centered_pos[1] -= center_y
            
            obj_points.append([centered_pos])
        
        return np.array(obj_points)
    
    def charuco_pose_to_transform_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Convert CharUco rvec, tvec to 4x4 transformation matrix with coordinate correction.
        This implements the coordinate system correction from utils.py
        """
        # Convert rotation vector to rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)
        
        # Apply coordinate system correction: 90° rotation around Z-axis
        # This fixes the systematic rotation error by aligning CharUco axes with robot axes
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
    
    def convert_to_center_origin(self, tvec: np.ndarray, rvec: np.ndarray) -> np.ndarray:
        """
        Convert bottom-left origin tvec to center origin using simple offset approach.
        """
        # Real offset values from center origin implementation
        center_x = (self.board_size[0] - 1) * self.square_length / 2 + self.square_length / 2
        center_y = (self.board_size[1] - 1) * self.square_length / 2 + self.square_length / 2
        offset_m = np.array([center_x, center_y, 0.0])  # meters
        
        # Get rotation matrix to transform offset to camera frame
        rot_matrix, _ = cv2.Rodrigues(rvec)
        offset_in_camera = rot_matrix @ offset_m
        
        # Apply offset (add to move from bottom-left to center position)
        tvec_center = tvec.flatten() + offset_in_camera
        
        return tvec_center.reshape(-1, 1)
    
    def pose_to_transform_matrix(self, robot_pose: List[float]) -> np.ndarray:
        """
        Convert robot pose [x, y, z, rx, ry, rz] to 4x4 transformation matrix.
        """
        x, y, z, rx, ry, rz = robot_pose
        
        # Convert to meters and radians
        x_m, y_m, z_m = x/1000.0, y/1000.0, z/1000.0
        rx_rad, ry_rad, rz_rad = np.radians([rx, ry, rz])
        
        # Create rotation matrix from Euler angles (XYZ intrinsic order)
        rotation = Rotation.from_euler('xyz', [rx_rad, ry_rad, rz_rad])
        rot_matrix = rotation.as_matrix()
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = [x_m, y_m, z_m]
        
        return transform
    
    def predict_robot_position(self, 
                             image: np.ndarray, 
                             robot_pose: List[float],
                             output_units: str = 'mm') -> Optional[np.ndarray]:
        """
        Predict CharUco pattern position in robot base coordinates.
        
        Args:
            image: Input BGR image
            robot_pose: Current robot pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            output_units: Output units ('mm' or 'meters')
            
        Returns:
            3D position in robot base coordinates [x, y, z] or None if not detected
        """
        if self.eye_in_hand_transform is None:
            raise RuntimeError("Eye-in-hand transformation not loaded. Call load_calibration() first.")
        
        # Detect CharUco pattern
        detection_results = self.detect_charuco_pose(image)
        
        if not detection_results['valid_pose']:
            return None
        
        # Get CharUco transformation matrix
        charuco_transform = self.charuco_pose_to_transform_matrix(
            detection_results['rvec'], 
            detection_results['tvec']
        )
        
        # Get robot transformation matrix
        robot_transform = self.pose_to_transform_matrix(robot_pose)
        
        # Apply transformation chain: Robot_base_to_gripper @ Eye_in_hand @ Camera_to_charuco
        pattern_in_robot = robot_transform @ self.eye_in_hand_transform @ charuco_transform
        
        # Extract position and convert to desired units
        position = pattern_in_robot[:3, 3]
        if output_units == 'mm':
            position = position * 1000.0
        elif output_units != 'meters':
            raise ValueError(f"Invalid output_units: {output_units}. Use 'meters' or 'mm'")
        
        # Apply prediction offset
        position = position + self.prediction_offset
        
        # Update position history for stability analysis
        self._update_position_history(position)
        
        return position
    
    def _update_position_history(self, position: np.ndarray):
        """Update position history for stability analysis."""
        self.recent_positions.append(position.copy())
        if len(self.recent_positions) > self.position_history_size:
            self.recent_positions.pop(0)
    
    def get_position_stability(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get position stability metrics."""
        if len(self.recent_positions) < 2:
            return None, None
        
        positions = np.array(self.recent_positions)
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        return mean_pos, std_pos
    
    def set_prediction_offset(self, offset: Union[List[float], np.ndarray]):
        """
        Set the prediction offset that will be added to all position predictions.
        
        Args:
            offset: 3D offset [x, y, z] in mm to add to all predictions
                   Use [0, 0, 0] to disable offset
        
        Example:
            # Add 5mm forward, 2mm left, 1mm up offset to all predictions
            transformer.set_prediction_offset([5.0, 2.0, 1.0])
            
            # Disable offset
            transformer.set_prediction_offset([0, 0, 0])
        """
        self.prediction_offset = np.array(offset, dtype=float)
        if self.prediction_offset.shape != (3,):
            raise ValueError(f"Offset must be [x, y, z], got shape {self.prediction_offset.shape}")
        
        print(f"[INFO] Prediction offset set to: [{self.prediction_offset[0]:.2f}, {self.prediction_offset[1]:.2f}, {self.prediction_offset[2]:.2f}] mm")
    
    def get_prediction_offset(self) -> np.ndarray:
        """
        Get the current prediction offset.
        
        Returns:
            Current 3D offset [x, y, z] in mm
        """
        return self.prediction_offset.copy()
    
    def clear_prediction_offset(self):
        """Clear the prediction offset (set to [0, 0, 0])."""
        self.set_prediction_offset([0.0, 0.0, 0.0])
    
    # Calibration Methods
    
    def collect_calibration_point(self, 
                                image: np.ndarray, 
                                robot_pose: List[float]) -> bool:
        """
        Collect a calibration point (robot pose + CharUco detection).
        
        Args:
            image: Input BGR image
            robot_pose: Current robot pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            
        Returns:
            bool: True if calibration point collected successfully
        """
        # Detect CharUco pattern
        detection_results = self.detect_charuco_pose(image)
        
        if not detection_results['valid_pose']:
            print("[WARN] No CharUco pose detected - cannot collect calibration point")
            return False
        
        print(f"\n[OK] Collecting calibration point {len(self.calibration_pairs) + 1}...")
        
        # Convert to transformation matrices
        robot_transform = self.pose_to_transform_matrix(robot_pose)
        charuco_transform = self.charuco_pose_to_transform_matrix(
            detection_results['rvec'], 
            detection_results['tvec']
        )
        
        # Store calibration pair
        calibration_point = {
            'robot_pose': robot_pose,
            'robot_transform': robot_transform,
            'charuco_rvec': detection_results['rvec'],
            'charuco_tvec': detection_results['tvec'],
            'charuco_transform': charuco_transform,
            'num_corners': detection_results['num_corners'],
            'subpixel': detection_results['subpixel_corners'],
            'timestamp': time.time()
        }
        
        self.calibration_pairs.append(calibration_point)
        
        print(f"     Robot position: [{robot_pose[0]:.1f}, {robot_pose[1]:.1f}, {robot_pose[2]:.1f}] mm")
        print(f"     CharUco corners: {detection_results['num_corners']} detected")
        print(f"     Total calibration points: {len(self.calibration_pairs)}")
        
        return True
    
    def calculate_eye_in_hand_calibration(self) -> Optional[np.ndarray]:
        """
        Calculate eye-in-hand calibration using collected calibration points.
        
        Returns:
            4x4 eye-in-hand transformation matrix or None if calculation failed
        """
        if len(self.calibration_pairs) < 3:
            print(f"[ERROR] Need at least 3 calibration points, have {len(self.calibration_pairs)}")
            return None
        
        print(f"\n[OK] Calculating eye-in-hand calibration from {len(self.calibration_pairs)} points...")
        
        # Prepare data for OpenCV calibrateHandEye
        robot_transforms = [pair['robot_transform'] for pair in self.calibration_pairs]
        charuco_transforms = [pair['charuco_transform'] for pair in self.calibration_pairs]
        
        # Convert to OpenCV format (rotation matrices and translation vectors)
        R_gripper2base = [T[:3, :3] for T in robot_transforms]
        t_gripper2base = [T[:3, 3].reshape(3, 1) for T in robot_transforms]
        R_target2cam = [T[:3, :3] for T in charuco_transforms]
        t_target2cam = [T[:3, 3].reshape(3, 1) for T in charuco_transforms]
        
        try:
            # Calculate hand-eye calibration using Tsai method
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # Create 4x4 transformation matrix
            self.eye_in_hand_transform = np.eye(4)
            self.eye_in_hand_transform[:3, :3] = R_cam2gripper
            self.eye_in_hand_transform[:3, 3] = t_cam2gripper.flatten()
            
            print("[OK] Eye-in-hand calibration successful!")
            self._print_transformation_info()
            
            return self.eye_in_hand_transform
            
        except Exception as e:
            print(f"[ERROR] Calibration failed: {e}")
            return None
    
    def validate_calibration(self, known_pattern_position: List[float] = None) -> Optional[Dict[str, Any]]:
        """
        Validate calibration quality using collected calibration points.
        
        Args:
            known_pattern_position: Known pattern position in robot coordinates [x, y, z] mm
                                  (defaults to [300, 100, 0] - 300mm forward, 100mm left)
        
        Returns:
            Dictionary with validation results or None if validation not possible
        """
        if len(self.calibration_pairs) < 3 or self.eye_in_hand_transform is None:
            return None
        
        if known_pattern_position is None:
            known_pattern_position = [300.0, 100.0, 0.0]  # Default known position
        
        print(f"\n[OK] Validating calibration quality...")
        print(f"     Known pattern position: [{known_pattern_position[0]:.1f}, {known_pattern_position[1]:.1f}, {known_pattern_position[2]:.1f}] mm")
        
        pattern_position_robot = np.array(known_pattern_position) / 1000.0  # Convert to meters
        
        translation_errors = []
        rotation_errors = []
        
        for i, pair in enumerate(self.calibration_pairs):
            # Calculate expected pattern position using eye-in-hand calibration
            robot_transform = pair['robot_transform']
            charuco_transform = pair['charuco_transform']
            
            expected_pattern_in_base = robot_transform @ self.eye_in_hand_transform @ charuco_transform
            expected_position = expected_pattern_in_base[:3, 3]
            
            # Calculate position error
            position_error = np.linalg.norm(expected_position - pattern_position_robot)
            translation_errors.append(position_error * 1000)  # Convert to mm
            
            # Calculate rotation error
            expected_rotation = np.eye(3)
            predicted_rotation = expected_pattern_in_base[:3, :3]
            
            rotation_error_matrix = predicted_rotation @ expected_rotation.T
            rotation_error_trace = np.trace(rotation_error_matrix)
            rotation_error_angle = np.arccos(np.clip((rotation_error_trace - 1) / 2, -1, 1))
            rotation_error_degrees = np.degrees(rotation_error_angle)
            rotation_errors.append(rotation_error_degrees)
        
        # Calculate summary statistics
        mean_translation_error = np.mean(translation_errors)
        max_translation_error = np.max(translation_errors)
        mean_rotation_error = np.mean(rotation_errors)
        
        # Quality assessment
        if mean_translation_error < 10.0:
            quality_score = 'Excellent'
        elif mean_translation_error < 25.0:
            quality_score = 'Good'
        elif mean_translation_error < 50.0:
            quality_score = 'Fair'
        else:
            quality_score = 'Poor'
        
        validation_results = {
            'mean_translation_error_mm': mean_translation_error,
            'max_translation_error_mm': max_translation_error,
            'mean_rotation_error_deg': mean_rotation_error,
            'translation_errors_mm': translation_errors,
            'rotation_errors_deg': rotation_errors,
            'quality_score': quality_score,
            'num_points': len(self.calibration_pairs)
        }
        
        print(f"     Mean translation error: {mean_translation_error:.2f} mm")
        print(f"     Mean rotation error: {mean_rotation_error:.2f}°")
        print(f"     Quality assessment: {quality_score}")
        
        return validation_results
    
    def save_calibration_results(self, output_dir: Union[str, Path] = None) -> Path:
        """
        Save calibration results to directory.
        
        Args:
            output_dir: Output directory (defaults to timestamped directory)
            
        Returns:
            Path to output directory
        """
        if self.eye_in_hand_transform is None:
            raise RuntimeError("No calibration results to save")
        
        if output_dir is None:
            timestamp = int(time.time())
            output_dir = Path(f"calibration_result_{timestamp}")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # Save transformation matrix
        np.savetxt(output_dir / 'eye_in_hand_transform.txt', self.eye_in_hand_transform)
        np.save(output_dir / 'eye_in_hand_transform.npy', self.eye_in_hand_transform)
        
        # Save camera intrinsics
        if self.camera_matrix is not None:
            np.savetxt(output_dir / 'camera_intrinsics.txt', self.camera_matrix)
        
        # Save calibration data
        calibration_data = {
            'timestamp': int(time.time()),
            'num_points': len(self.calibration_pairs),
            'board_config': {
                'board_size': self.board_size,
                'square_length': self.square_length,
                'marker_length': self.marker_length,
                'use_center_origin': self.use_center_origin
            },
            'camera_intrinsics': {
                'camera_matrix': self.camera_matrix.tolist() if self.camera_matrix is not None else None,
                'dist_coeffs': self.dist_coeffs.tolist(),
                'note': 'These intrinsics were used during calibration and MUST be used with this transformation matrix'
            },
            'eye_in_hand_transform': self.eye_in_hand_transform.tolist(),
            'points': []
        }
        
        # Save individual calibration points
        for i, pair in enumerate(self.calibration_pairs):
            point_data = {
                'id': i + 1,
                'robot_pose': pair['robot_pose'],
                'robot_transform': pair['robot_transform'].tolist(),
                'charuco_rvec': pair['charuco_rvec'].tolist(),
                'charuco_tvec': pair['charuco_tvec'].tolist(),
                'charuco_transform': pair['charuco_transform'].tolist(),
                'num_corners': pair['num_corners'],
                'subpixel': pair['subpixel'],
                'timestamp': pair['timestamp']
            }
            calibration_data['points'].append(point_data)
            
            # Save individual transformation matrices
            np.savetxt(output_dir / f'robot_transform_{i+1}.txt', pair['robot_transform'])
            np.savetxt(output_dir / f'charuco_transform_{i+1}.txt', pair['charuco_transform'])
        
        # Save JSON data
        with open(output_dir / 'calibration_data.json', 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        # Create README
        readme_content = f"""CharUco Calibration Result - {calibration_data['timestamp']}

IMPORTANT: This calibration result MUST be used with the specific camera intrinsics.

Files:
- eye_in_hand_transform.npy: Main transformation matrix  
- camera_intrinsics.txt: Camera intrinsics used during calibration
- calibration_data.json: Complete calibration data

USAGE:
transformer = CharUcoTransformer('{output_dir}')
robot_position = transformer.predict_robot_position(image, robot_pose)

Board Configuration:
- Size: {self.board_size[0]}x{self.board_size[1]} squares
- Square: {self.square_length*1000:.1f}mm, Marker: {self.marker_length*1000:.1f}mm  
- Origin: {'Center' if self.use_center_origin else 'Bottom-left'}

Number of calibration points: {len(self.calibration_pairs)}
"""
        
        with open(output_dir / 'README.txt', 'w') as f:
            f.write(readme_content)
        
        print(f"[OK] Calibration results saved to: {output_dir}")
        print(f"     Main transformation: {output_dir / 'eye_in_hand_transform.npy'}")
        
        return output_dir
    
    def is_ready_for_prediction(self) -> bool:
        """Check if transformer is ready for position prediction."""
        return (self.eye_in_hand_transform is not None and 
                self.camera_matrix is not None and
                self.charuco_detector is not None)
    
    def is_ready_for_calibration(self) -> bool:
        """Check if transformer is ready for calibration."""
        return (self.camera_matrix is not None and 
                self.charuco_detector is not None)
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get information about current calibration status."""
        info = {
            'calibration_loaded': self.eye_in_hand_transform is not None,
            'calibration_points_collected': len(self.calibration_pairs),
            'camera_setup': self.camera_matrix is not None,
            'board_config': {
                'size': self.board_size,
                'square_length_mm': self.square_length * 1000,
                'marker_length_mm': self.marker_length * 1000,
                'use_center_origin': self.use_center_origin
            },
            'prediction_offset_mm': self.prediction_offset.tolist()
        }
        
        if self.eye_in_hand_transform is not None:
            translation = self.eye_in_hand_transform[:3, 3]
            info['transformation_info'] = {
                'translation_mm': (translation * 1000).tolist(),
                'matrix_shape': self.eye_in_hand_transform.shape
            }
        
        if self.calibration_info:
            info['metadata'] = self.calibration_info
        
        return info
    
    def print_usage_examples(self):
        """Print comprehensive usage examples."""
        print("\n" + "="*70)
        print("CHARUCO TRANSFORMER - USAGE EXAMPLES")
        print("="*70)
        print("This transformer provides unified CharUco pattern detection and robot coordination:")
        print()
        print("# 1. Calibration Mode")
        print("transformer = CharUcoTransformer(board_size=(7,5), square_length=0.039)")
        print("transformer.setup_camera_and_detection(camera_matrix, dist_coeffs)")
        print("transformer.collect_calibration_point(image, robot_pose)")
        print("eye_in_hand = transformer.calculate_eye_in_hand_calibration()")
        print("transformer.save_calibration_results()")
        print()
        print("# 2. Prediction Mode")
        print("transformer = CharUcoTransformer('calibration_result_dir')")
        print("robot_pos = transformer.predict_robot_position(image, robot_pose)")
        print()
        print("# 3. Detection Only")
        print("results = transformer.detect_charuco_pose(image)")
        print("if results['valid_pose']:")
        print("    print(f\"Detected {results['num_corners']} corners\")")
        print()
        print("# 4. Validation")
        print("validation = transformer.validate_calibration([300, 100, 0])")
        print("print(f\"Quality: {validation['quality_score']}\")")
        print()
        print("# 5. Prediction Offset")
        print("# Add 5mm forward, 2mm left offset to all predictions")
        print("transformer.set_prediction_offset([5.0, 2.0, 0.0])")
        print("# Or create with offset")
        print("transformer = CharUcoTransformer('calib_dir', prediction_offset=[5, 2, 0])")
        print()
        print("Coordinate Systems:")
        print("  CharUco: Bottom-left or center origin, X=right, Y=up, Z=outward (meters)")
        print("  Camera:  X=right, Y=down, Z=forward (meters)")
        print("  Robot:   X=forward, Y=left, Z=up (millimeters)")
        print("="*70)


# Convenience functions for quick usage
def create_calibration_transformer(board_size: Tuple[int, int] = (7, 5),
                                 square_length: float = 0.039,
                                 marker_length: float = 0.029,
                                 use_center_origin: bool = False,
                                 prediction_offset: Union[List[float], np.ndarray] = None) -> CharUcoTransformer:
    """Create CharUco transformer for calibration workflow."""
    return CharUcoTransformer(
        calibration_path=None,
        board_size=board_size,
        square_length=square_length, 
        marker_length=marker_length,
        use_center_origin=use_center_origin,
        prediction_offset=prediction_offset
    )


def create_prediction_transformer(calibration_path: Union[str, Path],
                                prediction_offset: Union[List[float], np.ndarray] = None) -> CharUcoTransformer:
    """
    Create CharUco transformer for real-time prediction.
    
    Args:
        calibration_path: Path to calibration directory
        prediction_offset: Optional 3D offset [x, y, z] in mm to add to all predictions
        
    Returns:
        Ready-to-use CharUcoTransformer for prediction
        
    Raises:
        RuntimeError: If calibration cannot be loaded
    """
    transformer = CharUcoTransformer(calibration_path, prediction_offset=prediction_offset)
    if not transformer.is_ready_for_prediction():
        raise RuntimeError(f"Failed to load calibration from {calibration_path}")
    return transformer


def quick_charuco_prediction(image: np.ndarray,
                           robot_pose: List[float], 
                           calibration_path: Union[str, Path],
                           output_units: str = 'mm',
                           prediction_offset: Union[List[float], np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Quick CharUco position prediction without creating persistent transformer.
    
    Args:
        image: Input BGR image
        robot_pose: Current robot pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        calibration_path: Path to calibration directory
        output_units: Output units ('mm' or 'meters')
        prediction_offset: Optional 3D offset [x, y, z] in mm to add to prediction
        
    Returns:
        3D position in robot base coordinates [x, y, z] or None if not detected
    """
    transformer = create_prediction_transformer(calibration_path, prediction_offset)
    return transformer.predict_robot_position(image, robot_pose, output_units)


if __name__ == "__main__":
    # Example usage and demonstration
    print("CharUco Transformer - Unified CharUco Pattern Detection and Robot Coordination")
    print("=" * 80)
    
    # Example calibration directory (adjust for your system)
    example_calibration = "charuco-pose-estimation/calibration_result_1754622347"
    
    try:
        # Test prediction mode
        if Path(example_calibration).exists():
            print(f"\n[OK] Testing prediction mode with: {example_calibration}")
            
            transformer = CharUcoTransformer(example_calibration)
            
            if transformer.is_ready_for_prediction():
                print("[OK] Transformer ready for prediction!")
                
                # Show calibration info
                info = transformer.get_calibration_info()
                print(f"\nCalibration Info:")
                print(f"  Board: {info['board_config']['size']} squares")
                print(f"  Square: {info['board_config']['square_length_mm']:.1f}mm")
                print(f"  Origin: {'Center' if info['board_config']['use_center_origin'] else 'Bottom-left'}")
                if 'transformation_info' in info:
                    print(f"  Translation: {info['transformation_info']['translation_mm']} mm")
                
                transformer.print_usage_examples()
            else:
                print("[ERROR] Transformer not ready for prediction")
        else:
            print(f"[INFO] Calibration directory not found: {example_calibration}")
            print("[INFO] Showing calibration mode example...")
            
            # Test calibration mode
            transformer = create_calibration_transformer()
            
            print(f"\nCalibration Transformer Created:")
            print(f"  Board: {transformer.board_size} squares") 
            print(f"  Square: {transformer.square_length*1000:.1f}mm")
            print(f"  Marker: {transformer.marker_length*1000:.1f}mm")
            print(f"  Origin: {'Center' if transformer.use_center_origin else 'Bottom-left'}")
            print(f"  Ready for calibration: {transformer.is_ready_for_calibration()}")
            
            transformer.print_usage_examples()
        
    except Exception as e:
        print(f"[ERROR] Example failed: {e}")
        print("\nTo test with your calibration:")
        print("1. Update 'example_calibration' path to your calibration directory")
        print("2. Run: python charuco_transformer.py")
        print("\nOr use in your code:")
        print("transformer = CharUcoTransformer('your_calibration_dir')")
        print("position = transformer.predict_robot_position(image, robot_pose)")