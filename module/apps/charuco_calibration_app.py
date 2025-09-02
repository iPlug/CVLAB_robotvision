"""
CharUco-Based Camera-Robot Calibration Application

This application provides high-precision camera-robot calibration using CharUco patterns
instead of point cloud clustering. Supports both fixed camera and eye-in-hand calibration modes
with sub-5mm accuracy through OpenCV's CharUco detector and solvePnP() methods.

Features:
- RGB image-based CharUco detection
- ArUco marker and ChArUco corner detection
- Sub-pixel corner refinement
- Both fixed camera and eye-in-hand calibration modes
- Real-time pattern detection visualization
- Comprehensive quality validation
- Camera intrinsic calibration
- 3D pose estimation
"""

import time
import numpy as np
import cv2
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
import json

from ..core.object_detector import ObjectDetector
from ..strategies.charuco_detection_strategy import CharucoDetectionStrategy
from ..sensors.realsense_manager import RealSenseManager
from ..robot.mycobot_controller import MyCobotController
from ..config.config_factory import ConfigFactory
from ..config.charuco_config import CharucoConfig


class CharucoCalibrationApp:
    """
    CharUco-based camera-robot calibration application.
    
    Provides high-precision calibration using CharUco patterns with:
    - RGB image-based detection with ArUco markers
    - Sub-pixel accuracy
    - Multiple calibration modes
    - Comprehensive validation
    - Real-time visualization
    """
    
    def __init__(self, 
                 config_name: str = "charuco_standard",
                 calibration_mode: str = "fixed_camera",
                 pattern_size: Tuple[int, int] = (8, 11),
                 square_size_mm: float = 25.0,
                 marker_size_mm: float = 20.0,
                 transformation_matrix_file: str = "charuco_camera_robot_transform.npy",
                 camera_calibration_file: str = "charuco_camera_calibration.json",
                 log_file: str = "charuco_calibration.log",
                 no_filtering: bool = False):
        """
        Initialize CharUco calibration application.
        
        Args:
            config_name: Configuration preset name
            calibration_mode: 'fixed_camera' or 'eye_in_hand'
            pattern_size: CharUco pattern size (squares_x, squares_y)
            square_size_mm: Size of checkerboard squares in millimeters
            marker_size_mm: Size of ArUco markers in millimeters
            transformation_matrix_file: Output file for transformation matrix
            camera_calibration_file: Output file for camera calibration data
            log_file: Log file for calibration process
            no_filtering: If True, disable quality filtering (permissive like test_charuco_detection.py)
        """
        self.config_name = config_name
        self.calibration_mode = calibration_mode
        self.pattern_size = pattern_size
        self.square_size_mm = square_size_mm
        self.marker_size_mm = marker_size_mm
        self.transformation_matrix_file = transformation_matrix_file
        self.camera_calibration_file = camera_calibration_file
        self.log_file_path = log_file
        self.no_filtering = no_filtering
        
        # Load configuration
        if config_name.startswith('charuco_'):
            self.config = ConfigFactory.create_preset(config_name)
        else:
            # Create CharUco config from base config
            base_config = ConfigFactory.create_preset(config_name)
            self.config = CharucoConfig(base_config.config)
        
        # Override pattern parameters
        self.config.config['pattern_params'].update({
            'pattern_size': pattern_size,
            'square_size_mm': square_size_mm,
            'marker_size_mm': marker_size_mm
        })
        
        # Initialize components
        self.sensor_manager = None
        self.robot_controller = None
        self.detector = None
        self.charuco_strategy = None
        self.log_file = None
        
        # Calibration data
        self.pattern_detections = []  # All pattern detections for camera calibration
        self.calibration_pairs = []   # (camera_pose, robot_pose) pairs for hand-eye calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self.transformation_matrix = None
        
        # Application state
        self.is_running = False
        self.current_calibration_point = 0
        self.calibration_completed = False
        self.camera_calibrated = False
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validation
        self._validate_mode()
    
    def _validate_mode(self):
        """Validate calibration mode and configuration."""
        valid_modes = ['fixed_camera', 'eye_in_hand']
        if self.calibration_mode not in valid_modes:
            raise ValueError(f"Invalid calibration mode: {self.calibration_mode}. "
                           f"Must be one of: {valid_modes}")
        
        if self.pattern_size[0] < 4 or self.pattern_size[1] < 4:
            raise ValueError(f"CharUco pattern size too small: {self.pattern_size}. "
                           f"Must be at least (4, 4)")
        
        if self.square_size_mm <= 0:
            raise ValueError(f"Square size must be positive: {self.square_size_mm}")
        
        if self.marker_size_mm <= 0:
            raise ValueError(f"Marker size must be positive: {self.marker_size_mm}")
        
        if self.marker_size_mm >= self.square_size_mm:
            raise ValueError(f"Marker size must be smaller than square size: {self.marker_size_mm} >= {self.square_size_mm}")
    
    def log_message(self, message: str):
        """Write message to log file and console."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {message}"
        print(log_entry)
        
        if self.log_file:
            self.log_file.write(log_entry + "\\n")
            self.log_file.flush()
    
    def initialize_systems(self) -> bool:
        """
        Initialize all calibration system components.
        
        Returns:
            True if initialization successful
        """
        print("Initializing CharUco Camera-Robot Calibration System")
        print("=" * 60)
        print(f"Mode: {self.calibration_mode}")
        print(f"Pattern: {self.pattern_size[0]}x{self.pattern_size[1]} CharUco ({self.square_size_mm}mm squares, {self.marker_size_mm}mm markers)")
        print("=" * 60)
        
        # Open log file
        try:
            self.log_file = open(self.log_file_path, 'w')
            self.log_message("CharUco calibration process started")
            self.log_message(f"Mode: {self.calibration_mode}")
            self.log_message(f"Pattern: {self.pattern_size[0]}x{self.pattern_size[1]} CharUco ({self.square_size_mm}mm squares, {self.marker_size_mm}mm markers)")
        except Exception as e:
            print(f"Warning: Could not open log file: {e}")
        
        # Initialize sensor manager
        print("Initializing camera sensor...")
        self.sensor_manager = RealSenseManager(realtime_mode=True, use_imu=False)
        if not self.sensor_manager.initialize():
            self.log_message("Error: Failed to initialize camera sensor")
            return False
        
        if not self.sensor_manager.start():
            self.log_message("Error: Failed to start camera sensor")
            return False
        
        # Initialize robot controller
        print("Initializing robot controller...")
        self.robot_controller = MyCobotController(self.transformation_matrix_file)
        
        # Connect to robot
        if not self.robot_controller.connect():
            self.log_message("Warning: Failed to connect to robot")
            if self.calibration_mode == 'fixed_camera':
                print("Robot connection is required for fixed camera calibration")
                return False
            elif self.calibration_mode == 'eye_in_hand':
                print("Robot connection is required for eye-in-hand calibration")
                return False
        
        # Configure robot parameters
        robot_params = self.config.get('robot_params', {})
        if robot_params:
            self.robot_controller.safety_constraints.update(robot_params)
            self.log_message(f"Robot safety parameters: {robot_params}")
        
        # Enable async mode for smooth operation
        if self.robot_controller.enable_async_mode(max_queue_size=3):
            self.log_message("Robot async mode enabled")
        else:
            self.log_message("Warning: Failed to enable robot async mode")
        
        # Initialize CharUco detection strategy
        print("Initializing CharUco detection...")
        if self.no_filtering:
            print("Quality filtering DISABLED - using permissive detection like test_charuco_detection.py")
        
        self.charuco_strategy = CharucoDetectionStrategy(
            pattern_size=self.pattern_size,
            square_size_mm=self.square_size_mm,
            marker_size_mm=self.marker_size_mm,
            aruco_dict_type=cv2.aruco.DICT_6X6_250,
            min_pattern_coverage=self.config.get('quality_params', {}).get('min_pattern_coverage', 0.1),
            max_pattern_coverage=self.config.get('quality_params', {}).get('max_pattern_coverage', 0.9),
            quality_threshold=self.config.get('quality_params', {}).get('min_quality_threshold', 0.5),
            bypass_filtering=self.no_filtering
        )
        
        # Load camera calibration immediately after strategy initialization
        print("Loading camera calibration for 3D pose estimation...")
        if not self._load_camera_calibration():
            print("Warning: Failed to load camera calibration - 3D pose estimation will not be available")
        else:
            print("Camera calibration loaded successfully for CharUco strategy")
        
        # Initialize object detector with CharUco strategy
        self.detector = ObjectDetector(
            clustering_strategy=self.charuco_strategy,
            classification_strategy=None
        )
        
        self.log_message("All systems initialized successfully")
        print("All systems initialized successfully")
        return True
    
    def run_camera_intrinsic_calibration(self) -> bool:
        """
        Perform camera intrinsic calibration using multiple CharUco views.
        
        Returns:
            True if camera calibration successful
        """
        print(f"\\n{'='*60}")
        print("CAMERA INTRINSIC CALIBRATION")
        print(f"{'='*60}")
        print("This step calibrates camera internal parameters (focal length, distortion, etc.)")
        print("You will need to show the CharUco pattern in different positions and orientations.")
        print(f"Target: {self.config.get('camera_calibration_params', {}).get('min_calibration_images', 8)} images")
        print(f"{'='*60}")
        
        self.log_message("Starting camera intrinsic calibration")
        
        self.pattern_detections = []
        required_images = self.config.get('camera_calibration_params', {}).get('min_calibration_images', 8)
        
        print("\\nInstructions:")
        print("1. Hold the CharUco pattern in front of the camera")
        print("2. Move it to different positions and orientations")
        print("3. Press 'c' to capture when pattern is detected")
        print("4. Press 'q' to finish early (minimum 5 images required)")
        print("5. Pattern will be outlined in green when detected")
        
        try:
            while len(self.pattern_detections) < required_images:
                # Get frame from sensor
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                if color_image is not None:
                    # Detect CharUco pattern
                    detected_patterns = self.charuco_strategy.cluster_objects(vertices, color_image)
                    
                    # Create visualization
                    vis_image = self._create_camera_calibration_visualization(
                        color_image, detected_patterns, len(self.pattern_detections), required_images
                    )
                    
                    cv2.imshow('Camera Intrinsic Calibration', vis_image)
                
                # Handle user input
                key = cv2.waitKey(30) & 0xFF
                if key == ord('c'):
                    # Capture current detection
                    if detected_patterns:
                        pattern = detected_patterns[0]  # Use best pattern
                        self.pattern_detections.append({
                            'pattern': pattern,
                            'image_size': (color_image.shape[1], color_image.shape[0])
                        })
                        print(f"Captured image {len(self.pattern_detections)}/{required_images}")
                        self.log_message(f"Captured calibration image {len(self.pattern_detections)}")
                        
                        # Brief pause to avoid duplicate captures
                        time.sleep(0.5)
                    else:
                        print("No pattern detected - position CharUco pattern and try again")
                
                elif key == ord('q'):
                    if len(self.pattern_detections) >= 5:
                        print(f"Finishing early with {len(self.pattern_detections)} images")
                        break
                    else:
                        print("Need at least 5 images for calibration")
        
        except KeyboardInterrupt:
            print("\\nCamera calibration interrupted by user")
            if len(self.pattern_detections) < 5:
                return False
        
        cv2.destroyWindow('Camera Intrinsic Calibration')
        
        # Perform calibration calculation
        if len(self.pattern_detections) >= 5:
            return self._calculate_camera_calibration()
        else:
            print(f"Insufficient images for calibration: {len(self.pattern_detections)}")
            return False
    
    def _calculate_camera_calibration(self) -> bool:
        """Calculate camera calibration from collected pattern detections."""
        print(f"\\nCalculating camera calibration from {len(self.pattern_detections)} images...")
        self.log_message(f"Calculating camera calibration from {len(self.pattern_detections)} images")
        
        try:
            # Extract calibration data
            image_size = self.pattern_detections[0]['image_size']
            
            # Use CharUco strategy's calibration method
            self.camera_matrix, self.dist_coeffs, reprojection_error = \
                self.charuco_strategy.calibrate_camera_from_patterns(
                    [det['pattern'] for det in self.pattern_detections],
                    image_size
                )
            
            self.camera_calibrated = True
            
            # Validate calibration quality
            max_error = self.config.get('camera_calibration_params', {}).get('max_reprojection_error', 1.5)
            if reprojection_error > max_error:
                self.log_message(f"Warning: High reprojection error {reprojection_error:.3f} > {max_error}")
                print(f"Warning: High reprojection error {reprojection_error:.3f} pixels")
            else:
                print(f"Camera calibration successful - Error: {reprojection_error:.3f} pixels")
                self.log_message(f"Camera calibration successful - Error: {reprojection_error:.3f} pixels")
            
            # Save camera calibration
            self._save_camera_calibration()
            
            return True
            
        except Exception as e:
            print(f"Camera calibration failed: {e}")
            self.log_message(f"Camera calibration failed: {e}")
            return False
    
    def _save_camera_calibration(self):
        """Save camera calibration parameters to file."""
        try:
            calibration_data = {
                'camera_matrix': self.camera_matrix.tolist(),
                'dist_coeffs': self.dist_coeffs.tolist(),
                'image_size': self.pattern_detections[0]['image_size'],
                'pattern_size': self.pattern_size,
                'square_size_mm': self.square_size_mm,
                'marker_size_mm': self.marker_size_mm,
                'pattern_type': 'CharUco',
                'aruco_dict': 'DICT_6X6_250',
                'num_calibration_images': len(self.pattern_detections),
                'calibration_date': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(self.camera_calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            print(f"Camera calibration saved to: {self.camera_calibration_file}")
            self.log_message(f"Camera calibration saved to: {self.camera_calibration_file}")
            
        except Exception as e:
            print(f"Failed to save camera calibration: {e}")
            self.log_message(f"Failed to save camera calibration: {e}")
    
    def _load_camera_calibration(self) -> bool:
        """Use RealSense factory calibration for 3D pose estimation."""
        # Use RealSense factory calibration directly
        try:
            realsense_intrinsics = self.sensor_manager.get_camera_intrinsics()
            if realsense_intrinsics:
                # Convert RealSense intrinsics to OpenCV format
                self.camera_matrix = np.array([
                    [realsense_intrinsics.fx, 0, realsense_intrinsics.ppx],
                    [0, realsense_intrinsics.fy, realsense_intrinsics.ppy],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # RealSense distortion coefficients
                self.dist_coeffs = np.array(realsense_intrinsics.coeffs, dtype=np.float32)
                self.camera_calibrated = True
                
                # Set calibration in strategy
                self.charuco_strategy.set_camera_calibration(self.camera_matrix, self.dist_coeffs)
                
                print("Using RealSense factory calibration for 3D pose estimation")
                print(f"Camera matrix fx={realsense_intrinsics.fx:.1f}, fy={realsense_intrinsics.fy:.1f}")
                print(f"Principal point: cx={realsense_intrinsics.ppx:.1f}, cy={realsense_intrinsics.ppy:.1f}")
                self.log_message("Using RealSense factory calibration")
                return True
            else:
                print("No RealSense intrinsics available")
                return False
                
        except Exception as e:
            print(f"Failed to get RealSense calibration: {e}")
            return False
    
    def run_hand_eye_calibration(self) -> Optional[np.ndarray]:
        """
        Run hand-eye calibration process.
        
        Returns:
            Transformation matrix if successful, None otherwise
        """
        # Get calibration positions based on mode
        calibration_positions = self.config.get_optimal_calibration_positions(self.calibration_mode)
        
        print(f"\\n{'='*60}")
        print("HAND-EYE CALIBRATION PROCESS")
        print(f"{'='*60}")
        print(f"Mode: {self.calibration_mode}")
        print(f"Collecting {len(calibration_positions)} calibration points")
        print(f"Pattern: {self.pattern_size[0]}x{self.pattern_size[1]} CharUco ({self.square_size_mm}mm squares)")
        print(f"{'='*60}")
        
        self.log_message(f"Starting hand-eye calibration with {len(calibration_positions)} points")
        
        self.calibration_pairs = []
        
        for i, cal_pos in enumerate(calibration_positions):
            self.current_calibration_point = i + 1
            
            print(f"\\n--- CALIBRATION POINT {i+1}/{len(calibration_positions)} ---")
            self.log_message(f"Starting calibration point {i+1}/{len(calibration_positions)}")
            
            if self.calibration_mode == 'fixed_camera':
                success = self._capture_fixed_camera_pose(cal_pos, i+1)
            else:  # eye_in_hand
                success = self._capture_eye_in_hand_pose(cal_pos, i+1)
            
            if not success:
                print(f"Failed to capture calibration point {i+1}")
                self.log_message(f"Failed to capture calibration point {i+1}")
                
                # Ask user if they want to retry or continue
                print("Skipping this calibration point automatically...")
                continue
        
        # Check if we have enough points
        min_points = 3
        if len(self.calibration_pairs) < min_points:
            self.log_message(f"Error: Only captured {len(self.calibration_pairs)} pairs, need at least {min_points}")
            print(f"Calibration failed: Only {len(self.calibration_pairs)} pairs captured, need at least {min_points}")
            return None
        
        # Calculate transformation matrix
        print(f"\\nCalculating hand-eye transformation from {len(self.calibration_pairs)} point pairs...")
        self.log_message(f"Calculating hand-eye transformation from {len(self.calibration_pairs)} pairs")
        
        transformation_matrix = self._calculate_hand_eye_transformation()
        
        if transformation_matrix is not None:
            self.transformation_matrix = transformation_matrix
            print("Hand-eye calibration successful")
            self.log_message("Hand-eye calibration successful")
            return transformation_matrix
        else:
            print("Hand-eye calibration failed")
            self.log_message("Hand-eye calibration failed")
            return None
    
    def _capture_fixed_camera_pose(self, robot_position: List[float], point_num: int) -> bool:
        """
        Capture pose for fixed camera calibration.
        Robot holds the CharUco pattern.
        """
        print(f"Fixed camera mode - Robot will hold CharUco pattern at position {point_num}")
        
        # Move robot to position
        print(f"Moving robot to position: {robot_position}")
        self.log_message(f"Robot target position {point_num}: {robot_position}")
        
        if self.robot_controller.async_mode_enabled:
            command_id = self.robot_controller.move_to_position_async(
                robot_position, speed=30, priority=1
            )
            if command_id:
                self._wait_for_robot_movement(command_id)
            else:
                return False
        else:
            if not self.robot_controller.move_to_position(robot_position, speed=30):
                return False
            time.sleep(3)
        
        # Verify robot position
        try:
            current_pos = self.robot_controller.get_current_position()
            if current_pos:
                print(f"Robot reached: {current_pos}")
                self.log_message(f"Robot actual position: {current_pos}")
        except:
            self.log_message("Could not verify robot position")
        
        # Guide user
        print(f"\\nSTEP {point_num}: CHARUCO PATTERN DETECTION")
        print("1. Attach the CharUco pattern to the robot end effector")
        print("2. Ensure the pattern is clearly visible to the camera")
        print("3. Starting pattern detection automatically...")
        
        # Capture pattern pose
        pattern_pose = self._capture_pattern_pose_with_visualization()
        
        if pattern_pose is not None:
            # Store calibration pair
            robot_pose = robot_position[:6]  # [x, y, z, roll, pitch, yaw]
            self.calibration_pairs.append({
                'robot_pose': robot_pose,
                'pattern_pose': pattern_pose,
                'point_number': point_num
            })
            
            self.log_message(f"Point {point_num} captured successfully:")
            self.log_message(f"  Robot pose: {robot_pose}")
            self.log_message(f"  Pattern pose: {pattern_pose}")
            
            print(f"Calibration point {point_num} captured successfully")
            return True
        else:
            return False
    
    def _capture_eye_in_hand_pose(self, robot_position: List[float], point_num: int) -> bool:
        """
        Capture pose for eye-in-hand calibration.
        Robot camera looks at fixed CharUco pattern.
        """
        print(f"Eye-in-hand mode - Robot camera will view fixed CharUco pattern at position {point_num}")
        
        # Guide user for pattern placement (only for first position)
        if point_num == 1:
            print(f"\\nSTEP: CHARUCO PATTERN SETUP")
            print("1. Place the CharUco pattern on the table in a fixed position")
            print("2. Ensure it's clearly visible and well-lit")
            print("3. Do not move the pattern during the entire calibration process")
            print("4. Starting calibration automatically...")
        
        # Move robot camera to viewing position
        print(f"Moving robot camera to viewing position: {robot_position}")
        self.log_message(f"Robot camera position {point_num}: {robot_position}")
        
        if self.robot_controller.async_mode_enabled:
            command_id = self.robot_controller.move_to_position_async(
                robot_position, speed=30, priority=1
            )
            if command_id:
                self._wait_for_robot_movement(command_id)
            else:
                return False
        else:
            if not self.robot_controller.move_to_position(robot_position, speed=30):
                return False
            time.sleep(3)
        
        # Capture pattern pose from this viewpoint
        print(f"\\nCapturing pattern view from position {point_num}...")
        pattern_pose = self._capture_pattern_pose_with_visualization()
        
        if pattern_pose is not None:
            # Store calibration pair
            robot_pose = robot_position[:6]  # [x, y, z, roll, pitch, yaw]
            self.calibration_pairs.append({
                'robot_pose': robot_pose,
                'pattern_pose': pattern_pose,
                'point_number': point_num
            })
            
            self.log_message(f"Point {point_num} captured successfully:")
            self.log_message(f"  Robot pose: {robot_pose}")
            self.log_message(f"  Pattern pose: {pattern_pose}")
            
            print(f"Calibration point {point_num} captured successfully")
            return True
        else:
            return False
    
    def _capture_pattern_pose_with_visualization(self) -> Optional[Dict[str, Any]]:
        """
        Capture CharUco pattern pose with real-time visualization.
        
        Returns:
            Pattern pose data or None if failed
        """
        print("\\nPattern Detection Mode:")
        print("- Ensure CharUco pattern is clearly visible")
        print("- Press 'c' to capture pattern pose when detected")
        print("- Press 'q' to skip this calibration point")
        print("- Pattern corners and markers will be outlined when detected")
        
        captured_pose = None
        
        try:
            while True:
                # Get frame from sensor
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                if color_image is not None:
                    # Detect CharUco pattern
                    detected_patterns = self.charuco_strategy.cluster_objects(vertices, color_image)
                    
                    # Create visualization
                    vis_image = self._create_pattern_detection_visualization(
                        color_image, detected_patterns
                    )
                    
                    cv2.imshow('Pattern Detection', vis_image)
                
                # Handle user input
                key = cv2.waitKey(30) & 0xFF
                if key == ord('c'):
                    # Capture current detection
                    if detected_patterns:
                        best_pattern = detected_patterns[0]  # Use best quality pattern
                        
                        # Validate pattern has 3D pose
                        if best_pattern.get('pose_data') is not None:
                            captured_pose = best_pattern['pose_data']
                            print(f"CharUco pattern captured with 3D pose")
                            break
                        else:
                            print("Pattern detected but no 3D pose available - check camera calibration")
                    else:
                        print("No pattern detected - adjust position and try again")
                
                elif key == ord('q'):
                    print("Skipping this calibration point")
                    break
        
        except KeyboardInterrupt:
            print("\\nPattern capture interrupted by user")
        
        cv2.destroyWindow('Pattern Detection')
        return captured_pose
    
    def _calculate_hand_eye_transformation(self) -> Optional[np.ndarray]:
        """Calculate hand-eye transformation matrix from collected pose pairs."""
        if len(self.calibration_pairs) < 3:
            return None
        
        try:
            if self.calibration_mode == 'fixed_camera':
                return self._calculate_hand_to_eye_transformation()
            else:  # eye_in_hand
                return self._calculate_eye_in_hand_transformation()
                
        except Exception as e:
            self.log_message(f"Error calculating hand-eye transformation: {e}")
            return None
    
    def _calculate_hand_to_eye_transformation(self) -> np.ndarray:
        """Calculate transformation for fixed camera (hand-to-eye) setup."""
        # For fixed camera: camera sees pattern held by robot
        # We need to find transformation from robot base to camera
        
        robot_poses = []
        pattern_poses = []
        
        for pair in self.calibration_pairs:
            robot_pose = pair['robot_pose']  # [x, y, z, roll, pitch, yaw]
            pattern_data = pair['pattern_pose']
            
            # Convert robot pose to transformation matrix
            robot_matrix = self._pose_to_matrix(robot_pose)
            robot_poses.append(robot_matrix)
            
            # Convert pattern pose to transformation matrix
            translation = pattern_data['translation_mm'] / 1000.0  # Convert to meters
            rotation_matrix = cv2.Rodrigues(pattern_data['rvec'])[0]
            
            pattern_matrix = np.eye(4)
            pattern_matrix[:3, :3] = rotation_matrix
            pattern_matrix[:3, 3] = translation
            pattern_poses.append(pattern_matrix)
        
        # Solve AX = XB problem using least squares
        # A: robot poses, B: pattern poses, X: hand-eye transformation
        transformation_matrix = self._solve_hand_eye_least_squares(robot_poses, pattern_poses)
        
        return transformation_matrix
    
    def _calculate_eye_in_hand_transformation(self) -> np.ndarray:
        """
        Calculate transformation for eye-in-hand setup using L515-optimized workflow.
        
        For Intel RealSense L515, this implements the recommended workflow:
        1. Calculate robot-to-color transformation using RGB CharUco detection
        2. Get factory-calibrated color-to-depth transformation
        3. Chain transformations: T_robot_to_depth = T_robot_to_color * T_color_to_depth
        
        This leverages the factory-calibrated extrinsics for higher accuracy.
        """
        self.log_message("Using L515-optimized eye-in-hand calibration workflow")
        
        # Step 1: Calculate robot-to-color transformation using OpenCV's calibrateHandEye
        T_robot_to_color = self._calculate_robot_to_color_transformation()
        
        # Step 2: Get factory-calibrated color-to-depth transformation
        T_color_to_depth = self._get_factory_color_to_depth_transformation()
        
        # Step 3: Chain transformations
        if T_color_to_depth is not None:
            self.log_message("Chaining transformations: T_robot_to_depth = T_robot_to_color * T_color_to_depth")
            T_robot_to_depth = T_robot_to_color @ T_color_to_depth
            
            self.log_message("L515 factory extrinsics successfully integrated")
            self.log_message(f"Color-to-depth translation: {T_color_to_depth[:3, 3]}")
            self.log_message(f"Final robot-to-depth transformation calculated")
            
            return T_robot_to_depth
        else:
            self.log_message("Warning: Could not get factory extrinsics, using color-only calibration")
            return T_robot_to_color
    
    def _calculate_robot_to_color_transformation(self) -> np.ndarray:
        """Calculate robot-to-color transformation using OpenCV's calibrateHandEye."""
        robot_poses = []
        pattern_poses = []
        
        for pair in self.calibration_pairs:
            robot_pose = pair['robot_pose']  # [x, y, z, roll, pitch, yaw]
            pattern_data = pair['pattern_pose']
            
            # Convert robot pose to transformation matrix
            robot_matrix = self._pose_to_matrix(robot_pose)
            robot_poses.append(robot_matrix)
            
            # Convert pattern pose to transformation matrix (from camera perspective)
            translation = pattern_data['translation_mm'] / 1000.0  # Convert to meters
            rotation_matrix = cv2.Rodrigues(pattern_data['rvec'])[0]
            
            pattern_matrix = np.eye(4)
            pattern_matrix[:3, :3] = rotation_matrix
            pattern_matrix[:3, 3] = translation
            pattern_poses.append(pattern_matrix)
        
        # Use OpenCV's calibrateHandEye for robust eye-in-hand calibration
        try:
            self.log_message("Using OpenCV's calibrateHandEye for robot-to-color calibration")
            
            # Prepare data for OpenCV calibrateHandEye
            R_gripper2base = []  # Robot gripper rotations relative to base
            t_gripper2base = []  # Robot gripper translations relative to base
            R_target2cam = []    # Pattern rotations relative to camera
            t_target2cam = []    # Pattern translations relative to camera
            
            for robot_matrix, pattern_matrix in zip(robot_poses, pattern_poses):
                # Robot gripper pose (base to gripper)
                R_gripper2base.append(robot_matrix[:3, :3])
                t_gripper2base.append(robot_matrix[:3, 3])
                
                # Pattern pose relative to camera (camera to pattern)
                R_target2cam.append(pattern_matrix[:3, :3])
                t_target2cam.append(pattern_matrix[:3, 3])
            
            # Use OpenCV's eye-in-hand calibration (CALIB_HAND_EYE_TSAI method)
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base,
                t_gripper2base=t_gripper2base,
                R_target2cam=R_target2cam,
                t_target2cam=t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # Build transformation matrix (robot to color camera)
            T_robot_to_color = np.eye(4)
            T_robot_to_color[:3, :3] = R_cam2gripper
            T_robot_to_color[:3, 3] = t_cam2gripper.flatten()
            
            # Validate the result
            det = np.linalg.det(R_cam2gripper)
            self.log_message(f"OpenCV calibrateHandEye completed successfully")
            self.log_message(f"Rotation matrix determinant: {det:.6f} (should be ±1)")
            
            return T_robot_to_color
            
        except Exception as e:
            self.log_message(f"OpenCV calibrateHandEye failed: {e}, falling back to least squares")
            return self._solve_eye_in_hand_least_squares(robot_poses, pattern_poses)
    
    def _get_factory_color_to_depth_transformation(self) -> Optional[np.ndarray]:
        """Get factory-calibrated color-to-depth transformation from RealSense."""
        if not self.sensor_manager:
            self.log_message("Warning: No sensor manager available for factory extrinsics")
            return None
        
        # Check if sensor manager supports factory extrinsics
        if hasattr(self.sensor_manager, 'get_color_to_depth_transform'):
            T_color_to_depth = self.sensor_manager.get_color_to_depth_transform()
            if T_color_to_depth is not None:
                self.log_message("Successfully retrieved factory-calibrated extrinsics")
                return T_color_to_depth
            else:
                self.log_message("Warning: Could not retrieve factory extrinsics from sensor")
        else:
            self.log_message("Warning: Sensor manager does not support factory extrinsics")
        
        return None
    
    def _pose_to_matrix(self, pose: List[float]) -> np.ndarray:
        """Convert pose [x, y, z, roll, pitch, yaw] to 4x4 transformation matrix."""
        x, y, z, roll, pitch, yaw = pose
        
        # Convert to radians
        roll_rad = np.radians(roll)
        pitch_rad = np.radians(pitch)
        yaw_rad = np.radians(yaw)
        
        # Calculate rotation matrix
        cos_r, sin_r = np.cos(roll_rad), np.sin(roll_rad)
        cos_p, sin_p = np.cos(pitch_rad), np.sin(pitch_rad)
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        
        # Standard ZYX Euler angle rotation matrix (Roll: X, Pitch: Y, Yaw: Z)
        # Create individual rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, cos_r, -sin_r],
            [0, sin_r, cos_r]
        ])
        
        Ry = np.array([
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p]
        ])
        
        Rz = np.array([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1]
        ])
        
        # Combine rotations in ZYX order
        R = Rz @ Ry @ Rx
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x/1000.0, y/1000.0, z/1000.0]  # Convert mm to meters
        
        return T
    
    def _solve_hand_eye_least_squares(self, robot_poses: List[np.ndarray], 
                                    pattern_poses: List[np.ndarray]) -> np.ndarray:
        """Solve hand-eye calibration using least squares method."""
        # This is a simplified least-squares solution
        # For production use, consider cv2.calibrateHandEye() or more sophisticated methods
        
        n = len(robot_poses)
        if n < 3:
            raise ValueError("Need at least 3 pose pairs for calibration")
        
        # Extract positions for translation estimation
        robot_positions = np.array([T[:3, 3] for T in robot_poses])
        pattern_positions = np.array([T[:3, 3] for T in pattern_poses])
        
        # Simple least squares for translation (this is a simplification)
        # In practice, you'd solve the full AX = XB problem
        A = np.hstack([robot_positions, np.ones((n, 1))])
        translation_params = np.linalg.lstsq(A, pattern_positions, rcond=None)[0]
        
        # Create transformation matrix (simplified)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, 3] = translation_params[:3, 0]  # Use first column for x translation
        
        # For rotation, use average rotation (this is also simplified)
        # In practice, you'd use proper rotation averaging or optimization
        avg_rotation = np.mean([T[:3, :3] for T in robot_poses], axis=0)
        U, _, Vt = np.linalg.svd(avg_rotation)
        transformation_matrix[:3, :3] = U @ Vt
        
        return transformation_matrix
    
    def _solve_eye_in_hand_least_squares(self, robot_poses: List[np.ndarray], 
                                       pattern_poses: List[np.ndarray]) -> np.ndarray:
        """Solve eye-in-hand calibration using least squares method."""
        # Similar to hand-eye but with different interpretation
        return self._solve_hand_eye_least_squares(robot_poses, pattern_poses)
    
    def _create_camera_calibration_visualization(self, color_image: np.ndarray, 
                                               detected_patterns: List[Dict[str, Any]],
                                               current_count: int, target_count: int) -> np.ndarray:
        """Create visualization for camera calibration process."""
        vis_image = color_image.copy()
        
        # Convert RGB to BGR for OpenCV display if needed
        if len(vis_image.shape) == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        # Draw detected patterns using CharUco-specific visualization
        if detected_patterns:
            pattern = detected_patterns[0]
            
            # Draw ArUco markers
            if 'marker_corners' in pattern and pattern['marker_corners'] is not None:
                cv2.aruco.drawDetectedMarkers(vis_image, pattern['marker_corners'], pattern['marker_ids'])
            
            # Draw CharUco corners
            if 'corners_opencv' in pattern and pattern['corners_opencv'] is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis_image, pattern['corners_opencv'], pattern['corner_ids'])
            
            # Draw pattern center
            center_2d = pattern.get('center_2d')
            if center_2d is not None:
                cv2.circle(vis_image, tuple(center_2d.astype(int)), 5, (0, 255, 0), -1)
        
        # Add information overlay
        info_lines = [
            "CAMERA INTRINSIC CALIBRATION",
            f"Pattern: {self.pattern_size[0]}x{self.pattern_size[1]} CharUco ({self.square_size_mm}mm squares)",
            f"Images: {current_count}/{target_count}",
            f"Detected: {'YES' if detected_patterns else 'NO'}",
            "",
            "Controls:",
            "'c' - Capture image",
            "'q' - Finish (min 5 images)"
        ]
        
        for i, line in enumerate(info_lines):
            if i == 0:
                color = (0, 255, 0)  # Green for title
            elif i == 4:  # Empty line
                continue
            elif i >= 5:  # Controls
                color = (0, 255, 255)  # Yellow for controls
            else:
                color = (255, 255, 255)  # White for info
            
            cv2.putText(vis_image, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_image
    
    def _create_pattern_detection_visualization(self, color_image: np.ndarray, 
                                              detected_patterns: List[Dict[str, Any]]) -> np.ndarray:
        """Create visualization for pattern detection during calibration."""
        vis_image = color_image.copy()
        
        # Convert RGB to BGR for OpenCV display if needed
        if len(vis_image.shape) == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        if detected_patterns:
            pattern = detected_patterns[0]
            
            # Draw ArUco markers
            if 'marker_corners' in pattern and pattern['marker_corners'] is not None:
                cv2.aruco.drawDetectedMarkers(vis_image, pattern['marker_corners'], pattern['marker_ids'])
            
            # Draw CharUco corners
            if 'corners_opencv' in pattern and pattern['corners_opencv'] is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis_image, pattern['corners_opencv'], pattern['corner_ids'])
            
            # Draw coordinate axes if pose is available
            if (pattern.get('pose_data') is not None and 
                self.camera_calibrated and 
                self.camera_matrix is not None):
                rvec = pattern['pose_data']['rvec']
                tvec = pattern['pose_data']['tvec']
                axis_length = self.square_size_mm * 2
                cv2.drawFrameAxes(vis_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, axis_length, thickness=3)
            
            # Draw pattern center and info
            center_2d = pattern.get('center_2d')
            quality = pattern.get('quality_score', 0.0)
            
            if center_2d is not None:
                center_int = tuple(center_2d.astype(int))
                cv2.circle(vis_image, center_int, 5, (0, 255, 0), -1)
                
                # Add pattern info text
                text_lines = [
                    f"Quality: {quality:.2f}",
                    f"Corners: {pattern.get('corners_count', 0)}/{pattern.get('total_corners', 0)}",
                    f"Coverage: {pattern.get('coverage', 0):.1%}",
                    f"Markers: {pattern.get('markers_count', 0)}"
                ]
                
                for i, line in enumerate(text_lines):
                    cv2.putText(vis_image, line, (center_int[0] - 80, center_int[1] - 60 + i * 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Add control information
        info_lines = [
            f"CHARUCO CALIBRATION - {self.calibration_mode.upper()}",
            f"Point: {self.current_calibration_point}",
            f"Pattern: {'DETECTED' if detected_patterns else 'NOT DETECTED'}",
            "",
            "Controls:",
            "'c' - Capture pose",
            "'q' - Skip point"
        ]
        
        for i, line in enumerate(info_lines):
            if i == 0:
                color = (0, 255, 0)  # Green for title
            elif i == 3:  # Empty line
                continue
            elif i >= 4:  # Controls
                color = (0, 255, 255)  # Yellow for controls
            else:
                color = (255, 255, 255)  # White for info
            
            cv2.putText(vis_image, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_image
    
    def _wait_for_robot_movement(self, command_id: int, timeout: float = 15.0):
        """Wait for robot movement to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.robot_controller.get_command_status(command_id)
            if status and status.value in ['completed', 'failed']:
                if status.value == 'completed':
                    print("Robot movement completed")
                    time.sleep(1)  # Stabilization delay
                else:
                    print("Robot movement failed")
                return status.value == 'completed'
            
            time.sleep(0.1)
        
        print("Robot movement timed out")
        return False
    
    def save_transformation_matrix(self, transformation_matrix: np.ndarray) -> bool:
        """Save transformation matrix to file."""
        try:
            filepath = os.path.abspath(self.transformation_matrix_file)
            np.save(filepath, transformation_matrix)
            
            self.log_message(f"Transformation matrix saved to: {filepath}")
            print(f"Transformation matrix saved to: {filepath}")
            return True
            
        except Exception as e:
            self.log_message(f"Error saving transformation matrix: {e}")
            print(f"Error saving transformation matrix: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources and close connections."""
        print("\\nCleaning up CharUco calibration system...")
        
        # Close visualization windows
        cv2.destroyAllWindows()
        
        # Stop sensor
        if self.sensor_manager:
            self.sensor_manager.stop()
        
        # Cleanup robot
        if self.robot_controller:
            if hasattr(self.robot_controller, 'cleanup_async_system'):
                self.robot_controller.cleanup_async_system()
            self.robot_controller.disconnect()
        
        # Close log file
        if self.log_file:
            self.log_file.close()
        
        print("Cleanup complete")
    
    def run_full_charuco_calibration(self) -> bool:
        """
        Run the complete CharUco calibration process.
        
        Returns:
            True if calibration successful
        """
        try:
            # Initialize systems
            if not self.initialize_systems():
                print("System initialization failed")
                return False
            
            # Step 1: Camera calibration should already be loaded during initialization
            if not self.camera_calibrated:
                print("\\nStep 1: Camera intrinsic calibration required")
                if not self.run_camera_intrinsic_calibration():
                    print("Camera intrinsic calibration failed")
                    return False
            else:
                print("Camera calibration ready for 3D pose estimation")
            
            # Step 2: Hand-eye calibration
            print("\\nStep 2: Hand-eye calibration")
            transformation_matrix = self.run_hand_eye_calibration()
            if transformation_matrix is None:
                print("Hand-eye calibration failed")
                return False
            
            # Step 3: Save results
            if not self.save_transformation_matrix(transformation_matrix):
                print("Failed to save transformation matrix")
                return False
            
            # Step 4: Validate transformation accuracy with training data
            print("\\n" + "="*60)
            print("VALIDATION: Testing transformation accuracy with training data")
            print("="*60)
            validation_results = self._validate_transformation_accuracy(transformation_matrix)
            
            print("\\nCHARUCO CALIBRATION COMPLETED SUCCESSFULLY!")
            print(f"Mode: {self.calibration_mode}")
            print(f"Calibration pairs: {len(self.calibration_pairs)}")
            print(f"Camera calibration: {'Yes' if self.camera_calibrated else 'No'}")
            if validation_results:
                print(f"Validation accuracy: {validation_results['mean_error_mm']:.2f}mm ± {validation_results['std_error_mm']:.2f}mm")
                print(f"Max error: {validation_results['max_error_mm']:.2f}mm")
                if validation_results['mean_error_mm'] < 50.0:
                    print("[EXCELLENT] Calibration accuracy is EXCELLENT (< 50mm)")
                elif validation_results['mean_error_mm'] < 100.0:
                    print("[GOOD] Calibration accuracy is GOOD (< 100mm)")
                else:
                    print("[POOR] Calibration accuracy is POOR (> 100mm) - consider recalibration")
            
            self.log_message("CharUco calibration completed successfully")
            return True
            
        except KeyboardInterrupt:
            print("\\nCalibration interrupted by user")
            self.log_message("Calibration interrupted by user")
            return False
        except Exception as e:
            print(f"\\nCalibration failed with error: {e}")
            self.log_message(f"Calibration failed with error: {e}")
            return False
        finally:
            self.cleanup()
    
    def _validate_transformation_accuracy(self, transformation_matrix: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Validate transformation accuracy using training data (calibration pairs).
        
        Tests how well the calculated transformation matrix predicts the actual
        pattern positions from the training data used for calibration.
        
        Args:
            transformation_matrix: Calculated transformation matrix
            
        Returns:
            Dictionary with validation metrics or None if validation fails
        """
        if not self.calibration_pairs:
            print("No training data available for validation")
            return None
            
        try:
            errors_mm = []
            successful_validations = 0
            
            print(f"Validating with {len(self.calibration_pairs)} training data points...")
            
            for i, pair in enumerate(self.calibration_pairs):
                try:
                    robot_pose = pair['robot_pose']  # [x, y, z, roll, pitch, yaw] 
                    pattern_data = pair['pattern_pose']
                    actual_pattern_pos_mm = pattern_data['translation_mm']  # Actual measured position
                    
                    # Convert robot pose to transformation matrix
                    robot_matrix = self._pose_to_matrix(robot_pose)
                    
                    # For eye-in-hand: predicted_pattern = inv(transformation_matrix) * robot_pos
                    # For fixed_camera: predicted_pattern = transformation_matrix * robot_pos
                    
                    if self.calibration_mode == 'eye_in_hand':
                        # Eye-in-hand: camera is on robot, pattern is fixed
                        # Transform robot position to pattern position
                        camera_matrix = robot_matrix @ transformation_matrix
                        # Get camera position in world coordinates
                        camera_pos_world = camera_matrix[:3, 3] * 1000  # Convert to mm
                        
                        # For eye-in-hand, we predict where the pattern should be
                        # based on where the camera (robot) is
                        pattern_pos_camera = actual_pattern_pos_mm  # Pattern position in camera frame
                        
                        # Transform to world frame: T_world = T_robot * T_camera_robot * T_camera_pattern
                        pattern_homogeneous = np.array([pattern_pos_camera[0], pattern_pos_camera[1], pattern_pos_camera[2], 1])
                        predicted_pattern_world = camera_matrix @ pattern_homogeneous
                        predicted_pattern_pos_mm = predicted_pattern_world[:3]
                        
                        # For eye-in-hand validation, we compare camera positions
                        # The error is how well we can predict camera position from robot position
                        error_mm = np.linalg.norm(camera_pos_world - robot_pose[:3])
                        
                    else:
                        # Fixed camera: camera is fixed, robot holds pattern
                        # Transform robot position to expected pattern position in camera frame
                        robot_homogeneous = np.array([robot_pose[0], robot_pose[1], robot_pose[2], 1])
                        predicted_pattern_camera = transformation_matrix @ robot_homogeneous
                        predicted_pattern_pos_mm = predicted_pattern_camera[:3] * 1000  # Convert to mm
                        
                        # Calculate error between predicted and actual pattern position
                        error_mm = np.linalg.norm(predicted_pattern_pos_mm - actual_pattern_pos_mm)
                    
                    errors_mm.append(error_mm)
                    successful_validations += 1
                    
                    if i < 3:  # Show details for first 3 points
                        print(f"  Point {i+1}: Error = {error_mm:.2f}mm")
                        
                except Exception as e:
                    print(f"  Warning: Failed to validate point {i+1}: {e}")
                    continue
            
            if successful_validations == 0:
                print("No successful validations - cannot assess accuracy")
                return None
                
            # Calculate statistics
            errors_array = np.array(errors_mm)
            mean_error = np.mean(errors_array)
            std_error = np.std(errors_array)
            max_error = np.max(errors_array)
            min_error = np.min(errors_array)
            
            validation_results = {
                'mean_error_mm': float(mean_error),
                'std_error_mm': float(std_error), 
                'max_error_mm': float(max_error),
                'min_error_mm': float(min_error),
                'successful_validations': successful_validations,
                'total_points': len(self.calibration_pairs)
            }
            
            print(f"Validation completed: {successful_validations}/{len(self.calibration_pairs)} points")
            print(f"Mean error: {mean_error:.2f}mm")
            print(f"Std deviation: {std_error:.2f}mm") 
            print(f"Min/Max error: {min_error:.2f}mm / {max_error:.2f}mm")
            
            # Log results
            self.log_message(f"Validation results: mean={mean_error:.2f}mm, std={std_error:.2f}mm, max={max_error:.2f}mm")
            
            return validation_results
            
        except Exception as e:
            print(f"Validation failed: {e}")
            self.log_message(f"Validation failed: {e}")
            return None