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

# Import robot controller
try:
    from module.robot.mycobot_controller import MyCobotController
    ROBOT_AVAILABLE = True
except ImportError:
    print("Robot controller not available - will use manual input")
    ROBOT_AVAILABLE = False


class RealtimeCharucoPredictor:
    """Real-time CharUco position prediction system."""
    
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
        
        # Coordinate system option (now using simple method)
        self.use_center_origin = use_center_origin
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = np.zeros(4)
        
        # ArUco/CharUco setup
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.board = aruco.CharucoBoard(self.board_size, self.square_length, self.marker_length, self.dictionary)
        
        # Detector setup
        self.setup_detectors()
        
        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Robot controller
        self.robot = None
        if ROBOT_AVAILABLE:
            try:
                self.robot = MyCobotController()
                # Attempt to connect to robot
                if self.robot.connect():
                    print("Robot controller initialized and connected successfully")
                else:
                    print("Robot controller initialized but connection failed - using manual input")
                    # Don't set to None, keep for manual fallback
            except Exception as e:
                print(f"Failed to initialize robot: {e}")
                self.robot = None
        
        # Load camera intrinsics first (may be overridden by calibration)
        self.load_camera_intrinsics()
        
        # Load pre-calibrated eye-in-hand transformation (this will override intrinsics with calibration values)
        self.eye_in_hand_transform = self.load_eye_in_hand_transform()
        
        # Position tracking for stability analysis
        self.recent_positions = []
        self.position_history_size = 10
        
        # Robot movement presets
        self.load_robot_poses()
        self.current_pose_index = 0
        
        # Frame counter for saving
        self.frame_count = 0
    
    def load_eye_in_hand_transform(self):
        """Load pre-calibrated eye-in-hand transformation matrix and its intrinsics."""
        # Always use bottom-left calibration (center origin uses simple offset method)
        if self.use_center_origin:
            print("Using center origin (simple offset method)")
        else:
            print("Using bottom-left origin")
        calibration_dir = self.config_path / 'calibration_result_1754622347'
        transform_file = calibration_dir / 'eye_in_hand_transform.npy'
        intrinsics_file = calibration_dir / 'camera_intrinsics.txt'
        calibration_json = calibration_dir / 'calibration_data.json'
        
        try:
            # Load transformation matrix
            transform = np.load(transform_file)
            print(f"Loaded eye-in-hand transformation from: {transform_file}")
            print("Transformation matrix:")
            for row in transform:
                print(f"  [{row[0]:8.5f} {row[1]:8.5f} {row[2]:8.5f} {row[3]:8.5f}]")
            
            # Try to load calibration intrinsics
            if intrinsics_file.exists():
                calibration_intrinsics = np.loadtxt(intrinsics_file)
                print(f"\nFound calibration intrinsics: {intrinsics_file}")
                print("Calibration used these intrinsics:")
                print(f"  fx={calibration_intrinsics[0,0]:.2f}, fy={calibration_intrinsics[1,1]:.2f}")
                print(f"  cx={calibration_intrinsics[0,2]:.2f}, cy={calibration_intrinsics[1,2]:.2f}")
                
                # Override the loaded intrinsics with calibration intrinsics
                self.camera_matrix = calibration_intrinsics
                self.dist_coeffs = np.zeros(5)
                print("USING CALIBRATION INTRINSICS (recommended)")
                self.setup_detectors()  # Update detectors with correct intrinsics
                
            elif calibration_json.exists():
                # Try to load from JSON file
                try:
                    with open(calibration_json, 'r') as f:
                        calib_data = json.load(f)
                    if 'camera_intrinsics' in calib_data:
                        calibration_intrinsics = np.array(calib_data['camera_intrinsics']['camera_matrix'])
                        print(f"\nFound calibration intrinsics in JSON: {calibration_json}")
                        print("Calibration used these intrinsics:")
                        print(f"  fx={calibration_intrinsics[0,0]:.2f}, fy={calibration_intrinsics[1,1]:.2f}")
                        print(f"  cx={calibration_intrinsics[0,2]:.2f}, cy={calibration_intrinsics[1,2]:.2f}")
                        
                        self.camera_matrix = calibration_intrinsics
                        self.dist_coeffs = np.array(calib_data['camera_intrinsics']['dist_coeffs'])
                        print("USING CALIBRATION INTRINSICS (recommended)")
                        self.setup_detectors()
                except Exception as e:
                    print(f"Could not load intrinsics from JSON: {e}")
            else:
                print(f"⚠️  WARNING: No calibration intrinsics found!")
                print(f"   Looked for: {intrinsics_file}")  
                print(f"   This transformation was calibrated with different intrinsics")
                print(f"   Using currently loaded intrinsics may give wrong results")
            
            return transform
            
        except FileNotFoundError:
            print(f"Error: Eye-in-hand transformation not found at: {transform_file}")
            print("Please run the calibration first using charuco_realtime_pose.py")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading eye-in-hand transformation: {e}")
            sys.exit(1)
    
    def predict_charuco_position(self, charuco_transform, robot_transform):
        """
        Predict CharUco position in robot base coordinates using eye-in-hand calibration.
        
        Args:
            charuco_transform: 4x4 camera-to-charuco transformation matrix
            robot_transform: 4x4 robot base-to-gripper transformation matrix
            
        Returns:
            np.array: CharUco position in robot base coordinates [x, y, z] in mm
        """
        # Pattern in robot base = Robot_base_to_gripper @ Eye_in_hand @ Camera_to_pattern
        pattern_in_base = robot_transform @ self.eye_in_hand_transform @ charuco_transform
        position_mm = pattern_in_base[:3, 3] * 1000  # Convert to millimeters
        return position_mm
    
    def update_position_history(self, position):
        """Update position history for stability analysis."""
        self.recent_positions.append(position.copy())
        if len(self.recent_positions) > self.position_history_size:
            self.recent_positions.pop(0)
    
    def get_position_stability(self):
        """Get position stability metrics."""
        if len(self.recent_positions) < 2:
            return None, None
        
        positions = np.array(self.recent_positions)
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        return mean_pos, std_pos
    
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
                cv2.drawFrameAxes(vis_image, self.camera_matrix, self.dist_coeffs,
                                results['rvec'], results['tvec'], self.square_length, 3)
        
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
        
    def setup_detectors(self):
        """Setup ArUco and CharUco detectors."""
        # ArUco detector parameters
        detector_params = aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        self.aruco_detector = aruco.ArucoDetector(self.dictionary, detector_params)
        
        # CharUco detector parameters
        charuco_params = aruco.CharucoParameters()
        if self.camera_matrix is not None:
            charuco_params.cameraMatrix = self.camera_matrix
        self.charuco_detector = aruco.CharucoDetector(self.board, charuco_params)
        
    def load_camera_intrinsics(self):
        """Load camera intrinsics from file for comparison testing."""
        intrinsic_file = self.config_path / 'intrinsic.txt'
        if intrinsic_file.exists():
            try:
                self.camera_matrix = np.loadtxt(intrinsic_file)
                print(f"Loaded camera intrinsics from {intrinsic_file}")
                print(f"Camera matrix:")
                print(f"  fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
                print(f"  cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}")
                print(f"Matrix:\n{self.camera_matrix}")
                
                # Use minimal distortion for calibration intrinsics
                self.dist_coeffs = np.zeros(5)
                print(f"Distortion coefficients: {self.dist_coeffs}")
                
                # Update CharUco detector with camera matrix
                self.setup_detectors()
            except Exception as e:
                print(f"Warning: Could not load camera intrinsics: {e}")
                print("Using default camera parameters")
                # Use reasonable defaults for RealSense
                self.camera_matrix = np.array([
                    [615.0, 0.0, 320.0],
                    [0.0, 615.0, 240.0],
                    [0.0, 0.0, 1.0]
                ])
                self.dist_coeffs = np.zeros(5)
        else:
            print(f"No intrinsics file found at {intrinsic_file}")
            print("Using default camera parameters")
            self.camera_matrix = np.array([
                [615.0, 0.0, 320.0],
                [0.0, 615.0, 240.0],
                [0.0, 0.0, 1.0]
            ])
            self.dist_coeffs = np.zeros(5)
    
    def setup_camera(self):
        """Setup RealSense camera with L515-optimized settings."""
        try:
            # Try different resolutions for L515 (which supports higher resolutions)
            resolutions_to_try = [
                (1280, 720, 30),  # HD - good for L515
                (960, 540, 30),   # Good balance
                (640, 480, 30),   # Standard
            ]
            
            profile = None
            for width, height, fps in resolutions_to_try:
                try:
                    print(f"Trying resolution: {width}x{height}@{fps}fps")
                    self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                    profile = self.pipeline.start(self.config)
                    print(f"Successfully initialized at {width}x{height}@{fps}fps")
                    break
                except Exception as e:
                    print(f"Failed {width}x{height}: {e}")
                    self.pipeline.stop()
                    self.config.disable_all_streams()
                    continue
            
            if profile is None:
                raise Exception("No supported resolution found")
            
            # Get camera intrinsics from RealSense SDK
            color_profile = profile.get_stream(rs.stream.color)
            intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            # Update camera matrix with RealSense intrinsics
            self.camera_matrix = np.array([
                [intrinsics.fx, 0.0, intrinsics.ppx],
                [0.0, intrinsics.fy, intrinsics.ppy],
                [0.0, 0.0, 1.0]
            ])
            
            # Get distortion coefficients from RealSense
            self.dist_coeffs = np.array(intrinsics.coeffs[:5])  # k1, k2, p1, p2, k3
            
            # Update detectors with new camera matrix
            self.setup_detectors()
            
            # Get device info
            device = profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            serial_number = device.get_info(rs.camera_info.serial_number)
            firmware_version = device.get_info(rs.camera_info.firmware_version)
            
            print("RealSense camera initialized successfully")
            print(f"Device: {device_name} (S/N: {serial_number}, FW: {firmware_version})")
            print(f"Resolution: {intrinsics.width}x{intrinsics.height}")
            print(f"Camera intrinsics from RealSense SDK:")
            print(f"  fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
            print(f"  cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
            print(f"Camera matrix:\n{self.camera_matrix}")
            print(f"Distortion coefficients: {self.dist_coeffs}")
            
            # Save intrinsics to file for later use
            intrinsics_file = self.config_path / 'intrinsic_from_realsense.txt'
            np.savetxt(intrinsics_file, self.camera_matrix)
            
            # Save detailed calibration data
            calibration_data = {
                'camera_matrix': self.camera_matrix.tolist(),
                'distortion_coefficients': self.dist_coeffs.tolist(),
                'device_info': {
                    'name': device_name,
                    'serial_number': serial_number,
                    'firmware_version': firmware_version
                },
                'intrinsics': {
                    'fx': intrinsics.fx,
                    'fy': intrinsics.fy,
                    'ppx': intrinsics.ppx,
                    'ppy': intrinsics.ppy,
                    'width': intrinsics.width,
                    'height': intrinsics.height,
                    'model': str(intrinsics.model)
                }
            }
            
            import json
            with open(self.config_path / 'camera_calibration_realsense.json', 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            print(f"Intrinsics saved to: {intrinsics_file}")
            print(f"Detailed calibration saved to: camera_calibration_realsense.json")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize RealSense camera: {e}")
            return False
    
    def check_subpixel(self, corners):
        """Check if corners are subpixel accurate."""
        for xy in np.asarray(corners).flatten():
            if not xy.is_integer():
                return True
        return False
    
    def match_image_points_corrected(self, charuco_corners, charuco_ids):
        """
        Match image points with corrected coordinate system (bottom-left origin, outward Z).
        This matches the notebook's coordinate system and standard robotics convention.
        """
        objPoints = []
        imgPoints = []
        
        for i in range(len(charuco_ids)):
            index = charuco_ids[i]
            # Get the corner position from the board
            corner_pos = self.board.getChessboardCorners()[index]
            
            # Apply coordinate system correction (bottom-left origin)
            # Flip Y-axis to make bottom-left the origin (instead of top-left)
            corrected_pos = corner_pos.copy()
            corrected_pos[0][1] = self.board.getRightBottomCorner()[1] - corrected_pos[0][1]
            
            objPoints.append(corrected_pos)
            imgPoints.append(charuco_corners[i])
        
        return np.array(objPoints), np.array(imgPoints)
    
    def convert_to_center_origin_simple(self, tvec, rvec):
        """
        Convert bottom-left origin tvec to center origin using simple offset approach.
        
        Args:
            tvec: Translation vector from bottom-left origin detection
            rvec: Rotation vector from detection
            
        Returns:
            tvec_center: Translation vector for center origin
        """
        # Real offset values from --center-origin implementation
        # center_x = (board_size[0] - 1) * square_length / 2 + square_length / 2
        # center_y = (board_size[1] - 1) * square_length / 2 + square_length / 2
        center_x = (self.board_size[0] - 1) * self.square_length / 2 + self.square_length / 2
        center_y = (self.board_size[1] - 1) * self.square_length / 2 + self.square_length / 2
        offset_m = np.array([center_x, center_y, 0.0])  # meters
        
        # Get rotation matrix to transform offset to camera frame
        rot_matrix, _ = cv2.Rodrigues(rvec)
        offset_in_camera = rot_matrix @ offset_m
        
        # Apply offset (add to move from bottom-left to center position)
        tvec_center = tvec.flatten() + offset_in_camera
        
        return tvec_center.reshape(-1, 1)
    
    def get_robot_pose(self):
        """Get current robot end-effector pose."""
        if self.robot is not None:
            # Check robot connection status
            if hasattr(self.robot, 'is_connected') and not self.robot.is_connected:
                print("Tip: Make sure myCobot is powered on and connected via USB")
            else:
                try:
                    # Get robot pose [x, y, z, rx, ry, rz]
                    pose = self.robot.get_current_position()
                    if pose and len(pose) >= 6:
                        return pose[:6]  # [x, y, z, rx, ry, rz]
                    else:
                        print("Tip: Robot may need to be powered on or moved manually first")
                except Exception as e:
                    print(f"Failed to get robot pose: {e}")
        else:
            pass  # No robot controller available
        
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
        
        # Save calibration data
        calibration_data = {
            'num_points': len(self.calibration_pairs),
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
        """Validate calibration quality using proper hand-eye calibration validation."""
        if len(self.calibration_pairs) < 3:
            return None
        
        print("\nValidating calibration quality with known pattern position...")
        
        validation_results = {
            'reprojection_errors': [],
            'pose_errors': [],
            'mean_reprojection_error': 0.0,
            'max_reprojection_error': 0.0,
            'mean_translation_error': 0.0,
            'mean_rotation_error': 0.0,
            'quality_score': 'Unknown'
        }
        
        # Known CharUco pattern position in robot base coordinates  
        # myCobot 280: +X=forward, +Y=left side, +Z=up
        pattern_position_robot = np.array([0.3, 0.1, 0.0])  # [300mm forward, 100mm left, 0mm height]
        
        translation_errors = []
        rotation_errors = []
        
        for i, pair in enumerate(self.calibration_pairs):
            # Get transforms
            robot_transform = pair['robot_transform']  # Robot base to gripper
            charuco_transform = pair['charuco_transform']  # Camera to CharUco pattern
            
            # Calculate expected pattern position in robot base frame using eye-in-hand calibration
            # Pattern in robot base = Robot_base_to_gripper @ Eye_in_hand @ Camera_to_pattern
            expected_pattern_in_base = robot_transform @ eye_in_hand_transform @ charuco_transform
            expected_position = expected_pattern_in_base[:3, 3]
            
            # Calculate error from known pattern position
            position_error = np.linalg.norm(expected_position - pattern_position_robot)
            translation_errors.append(position_error * 1000)  # Convert to mm
            
            # Calculate proper rotation error with coordinate correction applied
            # With the 90° Z-rotation fix, the pattern should now have identity rotation in robot coordinates
            # This validates that the coordinate system correction is working properly
            expected_rotation = np.eye(3)
            predicted_rotation = expected_pattern_in_base[:3, :3]
            
            # Calculate rotation error from identity matrix
            rotation_error_matrix = predicted_rotation @ expected_rotation.T
            rotation_error_trace = np.trace(rotation_error_matrix)
            rotation_error_angle = np.arccos(np.clip((rotation_error_trace - 1) / 2, -1, 1))
            rotation_error_degrees = np.degrees(rotation_error_angle)
            rotation_errors.append(rotation_error_degrees)
            
            print(f"  Point {i+1}: Expected pattern at [{expected_position[0]*1000:.1f}, {expected_position[1]*1000:.1f}, {expected_position[2]*1000:.1f}]mm")
            print(f"           Known position at [300.0, 100.0, 0.0]mm (forward, left, height)")
            print(f"           Position error = {position_error*1000:.1f}mm")
        
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
    
    def print_validation_report(self, validation_results):
        """Print a detailed validation report."""
        if validation_results is None:
            print("No validation data available")
            return
        
        print("\n" + "="*50)
        print("CALIBRATION QUALITY REPORT")
        print("="*50)
        print(f"Number of calibration points: {len(self.calibration_pairs)}")
        print(f"Mean translation error: {validation_results['mean_translation_error']:.2f} mm")
        print(f"Mean rotation error: {validation_results['mean_rotation_error']:.2f}°")
        print(f"Max error: {validation_results['max_reprojection_error']:.2f} mm")
        print(f"Quality assessment: {validation_results['quality_score']}")
        
        print("\nPer-point errors:")
        for i, error in enumerate(validation_results['reprojection_errors']):
            print(f"  Point {i+1}: {error:.2f} mm")
        
        print("\nQuality Guidelines:")
        print("  Excellent: <5mm translation, <2° rotation")
        print("  Good:      <10mm translation, <5° rotation")
        print("  Fair:      <20mm translation, <10° rotation")
        print("  Poor:      >20mm translation or >10° rotation")
        
        if validation_results['quality_score'] in ['Poor', 'Fair']:
            print("\nSuggestions for improvement:")
            print("  - Collect more calibration points")
            print("  - Use more diverse robot poses (different angles/distances)")
            print("  - Ensure CharUco pattern is clearly visible in all poses")
            print("  - Check for camera calibration accuracy")
            print("  - Verify robot pose accuracy")
        
        print("="*50)
    
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