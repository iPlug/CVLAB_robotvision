#!/usr/bin/env python3
"""
CharUco Utils - Common functionality for CharUco pose estimation applications

This module provides utility classes to eliminate code duplication across:
- charuco_realtime_pose.py (calibration)
- charuco_realtime_prediction.py (real-time prediction)  
- lidar_realtime_prediction.py (LiDAR-based detection)

Classes:
- RobotUtils: Robot connection and movement management
- CameraUtils: Camera setup and intrinsics management
- CalibrationUtils: Eye-in-hand calibration workflow
- CoordinateUtils: Coordinate transformations and pose handling
- DetectionUtils: Common detection utilities and result saving
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

# Import robot controller
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from module.robot.mycobot_controller import MyCobotController
    ROBOT_AVAILABLE = True
except ImportError:
    ROBOT_AVAILABLE = False

from cv2 import aruco


class RobotUtils:
    """Utilities for robot connection, movement, and pose management."""
    
    def __init__(self):
        """Initialize robot utilities."""
        self.robot = None
        self.preset_poses = []
        self.current_pose_index = 0
        self.initialize_robot()
        
    def initialize_robot(self):
        """Initialize robot controller if available."""
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
    
    def get_robot_pose(self):
        """Get current robot end-effector pose."""
        if self.robot is not None:
            # Check robot connection status
            if hasattr(self.robot, 'is_connected') and not self.robot.is_connected:
                # Robot exists but not connected - silent operation
                pass
            else:
                try:
                    # Get robot pose [x, y, z, rx, ry, rz]
                    pose = self.robot.get_current_position()
                    if pose and len(pose) >= 6:
                        return pose[:6]  # [x, y, z, rx, ry, rz]
                except Exception as e:
                    # Silent failure - robot may be unavailable
                    pass
        else:
            # No robot controller - silent operation
            pass
        
        # Fallback: return None to retry automatically
        return None


class CameraUtils:
    """Utilities for camera setup, intrinsics management, and eye-in-hand calibration loading."""
    
    def __init__(self, config_path="charuco-pose-estimation"):
        """Initialize camera utilities."""
        self.config_path = Path(config_path)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros(4)
        self.color_intrinsics = None
        
    def load_camera_intrinsics(self):
        """Load camera intrinsics from file."""
        intrinsic_file = self.config_path / 'intrinsic_l515.txt'
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
    
    def setup_camera(self, pipeline, config):
        """Setup RealSense camera with L515 optimized settings."""
        try:
            # L515 LiDAR optimized configurations - using MATCHING resolutions for proper alignment
            # Different resolutions cause point cloud cropping and misalignment issues
            color_depth_configs = [
                ((640, 480, 30), (640, 480, 30)),      # Matching resolutions for perfect alignment
                ((1280, 720, 30), (1280, 720, 30)),    # High-res backup (if supported)
            ]
            
            profile = None
            for color_config, depth_config in color_depth_configs:
                color_w, color_h, color_fps = color_config
                depth_w, depth_h, depth_fps = depth_config
                try:
                    print(f"Trying color: {color_w}x{color_h}@{color_fps}fps, depth: {depth_w}x{depth_h}@{depth_fps}fps")
                    # Enable L515-optimized streams
                    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)
                    config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)
                    profile = pipeline.start(config)
                    
                    # Apply L515 Short Range Preset for optimal 0.25m-1.5m range
                    try:
                        device = profile.get_device()
                        depth_sensor = device.first_depth_sensor()
                        
                        # Set L515 Short Range visual preset
                        if hasattr(rs, 'l500_visual_preset') and hasattr(depth_sensor, 'set_option'):
                            depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.short_range)
                            print("✅ Applied L515 Short Range Preset (0.25m-1.5m optimal)")
                        else:
                            print("⚠️  L515 visual presets not available, using default settings")
                    except Exception as preset_error:
                        print(f"⚠️  Could not apply L515 preset: {preset_error}")
                    
                    print(f"Successfully initialized - Color: {color_w}x{color_h}@{color_fps}fps, Depth: {depth_w}x{depth_h}@{depth_fps}fps")
                    break
                except Exception as e:
                    print(f"Failed color:{color_w}x{color_h} depth:{depth_w}x{depth_h}: {e}")
                    try:
                        pipeline.stop()
                    except:
                        pass  # Ignore if pipeline wasn't started
                    config.disable_all_streams()
                    continue
            
            if profile is None:
                raise Exception("No supported resolution found")
            
            # Get camera intrinsics from RealSense SDK
            color_profile = profile.get_stream(rs.stream.color)
            intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            self.color_intrinsics = intrinsics
            
            # Check if we need to scale calibration intrinsics for different resolution
            color_width, color_height = intrinsics.width, intrinsics.height
            calibration_width, calibration_height = 1280, 720  # Original calibration resolution
            
            if (color_width, color_height) == (calibration_width, calibration_height):
                print("✅ Perfect match! Using original calibration resolution")
                print("Using original calibration resolution - no scaling needed")
                # Use the pre-calibrated intrinsics (they will override RealSense intrinsics if available)
                pass  # calibration intrinsics loaded in load_eye_in_hand_transform()
            else:
                print(f"⚠️  Resolution mismatch: Current {color_width}x{color_height}, Calibration {calibration_width}x{calibration_height}")
                print("Will use scaled calibration intrinsics for alignment (640x480 → 1280x720)")
                print("Note: Using matching color/depth resolutions to prevent point cloud cropping")
                
                # Calculate scaling factors
                scale_x = color_width / calibration_width
                scale_y = color_height / calibration_height
                
                # Scale the calibration intrinsics to current resolution
                if hasattr(self, 'camera_matrix') and self.camera_matrix is not None:
                    # We have calibration intrinsics - scale them
                    original_fx = self.camera_matrix[0, 0]
                    original_fy = self.camera_matrix[1, 1] 
                    original_cx = self.camera_matrix[0, 2]
                    original_cy = self.camera_matrix[1, 2]
                    
                    scaled_fx = original_fx * scale_x
                    scaled_fy = original_fy * scale_y
                    scaled_cx = original_cx * scale_x
                    scaled_cy = original_cy * scale_y
                    
                    self.camera_matrix = np.array([
                        [scaled_fx, 0.0, scaled_cx],
                        [0.0, scaled_fy, scaled_cy],
                        [0.0, 0.0, 1.0]
                    ])
                    
                    print(f"Scaled calibration intrinsics:")
                    print(f"  fx={scaled_fx:.2f} (was {original_fx:.2f})")
                    print(f"  fy={scaled_fy:.2f} (was {original_fy:.2f})")
                    print(f"  cx={scaled_cx:.2f} (was {original_cx:.2f})")
                    print(f"  cy={scaled_cy:.2f} (was {original_cy:.2f})")
                else:
                    print("No calibration intrinsics found - using RealSense intrinsics")
                    # Fall back to RealSense intrinsics
                    self.camera_matrix = np.array([
                        [intrinsics.fx, 0.0, intrinsics.ppx],
                        [0.0, intrinsics.fy, intrinsics.ppy],
                        [0.0, 0.0, 1.0]
                    ])
            
            # Get distortion coefficients from RealSense
            self.dist_coeffs = np.array(intrinsics.coeffs[:5])  # k1, k2, p1, p2, k3
            
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
            try:
                intrinsics_file = 'intrinsic_from_realsense.txt'
                np.savetxt(intrinsics_file, self.camera_matrix)
                print(f"Intrinsics saved to: {intrinsics_file}")
            except Exception as e:
                print(f"Could not save intrinsics file: {e}")
            
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
            
            try:
                with open('camera_calibration_realsense.json', 'w') as f:
                    json.dump(calibration_data, f, indent=2)
                print(f"Detailed calibration saved to: camera_calibration_realsense.json")
            except Exception as e:
                print(f"Could not save calibration data: {e}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize RealSense camera: {e}")
            return False
    
    def load_eye_in_hand_transform(self):
        """Load pre-calibrated eye-in-hand transformation matrix and its intrinsics."""
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


class CalibrationUtils:
    """Utilities for eye-in-hand calibration workflow and validation."""
    
    def __init__(self, config_path="charuco-pose-estimation"):
        """Initialize calibration utilities."""
        self.config_path = Path(config_path)
        self.calibration_pairs = []
    
    def collect_calibration_point(self, results, coordinate_utils, robot_utils):
        """Collect robot and CharUco poses for calibration."""
        if not results['valid_pose']:
            print("No CharUco pose detected - cannot collect calibration point")
            return False
        
        print(f"\nCollecting calibration point {len(self.calibration_pairs) + 1}...")
        
        # Get robot pose
        robot_pose = robot_utils.get_robot_pose()
        if robot_pose is None:
            return False
        
        # Get CharUco pose
        charuco_rvec = results['rvec']
        charuco_tvec = results['tvec']
        
        # Convert to transformation matrices
        robot_transform = coordinate_utils.pose_to_transform_matrix(robot_pose)
        charuco_transform = coordinate_utils.charuco_pose_to_transform_matrix(charuco_rvec, charuco_tvec)
        
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
    
    def save_calibration_results(self, eye_in_hand_transform, camera_utils, coordinate_info=None):
        """Save calibration results to file."""
        timestamp = int(time.time())
        output_dir = self.config_path / f"calibration_result_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        # Save transformation matrix
        np.savetxt(output_dir / 'eye_in_hand_transform.txt', eye_in_hand_transform)
        np.save(output_dir / 'eye_in_hand_transform.npy', eye_in_hand_transform)
        
        # Save camera intrinsics used for this calibration
        np.savetxt(output_dir / 'camera_intrinsics.txt', camera_utils.camera_matrix)
        
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
{camera_utils.camera_matrix}

Distortion Coefficients Used:
{camera_utils.dist_coeffs}

Number of calibration points: {len(self.calibration_pairs)}
"""
        
        with open(output_dir / 'README.txt', 'w') as f:
            f.write(readme_content)
        
        # Save calibration data including camera intrinsics
        calibration_data = {
            'timestamp': timestamp,
            'num_points': len(self.calibration_pairs),
            'camera_intrinsics': {
                'camera_matrix': camera_utils.camera_matrix.tolist(),
                'dist_coeffs': camera_utils.dist_coeffs.tolist(),
                'source': 'intrinsic.txt',
                'note': 'These intrinsics were used during calibration and MUST be used with this transformation matrix'
            },
            'eye_in_hand_transform': eye_in_hand_transform.tolist(),
            'points': []
        }
        
        if coordinate_info:
            calibration_data['coordinate_system'] = coordinate_info
        
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
            
            # Calculate proper rotation error
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
        robot_transforms = [pair['robot_transform'] for pair in self.calibration_pairs]
        
        # Check pose diversity
        positions = np.array([T[:3, 3] for T in robot_transforms])
        position_span = np.max(positions, axis=0) - np.min(positions, axis=0)
        position_diversity = np.linalg.norm(position_span)
        
        # Check rotation diversity  
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


class CoordinateUtils:
    """Utilities for coordinate transformations and pose handling."""
    
    def __init__(self, board_size=(7, 5), square_length=0.039, marker_length=0.029):
        """Initialize coordinate utilities."""
        self.board_size = board_size
        self.square_length = square_length
        self.marker_length = marker_length
        self.recent_positions = []
        self.position_history_size = 10
        
        # Create CharUco board for coordinate calculations
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.board = aruco.CharucoBoard(board_size, square_length, marker_length, dictionary)
    
    def pose_to_transform_matrix(self, pose):
        """Convert [x, y, z, rx, ry, rz] pose to 4x4 transformation matrix."""
        x, y, z, rx, ry, rz = pose
        
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
    
    def charuco_pose_to_transform_matrix(self, rvec, tvec):
        """Convert CharUco rvec, tvec to 4x4 transformation matrix with coordinate correction."""
        # Convert rotation vector to rotation matrix
        rot_matrix, _ = cv2.Rodrigues(rvec)
        
        # Apply coordinate system correction: 90° rotation around Z-axis
        # This fixes the systematic 95° rotation error by aligning CharUco axes with robot axes
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


class DetectionUtils:
    """Utilities for common detection operations and result handling."""
    
    def __init__(self, config_path="charuco-pose-estimation"):
        """Initialize detection utilities."""
        self.config_path = Path(config_path)
    
    def check_subpixel(self, corners):
        """Check if corners are subpixel accurate."""
        for xy in np.asarray(corners).flatten():
            if not xy.is_integer():
                return True
        return False
    
    def save_results(self, image, results, output_dir_name="realtime_output"):
        """Save detection results to file."""
        timestamp = int(time.time())
        output_dir = self.config_path / f"{output_dir_name}_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        # Save original image
        cv2.imwrite(str(output_dir / "image.png"), image)
        
        # Save pose data if available
        if results.get('valid_pose', False):
            # Save transformation matrices if we have rvec/tvec
            if 'rvec' in results and 'tvec' in results:
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
            
            print(f"Results saved to {output_dir}")
        else:
            print(f"Image saved to {output_dir} (no pose detected)")
        
        return output_dir