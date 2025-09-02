#!/usr/bin/env python3
"""
CharUco-Enhanced Virtual Fixed Camera Object Tracking Application

This application extends virtual_fixed_tracking.py with high-precision CharUco coordinate
transformation for better object position prediction accuracy. Uses pre-calibrated eye-in-hand 
transformation from the charuco-pose-estimation system for camera-to-robot coordinate conversion.

Key improvements over virtual_fixed_tracking.py:
- Uses CharUco-calibrated camera-to-robot coordinate transformation matrix (~20mm accuracy)
- Applies CharUco coordinate system corrections (90° Z-rotation fix)
- Uses L515 camera intrinsics that match the coordinate transformation calibration
- Provides position accuracy monitoring and stability metrics

Usage:
    python virtual_fixed_tracking_charuco.py [--config CONFIG_NAME] [--strategy STRATEGY] [--bag BAG_FILE]

Requirements:
    - CharUco eye-in-hand coordinate transformation calibration must be completed first
    - Intel RealSense L515 LiDAR camera mounted on robot end-effector  
    - Robot starts and returns to observation pose [87.5, -62.7, 367.2, -144.03, 29.89, -124.04]

Enhanced Features compared to virtual_fixed_tracking.py:
    - CharUco camera-to-robot coordinate transformation matrix with ~20mm position accuracy
    - Coordinate system corrections for proper robot alignment
    - L515 intrinsics consistency with coordinate transformation calibration
    - Position stability and accuracy metrics
    - Coordinate transformation quality monitoring
    - Same robot control and visualization features
"""

import argparse
import sys
import os
import time
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import RANSACRegressor
from pathlib import Path
import json

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.core.object_detector import ObjectDetector
from module.core.visualization_engine import VisualizationEngine
from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
from module.strategies.dbscan_clustering_strategy import DBSCANClusteringStrategy
from module.sensors.realsense_manager import RealSenseManager
from module.processing.temporal_filter import TemporalFilter
from module.config.config_factory import ConfigFactory
from module.robot.mycobot_controller import MyCobotController
from camera_to_robot_transformer import CameraToRobotTransformer

"""
IMPORTANT: Understanding CharUco System Purpose

CharUco is NOT camera calibration - it is COORDINATE TRANSFORMATION calibration.

What CharUco Does:
1. Uses a CharUco pattern (checkerboard + ArUco markers) as a known reference
2. Measures CharUco pattern position in camera coordinates  
3. Measures robot gripper position in robot coordinates
4. Calculates transformation matrix: Camera coordinates to Robot coordinates
5. This is called "eye-in-hand" calibration (camera mounted on robot end-effector)

L515 LiDAR Camera + CharUco:
- L515 is a LiDAR sensor (provides direct 3D distance measurements)
- L515 provides camera intrinsics (for coordinate system definition)
- CharUco uses L515's intrinsics during coordinate transformation calibration
- For coordinate consistency: Same L515 intrinsics must be used in production

Why Intrinsics Matter for L515:
- L515 distance measurements are accurate regardless of intrinsics
- BUT: CharUco calibration was done with specific L515 intrinsics
- The coordinate transformation matrix expects those same intrinsics
- Using different intrinsics = coordinate system mismatch = position errors

Accuracy Claims:
- "~20mm accuracy" refers to coordinate transformation precision
- This is the accuracy of converting camera coordinates to robot coordinates
- NOT the accuracy of L515 distance measurements (which are inherently precise)
"""


class CharucoVirtualFixedCameraTracker:
    """
    CharUco-enhanced virtual fixed camera tracking application.
    
    This system uses pre-calibrated CharUco coordinate transformation to convert
    L515 LiDAR camera coordinates to robot base coordinates with high precision.
    
    Features:
    - Robot maintains observation pose for continuous detection
    - Manual pointing trigger with 'p' key  
    - CharUco camera-to-robot coordinate transformation matrix for high accuracy (~20mm)
    - Coordinate system corrections for proper robot alignment
    - L515 LiDAR sensor with coordinate transformation consistency
    - Position stability and accuracy monitoring
    - Runtime strategy switching (BEV, DBSCAN)
    - Enhanced safety with minimal robot movement
    """
    
    def __init__(self, config_name: str = "virtual_fixed_camera",
                 initial_strategy: str = "bev", 
                 bag_file: Optional[str] = None,
                 charuco_calibration_path: str = "charuco-pose-estimation/calibration_result_1754622347",
                 visualize_raw_pointcloud: bool = True,
                 enable_auto_roi: bool = True,
                 floor_offset: float = 0.01):
        """
        Initialize CharUco-enhanced virtual fixed camera tracker.
        
        Args:
            config_name: Configuration preset name (default: virtual_fixed_camera)
            initial_strategy: Initial clustering strategy ('bev' or 'dbscan') 
            bag_file: Optional bag file for replay
            charuco_calibration_path: Path to CharUco calibration results directory
            visualize_raw_pointcloud: Enable raw point cloud visualization
            enable_auto_roi: Enable automatic ROI detection based on table surface
            floor_offset: Height offset above detected table surface in meters
        """
        self.config_name = config_name
        self.bag_file = bag_file
        self.charuco_calibration_path = Path(charuco_calibration_path)
        self.visualize_raw_pointcloud = visualize_raw_pointcloud
        self.enable_auto_roi = enable_auto_roi
        self.floor_offset = floor_offset
        
        # Load configuration
        self.config = self._load_configuration(config_name)
        
        # Initialize components
        self.sensor_manager = None
        self.robot_controller = None
        self.detector = None
        self.visualizer = VisualizationEngine()
        self.temporal_filter = TemporalFilter(window_size=5)
        
        # CharUco-specific initialization
        self.eye_in_hand_transform = None
        self.charuco_camera_matrix = None
        self.charuco_dist_coeffs = None
        self.calibration_quality = "Unknown"
        self.camera_to_robot_transformer = None
        
        # Strategy management
        self.available_strategies = {}
        self.current_strategy_name = initial_strategy
        
        # Application state
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.last_detection_objects = []  # Store for pointing
        
        # Robot state tracking
        self.robot_in_observation_pose = False
        self.pointing_in_progress = False
        self.pointing_start_time = 0
        
        # Auto ROI detection
        self.table_surface_height = None
        self.table_surface_normal = None
        self.pointing_duration = 3.0  # 3 seconds
        
        # Return-to-observer state tracking
        self.returning_to_observer = False
        self.return_start_time = 0
        self.stabilization_time = 3.0
        
        # Performance monitoring enhanced with CharUco metrics
        self.performance_stats = {
            'total_detections': 0,
            'pointing_commands': 0,
            'detection_times': [],
            'avg_fps': 0.0,
            'position_accuracy_mm': [],
            'position_stability_mm': []
        }
        
        # Position tracking for stability analysis (CharUco-style)
        self.recent_positions = []
        self.position_history_size = 10
        
        # Joint control parameters (for camera tilt functionality)
        self.joint_step = 30  # degrees per keypress for joints
        self.base_joint_angles = None  # Joint angles for look_at_table pose (base reference)
        self.last_movement_time = 0
        self.movement_cooldown = 0.1  # Minimum time between movements
        
        # UI controls
        self.show_help = False
        self.show_performance = True
        self.show_charuco_metrics = True
        
        # Periodic position reporting
        self.last_position_report_time = 0
        self.position_report_interval = 5.0  # 5 seconds
    
    def _load_configuration(self, config_name: str):
        """Load configuration from factory."""
        try:
            return ConfigFactory.create_preset(config_name)
        except:
            print(f"Unknown config '{config_name}', using default virtual_fixed_camera")
            return ConfigFactory.create_preset('virtual_fixed_camera')
    
    def _load_charuco_transformation(self) -> bool:
        """
        Load CharUco eye-in-hand coordinate transformation matrix and L515 camera intrinsics.
        
        The CharUco system calibrated the coordinate transformation from camera frame to robot
        frame using specific L515 camera intrinsics. Both the transformation matrix and the
        intrinsics used during calibration must be loaded for coordinate consistency.
        
        Returns:
            True if loading successful
        """
        print("Loading CharUco coordinate transformation calibration...")
        print("=" * 50)
        
        # Load transformation matrix
        transform_file = self.charuco_calibration_path / 'eye_in_hand_transform.npy'
        if not transform_file.exists():
            print(f"Error: CharUco transformation matrix not found at: {transform_file}")
            print("Please run CharUco calibration first using charuco_realtime_pose.py")
            return False
        
        try:
            self.eye_in_hand_transform = np.load(transform_file)
            print(f"[OK] Loaded CharUco transformation matrix from: {transform_file}")
            print("Transformation matrix:")
            for i, row in enumerate(self.eye_in_hand_transform):
                print(f"  [{row[0]:8.5f} {row[1]:8.5f} {row[2]:8.5f} {row[3]:8.5f}]")
        except Exception as e:
            print(f"Error loading transformation matrix: {e}")
            return False
        
        # Load L515 camera intrinsics used during coordinate transformation calibration
        intrinsics_file = self.charuco_calibration_path / 'camera_intrinsics.txt'
        if intrinsics_file.exists():
            try:
                self.charuco_camera_matrix = np.loadtxt(intrinsics_file)
                print(f"[OK] Loaded L515 camera intrinsics used during CharUco calibration: {intrinsics_file}")
                print("L515 camera matrix used during calibration:")
                print(f"  fx={self.charuco_camera_matrix[0,0]:.2f}, fy={self.charuco_camera_matrix[1,1]:.2f}")
                print(f"  cx={self.charuco_camera_matrix[0,2]:.2f}, cy={self.charuco_camera_matrix[1,2]:.2f}")
                
                # Use minimal distortion coefficients as per CharUco calibration
                self.charuco_dist_coeffs = np.zeros(5)
                
            except Exception as e:
                print(f"Warning: Could not load L515 intrinsics from CharUco calibration: {e}")
                print("Will use RealSense SDK intrinsics (coordinate consistency may be affected)")
        else:
            print(f"Warning: No L515 intrinsics file found at: {intrinsics_file}")
            print("Will use RealSense SDK intrinsics (coordinate consistency may be affected)")
        
        # Load calibration quality information
        calibration_json = self.charuco_calibration_path / 'calibration_data.json'
        if calibration_json.exists():
            try:
                with open(calibration_json, 'r') as f:
                    calib_data = json.load(f)
                num_points = calib_data.get('num_points', 'Unknown')
                print(f"[OK] CharUco calibration used {num_points} calibration points")
            except Exception as e:
                print(f"Could not load calibration metadata: {e}")
        
        # Load validation results if available
        validation_file = self.charuco_calibration_path / 'validation_report.json'
        if validation_file.exists():
            try:
                with open(validation_file, 'r') as f:
                    validation_data = json.load(f)
                self.calibration_quality = validation_data.get('quality_score', 'Unknown')
                mean_error = validation_data.get('mean_translation_error', 0)
                print(f"[OK] Calibration quality: {self.calibration_quality} ({mean_error:.1f}mm mean error)")
            except Exception as e:
                print(f"Could not load validation results: {e}")
        
        # Initialize camera-to-robot transformer with loaded calibration
        try:
            self.camera_to_robot_transformer = CameraToRobotTransformer(self.charuco_calibration_path)
            if self.camera_to_robot_transformer.is_ready():
                print(f"[OK] Camera-to-robot transformer initialized successfully")
            else:
                print(f"[ERROR] Camera-to-robot transformer failed to initialize")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to create camera-to-robot transformer: {e}")
            return False
        
        print("=" * 50)
        print("CharUco coordinate transformation calibration loaded successfully!")
        print("This system provides ~20mm position accuracy with proper camera-to-robot coordinate transformation.")
        print("=" * 50)
        
        return True
    
    def _predict_object_position_charuco(self, camera_position: np.ndarray, robot_pose: List[float]) -> Optional[np.ndarray]:
        """
        Predict object position in robot base coordinates using CharUco transformation.
        
        Args:
            camera_position: Object position in camera coordinates [x, y, z] in meters
            robot_pose: Current robot pose [x, y, z, rx, ry, rz]
            
        Returns:
            Object position in robot base coordinates [x, y, z] in mm, or None if prediction fails
        """
        if self.camera_to_robot_transformer is None or robot_pose is None:
            return None
        
        try:
            # Use the modular transformer (expects meters input, returns mm output)
            position_mm = self.camera_to_robot_transformer.transform_to_robot(
                camera_position, robot_pose, input_units='meters', output_units='mm'
            )
            return position_mm
            
        except Exception as e:
            print(f"Error predicting object position with CharUco transformer: {e}")
            return None
    
    def _update_position_history(self, position: np.ndarray):
        """Update position history for stability analysis (CharUco-style)."""
        self.recent_positions.append(position.copy())
        if len(self.recent_positions) > self.position_history_size:
            self.recent_positions.pop(0)
        
        # Update performance stats
        if len(self.recent_positions) >= 2:
            positions = np.array(self.recent_positions)
            std_pos = np.std(positions, axis=0)
            stability_metric = np.linalg.norm(std_pos)  # Overall stability
            self.performance_stats['position_stability_mm'].append(stability_metric)
            
            # Keep only recent samples
            if len(self.performance_stats['position_stability_mm']) > 100:
                self.performance_stats['position_stability_mm'].pop(0)
    
    def _get_position_stability_metrics(self):
        """Get CharUco-style position stability metrics."""
        if len(self.recent_positions) < 2:
            return None, None
        
        positions = np.array(self.recent_positions)
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        return mean_pos, std_pos
    
    def _report_positions_periodically(self, detected_objects: List[Dict[str, Any]]):
        """Print detected and transformed positions every 5 seconds."""
        current_time = time.time()
        
        # Check if it's time for a position report
        if current_time - self.last_position_report_time < self.position_report_interval:
            return
        
        # Only report if we have objects and robot connection
        if not detected_objects or not self.robot_controller.is_connected:
            return
        
        try:
            # Get current robot pose
            robot_pose = self.robot_controller.get_current_position()
            if robot_pose is None:
                return
            
            print(f"\n{'='*60}")
            print(f"POSITION REPORT - {time.strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            
            # Report up to 3 closest objects
            for i, obj in enumerate(detected_objects[:3]):
                camera_position = self._extract_object_position(obj)
                if camera_position:
                    # Convert to mm for display
                    cam_mm = [pos * 1000 for pos in camera_position]
                    
                    # Get transformed position
                    robot_position = self._predict_object_position_charuco(
                        np.array(camera_position), robot_pose
                    )
                    
                    print(f"Object {i+1}:")
                    print(f"  Camera:  [{cam_mm[0]:6.1f}, {cam_mm[1]:6.1f}, {cam_mm[2]:6.1f}] mm")
                    
                    if robot_position is not None:
                        print(f"  Robot:   [{robot_position[0]:6.1f}, {robot_position[1]:6.1f}, {robot_position[2]:6.1f}] mm")
                    else:
                        print(f"  Robot:   [transformation failed]")
                    print()
            
            # Show position stability if available
            mean_pos, std_pos = self._get_position_stability_metrics()
            if mean_pos is not None and std_pos is not None:
                stability = np.linalg.norm(std_pos)
                print(f"Position Stability: {stability:.1f}mm")
            
            print(f"Calibration Quality: {self.calibration_quality}")
            print(f"{'='*60}")
            
            self.last_position_report_time = current_time
            
        except Exception as e:
            print(f"Error in position reporting: {e}")
    
    def initialize(self) -> bool:
        """
        Initialize all system components with CharUco enhancement.
        
        Returns:
            True if initialization successful
        """
        print("Initializing CharUco-Enhanced Virtual Fixed Camera Tracking System")
        print("=" * 60)
        
        # Load CharUco calibration first
        if not self._load_charuco_transformation():
            print("Failed to load CharUco calibration data")
            return False
        
        # Initialize sensor manager with L515 intrinsics from CharUco calibration if available
        print("Initializing L515 LiDAR camera sensor...")
        self.sensor_manager = RealSenseManager(
            self.bag_file, 
            realtime_mode=True, 
            use_imu=True,
            external_intrinsics=self.charuco_camera_matrix
        )
        if not self.sensor_manager.initialize():
            print("Error: Failed to initialize L515 LiDAR camera sensor")
            return False
        
        if not self.sensor_manager.start():
            print("Error: Failed to start L515 LiDAR camera sensor")
            return False
        
        # Initialize robot controller first
        print("Initializing robot controller...")
        self.robot_controller = MyCobotController()
        
        # Set CharUco transformation matrix instead of loading from file
        if self.eye_in_hand_transform is not None:
            self.robot_controller.transformation_matrix = self.eye_in_hand_transform
            print(f"[OK] CharUco transformation matrix applied to robot controller")
            print(f"  Matrix shape: {self.robot_controller.transformation_matrix.shape}")
        else:
            print("Warning: No CharUco transformation matrix available. Robot commands will be disabled.")
        
        # Connect to robot
        if not self.robot_controller.connect():
            print("Warning: Failed to connect to robot. Running in visualization-only mode.")
        
        # Configure robot safety parameters and move to observation pose
        if self.robot_controller.is_connected:
            robot_params = self.config.get('robot_params', {})
            self.robot_controller.safety_constraints.update(robot_params)
            print(f"Robot safety parameters: {robot_params}")
            
            # Enable async mode for non-blocking robot commands
            if self.robot_controller.enable_async_mode(max_queue_size=5):
                print("[OK] Robot async mode enabled for smooth visualization")
            else:
                print("[WARNING] Failed to enable robot async mode - commands may block visualization")
            
            # Setup robot with safe home-first sequence
            from module.utils.robot_utils import setup_robot_with_stabilization
            if not setup_robot_with_stabilization(
                self.robot_controller,
                self.sensor_manager,
                pose_type="table",
                stabilization_time=3.0,
                window_name="CharUco Virtual Fixed Camera Tracking",
                home_first=True
            ):
                print("Warning: Robot setup failed")
            
            # Capture base joint angles for camera tilt control reference
            try:
                print("Capturing base joint angles for camera tilt control...")
                time.sleep(1)  # Wait for robot to fully stabilize
                self.base_joint_angles = self.robot_controller.get_joint_angles()
                if self.base_joint_angles and len(self.base_joint_angles) >= 6:
                    print(f"Base joint angles captured: {[f'{x:.1f}' for x in self.base_joint_angles]}")
                    print("[OK] Camera tilt controls (w/s keys) enabled")
                else:
                    print("Warning: Could not capture base joint angles - camera tilt controls disabled")
                    self.base_joint_angles = None
            except Exception as e:
                print(f"Warning: Failed to capture base joint angles: {e}")
                print("Camera tilt controls (w/s keys) will be disabled")
                self.base_joint_angles = None
        
        # L515 intrinsics integration for coordinate transformation consistency
        if self.charuco_camera_matrix is not None:
            print("[OK] L515 intrinsics from CharUco calibration integrated with RealSense manager")
            print("[OK] Point cloud generation will use same L515 intrinsics as coordinate transformation calibration")
            print("[OK] Full coordinate transformation consistency: Matching intrinsics + CharUco transformation")
        
        # Calibrate gravity alignment AFTER robot is in stable observation pose
        if self.sensor_manager.use_imu:
            print("Calibrating gravity alignment (robot is now stable)...")
            self.sensor_manager.calibrate_gravity_alignment()
        
        # Initialize detection strategies BEFORE auto ROI
        print("Initializing detection strategies...")
        self._initialize_strategies()
        
        # Perform auto ROI detection if enabled
        if self.enable_auto_roi:
            print("Performing auto ROI detection...")
            if self._detect_and_set_dynamic_roi():
                print("[OK] Auto ROI detection successful")
            else:
                print("! Auto ROI detection failed - using default parameters")
        
        # Set initial strategy
        if not self._switch_strategy(self.current_strategy_name):
            print(f"Error: Failed to initialize strategy '{self.current_strategy_name}'")
            return False
        
        print("[OK] CharUco coordinate transformation system initialized successfully")
        print(f"[OK] Using CharUco camera-to-robot coordinate transformation with {self.calibration_quality} quality")
        print(f"[OK] Expected position accuracy: ~20mm")
        return True
    
    def _detect_table_surface_ransac(self, vertices: np.ndarray, max_distance: float = 0.015) -> Optional[Dict[str, Any]]:
        """
        Detect table surface using RANSAC plane fitting.
        
        Args:
            vertices: Point cloud vertices [N, 3]
            max_distance: Maximum distance from plane to be considered inlier (meters)
            
        Returns:
            Dictionary with table surface info or None
        """
        if vertices is None or len(vertices) < 100:
            return None
        
        try:
            # Filter out invalid points
            valid_vertices = vertices[np.isfinite(vertices).all(axis=1)]
            if len(valid_vertices) < 100:
                return None
            
            # Filter points by distance from camera: focus on table surface range
            z_range = (0, 0.4)  # 0-400mm to focus on table area
            x_range = (-0.1, 0.1)  # X-axis range for table surface
            
            filtered_vertices = valid_vertices[
                (valid_vertices[:, 2] >= z_range[0]) & (valid_vertices[:, 2] <= z_range[1]) &
                (valid_vertices[:, 0] >= x_range[0]) & (valid_vertices[:, 0] <= x_range[1])
            ]
            
            if len(filtered_vertices) < 100:
                return None
            
            # RANSAC regression: fit plane to point cloud
            X = filtered_vertices[:, [0, 2]]  # [side, forward]
            y = filtered_vertices[:, 1]       # height
            
            # Use RANSAC to find the dominant plane (table surface)
            ransac = RANSACRegressor(
                max_trials=1000,
                min_samples=50,
                residual_threshold=max_distance,
                random_state=42
            )
            
            # Fit plane to point cloud
            ransac.fit(X, y)
            
            # Get inliers (points on table surface)
            inlier_mask = ransac.inlier_mask_
            table_points = filtered_vertices[inlier_mask]
            
            if len(table_points) < 50:  # Not enough inliers
                return None
            
            # Calculate table surface statistics
            table_height_mean = np.mean(table_points[:, 1])
            table_height_std = np.std(table_points[:, 1])
            
            # Calculate table surface normal from RANSAC plane coefficients
            # RANSAC fits: y = coef[0] * x + coef[1] * z + intercept
            # Plane equation: coef[0] * x - 1 * y + coef[1] * z + intercept = 0
            # Normal vector: [coef[0], -1, coef[1]]
            plane_coeffs = ransac.estimator_.coef_
            intercept = ransac.estimator_.intercept_
            
            # Calculate normal vector (pointing up from table surface)
            normal_vector = np.array([plane_coeffs[0], -1.0, plane_coeffs[1]])
            normal_vector = normal_vector / np.linalg.norm(normal_vector)
            
            # Ensure normal points upward (negative Y direction in camera coordinates)
            if normal_vector[1] > 0:
                normal_vector = -normal_vector
            
            # Calculate table bounds
            table_bounds = {
                'x_min': np.min(table_points[:, 0]),
                'x_max': np.max(table_points[:, 0]),
                'z_min': np.min(table_points[:, 2]),
                'z_max': np.max(table_points[:, 2])
            }
            
            print(f"+ Table surface detected:")
            print(f"  Height: {table_height_mean:.3f}m ± {table_height_std:.3f}m")
            print(f"  Normal: [{normal_vector[0]:.3f}, {normal_vector[1]:.3f}, {normal_vector[2]:.3f}]")
            print(f"  Inliers: {len(table_points)}/{len(filtered_vertices)} points (after distance filtering)")
            print(f"  Bounds: X[{table_bounds['x_min']:.2f}, {table_bounds['x_max']:.2f}], Z[{table_bounds['z_min']:.2f}, {table_bounds['z_max']:.2f}]")
            
            return {
                'table_height': table_height_mean,
                'table_height_std': table_height_std,
                'surface_normal': normal_vector,
                'inlier_count': len(table_points),
                'total_filtered_points': len(filtered_vertices),
                'bounds': table_bounds
            }
            
        except Exception as e:
            print(f"Error detecting table surface: {e}")
            return None
    
    def _detect_and_set_dynamic_roi(self) -> bool:
        """
        Detect table surface and set dynamic ROI parameters automatically.
        
        Returns:
            True if table detection and ROI setting successful
        """
        print("[AUTO-ROI] Detecting table surface...")
        
        try:
            # Allow camera to stabilize first (especially important on first run)
            print("[AUTO-ROI] Allowing camera to stabilize...")
            stabilization_frames = 10  # Discard first few frames to let camera settle
            for _ in range(stabilization_frames):
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                time.sleep(0.1)
            
            # Collect multiple frames for robust table detection
            table_detection_frames = 10  # Increased from 5 to 10 for better reliability
            all_vertices = []
            
            print(f"[AUTO-ROI] Collecting {table_detection_frames} frames for table detection...")
            for frame_idx in range(table_detection_frames):
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                if vertices is not None and len(vertices) > 100:
                    all_vertices.append(vertices)
                    print(f"[AUTO-ROI] Frame {frame_idx + 1}/{table_detection_frames}: {len(vertices)} points")
                else:
                    print(f"[AUTO-ROI] Frame {frame_idx + 1}/{table_detection_frames}: insufficient data")
                time.sleep(0.2)  # Increased delay between frames for better stability
            
            if not all_vertices:
                print("[AUTO-ROI] Warning: No valid frames for table detection")
                return False
            
            # Combine all frames for more robust table detection
            combined_vertices = np.vstack(all_vertices)
            print(f"[AUTO-ROI] Using {len(combined_vertices)} points from {len(all_vertices)} frames")
            
            # Detect table surface using RANSAC method
            table_info = self._detect_table_surface_ransac(combined_vertices, max_distance=0.015)
            
            if table_info is None:
                print("[AUTO-ROI] Warning: Failed to detect table surface")
                return False
            
            # Extract table height and surface normal
            detected_table_height = table_info['table_height']
            table_height_std = table_info['table_height_std']
            self.table_surface_normal = table_info['surface_normal']
            
            # Validate that this looks like a table surface (not floor)
            expected_table_range = (0.15, 0.5)  # Based on manual presets from calibration
            if not (expected_table_range[0] <= detected_table_height <= expected_table_range[1]):
                print(f"[AUTO-ROI] Detected surface at {detected_table_height:.3f}m outside expected table range {expected_table_range}")
                return False
            
            print(f"[AUTO-ROI] Detected table surface: {detected_table_height:.3f}m ± {table_height_std:.3f}m")
            
            # Set dynamic ROI based on detected table surface
            dynamic_roi = {
                'x_min': -1.0, 'x_max': 1.0,  # Keep wide X range
                'y_min': detected_table_height - self.floor_offset - 0.06,  # Below table surface
                'y_max': detected_table_height - self.floor_offset,  # At table surface
                'z_min': 0, 'z_max': 0.5    # Keep depth range
            }
            
            # Update all strategies' ROI parameters
            for strategy_name, strategy in self.available_strategies.items():
                if hasattr(strategy, 'roi_params'):
                    strategy.roi_params = dynamic_roi
                    print(f"[AUTO-ROI] Updated {strategy_name} strategy ROI parameters")
            
            self.table_surface_height = detected_table_height
            print(f"[AUTO-ROI] Set dynamic ROI: Y[{dynamic_roi['y_min']:.3f}, {dynamic_roi['y_max']:.3f}]m")
            print(f"[AUTO-ROI] Object detection at table surface level")
            return True
                
        except Exception as e:
            print(f"[AUTO-ROI] Error during table detection: {e}")
            return False
    
    def _recalibrate_auto_roi(self):
        """Recalibrate auto ROI during tracking."""
        try:
            # Recalibrate gravity alignment if available
            if self.sensor_manager.use_imu:
                print("Recalibrating gravity alignment...")
                self.sensor_manager.calibrate_gravity_alignment()
                print("+ Gravity alignment updated")
            
            # Re-detect table surface and update ROI
            if self.enable_auto_roi:
                print("Re-detecting table surface and updating ROI...")
                if self._detect_and_set_dynamic_roi():
                    print("+ Auto ROI updated successfully")
                else:
                    print("! Auto ROI re-detection failed")
                    
        except Exception as e:
            print(f"Error during auto ROI recalibration: {e}")
    
    def _initialize_strategies(self):
        """Initialize available detection strategies."""
        
        # BEV Clustering Strategy
        try:
            bev_strategy = BEVClusteringStrategy(
                roi_params=self.config.get('roi_params'),
                display_roi_params=self.config.get('display_roi_params'),
                clustering_params=self.config.get('clustering_params'),
                enable_temporal_filtering=True
            )
            self.available_strategies['bev'] = bev_strategy
            print("[OK] BEV clustering strategy initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize BEV strategy: {e}")
        
        # DBSCAN Clustering Strategy
        try:
            dbscan_strategy = DBSCANClusteringStrategy(
                eps=0.02,
                min_points=10
            )
            self.available_strategies['dbscan'] = dbscan_strategy
            print("[OK] DBSCAN clustering strategy initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize DBSCAN strategy: {e}")
    
    def _switch_strategy(self, strategy_name: str) -> bool:
        """Switch to a different clustering strategy."""
        if strategy_name not in self.available_strategies:
            print(f"Error: Strategy '{strategy_name}' not available")
            return False
        
        try:
            strategy = self.available_strategies[strategy_name]
            self.detector = ObjectDetector(
                clustering_strategy=strategy,
                classification_strategy=None  # No classification for real-time performance
            )
            self.current_strategy_name = strategy_name
            print(f"[OK] Switched to {strategy.get_strategy_name()}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to switch to strategy '{strategy_name}': {e}")
            return False
    
    def _extract_object_position(self, obj: Dict[str, Any]) -> Optional[List[float]]:
        """Extract 3D position from object in camera coordinates (returns in meters)."""
        try:
            if 'center_mm' in obj:
                # center_mm format: [x, y, z] = [side, height, forward] in mm (camera coordinates)
                center_mm = obj['center_mm']
                # Convert mm to meters for consistent transformer input
                return [center_mm[0]/1000.0, center_mm[1]/1000.0, center_mm[2]/1000.0]
            elif 'center' in obj:
                # center format: [x, y, z] = [side, height, forward] in meters (camera coordinates)
                center = obj['center']
                # Already in meters - return as-is
                return [center[0], center[1], center[2]]
            else:
                return None
        except Exception as e:
            print(f"Error extracting object position: {e}")
            return None
    
    def _find_closest_object(self, detected_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the closest detected object."""
        if not detected_objects:
            return None
        
        closest_object = None
        min_distance = float('inf')
        
        for obj in detected_objects:
            position = self._extract_object_position(obj)
            if position is None:
                continue
            
            # Calculate distance from camera origin (position is in meters)
            distance = np.sqrt(sum(p**2 for p in position))
            
            if distance < min_distance:
                min_distance = distance
                closest_object = obj
        
        return closest_object
    
    def _point_and_return(self):
        """Start pointing at closest detected object using CharUco-enhanced prediction."""
        if not self.robot_controller.is_connected:
            print("Robot not connected - cannot point at object")
            return
        
        if self.pointing_in_progress:
            print("Pointing action already in progress")
            return
        
        if not self.last_detection_objects:
            print("No objects detected - cannot point")
            return
        
        # Find closest object
        closest_object = self._find_closest_object(self.last_detection_objects)
        if closest_object is None:
            print("No valid target object found")
            return
        
        # Extract position in camera coordinates (now returns meters)
        camera_position = self._extract_object_position(closest_object)
        if camera_position is None:
            print("Could not extract object position")
            return
        
        # Get current robot pose for CharUco prediction
        try:
            robot_pose = self.robot_controller.get_current_position()
            if robot_pose is None:
                print("Could not get robot position for CharUco prediction")
                return
        except Exception as e:
            print(f"Error getting robot pose: {e}")
            return
        
        # Use CharUco-enhanced position prediction
        predicted_position = self._predict_object_position_charuco(
            np.array(camera_position), robot_pose
        )
        
        if predicted_position is None:
            print("CharUco position prediction failed, using direct camera coordinates")
            predicted_position = camera_position
        else:
            print(f"CharUco predicted position: [{predicted_position[0]:.1f}, {predicted_position[1]:.1f}, {predicted_position[2]:.1f}]mm")
            # Update position history for stability tracking
            self._update_position_history(predicted_position)
        
        print(f"\n{'='*60}")
        print(f"CHARUCO-ENHANCED POINTING ACTION - {time.strftime('%H:%M:%S')}")
        print(f"Camera position: ({camera_position[0]*1000:.1f}, {camera_position[1]*1000:.1f}, {camera_position[2]*1000:.1f}) mm")
        print(f"Predicted position: ({predicted_position[0]:.1f}, {predicted_position[1]:.1f}, {predicted_position[2]:.1f}) mm")
        print(f"Calibration quality: {self.calibration_quality}")
        print(f"{'='*60}")
        
        # Start non-blocking pointing sequence
        self.pointing_in_progress = True
        self.pointing_start_time = time.time()
        
        try:
            if self.robot_controller.async_mode_enabled:
                # Async pointing command using predicted position
                print("1. Pointing at CharUco-predicted position...")
                command_id = self.robot_controller.point_at_object_async(
                    predicted_position.tolist(),
                    priority=2,
                    timeout=15.0
                )
                
                if command_id:
                    print(f"[OK] CharUco-enhanced pointing command submitted (ID: {command_id})")
                    print("   RGB feed continues during pointing...")
                else:
                    print("[ERROR] Failed to submit pointing command")
                    self.pointing_in_progress = False
            else:
                # Synchronous pointing command
                print("1. Pointing at CharUco-predicted position...")
                success = self.robot_controller.point_at_object(predicted_position.tolist())
                
                if success:
                    print("[OK] Robot pointing at CharUco-predicted position")
                    print("   RGB feed continues during pointing...")
                else:
                    print("[ERROR] Failed to point at object")
                    self.pointing_in_progress = False
        
        except Exception as e:
            print(f"Error during CharUco-enhanced pointing sequence: {e}")
            self.pointing_in_progress = False
    
    def _check_pointing_timeout(self):
        """Check if pointing duration has elapsed and start return to observation pose."""
        if self.pointing_in_progress:
            elapsed_time = time.time() - self.pointing_start_time
            if elapsed_time >= self.pointing_duration:
                print(f"2. Pointing duration ({self.pointing_duration}s) elapsed, returning to observation pose...")
                from module.utils.robot_utils import move_to_standard_pose
                move_to_standard_pose(self.robot_controller, "table", async_mode=True)
                
                # Start return-to-observer sequence
                self.pointing_in_progress = False
                self.returning_to_observer = True
                self.return_start_time = time.time()
                self.robot_in_observation_pose = False  # Robot is moving, not yet stable
                print("[OK] Return to observer pose initiated, waiting for stabilization...")
    
    def _check_return_stabilization(self):
        """Check if robot has stabilized after returning to observer pose and recalibrate gravity."""
        if self.returning_to_observer:
            elapsed_time = time.time() - self.return_start_time
            if elapsed_time >= self.stabilization_time + self.pointing_duration/2:
                print(f"3. Robot stabilized after {self.stabilization_time}s, recalibrating gravity alignment...")
                
                # Recalibrate gravity alignment after stabilization
                if self.sensor_manager.use_imu:
                    self.sensor_manager.calibrate_gravity_alignment()
                    print("[OK] Gravity recalibration completed")
                else:
                    print("[INFO] IMU not available, skipping gravity recalibration")
                
                # Mark robot as back in observation pose and reset state
                self.returning_to_observer = False
                self.robot_in_observation_pose = True
                self.performance_stats['pointing_commands'] += 1
                print("[OK] CharUco point and return sequence fully completed")
    
    def _apply_camera_distance_filtering(self, vertices: np.ndarray) -> np.ndarray:
        """
        Apply camera distance and spatial filtering like LiDAR system.
        
        Args:
            vertices: Point cloud vertices [N, 3] in meters
            
        Returns:
            Filtered vertices within distance and spatial range
        """
        if vertices is None or len(vertices) == 0:
            return vertices
        
        try:
            # Calculate distance from camera origin (0, 0, 0) to each point
            distances = np.linalg.norm(vertices, axis=1)
            
            # Apply distance filtering: 0.25m to 0.8m
            min_distance = 0.25  # 250mm minimum
            max_distance = 0.8   # 800mm maximum
            distance_mask = (distances >= min_distance) & (distances <= max_distance)
            
            # Apply X-axis spatial filtering: -0.1m to 0.1m (±100mm from center)
            x_min = -0.1  # -100mm left
            x_max = 0.1   # +100mm right
            x_mask = (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max)
            
            # Combine both filters
            combined_mask = distance_mask & x_mask
            filtered_vertices = vertices[combined_mask]
            
            # Filtering applied silently for performance
            
            return filtered_vertices
            
        except Exception as e:
            print(f"Error in camera distance filtering: {e}")
            return vertices

    def _detect_objects(self, vertices: np.ndarray, color_image: Optional[np.ndarray]) -> Tuple[List[Dict[str, Any]], Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect objects using current strategy and get visualization data."""
        detection_start = time.time()
        
        try:
            # Skip heavy 3D processing during pointing mode
            if self.pointing_in_progress:
                # During pointing: return empty detection results but keep timing
                return [], {'total_time': time.time() - detection_start, 'num_objects': 0}, None, None
            
            # Note: Camera distance filtering is now applied before this method is called
            
            # Normal operation: full 3D detection
            # Check if current strategy supports visualization
            if hasattr(self.detector.clustering_strategy, 'cluster_objects') and \
               'return_visualization' in self.detector.clustering_strategy.cluster_objects.__code__.co_varnames:
                # Get clusters with visualization data
                result = self.detector.clustering_strategy.cluster_objects(vertices, color_image, return_visualization=True)
                if len(result) == 3:
                    detected_objects, bev_image, point_cloud_view = result
                else:
                    detected_objects = result
                    bev_image = point_cloud_view = None
                
                # Create timing info manually
                timing_info = {
                    'total_time': time.time() - detection_start,
                    'num_objects': len(detected_objects) if detected_objects else 0
                }
            else:
                # Fallback to original method
                detected_objects, timing_info = self.detector.detect_objects(vertices, color_image)
                bev_image = point_cloud_view = None
            
            # Apply temporal filtering
            if detected_objects:
                detected_objects = self.temporal_filter.stabilize_object_positions(detected_objects)
            
            self.performance_stats['total_detections'] += 1
            return detected_objects, timing_info, bev_image, point_cloud_view
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            return [], {'total_time': time.time() - detection_start, 'num_objects': 0}, None, None
    
    def _create_raw_point_cloud_view(self, vertices: np.ndarray) -> Optional[np.ndarray]:
        """Create raw point cloud visualization (camera perspective like RGB view)."""
        if vertices is None or len(vertices) == 0:
            return None
        
        try:
            # Filter out invalid points
            valid_mask = ~np.any(np.isnan(vertices) | np.isinf(vertices), axis=1)
            valid_vertices = vertices[valid_mask]
            
            if len(valid_vertices) < 100:
                return None
            
            # Create a 400x400 visualization canvas
            canvas_size = 400
            canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            
            # Camera perspective view (X-Y plane, looking along Z axis like RGB camera)
            x_coords = valid_vertices[:, 0]  # side (horizontal in camera view)
            y_coords = valid_vertices[:, 1]  # height (vertical in camera view)
            z_coords = valid_vertices[:, 2]  # depth (for color coding)
            
            # Normalize X and Y coordinates to canvas size with margins
            margin = 0.1
            x_range = np.max(x_coords) - np.min(x_coords)
            y_range = np.max(y_coords) - np.min(y_coords)
            
            if x_range > 0 and y_range > 0:
                # Scale to fit canvas with margin (maintain aspect ratio)
                scale = min((canvas_size * (1 - 2 * margin)) / max(x_range, y_range), 500)
                
                # Convert 3D coordinates to 2D canvas coordinates
                x_pixels = ((x_coords - np.min(x_coords)) * scale + canvas_size * margin).astype(int)
                # Flip Y coordinates to match camera view (Y increases downward in image)
                y_pixels = (canvas_size - ((y_coords - np.min(y_coords)) * scale + canvas_size * margin)).astype(int)
                
                # Clip to canvas bounds
                x_pixels = np.clip(x_pixels, 0, canvas_size - 1)
                y_pixels = np.clip(y_pixels, 0, canvas_size - 1)
                
                # Color code by depth (Z coordinate) - closer points brighter
                z_normalized = (z_coords - np.min(z_coords)) / (np.max(z_coords) - np.min(z_coords) + 1e-6)
                
                # Draw points with depth-based coloring (like a depth map)
                for i in range(len(x_pixels)):
                    # Create depth-based color: close = bright white/yellow, far = dark blue
                    depth_val = 1.0 - z_normalized[i]  # Invert so close is bright
                    color_intensity = int(depth_val * 255)
                    color = (color_intensity, color_intensity, 255)  # Blue-white gradient
                    cv2.circle(canvas, (x_pixels[i], y_pixels[i]), 1, color, -1)
            
            # Add grid lines for reference
            grid_spacing = canvas_size // 8
            for i in range(0, canvas_size, grid_spacing):
                cv2.line(canvas, (i, 0), (i, canvas_size), (40, 40, 40), 1)
                cv2.line(canvas, (0, i), (canvas_size, i), (40, 40, 40), 1)
            
            # Add text overlay
            cv2.putText(canvas, "RAW POINT CLOUD", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(canvas, "Camera View (X-Y)", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(canvas, f"Points: {len(valid_vertices)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Add depth range info
            if len(z_coords) > 0:
                z_min, z_max = np.min(z_coords), np.max(z_coords)
                cv2.putText(canvas, f"Depth: {z_min:.2f}-{z_max:.2f}m", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            
            # Color legend for depth
            cv2.putText(canvas, "Depth:", (10, canvas_size - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(canvas, "Close", (10, canvas_size - 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(canvas, "Far", (10, canvas_size - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            return canvas
            
        except Exception as e:
            print(f"Error creating raw point cloud view: {e}")
            return None
    
    def _create_visualization(self, vertices: np.ndarray, color_image: Optional[np.ndarray],
                            detected_objects: List[Dict[str, Any]], timing_info: Dict[str, float],
                            metadata: Dict[str, Any], bev_image: Optional[np.ndarray] = None, 
                            point_cloud_view: Optional[np.ndarray] = None, raw_vertices: Optional[np.ndarray] = None):
        """Create enhanced 2x2 visualization layout with CharUco information."""
        try:
            # Get strategy info
            strategy_info = self.detector.get_strategy_info() if self.detector else {}
            
            panel_size = 400
            
            # Initialize 2x2 grid panels
            top_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)    # RGB
            top_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)   # Raw Point Cloud
            bottom_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8) # BEV
            bottom_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8) # Side View
            
            # Top-left: RGB image with CharUco-enhanced tracking info overlay
            if color_image is not None:
                rgb_panel = color_image.copy()
                # Add tracking info overlay
                self._add_charuco_tracking_info_overlay(rgb_panel, detected_objects, timing_info, strategy_info)
                # Add object detection overlay with CharUco prediction
                self._add_charuco_object_detection_overlay(rgb_panel, detected_objects)
                
                # Resize to fit panel
                aspect_ratio = rgb_panel.shape[1] / rgb_panel.shape[0]
                if aspect_ratio > 1:
                    # Wider image
                    new_width = panel_size
                    new_height = int(panel_size / aspect_ratio)
                    rgb_resized = cv2.resize(rgb_panel, (new_width, new_height))
                    y_offset = (panel_size - new_height) // 2
                    top_left[y_offset:y_offset+new_height, :] = rgb_resized
                else:
                    # Taller image
                    new_height = panel_size
                    new_width = int(panel_size * aspect_ratio)
                    rgb_resized = cv2.resize(rgb_panel, (new_width, new_height))
                    x_offset = (panel_size - new_width) // 2
                    top_left[:, x_offset:x_offset+new_width] = rgb_resized
            else:
                cv2.putText(top_left, "NO RGB IMAGE", (50, panel_size//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            
            # Top-right: Raw point cloud (camera perspective)
            if self.visualize_raw_pointcloud and raw_vertices is not None:
                raw_pc_view = self._create_raw_point_cloud_view(raw_vertices)
                if raw_pc_view is not None:
                    top_right = raw_pc_view
                else:
                    cv2.putText(top_right, "RAW POINTS", (50, panel_size//2-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                    cv2.putText(top_right, "NO DATA", (80, panel_size//2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            else:
                cv2.putText(top_right, "RAW POINT CLOUD", (30, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
                cv2.putText(top_right, "DISABLED", (80, panel_size//2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            
            # Bottom-left: BEV image
            if bev_image is not None:
                bottom_left = cv2.resize(bev_image, (panel_size, panel_size))
                cv2.putText(bottom_left, "Bird's Eye View", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(bottom_left, "NO BEV VIEW", (50, panel_size//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            
            # Bottom-right: Side view point cloud
            if point_cloud_view is not None:
                bottom_right = cv2.resize(point_cloud_view, (panel_size, panel_size))
                cv2.putText(bottom_right, "Side View", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(bottom_right, "NO SIDE VIEW", (30, panel_size//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            
            # Combine into 2x2 grid
            top_row = np.hstack([top_left, top_right])
            bottom_row = np.hstack([bottom_left, bottom_right])
            combined = np.vstack([top_row, bottom_row])
            
            # Display the 2x2 visualization
            cv2.imshow('CharUco Virtual Fixed Camera Tracking', combined)
            
        except Exception as e:
            print(f"Warning: Visualization error: {e}")
            # Fallback to simple color image display
            if color_image is not None:
                cv2.imshow('CharUco Virtual Fixed Camera Tracking', color_image)
    
    def _add_charuco_tracking_info_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]],
                                         timing_info: Dict[str, float], strategy_info: Dict[str, str]):
        """Add CharUco-enhanced tracking information overlay to image."""
        
        # Get robot status info
        robot_status = "Disconnected"
        if self.robot_controller.is_connected:
            if self.pointing_in_progress:
                robot_status = "POINTING"
            elif self.returning_to_observer:
                robot_status = "RETURNING"
            elif self.robot_in_observation_pose:
                robot_status = "OBSERVING"
            else:
                robot_status = "Moving"
        
        # Add camera tilt status info
        camera_tilt_status = "Disabled"
        if self.base_joint_angles and self.robot_controller.is_connected:
            try:
                current_angles = self.robot_controller.get_joint_angles()
                if current_angles and len(current_angles) >= 4:
                    joint4_angle = current_angles[3]  # Joint4 for camera tilt
                    camera_tilt_status = f"J4: {joint4_angle:.0f}°"
                else:
                    camera_tilt_status = "Enabled"
            except:
                camera_tilt_status = "Enabled"

        info_lines = [
            "CHARUCO VIRTUAL FIXED CAMERA TRACKING",
            f"Strategy: {self.current_strategy_name.upper()} ({strategy_info.get('clustering_strategy', 'Unknown')})",
            f"Objects: {len(detected_objects)}",
            f"FPS: {self.temporal_filter.calculate_fps():.1f}",
            f"Robot: {robot_status}",
            f"CharUco Quality: {self.calibration_quality}",
            f"Camera Tilt: {camera_tilt_status}",
            f"Camera Range: 0.25-0.8m, X: ±0.1m",
            f"Mode: {'POINTING' if self.pointing_in_progress else 'RETURNING' if self.returning_to_observer else 'OBSERVING'}",
            "Press 'p' to point (w/s to tilt camera)"
        ]
        
        for i, line in enumerate(info_lines):
            if i == 0:
                color = (0, 255, 255)  # Yellow for CharUco title
            elif "CharUco Quality:" in line:
                if self.calibration_quality == "Excellent":
                    color = (0, 255, 0)  # Green for excellent
                elif self.calibration_quality == "Good":
                    color = (0, 255, 255)  # Yellow for good
                else:
                    color = (0, 165, 255)  # Orange for fair/poor
            elif "POINTING" in line or "RETURNING" in line:
                color = (0, 255, 255)  # Yellow for pointing/returning status
            else:
                color = (255, 255, 255)  # White for other info
                
            cv2.putText(image, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add prominent mode indicators
        if self.pointing_in_progress:
            cv2.putText(image, "3D DETECTION PAUSED - CHARUCO POINTING MODE", 
                       (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif self.returning_to_observer:
            cv2.putText(image, "RETURNING TO OBSERVER POSE - STABILIZING", 
                       (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    def _add_charuco_object_detection_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]]):
        """Add CharUco-enhanced object detection information overlay."""
        if not detected_objects:
            return
        
        # Get current robot pose for position prediction
        robot_pose = None
        try:
            if self.robot_controller.is_connected:
                robot_pose = self.robot_controller.get_current_position()
        except:
            pass
        
        y_offset = 280
        for i, obj in enumerate(detected_objects[:3]):  # Show first 3 objects
            camera_position = self._extract_object_position(obj)
            if camera_position:
                # Show both camera and predicted positions (camera_position now in meters)
                info_text = f"Obj{i+1} Cam: ({camera_position[0]*1000:.0f}, {camera_position[1]*1000:.0f}, {camera_position[2]*1000:.0f})mm"
                cv2.putText(image, info_text, (10, y_offset + i * 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                
                # Add CharUco-predicted position if available
                if robot_pose is not None and self.camera_to_robot_transformer is not None:
                    predicted_pos = self._predict_object_position_charuco(
                        np.array(camera_position), robot_pose
                    )
                    if predicted_pos is not None:
                        pred_text = f"     Robot: ({predicted_pos[0]:.0f}, {predicted_pos[1]:.0f}, {predicted_pos[2]:.0f})mm"
                        cv2.putText(image, pred_text, (10, y_offset + i * 40 + 18),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    else:
                        pred_text = f"     Robot: Prediction failed"
                        cv2.putText(image, pred_text, (10, y_offset + i * 40 + 18),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                else:
                    pred_text = f"     Robot: Not available"
                    cv2.putText(image, pred_text, (10, y_offset + i * 40 + 18),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)
        
        # Add position stability info if available
        mean_pos, std_pos = self._get_position_stability_metrics()
        if mean_pos is not None and std_pos is not None:
            stability_text = f"Position Stability: {np.linalg.norm(std_pos):.1f}mm"
            cv2.putText(image, stability_text, (10, y_offset + 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Main CharUco-enhanced tracking loop."""
        if not self.initialize():
            print("Failed to initialize CharUco-enhanced system")
            return
        
        print("\nCharUco-Enhanced Virtual Fixed Camera Tracking Started!")
        print("=" * 60)
        print("Controls:")
        print("  'q' - Quit")
        print("  'h' - Toggle help")
        print("  'p' - POINT AT OBJECT (CharUco-enhanced, 3 sec, return to observation)")
        print("  'space' - Pause/Resume")
        print("  '1' - Switch to BEV strategy")
        print("  '2' - Switch to DBSCAN strategy")
        print("  'r' - Return to observation pose")
        print("  'f' - Look forward (horizontal view)")
        print("  's' - Stop robot movement (or tilt camera down when idle)")
        print("  'w' - Tilt camera UP (Joint4 +30°)")
        print("  's' - Tilt camera DOWN (Joint4 -30°)")
        print("  'i' - Toggle performance info")
        print("  'a' - Recalibrate auto ROI")
        print("  'm' - Toggle CharUco metrics display")
        print("  't' - Toggle position reporting interval (1s/5s/10s)")
        print("-" * 60)
        print("Enhanced with CharUco calibration for ~20mm position accuracy!")
        print(f"Calibration quality: {self.calibration_quality}")
        if self.base_joint_angles:
            print("Camera tilt controls (w/s) enabled - adjust view angle while tracking!")
        else:
            print("Camera tilt controls (w/s) disabled - robot connection issue")
        print("Robot stays in observation pose for continuous detection.")
        print("Press 'p' to point at detected objects with CharUco precision.")
        print("-" * 60)
        
        self.is_running = True
        
        try:
            while self.is_running:
                frame_start = time.time()
                
                # Handle pause
                if self.is_paused:
                    cv2.waitKey(30)
                    continue
                
                # Get frame from sensor
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                # Get raw vertices (before alignment) if raw visualization is enabled
                raw_vertices = None
                if self.visualize_raw_pointcloud:
                    raw_vertices = self.sensor_manager.get_raw_vertices()
                
                # Check for end of stream
                if metadata.get('end_of_stream', False):
                    print("End of stream reached")
                    break
                
                if vertices is not None:
                    # Apply camera distance and spatial filtering BEFORE detection and visualization
                    vertices = self._apply_camera_distance_filtering(vertices)
                    
                    # Also apply filtering to raw vertices for consistent visualization
                    if raw_vertices is not None:
                        raw_vertices = self._apply_camera_distance_filtering(raw_vertices)
                    
                    # Always run detection pipeline, but skip heavy 3D processing during pointing
                    detected_objects, timing_info, bev_image, point_cloud_view = self._detect_objects(vertices, color_image)
                    
                    # Store detected objects for pointing (only when not pointing)
                    if not self.pointing_in_progress:
                        self.last_detection_objects = detected_objects
                    
                    # Periodic position reporting to console
                    self._report_positions_periodically(detected_objects)
                    
                    # Create visualization (RGB feed continues, 3D panels may be empty during pointing)
                    self._create_visualization(vertices, color_image, detected_objects, timing_info, metadata, bev_image, point_cloud_view, raw_vertices)
                    
                    self.frame_count += 1
                    
                    # Update temporal filter with frame timestamp for FPS calculation
                    current_time = time.time()
                    self.temporal_filter.add_frame_to_history(
                        objects=detected_objects,
                        timestamp=current_time
                    )
                    
                    # Update performance stats
                    frame_time = current_time - frame_start
                    self._update_performance_stats(frame_time, timing_info)
                
                # Check if pointing duration has elapsed
                self._check_pointing_timeout()
                
                # Check if robot has stabilized after returning to observer pose
                self._check_return_stabilization()
                
                # Handle user input
                if not self._handle_user_input():
                    break
                
        except KeyboardInterrupt:
            print("\nCharUco tracking interrupted by user")
        except Exception as e:
            print(f"Error in CharUco main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _rotate_joint(self, joint_index: int, angle_delta: float, message: str):
        """
        Rotate a specific joint by the given angle delta.
        Adapted from robot_camera_explorer.py for CharUco system.
        
        Args:
            joint_index: Index of joint to rotate (0-5)
            angle_delta: Angle change in degrees (positive/negative)
            message: Message to display for this movement
        """
        if not self.robot_controller.is_connected:
            print("Cannot move: Robot not connected")
            return
        
        if not self.base_joint_angles:
            print("Cannot move: Base joint angles unknown")
            return
        
        # Get current joint angles
        try:
            current_angles = self.robot_controller.get_joint_angles()
        except Exception as e:
            print(f"Error getting joint angles: {e}")
            return
        
        # Check if current_angles is valid, otherwise use base angles as fallback
        if not current_angles or not isinstance(current_angles, list) or len(current_angles) < 6:
            print(f"Cannot get valid joint angles (got: {current_angles}), using base angles as fallback")
            current_angles = self.base_joint_angles.copy()
        
        # Calculate new angle for the specified joint
        new_angles = current_angles.copy()
        new_angles[joint_index] += angle_delta
        
        print(f"{message}: Joint {joint_index+1} = {new_angles[joint_index]:.1f}°")
        
        # Execute joint movement using async system if available
        try:
            if self.robot_controller.async_mode_enabled:
                from module.robot.command_queue import CommandType
                command_id = self.robot_controller.submit_async_command(
                    CommandType.MOVE_TO_ANGLES, new_angles, speed=50, priority=2)
                if command_id:
                    print(f"Joint movement command queued (ID: {command_id})")
                else:
                    print("Failed to queue joint movement command")
            else:
                # Synchronous movement as fallback
                self.robot_controller.move_to_angles(new_angles, speed=50)
                print("Joint movement executed synchronously")
        except Exception as e:
            print(f"Error executing joint movement: {e}")
        
        self.last_movement_time = time.time()

    def _handle_user_input(self) -> bool:
        """Handle user keyboard input."""
        key = cv2.waitKey(30) & 0xFF
        
        current_time = time.time()
        
        if key == ord('q'):
            return False
        elif key == ord('h'):
            self.show_help = not self.show_help
            print(f"Help {'enabled' if self.show_help else 'disabled'}")
        elif key == ord('p'):
            # CharUco-enhanced pointing action
            print("CharUco-enhanced pointing action triggered!")
            self._point_and_return()
        elif key == ord('i'):
            # Performance info toggle
            self.show_performance = not self.show_performance
            print(f"Performance info {'enabled' if self.show_performance else 'disabled'}")
        elif key == ord('m'):
            # CharUco metrics toggle
            self.show_charuco_metrics = not self.show_charuco_metrics
            print(f"CharUco metrics {'enabled' if self.show_charuco_metrics else 'disabled'}")
        elif key == ord(' '):
            self.is_paused = not self.is_paused
            print(f"{'Paused' if self.is_paused else 'Resumed'}")
        elif key == ord('1'):
            self._switch_strategy('bev')
        elif key == ord('2'):
            self._switch_strategy('dbscan')
        elif key == ord('r'):
            if self.robot_controller.is_connected:
                print("Returning to observation pose...")
                from module.utils.robot_utils import move_to_standard_pose
                move_to_standard_pose(self.robot_controller, "table", async_mode=True)
        elif key == ord('f'):
            if self.robot_controller.is_connected:
                print("Looking forward (horizontal view)...")
                if self.robot_controller.async_mode_enabled:
                    command_id = self.robot_controller.look_forward_async(priority=1)
                    if command_id:
                        print(f"Look forward command queued (ID: {command_id})")
                else:
                    self.robot_controller.look_forward()
                    print("Robot moved to forward view")
        elif key == ord('a'):
            # Auto ROI recalibration
            if self.enable_auto_roi:
                print("Recalibrating auto ROI...")
                self._recalibrate_auto_roi()
            else:
                print("Auto ROI is disabled")
        elif key == ord('t'):
            # Toggle position reporting interval
            if self.position_report_interval == 5.0:
                self.position_report_interval = 1.0
                print("Position reporting: Every 1 second (fast mode)")
            elif self.position_report_interval == 1.0:
                self.position_report_interval = 10.0
                print("Position reporting: Every 10 seconds (slow mode)")
            else:
                self.position_report_interval = 5.0
                print("Position reporting: Every 5 seconds (normal mode)")
        
        # Camera tilt controls (Joint4) - only if robot connected and not in cooldown
        elif key == ord('w') and self.robot_controller.is_connected and \
             current_time - self.last_movement_time > self.movement_cooldown:
            self._rotate_joint(3, self.joint_step, "Tilting camera up (joint4)")
        elif key == ord('s') and self.robot_controller.is_connected and \
             current_time - self.last_movement_time > self.movement_cooldown:
            # Check if this is the stop movement command or camera tilt down
            # If robot is moving (pointing, returning), 's' stops movement
            if self.pointing_in_progress or self.returning_to_observer:
                if self.robot_controller.async_mode_enabled:
                    command_id = self.robot_controller.stop_movement_async(priority=3)
                    print(f"Stop movement command submitted (ID: {command_id})")
                else:
                    print("Stopping robot movement...")
                    self.robot_controller.stop_movement()
                self.pointing_in_progress = False
                self.returning_to_observer = False
            else:
                # Camera tilt down when not moving
                self._rotate_joint(3, -self.joint_step, "Tilting camera down (joint4)")
        
        return True
    
    def _update_performance_stats(self, frame_time: float, timing_info: Dict[str, float]):
        """Update performance statistics with CharUco metrics."""
        detection_time = timing_info.get('total_time', 0)
        self.performance_stats['detection_times'].append(detection_time)
        
        # Calculate FPS using temporal filter for more accurate averaging
        self.performance_stats['avg_fps'] = self.temporal_filter.calculate_fps()
        
        # Keep only recent samples
        max_samples = 100
        if len(self.performance_stats['detection_times']) > max_samples:
            self.performance_stats['detection_times'].pop(0)
    
    def _cleanup(self):
        """Cleanup resources."""
        print("\nShutting down CharUco-Enhanced Virtual Fixed Camera Tracking System...")
        
        # Return to observation pose before shutdown
        if self.robot_controller and self.robot_controller.is_connected:
            print("Returning to observation pose for shutdown...")
            from module.utils.robot_utils import move_to_standard_pose
            move_to_standard_pose(self.robot_controller, "table", async_mode=False)
        
        # Stop sensor
        if self.sensor_manager:
            self.sensor_manager.stop()
        
        # Disconnect robot and cleanup async system
        if self.robot_controller:
            self.robot_controller.cleanup_async_system()
            self.robot_controller.disconnect()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Print final statistics
        self._print_final_statistics()
        
        print("[OK] CharUco-enhanced shutdown complete")
    
    def _print_final_statistics(self):
        """Print final performance statistics with CharUco metrics."""
        stats = self.performance_stats
        
        print("\n" + "=" * 60)
        print("FINAL CHARUCO-ENHANCED PERFORMANCE STATISTICS")
        print("=" * 60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Pointing commands: {stats['pointing_commands']}")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print(f"CharUco calibration quality: {self.calibration_quality}")
        
        if stats['detection_times']:
            avg_detection = np.mean(stats['detection_times']) * 1000
            print(f"Average detection time: {avg_detection:.1f}ms")
        
        if stats['position_stability_mm']:
            avg_stability = np.mean(stats['position_stability_mm'])
            print(f"Average position stability: {avg_stability:.1f}mm")
        
        print("=" * 60)
        print("CharUco Enhancement Summary:")
        print("- High-precision eye-in-hand calibration (~20mm accuracy)")
        print("- Coordinate system corrections (90° Z-rotation fix)")
        print("- Calibration-specific camera intrinsics")
        print("- Position stability monitoring")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CharUco-Enhanced Virtual Fixed Camera Tracking Application')
    parser.add_argument('--config', default='virtual_fixed_camera',
                       help='Configuration preset (default: virtual_fixed_camera)')
    parser.add_argument('--strategy', default='bev', choices=['bev', 'dbscan'],
                       help='Initial clustering strategy')
    parser.add_argument('--bag', help='RealSense bag file for replay')
    parser.add_argument('--charuco-calibration', default='charuco-pose-estimation/calibration_result_1754622347',
                       help='Path to CharUco calibration results directory')
    parser.add_argument('--visualize-raw-pointcloud', action='store_true', default=True,
                       help='Enable raw point cloud visualization (default: enabled)')
    parser.add_argument('--no-raw-pointcloud', action='store_true',
                       help='Disable raw point cloud visualization')
    parser.add_argument('--disable-auto-roi', action='store_true',
                       help='Disable automatic ROI detection based on table surface')
    parser.add_argument('--floor-offset', type=float, default=0.01,
                       help='Height offset above detected table surface in meters (default: 0.01m = 10mm)')
    
    args = parser.parse_args()
    
    # Handle raw pointcloud visualization logic
    if args.no_raw_pointcloud:
        args.visualize_raw_pointcloud = False
    
    print("CharUco-Enhanced Virtual Fixed Camera Tracking Application")
    print("=" * 55)
    print(f"Configuration: {args.config}")
    print(f"Initial strategy: {args.strategy}")
    print(f"CharUco calibration: {args.charuco_calibration}")
    if args.bag:
        print(f"Bag file: {args.bag}")
    print(f"Auto ROI detection: {'Disabled' if args.disable_auto_roi else 'Enabled'}")
    if not args.disable_auto_roi:
        print(f"Floor offset: {args.floor_offset}m")
    print("=" * 55)
    print("Enhancement: Using CharUco calibration for ~20mm position accuracy!")
    print("=" * 55)
    
    try:
        tracker = CharucoVirtualFixedCameraTracker(
            config_name=args.config,
            initial_strategy=args.strategy,
            bag_file=args.bag,
            charuco_calibration_path=args.charuco_calibration,
            visualize_raw_pointcloud=args.visualize_raw_pointcloud,
            enable_auto_roi=not args.disable_auto_roi,
            floor_offset=args.floor_offset
        )
        
        tracker.run()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()