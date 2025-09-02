#!/usr/bin/env python3
"""
Virtual Fixed Camera Object Tracking Application

This application uses an end-effector mounted camera in a "virtual fixed camera" mode.
The robot stays in a fixed observation pose most of the time for continuous detection,
and only moves to point at objects when manually triggered with the 'p' key.

Usage:
    python virtual_fixed_tracking.py [--config CONFIG_NAME] [--strategy STRATEGY] [--bag BAG_FILE]

Requirements:
    - Run modular_calibrate_camera.py with virtual_fixed_camera config first
    - Camera mounted on robot end-effector
    - Robot starts and returns to observation pose [87.5, -62.7, 367.2, -144.03, 29.89, -124.04]

New Features compared to modular_track_object.py:
    - Robot maintains observation pose for continuous detection
    - On-demand pointing with 'p' key (point for 3 seconds, return to observation)
    - Minimal robot movement for safer operation
    - Same detection strategies and visualization
"""

import argparse
import sys
import os
import time
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import RANSACRegressor

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


class VirtualFixedCameraTracker:
    """
    Virtual fixed camera tracking application with modular transformer integration.
    
    Features:
    - Robot maintains observation pose for continuous detection
    - Manual pointing trigger with 'p' key
    - Modular camera-to-robot transformer for precise coordinate conversion
    - Configurable detection strategies (BEV, DBSCAN)
    - Runtime strategy switching
    - Enhanced safety with minimal robot movement
    - Consistent unit handling (meters ↔ mm) throughout the system
    """
    
    def _resolve_transformation_matrix_file(self, transformation_matrix_file: str) -> str:
        """
        Resolve transformation matrix file path with automatic detection.
        
        Prioritizes depth-based calibration matrices over RGB-based ones for
        better compatibility with LiDAR detection systems.
        
        Args:
            transformation_matrix_file: File path or "auto" for automatic detection
            
        Returns:
            str: Resolved file path to transformation matrix
        """
        if transformation_matrix_file != "auto":
            return transformation_matrix_file
        
        import glob
        from pathlib import Path
        
        print("[AUTO-DETECTION] Searching for optimal transformation matrix...")
        
        # Priority 1: Look for depth-based calibration results
        depth_patterns = [
            "charuco-pose-estimation/depth_calibration_result_*/eye_in_hand_transform.npy",
            "charuco-pose-estimation/*/eye_in_hand_transform_depth.npy",
            "*depth*calibration*/eye_in_hand_transform.npy"
        ]
        
        for pattern in depth_patterns:
            files = glob.glob(pattern)
            if files:
                # Get the most recent depth-based calibration
                latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
                print(f"[AUTO-DETECTION] + Found DEPTH-BASED calibration: {latest_file}")
                print("[AUTO-DETECTION]   This is optimal for LiDAR object detection!")
                return latest_file
        
        # Priority 2: Look for recent RGB-based calibration results  
        rgb_patterns = [
            "charuco-pose-estimation/calibration_result_*/eye_in_hand_transform.npy",
            "charuco_eye_in_hand_transform.npy",
            "eye_in_hand_transform.npy"
        ]
        
        for pattern in rgb_patterns:
            files = glob.glob(pattern)
            if files:
                # Get the most recent RGB-based calibration
                latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
                print(f"[AUTO-DETECTION] ! Found RGB-based calibration: {latest_file}")
                print("[AUTO-DETECTION]   WARNING: This may have coordinate system mismatch with LiDAR!")
                print("[AUTO-DETECTION]   Run charuco_depth_calibration.py for better accuracy.")
                return latest_file
        
        # Fallback: Use the original default
        fallback = "charuco-pose-estimation/calibration_result_1754622347/eye_in_hand_transform.npy"
        print(f"[AUTO-DETECTION] - No calibration matrices found, using fallback: {fallback}")
        print("[AUTO-DETECTION]   Run charuco_depth_calibration.py to create proper calibration.")
        return fallback
    
    def __init__(self, config_name: str = "virtual_fixed_camera", 
                 initial_strategy: str = "bev",
                 bag_file: Optional[str] = None,
                 transformation_matrix_file: str = "auto",
                 visualize_raw_pointcloud: bool = True,
                 enable_auto_roi: bool = True,
                 floor_offset: float = 0.01):
        """
        Initialize virtual fixed camera tracker.
        
        Args:
            config_name: Configuration preset name (default: virtual_fixed_camera)
            initial_strategy: Initial clustering strategy ('bev' or 'dbscan')
            bag_file: Optional bag file for replay
            transformation_matrix_file: Path to transformation matrix or "auto" for automatic detection
            visualize_raw_pointcloud: Enable raw point cloud visualization
            enable_auto_roi: Enable automatic ROI detection based on table surface
            floor_offset: Height offset above detected table surface in meters
        """
        self.config_name = config_name
        self.bag_file = bag_file
        self.transformation_matrix_file = self._resolve_transformation_matrix_file(transformation_matrix_file)
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
        self.pointing_duration = 3.0  # 3 seconds
        
        # Return-to-observer state tracking
        self.returning_to_observer = False
        self.return_start_time = 0
        self.stabilization_time = 3.0
        
        # Performance monitoring
        self.performance_stats = {
            'total_detections': 0,
            'pointing_commands': 0,
            'detection_times': [],
            'avg_fps': 0.0
        }
        
        # Joint control parameters (for camera tilt functionality)
        self.joint_step = 30  # degrees per keypress for joints
        self.base_joint_angles = None  # Joint angles for look_at_table pose (base reference)
        self.last_movement_time = 0
        self.movement_cooldown = 0.1  # Minimum time between movements
        
        # Camera-to-robot transformer
        self.camera_to_robot_transformer = None
        
        # UI controls
        self.show_help = False
        self.show_performance = True
        
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
    
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful
        """
        print("Initializing Virtual Fixed Camera Tracking System")
        print("=" * 50)
        
        # Initialize sensor manager
        print("Initializing camera sensor...")
        self.sensor_manager = RealSenseManager(self.bag_file, realtime_mode=True, use_imu=True)
        if not self.sensor_manager.initialize():
            print("Error: Failed to initialize camera sensor")
            return False
        
        if not self.sensor_manager.start():
            print("Error: Failed to start camera sensor")
            return False
        
        # Initialize robot controller first
        print("Initializing robot controller...")
        self.robot_controller = MyCobotController(self.transformation_matrix_file)
        
        # Load transformation matrix
        if self.robot_controller.transformation_matrix is None:
            print("Warning: No transformation matrix loaded. Robot commands will be disabled.")
        else:
            print(f"✓ Transformation matrix loaded from: {self.transformation_matrix_file}")
            print(f"  Matrix shape: {self.robot_controller.transformation_matrix.shape}")
        
        # Initialize camera-to-robot transformer
        try:
            self.camera_to_robot_transformer = CameraToRobotTransformer(self.transformation_matrix_file)
            if self.camera_to_robot_transformer.is_ready():
                print(f"[OK] Camera-to-robot transformer initialized successfully")
            else:
                print(f"[ERROR] Camera-to-robot transformer failed to initialize")
        except Exception as e:
            print(f"[ERROR] Failed to create camera-to-robot transformer: {e}")
            self.camera_to_robot_transformer = None
        
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
                window_name="Virtual Fixed Camera Tracking",
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
        
        # Calibrate gravity alignment AFTER robot is in stable observation pose
        if self.sensor_manager.use_imu:
            print("Calibrating gravity alignment (robot is now stable)...")
            self.sensor_manager.calibrate_gravity_alignment()
        
        # Initialize detection strategies BEFORE auto ROI (need strategies available)
        print("Initializing detection strategies...")
        self._initialize_strategies()
        
        # Perform auto ROI detection if enabled
        if self.enable_auto_roi:
            print("Performing auto ROI detection...")
            if self._detect_and_set_dynamic_roi():
                print("✓ Auto ROI detection successful")
            else:
                print("! Auto ROI detection failed - using default parameters")
        
        # Set initial strategy
        if not self._switch_strategy(self.current_strategy_name):
            print(f"Error: Failed to initialize strategy '{self.current_strategy_name}'")
            return False
        
        print("[OK] All systems initialized successfully")
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
            
            # Calculate table bounds
            table_bounds = {
                'x_min': np.min(table_points[:, 0]),
                'x_max': np.max(table_points[:, 0]),
                'z_min': np.min(table_points[:, 2]),
                'z_max': np.max(table_points[:, 2])
            }
            
            print(f"+ Table surface detected:")
            print(f"  Height: {table_height_mean:.3f}m ± {table_height_std:.3f}m")
            print(f"  Inliers: {len(table_points)}/{len(filtered_vertices)} points (after distance filtering)")
            print(f"  Bounds: X[{table_bounds['x_min']:.2f}, {table_bounds['x_max']:.2f}], Z[{table_bounds['z_min']:.2f}, {table_bounds['z_max']:.2f}]")
            
            return {
                'table_height': table_height_mean,
                'table_height_std': table_height_std,
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
            
            # Extract table height
            detected_table_height = table_info['table_height']
            table_height_std = table_info['table_height_std']
            
            # Validate that this looks like a table surface (not floor)
            expected_table_range = (0.15, 0.6)  # Based on manual presets from calibration
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
    
    def _grasp_and_return(self):
        """Grasp closest detected object and return to observation pose."""
        if not self.robot_controller.is_connected:
            print("Robot not connected - cannot grasp object")
            return
        
        if self.pointing_in_progress:
            print("Robot is already busy - cannot start grasping")
            return
        
        # Find closest object
        closest_object = self._find_closest_object(self.last_detection_objects)
        if closest_object is None:
            print("No valid target object found for grasping")
            return
        
        # Extract position
        camera_position = self._extract_object_position(closest_object)
        if camera_position is None:
            print("Could not extract object position")
            return
        
        print(f"\n{'='*60}")
        print(f"GRASPING ACTION - {time.strftime('%H:%M:%S')}")
        print(f"Target: ({camera_position[0]*1000:.1f}, {camera_position[1]*1000:.1f}, {camera_position[2]*1000:.1f}) mm")
        print(f"{'='*60}")
        
        # Start grasping sequence
        self.pointing_in_progress = True  # Use same flag to prevent conflicts
        self.pointing_start_time = time.time()
        
        try:
            if self.robot_controller.async_mode_enabled:
                # Async grasping sequence
                print("1. Moving to hover position above object...")
                
                # Transform camera coordinates to robot coordinates using new transformer
                if self.camera_to_robot_transformer is None:
                    print("Camera-to-robot transformer not available")
                    self.pointing_in_progress = False
                    return
                
                try:
                    robot_pose = self.robot_controller.get_current_position()
                    if robot_pose is None:
                        print("Could not get current robot position")
                        self.pointing_in_progress = False
                        return
                    
                    # Normalize robot pose units
                    robot_pose = self._normalize_robot_pose(robot_pose)
                    
                    robot_position = self.camera_to_robot_transformer.transform_to_robot(
                        camera_position, robot_pose, input_units='meters', output_units='mm'
                    )
                    print(f"Transformed position: {robot_position} mm (using new transformer)")
                except Exception as e:
                    print(f"Failed to transform camera coordinates to robot coordinates: {e}")
                    self.pointing_in_progress = False
                    return
                
                # Create hover position (120mm height as requested)
                hover_height = 150  # mm absolute height
                hover_position = [robot_position[0], robot_position[1], hover_height]
                hover_pose = hover_position + [0, 180, 45]  # Default orientation
                
                # Step 1: Move to hover position
                hover_cmd = self.robot_controller.move_to_position_async(hover_pose, speed=30, priority=2)
                if hover_cmd:
                    self._wait_for_robot_movement(hover_cmd)
                    print("✓ Reached hover position")
                else:
                    print("✗ Failed to move to hover position")
                    self.pointing_in_progress = False
                    return
                
                # Step 2: Open gripper
                print("2. Opening gripper...")
                if self.robot_controller.open_gripper():
                    time.sleep(1)  # Wait for gripper to open
                    print("✓ Gripper opened")
                else:
                    print("✗ Failed to open gripper")
                    self.pointing_in_progress = False
                    return
                
                # Step 3: Move down to grasp position
                print("3. Moving down to grasp object...")
                # Use a safe grasp height that's above the robot's minimum Z bound
                grasp_height = max(robot_position[2] + 20, 70)  # At least 60mm height for safety
                grasp_position = [robot_position[0], robot_position[1], grasp_height]
                grasp_pose = grasp_position + [0, 180, 45]
                
                grasp_cmd = self.robot_controller.move_to_position_async(grasp_pose, speed=20, priority=2)
                if grasp_cmd:
                    self._wait_for_robot_movement(grasp_cmd)
                    print("✓ Reached grasp position")
                else:
                    print("✗ Failed to move to grasp position")
                    self.pointing_in_progress = False
                    return
                
                # Step 4: Close gripper
                print("4. Closing gripper...")
                if self.robot_controller.close_gripper():
                    time.sleep(2)  # Wait for gripper to close and secure object
                    print("✓ Gripper closed - object grasped")
                else:
                    print("✗ Failed to close gripper")
                    self.pointing_in_progress = False
                    return
                
                # Step 5: Lift object
                print("5. Lifting object...")
                lift_height = 150  # Lift to 150mm height for safe transport
                lift_position = [robot_position[0], robot_position[1], lift_height]
                lift_pose = lift_position + [0, 180, 45]
                
                lift_cmd = self.robot_controller.move_to_position_async(lift_pose, speed=25, priority=2)
                if lift_cmd:
                    self._wait_for_robot_movement(lift_cmd)
                    print("✓ Object lifted")
                else:
                    print("✗ Failed to lift object")
                
                # Step 6: Return to observation pose
                print("6. Returning to observation pose...")
                from module.utils.robot_utils import move_to_standard_pose
                move_to_standard_pose(self.robot_controller, "table", async_mode=True)
                print("✓ Returned to observation pose")
                
                print(f"{'='*60}")
                print("GRASPING COMPLETED SUCCESSFULLY")
                print(f"{'='*60}")
            
            else:
                print("Async mode not enabled - grasping requires async mode")
                
        except Exception as e:
            print(f"Error during grasping: {e}")
        finally:
            self.pointing_in_progress = False
    
    def _wait_for_robot_movement(self, command_id: int, timeout: float = 10.0, stabilization_delay: float = 2.0):
        """Wait for robot movement command to complete and stabilize."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.robot_controller.get_command_status(command_id)
            if status and status.value in ['completed', 'failed']:
                if status.value == 'completed':
                    print("[OK] Robot movement completed")
                    # Add stabilization delay to ensure robot has fully stopped
                    if stabilization_delay > 0:
                        print(f"Waiting {stabilization_delay}s for robot stabilization...")
                        time.sleep(stabilization_delay)
                        print("[OK] Robot stabilized")
                else:
                    print("[ERROR] Robot movement failed")
                return status.value == 'completed'
            
            time.sleep(0.1)
        
        print("[WARNING] Robot movement timed out")
        return False
    
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
    
    def run(self):
        """Main tracking loop."""
        if not self.initialize():
            print("Failed to initialize system")
            return
        
        print("\nVirtual Fixed Camera Tracking Started! (with Modular Transformer)")
        print("=" * 65)
        print("Controls:")
        print("  'q' - Quit")
        print("  'h' - Toggle help")
        print("  'p' - POINT AT OBJECT (3 sec, return to observation)")
        print("  'space' - Pause/Resume")
        print("  '1' - Switch to BEV strategy")
        print("  '2' - Switch to DBSCAN strategy")
        print("  'r' - Return to observation pose")
        print("  'f' - Look forward (horizontal view)")
        print("  's' - Stop robot movement (or tilt camera down when idle)")
        print("  'w' - Tilt camera UP (Joint4 +30°)")
        print("  'i' - Toggle performance info")
        print("  'a' - Recalibrate auto ROI")
        print("  't' - Toggle position reporting interval (1s/5s/10s)")
        print("  'g' - GRASP OBJECT (hover, open gripper, grasp, lift, return)")
        print("-" * 50)
        print("Robot stays in observation pose for continuous detection.")
        print("Press 'p' to point at detected objects when needed.")
        print("-" * 50)
        
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
            print("\nTracking interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _detect_objects(self, vertices: np.ndarray, color_image: Optional[np.ndarray]) -> Tuple[List[Dict[str, Any]], Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect objects using current strategy and get visualization data."""
        detection_start = time.time()
        
        try:
            # Skip heavy 3D processing during pointing mode
            if self.pointing_in_progress:
                # During pointing: return empty detection results but keep timing
                return [], {'total_time': time.time() - detection_start, 'num_objects': 0}, None, None
            
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
    
    def _point_and_return(self):
        """Start pointing at closest detected object (non-blocking)."""
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
        
        # Extract position
        camera_position = self._extract_object_position(closest_object)

        if camera_position is None:
            print("Could not extract object position")
            return
        
        print(f"\n{'='*60}")
        print(f"POINTING ACTION - {time.strftime('%H:%M:%S')}")
        print(f"Target: ({camera_position[0]*1000:.1f}, {camera_position[1]*1000:.1f}, {camera_position[2]*1000:.1f}) mm")
        print(f"{'='*60}")
        
        # Start non-blocking pointing sequence
        self.pointing_in_progress = True
        self.pointing_start_time = time.time()
        
        try:
            # Transform camera coordinates to robot coordinates using new transformer
            if self.camera_to_robot_transformer is None:
                print("Camera-to-robot transformer not available")
                self.pointing_in_progress = False
                return
            
            try:
                robot_pose = self.robot_controller.get_current_position()
                if robot_pose is None:
                    print("Could not get current robot position")
                    self.pointing_in_progress = False
                    return
                
                # Normalize robot pose units
                robot_pose = self._normalize_robot_pose(robot_pose)
                
                robot_position = self.camera_to_robot_transformer.transform_to_robot(
                    camera_position, robot_pose, input_units='meters', output_units='mm'
                )
                print(f"Transformed position: {robot_position} mm (using new transformer)")
                
                if self.robot_controller.async_mode_enabled:
                    # Async pointing command with transformed coordinates
                    print("1. Pointing at object...")
                    from module.robot.command_queue import CommandType
                    command_id = self.robot_controller.submit_async_command(
                        CommandType.POINT_AT_POSITION, robot_position, priority=2
                    )
                    
                    if command_id:
                        print(f"[OK] Pointing command submitted (ID: {command_id})")
                        print("   RGB feed continues during pointing...")
                    else:
                        print("[ERROR] Failed to submit pointing command")
                        self.pointing_in_progress = False
                else:
                    # Synchronous pointing command (still blocking, but minimal)
                    print("1. Pointing at object...")
                    success = self.robot_controller.point_at_position(robot_position)
                    
                    if success:
                        print("[OK] Robot pointing at object")
                        print("   RGB feed continues during pointing...")
                    else:
                        print("[ERROR] Failed to point at object")
                        self.pointing_in_progress = False
                        
            except Exception as e:
                print(f"Failed to transform camera coordinates to robot coordinates: {e}")
                self.pointing_in_progress = False
                return
        
        except Exception as e:
            print(f"Error during pointing sequence: {e}")
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
                print("[OK] Point and return sequence fully completed")
    
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
            
            # Calculate distance from camera origin
            distance = np.sqrt(sum(p**2 for p in position))
            
            if distance < min_distance:
                min_distance = distance
                closest_object = obj
        
        return closest_object
    
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
    
    def _normalize_robot_pose(self, robot_pose: List[float]) -> List[float]:
        """
        Normalize robot pose to expected units (mm for position, degrees for angles).
        
        Based on pymycobot source analysis, the controller sometimes returns raw integer 
        values that need to be processed through _int2coord (divide by 10) conversion.
        """
        if not robot_pose or len(robot_pose) < 6:
            return robot_pose
        
        x, y, z, rx, ry, rz = robot_pose[:6]
        
        # Detect if positions are raw integer values that need _int2coord conversion
        # Raw values are typically in the hundreds of thousands to millions range
        if abs(x) > 50000 or abs(y) > 50000 or abs(z) > 50000:
            # Apply _int2coord conversion: divide by 10 to get millimeters
            x, y, z = x / 10.0, y / 10.0, z / 10.0
            print(f"[DEBUG] Applied _int2coord conversion (÷10): [{x:.1f}, {y:.1f}, {z:.1f}] mm")
        
        # Detect if positions are in micrometers (values > 5000 but < 50000 suggest micrometers)
        elif abs(x) > 5000 or abs(y) > 5000 or abs(z) > 5000:
            # Convert micrometers to millimeters
            x, y, z = x / 1000.0, y / 1000.0, z / 1000.0
            print(f"[DEBUG] Converted robot pose from micrometers to mm: [{x:.1f}, {y:.1f}, {z:.1f}]")
        
        # Detect if positions are in meters (values < 1 suggest meters)
        elif abs(x) < 1 and abs(y) < 1 and abs(z) < 1:
            # Convert meters to millimeters
            x, y, z = x * 1000.0, y * 1000.0, z * 1000.0
            print(f"[DEBUG] Converted robot pose from meters to mm: [{x:.1f}, {y:.1f}, {z:.1f}]")
        
        # Detect if angles are in radians (values > 6.28 suggest they're not radians)
        if abs(rx) > 6.28 or abs(ry) > 6.28 or abs(rz) > 6.28:
            # Angles are likely already in degrees
            pass
        else:
            # Convert radians to degrees
            rx, ry, rz = np.degrees([rx, ry, rz])
            print(f"[DEBUG] Converted robot angles from radians to degrees: [{rx:.1f}, {ry:.1f}, {rz:.1f}]")
        
        return [x, y, z, rx, ry, rz]
    
    def _report_positions_periodically(self, detected_objects: List[Dict[str, Any]]):
        """Print comprehensive position report with robot pose, transformations, and coordinate details."""
        current_time = time.time()
        
        # Check if it's time for a position report
        if current_time - self.last_position_report_time < self.position_report_interval:
            return
        
        # Report basic info even without objects or robot
        try:
            print(f"\n{'='*80}")
            print(f"COMPREHENSIVE POSITION REPORT - {time.strftime('%H:%M:%S')}")
            print(f"{'='*80}")
            
            # Robot status and pose information
            if self.robot_controller.is_connected:
                robot_pose = self.robot_controller.get_current_position()
                if robot_pose is not None:
                    # Normalize robot pose units
                    robot_pose = self._normalize_robot_pose(robot_pose)
                    
                    print(f"ROBOT STATUS:")
                    print(f"  Connected: YES")
                    print(f"  Current Pose: [{robot_pose[0]:7.1f}, {robot_pose[1]:7.1f}, {robot_pose[2]:7.1f}, "
                          f"{robot_pose[3]:6.1f}, {robot_pose[4]:6.1f}, {robot_pose[5]:6.1f}]")
                    print(f"                (X=forward, Y=left, Z=up in mm; RX,RY,RZ in degrees)")
                    print(f"  Mode: {'POINTING' if self.pointing_in_progress else 'RETURNING' if self.returning_to_observer else 'OBSERVING'}")
                    print()
                else:
                    print(f"ROBOT STATUS:")
                    print(f"  Connected: YES")
                    print(f"  Current Pose: [UNAVAILABLE - position read failed]")
                    print()
            else:
                robot_pose = None
                print(f"ROBOT STATUS:")
                print(f"  Connected: NO")
                print(f"  Current Pose: [NOT AVAILABLE]")
                print()
            
            # Transformation system status
            print(f"TRANSFORMATION SYSTEM:")
            if self.camera_to_robot_transformer is not None:
                transformer_ready = self.camera_to_robot_transformer.is_ready()
                print(f"  Transformer: {'READY' if transformer_ready else 'NOT READY'}")
                print(f"  Matrix File: {self.transformation_matrix_file}")
                if hasattr(self.camera_to_robot_transformer, 'transformation_matrix') and \
                   self.camera_to_robot_transformer.transformation_matrix is not None:
                    matrix = self.camera_to_robot_transformer.transformation_matrix
                    print(f"  Matrix Size: {matrix.shape}")
                    # Show first row as example
                    print(f"  Matrix[0]:   [{matrix[0,0]:8.4f}, {matrix[0,1]:8.4f}, {matrix[0,2]:8.4f}, {matrix[0,3]:8.1f}]")
                else:
                    print(f"  Matrix: [NOT LOADED]")
            else:
                print(f"  Transformer: NOT AVAILABLE")
                print(f"  Matrix File: {self.transformation_matrix_file}")
            print()
            
            # Detection information
            print(f"DETECTION STATUS:")
            print(f"  Strategy: {self.current_strategy_name.upper()}")
            print(f"  Objects Detected: {len(detected_objects) if detected_objects else 0}")
            print(f"  Detection FPS: {self.temporal_filter.calculate_fps():.1f}")
            if hasattr(self, 'table_surface_height') and self.table_surface_height is not None:
                print(f"  Table Height: {self.table_surface_height:.3f}m ({self.table_surface_height*1000:.0f}mm)")
            print(f"  Auto ROI: {'ENABLED' if self.enable_auto_roi else 'DISABLED'}")
            print()
            
            # Object position details
            if detected_objects and len(detected_objects) > 0:
                print(f"DETECTED OBJECT POSITIONS:")
                print(f"  Coordinate Systems:")
                print(f"    Camera: X=right, Y=down, Z=forward (standard camera coordinates)")
                print(f"    Robot:  X=forward, Y=left, Z=up (robot base coordinates)")
                print()
                
                # Report up to 5 objects with detailed step-by-step transformation
                for i, obj in enumerate(detected_objects[:5]):
                    camera_position = self._extract_object_position(obj)
                    if camera_position:
                        print(f"  Object {i+1} - DETAILED TRANSFORMATION:")
                        print(f"    ========================================")
                        
                        # Step 1: Original object detection data
                        print(f"    STEP 1: Original Detection Data")
                        if 'center_mm' in obj:
                            original_mm = obj['center_mm']
                            print(f"      Detection Format:   center_mm")
                            print(f"      Raw Values:         [{original_mm[0]:7.1f}, {original_mm[1]:7.1f}, {original_mm[2]:7.1f}] mm")
                            print(f"      Converted to m:     [{original_mm[0]/1000:.6f}, {original_mm[1]/1000:.6f}, {original_mm[2]/1000:.6f}] m")
                        elif 'center' in obj:
                            original_m = obj['center']
                            print(f"      Detection Format:   center")
                            print(f"      Raw Values:         [{original_m[0]:8.6f}, {original_m[1]:8.6f}, {original_m[2]:8.6f}] m")
                            print(f"      Converted to mm:    [{original_m[0]*1000:7.1f}, {original_m[1]*1000:7.1f}, {original_m[2]*1000:7.1f}] mm")
                        
                        # Step 2: Normalized camera position
                        cam_mm = [pos * 1000 for pos in camera_position]
                        print(f"    STEP 2: Normalized Camera Position (meters)")
                        print(f"      Camera Position:    [{camera_position[0]:8.6f}, {camera_position[1]:8.6f}, {camera_position[2]:8.6f}] m")
                        print(f"      Camera Position:    [{cam_mm[0]:7.1f}, {cam_mm[1]:7.1f}, {cam_mm[2]:7.1f}] mm (for display)")
                        print(f"      Coordinate System:  X=right, Y=down, Z=forward")
                        
                        # Calculate distance from camera
                        distance_m = np.sqrt(sum(p**2 for p in camera_position))
                        distance_mm = distance_m * 1000
                        print(f"      Distance from Cam:  {distance_mm:.1f}mm ({distance_m:.6f}m)")
                        
                        # Step 3: Robot pose information
                        print(f"    STEP 3: Current Robot Pose")
                        if robot_pose is not None:
                            print(f"      Raw Robot Pose:     [{robot_pose[0]:7.1f}, {robot_pose[1]:7.1f}, {robot_pose[2]:7.1f}, {robot_pose[3]:6.1f}, {robot_pose[4]:6.1f}, {robot_pose[5]:6.1f}]")
                            print(f"      Position (mm):      [{robot_pose[0]:7.1f}, {robot_pose[1]:7.1f}, {robot_pose[2]:7.1f}]")
                            print(f"      Orientation (deg):  [{robot_pose[3]:6.1f}, {robot_pose[4]:6.1f}, {robot_pose[5]:6.1f}]")
                            print(f"      Coordinate System:  X=forward, Y=left, Z=up")
                        else:
                            print(f"      Robot Pose:         [NOT AVAILABLE]")
                        
                        # Step 4: Transformation matrix information
                        print(f"    STEP 4: Transformation Matrix Info")
                        if self.camera_to_robot_transformer is not None and hasattr(self.camera_to_robot_transformer, 'eye_in_hand_transform'):
                            transform_matrix = self.camera_to_robot_transformer.eye_in_hand_transform
                            print(f"      Matrix File:        {self.transformation_matrix_file}")
                            print(f"      Matrix Shape:       {transform_matrix.shape}")
                            print(f"      Translation:        [{transform_matrix[0,3]*1000:6.2f}, {transform_matrix[1,3]*1000:6.2f}, {transform_matrix[2,3]*1000:6.2f}] mm")
                            print(f"      Rotation (top-left 3x3):")
                            for row_idx in range(3):
                                row = transform_matrix[row_idx, :3]
                                print(f"        [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}]")
                        else:
                            print(f"      Transformation:     [NOT AVAILABLE]")
                        
                        # Step 5: Detailed transformation calculation
                        if self.camera_to_robot_transformer is not None and robot_pose is not None:
                            try:
                                print(f"    STEP 5: Transformation Calculation")
                                
                                # Create homogeneous coordinates
                                position_homogeneous = np.array([camera_position[0], camera_position[1], camera_position[2], 1.0])
                                print(f"      Homogeneous Pos:    [{position_homogeneous[0]:8.6f}, {position_homogeneous[1]:8.6f}, {position_homogeneous[2]:8.6f}, {position_homogeneous[3]:1.0f}]")
                                
                                # Get robot transformation matrix
                                robot_transform = self.camera_to_robot_transformer.pose_to_transform_matrix(robot_pose)
                                print(f"      Robot Transform Matrix (4x4):")
                                for row_idx in range(4):
                                    row = robot_transform[row_idx, :]
                                    print(f"        [{row[0]:8.5f}, {row[1]:8.5f}, {row[2]:8.5f}, {row[3]:8.5f}]")
                                
                                # Apply eye-in-hand transformation first
                                eye_transform = self.camera_to_robot_transformer.eye_in_hand_transform
                                position_after_eye = eye_transform @ position_homogeneous
                                print(f"      After Eye Transform: [{position_after_eye[0]:8.6f}, {position_after_eye[1]:8.6f}, {position_after_eye[2]:8.6f}, {position_after_eye[3]:6.3f}]")
                                print(f"      After Eye (mm):      [{position_after_eye[0]*1000:7.2f}, {position_after_eye[1]*1000:7.2f}, {position_after_eye[2]*1000:7.2f}]")
                                
                                # Apply robot transformation
                                position_final = robot_transform @ position_after_eye
                                print(f"      Final Position (m):  [{position_final[0]:8.6f}, {position_final[1]:8.6f}, {position_final[2]:8.6f}]")
                                
                                # Final result
                                robot_position = self.camera_to_robot_transformer.transform_to_robot(
                                    camera_position, robot_pose, input_units='meters', output_units='mm'
                                )
                                
                                print(f"    STEP 6: Final Results")
                                print(f"      Robot Coords (mm):  [{robot_position[0]:7.1f}, {robot_position[1]:7.1f}, {robot_position[2]:7.1f}]")
                                
                                # Calculate distance from robot base
                                robot_distance = np.sqrt(robot_position[0]**2 + robot_position[1]**2 + robot_position[2]**2)
                                print(f"      Robot Distance:     {robot_distance:.1f}mm ({robot_distance/1000:.6f}m)")
                                
                                # Check if within robot workspace (approximate)
                                reach_limit = 320  # mm - approximate reach for myCobot 280
                                xy_distance = np.sqrt(robot_position[0]**2 + robot_position[1]**2)
                                within_reach = xy_distance <= reach_limit and robot_position[2] >= 50  # minimum safe height
                                print(f"      XY Distance:        {xy_distance:.1f}mm")
                                print(f"      Z Height:           {robot_position[2]:.1f}mm")
                                print(f"      Within Reach:       {'YES' if within_reach else 'NO'} (XY limit: {reach_limit}mm, Z min: 50mm)")
                                
                                # Additional validation calculations
                                print(f"    STEP 7: Validation & Analysis")
                                print(f"      Camera to Robot Distance Change: {distance_mm:.1f}mm -> {robot_distance:.1f}mm")
                                print(f"      Coordinate System Change:")
                                print(f"        Camera: [{cam_mm[0]:7.1f}, {cam_mm[1]:7.1f}, {cam_mm[2]:7.1f}] (right, down, forward)")
                                print(f"        Robot:  [{robot_position[0]:7.1f}, {robot_position[1]:7.1f}, {robot_position[2]:7.1f}] (forward, left, up)")
                                
                                # Show transformation quality
                                if robot_distance < 1000 and robot_distance > 10:  # Reasonable range
                                    quality = "GOOD"
                                elif robot_distance < 10:
                                    quality = "SUSPICIOUSLY CLOSE - Check calibration"
                                else:
                                    quality = "SUSPICIOUSLY FAR - Check calibration"
                                print(f"      Transformation Quality: {quality}")
                                
                            except Exception as e:
                                print(f"    STEP 5: Transformation FAILED")
                                print(f"      Error: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"    STEP 5: Transformation NOT POSSIBLE")
                            if robot_pose is None:
                                print(f"      Reason: Robot pose unavailable")
                            else:
                                print(f"      Reason: Transformer not available")
                        
                        print(f"    ========================================")
                        print()
                        
                if len(detected_objects) > 5:
                    print(f"  ... and {len(detected_objects) - 5} more objects")
                    print()
            else:
                print(f"DETECTED OBJECT POSITIONS:")
                print(f"  No objects currently detected")
                print()
            
            # Performance metrics
            print(f"SYSTEM PERFORMANCE:")
            print(f"  Total Frames: {self.frame_count}")
            print(f"  Pointing Actions: {self.performance_stats.get('pointing_commands', 0)}")
            print(f"  Report Interval: {self.position_report_interval}s")
            if hasattr(self.performance_stats, 'detection_times') and self.performance_stats['detection_times']:
                avg_detection = np.mean(self.performance_stats['detection_times']) * 1000
                print(f"  Avg Detection Time: {avg_detection:.1f}ms")
            print()
            
            print(f"{'='*80}")
            
            self.last_position_report_time = current_time
            
        except Exception as e:
            print(f"Error in comprehensive position reporting: {e}")
            import traceback
            traceback.print_exc()
    
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
        """Create enhanced 2x2 visualization layout with raw point cloud support."""
        try:
            # Get strategy info
            strategy_info = self.detector.get_strategy_info() if self.detector else {}
            
            panel_size = 400
            
            # Initialize 2x2 grid panels
            top_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)    # RGB
            top_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)   # Raw Point Cloud
            bottom_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8) # BEV
            bottom_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8) # Side View
            
            # Top-left: RGB image with tracking info overlay
            if color_image is not None:
                rgb_panel = color_image.copy()
                # Add tracking info overlay
                self._add_tracking_info_overlay(rgb_panel, detected_objects, timing_info, strategy_info)
                # Add object detection overlay
                self._add_object_detection_overlay(rgb_panel, detected_objects)
                
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
            cv2.imshow('Virtual Fixed Camera Tracking', combined)
            
        except Exception as e:
            print(f"Warning: Visualization error: {e}")
            # Fallback to simple color image display
            if color_image is not None:
                cv2.imshow('Virtual Fixed Camera Tracking', color_image)
    
    def _add_tracking_info_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]],
                                 timing_info: Dict[str, float], strategy_info: Dict[str, str]):
        """Add tracking information overlay to image."""
        
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
        
        info_lines = [
            "VIRTUAL FIXED CAMERA TRACKING",
            f"Strategy: {self.current_strategy_name.upper()} ({strategy_info.get('clustering_strategy', 'Unknown')})",
            f"Objects: {len(detected_objects)}",
            f"FPS: {self.temporal_filter.calculate_fps():.1f}",
            f"Robot: {robot_status}",
            f"Mode: {'POINTING' if self.pointing_in_progress else 'RETURNING' if self.returning_to_observer else 'OBSERVING'}",
            "Press 'p' to point at object"
        ]
        
        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            if "POINTING" in line or "RETURNING" in line:
                color = (0, 255, 255)  # Yellow for pointing/returning status
            cv2.putText(image, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add prominent mode indicators
        if self.pointing_in_progress:
            cv2.putText(image, "3D DETECTION PAUSED - POINTING MODE", 
                       (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif self.returning_to_observer:
            cv2.putText(image, "RETURNING TO OBSERVER POSE - STABILIZING", 
                       (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    def _add_object_detection_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]]):
        """Add object detection information overlay."""
        if not detected_objects:
            return
        
        y_offset = 200
        for i, obj in enumerate(detected_objects[:3]):  # Show first 3 objects
            position = self._extract_object_position(obj)
            if position:
                # Format: (x, y, z) in camera coordinates, convert to mm for display
                info_text = f"Obj{i+1}: ({position[0]*1000:.0f}, {position[1]*1000:.0f}, {position[2]*1000:.0f})mm"
                cv2.putText(image, info_text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _rotate_joint(self, joint_index: int, angle_delta: float, message: str):
        """
        Rotate a specific joint by the given angle delta.
        Adapted from CharUco system for camera tilt control.
        
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
        
        if key == ord('q'):
            return False
        elif key == ord('h'):
            self.show_help = not self.show_help
            print(f"Help {'enabled' if self.show_help else 'disabled'}")
        elif key == ord('p'):
            # CHANGED: 'p' now triggers pointing action instead of performance toggle
            print("Pointing action triggered!")
            self._point_and_return()
        elif key == ord('i'):
            # NEW: 'i' for performance info (moved from 'p')
            self.show_performance = not self.show_performance
            print(f"Performance info {'enabled' if self.show_performance else 'disabled'}")
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
        elif key == ord('a'):
            # Auto ROI recalibration
            if self.enable_auto_roi:
                print("Recalibrating auto ROI...")
                self._recalibrate_auto_roi()
            else:
                print("Auto ROI is disabled")
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
        elif key == ord('g'):
            # Grasp object action
            if self.robot_controller.is_connected:
                print("Grasping object...")
                self._grasp_and_return()
            else:
                print("Robot not connected - cannot grasp object")
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
             time.time() - self.last_movement_time > self.movement_cooldown:
            self._rotate_joint(3, self.joint_step, "Tilting camera up (joint4)")
        elif key == ord('s') and self.robot_controller.is_connected and \
             time.time() - self.last_movement_time > self.movement_cooldown:
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
        """Update performance statistics."""
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
        print("\nShutting down Virtual Fixed Camera Tracking System...")
        
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
        
        print("[OK] Shutdown complete")
    
    def _print_final_statistics(self):
        """Print final performance statistics."""
        stats = self.performance_stats
        
        print("\n" + "=" * 50)
        print("FINAL PERFORMANCE STATISTICS")
        print("=" * 50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Pointing commands: {stats['pointing_commands']}")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        
        if stats['detection_times']:
            avg_detection = np.mean(stats['detection_times']) * 1000
            print(f"Average detection time: {avg_detection:.1f}ms")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Virtual Fixed Camera Tracking Application')
    parser.add_argument('--config', default='virtual_fixed_camera',
                       help='Configuration preset (default: virtual_fixed_camera)')
    parser.add_argument('--strategy', default='bev', choices=['bev', 'dbscan'],
                       help='Initial clustering strategy')
    parser.add_argument('--bag', help='RealSense bag file for replay')
    parser.add_argument('--transform', default='auto',
                       help='Camera-robot transformation matrix file or "auto" for automatic detection')
    parser.add_argument('--visualize-raw-pointcloud', action='store_true', default=True,
                       help='Enable raw point cloud visualization (shows unprocessed RealSense data, default: enabled)')
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
    
    print("Virtual Fixed Camera Tracking Application")
    print("=" * 40)
    print(f"Configuration: {args.config}")
    print(f"Initial strategy: {args.strategy}")
    if args.bag:
        print(f"Bag file: {args.bag}")
    print(f"Auto ROI detection: {'Disabled' if args.disable_auto_roi else 'Enabled'}")
    if not args.disable_auto_roi:
        print(f"Floor offset: {args.floor_offset}m")
    print("=" * 40)
    
    try:
        tracker = VirtualFixedCameraTracker(
            config_name=args.config,
            initial_strategy=args.strategy,
            bag_file=args.bag,
            transformation_matrix_file=args.transform,
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