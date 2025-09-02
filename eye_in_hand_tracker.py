#!/usr/bin/env python3
"""
Eye-in-Hand Object Tracking Application

This application adapts the modular tracking system for eye-in-hand configuration
where the camera is mounted on the robot's end effector. In this setup:
- Camera moves with the robot
- Objects are detected in camera frame 
- Robot uses camera feedback to approach and manipulate objects
- LOOK_AT_TABLE positions robot to view the workspace

Usage:
    python eye_in_hand_tracker.py [--config CONFIG_NAME] [--strategy STRATEGY]

Key Differences from Hand-to-Eye:
- No coordinate transformation between camera and robot needed
- Camera position changes with robot movement
- Objects are tracked relative to end effector
- Direct camera-to-robot commands for approaching objects
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


class EyeInHandTracker:
    """
    Eye-in-hand object tracking application.
    
    Features:
    - Camera mounted on robot end effector
    - Direct camera-to-robot control without coordinate transformation
    - Robot approaches objects detected in camera frame
    - LOOK_AT_TABLE for initial workspace viewing
    """
    
    def __init__(self, config_name: str = "realtime_tracking", 
                 initial_strategy: str = "bev",
                 transformation_matrix_file: str = "eye_in_hand_transform.npy",
                 visualize_raw_pointcloud: bool = True):
        """
        Initialize eye-in-hand tracker.
        
        Args:
            config_name: Configuration preset name
            initial_strategy: Initial clustering strategy ('bev' or 'dbscan')
            transformation_matrix_file: Path to eye-in-hand transformation matrix
            visualize_raw_pointcloud: Enable raw point cloud visualization
        """
        self.config_name = config_name
        self.transformation_matrix_file = transformation_matrix_file
        self.visualize_raw_pointcloud = visualize_raw_pointcloud
        
        # Load configuration
        self.config = self._load_configuration(config_name)
        
        # Load eye-in-hand transformation matrix
        self.transformation_matrix = None
        self._load_transformation_matrix()
        
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
        self.last_update_time = 0
        self.last_command_had_object = True
        self.last_detection_objects = []  # Store for cross-mode usage
        
        # Eye-in-hand specific parameters
        self.camera_offset = [0, 0, -50]  # Camera offset from end effector (mm)
        self.approach_distance = 100  # Distance to maintain from objects (mm)
        self.min_object_distance = 250  # Minimum distance for lidar effective range (mm)
        self.current_robot_mode = "observe"  # "observe", "approach", "manipulate"
        
        # Robot command tracking
        self.current_robot_command_id = None
        self.robot_command_history = []
        
        # Performance monitoring
        self.performance_stats = {
            'total_detections': 0,
            'successful_robot_commands': 0,
            'detection_times': [],
            'robot_command_times': [],
            'avg_fps': 0.0
        }
        
        # UI controls
        self.show_help = False
        self.show_performance = True
        self.adaptive_timing = True
        
        # Auto ROI detection
        self.auto_roi_enabled = True
        self.table_surface_height = None
        self.floor_offset = 0.01  # 10mm above detected table surface
    
    def _load_transformation_matrix(self):
        """Load eye-in-hand transformation matrix."""
        try:
            if os.path.exists(self.transformation_matrix_file):
                self.transformation_matrix = np.load(self.transformation_matrix_file)
                print(f"+ Eye-in-hand transformation matrix loaded: {self.transformation_matrix_file}")
                print(f"  Matrix shape: {self.transformation_matrix.shape}")
            else:
                print(f"Warning: Transformation matrix file not found: {self.transformation_matrix_file}")
                print("Running without transformation - objects will be in camera coordinates")
        except Exception as e:
            print(f"Error loading transformation matrix: {e}")
            self.transformation_matrix = None
    
    def transform_camera_to_end_effector(self, camera_position: List[float]) -> Optional[List[float]]:
        """
        Transform camera coordinates to end effector coordinates.
        
        Args:
            camera_position: Position in camera frame [x, y, z]
            
        Returns:
            Position in end effector frame [x, y, z] or None if transformation fails
        """
        if self.transformation_matrix is None:
            # No transformation available - return camera coordinates
            return camera_position
        
        try:
            # Convert to homogeneous coordinates
            camera_homogeneous = np.array([camera_position[0], camera_position[1], camera_position[2], 1])
            
            # Apply transformation
            ee_homogeneous = np.dot(self.transformation_matrix, camera_homogeneous)
            
            # Return 3D coordinates
            return ee_homogeneous[:3].tolist()
        except Exception as e:
            print(f"Error in camera-to-end-effector transformation: {e}")
            return None
    
    def calculate_robot_target_pose(self, camera_position: List[float], mode: str = "approach") -> Optional[List[float]]:
        """
        Calculate robot target pose to position end effector relative to detected object.
        
        Args:
            camera_position: Object position in camera coordinates [forward, side, height]
            mode: Operating mode ('observe', 'approach', 'manipulate')
            
        Returns:
            Target robot pose [x, y, z, rx, ry, rz] in robot base frame
        """
        try:
            # Get current robot pose (end effector position in base frame)
            current_robot_pose = self.robot_controller.get_current_position()
            if current_robot_pose is None:
                print("Cannot get current robot position")
                return None
            
            # Transform object position from camera frame to end effector frame
            ee_position = self.transform_camera_to_end_effector(camera_position)
            if ee_position is None:
                print("Failed to transform camera position to end effector frame")
                # Fallback: use camera position directly (assumes camera = end effector)
                ee_position = camera_position
            
            # Get mode-specific target distance (respecting lidar minimum range)
            robot_params = getattr(self.config, 'robot_params', {})
            if mode == "observe":
                target_distance = max(robot_params.get('retreat_distance', 300), self.min_object_distance)
            elif mode == "approach":
                target_distance = max(robot_params.get('approach_distance', 280), self.min_object_distance)
            elif mode == "manipulate":
                target_distance = max(250, self.min_object_distance)  # At minimum lidar range for manipulation
            else:
                target_distance = max(280, self.min_object_distance)  # Default safe distance
            
            # Calculate desired end effector position relative to object
            # Move end effector to be at target_distance from object in Z direction
            object_direction = np.array(ee_position)
            object_distance = np.linalg.norm(object_direction)
            
            if object_distance < 10:  # Too close, avoid division by zero
                print("Object too close to end effector")
                return None
            
            # Normalize direction and scale to target distance
            normalized_direction = object_direction / object_distance
            desired_ee_offset = normalized_direction * target_distance
            
            # Calculate target robot pose
            # The target is to position the end effector at the desired offset from object
            target_robot_pose = current_robot_pose.copy()
            
            # For now, keep current orientation (can be enhanced later)
            # Just adjust position to maintain target distance from object
            position_adjustment = desired_ee_offset - object_direction
            
            target_robot_pose[0] += position_adjustment[0]  # X
            target_robot_pose[1] += position_adjustment[1]  # Y  
            target_robot_pose[2] += position_adjustment[2]  # Z
            
            return target_robot_pose
            
        except Exception as e:
            print(f"Error calculating robot target pose: {e}")
            return None
    
    def _load_configuration(self, config_name: str):
        """Load configuration optimized for eye-in-hand setup."""
        
        # Eye-in-hand specific configurations
        eye_in_hand_configs = {
            'eye_in_hand_realtime': {
                'type': 'bev',
                'config': {
                    'clustering_params': {
                        'min_cluster_size': 15,  # Smaller for closer objects
                        'max_cluster_size': 80,  # Adjusted for typical object sizes
                        'morphology_kernel_size': 3,
                        'window_size': 3  # Faster response
                    },
                    'tracking_params': {
                        'update_interval': 2.0,  # Track once every 2 seconds as requested
                        'fast_update_interval': 2.0,  # Keep consistent timing
                        'adaptive_timing': False,  # Disable adaptive timing for consistent 2s interval
                        'max_no_object_time': 6.0  # Increased to match slower interval
                    },
                    'robot_params': {
                        'min_height': 150,  # Higher minimum for eye-in-hand safety
                        'max_speed': 60,    # Slower for precise positioning
                        'approach_distance': 280,  # Distance to maintain from objects (>250mm lidar minimum)
                        'retreat_distance': 350   # Distance to retreat when no objects (safe margin)
                    }
                }
            },
            'eye_in_hand_precision': {
                'type': 'bev',
                'config': {
                    'clustering_params': {
                        'min_cluster_size': 10,
                        'max_cluster_size': 150,
                        'morphology_kernel_size': 5,
                        'window_size': 5
                    },
                    'tracking_params': {
                        'update_interval': 2.0,  # Track once every 2 seconds as requested
                        'fast_update_interval': 2.0,  # Keep consistent timing
                        'adaptive_timing': False,  # Disable adaptive timing for consistent 2s interval
                        'max_no_object_time': 8.0
                    },
                    'robot_params': {
                        'min_height': 180,
                        'max_speed': 30,
                        'approach_distance': 260,    # Closer approach (still >250mm lidar minimum)
                        'retreat_distance': 400      # Further retreat for precision
                    }
                }
            }
        }
        
        if config_name in eye_in_hand_configs:
            preset = eye_in_hand_configs[config_name]
            config = ConfigFactory.create_config(preset['type'], preset['config'])
            # Add robot params as attributes to the config object
            robot_params = preset['config'].get('robot_params', {})
            setattr(config, 'robot_params', robot_params)
            setattr(config, 'tracking_params', preset['config'].get('tracking_params', {}))
            return config
        else:
            # Try to load from existing presets and adapt for eye-in-hand
            try:
                base_config = ConfigFactory.create_preset(config_name)
                # Add eye-in-hand specific parameters
                robot_params = {
                    'approach_distance': 280,  # Respect 250mm lidar minimum
                    'retreat_distance': 350,   # Safe margin above minimum
                    'min_height': 150
                }
                tracking_params = {
                    'update_interval': 2.0,  # Track once every 2 seconds as requested
                    'fast_update_interval': 2.0,  # Keep consistent timing
                    'adaptive_timing': False,  # Disable adaptive timing for consistent 2s interval
                    'max_no_object_time': 6.0
                }
                setattr(base_config, 'robot_params', robot_params)
                setattr(base_config, 'tracking_params', tracking_params)
                return base_config
            except:
                print(f"Unknown config '{config_name}', using default eye_in_hand_realtime")
                preset = eye_in_hand_configs['eye_in_hand_realtime']
                config = ConfigFactory.create_config(preset['type'], preset['config'])
                # Add robot params as attributes to the config object
                robot_params = preset['config'].get('robot_params', {})
                setattr(config, 'robot_params', robot_params)
                setattr(config, 'tracking_params', preset['config'].get('tracking_params', {}))
                return config
    
    def initialize(self) -> bool:
        """Initialize all system components for eye-in-hand setup."""
        print("Initializing Eye-in-Hand Object Tracking System")
        print("=" * 50)
        
        # Initialize sensor manager
        print("Initializing camera sensor (mounted on end effector)...")
        self.sensor_manager = RealSenseManager(None, realtime_mode=True, use_imu=True)
        if not self.sensor_manager.initialize():
            print("Error: Failed to initialize camera sensor")
            return False
        
        if not self.sensor_manager.start():
            print("Error: Failed to start camera sensor")
            return False
        
        # NOTE: Gravity calibration moved to after robot positioning and stabilization
        
        # Initialize robot controller 
        print("Initializing robot controller for eye-in-hand setup...")
        self.robot_controller = MyCobotController(transformation_matrix_file=None)
        
        # Connect to robot
        if not self.robot_controller.connect():
            print("Warning: Failed to connect to robot. Running in visualization-only mode.")
        
        # Configure robot safety parameters for eye-in-hand
        if self.robot_controller.is_connected:
            # Get robot params from config object
            if hasattr(self.config, 'robot_params'):
                robot_params = self.config.robot_params
            else:
                robot_params = getattr(self.config, 'robot_params', {})
            
            # Extract robot parameters with defaults
            min_height = getattr(robot_params, 'min_height', 150) if hasattr(robot_params, 'min_height') else robot_params.get('min_height', 150)
            max_speed = getattr(robot_params, 'max_speed', 60) if hasattr(robot_params, 'max_speed') else robot_params.get('max_speed', 60)
            approach_distance = getattr(robot_params, 'approach_distance', 100) if hasattr(robot_params, 'approach_distance') else robot_params.get('approach_distance', 100)
            retreat_distance = getattr(robot_params, 'retreat_distance', 200) if hasattr(robot_params, 'retreat_distance') else robot_params.get('retreat_distance', 200)
            
            # Update safety constraints for eye-in-hand configuration
            eye_in_hand_safety = {
                'min_height': min_height,
                'max_speed': max_speed,
                'approach_distance': approach_distance,
                'retreat_distance': retreat_distance
            }
            self.robot_controller.safety_constraints.update(eye_in_hand_safety)
            print(f"Eye-in-hand safety parameters: {eye_in_hand_safety}")
            
            # Enable async mode for smooth camera movement
            if self.robot_controller.enable_async_mode(max_queue_size=3):
                print("[OK] Robot async mode enabled for smooth eye-in-hand control")
            else:
                print("[WARNING] Failed to enable robot async mode")
            
            # Setup robot with safe home-first sequence
            from module.utils.robot_utils import setup_robot_with_stabilization
            if not setup_robot_with_stabilization(
                self.robot_controller,
                self.sensor_manager,
                pose_type="table",
                stabilization_time=3.0,
                window_name="Eye-in-Hand Object Tracking",
                home_first=True
            ):
                print("Warning: Robot setup failed")
        
        # Calibrate gravity alignment AFTER robot is in stable observer pose  
        if self.sensor_manager.use_imu:
            print("Calibrating gravity alignment (robot is now stable)...")
            self.sensor_manager.calibrate_gravity_alignment()
        
        # Initialize detection strategies
        print("Initializing detection strategies...")
        self._initialize_strategies()
        
        # Set initial strategy
        if not self._switch_strategy(self.current_strategy_name):
            print(f"Error: Failed to initialize strategy '{self.current_strategy_name}'")
            return False
        
        # Detect table surface and set dynamic ROI
        if self.auto_roi_enabled:
            print("Detecting table surface for automatic ROI adjustment...")
            if self._detect_and_set_dynamic_roi():
                print("✓ Auto ROI detection successful")
            else:
                print("! Auto ROI detection failed - using default ROI parameters")
        
        print("✓ Eye-in-hand system initialized successfully")
        return True
    
    def _initialize_strategies(self):
        """Initialize available detection strategies."""
        
        # BEV Clustering Strategy (optimized for eye-in-hand)
        try:
            bev_strategy = BEVClusteringStrategy(
                roi_params=getattr(self.config, 'roi_params', None),
                display_roi_params=getattr(self.config, 'display_roi_params', None),
                clustering_params=getattr(self.config, 'clustering_params', None),
                enable_temporal_filtering=True
            )
            self.available_strategies['bev'] = bev_strategy
            print("✓ BEV clustering strategy initialized for eye-in-hand")
        except Exception as e:
            print(f"✗ Failed to initialize BEV strategy: {e}")
        
        # DBSCAN Clustering Strategy
        try:
            dbscan_strategy = DBSCANClusteringStrategy(
                eps=0.015,  # Smaller eps for closer objects
                min_points=8
            )
            self.available_strategies['dbscan'] = dbscan_strategy
            print("✓ DBSCAN clustering strategy initialized")
        except Exception as e:
            print(f"✗ Failed to initialize DBSCAN strategy: {e}")
    
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
            print(f"✓ Switched to {strategy.get_strategy_name()} for eye-in-hand")
            return True
        except Exception as e:
            print(f"✗ Failed to switch to strategy '{strategy_name}': {e}")
            return False
    
    def run(self):
        """Main eye-in-hand tracking loop."""
        if not self.initialize():
            print("Failed to initialize eye-in-hand system")
            return
        
        print("\nEye-in-Hand Object Tracking Started!")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'h' - Toggle help")
        print("  'p' - Toggle performance info")
        print("  'space' - Pause/Resume")
        print("  '1' - Switch to BEV strategy")
        print("  '2' - Switch to DBSCAN strategy")
        print("  'l' - LOOK_AT_TABLE (position robot to view workspace)")
        print("  's' - Stop robot movement")
        print("  'o' - Switch to observe mode")
        print("  'a' - Switch to approach mode")
        print("  'm' - Switch to manipulate mode")
        print("-" * 50)
        
        self.is_running = True
        
        # Start in LOOK_AT_TABLE position
        if self.robot_controller.is_connected:
            print("Moving to LOOK_AT_TABLE position for initial workspace view...")
            self.robot_controller.look_at_table_async(priority=1)
        
        try:
            while self.is_running:
                frame_start = time.time()
                
                # Handle pause
                if self.is_paused:
                    cv2.waitKey(30)
                    continue
                
                # Get frame from sensor
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                if vertices is not None:
                    # Detect objects with visualization data
                    detected_objects, timing_info, bev_image, point_cloud_view = self._detect_objects(vertices, color_image)
                    
                    # Process robot movement based on current mode
                    if self.robot_controller.is_connected:
                        self._process_eye_in_hand_movement(detected_objects)
                    
                    # Create visualization
                    self._create_visualization(vertices, color_image, detected_objects, timing_info, metadata, bev_image, point_cloud_view)
                    
                    self.frame_count += 1
                    
                    # Update temporal filter
                    current_time = time.time()
                    self.temporal_filter.add_frame_to_history(
                        objects=detected_objects,
                        timestamp=current_time
                    )
                    
                    # Update performance stats
                    frame_time = current_time - frame_start
                    self._update_performance_stats(frame_time, timing_info)
                
                # Handle user input
                if not self._handle_user_input():
                    break
                
        except KeyboardInterrupt:
            print("\nEye-in-hand tracking interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
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
            # Remove invalid points
            valid_mask = ~np.any(np.isnan(vertices) | np.isinf(vertices), axis=1)
            valid_vertices = vertices[valid_mask]
            
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
            
            # Use X and Z as input features, Y as target (height) - using filtered vertices
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
            
            # Get table bounds
            table_bounds = {
                'x_min': np.min(table_points[:, 0]),
                'x_max': np.max(table_points[:, 0]),
                'y_mean': table_height_mean,
                'y_std': table_height_std,
                'z_min': np.min(table_points[:, 2]),
                'z_max': np.max(table_points[:, 2])
            }
            
            return {
                'plane_model': ransac,
                'table_points': table_points,
                'table_height': table_height_mean,
                'table_height_std': table_height_std,
                'bounds': table_bounds,
                'inlier_count': len(table_points)
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
        if not self.auto_roi_enabled:
            return True  # Skip if auto ROI is disabled
        
        try:
            # Get multiple frames for better table detection
            table_detection_frames = 3
            all_vertices = []
            
            for frame_idx in range(table_detection_frames):
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                if vertices is not None and len(vertices) > 100:
                    all_vertices.append(vertices)
                time.sleep(0.1)  # Brief pause between frames
            
            if not all_vertices:
                return False
            
            # Combine all frames for more robust table detection
            combined_vertices = np.vstack(all_vertices)
            
            # Detect table surface using existing RANSAC method
            table_info = self._detect_table_surface_ransac(combined_vertices, max_distance=0.015)
            
            if table_info is None:
                return False
            
            # Extract table height
            detected_table_height = table_info['table_height']
            table_height_std = table_info['table_height_std']
            
            # Validate that this looks like a table surface (not floor)
            expected_table_range = (0.22, 0.40)  # Based on manual presets from calibration
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
            
            # Update detector's ROI parameters
            if hasattr(self.detector.clustering_strategy, 'roi_params'):
                self.detector.clustering_strategy.roi_params = dynamic_roi
                self.table_surface_height = detected_table_height
                print(f"[AUTO-ROI] Set dynamic ROI: Y[{dynamic_roi['y_min']:.3f}, {dynamic_roi['y_max']:.3f}]m")
                print(f"[AUTO-ROI] Object detection at table surface level")
                return True
            else:
                print(f"[AUTO-ROI] Warning: Detector does not support ROI parameter updates")
                return False
                
        except Exception as e:
            print(f"[AUTO-ROI] Error during table detection: {e}")
            return False
    
    def _detect_objects(self, vertices: np.ndarray, color_image: Optional[np.ndarray]) -> Tuple[List[Dict[str, Any]], Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect objects using current strategy and get visualization data."""
        detection_start = time.time()
        
        try:
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
    
    def _process_eye_in_hand_movement(self, detected_objects: List[Dict[str, Any]]):
        """Process robot movement for eye-in-hand configuration."""
        current_time = time.time()
        
        # Get timing parameters
        tracking_params = getattr(self.config, 'tracking_params', {})
        robot_params = getattr(self.config, 'robot_params', {})
        
        normal_interval = tracking_params.get('update_interval', 1.0)
        fast_interval = tracking_params.get('fast_update_interval', 0.5)
        adaptive_timing = tracking_params.get('adaptive_timing', True)
        
        # Determine update interval
        if adaptive_timing and self.adaptive_timing:
            effective_interval = fast_interval if not self.last_command_had_object else normal_interval
        else:
            effective_interval = normal_interval
        
        # Check if it's time for robot update
        if current_time - self.last_update_time < effective_interval:
            return
        
        print(f"\n{'='*60}")
        print(f"Eye-in-Hand Update - {time.strftime('%H:%M:%S')} - Mode: {self.current_robot_mode.upper()}")
        print(f"Strategy: {self.current_strategy_name.upper()}")
        print(f"{'='*60}")
        
        # Process based on current mode
        if self.current_robot_mode == "observe":
            self._handle_observe_mode(detected_objects, robot_params)
        elif self.current_robot_mode == "approach":
            self._handle_approach_mode(detected_objects, robot_params)
        elif self.current_robot_mode == "manipulate":
            self._handle_manipulate_mode(detected_objects, robot_params)
        
        self.last_update_time = current_time
    
    def _handle_observe_mode(self, detected_objects: List[Dict[str, Any]], robot_params: Dict[str, Any]):
        """Handle robot behavior in observe mode - visualize only, no robot movement."""
        if not detected_objects:
            # In observe mode, just maintain current position and visualize
            self.last_command_had_object = False
            return
        
        # Find closest object for display purposes
        closest_object = self._find_closest_object(detected_objects)
        if closest_object is None:
            self.last_command_had_object = False
            return
        
        # Extract position for display only
        camera_position = self._extract_object_position(closest_object)
        if camera_position is None:
            self.last_command_had_object = False
            return
        
        # In observe mode, just report object position but don't move robot
        print(f"Observe mode: Object detected at {camera_position} - maintaining observation pose")
        
        # Store detected objects for potential use in other modes
        self.last_detection_objects = detected_objects
        self.last_command_had_object = True
    
    def _handle_approach_mode(self, detected_objects: List[Dict[str, Any]], robot_params: Dict[str, Any]):
        """Handle robot movement in approach mode - move closer to objects."""
        if not detected_objects:
            print("No objects detected - switching to observe mode")
            self.current_robot_mode = "observe"
            self.last_command_had_object = False
            return
        
        # Find closest object
        closest_object = self._find_closest_object(detected_objects)
        if closest_object is None:
            print("No valid target object found")
            self.last_command_had_object = False
            return
        
        # Extract position in camera frame
        camera_position = self._extract_object_position(closest_object)
        if camera_position is None:
            print("Could not extract object position")
            self.last_command_had_object = False
            return
        
        # Calculate robot target pose for approach mode
        target_pose = self.calculate_robot_target_pose(camera_position, mode="approach")
        if target_pose is None:
            print("Failed to calculate target pose for approach mode")
            self.last_command_had_object = False
            return
        
        print(f"Approach mode: Object at {camera_position}")
        print(f"Target robot pose: {[f'{x:.1f}' for x in target_pose]}")
        
        # Submit robot movement command
        if self.robot_controller.async_mode_enabled:
            command_id = self.robot_controller.move_to_position_async(
                target_pose,
                speed=40,  # Slower for approach
                priority=2,
                timeout=15.0,
                callback=self._robot_command_callback
            )
            if command_id:
                print(f"+ Approach command submitted (ID: {command_id})")
                self.current_robot_command_id = command_id
                self.last_command_had_object = True
                self.performance_stats['successful_robot_commands'] += 1
        
        self.last_command_had_object = True
    
    def _handle_manipulate_mode(self, detected_objects: List[Dict[str, Any]], robot_params: Dict[str, Any]):
        """Handle robot movement in manipulate mode - fine positioning for manipulation."""
        if not detected_objects:
            print("No objects detected - switching to observe mode")
            self.current_robot_mode = "observe"
            self.last_command_had_object = False
            return
        
        # Find closest object
        closest_object = self._find_closest_object(detected_objects)
        if closest_object is None:
            print("No valid target object found")
            self.last_command_had_object = False
            return
        
        # Extract position in camera frame
        camera_position = self._extract_object_position(closest_object)
        if camera_position is None:
            print("Could not extract object position")
            self.last_command_had_object = False
            return
        
        # Calculate robot target pose for manipulate mode
        target_pose = self.calculate_robot_target_pose(camera_position, mode="manipulate")
        if target_pose is None:
            print("Failed to calculate target pose for manipulate mode")
            self.last_command_had_object = False
            return
        
        print(f"Manipulate mode: Object at {camera_position}")
        print(f"Target robot pose: {[f'{x:.1f}' for x in target_pose]}")
        
        # Submit robot movement command
        if self.robot_controller.async_mode_enabled:
            command_id = self.robot_controller.move_to_position_async(
                target_pose,
                speed=20,  # Very slow for precision
                priority=3,
                timeout=20.0,
                callback=self._robot_command_callback
            )
            if command_id:
                print(f"+ Manipulation positioning command submitted (ID: {command_id})")
                self.current_robot_command_id = command_id
                self.last_command_had_object = True
                self.performance_stats['successful_robot_commands'] += 1
        
        self.last_command_had_object = True
    
    
    def _robot_command_callback(self, command):
        """Callback function for completed robot commands."""
        from module.robot.command_queue import CommandStatus
        
        # Add to command history
        self.robot_command_history.append({
            'command_id': command.command_id,
            'type': command.command_type.value,
            'status': command.status.value,
            'execution_time': command.get_execution_time(),
            'timestamp': time.time()
        })
        
        # Keep history limited
        if len(self.robot_command_history) > 20:
            self.robot_command_history.pop(0)
        
        # Clear current command if it's this one
        if self.current_robot_command_id == command.command_id:
            self.current_robot_command_id = None
        
        # Log result
        if command.status == CommandStatus.COMPLETED:
            exec_time = command.get_execution_time()
            print(f"Eye-in-hand command {command.command_id} completed in {exec_time:.2f}s")
            
            # Recalibrate gravity and ROI after robot movement (camera orientation changed)
            self._recalibrate_after_robot_movement()
            
        elif command.status == CommandStatus.FAILED:
            print(f"Eye-in-hand command {command.command_id} failed: {command.error}")
    
    def _recalibrate_after_robot_movement(self):
        """Recalibrate gravity alignment and auto ROI after robot movement."""
        try:
            print("Recalibrating after robot movement...")
            
            # Recalibrate gravity alignment (camera orientation changed)
            if self.sensor_manager and self.sensor_manager.use_imu:
                print("- Recalibrating gravity alignment...")
                self.sensor_manager.calibrate_gravity_alignment()
                print("+ Gravity alignment updated")
            
            # Re-detect and update auto ROI (table surface from new camera perspective)
            if hasattr(self, '_detect_and_set_dynamic_roi'):
                print("- Re-detecting table surface and updating ROI...")
                if self._detect_and_set_dynamic_roi():
                    print("+ Auto ROI updated successfully")
                else:
                    print("! Auto ROI re-detection failed")
            else:
                print("- Auto ROI detection not available in this tracker")
                
        except Exception as e:
            print(f"Error during post-movement recalibration: {e}")
    
    def _detect_and_set_dynamic_roi(self) -> bool:
        """
        Detect table surface and set dynamic ROI parameters automatically.
        
        Returns:
            True if table detection and ROI setting successful
        """
        print("[AUTO-ROI] Detecting table surface...")
        floor_offset = 0.01  # Default floor offset
        
        try:
            # Allow camera to stabilize first
            print("[AUTO-ROI] Allowing camera to stabilize...")
            stabilization_frames = 5  # Fewer frames for faster tracking
            for _ in range(stabilization_frames):
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                time.sleep(0.05)
            
            # Get multiple frames for table detection  
            table_detection_frames = 5  # Reduced for faster operation
            all_vertices = []
            
            for frame_idx in range(table_detection_frames):
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                if vertices is not None and len(vertices) > 100:
                    all_vertices.append(vertices)
                time.sleep(0.1)
            
            if not all_vertices:
                print("[AUTO-ROI] Warning: No valid frames for table detection")
                return False
            
            # Combine all frames for more robust table detection
            combined_vertices = np.vstack(all_vertices)
            
            # Detect table surface using RANSAC method
            table_info = self._detect_table_surface_ransac(combined_vertices, max_distance=0.015)
            
            if table_info is None:
                print("[AUTO-ROI] Warning: Failed to detect table surface")
                return False
            
            # Extract table height
            detected_table_height = table_info['table_height']
            table_height_std = table_info['table_height_std']
            
            # Validate that this looks like a table surface
            expected_table_range = (0.15, 0.6)
            if not (expected_table_range[0] <= detected_table_height <= expected_table_range[1]):
                print(f"[AUTO-ROI] Detected surface at {detected_table_height:.3f}m outside expected table range {expected_table_range}")
                return False
            
            print(f"[AUTO-ROI] Detected table surface: {detected_table_height:.3f}m ± {table_height_std:.3f}m")
            
            # Set dynamic ROI based on detected table surface
            dynamic_roi = {
                'x_min': -1.0, 'x_max': 1.0,
                'y_min': detected_table_height - floor_offset - 0.06,
                'y_max': detected_table_height - floor_offset,
                'z_min': 0, 'z_max': 0.5
            }
            
            # Update detector's ROI parameters
            if hasattr(self.detector.clustering_strategy, 'roi_params'):
                self.detector.clustering_strategy.roi_params = dynamic_roi
                print(f"[AUTO-ROI] Set dynamic ROI: Y[{dynamic_roi['y_min']:.3f}, {dynamic_roi['y_max']:.3f}]m")
                return True
            else:
                print("[AUTO-ROI] Warning: Detector does not support ROI parameter updates")
                return False
                
        except Exception as e:
            print(f"[AUTO-ROI] Error during table detection: {e}")
            return False
    
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
            
            # Calculate distance from camera origin (end effector)
            distance = np.sqrt(sum(p**2 for p in position))
            
            if distance < min_distance:
                min_distance = distance
                closest_object = obj
        
        return closest_object
    
    def _extract_object_position(self, obj: Dict[str, Any]) -> Optional[List[float]]:
        """Extract 3D position from object in camera coordinates."""
        try:
            if 'center_mm' in obj:
                # center_mm format: [x, y, z] = [side, height, forward]
                center_mm = obj['center_mm']
                return [center_mm[2], center_mm[0], center_mm[1]]  # [forward, side, height]
            elif 'center' in obj:
                # center format: [x, y, z] = [side, height, forward] in meters
                center = obj['center']
                return [center[2]*1000, center[0]*1000, center[1]*1000]  # Convert to mm
            else:
                return None
        except:
            return None
    
    def _create_raw_point_cloud_view(self, vertices: np.ndarray) -> Optional[np.ndarray]:
        """Create raw point cloud view showing camera perspective."""
        if vertices is None or len(vertices) == 0:
            return None
        
        try:
            # Get raw vertices from sensor manager (before gravity alignment)
            raw_vertices = self.sensor_manager.get_raw_vertices()
            if raw_vertices is None or len(raw_vertices) == 0:
                return None
            
            # Remove invalid points
            valid_mask = ~np.any(np.isnan(raw_vertices) | np.isinf(raw_vertices), axis=1)
            valid_vertices = raw_vertices[valid_mask]
            
            if len(valid_vertices) == 0:
                return None
            
            # Create image for camera perspective view (X-Y plane, looking along Z axis like RGB camera)
            img_size = 400
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            # Define the view bounds (in meters)
            x_range = (-0.2, 0.2)  # side (horizontal in camera view)
            y_range = (-0.2, 0.2)  # height (vertical in camera view) 
            
            # Extract coordinates
            x_coords = valid_vertices[:, 0]  # side (horizontal in camera view)
            y_coords = valid_vertices[:, 1]  # height (vertical in camera view)
            z_coords = valid_vertices[:, 2]  # depth (for color coding)
            
            # Filter points within view bounds
            mask = ((x_coords >= x_range[0]) & (x_coords <= x_range[1]) & 
                   (y_coords >= y_range[0]) & (y_coords <= y_range[1]))
            
            if not np.any(mask):
                # No points in view, add title and return empty view
                cv2.putText(img, "Raw Point Cloud", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, "(Camera View)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(img, "No points in view", (10, img_size-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                return img
            
            filtered_x = x_coords[mask]
            filtered_y = y_coords[mask]
            filtered_z = z_coords[mask]
            
            # Convert to image coordinates
            # X maps to horizontal (left-right), Y maps to vertical (top-bottom)
            img_x = ((filtered_x - x_range[0]) / (x_range[1] - x_range[0]) * (img_size - 1)).astype(int)
            img_y = ((y_range[1] - filtered_y) / (y_range[1] - y_range[0]) * (img_size - 1)).astype(int)  # Flip Y
            
            # Clamp coordinates
            img_x = np.clip(img_x, 0, img_size - 1)
            img_y = np.clip(img_y, 0, img_size - 1)
            
            # Color points based on depth (Z coordinate)
            if len(filtered_z) > 0:
                z_min, z_max = np.min(filtered_z), np.max(filtered_z)
                if z_max > z_min:
                    # Normalize depth values to [0, 1]
                    z_normalized = (filtered_z - z_min) / (z_max - z_min)
                    
                    # Create color map: blue (far) to white/yellow (close)
                    colors = np.zeros((len(filtered_z), 3))
                    colors[:, 0] = 255 * (1 - z_normalized)  # Blue component (high for far)
                    colors[:, 1] = 255 * z_normalized       # Green component (high for close)
                    colors[:, 2] = 255 * z_normalized       # Red component (high for close)
                else:
                    # All points at same depth
                    colors = np.full((len(filtered_z), 3), [255, 255, 255])  # White
                
                # Draw points
                for i in range(len(img_x)):
                    color = tuple(map(int, colors[i]))
                    cv2.circle(img, (img_x[i], img_y[i]), 1, color, -1)
            
            # Add title and labels
            cv2.putText(img, "Raw Point Cloud", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, "(Camera View)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Add depth range info if available
            if len(filtered_z) > 0:
                depth_text = f"Depth: {z_min*1000:.0f}-{z_max*1000:.0f}mm"
                cv2.putText(img, depth_text, (10, img_size-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            # Add point count
            count_text = f"Points: {len(filtered_x)}"
            cv2.putText(img, count_text, (10, img_size-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            
            return img
            
        except Exception as e:
            print(f"Error creating raw point cloud view: {e}")
            return None
    
    def _create_visualization(self, vertices: np.ndarray, color_image: Optional[np.ndarray],
                            detected_objects: List[Dict[str, Any]], timing_info: Dict[str, float],
                            metadata: Dict[str, Any], bev_image: Optional[np.ndarray] = None, 
                            point_cloud_view: Optional[np.ndarray] = None):
        """Create 2x2 visualization layout for eye-in-hand setup."""
        try:
            # Get strategy info
            strategy_info = self.detector.get_strategy_info() if self.detector else {}
            
            # Create 2x2 grid layout with 400x400 panels
            panel_size = 400
            
            # Top-left: RGB Image
            rgb_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
            if color_image is not None:
                color_panel = color_image.copy()
                # Add eye-in-hand specific info overlay
                self._add_eye_in_hand_info_overlay(color_panel, detected_objects, timing_info, strategy_info)
                # Add object detection overlay
                self._add_object_detection_overlay(color_panel, detected_objects)
                # Resize to panel size
                rgb_panel = cv2.resize(color_panel, (panel_size, panel_size))
            else:
                cv2.putText(rgb_panel, "No RGB Image", (100, panel_size//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Top-right: Raw Point Cloud
            if self.visualize_raw_pointcloud:
                raw_panel = self._create_raw_point_cloud_view(vertices)
                if raw_panel is None:
                    raw_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                    cv2.putText(raw_panel, "Raw Point Cloud", (80, panel_size//2-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(raw_panel, "No Data", (150, panel_size//2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                else:
                    raw_panel = cv2.resize(raw_panel, (panel_size, panel_size))
            else:
                raw_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                cv2.putText(raw_panel, "Raw Point Cloud", (80, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(raw_panel, "Disabled", (140, panel_size//2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            
            # Bottom-left: BEV View
            if bev_image is not None:
                bev_panel = cv2.resize(bev_image, (panel_size, panel_size))
                # Add BEV title
                cv2.putText(bev_panel, "Bird's Eye View", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                bev_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                cv2.putText(bev_panel, "Bird's Eye View", (80, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(bev_panel, "No Data", (150, panel_size//2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            
            # Bottom-right: Point Cloud Side View
            if point_cloud_view is not None:
                side_panel = cv2.resize(point_cloud_view, (panel_size, panel_size))
                # Add side view title
                cv2.putText(side_panel, "Side View", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                side_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                cv2.putText(side_panel, "Side View", (130, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(side_panel, "No Data", (150, panel_size//2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            
            # Combine into 2x2 layout
            top_row = np.hstack([rgb_panel, raw_panel])
            bottom_row = np.hstack([bev_panel, side_panel])
            combined = np.vstack([top_row, bottom_row])
            
            # Display combined visualization
            cv2.imshow('Eye-in-Hand Object Tracking', combined)
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            # Fallback to simple color image display
            if color_image is not None:
                color_display = color_image.copy()
                self._add_eye_in_hand_info_overlay(color_display, detected_objects, timing_info, strategy_info)
                self._add_object_detection_overlay(color_display, detected_objects)
                cv2.imshow('Eye-in-Hand Object Tracking', color_display)
    
    def _add_eye_in_hand_info_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]],
                                    timing_info: Dict[str, float], strategy_info: Dict[str, str]):
        """Add eye-in-hand specific information overlay to image."""
        tracking_params = getattr(self.config, 'tracking_params', {})
        
        # Get robot status info
        robot_status = "Disconnected"
        if self.robot_controller.is_connected:
            if self.robot_controller.async_mode_enabled:
                queue_status = self.robot_controller.get_queue_status()
                active_cmd = queue_status.get('active_command', {})
                if active_cmd and active_cmd.get('command_id'):
                    robot_status = f"Moving ({active_cmd.get('type', 'unknown')})"
                else:
                    pending_count = queue_status.get('queue_size', 0)
                    if pending_count > 0:
                        robot_status = f"Ready ({pending_count} pending)"
                    else:
                        robot_status = "Ready (async)"
            else:
                robot_status = "Connected (sync)"
        
        info_lines = [
            "EYE-IN-HAND TRACKING",
            f"Mode: {self.current_robot_mode.upper()}",
            f"Strategy: {self.current_strategy_name.upper()}",
            f"Objects: {len(detected_objects)}",
            f"FPS: {self.temporal_filter.calculate_fps():.1f}",
            f"Robot: {robot_status}",
            f"Adaptive: {'ON' if self.adaptive_timing else 'OFF'}",
        ]
        
        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(image, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def _add_object_detection_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]]):
        """Add object detection information overlay."""
        if not detected_objects:
            return
        
        y_offset = 200
        for i, obj in enumerate(detected_objects[:3]):  # Show first 3 objects
            position = self._extract_object_position(obj)
            if position:
                # Format: (forward, side, height) relative to end effector
                info_text = f"Obj{i+1}: ({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f})"
                cv2.putText(image, info_text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _handle_user_input(self) -> bool:
        """Handle user keyboard input."""
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('h'):
            self.show_help = not self.show_help
            print(f"Help {'enabled' if self.show_help else 'disabled'}")
        elif key == ord('p'):
            self.show_performance = not self.show_performance
            print(f"Performance info {'enabled' if self.show_performance else 'disabled'}")
        elif key == ord(' '):
            self.is_paused = not self.is_paused
            print(f"{'Paused' if self.is_paused else 'Resumed'}")
        elif key == ord('1'):
            self._switch_strategy('bev')
        elif key == ord('2'):
            self._switch_strategy('dbscan')
        elif key == ord('l'):
            if self.robot_controller.is_connected:
                if self.robot_controller.async_mode_enabled:
                    command_id = self.robot_controller.look_at_table_async(priority=1)
                    print(f"LOOK_AT_TABLE command submitted (ID: {command_id})")
                else:
                    print("Moving to LOOK_AT_TABLE position...")
                    self.robot_controller.look_at_table()
        elif key == ord('s'):
            if self.robot_controller.is_connected:
                if self.robot_controller.async_mode_enabled:
                    command_id = self.robot_controller.stop_movement_async(priority=3)
                    print(f"Stop movement command submitted (ID: {command_id})")
                else:
                    print("Stopping robot movement...")
                    self.robot_controller.stop_movement()
        elif key == ord('o'):
            self.current_robot_mode = "observe"
            print("Switched to OBSERVE mode - maintaining viewing distance")
        elif key == ord('a'):
            self.current_robot_mode = "approach"
            print("Switched to APPROACH mode - moving closer to objects")
        elif key == ord('m'):
            self.current_robot_mode = "manipulate"
            print("Switched to MANIPULATE mode - precise positioning")
        
        return True
    
    def _update_performance_stats(self, frame_time: float, timing_info: Dict[str, float]):
        """Update performance statistics."""
        detection_time = timing_info.get('total_time', 0)
        self.performance_stats['detection_times'].append(detection_time)
        
        # Calculate FPS using temporal filter
        self.performance_stats['avg_fps'] = self.temporal_filter.calculate_fps()
        
        # Keep only recent samples
        max_samples = 100
        if len(self.performance_stats['detection_times']) > max_samples:
            self.performance_stats['detection_times'].pop(0)
        if len(self.performance_stats['robot_command_times']) > max_samples:
            self.performance_stats['robot_command_times'].pop(0)
    
    
    def _cleanup(self):
        """Cleanup resources."""
        print("\nShutting down Eye-in-Hand Tracking System...")
        
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
        
        print("✓ Eye-in-hand shutdown complete")
    
    def _print_final_statistics(self):
        """Print final performance statistics."""
        stats = self.performance_stats
        
        print("\n" + "=" * 50)
        print("EYE-IN-HAND FINAL STATISTICS")
        print("=" * 50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Successful robot commands: {stats['successful_robot_commands']}")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        
        if stats['detection_times']:
            avg_detection = np.mean(stats['detection_times']) * 1000
            print(f"Average detection time: {avg_detection:.1f}ms")
        
        if stats['robot_command_times']:
            avg_robot = np.mean(stats['robot_command_times']) * 1000
            print(f"Average robot command time: {avg_robot:.1f}ms")
        
        success_rate = (stats['successful_robot_commands'] / max(1, stats['total_detections'])) * 100
        print(f"Robot command success rate: {success_rate:.1f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Eye-in-Hand Object Tracking Application')
    parser.add_argument('--config', default='eye_in_hand_realtime',
                       help='Configuration preset (eye_in_hand_realtime, eye_in_hand_precision)')
    parser.add_argument('--strategy', default='bev', choices=['bev', 'dbscan'],
                       help='Initial clustering strategy')
    parser.add_argument('--transform', default='eye_in_hand_transform.npy',
                       help='Eye-in-hand transformation matrix file')
    parser.add_argument('--visualize-raw-pointcloud', action='store_true', default=True,
                       help='Enable raw point cloud visualization (default: True)')
    parser.add_argument('--no-raw-pointcloud', dest='visualize_raw_pointcloud', action='store_false',
                       help='Disable raw point cloud visualization')
    
    args = parser.parse_args()
    
    print("Eye-in-Hand Object Tracking Application")
    print("=" * 40)
    print(f"Configuration: {args.config}")
    print(f"Initial strategy: {args.strategy}")
    print("Camera mounted on robot end effector")
    print("=" * 40)
    
    try:
        tracker = EyeInHandTracker(
            config_name=args.config,
            initial_strategy=args.strategy,
            transformation_matrix_file=args.transform,
            visualize_raw_pointcloud=args.visualize_raw_pointcloud
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