#!/usr/bin/env python3
"""
LiDAR-based Object Tracking Application

This application uses LiDAR point cloud data to detect and track objects without
markers, integrating with the existing hand-eye calibration matrix for accurate
robot coordination.

Key Features:
- Markerless object detection using LiDAR point clouds
- RANSAC-based floor detection and removal
- 3D DBSCAN clustering for object separation
- Distance filtering (0.25-0.8m LiDAR effective range)
- Integration with existing hand-eye calibration matrix
- Real-time visualization with multiple views
- Async robot control for smooth operation

Usage:
    python lidar_track_object.py [--config CONFIG_NAME] [--bag BAG_FILE] [--matrix MATRIX_FILE]

Requirements:
    - Existing hand-eye calibration matrix (from CharUco calibration)
    - RealSense camera for depth data
    - Robot and camera in same positions as during calibration

Author: LiDAR Integration System
"""

import argparse
import sys
import os
import time
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.sensors.realsense_manager import RealSenseManager
from module.strategies.enhanced_lidar_strategy import EnhancedLiDARStrategy
from module.processing.lidar_coordinator import LiDARCoordinator
from module.processing.position_outlier_filter import PositionOutlierFilter
from module.robot.mycobot_controller import MyCobotController
from module.core.visualization_engine import VisualizationEngine


class LiDARObjectTracker:
    """
    LiDAR-based object tracking application with markerless detection.
    
    Features:
    - Real-time LiDAR point cloud processing
    - Floor detection and object clustering
    - Coordinate transformation using existing hand-eye calibration
    - Multi-view visualization (LiDAR, Floor, Clusters, Coordinate)
    - Async robot control with safety constraints
    - Performance monitoring and statistics
    """
    
    def __init__(self, 
                 config_name: str = "lidar_tracking",
                 bag_file: Optional[str] = None,
                 transformation_matrix_file: str = "charuco_eye_in_hand_transform.npy",
                 enable_robot_control: bool = True,
                 distance_range: Optional[Tuple[float, float]] = None):
        """
        Initialize LiDAR object tracker.
        
        Args:
            config_name: Configuration preset name
            bag_file: Optional bag file for replay
            transformation_matrix_file: Path to hand-eye transformation matrix
        """
        self.config_name = config_name
        self.bag_file = bag_file
        self.transformation_matrix_file = transformation_matrix_file
        
        # Core components
        self.sensor_manager = None
        self.clustering_strategy = None
        self.coordinator = None
        self.robot_controller = None
        self.visualizer = VisualizationEngine()
        
        # Application state
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.last_detection_time = 0
        self.last_robot_command_time = 0
        
        # Configuration parameters
        self.config = self._load_configuration(config_name)
        
        # Note: Enhanced strategy uses spatial_filter_params instead of distance_range
        if distance_range is not None:
            print(f"Warning: distance_range parameter ignored. Enhanced strategy uses spatial_filter_params['z_range'].")
        
        # Performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'robot_commands_sent': 0,
            'avg_processing_time': 0.0,
            'avg_fps': 0.0,
            'detection_rate': 0.0
        }
        
        # Visualization settings
        self.show_rgb_view = True        # RGB camera feed
        self.show_lidar_view = True      # LiDAR point cloud
        self.show_floor_view = True      # Floor detection
        self.show_cluster_view = True    # Object clustering
        self.show_coordinate_view = True # Robot coordinates
        self.show_stats = True
        
        # Robot control settings
        self.enable_robot_control = enable_robot_control
        self.robot_update_interval = 1.0  # Seconds between robot commands
        self.min_confidence_threshold = 0.7
        
        # Floor redetection settings
        self.floor_redetection_interval = 30.0  # Seconds between floor redetections (less frequent)
        self.last_floor_detection_time = 0
        
        # Position printing settings
        self.enable_position_printing = True
        self.position_print_interval = 2.0  # Print every 2 seconds (less clutter)
        self.last_position_print_time = 0
        self.position_history = {
            'camera_frame': [],
            'world_frame': [], 
            'robot_frame': []
        }
        self.max_history_size = 5  # Keep last 5 positions for stability
        
        # Initialize components
        self._initialize_components()
    
    def _load_configuration(self, config_name: str) -> Dict[str, Any]:
        """Load enhanced LiDAR tracking configuration parameters."""
        
        configurations = {
            'enhanced_lidar_precise': {
                'strategy_type': 'enhanced',  # Use EnhancedLiDARStrategy
                'spatial_filter_params': {
                    'z_range': (0.2, 0.6),  # Depth filtering in meters
                    'x_range': (-0.2, 0.2), # Lateral filtering in meters
                    'y_range': None,        # No Y filtering
                    'min_points_after_filter': 100
                },
                'gravity_params': {
                    'gravity_vector': [0, 0, -1],  # Target gravity in world frame (Z-up)
                    'alignment_samples': 10,
                    'alignment_tolerance': 0.1,
                    'use_cached_rotation': True
                },
                'floor_params': {
                    'distance_threshold': 0.001,    # 0.1cm RANSAC tolerance (very strict)
                    'max_trials': 1500,
                    'min_floor_points': 300,
                    'floor_normal_tolerance': 0.2,  # Stricter horizontal tolerance
                    'min_floor_area': 0.05
                },
                'object_height_params': {
                    'min_height_above_floor': 0.01,  # 1cm minimum above floor
                    'max_height_above_floor': 0.20,  # 20cm maximum above floor
                    'height_tolerance': 0.005,
                    'min_object_points': 30
                },
                'clustering_params': {
                    'eps': 0.025,                    # 2.5cm clustering
                    'min_samples': 8,
                    'min_cluster_size': 50,
                    'max_cluster_size': 1500,
                    'outlier_removal': True
                },
                'visualization_params': {
                    'enable_all_steps': True,
                    'point_cloud_colors': {
                        'raw': (128, 128, 128),
                        'filtered': (0, 255, 0),
                        'rotated': (255, 255, 0),
                        'floor': (255, 0, 0),
                        'objects': (0, 0, 255),
                        'projected': (255, 0, 255)
                    }
                },
                'fixed_view_params': {
                    'x_range': (-0.3, 0.3),
                    'y_range': (-0.25, 0.25),
                    'z_range': (0.15, 0.65)
                }
            },
            'enhanced_lidar_demo': {
                'strategy_type': 'enhanced',  # Use EnhancedLiDARStrategy
                'spatial_filter_params': {
                    'z_range': (0.2, 0.6),  # Depth filtering in meters
                    'x_range': (-0.2, 0.2), # Lateral filtering in meters  
                    'y_range': None,        # No Y filtering
                    'min_points_after_filter': 80
                },
                'gravity_params': {
                    'gravity_vector': [0, 0, -1],  # Target gravity in world frame (Z-up)
                    'alignment_samples': 5,     # Faster calibration
                    'alignment_tolerance': 0.15, # More tolerant
                    'use_cached_rotation': True
                },
                'floor_params': {
                    'distance_threshold': 0.001,     # 0.1cm RANSAC tolerance (very strict)
                    'max_trials': 1000,
                    'min_floor_points': 200,        # Fewer points needed
                    'floor_normal_tolerance': 0.25, # More tolerance
                    'min_floor_area': 0.03
                },
                'object_height_params': {
                    'min_height_above_floor': 0.01,  # 1cm minimum above floor
                    'max_height_above_floor': 0.25,  # 25cm maximum above floor (more lenient)
                    'height_tolerance': 0.01,        # 1cm tolerance
                    'min_object_points': 25          # Fewer points needed
                },
                'clustering_params': {
                    'eps': 0.03,                     # 3cm clustering (more lenient)
                    'min_samples': 6,                # Fewer samples needed
                    'min_cluster_size': 40,          # Smaller clusters allowed
                    'max_cluster_size': 2000,        # Larger clusters allowed
                    'outlier_removal': False         # Disabled for demo
                },
                'visualization_params': {
                    'enable_all_steps': True,
                    'point_cloud_colors': {
                        'raw': (128, 128, 128),
                        'filtered': (0, 255, 0),
                        'rotated': (255, 255, 0),
                        'floor': (255, 0, 0),
                        'objects': (0, 0, 255),
                        'projected': (255, 0, 255)
                    }
                },
                'fixed_view_params': {
                    'x_range': (-0.3, 0.3),
                    'y_range': (-0.25, 0.25),
                    'z_range': (0.15, 0.65)
                }
            },
            'enhanced_lidar_debug': {
                'strategy_type': 'enhanced',  # Use EnhancedLiDARStrategy  
                'spatial_filter_params': {
                    'z_range': (0.15, 0.8),  # Wider depth range for debugging
                    'x_range': (-0.3, 0.3),  # Wider lateral range
                    'y_range': None,         # No Y filtering
                    'min_points_after_filter': 50
                },
                'gravity_params': {
                    'gravity_vector': [0, 0, -1],  # Target gravity in world frame (Z-up)
                    'alignment_samples': 15,    # More samples for accuracy
                    'alignment_tolerance': 0.2, # Very tolerant
                    'use_cached_rotation': False # Don't cache for debugging
                },
                'floor_params': {
                    'distance_threshold': 0.001,     # 0.1cm RANSAC tolerance (very strict)
                    'max_trials': 800,
                    'min_floor_points': 150,        # Fewer points needed
                    'floor_normal_tolerance': 0.4,  # Very tolerant
                    'min_floor_area': 0.02
                },
                'object_height_params': {
                    'min_height_above_floor': 0.005, # 5mm minimum (very low)
                    'max_height_above_floor': 0.30,  # 30cm maximum (high)
                    'height_tolerance': 0.015,       # 1.5cm tolerance
                    'min_object_points': 20          # Very few points needed
                },
                'clustering_params': {
                    'eps': 0.04,                     # 4cm clustering (very lenient)
                    'min_samples': 5,                # Few samples needed
                    'min_cluster_size': 25,          # Small clusters allowed
                    'max_cluster_size': 3000,        # Very large clusters allowed
                    'outlier_removal': False         # Disabled for debugging
                },
                'visualization_params': {
                    'enable_all_steps': True,
                    'point_cloud_colors': {
                        'raw': (128, 128, 128),
                        'filtered': (0, 255, 0),
                        'rotated': (255, 255, 0),
                        'floor': (255, 0, 0),
                        'objects': (0, 0, 255),
                        'projected': (255, 0, 255)
                    }
                },
                'fixed_view_params': {
                    'x_range': (-0.4, 0.4),
                    'y_range': (-0.3, 0.3),
                    'z_range': (0.1, 0.85)
                }
            }
        }
        
        if config_name not in configurations:
            print(f"Warning: Unknown config '{config_name}', using default 'enhanced_lidar_demo'")
            config_name = 'enhanced_lidar_demo'
        
        config = configurations[config_name]
        print(f"Loaded configuration: {config_name}")
        
        return config
    
    def _initialize_components(self):
        """Initialize all system components."""
        print("Initializing LiDAR object tracking system...")
        
        # Initialize sensor manager
        self.sensor_manager = RealSenseManager(
            bag_file=self.bag_file,
            realtime_mode=True,
            use_imu=True
        )
        
        # Initialize enhanced LiDAR strategy (only strategy available)
        self.clustering_strategy = EnhancedLiDARStrategy(
            spatial_filter_params=self.config.get('spatial_filter_params'),
            gravity_params=self.config.get('gravity_params'),
            floor_params=self.config.get('floor_params'),
            object_height_params=self.config.get('object_height_params'),
            clustering_params=self.config.get('clustering_params'),
            visualization_params=self.config.get('visualization_params')
        )
        
        # Set sensor manager reference for IMU access
        self.clustering_strategy.set_sensor_manager(self.sensor_manager)
        print(f"Initialized Enhanced LiDAR Strategy with 5-step pipeline")
        
        # Initialize coordinate transformation
        self.coordinator = LiDARCoordinator(
            transformation_matrix_file=self.transformation_matrix_file,
            coordinate_system="standard"
        )
        
        # Set sensor manager for coordinate transformation  
        self.coordinator.set_sensor_manager(self.sensor_manager)
        
        # Initialize position outlier filter
        outlier_config = self.config.get('outlier_filter', {})
        self.outlier_filter = PositionOutlierFilter(
            max_distance_from_base=outlier_config.get('max_distance_from_base', 500.0),
            max_velocity=outlier_config.get('max_velocity', 1000.0),
            max_position_jump=outlier_config.get('max_position_jump', 100.0),
            temporal_window_size=outlier_config.get('temporal_window_size', 10),
            enable_statistical_filtering=outlier_config.get('enable_statistical_filtering', True),
            statistical_threshold=outlier_config.get('statistical_threshold', 2.0),
            workspace_bounds=outlier_config.get('workspace_bounds', self.config.get('robot_params', {}).get('safety_bounds'))
        )
        
        # Initialize robot controller (only if robot control is enabled)
        if self.enable_robot_control:
            try:
                self.robot_controller = MyCobotController()
                if self.robot_controller.connect():
                    print("Robot controller connected successfully")
                else:
                    print("Robot controller initialized but not connected")
            except Exception as e:
                print(f"Robot controller initialization failed: {e}")
                self.robot_controller = None
                self.enable_robot_control = False
        else:
            print("Robot control disabled - skipping robot initialization")
            self.robot_controller = None
        
        print("Component initialization complete")
    
    def _init_charuco_style_transformation(self):
        """Initialize CharUco-style coordinate transformation (following charuco_predict_position.py method)."""
        try:
            # Load the same transformation matrix as CharUco
            self.T_robot_to_depth = np.load(self.transformation_matrix_file)
            print(f"Loaded T_robot_to_depth from: {self.transformation_matrix_file}")
            
            # Get camera extrinsics (same as CharUco script)
            if self.sensor_manager:
                extrinsics_data = self.sensor_manager.get_factory_extrinsics()
                if extrinsics_data is not None:
                    T_depth_to_color = extrinsics_data['transformation_matrix']
                    T_color_to_depth = np.linalg.inv(T_depth_to_color)
                    
                    # Calculate T_robot_to_color (EXACTLY same as CharUco script)
                    # T_robot_to_depth = T_robot_to_color @ T_color_to_depth
                    # T_robot_to_color = T_robot_to_depth @ inv(T_color_to_depth)  
                    self.T_robot_to_color = self.T_robot_to_depth @ np.linalg.inv(T_color_to_depth)
                    print("Calculated T_robot_to_color using CharUco method")
                    
                    self.charuco_style_transform_ready = True
                else:
                    print("Warning: Could not get camera extrinsics for CharUco-style transform")
                    self.charuco_style_transform_ready = False
            else:
                print("Warning: No sensor manager for CharUco-style transform")
                self.charuco_style_transform_ready = False
                
        except Exception as e:
            print(f"Error initializing CharUco-style transformation: {e}")
            self.charuco_style_transform_ready = False
    
    def _transform_using_charuco_method(self, object_position_depth: np.ndarray) -> Optional[np.ndarray]:
        """Transform object position using exact CharUco coordinate method."""
        if not hasattr(self, 'charuco_style_transform_ready') or not self.charuco_style_transform_ready:
            return None
        
        try:
            # Get camera extrinsics for depth->color transformation
            extrinsics_data = self.sensor_manager.get_factory_extrinsics()
            if extrinsics_data is None:
                return None
                
            T_depth_to_color = extrinsics_data['transformation_matrix']
            
            # Transform object position from depth camera to color camera coordinates
            # (CharUco works in color camera frame)
            depth_position_homo = np.append(object_position_depth, 1.0)
            color_position_homo = T_depth_to_color @ depth_position_homo
            object_position_color = color_position_homo[:3]
            
            # Create transformation matrix for object position (like rvec_tvec_to_matrix in CharUco)
            T_color_to_object = np.eye(4)
            T_color_to_object[0:3, 3] = object_position_color
            
            # Transform to robot base frame (EXACTLY same as CharUco)
            T_robot_to_object = self.T_robot_to_color @ T_color_to_object
            
            # Extract position and convert to mm (same as CharUco)
            position_meters = T_robot_to_object[:3, 3]
            position_mm = position_meters * 1000.0
            
            # STEP 7: Print the literal center point of the object from LiDAR sensor (no transformations)
            print(f"STEP 7 - Object center (LiDAR sensor): {object_position_depth} meters")
            print(f"STEP 7 - Object center (LiDAR sensor): {object_position_depth * 1000} mm")
            
            return position_mm
            
        except Exception as e:
            print(f"Error in CharUco-style transformation: {e}")
            return None

    def start_tracking(self) -> bool:
        """Start the LiDAR object tracking system."""
        if not self._start_sensor():
            return False
        
        if not self._verify_transformation_matrix():
            return False
        
        print("\nStarting LiDAR object tracking...")
        if self.enable_position_printing:
            print(f"Position printing: ENABLED (every {self.position_print_interval:.1f}s)")
        else:
            print("Position printing: DISABLED (press 'n' to enable)")
        print("Controls:")
        print("  'q' - Quit application") 
        print("  'p' - Pause/resume tracking")
        print("  'r' - Toggle robot control")
        print("  'f' - Force floor redetection")
        print("  '1' - Toggle RGB camera view")
        print("  '2' - Toggle LiDAR point cloud view")
        print("  '3' - Toggle floor detection view")
        print("  '4' - Toggle object clustering view")
        print("  '5' - Toggle robot coordinates view")
        print("  's' - Toggle statistics display")
        print("  'n' - Toggle position printing")
        print("  't' - Change position print timing (2.0s/1.0s/5.0s)")
        print("  'h' - Show/hide this help")
        print(f"Floor redetection interval: {self.floor_redetection_interval}s")
        print(f"Position printing: {'ON' if self.enable_position_printing else 'OFF'} (every {self.position_print_interval}s)")
        print()
        
        self.is_running = True
        self._run_tracking_loop()
        
        return True
    
    def _start_sensor(self) -> bool:
        """Initialize and start the sensor."""
        if not self.sensor_manager.initialize():
            print("Error: Failed to initialize RealSense sensor")
            return False
        
        if not self.sensor_manager.start():
            print("Error: Failed to start RealSense pipeline")
            return False
        
        print("RealSense sensor started successfully")
        return True
    
    def _verify_transformation_matrix(self) -> bool:
        """Verify that transformation matrix is loaded and valid."""
        info = self.coordinator.get_transformation_info()
        
        if not info['matrix_loaded']:
            print("Error: No transformation matrix loaded!")
            print("Please run calibration first (e.g., charuco_eye_in_hand.py)")
            return False
        
        print(f"Using transformation matrix: {info['matrix_source']}")
        print(f"Matrix determinant: {info['matrix_determinant']:.6f}")
        
        # Initialize CharUco-style transformation now that camera is started
        self._init_charuco_style_transformation()
        
        return True
    
    def _run_tracking_loop(self):
        """Main tracking loop with visualization and robot control."""
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while self.is_running:
                loop_start_time = time.time()
                
                # Get LiDAR-optimized frame
                # Use Z-range from spatial filter parameters
                z_range = self.config.get('spatial_filter_params', {}).get('z_range', (0.2, 0.6))
                vertices, color_image, metadata = self.sensor_manager.get_lidar_optimized_frame(
                    distance_range=z_range,
                    enable_filtering=True
                )
                
                # Check for end of stream (bag files)
                if metadata.get('end_of_stream', False):
                    print("End of bag file reached")
                    break
                
                # Skip frame if no data, but show empty visualization
                if vertices is None or len(vertices) == 0:
                    # Still show the display with status message
                    empty_vis = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(empty_vis, "LiDAR TRACKING ACTIVE", (180, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(empty_vis, "Waiting for depth data...", (200, 260), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow('LiDAR Object Tracking', empty_vis)
                    
                    # Handle keyboard input even when no data
                    key = cv2.waitKey(1) & 0xFF
                    if not self._handle_keyboard_input(key):
                        break
                    continue
                
                self.frame_count += 1
                fps_counter += 1
                
                # Process frame if not paused
                detection_result = None
                if not self.is_paused:
                    detection_result = self._process_frame(vertices, color_image)
                
                # Create visualization
                vis_image = self._create_visualization(vertices, color_image, detection_result)
                
                # Display visualization
                cv2.imshow('LiDAR Object Tracking', vis_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keyboard_input(key):
                    break
                
                # Update FPS calculation
                if time.time() - fps_start_time >= 1.0:
                    self.performance_stats['avg_fps'] = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Update performance stats
                processing_time = time.time() - loop_start_time
                self._update_performance_stats(processing_time, detection_result is not None)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in tracking loop: {e}")
        finally:
            self._cleanup()
    
    def _process_frame(self, vertices: np.ndarray, color_image: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Process a single frame for object detection and robot control."""
        current_time = time.time()
        
        # Check if it's time to force floor redetection
        if (current_time - self.last_floor_detection_time) >= self.floor_redetection_interval:
            # Reset enhanced strategy caches to force fresh detection
            if hasattr(self.clustering_strategy, 'cached_rotation_matrix'):
                self.clustering_strategy.cached_rotation_matrix = None
                self.clustering_strategy.rotation_matrix_valid = False
                if self.enable_position_printing:
                    print(f"Floor redetection triggered (every {self.floor_redetection_interval:.0f}s)")
            self.last_floor_detection_time = current_time
        
        # Perform clustering with visualization data
        clusters = self.clustering_strategy.cluster_objects(
            vertices, color_image, return_visualization=False
        )
        
        if not clusters:
            # Print occasionally when no objects detected
            if self.enable_position_printing and current_time - self.last_position_print_time > self.position_print_interval:
                print(f"No objects detected (frame {self.frame_count}) - check detection range and scene")
                self.last_position_print_time = current_time
            return None
        
        # Get the selected object (first cluster returned)
        selected_cluster = clusters[0]
        
        # Debug: Print that we found a cluster
        if self.enable_position_printing:
            print(f"DEBUG: Found cluster with {selected_cluster.get('size', 0)} points at center {selected_cluster.get('center', [0,0,0])}")
        
        # Transform to robot coordinates using CHARUCO METHOD (exact same as charuco_predict_position.py)
        
        # First try CharUco-style transformation
        if hasattr(selected_cluster, 'center') or 'center' in selected_cluster:
            object_position_depth = selected_cluster.get('center', selected_cluster.get('center'))
            charuco_robot_pos = self._transform_using_charuco_method(object_position_depth)
            
            if charuco_robot_pos is not None:
                # Create robot_coords dict in the expected format using CharUco result
                robot_coords = {
                    'robot_position': charuco_robot_pos,  # [x, y, z] in mm from CharUco method
                    'camera_position': object_position_depth,  # original camera position
                    'distance_from_base': np.linalg.norm(charuco_robot_pos),
                    'is_valid': True,
                    'transformation_method': 'charuco_exact',
                    'matrix_source': self.transformation_matrix_file
                }
                
                if self.enable_position_printing:
                    print(f"DEBUG: CharUco method → Robot: {charuco_robot_pos}")
            else:
                # Fallback to original method if CharUco method fails
                robot_coords = self.coordinator.transform_cluster_center(selected_cluster)
                if robot_coords and self.enable_position_printing:
                    print(f"DEBUG: Fallback method → Robot: {robot_coords.get('robot_position', 'None')}")
        else:
            # Fallback to original method if no center available
            robot_coords = self.coordinator.transform_cluster_center(selected_cluster)
        
        # Debug: Check robot coordinate transformation
        if self.enable_position_printing:
            if robot_coords and robot_coords['is_valid']:
                robot_pos = robot_coords['robot_position']
                print(f"DEBUG: Robot coords: [{robot_pos[0]:+6.0f}, {robot_pos[1]:+6.0f}, {robot_pos[2]:+6.0f}]mm")
            else:
                print(f"DEBUG: Robot coordinate transformation FAILED")
        
        if not robot_coords or not robot_coords['is_valid']:
            return None
        
        # TEMPORARILY DISABLE outlier filtering for debugging
        is_valid_position = True
        filter_details = {'reason': 'disabled_for_debug'}
        
        # Apply outlier filtering after coordinate transformation
        # is_valid_position, filter_details = self.outlier_filter.filter_position(robot_coords, current_time)
        
        # Debug: Check if outlier filter is blocking
        if self.enable_position_printing and not is_valid_position:
            print(f"DEBUG: Outlier filter BLOCKED position - reason: {filter_details['reason']}")
        
        if not is_valid_position:
            # Position was filtered out as outlier - provide detailed feedback
            robot_pos = robot_coords['robot_position']
            reason = filter_details['reason']
            print(f"OUTLIER FILTERED: Position ({robot_pos[0]:.0f}, {robot_pos[1]:.0f}, {robot_pos[2]:.0f}) - {reason}")
            
            # Show filter details for debugging
            if reason == 'distance_outlier':
                distance = filter_details.get('distance_from_base', 0)
                print(f"  Distance {distance:.0f}mm exceeds limit")
            elif reason == 'velocity_outlier':
                print(f"  Velocity too high between consecutive detections")
            elif reason == 'position_jump_outlier':
                print(f"  Position jump too large from previous detection")
            elif reason == 'statistical_outlier':
                print(f"  Position is statistical outlier from recent history")
            elif reason == 'workspace_outlier':
                print(f"  Position outside robot workspace bounds")
            
            # Return filtered result for visualization (but mark as invalid)
            return {
                'cluster': selected_cluster,
                'robot_coords': robot_coords,
                'detection_time': current_time,
                'filtered_as_outlier': True,
                'filter_reason': reason
            }
        
        # Add filter information to robot_coords
        robot_coords['outlier_filter'] = filter_details
        
        # Send robot command if enabled and enough time has passed (only for valid positions)
        if (is_valid_position and  # Only send robot commands for valid positions
            self.enable_robot_control and 
            self.robot_controller and 
            self.robot_controller.is_connected and
            current_time - self.last_robot_command_time > self.robot_update_interval):
            
            self._send_robot_command(robot_coords)
            self.last_robot_command_time = current_time
        
        # Print position information if enabled and enough time has passed
        if self.enable_position_printing and current_time - self.last_position_print_time > self.position_print_interval:
            if is_valid_position:
                self._print_position_information(selected_cluster, robot_coords, current_time)
                self.last_position_print_time = current_time
            else:
                # Print why position is invalid
                if detection_result and detection_result.get('filtered_as_outlier', False):
                    reason = detection_result.get('filter_reason', 'unknown')
                    print(f"Object detected but filtered as outlier: {reason}")
                else:
                    print(f"Object detected but position invalid")
                self.last_position_print_time = current_time
        
        return {
            'cluster': selected_cluster,
            'robot_coords': robot_coords,
            'detection_time': current_time
        }
    
    def _send_robot_command(self, robot_coords: Dict[str, Any]):
        """Send robot command to interact with detected object using 6DOF pose."""
        if not robot_coords.get('suggested_pose'):
            return
        
        try:
            pose = robot_coords['suggested_pose']
            target_pose = pose['position'] + pose['orientation']
            
            # Determine command type based on pose information
            has_orientation = robot_coords.get('has_orientation', False)
            approach_type = pose.get('approach_type', 'pointing')
            
            # Send async command to avoid blocking visualization
            success = self.robot_controller.send_command_async(
                command_type='move_to_position',
                parameters={'target_pose': target_pose}
            )
            
            if success:
                self.performance_stats['robot_commands_sent'] += 1
                
                if has_orientation and approach_type == '6DOF_floor_aligned':
                    # Enhanced 6DOF pose command
                    target_pos = pose['target_position']
                    target_orient = pose.get('target_orientation', [0, 0, 0])
                    print(f"Robot 6DOF command sent: Approach object at ({target_pos[0]:.0f}, {target_pos[1]:.0f}, {target_pos[2]:.0f}) "
                          f"with orientation ({target_orient[0]:.1f}°, {target_orient[1]:.1f}°, {target_orient[2]:.1f}°)")
                    
                    # Display floor alignment information
                    if robot_coords.get('floor_normal_robot') is not None:
                        floor_normal = robot_coords['floor_normal_robot']
                        print(f"  Floor normal (robot frame): [{floor_normal[0]:.3f}, {floor_normal[1]:.3f}, {floor_normal[2]:.3f}]")
                else:
                    # Standard pointing command
                    print(f"Robot pointing command sent: Point at ({pose['position'][0]:.0f}, {pose['position'][1]:.0f}, {pose['position'][2]:.0f})")
            
        except Exception as e:
            print(f"Error sending robot command: {e}")
    
    def _print_position_information(self, cluster: Dict[str, Any], robot_coords: Dict[str, Any], current_time: float):
        """Print concise robot position information."""
        try:
            # Get robot position (this is the main result)
            robot_pos = robot_coords['robot_position']  # [x, y, z] in mm
            distance = robot_coords['distance_from_base']
            
            # Simple, clean output focused on robot coordinates
            print(f"Object detected → Robot position: [{robot_pos[0]:+6.0f}, {robot_pos[1]:+6.0f}, {robot_pos[2]:+6.0f}]mm (distance: {distance:3.0f}mm)")
            
            # Update history for minimal tracking (camera frame only for simplicity)
            camera_center = cluster['center']
            self.position_history['robot_frame'].append(robot_pos.copy())
            
            # Limit history size  
            if len(self.position_history['robot_frame']) > self.max_history_size:
                self.position_history['robot_frame'].pop(0)
                
        except Exception as e:
            print(f"Error printing position: {e}")
    
    def _update_position_history(self, camera_pos: np.ndarray, world_pos: Optional[np.ndarray], robot_pos: np.ndarray):
        """Update position history for stability analysis."""
        self.position_history['camera_frame'].append(camera_pos.copy())
        self.position_history['robot_frame'].append(robot_pos.copy())
        
        if world_pos is not None:
            self.position_history['world_frame'].append(world_pos.copy())
        
        # Limit history size
        for frame_type in self.position_history:
            if len(self.position_history[frame_type]) > self.max_history_size:
                self.position_history[frame_type].pop(0)
    
    def _print_position_stability(self):
        """Print simple position stability for robot frame only."""
        robot_positions = self.position_history.get('robot_frame', [])
        
        if len(robot_positions) >= 2:
            positions_array = np.array(robot_positions)
            std_devs = np.std(positions_array, axis=0) 
            total_std = np.linalg.norm(std_devs)
            print(f"Position stability: σ={total_std:4.1f}mm (n={len(robot_positions)})")
        else:
            print("Position stability: collecting data...")
    
    def _create_visualization(self, vertices: np.ndarray, color_image: Optional[np.ndarray], 
                            detection_result: Optional[Dict[str, Any]]) -> np.ndarray:
        """Create comprehensive visualization of tracking results."""
        
        # Get clustering visualization data
        clusters = self.clustering_strategy.cluster_objects(
            vertices, color_image, return_visualization=True
        )
        
        # Enhanced strategy always returns 6-step visualization
        if isinstance(clusters, tuple) and len(clusters) == 7:
            # Enhanced strategy returns 6 visualization panels + cluster list
            cluster_list, step1_vis, step2_vis, step3_vis, step4_vis, step5_vis, step6_vis = clusters
            return self._create_enhanced_6panel_visualization(
                cluster_list, detection_result, color_image,
                [step1_vis, step2_vis, step3_vis, step4_vis, step5_vis, step6_vis]
            )
        else:
            # Fallback: if enhanced strategy doesn't return visualization data
            cluster_list = clusters if isinstance(clusters, list) else []
            # Create empty 6-panel layout
            empty_panels = []
            for i in range(6):
                panel = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(panel, f"Step {i+1}: No data", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                empty_panels.append(panel)
            return self._create_enhanced_6panel_visualization(
                cluster_list, detection_result, color_image, empty_panels
            )
        
        # Create coordinate transformation view
        coord_view = self._create_coordinate_view(detection_result)
        
        # Create RGB camera view
        rgb_view = self._create_rgb_view(color_image, detection_result)
        
        # Create layout based on enabled views
        views_to_show = []
        view_labels = []
        
        if self.show_rgb_view:
            views_to_show.append(rgb_view)
            view_labels.append("RGB Camera")
        
        if self.show_lidar_view:
            views_to_show.append(lidar_view)
            view_labels.append("LiDAR View")
        
        if self.show_floor_view:
            views_to_show.append(floor_view)
            view_labels.append("Floor Detection")
        
        if self.show_cluster_view:
            views_to_show.append(cluster_view)
            view_labels.append("Object Clustering")
        
        if self.show_coordinate_view:
            views_to_show.append(coord_view)
            view_labels.append("Robot Coordinates")
        
        # Ensure we have valid views to show
        if not views_to_show:
            # Create fallback view if no valid visualization data
            combined_view = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(combined_view, "LiDAR SYSTEM ACTIVE", (200, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(combined_view, "No visualization data", (220, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        elif len(views_to_show) == 1:
            combined_view = views_to_show[0].copy()
        elif len(views_to_show) == 2:
            # Side by side
            combined_view = np.hstack(views_to_show)
        else:
            # Multi-view grid layout
            if len(views_to_show) == 3:
                # Add empty view for 2x2 grid
                empty_view = np.zeros_like(views_to_show[0])
                cv2.putText(empty_view, "Reserved", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                views_to_show.append(empty_view)
                view_labels.append("Reserved")
            elif len(views_to_show) == 5:
                # Add empty view for 3x2 grid (6 total positions)
                empty_view = np.zeros_like(views_to_show[0])
                cv2.putText(empty_view, "Reserved", (200, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                views_to_show.append(empty_view)
                view_labels.append("Reserved")
            
            # Make sure all views have the same size
            target_height = views_to_show[0].shape[0]
            target_width = views_to_show[0].shape[1]
            for i, view in enumerate(views_to_show):
                if view.shape[:2] != (target_height, target_width):
                    views_to_show[i] = cv2.resize(view, (target_width, target_height))
            
            if len(views_to_show) <= 4:
                # 2x2 grid for up to 4 views
                top_row = np.hstack(views_to_show[:2])
                bottom_row = np.hstack(views_to_show[2:4])
                combined_view = np.vstack([top_row, bottom_row])
            else:
                # 3x2 grid for 5-6 views
                top_row = np.hstack(views_to_show[:3])
                bottom_row = np.hstack(views_to_show[3:6])
                combined_view = np.vstack([top_row, bottom_row])
        
        # Add labels to views
        if len(views_to_show) > 1:
            label_y_positions = []
            label_x_positions = []
            
            if len(views_to_show) <= 2:
                # Side by side
                label_y_positions = [30, 30]
                label_x_positions = [10, combined_view.shape[1]//2 + 10]
            elif len(views_to_show) <= 4:
                # 2x2 grid
                label_y_positions = [30, 30, combined_view.shape[0]//2 + 30, combined_view.shape[0]//2 + 30]
                label_x_positions = [10, combined_view.shape[1]//2 + 10, 10, combined_view.shape[1]//2 + 10]
            else:
                # 3x2 grid (for 5-6 views)
                third_width = combined_view.shape[1]//3
                half_height = combined_view.shape[0]//2
                label_y_positions = [30, 30, 30, half_height + 30, half_height + 30, half_height + 30]
                label_x_positions = [10, third_width + 10, third_width*2 + 10, 10, third_width + 10, third_width*2 + 10]
            
            for i, (label, x_pos, y_pos) in enumerate(zip(view_labels, label_x_positions, label_y_positions)):
                if i < len(label_x_positions) and i < len(label_y_positions) and label:  # Skip empty labels and bounds check
                    cv2.putText(combined_view, label, (x_pos, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add statistics overlay
        if self.show_stats:
            combined_view = self._add_statistics_overlay(combined_view, detection_result)
        
        return combined_view
    
    def _create_enhanced_6panel_visualization(self, cluster_list: List[Dict[str, Any]], 
                                            detection_result: Optional[Dict[str, Any]],
                                            color_image: Optional[np.ndarray],
                                            step_visualizations: List[np.ndarray]) -> np.ndarray:
        """
        Create enhanced 6-panel visualization for the full pipeline.
        
        Layout:
        [RGB Camera]  [Step 2: Filtered]  [Step 3: Aligned]
        [Step 4: Floor] [Step 5: Objects] [Step 6: Final]
        """
        # Ensure we have 6 step visualizations
        if len(step_visualizations) < 6:
            # Fill missing panels with empty ones
            while len(step_visualizations) < 6:
                empty_vis = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(empty_vis, "No Data", (280, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                step_visualizations.append(empty_vis)
        
        # Create RGB camera view with enhanced overlays
        rgb_view = self._create_enhanced_rgb_view(color_image, detection_result)
        
        # Resize all panels to consistent size (320x240 for 3x2 layout)
        target_size = (480, 360)
        
        # Assemble the panels in the new order
        rgb_panel = cv2.resize(rgb_view, target_size)
        step2_panel = cv2.resize(step_visualizations[1], target_size) # Index 1 is Step 2
        step3_panel = cv2.resize(step_visualizations[2], target_size) # Index 2 is Step 3
        step4_panel = cv2.resize(step_visualizations[3], target_size) # Index 3 is Step 4
        step5_panel = cv2.resize(step_visualizations[4], target_size) # Index 4 is Step 5
        step6_panel = cv2.resize(step_visualizations[5], target_size) # Index 5 is Step 6

        top_row = np.hstack([rgb_panel, step2_panel, step3_panel])
        bottom_row = np.hstack([step4_panel, step5_panel, step6_panel])
        combined_view = np.vstack([top_row, bottom_row])
        
        # Add new panel titles
        panel_titles = [
            "RGB Camera + Tracking",
            "Step 2: Spatial Filter",
            "Step 3: Gravity Aligned",
            "Step 4: Floor Detection", 
            "Step 5: Filtered Objects",
            "Step 6: Final Objects"
        ]
        
        # Add titles to each panel
        third_width = combined_view.shape[1] // 3
        half_height = combined_view.shape[0] // 2
        
        title_positions = [
            (10, 25),                           # Top-left
            (third_width + 10, 25),             # Top-center
            (third_width * 2 + 10, 25),         # Top-right
            (10, half_height + 25),             # Bottom-left
            (third_width + 10, half_height + 25), # Bottom-center
            (third_width * 2 + 10, half_height + 25) # Bottom-right
        ]
        
        for title, (x, y) in zip(panel_titles, title_positions):
            cv2.putText(combined_view, title, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add comprehensive statistics overlay
        if self.show_stats:
            combined_view = self._add_enhanced_statistics_overlay(combined_view, detection_result, cluster_list)
        
        return combined_view
    
    def _create_enhanced_rgb_view(self, color_image: Optional[np.ndarray], 
                                detection_result: Optional[Dict[str, Any]]) -> np.ndarray:
        """Create enhanced RGB camera view with comprehensive LiDAR tracking overlays."""
        if color_image is None:
            rgb_view = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(rgb_view, "No RGB Camera Data", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            return rgb_view
        
        # Copy and resize color image
        rgb_view = color_image.copy()
        if rgb_view.shape[:2] != (480, 640):
            rgb_view = cv2.resize(rgb_view, (640, 480))
        
        # Add enhanced tracking information
        cv2.putText(rgb_view, "5-STEP LiDAR PIPELINE", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add detection information if available
        if detection_result and 'robot_coords' in detection_result:
            robot_coords = detection_result['robot_coords']
            
            # Position information
            if 'robot_position' in robot_coords:
                pos = robot_coords['robot_position'] 
                cv2.putText(rgb_view, f"Target: ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})mm", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Distance information
            if 'distance_from_base' in robot_coords:
                distance = robot_coords['distance_from_base']
                cv2.putText(rgb_view, f"Distance: {distance:.0f}mm from base", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Pipeline-specific information
            if 'pipeline_metadata' in robot_coords:
                metadata = robot_coords['pipeline_metadata']
                y_offset = 120
                
                if metadata.get('gravity_aligned', False):
                    cv2.putText(rgb_view, "Gravity-aligned detection", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 25
                
                if metadata.get('floor_normal') is not None:
                    floor_normal = metadata['floor_normal']
                    cv2.putText(rgb_view, f"Floor normal: [{floor_normal[0]:.2f}, {floor_normal[1]:.2f}, {floor_normal[2]:.2f}]", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    y_offset += 20
        else:
            cv2.putText(rgb_view, "No objects detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add crosshair at center (where sensor is pointing)
        center_x, center_y = rgb_view.shape[1] // 2, rgb_view.shape[0] // 2
        cv2.line(rgb_view, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(rgb_view, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
        cv2.circle(rgb_view, (center_x, center_y), 5, (0, 255, 0), -1)
        
        # Add spatial filter ranges
        cv2.putText(rgb_view, "Z: 0.2-0.6m, X: -0.2-0.2m", (10, rgb_view.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return rgb_view
    
    def _add_enhanced_statistics_overlay(self, image: np.ndarray, 
                                       detection_result: Optional[Dict[str, Any]], 
                                       cluster_list: List[Dict[str, Any]]) -> np.ndarray:
        """Add enhanced statistics overlay for 6-panel visualization."""
        overlay = image.copy()
        
        # Semi-transparent background for statistics
        cv2.rectangle(overlay, (10, image.shape[0] - 120), (500, image.shape[0] - 10), (0, 0, 0), -1)
        cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
        
        # Performance statistics
        stats = self.performance_stats
        y_offset = image.shape[0] - 100
        
        cv2.putText(image, f"Frame: {self.frame_count} | FPS: {stats['avg_fps']:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        # Pipeline-specific statistics
        if hasattr(self.clustering_strategy, 'get_performance_stats'):
            pipeline_stats = self.clustering_strategy.get_performance_stats()
            
            cv2.putText(image, f"Pipeline Success: {pipeline_stats.get('success_rate', 0):.1%} | Objects: {len(cluster_list)}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            # Step timing information
            step_times = pipeline_stats.get('pipeline_step_times', {})
            total_time = sum(step_times.values()) * 1000  # Convert to ms
            cv2.putText(image, f"Pipeline Time: {total_time:.1f}ms | Robot Cmds: {stats['robot_commands_sent']}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Current detection status
        if detection_result:
            # Show current detection information on right side
            cv2.putText(image, "Detection: ACTIVE", (image.shape[1] - 150, y_offset - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(image, "Detection: NONE", (image.shape[1] - 150, y_offset - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # System status
        status_x = image.shape[1] - 150
        if self.is_paused:
            cv2.putText(image, "PAUSED", (status_x, y_offset - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        if not self.enable_robot_control:
            cv2.putText(image, "ROBOT DISABLED", (status_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return image
    
    def _create_rgb_view(self, color_image: Optional[np.ndarray], detection_result: Optional[Dict[str, Any]]) -> np.ndarray:
        """Create RGB camera view with tracking overlays."""
        if color_image is None:
            # Create empty RGB view
            rgb_view = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(rgb_view, "No RGB Camera Data", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            return rgb_view
        
        # Copy color image and resize to standard size
        rgb_view = color_image.copy()
        if rgb_view.shape[:2] != (480, 640):
            rgb_view = cv2.resize(rgb_view, (640, 480))
        
        # Add tracking overlays
        self._add_rgb_tracking_overlay(rgb_view, detection_result)
        
        return rgb_view
    
    def _add_rgb_tracking_overlay(self, image: np.ndarray, detection_result: Optional[Dict[str, Any]]):
        """Add tracking information overlay to RGB image."""
        # Add title
        cv2.putText(image, "RGB CAMERA + LiDAR TRACKING", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add detection info
        if detection_result:
            robot_coords = detection_result['robot_coords']
            robot_pos = robot_coords['robot_position']
            
            cv2.putText(image, f"Target: ({robot_pos[0]:.0f}, {robot_pos[1]:.0f}, {robot_pos[2]:.0f})mm", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            distance = robot_coords['distance_from_base']
            cv2.putText(image, f"Distance: {distance:.0f}mm from base", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            
            # Add orientation information if available
            y_offset = 120
            if robot_coords.get('has_orientation', False):
                robot_orient = robot_coords.get('robot_orientation_degrees', [0, 0, 0])
                cv2.putText(image, f"Orientation: ({robot_orient[0]:.1f}°, {robot_orient[1]:.1f}°, {robot_orient[2]:.1f}°)", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 30
                
                # Floor information
                if robot_coords.get('floor_normal_robot') is not None:
                    cv2.putText(image, "Floor-aligned pose detected", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    y_offset += 25
            else:
                cv2.putText(image, "Position-only tracking", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 30
            
            # Add crosshair at center (where LiDAR is looking)
            center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
            cv2.line(image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
            cv2.line(image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
            cv2.circle(image, (center_x, center_y), 5, (0, 255, 0), -1)
        else:
            cv2.putText(image, "No LiDAR objects detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add LiDAR detection range indicator
        cv2.putText(image, "LiDAR Range: 0.25-0.8m", (10, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _create_coordinate_view(self, detection_result: Optional[Dict[str, Any]]) -> np.ndarray:
        """Create coordinate transformation visualization with outlier filter information."""
        if detection_result is None:
            # Create empty coordinate view with outlier filter stats
            coord_view = self.coordinator.create_transformation_visualization([])
            
            # Add outlier filter information
            filter_stats = self.outlier_filter.get_filter_statistics()
            y_offset = coord_view.shape[0] - 120
            
            cv2.putText(coord_view, "Outlier Filter Status:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            cv2.putText(coord_view, f"Positions processed: {filter_stats['total_positions']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 15
            
            cv2.putText(coord_view, f"Success rate: {filter_stats['success_rate']:.1%}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_offset += 15
            
            cv2.putText(coord_view, f"Outliers removed: {filter_stats['total_outliers_removed']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
            
            return coord_view
        
        transformations = [detection_result['robot_coords']]
        coord_view = self.coordinator.create_transformation_visualization(transformations)
        
        # Add outlier filter information overlay
        filter_info = detection_result['robot_coords'].get('outlier_filter', {})
        filter_stats = self.outlier_filter.get_filter_statistics()
        
        # Add semi-transparent background for filter info
        overlay = coord_view.copy()
        cv2.rectangle(overlay, (10, coord_view.shape[0] - 140), (300, coord_view.shape[0] - 10), (0, 0, 0), -1)
        cv2.addWeighted(coord_view, 0.8, overlay, 0.2, 0, coord_view)
        
        y_offset = coord_view.shape[0] - 130
        
        # Current position filter status
        cv2.putText(coord_view, "Outlier Filter:", (15, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += 18
        
        # Check if this detection was filtered as outlier
        is_filtered = detection_result and detection_result.get('filtered_as_outlier', False)
        
        if is_filtered:
            reason = detection_result.get('filter_reason', 'unknown')
            cv2.putText(coord_view, f"OUTLIER: {reason.replace('_', ' ').upper()}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)
        elif filter_info.get('reason') == 'valid':
            cv2.putText(coord_view, "Position VALID", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            reason = filter_info.get('reason', 'unknown')
            cv2.putText(coord_view, f"Filtered: {reason}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        y_offset += 15
        
        # Filter statistics
        cv2.putText(coord_view, f"Total: {filter_stats['total_positions']} | Success: {filter_stats['success_rate']:.1%}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y_offset += 12
        
        # Outlier breakdown
        cv2.putText(coord_view, f"Distance: {filter_stats['distance_outliers']} | Velocity: {filter_stats['velocity_outliers']}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 100), 1)
        y_offset += 12
        
        cv2.putText(coord_view, f"Jump: {filter_stats['position_jump_outliers']} | Statistical: {filter_stats['statistical_outliers']}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 100, 100), 1)
        y_offset += 12
        
        # Position confidence if available
        confidence = filter_stats.get('position_confidence', 0.0)
        confidence_color = (0, 255, 0) if confidence > 0.8 else (255, 255, 0) if confidence > 0.5 else (255, 0, 0)
        cv2.putText(coord_view, f"Confidence: {confidence:.2f}", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.35, confidence_color, 1)
        
        return coord_view
    
    def _add_statistics_overlay(self, image: np.ndarray, detection_result: Optional[Dict[str, Any]]) -> np.ndarray:
        """Add performance statistics overlay to visualization including outlier filter stats."""
        overlay = image.copy()
        
        # Semi-transparent background for text (expanded for outlier filter info)
        cv2.rectangle(overlay, (10, image.shape[0] - 150), (450, image.shape[0] - 10), (0, 0, 0), -1)
        cv2.addWeighted(image, 0.7, overlay, 0.3, 0, image)
        
        # Statistics text
        stats = self.performance_stats
        y_offset = image.shape[0] - 130
        
        cv2.putText(image, f"Frame: {self.frame_count}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(image, f"FPS: {stats['avg_fps']:.1f}", (20, y_offset + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        success_rate = (stats['successful_detections'] / max(stats['total_frames'], 1)) * 100
        cv2.putText(image, f"Detection Rate: {success_rate:.1f}%", (20, y_offset + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(image, f"Robot Commands: {stats['robot_commands_sent']}", (20, y_offset + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Outlier filter statistics
        filter_stats = self.outlier_filter.get_filter_statistics()
        cv2.putText(image, f"Filter Success: {filter_stats['success_rate']:.1%}", (20, y_offset + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 100), 1)
        
        cv2.putText(image, f"Outliers Removed: {filter_stats['total_outliers_removed']}", (20, y_offset + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
        
        # Floor redetection timing
        time_since_floor_detection = time.time() - self.last_floor_detection_time
        time_until_next = max(0, self.floor_redetection_interval - time_since_floor_detection)
        cv2.putText(image, f"Next Floor Redetect: {time_until_next:.1f}s", (20, y_offset + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 255), 1)
        
        # Current detection info (moved to right side)
        if detection_result:
            robot_pos = detection_result['robot_coords']['robot_position']
            
            # Use different colors based on whether position was filtered
            is_filtered = detection_result.get('filtered_as_outlier', False)
            position_color = (255, 100, 100) if is_filtered else (0, 255, 0)  # Red if filtered, green if valid
            
            cv2.putText(image, f"Target: ({robot_pos[0]:.0f}, {robot_pos[1]:.0f}, {robot_pos[2]:.0f})", 
                       (250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, position_color, 1)
            
            distance = detection_result['robot_coords']['distance_from_base']
            cv2.putText(image, f"Distance: {distance:.0f}mm", (250, y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, position_color, 1)
            
            # Current outlier filter status
            if is_filtered:
                reason = detection_result.get('filter_reason', 'unknown')
                cv2.putText(image, f"FILTERED: {reason.replace('_', ' ').upper()}", (250, y_offset + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            else:
                filter_info = detection_result['robot_coords'].get('outlier_filter', {})
                if filter_info.get('reason') == 'valid':
                    cv2.putText(image, "Position: VALID", (250, y_offset + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                else:
                    reason = filter_info.get('reason', 'filtered')
                    cv2.putText(image, f"Position: {reason.upper()}", (250, y_offset + 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Position confidence
            confidence = filter_stats.get('position_confidence', 0.0)
            confidence_color = (0, 255, 0) if confidence > 0.8 else (255, 255, 0) if confidence > 0.5 else (255, 0, 0)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (250, y_offset + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, confidence_color, 1)
        
        # Status indicators (moved down)
        status_x = 250
        status_y_offset = y_offset + 80
        
        if self.is_paused:
            cv2.putText(image, "PAUSED", (status_x, status_y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            status_y_offset += 15
        
        if not self.enable_robot_control:
            cv2.putText(image, "ROBOT DISABLED", (status_x, status_y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return image
    
    def _handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input for controlling the application."""
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        
        elif key == ord('p'):  # Pause/resume
            self.is_paused = not self.is_paused
            print(f"{'Paused' if self.is_paused else 'Resumed'} tracking")
        
        elif key == ord('r'):  # Toggle robot control
            if self.robot_controller:
                self.enable_robot_control = not self.enable_robot_control
                print(f"Robot control {'enabled' if self.enable_robot_control else 'disabled'}")
            else:
                print("Robot controller not initialized - cannot toggle robot control")
        
        elif key == ord('1'):  # Toggle RGB view
            self.show_rgb_view = not self.show_rgb_view
            print(f"RGB camera view {'enabled' if self.show_rgb_view else 'disabled'}")
        
        elif key == ord('2'):  # Toggle LiDAR view
            self.show_lidar_view = not self.show_lidar_view
            print(f"LiDAR view {'enabled' if self.show_lidar_view else 'disabled'}")
        
        elif key == ord('3'):  # Toggle floor detection view
            self.show_floor_view = not self.show_floor_view
            print(f"Floor detection view {'enabled' if self.show_floor_view else 'disabled'}")
        
        elif key == ord('4'):  # Toggle cluster view
            self.show_cluster_view = not self.show_cluster_view
            print(f"Cluster view {'enabled' if self.show_cluster_view else 'disabled'}")
        
        elif key == ord('5'):  # Toggle coordinate view
            self.show_coordinate_view = not self.show_coordinate_view
            print(f"Coordinate view {'enabled' if self.show_coordinate_view else 'disabled'}")
        
        elif key == ord('s'):  # Toggle statistics
            self.show_stats = not self.show_stats
            print(f"Statistics {'enabled' if self.show_stats else 'disabled'}")
        
        elif key == ord('f'):  # Force floor redetection
            print("[MANUAL FLOOR REDETECTION] Forcing immediate floor redetection...")
            # Reset enhanced strategy caches for fresh detection
            if hasattr(self.clustering_strategy, 'cached_rotation_matrix'):
                self.clustering_strategy.cached_rotation_matrix = None
                self.clustering_strategy.rotation_matrix_valid = False
                print("[MANUAL FLOOR REDETECTION] Cleared gravity rotation cache")
            self.last_floor_detection_time = time.time()
        
        elif key == ord('h'):  # Show help
            self._show_help()
        
        elif key == ord('n'):  # Toggle position printing
            self.enable_position_printing = not self.enable_position_printing
            print(f"Position printing {'enabled' if self.enable_position_printing else 'disabled'}")
        
        elif key == ord('t'):  # Change position print timing
            if self.position_print_interval == 2.0:
                self.position_print_interval = 1.0
                print("Position print interval: 1.0 seconds (fast)")
            elif self.position_print_interval == 1.0:
                self.position_print_interval = 5.0
                print("Position print interval: 5.0 seconds (slow)")
            else:
                self.position_print_interval = 2.0
                print("Position print interval: 2.0 seconds (default)")
        
        return True
    
    def _show_help(self):
        """Display help information."""
        print("\n" + "="*50)
        print("LiDAR OBJECT TRACKING - CONTROLS")
        print("="*50)
        print("  'q' / ESC - Quit application")
        print("  'p'       - Pause/resume tracking")
        print("  'r'       - Toggle robot control")
        print("  'f'       - Force floor redetection")
        print("  '1'       - Toggle RGB camera view")
        print("  '2'       - Toggle LiDAR point cloud view")
        print("  '3'       - Toggle floor detection view")
        print("  '4'       - Toggle object clustering view")
        print("  '5'       - Toggle robot coordinates view")
        print("  's'       - Toggle statistics display")
        print("  'n'       - Toggle position printing")
        print("  't'       - Change position print timing")
        print("  'h'       - Show this help")
        print("="*50)
        print(f"Config: {self.config_name}")
        z_range = self.config.get('spatial_filter_params', {}).get('z_range', 'N/A')
        x_range = self.config.get('spatial_filter_params', {}).get('x_range', 'N/A')
        print(f"Spatial Filter - Z: {z_range}, X: {x_range}")
        print(f"Floor Redetection Interval: {self.floor_redetection_interval}s")
        print(f"Position Printing: {'ON' if self.enable_position_printing else 'OFF'} (every {self.position_print_interval}s)")
        print(f"Matrix: {self.coordinator.matrix_source}")
        print("="*50 + "\n")
    
    def _update_performance_stats(self, processing_time: float, detection_success: bool):
        """Update performance statistics."""
        self.performance_stats['total_frames'] += 1
        
        if detection_success:
            self.performance_stats['successful_detections'] += 1
        
        # Update running average for processing time
        total = self.performance_stats['total_frames']
        prev_avg = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (
            (prev_avg * (total - 1) + processing_time) / total
        )
    
    def _cleanup(self):
        """Clean up resources on exit."""
        print("\nCleaning up...")
        
        if self.sensor_manager:
            self.sensor_manager.stop()
        
        if self.robot_controller:
            self.robot_controller.disconnect()
            print("Robot disconnected")
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        self._print_final_statistics()
        
        print("LiDAR object tracking stopped")
    
    def _print_final_statistics(self):
        """Print final performance statistics."""
        stats = self.performance_stats
        
        print("\n" + "="*50)
        print("FINAL STATISTICS")
        print("="*50)
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Successful detections: {stats['successful_detections']}")
        print(f"Detection rate: {(stats['successful_detections']/max(stats['total_frames'], 1)*100):.1f}%")
        print(f"Robot commands sent: {stats['robot_commands_sent']}")
        print(f"Average processing time: {stats['avg_processing_time']*1000:.1f}ms")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        
        # Component statistics
        clustering_stats = self.clustering_strategy.get_performance_stats()
        print(f"\nClustering success rate: {clustering_stats['success_rate']:.1%}")
        print(f"Average clusters found: {clustering_stats['avg_clusters_found']:.1f}")
        
        coord_stats = self.coordinator.get_transformation_info()['statistics']
        coord_success = coord_stats['successful_transformations'] / max(coord_stats['total_transformations'], 1)
        print(f"Coordinate transformation success rate: {coord_success:.1%}")
        
        print("="*50)


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="LiDAR-based Object Tracking")
    
    parser.add_argument('--config', type=str, default='enhanced_lidar_demo',
                       choices=['enhanced_lidar_precise', 'enhanced_lidar_demo', 'enhanced_lidar_debug'],
                       help='Enhanced LiDAR configuration preset (default: enhanced_lidar_demo)')
    
    parser.add_argument('--bag', type=str, default=None,
                       help='RealSense bag file for replay (optional)')
    
    parser.add_argument('--matrix', type=str, default='charuco_eye_in_hand_transform.npy',
                       help='Hand-eye transformation matrix file (default: charuco_eye_in_hand_transform.npy)')
    
    parser.add_argument('--no-robot', action='store_true',
                       help='Disable robot control (detection only)')
    
    parser.add_argument('--min-distance', type=float, default=0.25,
                       help='Minimum detection distance in meters (default: 0.25)')
    
    parser.add_argument('--max-distance', type=float, default=0.8,
                       help='Maximum detection distance in meters (default: 0.8)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LiDAR OBJECT TRACKING SYSTEM")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Transformation matrix: {args.matrix}")
    
    if args.bag:
        print(f"Bag file: {args.bag}")
    else:
        print("Live camera mode")
    
    if args.no_robot:
        print("Robot control: DISABLED")
    
    print("="*60)
    
    # Create and start tracker
    tracker = LiDARObjectTracker(
        config_name=args.config,
        bag_file=args.bag,
        transformation_matrix_file=args.matrix,
        enable_robot_control=not args.no_robot,  # Disable robot control if --no-robot flag is used
        distance_range=(args.min_distance, args.max_distance)
    )
    
    try:
        if not tracker.start_tracking():
            print("Failed to start tracking system")
            return 1
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())