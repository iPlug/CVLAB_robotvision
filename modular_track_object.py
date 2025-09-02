#!/usr/bin/env python3
"""
Modular Real-time Object Tracking Application

This application uses the modular detection system to track objects detected
by the camera and command the robot to point at them. It provides configurable
strategies, enhanced safety features, and performance optimization.

Usage:
    python modular_track_object.py [--config CONFIG_NAME] [--strategy STRATEGY] [--bag BAG_FILE]

Requirements:
    - Run calibrate_camera.py first to generate camera_robot_transform.npy
    - Robot and camera must be in the same positions as during calibration

New Features compared to track_object.py:
    - Modular architecture with pluggable strategies
    - Runtime strategy switching (BEV ↔ DBSCAN ↔ LiDAR)
    - LiDAR-based markerless object detection with floor removal
    - Configuration presets for different scenarios
    - Enhanced safety constraints and workspace validation
    - Performance monitoring and adaptive timing
    - Better error handling and recovery
"""

import argparse
import sys
import os
import time
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.core.object_detector import ObjectDetector
from module.core.visualization_engine import VisualizationEngine
from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
from module.strategies.dbscan_clustering_strategy import DBSCANClusteringStrategy
from module.strategies.lidar_clustering_strategy import LiDARClusteringStrategy
from module.sensors.realsense_manager import RealSenseManager
from module.processing.temporal_filter import TemporalFilter
from module.processing.lidar_coordinator import LiDARCoordinator
from module.config.config_factory import ConfigFactory
from module.robot.mycobot_controller import MyCobotController


class ModularObjectTracker:
    """
    Modular object tracking application with pluggable strategies and enhanced features.
    
    Features:
    - Configurable detection strategies (BEV, DBSCAN, LiDAR)
    - Runtime strategy switching with markerless LiDAR detection
    - Floor detection and removal for clean object isolation
    - Enhanced safety constraints
    - Performance monitoring
    - Adaptive update intervals
    - Comprehensive error handling
    """
    
    def __init__(self, config_name: str = "realtime_tracking", 
                 initial_strategy: str = "bev",
                 bag_file: Optional[str] = None,
                 transformation_matrix_file: str = "charuco_eye_in_hand_transform.npy"):
        """
        Initialize modular object tracker.
        
        Args:
            config_name: Configuration preset name
            initial_strategy: Initial clustering strategy ('bev' or 'dbscan')
            bag_file: Optional bag file for replay
            transformation_matrix_file: Path to transformation matrix
        """
        self.config_name = config_name
        self.bag_file = bag_file
        self.transformation_matrix_file = transformation_matrix_file
        
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
        self.last_update_time = 0
        self.last_command_had_object = True
        
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
    
    def _load_configuration(self, config_name: str):
        """Load configuration from factory or create custom tracking configs."""
        
        # Define custom tracking configurations
        tracking_configs = {
            'realtime_tracking': {
                'type': 'bev',
                'config': {
                    'clustering_params': {
                        'min_cluster_size': 20,
                        'max_cluster_size': 100,
                        'morphology_kernel_size': 5,
                        'window_size': 5
                    },
                    'tracking_params': {
                        'update_interval': 1.0,
                        'fast_update_interval': 0.5,
                        'adaptive_timing': True,
                        'max_no_object_time': 5.0
                    },
                    'robot_params': {
                        'min_height': 100,
                        'max_speed': 90,
                        'pointing_angles': [0, 180, 0]
                    }
                }
            },
            'precision_tracking': {
                'type': 'bev',
                'config': {
                    'clustering_params': {
                        'min_cluster_size': 15,
                        'max_cluster_size': 200,
                        'morphology_kernel_size': 5,
                        'window_size': 7
                    },
                    'tracking_params': {
                        'update_interval': 2.0,
                        'fast_update_interval': 1.0,
                        'adaptive_timing': True,
                        'max_no_object_time': 10.0
                    },
                    'robot_params': {
                        'min_height': 120,
                        'max_speed': 50,
                        'pointing_angles': [0, 180, 0]
                    }
                }
            },
            'debug_tracking': {
                'type': 'bev',
                'config': {
                    'clustering_params': {
                        'min_cluster_size': 10,
                        'max_cluster_size': 300,
                        'morphology_kernel_size': 3,
                        'window_size': 5
                    },
                    'tracking_params': {
                        'update_interval': 3.0,
                        'fast_update_interval': 1.5,
                        'adaptive_timing': False,
                        'max_no_object_time': 15.0
                    },
                    'robot_params': {
                        'min_height': 150,
                        'max_speed': 30,
                        'pointing_angles': [0, 180, 0]
                    },
                    'debug_mode': True,
                    'verbose_logging': True
                }
            }
        }
        
        if config_name in tracking_configs:
            preset = tracking_configs[config_name]
            return ConfigFactory.create_config(preset['type'], preset['config'])
        else:
            # Try to load from existing presets
            try:
                return ConfigFactory.create_preset(config_name)
            except:
                print(f"Unknown config '{config_name}', using default realtime_tracking")
                preset = tracking_configs['realtime_tracking']
                return ConfigFactory.create_config(preset['type'], preset['config'])
    
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful
        """
        print("Initializing Modular Object Tracking System")
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
        
        # Calibrate gravity alignment if using IMU
        if self.sensor_manager.use_imu:
            print("Calibrating gravity alignment...")
            self.sensor_manager.calibrate_gravity_alignment()
        
        # Initialize robot controller
        print("Initializing robot controller...")
        self.robot_controller = MyCobotController(self.transformation_matrix_file)
        
        # Load transformation matrix
        if self.robot_controller.transformation_matrix is None:
            print("Warning: No transformation matrix loaded. Robot commands will be disabled.")
        
        # Connect to robot
        if not self.robot_controller.connect():
            print("Warning: Failed to connect to robot. Running in visualization-only mode.")
        
        # Configure robot safety parameters
        if self.robot_controller.is_connected:
            robot_params = self.config.get('robot_params', {})
            self.robot_controller.safety_constraints.update(robot_params)
            print(f"Robot safety parameters: {robot_params}")
            
            # Enable async mode for non-blocking robot commands
            if self.robot_controller.enable_async_mode(max_queue_size=5):
                print("✓ Robot async mode enabled for smooth visualization")
            else:
                print("⚠ Failed to enable robot async mode - commands may block visualization")
        
        # Initialize detection strategies
        print("Initializing detection strategies...")
        self._initialize_strategies()
        
        # Set initial strategy
        if not self._switch_strategy(self.current_strategy_name):
            print(f"Error: Failed to initialize strategy '{self.current_strategy_name}'")
            return False
        
        print("✓ All systems initialized successfully")
        return True
    
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
            print("✓ BEV clustering strategy initialized")
        except Exception as e:
            print(f"✗ Failed to initialize BEV strategy: {e}")
        
        # DBSCAN Clustering Strategy
        try:
            dbscan_strategy = DBSCANClusteringStrategy(
                eps=0.02,
                min_points=10
            )
            self.available_strategies['dbscan'] = dbscan_strategy
            print("✓ DBSCAN clustering strategy initialized")
        except Exception as e:
            print(f"✗ Failed to initialize DBSCAN strategy: {e}")
        
        # LiDAR Clustering Strategy
        try:
            lidar_strategy = LiDARClusteringStrategy(
                distance_range=(0.25, 0.8),
                floor_params=self.config.get('floor_params', {}),
                clustering_params=self.config.get('clustering_params', {}),
                selection_params=self.config.get('selection_params', {}),
                enable_temporal_filtering=True
            )
            self.available_strategies['lidar'] = lidar_strategy
            print("✓ LiDAR clustering strategy initialized")
        except Exception as e:
            print(f"✗ Failed to initialize LiDAR strategy: {e}")
    
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
            print(f"✓ Switched to {strategy.get_strategy_name()}")
            return True
        except Exception as e:
            print(f"✗ Failed to switch to strategy '{strategy_name}': {e}")
            return False
    
    def run(self):
        """Main tracking loop."""
        if not self.initialize():
            print("Failed to initialize system")
            return
        
        print("\nModular Object Tracking Started!")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'h' - Toggle help")
        print("  'p' - Toggle performance info")
        print("  'space' - Pause/Resume")
        print("  '1' - Switch to BEV strategy")
        print("  '2' - Switch to DBSCAN strategy")
        print("  '3' - Switch to LiDAR strategy")
        print("  'r' - Reset robot to home position")
        print("  's' - Stop robot movement")
        print("  'a' - Toggle adaptive timing")
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
                
                # Check for end of stream
                if metadata.get('end_of_stream', False):
                    print("End of stream reached")
                    break
                
                if vertices is not None:
                    # Detect objects with visualization data
                    detected_objects, timing_info, bev_image, point_cloud_view = self._detect_objects(vertices, color_image)
                    
                    # Process robot movement
                    if self.robot_controller.is_connected:
                        self._process_robot_movement(detected_objects)
                    
                    # Create visualization
                    self._create_visualization(vertices, color_image, detected_objects, timing_info, metadata, bev_image, point_cloud_view)
                    
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
    
    def _process_robot_movement(self, detected_objects: List[Dict[str, Any]]):
        """Process robot movement with adaptive timing."""
        current_time = time.time()
        
        # Get timing parameters
        tracking_params = self.config.get('tracking_params', {})
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
        print(f"Robot Update - {time.strftime('%H:%M:%S')} - Strategy: {self.current_strategy_name.upper()}")
        print(f"{'='*60}")
        
        if not detected_objects:
            print("No objects detected")
            self.last_command_had_object = False
            self.last_update_time = current_time
            return
        
        # Find closest object
        closest_object = self._find_closest_object(detected_objects)
        if closest_object is None:
            print("No valid target object found")
            self.last_command_had_object = False
            self.last_update_time = current_time
            return
        
        # Extract position
        camera_position = self._extract_object_position(closest_object)
        if camera_position is None:
            print("Could not extract object position")
            self.last_command_had_object = False
            self.last_update_time = current_time
            return
        
        # Submit async robot command to point at object
        if self.robot_controller.async_mode_enabled:
            # Use async command for non-blocking execution
            command_id = self.robot_controller.point_at_object_async(
                camera_position,
                priority=2,
                timeout=15.0,
                callback=self._robot_command_callback
            )
            
            if command_id:
                print(f"✓ Robot movement command submitted (ID: {command_id})")
                self.current_robot_command_id = command_id
                self.last_command_had_object = True
                self.performance_stats['successful_robot_commands'] += 1
            else:
                print("✗ Failed to submit robot movement command")
                self.last_command_had_object = False
        else:
            # Fallback to synchronous command if async mode disabled
            robot_start = time.time()
            success = self.robot_controller.point_at_object(camera_position)
            robot_time = time.time() - robot_start
            
            if success:
                print("✓ Robot movement command sent successfully (sync)")
                self.last_command_had_object = True
                self.performance_stats['successful_robot_commands'] += 1
            else:
                print("✗ Robot movement command failed (sync)")
                self.last_command_had_object = False
            
            self.performance_stats['robot_command_times'].append(robot_time)
        
        self.last_update_time = current_time
    
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
            print(f"Robot command {command.command_id} completed in {exec_time:.2f}s")
        elif command.status == CommandStatus.FAILED:
            print(f"Robot command {command.command_id} failed: {command.error}")
    
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
    
    def _create_visualization(self, vertices: np.ndarray, color_image: Optional[np.ndarray],
                            detected_objects: List[Dict[str, Any]], timing_info: Dict[str, float],
                            metadata: Dict[str, Any], bev_image: Optional[np.ndarray] = None, 
                            point_cloud_view: Optional[np.ndarray] = None):
        """Create comprehensive visualization with 3-panel display."""
        try:
            # Get strategy info
            strategy_info = self.detector.get_strategy_info() if self.detector else {}
            
            # Create 3-panel visualization like original system
            panels = []
            panel_height = 400
            
            # Panel 1: Color image with overlays
            if color_image is not None:
                color_panel = color_image.copy()
                # Add tracking info overlay
                self._add_tracking_info_overlay(color_panel, detected_objects, timing_info, strategy_info)
                # Add object detection overlay
                self._add_object_detection_overlay(color_panel, detected_objects)
                # Resize to standard height
                aspect_ratio = color_panel.shape[1] / color_panel.shape[0]
                panel_width = int(panel_height * aspect_ratio)
                color_panel = cv2.resize(color_panel, (panel_width, panel_height))
                panels.append(color_panel)
            
            # Panel 2: BEV image
            if bev_image is not None:
                bev_panel = cv2.resize(bev_image, (panel_height, panel_height))
                # Add BEV title
                cv2.putText(bev_panel, "Bird's Eye View", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                panels.append(bev_panel)
            
            # Panel 3: Point cloud side view
            if point_cloud_view is not None:
                pc_panel = cv2.resize(point_cloud_view, (panel_height, panel_height))
                panels.append(pc_panel)
            
            # Combine panels horizontally
            if panels:
                if len(panels) == 1:
                    combined = panels[0]
                else:
                    combined = np.hstack(panels)
                
                # Display combined visualization
                cv2.imshow('Modular Object Tracking', combined)
            elif color_image is not None:
                # Fallback to color image only if no BEV/PC views available
                color_display = color_image.copy()
                self._add_tracking_info_overlay(color_display, detected_objects, timing_info, strategy_info)
                self._add_object_detection_overlay(color_display, detected_objects)
                cv2.imshow('Modular Object Tracking', color_display)
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    def _add_tracking_info_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]],
                                 timing_info: Dict[str, float], strategy_info: Dict[str, str]):
        """Add tracking information overlay to image."""
        tracking_params = self.config.get('tracking_params', {})
        
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
            "MODULAR OBJECT TRACKING",
            f"Strategy: {self.current_strategy_name.upper()} ({strategy_info.get('clustering_strategy', 'Unknown')})",
            f"Objects: {len(detected_objects)}",
            f"FPS: {self.temporal_filter.calculate_fps():.1f}",
            f"Robot: {robot_status}",
            f"Adaptive: {'ON' if self.adaptive_timing else 'OFF'}",
            f"Update: {tracking_params.get('update_interval', 1.0):.1f}s"
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
                # Format: (forward, side, height) to match original system
                info_text = f"Obj{i+1}: ({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f})"
                cv2.putText(image, info_text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _show_performance_window(self, timing_info: Dict[str, float]):
        """Show performance monitoring window."""
        # This could create a separate performance monitoring window
        # For now, we'll just update internal stats
        pass
    
    def _show_help_window(self):
        """Show help information window."""
        # This could create a separate help window
        # For now, help is shown in console
        pass
    
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
        elif key == ord('3'):
            self._switch_strategy('lidar')
        elif key == ord('r'):
            if self.robot_controller.is_connected:
                if self.robot_controller.async_mode_enabled:
                    command_id = self.robot_controller.look_at_table_async(priority=1)
                    print(f"Home position command submitted (ID: {command_id})")
                else:
                    print("Resetting robot to home position...")
                    self.robot_controller.look_at_table()
        elif key == ord('s'):
            if self.robot_controller.is_connected:
                if self.robot_controller.async_mode_enabled:
                    command_id = self.robot_controller.stop_movement_async(priority=3)
                    print(f"Stop movement command submitted (ID: {command_id})")
                else:
                    print("Stopping robot movement...")
                    self.robot_controller.stop_movement()
        elif key == ord('a'):
            self.adaptive_timing = not self.adaptive_timing
            print(f"Adaptive timing {'enabled' if self.adaptive_timing else 'disabled'}")
        
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
        if len(self.performance_stats['robot_command_times']) > max_samples:
            self.performance_stats['robot_command_times'].pop(0)
    
    def _cleanup(self):
        """Cleanup resources."""
        print("\nShutting down Modular Object Tracking System...")
        
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
        
        print("✓ Shutdown complete")
    
    def _print_final_statistics(self):
        """Print final performance statistics."""
        stats = self.performance_stats
        
        print("\n" + "=" * 50)
        print("FINAL PERFORMANCE STATISTICS")
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
    parser = argparse.ArgumentParser(description='Modular Object Tracking Application')
    parser.add_argument('--config', default='realtime_tracking',
                       help='Configuration preset (realtime_tracking, precision_tracking, debug_tracking)')
    parser.add_argument('--strategy', default='bev', choices=['bev', 'dbscan', 'lidar'],
                       help='Initial clustering strategy')
    parser.add_argument('--bag', help='RealSense bag file for replay')
    parser.add_argument('--transform', default='charuco_eye_in_hand_transform.npy',
                       help='Camera-robot transformation matrix file')
    
    args = parser.parse_args()
    
    print("Modular Object Tracking Application")
    print("=" * 40)
    print(f"Configuration: {args.config}")
    print(f"Initial strategy: {args.strategy}")
    if args.bag:
        print(f"Bag file: {args.bag}")
    print("=" * 40)
    
    try:
        tracker = ModularObjectTracker(
            config_name=args.config,
            initial_strategy=args.strategy,
            bag_file=args.bag,
            transformation_matrix_file=args.transform
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