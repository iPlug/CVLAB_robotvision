"""
Modular Camera-Robot Calibration Application

This application provides a modular implementation of camera-robot calibration
using the enhanced modular architecture. It supports multiple calibration modes,
async robot control, and real-time visualization feedback.

Features:
- Multi-point calibration with configurable positions
- Async robot movement for smooth UI operation
- Real-time object detection visualization
- Multiple calibration presets (standard, precise, quick)
- Enhanced error handling and validation
- Comprehensive logging and reporting
"""

import time
import numpy as np
import cv2
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.linear_model import RANSACRegressor

from ..core.object_detector import ObjectDetector
from ..strategies.bev_clustering_strategy import BEVClusteringStrategy
from ..sensors.realsense_manager import RealSenseManager
from ..robot.mycobot_controller import MyCobotController
from ..config.config_factory import ConfigFactory


class ModularCalibrationApp:
    """
    Modular camera-robot calibration application.
    
    Provides a complete calibration workflow using the modular architecture:
    - Configurable calibration strategies
    - Async robot movement
    - Real-time visualization
    - Multi-point calibration
    - Validation testing
    """
    
    def __init__(self, config_name: str = "calibration_standard",
                 transformation_matrix_file: str = "camera_robot_transform.npy",
                 log_file: str = "calibration_report.log",
                 enable_auto_roi: bool = True,
                 floor_offset: float = 0.01):
        """
        Initialize modular calibration application.
        
        Args:
            config_name: Calibration configuration preset name
            transformation_matrix_file: Output file for transformation matrix
            log_file: Log file for calibration process
            enable_auto_roi: Enable automatic ROI detection based on table surface
            floor_offset: Height offset above detected table surface in meters
        """
        self.config_name = config_name
        self.transformation_matrix_file = transformation_matrix_file
        self.log_file_path = log_file
        self.enable_auto_roi = enable_auto_roi
        self.floor_offset = floor_offset
        
        # Load configuration
        self.config = ConfigFactory.create_preset(config_name)
        
        # Initialize components
        self.sensor_manager = None
        self.robot_controller = None
        self.detector = None
        self.log_file = None
        
        # Calibration data
        self.camera_points = []
        self.robot_points = []
        self.transformation_matrix = None
        
        # Application state
        self.is_running = False
        self.current_calibration_point = 0
        self.calibration_completed = False
        
        # Auto ROI detection
        self.table_surface_height = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
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
        print("Initializing Modular Camera-Robot Calibration System")
        print("=" * 60)
        
        # Open log file
        try:
            self.log_file = open(self.log_file_path, 'w')
            self.log_message("Modular calibration process started")
        except Exception as e:
            print(f"Warning: Could not open log file: {e}")
        
        # Initialize sensor manager
        print("Initializing camera sensor...")
        self.sensor_manager = RealSenseManager(realtime_mode=True, use_imu=True)
        if not self.sensor_manager.initialize():
            self.log_message("Error: Failed to initialize camera sensor")
            return False
        
        if not self.sensor_manager.start():
            self.log_message("Error: Failed to start camera sensor")
            return False
        
        # Calibrate gravity alignment
        if self.sensor_manager.use_imu:
            print("Calibrating gravity alignment...")
            self.sensor_manager.calibrate_gravity_alignment()
            self.log_message("Camera gravity alignment calibrated")
        
        # Initialize robot controller
        print("Initializing robot controller...")
        self.robot_controller = MyCobotController(self.transformation_matrix_file)
        
        # Connect to robot
        if not self.robot_controller.connect():
            self.log_message("Warning: Failed to connect to robot")
            return False
        
        # Configure robot parameters
        robot_params = self.config.get('robot_params', {})
        self.robot_controller.safety_constraints.update(robot_params)
        self.log_message(f"Robot safety parameters: {robot_params}")
        
        # Enable async mode for smooth operation
        if self.robot_controller.enable_async_mode(max_queue_size=3):
            self.log_message("Robot async mode enabled")
        else:
            self.log_message("Warning: Failed to enable robot async mode")
        
        # Initialize object detector
        print("Initializing object detection...")
        bev_strategy = BEVClusteringStrategy(
            roi_params=self.config.get('roi_params'),
            display_roi_params=self.config.get('display_roi_params'),
            clustering_params=self.config.get('clustering_params'),
            enable_temporal_filtering=False  # Disable for real-time calibration
        )
        
        self.detector = ObjectDetector(
            clustering_strategy=bev_strategy,
            classification_strategy=None
        )
        
        # Perform auto ROI detection if enabled
        if self.enable_auto_roi:
            if self._detect_and_set_dynamic_roi():
                self.log_message("Auto ROI detection successful")
                print("✓ Auto ROI detection successful")
            else:
                self.log_message("Auto ROI detection failed - using default parameters")
                print("! Auto ROI detection failed - using default parameters")
        
        self.log_message("All systems initialized successfully")
        print("✓ All systems initialized successfully")
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
        self.log_message("Starting auto ROI detection")
        
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
                self.log_message("Auto ROI failed: no valid frames")
                return False
            
            # Combine all frames for more robust table detection
            combined_vertices = np.vstack(all_vertices)
            print(f"[AUTO-ROI] Using {len(combined_vertices)} points from {len(all_vertices)} frames")
            
            # Detect table surface using RANSAC method
            table_info = self._detect_table_surface_ransac(combined_vertices, max_distance=0.015)
            
            if table_info is None:
                print("[AUTO-ROI] Warning: Failed to detect table surface")
                self.log_message("Auto ROI failed: table surface not detected")
                return False
            
            # Extract table height
            detected_table_height = table_info['table_height']
            table_height_std = table_info['table_height_std']
            
            # Validate that this looks like a table surface (not floor)
            expected_table_range = (0.15, 0.6)  # Based on manual presets from calibration
            if not (expected_table_range[0] <= detected_table_height <= expected_table_range[1]):
                print(f"[AUTO-ROI] Detected surface at {detected_table_height:.3f}m outside expected table range {expected_table_range}")
                self.log_message(f"Auto ROI failed: detected height {detected_table_height:.3f}m outside expected range")
                return False
            
            print(f"[AUTO-ROI] Detected table surface: {detected_table_height:.3f}m ± {table_height_std:.3f}m")
            self.log_message(f"Auto ROI detected table at {detected_table_height:.3f}m")
            
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
                self.log_message(f"Auto ROI set dynamic range Y[{dynamic_roi['y_min']:.3f}, {dynamic_roi['y_max']:.3f}]m")
                return True
            else:
                print("[AUTO-ROI] Warning: Detector does not support ROI parameter updates")
                self.log_message("Auto ROI failed: detector does not support ROI updates")
                return False
                
        except Exception as e:
            print(f"[AUTO-ROI] Error during table detection: {e}")
            self.log_message(f"Auto ROI error: {e}")
            return False
    
    def _recalibrate_auto_roi(self):
        """Recalibrate auto ROI during calibration process."""
        try:
            # Recalibrate gravity alignment if available
            if self.sensor_manager.use_imu:
                print("Recalibrating gravity alignment...")
                self.sensor_manager.calibrate_gravity_alignment()
                print("+ Gravity alignment updated")
                self.log_message("Gravity alignment recalibrated")
            
            # Re-detect table surface and update ROI
            if self.enable_auto_roi:
                print("Re-detecting table surface and updating ROI...")
                if self._detect_and_set_dynamic_roi():
                    print("+ Auto ROI updated successfully")
                    self.log_message("Auto ROI recalibrated successfully")
                else:
                    print("! Auto ROI re-detection failed")
                    self.log_message("Auto ROI recalibration failed")
                    
        except Exception as e:
            print(f"Error during auto ROI recalibration: {e}")
            self.log_message(f"Auto ROI recalibration error: {e}")
    
    def run_multipoint_calibration(self) -> Optional[np.ndarray]:
        """
        Run multi-point calibration process.
        
        Returns:
            Transformation matrix if successful, None otherwise
        """
        calibration_positions = self.config.get('calibration_positions', [])
        
        print(f"\\n{'='*60}")
        print("MULTI-POINT CALIBRATION PROCESS")
        print(f"{'='*60}")
        print(f"Collecting {len(calibration_positions)} calibration points")
        print(f"Configuration: {self.config_name}")
        print(f"{'='*60}")
        
        self.log_message(f"Starting multi-point calibration with {len(calibration_positions)} points")
        
        self.camera_points = []
        self.robot_points = []
        
        for i, cal_pos in enumerate(calibration_positions):
            self.current_calibration_point = i + 1
            
            print(f"\\n--- CALIBRATION POINT {i+1}/{len(calibration_positions)} ---")
            self.log_message(f"Starting calibration point {i+1}/{len(calibration_positions)}")
            
            # Move robot to calibration position
            print(f"Moving robot to position {i+1}: {cal_pos}")
            self.log_message(f"Robot target position {i+1}: {cal_pos}")
            
            # Submit async robot movement
            if self.robot_controller.async_mode_enabled:
                command_id = self.robot_controller.move_to_position_async(
                    cal_pos, speed=40, priority=1
                )
                if command_id:
                    print(f"Robot movement command submitted (ID: {command_id})")
                    # Wait for robot to reach position
                    self._wait_for_robot_movement(command_id)
                else:
                    self.log_message(f"Failed to submit robot movement for position {i+1}")
                    continue
            else:
                # Fallback to synchronous movement
                if not self.robot_controller.move_to_position(cal_pos, speed=40):
                    self.log_message(f"Failed to move robot to position {i+1}")
                    continue
                time.sleep(3)  # Wait for movement
            
            # Verify robot position
            try:
                current_pos = self.robot_controller.get_current_position()
                if current_pos:
                    print(f"Robot reached: {current_pos}")
                    self.log_message(f"Robot actual position: {current_pos}")
            except:
                self.log_message("Could not verify robot position")
            
            # Guide user for object placement
            print(f"\\nSTEP {i+1}: OBJECT PLACEMENT")
            print("1. Place your calibration object DIRECTLY BELOW the robot end effector")
            print("2. Make sure the object is clearly visible and well-lit")
            print("3. Press ENTER when the object is positioned correctly...")
            input()
            
            # Move robot to home position to clear camera view
            robot_params = self.config.get('robot_params', {})
            home_method = robot_params.get('home_method', 'home_position')
            
            print(f"Moving robot to {home_method} position...")
            if home_method == 'look_at_table':
                if self.robot_controller.async_mode_enabled:
                    home_cmd = self.robot_controller.look_at_table_async(priority=1)
                    if home_cmd:
                        self._wait_for_robot_movement(home_cmd)
                else:
                    self.robot_controller.look_at_table()
                    time.sleep(2)
            else:
                if self.robot_controller.async_mode_enabled:
                    home_cmd = self.robot_controller.home_position_async(priority=1)
                    if home_cmd:
                        self._wait_for_robot_movement(home_cmd)
                else:
                    self.robot_controller.home_position()
                    time.sleep(2)
            
            # Calibrate gravity alignment
            if self.sensor_manager.use_imu:
                print("Calibrating gravity alignment...")
                self.sensor_manager.calibrate_gravity_alignment()
                self.log_message("Camera gravity alignment calibrated")
            
            # Auto ROI re-detection after robot movement
            if self.enable_auto_roi:
                print("Re-detecting table surface after robot movement...")
                if self._detect_and_set_dynamic_roi():
                    print("+ Auto ROI updated successfully after movement")
                    self.log_message("Auto ROI recalibrated after robot movement")
                else:
                    print("! Auto ROI re-detection failed after movement")
                    self.log_message("Auto ROI recalibration failed after robot movement")

            # Capture object position
            print(f"Capturing object position for calibration point {i+1}...")
            camera_position = self.capture_object_position()
            
            if camera_position is not None:
                self.camera_points.append(camera_position)
                # Use robot XY position but set Z to 0 (object is on table, not at end effector height)
                actual_object_position = [cal_pos[0], cal_pos[1], 30]
                self.robot_points.append(actual_object_position)
                
                self.log_message(f"Point {i+1} captured successfully:")
                self.log_message(f"  Robot end effector: {cal_pos[:3]}")
                self.log_message(f"  Object position: {actual_object_position}")
                self.log_message(f"  Camera: {camera_position}")
                
                print(f"✓ Calibration point {i+1} captured successfully")
            else:
                self.log_message(f"Failed to capture calibration point {i+1}")
                print(f"✗ Failed to capture calibration point {i+1}")
                
                # Ask user if they want to retry or continue
                retry = input("Retry this point? (y/n): ").lower().strip()
                if retry == 'y':
                    i -= 1  # Retry current point
                    continue
        
        # Check if we have enough points
        min_points = self.config.get('accuracy_thresholds', {}).get('min_calibration_points', 3)
        if len(self.camera_points) < min_points:
            self.log_message(f"Error: Only captured {len(self.camera_points)} points, need at least {min_points}")
            print(f"\\n✗ Calibration failed: Only {len(self.camera_points)} points captured, need at least {min_points}")
            return None
        
        # Calculate transformation matrix
        print(f"\\nCalculating transformation matrix from {len(self.camera_points)} points...")
        self.log_message(f"Calculating transformation matrix from {len(self.camera_points)} calibration points")
        
        transformation_matrix = self.calculate_transformation_matrix(self.camera_points, self.robot_points)
        
        if transformation_matrix is not None:
            self.transformation_matrix = transformation_matrix
            print("✓ Transformation matrix calculated successfully")
            self.log_message("Transformation matrix calculated successfully")
            return transformation_matrix
        else:
            print("✗ Failed to calculate transformation matrix")
            self.log_message("Failed to calculate transformation matrix")
            return None
    
    def capture_object_position(self) -> Optional[List[float]]:
        """
        Capture object position from camera with real-time visualization.
        
        Returns:
            Object position in camera coordinates [forward, side, height] or None
        """
        print("\\nObject Detection Mode:")
        print("- Position camera to clearly see the calibration object")
        print("- Press 'c' to capture object position when ready")
        print("- Press 'q' to skip this calibration point")
        print("- Press 'r' to recalibrate auto ROI (if camera moved)")
        print("- Real-time detection will be shown...")
        
        captured_position = None
        
        try:
            while True:
                # Get frame from sensor
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                if vertices is not None:
                    # Detect objects
                    detected_objects, timing_info, bev_image, point_cloud_view = self._detect_objects_with_visualization(vertices, color_image)
                    
                    # Create calibration visualization
                    self._create_calibration_visualization(
                        vertices, color_image, detected_objects, 
                        timing_info, bev_image, point_cloud_view
                    )
                
                # Handle user input
                key = cv2.waitKey(30) & 0xFF
                if key == ord('c'):
                    # Capture current detection
                    if detected_objects:
                        # Find the largest/most central object
                        best_object = self._find_best_calibration_object(detected_objects)
                        if best_object:
                            captured_position = self._extract_object_position(best_object)
                            print(f"✓ Object captured at position: {captured_position}")
                            break
                        else:
                            print("No suitable calibration object detected")
                    else:
                        print("No objects detected - position camera and try again")
                
                elif key == ord('r'):
                    # Recalibrate auto ROI
                    if self.enable_auto_roi:
                        print("Recalibrating auto ROI...")
                        self._recalibrate_auto_roi()
                    else:
                        print("Auto ROI is disabled")
                
                elif key == ord('q'):
                    print("Skipping this calibration point")
                    break
        
        except KeyboardInterrupt:
            print("\\nCapture interrupted by user")
        except Exception as e:
            self.log_message(f"Error during object capture: {e}")
            print(f"Error during object capture: {e}")
        
        return captured_position
    
    def _detect_objects_with_visualization(self, vertices: np.ndarray, color_image: Optional[np.ndarray]):
        """Detect objects and get visualization data."""
        try:
            # Check if strategy supports visualization
            if hasattr(self.detector.clustering_strategy, 'cluster_objects'):
                result = self.detector.clustering_strategy.cluster_objects(
                    vertices, color_image, return_visualization=True
                )
                if len(result) == 3:
                    detected_objects, bev_image, point_cloud_view = result
                else:
                    detected_objects = result
                    bev_image = point_cloud_view = None
                
                timing_info = {'total_time': 0.0, 'num_objects': len(detected_objects) if detected_objects else 0}
                return detected_objects, timing_info, bev_image, point_cloud_view
            else:
                detected_objects, timing_info = self.detector.detect_objects(vertices, color_image)
                return detected_objects, timing_info, None, None
                
        except Exception as e:
            self.log_message(f"Error during object detection: {e}")
            return [], {'total_time': 0.0, 'num_objects': 0}, None, None
    
    def _create_calibration_visualization(self, vertices: np.ndarray, color_image: Optional[np.ndarray],
                                        detected_objects: List[Dict[str, Any]], timing_info: Dict[str, float],
                                        bev_image: Optional[np.ndarray], point_cloud_view: Optional[np.ndarray]):
        """Create calibration-specific visualization."""
        try:
            panels = []
            panel_height = 400
            
            # Panel 1: Color image with calibration info
            if color_image is not None:
                color_panel = color_image.copy()
                self._add_calibration_overlay(color_panel, detected_objects)
                
                # Resize to standard height
                aspect_ratio = color_panel.shape[1] / color_panel.shape[0]
                panel_width = int(panel_height * aspect_ratio)
                color_panel = cv2.resize(color_panel, (panel_width, panel_height))
                panels.append(color_panel)
            
            # Panel 2: BEV image
            if bev_image is not None:
                bev_panel = cv2.resize(bev_image, (panel_height, panel_height))
                cv2.putText(bev_panel, "Bird's Eye View", (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                panels.append(bev_panel)
            
            # Panel 3: Point cloud view
            if point_cloud_view is not None:
                pc_panel = cv2.resize(point_cloud_view, (panel_height, panel_height))
                panels.append(pc_panel)
            
            # Combine panels
            if panels:
                if len(panels) == 1:
                    combined = panels[0]
                else:
                    combined = np.hstack(panels)
                
                cv2.imshow('Modular Camera Calibration', combined)
            
        except Exception as e:
            self.log_message(f"Error creating calibration visualization: {e}")
    
    def _add_calibration_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]]):
        """Add calibration-specific overlay information."""
        info_lines = [
            "MODULAR CAMERA CALIBRATION",
            f"Config: {self.config_name}",
            f"Point: {self.current_calibration_point}/{len(self.config.get('calibration_positions', []))}",
            f"Objects: {len(detected_objects)}",
            "",
            "Controls:",
            "'c' - Capture object position",
            "'q' - Skip this point"
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
            
            cv2.putText(image, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add object information
        if detected_objects:
            y_offset = 250
            for i, obj in enumerate(detected_objects[:3]):
                position = self._extract_object_position(obj)
                if position:
                    obj_text = f"Obj{i+1}: ({position[0]:.0f}, {position[1]:.0f}, {position[2]:.0f})mm"
                    cv2.putText(image, obj_text, (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _find_best_calibration_object(self, detected_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the best object for calibration (largest and most central)."""
        if not detected_objects:
            return None
        
        best_object = None
        best_score = -1
        
        for obj in detected_objects:
            # Score based on size and centrality
            num_points = obj.get('num_points', 0)
            position = self._extract_object_position(obj)
            
            if position and num_points > 0:
                # Distance from center (prefer central objects)
                center_distance = np.sqrt(position[0]**2 + position[1]**2)
                
                # Combined score: size bonus - distance penalty
                score = num_points - center_distance * 1
                
                if score > best_score:
                    best_score = score
                    best_object = obj
        
        # if nothing found, return the first object
        if best_object is None:
            print("No suitable calibration object found, using first detected object")
            return detected_objects[0] if detected_objects else None
        
        return best_object
    
    def _extract_object_position(self, obj: Dict[str, Any]) -> Optional[List[float]]:
        """Extract 3D position from object in camera coordinates."""
        try:
            if 'center_mm' in obj:
                center_mm = obj['center_mm']
                return [center_mm[2], center_mm[0], center_mm[1]]  # [forward, side, height]
            elif 'center' in obj:
                center = obj['center']
                return [center[2]*1000, center[0]*1000, center[1]*1000]  # Convert to mm
            else:
                return None
        except:
            return None
    
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
    
    def calculate_transformation_matrix(self, camera_points: List[List[float]], 
                                      robot_points: List[List[float]]) -> Optional[np.ndarray]:
        """Calculate transformation matrix using least-squares fitting."""
        if len(camera_points) < 3:
            self.log_message("Error: Need at least 3 calibration points")
            return None
        
        try:
            # Convert to numpy arrays
            camera_pts = np.array(camera_points)  # Nx3
            robot_pts = np.array(robot_points)    # Nx3
            
            self.log_message(f"Calculating transformation from {len(camera_points)} points:")
            for i, (cam, rob) in enumerate(zip(camera_points, robot_points)):
                self.log_message(f"  Point {i+1}: Camera{cam} -> Robot{rob}")
            
            # Add homogeneous coordinate (ones column)
            camera_homogeneous = np.hstack([camera_pts, np.ones((len(camera_pts), 1))])  # Nx4
            
            # Solve for transformation matrix using least-squares
            # camera_homogeneous @ T.T = robot_pts
            # T.T = pinv(camera_homogeneous) @ robot_pts
            transformation_3x4 = np.linalg.lstsq(camera_homogeneous, robot_pts, rcond=None)[0].T
            
            # Convert to 4x4 homogeneous transformation matrix
            transformation_matrix = np.vstack([transformation_3x4, [0, 0, 0, 1]])
            
            self.log_message("Calculated transformation matrix:")
            for row in transformation_matrix:
                self.log_message(f"  {row}")
            
            # Calculate and log calibration errors
            self._calculate_calibration_errors(camera_points, robot_points, transformation_matrix)
            
            return transformation_matrix
            
        except Exception as e:
            self.log_message(f"Error calculating transformation matrix: {e}")
            return None
    
    def _calculate_calibration_errors(self, camera_points: List[List[float]], 
                                    robot_points: List[List[float]], 
                                    transformation_matrix: np.ndarray):
        """Calculate and log calibration errors."""
        errors = []
        
        for i, (cam_pt, rob_pt) in enumerate(zip(camera_points, robot_points)):
            # Transform camera point to robot coordinates
            cam_homogeneous = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1])
            predicted_robot = np.dot(transformation_matrix, cam_homogeneous)[:3]
            
            # Calculate error
            error = np.linalg.norm(np.array(predicted_robot) - np.array(rob_pt))
            errors.append(error)
            
            self.log_message(f"Point {i+1} error: {error:.2f}mm")
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        self.log_message(f"Calibration errors - Mean: {mean_error:.2f}mm, Max: {max_error:.2f}mm")
        
        # Check against thresholds
        max_allowed = self.config.get('accuracy_thresholds', {}).get('max_calibration_error', 100.0)
        if max_error > max_allowed:
            self.log_message(f"Warning: Max error {max_error:.2f}mm exceeds threshold {max_allowed}mm")
    
    def save_transformation_matrix(self, transformation_matrix: np.ndarray) -> bool:
        """Save transformation matrix to file."""
        try:
            filepath = os.path.abspath(self.transformation_matrix_file)
            np.save(filepath, transformation_matrix)
            
            self.log_message(f"Transformation matrix saved to: {filepath}")
            print(f"✓ Transformation matrix saved to: {filepath}")
            return True
            
        except Exception as e:
            self.log_message(f"Error saving transformation matrix: {e}")
            print(f"✗ Error saving transformation matrix: {e}")
            return False
    
    def run_validation_tests(self) -> bool:
        """Run validation tests to verify calibration accuracy."""
        if self.transformation_matrix is None:
            print("No transformation matrix to validate")
            return False
        
        test_positions = self.config.get('test_positions', [])
        if not test_positions:
            print("No test positions defined, skipping validation")
            return True
        
        print(f"\\n{'='*60}")
        print("CALIBRATION VALIDATION")
        print(f"{'='*60}")
        print(f"Running {len(test_positions)} validation tests...")
        
        validation_errors = []
        
        for i, test_pos in enumerate(test_positions):
            print(f"\\n--- Validation Test {i+1}/{len(test_positions)} ---")
            self.log_message(f"Validation test {i+1}: {test_pos}")
            
            # Move robot to test position
            print(f"Moving robot to test position: {test_pos}")
            if self.robot_controller.async_mode_enabled:
                cmd_id = self.robot_controller.move_to_position_async(test_pos, speed=40)
                if cmd_id:
                    self._wait_for_robot_movement(cmd_id)
            else:
                self.robot_controller.move_to_position(test_pos, speed=40)
                time.sleep(3)
            
            # Guide user
            print("Place calibration object below robot and press ENTER...")
            input()
            
            # Move robot away
            robot_params = self.config.get('robot_params', {})
            home_method = robot_params.get('home_method', 'home_position')
            
            if home_method == 'look_at_table':
                if self.robot_controller.async_mode_enabled:
                    home_cmd = self.robot_controller.look_at_table_async(priority=1)
                    if home_cmd:
                        self._wait_for_robot_movement(home_cmd)
                else:
                    self.robot_controller.look_at_table()
                    time.sleep(2)
            else:
                if self.robot_controller.async_mode_enabled:
                    home_cmd = self.robot_controller.home_position_async()
                    if home_cmd:
                        self._wait_for_robot_movement(home_cmd)
                else:
                    self.robot_controller.home_position()
                    time.sleep(2)
            
            # Capture object position
            camera_position = self.capture_object_position()
            
            if camera_position:
                # Transform to robot coordinates
                cam_homogeneous = np.array([camera_position[0], camera_position[1], camera_position[2], 1])
                predicted_robot = np.dot(self.transformation_matrix, cam_homogeneous)[:3]
                
                # Calculate error
                actual_robot = test_pos[:3]
                error = np.linalg.norm(predicted_robot - actual_robot)
                validation_errors.append(error)
                
                print(f"Validation {i+1} - Expected: {actual_robot}, Predicted: {predicted_robot}")
                print(f"Validation {i+1} - Error: {error:.2f}mm")
                self.log_message(f"Validation {i+1} error: {error:.2f}mm")
            else:
                print(f"Failed to capture validation point {i+1}")
                self.log_message(f"Failed to capture validation point {i+1}")
        
        # Analyze validation results
        if validation_errors:
            mean_error = np.mean(validation_errors)
            max_error = np.max(validation_errors)
            
            print(f"\\nValidation Results:")
            print(f"Mean error: {mean_error:.2f}mm")
            print(f"Max error: {max_error:.2f}mm")
            
            self.log_message(f"Validation results - Mean: {mean_error:.2f}mm, Max: {max_error:.2f}mm")
            
            # Check against threshold
            max_allowed = self.config.get('accuracy_thresholds', {}).get('max_validation_error', 50.0)
            if max_error <= max_allowed:
                print(f"✓ Validation PASSED (error ≤ {max_allowed}mm)")
                self.log_message("Validation PASSED")
                return True
            else:
                print(f"✗ Validation FAILED (error > {max_allowed}mm)")
                self.log_message("Validation FAILED")
                return False
        else:
            print("No validation data collected")
            return False
    
    def cleanup(self):
        """Cleanup resources and close connections."""
        print("\\nCleaning up calibration system...")
        
        # Close visualization windows
        cv2.destroyAllWindows()
        
        # Stop sensor
        if self.sensor_manager:
            self.sensor_manager.stop()
        
        # Cleanup robot
        if self.robot_controller:
            self.robot_controller.cleanup_async_system()
            self.robot_controller.disconnect()
        
        # Close log file
        if self.log_file:
            self.log_file.close()
        
        print("✓ Cleanup complete")
    
    def run_full_calibration(self) -> bool:
        """
        Run the complete calibration process.
        
        Returns:
            True if calibration successful
        """
        try:
            # Initialize systems
            if not self.initialize_systems():
                print("System initialization failed")
                return False
            
            # Run calibration
            transformation_matrix = self.run_multipoint_calibration()
            if transformation_matrix is None:
                print("Calibration failed")
                return False
            
            # Save transformation matrix
            if not self.save_transformation_matrix(transformation_matrix):
                print("Failed to save transformation matrix")
                return False
            
            # Run validation
            print("\\nRunning validation tests...")
            validation_passed = self.run_validation_tests()
            
            if validation_passed:
                print("\\n🎉 CALIBRATION COMPLETED SUCCESSFULLY!")
                self.log_message("Calibration completed successfully")
            else:
                print("\\n⚠ CALIBRATION COMPLETED WITH VALIDATION WARNINGS")
                self.log_message("Calibration completed with validation warnings")
            
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