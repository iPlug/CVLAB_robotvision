#!/usr/bin/env python3
"""
Enhanced Real-time LiDAR Object Position Prediction with Advanced Robot Integration

Real-time markerless object detection and position prediction using LiDAR point cloud data
with comprehensive robot functionality including pointing, grasping, and camera control.

Features:
- Real-time object position prediction relative to robot base (markerless)
- Virtual fixed camera mode - robot maintains observation pose for continuous detection
- On-demand pointing at closest detected object ('p' key)
- Full grasp sequence with gripper control and drop at specified position ('g' key)  
- Camera tilt controls for optimal viewing angles ('w'/'s' keys)
- Advanced robot state management with async operations
- Surface plane detection for objects on tables/floors
- Point cloud clustering for robust multi-object detection
- Live visualization with position display and robot status
- Robot coordinate system (X=forward, Y=left, Z=up)

Usage:
    python lidar_realtime_prediction_with_robot.py [options]

Controls:
    'p' - Point at closest detected object (3 sec, return to observation)
    'g' - Grasp closest object (hover, open, grasp, lift, drop, close gripper, return)
    '1' - Grasp Object 1 (specific target)
    '2' - Grasp Object 2 (specific target)
    '3' - Grasp Object 3 (specific target)
    '4-9' - Grasp Objects 4-9 (specific targets)
    'w' - Tilt camera UP (Joint4 +30°)
    's' - Tilt camera DOWN (Joint4 -30°) or Stop robot movement
    'f' - Look forward (horizontal view)  
    'r' - Return to observation pose
    'c' - Move robot to next preset position
    'a' - Recalibrate auto ROI
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
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities and robot controller
from utils import RobotUtils, CameraUtils, CoordinateUtils, DetectionUtils
from module.robot.mycobot_controller import MyCobotController


class EnhancedLidarPredictor:
    """Enhanced real-time LiDAR-based markerless object position prediction system with advanced robot integration."""
    
    def __init__(self, config_path="charuco-pose-estimation", prediction_offset=None):
        """Initialize the enhanced real-time LiDAR object detection system.
        
        Args:
            config_path: Path to configuration directory
            prediction_offset: 3D offset [x, y, z] in mm to add to all predictions (default: [0, 0, 0])
                             Useful for fine-tuning predictions or accounting for mounting offsets
        """
        self.config_path = Path(config_path)
        
        # Prediction offset configuration
        if prediction_offset is None:
            self.prediction_offset = np.array([0.0, 0.0, 0.0])
        else:
            self.prediction_offset = np.array(prediction_offset, dtype=float)
            if self.prediction_offset.shape != (3,):
                raise ValueError(f"prediction_offset must be [x, y, z], got shape {self.prediction_offset.shape}")
        
        print(f"[INFO] Prediction offset: [{self.prediction_offset[0]:.1f}, {self.prediction_offset[1]:.1f}, {self.prediction_offset[2]:.1f}] mm")
        
        # LiDAR detection parameters
        self.surface_distance_threshold = 0.01  # 1cm threshold for surface plane detection
        self.object_height_threshold = 0.005    # 5mm minimum object height above surface
        self.clustering_eps = 0.02               # 2cm clustering tolerance
        self.clustering_min_samples = 10         # Minimum points for object cluster
        
        # KNN detection parameters (simplified approach)
        self.knn_radius = 0.05                   # 5cm radius for KNN search
        self.knn_min_neighbors = 5               # Minimum neighbors for averaging
        
        self.roi_bounds = {                      # Region of interest bounds (meters)
            'x_min': -0.5, 'x_max': 0.5,
            'y_min': -0.5, 'y_max': 0.5, 
            'z_min': 0.1,  'z_max': 0.5     # Close range for tabletop objects (10-50cm)
        }
        
        # Visualization optimization parameters
        self.viz_max_points = 5000               # Maximum points to render for visualization (reduces lag)
        self.viz_downsample_factor = 3           # Every Nth point for visualization (1=all points, 3=every 3rd point)
        self.viz_show_performance = False        # Disable verbose performance metrics (reduces console noise)
        
        # Initialize utility classes
        self.robot_utils = RobotUtils()
        self.camera_utils = CameraUtils(config_path)
        self.coordinate_utils = CoordinateUtils()
        self.detection_utils = DetectionUtils(config_path)
        
        # Enhanced robot controller setup - reuse the robot from robot_utils to avoid COM port conflicts
        self.enhanced_robot_controller = None
        self.base_joint_angles = None  # Joint angles for camera tilt control reference
        self._initialize_enhanced_robot_controller()
        
        # Load robot poses
        self.robot_utils.load_robot_poses()
        
        # Load camera intrinsics first (may be overridden by calibration)
        self.camera_utils.load_camera_intrinsics()
        
        # Load pre-calibrated eye-in-hand transformation (this will override intrinsics with calibration values)
        self.eye_in_hand_transform = self.camera_utils.load_eye_in_hand_transform()
        
        # Point cloud processing
        self.pc = rs.pointcloud()
        self.points = rs.points()
        
        # Detection state
        self.current_surface_plane = None
        self.detection_confidence = 0.0
        
        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Frame counter for saving and output control
        self.frame_count = 0
        self.position_print_counter = 0
        
        # Enhanced robot state tracking
        self.robot_in_observation_pose = False
        self.pointing_in_progress = False
        self.grasping_in_progress = False
        self.returning_to_observer = False
        self.pointing_start_time = 0
        self.return_start_time = 0
        
        # Non-blocking grasping state machine
        self.grasping_state = None  # 'starting', 'hovering', 'opening', 'descending', 'waiting_to_close', 'closing', 'lifting', 'dropping', 'waiting_to_drop', 'drop_opening', 'drop_closing', 'returning'
        self.grasping_command_id = None
        self.grasping_object = None
        self.grasping_step_start_time = None
        
        # Drop position configuration (X, Y, Z, RX, RY, RZ)
        self.drop_position = [40, -200, 150, -180, 0, -135]
        
        # Timing parameters
        self.pointing_duration = 20.0  # 20 seconds for complete grasping sequence
        self.stabilization_time = 3.0
        
        # Movement speed parameters
        self.movement_speed = 100      # Standard movement speed (0-100) - used for hovering, dropping, joint movements
        self.careful_speed = 100       # Careful/precision movement speed (0-100) - used for descending to grasp, lifting objects
        
        # Grasping sequence timing parameters (all in seconds)
        self.grasping_timeout = 5.0           # Master timeout for each grasping step
        self.hover_stability_wait = 0        # Wait time for hover position stability (with status)
        self.hover_fallback_wait = 6.0         # Wait time for hover position (fallback without status)
        self.gripper_open_wait = 0           # Wait time for gripper opening
        self.object_positioning_wait = 0     # Wait before closing gripper (object positioning)
        self.gripper_close_wait = 0          # Wait time for gripper closing and securing object
        self.descent_timeout = 0             # Timeout for descent movement
        self.lift_timeout = 0                # Timeout for lift movement
        self.drop_movement_timeout = 0       # Timeout for drop movement
        self.drop_stabilization_wait = .75     # Wait at drop position for robot stabilization
        self.drop_gripper_open_wait = 0      # Wait time for gripper opening at drop
        self.drop_gripper_close_wait = 0     # Wait time for gripper closing after drop
        self.return_timeout = 4              # Timeout for return movement
        
        # Joint control parameters (for camera tilt functionality)
        self.joint_step = 30  # degrees per keypress for joints
        self.last_movement_time = 0
        self.movement_cooldown = 0.1  # Minimum time between movements
        
        # Detection state for robot actions
        self.last_detection_objects = []  # Store for pointing/grasping
        self.last_detection_results = {}  # Store complete detection results for UI during robot actions
        
        # Performance monitoring
        self.performance_stats = {
            'total_detections': 0,
            'pointing_commands': 0,
            'grasping_commands': 0,
            'detection_times': [],
            'avg_fps': 0.0
        }
    
    def _is_robot_available(self):
        """Check if robot is available for enhanced commands."""
        if self.enhanced_robot_controller:
            # Check if it has is_connected attribute
            if hasattr(self.enhanced_robot_controller, 'is_connected'):
                return self.enhanced_robot_controller.is_connected
            else:
                # Fallback: assume available if we have the controller
                return True
        return False
    
    def _initialize_enhanced_robot_controller(self):
        """Initialize enhanced robot controller using existing robot connection from robot_utils."""
        try:
            # Use the existing robot connection from robot_utils to avoid COM port conflicts
            if self.robot_utils.robot and hasattr(self.robot_utils.robot, 'is_connected'):
                if self.robot_utils.robot.is_connected:
                    print("✓ Using existing robot connection from robot_utils")
                    self.enhanced_robot_controller = self.robot_utils.robot
                    
                    # Try to enable async mode if available
                    if hasattr(self.enhanced_robot_controller, 'enable_async_mode'):
                        if self.enhanced_robot_controller.enable_async_mode(max_queue_size=5):
                            print("✓ Robot async mode enabled for smooth visualization")
                        else:
                            print("! Async mode not available - using synchronous commands")
                    else:
                        print("! Async mode not supported by this robot controller")
                    
                    # Try to capture base joint angles for camera tilt control
                    try:
                        if hasattr(self.enhanced_robot_controller, 'get_joint_angles'):
                            self.base_joint_angles = self.enhanced_robot_controller.get_joint_angles()
                            if self.base_joint_angles and len(self.base_joint_angles) >= 6:
                                print(f"✓ Base joint angles captured: {[f'{x:.1f}' for x in self.base_joint_angles]}")
                                print("✓ Camera tilt controls (w/s keys) enabled")
                                self.robot_in_observation_pose = True
                            else:
                                print("! Could not capture base joint angles - camera tilt controls disabled")
                                self.base_joint_angles = None
                        else:
                            print("! get_joint_angles not available - camera tilt controls disabled")
                            self.base_joint_angles = None
                    except Exception as e:
                        print(f"! Failed to capture base joint angles: {e}")
                        self.base_joint_angles = None
                        
                    print("✓ Enhanced robot functionality initialized with existing connection")
                else:
                    print("! Robot connection exists but not connected - enhanced features disabled")
                    self.enhanced_robot_controller = None
            else:
                print("! No robot connection available from robot_utils - enhanced features disabled")
                self.enhanced_robot_controller = None
                
        except Exception as e:
            print(f"! Error setting up enhanced robot functionality: {e}")
            self.enhanced_robot_controller = None
    
    def predict_object_position(self, object_center, robot_transform):
        """
        Predict object position in robot base coordinates using eye-in-hand calibration.
        
        COORDINATE SYSTEM NOTES:
        - LiDAR detection provides object center in camera coordinates (relative to camera center)
        - Calibration matrix was created with CENTER ORIGIN (as per README.txt)
        - This means both systems use center-based coordinates - they should be compatible
        
        Args:
            object_center: 3D center point in camera coordinates [x, y, z] in meters
            robot_transform: 4x4 robot base-to-gripper transformation matrix
            
        Returns:
            np.array: Object position in robot base coordinates [x, y, z] in mm
        """
        # Create transformation matrix for object in camera frame
        object_transform = np.eye(4)
        object_transform[:3, 3] = object_center
        
        # Object in robot base = Robot_base_to_gripper @ Eye_in_hand @ Camera_to_object
        object_in_base = robot_transform @ self.eye_in_hand_transform @ object_transform
        position_mm = object_in_base[:3, 3] * 1000  # Convert to millimeters
        
        # Apply prediction offset
        position_mm = position_mm + self.prediction_offset
        
        return position_mm
    
    def update_position_history(self, position):
        """Update position history for stability analysis."""
        return self.coordinate_utils.update_position_history(position)
    
    def get_position_stability(self):
        """Get position stability metrics."""
        return self.coordinate_utils.get_position_stability()
    
    def _project_point_to_pixel(self, point_3d, intrinsics):
        """Project a 3D point to 2D pixel coordinates."""
        # Use the color intrinsics for projection (since depth is aligned to color)
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy
        
        x, y, z = point_3d
        if z == 0:
            return None
        
        # Project to image plane
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        
        # Check if point is within image bounds
        if 0 <= u < intrinsics.width and 0 <= v < intrinsics.height:
            return (u, v)
        return None
    
    def _point_and_return(self):
        """Start pointing at closest detected object (non-blocking)."""
        if not self._is_robot_available():
            print("Robot not connected - cannot point at object")
            return
        
        if self.pointing_in_progress or self.grasping_in_progress:
            print("Robot action already in progress")
            return
        
        if not self.last_detection_objects:
            print("No objects detected - cannot point")
            return
        
        # Find closest object
        closest_object = self._find_closest_object(self.last_detection_objects)
        if closest_object is None:
            print("No valid target object found")
            return
        
        # Extract position (use center directly, same as main loop visualization)
        camera_position_m = self._extract_object_position(closest_object)

        if camera_position_m is None:
            print("Could not extract object position")
            return
        
        # Convert to mm for display
        camera_position_mm = [pos * 1000 for pos in camera_position_m]
        
        print(f"\n{'='*60}")
        print(f"POINTING ACTION - {time.strftime('%H:%M:%S')}")
        print(f"Target: ({camera_position_mm[0]:.1f}, {camera_position_mm[1]:.1f}, {camera_position_mm[2]:.1f}) mm")
        print(f"{'='*60}")
        
        # Start non-blocking pointing sequence
        self.pointing_in_progress = True
        self.pointing_start_time = time.time()
        
        try:
            # Try async pointing if available
            if hasattr(self.enhanced_robot_controller, 'point_at_object_async'):
                # Async pointing command
                print("1. Pointing at object...")
                command_id = self.enhanced_robot_controller.point_at_object_async(
                    camera_position_m,  # Use meters as expected by the robot controller
                    priority=2,
                    timeout=15.0
                )
                
                if command_id:
                    print(f"[OK] Pointing command submitted (ID: {command_id})")
                    print("   RGB feed continues during pointing...")
                    self.performance_stats['pointing_commands'] += 1
                else:
                    print("[ERROR] Failed to submit pointing command - trying sync")
                    # Fallback to sync pointing
                    success = self.enhanced_robot_controller.point_at_object(camera_position_m)
                    if success:
                        print("[OK] Robot pointing at object (sync)")
                        self.performance_stats['pointing_commands'] += 1
                    else:
                        print("[ERROR] Failed to point at object")
                        self.pointing_in_progress = False
            elif hasattr(self.enhanced_robot_controller, 'point_at_object'):
                # Synchronous pointing command
                print("1. Pointing at object...")
                success = self.enhanced_robot_controller.point_at_object(camera_position_m)
                
                if success:
                    print("[OK] Robot pointing at object (sync)")
                    print("   RGB feed continues during pointing...")
                    self.performance_stats['pointing_commands'] += 1
                else:
                    print("[ERROR] Failed to point at object")
                    self.pointing_in_progress = False
            else:
                print("[ERROR] No pointing method available on robot controller")
                self.pointing_in_progress = False
        
        except Exception as e:
            print(f"Error during pointing sequence: {e}")
            self.pointing_in_progress = False
    
    def _grasp_and_return(self):
        """Start non-blocking grasping sequence (state machine approach)."""
        if not self._is_robot_available():
            print("Robot not connected - cannot grasp object")
            return
        
        if self.pointing_in_progress or self.grasping_in_progress:
            print("Robot action already in progress")
            return
        
        # Print all detected object centers before grasping
        print(f"\n{'='*60}")
        print(f"GRASPING ACTION - {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        if not self.last_detection_objects:
            print("No objects detected for grasping")
            return
        
        # Get current robot pose for coordinate transformation (same as main loop)
        try:
            robot_pose = self.get_robot_pose()
            robot_transform = self.pose_to_transform_matrix(robot_pose) if robot_pose else None
        except Exception as e:
            print(f"Warning: Could not get robot transform for consistent coordinates: {e}")
            robot_transform = None
        
        print(f"Detected {len(self.last_detection_objects)} object(s):")
        for i, obj in enumerate(self.last_detection_objects):
            obj_pos = self._extract_object_position(obj)
            if obj_pos is not None:
                # Use same coordinate transformation as main loop for consistency
                if robot_transform is not None:
                    # Transform to robot coordinates (in mm) - same as main loop
                    obj_pos_robot_mm = self.predict_object_position(obj_pos, robot_transform)
                    confidence = obj.get('confidence', 0.0)
                    size = obj.get('size', 0)
                    print(f"  Object #{obj.get('id', i+1)}: ({obj_pos_robot_mm[0]:6.1f}, {obj_pos_robot_mm[1]:6.1f}, {obj_pos_robot_mm[2]:6.1f}) mm, size={size:3d}, conf={confidence:.2f}")
                else:
                    # Fallback: use camera coordinates in mm (may be inconsistent)
                    obj_pos_mm = [pos * 1000 for pos in obj_pos]
                    confidence = obj.get('confidence', 0.0)
                    size = obj.get('size', 0)
                    print(f"  Object #{obj.get('id', i+1)}: ({obj_pos_mm[0]:6.1f}, {obj_pos_mm[1]:6.1f}, {obj_pos_mm[2]:6.1f}) mm [camera], size={size:3d}, conf={confidence:.2f}")
        
        # Find closest object
        closest_object = self._find_closest_object(self.last_detection_objects)
        if closest_object is None:
            print("No valid target object found for grasping")
            return
        
        # Extract position
        camera_position_m = self._extract_object_position(closest_object)
        if camera_position_m is None:
            print("Could not extract object position")
            return
        
        # Get robot coordinates for display consistency (same as main loop)
        if robot_transform is not None:
            robot_position_mm = self.predict_object_position(camera_position_m, robot_transform)
            display_position_mm = robot_position_mm
            coord_label = "robot"
        else:
            # Fallback to camera coordinates
            display_position_mm = [pos * 1000 for pos in camera_position_m]
            coord_label = "camera"
        
        # Find which object index this is
        target_index = -1
        closest_center = self._extract_object_position(closest_object)
        for i, obj in enumerate(self.last_detection_objects):
            obj_center = self._extract_object_position(obj)
            if obj_center is not None and closest_center is not None:
                # Compare positions with small tolerance for floating point precision
                if abs(obj_center[0] - closest_center[0]) < 1e-6 and \
                   abs(obj_center[1] - closest_center[1]) < 1e-6 and \
                   abs(obj_center[2] - closest_center[2]) < 1e-6:
                    target_index = i + 1
                    break
        
        # Fallback if target index not found
        if target_index == -1:
            target_index = 1  # Default to first object
            print(f"[Note: Using default target selection]")
        
        print(f"\n→ TARGETING Object {target_index} (closest to camera):")
        print(f"  Position: ({display_position_mm[0]:6.1f}, {display_position_mm[1]:6.1f}, {display_position_mm[2]:6.1f}) mm [{coord_label}]")
        print(f"  Size: {closest_object.get('size', 0)} points")
        print(f"  Confidence: {closest_object.get('confidence', 0.0):.2f}")
        print(f"{'='*60}")
        
        # Initialize non-blocking grasping state machine
        self.grasping_in_progress = True
        self.grasping_state = 'starting'
        self.grasping_object = {
            'camera_position_m': camera_position_m,
            'display_position_mm': display_position_mm,
            'coord_label': coord_label,
            'closest_object': closest_object
        }
        self.grasping_step_start_time = time.time()
        self.pointing_start_time = time.time()  # For consistency with pointing timeout
        
        print("\n✓ Grasping sequence started - RGB feed will continue updating")
        if target_index > 0:
            print(f"  Target object: Object {target_index} at ({display_position_mm[0]:.1f}, {display_position_mm[1]:.1f}, {display_position_mm[2]:.1f}) mm [{coord_label}]")
        else:
            print(f"  Target object at ({display_position_mm[0]:.1f}, {display_position_mm[1]:.1f}, {display_position_mm[2]:.1f}) mm [{coord_label}]")
    
    def _grasp_specific_object(self, object_id):
        """Start non-blocking grasping sequence for a specific object ID."""
        if not self._is_robot_available():
            print("Robot not connected - cannot grasp object")
            return
        
        if self.pointing_in_progress or self.grasping_in_progress:
            print("Robot action already in progress")
            return
        
        if not self.last_detection_objects:
            print("No objects detected for grasping")
            return
        
        # Find object by visual cluster ID (not array index)
        target_object = None
        for obj in self.last_detection_objects:
            if obj.get('id') == object_id:
                target_object = obj
                break
        
        if target_object is None:
            # Show available object IDs for better error message
            available_ids = [obj.get('id', 'Unknown') for obj in self.last_detection_objects]
            print(f"Invalid object ID: {object_id}. Available objects: {available_ids}")
            return
        
        # Print all detected object centers before grasping
        print(f"\n{'='*60}")
        print(f"GRASPING SPECIFIC OBJECT {object_id} - {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        # Get current robot pose for coordinate transformation
        try:
            robot_pose = self.get_robot_pose()
            robot_transform = self.pose_to_transform_matrix(robot_pose) if robot_pose else None
        except Exception as e:
            print(f"Warning: Could not get robot transform for consistent coordinates: {e}")
            robot_transform = None
        
        print(f"Detected {len(self.last_detection_objects)} object(s):")
        for i, obj in enumerate(self.last_detection_objects):
            obj_pos = self._extract_object_position(obj)
            if obj_pos is not None:
                marker = "→ TARGET" if obj.get('id') == object_id else "  "
                if robot_transform is not None:
                    obj_pos_robot_mm = self.predict_object_position(obj_pos, robot_transform)
                    confidence = obj.get('confidence', 0.0)
                    size = obj.get('size', 0)
                    print(f"{marker} Object #{obj.get('id', i+1)}: ({obj_pos_robot_mm[0]:6.1f}, {obj_pos_robot_mm[1]:6.1f}, {obj_pos_robot_mm[2]:6.1f}) mm, size={size:3d}, conf={confidence:.2f}")
                else:
                    obj_pos_mm = [pos * 1000 for pos in obj_pos]
                    confidence = obj.get('confidence', 0.0)
                    size = obj.get('size', 0)
                    print(f"{marker} Object #{obj.get('id', i+1)}: ({obj_pos_mm[0]:6.1f}, {obj_pos_mm[1]:6.1f}, {obj_pos_mm[2]:6.1f}) mm [camera], size={size:3d}, conf={confidence:.2f}")
        
        # Extract target object position
        camera_position_m = self._extract_object_position(target_object)
        if camera_position_m is None:
            print("Could not extract target object position")
            return
        
        # Get robot coordinates for display consistency
        if robot_transform is not None:
            robot_position_mm = self.predict_object_position(camera_position_m, robot_transform)
            display_position_mm = robot_position_mm
            coord_label = "robot"
        else:
            display_position_mm = [pos * 1000 for pos in camera_position_m]
            coord_label = "camera"
        
        print(f"\n→ TARGETING Object {object_id} (user selected):")
        print(f"  Position: ({display_position_mm[0]:6.1f}, {display_position_mm[1]:6.1f}, {display_position_mm[2]:6.1f}) mm [{coord_label}]")
        print(f"  Size: {target_object.get('size', 0)} points")
        print(f"  Confidence: {target_object.get('confidence', 0.0):.2f}")
        print(f"{'='*60}")
        
        # Initialize non-blocking grasping state machine
        self.grasping_in_progress = True
        self.grasping_state = 'starting'
        self.grasping_object = {
            'camera_position_m': camera_position_m,
            'display_position_mm': display_position_mm,
            'coord_label': coord_label,
            'closest_object': target_object,
            'robot_position_mm': display_position_mm if coord_label == "robot" else None
        }
        self.grasping_step_start_time = time.time()
        self.pointing_start_time = time.time()
        
        print(f"\n✓ Grasping sequence started for Object {object_id}")
        print(f"  Target object: Object {object_id} at ({display_position_mm[0]:.1f}, {display_position_mm[1]:.1f}, {display_position_mm[2]:.1f}) mm [{coord_label}]")

    def _process_grasping_state_machine(self):
        """Process non-blocking grasping state machine - called every frame."""
        if not self.grasping_in_progress or not self.grasping_state:
            return
            
        current_time = time.time()
        step_elapsed = current_time - self.grasping_step_start_time
        
        # Check for timeout
        if step_elapsed > self.grasping_timeout:
            print(f"Grasping step '{self.grasping_state}' timed out after {self.grasping_timeout}s")
            self._finish_grasping(success=False)
            return
        
        try:
            if self.grasping_state == 'starting':
                self._start_hover_movement()
            elif self.grasping_state == 'hovering':
                self._check_hover_completion()
            elif self.grasping_state == 'opening':
                self._check_gripper_opening()
            elif self.grasping_state == 'descending':
                self._check_descent_completion()
            elif self.grasping_state == 'waiting_to_close':
                self._check_waiting_to_close()
            elif self.grasping_state == 'closing':
                self._check_gripper_closing()
            elif self.grasping_state == 'lifting':
                self._check_lift_movement()
            elif self.grasping_state == 'dropping':
                self._check_drop_movement()
            elif self.grasping_state == 'waiting_to_drop':
                self._check_waiting_to_drop()
            elif self.grasping_state == 'drop_opening':
                self._check_drop_opening()
            elif self.grasping_state == 'drop_closing':
                self._check_drop_closing()
            elif self.grasping_state == 'returning':
                self._check_return_movement()
        except Exception as e:
            print(f"Error in grasping state '{self.grasping_state}': {e}")
            self._finish_grasping(success=False)
    
    def _start_hover_movement(self):
        """Start movement to hover position."""
        try:
            # Transform camera coordinates to robot coordinates
            robot_pose = self.get_robot_pose()
            if robot_pose is None:
                print("Robot pose not available, will retry on next frame...")
                # Don't finish grasping, just return and try again on next iteration
                return
            
            robot_transform = self.pose_to_transform_matrix(robot_pose)
            object_center_camera = self.grasping_object['camera_position_m']
            robot_position_mm = self.predict_object_position(object_center_camera, robot_transform)
            
            # Store robot position for later steps
            self.grasping_object['robot_position_mm'] = robot_position_mm
            
            print(f"Coordinate transformation: Camera {[f'{x:.1f}' for x in [pos * 1000 for pos in self.grasping_object['camera_position_m']]]} → Robot [{robot_position_mm[0]:.1f}, {robot_position_mm[1]:.1f}, {robot_position_mm[2]:.1f}] mm")
            
            # Create hover position (150mm height)
            hover_height = 150
            hover_position = [robot_position_mm[0], robot_position_mm[1], hover_height]
            hover_pose = hover_position + [0, 180, 45]
            
            print("1. Moving to hover position above object...")
            
            # Submit async command
            if hasattr(self.enhanced_robot_controller, 'move_to_position_async'):
                self.grasping_command_id = self.enhanced_robot_controller.move_to_position_async(
                    hover_pose, speed=self.movement_speed, priority=2)
                if self.grasping_command_id:
                    self.grasping_state = 'hovering'
                    self.grasping_step_start_time = time.time()
                else:
                    print("✗ Failed to submit hover command")
                    self._finish_grasping(success=False)
            else:
                print("✗ Async movement not available")
                self._finish_grasping(success=False)
                
        except Exception as e:
            print(f"Error starting hover movement: {e}")
            self._finish_grasping(success=False)
    
    def _check_hover_completion(self):
        """Check if hover movement is complete and stable."""
        elapsed = time.time() - self.grasping_step_start_time
        
        # Debug: Show progress every 2 seconds
        if int(elapsed) % 2 == 0 and int(elapsed) != getattr(self, '_last_debug_time', -1):
            self._last_debug_time = int(elapsed)
            print(f"  Hover progress: {elapsed:.1f}s elapsed, waiting for completion...")
        
        if self.grasping_command_id and hasattr(self.enhanced_robot_controller, 'get_command_status'):
            status = self.enhanced_robot_controller.get_command_status(self.grasping_command_id)
            if status and hasattr(status, 'value'):
                if status.value == 'completed':
                    # Wait additional time for stability after movement completion
                    if elapsed > self.hover_stability_wait:  # Minimum wait for stability
                        print("✓ Reached hover position and stabilized")
                        self._start_gripper_opening()
                elif status.value == 'failed':
                    print("✗ Failed to reach hover position")
                    self._finish_grasping(success=False)
        else:
            # Fallback: assume completed after longer time for stability
            if elapsed > self.hover_fallback_wait:
                print("✓ Hover movement assumed complete and stable")
                self._start_gripper_opening()
    
    def _start_gripper_opening(self):
        """Start gripper opening."""
        print("2. Opening gripper...")
        try:
            if self.enhanced_robot_controller.open_gripper():
                self.grasping_state = 'opening'
                self.grasping_step_start_time = time.time()
                # Allow 1.5 seconds for gripper opening
            else:
                print("✗ Failed to start gripper opening")
                self._finish_grasping(success=False)
        except Exception as e:
            print(f"Error opening gripper: {e}")
            self._finish_grasping(success=False)
    
    def _check_gripper_opening(self):
        """Check if gripper opening is complete."""
        if time.time() - self.grasping_step_start_time > self.gripper_open_wait:
            print("✓ Gripper opened")
            self._start_grasp_movement()
    
    def _start_grasp_movement(self):
        """Start straight-down descent to grasp position using current robot position."""
        print("3. Descending straight down to grasp object...")
        try:
            # Get current actual robot position after hover is stable
            current_pose = self.enhanced_robot_controller.get_current_position()
            if not current_pose:
                print("✗ Failed to get current robot position for descent")
                self._finish_grasping(success=False)
                return
            
            # Calculate grasp height based on original object detection - go much lower
            robot_position_mm = self.grasping_object['robot_position_mm']
            grasp_height = max(robot_position_mm[2] - 40, 0)  # Go 10mm below object, minimum 40mm
            
            # Use current X,Y coordinates but change only Z for straight down movement
            descent_position = [current_pose[0], current_pose[1], grasp_height]
            descent_pose = descent_position + [0, 180, 45]  # Keep same orientation
            
            print(f"Straight descent: X={current_pose[0]:.1f}, Y={current_pose[1]:.1f}, Z: {current_pose[2]:.1f}mm → {grasp_height}mm")
            
            self.grasping_command_id = self.enhanced_robot_controller.move_to_position_async(
                descent_pose, speed=self.careful_speed, priority=2, linear_mode=True)  # Linear mode for straight descent
            if self.grasping_command_id:
                self.grasping_state = 'descending'
                self.grasping_step_start_time = time.time()
            else:
                print("✗ Failed to submit descent command")
                self._finish_grasping(success=False)
        except Exception as e:
            print(f"Error starting descent movement: {e}")
            self._finish_grasping(success=False)
    
    def _check_descent_completion(self):
        """Check if straight-down descent is complete."""
        if self.grasping_command_id and hasattr(self.enhanced_robot_controller, 'get_command_status'):
            status = self.enhanced_robot_controller.get_command_status(self.grasping_command_id)
            if status and hasattr(status, 'value'):
                if status.value == 'completed':
                    print("✓ Reached grasp position via straight descent")
                    print(f"⏰ Waiting {self.object_positioning_wait} seconds for object positioning before closing gripper...")
                    self.grasping_state = 'waiting_to_close'  # New intermediate state
                    self.grasping_step_start_time = time.time()  # Reset timer for wait period
                elif status.value == 'failed':
                    print("✗ Failed to complete descent")
                    self._finish_grasping(success=False)
        else:
            # Fallback: assume completed after reasonable time
            if time.time() - self.grasping_step_start_time > self.descent_timeout:  # Descent movement timeout
                print("✓ Descent movement assumed complete")
                print(f"⏰ Waiting {self.object_positioning_wait} seconds for object positioning before closing gripper...")
                self.grasping_state = 'waiting_to_close'  # New intermediate state
                self.grasping_step_start_time = time.time()  # Reset timer for wait period
    
    def _check_waiting_to_close(self):
        """Wait for object to stabilize before closing gripper."""
        elapsed = time.time() - self.grasping_step_start_time
        if elapsed > self.object_positioning_wait:  # Wait for object positioning
            print("✓ Wait period complete - ready to close gripper")
            self._start_gripper_closing()
    
    def _start_gripper_closing(self):
        """Start gripper closing."""
        print("4. Closing gripper...")
        try:
            if self.enhanced_robot_controller.close_gripper():
                self.grasping_state = 'closing'
                self.grasping_step_start_time = time.time()
                # Allow 2.5 seconds for gripper closing and securing object
            else:
                print("✗ Failed to start gripper closing")
                self._finish_grasping(success=False)
        except Exception as e:
            print(f"Error closing gripper: {e}")
            self._finish_grasping(success=False)
    
    def _check_gripper_closing(self):
        """Check if gripper closing is complete."""
        if time.time() - self.grasping_step_start_time > self.gripper_close_wait:
            print("✓ Gripper closed - object grasped")
            self._start_lift_movement()
    
    def _start_lift_movement(self):
        """Start lifting movement."""
        print("5. Lifting object...")
        try:
            robot_position_mm = self.grasping_object['robot_position_mm']
            lift_height = 250
            lift_position = [robot_position_mm[0], robot_position_mm[1], lift_height]
            lift_pose = lift_position + [0, 180, 45]
            
            self.grasping_command_id = self.enhanced_robot_controller.move_to_position_async(
                lift_pose, speed=self.careful_speed, priority=2)
            if self.grasping_command_id:
                self.grasping_state = 'lifting'
                self.grasping_step_start_time = time.time()
            else:
                print("✗ Failed to submit lift command")
                self._finish_grasping(success=False)
        except Exception as e:
            print(f"Error starting lift movement: {e}")
            self._finish_grasping(success=False)
    
    def _check_lift_movement(self):
        """Check if lift movement is complete."""
        if self.grasping_command_id and hasattr(self.enhanced_robot_controller, 'get_command_status'):
            status = self.enhanced_robot_controller.get_command_status(self.grasping_command_id)
            if status and hasattr(status, 'value'):
                if status.value == 'completed':
                    print("✓ Object lifted")
                    self._start_drop_movement()
                elif status.value == 'failed':
                    print("✗ Failed to lift object")
                    self._finish_grasping(success=False)
        else:
            # Fallback: assume completed after reasonable time
            if time.time() - self.grasping_step_start_time > self.lift_timeout:
                print("✓ Lift assumed complete")
                self._start_drop_movement()
    
    def _start_drop_movement(self):
        """Start movement to drop position."""
        print("5. Moving to drop position...")
        try:
            self.grasping_command_id = self.enhanced_robot_controller.move_to_position_async(
                self.drop_position, speed=self.movement_speed, priority=2)
            if self.grasping_command_id:
                self.grasping_state = 'dropping'
                self.grasping_step_start_time = time.time()
            else:
                print("✗ Failed to submit drop movement command")
                self._finish_grasping(success=False)
        except Exception as e:
            print(f"Error starting drop movement: {e}")
            self._finish_grasping(success=False)
    
    def _check_drop_movement(self):
        """Check if drop movement is complete."""
        if self.grasping_command_id and hasattr(self.enhanced_robot_controller, 'get_command_status'):
            status = self.enhanced_robot_controller.get_command_status(self.grasping_command_id)
            if status and hasattr(status, 'value'):
                if status.value == 'completed':
                    print("✓ Reached drop position")
                    print(f"⏰ Waiting {self.drop_stabilization_wait} seconds for robot stabilization at drop position...")
                    self.grasping_state = 'waiting_to_drop'
                    self.grasping_step_start_time = time.time()
                elif status.value == 'failed':
                    print("✗ Failed to reach drop position")
                    self._finish_grasping(success=False)
        else:
            # Fallback: assume completed after reasonable time
            if time.time() - self.grasping_step_start_time > self.drop_movement_timeout:
                print("✓ Drop movement assumed complete")
                print(f"⏰ Waiting {self.drop_stabilization_wait} seconds for robot stabilization at drop position...")
                self.grasping_state = 'waiting_to_drop'
                self.grasping_step_start_time = time.time()
    
    def _check_waiting_to_drop(self):
        """Wait for robot to stabilize at drop position before opening gripper."""
        if time.time() - self.grasping_step_start_time > self.drop_stabilization_wait:
            print("✓ Robot stabilized at drop position - ready to drop object")
            self._start_drop_opening()
    
    def _start_drop_opening(self):
        """Start gripper opening to drop the object."""
        print("6. Opening gripper to drop object...")
        try:
            if self.enhanced_robot_controller.open_gripper():
                self.grasping_state = 'drop_opening'
                self.grasping_step_start_time = time.time()
                # Allow 2.0 seconds for gripper opening
            else:
                print("✗ Failed to start gripper opening for drop")
                self._finish_grasping(success=False)
        except Exception as e:
            print(f"Error opening gripper for drop: {e}")
            self._finish_grasping(success=False)
    
    def _check_drop_opening(self):
        """Check if gripper opening for drop is complete."""
        if time.time() - self.grasping_step_start_time > self.drop_gripper_open_wait:
            print("✓ Gripper opened - object dropped")
            self._start_drop_closing()

    def _start_drop_closing(self):
        """Start gripper closing after dropping object."""
        print("7. Closing gripper after drop...")
        try:
            if self.enhanced_robot_controller.close_gripper():
                self.grasping_state = 'drop_closing'
                self.grasping_step_start_time = time.time()
                # Allow 2.0 seconds for gripper closing
            else:
                print("✗ Failed to start gripper closing after drop")
                self._finish_grasping(success=False)
        except Exception as e:
            print(f"Error closing gripper after drop: {e}")
            self._finish_grasping(success=False)
    
    def _check_drop_closing(self):
        """Check if gripper closing after drop is complete."""
        if time.time() - self.grasping_step_start_time > self.drop_gripper_close_wait:
            print("✓ Gripper closed after drop")
            self._start_return_movement()

    def _start_return_movement(self):
        """Start return to observation pose."""
        print("8. Returning to observation pose...")
        try:
            from module.utils.robot_utils import move_to_standard_pose
            success = move_to_standard_pose(self.enhanced_robot_controller, "table", async_mode=True)
            if success:
                self.grasping_state = 'returning'
                self.grasping_step_start_time = time.time()
            else:
                print("✗ Failed to start return movement")
                self._finish_grasping(success=False)
        except Exception as e:
            print(f"Error starting return movement: {e}")
            self._finish_grasping(success=False)
    
    def _check_return_movement(self):
        """Check if return movement is complete."""
        # Give reasonable time for return movement
        if time.time() - self.grasping_step_start_time > self.return_timeout:
            print("✓ Returned to observation pose")
            self._finish_grasping(success=True)
    
    def _finish_grasping(self, success=True):
        """Finish grasping sequence and reset state."""
        self.grasping_in_progress = False
        self.grasping_state = None
        self.grasping_command_id = None
        self.grasping_object = None
        self.grasping_step_start_time = None
        
        if success:
            print(f"{'='*60}")
            print("GRASPING COMPLETED SUCCESSFULLY")
            print(f"{'='*60}")
            self.performance_stats['grasping_commands'] += 1
        else:
            print(f"{'='*60}")
            print("GRASPING FAILED - SEQUENCE ABORTED")
            print(f"{'='*60}")
            
    def _perform_sync_grasping_sequence(self, camera_position_m):
        """Perform synchronous grasping sequence that doesn't block RGB feed."""
        try:
            # Transform camera coordinates to robot coordinates using the eye-in-hand calibration
            robot_pose = self.get_robot_pose()
            if robot_pose is None:
                print("Failed to get current robot pose")
                return
            
            robot_transform = self.pose_to_transform_matrix(robot_pose)
            
            # Use camera position directly (already in meters)
            object_center_camera = camera_position_m  # Already in correct format
            
            # Transform to robot base coordinates
            robot_position_mm = self.predict_object_position(object_center_camera, robot_transform)
            robot_position = [robot_position_mm[0], robot_position_mm[1], robot_position_mm[2]]
            
            # Convert to mm for display
            camera_position_mm = [pos * 1000 for pos in camera_position_m]
            print(f"Sync grasping - Camera: {camera_position_mm} -> Robot: [{robot_position[0]:.1f}, {robot_position[1]:.1f}, {robot_position[2]:.1f}] mm")
            
            # Simple synchronous sequence - each step is quick to not block RGB
            print("1. Quick grasp attempt (sync mode)")
            
            # Create grasp position (safe height)
            grasp_height = max(robot_position[2] + 30, 80)  # At least 80mm height for safety
            grasp_position = [robot_position[0], robot_position[1], grasp_height]
            grasp_pose = grasp_position + [0, 180, 45]
            
            # Quick movement to grasp position
            if hasattr(self.enhanced_robot_controller, 'move_to_position'):
                success = self.enhanced_robot_controller.move_to_position(grasp_pose, speed=self.movement_speed)
                if success:
                    print("✓ Reached grasp position")
                    
                    # Quick gripper operations
                    if hasattr(self.enhanced_robot_controller, 'open_gripper'):
                        self.enhanced_robot_controller.open_gripper()
                        time.sleep(0.5)  # Quick pause
                        
                    if hasattr(self.enhanced_robot_controller, 'close_gripper'):
                        self.enhanced_robot_controller.close_gripper()
                        time.sleep(0.5)  # Quick pause
                        
                    print("✓ Synchronous grasp completed")
                else:
                    print("✗ Failed to reach grasp position")
            else:
                print("✗ No movement method available")
                
        except Exception as e:
            print(f"Error in sync grasping: {e}")
    
    def _wait_for_robot_movement(self, command_id: int, timeout: float = 10.0, stabilization_delay: float = 2.0):
        """Wait for robot movement command to complete and stabilize."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.enhanced_robot_controller.get_command_status(command_id)
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
    
    def _rotate_joint(self, joint_index: int, angle_delta: float, message: str):
        """
        Rotate a specific joint by the given angle delta.
        Adapted from virtual_fixed_tracking system for camera tilt control.
        
        Args:
            joint_index: Index of joint to rotate (0-5)
            angle_delta: Angle change in degrees (positive/negative)
            message: Message to display for this movement
        """
        if not self.enhanced_robot_controller or not self.enhanced_robot_controller.is_connected:
            print("Cannot move: Robot not connected")
            return
        
        if not self.base_joint_angles:
            print("Cannot move: Base joint angles unknown")
            return
        
        # Get current joint angles
        try:
            current_angles = self.enhanced_robot_controller.get_joint_angles()
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
            # Try async first if available
            if hasattr(self.enhanced_robot_controller, 'move_to_angles_async'):
                command_id = self.enhanced_robot_controller.move_to_angles_async(new_angles, speed=self.movement_speed)
                if command_id:
                    print(f"Joint movement command queued (ID: {command_id})")
                else:
                    print("Failed to queue joint movement command - trying sync")
                    # Fallback to sync
                    self.enhanced_robot_controller.move_to_angles(new_angles, speed=self.movement_speed)
                    print("Joint movement executed synchronously")
            else:
                # Synchronous movement as fallback
                self.enhanced_robot_controller.move_to_angles(new_angles, speed=self.movement_speed)
                print("Joint movement executed synchronously")
        except Exception as e:
            print(f"Error executing joint movement: {e}")
        
        self.last_movement_time = time.time()
    
    def _find_closest_object(self, detected_objects):
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
            distance = np.sqrt(np.sum(np.array(position)**2))
            
            if distance < min_distance:
                min_distance = distance
                closest_object = obj
        
        return closest_object
    
    def _extract_object_position(self, obj):
        """Extract 3D position from object in camera coordinates (returns in meters, same format as obj['center'])."""
        try:
            if 'center' in obj:
                # center format: [x, y, z] in meters - use directly (no reordering)
                center = obj['center']
                return center  # Return as-is in meters
            elif 'center_mm' in obj:
                # center_mm format: [x, y, z] in mm - convert to meters
                center_mm = obj['center_mm']
                return [center_mm[0]/1000.0, center_mm[1]/1000.0, center_mm[2]/1000.0]
            else:
                return None
        except Exception as e:
            print(f"Error extracting object position: {e}")
            return None
    
    def _check_pointing_timeout(self):
        """Check if pointing duration has elapsed and start return to observation pose."""
        if self.pointing_in_progress or self.grasping_in_progress:
            elapsed_time = time.time() - self.pointing_start_time
            if elapsed_time >= self.pointing_duration:
                action_type = "Grasping" if self.grasping_in_progress else "Pointing"
                print(f"2. {action_type} duration ({self.pointing_duration}s) elapsed, returning to observation pose...")
                
                if self.enhanced_robot_controller:
                    from module.utils.robot_utils import move_to_standard_pose
                    move_to_standard_pose(self.enhanced_robot_controller, "table", async_mode=True)
                
                # Start return-to-observer sequence
                self.pointing_in_progress = False
                self.grasping_in_progress = False
                self.returning_to_observer = True
                self.return_start_time = time.time()
                self.robot_in_observation_pose = False  # Robot is moving, not yet stable
                print("[OK] Return to observer pose initiated, waiting for stabilization...")
    
    def _check_return_stabilization(self):
        """Check if robot has stabilized after returning to observer pose."""
        if self.returning_to_observer:
            elapsed_time = time.time() - self.return_start_time
            if elapsed_time >= self.stabilization_time:
                print(f"3. Robot stabilized after {self.stabilization_time}s")
                
                # Mark robot as back in observation pose and reset state
                self.returning_to_observer = False
                self.robot_in_observation_pose = True
                print("[OK] Point/grasp and return sequence fully completed")
    
    def create_2x2_visualization(self, color_image, depth_frame, results, robot_pose, position_in_base, object_positions=None):
        """
        Create enhanced 2x2 visualization layout: RGB + Depth + ROI Cloud + Object Cluster.
        
        Layout:
        ┌─────────────┬─────────────┐
        │ RGB Camera  │ Depth Image │ ← Fixed: No double resize
        ├─────────────┼─────────────┤
        │ ROI Cloud   │ Object View │ ← Fixed: ROI points instead of noisy full cloud
        └─────────────┴─────────────┘
        
        Args:
            color_image: RGB camera image
            depth_frame: RealSense depth frame (aligned to color)
            results: Detection results from detect_tabletop_object
            robot_pose: Current robot pose information
            position_in_base: Predicted object position in robot base coordinates (mm) [legacy]
            object_positions: List of all object positions with IDs [new multi-object support]
            
        Returns:
            np.array: Combined 2x2 visualization image
        """
        try:
            panel_size = 400
            
            # Get object clusters for all visualizations
            object_clusters = results.get('objects', [])
            
            # Initialize 2x2 grid panels (reorganized layout)
            top_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)    # RGB
            top_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)   # Depth Image  
            bottom_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8) # Point Cloud (always visible)
            bottom_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8) # Object Cluster
            
            # Top-left: RGB image with tracking info overlay
            if color_image is not None:
                rgb_panel = color_image.copy()
                # Add tracking info overlay with robot status
                self._add_tracking_info_overlay(rgb_panel, results, robot_pose, position_in_base)
                # Add object detection overlay with position information
                self._add_object_detection_overlay(rgb_panel, results, object_positions)
                
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
            
            # Top-right: Depth image view with cluster markers
            depth_view = self._create_depth_image_view(depth_frame, object_clusters)
            if depth_view is not None:
                # _create_depth_image_view already returns a properly sized panel, don't resize again
                top_right = depth_view
            else:
                cv2.putText(top_right, "DEPTH IMAGE", (80, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                cv2.putText(top_right, "NO DATA", (120, panel_size//2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            
            # Bottom-left: ROI point cloud view (filtered objects) with cluster coloring
            roi_points = results.get('roi_points', results.get('full_point_cloud'))  # Use ROI points if available
            lidar_view = self._create_lidar_point_cloud_view(roi_points, object_clusters)
            if lidar_view is not None:
                bottom_left = lidar_view
            else:
                cv2.putText(bottom_left, "ROI POINT CLOUD", (60, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                cv2.putText(bottom_left, "NO DATA", (120, panel_size//2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            
            # Bottom-right: Individual cluster view (show largest cluster or all clusters)
            if object_clusters:
                # Show object points with cluster coloring
                object_cluster_view = self._create_lidar_point_cloud_view(results.get('object_points'), object_clusters)
                if object_cluster_view is not None:
                    bottom_right = object_cluster_view
                else:
                    cv2.putText(bottom_right, f"{len(object_clusters)} CLUSTERS", (70, panel_size//2-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                    cv2.putText(bottom_right, f"DETECTED", (120, panel_size//2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            else:
                cv2.putText(bottom_right, "OBJECT CLUSTERS", (60, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                cv2.putText(bottom_right, "NO OBJECTS DETECTED", (40, panel_size//2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
            
            # Combine into 2x2 grid
            top_row = np.hstack([top_left, top_right])
            bottom_row = np.hstack([bottom_left, bottom_right])
            combined = np.vstack([top_row, bottom_row])
            
            return combined
            
        except Exception as e:
            print(f"Warning: 2x2 visualization error: {e}")
            # Fallback to simple color image display
            if color_image is not None:
                return color_image
            else:
                fallback = np.zeros((400, 400, 3), dtype=np.uint8)
                cv2.putText(fallback, "VISUALIZATION ERROR", (50, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                return fallback
    
    def _add_tracking_info_overlay(self, image, results, robot_pose, position_in_base):
        """Add enhanced tracking information overlay to the image with robot status."""
        h, w = image.shape[:2]
        
        # Create overlay for text information
        overlay = image.copy()
        alpha = 0.7
        
        # Main info panel
        panel_width = min(400, w - 20)
        panel_height = min(240, h - 40)  # Increased height for robot status
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (255, 255, 255), 2)
        
        # Blend overlay
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Title
        cv2.putText(image, "Enhanced LiDAR Object Position Prediction", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Detection status
        y = 60
        if results.get('valid_detection', False):
            object_clusters = results.get('objects', [])
            num_objects = len(object_clusters)
            
            # Multi-object status
            if num_objects > 1:
                cv2.putText(image, f"Status: {num_objects} OBJECTS DETECTED", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(image, "Status: 1 OBJECT DETECTED", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show brief info for each object (limit to top 3 to fit on screen)
            y += 20
            max_objects_to_show = min(3, num_objects)
            
            for i, cluster in enumerate(object_clusters[:max_objects_to_show]):
                cluster_id = cluster['id']
                cluster_size = cluster['size']
                cluster_color = cluster['color']
                color_name = self._get_color_name(cluster_color)
                
                # Object info line
                y += 16
                cv2.putText(image, f"#{cluster_id}: {color_name} ({cluster_size} pts)", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, cluster_color, 1)
            
            # Show "and X more..." if we have more than 3 objects
            if num_objects > max_objects_to_show:
                y += 16
                remaining = num_objects - max_objects_to_show
                cv2.putText(image, f"... and {remaining} more", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show main object position (largest cluster)
            if position_in_base is not None and object_clusters:
                main_cluster = object_clusters[0]  # Largest cluster
                y += 25
                cv2.putText(image, f"Main object position (mm):", (20, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y += 18
                cv2.putText(image, f"  X(forward): {position_in_base[0]:7.1f}", (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                y += 16
                cv2.putText(image, f"  Y(left):    {position_in_base[1]:7.1f}", (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y += 16
                cv2.putText(image, f"  Z(up):      {position_in_base[2]:7.1f}", (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Position stability for main object
                mean_pos, std_pos = self.get_position_stability()
                if mean_pos is not None:
                    y += 18
                    cv2.putText(image, f"Stability: ±{std_pos[0]:.1f}, ±{std_pos[1]:.1f}, ±{std_pos[2]:.1f}mm",
                               (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        else:
            cv2.putText(image, "Status: Searching for objects...", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Enhanced Robot status
        y += 30
        if self._is_robot_available():
            if self.grasping_in_progress:
                robot_status = "GRASPING OBJECT"
                status_color = (0, 165, 255)  # Orange
            elif self.pointing_in_progress:
                robot_status = "POINTING AT OBJECT"
                status_color = (0, 255, 255)  # Yellow
            elif self.returning_to_observer:
                robot_status = "RETURNING TO POSE"
                status_color = (255, 255, 0)  # Cyan
            elif self.robot_in_observation_pose:
                robot_status = "OBSERVING"
                status_color = (0, 255, 0)  # Green
            else:
                robot_status = "CONNECTED"
                status_color = (0, 255, 0)  # Green
                
            cv2.putText(image, f"Robot: {robot_status}", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        else:
            cv2.putText(image, f"Robot: Not connected", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Controls info with enhanced controls
        cv2.putText(image, "Controls: 'p'=Point, 'g'=Grasp, 'w'/'s'=Tilt, 'f'=Forward, 'r'=Return, 'q'=Quit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Add prominent mode indicators for robot actions
        if self.grasping_in_progress:
            cv2.putText(image, "*** GRASPING MODE - RGB LIVE, 3D PAUSED ***", 
                       (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        elif self.pointing_in_progress:
            cv2.putText(image, "*** POINTING MODE - RGB LIVE, 3D PAUSED ***", 
                       (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif self.returning_to_observer:
            cv2.putText(image, "*** RETURNING TO POSE - RGB LIVE ***", 
                       (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        return image
    
    def _add_object_detection_overlay(self, image, results, object_positions=None):
        """Add multi-object detection overlay with colored markers, IDs, and positions."""
        if not results['valid_detection']:
            return
        
        h, w = image.shape[:2]
        object_clusters = results.get('objects', [])
        
        # Create position lookup for easy access
        position_lookup = {}
        if object_positions:
            for pos_info in object_positions:
                position_lookup[pos_info['id']] = pos_info['position']
        
        # Draw markers for each detected object/cluster
        for cluster in object_clusters:
            center = cluster['center']
            cluster_color = cluster['color']
            cluster_id = cluster['id']
            cluster_size = cluster['size']
            
            # Project the 3D object center to 2D pixel coordinates
            pixel = self._project_point_to_pixel(center, self.camera_utils.color_intrinsics)
            if pixel is not None:
                center_x, center_y = pixel
                # Ensure the pixel is within the image bounds before drawing
                if 0 <= center_x < w and 0 <= center_y < h:
                    # Draw colored circles for the cluster
                    cv2.circle(image, (center_x, center_y), 15, cluster_color, 3)
                    cv2.circle(image, (center_x, center_y), 25, cluster_color, 2)
                    
                    # Add white outline for visibility
                    cv2.circle(image, (center_x, center_y), 27, (255, 255, 255), 1)
                    
                    # Add cluster ID label
                    label_text = f"#{cluster_id}"
                    cv2.putText(image, label_text, (center_x + 30, center_y - 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(image, label_text, (center_x + 30, center_y - 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, cluster_color, 1)
                    
                    # Add size information
                    size_text = f"{cluster_size}pts"
                    cv2.putText(image, size_text, (center_x + 30, center_y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Add predicted position if available
                    if cluster_id in position_lookup:
                        pos = position_lookup[cluster_id]
                        pos_text = f"({pos[0]:3.0f},{pos[1]:3.0f},{pos[2]:3.0f})"
                        cv2.putText(image, pos_text, (center_x + 30, center_y + 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    def _downsample_points_for_visualization(self, point_cloud):
        """Downsample point cloud for visualization to improve performance."""
        if point_cloud is None or len(point_cloud) == 0:
            return point_cloud
        
        # If point cloud is small enough, return as-is
        if len(point_cloud) <= self.viz_max_points:
            return point_cloud
        
        # Method 1: Regular downsampling (every Nth point)
        if len(point_cloud) <= self.viz_max_points * self.viz_downsample_factor:
            return point_cloud[::self.viz_downsample_factor]
        
        # Method 2: Random sampling for very dense point clouds
        indices = np.random.choice(len(point_cloud), self.viz_max_points, replace=False)
        return point_cloud[indices]

    def _create_lidar_point_cloud_view(self, point_cloud, object_clusters=None):
        """Create LiDAR point cloud visualization with cluster-based color coding."""
        if point_cloud is None or len(point_cloud) == 0:
            return None
        
        try:
            viz_start_time = time.time()
            
            # Filter out invalid points
            valid_mask = ~np.any(np.isnan(point_cloud) | np.isinf(point_cloud), axis=1)
            valid_vertices = point_cloud[valid_mask]
            if len(valid_vertices) < 10:
                return None
            
            # Downsample for visualization performance
            valid_vertices = self._downsample_points_for_visualization(valid_vertices)
            
            # --- Use Color Intrinsics for Projection (since depth is aligned to color) ---
            if not hasattr(self, 'camera_utils') or self.camera_utils.color_intrinsics is None:
                print("Warning: Color intrinsics not available for point cloud visualization.")
                return None
            
            intrinsics = self.camera_utils.color_intrinsics
            original_width = intrinsics.width
            original_height = intrinsics.height
            
            # OPTIMIZED: Create smaller projection canvas for faster rendering
            viz_scale_factor = 0.5  # Use half resolution for visualization
            viz_width = int(original_width * viz_scale_factor)
            viz_height = int(original_height * viz_scale_factor)
            proj_canvas = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)
            
            # Scale intrinsics for visualization canvas
            fx = intrinsics.fx * viz_scale_factor
            fy = intrinsics.fy * viz_scale_factor
            cx = intrinsics.ppx * viz_scale_factor
            cy = intrinsics.ppy * viz_scale_factor
            
            # If we have cluster information, use cluster colors
            if object_clusters and len(object_clusters) > 0:
                # OPTIMIZED: Use vectorized operations instead of point-by-point mapping
                # Project all points first
                x_3d = valid_vertices[:, 0]
                y_3d = valid_vertices[:, 1] 
                z_3d = np.maximum(valid_vertices[:, 2], 0.001)
                
                x_2d = (fx * x_3d / z_3d) + cx
                y_2d = (fy * y_3d / z_3d) + cy
                x_pixels = np.clip(x_2d.astype(int), 0, viz_width - 1)
                y_pixels = np.clip(y_2d.astype(int), 0, viz_height - 1)
                
                # Create mask for valid projections
                valid_proj = (x_pixels >= 0) & (x_pixels < viz_width) & (y_pixels >= 0) & (y_pixels < viz_height)
                
                # Default all points to gray
                colors = np.full((len(valid_vertices), 3), (128, 128, 128), dtype=np.uint8)
                
                # Assign cluster colors using distance-based assignment (faster than dictionary lookup)
                for cluster in object_clusters:
                    if len(cluster['points']) > 0:
                        cluster_points = np.array(cluster['points'])
                        cluster_color = cluster['color']
                        
                        # Use broadcasting to find closest cluster points (simplified)
                        # For visualization, we'll use a radius-based approach instead of exact matching
                        center = np.mean(cluster_points, axis=0)
                        distances = np.linalg.norm(valid_vertices - center, axis=1)
                        radius = np.std(np.linalg.norm(cluster_points - center, axis=1)) * 2.0  # 2 std devs
                        cluster_mask = distances < max(radius, 0.05)  # Minimum 5cm radius
                        
                        colors[cluster_mask] = cluster_color
                
                # Apply colors to canvas using vectorized operations
                valid_idx = np.where(valid_proj)[0]
                if len(valid_idx) > 0:
                    proj_canvas[y_pixels[valid_idx], x_pixels[valid_idx]] = colors[valid_idx]
                
                # Draw cluster centers
                for cluster in object_clusters:
                    center = cluster['center']
                    pixel = self._project_point_to_pixel(center, intrinsics)
                    if pixel is not None:
                        obj_x_pixel, obj_y_pixel = pixel
                        cluster_color = cluster['color']
                        cluster_id = cluster['id']
                        
                        # Draw center marker
                        cv2.circle(proj_canvas, (obj_x_pixel, obj_y_pixel), 8, cluster_color, 3)
                        cv2.circle(proj_canvas, (obj_x_pixel, obj_y_pixel), 15, (255, 255, 255), 2)
                        
                        # Add cluster ID label
                        cv2.putText(proj_canvas, str(cluster_id), 
                                  (obj_x_pixel + 20, obj_y_pixel - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Fallback: use depth-based coloring
                x_3d = valid_vertices[:, 0]
                y_3d = valid_vertices[:, 1]
                z_3d = np.maximum(valid_vertices[:, 2], 0.001)
                x_2d = (fx * x_3d / z_3d) + cx
                y_2d = (fy * y_3d / z_3d) + cy
                x_pixels = np.clip(x_2d.astype(int), 0, viz_width - 1)
                y_pixels = np.clip(y_2d.astype(int), 0, viz_height - 1)
                
                # Color points by depth
                z_min, z_max = np.min(z_3d), np.max(z_3d)
                if z_max > z_min:
                    z_normalized = (z_3d - z_min) / (z_max - z_min)
                    depth_colors = cv2.applyColorMap((z_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                    proj_canvas[y_pixels, x_pixels] = depth_colors.reshape(-1, 3)[range(len(x_pixels))]
            
            # Resize the projection canvas to fit the panel, preserving aspect ratio
            panel_size = 400
            canvas = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
            aspect_ratio = original_width / original_height
            
            if aspect_ratio > 1:  # Landscape
                new_width = panel_size
                new_height = int(panel_size / aspect_ratio)
                resized_proj = cv2.resize(proj_canvas, (new_width, new_height))
                y_offset = (panel_size - new_height) // 2
                canvas[y_offset:y_offset + new_height, :] = resized_proj
            else:  # Portrait or square
                new_height = panel_size
                new_width = int(panel_size * aspect_ratio)
                resized_proj = cv2.resize(proj_canvas, (new_width, new_height))
                x_offset = (panel_size - new_width) // 2
                canvas[:, x_offset:x_offset + new_width] = resized_proj
                
            # Performance monitoring (silent)
            if self.viz_show_performance:
                viz_time = (time.time() - viz_start_time) * 1000  # Convert to milliseconds
                # Timing is tracked but not printed to reduce console noise
            
            return canvas
            
        except Exception as e:
            print(f"Error creating LiDAR point cloud view: {e}")
            return None
    
    def _create_depth_image_view(self, depth_frame, object_clusters=None):
        """Create colorized depth image visualization with multi-object support."""
        if depth_frame is None:
            return None
        
        try:
            # Get depth data as numpy array
            depth_image = np.asanyarray(depth_frame.get_data())
            
            if depth_image is None or depth_image.size == 0:
                return None
            
            canvas_size = 400
            
            # Handle aspect ratio properly when resizing for the panel
            original_height, original_width = depth_image.shape[:2]
            aspect_ratio = original_width / original_height
            
            x_offset, y_offset = 0, 0
            if aspect_ratio > 1:  # Landscape image
                new_width = canvas_size
                new_height = int(canvas_size / aspect_ratio)
                y_offset = (canvas_size - new_height) // 2
            else:  # Portrait image
                new_height = canvas_size  
                new_width = int(canvas_size * aspect_ratio)
                x_offset = (canvas_size - new_width) // 2
            depth_resized = cv2.resize(depth_image, (new_width, new_height))
            
            # Create a canvas and place the resized image in it to maintain aspect ratio
            canvas_with_image = np.zeros((canvas_size, canvas_size), dtype=np.uint16)
            canvas_with_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = depth_resized
            
            # Convert the full canvas to meters for colorization
            depth_m = canvas_with_image.astype(np.float32) / 1000.0
            
            valid_mask = (depth_m > 0)
            
            depth_colorized = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            min_depth, max_depth = 0, 0
            if np.any(valid_mask):
                valid_depths = depth_m[valid_mask]
                min_depth, max_depth = np.min(valid_depths), np.max(valid_depths)
                
                if max_depth > min_depth:
                    depth_normalized = (depth_m - min_depth) / (max_depth - min_depth)
                    depth_normalized = np.clip(depth_normalized, 0, 1)
                    depth_colorized = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                depth_colorized[~valid_mask] = [0, 0, 0]
            
            # Highlight detected object centers if available
            if object_clusters and self.camera_utils.color_intrinsics is not None:
                for cluster in object_clusters:
                    object_center = cluster['center']
                    cluster_color = cluster['color']
                    cluster_id = cluster['id']
                    
                    # Project the 3D object center to 2D pixel coordinates in the *original* frame
                    pixel = self._project_point_to_pixel(object_center, self.camera_utils.color_intrinsics)
                    if pixel is not None:
                        # Map the original coordinates to the final 400x400 panel, accounting for resizing and letterboxing
                        original_pixel_x, original_pixel_y = pixel
                        original_h, original_w = self.camera_utils.color_intrinsics.height, self.camera_utils.color_intrinsics.width
                        # The scaling factor is based on how the original image was fitted into the new dimensions
                        scale = new_width / original_w
                        
                        mapped_x = int(original_pixel_x * scale) + x_offset
                        mapped_y = int(original_pixel_y * scale) + y_offset
                        
                        # Draw object center marker with cluster color
                        cv2.circle(depth_colorized, (mapped_x, mapped_y), 8, cluster_color, 3)
                        cv2.circle(depth_colorized, (mapped_x, mapped_y), 15, (255, 255, 255), 2)
                        cv2.drawMarker(depth_colorized, (mapped_x, mapped_y), cluster_color, 
                                     cv2.MARKER_CROSS, 20, 3)
                        
                        # Add cluster ID label
                        cv2.putText(depth_colorized, f"#{cluster_id}", 
                                  (mapped_x + 25, mapped_y - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return depth_colorized
            
        except Exception as e:
            print(f"Error creating depth image view: {e}")
            return None
    
    def _get_color_name(self, color_tuple):
        """Get human-readable color name from RGB tuple."""
        color_map = {
            (255, 0, 0): "Red",
            (0, 255, 0): "Green", 
            (0, 0, 255): "Blue",
            (0, 255, 255): "Cyan",
            (255, 0, 255): "Magenta",
            (255, 255, 0): "Yellow",
            (255, 165, 0): "Orange",
            (128, 0, 128): "Purple",
            (255, 192, 203): "Pink",
            (0, 128, 128): "Teal"
        }
        return color_map.get(color_tuple, "Unknown")
    
    # === Detection Methods (from original lidar_realtime_prediction.py) ===
    
    def setup_alignment(self):
        """Setup RealSense frame alignment."""
        # Create alignment object to align depth frames to color frames
        self.align = rs.align(rs.stream.color)
    
    def setup_camera(self):
        """Setup RealSense camera with depth and color streams."""
        success = self.camera_utils.setup_camera(self.pipeline, self.config)
        if success:
            # Setup alignment for depth-color frame alignment
            self.setup_alignment()
        return success
    
    def get_robot_pose(self):
        """Get current robot end-effector pose."""
        # Use robot_utils which already handles the robot connection properly
        return self.robot_utils.get_robot_pose()
    
    def pose_to_transform_matrix(self, pose):
        """Convert [x, y, z, rx, ry, rz] pose to 4x4 transformation matrix."""
        return self.coordinate_utils.pose_to_transform_matrix(pose)
    
    def detect_object_on_surface(self, color_frame, depth_frame):
        """
        Detect objects on a surface using point cloud analysis.
        
        Returns:
            dict: Detection results with object center and surface information
        """
        # Generate point cloud from depth data
        point_cloud = self.generate_point_cloud(depth_frame, color_frame)
        
        if point_cloud is None or len(point_cloud) < 50:
            return self.create_empty_detection_result()
        
        # Filter to region of interest
        roi_points = self.filter_roi(point_cloud)
        
        if len(roi_points) < 50:
            return self.create_empty_detection_result(point_cloud)
        
        # Detect surface plane
        surface_plane = self.detect_surface_plane(roi_points)
        
        if surface_plane is None:
            return self.create_empty_detection_result(point_cloud, surface_plane)
        
        # Filter points above surface
        object_points = self.filter_points_above_surface(roi_points, surface_plane)
        
        if len(object_points) < self.clustering_min_samples:
            return self.create_empty_detection_result(point_cloud, surface_plane)
        
        # Find all significant clusters
        clusters = self.find_all_significant_clusters(object_points)
        
        if not clusters:
            return self.create_empty_detection_result(point_cloud, surface_plane)
        
        # Calculate overall confidence based on total object points and cluster count
        total_points = sum(cluster['size'] for cluster in clusters)
        confidence = min(1.0, total_points / 100.0)
        
        return {
            'valid_detection': True,
            'objects': clusters,
            'num_objects': len(clusters),
            'object_center': clusters[0]['center'],  # Legacy - largest cluster center
            'object_points': object_points,
            'surface_plane': surface_plane,
            'full_point_cloud': point_cloud,
            'roi_points': roi_points,
            'confidence': confidence,
            'num_points': total_points
        }
    
    def detect_object_knn_simple(self, point_cloud, intrinsics, knn_radius=0.05, min_neighbors=5, roi_size=50):
        """
        Simplified object detection using KNN approach focused on image center.
        
        Args:
            point_cloud: Generated point cloud data
            intrinsics: Camera intrinsic parameters
            knn_radius: Radius for KNN search (meters)
            min_neighbors: Minimum neighbors required for valid detection
            roi_size: Size of ROI around image center (pixels)
            
        Returns:
            dict: Detection results
        """
        if point_cloud is None or len(point_cloud) < 10:
            return self.create_empty_detection_result()
        
        # Define center region of image
        center_u, center_v = intrinsics.width // 2, intrinsics.height // 2
        u_min, u_max = center_u - roi_size, center_u + roi_size
        v_min, v_max = center_v - roi_size, center_v + roi_size
        
        # Filter point cloud to center region and reasonable depth range
        center_points = []
        for point in point_cloud:
            x, y, z = point[:3]
            
            # Skip points too close or too far
            if z < 0.1 or z > 0.8:
                continue
            
            # Project to pixel coordinates for ROI filtering
            pixel_coords = self._project_point_to_pixel([x, y, z], intrinsics)
            if pixel_coords:
                u, v = pixel_coords
                if u_min <= u <= u_max and v_min <= v <= v_max:
                    center_points.append([x, y, z])
        
        if len(center_points) < min_neighbors:
            return self.create_empty_detection_result(point_cloud)
        
        # Find all significant clusters in center points
        center_points = np.array(center_points)
        clusters = self.find_all_significant_clusters(center_points, min_cluster_size=max(5, min_neighbors))
        
        if not clusters:
            # Fallback: treat all center points as single cluster
            object_center = np.mean(center_points, axis=0)
            confidence = min(1.0, len(center_points) / 50.0)
            clusters = [{
                'id': 1,
                'center': object_center,
                'points': center_points,
                'color': (255, 0, 0),  # Red default
                'size': len(center_points),
                'confidence': confidence,
                'label': 0
            }]
        
        # Calculate overall confidence
        total_points = sum(cluster['size'] for cluster in clusters)
        confidence = min(1.0, total_points / 50.0)
        
        return {
            'valid_detection': True,
            'objects': clusters,
            'num_objects': len(clusters),
            'object_center': clusters[0]['center'],  # Legacy - largest cluster center
            'object_points': center_points,
            'surface_plane': None,  # Not detected in KNN mode
            'full_point_cloud': point_cloud,
            'roi_points': center_points,
            'confidence': confidence,
            'num_points': total_points
        }
    
    def detect_tabletop_object(self, color_frame, depth_frame):
        """
        Main object detection method using surface detection and clustering.
        
        Returns:
            dict: Detection results with object information
        """
        try:
            # Skip detection during robot grasping to avoid interference  
            if self.grasping_in_progress:
                return self.create_empty_detection_result()
            # Try surface-based detection first (more robust)
            results = self.detect_object_on_surface(color_frame, depth_frame)
            
            if results['valid_detection']:
                return results
            
            # Fallback to KNN-based detection if surface detection fails
            print("Surface detection failed, trying KNN approach...")
            point_cloud = results.get('full_point_cloud')
            
            if point_cloud is not None and self.camera_utils.color_intrinsics is not None:
                knn_results = self.detect_object_knn_simple(
                    point_cloud, 
                    self.camera_utils.color_intrinsics,
                    self.knn_radius,
                    self.knn_min_neighbors
                )
                return knn_results
            
            return self.create_empty_detection_result()
            
        except Exception as e:
            print(f"Detection error: {e}")
            return self.create_empty_detection_result()
    
    def create_empty_detection_result(self, point_cloud=None, surface_plane=None, full_point_cloud=None):
        """Create empty detection result structure."""
        return {
            'valid_detection': False,
            'objects': [],
            'num_objects': 0,
            'object_center': None,  # Legacy compatibility
            'object_points': [],
            'surface_plane': surface_plane,
            'full_point_cloud': full_point_cloud or point_cloud,
            'roi_points': [],
            'confidence': 0.0,
            'num_points': 0
        }
    
    def generate_point_cloud(self, depth_frame, color_frame=None):
        """Generate point cloud from depth frame."""
        # Generate point cloud
        self.pc.map_to(color_frame if color_frame else depth_frame)
        points = self.pc.calculate(depth_frame)
        
        # Get vertices as numpy array
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        
        # Filter out invalid points
        valid_mask = (vertices[:, 2] > 0) & (vertices[:, 2] < 2.0)  # Z between 0 and 2 meters
        valid_points = vertices[valid_mask]
        
        return valid_points
    
    def filter_roi(self, point_cloud):
        """Filter point cloud to region of interest."""
        # Apply ROI filtering
        roi_mask = (
            (point_cloud[:, 0] >= self.roi_bounds['x_min']) &
            (point_cloud[:, 0] <= self.roi_bounds['x_max']) &
            (point_cloud[:, 1] >= self.roi_bounds['y_min']) &
            (point_cloud[:, 1] <= self.roi_bounds['y_max']) &
            (point_cloud[:, 2] >= self.roi_bounds['z_min']) &
            (point_cloud[:, 2] <= self.roi_bounds['z_max'])
        )
        
        return point_cloud[roi_mask]
    
    def detect_surface_plane(self, point_cloud):
        """Detect surface plane using RANSAC."""
        if len(point_cloud) < 3:
            return None
        
        # Use only X and Y coordinates to fit a horizontal plane (Z as target)
        X = point_cloud[:, :2]  # X, Y coordinates
        y = point_cloud[:, 2]   # Z coordinates
        
        # RANSAC plane fitting
        ransac = RANSACRegressor(
            residual_threshold=self.surface_distance_threshold,
            min_samples=3,
            max_trials=100,
            random_state=42
        )
        
        try:
            ransac.fit(X, y)
            
            # Get plane coefficients
            # Plane equation: Z = ax + by + c
            # Convert to general form: ax + by - z + c = 0
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_
            
            return np.array([a, b, -1.0, c])
            
        except Exception as e:
            print(f"Plane detection failed: {e}")
            return None
    
    def filter_points_above_surface(self, point_cloud, surface_plane, min_height=0.01, max_height=0.1):
        """Filter points that are above the detected surface plane."""
        if surface_plane is None:
            return np.array([])
        
        a, b, c, d = surface_plane
        
        # Calculate distance from each point to the plane
        # Distance = |ax + by + cz + d| / sqrt(a² + b² + c²)
        norm = np.sqrt(a**2 + b**2 + c**2)
        distances = np.abs(
            a * point_cloud[:, 0] + 
            b * point_cloud[:, 1] + 
            c * point_cloud[:, 2] + 
            d
        ) / norm
        
        # Filter points that are above the surface by the specified height range
        above_surface_mask = (distances >= min_height) & (distances <= max_height)
        
        return point_cloud[above_surface_mask]
    
    def find_all_significant_clusters(self, object_points, min_cluster_size=15):
        """Find all significant clusters in the object points with color assignments."""
        if len(object_points) < self.clustering_min_samples:
            return []
        
        # Define color palette for different clusters
        color_palette = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Yellow
            (255, 165, 0),  # Orange
            (128, 0, 128),  # Purple
            (255, 192, 203), # Pink
            (0, 128, 128),  # Teal
        ]
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=self.clustering_eps, min_samples=self.clustering_min_samples)
        labels = clustering.fit_predict(object_points)
        
        # Find all significant clusters
        unique_labels = np.unique(labels)
        clusters = []
        cluster_id = 1
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
            
            # Get points in this cluster
            cluster_mask = labels == label
            cluster_points = object_points[cluster_mask]
            cluster_size = len(cluster_points)
            
            # Only include clusters above minimum size
            if cluster_size >= min_cluster_size:
                # Calculate cluster center
                cluster_center = np.mean(cluster_points, axis=0)
                
                # Assign color (cycle through palette if more clusters than colors)
                color_index = (cluster_id - 1) % len(color_palette)
                cluster_color = color_palette[color_index]
                
                # Calculate confidence based on cluster size
                confidence = min(1.0, cluster_size / 50.0)
                
                clusters.append({
                    'id': cluster_id,
                    'center': cluster_center,
                    'points': cluster_points,
                    'color': cluster_color,
                    'size': cluster_size,
                    'confidence': confidence,
                    'label': label  # Original DBSCAN label for visualization
                })
                
                cluster_id += 1
        
        # Sort clusters by size (largest first)
        clusters.sort(key=lambda x: x['size'], reverse=True)
        
        return clusters
    
    # === Enhanced User Interface and Controls ===
    
    def handle_enhanced_user_input(self):
        """Enhanced user input handler with robot control capabilities."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        
        # Enhanced robot controls
        elif key == ord('p'):  # Point at closest object
            if not (self.pointing_in_progress or self.grasping_in_progress):
                self._point_and_return()
            else:
                print("Robot action already in progress")
        
        elif key == ord('g'):  # Grasp closest object
            if not (self.pointing_in_progress or self.grasping_in_progress):
                self._grasp_and_return()
            else:
                print("Robot action already in progress")
        
        # Number keys for specific object grasping
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]:
            object_id = int(chr(key))  # Convert key to object ID (1-9)
            if not (self.pointing_in_progress or self.grasping_in_progress):
                self._grasp_specific_object(object_id)
            else:
                print("Robot action already in progress")
        
        # Camera tilt controls (requires base joint angles)
        elif key == ord('w'):  # Camera UP (joint4 +30°)
            if self.base_joint_angles and time.time() - self.last_movement_time > self.movement_cooldown:
                self._rotate_joint(3, self.joint_step, "Camera UP")  # Joint 4 (index 3)
            else:
                print("Camera tilt not available (no base joint angles)" if not self.base_joint_angles else "Movement cooling down...")
        
        elif key == ord('s'):  # Camera DOWN (joint4 -30°) or STOP robot
            if self.pointing_in_progress or self.grasping_in_progress:
                print("Stopping robot movement...")
                # Implement emergency stop if available
                if self._is_robot_available():
                    try:
                        if hasattr(self.enhanced_robot_controller, 'stop_movement'):
                            self.enhanced_robot_controller.stop_movement()
                        self.pointing_in_progress = False
                        self.grasping_in_progress = False
                        print("Robot movement stopped")
                    except Exception as e:
                        print(f"Error stopping robot: {e}")
            elif self.base_joint_angles and time.time() - self.last_movement_time > self.movement_cooldown:
                self._rotate_joint(3, -self.joint_step, "Camera DOWN")  # Joint 4 (index 3)
            else:
                print("Camera tilt not available (no base joint angles)" if not self.base_joint_angles else "Movement cooling down...")
        
        # Standard robot position controls
        elif key == ord('f'):  # Look forward (horizontal view)
            if self._is_robot_available():
                print("Moving to forward-looking pose...")
                if self.base_joint_angles:
                    # Use base angles with joint4 at 0° for forward look
                    forward_angles = self.base_joint_angles.copy()
                    forward_angles[3] = 0  # Joint 4 horizontal
                    try:
                        # Try async first if available
                        if hasattr(self.enhanced_robot_controller, 'move_to_angles_async'):
                            command_id = self.enhanced_robot_controller.move_to_angles_async(forward_angles, speed=self.movement_speed)
                            if command_id:
                                print(f"Forward look command submitted (ID: {command_id})")
                            else:
                                print("Failed to submit forward look command - trying sync")
                                # Fallback to sync
                                self.enhanced_robot_controller.move_to_angles(forward_angles, speed=self.movement_speed)
                                print("Moved to forward-looking position (sync)")
                        else:
                            # Use synchronous movement
                            self.enhanced_robot_controller.move_to_angles(forward_angles, speed=self.movement_speed)
                            print("Moved to forward-looking position (sync)")
                    except Exception as e:
                        print(f"Error moving to forward pose: {e}")
                else:
                    print("Cannot look forward: base joint angles not available")
            else:
                print("Robot not connected")
        
        elif key == ord('r'):  # Return to observation pose
            if self._is_robot_available():
                print("Returning to observation pose...")
                try:
                    from module.utils.robot_utils import move_to_standard_pose
                    success = move_to_standard_pose(self.enhanced_robot_controller, "table", async_mode=True)
                    if success:
                        print("Returned to observation pose")
                        self.robot_in_observation_pose = True
                    else:
                        print("Failed to return to observation pose")
                except Exception as e:
                    print(f"Error returning to observation pose: {e}")
            else:
                print("Robot not connected")
        
        elif key == ord('c'):  # Move robot to next preset position
            if self.robot_utils.move_robot_to_next_pose():
                print("Robot moved to next preset position")
            else:
                print("Failed to move robot to next preset position")
        
        elif key == ord('a'):  # Recalibrate auto ROI (placeholder for future implementation)
            print("Auto ROI recalibration not yet implemented")
            print("Current ROI bounds:")
            for bound_name, value in self.roi_bounds.items():
                print(f"  {bound_name}: {value:.3f}m")
        
        return True  # Continue main loop
    
    def update_robot_state_tracking(self):
        """Update robot state tracking for timeouts and transitions."""
        # Check for pointing timeout
        self._check_pointing_timeout()
        
        # Check for return stabilization completion
        self._check_return_stabilization()
    
    def run(self):
        """Enhanced main application loop with robot functionality."""
        print("Starting Enhanced LiDAR Real-time Prediction System...")
        print("Enhanced Features:")
        print("  • Real-time markerless object detection and clustering")
        print("  • Advanced robot integration with pointing and grasping")
        print("  • Camera tilt controls for optimal viewing angles")
        print("  • Async robot operations - visualization never blocks")
        print("  • Multi-object detection with color-coded visualization")
        
        # Setup camera
        if not self.setup_camera():
            print("Failed to initialize camera. Exiting...")
            return
        
        # Initialize frame timing
        fps_counter = 0
        fps_timer = time.time()
        last_frame_time = time.time()
        
        try:
            while True:
                frame_start_time = time.time()
                
                # Update robot state tracking
                self.update_robot_state_tracking()
                
                # Process non-blocking grasping state machine
                if self.grasping_in_progress and self.grasping_state:
                    self._process_grasping_state_machine()
                
                # Get frames from camera
                frames = self.pipeline.wait_for_frames()
                
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                
                # Skip depth processing during robot actions for better performance
                if self.pointing_in_progress or self.grasping_in_progress or self.returning_to_observer:
                    depth_frame = None  # No depth processing needed during robot actions
                    # During robot actions: use last detection results to maintain UI but keep RGB updating
                    detection_results = self.create_empty_detection_result()
                    # Keep using the last detected objects for visualization and robot status
                    if hasattr(self, 'last_detection_results'):
                        detection_results['objects'] = self.last_detection_results.get('objects', [])
                        detection_results['valid_detection'] = len(detection_results['objects']) > 0
                        detection_results['num_objects'] = len(detection_results['objects'])
                        if detection_results['objects']:
                            detection_results['object_center'] = detection_results['objects'][0]['center']
                else:
                    # Normal operation: process depth and do full detection
                    aligned_frames = self.align.process(frames)
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if not depth_frame:
                        continue
                        
                    # Update color frame to aligned version for consistency
                    color_frame = aligned_frames.get_color_frame()
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Full detection with depth processing
                    detection_results = self.detect_tabletop_object(color_frame, depth_frame)
                    
                    # Store detection results for robot actions
                    if detection_results['valid_detection']:
                        self.last_detection_objects = detection_results['objects']
                        self.last_detection_results = detection_results  # Store complete results
                        self.performance_stats['total_detections'] += 1
                
                # Get robot pose for coordinate transformation
                robot_pose = None
                robot_transform = None
                position_in_base = None
                
                try:
                    robot_pose = self.get_robot_pose()
                    if robot_pose:
                        robot_transform = self.pose_to_transform_matrix(robot_pose)
                        
                        # Get position prediction for all detected objects
                        if detection_results['valid_detection'] and detection_results['objects']:
                            # Use main object (largest cluster) for position history tracking
                            main_object = detection_results['objects'][0]  # Largest cluster
                            main_object_center = main_object['center']
                            main_position_in_base = self.predict_object_position(main_object_center, robot_transform)
                            
                            # Update position history for stability analysis (use main object only)
                            self.update_position_history(main_position_in_base)
                            
                            # Print positions for ALL objects periodically (skip during robot actions)
                            if not (self.grasping_in_progress or self.pointing_in_progress or self.returning_to_observer):
                                self.position_print_counter += 1
                                if self.position_print_counter % 30 == 0:  # Every 30 frames (~1 second)
                                    mean_pos, std_pos = self.get_position_stability()
                                    print(f"\n--- Object Positions (Robot Coordinates) ---")
                                    for i, obj in enumerate(detection_results['objects']):
                                        obj_center = obj['center']
                                        obj_position_in_base = self.predict_object_position(obj_center, robot_transform)
                                        obj_id = obj.get('id', i+1)
                                        obj_size = obj.get('size', 0)
                                        obj_conf = obj.get('confidence', 0.0)
                                        
                                        # Show stability info only for main object (largest)
                                        if i == 0 and mean_pos is not None:
                                            print(f"Object #{obj_id}: ({obj_position_in_base[0]:6.1f}, {obj_position_in_base[1]:6.1f}, {obj_position_in_base[2]:6.1f}) mm "
                                                  f"(±{std_pos[0]:.1f}, ±{std_pos[1]:.1f}, ±{std_pos[2]:.1f}) [main] size={obj_size}, conf={obj_conf:.2f}")
                                        else:
                                            print(f"Object #{obj_id}: ({obj_position_in_base[0]:6.1f}, {obj_position_in_base[1]:6.1f}, {obj_position_in_base[2]:6.1f}) mm "
                                                  f"size={obj_size}, conf={obj_conf:.2f}")
                                    print("" if len(detection_results['objects']) > 1 else "", end="")
                except Exception as e:
                    print(f"Robot pose error: {e}")
                
                # Create multi-object position data for visualization
                object_positions = []
                if detection_results['valid_detection'] and robot_transform is not None:
                    for obj in detection_results['objects']:
                        obj_position = self.predict_object_position(obj['center'], robot_transform)
                        object_positions.append({
                            'id': obj.get('id', 0),
                            'position': obj_position,
                            'confidence': obj.get('confidence', 0.0)
                        })
                
                # Create enhanced 2x2 visualization
                visualization = self.create_2x2_visualization(
                    color_image, depth_frame, detection_results, robot_pose, position_in_base, object_positions
                )
                
                # Display the visualization
                cv2.imshow('Enhanced LiDAR Object Position Prediction', visualization)
                
                # Handle enhanced user input (including robot controls)
                if not self.handle_enhanced_user_input():
                    break
                
                # Calculate and update FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_timer >= 1.0:
                    self.performance_stats['avg_fps'] = fps_counter
                    fps_counter = 0
                    fps_timer = current_time
                
                # Track frame timing for performance monitoring
                frame_time = current_time - frame_start_time
                self.performance_stats['detection_times'].append(frame_time)
                
                # Keep only recent timing data
                if len(self.performance_stats['detection_times']) > 100:
                    self.performance_stats['detection_times'].pop(0)
                
                # Frame rate control
                target_fps = 30
                frame_duration = 1.0 / target_fps
                elapsed = current_time - last_frame_time
                if elapsed < frame_duration:
                    time.sleep(frame_duration - elapsed)
                last_frame_time = time.time()
        
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
        
        finally:
            self.cleanup()
    
    def set_safety_validation(self, enabled: bool = True):
        """
        Enable or disable robot safety validation.
        
        Args:
            enabled: True to enable safety validation (default), False to skip all safety checks
        
        Warning:
            Disabling safety validation removes all movement constraints and can be dangerous.
            Use with extreme caution and manual oversight.
        """
        if self.enhanced_robot_controller and hasattr(self.enhanced_robot_controller, 'set_safety_validation'):
            self.enhanced_robot_controller.set_safety_validation(enabled)
        else:
            print("Robot controller not available or doesn't support safety validation control")
    
    def cleanup(self):
        """Enhanced cleanup with robot return to safe position."""
        print("Cleaning up Enhanced LiDAR Prediction System...")
        
        # Return robot to observation pose if connected
        if self._is_robot_available():
            try:
                print("Returning robot to safe observation pose...")
                from module.utils.robot_utils import move_to_standard_pose
                move_to_standard_pose(self.enhanced_robot_controller, "table", async_mode=False)
                print("✓ Robot returned to safe position")
            except Exception as e:
                print(f"Warning: Could not return robot to safe position: {e}")
        
        # Stop camera pipeline
        try:
            self.pipeline.stop()
            print("✓ Camera pipeline stopped")
        except Exception as e:
            print(f"Warning: Camera cleanup error: {e}")
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        # Print performance summary
        print("\nPerformance Summary:")
        print(f"  Total detections: {self.performance_stats['total_detections']}")
        print(f"  Pointing commands: {self.performance_stats['pointing_commands']}")
        print(f"  Grasping commands: {self.performance_stats['grasping_commands']}")
        print(f"  Average FPS: {self.performance_stats['avg_fps']:.1f}")
        
        if self.performance_stats['detection_times']:
            avg_detection_time = np.mean(self.performance_stats['detection_times'])
            print(f"  Average detection time: {avg_detection_time*1000:.1f}ms")
        
        print("Enhanced LiDAR Prediction System shutdown complete.")


def main():
    """Main function to run the Enhanced LiDAR Real-time Prediction System."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Enhanced Real-time LiDAR Object Position Prediction with Advanced Robot Integration"
    )
    parser.add_argument('--config-path', default="charuco-pose-estimation",
                       help='Path to configuration directory')
    parser.add_argument('--roi-x-min', type=float, default=-0.5,
                       help='ROI minimum X bound (meters)')
    parser.add_argument('--roi-x-max', type=float, default=0.5,
                       help='ROI maximum X bound (meters)')  
    parser.add_argument('--roi-y-min', type=float, default=-0.5,
                       help='ROI minimum Y bound (meters)')
    parser.add_argument('--roi-y-max', type=float, default=0.5,
                       help='ROI maximum Y bound (meters)')
    parser.add_argument('--roi-z-min', type=float, default=0.1,
                       help='ROI minimum Z bound (meters)')
    parser.add_argument('--roi-z-max', type=float, default=0.5,
                       help='ROI maximum Z bound (meters)')
    parser.add_argument('--enable-safety', action='store_true',
                       help='Enable robot safety validation (safety is disabled by default)')
    parser.add_argument('--offset-x', type=float, default=-15.0,
                       help='Prediction offset X (forward/backward) in mm (default: 0.0)')
    parser.add_argument('--offset-y', type=float, default=-7.5,
                       help='Prediction offset Y (left/right) in mm (default: 0.0)')
    parser.add_argument('--offset-z', type=float, default=0.0,
                       help='Prediction offset Z (up/down) in mm (default: 0.0)')
    parser.add_argument('--offset', nargs=3, type=float, metavar=('X', 'Y', 'Z'),
                       help='Prediction offset as X Y Z in mm (alternative to --offset-x/y/z)')
    
    args = parser.parse_args()
    
    # Process offset arguments
    if args.offset is not None:
        # Use --offset X Y Z format
        prediction_offset = args.offset
    else:
        # Use individual --offset-x/y/z arguments
        prediction_offset = [args.offset_x, args.offset_y, args.offset_z]
    
    # Only pass offset if it's not [0, 0, 0]
    if not all(x == 0.0 for x in prediction_offset):
        print(f"[CONFIG] Using prediction offset: [{prediction_offset[0]:.1f}, {prediction_offset[1]:.1f}, {prediction_offset[2]:.1f}] mm")
        print(f"         X={prediction_offset[0]:.1f}mm ({'forward' if prediction_offset[0] > 0 else 'backward' if prediction_offset[0] < 0 else 'no change'})")
        print(f"         Y={prediction_offset[1]:.1f}mm ({'left' if prediction_offset[1] > 0 else 'right' if prediction_offset[1] < 0 else 'no change'})")
        print(f"         Z={prediction_offset[2]:.1f}mm ({'up' if prediction_offset[2] > 0 else 'down' if prediction_offset[2] < 0 else 'no change'})")
    
    # Create enhanced predictor instance
    predictor = EnhancedLidarPredictor(
        config_path=args.config_path,
        prediction_offset=prediction_offset if not all(x == 0.0 for x in prediction_offset) else None
    )
    
    # Configure ROI bounds from command line arguments
    predictor.roi_bounds = {
        'x_min': args.roi_x_min,
        'x_max': args.roi_x_max,
        'y_min': args.roi_y_min,
        'y_max': args.roi_y_max,
        'z_min': args.roi_z_min,
        'z_max': args.roi_z_max
    }
    
    print(f"ROI configured: X=[{args.roi_x_min:.1f}, {args.roi_x_max:.1f}], "
          f"Y=[{args.roi_y_min:.1f}, {args.roi_y_max:.1f}], "
          f"Z=[{args.roi_z_min:.1f}, {args.roi_z_max:.1f}] meters")
    
    # Apply safety validation setting from command line argument
    # Safety is disabled by default, enable only if flag is provided
    if args.enable_safety:
        print("\n✅ Safety validation enabled by --enable-safety flag")
        predictor.set_safety_validation(enabled=True)
    else:
        print("\n⚠️  Safety validation disabled by default")
        print("    Use EXTREME CAUTION - manual oversight required")
        predictor.set_safety_validation(enabled=False)
    
    print("\n=== Enhanced LiDAR Real-time Prediction System with Robot Integration ===")
    safety_status = "ENABLED ✅" if args.enable_safety else "DISABLED ⚠️ (Default)"
    print(f"Safety Control: {safety_status}")
    if args.enable_safety:
        print("  - Safety validation enabled - robot movements will be constrained by safety limits")
        print("  - To disable safety checks, remove --enable-safety flag")
    else:
        print("  - Safety validation disabled by default for maximum grasping flexibility")
        print("  - Robot will move to exact calculated positions without constraints")
        print("  - To enable safety checks, use --enable-safety flag")
        print("  - Monitor robot movements carefully to prevent collisions")
    print("\nGrasping Controls:")
    print("  - 'g' key: Start non-blocking grasping sequence")
    print("  - 'p' key: Point at closest detected object")
    print("  - 's' key: Emergency stop (during robot actions)")
    print("\nStarting system...\n")
    
    # Run the enhanced system
    predictor.run()


if __name__ == "__main__":
    main()