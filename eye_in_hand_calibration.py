#!/usr/bin/env python3
"""
Eye-in-Hand Calibration for Camera Mounted on Robot End Effector

This script calibrates the transformation between:
1. Camera coordinate frame (mounted on end effector)  
2. Robot end effector coordinate frame
3. Robot base coordinate frame

The camera is physically mounted on the robot's end effector, typically:
- With some offset from end effector center
- Angled downward (e.g., 45 degrees) to look at workspace
- Fixed relative to end effector (moves with robot)

Usage:
    python eye_in_hand_calibration.py [--positions N] [--output FILE] [--visualize-raw-pointcloud]
"""

import argparse
import sys
import os
import time
import threading
import queue
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eye_in_hand_calibrator_base import EyeInHandCalibratorBase
from module.sensors.realsense_manager import RealSenseManager
from module.robot.mycobot_controller import MyCobotController
from module.core.object_detector import ObjectDetector
from module.core.visualization_engine import VisualizationEngine
from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
from module.robot.command_queue import CommandStatus
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures


class EyeInHandCalibrator(EyeInHandCalibratorBase):
    """
    Eye-in-hand calibration system.
    
    Calibrates the transformation from camera frame to robot end effector frame.
    The camera is mounted on the robot's end effector and moves with it.
    
    Process:
    1. Move robot to different positions/orientations
    2. Detect calibration object in camera frame at each position
    3. Record robot end effector pose for each detection
    4. Solve for camera-to-end-effector transformation
    """
    
    def __init__(self, num_positions: int = 5, output_file: str = "eye_in_hand_transform.npy",
                 auto_floor_detection: bool = False, floor_offset: float = 0.01,
                 visualize_raw_pointcloud: bool = False):
        """
        Initialize eye-in-hand calibrator.
        
        Args:
            num_positions: Number of calibration positions to use
            output_file: Output file for transformation matrix
            auto_floor_detection: Enable automatic floor detection and ROI adjustment
            floor_offset: Height offset above detected floor in meters (default: 40mm)
            visualize_raw_pointcloud: Enable raw point cloud visualization
        """
        # Initialize base class
        super().__init__()
        
        self.num_positions = num_positions
        self.output_file = output_file
        self.auto_floor_detection = auto_floor_detection
        self.floor_offset = floor_offset
        self.visualize_raw_pointcloud = visualize_raw_pointcloud
        
        # Components
        self.sensor_manager = None
        self.robot_controller = None
        self.detector = None
        self.visualizer = VisualizationEngine()
        
        # Calibration data
        self.calibration_data = []
        self.current_position = 0
        
        # Camera orientation tracking for efficient gravity alignment updates
        self.initial_robot_pose = None
        self.initial_rotation_matrix = None
        
        # Direct visualization system (no threading needed)
        # Using approach similar to virtual_fixed_tracking.py
        
        print("Eye-in-Hand Calibration System")
        print("=" * 50)
        print(f"Target positions: {num_positions}")
        print(f"Output file: {output_file}")
    
    def initialize(self) -> bool:
        """Initialize all system components."""
        print("\nInitializing calibration system...")
        
        # Initialize camera
        print("Initializing camera (mounted on end effector)...")
        self.sensor_manager = RealSenseManager(None, realtime_mode=True, use_imu=True)
        if not self.sensor_manager.initialize():
            print("Error: Failed to initialize camera")
            return False
        
        if not self.sensor_manager.start():
            print("Error: Failed to start camera")
            return False
        
        # Initialize robot controller first (no transformation matrix needed yet)
        print("Initializing robot controller...")
        self.robot_controller = MyCobotController(transformation_matrix_file=None)
        
        if not self.robot_controller.connect():
            print("Error: Failed to connect to robot")
            return False
        
        # Enable async mode for non-blocking robot commands (like virtual_fixed_tracking.py)
        if self.robot_controller.enable_async_mode(max_queue_size=5):
            print("[OK] Robot async mode enabled - visualization continues during movement")
        else:
            print("[WARNING] Failed to enable robot async mode - commands may block visualization")
        
        # Move robot to home position first and wait for stabilization
        print("Moving robot to home position...")
        if self.robot_controller.async_mode_enabled:
            command_id = self.robot_controller.home_position_async(priority=1)
            if command_id:
                print("[OK] Home position command submitted")
                self._wait_for_movement_with_visualization(command_id)
            else:
                print("[ERROR] Failed to send home position command")
        else:
            if self.robot_controller.home_position():
                print("[OK] Robot moved to home position")
            else:
                print("[WARNING] Failed to move robot to home position")
        
        # Wait for robot stabilization with visualization
        self._wait_with_visualization(2.0, "Robot stabilizing...")
        
        # Note: Gravity calibration will happen at each calibration position, not here
        
        # Initialize object detector for calibration target detection
        print("Initializing object detector...")
        try:
            strategy = BEVClusteringStrategy(
                roi_params=None,
                display_roi_params=None, 
                clustering_params={
                    'min_cluster_size': self.min_object_size,
                    'max_cluster_size': 200,
                    'morphology_kernel_size': 3,
                    'window_size': 3
                },
                enable_temporal_filtering=False  # No temporal filtering for calibration
            )
            self.detector = ObjectDetector(clustering_strategy=strategy)
            print("+ Object detector initialized")
        except Exception as e:
            print(f"Error initializing detector: {e}")
            return False
        
        print("+ All systems initialized successfully")
        print("Note: First calibration position will be LOOK_AT_TABLE coordinates")
        print("+ Live visualization enabled - 3-panel display like virtual_fixed_tracking.py")
        
        return True
    
    def get_calibration_positions(self) -> List[List[float]]:
        """
        Generate calibration positions for eye-in-hand setup.
        
        These positions should:
        1. Start from LOOK_AT_TABLE position as reference
        2. Keep calibration object in camera view
        3. Maintain minimum 250mm distance for lidar effective range
        4. Provide different perspectives (positions and orientations)
        5. Stay within robot workspace limits
        
        All positions are designed to keep camera at least 250mm from table objects.
        
        Returns:
            List of robot positions [x, y, z, rx, ry, rz]
        """
        positions = []
        
        
        if self.num_positions == 3:
            # Quick calibration - 3 positions
            # All positions use coordinates, starting with actual LOOK_AT_TABLE coordinates
            positions = [
                [87.5, -62.7, 367.2, -144.03, 29.89, -124.04],                 # Actual LOOK_AT_TABLE coordinates
                [150, -100, 320, -144, 30, -124],    # Left side, similar orientation
                [150, 100, 320, -144, 30, -124],     # Right side, similar orientation
            ]
        elif self.num_positions == 5:
            # Standard calibration - 5 positions with DIVERSE rotations for OpenCV calibrateHandEye
            # FIXED: All positions maintain downward camera angle to keep object in view
            positions = [
                [87.5, -62.7, 367.2, -144.03, 29.89, -124.04],   # Actual LOOK_AT_TABLE coordinates
                [150, 0, 300, -144, 30, -124],                   # Center, same orientation (reference)
                [120, -80, 320, -160, 45, -140],                # Left, rotated but still looking down
                [120, 80, 320, -130, 15, -110],                 # Right, rotated but still looking down  
                [180, 0, 280, -155, 50, -115],                  # Center, angled but still looking down
            ]
        elif self.num_positions >= 7:
            # Precision calibration - 7+ positions with maximum rotational diversity
            # FIXED: All positions maintain downward camera angle to keep object in view
            positions = [
                [87.5, -62.7, 367.2, -144.03, 29.89, -124.04],   # Actual LOOK_AT_TABLE coordinates
                [150, 0, 300, -144, 30, -124],                   # Center, same orientation (reference)
                [120, -120, 320, -170, 50, -150],               # Far left, rotated but looking down
                [130, -60, 310, -135, 10, -130],                # Near left, slight rotation
                [130, 60, 310, -150, 45, -115],                 # Near right, angled but looking down
                [120, 120, 320, -125, 20, -105],                # Far right, rotated but looking down
                [100, 0, 340, -160, 60, -120],                  # High position, steep downward angle
            ]
            
            # Add extra positions if requested
            if self.num_positions > 7:
                extra_positions = [
                    [140, -50, 290, -155, 35, -125],  # Tilted left, looking down
                    [140, 50, 290, -135, 25, -115],   # Tilted right, looking down
                    [170, 0, 270, -165, 55, -125],    # Extended reach, steep downward
                ]
                positions.extend(extra_positions[:self.num_positions - 7])
        
        return positions[:self.num_positions]
    
    
    def _create_dynamic_roi_from_table(self, table_info: Dict[str, Any], object_height_range: Tuple[float, float] = (0.01, 0.3)) -> Dict[str, float]:
        """
        Create dynamic ROI parameters based on detected table surface.
        
        Args:
            table_info: Table surface detection results
            object_height_range: Expected object height range above table (min, max) in meters
            
        Returns:
            ROI parameters for filtering
        """
        table_height = table_info['table_height']
        bounds = table_info['bounds']
        
        # Create ROI above table surface
        roi_params = {
            'x_min': bounds['x_min'] - 0.1,  # Add 10cm margin
            'x_max': bounds['x_max'] + 0.1,
            'y_min': table_height + object_height_range[0],  # Objects start just above table
            'y_max': table_height + object_height_range[1],  # Objects up to 15cm above table
            'z_min': bounds['z_min'] - 0.1,  # Add 10cm margin
            'z_max': bounds['z_max'] + 0.1
        }
        
        print(f"+ Dynamic ROI created:")
        print(f"  Object height range: {object_height_range[0]*1000:.0f}-{object_height_range[1]*1000:.0f}mm above table")
        print(f"  Y range: [{roi_params['y_min']:.3f}, {roi_params['y_max']:.3f}]m")
        
        return roi_params
    
    def _detect_and_set_dynamic_roi(self, position_index: int) -> bool:
        """
        Detect floor surface and set dynamic ROI parameters automatically.
        
        Args:
            position_index: Current calibration position index (for logging)
            
        Returns:
            True if floor detection and ROI setting successful
        """
        if not self.auto_floor_detection:
            return True  # Skip if auto detection is disabled
        
        print(f"[AUTO-ROI] Detecting floor surface at position {position_index + 1}...")
        
        try:
            # Get multiple frames for better floor detection
            floor_detection_frames = 5
            all_vertices = []
            
            for frame_idx in range(floor_detection_frames):
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                if vertices is not None and len(vertices) > 100:
                    all_vertices.append(vertices)
                time.sleep(0.1)  # Brief pause between frames
            
            if not all_vertices:
                print(f"[AUTO-ROI] Warning: No valid frames for floor detection at position {position_index + 1}")
                return False
            
            # Combine all frames for more robust floor detection
            combined_vertices = np.vstack(all_vertices)
            print(f"[AUTO-ROI] Using {len(combined_vertices)} points from {len(all_vertices)} frames")
            
            # Detect floor surface using existing RANSAC method
            table_info = self._detect_table_surface_ransac(combined_vertices, max_distance=0.015)
            
            if table_info is None:
                print(f"[AUTO-ROI] Warning: Failed to detect floor surface at position {position_index + 1}")
                return False
            
            # Extract floor height
            detected_floor_height = table_info['table_height']
            floor_height_std = table_info['table_height_std']
            
            print(f"[AUTO-ROI] Detected floor height: {detected_floor_height:.3f}m ± {floor_height_std:.3f}m")
            
            # Set dynamic ROI based on detected floor
            dynamic_roi = {
                'x_min': -1.0, 'x_max': 1.0,  # Keep wide X range
                'y_min': detected_floor_height - self.floor_offset - 0.06 ,  # Start at detected floor
                'y_max': detected_floor_height - self.floor_offset,  # Add configurable offset
                'z_min': 0, 'z_max': 0.5    # Keep depth range
            }
            
            # Update detector's ROI parameters
            if hasattr(self.detector.clustering_strategy, 'roi_params'):
                self.detector.clustering_strategy.roi_params = dynamic_roi
                print(f"[AUTO-ROI] Set dynamic ROI: Y[{dynamic_roi['y_min']:.3f}, {dynamic_roi['y_max']:.3f}]m")
                print(f"[AUTO-ROI] Object detection height: {self.floor_offset*1000:.0f}mm above floor")
                return True
            else:
                print(f"[AUTO-ROI] Warning: Detector does not support ROI parameter updates")
                return False
                
        except Exception as e:
            print(f"[AUTO-ROI] Error during floor detection at position {position_index + 1}: {e}")
            return False
    
    def _apply_manual_roi_preset(self, position_index: int):
        """
        Apply manual ROI presets for each calibration position.
        This preserves the original hardcoded ROI behavior.
        
        Args:
            position_index: Current calibration position index (0-based)
        """
        if not hasattr(self.detector.clustering_strategy, 'roi_params'):
            print(f"[MANUAL-ROI] Warning: Detector does not support ROI parameter updates")
            return
        
        print(f"[MANUAL-ROI] Applying manual ROI preset for position {position_index + 1}")
        
        # Original hardcoded ROI presets for each position
        if position_index == 0:
            roi_params = {
                'x_min': -1.0, 'x_max': 1.0,
                'y_min': 0.34, 'y_max': .38,  # Height range
                'z_min': 0, 'z_max': .5    # Depth range
            }
        elif position_index == 1:
            roi_params = {
                'x_min': -1.0, 'x_max': 1.0,
                'y_min': 0.26, 'y_max': .30,  # Height range
                'z_min': 0, 'z_max': .5    # Depth range
            }
        elif position_index == 2:
            roi_params = {
                'x_min': -1.0, 'x_max': 1.0,
                'y_min': 0.26, 'y_max': .30,  # Height range
                'z_min': 0, 'z_max': .5    # Depth range
            }
        elif position_index == 3:
            roi_params = {
                'x_min': -1.0, 'x_max': 1.0,
                'y_min': 0.26, 'y_max': .30,  # Height range
                'z_min': 0, 'z_max': .5    # Depth range
            }
        elif position_index == 4:
            roi_params = {
                'x_min': -1.0, 'x_max': 1.0,
                'y_min': 0.22, 'y_max': .26,  # Height range
                'z_min': 0, 'z_max': .5    # Depth range
            }
        else:
            # Default ROI for positions beyond predefined presets
            roi_params = {
                'x_min': -1.0, 'x_max': 1.0,
                'y_min': 0.25, 'y_max': .35,  # Default height range
                'z_min': 0, 'z_max': .5    # Depth range
            }
        
        # Apply the ROI parameters
        self.detector.clustering_strategy.roi_params = roi_params
        print(f"[MANUAL-ROI] Set ROI: Y[{roi_params['y_min']:.3f}, {roi_params['y_max']:.3f}]m")
    
    def move_to_calibration_position(self, position: List[float]) -> bool:
        """
        Move robot to calibration position with continuous visualization.
        Uses async commands to keep camera feed running during movement.
        
        Args:
            position: Target position [x, y, z, rx, ry, rz]
            
        Returns:
            True if movement successful
        """
        print(f"Moving to position: {[f'{x:.1f}' for x in position]}")
        
        # Validate position is safe
        is_safe, reason = self.robot_controller.validate_position_safety(position)
        if not is_safe:
            print(f"Unsafe position: {reason}")
            return False
        
        # Move robot to position (async if possible for continuous visualization)
        if self.robot_controller.async_mode_enabled:
            print("Sending async movement command - visualization continues...")
            command_id = self.robot_controller.move_to_position_async(position, speed=30, priority=1)
            if not command_id:
                print("Failed to send async movement command")
                return False
            
            # Wait for movement completion with live visualization
            print("Robot moving... (live feed continues)")
            self._wait_for_movement_with_visualization(command_id)
            
            # Wait for robot to settle after async movement completion
            print("Waiting for robot to settle...")
            self._wait_with_visualization(2.0, "Robot settling...")
            
        else:
            # Fallback to synchronous movement
            if not self.robot_controller.move_to_position(position, speed=30):
                print("Failed to send movement command")
                return False
            
            print("Waiting for robot to settle...")
            self._wait_with_visualization(3.0, "Robot settling...")
        
        # Recalibrate gravity alignment for new camera orientation (quick recalibration)
        if self.sensor_manager.use_imu:
            print("Recalibrating gravity alignment for new camera orientation...")
            if self.sensor_manager.calibrate_gravity_alignment(num_samples=10):  # Fast re-calibration for robot movement
                print("+ Gravity alignment updated for current robot position")
            else:
                print("Warning: Failed to recalibrate gravity alignment")
        
        # Verify robot reached target position
        current_pos = self.robot_controller.get_current_position()
        if current_pos:
            position_error = np.linalg.norm(np.array(current_pos[:3]) - np.array(position[:3]))
            print(f"Position error: {position_error:.1f}mm")
            if position_error > 50:  # 50mm tolerance
                print("Warning: Large position error detected")
        
        return True
    
    

    def _create_calibration_visualization(self, vertices: np.ndarray, color_image: Optional[np.ndarray],
                                        detected_objects: List[Dict[str, Any]], timing_info: Dict[str, float],
                                        bev_image: Optional[np.ndarray] = None, point_cloud_view: Optional[np.ndarray] = None,
                                        raw_vertices: Optional[np.ndarray] = None):
        """Create enhanced 2x2 visualization layout with raw point cloud support."""
        try:
            panel_size = 400
            
            # Initialize 2x2 grid panels
            top_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)    # RGB
            top_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)   # Raw Point Cloud
            bottom_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8) # BEV
            bottom_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8) # Side View
            
            # Top-left: RGB image with calibration info overlay
            if color_image is not None:
                rgb_panel = color_image.copy()
                self._add_calibration_info_overlay(rgb_panel, detected_objects, timing_info)
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
            cv2.imshow('Eye-in-Hand Calibration', combined)
            
        except Exception as e:
            print(f"Warning: Visualization error: {e}")
            # Fallback to simple color image display
            if color_image is not None:
                cv2.imshow('Eye-in-Hand Calibration', color_image)
    
    def _add_calibration_info_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]], 
                                    timing_info: Dict[str, float]):
        """Add calibration information overlay to image (similar to virtual_fixed_tracking.py)."""
        # Get robot status
        robot_status = "Disconnected"
        if self.robot_controller and self.robot_controller.is_connected:
            robot_status = "Connected"
        
        # Get ROI mode information
        roi_mode = "AUTO Floor Detection" if self.auto_floor_detection else "Manual ROI Presets"
        roi_info = ""
        if hasattr(self.detector.clustering_strategy, 'roi_params') and self.detector.clustering_strategy.roi_params:
            roi_params = self.detector.clustering_strategy.roi_params
            roi_info = f"Y: [{roi_params['y_min']:.3f}, {roi_params['y_max']:.3f}]m"
            if self.auto_floor_detection:
                floor_height = roi_params['y_min']
                offset_mm = (roi_params['y_max'] - roi_params['y_min']) * 1000
                roi_info += f" (Floor: {floor_height:.3f}m, +{offset_mm:.0f}mm)"
        
        info_lines = [
            "EYE-IN-HAND CALIBRATION",
            f"Position: {self.current_position}/{self.num_positions}",
            f"ROI Mode: {roi_mode}",
            f"ROI: {roi_info}" if roi_info else "ROI: Not set",
            f"Objects: {len(detected_objects)}",
            f"Detection time: {timing_info.get('total_time', 0)*1000:.1f}ms",
            f"Robot: {robot_status}",
            "",
            "CONTROLS:",
            "'c' - Confirm calibration point",
            "'s' - Skip this position", 
            "'q' - Quit calibration",
            "'r' - Retry object detection",
            "'i' - Recalibrate IMU/gravity"
        ]
        
        for i, line in enumerate(info_lines):
            if not line:  # Skip empty lines
                continue
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            if "CONTROLS:" in line:
                color = (0, 255, 255)  # Yellow for controls section
            cv2.putText(image, line, (10, 30 + i * 22), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _add_object_detection_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]]):
        """Add object detection information overlay."""
        if not detected_objects:
            cv2.putText(image, "NO OBJECTS DETECTED", (10, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return
        
        y_offset = 300
        for i, obj in enumerate(detected_objects[:3]):  # Show first 3 objects
            position = self._extract_object_position(obj)
            if position:
                distance = np.linalg.norm(position)
                # Color based on distance (green if within lidar range)
                color = (0, 255, 0) if distance >= self.min_object_distance else (0, 0, 255)
                info_text = f"Obj{i+1}: {distance:.0f}mm"
                if distance >= self.min_object_distance:
                    info_text += " [VALID]"
                else:
                    info_text += " [TOO CLOSE]"
                cv2.putText(image, info_text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _recalibrate_imu_manual(self):
        """Manually recalibrate IMU/gravity alignment and auto ROI during calibration."""
        if not self.sensor_manager or not self.sensor_manager.use_imu:
            print("[INFO] IMU not available, skipping gravity recalibration")
            return
        
        print("[IMU] Recalibrating gravity alignment...")
        print("[IMU] Keep camera steady for 2 seconds...")
        
        if self.sensor_manager.calibrate_gravity_alignment(num_samples=20):
            print("[OK] Gravity recalibration completed")
            
            # Also trigger auto ROI detection after IMU recalibration
            position_index = self.current_position - 1  # Convert to 0-based index
            print("[AUTO-ROI] Recalibrating ROI after gravity alignment...")
            
            if self.auto_floor_detection:
                # Attempt auto floor detection
                roi_success = self._detect_and_set_dynamic_roi(position_index)
                if not roi_success:
                    print("[AUTO-ROI] Auto detection failed, falling back to manual ROI preset")
                    self._apply_manual_roi_preset(position_index)
            else:
                # Apply manual ROI preset
                print("[MANUAL-ROI] Applying manual ROI preset")
                self._apply_manual_roi_preset(position_index)
                
        else:
            print("[ERROR] Failed to recalibrate gravity alignment")
    
    def detect_calibration_object_continuous(self) -> Optional[Dict[str, Any]]:
        """
        Continuous object detection with live visualization and user confirmation.
        Similar to virtual_fixed_tracking.py approach.
        
        Returns:
            Object data with position in camera coordinates, or None if user quits
        """
        print(f"\n--- CALIBRATION POINT {self.current_position}/{self.num_positions} ---")
        print("Live detection mode - press 'c' when ready to confirm calibration point")
        print("Controls: 'c'=confirm, 's'=skip, 'q'=quit, 'r'=retry, 'i'=recalibrate IMU")
        
        best_detected_object = None
        user_confirmed = False
        user_response = None
        
        # Continuous detection loop with live visualization
        while not user_confirmed and user_response != 'q':
            try:
                # Get frame from camera
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                # Get raw vertices (before alignment) if raw visualization is enabled
                raw_vertices = None
                if self.visualize_raw_pointcloud:
                    raw_vertices = self.sensor_manager.get_raw_vertices()
                
                if vertices is None:
                    cv2.waitKey(30)  # Keep visualization responsive
                    continue
                
                # Detect objects with visualization
                detected_objects, timing_info, bev_image, point_cloud_view = self._detect_objects_with_viz(vertices, color_image)
                
                
                # Update best detected object (continuously track the best one)
                if detected_objects:
                    current_best = self._select_best_calibration_object(detected_objects)
                    if current_best and self._is_object_valid_for_calibration(current_best):
                        best_detected_object = current_best
                
                # Add status information to timing_info
                if best_detected_object:
                    position = self._extract_object_position(best_detected_object)
                    if position:
                        distance = np.linalg.norm(position)
                        timing_info['status_message'] = f"READY - Press 'c' to confirm (distance: {distance:.0f}mm)"
                else:
                    timing_info['status_message'] = "Waiting for valid object detection..."
                
                # Create live visualization (always showing)
                self._create_calibration_visualization(vertices, color_image, detected_objects, timing_info, bev_image, point_cloud_view, raw_vertices)
                
                # Handle non-blocking user input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    if best_detected_object:
                        user_confirmed = True
                        user_response = 'c'
                        print("[CONFIRMED] Calibration point confirmed!")
                    else:
                        print("[WARNING] No valid object detected - place object in view and try again")
                elif key == ord('s'):
                    user_response = 's'
                    print("- Skipping this calibration position")
                    break
                elif key == ord('q'):
                    user_response = 'q'
                    print("- Quitting calibration")
                    break
                elif key == ord('r'):
                    print("- Retrying object detection...")
                    best_detected_object = None
                elif key == ord('i'):
                    print("- Recalibrating IMU/gravity alignment...")
                    self._recalibrate_imu_manual()
                
            except Exception as e:
                print(f"Detection error: {e}")
                cv2.waitKey(30)  # Keep responsive even on errors
                continue
        
        # Return based on user action
        if user_response == 'q':
            return None  # Signal quit
        elif user_response == 's':
            return "SKIP"  # Signal skip
        elif user_confirmed and best_detected_object:
            return best_detected_object
        else:
            return None
    
    def _detect_objects_with_viz(self, vertices: np.ndarray, color_image: Optional[np.ndarray]) -> Tuple[List[Dict[str, Any]], Dict[str, float], Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect objects and return visualization data."""
        detection_start = time.time()
        
        try:
            # Check if strategy supports visualization data
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
            
            return detected_objects, timing_info, bev_image, point_cloud_view
            
        except Exception as e:
            print(f"Detection error: {e}")
            return [], {'total_time': time.time() - detection_start, 'num_objects': 0}, None, None
    
    
    
    def collect_calibration_data(self) -> bool:
        """Collect calibration data from all positions."""
        print(f"\nStarting calibration data collection...")
        print("=" * 50)
        
        # Get calibration positions
        positions = self.get_calibration_positions()
        print(f"Calibration positions: {len(positions)}")
        
        for i, position in enumerate(positions):
            print(f"\n--- Calibration Point {i + 1}/{len(positions)} ---")
            self.current_position = i + 1
            
            
            # Move to calibration position
            if not self.move_to_calibration_position(position):
                print(f"Failed to move to position {i + 1}")
                continue
            
            # Get actual robot position (may differ slightly from target)
            actual_robot_pose = self.robot_controller.get_current_position()
            if not actual_robot_pose:
                print(f"Failed to get robot position for point {i + 1}")
                continue
            
            # ROI adjustment - choose between automatic floor detection or manual presets
            if self.auto_floor_detection:
                # Automatic floor detection and ROI setting
                floor_detection_success = self._detect_and_set_dynamic_roi(i)
                if not floor_detection_success:
                    print(f"[AUTO-ROI] Falling back to manual ROI presets for position {i + 1}")
                    # Fall back to manual ROI if auto detection fails
                    self._apply_manual_roi_preset(i)
            else:
                # Manual ROI presets (original behavior)
                self._apply_manual_roi_preset(i)
                
            # Use continuous detection with live visualization and user confirmation
            calibration_object = self.detect_calibration_object_continuous()
            
            if calibration_object is None:
                print("Calibration cancelled by user")
                return False
            elif calibration_object == "SKIP":
                print(f"Skipping position {i + 1}")
                continue
            
            # Extract object position in RAW camera coordinates for eye-in-hand calibration
            raw_camera_position = self._extract_object_position(calibration_object, use_raw_coordinates=True)
            if not raw_camera_position:
                print(f"Failed to extract raw camera position for point {i + 1}")
                continue
            
            # Also get world-aligned position for debugging/display
            world_camera_position = self._extract_object_position(calibration_object, use_raw_coordinates=False)
            
            # Store calibration data
            calibration_point = {
                'index': i + 1,
                'robot_pose': actual_robot_pose,
                'camera_position': raw_camera_position,  # Use RAW coordinates for calibration
                'camera_position_world': world_camera_position,  # Store world coordinates for debugging
                'object_data': calibration_object
            }
            
            self.calibration_data.append(calibration_point)
            
            print(f"+ Calibration point {i + 1} recorded:")
            print(f"  Robot pose: {[f'{x:.1f}' for x in actual_robot_pose]}")
            print(f"  Raw camera position: {[f'{x:.1f}' for x in raw_camera_position]} (for calibration)")
            print(f"  World camera position: {[f'{x:.1f}' for x in world_camera_position]} (for reference)")
        
        print(f"\nCalibration data collection complete: {len(self.calibration_data)}/{len(positions)} points")
        
        if len(self.calibration_data) < 3:
            print("Error: Need at least 3 calibration points for reliable calibration")
            return False
        
        return True
    
    def calculate_eye_in_hand_transformation(self) -> Optional[np.ndarray]:
        """
        Calculate eye-in-hand transformation matrix using OpenCV's proven algorithm.
        
        Uses OpenCV's cv2.calibrateHandEye() with Tsai-Lenz method to solve the 
        AX = XB eye-in-hand calibration problem properly.
        
        Where:
        - A: Robot gripper motion transformations  
        - B: Camera observations of calibration object
        - X: Camera-to-end-effector transformation (what we solve for)
        
        Returns:
            4x4 transformation matrix from camera to end effector, or None
        """
        print("\nCalculating eye-in-hand transformation using OpenCV Tsai-Lenz method...")
        
        if len(self.calibration_data) < 3:
            print("Error: Need at least 3 calibration points")
            return None
        
        try:
            import cv2
            print("+ Using OpenCV's calibrateHandEye() for robust eye-in-hand calibration")
        except ImportError:
            print("Error: OpenCV not available, falling back to simple method")
            return self._calculate_simple_transformation()
        
        try:
            # Extract robot poses and camera observations
            robot_poses = []
            camera_positions = []
            
            for data_point in self.calibration_data:
                robot_poses.append(data_point['robot_pose'])
                camera_positions.append(data_point['camera_position'])
            
            robot_poses = np.array(robot_poses)
            camera_positions = np.array(camera_positions)
            
            print(f"Using {len(robot_poses)} calibration points")
            print(f"Robot poses shape: {robot_poses.shape}")
            print(f"Camera positions shape: {camera_positions.shape}")
            
            # Debug: Show that we're using RAW camera coordinates
            print(f"+ Using RAW camera coordinates (before gravity alignment) for eye-in-hand calibration")
            print(f"+ Sample raw camera position: {camera_positions[0]} mm")
            if len(self.calibration_data) > 0 and 'camera_position_world' in self.calibration_data[0]:
                world_pos = self.calibration_data[0]['camera_position_world']
                print(f"+ Sample world position (reference): {world_pos} mm")
            
            # Convert data to OpenCV eye-in-hand calibration format
            R_gripper2base = []  # Robot gripper rotations relative to base
            t_gripper2base = []  # Robot gripper translations relative to base
            R_target2cam = []    # Target rotations relative to camera
            t_target2cam = []    # Target translations relative to camera (our object detections)
            
            # Convert robot poses to transformation matrices
            for i, pose in enumerate(robot_poses):
                # Robot gripper pose (gripper to base frame)
                # Convert position from mm to meters for OpenCV
                t_gripper2base.append(np.array(pose[:3]) / 1000.0)  # mm to meters
                
                # Convert rotation angles to rotation matrix
                rx, ry, rz = np.radians(pose[3:6])  # degrees to radians
                R_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
                R_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
                R_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
                R_gripper = R_z @ R_y @ R_x
                R_gripper2base.append(R_gripper)
                
                # Target position relative to camera (object detection in RAW camera frame)
                # Now using RAW camera coordinates (before gravity alignment) - this is correct for eye-in-hand
                # Convert from mm to meters for OpenCV
                cam_pos = np.array(camera_positions[i]) / 1000.0  # mm to meters  
                t_target2cam.append(cam_pos)
                
                # For target rotation, we assume the calibration object has fixed orientation
                # relative to the camera (since we're detecting the same object each time)
                R_target2cam.append(np.eye(3))  # Identity - object has consistent orientation
            
            print(f"+ Converted {len(R_gripper2base)} poses to OpenCV format")
            
            # Format position ranges properly
            robot_positions = np.array([t[:2] for t in t_gripper2base])
            camera_positions = np.array([t[:2] for t in t_target2cam])
            
            robot_min = np.min(robot_positions, axis=0) * 1000
            robot_max = np.max(robot_positions, axis=0) * 1000
            cam_min = np.min(camera_positions, axis=0) * 1000  
            cam_max = np.max(camera_positions, axis=0) * 1000
            
            print(f"+ Robot position range: [{robot_min[0]:.1f}, {robot_min[1]:.1f}] to [{robot_max[0]:.1f}, {robot_max[1]:.1f}] mm")
            print(f"+ Camera detection range: [{cam_min[0]:.1f}, {cam_min[1]:.1f}] to [{cam_max[0]:.1f}, {cam_max[1]:.1f}] mm")
            
            # Use OpenCV's eye-in-hand calibration with Tsai-Lenz method
            print("+ Running OpenCV calibrateHandEye() with CALIB_HAND_EYE_TSAI method...")
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base,
                t_gripper2base=t_gripper2base,
                R_target2cam=R_target2cam,
                t_target2cam=t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI  # Proven Tsai-Lenz algorithm
            )
            
            # Build 4x4 transformation matrix from rotation and translation
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R_cam2gripper
            transformation_matrix[:3, 3] = t_cam2gripper.flatten() * 1000.0  # Convert back to mm
            
            print("+ OpenCV eye-in-hand calibration completed successfully!")
            print("Calculated transformation matrix:")
            print(transformation_matrix)
            
            # Validate the OpenCV result
            det = np.linalg.det(R_cam2gripper)
            orthogonality_error = np.linalg.norm(R_cam2gripper @ R_cam2gripper.T - np.eye(3))
            print(f"+ Rotation matrix determinant: {det:.6f} (should be ±1)")
            print(f"+ Orthogonality error: {orthogonality_error:.6f} (should be <0.01)")
            
            # Validate the transformation with our calibration data
            self._validate_transformation(transformation_matrix)
            
            return transformation_matrix
            
        except Exception as e:
            print(f"Error in OpenCV calibration: {e}")
            if "Not enough informative motions" in str(e):
                print("\nTROUBLESHOoting: OpenCV requires more diverse robot rotations!")
                print("Solutions:")
                print("1. Use --positions 7 for precision mode with diverse rotations")
                print("2. Ensure robot actually reaches all calibration positions")
                print("3. Check that rotational differences are at least 20-30 degrees")
                print("4. Make sure calibration object is detected at all positions")
            print("Falling back to simple transformation method...")
            import traceback
            traceback.print_exc()
            return self._calculate_simple_transformation()
    
    def _calculate_simple_transformation(self) -> Optional[np.ndarray]:
        """
        Fallback simple transformation calculation (original method).
        Used when OpenCV is not available.
        """
        try:
            # Extract robot poses and camera observations
            robot_poses = []
            camera_positions = []
            
            for data_point in self.calibration_data:
                robot_poses.append(data_point['robot_pose'])
                camera_positions.append(data_point['camera_position'])
            
            robot_poses = np.array(robot_poses)
            camera_positions = np.array(camera_positions)
            
            print(f"Using simple method with {len(robot_poses)} calibration points")
            
            # Create homogeneous coordinates  
            camera_homogeneous = np.column_stack([camera_positions, np.ones(len(camera_positions))])
            robot_homogeneous = np.column_stack([robot_poses[:, :3], np.ones(len(robot_poses))])
            
            # Solve least squares: camera = T * robot
            robot_pinv = np.linalg.pinv(robot_homogeneous)
            T_solution = camera_homogeneous.T @ robot_pinv.T
            
            # Create 4x4 transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :] = T_solution[:3, :]
            
            print("Simple transformation matrix:")
            print(transformation_matrix)
            
            return transformation_matrix
            
        except Exception as e:
            print(f"Error in simple transformation: {e}")
            return None
    
    def _validate_transformation(self, transformation_matrix: np.ndarray):
        """Validate the calculated transformation matrix."""
        print("\nValidating transformation...")
        print("=" * 50)
        print("COORDINATE SYSTEM VALIDATION:")
        print("+ Transformation matrix calculated using RAW camera coordinates")
        print("+ This represents the camera-to-end-effector transformation for eye-in-hand setup")
        print("=" * 50)
        
        errors = []
        
        for i, data_point in enumerate(self.calibration_data):
            robot_pose = data_point['robot_pose']
            camera_position = data_point['camera_position']  # This is now RAW camera coordinates
            
            # Transform robot position to camera frame
            robot_homogeneous = np.array([robot_pose[0], robot_pose[1], robot_pose[2], 1])
            predicted_camera = transformation_matrix @ robot_homogeneous
            predicted_camera = predicted_camera[:3]
            
            # Calculate error
            error = np.linalg.norm(np.array(camera_position) - predicted_camera)
            errors.append(error)
            
            print(f"Point {i + 1}:")
            print(f"  Measured (raw camera): {[f'{x:.1f}' for x in camera_position]} mm")
            print(f"  Predicted: {[f'{x:.1f}' for x in predicted_camera]} mm")
            print(f"  Error: {error:.2f}mm")
            
            # Show world coordinates if available for comparison
            if 'camera_position_world' in data_point:
                world_pos = data_point['camera_position_world']
                print(f"  Reference (world): {[f'{x:.1f}' for x in world_pos]} mm")
            print()
        
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"VALIDATION RESULTS:")
        print(f"Mean error: {mean_error:.2f}mm")
        print(f"Max error: {max_error:.2f}mm")
        print(f"Object distance range: {self.min_object_distance}-{self.max_object_distance}mm (lidar effective range)")
        
        if mean_error < 30:  # 3cm tolerance
            print("+ Calibration quality: EXCELLENT - Ready for precision eye-in-hand tracking")
        elif mean_error < 50:  # 5cm tolerance
            print("+ Calibration quality: GOOD - Suitable for most eye-in-hand applications")
        elif mean_error < 100:  # 10cm tolerance
            print("! Calibration quality: ACCEPTABLE - May work for coarse positioning")
        else:
            print("- Calibration quality: POOR - Consider recalibrating with more diverse poses")
    
    def save_transformation(self, transformation_matrix: np.ndarray) -> bool:
        """Save transformation matrix to file."""
        try:
            np.save(self.output_file, transformation_matrix)
            print(f"\n+ Transformation matrix saved to: {self.output_file}")
            
            # Also save calibration data for future reference
            data_file = self.output_file.replace('.npy', '_calibration_data.npy')
            np.save(data_file, self.calibration_data)
            print(f"+ Calibration data saved to: {data_file}")
            
            return True
        except Exception as e:
            print(f"Error saving transformation: {e}")
            return False
    
    
    def run_calibration(self) -> bool:
        """Run the complete eye-in-hand calibration process."""
        if not self.initialize():
            print("Failed to initialize calibration system")
            return False
        
        print("\n" + "=" * 60)
        print("EYE-IN-HAND CALIBRATION PROCESS")
        print("=" * 60)
        print("Instructions:")
        print("1. Place a calibration object in the robot workspace")
        print("2. Object must be at least 250mm from camera (lidar effective range)")
        print("3. Robot starts from LOOK_AT_TABLE position and moves to different positions")
        print("4. Keep the object visible to the camera at each position")
        print("5. The system will automatically detect the object")
        print("6. A transformation matrix will be calculated and saved")
        print("")
        print("IMPORTANT: Lidar camera minimum effective range is 250mm!")
        print("Make sure calibration object is placed on table surface.")
        print("")
        print("ROI DETECTION MODE:")
        if self.auto_floor_detection:
            print(f"- AUTO: Floor detection enabled (RANSAC plane fitting)")
            print(f"- Detection height: {self.floor_offset*1000:.0f}mm above detected floor surface")
            print(f"- Automatic fallback to manual presets if detection fails")
        else:
            print("- MANUAL: Using predefined ROI presets for each position")
            print("- Enable auto mode with: --auto-floor-detection")
        print("")
        print("VISUALIZATION:")
        print("- Always-on 2x2 grid display:")
        print("  ┌─────────────┬─────────────┐")
        if self.visualize_raw_pointcloud:
            print("  │  RGB Image  │ Raw Points  │")
        else:
            print("  │  RGB Image  │ (Disabled)  │")
        print("  ├─────────────┼─────────────┤")
        print("  │  BEV View   │ Side View   │")
        print("  └─────────────┴─────────────┘")
        print("- Live camera feed continues during robot movement")
        print("- Objects within lidar range (>250mm) shown in GREEN")
        print("- Objects too close (<250mm) shown in RED")
        print("- ROI information displayed in real-time")
        if self.visualize_raw_pointcloud:
            print("- Raw point cloud shows camera perspective with depth color-coding")
        print("")
        print("INTERACTIVE CONTROLS:")
        print("- 'c' = Confirm calibration point when object detected")
        print("- 's' = Skip current position")
        print("- 'q' = Quit calibration")  
        print("- 'r' = Retry object detection")
        print("- 'i' = Recalibrate IMU/gravity alignment")
        print("-" * 60)
        
        
        try:
            # Collect calibration data
            if not self.collect_calibration_data():
                print("Failed to collect calibration data")
                return False
            
            # Calculate transformation
            transformation_matrix = self.calculate_eye_in_hand_transformation()
            if transformation_matrix is None:
                print("Failed to calculate transformation matrix")
                return False
            
            # Save transformation
            if not self.save_transformation(transformation_matrix):
                print("Failed to save transformation matrix")
                return False
            
            print("\n" + "=" * 60)
            print("EYE-IN-HAND CALIBRATION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"Transformation matrix saved to: {self.output_file}")
            print("You can now use the eye-in-hand tracking system with:")
            print(f"  python eye_in_hand_tracker.py --transform {self.output_file}")
            
            return True
            
        except KeyboardInterrupt:
            print("\nCalibration interrupted by user")
            return False
        except Exception as e:
            print(f"Calibration failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self._cleanup()
    
    def run_calibration_with_existing_data(self) -> bool:
        """Run calibration using existing calibration data to test new OpenCV algorithm."""
        print("\n" + "=" * 60)
        print("TESTING NEW OPENCV ALGORITHM WITH EXISTING DATA")
        print("=" * 60)
        
        # Load existing calibration data
        data_file = self.output_file.replace('.npy', '_calibration_data.npy')
        
        if not os.path.exists(data_file):
            print(f"Error: Calibration data file not found: {data_file}")
            print("Run a fresh calibration first without --use-existing-data")
            return False
        
        try:
            print(f"Loading calibration data from: {data_file}")
            self.calibration_data = np.load(data_file, allow_pickle=True).tolist()
            print(f"+ Loaded {len(self.calibration_data)} calibration points")
            
            # Show the data we're using
            for i, data_point in enumerate(self.calibration_data):
                print(f"Point {i + 1}:")
                print(f"  Robot pose: {[f'{x:.1f}' for x in data_point['robot_pose']]}")
                print(f"  Camera pos (raw): {[f'{x:.1f}' for x in data_point['camera_position']]}")
                if 'camera_position_world' in data_point:
                    print(f"  Camera pos (world): {[f'{x:.1f}' for x in data_point['camera_position_world']]}")
            
            print("\n" + "=" * 60)
            print("RECALCULATING WITH NEW OPENCV ALGORITHM")
            print("=" * 60)
            
            # Calculate transformation with new OpenCV method
            transformation_matrix = self.calculate_eye_in_hand_transformation()
            if transformation_matrix is None:
                print("Failed to calculate transformation matrix with OpenCV")
                return False
            
            # Save the new transformation
            new_output_file = self.output_file.replace('.npy', '_opencv.npy')
            old_matrix = None
            
            # Compare with old matrix if it exists
            if os.path.exists(self.output_file):
                try:
                    old_matrix = np.load(self.output_file)
                    print(f"\n=== COMPARISON WITH OLD MATRIX ===")
                    print("Old matrix:")
                    print(old_matrix)
                    print(f"Old determinant: {np.linalg.det(old_matrix[:3, :3]):.6f}")
                    
                    print("\nNew OpenCV matrix:")
                    print(transformation_matrix)
                    print(f"New determinant: {np.linalg.det(transformation_matrix[:3, :3]):.6f}")
                    
                    # Show improvement
                    old_det = np.linalg.det(old_matrix[:3, :3])
                    new_det = np.linalg.det(transformation_matrix[:3, :3])
                    print(f"\nImprovement:")
                    print(f"  Old determinant error: {abs(abs(old_det) - 1):.6f}")
                    print(f"  New determinant error: {abs(abs(new_det) - 1):.6f}")
                    
                except Exception as e:
                    print(f"Could not load old matrix: {e}")
            
            # Save new transformation
            try:
                np.save(new_output_file, transformation_matrix)
                print(f"\n+ New OpenCV transformation saved to: {new_output_file}")
                
                # Also overwrite the main file
                np.save(self.output_file, transformation_matrix)
                print(f"+ Main transformation file updated: {self.output_file}")
                
            except Exception as e:
                print(f"Error saving transformation: {e}")
                return False
            
            print("\n" + "=" * 60)
            print("OPENCV CALIBRATION TEST COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"New transformation matrix saved to: {self.output_file}")
            print("The eye-in-hand system should now work with a mathematically valid matrix!")
            
            return True
            
        except Exception as e:
            print(f"Error processing existing data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _cleanup(self):
        """Cleanup resources."""
        print("\nCleaning up calibration system...")
        
        if self.sensor_manager:
            self.sensor_manager.stop()
        
        if self.robot_controller:
            # Cleanup async system if enabled
            if self.robot_controller.async_mode_enabled:
                self.robot_controller.cleanup_async_system()
            self.robot_controller.disconnect()
        
        cv2.destroyAllWindows()
        print("+ Visualization windows closed")
        print("+ Calibration system shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Eye-in-Hand Calibration')
    parser.add_argument('--positions', type=int, default=5,
                       help='Number of calibration positions (3=quick, 5=standard, 7+=precision)')
    parser.add_argument('--output', default='eye_in_hand_transform.npy',
                       help='Output transformation matrix file')
    parser.add_argument('--use-existing-data', action='store_true',
                       help='Use existing calibration data to test new OpenCV algorithm')
    parser.add_argument('--auto-floor-detection', action='store_true', default=True,
                       help='Enable automatic floor detection and ROI adjustment (default: manual ROI)')
    parser.add_argument('--floor-offset', type=float, default=0.01,
                       help='Height offset above detected floor in meters (default: 0.04m = 40mm)')
    parser.add_argument('--visualize-raw-pointcloud', action='store_true', default=True,
                       help='Enable raw point cloud visualization (shows unprocessed RealSense data, default: enabled)')
    parser.add_argument('--no-raw-pointcloud', action='store_true',
                       help='Disable raw point cloud visualization')
    
    args = parser.parse_args()
    
    # Handle raw pointcloud visualization logic
    if args.no_raw_pointcloud:
        args.visualize_raw_pointcloud = False
    
    print("Eye-in-Hand Calibration System")
    print("=" * 40)
    print(f"Calibration positions: {args.positions}")
    print(f"Output file: {args.output}")
    print(f"Auto floor detection: {'Enabled' if args.auto_floor_detection else 'Disabled (Manual ROI)'}")
    if args.auto_floor_detection:
        print(f"Floor offset: {args.floor_offset*1000:.0f}mm above detected floor")
    print(f"Raw point cloud visualization: {'Enabled' if args.visualize_raw_pointcloud else 'Disabled'}")
    print("=" * 40)
    
    try:
        calibrator = EyeInHandCalibrator(
            num_positions=args.positions,
            output_file=args.output,
            auto_floor_detection=args.auto_floor_detection,
            floor_offset=args.floor_offset,
            visualize_raw_pointcloud=args.visualize_raw_pointcloud
        )
        
        if args.use_existing_data:
            success = calibrator.run_calibration_with_existing_data()
        else:
            success = calibrator.run_calibration()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()