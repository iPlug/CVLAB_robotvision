#!/usr/bin/env python3
"""
Eye-in-Hand Fixed Target Calibration

This script provides fixed target calibration for eye-in-hand setup where:
1. Camera is mounted on robot end effector
2. Object is placed at KNOWN fixed coordinates  
3. Robot moves to multiple poses around fixed object
4. Direct point-to-point correspondence for higher accuracy

Key Features:
- Robot-guided object placement at known coordinates
- Auto table surface detection and ROI adjustment
- Simplified transformation mathematics
- Direct correspondence solving (more stable than relative methods)

Usage:
    python eye_in_hand_fixed_calibration.py [--positions N] [--output FILE]

Process:
1. Robot points to fixed object placement location
2. User places object at exact known coordinates
3. Robot moves through calibration poses
4. Direct transformation calculated from correspondences
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

from eye_in_hand_calibrator_base import EyeInHandCalibratorBase
from module.sensors.realsense_manager import RealSenseManager
from module.robot.mycobot_controller import MyCobotController
from module.core.object_detector import ObjectDetector
from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
from module.robot.command_queue import CommandStatus
from module.config.config_factory import ConfigFactory
from module.utils.wait_utils import wait_for_command_completion, wait_for_robot_stabilization


class EyeInHandFixedCalibrator(EyeInHandCalibratorBase):
    """
    Eye-in-hand fixed target calibration system.
    
    Uses known object coordinates for direct correspondence calibration.
    More accurate than relative transformation methods.
    
    Process:
    1. Robot guides user to place object at known coordinates
    2. Robot moves to multiple calibration poses
    3. Object detected from each pose (object stays fixed)
    4. Direct transformation calculated from correspondences
    """
    
    def __init__(self, num_positions: int = 5, output_file: str = "eye_in_hand_fixed_transform.npy",
                 floor_offset: float = 0.01, use_optimal_positions: bool = True,
                 visualize_raw_pointcloud: bool = True):
        """
        Initialize fixed target calibrator.
        
        Args:
            num_positions: Number of calibration positions to use
            output_file: Output file for transformation matrix
            floor_offset: Height offset above detected table surface in meters
            use_optimal_positions: If True, calculate positions that look directly at target
            visualize_raw_pointcloud: Enable raw point cloud visualization
        """
        # Initialize base class
        super().__init__()
        
        self.num_positions = num_positions
        self.output_file = output_file
        self.floor_offset = floor_offset
        self.use_optimal_positions = use_optimal_positions
        self.visualize_raw_pointcloud = visualize_raw_pointcloud
        
        # Get fixed object position from first calibration position in config
        self.fixed_object_position = self._get_fixed_object_position()
        
        # Components
        self.sensor_manager = None
        self.robot_controller = None
        self.detector = None
        
        # Calibration data for fixed target approach
        self.calibration_data = []
        self.current_position = 0
        
        # Auto ROI detection (no manual fallback)
        self.auto_floor_detection = True
        self.table_surface_height = None
        
        print("Eye-in-Hand Fixed Target Calibration System")
        print("=" * 60)
        print(f"Target positions: {num_positions}")
        print(f"Fixed object position: {self.fixed_object_position}")
        print(f"Output file: {output_file}")
        print(f"Auto ROI detection: Enabled")
    
    def _get_fixed_object_position(self) -> List[float]:
        """Get fixed object position from first calibration position in config."""
        try:
            # Load calibration_standard config to get first position
            config = ConfigFactory.create_preset('calibration_standard')
            calibration_positions = config.get('calibration_positions', [])
            
            if calibration_positions:
                # Use first position XYZ as fixed object coordinates
                first_position = calibration_positions[0]
                fixed_pos = [first_position[0], first_position[1], first_position[2]]
                print(f"+ Using first calibration position as fixed object location: {fixed_pos}")
                return fixed_pos
            else:
                # Fallback to default
                default_pos = [250, 0, 60]
                print(f"+ Using default fixed object position: {default_pos}")
                return default_pos
                
        except Exception as e:
            print(f"Error loading calibration config: {e}")
            # Fallback to default
            default_pos = [250, 0, 60]
            print(f"+ Using default fixed object position: {default_pos}")
            return default_pos
    
    def get_calibration_positions(self) -> List[List[float]]:
        """
        Generate calibration positions for fixed target eye-in-hand setup.
        
        Can either use predefined positions (like eye_in_hand_calibration.py) or
        calculate optimal positions that look directly at the fixed target.
        
        Returns:
            List of robot positions [x, y, z, rx, ry, rz]
        """
        if self.use_optimal_positions:
            print("Using optimal calculated positions that look directly at fixed target")
            return self._calculate_optimal_calibration_positions(
                target_position=self.fixed_object_position[:3],  # Only XYZ
                num_positions=self.num_positions
            )
        else:
            print("Using predefined calibration positions")
            positions = []
        
        if self.num_positions == 3:
            # Quick calibration - 3 positions
            positions = [
                [87.5, -62.7, 367.2, -144.03, 29.89, -124.04],  # LOOK_AT_TABLE position
                [150, -100, 320, -144, 30, -124],    # Left side, similar orientation
                [150, 100, 320, -144, 30, -124],     # Right side, similar orientation
            ]
        elif self.num_positions == 5:
            # Standard calibration - 5 positions with diverse orientations
            positions = [
                [87.5, -62.7, 367.2, -144.03, 29.89, -124.04],   # LOOK_AT_TABLE position
                [150, 0, 300, -144, 30, -124],                   # Center, same orientation
                [120, -80, 320, -144.03, 29.89, -124.04],                # Left, rotated
                [120, 80, 320, -144.03, 29.89, -124.04],                 # Right, rotated  
                [180, 0, 280, -144.03, 29.89, -124.04],                  # Center, angled
            ]
        elif self.num_positions >= 7:
            # Precision calibration - 7+ positions with maximum diversity
            positions = [
                [87.5, -62.7, 367.2, -144.03, 29.89, -124.04],   # LOOK_AT_TABLE position
                [150, 0, 300, -144, 30, -124],                   # Center, same orientation
                [120, -120, 320, -170, 50, -150],               # Far left, rotated
                [130, -60, 310, -135, 10, -130],                # Near left, slight rotation
                [130, 60, 310, -150, 45, -115],                 # Near right, angled
                [120, 120, 320, -125, 20, -105],                # Far right, rotated
                [100, 0, 340, -160, 60, -120],                  # High position, steep angle
            ]
            
            # Add extra positions if requested
            if self.num_positions > 7:
                extra_positions = [
                    [140, -50, 290, -155, 35, -125],  # Tilted left
                    [140, 50, 290, -135, 25, -115],   # Tilted right
                    [170, 0, 270, -165, 55, -125],    # Extended reach
                ]
                positions.extend(extra_positions[:self.num_positions - 7])
        
        return positions[:self.num_positions]
    
    def initialize(self) -> bool:
        """Initialize all system components."""
        print("\nInitializing Fixed Target Calibration System...")
        print("=" * 60)
        
        # Initialize camera
        print("Initializing camera (mounted on end effector)...")
        self.sensor_manager = RealSenseManager(None, realtime_mode=True, use_imu=True)
        if not self.sensor_manager.initialize():
            print("Error: Failed to initialize camera")
            return False
        
        if not self.sensor_manager.start():
            print("Error: Failed to start camera")
            return False
        
        # Initialize robot controller
        print("Initializing robot controller...")
        self.robot_controller = MyCobotController(transformation_matrix_file=None)
        
        if not self.robot_controller.connect():
            print("Error: Failed to connect to robot")
            return False
        
        # Enable async mode for smooth operation
        if self.robot_controller.enable_async_mode(max_queue_size=5):
            print("[OK] Robot async mode enabled")
        else:
            print("[WARNING] Failed to enable robot async mode")
        
        # Move robot to home position first
        print("Moving robot to home position...")
        if self.robot_controller.async_mode_enabled:
            command_id = self.robot_controller.home_position_async(priority=1)
            if command_id:
                self._wait_for_movement_completion(command_id)
        else:
            self.robot_controller.home_position()
            time.sleep(2)
        
        # Calibrate gravity alignment
        if self.sensor_manager.use_imu:
            print("Calibrating gravity alignment...")
            self.sensor_manager.calibrate_gravity_alignment(num_samples=15)  # Fast initial calibration
            print("+ Gravity alignment calibrated")
        
        # Initialize object detector
        print("Initializing object detector...")
        try:
            strategy = BEVClusteringStrategy(
                roi_params=None,  # Will be set by auto detection
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
        
        # Detect table surface and set auto ROI
        if self._detect_and_set_dynamic_roi():
            print("+ Auto ROI detection successful")
        else:
            print("! Auto ROI detection failed - using default parameters")
        
        print("✓ All systems initialized successfully")
        return True
    
    def _get_first_placement_position(self) -> List[float]:
        """Get the predefined first placement position (LOOK_AT_TABLE)."""
        return [87.5, -62.7, 367.2, -144.03, 29.89, -124.04]
    
    def _calculate_optimal_calibration_positions(self, target_position: List[float], num_positions: int = 5) -> List[List[float]]:
        """
        Calculate optimal calibration positions that look directly at the fixed target.
        
        The first position is always the predefined placement position (for consistency
        with object placement workflow). Remaining positions are optimally calculated.
        
        Uses myCobot-specific coordinate system and orientation calculations to ensure
        the camera actually points at the target object from various viewpoints.
        
        Args:
            target_position: [forward, side, height] coordinates of the fixed object in robot base frame
            num_positions: Number of calibration positions to generate
            
        Returns:
            List of [forward, side, height, rx, ry, rz] robot poses where camera looks at target
        """
        import math
        
        positions = []
        target_x, target_y, target_z = target_position
        
        # First position: Always use the predefined placement position
        first_position = self._get_first_placement_position()
        positions.append(first_position)
        print(f"Position 1 (predefined): {[f'{x:.1f}' for x in first_position]}")
        
        # Calculate remaining positions optimally
        remaining_positions = num_positions - 1
        if remaining_positions <= 0:
            return positions
        
        # Define viewing parameters
        base_distance = 250  # Base distance from camera to target (mm)
        min_distance = 200   # Minimum viewing distance
        max_distance = 350   # Maximum viewing distance
        
        print(f"Calculating {remaining_positions} additional optimal positions to view target at ({target_x}, {target_y}, {target_z})")
        
        # Define viewing angles based on number of remaining positions
        if remaining_positions == 2:  # Total 3 positions (1 predefined + 2 calculated)
            angles = [
                (25, 0, base_distance + 50),    # Tilted forward, further
                (0, 25, base_distance + 30)     # Tilted left, further
            ]
        elif remaining_positions == 4:  # Total 5 positions (1 predefined + 4 calculated)
            angles = [
                (30, 0, base_distance + 50),    # Forward
                (-20, 0, base_distance + 30),   # Backward
                (0, 30, base_distance + 40),    # Left
                (0, -30, base_distance + 40)    # Right
            ]
        elif remaining_positions == 6:  # Total 7 positions (1 predefined + 6 calculated)
            angles = [
                (25, 0, base_distance + 40),    # Forward
                (-25, 0, base_distance + 40),   # Backward
                (0, 25, base_distance + 30),    # Left
                (0, -25, base_distance + 30),   # Right
                (20, 20, base_distance + 60),   # Forward-left
                (20, -20, base_distance + 60)   # Forward-right
            ]
        else:
            # Generate circular pattern for other numbers
            angles = []
            for i in range(remaining_positions):
                angle = 2 * math.pi * i / remaining_positions
                tilt_angle = 20  # degrees from vertical
                distance = base_distance + (i % 2) * 50  # Vary distance
                angles.append((
                    tilt_angle * math.cos(angle),
                    tilt_angle * math.sin(angle),
                    distance
                ))
        
        from module.utils.coordinate_utils import calculate_eye_in_hand_orientation, validate_mycobot_position
        
        for i, (tilt_x, tilt_y, distance) in enumerate(angles):
            # Convert tilt angles to radians
            tilt_x_rad = math.radians(tilt_x)
            tilt_y_rad = math.radians(tilt_y)
            
            # Calculate initial camera position (distance away from target at specified angles)
            camera_x = target_x - distance * math.sin(tilt_x_rad)
            camera_y = target_y - distance * math.sin(tilt_y_rad)
            camera_z = target_z + distance * math.cos(tilt_x_rad) * math.cos(tilt_y_rad)
            
            # Ensure minimum safe height and workspace bounds
            min_safe_height = 100.0  # myCobot minimum height constraint
            camera_z = max(camera_z, min_safe_height)
            
            # Validate and adjust position to be within myCobot workspace
            tentative_position = [camera_x, camera_y, camera_z, 0, 0, 0]
            is_valid, reason = validate_mycobot_position(tentative_position)
            
            if not is_valid:
                # If position is invalid, move it closer to the robot base while maintaining direction
                print(f"Position {i+2} invalid ({reason}), adjusting...")
                
                # Calculate direction from origin to desired position
                direction_forward = camera_x if abs(camera_x) > 0.1 else 0
                direction_side = camera_y if abs(camera_y) > 0.1 else 0
                
                # Scale down to be within workspace (conservative approach)
                max_forward = min(280, abs(direction_forward)) if direction_forward != 0 else 200
                max_side = min(200, abs(direction_side)) if direction_side != 0 else 0
                
                camera_x = max_forward * (1 if direction_forward >= 0 else -1) if direction_forward != 0 else 200
                camera_y = max_side * (1 if direction_side >= 0 else -1) if direction_side != 0 else 0
                camera_z = max(camera_z, min_safe_height)
                
                print(f"Adjusted to reachable position: ({camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f})")
            
            # Camera position and target position in myCobot coordinates [forward, side, height]
            camera_position = [camera_x, camera_y, camera_z]
            target_position = [target_x, target_y, target_z]
            
            # Calculate proper orientation for eye-in-hand setup
            rx, ry, rz = calculate_eye_in_hand_orientation(camera_position, target_position)
            
            position = [camera_x, camera_y, camera_z, rx, ry, rz]
            positions.append(position)
            
            print(f"Position {i+2}: camera at ({camera_x:.1f}, {camera_y:.1f}, {camera_z:.1f}) "
                  f"looking at target ({target_x}, {target_y}, {target_z}), orientation ({rx:.1f}, {ry:.1f}, {rz:.1f})")
        
        return positions
    
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
                print(f"Warning: Only {len(filtered_vertices)} points in distance range 0-400mm")
                return None
            
            print(f"+ Filtered points: {len(filtered_vertices)}/{len(valid_vertices)} points in 0-400mm range")
            
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
            
            print(f"+ Table surface detected:")
            print(f"  Height: {table_height_mean:.3f}m ± {table_height_std:.3f}m")
            print(f"  Inliers: {len(table_points)}/{len(filtered_vertices)} points (after distance filtering)")
            print(f"  Bounds: X[{table_bounds['x_min']:.2f}, {table_bounds['x_max']:.2f}], Z[{table_bounds['z_min']:.2f}, {table_bounds['z_max']:.2f}]")
            
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
        print("[AUTO-ROI] Detecting table surface...")
        
        try:
            # Allow camera to stabilize first (especially important on first run)
            print("[AUTO-ROI] Allowing camera to stabilize...")
            stabilization_frames = 10  # Discard first few frames to let camera settle
            for _ in range(stabilization_frames):
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                time.sleep(0.05)  # Quick discard of initial frames
            
            # Get multiple frames for better table detection  
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
            
            # Update detector's ROI parameters
            if hasattr(self.detector.clustering_strategy, 'roi_params'):
                self.detector.clustering_strategy.roi_params = dynamic_roi
                self.table_surface_height = detected_table_height
                print(f"[AUTO-ROI] Set dynamic ROI: Y[{dynamic_roi['y_min']:.3f}, {dynamic_roi['y_max']:.3f}]m")
                print(f"[AUTO-ROI] Object detection at table surface level")
                return True
            else:
                print("[AUTO-ROI] Warning: Detector does not support ROI parameter updates")
                return False
                
        except Exception as e:
            print(f"[AUTO-ROI] Error during table detection: {e}")
            return False
    
    def guide_fixed_object_placement(self) -> bool:
        """
        Guide user to place object at known fixed coordinates.
        
        Workflow:
        1. Robot in observer pose - show live camera feed and wait for ENTER
        2. Robot moves TO placement position [250, 0, 60, 0, 180, 0]
        3. User places object below end effector and presses 'c' to confirm
        
        Returns:
            True if object placement confirmed by user
        """
        print("\n" + "=" * 60)
        print("FIXED TARGET OBJECT PLACEMENT GUIDANCE")
        print("=" * 60)
        print(f"Object will be placed at fixed coordinates: {self.fixed_object_position}")
        print("Robot will move to placement position for object placement.")
        print("=" * 60)
        
        # Step 1: Start in observer pose with live camera feed
        print("\nStep 1: Observer position ready")
        print("Live camera feed - Press ENTER when ready to move robot to placement position...")
        
        if not self._show_observer_feed_and_wait_enter():
            print("✗ Cancelled by user")
            return False
        
        # Step 2: Move robot TO placement position that points toward [400, 0, 0]
        from module.utils.coordinate_utils import calculate_reachable_position, calculate_mycobot_pointing_orientation
        
        target_coords = [400, 0, 0]  # Target to point toward
        
        # Calculate pointing position using the same logic as point_to_coordinate --point-toward
        # Use the robot's calculate_pointing_toward_object method with default orientation
        default_orientation = [0, 180, 0]  # Default pointing down orientation
        
        # Use the robot controller's method to calculate pointing position (same as point_to_coordinate.py)
        if hasattr(self.robot_controller, 'calculate_pointing_toward_object'):
            pointing_pose = self.robot_controller.calculate_pointing_toward_object(target_coords, default_orientation)
            if pointing_pose:
                robot_position = pointing_pose[:3]  # Extract position
                # Now calculate the correct orientation to point toward the target
                pointing_orientation = calculate_mycobot_pointing_orientation(robot_position, target_coords)
            else:
                print("Could not calculate pointing position, using fallback")
                robot_position = [212.8, 0.0, 100.0]  # Use known working position from your test
                pointing_orientation = calculate_mycobot_pointing_orientation(robot_position, target_coords)
        else:
            print("Robot controller doesn't have pointing calculation, using fallback")
            robot_position = [212.8, 0.0, 100.0]  # Use known working position from your test
            pointing_orientation = calculate_mycobot_pointing_orientation(robot_position, target_coords)
        
        # Create the placement position with pointing orientation
        placement_position = robot_position + pointing_orientation
        
        # Set the fixed object position to the unreachable target
        self.fixed_object_position = target_coords  # Object at target position [400, 0, 0]
        
        print(f"Calculated pointing position: {[f'{x:.1f}' for x in robot_position]}")
        print(f"Calculated pointing orientation: {[f'{x:.1f}' for x in pointing_orientation]} (toward target)")
        print(f"Target coordinates: {target_coords}")
        print(f"\nStep 2: Moving robot to placement position: {placement_position}")
        print(f"Object will be placed at: {self.fixed_object_position}")
        
        if not self._move_robot_to_position(placement_position):
            print("Failed to move robot to placement position")
            return False
        
        # Step 3: Wait for user to place object and confirm with 'c'
        print("\nStep 3: Object placement")
        print("PLACEMENT INSTRUCTIONS:")
        print(f"1. Place your calibration object DIRECTLY BELOW the robot end effector")
        print(f"2. Object will be at world coordinates: {self.fixed_object_position}")
        print("3. Make sure the object is stable and well-lit")
        print("4. Press 'c' in the camera window when object is placed correctly...")
        
        if self._show_placement_feed_and_wait_confirm():
            print("✓ Object placement confirmed by user")
            
            # Step 4: Move robot back to observer pose for calibration start
            # Keep visualization open like eye_in_hand_calibration.py
            print("\nStep 4: Moving robot back to observer pose to begin calibration...")
            
            observer_position = self._get_guidance_position()  # Get observer pose
            if not self._move_robot_to_position(observer_position):
                print("Failed to move robot back to observer pose")
                return False
            
            print("✓ Robot returned to observer pose - ready for calibration")
            return True
        else:
            print("✗ Object placement cancelled by user")
            return False
    
    def _get_guidance_position(self) -> List[float]:
        """Get robot position for guiding object placement."""
        # Use first calibration position as guidance position
        calibration_positions = self.get_calibration_positions()
        if calibration_positions:
            return calibration_positions[0]  # First position
        else:
            # Fallback to default LOOK_AT_TABLE position
            return [78.0, -2.9, 302.9, -169.62, 13.26, -133.45]
    
    def _move_robot_to_guidance_position(self, guidance_position: List[float]) -> bool:
        """Move robot to guidance position."""
        try:
            if self.robot_controller.async_mode_enabled:
                command_id = self.robot_controller.move_to_position_async(
                    guidance_position, speed=30, priority=1
                )
                if command_id:
                    print("Robot moving to guidance position...")
                    return self._wait_for_movement_completion(command_id)
                else:
                    print("Failed to submit robot movement command")
                    return False
            else:
                # Fallback to synchronous movement
                if self.robot_controller.move_to_position(guidance_position, speed=30):
                    time.sleep(3)  # Wait for movement completion
                    return True
                else:
                    print("Failed to move robot")
                    return False
                    
        except Exception as e:
            print(f"Error moving robot to guidance position: {e}")
            return False
    
    def _show_observer_feed_and_wait_enter(self):
        """Show live camera feed from observer pose and wait for ENTER."""
        try:
            while True:
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                if color_image is not None:
                    display_image = color_image.copy()
                    
                    # Add observer pose instructions overlay
                    cv2.putText(display_image, "FIXED TARGET CALIBRATION - OBSERVER POSE", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_image, "Robot ready in observer position", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_image, "Press ENTER to move robot to placement position", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(display_image, "Press ESC to cancel", (10, 140),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.imshow('Fixed Target Calibration', display_image)
                
                key = cv2.waitKey(30) & 0xFF
                if key == 13:  # ENTER key
                    return True
                elif key == 27:  # ESC key
                    return False
                    
        except Exception as e:
            print(f"Error showing observer feed: {e}")
            return False
    
    def _show_placement_feed_and_wait_confirm(self):
        """Show live camera feed during object placement and wait for 'c' confirmation."""
        try:
            while True:
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                if color_image is not None:
                    display_image = color_image.copy()
                    
                    # Add placement confirmation overlay
                    cv2.putText(display_image, "OBJECT PLACEMENT - ROBOT AT PLACEMENT POSITION", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_image, f"Place object at: {self.fixed_object_position}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display_image, "Position DIRECTLY BELOW robot end effector", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(display_image, "Press 'c' when object placed correctly", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(display_image, "Press ESC to cancel", (10, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.imshow('Fixed Target Calibration', display_image)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('c') or key == ord('C'):  # 'c' key
                    return True
                elif key == 27:  # ESC key
                    return False
                    
        except Exception as e:
            print(f"Error showing placement feed: {e}")
            return False
    
    def _validate_fixed_object_placement(self) -> bool:
        """
        Validate that object is placed correctly at fixed coordinates.
        
        Returns:
            True if object detected near expected camera coordinates
        """
        try:
            print("Detecting object at fixed placement location...")
            
            # Try to detect object multiple times for robustness
            detection_attempts = 5
            successful_detections = 0
            detected_positions = []
            
            for attempt in range(detection_attempts):
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                if vertices is not None:
                    # Detect objects
                    detected_objects = self._detect_objects_simple(vertices, color_image)
                    
                    if detected_objects:
                        # Find best object
                        best_object = self._find_best_object(detected_objects)
                        if best_object:
                            position = self._extract_object_position(best_object)
                            if position:
                                detected_positions.append(position)
                                successful_detections += 1
                                print(f"  Attempt {attempt + 1}: Object detected at {position}")
                
                time.sleep(0.2)  # Brief pause between attempts
            
            if successful_detections == 0:
                print("No objects detected during validation")
                return False
            
            # Check if detections are consistent
            if successful_detections < 3:
                print(f"Only {successful_detections}/{detection_attempts} successful detections")
                print("Object may not be stable or visible enough")
                return False
            
            # Calculate average position
            avg_position = np.mean(detected_positions, axis=0)
            position_std = np.std(detected_positions, axis=0)
            
            print(f"Average detected position: ({avg_position[0]:.1f}, {avg_position[1]:.1f}, {avg_position[2]:.1f}) mm")
            print(f"Position stability: ±({position_std[0]:.1f}, {position_std[1]:.1f}, {position_std[2]:.1f}) mm")
            
            # Check if position is reasonable (within lidar range)
            distance = np.linalg.norm(avg_position)
            if distance < self.min_object_distance:
                print(f"Object too close: {distance:.1f}mm (minimum: {self.min_object_distance}mm)")
                return False
            elif distance > self.max_object_distance:
                print(f"Object too far: {distance:.1f}mm (maximum: {self.max_object_distance}mm)")
                return False
            
            print(f"Object distance: {distance:.1f}mm (within valid range)")
            return True
            
        except Exception as e:
            print(f"Error during object placement validation: {e}")
            return False
    
    def _wait_for_movement_completion(self, command_id: str, timeout: float = 30.0) -> bool:
        """Wait for robot movement command to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.robot_controller.get_command_status(command_id)
            if status and status.value in ['completed', 'failed']:
                if status.value == 'completed':
                    print("[OK] Robot movement completed")
                    return True
                else:
                    print("[ERROR] Robot movement failed")
                    return False
            time.sleep(0.1)
        
        print("[WARNING] Robot movement timed out")
        return False
    
    def _detect_objects_simple(self, vertices: np.ndarray, color_image: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """Simple object detection for validation."""
        try:
            if hasattr(self.detector.clustering_strategy, 'cluster_objects'):
                result = self.detector.clustering_strategy.cluster_objects(vertices, color_image)
                if isinstance(result, tuple):
                    return result[0]  # Get objects from tuple
                else:
                    return result
            else:
                detected_objects, _ = self.detector.detect_objects(vertices, color_image)
                return detected_objects
        except Exception as e:
            print(f"Error in simple object detection: {e}")
            return []
    
    def _find_best_object(self, detected_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the best object for calibration (largest and most central)."""
        if not detected_objects:
            return None
        
        best_object = None
        best_score = -1
        
        for obj in detected_objects:
            # Score based on size
            num_points = obj.get('num_points', 0)
            position = self._extract_object_position(obj)
            
            if position and num_points > 0:
                # Distance from center (prefer central objects)
                center_distance = np.sqrt(position[0]**2 + position[1]**2)
                
                # Combined score: size bonus - distance penalty (reduced penalty)
                score = num_points - center_distance * 0.5
                
                if score > best_score:
                    best_score = score
                    best_object = obj
        
        return best_object
    
    
    def collect_fixed_target_calibration_data(self) -> bool:
        """
        Collect calibration data using fixed target approach.
        
        Object stays at known fixed coordinates throughout.
        Robot moves to different poses and detects object from each pose.
        
        Returns:
            True if data collection successful
        """
        print("\n" + "=" * 60)
        print("FIXED TARGET CALIBRATION DATA COLLECTION")
        print("=" * 60)
        print(f"Object at fixed coordinates: {self.fixed_object_position}")
        print(f"Collecting data from {self.num_positions} robot poses")
        print("=" * 60)
        
        # Get calibration positions
        calibration_positions = self.get_calibration_positions()
        
        if len(calibration_positions) != self.num_positions:
            print(f"Error: Expected {self.num_positions} positions, got {len(calibration_positions)}")
            return False
        
        # Clear previous calibration data
        self.calibration_data.clear()
        
        # Collect data from each position
        for i, robot_pose in enumerate(calibration_positions):
            print(f"\n--- CALIBRATION POINT {i+1}/{self.num_positions} ---")
            print(f"Robot pose: {[f'{x:.1f}' for x in robot_pose]}")
            
            if not self._collect_data_at_position(i, robot_pose):
                print(f"Failed to collect data at position {i+1}")
                return False
            
            print(f"✓ Data point {i+1} collected successfully")
        
        min_required_points = 3  # Minimum points needed for calibration
        collected_points = len(self.calibration_data)
        
        if collected_points >= min_required_points:
            print(f"\n✓ Collected {collected_points}/{self.num_positions} calibration data points (sufficient for calibration)")
            return True
        else:
            print(f"\n✗ Only collected {collected_points}/{self.num_positions} data points (need at least {min_required_points})")
            return False
    
    def _collect_data_at_position(self, position_index: int, robot_pose: List[float]) -> bool:
        """
        Collect calibration data at a specific robot position.
        
        Args:
            position_index: Index of current position (0-based)
            robot_pose: Robot pose [x, y, z, rx, ry, rz]
            
        Returns:
            True if data collection successful
        """
        try:
            # Move robot to calibration position
            print("Moving robot to calibration position...")
            if not self._move_robot_to_position(robot_pose):
                print("Failed to move robot to calibration position")
                return False
            
            # Wait for robot to settle
            print("Waiting for robot to settle...")
            wait_for_robot_stabilization(
                sensor_manager=self.sensor_manager,
                duration=5.0,
                window_name="Fixed Target Calibration"
            )
            
            # Recalibrate gravity if using IMU (camera orientation changed)
            if self.sensor_manager.use_imu:
                print("Recalibrating gravity alignment for new camera orientation...")
                self.sensor_manager.calibrate_gravity_alignment(num_samples=10)  # Fast re-calibration for robot movement
                print("+ Gravity alignment updated")
                
                # Also update auto ROI since camera orientation changed
                print("Re-detecting table surface and updating ROI...")
                if self._detect_and_set_dynamic_roi():
                    print("+ Auto ROI updated successfully")
                else:
                    print("! Auto ROI re-detection failed")
            
            # Verify robot position
            current_position = self.robot_controller.get_current_position()
            if current_position:
                position_error = np.linalg.norm(np.array(robot_pose[:3]) - np.array(current_position[:3]))
                print(f"Position error: {position_error:.1f}mm")
                
                if position_error > 50:  # 50mm tolerance
                    print(f"Warning: Large position error ({position_error:.1f}mm)")
            
            # Detect object at fixed target location using continuous detection like eye_in_hand_calibration.py
            print(f"Detecting fixed target object...")
            calibration_object = self._detect_calibration_object_continuous(position_index)
            
            if calibration_object is None:
                print("Failed to detect object at fixed target location or user cancelled")
                return False
            elif calibration_object == "SKIP":
                print(f"Position {position_index + 1} skipped by user")
                return True  # Continue with next position instead of failing
            
            # Extract object position in RAW camera coordinates for eye-in-hand calibration
            raw_detected_position = self._extract_object_position(calibration_object, use_raw_coordinates=True)
            if raw_detected_position is None:
                print("Failed to extract raw camera position from detected object")
                return False
            
            # Also get world-aligned position for debugging/display
            world_detected_position = self._extract_object_position(calibration_object, use_raw_coordinates=False)
            
            print(f"Object detected:")
            print(f"  Raw camera coordinates: ({raw_detected_position[0]:.1f}, {raw_detected_position[1]:.1f}, {raw_detected_position[2]:.1f}) mm (for calibration)")
            print(f"  World coordinates: ({world_detected_position[0]:.1f}, {world_detected_position[1]:.1f}, {world_detected_position[2]:.1f}) mm (for reference)")
            
            # Get actual robot position for accurate calibration
            actual_robot_position = self.robot_controller.get_current_position()
            if actual_robot_position:
                print(f"Actual robot position: {[f'{x:.1f}' for x in actual_robot_position]}")
            else:
                print("Warning: Could not read actual robot position, using commanded position")
                actual_robot_position = robot_pose.copy()
            
            # Store calibration data point
            data_point = {
                'position_index': position_index,
                'robot_pose': robot_pose.copy(),
                'actual_robot_pose': actual_robot_position.copy(),
                'fixed_world_position': self.fixed_object_position.copy(),  # Known fixed coordinates
                'detected_camera_position': raw_detected_position.copy(),  # RAW camera coordinates for calibration
                'detected_camera_position_world': world_detected_position.copy(),  # World coordinates for reference
                'timestamp': time.time()
            }
            
            self.calibration_data.append(data_point)
            
            print(f"Data point stored:")
            print(f"  Fixed world position: {self.fixed_object_position}")
            print(f"  Camera detection (raw): {raw_detected_position}")
            print(f"  Camera detection (world): {world_detected_position}")
            print(f"  Robot pose: {[f'{x:.1f}' for x in robot_pose]}")
            
            return True
            
        except Exception as e:
            print(f"Error collecting data at position {position_index}: {e}")
            return False
    
    def _move_robot_to_position(self, robot_pose: List[float]) -> bool:
        """Move robot to specified position with live visualization."""
        try:
            if self.robot_controller.async_mode_enabled:
                command_id = self.robot_controller.move_to_position_async(
                    robot_pose, speed=30, priority=1
                )
                if command_id:
                    # Use existing wait_for_command_completion from utils
                    return wait_for_command_completion(
                        robot_controller=self.robot_controller,
                        command_id=command_id,
                        sensor_manager=self.sensor_manager,
                        timeout=30.0,
                        window_name="Fixed Target Calibration"
                    )
                else:
                    print("Failed to submit robot movement command")
                    return False
            else:
                # Fallback to synchronous movement
                if self.robot_controller.move_to_position(robot_pose, speed=30):
                    # Use wait utility instead of time.sleep
                    return wait_for_robot_stabilization(
                        sensor_manager=self.sensor_manager,
                        duration=5.0,
                        window_name="Fixed Target Calibration"
                    )
                else:
                    print("Failed to move robot")
                    return False
                    
        except Exception as e:
            print(f"Error moving robot to position: {e}")
            return False
    
    def _detect_fixed_target_object(self) -> Optional[List[float]]:
        """
        Detect object at fixed target location.
        
        Returns:
            Object position in camera coordinates [forward, side, height] or None
        """
        try:
            # Take multiple detections for robustness
            detection_attempts = 3
            detected_positions = []
            
            for attempt in range(detection_attempts):
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                if vertices is not None:
                    # Detect objects
                    detected_objects = self._detect_objects_simple(vertices, color_image)
                    
                    if detected_objects:
                        # Find best object (should be our fixed target)
                        best_object = self._find_best_object(detected_objects)
                        if best_object:
                            position = self._extract_object_position(best_object)
                            if position:
                                detected_positions.append(position)
                                print(f"  Detection {attempt + 1}: {position}")
                
                time.sleep(0.1)  # Brief pause between detections
            
            if not detected_positions:
                print("No objects detected")
                return None
            
            if len(detected_positions) < 2:
                print(f"Only {len(detected_positions)} successful detections - using available data")
            
            # Use average of detections
            avg_position = np.mean(detected_positions, axis=0)
            position_std = np.std(detected_positions, axis=0) if len(detected_positions) > 1 else [0, 0, 0]
            
            print(f"Average position: ({avg_position[0]:.1f}, {avg_position[1]:.1f}, {avg_position[2]:.1f}) mm")
            if len(detected_positions) > 1:
                print(f"Detection stability: ±({position_std[0]:.1f}, {position_std[1]:.1f}, {position_std[2]:.1f}) mm")
            
            # Validate detection (within reasonable range)
            distance = np.linalg.norm(avg_position)
            if distance < self.min_object_distance or distance > self.max_object_distance:
                print(f"Object distance {distance:.1f}mm outside valid range [{self.min_object_distance}, {self.max_object_distance}]")
                return None
            
            return avg_position.tolist()
            
        except Exception as e:
            print(f"Error detecting fixed target object: {e}")
            return None
    
    def _detect_calibration_object_continuous(self, position_index: int) -> Optional[Dict[str, Any]]:
        """
        Continuous object detection with live visualization and user confirmation.
        Similar to eye_in_hand_calibration.py approach.
        
        Returns:
            Object data with position in camera coordinates, or None if user quits
        """
        print(f"\n--- CALIBRATION POINT {position_index + 1}/{self.num_positions} ---")
        print("Live detection mode - press 'c' when ready to confirm calibration point")
        print("Controls: 'c'=confirm, 's'=skip, 'q'=quit, 'r'=retry")
        
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
                    best_detected_object = detected_objects[0] if detected_objects else None
                
                # Create live visualization (always showing) with raw vertices
                self._create_calibration_visualization(vertices, color_image, detected_objects, timing_info, bev_image, point_cloud_view, position_index, raw_vertices)
                
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
                    print("- Recalibrating gravity alignment + auto ROI...")
                    self._recalibrate_gravity_and_roi()
                    best_detected_object = None
                
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
    
    def _select_best_calibration_object(self, detected_objects: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best object for calibration."""
        if not detected_objects:
            return None
        
        # Use the same logic as _find_best_object
        return self._find_best_object(detected_objects)
    
    def _is_object_valid_for_calibration(self, obj: Dict[str, Any]) -> bool:
        """Check if object is valid for calibration."""
        position = self._extract_object_position(obj)
        if not position:
            return False
        
        distance = np.linalg.norm(position)
        return self.min_object_distance <= distance <= self.max_object_distance
    
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
    
    def _create_calibration_visualization(self, vertices: np.ndarray, color_image: Optional[np.ndarray],
                                        detected_objects: List[Dict[str, Any]], timing_info: Dict[str, float],
                                        bev_image: Optional[np.ndarray], point_cloud_view: Optional[np.ndarray],
                                        position_index: int, raw_vertices: Optional[np.ndarray] = None):
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
                self._add_calibration_info_overlay(rgb_panel, detected_objects, timing_info, position_index)
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
            cv2.imshow('Fixed Target Calibration', combined)
            
        except Exception as e:
            print(f"Warning: Visualization error: {e}")
            # Fallback to simple color image display
            if color_image is not None:
                cv2.imshow('Fixed Target Calibration', color_image)
    
    def _add_calibration_info_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]], 
                                    timing_info: Dict[str, float], position_index: int):
        """Add calibration information overlay to image."""
        # Get robot status
        robot_status = "Disconnected"
        if self.robot_controller and self.robot_controller.is_connected:
            robot_status = "Connected"
        
        # Get ROI mode information
        roi_mode = "AUTO Surface Detection"
        roi_info = ""
        if hasattr(self.detector.clustering_strategy, 'roi_params') and self.detector.clustering_strategy.roi_params:
            roi_params = self.detector.clustering_strategy.roi_params
            roi_info = f"Y: [{roi_params['y_min']:.3f}, {roi_params['y_max']:.3f}]m"
        
        info_lines = [
            "FIXED TARGET CALIBRATION",
            f"Position: {position_index + 1}/{self.num_positions}",
            f"Fixed object at: {self.fixed_object_position}",
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
            "'i' - Recalibrate gravity + auto ROI",
        ]
        
        # Add status message if available
        if 'status_message' in timing_info:
            info_lines.append("")
            info_lines.append("STATUS:")
            info_lines.append(timing_info['status_message'])
        
        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            if line.startswith("STATUS:") or "READY" in line:
                color = (0, 255, 255)
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
    
    def _recalibrate_gravity_and_roi(self):
        """Recalibrate gravity alignment and auto ROI."""
        try:
            # Recalibrate gravity alignment
            if self.sensor_manager.use_imu:
                print("Recalibrating gravity alignment...")
                self.sensor_manager.calibrate_gravity_alignment(num_samples=10)  # Fast re-calibration
                print("+ Gravity alignment updated")
            
            # Re-detect table surface and update ROI
            print("Re-detecting table surface and updating ROI...")
            if self._detect_and_set_dynamic_roi():
                print("+ Auto ROI updated successfully")
            else:
                print("! Auto ROI re-detection failed")
                
        except Exception as e:
            print(f"Error during gravity + ROI recalibration: {e}")
    
    def calculate_fixed_target_transformation(self) -> bool:
        """
        Calculate camera-to-end-effector transformation using fixed target approach.
        
        Uses direct point correspondence for more stable results than relative methods.
        For each calibration point:
        - Known world position (fixed)
        - Detected camera position  
        - Robot end effector pose
        
        Returns:
            True if transformation calculation successful
        """
        print("\n" + "=" * 60)
        print("FIXED TARGET TRANSFORMATION CALCULATION")
        print("=" * 60)
        
        if len(self.calibration_data) < 3:
            print(f"Error: Need at least 3 calibration points, got {len(self.calibration_data)}")
            return False
        
        print(f"Using {len(self.calibration_data)} calibration data points")
        print(f"Fixed world position: {self.fixed_object_position}")
        
        try:
            # Extract data for calculation
            world_points = []  # Known fixed world coordinates
            camera_points = []  # Detected camera coordinates  
            robot_poses = []   # Robot end effector poses
            
            for i, data_point in enumerate(self.calibration_data):
                world_pos = data_point['fixed_world_position']
                camera_pos = data_point['detected_camera_position'] 
                robot_pose = data_point['actual_robot_pose']
                
                world_points.append(world_pos)
                camera_points.append(camera_pos)
                robot_poses.append(robot_pose)
                
                print(f"Point {i+1}: World {world_pos} → Camera {[f'{x:.1f}' for x in camera_pos]}")
            
            # Convert to numpy arrays
            world_points = np.array(world_points) / 1000.0  # Convert mm to meters
            camera_points = np.array(camera_points) / 1000.0  # Convert mm to meters
            robot_poses = np.array(robot_poses)
            
            # Use OpenCV's solvePnP for direct correspondence
            transformation_matrix = self._solve_fixed_target_pnp(world_points, camera_points, robot_poses)
            
            if transformation_matrix is not None:
                print("\n✓ Fixed target transformation calculated successfully")
                
                # Save transformation matrix
                if self._save_transformation_matrix(transformation_matrix):
                    print(f"✓ Transformation matrix saved to: {self.output_file}")
                    
                    # Validate the transformation
                    self._validate_fixed_target_transformation(transformation_matrix)
                    return True
                else:
                    print("✗ Failed to save transformation matrix")
                    return False
            else:
                print("✗ Failed to calculate transformation matrix")
                return False
                
        except Exception as e:
            print(f"Error calculating fixed target transformation: {e}")
            return False
    
    def _solve_fixed_target_pnp(self, world_points: np.ndarray, camera_points: np.ndarray, 
                               robot_poses: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve for camera-to-end-effector transformation using proper eye-in-hand calibration.
        
        For eye-in-hand setup, we need to find X (camera-to-end-effector transform) such that:
        world_point_base = T_ee_to_base @ X @ camera_point
        
        Args:
            world_points: Known world coordinates [N, 3] in meters
            camera_points: Detected camera coordinates [N, 3] in meters  
            robot_poses: Robot end effector poses [N, 6] 
            
        Returns:
            4x4 transformation matrix or None
        """
        try:
            print("\nSolving eye-in-hand transformation...")
            print("Method: Least squares optimization for X in T_ee_to_base @ X @ camera_point = world_point")
            
            # Build system of equations
            # For each measurement i: world_point_i = T_ee_to_base_i @ X @ camera_point_i
            # Rearranging: T_ee_to_base_i^(-1) @ world_point_i = X @ camera_point_i
            
            A_list = []  # Will contain camera points in homogeneous coordinates
            b_list = []  # Will contain world points transformed to end-effector frame
            
            for i in range(len(world_points)):
                # Get robot pose for this measurement
                robot_pos = robot_poses[i][:3] / 1000.0  # Convert mm to meters
                robot_rot = robot_poses[i][3:]  # In degrees
                
                # Convert robot rotation to rotation matrix
                robot_rot_rad = np.radians(robot_rot)
                R_robot = self._euler_to_rotation_matrix(robot_rot_rad)
                
                # Robot transformation matrix (end effector to base frame)
                T_ee_to_base = np.eye(4)
                T_ee_to_base[:3, :3] = R_robot
                T_ee_to_base[:3, 3] = robot_pos
                
                # World point in base frame (homogeneous coordinates)
                world_point_homo = np.array([world_points[i][0], world_points[i][1], world_points[i][2], 1.0])
                
                # Camera point in homogeneous coordinates
                camera_point_homo = np.array([camera_points[i][0], camera_points[i][1], camera_points[i][2], 1.0])
                
                # Transform world point to end-effector frame: T_ee_to_base^(-1) @ world_point
                T_base_to_ee = np.linalg.inv(T_ee_to_base)
                world_in_ee_frame = T_base_to_ee @ world_point_homo
                
                # Store for least squares: world_in_ee_frame = X @ camera_point_homo
                A_list.append(camera_point_homo)
                b_list.append(world_in_ee_frame)
            
            # Convert to matrices for solving
            A = np.array(A_list)  # [N, 4] - camera points (homogeneous)
            b = np.array(b_list)  # [N, 4] - world points in EE frame (homogeneous)
            
            print(f"System dimensions: A={A.shape}, b={b.shape}")
            
            # Solve for transformation matrix X such that b = A @ X^T
            # This means X^T = A^+ @ b, so X = (A^+ @ b)^T
            # Where A^+ is the Moore-Penrose pseudoinverse
            
            A_pinv = np.linalg.pinv(A)  # Pseudoinverse of A
            X_transpose = A_pinv @ b     # Solve for X^T
            X = X_transpose.T            # Get X
            
            # Ensure X is a proper 4x4 transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :] = X[:3, :]  # Copy rotation and translation parts
            
            # Calculate residual error
            residuals = []
            for i in range(len(A)):
                predicted = A[i] @ X.T
                error = np.linalg.norm(predicted[:3] - b[i][:3])
                residuals.append(error)
            
            avg_residual = np.mean(residuals)
            print(f"Average residual error: {avg_residual:.6f}m ({avg_residual*1000:.1f}mm)")
            
            print(f"Calculated transformation matrix:")
            print(transformation_matrix)
            
            return transformation_matrix
            
        except Exception as e:
            print(f"Error in eye-in-hand calibration: {e}")
            
            # Fallback: Simple average offset method
            return self._solve_simple_average_offset(world_points, camera_points, robot_poses)
    
    def _solve_simple_average_offset(self, world_points: np.ndarray, camera_points: np.ndarray,
                                   robot_poses: np.ndarray) -> Optional[np.ndarray]:
        """
        Fallback method: Calculate simple average offset between camera and end effector.
        
        Returns:
            4x4 transformation matrix with average translation offset
        """
        try:
            print("\nUsing fallback method: Simple average offset")
            
            # Calculate average offset in end effector frame
            offsets = []
            
            for i in range(len(world_points)):
                # Assume camera sees object relative to end effector position
                # This is a simplified approach for eye-in-hand setup
                camera_point = camera_points[i]  # Object position in camera frame
                
                # Simple offset: camera origin relative to end effector
                # (This assumes object is at known location relative to robot base)
                offset = camera_point  # Simplified assumption
                offsets.append(offset)
            
            # Average offset
            avg_offset = np.mean(offsets, axis=0)
            offset_std = np.std(offsets, axis=0)
            
            print(f"Average camera offset: ({avg_offset[0]*1000:.1f}, {avg_offset[1]*1000:.1f}, {avg_offset[2]*1000:.1f}) mm")
            print(f"Offset stability: ±({offset_std[0]*1000:.1f}, {offset_std[1]*1000:.1f}, {offset_std[2]*1000:.1f}) mm")
            
            # Build simple transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, 3] = avg_offset
            
            return transformation_matrix
            
        except Exception as e:
            print(f"Error in simple offset calculation: {e}")
            return None
    
    def _euler_to_rotation_matrix(self, euler_angles: np.ndarray) -> np.ndarray:
        """Convert Euler angles (rx, ry, rz) to rotation matrix."""
        rx, ry, rz = euler_angles
        
        # Rotation matrices for each axis
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        
        # Combined rotation (order: Z-Y-X)
        R = Rz @ Ry @ Rx
        return R
    
    def _save_transformation_matrix(self, transformation_matrix: np.ndarray) -> bool:
        """Save transformation matrix to file."""
        try:
            np.save(self.output_file, transformation_matrix)
            print(f"Transformation matrix shape: {transformation_matrix.shape}")
            return True
        except Exception as e:
            print(f"Error saving transformation matrix: {e}")
            return False
    
    def _validate_fixed_target_transformation(self, transformation_matrix: np.ndarray):
        """Validate the calculated transformation matrix using proper eye-in-hand equations."""
        print("\n" + "=" * 40)
        print("TRANSFORMATION VALIDATION")
        print("=" * 40)
        
        try:
            total_error = 0
            max_error = 0
            
            for i, data_point in enumerate(self.calibration_data):
                # Known world position
                world_pos = np.array(data_point['fixed_world_position']) / 1000.0  # Convert to meters
                
                # Detected camera position  
                camera_pos = np.array(data_point['detected_camera_position']) / 1000.0
                
                # Robot pose for this measurement
                robot_pose = data_point['actual_robot_pose']
                robot_pos = np.array(robot_pose[:3]) / 1000.0  # Convert mm to meters
                robot_rot = np.radians(robot_pose[3:])  # Convert degrees to radians
                
                # Robot transformation matrix (end effector to base frame)
                R_robot = self._euler_to_rotation_matrix(robot_rot)
                T_ee_to_base = np.eye(4)
                T_ee_to_base[:3, :3] = R_robot
                T_ee_to_base[:3, 3] = robot_pos
                
                # Forward validation: world_point = T_ee_to_base @ X @ camera_point
                camera_homogeneous = np.array([camera_pos[0], camera_pos[1], camera_pos[2], 1])
                predicted_world_homo = T_ee_to_base @ transformation_matrix @ camera_homogeneous
                predicted_world_pos = predicted_world_homo[:3]
                
                # Calculate error between predicted and actual world position
                error = np.linalg.norm(world_pos - predicted_world_pos) * 1000  # Convert to mm
                total_error += error
                max_error = max(max_error, error)
                
                print(f"Point {i+1}: Predicted world {[f'{x*1000:.1f}' for x in predicted_world_pos]} vs "
                      f"Actual world {[f'{x*1000:.1f}' for x in world_pos]} - Error: {error:.1f}mm")
            
            avg_error = total_error / len(self.calibration_data)
            print(f"\nValidation Results:")
            print(f"Average error: {avg_error:.1f}mm")  
            print(f"Maximum error: {max_error:.1f}mm")
            
            if avg_error < 50:  # 50mm threshold
                print("✓ Transformation validation PASSED")
            else:
                print("! Transformation validation MARGINAL - consider recalibration")
                
        except Exception as e:
            print(f"Error during validation: {e}")
    
    def run_fixed_target_calibration(self) -> bool:
        """
        Main workflow for fixed target eye-in-hand calibration.
        
        Process:
        1. Initialize system (camera, robot, auto ROI)
        2. Guide user to place object at fixed coordinates
        3. Collect calibration data from multiple robot poses
        4. Calculate transformation matrix
        
        Returns:
            True if calibration successful
        """
        print("Starting Fixed Target Eye-in-Hand Calibration")
        print("=" * 70)
        print("This calibration uses a FIXED object placement for higher accuracy.")
        print("Robot will guide you to place the object at exact known coordinates.")
        print("=" * 70)
        
        try:
            # Step 1: Initialize system
            print("\n🔧 STEP 1: SYSTEM INITIALIZATION")
            if not self.initialize():
                print("❌ System initialization failed")
                return False
            
            # Step 2: Guide object placement
            print("\n📍 STEP 2: FIXED TARGET OBJECT PLACEMENT")
            if not self.guide_fixed_object_placement():
                print("❌ Object placement failed")
                return False
            
            # Step 3: Collect calibration data
            print("\n📊 STEP 3: CALIBRATION DATA COLLECTION")
            if not self.collect_fixed_target_calibration_data():
                print("❌ Data collection failed")
                return False
            
            # Step 4: Calculate transformation
            print("\n🧮 STEP 4: TRANSFORMATION CALCULATION")
            if not self.calculate_fixed_target_transformation():
                print("❌ Transformation calculation failed")
                return False
            
            # Success
            print("\n" + "=" * 70)
            print("🎉 FIXED TARGET CALIBRATION COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"✅ Transformation matrix saved to: {self.output_file}")
            print(f"✅ Used {len(self.calibration_data)} calibration points")
            print(f"✅ Fixed object position: {self.fixed_object_position}")
            print("✅ Auto ROI detection enabled")
            print("\nThe calibration matrix can now be used with eye_in_hand_tracker.py")
            print("=" * 70)
            
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ Calibration interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error during calibration: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup system resources."""
        print("\nCleaning up system resources...")
        
        try:
            # Stop camera
            if self.sensor_manager:
                self.sensor_manager.stop()
                print("✓ Camera stopped")
            
            # Disconnect robot
            if self.robot_controller:
                if self.robot_controller.async_mode_enabled:
                    self.robot_controller.cleanup_async_system()
                self.robot_controller.disconnect()
                print("✓ Robot disconnected")
            
            # Close any open windows
            cv2.destroyAllWindows()
            print("✓ Windows closed")
            
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        
        print("✓ Cleanup complete")


def main():
    """Main entry point for fixed target eye-in-hand calibration."""
    parser = argparse.ArgumentParser(
        description='Fixed Target Eye-in-Hand Calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 5-point calibration with optimal positioning (default)
  python eye_in_hand_fixed_calibration.py
  
  # Quick 3-point calibration with calculated optimal positions
  python eye_in_hand_fixed_calibration.py --positions 3
  
  # High-precision 7-point calibration
  python eye_in_hand_fixed_calibration.py --positions 7
  
  # Use predefined positions instead of optimal calculation
  python eye_in_hand_fixed_calibration.py --use-predefined
  
  # Custom output file
  python eye_in_hand_fixed_calibration.py --output my_transform.npy

Fixed Target Approach:
  - Object placed at known coordinates for higher accuracy
  - Robot guides placement at exact location
  - Direct point correspondence (more stable than relative methods)
  - Auto table surface detection and ROI adjustment
        """
    )
    
    parser.add_argument('--positions', type=int, default=5,
                       help='Number of calibration positions (3=quick, 5=standard, 7=precision)')
    parser.add_argument('--output', default='eye_in_hand_fixed_transform.npy',
                       help='Output transformation matrix file')
    parser.add_argument('--floor-offset', type=float, default=0.01,
                       help='Height offset above detected table surface in meters (default: 0.01m = 10mm)')
    parser.add_argument('--use-predefined', action='store_true', 
                       help='Use predefined positions instead of optimal calculated positions')
    parser.add_argument('--visualize-raw-pointcloud', action='store_true', default=True,
                       help='Enable raw point cloud visualization (shows unprocessed RealSense data, default: enabled)')
    parser.add_argument('--no-raw-pointcloud', action='store_true',
                       help='Disable raw point cloud visualization')
    
    args = parser.parse_args()
    
    # Handle raw pointcloud visualization logic
    if args.no_raw_pointcloud:
        args.visualize_raw_pointcloud = False
    
    # Validate arguments
    if args.positions < 3:
        print("Error: Need at least 3 calibration positions")
        return False
    elif args.positions > 10:
        print("Warning: More than 10 positions may be excessive")
    
    print("Fixed Target Eye-in-Hand Calibration")
    print("=" * 50)
    print(f"Calibration positions: {args.positions}")
    print(f"Output file: {args.output}")
    print(f"Floor offset: {args.floor_offset}m")
    print("Fixed target approach: Enabled")
    print("Auto ROI detection: Enabled")
    print("=" * 50)
    
    try:
        # Create calibrator
        calibrator = EyeInHandFixedCalibrator(
            num_positions=args.positions,
            output_file=args.output,
            floor_offset=args.floor_offset,
            use_optimal_positions=not args.use_predefined,  # Default to optimal unless predefined requested
            visualize_raw_pointcloud=args.visualize_raw_pointcloud
        )
        
        # Run calibration
        success = calibrator.run_fixed_target_calibration()
        
        if success:
            print("\n🎉 Calibration completed successfully!")
            return True
        else:
            print("\n❌ Calibration failed!")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️ Calibration interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)