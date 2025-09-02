#!/usr/bin/env python3
"""
Real-time LiDAR Object Position Prediction

Real-time markerless object detection and position prediction using LiDAR point cloud data.
Uses pre-calibrated eye-in-hand transformation for accurate position prediction.

Features:
- Real-time object position prediction relative to robot base (markerless)
- Surface plane detection for objects on tables/floors
- Point cloud clustering for robust object detection
- Live visualization with position display
- Robot coordinate system (X=forward, Y=left, Z=up)

Usage:
    python lidar_realtime_prediction.py [options]

Controls:
    'c' - Move robot to next preset position
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

# Import robot controller
try:
    from module.robot.mycobot_controller import MyCobotController
    ROBOT_AVAILABLE = True
except ImportError:
    print("Robot controller not available - will use manual input")
    ROBOT_AVAILABLE = False


class RealtimeLidarPredictor:
    """Real-time LiDAR-based markerless object position prediction system."""
    
    def __init__(self, config_path="charuco-pose-estimation"):
        """Initialize the real-time LiDAR object detection system.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = Path(config_path)
        
        # LiDAR detection parameters
        self.surface_distance_threshold = 0.01  # 1cm threshold for surface plane detection
        self.object_height_threshold = 0.005    # 5mm minimum object height above surface
        self.clustering_eps = 0.02               # 2cm clustering tolerance
        self.clustering_min_samples = 10         # Minimum points for object cluster
        
        # KNN detection parameters (simplified approach)
        self.knn_radius = 0.05                   # 5cm radius for KNN search
        self.knn_min_neighbors = 5               # Minimum neighbors for averaging
        
        # Visualization options
        # Removed use_pointcloud_view - both depth and point cloud are always visible in 2x2 layout
        
        self.roi_bounds = {                      # Region of interest bounds (meters)
            'x_min': -0.5, 'x_max': 0.5,
            'y_min': -0.5, 'y_max': 0.5, 
            'z_min': 0.1,  'z_max': 0.5     # Close range for tabletop objects (10-50cm)
        }
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = np.zeros(4)
        self.color_intrinsics = None
        
        # Point cloud processing
        self.pc = rs.pointcloud()
        self.points = rs.points()
        
        # Detection state
        self.current_surface_plane = None
        self.detection_confidence = 0.0
        
        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Robot controller
        self.robot = None
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
        
        # Load camera intrinsics first (may be overridden by calibration)
        self.load_camera_intrinsics()
        
        # Load pre-calibrated eye-in-hand transformation (this will override intrinsics with calibration values)
        self.eye_in_hand_transform = self.load_eye_in_hand_transform()
        
        # Position tracking for stability analysis
        self.recent_positions = []
        self.position_history_size = 10
        
        # Robot movement presets
        self.load_robot_poses()
        self.current_pose_index = 0
        
        # Frame counter for saving
        self.frame_count = 0
    
    def load_eye_in_hand_transform(self):
        """Load pre-calibrated eye-in-hand transformation matrix and its intrinsics."""
        print("Loading eye-in-hand calibration for LiDAR object detection")
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
                # Calibration intrinsics loaded successfully (no detectors needed for LiDAR)
                
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
                        # Calibration intrinsics loaded from JSON (no detectors needed for LiDAR)
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
    
    def predict_object_position(self, object_center, robot_transform):
        """
        Predict object position in robot base coordinates using eye-in-hand calibration.
        
        COORDINATE SYSTEM NOTES:
        - LiDAR detection provides object center in camera coordinates (relative to camera center)
        - Calibration matrix was created with CENTER ORIGIN (as per README.txt)
        - This means both systems use center-based coordinates - they should be compatible
        
        Args:
            object_center: [x, y, z] object center in camera coordinates (meters)
            robot_transform: 4x4 robot base-to-gripper transformation matrix
            
        Returns:
            np.array: Object position in robot base coordinates [x, y, z] in mm
        """
        # Check coordinate system compatibility
        print(f"[COORD] Object center (camera): [{object_center[0]:.3f}, {object_center[1]:.3f}, {object_center[2]:.3f}]m")
        print(f"[COORD] Using CENTER ORIGIN calibration (both systems use center-based coordinates)")
        
        # Create simple transformation matrix (position only, no rotation)
        object_transform = np.eye(4)
        object_transform[:3, 3] = object_center
        
        # Object in robot base = Robot_base_to_gripper @ Eye_in_hand @ Camera_to_object
        object_in_base = robot_transform @ self.eye_in_hand_transform @ object_transform
        position_mm = object_in_base[:3, 3] * 1000  # Convert to millimeters
        
        print(f"[COORD] Object in robot base: [{position_mm[0]:.1f}, {position_mm[1]:.1f}, {position_mm[2]:.1f}]mm")
        return position_mm
    
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

    def _project_point_to_pixel(self, point_3d, intrinsics):
        """Project a 3D point to a 2D pixel coordinate using camera intrinsics."""
        if point_3d is None or intrinsics is None:
            return None

        x_3d, y_3d, z_3d = point_3d
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy

        if z_3d <= 0:
            return None

        x_2d = (fx * x_3d / z_3d) + cx
        y_2d = (fy * y_3d / z_3d) + cy

        return int(x_2d), int(y_2d)
    
    def create_2x2_visualization(self, color_image, depth_frame, results, robot_pose, position_in_base):
        """
        Create enhanced 2x2 visualization layout: RGB + Depth + Point Cloud + BEV.
        
        Layout:
        ┌─────────────┬─────────────┐
        │ RGB Camera  │ Depth Image │
        ├─────────────┼─────────────┤
        │ Point Cloud │ BEV View    │
        └─────────────┴─────────────┘
        
        Args:
            color_image: RGB camera image
            depth_frame: RealSense depth frame (aligned to color)
            results: Detection results from detect_object_on_surface
            robot_pose: Current robot pose information
            position_in_base: Predicted object position in robot base coordinates (mm)
            
        Returns:
            np.array: Combined 2x2 visualization image
        """
        try:
            panel_size = 400
            
            # Initialize 2x2 grid panels (reorganized layout)
            top_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)    # RGB
            top_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)   # Depth Image  
            bottom_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8) # Point Cloud (always visible)
            bottom_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8) # BEV View
            
            # Top-left: RGB image with tracking info overlay
            if color_image is not None:
                rgb_panel = color_image.copy()
                # Add tracking info overlay
                self._add_tracking_info_overlay(rgb_panel, results, robot_pose, position_in_base)
                # Add object detection overlay
                self._add_object_detection_overlay(rgb_panel, results)
                
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
            
            # Top-right: Depth image (always visible)
            depth_view = self._create_depth_image_view(depth_frame, results.get('object_center'))
            if depth_view is not None:
                top_right = depth_view
            else:
                cv2.putText(top_right, "DEPTH IMAGE", (50, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                cv2.putText(top_right, "NO DATA", (80, panel_size//2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            
            # Bottom-left: Full point cloud (always visible)
            lidar_view = self._create_lidar_point_cloud_view(results.get('full_point_cloud'), results.get('object_center'))
            if lidar_view is not None:
                bottom_left = lidar_view
            else:
                cv2.putText(bottom_left, "POINT CLOUD", (80, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                cv2.putText(bottom_left, "NO DATA", (120, panel_size//2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            
            # Bottom-right: Object cluster point cloud
            object_cluster_view = self._create_lidar_point_cloud_view(results.get('object_points'), results.get('object_center'))
            if object_cluster_view is not None:
                bottom_right = object_cluster_view
            else:
                cv2.putText(bottom_right, "OBJECT CLUSTER", (80, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                cv2.putText(bottom_right, "NO OBJECT DETECTED", (50, panel_size//2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            
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
        """Add tracking information overlay to RGB image."""
        info_lines = [
            "REAL-TIME LIDAR OBJECT PREDICTION",
            f"Status: {'OBJECT DETECTED' if results['valid_detection'] else 'SEARCHING...'}",
            f"Method: KNN Point Clustering",
            "Layout: RGB + Depth + Point Cloud + BEV (no toggle needed)",
            f"Robot: {'Connected' if robot_pose is not None else 'Disconnected'}",
            "Press 'c'=move robot, 'q'=quit"
        ]
        
        for i, line in enumerate(info_lines):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            if "DETECTED" in line:
                color = (0, 255, 0)
            elif "SEARCHING" in line:
                color = (0, 255, 255)
            cv2.putText(image, line, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add position prediction if available
        if results['valid_detection'] and position_in_base is not None:
            y = 160
            cv2.putText(image, f"Robot Base Position (mm):", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20
            cv2.putText(image, f"X(forward): {position_in_base[0]:7.1f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            y += 18
            cv2.putText(image, f"Y(left):    {position_in_base[1]:7.1f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y += 18
            cv2.putText(image, f"Z(up):      {position_in_base[2]:7.1f}", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    def _add_object_detection_overlay(self, image, results):
        """Add object detection information overlay."""
        if not results['valid_detection'] or results['object_center'] is None:
            return

        # Project the 3D object center to 2D pixel coordinates
        pixel = self._project_point_to_pixel(results['object_center'], self.color_intrinsics)

        if pixel is not None:
            center_x, center_y = pixel
            h, w = image.shape[:2]

            # Ensure the pixel is within the image bounds before drawing
            if 0 <= center_x < w and 0 <= center_y < h:
                cv2.circle(image, (center_x, center_y), 15, (0, 255, 0), 3)
                cv2.circle(image, (center_x, center_y), 30, (0, 255, 0), 2)

                # Draw crosshairs for detected object
                cv2.line(image, (center_x - 40, center_y), (center_x + 40, center_y), (0, 255, 0), 2)
                cv2.line(image, (center_x, center_y - 40), (center_x, center_y + 40), (0, 255, 0), 2)

        # Add detection info text
        obj_pos = results['object_center']
        cv2.putText(image, f"Camera: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]m",
                   (10, image.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    def _create_lidar_point_cloud_view(self, point_cloud, object_center):
        """Create LiDAR point cloud visualization with correct aspect ratio using color intrinsics."""
        if point_cloud is None or len(point_cloud) == 0:
            return None

        try:
            # Filter out invalid points
            valid_mask = ~np.any(np.isnan(point_cloud) | np.isinf(point_cloud), axis=1)
            valid_vertices = point_cloud[valid_mask]

            if len(valid_vertices) < 10:
                return None

            # --- Use Color Intrinsics for Projection (since depth is aligned to color) ---
            if not hasattr(self, 'color_intrinsics') or self.color_intrinsics is None:
                print("Warning: Color intrinsics not available for point cloud visualization.")
                return None

            intrinsics = self.color_intrinsics
            original_width = intrinsics.width
            original_height = intrinsics.height

            # Create a projection canvas with the original color sensor resolution
            proj_canvas = np.zeros((original_height, original_width, 3), dtype=np.uint8)

            # Use unscaled color intrinsics for projection
            fx = intrinsics.fx
            fy = intrinsics.fy
            cx = intrinsics.ppx
            cy = intrinsics.ppy

            # Project 3D points to 2D
            x_3d = valid_vertices[:, 0]
            y_3d = valid_vertices[:, 1]
            z_3d = np.maximum(valid_vertices[:, 2], 0.001)

            x_2d = (fx * x_3d / z_3d) + cx
            y_2d = (fy * y_3d / z_3d) + cy

            x_pixels = np.clip(x_2d.astype(int), 0, original_width - 1)
            y_pixels = np.clip(y_2d.astype(int), 0, original_height - 1)

            # Color points by depth
            z_min, z_max = np.min(z_3d), np.max(z_3d)
            if z_max > z_min:
                z_normalized = (z_3d - z_min) / (z_max - z_min)
                depth_colors = cv2.applyColorMap((z_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
                proj_canvas[y_pixels, x_pixels] = depth_colors.reshape(-1, 3)[range(len(x_pixels))]

            # Draw object center
            if object_center is not None:
                pixel = self._project_point_to_pixel(object_center, intrinsics)
                if pixel is not None:
                    obj_x_pixel, obj_y_pixel = pixel
                    cv2.circle(proj_canvas, (obj_x_pixel, obj_y_pixel), 8, (0, 255, 0), 3)
                    cv2.circle(proj_canvas, (obj_x_pixel, obj_y_pixel), 15, (0, 255, 0), 2)

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

            return canvas

        except Exception as e:
            print(f"Error creating LiDAR point cloud view: {e}")
            return None
            
        except Exception as e:
            print(f"Error creating LiDAR point cloud view: {e}")
            return None
    
    def _create_depth_image_view(self, depth_frame, object_center=None):
        """Create colorized depth image visualization (aligned with RGB camera view)."""
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
            
            # Highlight detected object center if available
            if object_center is not None and self.color_intrinsics is not None:
                # Project the 3D object center to 2D pixel coordinates in the *original* frame
                pixel = self._project_point_to_pixel(object_center, self.color_intrinsics)

                if pixel is not None:
                    # Map the original coordinates to the final 400x400 panel, accounting for resizing and letterboxing
                    original_pixel_x, original_pixel_y = pixel
                    original_h, original_w = self.color_intrinsics.height, self.color_intrinsics.width

                    # The scaling factor is based on how the original image was fitted into the new dimensions
                    scale = new_width / original_w

                    # Calculate the position on the resized image
                    scaled_x = original_pixel_x * scale
                    scaled_y = original_pixel_y * scale

                    # Add the offset to get the final position on the 400x400 canvas
                    final_x = int(scaled_x + x_offset)
                    final_y = int(scaled_y + y_offset)

                    # Draw object center marker at the corrected position
                    if 0 <= final_x < canvas_size and 0 <= final_y < canvas_size:
                        cv2.circle(depth_colorized, (final_x, final_y), 8, (0, 255, 0), 3)
                        cv2.circle(depth_colorized, (final_x, final_y), 15, (0, 255, 0), 2)
                        cv2.circle(depth_colorized, (final_x, final_y), 25, (255, 255, 255), 1)
            
            # Add text overlay
            cv2.putText(depth_colorized, "DEPTH IMAGE", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(depth_colorized, "Fixed Colors (Blue=Near)", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Add depth range info
            if np.any(valid_mask):
                cv2.putText(depth_colorized, f"Range: {min_depth:.2f}-{max_depth:.2f}m", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
                
                valid_count = np.sum(valid_mask)
                total_count = depth_m.size
                cv2.putText(depth_colorized, f"Valid: {valid_count}/{total_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            else:
                cv2.putText(depth_colorized, "No valid depth data", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Color legend
            cv2.putText(depth_colorized, "Color: Blue=Near, Red=Far", (10, canvas_size - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            return depth_colorized
            
        except Exception as e:
            print(f"Error creating depth image view: {e}")
            return None
        
    def setup_alignment(self):
        """Setup depth-to-color alignment for point cloud generation."""
        # Create alignment object to align depth to color
        self.align = rs.align(rs.stream.color)
        print("Depth-to-color alignment initialized")
        
    def load_camera_intrinsics(self):
        """Load camera intrinsics from file for comparison testing."""
        intrinsic_file = self.config_path / 'intrinsic.txt'
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
                
                # Camera matrix loaded successfully (no detectors needed for LiDAR)
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
    
    def setup_camera(self):
        """Setup RealSense camera with L515 Short Range Preset optimized settings."""
        try:
            # L515 LiDAR Short Range Preset - using MATCHING resolutions for proper alignment
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
                    self.config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)
                    self.config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)
                    profile = self.pipeline.start(self.config)
                    
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
                    self.pipeline.stop()
                    self.config.disable_all_streams()
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
                # Use the pre-calibrated intrinsics (they will override RealSense intrinsics)
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
                    self.dist_coeffs = np.array(intrinsics.coeffs[:5])
            
            print(f"Final camera matrix:")
            print(f"  fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
            print(f"  cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}")
            
            # Get depth intrinsics for point cloud generation
            depth_profile = profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
            
            # Initialize depth-to-color alignment
            self.align = rs.align(rs.stream.color)
            
            # Note: Point cloud mapping will be set up during runtime with actual frames
            # since we're using different resolutions for color and depth streams
            
            print(f"Point cloud processing initialized with depth intrinsics:")
            print(f"  fx={self.depth_intrinsics.fx:.2f}, fy={self.depth_intrinsics.fy:.2f}")
            print(f"  ppx={self.depth_intrinsics.ppx:.2f}, ppy={self.depth_intrinsics.ppy:.2f}")
            
            # Get device info
            device = profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            serial_number = device.get_info(rs.camera_info.serial_number)
            firmware_version = device.get_info(rs.camera_info.firmware_version)
            
            print("🎯 RealSense L515 Camera initialized successfully with Short Range Preset")
            print(f"Device: {device_name} (S/N: {serial_number}, FW: {firmware_version})")
            print(f"Resolution: {intrinsics.width}x{intrinsics.height} (color/depth matched for alignment)")
            print(f"Optimized range: 0.25m - 1.5m (perfect for tabletop objects)")
            print(f"Camera intrinsics from RealSense SDK:")
            print(f"  fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
            print(f"  cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
            print(f"Camera matrix:\n{self.camera_matrix}")
            print(f"Distortion coefficients: {self.dist_coeffs}")
            
            # Save intrinsics to file for later use
            intrinsics_file = self.config_path / 'intrinsic_from_realsense.txt'
            np.savetxt(intrinsics_file, self.camera_matrix)
            
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
            
            import json
            with open(self.config_path / 'camera_calibration_realsense.json', 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            print(f"Intrinsics saved to: {intrinsics_file}")
            print(f"Detailed calibration saved to: camera_calibration_realsense.json")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize RealSense camera: {e}")
            return False
    
    
    
    def get_robot_pose(self):
        """Get current robot end-effector pose."""
        if self.robot is not None:
            # Check robot connection status
            if hasattr(self.robot, 'is_connected') and not self.robot.is_connected:
                print("Tip: Make sure myCobot is powered on and connected via USB")
            else:
                try:
                    # Get robot pose [x, y, z, rx, ry, rz]
                    pose = self.robot.get_current_position()
                    if pose and len(pose) >= 6:
                        return pose[:6]  # [x, y, z, rx, ry, rz]
                    else:
                        print("Tip: Robot may need to be powered on or moved manually first")
                except Exception as e:
                    print(f"Failed to get robot pose: {e}")
        else:
            pass  # No robot controller available
        
        # Fallback: manual input
        print("\nRobot pose not available. Enter robot end-effector pose manually:")
        try:
            x = float(input("X (mm): "))
            y = float(input("Y (mm): "))
            z = float(input("Z (mm): "))
            rx = float(input("RX (degrees): "))
            ry = float(input("RY (degrees): "))
            rz = float(input("RZ (degrees): "))
            return [x, y, z, rx, ry, rz]
        except (ValueError, KeyboardInterrupt):
            print("Input cancelled, skipping pose collection")
            return None
    
    def pose_to_transform_matrix(self, pose):
        """Convert [x, y, z, rx, ry, rz] pose to 4x4 transformation matrix."""
        x, y, z, rx, ry, rz = pose
        
        # Convert to meters and radians
        x_m, y_m, z_m = x/1000.0, y/1000.0, z/1000.0
        rx_rad, ry_rad, rz_rad = np.radians([rx, ry, rz])
        
        # Create rotation matrix from Euler angles (XYZ intrinsic order - test original)
        from scipy.spatial.transform import Rotation
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
        # Based on analysis: CharUco X/Y axes need 90° Z-rotation to match robot coordinates
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
    
    def collect_calibration_point(self, results):
        """Collect robot and CharUco poses for calibration."""
        if not results['valid_pose']:
            print("No CharUco pose detected - cannot collect calibration point")
            return False
        
        print(f"\nCollecting calibration point {len(self.calibration_pairs) + 1}...")
        
        # Get robot pose
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return False
        
        # Get CharUco pose
        charuco_rvec = results['rvec']
        charuco_tvec = results['tvec']
        
        # Convert to transformation matrices
        robot_transform = self.pose_to_transform_matrix(robot_pose)
        charuco_transform = self.charuco_pose_to_transform_matrix(charuco_rvec, charuco_tvec)
        
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
    
    def save_calibration_results(self, eye_in_hand_transform):
        """Save calibration results to file."""
        timestamp = int(time.time())
        output_dir = self.config_path / f"calibration_result_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        # Save transformation matrix
        np.savetxt(output_dir / 'eye_in_hand_transform.txt', eye_in_hand_transform)
        np.save(output_dir / 'eye_in_hand_transform.npy', eye_in_hand_transform)
        
        # Save calibration data
        calibration_data = {
            'num_points': len(self.calibration_pairs),
            'points': []
        }
        
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
            
            # Calculate proper rotation error with coordinate correction applied
            # With the 90° Z-rotation fix, the pattern should now have identity rotation in robot coordinates
            # This validates that the coordinate system correction is working properly
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
        print("  Excellent: <5mm translation, <2° rotation")
        print("  Good:      <10mm translation, <5° rotation")
        print("  Fair:      <20mm translation, <10° rotation")
        print("  Poor:      >20mm translation or >10° rotation")
        
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
        # This indicates how well-conditioned the calibration problem is
        robot_transforms = [pair['robot_transform'] for pair in self.calibration_pairs]
        
        # Check pose diversity
        positions = np.array([T[:3, 3] for T in robot_transforms])
        position_span = np.max(positions, axis=0) - np.min(positions, axis=0)
        position_diversity = np.linalg.norm(position_span)
        
        # Check rotation diversity  
        from scipy.spatial.transform import Rotation
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
    
    def detect_object_on_surface(self, color_frame, depth_frame):
        """Detect object using a center-view KNN approach."""
        try:
            # Generate point cloud from depth data (with color frame for mapping)
            point_cloud = self.generate_point_cloud(depth_frame, color_frame)

            if point_cloud is None or len(point_cloud) < 100:
                return self.create_empty_detection_result()

            # Filter point cloud to ROI
            filtered_cloud = self.filter_roi(point_cloud)

            if len(filtered_cloud) < 50:
                return self.create_empty_detection_result()

            # Detect object by finding the average of points in the center of the view
            object_center = self.detect_object_knn_simple(
                filtered_cloud, self.color_intrinsics, self.knn_radius, self.knn_min_neighbors
            )

            if object_center is None:
                return self.create_empty_detection_result()

            # Create results with object position
            results = {
                'object_center': object_center,
                'object_points': None,
                'surface_plane': None,
                'point_cloud': filtered_cloud,
                'valid_detection': True,
                'num_object_points': 1
            }

            self.detection_confidence = 1.0
            return results

        except Exception as e:
            print(f"LiDAR detection error: {e}")
            return self.create_empty_detection_result()
    
    def detect_object_knn_simple(self, point_cloud, intrinsics, knn_radius=0.05, min_neighbors=5, roi_size=50):
        """
        Detects an object by averaging points within the center of the camera's view.

        1. Projects all 3D points to 2D pixel coordinates.
        2. Selects points that fall within a central Region of Interest (ROI).
        3. Averages the 3D coordinates of these central points to find the object center.

        Args:
            point_cloud: The full 3D point cloud.
            intrinsics: The camera intrinsics for projection.
            knn_radius: Not used in this method, but kept for compatibility.
            min_neighbors: The minimum number of points required in the center ROI.
            roi_size: The size of the square ROI in the center of the image (pixels).

        Returns:
            np.array: The averaged 3D center of the object in camera coordinates, or None.
        """
        if point_cloud is None or len(point_cloud) < min_neighbors or intrinsics is None:
            return None

        try:
            # Project all points to 2D
            z_3d = np.maximum(point_cloud[:, 2], 0.001)
            x_2d = (intrinsics.fx * point_cloud[:, 0] / z_3d) + intrinsics.ppx
            y_2d = (intrinsics.fy * point_cloud[:, 1] / z_3d) + intrinsics.ppy

            # Define the center ROI
            center_x, center_y = intrinsics.width / 2, intrinsics.height / 2
            roi_x_min, roi_x_max = center_x - roi_size / 2, center_x + roi_size / 2
            roi_y_min, roi_y_max = center_y - roi_size / 2, center_y + roi_size / 2

            # Find points within the center ROI
            center_mask = (
                (x_2d >= roi_x_min) & (x_2d <= roi_x_max) &
                (y_2d >= roi_y_min) & (y_2d <= roi_y_max)
            )
            center_points = point_cloud[center_mask]

            print(f"[CENTER_VIEW] Found {len(center_points)} points in the center {roi_size}x{roi_size} ROI.")

            # Check if we have enough points
            if len(center_points) < min_neighbors:
                print(f"[CENTER_VIEW] Not enough points in ROI ({len(center_points)} < {min_neighbors}).")
                return None

            # Average the 3D positions of the points in the center of the view
            object_center = np.mean(center_points, axis=0)

            print(f"[CENTER_VIEW] Object center (averaged): [{object_center[0]:.3f}, {object_center[1]:.3f}, {object_center[2]:.3f}]m")

            return object_center

        except Exception as e:
            print(f"[CENTER_VIEW] Error in center view detection: {e}")
            return None

    def remove_statistical_outliers(self, point_cloud, num_neighbors=20, std_ratio=1.0):
        """
        Removes statistical outliers from a point cloud using a numpy-based approach.

        Args:
            point_cloud: The input point cloud (numpy array).
            num_neighbors: The number of neighbors to use for mean distance estimation.
            std_ratio: The standard deviation ratio for outlier threshold.

        Returns:
            The filtered point cloud (numpy array).
        """
        if point_cloud is None or len(point_cloud) == 0:
            return None

        from sklearn.neighbors import NearestNeighbors
        
        k = num_neighbors
        if len(point_cloud) < k + 1:
            return point_cloud # Not enough points

        # Find the k nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(point_cloud)
        distances, indices = nbrs.kneighbors(point_cloud)

        # Calculate the mean distance for each point to its neighbors
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # Calculate the global mean and standard deviation of these distances
        global_mean_dist = np.mean(mean_distances)
        global_std_dist = np.std(mean_distances)

        # Define the threshold for identifying outliers
        threshold = global_mean_dist + std_ratio * global_std_dist

        # Keep only the points that are not outliers
        inlier_mask = mean_distances < threshold
        return point_cloud[inlier_mask]

    def detect_tabletop_object(self, color_frame, depth_frame):
        """
        Detects tabletop objects using a RANSAC plane fitting and DBSCAN clustering approach.
        """
        try:
            # Generate point cloud
            point_cloud = self.generate_point_cloud(depth_frame, color_frame)
            if point_cloud is None or len(point_cloud) < 100:
                return self.create_empty_detection_result()

            # 1. Filter to a reasonable ROI
            filtered_cloud = self.filter_roi(point_cloud)
            if len(filtered_cloud) < 50:
                return self.create_empty_detection_result(full_point_cloud=point_cloud)

            # 2. Remove statistical outliers
            denoised_cloud = self.remove_statistical_outliers(filtered_cloud)
            if denoised_cloud is None or len(denoised_cloud) < 50:
                return self.create_empty_detection_result(full_point_cloud=point_cloud)

            # 3. Find the table plane using RANSAC on the denoised cloud
            surface_plane = self.detect_surface_plane(denoised_cloud)
            if surface_plane is None:
                print("[TABLETOP] Could not detect a dominant plane (table).")
                return self.create_empty_detection_result(full_point_cloud=point_cloud)

            # 4. Filter points that are on top of the table
            object_points = self.filter_points_above_surface(
                denoised_cloud, surface_plane, min_height=0.02, max_height=0.1
            )
            if object_points is None or len(object_points) < self.clustering_min_samples:
                print("[TABLETOP] No points found above the table in the specified height range.")
                return self.create_empty_detection_result(full_point_cloud=point_cloud, surface_plane=surface_plane)

            # 5. Find the largest cluster of object points
            object_center = self.find_largest_cluster_center(object_points)
            if object_center is None:
                print("[TABLETOP] Could not find a significant object cluster.")
                return self.create_empty_detection_result(full_point_cloud=point_cloud, surface_plane=surface_plane)

            # 6. Success! Return the results
            results = {
                'object_center': object_center,
                'object_points': object_points,
                'surface_plane': surface_plane,
                'point_cloud': denoised_cloud, # Pass the denoised cloud for visualization
                'full_point_cloud': point_cloud,
                'valid_detection': True,
                'num_object_points': len(object_points)
            }
            self.detection_confidence = 1.0
            return results

        except Exception as e:
            print(f"Tabletop detection error: {e}")
            return self.create_empty_detection_result()
    
    def create_empty_detection_result(self, point_cloud=None, surface_plane=None, full_point_cloud=None):
        """Create empty detection result structure."""
        return {
            'object_center': None,
            'object_points': None,
            'surface_plane': surface_plane,
            'point_cloud': point_cloud,
            'full_point_cloud': full_point_cloud,
            'valid_detection': False,
            'num_object_points': 0
        }
    
    def generate_point_cloud(self, depth_frame, color_frame=None):
        """Generate 3D point cloud from depth frame."""
        try:
            # Map point cloud to color frame if available (for texture mapping)
            if color_frame is not None:
                self.pc.map_to(color_frame)
            
            # Calculate point cloud using RealSense SDK
            points = self.pc.calculate(depth_frame)
            vertices = np.asanyarray(points.get_vertices())
            
            # Convert to numpy array format
            point_cloud = np.array([[v[0], v[1], v[2]] for v in vertices])
            
            # Filter out invalid points (z <= 0)
            valid_mask = point_cloud[:, 2] > 0
            point_cloud = point_cloud[valid_mask]
            
            return point_cloud
            
        except Exception as e:
            print(f"Point cloud generation error: {e}")
            return None
    
    def filter_roi(self, point_cloud):
        """Filter point cloud to region of interest."""
        bounds = self.roi_bounds
        
        mask = (
            (point_cloud[:, 0] >= bounds['x_min']) & (point_cloud[:, 0] <= bounds['x_max']) &
            (point_cloud[:, 1] >= bounds['y_min']) & (point_cloud[:, 1] <= bounds['y_max']) &
            (point_cloud[:, 2] >= bounds['z_min']) & (point_cloud[:, 2] <= bounds['z_max'])
        )
        
        return point_cloud[mask]
    
    def detect_surface_plane(self, point_cloud):
        """Detect surface plane using RANSAC."""
        if len(point_cloud) < 50:
            return None
            
        try:
            # Fit plane: z = ax + by + c
            X = point_cloud[:, :2]  # x, y coordinates
            y = point_cloud[:, 2]   # z coordinates
            
            ransac = RANSACRegressor(
                residual_threshold=self.surface_distance_threshold,
                max_trials=1000,
                random_state=42
            )
            ransac.fit(X, y)
            
            # Get plane parameters
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_
            
            return {
                'a': a, 'b': b, 'c': c,
                'inliers': ransac.inlier_mask_,
                'num_inliers': np.sum(ransac.inlier_mask_)
            }
            
        except Exception as e:
            print(f"Surface plane detection error: {e}")
            return None
    
    def filter_points_above_surface(self, point_cloud, surface_plane, min_height=0.01, max_height=0.1):
        """
        Filter points above the detected surface by rotating the cloud to align the surface with the XY plane.
        """
        a, b, c = surface_plane['a'], surface_plane['b'], surface_plane['c']
        
        # 1. The plane equation from RANSAC is z = ax + by + c, which we can write as ax + by - z + c = 0.
        # The normal vector to this plane is (a, b, -1).
        normal_vector = np.array([a, b, -1.0])
        normal_vector /= np.linalg.norm(normal_vector)

        # The vector we want to align the plane's normal to is the Z-axis.
        # Let's align it with (0,0,-1) to keep the orientation consistent. This vector points away from the camera
        # into the scene, which is a common convention for camera coordinates.
        target_vector = np.array([0.0, 0.0, -1.0])

        # 2. Calculate the rotation required to align the plane's normal with the target Z-axis.
        rot, _ = Rotation.align_vectors(target_vector[np.newaxis, :], normal_vector[np.newaxis, :])

        # 3. Apply this rotation to the entire point cloud.
        rotated_cloud = rot.apply(point_cloud)

        # 4. In this new rotated space, the table surface is flat (parallel to the XY plane).
        # We find its Z-coordinate by rotating the original inlier points from RANSAC and taking their median Z value.
        plane_points = point_cloud[surface_plane['inliers']]
        rotated_plane_points = rot.apply(plane_points)
        plane_z = np.median(rotated_plane_points[:, 2])

        # 5. Filter the points based on their height (Z-value) in the rotated space.
        # Since the normal vector was aligned to point in the negative Z direction, points "above" the surface
        # (i.e., further from the camera along the normal) will have a more negative Z value.
        mask = (rotated_cloud[:, 2] < plane_z - min_height) & (rotated_cloud[:, 2] > plane_z - max_height)
        object_points_rotated = rotated_cloud[mask]

        if len(object_points_rotated) == 0:
            return np.array([])

        # 6. Rotate the filtered points back to the original coordinate system using the inverse rotation.
        inv_rot = rot.inv()
        object_points = inv_rot.apply(object_points_rotated)
        
        return object_points
    
    def find_largest_cluster_center(self, object_points):
        """Find center of largest object cluster using DBSCAN."""
        if len(object_points) < self.clustering_min_samples:
            return None
        
        try:
            # Cluster object points
            clustering = DBSCAN(
                eps=self.clustering_eps,
                min_samples=self.clustering_min_samples
            )
            labels = clustering.fit_predict(object_points)
            
            # Find largest cluster (ignore noise points with label -1)
            unique_labels = np.unique(labels[labels >= 0])
            
            if len(unique_labels) == 0:
                # No clusters found, return centroid of all points
                return np.mean(object_points, axis=0)
            
            # Find largest cluster
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            largest_cluster_label = unique_labels[np.argmax(cluster_sizes)]
            
            # Get center of largest cluster
            cluster_points = object_points[labels == largest_cluster_label]
            return np.mean(cluster_points, axis=0)
            
        except Exception as e:
            print(f"Clustering error: {e}")
            # Fallback: return centroid of all object points
            return np.mean(object_points, axis=0)
    
    def draw_lidar_results(self, image, results):
        """Draw LiDAR detection results on the image."""
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        # Draw object center if detected
        if results['valid_detection'] and results['object_center'] is not None:
            # Project object center to image coordinates for visualization
            object_center_3d = results['object_center']
            
            # Create a simple visualization marker at image center
            center_x, center_y = w // 2, h // 2
            cv2.circle(vis_image, (center_x, center_y), 10, (0, 255, 0), 3)
            cv2.circle(vis_image, (center_x, center_y), 20, (0, 255, 0), 2)
            
            # Draw crosshairs
            cv2.line(vis_image, (center_x - 30, center_y), (center_x + 30, center_y), (0, 255, 0), 2)
            cv2.line(vis_image, (center_x, center_y - 30), (center_x, center_y + 30), (0, 255, 0), 2)
            
            # Display object center coordinates
            cv2.putText(vis_image, f"Object: [{object_center_3d[0]:.3f}, {object_center_3d[1]:.3f}, {object_center_3d[2]:.3f}]m",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display detection confidence
            cv2.putText(vis_image, f"Points: {results['num_object_points']} | Confidence: {self.detection_confidence:.2f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add detection status
        if results['valid_detection']:
            status_text = f"Object detected (LiDAR)"
            color = (0, 255, 0)
        elif results['surface_plane'] is not None:
            status_text = f"Surface detected, searching for objects..."
            color = (0, 255, 255)
        else:
            status_text = "Searching for surface and objects..."
            color = (0, 0, 255)
            
        cv2.putText(vis_image, status_text, (10, h - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add surface plane info
        if results['surface_plane'] is not None:
            plane = results['surface_plane']
            cv2.putText(vis_image, f"Surface: z = {plane['a']:.3f}x + {plane['b']:.3f}y + {plane['c']:.3f}",
                       (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add robot movement status
        robot_status = self.get_robot_movement_status()
        cv2.putText(vis_image, robot_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add detection parameters info
        cv2.putText(vis_image, f"LiDAR Detection: eps={self.clustering_eps:.3f}m, height_thresh={self.object_height_threshold:.3f}m",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Controls info
        cv2.putText(vis_image, "Controls: 'c'=Move robot, 'q'=Quit", 
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_image, "LiDAR markerless object detection using pre-calibrated eye-in-hand transformation", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_image
    
    def save_results(self, image, results):
        """Save current frame and results."""
        timestamp = int(time.time())
        output_dir = self.config_path / f"realtime_output_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        # Save original image
        cv2.imwrite(str(output_dir / "image.png"), image)
        
        # Save visualization
        vis_image = self.draw_results(image, results)
        cv2.imwrite(str(output_dir / "detection.png"), vis_image)
        
        # Save pose data if available
        if results['valid_pose']:
            # Save transformation matrices
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
            
            # Save corner points (using corrected coordinate system)
            if results['charuco_corners'] is not None:
                obj_points, img_points = self.match_image_points_corrected(results['charuco_corners'], results['charuco_ids'])
                np.savetxt(output_dir / 'objPoints.txt', obj_points.reshape(-1, 3))
                np.savetxt(output_dir / 'imgPoints.txt', img_points.reshape(-1, 2))
                
                # Also save original OpenCV coordinates for comparison
                obj_points_orig, img_points_orig = self.board.matchImagePoints(results['charuco_corners'], results['charuco_ids'])
                np.savetxt(output_dir / 'objPoints_opencv_original.txt', obj_points_orig.reshape(-1, 3))
                np.savetxt(output_dir / 'imgPoints_opencv_original.txt', img_points_orig.reshape(-1, 2))
            
            print(f"Results saved to {output_dir}")
        else:
            print(f"Image saved to {output_dir} (no pose detected)")
    
    def run(self):
        """Run the real-time position prediction."""
        if not self.setup_camera():
            print("Failed to setup camera. Exiting.")
            return
        
        print("\nStarting real-time LiDAR object position prediction...")
        print("Controls:")
        print("  'c' - Move robot to next preset position")
        print("  'q' - Quit application")
        print("  ESC - Exit")
        print("\nReal-time markerless object detection active - Object position will be shown in robot coordinates.")
        print("Using LiDAR point cloud data for surface detection and object clustering.")
        print("Robot coordinates: X=forward(+), Y=left(+), Z=up(+), units in mm")
        print()
        
        try:
            while True:
                # Get frames from camera (both color and depth)
                frames = self.pipeline.wait_for_frames()
                
                # Align depth to color
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy array
                image = np.asanyarray(color_frame.get_data())
                
                # Detect object using the new tabletop detection method
                results = self.detect_tabletop_object(color_frame, depth_frame)
                
                # Get current robot position for prediction
                robot_pose = self.get_robot_pose()
                
                # Predict object position if detected
                position_in_base = None
                if results['valid_detection'] and robot_pose is not None:
                    # Get object center from LiDAR detection
                    object_center = results['object_center']
                    # Convert robot pose to transformation matrix
                    robot_transform = self.pose_to_transform_matrix(robot_pose)
                    
                    # Predict position in robot base coordinates (no rvec needed!)
                    position_in_base = self.predict_object_position(object_center, robot_transform)
                    self.update_position_history(position_in_base)
                
                # Create 2x2 visualization with RGB, Depth/LiDAR, BEV, and Side View
                vis_image = self.create_2x2_visualization(image, depth_frame, results, robot_pose, position_in_base)
                
                # Display
                cv2.imshow('LiDAR Real-time Object Position Prediction', vis_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # 'q' - quit
                    break
                elif key == 27:  # ESC - exit
                    print("Exiting...")
                    self.pipeline.stop()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('c'):  # 'c' - move robot to next pose
                    print("\nMoving robot to next position...")
                    self.move_robot_to_next_pose()
                # Removed 'v' key toggle - both depth and point cloud views are always visible in 2x2 layout
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.pipeline.stop()
            cv2.destroyAllWindows()
            
            print("Real-time LiDAR object detection stopped")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time LiDAR object detection and position prediction")
    parser.add_argument('--config-path', type=str, default='charuco-pose-estimation',
                       help='Path to configuration directory')
    parser.add_argument('--surface-threshold', type=float, default=0.01,
                       help='Surface plane detection threshold in meters (default: 0.01)')
    parser.add_argument('--object-height', type=float, default=0.005,
                       help='Minimum object height above surface in meters (default: 0.005)')
    parser.add_argument('--clustering-eps', type=float, default=0.02,
                       help='DBSCAN clustering tolerance in meters (default: 0.02)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum points for object cluster (default: 10)')
    parser.add_argument('--knn-radius', type=float, default=0.05,
                       help='KNN search radius in meters (default: 0.05 = 5cm)')
    parser.add_argument('--knn-min-neighbors', type=int, default=5,
                       help='Minimum KNN neighbors required (default: 5)')
    parser.add_argument('--use-pointcloud', action='store_true',
                       help='Use point cloud view instead of depth image (depth image is aligned with RGB, recommended)')
    
    args = parser.parse_args()
    
    # Create LiDAR prediction system
    lidar_detector = RealtimeLidarPredictor(args.config_path)
    
    # Update parameters from command line
    lidar_detector.surface_distance_threshold = args.surface_threshold
    lidar_detector.object_height_threshold = args.object_height
    lidar_detector.clustering_eps = args.clustering_eps
    lidar_detector.clustering_min_samples = args.min_samples
    
    # Update KNN parameters
    lidar_detector.knn_radius = args.knn_radius
    lidar_detector.knn_min_neighbors = args.knn_min_neighbors
    
    # Update visualization options
    # No longer need to set use_pointcloud_view - both views always visible
    
    print(f"LiDAR Detection Parameters:")
    print(f"  KNN radius: {args.knn_radius}m")
    print(f"  KNN min neighbors: {args.knn_min_neighbors}")
    print(f"  ROI bounds: X±0.5m, Y±0.5m, Z=0.1-0.5m (tabletop objects)")
    print(f"  Detection method: Simplified KNN (nearest point + radius averaging)")
    print(f"  L515 Short Range: 0.25m-1.5m optimal, ROI focuses on 0.1-0.5m")
    print(f"  Visualization: 2x2 layout with RGB + Depth + Point Cloud + BEV")
    
    # Run the system
    lidar_detector.run()


if __name__ == "__main__":
    main()
