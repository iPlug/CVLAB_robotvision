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

# Import utilities
from utils import RobotUtils, CameraUtils, CoordinateUtils, DetectionUtils


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
        
        self.roi_bounds = {                      # Region of interest bounds (meters)
            'x_min': -0.5, 'x_max': 0.5,
            'y_min': -0.5, 'y_max': 0.5, 
            'z_min': 0.1,  'z_max': 0.5     # Close range for tabletop objects (10-50cm)
        }
        
        # Initialize utility classes
        self.robot_utils = RobotUtils()
        self.camera_utils = CameraUtils(config_path)
        self.coordinate_utils = CoordinateUtils()
        self.detection_utils = DetectionUtils(config_path)
        
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
    
    def create_2x2_visualization(self, color_image, depth_frame, results, robot_pose, position_in_base, object_positions=None):
        """
        Create enhanced 2x2 visualization layout: RGB + Depth + Point Cloud + Object Cluster.
        
        Layout:
        ┌─────────────┬─────────────┐
        │ RGB Camera  │ Depth Image │
        ├─────────────┼─────────────┤
        │ Point Cloud │ Object View │
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
                # Add tracking info overlay
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
                top_right = cv2.resize(depth_view, (panel_size, panel_size))
            else:
                cv2.putText(top_right, "DEPTH IMAGE", (80, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                cv2.putText(top_right, "NO DATA", (120, panel_size//2+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
            
            # Bottom-left: Full point cloud view with cluster coloring
            lidar_view = self._create_lidar_point_cloud_view(results.get('full_point_cloud'), object_clusters)
            if lidar_view is not None:
                bottom_left = lidar_view
            else:
                cv2.putText(bottom_left, "POINT CLOUD", (80, panel_size//2-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
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
        """Add tracking information overlay to the image."""
        h, w = image.shape[:2]
        
        # Create overlay for text information
        overlay = image.copy()
        alpha = 0.7
        
        # Main info panel
        panel_width = min(400, w - 20)
        panel_height = min(200, h - 40)
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (255, 255, 255), 2)
        
        # Blend overlay
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Title
        cv2.putText(image, "LiDAR Object Position Prediction", (20, 35),
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
        
        # Robot status
        y += 30
        if robot_pose is not None:
            cv2.putText(image, f"Robot Connected", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        else:
            cv2.putText(image, f"Robot: Not connected", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Controls info
        cv2.putText(image, "Controls: 'c'=Move robot, 'q'=Quit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
    
    def _create_lidar_point_cloud_view(self, point_cloud, object_clusters=None):
        """Create LiDAR point cloud visualization with cluster-based color coding."""
        if point_cloud is None or len(point_cloud) == 0:
            return None
        
        try:
            # Filter out invalid points
            valid_mask = ~np.any(np.isnan(point_cloud) | np.isinf(point_cloud), axis=1)
            valid_vertices = point_cloud[valid_mask]
            if len(valid_vertices) < 10:
                return None
            
            # --- Use Color Intrinsics for Projection (since depth is aligned to color) ---
            if not hasattr(self, 'camera_utils') or self.camera_utils.color_intrinsics is None:
                print("Warning: Color intrinsics not available for point cloud visualization.")
                return None
            
            intrinsics = self.camera_utils.color_intrinsics
            original_width = intrinsics.width
            original_height = intrinsics.height
            
            # Create a projection canvas with the original color sensor resolution
            proj_canvas = np.zeros((original_height, original_width, 3), dtype=np.uint8)
            
            # Use unscaled color intrinsics for projection
            fx = intrinsics.fx
            fy = intrinsics.fy
            cx = intrinsics.ppx
            cy = intrinsics.ppy
            
            # If we have cluster information, use cluster colors
            if object_clusters and len(object_clusters) > 0:
                # Create a mapping from points to clusters
                point_to_color = {}
                
                for cluster in object_clusters:
                    cluster_points = cluster['points']
                    cluster_color = cluster['color']
                    
                    # Map each point in the cluster to its color
                    for point in cluster_points:
                        point_key = tuple(point.round(6))  # Round to avoid floating point issues
                        point_to_color[point_key] = cluster_color
                
                # Project points with cluster colors
                for i, point_3d in enumerate(valid_vertices):
                    x_3d, y_3d, z_3d = point_3d[0], point_3d[1], max(point_3d[2], 0.001)
                    
                    # Project to 2D
                    x_2d = int((fx * x_3d / z_3d) + cx)
                    y_2d = int((fy * y_3d / z_3d) + cy)
                    
                    if 0 <= x_2d < original_width and 0 <= y_2d < original_height:
                        # Check if this point belongs to a cluster
                        point_key = tuple(point_3d.round(6))
                        if point_key in point_to_color:
                            # Use cluster color
                            color = point_to_color[point_key]
                        else:
                            # Use gray for unclustered points
                            color = (128, 128, 128)
                        
                        proj_canvas[y_2d, x_2d] = color
                
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
                x_pixels = np.clip(x_2d.astype(int), 0, original_width - 1)
                y_pixels = np.clip(y_2d.astype(int), 0, original_height - 1)
                
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
        return self.robot_utils.get_robot_pose()
    
    def pose_to_transform_matrix(self, pose):
        """Convert [x, y, z, rx, ry, rz] pose to 4x4 transformation matrix."""
        return self.coordinate_utils.pose_to_transform_matrix(pose)
    
    def charuco_pose_to_transform_matrix(self, rvec, tvec):
        """Convert CharUco rvec, tvec to 4x4 transformation matrix with coordinate correction."""
        return self.coordinate_utils.charuco_pose_to_transform_matrix(rvec, tvec)
    
    def collect_calibration_point(self, results):
        """Collect robot and pattern poses for calibration (not used in LiDAR mode)."""
        print("Calibration not supported in LiDAR mode")
        return False
    
    def calculate_eye_in_hand_calibration(self):
        """Calculate eye-in-hand calibration (not used in LiDAR mode)."""
        print("Calibration not supported in LiDAR mode")
        return None
    
    def save_calibration_results(self, eye_in_hand_transform):
        """Save calibration results (not used in LiDAR mode)."""
        print("Calibration not supported in LiDAR mode")
        return None
    
    def validate_calibration(self, eye_in_hand_transform):
        """Validate calibration quality (not used in LiDAR mode)."""
        print("Calibration not supported in LiDAR mode")
        return None
    
    def print_validation_report(self, validation_results):
        """Print validation report (not used in LiDAR mode)."""
        print("Calibration not supported in LiDAR mode")
        return
    
    def interactive_validation_review(self, validation_results):
        """Interactive validation review (not used in LiDAR mode)."""
        print("Calibration not supported in LiDAR mode")
        return None
    
    def advanced_calibration_metrics(self, eye_in_hand_transform):
        """Calculate advanced calibration metrics (not used in LiDAR mode)."""
        print("Calibration not supported in LiDAR mode")
        return None
    
    def load_robot_poses(self):
        """Load predefined robot poses from JSON file."""
        return self.robot_utils.load_robot_poses()
    
    def move_robot_to_next_pose(self):
        """Move robot to the next preset pose."""
        return self.robot_utils.move_robot_to_next_pose()
    
    def get_robot_movement_status(self):
        """Get current robot movement status for display."""
        return self.robot_utils.get_robot_movement_status()
    
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
    
    def remove_statistical_outliers(self, point_cloud, num_neighbors=20, std_ratio=1.0):
        """Remove statistical outliers from point cloud."""
        if len(point_cloud) < num_neighbors:
            return point_cloud
        
        from sklearn.neighbors import NearestNeighbors
        
        # Fit KNN model
        nbrs = NearestNeighbors(n_neighbors=num_neighbors).fit(point_cloud)
        distances, _ = nbrs.kneighbors(point_cloud)
        
        # Calculate mean distance for each point
        mean_distances = np.mean(distances, axis=1)
        
        # Calculate threshold
        overall_mean = np.mean(mean_distances)
        std_dev = np.std(mean_distances)
        threshold = overall_mean + std_ratio * std_dev
        
        # Filter outliers
        inlier_mask = mean_distances < threshold
        filtered_points = point_cloud[inlier_mask]
        
        return filtered_points
    
    def detect_tabletop_object(self, color_frame, depth_frame):
        """
        Main object detection method using surface detection and clustering.
        
        Returns:
            dict: Detection results with object information
        """
        try:
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
    
    def find_largest_cluster_center(self, object_points):
        """Legacy method - find center of largest cluster (for backward compatibility)."""
        clusters = self.find_all_significant_clusters(object_points)
        if clusters:
            return clusters[0]['center']  # Return center of largest cluster
        elif len(object_points) > 0:
            return np.mean(object_points, axis=0)  # Fallback to centroid
        else:
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
    
    def draw_lidar_results(self, image, results):
        """Draw LiDAR detection results on image."""
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        # Add detection status
        if results.get('valid_detection', False):
            status_text = f"Object detected ({results.get('num_points', 0)} points)"
            color = (0, 255, 0)
            
            # Draw object center if available
            object_center = results.get('object_center')
            if object_center is not None and self.camera_utils.color_intrinsics is not None:
                pixel_coords = self._project_point_to_pixel(object_center, self.camera_utils.color_intrinsics)
                if pixel_coords:
                    u, v = pixel_coords
                    cv2.drawMarker(vis_image, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 3)
                    cv2.circle(vis_image, (u, v), 10, (0, 255, 0), 2)
        else:
            status_text = "Searching for objects..."
            color = (0, 255, 255)
        
        cv2.putText(vis_image, status_text, (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add robot movement status
        robot_status = self.get_robot_movement_status()
        cv2.putText(vis_image, robot_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return vis_image
    
    def save_results(self, image, results):
        """Save current frame and LiDAR results."""
        return self.detection_utils.save_results(image, results, "lidar_output")
    
    def run(self):
        """Run the real-time LiDAR object detection and position prediction."""
        if not self.setup_camera():
            print("Failed to setup camera. Exiting.")
            return
        
        print("\nStarting real-time LiDAR object position prediction...")
        print("Controls:")
        print("  'c' - Move robot to next preset position")
        print("  'q' - Quit application")
        print("  ESC - Exit")
        print("\nLiDAR-based object detection active - objects will be detected without markers.")
        print("Robot coordinates: X=forward(+), Y=left(+), Z=up(+), units in mm")
        print()
        
        try:
            while True:
                # Get frame from camera
                frames = self.pipeline.wait_for_frames()
                
                # Align depth frame to color frame
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                
                # Detect objects using LiDAR point cloud analysis
                results = self.detect_tabletop_object(color_frame, depth_frame)
                
                # Get current robot position for prediction
                robot_pose = self.get_robot_pose()
                
                # Predict positions for all detected objects
                object_positions = []
                position_in_base = None  # Legacy - largest object position
                
                if results['valid_detection'] and robot_pose is not None:
                    object_clusters = results.get('objects', [])
                    
                    if object_clusters:
                        # Convert robot pose to transformation matrix
                        robot_transform = self.pose_to_transform_matrix(robot_pose)
                        
                        # Print console output periodically (every 30 frames ≈ 1 second)
                        self.position_print_counter += 1
                        should_print = (self.position_print_counter % 30 == 0)
                        
                        if should_print:
                            print(f"\n=== Detected {len(object_clusters)} objects ===")
                        
                        for cluster in object_clusters:
                            object_center = cluster['center']
                            cluster_id = cluster['id']
                            cluster_size = cluster['size']
                            cluster_color = cluster['color']
                            
                            # Predict position in robot base coordinates
                            pos_in_base = self.predict_object_position(object_center, robot_transform)
                            object_positions.append({
                                'id': cluster_id,
                                'position': pos_in_base,
                                'size': cluster_size,
                                'color': cluster_color
                            })
                            
                            # Legacy support - use largest cluster (first in sorted list)
                            if cluster_id == object_clusters[0]['id']:
                                position_in_base = pos_in_base
                                self.update_position_history(position_in_base)
                            
                            # Print position info for each object
                            if should_print:
                                color_name = self._get_color_name(cluster_color)
                                print(f"  Object #{cluster_id} ({color_name}, {cluster_size}pts): X={pos_in_base[0]:6.1f}mm, Y={pos_in_base[1]:6.1f}mm, Z={pos_in_base[2]:6.1f}mm")
                        
                        if should_print:
                            # Show stability for main object
                            mean_pos, std_pos = self.get_position_stability()
                            if mean_pos is not None and len(mean_pos) >= 3:
                                print(f"  Main object stability: ±{std_pos[0]:4.1f}, ±{std_pos[1]:4.1f}, ±{std_pos[2]:4.1f}mm")
                            print("=" * 40)
                
                # Create 2x2 visualization with multi-object support
                vis_image = self.create_2x2_visualization(color_image, depth_frame, results, robot_pose, position_in_base, object_positions)
                
                # Display
                cv2.imshow('LiDAR Real-time Object Detection', vis_image)
                
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
    
    parser = argparse.ArgumentParser(description="Real-time LiDAR object detection")
    parser.add_argument('--config-path', type=str, default='charuco-pose-estimation',
                       help='Path to configuration directory')
    
    args = parser.parse_args()
    
    # Create LiDAR prediction system
    lidar_detector = RealtimeLidarPredictor(args.config_path)
    
    # Run the system
    lidar_detector.run()


if __name__ == "__main__":
    main()