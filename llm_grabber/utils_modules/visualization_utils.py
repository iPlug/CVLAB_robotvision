#!/usr/bin/env python3
"""
Visualization utilities for LiDAR-based object detection and robot vision.
Handles 2D/3D visualization, overlay generation, and multi-panel displays.
"""

import cv2
import numpy as np
import time


class VisualizationUtils:
    """Utilities for creating visualizations and overlays for robot vision systems."""
    
    def __init__(self, config_path="."):
        """Initialize visualization utilities."""
        self.config_path = config_path
        self.viz_show_performance = False
        
    def project_point_to_pixel(self, point_3d, intrinsics):
        """Project 3D point to 2D pixel coordinates."""
        if point_3d is None or len(point_3d) != 3:
            return None
        
        x, y, z = point_3d
        if z <= 0:
            return None
        
        # Project to pixel coordinates
        pixel_x = int((intrinsics.fx * x / z) + intrinsics.ppx)
        pixel_y = int((intrinsics.fy * y / z) + intrinsics.ppy)
        
        # Check if pixel is within image bounds
        if 0 <= pixel_x < intrinsics.width and 0 <= pixel_y < intrinsics.height:
            return (pixel_x, pixel_y)
        
        return None
    
    def create_lidar_point_cloud_view(self, point_cloud, object_clusters=None, intrinsics=None, downsample_func=None):
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
            if downsample_func:
                valid_vertices = downsample_func(valid_vertices)
            
            # Check intrinsics availability
            if intrinsics is None:
                print("Warning: Intrinsics not available for point cloud visualization.")
                return None
            
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
                
                # Assign cluster colors using distance-based assignment
                for cluster in object_clusters:
                    if len(cluster['points']) > 0:
                        cluster_points = np.array(cluster['points'])
                        cluster_color = cluster['color']
                        
                        # Use broadcasting to find closest cluster points
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
                    pixel = self.project_point_to_pixel(center, intrinsics)
                    if pixel is not None:
                        obj_x_pixel, obj_y_pixel = pixel
                        # Scale pixel coordinates for visualization canvas
                        obj_x_pixel = int(obj_x_pixel * viz_scale_factor)
                        obj_y_pixel = int(obj_y_pixel * viz_scale_factor)
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
            
            return canvas
            
        except Exception as e:
            print(f"Error creating LiDAR point cloud view: {e}")
            return None
    
    def create_depth_image_view(self, depth_frame, object_clusters=None):
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
                min_depth = np.min(valid_depths)
                max_depth = np.max(valid_depths)
                
                if max_depth > min_depth:
                    # Normalize depth values to 0-255 range
                    normalized_depth = np.zeros_like(depth_m)
                    normalized_depth[valid_mask] = ((valid_depths - min_depth) / (max_depth - min_depth)) * 255
                    
                    # Apply colormap
                    depth_colorized = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)
            
            return depth_colorized
            
        except Exception as e:
            print(f"Error creating depth image view: {e}")
            return None
    
    def get_color_name(self, color_tuple):
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
            (0, 128, 128): "Teal",
        }
        
        return color_map.get(color_tuple, f"RGB{color_tuple}")
    
    def add_tracking_info_overlay(self, image, results, robot_pose, position_in_base):
        """Add tracking information overlay to image."""
        overlay_y = 30
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Detection status
        status_text = "✓ OBJECT DETECTED" if results.get('valid_pose', False) else "✗ NO DETECTION"
        color = (0, 255, 0) if results.get('valid_pose', False) else (0, 0, 255)
        cv2.putText(image, status_text, (10, overlay_y), font, font_scale, color, thickness)
        overlay_y += line_height
        
        # Detection confidence
        if 'confidence' in results:
            confidence_text = f"Confidence: {results['confidence']:.2f}"
            cv2.putText(image, confidence_text, (10, overlay_y), font, font_scale, (255, 255, 255), thickness)
            overlay_y += line_height
        
        # Object position
        if results.get('valid_pose', False) and 'center' in results:
            center = results['center']
            pos_text = f"Pos: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})m"
            cv2.putText(image, pos_text, (10, overlay_y), font, font_scale, (255, 255, 255), thickness)
            overlay_y += line_height
        
        # Robot position in base coordinates
        if position_in_base is not None:
            coord_system = position_in_base[1] if isinstance(position_in_base, tuple) else "robot"
            pos_values = position_in_base[0] if isinstance(position_in_base, tuple) else position_in_base
            
            robot_text = f"Robot ({coord_system}): ({pos_values[0]:.1f}, {pos_values[1]:.1f}, {pos_values[2]:.1f})mm"
            cv2.putText(image, robot_text, (10, overlay_y), font, font_scale, (0, 255, 255), thickness)
            overlay_y += line_height
        
        # Current robot pose
        if robot_pose:
            pose_text = f"Joint Pose: [{robot_pose[0]:.1f}, {robot_pose[1]:.1f}, {robot_pose[2]:.1f}, {robot_pose[3]:.1f}, {robot_pose[4]:.1f}, {robot_pose[5]:.1f}]"
            cv2.putText(image, pose_text, (10, overlay_y), font, 0.4, (128, 128, 128), 1)
        
        return image
    
    def add_object_detection_overlay(self, image, results, object_positions=None):
        """Add object detection overlay with multi-object support."""
        if not results.get('valid_pose', False):
            return image
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Handle multiple objects if available
        if object_positions and len(object_positions) > 0:
            for i, obj_pos in enumerate(object_positions):
                if obj_pos.get('pixel_position'):
                    x, y = obj_pos['pixel_position']
                    
                    # Draw object marker
                    object_color = obj_pos.get('color', (0, 255, 0))
                    cv2.circle(image, (x, y), 12, object_color, 3)
                    cv2.circle(image, (x, y), 20, (255, 255, 255), 2)
                    
                    # Add object ID
                    object_id = obj_pos.get('id', i + 1)
                    cv2.putText(image, str(object_id), (x + 25, y - 10), 
                              font, 0.7, (255, 255, 255), 2)
                    
                    # Add distance info
                    if obj_pos.get('distance'):
                        dist_text = f"{obj_pos['distance']:.3f}m"
                        cv2.putText(image, dist_text, (x + 25, y + 15), 
                                  font, 0.5, (255, 255, 255), 1)
        else:
            # Single object detection (legacy support)
            if 'pixel_position' in results:
                x, y = results['pixel_position']
                
                # Draw crosshair
                cv2.line(image, (x-20, y), (x+20, y), (0, 255, 0), 3)
                cv2.line(image, (x, y-20), (x, y+20), (0, 255, 0), 3)
                cv2.circle(image, (x, y), 25, (0, 255, 0), 2)
                
                # Add center marker
                cv2.circle(image, (x, y), 3, (255, 255, 255), -1)
        
        return image
    
    def create_2x2_visualization(self, color_image, depth_frame, results, robot_pose, position_in_base, 
                                object_positions=None, point_cloud=None, object_clusters=None, 
                                intrinsics=None, downsample_func=None):
        """Create 2x2 grid visualization with color, depth, and point cloud views."""
        panel_size = 400
        
        # Top-left: Color image with overlays
        color_panel = color_image.copy()
        color_panel = cv2.resize(color_panel, (panel_size, panel_size))
        color_panel = self.add_tracking_info_overlay(color_panel, results, robot_pose, position_in_base)
        color_panel = self.add_object_detection_overlay(color_panel, results, object_positions)
        
        # Top-right: Depth visualization
        depth_panel = self.create_depth_image_view(depth_frame, object_clusters)
        if depth_panel is None:
            depth_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
        
        # Bottom-left: Point cloud visualization
        if point_cloud is not None and intrinsics is not None:
            lidar_panel = self.create_lidar_point_cloud_view(point_cloud, object_clusters, intrinsics, downsample_func)
        else:
            lidar_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
        
        if lidar_panel is None:
            lidar_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
        
        # Bottom-right: Status/Info panel
        info_panel = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
        
        # Add panel labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(color_panel, "Color + Detection", (10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(depth_panel, "Depth View", (10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(lidar_panel, "Point Cloud View", (10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(info_panel, "System Status", (10, 25), font, 0.7, (255, 255, 255), 2)
        
        # Combine panels into 2x2 grid
        top_row = np.hstack([color_panel, depth_panel])
        bottom_row = np.hstack([lidar_panel, info_panel])
        combined = np.vstack([top_row, bottom_row])
        
        return combined