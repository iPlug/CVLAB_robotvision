#!/usr/bin/env python3
"""
Base class for eye-in-hand calibration systems.

This module contains shared functionality for both:
- Standard eye-in-hand calibration (moving target)
- Fixed target eye-in-hand calibration

Provides coordinate system handling, visualization, IMU calibration,
and other common utilities.
"""

import time
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple
from sklearn.linear_model import RANSACRegressor

from module.robot.command_queue import CommandStatus


class EyeInHandCalibratorBase:
    """
    Base class for eye-in-hand calibration systems.
    
    Contains shared functionality:
    - Object position extraction with raw/world coordinate support
    - Table surface detection using RANSAC
    - Visualization systems with raw point cloud support
    - IMU calibration and gravity alignment
    - Robot movement with live visualization
    - Coordinate system debugging and validation
    """
    
    def __init__(self):
        # These will be set by subclasses
        self.sensor_manager = None
        self.robot_controller = None
        self.detector = None
        self.visualizer = None
        
        # Calibration object requirements (shared)
        self.min_object_size = 20  # Minimum object size for reliable detection
        self.min_object_distance = 250  # Minimum distance from camera (mm) - lidar effective range
        self.max_object_distance = 500  # Maximum distance from camera (mm)
        
        # Common visualization settings
        self.visualize_raw_pointcloud = True
    
    def _extract_object_position(self, obj: Dict[str, Any], use_raw_coordinates: bool = False) -> Optional[List[float]]:
        """
        Extract 3D position from object in camera coordinates.
        
        Args:
            obj: Object detection data
            use_raw_coordinates: If True, return position in raw camera frame (for eye-in-hand calibration)
                                If False, return position in world-aligned frame (for display/tracking)
                                
        Returns:
            Position in [forward, side, height] format in mm, or None if extraction fails
        """
        try:
            world_aligned_position = None
            
            if 'center_mm' in obj:
                # center_mm format: [x, y, z] = [side, height, forward]
                center_mm = obj['center_mm']
                world_aligned_position = np.array([center_mm[2], center_mm[0], center_mm[1]])  # [forward, side, height]
            elif 'center' in obj:
                # center format in meters
                center = obj['center']
                world_aligned_position = np.array([center[2]*1000, center[0]*1000, center[1]*1000])  # Convert to mm
            else:
                return None
            
            if use_raw_coordinates and self.sensor_manager:
                # Transform from world-aligned back to raw camera coordinates
                # The detected position is in world-aligned coordinates, we need raw camera coordinates
                world_aligned_3d = np.array([world_aligned_position[1], world_aligned_position[2], world_aligned_position[0]])  # [side, height, forward]
                raw_camera_3d = self.sensor_manager.transform_to_raw_camera_frame(world_aligned_3d.reshape(1, -1))[0]
                # Convert back to [forward, side, height] format
                return [raw_camera_3d[2], raw_camera_3d[0], raw_camera_3d[1]]
            else:
                # Return world-aligned coordinates
                return world_aligned_position.tolist()
                
        except Exception as e:
            print(f"Error extracting object position: {e}")
            return None
    
    def _detect_table_surface_ransac(self, vertices: np.ndarray, max_distance: float = 0.01) -> Optional[Dict[str, Any]]:
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
            
            # Filter points by distance from camera (Z-axis): 250mm to 500mm
            z_range = (0, 0.4)  # 250mm to 500mm
            x_range = (-0.1, 0.1)  # X-axis range for table surface
            
            filtered_vertices = valid_vertices[
                (valid_vertices[:, 2] >= z_range[0]) & (valid_vertices[:, 2] <= z_range[1]) &
                (valid_vertices[:, 0] >= x_range[0]) & (valid_vertices[:, 0] <= x_range[1])
            ]

            if len(filtered_vertices) < 100:
                print(f"Warning: Only {len(filtered_vertices)} points in distance range 250-500mm")
                return None
            
            print(f"+ Filtered points: {len(filtered_vertices)}/{len(valid_vertices)} points in 250-500mm range")
            
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
    
    def _wait_for_movement_with_visualization(self, command_id: str, timeout: float = 30.0):
        """Wait for robot movement completion while maintaining live visualization."""
        start_time = time.time()
        
        print("Maintaining live visualization during robot movement...")
        
        while time.time() - start_time < timeout:
            # Check if movement is complete
            command_status = self.robot_controller.get_command_status(command_id)
            if command_status in [CommandStatus.COMPLETED, CommandStatus.FAILED, CommandStatus.CANCELLED, None]:
                if command_status == CommandStatus.COMPLETED:
                    print("[OK] Robot movement completed")
                elif command_status == CommandStatus.FAILED:
                    print("[ERROR] Robot movement failed")
                elif command_status == CommandStatus.CANCELLED:
                    print("[CANCELLED] Robot movement cancelled")
                else:
                    print("[UNKNOWN] Robot movement status unknown")
                break
            
            try:
                # Get frame and show live visualization during movement
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                # Get raw vertices if raw visualization is enabled
                raw_vertices = None
                if self.visualize_raw_pointcloud:
                    raw_vertices = self.sensor_manager.get_raw_vertices()
                
                if vertices is not None:
                    # Quick detection for live feedback (no heavy processing during movement)
                    timing_info = {
                        'total_time': 0.01,
                        'status_message': f"Robot moving... (live feed active)"
                    }
                    
                    # Simple visualization during movement
                    if color_image is not None:
                        display_image = color_image.copy()
                        cv2.putText(display_image, "ROBOT MOVING - LIVE FEED", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.putText(display_image, "Visualization continues during movement", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Use simplified 2x2 layout during movement
                        panel_size = 400  # Keep consistent with main visualization
                        
                        # Create simple 2x2 layout
                        top_left = cv2.resize(display_image, (panel_size, panel_size))
                        top_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                        bottom_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                        bottom_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                        
                        # Add placeholders
                        cv2.putText(bottom_left, "BEV", (panel_size//2-20, panel_size//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                        cv2.putText(bottom_right, "SIDE", (panel_size//2-30, panel_size//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                        
                        # Add raw point cloud if enabled (top-right)
                        if self.visualize_raw_pointcloud and raw_vertices is not None:
                            raw_panel = self._create_raw_point_cloud_view(raw_vertices)
                            if raw_panel is not None:
                                top_right = cv2.resize(raw_panel, (panel_size, panel_size))
                            else:
                                cv2.putText(top_right, "RAW", (panel_size//2-20, panel_size//2), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                        else:
                            cv2.putText(top_right, "RAW OFF", (panel_size//2-40, panel_size//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
                        
                        # Combine and display
                        top_row = np.hstack([top_left, top_right])
                        bottom_row = np.hstack([bottom_left, bottom_right])
                        movement_display = np.vstack([top_row, bottom_row])
                        
                        cv2.imshow('Eye-in-Hand Calibration', movement_display)
                
                # Check for user interrupt
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("Movement interrupted by user")
                    self.robot_controller.stop_movement_async(priority=3)
                    break
                    
            except Exception as e:
                print(f"Visualization error during movement: {e}")
                cv2.waitKey(30)  # Keep responsive
        
        # Final wait for settlement
        print("Allowing time for vibration settling...")
        self._wait_with_visualization(2.0, "Vibration settling...")
    
    def _wait_with_visualization(self, duration: float, status_message: str):
        """Wait for specified duration while maintaining live visualization."""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                # Get frame and show live visualization during wait
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                # Get raw vertices if raw visualization is enabled
                raw_vertices = None
                if self.visualize_raw_pointcloud:
                    raw_vertices = self.sensor_manager.get_raw_vertices()
                
                if vertices is not None:
                    # Quick timing info for status
                    timing_info = {
                        'total_time': 0.01,
                        'status_message': status_message
                    }
                    
                    # Simple visualization during wait
                    if color_image is not None:
                        display_image = color_image.copy()
                        cv2.putText(display_image, status_message.upper(), (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        remaining_time = duration - (time.time() - start_time)
                        cv2.putText(display_image, f"Remaining: {remaining_time:.1f}s", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Use simplified 2x2 layout during wait
                        panel_size = 400  # Keep consistent with main visualization
                        
                        # Create simple 2x2 layout
                        top_left = cv2.resize(display_image, (panel_size, panel_size))
                        top_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                        bottom_left = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                        bottom_right = np.zeros((panel_size, panel_size, 3), dtype=np.uint8)
                        
                        # Add placeholders
                        cv2.putText(bottom_left, "BEV", (panel_size//2-20, panel_size//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                        cv2.putText(bottom_right, "SIDE", (panel_size//2-30, panel_size//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                        
                        # Add raw point cloud if enabled (top-right)
                        if self.visualize_raw_pointcloud and raw_vertices is not None:
                            raw_panel = self._create_raw_point_cloud_view(raw_vertices)
                            if raw_panel is not None:
                                top_right = cv2.resize(raw_panel, (panel_size, panel_size))
                            else:
                                cv2.putText(top_right, "RAW", (panel_size//2-20, panel_size//2), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
                        else:
                            cv2.putText(top_right, "RAW OFF", (panel_size//2-40, panel_size//2), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 2)
                        
                        # Combine and display
                        top_row = np.hstack([top_left, top_right])
                        bottom_row = np.hstack([bottom_left, bottom_right])
                        wait_display = np.vstack([top_row, bottom_row])
                        
                        cv2.imshow('Eye-in-Hand Calibration', wait_display)
                
                # Check for user interrupt
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("Wait interrupted by user")
                    break
                    
            except Exception as e:
                print(f"Visualization error during wait: {e}")
                cv2.waitKey(30)  # Keep responsive
    
    def _is_object_valid_for_calibration(self, obj: Dict[str, Any]) -> bool:
        """Check if object is valid for calibration."""
        position = self._extract_object_position(obj)
        if not position:
            return False
        
        distance = np.linalg.norm(position)
        return self.min_object_distance <= distance <= self.max_object_distance
    
    def _select_best_calibration_object(self, objects_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best object for calibration from multiple detections."""
        if not objects_list:
            return None
        
        # For now, select the largest object (most reliable detection)
        best_object = None
        max_size = 0
        
        for obj in objects_list:
            # Try to get object size
            size = 0
            if 'num_points' in obj:
                size = obj['num_points']
            elif 'area' in obj:
                size = obj['area']
            elif 'bbox' in obj:
                bbox = obj['bbox']
                size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # width * height
            
            if size > max_size:
                max_size = size
                best_object = obj
        
        return best_object