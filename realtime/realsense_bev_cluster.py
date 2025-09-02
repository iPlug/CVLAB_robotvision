#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
from bag_inspector import inspect_bag_file
from scipy import ndimage
from sklearn.cluster import DBSCAN
import time

class BEVClusteringVisualizer:
    def __init__(self, bag_file=None, realtime_mode=False, use_imu=True):
        self.bag_file = bag_file
        self.realtime_mode = realtime_mode
        self.use_imu = use_imu
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pc = rs.pointcloud()
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2)
        
        # Rotation matrix for alignment (identity initially)
        self.rotation_matrix = np.eye(3)
        self.gravity_vector = np.array([0, -1, 0])  # Default gravity pointing down
        self.alignment_calibrated = False
        
        # ROI parameters (in meters, after rotation) - for filtering points
        self.roi_params = {
            'x_min': -1.0, 'x_max': 1.0,
            'y_min':0.3, 'y_max': .38,  # Height range
            'z_min': 0, 'z_max': .5    # Depth range
        }
        
        # Display ROI parameters (in meters) - for limiting BEV display area
        self.display_roi_params = {
            'x_min': -1.5, 'x_max': 1.5,
            'y_min': -0.8, 'y_max': 0.5,  # Height range
            'z_min': 0, 'z_max': 1.0    # Depth range
        }
        
        # RGB color filter parameters (BGR format for OpenCV)
        # Example: Yellow to Green filter (0-1 range)
        self.color_filter_params = {
            'enabled': False,
            'r_min': 0.2, 'r_max': 1.0,   # Red channel (moderate to high for yellow)
            'g_min': 0.6, 'g_max': 1.0,   # Green channel (high for both yellow and green)
            'b_min': 0.0, 'b_max': 0.4    # Blue channel (low for yellow/green)
        }
        
        # BEV clustering parameters
        self.clustering_params = {
            'min_cluster_size': 30,  # Minimum pixels for a valid cluster
            'max_cluster_size': 500,  # Maximum pixels for a valid cluster (filter out large surfaces)
            'morphology_kernel_size': 5,  # Kernel size for morphological operations
            'window_size': 5,  # Number of frames to average for noise reduction
        }
        
        # 3D Statistical outlier filtering parameters (early stage)
        self.outlier_params = {
            'enabled': True,  # Enable/disable 3D statistical outlier removal
            'k_neighbors': 15,  # Number of neighbors to consider
            'std_ratio': 0.1  # Standard deviation ratio for outlier detection
        }
        
        # Frame history for temporal filtering
        self.frame_history = []
        
        # FPS tracking
        self.fps_history = []
        self.fps_window_size = 30
        
        # Cluster position history for stable positioning
        self.cluster_position_history = {}
        
        # Visualization parameters
        self.show_bounding_boxes = True
        self.show_cluster_ids = True
        self.show_smooth_bev = True  # Toggle between smooth BEV and pillar visualization
        
        # Configure streams
        self._configure_streams(bag_file)
        
    def _configure_streams(self, bag_file):
        """Configure camera and IMU streams"""
        if bag_file:
            if not os.path.exists(bag_file):
                raise FileNotFoundError(f"Bag file '{bag_file}' not found!")
            rs.config.enable_device_from_file(self.config, bag_file)
        
            available_streams = inspect_bag_file(bag_file)
            self._enable_streams_from_bag(available_streams)
        else:
            # Configure for live camera
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Enable IMU streams if available
            if self.use_imu:
                try:
                    self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
                    self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
                    print("IMU streams enabled for gravity-based alignment")
                except:
                    print("IMU not available, using default orientation")
                    self.use_imu = False

    def _enable_streams_from_bag(self, available_streams):
        """Enable streams based on bag file contents"""
        has_depth = False
        has_color = False
        has_imu = False
        
        if not available_streams:
            print("Could not determine available streams. Exiting.")
            return
        
        for stream_info in available_streams:
            stream_type = stream_info['stream']
            
            if stream_type == rs.stream.depth and 'width' in stream_info:
                print(f"Enabling depth stream: {stream_info['width']}x{stream_info['height']} @ {stream_info['fps']}fps")
                self.config.enable_stream(rs.stream.depth, stream_info['width'], stream_info['height'], 
                                stream_info['format'], stream_info['fps'])
                has_depth = True
            elif stream_type == rs.stream.color and 'width' in stream_info:
                print(f"Enabling color stream: {stream_info['width']}x{stream_info['height']} @ {stream_info['fps']}fps")
                self.config.enable_stream(rs.stream.color, stream_info['width'], stream_info['height'], 
                                stream_info['format'], stream_info['fps'])
                has_color = True
            elif stream_type == rs.stream.accel and self.use_imu and 'format' in stream_info:
                print(f"Enabling IMU stream: {stream_type}")
                self.config.enable_stream(rs.stream.accel, stream_info['format'])
                has_imu = True
            elif stream_type == rs.stream.gyro and self.use_imu and 'format' in stream_info:
                print(f"Enabling IMU stream: {stream_type}")
                self.config.enable_stream(rs.stream.gyro, stream_info['format'])
                has_imu = True
        
        if not has_depth:
            print("No depth stream found in bag file!")
            return
        
        if has_imu:
            print("IMU data available for gravity-based alignment")
        else:
            print("No IMU data in bag file, using default orientation")
            self.use_imu = False
        
    def start(self):
        """Start the pipeline with error handling"""
        try:
            profile = self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error starting pipeline with specific configuration: {e}")
            print("Trying with automatic stream detection...")
            
            # Try with automatic configuration
            self.config = rs.config()
            if self.bag_file:
                rs.config.enable_device_from_file(self.config, self.bag_file)
            else:
                # For live camera, try basic configuration
                self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            try:
                profile = self.pipeline.start(self.config)
                print("Successfully started with automatic configuration")
                
                # Check if IMU is available in automatic mode
                device = profile.get_device()
                has_imu = False
                for sensor in device.sensors:
                    for stream_profile in sensor.get_stream_profiles():
                        if stream_profile.stream_type() in [rs.stream.accel, rs.stream.gyro]:
                            has_imu = True
                            break
                
                if not has_imu:
                    print("No IMU data available in automatic mode")
                    self.use_imu = False
                    
            except Exception as e2:
                print(f"Failed with automatic configuration too: {e2}")
                raise e2
        
        if self.bag_file:
            playback = profile.get_device().as_playback()
            playback.set_real_time(self.realtime_mode)
            print(f"Playing bag file: {self.bag_file}")
            if not self.realtime_mode:
                print("Non-real-time mode: Processing all frames")
        else:
            print("Live camera mode")
        
        return profile
    
    def calibrate_gravity_alignment(self, num_samples=100):
        """Calibrate gravity direction from IMU data"""
        if not self.use_imu or self.alignment_calibrated:
            return
        
        accel_samples = []
        
        print(f"Calibrating gravity alignment... collecting {num_samples} samples")
        
        for i in range(num_samples):
            try:
                if self.bag_file:
                    frames_available, frames = self.pipeline.try_wait_for_frames(timeout_ms=100)
                    if not frames_available:
                        continue
                else:
                    frames = self.pipeline.wait_for_frames()
                
                # Get accelerometer data
                accel_frame = frames.first_or_default(rs.stream.accel)
                if accel_frame:
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    accel_samples.append([accel_data.x, accel_data.y, accel_data.z])
                
                # if i % 20 == 0:
                #     print(f"  Collected {len(accel_samples)}/{num_samples} samples...")
                    
            except Exception as e:
                print(f"  Error collecting sample {i}: {e}")
                continue
        
        if len(accel_samples) > 10:  # Need at least some samples
            # Average accelerometer readings to get gravity vector
            accel_avg = np.mean(accel_samples, axis=0)
            gravity_magnitude = np.linalg.norm(accel_avg)
            
            if gravity_magnitude > 0.5:  # Reasonable gravity reading (should be ~9.8 m/s²)
                # Normalize gravity vector
                self.gravity_vector = accel_avg / gravity_magnitude
                
                # Calculate rotation matrix to align gravity with -Y axis
                target_up = np.array([0, -1, 0])  # Y-up in camera coordinates
                self.rotation_matrix = self._calculate_alignment_rotation(self.gravity_vector, target_up)
                
                print(f"Gravity vector detected: {self.gravity_vector}")
                print(f"Gravity magnitude: {gravity_magnitude:.2f} m/s²")
                print(f"Alignment calibrated successfully!")
                self.alignment_calibrated = True
            else:
                print(f"Could not detect reliable gravity vector (magnitude: {gravity_magnitude:.2f})")
        else:
            print(f"Insufficient IMU data for calibration (only {len(accel_samples)} samples)")
        
        if not self.alignment_calibrated:
            print("Using default orientation (no alignment)")
            self.use_imu = False
    
    def _calculate_alignment_rotation(self, from_vec, to_vec):
        """Calculate rotation matrix to align from_vec to to_vec"""
        from_vec = from_vec / np.linalg.norm(from_vec)
        to_vec = to_vec / np.linalg.norm(to_vec)
        
        # If vectors are already aligned
        if np.allclose(from_vec, to_vec):
            return np.eye(3)
        
        # If vectors are opposite
        if np.allclose(from_vec, -to_vec):
            # Find any perpendicular vector
            perp = np.array([1, 0, 0]) if abs(from_vec[0]) < 0.9 else np.array([0, 1, 0])
            perp = perp - np.dot(perp, from_vec) * from_vec
            perp = perp / np.linalg.norm(perp)
            # 180-degree rotation around perpendicular axis
            return 2 * np.outer(perp, perp) - np.eye(3)
        
        # General case: Rodrigues' rotation formula
        v = np.cross(from_vec, to_vec)
        s = np.linalg.norm(v)
        c = np.dot(from_vec, to_vec)
        
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        
        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
        return R
    
    def apply_roi_filter(self, vertices):
        """Apply ROI filtering to point cloud"""
        if vertices is None or len(vertices) == 0:
            return vertices
        
        # Apply ROI filtering
        mask = ((vertices[:, 0] >= self.roi_params['x_min']) & 
                (vertices[:, 0] <= self.roi_params['x_max']) &
                (vertices[:, 1] >= self.roi_params['y_min']) & 
                (vertices[:, 1] <= self.roi_params['y_max']) &
                (vertices[:, 2] >= self.roi_params['z_min']) & 
                (vertices[:, 2] <= self.roi_params['z_max']))
        
        return vertices[mask]
    
    def apply_3d_statistical_outlier_filter(self, vertices, k_neighbors=20, std_ratio=2.0):
        """Remove 3D statistical outliers based on distance to neighbors"""
        if vertices is None or len(vertices) < k_neighbors:
            return vertices
        
        # Calculate distances to k nearest neighbors for each point
        from sklearn.neighbors import NearestNeighbors
        
        # Use all 3D coordinates for neighbor calculation
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto').fit(vertices)
        distances, indices = nbrs.kneighbors(vertices)
        
        # Calculate mean distance to neighbors (excluding self at index 0)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # Statistical outlier detection
        overall_mean = np.mean(mean_distances)
        overall_std = np.std(mean_distances)
        
        # Keep points within std_ratio standard deviations
        threshold = overall_mean + std_ratio * overall_std
        inlier_mask = mean_distances <= threshold
        
        return vertices[inlier_mask]
    
    def apply_color_filter(self, vertices, texcoords, color_image):
        """Apply RGB color filtering to point cloud (optimized)"""
        if not self.color_filter_params['enabled'] or color_image is None or texcoords is None:
            return vertices
        
        if len(vertices) == 0 or len(texcoords) == 0:
            return vertices
        
        # Get color values for each point (vectorized)
        height, width = color_image.shape[:2]
        
        # Vectorized texture coordinate to pixel coordinate conversion
        u = np.clip((texcoords[:, 0] * width).astype(np.int32), 0, width - 1)
        v = np.clip((texcoords[:, 1] * height).astype(np.int32), 0, height - 1)
        
        # Vectorized color sampling (much faster than individual lookups)
        colors = color_image[v, u]
        
        # Get filter parameters once
        r_min, r_max = self.color_filter_params['r_min'], self.color_filter_params['r_max']
        g_min, g_max = self.color_filter_params['g_min'], self.color_filter_params['g_max']
        b_min, b_max = self.color_filter_params['b_min'], self.color_filter_params['b_max']
        
        # Vectorized color filtering (BGR format) - much faster than individual comparisons
        color_mask = ((colors[:, 2] >= r_min) & (colors[:, 2] <= r_max) &  # R channel
                     (colors[:, 1] >= g_min) & (colors[:, 1] <= g_max) &   # G channel
                     (colors[:, 0] >= b_min) & (colors[:, 0] <= b_max))    # B channel
        
        return vertices[color_mask]
    
    def get_3d_points(self, frames, apply_alignment=True):
        """Extract and optionally align 3D points from frames"""
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame:
            return None, None, None
        
        # Apply decimation filter to reduce the number of points
        depth_frame = self.decimate.process(depth_frame)
        
        # Generate point cloud
        if color_frame:
            self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)
        
        # Get vertices
        v = points.get_vertices()
        vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        
        # Apply gravity-based alignment if calibrated
        if apply_alignment and self.alignment_calibrated:
            vertices = self._rotate_points(vertices)
        
        # Get texture coordinates and color image first (needed for color filtering)
        texcoords = None
        color_image = None
        
        if color_frame:
            t = points.get_texture_coordinates()
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
            color_image = np.asanyarray(color_frame.get_data())
            
            # Handle different color formats (RGB8 vs BGR8)
            color_profile = color_frame.get_profile().as_video_stream_profile()
            if color_profile.format() == rs.format.rgb8:
                # Convert RGB to BGR for OpenCV
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
        # Apply ROI filtering
        vertices = self.apply_roi_filter(vertices)
        
        # Apply 3D statistical outlier filtering (remove isolated points)
        if self.outlier_params['enabled'] and vertices is not None and len(vertices) > 0:
            vertices = self.apply_3d_statistical_outlier_filter(
                vertices, 
                k_neighbors=self.outlier_params['k_neighbors'], 
                std_ratio=self.outlier_params['std_ratio'])
        
        return vertices, texcoords, color_image
    
    def _rotate_points(self, vertices):
        """Apply rotation matrix to align coordinate system"""
        if vertices is None or len(vertices) == 0:
            return vertices
        
        # Apply rotation to all points
        rotated_vertices = np.dot(vertices, self.rotation_matrix.T)
        return rotated_vertices
    
    
    def create_point_cloud_view(self, vertices):
        """Create a 3D point cloud visualization from filtered vertices"""
        if vertices is None or len(vertices) == 0:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Filter out invalid points
        valid_mask = vertices[:, 2] > 0
        valid_points = vertices[valid_mask]
        
        if len(valid_points) == 0:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Create a side view (X-Y projection, looking from Z direction)
        img_size = 400
        
        # Get coordinate ranges
        x_coords = valid_points[:, 0]  # Left-right
        y_coords = valid_points[:, 1]  # Height (up-down)
        z_coords = valid_points[:, 2]  # Depth (forward-backward)
        
        # Calculate ranges with some padding
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add padding
        x_padding = max(0.1, (x_max - x_min) * 0.1)
        y_padding = max(0.1, (y_max - y_min) * 0.1)
        
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        # Ensure we have a valid range
        if x_max <= x_min:
            x_min, x_max = -1, 1
        if y_max <= y_min:
            y_min, y_max = -1, 1
        
        # Convert to image coordinates
        x_img = ((x_coords - x_min) / (x_max - x_min) * img_size).astype(int)
        y_img = ((y_coords - y_min) / (y_max - y_min) * img_size).astype(int)
        
        # Flip Y for image display (higher Y values at top)
        y_img = img_size - 1 - y_img
        
        # Filter points within image bounds
        valid_img_mask = (x_img >= 0) & (x_img < img_size) & (y_img >= 0) & (y_img < img_size)
        x_img = x_img[valid_img_mask]
        y_img = y_img[valid_img_mask]
        z_valid = z_coords[valid_img_mask]
        
        # Create point cloud image
        pc_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Color points based on depth (Z coordinate)
        if len(z_valid) > 0:
            # Normalize depth for coloring
            z_min, z_max = np.min(z_valid), np.max(z_valid)
            if z_max > z_min:
                z_normalized = (z_valid - z_min) / (z_max - z_min)
            else:
                z_normalized = np.zeros_like(z_valid)
            
            # Apply colormap (red for near, blue for far)
            colors = plt.cm.jet(z_normalized)[:, :3] * 255
            
            # Draw points
            point_size = 2
            for i in range(len(x_img)):
                cv2.circle(pc_image, (x_img[i], y_img[i]), point_size, 
                          (int(colors[i][2]), int(colors[i][1]), int(colors[i][0])), -1)
        
        # Add axis labels and grid
        cv2.putText(pc_image, "Point Cloud (Side View)", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(pc_image, f"X: [{x_min:.2f}, {x_max:.2f}]m", (10, img_size - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(pc_image, f"Y: [{y_min:.2f}, {y_max:.2f}]m", (10, img_size - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw coordinate lines
        # Y=0 line (camera height level after gravity alignment)
        if y_min <= 0 <= y_max:
            camera_level_y = int((0 - y_min) / (y_max - y_min) * img_size)
            camera_level_y = img_size - 1 - camera_level_y  # Flip for display
            cv2.line(pc_image, (0, camera_level_y), (img_size-1, camera_level_y), (0, 255, 0), 1)
            cv2.putText(pc_image, "Camera Level (Y=0)", (img_size-130, camera_level_y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # ROI boundaries
        roi_y_min_img = int((self.roi_params['y_min'] - y_min) / (y_max - y_min) * img_size)
        roi_y_max_img = int((self.roi_params['y_max'] - y_min) / (y_max - y_min) * img_size)
        roi_y_min_img = img_size - 1 - roi_y_min_img  # Flip for display
        roi_y_max_img = img_size - 1 - roi_y_max_img  # Flip for display
        
        # Clamp to image bounds
        roi_y_min_img = max(0, min(img_size-1, roi_y_min_img))
        roi_y_max_img = max(0, min(img_size-1, roi_y_max_img))
        
        # Draw ROI boundaries
        cv2.line(pc_image, (0, roi_y_min_img), (img_size-1, roi_y_min_img), (255, 0, 0), 2)
        cv2.line(pc_image, (0, roi_y_max_img), (img_size-1, roi_y_max_img), (255, 0, 0), 2)
        cv2.putText(pc_image, f"Y_min={self.roi_params['y_min']:.2f}", (5, roi_y_min_img-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        cv2.putText(pc_image, f"Y_max={self.roi_params['y_max']:.2f}", (5, roi_y_max_img+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        return pc_image
    
    def create_smooth_bev_image(self, vertices, x_range, z_range):
        """Create smooth BEV image using realsense pointcloud viewer method"""
        if vertices is None or len(vertices) == 0:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Filter out invalid points
        valid_mask = vertices[:, 2] > 0
        valid_points = vertices[valid_mask]
        
        if len(valid_points) == 0:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Use aligned coordinates for top-down view
        x_coords = valid_points[:, 0]  # Left-right (aligned)
        z_coords = valid_points[:, 2]  # Forward-backward (aligned)
        y_coords = valid_points[:, 1]  # Height (aligned, up-down)
        
        # Calculate aspect ratio and image dimensions
        x_span = x_range[1] - x_range[0]
        z_span = z_range[1] - z_range[0]
        
        # Target image size
        max_img_size = 400
        
        if x_span == 0 or z_span == 0:
            return np.zeros((max_img_size, max_img_size, 3), dtype=np.uint8)
        
        # Maintain aspect ratio
        aspect_ratio = x_span / z_span
        
        if aspect_ratio > 1:  # Wider than tall
            img_width = max_img_size
            img_height = int(max_img_size / aspect_ratio)
        else:  # Taller than wide
            img_height = max_img_size
            img_width = int(max_img_size * aspect_ratio)
        
        # Ensure minimum size
        img_width = max(img_width, 100)
        img_height = max(img_height, 100)
        
        # Convert to image coordinates
        x_img = ((x_coords - x_range[0]) / x_span * img_width).astype(int)
        z_img = ((z_coords - z_range[0]) / z_span * img_height).astype(int)
        
        # Filter points within image bounds
        valid_img_mask = (x_img >= 0) & (x_img < img_width) & (z_img >= 0) & (z_img < img_height)
        x_img = x_img[valid_img_mask]
        z_img = z_img[valid_img_mask]
        y_valid = y_coords[valid_img_mask]
        
        # Create top-down image
        bev_image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Color points based on height
        if len(y_valid) > 0:
            # Normalize height for coloring within ROI range
            y_min, y_max = self.roi_params['y_min'], self.roi_params['y_max']
            
            # Use actual data range if ROI range is too restrictive
            actual_y_min, actual_y_max = np.min(y_valid), np.max(y_valid)
            if actual_y_max - actual_y_min > y_max - y_min:
                y_min, y_max = actual_y_min, actual_y_max
            
            if y_max > y_min:
                y_normalized = np.clip((y_valid - y_min) / (y_max - y_min), 0, 1)
            else:
                y_normalized = np.zeros_like(y_valid)
            
            # Apply colormap (red for high, blue for low)
            colors = plt.cm.jet(y_normalized)[:, :3] * 255
            
            # Draw points
            point_size = max(1, min(3, max_img_size // 200))
            
            for i in range(len(x_img)):
                cv2.circle(bev_image, (x_img[i], img_height - z_img[i] - 1), point_size, 
                          (int(colors[i][2]), int(colors[i][1]), int(colors[i][0])), -1)
        
        return bev_image
    
    def apply_realsense_noise_reduction(self, bev_image):
        """Apply the same noise reduction as realsense pointcloud viewer"""
        # Convert to grayscale for binary thresholding
        binary_bev = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold (any non-zero pixel becomes 255)
        _, binary_bev = cv2.threshold(binary_bev, 1, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the BEV
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_bev = cv2.morphologyEx(binary_bev, cv2.MORPH_OPEN, kernel)
        binary_bev = cv2.morphologyEx(binary_bev, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR for consistency
        cleaned_bev = cv2.cvtColor(binary_bev, cv2.COLOR_GRAY2BGR)
        
        return cleaned_bev
    
    def apply_temporal_filtering(self, cleaned_bev):
        """Apply temporal filtering - only keep pixels that appear in ALL frames"""
        # Add current frame to history
        self.frame_history.append(cleaned_bev.copy())
        
        # Maintain window size
        window_size = self.clustering_params['window_size']
        if len(self.frame_history) > window_size:
            self.frame_history.pop(0)
        
        # If we don't have enough frames yet, return current frame
        if len(self.frame_history) < window_size:
            return cleaned_bev
        
        # Convert all frames to grayscale binary
        binary_frames = []
        for frame in self.frame_history:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            binary_frames.append(binary)
        
        # Stack frames and find pixels that are white (255) in ALL frames
        frame_stack = np.stack(binary_frames, axis=0)
        
        # Only keep pixels that are 255 in all frames
        persistent_pixels = np.all(frame_stack == 255, axis=0)
        
        # Create output image
        result = np.zeros_like(cleaned_bev)
        result[persistent_pixels] = 255
        
        return result
    
    def calculate_fps(self, current_time):
        """Calculate FPS from frame timestamps"""
        self.fps_history.append(current_time)
        
        # Keep only recent frames
        if len(self.fps_history) > self.fps_window_size:
            self.fps_history.pop(0)
        
        # Calculate FPS if we have enough samples
        if len(self.fps_history) >= 2:
            time_span = self.fps_history[-1] - self.fps_history[0]
            if time_span > 0:
                fps = (len(self.fps_history) - 1) / time_span
                return fps
        
        return 0.0
    
    def stabilize_cluster_positions(self, clusters):
        """Stabilize cluster positions using temporal averaging"""
        window_size = self.clustering_params['window_size']
        stabilized_clusters = []
        
        for cluster in clusters:
            cluster_id = cluster['id']
            current_center = cluster['center']
            
            # Initialize history for new clusters
            if cluster_id not in self.cluster_position_history:
                self.cluster_position_history[cluster_id] = []
            
            # Add current position to history
            self.cluster_position_history[cluster_id].append(current_center.copy())
            
            # Maintain window size
            if len(self.cluster_position_history[cluster_id]) > window_size:
                self.cluster_position_history[cluster_id].pop(0)
            
            # Calculate averaged position
            positions = self.cluster_position_history[cluster_id]
            avg_center = np.mean(positions, axis=0)
            
            # Create stabilized cluster
            stabilized_cluster = cluster.copy()
            stabilized_cluster['center'] = avg_center
            stabilized_cluster['center_mm'] = avg_center * 1000  # Convert to mm
            
            stabilized_clusters.append(stabilized_cluster)
        
        # Clean up old cluster histories (clusters that no longer exist)
        current_cluster_ids = {cluster['id'] for cluster in clusters}
        old_ids = set(self.cluster_position_history.keys()) - current_cluster_ids
        for old_id in old_ids:
            del self.cluster_position_history[old_id]
        
        return stabilized_clusters
    
    def get_2d_center_coordinates(self, clusters, x_range, z_range, img_width, img_height):
        """Get 2D center coordinates of bounding boxes in image space"""
        centers_2d = []
        
        for cluster in clusters:
            # Get 3D center
            center_3d = cluster['center']
            
            # Convert to 2D image coordinates
            center_x_2d = int((center_3d[0] - x_range[0]) / (x_range[1] - x_range[0]) * img_width)
            center_z_2d = int((center_3d[2] - z_range[0]) / (z_range[1] - z_range[0]) * img_height)
            
            # Flip Z coordinate for image display
            center_z_2d = img_height - center_z_2d
            
            # Ensure coordinates are within image bounds
            center_x_2d = max(0, min(img_width - 1, center_x_2d))
            center_z_2d = max(0, min(img_height - 1, center_z_2d))
            
            centers_2d.append({
                'cluster_id': cluster['id'],
                'center_2d': (center_x_2d, center_z_2d),
                'center_3d': center_3d,
                'dimensions': cluster['dimensions'],
                'num_points': cluster['num_points']
            })
        
        return centers_2d
    
    def crop_cluster_point_clouds(self, clusters, vertices, x_range, z_range, img_width, img_height):
        """Crop point clouds for each cluster using 2D cluster regions"""
        if not clusters:
            return {}
        
        # Get valid points
        valid_mask = vertices[:, 2] > 0
        valid_points = vertices[valid_mask]
        
        if len(valid_points) == 0:
            return {}
        
        # Calculate spans
        x_span = x_range[1] - x_range[0]
        z_span = z_range[1] - z_range[0]
        
        if x_span == 0 or z_span == 0:
            return {}
        
        # Map all 3D points to 2D image coordinates
        x_coords = valid_points[:, 0]
        z_coords = valid_points[:, 2]
        
        x_img = ((x_coords - x_range[0]) / x_span * img_width).astype(int)
        z_img = ((z_coords - z_range[0]) / z_span * img_height).astype(int)
        
        # Filter points within image bounds
        valid_img_mask = (x_img >= 0) & (x_img < img_width) & (z_img >= 0) & (z_img < img_height)
        x_img = x_img[valid_img_mask]
        z_img = z_img[valid_img_mask]
        valid_points_in_img = valid_points[valid_img_mask]
        
        # Create a mapping of each cluster to its point cloud
        cluster_point_clouds = {}
        
        for cluster in clusters:
            cluster_id = cluster['id']
            cluster_points = cluster['points']
            
            # For each cluster, find the corresponding 2D region and crop points
            # Use the cluster's bounding box to define the crop region
            min_coords = cluster['min_coords']
            max_coords = cluster['max_coords']
            
            # Find points within the cluster's 3D bounding box
            mask = ((cluster_points[:, 0] >= min_coords[0]) & 
                   (cluster_points[:, 0] <= max_coords[0]) &
                   (cluster_points[:, 1] >= min_coords[1]) & 
                   (cluster_points[:, 1] <= max_coords[1]) &
                   (cluster_points[:, 2] >= min_coords[2]) & 
                   (cluster_points[:, 2] <= max_coords[2]))
            
            cropped_points = cluster_points[mask]
            
            cluster_point_clouds[cluster_id] = {
                'points': cropped_points,
                'center_3d': cluster['center'],
                'dimensions': cluster['dimensions'],
                'num_points': len(cropped_points),
                'bounds': {
                    'min': min_coords,
                    'max': max_coords
                }
            }
        
        return cluster_point_clouds
    
    def cluster_from_cleaned_bev(self, cleaned_bev, vertices, x_range, z_range):
        """Perform clustering on the cleaned BEV image"""
        # Convert to grayscale for connected components
        gray = cv2.cvtColor(cleaned_bev, cv2.COLOR_BGR2GRAY)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(gray)
        
        # Filter out small and large components
        min_size = self.clustering_params['min_cluster_size']
        max_size = self.clustering_params['max_cluster_size']
        clusters = []
        
        for label in range(1, num_labels):  # Skip background (0)
            mask = labels == label
            component_size = np.sum(mask)
            
            if min_size <= component_size <= max_size:
                # Map image pixels back to 3D coordinates
                cluster_points = self.map_image_pixels_to_3d(mask, vertices, x_range, z_range)
                
                if len(cluster_points) > 0:
                    # Compute bounding box
                    min_coords = np.min(cluster_points, axis=0)
                    max_coords = np.max(cluster_points, axis=0)
                    center = (min_coords + max_coords) / 2
                    dimensions = max_coords - min_coords
                    
                    cluster_info = {
                        'id': label,
                        'points': cluster_points,
                        'center': center,
                        'min_coords': min_coords,
                        'max_coords': max_coords,
                        'dimensions': dimensions,
                        'num_points': len(cluster_points)
                    }
                    clusters.append(cluster_info)
        
        return clusters
    
    def apply_post_clustering_color_filter(self, clusters, vertices_full, color_image):
        """Apply RGB color filtering as post-processing after clustering to remove unrelated clusters"""
        if not self.color_filter_params['enabled'] or color_image is None:
            return clusters
        
        if not clusters:
            return clusters
        
        filtered_clusters = []
        height, width = color_image.shape[:2]
        
        # Debug: Check color image format once
        if not hasattr(self, '_color_format_checked'):
            sample_color = color_image[height//2, width//2]
            max_val = np.max(color_image)
            min_val = np.min(color_image)
            print(f"Color image format - Min: {min_val:.3f}, Max: {max_val:.3f}, Data type: {color_image.dtype}")
            print(f"Color image shape: {color_image.shape}")
            if max_val <= 1.0:
                print("Color image appears to be normalized (0-1 range)")
            elif max_val > 100:
                print("Color image appears to be in 0-255 range")
            else:
                print(f"Color image has limited range (0-{max_val:.0f}) - possibly low light conditions")
            self._color_format_checked = True
            
            # Store the actual range for normalization
            self._color_range = (min_val, max_val)
        
        # Get filter parameters
        r_min, r_max = self.color_filter_params['r_min'], self.color_filter_params['r_max']
        g_min, g_max = self.color_filter_params['g_min'], self.color_filter_params['g_max']
        b_min, b_max = self.color_filter_params['b_min'], self.color_filter_params['b_max']
        
        for cluster in clusters:
            cluster_points = cluster['points']
            center = cluster['center']
            
            # Sample colors from cluster region by projecting 3D cluster center to image
            # This is a simplified approach - sampling the center region of the cluster
            
            # Project cluster center to image coordinates
            # We need to reverse the camera projection to get pixel coordinates
            # For simplicity, we'll sample around the cluster center in 3D and project to 2D
            
            valid_color_count = 0
            total_samples = 0
            
            # Sample multiple points within the cluster for color analysis
            sample_points = []
            min_coords = cluster['min_coords'] 
            max_coords = cluster['max_coords']
            
            # Create a grid of sample points within the cluster bounding box
            for x in np.linspace(min_coords[0], max_coords[0], 3):
                for y in np.linspace(min_coords[1], max_coords[1], 2):
                    for z in np.linspace(min_coords[2], max_coords[2], 3):
                        sample_points.append([x, y, z])
            
            # For each sample point, estimate the color by projecting to image space
            for sample_point in sample_points:
                # Simple projection: use X and Z coordinates to estimate image position
                # This is approximate since we don't have the exact camera intrinsics here
                
                # Estimate pixel coordinates (very rough approximation)
                # Assuming a typical camera FOV and image center
                focal_length_approx = 500  # Approximate focal length in pixels
                cx, cy = width // 2, height // 2  # Image center
                
                # Project 3D point to 2D (simplified pinhole model)
                if sample_point[2] > 0:  # Valid depth
                    u = int(cx + (sample_point[0] / sample_point[2]) * focal_length_approx)
                    v = int(cy + (sample_point[1] / sample_point[2]) * focal_length_approx)
                    
                    # Check if projection is within image bounds
                    if 0 <= u < width and 0 <= v < height:
                        # Sample color at this pixel
                        color = color_image[v, u]  # BGR format
                        
                        # Normalize colors to 0-1 range using actual detected range
                        if hasattr(self, '_color_range'):
                            min_range, max_range = self._color_range
                            if max_range > min_range:
                                # Normalize to 0-1 using actual range
                                b = (color[0] - min_range) / (max_range - min_range)
                                g = (color[1] - min_range) / (max_range - min_range)
                                r = (color[2] - min_range) / (max_range - min_range)
                            else:
                                # Fallback if range is too small
                                b, g, r = 0.5, 0.5, 0.5
                        else:
                            # Fallback to standard 0-255 normalization
                            b, g, r = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
                        
                        total_samples += 1
                        
                        # Check if color passes the filter
                        if (r_min <= r <= r_max and 
                            g_min <= g <= g_max and 
                            b_min <= b <= b_max):
                            valid_color_count += 1
            
            # Decide whether to keep the cluster based on color filter results
            if total_samples > 0:
                color_ratio = valid_color_count / total_samples
                # Keep cluster if at least 30% of sampled points pass the color filter
                if color_ratio >= 0.01:
                    filtered_clusters.append(cluster)
                # else:
                #     print(f"Cluster {cluster['id']} removed by RGB filter (ratio: {color_ratio:.2f})")
            else:
                # If no valid samples, keep the cluster (fallback)
                filtered_clusters.append(cluster)
        
        return filtered_clusters
    
    def remove_statistical_outliers_from_clusters(self, clusters, std_threshold=1.5, min_points=5):
        """Fast statistical outlier removal: use cluster info → select 3D points → filter → validate"""
        if not clusters:
            return clusters
        
        cleaned_clusters = []
        
        for cluster in clusters:
            # Step 1: Get 3D points for this cluster (already done in clustering)
            cluster_points = cluster['points']
            
            if len(cluster_points) < min_points:
                continue  # Skip tiny clusters
            
            # Step 2: Statistical outlier removal on height (Y) coordinate
            y_coords = cluster_points[:, 1]
            y_mean = np.mean(y_coords)
            y_std = np.std(y_coords)
            
            if y_std > 0:  # Avoid division by zero
                # Keep points within std_threshold standard deviations
                z_scores = np.abs(y_coords - y_mean) / y_std
                inlier_mask = z_scores <= std_threshold
                filtered_points = cluster_points[inlier_mask]
            else:
                # All points at same height
                filtered_points = cluster_points
            
            # Step 3: Check minimum points requirement
            if len(filtered_points) >= min_points:
                # Recompute cluster properties with cleaned points
                min_coords = np.min(filtered_points, axis=0)
                max_coords = np.max(filtered_points, axis=0)
                center = (min_coords + max_coords) / 2
                dimensions = max_coords - min_coords
                
                # Create cleaned cluster
                cleaned_cluster = cluster.copy()
                cleaned_cluster['points'] = filtered_points
                cleaned_cluster['center'] = center
                cleaned_cluster['min_coords'] = min_coords
                cleaned_cluster['max_coords'] = max_coords
                cleaned_cluster['dimensions'] = dimensions
                cleaned_cluster['num_points'] = len(filtered_points)
                
                cleaned_clusters.append(cleaned_cluster)
        
        return cleaned_clusters
    
    def map_image_pixels_to_3d(self, pixel_mask, vertices, x_range, z_range):
        """Map BEV image pixels back to 3D points"""
        # Get valid points
        valid_mask = vertices[:, 2] > 0
        valid_points = vertices[valid_mask]
        
        if len(valid_points) == 0:
            return np.array([])
        
        # Get image dimensions
        img_height, img_width = pixel_mask.shape
        
        # Calculate spans
        x_span = x_range[1] - x_range[0]
        z_span = z_range[1] - z_range[0]
        
        if x_span == 0 or z_span == 0:
            return np.array([])
        
        # Map 3D points to image coordinates
        x_coords = valid_points[:, 0]
        z_coords = valid_points[:, 2]
        
        x_img = ((x_coords - x_range[0]) / x_span * img_width).astype(int)
        z_img = ((z_coords - z_range[0]) / z_span * img_height).astype(int)
        
        # Filter points within image bounds
        valid_img_mask = (x_img >= 0) & (x_img < img_width) & (z_img >= 0) & (z_img < img_height)
        x_img = x_img[valid_img_mask]
        z_img = z_img[valid_img_mask]
        valid_points_in_img = valid_points[valid_img_mask]
        
        # Find points that fall within the cluster mask
        cluster_points = []
        for i in range(len(x_img)):
            # Flip z coordinate for image indexing
            z_flipped = img_height - z_img[i] - 1
            if 0 <= z_flipped < img_height and pixel_mask[z_flipped, x_img[i]]:
                cluster_points.append(valid_points_in_img[i])
        
        return np.array(cluster_points) if cluster_points else np.array([])
    
    def draw_clusters_on_smooth_bev(self, bev_image, clusters, x_range, z_range):
        """Draw cluster bounding boxes on smooth BEV image"""
        if not clusters:
            return bev_image
        
        height, width = bev_image.shape[:2]
        result_image = bev_image.copy()
        
        # Generate colors for each cluster
        colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))[:, :3] * 255
        
        for i, cluster in enumerate(clusters):
            color = colors[i % len(colors)]
            
            # Project bounding box corners to image coordinates
            min_coords = cluster['min_coords']
            max_coords = cluster['max_coords']
            
            # Convert 3D coordinates to image coordinates
            x_img = int((min_coords[0] - x_range[0]) / (x_range[1] - x_range[0]) * width)
            x_img_max = int((max_coords[0] - x_range[0]) / (x_range[1] - x_range[0]) * width)
            z_img = int((min_coords[2] - z_range[0]) / (z_range[1] - z_range[0]) * height)
            z_img_max = int((max_coords[2] - z_range[0]) / (z_range[1] - z_range[0]) * height)
            
            # Flip z coordinates for image display
            z_img = height - z_img
            z_img_max = height - z_img_max
            
            # Ensure coordinates are within image bounds
            x_img = max(0, min(width-1, x_img))
            x_img_max = max(0, min(width-1, x_img_max))
            z_img = max(0, min(height-1, z_img))
            z_img_max = max(0, min(height-1, z_img_max))
            
            if self.show_bounding_boxes:
                # Draw bounding box rectangle
                cv2.rectangle(result_image, (x_img, z_img_max), (x_img_max, z_img), 
                            (int(color[2]), int(color[1]), int(color[0])), 2)
                
                # Add cluster information (commented out to reduce clutter)
                # info_text = f"ID:{cluster['id']} P:{cluster['num_points']}"
                # dims_text = f"({cluster['dimensions'][0]:.2f}x{cluster['dimensions'][2]:.2f}x{cluster['dimensions'][1]:.2f})"
                
                # cv2.putText(result_image, info_text, (x_img, z_img_max - 20), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (int(color[2]), int(color[1]), int(color[0])), 1)
                # cv2.putText(result_image, dims_text, (x_img, z_img_max - 5), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (int(color[2]), int(color[1]), int(color[0])), 1)
            
            if self.show_cluster_ids:
                # Draw 3D coordinates at center
                center_x = int((cluster['center'][0] - x_range[0]) / (x_range[1] - x_range[0]) * width)
                center_z = int((cluster['center'][2] - z_range[0]) / (z_range[1] - z_range[0]) * height)
                center_z = height - center_z  # Flip z coordinate
                
                center_x = max(0, min(width-1, center_x))
                center_z = max(0, min(height-1, center_z))
                
                # Draw white dot at center
                cv2.circle(result_image, (center_x, center_z), 3, (255, 255, 255), -1)
                
                # Display 3D coordinates in millimeters (depth, left-right, height)
                if 'center_mm' in cluster:
                    coord_text = f"({cluster['center_mm'][2]:.0f}, {cluster['center_mm'][0]:.0f}, {cluster['center_mm'][1]:.0f})"
                else:
                    coord_text = f"({cluster['center'][2]*1000:.0f}, {cluster['center'][0]*1000:.0f}, {cluster['center'][1]*1000:.0f})"
                cv2.putText(result_image, coord_text, (center_x + 5, center_z - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return result_image
    
    def project_to_2d_with_clustering(self, vertices, color_image):
        """Create BEV with clustering and object detection"""
        start_time = time.time()
        
        if vertices is None or len(vertices) == 0:
            return np.zeros((400, 400, 3), dtype=np.uint8), [], np.zeros((400, 400, 3), dtype=np.uint8)

        # Calculate ranges for consistent coordinate system
        valid_mask = vertices[:, 2] > 0
        valid_points = vertices[valid_mask]
        
        if len(valid_points) == 0:
            return np.zeros((400, 400, 3), dtype=np.uint8), [], np.zeros((400, 400, 3), dtype=np.uint8)

        x_coords = valid_points[:, 0]
        z_coords = valid_points[:, 2]
        
        actual_x_min, actual_x_max = np.min(x_coords), np.max(x_coords)
        actual_z_min, actual_z_max = np.min(z_coords), np.max(z_coords)
        
        # Use display ROI bounds for consistent BEV dimensions (separate from filter ROI)
        # Filter ROI is for point filtering, display ROI is for visualization area
        x_range = (self.display_roi_params['x_min'], self.display_roi_params['x_max'])
        z_range = (self.display_roi_params['z_min'], self.display_roi_params['z_max'])
        
        # Create smooth BEV image
        bev_image = self.create_smooth_bev_image(vertices, x_range, z_range)
        
        # Apply realsense pointcloud viewer's noise reduction
        cleaned_bev = self.apply_realsense_noise_reduction(bev_image)
        
        # Apply temporal filtering to reduce frame-to-frame noise
        filtered_bev = self.apply_temporal_filtering(cleaned_bev)
        
        # Perform clustering on the filtered BEV image
        clusters = self.cluster_from_cleaned_bev(filtered_bev, vertices, x_range, z_range)
        
        # Apply post-clustering RGB color filter to remove unrelated clusters
        color_filtered_clusters = self.apply_post_clustering_color_filter(clusters, vertices, color_image)
        
        # Stabilize cluster positions using temporal averaging
        stabilized_clusters = self.stabilize_cluster_positions(color_filtered_clusters)
        
        # Draw clusters on filtered BEV
        final_bev = self.draw_clusters_on_smooth_bev(filtered_bev, stabilized_clusters, x_range, z_range)
        
        # Create point cloud view of filtered vertices
        point_cloud_view = self.create_point_cloud_view(vertices)
        
        # Add clustering performance info
        processing_time = (time.time() - start_time) * 1000
        cv2.putText(final_bev, f'Clustering: {processing_time:.1f}ms', 
                   (10, final_bev.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(final_bev, f'Raw clusters: {len(clusters)}', 
                   (10, final_bev.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(final_bev, f'Filtered: {len(stabilized_clusters)}', 
                   (10, final_bev.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return final_bev, stabilized_clusters, point_cloud_view
    
    def print_cluster_info(self, clusters):
        """Print detailed information about detected clusters"""
        if not clusters:
            return
        
        print(f"\n=== Detected {len(clusters)} Objects ===")
        for cluster in clusters:
            print(f"Object {cluster['id']}:")
            print(f"  Points: {cluster['num_points']}")
            print(f"  Center: ({cluster['center'][0]:.3f}, {cluster['center'][1]:.3f}, {cluster['center'][2]:.3f})")
            print(f"  Dimensions (WxHxD): {cluster['dimensions'][0]:.3f} x {cluster['dimensions'][1]:.3f} x {cluster['dimensions'][2]:.3f}")
            print(f"  Volume: {np.prod(cluster['dimensions']):.6f} m³")
    
    def visualize(self):
        """Main visualization loop"""
        profile = self.start()
        frame_count = 0
        
        # Calibrate gravity alignment if using IMU
        if self.use_imu and not self.alignment_calibrated:
            print("Calibrating gravity alignment...")
            self.calibrate_gravity_alignment()
        
        try:
            while True:
                # Calculate FPS
                current_time = time.time()
                fps = self.calculate_fps(current_time)
                
                if self.bag_file:
                    frames_available, frames = self.pipeline.try_wait_for_frames(timeout_ms=100)
                    if not frames_available:
                        print(f"End of bag file. Processed {frame_count} frames.")
                        break
                else:
                    frames = self.pipeline.wait_for_frames()
                
                frame_count += 1
                
                # Get aligned 3D points
                vertices, texcoords, color_image = self.get_3d_points(frames)
                
                if vertices is not None and len(vertices) > 0:
                    # Create BEV with clustering
                    clustered_bev, clusters, point_cloud_view = self.project_to_2d_with_clustering(vertices, color_image)
                    
                    # Get 2D center coordinates for all clusters
                    if clusters:
                        bev_height, bev_width = clustered_bev.shape[:2]
                        # Calculate x_range and z_range (same as in project_to_2d_with_clustering)
                        valid_mask = vertices[:, 2] > 0
                        valid_points = vertices[valid_mask]
                        x_coords = valid_points[:, 0]
                        z_coords = valid_points[:, 2]
                        actual_x_min, actual_x_max = np.min(x_coords), np.max(x_coords)
                        actual_z_min, actual_z_max = np.min(z_coords), np.max(z_coords)
                        x_range = (min(self.roi_params['x_min'], actual_x_min - 0.1), 
                                   max(self.roi_params['x_max'], actual_x_max + 0.1))
                        z_range = (min(self.roi_params['z_min'], actual_z_min - 0.1), 
                                   max(self.roi_params['z_max'], actual_z_max + 0.1))
                        
                        # Crop point clouds for each cluster
                        cluster_point_clouds = self.crop_cluster_point_clouds(clusters, vertices, x_range, z_range, bev_width, bev_height)
                        
                        # Print cluster information every 30 frames
                        # if frame_count % 30 == 0 and cluster_point_clouds:
                        #     print(f"\n=== Cluster Positions (Frame {frame_count}) ===")
                        #     for cluster in clusters:
                        #         if 'center_mm' in cluster:
                        #             center_mm = cluster['center_mm']
                        #             print(f"Cluster {cluster['id']}: Position=({center_mm[2]:.0f}, {center_mm[0]:.0f}, {center_mm[1]:.0f}) mm")
                        #         else:
                        #             center = cluster['center']
                        #             print(f"Cluster {cluster['id']}: Position=({center[2]*1000:.0f}, {center[0]*1000:.0f}, {center[1]*1000:.0f}) mm")
                    
                    # Display point cloud statistics
                    if len(vertices) > 0:
                        min_dist = np.min(vertices[:, 2]) if len(vertices) > 0 else 0
                        max_dist = np.max(vertices[:, 2]) if len(vertices) > 0 else 0
                        avg_dist = np.mean(vertices[:, 2]) if len(vertices) > 0 else 0
                        
                        # Add stats to color image (or create dummy if no color)
                        if color_image is None:
                            color_image = np.zeros((480, 640, 3), dtype=np.uint8)
                        
                        # Add Y-axis debugging info
                        y_coords = vertices[:, 1]
                        y_min_actual, y_max_actual = np.min(y_coords), np.max(y_coords)
                        points_in_y_roi = np.sum((y_coords >= self.roi_params['y_min']) & 
                                                (y_coords <= self.roi_params['y_max']))
                        
                        stats_text = [
                            f'Frame: {frame_count}',
                            f'FPS: {fps:.1f}',
                            f'Points (ROI): {len(vertices)}',
                            f'Objects: {len(clusters)}',
                            f'Min dist: {min_dist:.2f}m',
                            f'Max dist: {max_dist:.2f}m',
                            f'Avg dist: {avg_dist:.2f}m',
                            f'Aligned: {"Yes" if self.alignment_calibrated else "No"}',
                            f'Y range: [{y_min_actual:.3f}, {y_max_actual:.3f}]',
                            f'Y ROI: [{self.roi_params["y_min"]:.2f}, {self.roi_params["y_max"]:.2f}]',
                            f'Y filtered: {points_in_y_roi}/{len(vertices)}'
                        ]
                        
                        for i, text in enumerate(stats_text):
                            cv2.putText(color_image, text, (10, 30 + i * 25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Stack images horizontally: Color | BEV | Point Cloud
                    if color_image is not None:
                        # Resize all images to same height
                        height = color_image.shape[0]
                        clustered_bev_resized = cv2.resize(clustered_bev, (height, height))
                        point_cloud_resized = cv2.resize(point_cloud_view, (height, height))
                        combined = np.hstack((color_image, clustered_bev_resized, point_cloud_resized))
                    else:
                        # No color image, just show BEV and point cloud
                        combined = np.hstack((clustered_bev, point_cloud_view))
                    
                    # Display
                    cv2.namedWindow('ROI Analysis: Color | BEV | Point Cloud', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('ROI Analysis: Color | BEV | Point Cloud', combined)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' ') and self.bag_file:  # Pause/resume for bag files
                    playback = profile.get_device().as_playback()
                    playback.pause()
                    print("Paused. Press space to resume.")
                    while cv2.waitKey(30) & 0xFF != ord(' '):
                        pass
                    playback.resume()
                    print("Resumed.")
                elif key == ord('r'):  # Reset/recalibrate alignment
                    if self.use_imu:
                        print("Recalibrating alignment...")
                        self.alignment_calibrated = False
                        self.calibrate_gravity_alignment()
                elif key == ord('t'):  # Toggle alignment
                    if self.use_imu:
                        self.alignment_calibrated = not self.alignment_calibrated
                        print(f"Alignment {'enabled' if self.alignment_calibrated else 'disabled'}")
                    else:
                        print("IMU not available - cannot toggle alignment")
                elif key == ord('b'):  # Toggle bounding boxes
                    self.show_bounding_boxes = not self.show_bounding_boxes
                    print(f"Bounding boxes {'enabled' if self.show_bounding_boxes else 'disabled'}")
                elif key == ord('c'):  # Toggle cluster IDs
                    self.show_cluster_ids = not self.show_cluster_ids
                    print(f"Cluster IDs {'enabled' if self.show_cluster_ids else 'disabled'}")
                elif key == ord('='):  # Increase min cluster size
                    self.clustering_params['min_cluster_size'] += 1
                    print(f"Min cluster size: {self.clustering_params['min_cluster_size']} pixels")
                elif key == ord('-'):  # Decrease min cluster size
                    self.clustering_params['min_cluster_size'] = max(1, self.clustering_params['min_cluster_size'] - 1)
                    print(f"Min cluster size: {self.clustering_params['min_cluster_size']} pixels")
                elif key == ord(']'):  # Increase max cluster size
                    self.clustering_params['max_cluster_size'] += 50
                    print(f"Max cluster size: {self.clustering_params['max_cluster_size']} pixels")
                elif key == ord('['):  # Decrease max cluster size
                    self.clustering_params['max_cluster_size'] = max(self.clustering_params['min_cluster_size'] + 10, 
                                                                     self.clustering_params['max_cluster_size'] - 50)
                    print(f"Max cluster size: {self.clustering_params['max_cluster_size']} pixels")
                elif key == ord('w'):  # Increase window size
                    self.clustering_params['window_size'] = min(20, self.clustering_params['window_size'] + 1)
                    print(f"Window size: {self.clustering_params['window_size']} frames")
                elif key == ord('s'):  # Decrease window size
                    self.clustering_params['window_size'] = max(1, self.clustering_params['window_size'] - 1)
                    # Clear frame history when window size changes
                    self.frame_history = []
                    print(f"Window size: {self.clustering_params['window_size']} frames")
                elif key == ord('f'):  # Toggle RGB color filter
                    self.color_filter_params['enabled'] = not self.color_filter_params['enabled']
                    print(f"RGB color filter {'enabled' if self.color_filter_params['enabled'] else 'disabled'}")
                elif key == ord('d'):  # Increase display ROI Z (depth)
                    self.display_roi_params['z_max'] += 0.1
                    print(f"Display ROI Z: [0, {self.display_roi_params['z_max']:.1f}]")
                elif key == ord('a'):  # Decrease display ROI Z (depth)
                    self.display_roi_params['z_max'] = max(0.5, self.display_roi_params['z_max'] - 0.1)
                    print(f"Display ROI Z: [0, {self.display_roi_params['z_max']:.1f}]")
                elif key == ord('x'):  # Increase display ROI X (side)
                    self.display_roi_params['x_max'] += 0.1
                    self.display_roi_params['x_min'] -= 0.1
                    print(f"Display ROI X: [{self.display_roi_params['x_min']:.1f}, {self.display_roi_params['x_max']:.1f}]")
                elif key == ord('z'):  # Decrease display ROI X (side)
                    if self.display_roi_params['x_max'] - self.display_roi_params['x_min'] > 0.5:
                        self.display_roi_params['x_max'] -= 0.1
                        self.display_roi_params['x_min'] += 0.1
                        print(f"Display ROI X: [{self.display_roi_params['x_min']:.1f}, {self.display_roi_params['x_max']:.1f}]")
                elif key == ord('1'):  # Preset: Yellow-Green filter
                    self.color_filter_params.update({
                        'r_min': 0.2, 'r_max': 1.0,   # Red: moderate to high
                        'g_min': 0.6, 'g_max': 1.0,   # Green: high
                        'b_min': 0.0, 'b_max': 0.4    # Blue: low
                    })
                    print("Color filter: Yellow-Green objects")
                elif key == ord('2'):  # Preset: Red filter
                    self.color_filter_params.update({
                        'r_min': 0.6, 'r_max': 1.0,   # Red: high
                        'g_min': 0.0, 'g_max': 0.4,   # Green: low
                        'b_min': 0.0, 'b_max': 0.4    # Blue: low
                    })
                    print("Color filter: Red objects")
                elif key == ord('3'):  # Preset: Blue filter
                    self.color_filter_params.update({
                        'r_min': 0.0, 'r_max': 0.4,   # Red: low
                        'g_min': 0.0, 'g_max': 0.6,   # Green: low to moderate
                        'b_min': 0.6, 'b_max': 1.0    # Blue: high
                    })
                    print("Color filter: Blue objects")
                elif key == ord('0'):  # Reset color filter (all colors)
                    self.color_filter_params.update({
                        'r_min': 0.0, 'r_max': 1.0,
                        'g_min': 0.0, 'g_max': 1.0,
                        'b_min': 0.0, 'b_max': 1.0
                    })
                    print("Color filter: All colors (disabled)")
                elif key == ord('y'):  # Increase Y ROI max (allow more points above floor)
                    self.roi_params['y_max'] += 0.02
                    print(f"Y ROI: [{self.roi_params['y_min']:.2f}, {self.roi_params['y_max']:.2f}] (height)")
                elif key == ord('h'):  # Decrease Y ROI max (filter points above floor)
                    self.roi_params['y_max'] = max(self.roi_params['y_min'] + 0.02, self.roi_params['y_max'] - 0.02)
                    print(f"Y ROI: [{self.roi_params['y_min']:.2f}, {self.roi_params['y_max']:.2f}] (height)")
                elif key == ord('u'):  # Increase Y ROI min (filter points below floor)
                    self.roi_params['y_min'] = min(self.roi_params['y_max'] - 0.02, self.roi_params['y_min'] + 0.02)
                    print(f"Y ROI: [{self.roi_params['y_min']:.2f}, {self.roi_params['y_max']:.2f}] (height)")
                elif key == ord('j'):  # Decrease Y ROI min (allow more points below floor)
                    self.roi_params['y_min'] -= 0.02
                    print(f"Y ROI: [{self.roi_params['y_min']:.2f}, {self.roi_params['y_max']:.2f}] (height)")
                elif key == ord('o'):  # Toggle 3D statistical outlier filtering
                    self.outlier_params['enabled'] = not self.outlier_params['enabled']
                    print(f"3D statistical outlier filtering {'enabled' if self.outlier_params['enabled'] else 'disabled'}")
                elif key == ord('p'):  # Increase outlier std ratio (more permissive)
                    self.outlier_params['std_ratio'] += 0.2
                    print(f"Outlier std ratio: {self.outlier_params['std_ratio']:.1f}")
                elif key == ord('l'):  # Decrease outlier std ratio (more strict)
                    self.outlier_params['std_ratio'] = max(0.5, self.outlier_params['std_ratio'] - 0.2)
                    print(f"Outlier std ratio: {self.outlier_params['std_ratio']:.1f}")
                elif key == ord('k'):  # Increase k_neighbors
                    self.outlier_params['k_neighbors'] = min(50, self.outlier_params['k_neighbors'] + 5)
                    print(f"K neighbors: {self.outlier_params['k_neighbors']}")
                elif key == ord('i'):  # Decrease k_neighbors
                    self.outlier_params['k_neighbors'] = max(5, self.outlier_params['k_neighbors'] - 5)
                    print(f"K neighbors: {self.outlier_params['k_neighbors']}")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Smooth BEV Point Cloud Clustering Visualizer')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to bag file (leave empty for live camera)')
    parser.add_argument('--realtime', '-rt', action='store_true',
                        help='Play bag file at real-time speed')
    parser.add_argument('--no-imu', action='store_true',
                        help='Disable IMU-based alignment')
    parser.add_argument('--min-cluster-size', type=int, default=25,
                        help='Minimum cluster size in pixels (default: 5)')
    parser.add_argument('--window-size', type=int, default=15,
                        help='Temporal window size for noise reduction (default: 5)')
    args = parser.parse_args()
    
    try:
        visualizer = BEVClusteringVisualizer(args.input, args.realtime, not args.no_imu)
        visualizer.clustering_params['min_cluster_size'] = args.min_cluster_size
        visualizer.clustering_params['window_size'] = args.window_size
        
        print("Smooth BEV Point Cloud Clustering Visualizer")
        print("Controls:")
        print("  'q' - Quit")
        print("  'space' - Pause/Resume (bag files only)")
        print("  'r' - Recalibrate alignment (if IMU available)")
        print("  't' - Toggle alignment on/off")
        print("  'b' - Toggle bounding boxes")
        print("  'c' - Toggle cluster IDs")
        print("  '='/'−' - Increase/Decrease minimum cluster size")
        print("  ']'/'[' - Increase/Decrease maximum cluster size")
        print("  'w'/'s' - Increase/Decrease temporal window size")
        print("  'f' - Toggle RGB color filter (post-clustering)")
        print("  '1' - Yellow-Green filter preset")
        print("  '2' - Red filter preset") 
        print("  '3' - Blue filter preset")
        print("  '0' - Reset color filter (all colors)")
        print("  'd'/'a' - Increase/Decrease display depth range")
        print("  'x'/'z' - Increase/Decrease display side range")
        print("  'y'/'h' - Increase/Decrease Y ROI max (height above floor)")
        print("  'u'/'j' - Increase/Decrease Y ROI min (height below floor)")
        print("  'o' - Toggle 3D statistical outlier filtering (remove isolated points)")
        print("  'p'/'l' - Increase/Decrease outlier std ratio (permissive/strict)")
        print("  'k'/'i' - Increase/Decrease k-neighbors for outlier detection")
        print("Features:")
        print("  • Smooth BEV projection with clustering")
        print("  • Real-time object detection and tracking")
        print("  • Axis-aligned 3D bounding boxes")
        print("  • Image-based clustering (no pillars)")
        print("Left: Color view with stats")
        print("Right: Smooth BEV clustered view with bounding boxes")
        visualizer.visualize()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()