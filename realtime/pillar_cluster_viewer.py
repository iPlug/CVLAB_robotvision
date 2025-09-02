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
        
        # ROI parameters (in meters, after rotation)
        self.roi_params = {
            'x_min': -1.0, 'x_max': 1.0,
            'y_min': -.5, 'y_max': .1,  # Height range
            'z_min': 0, 'z_max': 1    # Depth range
        }
        
        # BEV clustering parameters
        self.clustering_params = {
            'pillar_size': 0.05,  # Size of each pillar in meters
            'min_points_per_pillar': 3,  # Minimum points to consider a pillar active
            'cluster_method': 'morphology',  # 'morphology' or 'dbscan'
            'min_cluster_size': 3,  # Minimum pixels for a valid cluster
            'morphology_kernel_size': 3,  # Kernel size for morphological operations
            'dbscan_eps': 2,  # DBSCAN epsilon in pixels
            'dbscan_min_samples': 10,  # DBSCAN minimum samples
            'height_threshold': 0.05,  # Minimum height difference for object detection
        }
        
        # Visualization parameters
        self.show_bounding_boxes = True
        self.show_cluster_ids = True
        
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
                
                if i % 20 == 0:
                    print(f"  Collected {len(accel_samples)}/{num_samples} samples...")
                    
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
        
        # Apply ROI filtering
        vertices = self.apply_roi_filter(vertices)
        
        # Get texture coordinates and color image
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
        
        return vertices, texcoords, color_image
    
    def _rotate_points(self, vertices):
        """Apply rotation matrix to align coordinate system"""
        if vertices is None or len(vertices) == 0:
            return vertices
        
        # Apply rotation to all points
        rotated_vertices = np.dot(vertices, self.rotation_matrix.T)
        return rotated_vertices
    
    def create_bev_pillars(self, vertices):
        """Create BEV representation using pillar clustering"""
        if vertices is None or len(vertices) == 0:
            return None, None, None, None
        
        # Filter out invalid points
        valid_mask = vertices[:, 2] > 0
        valid_points = vertices[valid_mask]
        
        if len(valid_points) == 0:
            return None, None, None, None
        
        # Use aligned coordinates for BEV with intuitive naming
        side_coords = valid_points[:, 0]    # Side (left-right)
        height_coords = valid_points[:, 1]  # Height (up-down)
        forward_coords = valid_points[:, 2] # Forward (depth)
        
        # Calculate actual data bounds
        actual_side_min, actual_side_max = np.min(side_coords), np.max(side_coords)
        actual_forward_min, actual_forward_max = np.min(forward_coords), np.max(forward_coords)
        
        # Use ROI bounds but expand to include actual data if needed
        side_range = (min(self.roi_params['x_min'], actual_side_min - 0.1), 
                      max(self.roi_params['x_max'], actual_side_max + 0.1))
        forward_range = (min(self.roi_params['z_min'], actual_forward_min - 0.1), 
                         max(self.roi_params['z_max'], actual_forward_max + 0.1))
        
        # Calculate pillar grid dimensions
        pillar_size = self.clustering_params['pillar_size']
        n_side_pillars = int(np.ceil((side_range[1] - side_range[0]) / pillar_size))
        n_forward_pillars = int(np.ceil((forward_range[1] - forward_range[0]) / pillar_size))
        
        # Create pillar grid (forward x side)
        pillar_grid = np.zeros((n_forward_pillars, n_side_pillars), dtype=np.int32)
        pillar_heights = np.full((n_forward_pillars, n_side_pillars), -np.inf)
        pillar_min_heights = np.full((n_forward_pillars, n_side_pillars), np.inf)
        
        # Assign points to pillars
        side_indices = np.clip(((side_coords - side_range[0]) / pillar_size).astype(int), 0, n_side_pillars - 1)
        forward_indices = np.clip(((forward_coords - forward_range[0]) / pillar_size).astype(int), 0, n_forward_pillars - 1)
        
        # Count points per pillar and track height statistics
        for i in range(len(valid_points)):
            side_idx, forward_idx = side_indices[i], forward_indices[i]
            pillar_grid[forward_idx, side_idx] += 1
            pillar_heights[forward_idx, side_idx] = max(pillar_heights[forward_idx, side_idx], height_coords[i])
            pillar_min_heights[forward_idx, side_idx] = min(pillar_min_heights[forward_idx, side_idx], height_coords[i])
        
        # Create occupancy mask based on minimum points per pillar and height difference
        min_points = self.clustering_params['min_points_per_pillar']
        height_threshold = self.clustering_params['height_threshold']
        
        occupancy_mask = (pillar_grid >= min_points)
        
        # Additional filtering based on height difference (detect objects above ground)
        height_diff = pillar_heights - pillar_min_heights
        height_diff[pillar_heights == -np.inf] = 0
        height_mask = height_diff >= height_threshold
        
        final_occupancy = occupancy_mask & height_mask
        
        # Debug output
        print(f"DEBUG: Total pillars: {pillar_grid.shape[0]}x{pillar_grid.shape[1]}")
        print(f"DEBUG: Pillars with ≥{min_points} points: {np.sum(occupancy_mask)}")
        print(f"DEBUG: Pillars with ≥{height_threshold}m height diff: {np.sum(height_mask)}")
        print(f"DEBUG: Final occupied pillars: {np.sum(final_occupancy)}")
        if np.sum(final_occupancy) > 0:
            print(f"DEBUG: Height diff range: {np.min(height_diff[height_diff > 0]):.3f} to {np.max(height_diff):.3f}")
        print(f"DEBUG: Point count range: {np.min(pillar_grid)} to {np.max(pillar_grid)}")
        
        return final_occupancy, pillar_grid, (side_range, forward_range), (side_indices, forward_indices, valid_points)
    
    def cluster_bev_objects(self, occupancy_mask):
        """Cluster objects in BEV using morphological operations or DBSCAN"""
        if occupancy_mask is None:
            return None
        
        method = self.clustering_params['cluster_method']
        
        if method == 'morphology':
            return self._cluster_morphology(occupancy_mask)
        elif method == 'dbscan':
            return self._cluster_dbscan(occupancy_mask)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def _cluster_morphology(self, occupancy_mask):
        """Cluster using morphological operations (faster)"""
        # Convert to uint8 for OpenCV
        mask_uint8 = (occupancy_mask * 255).astype(np.uint8)
        
        # Apply morphological operations to connect nearby pixels
        kernel_size = self.clustering_params['morphology_kernel_size']
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Close small gaps
        closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(closed)
        
        # Filter out small clusters
        min_size = self.clustering_params['min_cluster_size']
        filtered_labels = np.zeros_like(labels)
        cluster_id = 1
        
        for label in range(1, num_labels):  # Skip background (0)
            mask = labels == label
            component_size = np.sum(mask)
            print(f"DEBUG: Component {label} has {component_size} pixels (min_size={min_size})")
            if component_size >= min_size:
                filtered_labels[mask] = cluster_id
                cluster_id += 1
        
        print(f"DEBUG: Morphology found {num_labels-1} components, {cluster_id-1} clusters after size filtering")
        return filtered_labels
    
    def _cluster_dbscan(self, occupancy_mask):
        """Cluster using DBSCAN on pillar centers"""
        # Get coordinates of occupied pillars
        occupied_coords = np.argwhere(occupancy_mask)
        
        if len(occupied_coords) == 0:
            return np.zeros_like(occupancy_mask, dtype=np.int32)
        
        # Apply DBSCAN clustering
        eps = self.clustering_params['dbscan_eps']
        min_samples = self.clustering_params['dbscan_min_samples']
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(occupied_coords)
        
        # Create cluster mask
        cluster_mask = np.zeros_like(occupancy_mask, dtype=np.int32)
        
        for i, (z, x) in enumerate(occupied_coords):
            if cluster_labels[i] != -1:  # Not noise
                cluster_mask[z, x] = cluster_labels[i] + 1  # +1 to avoid 0 (background)
        
        return cluster_mask
    
    def extract_cluster_3d_points(self, cluster_mask, point_data):
        """Extract 3D points for each cluster and compute bounding boxes"""
        if cluster_mask is None or point_data is None:
            return []
        
        x_indices, z_indices, valid_points = point_data
        clusters = []
        
        # Get unique cluster IDs (excluding background 0)
        cluster_ids = np.unique(cluster_mask)
        cluster_ids = cluster_ids[cluster_ids > 0]
        
        for cluster_id in cluster_ids:
            # Find points belonging to this cluster
            cluster_pillar_mask = cluster_mask == cluster_id
            cluster_coords = np.argwhere(cluster_pillar_mask)
            
            # Find all 3D points that belong to these pillars
            point_mask = np.zeros(len(valid_points), dtype=bool)
            
            for z_idx, x_idx in cluster_coords:
                pillar_point_mask = (z_indices == z_idx) & (x_indices == x_idx)
                point_mask |= pillar_point_mask
            
            cluster_points = valid_points[point_mask]
            
            if len(cluster_points) > 0:
                # Compute axis-aligned bounding box
                min_coords = np.min(cluster_points, axis=0)
                max_coords = np.max(cluster_points, axis=0)
                
                # Calculate bounding box center and dimensions
                center = (min_coords + max_coords) / 2
                dimensions = max_coords - min_coords
                
                cluster_info = {
                    'id': int(cluster_id),
                    'points': cluster_points,
                    'center': center,
                    'min_coords': min_coords,
                    'max_coords': max_coords,
                    'dimensions': dimensions,
                    'num_points': len(cluster_points)
                }
                
                clusters.append(cluster_info)
        
        return clusters
    
    def visualize_clusters_on_bev(self, bev_image, cluster_mask, clusters):
        """Visualize clusters and bounding boxes on BEV image"""
        if cluster_mask is None:
            return bev_image
        
        # Create colored cluster visualization
        colored_bev = bev_image.copy()
        
        # Get unique cluster IDs
        cluster_ids = np.unique(cluster_mask)
        cluster_ids = cluster_ids[cluster_ids > 0]
        
        if len(cluster_ids) == 0:
            return colored_bev
        
        # Generate colors for each cluster
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))[:, :3] * 255
        
        # Get BEV image dimensions
        bev_height, bev_width = bev_image.shape[:2]
        mask_height, mask_width = cluster_mask.shape
        
        # Scale cluster mask to match BEV image dimensions
        scale_y = bev_height / mask_height
        scale_x = bev_width / mask_width
        
        for i, cluster_id in enumerate(cluster_ids):
            cluster_pixels = cluster_mask == cluster_id
            color = colors[i % len(colors)]
            
            # Find cluster pixel coordinates in mask space
            cluster_coords = np.argwhere(cluster_pixels)
            
            if len(cluster_coords) > 0:
                # Convert to image coordinates (flip Y-axis)
                for coord in cluster_coords:
                    y_img = int((mask_height - 1 - coord[0]) * scale_y)  # Flip Y-axis
                    x_img = int(coord[1] * scale_x)
                    
                    # Draw filled circle to represent pillar
                    radius = max(1, int(min(scale_x, scale_y) / 2))
                    cv2.circle(colored_bev, (x_img, y_img), radius, 
                             (int(color[2]), int(color[1]), int(color[0])), -1)
                
                if self.show_cluster_ids:
                    # Find cluster centroid for label placement
                    centroid = np.mean(cluster_coords, axis=0).astype(int)
                    centroid_y = int((mask_height - 1 - centroid[0]) * scale_y)  # Flip Y-axis
                    centroid_x = int(centroid[1] * scale_x)
                    
                    cv2.putText(colored_bev, str(cluster_id), 
                               (centroid_x, centroid_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return colored_bev
    
    def draw_3d_bounding_boxes_on_image(self, image, clusters, x_range, z_range):
        """Draw 3D bounding box projections on the BEV image"""
        if not self.show_bounding_boxes or not clusters:
            return image
        
        height, width = image.shape[:2]
        
        for cluster in clusters:
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
            
            # Draw bounding box rectangle
            cv2.rectangle(image, (x_img, z_img_max), (x_img_max, z_img), (0, 255, 0), 2)
            
            # Add cluster information
            info_text = f"ID:{cluster['id']} P:{cluster['num_points']}"
            dims_text = f"({cluster['dimensions'][0]:.2f}x{cluster['dimensions'][2]:.2f}x{cluster['dimensions'][1]:.2f})"
            
            cv2.putText(image, info_text, (x_img, z_img_max - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(image, dims_text, (x_img, z_img_max - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        return image
    
    def project_to_2d_with_clustering(self, vertices, color_image):
        """Create BEV with clustering and object detection"""
        start_time = time.time()
        
        # Create BEV pillars
        occupancy_mask, pillar_grid, ranges, point_data = self.create_bev_pillars(vertices)
        
        if occupancy_mask is None:
            return np.zeros((400, 400, 3), dtype=np.uint8), []
        
        x_range, z_range = ranges
        
        # Cluster objects in BEV
        cluster_mask = self.cluster_bev_objects(occupancy_mask)
        
        # Extract 3D points and bounding boxes for each cluster
        clusters = self.extract_cluster_3d_points(cluster_mask, point_data)
        
        # Create base BEV image (similar to original but with clustering)
        bev_image = self.create_base_bev_image(vertices, x_range, z_range)
        
        # Visualize clusters
        clustered_bev = self.visualize_clusters_on_bev(bev_image, cluster_mask, clusters)
        
        # Draw bounding boxes
        final_bev = self.draw_3d_bounding_boxes_on_image(clustered_bev, clusters, x_range, z_range)
        
        # Add clustering performance info
        processing_time = (time.time() - start_time) * 1000
        cv2.putText(final_bev, f'Clustering: {processing_time:.1f}ms', 
                   (10, final_bev.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(final_bev, f'Objects: {len(clusters)}', 
                   (10, final_bev.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return final_bev, clusters
    
    def create_base_bev_image(self, vertices, x_range, z_range):
        """Create base BEV image similar to original method"""
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
                    clustered_bev, clusters = self.project_to_2d_with_clustering(vertices, color_image)
                    
                    # Print cluster information (every 30 frames to avoid spam)
                    if frame_count % 30 == 0 and clusters:
                        self.print_cluster_info(clusters)
                    
                    # Display point cloud statistics
                    if len(vertices) > 0:
                        min_dist = np.min(vertices[:, 2]) if len(vertices) > 0 else 0
                        max_dist = np.max(vertices[:, 2]) if len(vertices) > 0 else 0
                        avg_dist = np.mean(vertices[:, 2]) if len(vertices) > 0 else 0
                        
                        # Add stats to color image (or create dummy if no color)
                        if color_image is None:
                            color_image = np.zeros((480, 640, 3), dtype=np.uint8)
                        
                        stats_text = [
                            f'Frame: {frame_count}',
                            f'Points (ROI): {len(vertices)}',
                            f'Objects: {len(clusters)}',
                            f'Min dist: {min_dist:.2f}m',
                            f'Max dist: {max_dist:.2f}m',
                            f'Avg dist: {avg_dist:.2f}m',
                            f'Aligned: {"Yes" if self.alignment_calibrated else "No"}',
                            f'Cluster: {self.clustering_params["cluster_method"]}'
                        ]
                        
                        for i, text in enumerate(stats_text):
                            cv2.putText(color_image, text, (10, 30 + i * 25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Stack images
                    if color_image is not None:
                        combined = np.hstack((color_image, cv2.resize(clustered_bev, (color_image.shape[1], color_image.shape[0]))))
                    else:
                        combined = clustered_bev
                    
                    # Display
                    cv2.namedWindow('BEV Clustering Visualization', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('BEV Clustering Visualization', combined)
                
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
                elif key == ord('m'):  # Toggle clustering method
                    if self.clustering_params['cluster_method'] == 'morphology':
                        self.clustering_params['cluster_method'] = 'dbscan'
                    else:
                        self.clustering_params['cluster_method'] = 'morphology'
                    print(f"Clustering method: {self.clustering_params['cluster_method']}")
                elif key == ord('p'):  # Increase pillar size
                    self.clustering_params['pillar_size'] = min(0.2, self.clustering_params['pillar_size'] + 0.01)
                    print(f"Pillar size: {self.clustering_params['pillar_size']:.3f}m")
                elif key == ord('o'):  # Decrease pillar size
                    self.clustering_params['pillar_size'] = max(0.02, self.clustering_params['pillar_size'] - 0.01)
                    print(f"Pillar size: {self.clustering_params['pillar_size']:.3f}m")
                elif key == ord('1'):  # Increase X ROI
                    self.roi_params['x_max'] += 0.5
                    self.roi_params['x_min'] -= 0.5
                    print(f"X ROI: [{self.roi_params['x_min']:.1f}, {self.roi_params['x_max']:.1f}]")
                elif key == ord('2'):  # Decrease X ROI
                    if self.roi_params['x_max'] - self.roi_params['x_min'] > 1.0:
                        self.roi_params['x_max'] -= 0.5
                        self.roi_params['x_min'] += 0.5
                        print(f"X ROI: [{self.roi_params['x_min']:.1f}, {self.roi_params['x_max']:.1f}]")
                elif key == ord('3'):  # Increase Z ROI
                    self.roi_params['z_max'] += 0.5
                    print(f"Z ROI: [{self.roi_params['z_min']:.1f}, {self.roi_params['z_max']:.1f}]")
                elif key == ord('4'):  # Decrease Z ROI
                    if self.roi_params['z_max'] - self.roi_params['z_min'] > 1.0:
                        self.roi_params['z_max'] -= 0.5
                        print(f"Z ROI: [{self.roi_params['z_min']:.1f}, {self.roi_params['z_max']:.1f}]")
                elif key == ord('5'):  # Increase Y ROI
                    self.roi_params['y_max'] += 0.01
                    print(f"Y ROI: [{self.roi_params['y_min']:.1f}, {self.roi_params['y_max']:.1f}]")
                elif key == ord('6'):  # Decrease Y ROI
                    if self.roi_params['y_max'] - self.roi_params['y_min'] > 0.5:
                        self.roi_params['y_max'] -= 0.01
                        print(f"Y ROI: [{self.roi_params['y_min']:.1f}, {self.roi_params['y_max']:.1f}]")
                elif key == ord('0'):  # Reset ROI to defaults
                    self.roi_params = {
                        'x_min': -2.0, 'x_max': 2.0,
                        'y_min': -0.5, 'y_max': 2.0,
                        'z_min': 0.0, 'z_max': 4.0
                    }
                    print("ROI reset to defaults")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='BEV Point Cloud Clustering Visualizer')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to bag file (leave empty for live camera)')
    parser.add_argument('--realtime', '-rt', action='store_true',
                        help='Play bag file at real-time speed')
    parser.add_argument('--no-imu', action='store_true',
                        help='Disable IMU-based alignment')
    parser.add_argument('--pillar-size', type=float, default=0.05,
                        help='Pillar size for clustering (default: 0.05m)')
    parser.add_argument('--cluster-method', choices=['morphology', 'dbscan'], default='morphology',
                        help='Clustering method (default: morphology)')
    args = parser.parse_args()
    
    try:
        visualizer = BEVClusteringVisualizer(args.input, args.realtime, not args.no_imu)
        visualizer.clustering_params['pillar_size'] = args.pillar_size
        visualizer.clustering_params['cluster_method'] = args.cluster_method
        
        print("BEV Point Cloud Clustering Visualizer")
        print("Controls:")
        print("  'q' - Quit")
        print("  'space' - Pause/Resume (bag files only)")
        print("  'r' - Recalibrate alignment (if IMU available)")
        print("  't' - Toggle alignment on/off")
        print("  'b' - Toggle bounding boxes")
        print("  'c' - Toggle cluster IDs")
        print("  'm' - Toggle clustering method (morphology/dbscan)")
        print("  'p'/'o' - Increase/Decrease pillar size")
        print("  '1'/'2' - Expand/Contract X ROI")
        print("  '3'/'4' - Expand/Contract Z ROI")
        print("  '5'/'6' - Expand/Contract Y ROI")
        print("  '0' - Reset ROI to defaults")
        print("Features:")
        print("  • Fast BEV-based pillar clustering")
        print("  • Real-time object detection and tracking")
        print("  • Axis-aligned 3D bounding boxes")
        print("  • Multiple clustering algorithms")
        print("Left: Color view with stats")
        print("Right: BEV clustered view with bounding boxes")
        visualizer.visualize()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()