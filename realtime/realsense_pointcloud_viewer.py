#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
from bag_inspector import inspect_bag_file

class PointCloudVisualizer:
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

                depth_sensor = profile.get_device().first_depth_sensor()
                if not self.bag_file:
                    depth_sensor.set_option(rs.option.visual_preset, int(rs.l500_visual_preset.short_range))
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
    
    def project_to_2d(self, vertices, color_image):
        """Project aligned 3D points to 2D for top-down view with proper aspect ratio"""
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
        
        # Calculate actual data bounds after ROI filtering
        actual_x_min, actual_x_max = np.min(x_coords), np.max(x_coords)
        actual_z_min, actual_z_max = np.min(z_coords), np.max(z_coords)
        
        # Use ROI bounds but expand to include actual data if needed
        x_range = (min(self.roi_params['x_min'], actual_x_min - 0.1), 
                   max(self.roi_params['x_max'], actual_x_max + 0.1))
        z_range = (min(self.roi_params['z_min'], actual_z_min - 0.1), 
                   max(self.roi_params['z_max'], actual_z_max + 0.1))
        
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
        top_down = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
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
            
            # Draw points with size based on density
            point_size = max(1, min(3, max_img_size // 200))
            
            for i in range(len(x_img)):
                cv2.circle(top_down, (x_img[i], img_height - z_img[i] - 1), point_size, 
                          (int(colors[i][2]), int(colors[i][1]), int(colors[i][0])), -1)
        
        # # Add grid and labels
        # self.add_grid_to_topdown(top_down, x_range, z_range)
        
        # # Add data statistics
        # self.add_data_info_to_topdown(top_down, len(y_valid), x_range, z_range, 
        #                              (actual_y_min, actual_y_max) if len(y_valid) > 0 else (0, 0))
        
        return top_down
    
    def add_grid_to_topdown(self, image, x_range, z_range):
        """Add grid lines and labels to top-down view with dynamic sizing"""
        height, width = image.shape[:2]
        
        # Calculate grid spacing based on range and image size
        x_span = x_range[1] - x_range[0]
        z_span = z_range[1] - z_range[0]
        
        # Adaptive grid spacing (aim for ~4-8 grid lines)
        x_step = self._calculate_grid_step(x_span)
        z_step = self._calculate_grid_step(z_span)
        
        # Draw vertical grid lines (X direction)
        x_start = int(np.ceil(x_range[0] / x_step)) * x_step
        x_pos = x_start
        while x_pos <= x_range[1]:
            if abs(x_pos) < 0.01:  # Near zero
                color = (0, 255, 0)  # Green for center line
                thickness = 2
            else:
                color = (100, 100, 100)  # Gray for other lines
                thickness = 1
            
            x_pixel = int((x_pos - x_range[0]) / x_span * width)
            if 0 <= x_pixel < width:
                cv2.line(image, (x_pixel, 0), (x_pixel, height), color, thickness)
                
                # Add label at bottom
                if x_pos != 0:  # Don't label center line
                    cv2.putText(image, f'{x_pos:.1f}', (x_pixel + 2, height - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            x_pos += x_step
        
        # Draw horizontal grid lines (Z direction)
        z_start = int(np.ceil(z_range[0] / z_step)) * z_step
        z_pos = z_start
        while z_pos <= z_range[1]:
            if abs(z_pos) < 0.01:  # Near zero
                color = (0, 255, 0)  # Green for center line
                thickness = 2
            else:
                color = (100, 100, 100)  # Gray for other lines
                thickness = 1
                
            z_pixel = int((z_pos - z_range[0]) / z_span * height)
            if 0 <= z_pixel < height:
                cv2.line(image, (0, height - z_pixel), (width, height - z_pixel), color, thickness)
                
                # Add label at left
                if z_pos != 0:  # Don't label center line
                    cv2.putText(image, f'{z_pos:.1f}', (2, height - z_pixel - 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            z_pos += z_step
    
    def _calculate_grid_step(self, span):
        """Calculate appropriate grid step size"""
        if span <= 0:
            return 1.0
        
        # Target 4-8 grid lines
        rough_step = span / 6
        
        # Round to nice numbers
        magnitude = 10 ** np.floor(np.log10(rough_step))
        normalized = rough_step / magnitude
        
        if normalized <= 1:
            nice_step = 1
        elif normalized <= 2:
            nice_step = 2
        elif normalized <= 5:
            nice_step = 5
        else:
            nice_step = 10
            
        return nice_step * magnitude
    
    def add_data_info_to_topdown(self, image, num_points, x_range, z_range, y_range):
        """Add data statistics to top-down view"""
        height, width = image.shape[:2]
        
        # Calculate text size based on image size
        font_scale = max(0.3, min(0.6, width / 600))
        
        info_lines = [
            f'Points: {num_points}',
            f'X: [{x_range[0]:.1f}, {x_range[1]:.1f}]m',
            f'Z: [{z_range[0]:.1f}, {z_range[1]:.1f}]m',
            f'Y: [{y_range[0]:.1f}, {y_range[1]:.1f}]m'
        ]
        
        # Add semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, len(info_lines) * 15 + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Add text
        for i, line in enumerate(info_lines):
            cv2.putText(image, line, (5, 15 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    
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
                    # Create top-down view
                    top_down = self.project_to_2d(vertices, color_image)
                    
                    binary_topdown = cv2.cvtColor(top_down, cv2.COLOR_BGR2GRAY)
                    _, binary_topdown = cv2.threshold(binary_topdown, 1, 255, cv2.THRESH_BINARY)
                    # Apply morphological operations to clean up the top-down view
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    top_down = cv2.morphologyEx(binary_topdown, cv2.MORPH_OPEN, kernel)
                    top_down = cv2.morphologyEx(top_down, cv2.MORPH_CLOSE, kernel)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                    top_down = cv2.cvtColor(top_down, cv2.COLOR_GRAY2BGR)
                    
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
                            f'Min dist: {min_dist:.2f}m',
                            f'Max dist: {max_dist:.2f}m',
                            f'Avg dist: {avg_dist:.2f}m',
                            f'Aligned: {"Yes" if self.alignment_calibrated else "No"}'
                        ]
                        
                        for i, text in enumerate(stats_text):
                            cv2.putText(color_image, text, (10, 30 + i * 25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Stack images
                    if color_image is not None:
                        combined = np.hstack((color_image, cv2.resize(top_down, (color_image.shape[1], color_image.shape[0]))))
                    else:
                        combined = top_down
                    
                    
                    
                    # Display
                    cv2.namedWindow('Aligned Point Cloud Visualization', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Aligned Point Cloud Visualization', combined)
                
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
                    self.roi_params['y_max'] += 0.05
                    print(f"Y ROI: [{self.roi_params['y_min']:.1f}, {self.roi_params['y_max']:.1f}]")
                elif key == ord('6'):  # Decrease Y ROI
                    if self.roi_params['y_max'] - self.roi_params['y_min'] > 0.5:
                        self.roi_params['y_max'] -= 0.05
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
    parser = argparse.ArgumentParser(description='Enhanced RealSense Point Cloud Visualizer with IMU Alignment')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to bag file (leave empty for live camera)')
    parser.add_argument('--realtime', '-rt', action='store_true',
                        help='Play bag file at real-time speed')
    parser.add_argument('--no-imu', action='store_true',
                        help='Disable IMU-based alignment')
    args = parser.parse_args()
    
    try:
        visualizer = PointCloudVisualizer(args.input, args.realtime, not args.no_imu)
        print("Enhanced Point Cloud Visualizer with IMU Alignment")
        print("Controls:")
        print("  'q' - Quit")
        print("  'space' - Pause/Resume (bag files only)")
        print("  'r' - Recalibrate alignment (if IMU available)")
        print("  't' - Toggle alignment on/off")
        print("  '1'/'2' - Expand/Contract X ROI (left-right)")
        print("  '3'/'4' - Expand/Contract Z ROI (forward-back)")
        print("  '5'/'6' - Expand/Contract Y ROI (up-down)")
        print("  '0' - Reset ROI to defaults")
        print("Features:")
        print("  • Gravity-based coordinate alignment using IMU")
        print("  • ROI filtering before 2D projection")
        print("  • Proper aspect ratio maintenance")
        print("  • Axis-aligned coordinate system")
        print("Left: Color view with stats")
        print("Right: Top-down aligned point cloud view")
        visualizer.visualize()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    # --input "F:\LiDAR Videos\objects.bag"