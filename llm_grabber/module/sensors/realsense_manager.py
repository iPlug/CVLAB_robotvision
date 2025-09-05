"""
RealSense camera sensor management extracted from realsense_bev_cluster.py
"""

import pyrealsense2 as rs
import numpy as np
import os
from typing import Tuple, Optional, Dict, Any, List

from .sensor_manager import SensorManager


class RealSenseManager(SensorManager):
    """
    RealSense camera sensor manager for real-time and bag file processing.
    
    Supports:
    - Live camera streaming
    - Bag file playback  
    - IMU data for gravity alignment
    - Depth and color stream configuration
    """
    
    def __init__(self, bag_file: Optional[str] = None, 
                 realtime_mode: bool = False,
                 use_imu: bool = True,
                 external_intrinsics: Optional[np.ndarray] = None):
        """
        Initialize RealSense manager.
        
        Args:
            bag_file: Path to bag file for playback (None for live camera)
            realtime_mode: Whether to play bag files at real-time speed
            use_imu: Whether to enable IMU streams for alignment
            external_intrinsics: External camera intrinsics matrix (3x3) for CharUco calibration
        """
        super().__init__()
        
        self.bag_file = bag_file
        self.realtime_mode = realtime_mode
        self.use_imu = use_imu
        self.external_intrinsics = external_intrinsics
        
        # RealSense objects
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.pc = rs.pointcloud()
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2)
        
        # Profile for accessing device information
        self.profile = None
        
        # Gravity alignment (from original BEVClusteringVisualizer)
        self.rotation_matrix = np.eye(3)
        self.gravity_vector = np.array([0, -1, 0])
        self.alignment_calibrated = False
        
        # Store raw vertices for visualization
        self.last_raw_vertices = None
    
    def initialize(self, **kwargs) -> bool:
        """
        Initialize RealSense sensor.
        
        Kwargs:
            depth_width: Depth stream width (default: 640)
            depth_height: Depth stream height (default: 480)
            color_width: Color stream width (default: 640)
            color_height: Color stream height (default: 480)
            fps: Stream FPS (default: 30)
            
        Returns:
            True if initialization successful
        """
        try:
            # Get parameters
            depth_width = kwargs.get('depth_width', 640)
            depth_height = kwargs.get('depth_height', 480)
            color_width = kwargs.get('color_width', 640)
            color_height = kwargs.get('color_height', 480)
            fps = kwargs.get('fps', 30)
            
            if self.bag_file:
                # Configure for bag file playback
                if not os.path.exists(self.bag_file):
                    print(f"Error: Bag file '{self.bag_file}' not found!")
                    return False
                
                rs.config.enable_device_from_file(self.config, self.bag_file)
                
                # Try to get stream information from bag file
                available_streams = self._inspect_bag_file()
                if available_streams:
                    self._enable_streams_from_bag(available_streams)
                else:
                    # Fallback to automatic configuration
                    print("Warning: Could not inspect bag file, using automatic configuration")
            else:
                # Configure for live camera
                self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
                self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
                
                # Enable IMU streams if available
                if self.use_imu:
                    try:
                        self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
                        self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
                        print("IMU streams enabled for gravity-based alignment")
                    except:
                        print("IMU not available, using default orientation")
                        self.use_imu = False
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing RealSense: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start RealSense pipeline.
        
        Returns:
            True if started successfully
        """
        if not self.is_initialized:
            print("Error: RealSense not initialized")
            return False
        
        try:
            self.profile = self.pipeline.start(self.config)
            
            if self.bag_file:
                playback = self.profile.get_device().as_playbook()
                playback.set_real_time(self.realtime_mode)
                print(f"Playing bag file: {self.bag_file}")
                if not self.realtime_mode:
                    print("Non-real-time mode: Processing all frames")
            else:
                print("Live camera mode")
            
            self.is_running = True
            return True
            
        except Exception as e:
            print(f"Error starting RealSense pipeline: {e}")
            
            # Try fallback configuration
            try:
                print("Trying with automatic configuration...")
                self.config = rs.config()
                if self.bag_file:
                    rs.config.enable_device_from_file(self.config, self.bag_file)
                else:
                    self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                    self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                
                self.profile = self.pipeline.start(self.config)
                print("Successfully started with automatic configuration")
                self.is_running = True
                return True
                
            except Exception as e2:
                print(f"Failed with automatic configuration too: {e2}")
                return False
    
    def stop(self):
        """Stop RealSense pipeline."""
        if self.is_running:
            try:
                self.pipeline.stop()
                self.is_running = False
                print("RealSense pipeline stopped")
            except Exception as e:
                print(f"Error stopping RealSense pipeline: {e}")
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Get next frame from RealSense.
        
        Returns:
            Tuple of (vertices, color_image, metadata)
        """
        if not self.is_running:
            return None, None, {'error': 'Sensor not running'}
        
        try:
            # Get frames
            if self.bag_file:
                frames_available, frames = self.pipeline.try_wait_for_frames(timeout_ms=100)
                if not frames_available:
                    return None, None, {'end_of_stream': True}
            else:
                frames = self.pipeline.wait_for_frames()
            
            self.frame_count += 1
            
            # Extract 3D points and color image
            vertices, color_image = self._extract_3d_points(frames)
            
            # Create metadata
            metadata = {
                'frame_id': self.frame_count,
                'timestamp': frames.get_timestamp(),
                'has_depth': frames.get_depth_frame() is not None,
                'has_color': frames.get_color_frame() is not None,
                'alignment_calibrated': self.alignment_calibrated
            }
            
            # Add IMU data if available
            if self.use_imu:
                accel_frame = frames.first_or_default(rs.stream.accel)
                if accel_frame:
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    metadata['imu_accel'] = [accel_data.x, accel_data.y, accel_data.z]
            
            return vertices, color_image, metadata
            
        except Exception as e:
            return None, None, {'error': str(e)}
    
    def _extract_3d_points(self, frames) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract 3D points and color image from frames"""
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame:
            return None, None
        
        # Apply decimation filter to reduce number of points
        depth_frame = self.decimate.process(depth_frame)
        
        # Generate point cloud
        if self.external_intrinsics is not None:
            # Use external intrinsics for point cloud generation
            vertices = self._calculate_vertices_with_external_intrinsics(depth_frame)
        else:
            # Use default RealSense intrinsics
            if color_frame:
                self.pc.map_to(color_frame)
            points = self.pc.calculate(depth_frame)
            
            # Get vertices
            v = points.get_vertices()
            vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        
        # Apply depth filtering (0.25m to 0.8m from camera - LiDAR effective range)
        if vertices is not None and len(vertices) > 0:
            # Filter based on Z coordinate (depth from camera)
            depth_mask = (vertices[:, 2] >= 0.25) & (vertices[:, 2] <= 0.8)
            vertices = vertices[depth_mask]

        # Store raw vertices before any processing (for visualization)
        self.last_raw_vertices = vertices.copy() if vertices is not None else None
        
        # Apply gravity-based alignment if calibrated
        if self.alignment_calibrated:
            vertices = self._apply_alignment(vertices)
        
        
        
        # Get color image
        color_image = None
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            
            # Handle different color formats
            color_profile = color_frame.get_profile().as_video_stream_profile()
            if color_profile.format() == rs.format.rgb8:
                # Convert RGB to BGR for OpenCV
                import cv2
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
        return vertices, color_image
    
    def get_raw_vertices(self) -> Optional[np.ndarray]:
        """
        Get the last captured raw vertices before any alignment or filtering.
        
        Returns:
            Raw vertices from the RealSense sensor before gravity alignment,
            or None if no vertices have been captured yet.
        """
        return self.last_raw_vertices
    
    def _apply_alignment(self, vertices: np.ndarray) -> np.ndarray:
        """Apply gravity-based alignment to vertices"""
        if vertices is None or len(vertices) == 0:
            return vertices
        
        # Apply rotation to all points
        rotated_vertices = np.dot(vertices, self.rotation_matrix.T)
        return rotated_vertices
    
    def get_inverse_rotation_matrix(self) -> np.ndarray:
        """
        Get the inverse rotation matrix to transform from world-aligned back to raw camera coordinates.
        
        Returns:
            3x3 inverse rotation matrix (transpose of rotation_matrix since it's orthogonal)
        """
        return self.rotation_matrix.T
    
    def transform_to_raw_camera_frame(self, world_aligned_points: np.ndarray) -> np.ndarray:
        """
        Transform points from world-aligned coordinates back to raw camera coordinates.
        
        Args:
            world_aligned_points: Points in world-aligned coordinate system
            
        Returns:
            Points in raw camera coordinate system
        """
        if world_aligned_points is None or len(world_aligned_points) == 0:
            return world_aligned_points
            
        # Apply inverse rotation to transform back to camera frame
        return np.dot(world_aligned_points, self.rotation_matrix)  # Transpose of transpose = original

    def calibrate_gravity_alignment(self, num_samples: int = 15) -> bool:
        """
        Calibrate gravity direction from IMU data.
        
        Args:
            num_samples: Number of samples to collect
            
        Returns:
            True if calibration successful
        """
        if not self.use_imu or not self.is_running:
            return False
        
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
                    
            except Exception as e:
                print(f"Error collecting sample {i}: {e}")
                continue
        
        if len(accel_samples) > 10:  # Need at least some samples
            # Average accelerometer readings to get gravity vector
            accel_avg = np.mean(accel_samples, axis=0)
            gravity_magnitude = np.linalg.norm(accel_avg)
            
            if gravity_magnitude > 0.5:  # Reasonable gravity reading
                # Normalize gravity vector
                self.gravity_vector = accel_avg / gravity_magnitude
                
                # Calculate rotation matrix to align gravity with -Y axis
                target_up = np.array([0, -1, 0])
                self.rotation_matrix = self._calculate_alignment_rotation(self.gravity_vector, target_up)
                
                print(f"Gravity vector detected: {self.gravity_vector}")
                print(f"Gravity magnitude: {gravity_magnitude:.2f} m/s^2")
                print(f"Alignment calibrated successfully!")
                self.alignment_calibrated = True
                return True
            else:
                print(f"Could not detect reliable gravity vector (magnitude: {gravity_magnitude:.2f})")
        else:
            print(f"Insufficient IMU data for calibration (only {len(accel_samples)} samples)")
        
        return False
    
    def get_current_gravity_vector(self) -> Optional[np.ndarray]:
        """
        Get current gravity vector from IMU data.
        
        Returns:
            Normalized gravity vector as numpy array, or None if unavailable
        """
        if not self.use_imu or not self.is_running:
            return None
        
        try:
            if self.bag_file:
                frames_available, frames = self.pipeline.try_wait_for_frames(timeout_ms=100)
                if not frames_available:
                    return None
            else:
                frames = self.pipeline.wait_for_frames()
            
            # Get accelerometer data
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                gravity_raw = np.array([accel_data.x, accel_data.y, accel_data.z])
                
                # Normalize and return gravity vector
                gravity_magnitude = np.linalg.norm(gravity_raw)
                if gravity_magnitude > 0.5:  # Reasonable gravity reading
                    return gravity_raw / gravity_magnitude
                    
        except Exception as e:
            print(f"Warning: Failed to get current gravity vector: {e}")
            
        return None
    
    def get_gravity_alignment_status(self) -> Dict[str, Any]:
        """
        Get current gravity alignment status and information.
        
        Returns:
            Dictionary with alignment status and parameters
        """
        return {
            'alignment_calibrated': self.alignment_calibrated,
            'current_gravity_vector': self.gravity_vector.tolist() if self.gravity_vector is not None else None,
            'rotation_matrix_determinant': np.linalg.det(self.rotation_matrix),
            'use_imu': self.use_imu,
            'is_running': self.is_running
        }
    
    def force_gravity_recalibration(self, num_samples: int = 15) -> bool:
        """
        Force recalibration of gravity alignment.
        
        Args:
            num_samples: Number of samples to collect for recalibration
            
        Returns:
            True if recalibration successful
        """
        print("Forcing gravity recalibration...")
        self.alignment_calibrated = False
        self.rotation_matrix = np.eye(3)
        self.gravity_vector = np.array([0, -1, 0])
        
        return self.calibrate_gravity_alignment(num_samples)
    
    def _calculate_alignment_rotation(self, from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
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
    
    def _inspect_bag_file(self) -> Optional[List[Dict[str, Any]]]:
        """Inspect bag file for available streams"""
        try:
            # This would require importing the bag_inspector from the original code
            # For now, return None to use automatic configuration
            return None
        except:
            return None
    
    def _enable_streams_from_bag(self, available_streams: List[Dict[str, Any]]):
        """Enable streams based on bag file contents"""
        has_depth = False
        has_color = False
        has_imu = False
        
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
        
        if has_imu:
            print("IMU data available for gravity-based alignment")
        else:
            print("No IMU data in bag file, using default orientation")
            self.use_imu = False
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """Get RealSense sensor information"""
        info = {
            'sensor_type': 'RealSense',
            'bag_file': self.bag_file,
            'realtime_mode': self.realtime_mode,
            'use_imu': self.use_imu,
            'alignment_calibrated': self.alignment_calibrated
        }
        
        if self.profile:
            try:
                device = self.profile.get_device()
                info['device_name'] = device.get_info(rs.camera_info.name)
                info['serial_number'] = device.get_info(rs.camera_info.serial_number)
                info['firmware_version'] = device.get_info(rs.camera_info.firmware_version)
            except:
                pass
        
        return info
    
    def is_available(self) -> bool:
        """Check if RealSense is available"""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            return len(devices) > 0 or self.bag_file is not None
        except:
            return False
    
    def get_factory_extrinsics(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get factory-calibrated extrinsics between color and depth sensors.
        
        For Intel RealSense L515, this returns the factory-calibrated transformation
        from color sensor to depth sensor coordinate frame. This eliminates the need
        for manual extrinsic calibration between internal sensors.
        
        Returns:
            Dictionary containing:
            - 'rotation': 3x3 rotation matrix from color to depth
            - 'translation': 3x1 translation vector from color to depth (meters)
            - 'transformation_matrix': 4x4 homogeneous transformation matrix
            Or None if extrinsics cannot be retrieved
        """
        if not self.profile:
            print("Warning: RealSense pipeline not started, cannot get extrinsics")
            return None
        
        try:
            # Get stream profiles for color and depth
            streams = self.profile.get_streams()
            color_profile = None
            depth_profile = None
            
            for stream in streams:
                if stream.stream_type() == rs.stream.color:
                    color_profile = stream
                elif stream.stream_type() == rs.stream.depth:
                    depth_profile = stream
            
            if not color_profile or not depth_profile:
                print("Warning: Could not find both color and depth streams")
                return None
            
            # Get extrinsics from color to depth using RealSense SDK
            extrinsics = color_profile.get_extrinsics_to(depth_profile)
            
            # Extract rotation matrix (3x3) and translation vector (3x1)
            rotation = np.array(extrinsics.rotation).reshape(3, 3)
            translation = np.array(extrinsics.translation).reshape(3, 1)
            
            # Create 4x4 homogeneous transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation
            transformation_matrix[:3, 3] = translation.flatten()
            
            print("Factory extrinsics retrieved successfully:")
            print(f"  Translation: [{translation[0,0]:.6f}, {translation[1,0]:.6f}, {translation[2,0]:.6f}] meters")
            print(f"  Rotation determinant: {np.linalg.det(rotation):.6f} (should be ±1)")
            
            return {
                'rotation': rotation,
                'translation': translation,
                'transformation_matrix': transformation_matrix
            }
            
        except Exception as e:
            print(f"Error retrieving factory extrinsics: {e}")
            return None
    
    def get_color_to_depth_transform(self) -> Optional[np.ndarray]:
        """
        Get the 4x4 transformation matrix from color to depth sensor.
        
        This is a convenience method that returns just the transformation matrix
        from the factory extrinsics.
        
        Returns:
            4x4 numpy array representing T_color_to_depth, or None if unavailable
        """
        extrinsics = self.get_factory_extrinsics()
        if extrinsics:
            return extrinsics['transformation_matrix']
        return None
    
    def get_camera_intrinsics(self):
        """
        Get camera intrinsic parameters from RealSense factory calibration.
        
        Returns:
            RealSense intrinsics object or None if unavailable
        """
        if not self.pipeline:
            print("RealSense pipeline not initialized")
            return None
        
        try:
            # Get profile for color stream
            profile = self.pipeline.get_active_profile()
            color_profile = profile.get_stream(rs.stream.color)
            
            if color_profile:
                video_profile = color_profile.as_video_stream_profile()
                intrinsics = video_profile.get_intrinsics()
                return intrinsics
            else:
                print("No color stream profile available")
                return None
                
        except Exception as e:
            print(f"Error getting camera intrinsics: {e}")
            return None
    
    def get_lidar_optimized_frame(self, distance_range: Tuple[float, float] = (0.25, 0.8),
                                enable_filtering: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Get frame optimized for LiDAR-style processing with enhanced filtering.
        
        Args:
            distance_range: Min/max distance filtering range (meters)
            enable_filtering: Whether to apply advanced filtering
            
        Returns:
            Tuple of (vertices, color_image, metadata) with LiDAR optimizations
        """
        if not self.is_running:
            return None, None, {'error': 'Sensor not running'}
        
        try:
            # Get frames
            if self.bag_file:
                frames_available, frames = self.pipeline.try_wait_for_frames(timeout_ms=100)
                if not frames_available:
                    return None, None, {'end_of_stream': True}
            else:
                frames = self.pipeline.wait_for_frames()
            
            self.frame_count += 1
            
            # Extract 3D points with LiDAR-specific processing
            vertices, color_image = self._extract_lidar_optimized_points(frames, distance_range, enable_filtering)
            
            # Create enhanced metadata for LiDAR processing
            metadata = {
                'frame_id': self.frame_count,
                'timestamp': frames.get_timestamp(),
                'has_depth': frames.get_depth_frame() is not None,
                'has_color': frames.get_color_frame() is not None,
                'alignment_calibrated': self.alignment_calibrated,
                'distance_range': distance_range,
                'filtering_enabled': enable_filtering,
                'lidar_optimized': True
            }
            
            # Add depth frame metadata
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_profile = depth_frame.get_profile().as_video_stream_profile()
                metadata['depth_width'] = depth_profile.width()
                metadata['depth_height'] = depth_profile.height()
                metadata['depth_fps'] = depth_profile.fps()
            
            # Add IMU data if available
            if self.use_imu:
                accel_frame = frames.first_or_default(rs.stream.accel)
                if accel_frame:
                    accel_data = accel_frame.as_motion_frame().get_motion_data()
                    metadata['imu_accel'] = [accel_data.x, accel_data.y, accel_data.z]
            
            return vertices, color_image, metadata
            
        except Exception as e:
            return None, None, {'error': str(e)}
    
    def _extract_lidar_optimized_points(self, frames, distance_range: Tuple[float, float],
                                      enable_filtering: bool) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract and filter 3D points optimized for LiDAR-style processing."""
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame:
            return None, None
        
        # Apply decimation filter to reduce number of points for performance
        depth_frame = self.decimate.process(depth_frame)
        
        # Generate point cloud
        if color_frame:
            self.pc.map_to(color_frame)
        points = self.pc.calculate(depth_frame)
        
        # Get vertices
        v = points.get_vertices()
        vertices = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        
        if vertices is None or len(vertices) == 0:
            return None, None
        
        # Apply distance filtering for LiDAR effective range
        min_dist, max_dist = distance_range
        depth_mask = (vertices[:, 2] >= min_dist) & (vertices[:, 2] <= max_dist)
        vertices = vertices[depth_mask]
        
        # Apply advanced filtering if enabled
        if enable_filtering and len(vertices) > 0:
            vertices = self._apply_lidar_advanced_filtering(vertices)
        
        # Store raw vertices before alignment (for visualization)
        self.last_raw_vertices = vertices.copy() if vertices is not None else None
        
        # Apply gravity-based alignment if calibrated
        if self.alignment_calibrated:
            vertices = self._apply_alignment(vertices)
        
        # Get color image
        color_image = None
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
            
            # Handle different color formats
            color_profile = color_frame.get_profile().as_video_stream_profile()
            if color_profile.format() == rs.format.rgb8:
                # Convert RGB to BGR for OpenCV
                import cv2
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
        return vertices, color_image
    
    def _apply_lidar_advanced_filtering(self, vertices: np.ndarray) -> np.ndarray:
        """Apply advanced filtering optimized for LiDAR-style object detection."""
        if vertices is None or len(vertices) < 10:
            return vertices
        
        # 1. Remove points with invalid coordinates
        valid_mask = (
            np.isfinite(vertices[:, 0]) & 
            np.isfinite(vertices[:, 1]) & 
            np.isfinite(vertices[:, 2])
        )
        vertices = vertices[valid_mask]
        
        if len(vertices) < 10:
            return vertices
        
        # 2. Apply statistical outlier removal (simplified for performance)
        from sklearn.neighbors import NearestNeighbors
        
        try:
            k_neighbors = min(10, len(vertices) - 1)
            if k_neighbors > 0:
                nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(vertices)
                distances, _ = nbrs.kneighbors(vertices)
                
                # Calculate mean distance to neighbors (excluding self)
                mean_distances = np.mean(distances[:, 1:], axis=1)
                
                # Remove statistical outliers
                distance_mean = np.mean(mean_distances)
                distance_std = np.std(mean_distances)
                threshold = distance_mean + 2.0 * distance_std  # 2 sigma threshold
                
                inlier_mask = mean_distances <= threshold
                vertices = vertices[inlier_mask]
        except Exception as e:
            # If filtering fails, continue with original points
            print(f"Warning: Advanced filtering failed: {e}")
        
        return vertices
    
    def get_depth_intrinsics(self):
        """
        Get depth camera intrinsic parameters for accurate 3D reconstruction.
        
        Returns:
            RealSense depth intrinsics object or None if unavailable
        """
        if not self.profile:
            print("Warning: RealSense pipeline not started, cannot get depth intrinsics")
            return None
        
        try:
            # Get depth stream profile
            streams = self.profile.get_streams()
            depth_profile = None
            
            for stream in streams:
                if stream.stream_type() == rs.stream.depth:
                    depth_profile = stream
                    break
            
            if not depth_profile:
                print("Warning: Could not find depth stream")
                return None
            
            # Get depth intrinsics
            video_profile = depth_profile.as_video_stream_profile()
            intrinsics = video_profile.get_intrinsics()
            
            print("Depth intrinsics retrieved successfully:")
            print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")
            print(f"  Focal length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
            print(f"  Principal point: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
            
            return intrinsics
            
        except Exception as e:
            print(f"Error retrieving depth intrinsics: {e}")
            return None
    
    def _calculate_vertices_with_external_intrinsics(self, depth_frame) -> Optional[np.ndarray]:
        """
        Calculate 3D vertices from depth frame using external intrinsics.
        This is used when CharUco calibration intrinsics should be used instead of factory ones.
        
        Args:
            depth_frame: RealSense depth frame
            
        Returns:
            3D vertices array [N, 3] or None if calculation fails
        """
        try:
            # Get depth data as numpy array
            depth_data = np.asanyarray(depth_frame.get_data(), dtype=np.float32)
            height, width = depth_data.shape
            
            # Extract intrinsics from external matrix
            fx = self.external_intrinsics[0, 0]
            fy = self.external_intrinsics[1, 1]
            cx = self.external_intrinsics[0, 2]
            cy = self.external_intrinsics[1, 2]
            
            # Apply decimation scaling (since we applied decimation filter)
            scale_factor = 2  # Decimation filter magnitude
            fx /= scale_factor
            fy /= scale_factor
            cx /= scale_factor
            cy /= scale_factor
            
            # Create coordinate grids
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            
            # Convert depth to meters (RealSense uses depth scale)
            # Get depth scale from profile/sensor instead of frame
            if hasattr(self, 'profile') and self.profile:
                depth_sensor = self.profile.get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()
            else:
                depth_scale = 0.001  # Default 1mm scale for L515/most RealSense cameras
            depth_m = depth_data * depth_scale
            
            # Calculate 3D coordinates using external intrinsics
            # Camera coordinate system: X=right, Y=down, Z=forward
            x = (u - cx) * depth_m / fx
            y = (v - cy) * depth_m / fy
            z = depth_m
            
            # Stack and reshape to [N, 3] format
            vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
            
            # Filter out invalid points (depth == 0)
            valid_mask = vertices[:, 2] > 0
            vertices = vertices[valid_mask]
            
            return vertices
            
        except Exception as e:
            print(f"Error calculating vertices with external intrinsics: {e}")
            return None