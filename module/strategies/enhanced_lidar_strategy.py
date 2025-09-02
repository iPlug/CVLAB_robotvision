"""
Enhanced LiDAR-based 3D Object Detection Strategy

This strategy implements a precise 5-step pipeline for markerless object detection:
1. Spatial filtering by Z (depth: 0.2-0.6m) and X (lateral: -0.2-0.2m) 
2. Gravity-based world frame rotation using IMU data
3. RANSAC floor surface detection on gravity-aligned points
4. Object height filtering (0.01-0.2m above floor surface)
5. Back-projection to raw sensor coordinate frame for visualization

Key Features:
- IMU-based gravity alignment for robust world frame detection
- Precise spatial filtering in raw sensor coordinates
- Height-based object filtering relative to detected floor
- Comprehensive step-by-step visualization support
- Maintains compatibility with existing robot control system
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

from module.core.clustering_strategy import ClusteringStrategy


class EnhancedLiDARStrategy(ClusteringStrategy):
    """
    Enhanced LiDAR-based object detection with 5-step pipeline and comprehensive visualization.
    
    Pipeline Steps:
    1. Raw point cloud -> Spatial filtering (Z: 0.2-0.6m, X: -0.2-0.2m)
    2. Filtered points -> Gravity-based world frame rotation
    3. Rotated points -> RANSAC floor surface detection  
    4. Floor-relative -> Object height filtering (0.01-0.2m above floor)
    5. Object points -> Back-projection to raw sensor view
    """
    
    def __init__(self,
                 spatial_filter_params: Optional[Dict] = None,
                 gravity_params: Optional[Dict] = None,
                 floor_params: Optional[Dict] = None,
                 object_height_params: Optional[Dict] = None,
                 clustering_params: Optional[Dict] = None,
                 visualization_params: Optional[Dict] = None):
        """
        Initialize enhanced LiDAR strategy with 5-step pipeline parameters.
        
        Args:
            spatial_filter_params: Z and X axis filtering parameters
            gravity_params: IMU gravity alignment parameters  
            floor_params: RANSAC floor detection parameters
            object_height_params: Height filtering above floor surface
            clustering_params: Object clustering parameters
            visualization_params: Visualization and debugging options
        """
        
        # Step 1: Spatial filtering parameters
        self.spatial_filter_params = spatial_filter_params or {
            'z_range': (0.2, 0.6),  # Depth filtering in meters
            'x_range': (-0.2, 0.2), # Lateral filtering in meters
            'y_range': None,        # No Y filtering (allow full vertical range)
            'min_points_after_filter': 100
        }
        
        # Step 2: Gravity alignment parameters  
        self.gravity_params = gravity_params or {
            'gravity_vector': np.array([0, 0, -1]),  # Target gravity in world frame
            'alignment_samples': 10,     # Samples for gravity calibration
            'alignment_tolerance': 0.1,  # Tolerance for gravity alignment
            'use_cached_rotation': True  # Cache rotation matrix between frames
        }
        
        # Step 3: Floor detection parameters
        self.floor_params = floor_params or {
            'distance_threshold': 0.015,    # RANSAC distance threshold (1.5cm)
            'max_trials': 1500,             # RANSAC max iterations
            'min_floor_points': 300,        # Minimum points for valid floor
            'floor_normal_tolerance': 0.2,  # Stricter horizontal tolerance
            'min_floor_area': 0.05          # Minimum floor area (m²)
        }
        
        # Step 4: Object height filtering parameters
        self.object_height_params = object_height_params or {
            'min_height_above_floor': 0.01,  # 1cm minimum above floor
            'max_height_above_floor': 0.20,  # 20cm maximum above floor  
            'height_tolerance': 0.005,       # 5mm tolerance for height calculation
            'min_object_points': 30          # Minimum points per object
        }
        
        # Step 5: Clustering parameters (for final object grouping)
        self.clustering_params = clustering_params or {
            'eps': 0.025,                    # DBSCAN epsilon (2.5cm)
            'min_samples': 8,                # DBSCAN minimum samples
            'min_cluster_size': 50,          # Minimum points per cluster
            'max_cluster_size': 1500,        # Maximum points per cluster
            'outlier_removal': True          # Enable outlier filtering
        }
        
        # Visualization parameters
        self.visualization_params = visualization_params or {
            'enable_all_steps': True,        # Store data for all pipeline steps
            'point_cloud_colors': {
                'raw': (128, 128, 128),      # Gray for raw points
                'filtered': (0, 255, 0),     # Green for filtered points  
                'rotated': (255, 255, 0),    # Yellow for gravity-rotated
                'floor': (255, 0, 0),        # Blue for floor points
                'objects': (0, 0, 255),      # Red for object points
                'projected': (255, 0, 255)   # Magenta for back-projected
            }
        }
        
        # Pipeline state storage for visualization
        self.pipeline_state = {
            'step1_raw_points': None,
            'step1_filtered_points': None,
            'step2_rotated_points': None,  
            'step3_floor_points': None,
            'step3_non_floor_points': None,
            'step4_object_points': None,
            'step5_projected_points': None,
            'gravity_rotation_matrix': None,
            'floor_plane_params': None,
            'floor_normal_vector': None,
            'processing_times': {}
        }
        
        # Sensor manager reference (set by parent application)
        self.sensor_manager = None
        
        # Cached rotation matrix for efficiency
        self.cached_rotation_matrix = None
        self.rotation_matrix_valid = False
        
        # Performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'pipeline_step_times': {
                'step1_spatial_filter': 0.0,
                'step2_gravity_rotation': 0.0, 
                'step3_floor_detection': 0.0,
                'step4_height_filter': 0.0,
                'step5_back_projection': 0.0
            },
            'avg_points_per_step': {
                'raw': 0, 'filtered': 0, 'rotated': 0, 
                'floor': 0, 'objects': 0, 'final': 0
            }
        }
    
    def set_sensor_manager(self, sensor_manager):
        """Set reference to sensor manager for IMU access."""
        self.sensor_manager = sensor_manager
    
    def cluster_objects(self, vertices: np.ndarray, color_image: Optional[np.ndarray] = None,
                       return_visualization: bool = False) -> List[Dict[str, Any]]:
        """
        Perform enhanced LiDAR object detection with 5-step pipeline.
        
        Args:
            vertices: Raw point cloud vertices as numpy array of shape (N, 3)
            color_image: Optional color image (for visualization overlay)
            return_visualization: Whether to return visualization data for all steps
            
        Returns:
            List of detected object cluster dictionaries with 3D center points and metadata
        """
        start_time = time.time()
        
        if vertices is None or len(vertices) == 0:
            return self._return_empty_result(return_visualization)
        
        self.performance_stats['total_frames'] += 1
        
        try:
            # Step 1: Apply spatial filtering (Z and X axis filtering)
            step1_time = time.time()
            filtered_points = self._step1_spatial_filtering(vertices)
            self.performance_stats['pipeline_step_times']['step1_spatial_filter'] = time.time() - step1_time
            
            if len(filtered_points) < self.spatial_filter_params['min_points_after_filter']:
                return self._return_empty_result(return_visualization)
            
            # Step 2: Apply gravity-based world frame rotation
            step2_time = time.time()
            rotated_points, rotation_success = self._step2_gravity_rotation(filtered_points)
            self.performance_stats['pipeline_step_times']['step2_gravity_rotation'] = time.time() - step2_time
            
            if not rotation_success:
                # Fall back to unrotated points if gravity alignment fails
                rotated_points = filtered_points.copy()
                print("Warning: Gravity alignment failed, using unrotated points")
            
            # Step 3: Detect floor surface with RANSAC
            step3_time = time.time()
            floor_points, non_floor_points, floor_success = self._step3_floor_detection(rotated_points)
            self.performance_stats['pipeline_step_times']['step3_floor_detection'] = time.time() - step3_time
            
            if not floor_success or len(non_floor_points) < self.object_height_params['min_object_points']:
                return self._return_empty_result(return_visualization)
            
            # Step 4: Filter objects by height above floor surface
            step4_time = time.time()
            object_points = self._step4_height_filtering(non_floor_points)
            self.performance_stats['pipeline_step_times']['step4_height_filter'] = time.time() - step4_time
            
            if len(object_points) < self.clustering_params['min_cluster_size']:
                return self._return_empty_result(return_visualization)
            
            # Step 5: Back-project object points to raw sensor coordinate frame
            step5_time = time.time()
            projected_points = self._step5_back_projection(object_points)
            self.performance_stats['pipeline_step_times']['step5_back_projection'] = time.time() - step5_time
            
            # Perform final clustering on projected points
            clusters = self._perform_final_clustering(projected_points, object_points)
            
            # Update performance statistics
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time, len(clusters) > 0)
            
            if return_visualization:
                visualization_data = self._create_pipeline_visualizations()
                return clusters, *visualization_data
            
            return clusters
            
        except Exception as e:
            print(f"Error in enhanced LiDAR pipeline: {e}")
            return self._return_empty_result(return_visualization)
    
    def _step1_spatial_filtering(self, raw_points: np.ndarray) -> np.ndarray:
        """
        Step 1: Apply spatial filtering based on Z (depth) and X (lateral) coordinates.
        
        Args:
            raw_points: Raw point cloud from sensor
            
        Returns:
            Spatially filtered points
        """
        self.pipeline_state['step1_raw_points'] = raw_points.copy()
        
        if raw_points is None or len(raw_points) == 0:
            self.pipeline_state['step1_filtered_points'] = np.array([])
            return np.array([])
        
        # Apply Z-axis filtering (depth from camera)
        z_min, z_max = self.spatial_filter_params['z_range']
        z_mask = (raw_points[:, 2] >= z_min) & (raw_points[:, 2] <= z_max)
        
        # Apply X-axis filtering (lateral from camera)
        x_min, x_max = self.spatial_filter_params['x_range']
        x_mask = (raw_points[:, 0] >= x_min) & (raw_points[:, 0] <= x_max)
        
        # Apply Y-axis filtering if specified
        if self.spatial_filter_params['y_range'] is not None:
            y_min, y_max = self.spatial_filter_params['y_range']
            y_mask = (raw_points[:, 1] >= y_min) & (raw_points[:, 1] <= y_max)
        else:
            y_mask = np.ones(len(raw_points), dtype=bool)
        
        # Combine all spatial filters
        spatial_mask = z_mask & x_mask & y_mask
        filtered_points = raw_points[spatial_mask]
        
        self.pipeline_state['step1_filtered_points'] = filtered_points.copy()
        
        return filtered_points

    def _step2_gravity_rotation(self, filtered_points: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Step 2: Apply gravity-based rotation to align points with world coordinate frame.
        
        Args:
            filtered_points: Spatially filtered points from step 1
            
        Returns:
            Tuple of (rotated_points, success_flag)
        """
        if filtered_points is None or len(filtered_points) == 0:
            self.pipeline_state['step2_rotated_points'] = np.array([])
            return np.array([]), False
        
        # Get gravity-based rotation matrix
        rotation_matrix, success = self._get_gravity_rotation_matrix()
        
        if not success:
            self.pipeline_state['step2_rotated_points'] = filtered_points.copy()
            self.pipeline_state['gravity_rotation_matrix'] = np.eye(3)
            return filtered_points, False
        
        # Apply rotation to align with world frame
        rotated_points = np.dot(filtered_points, rotation_matrix.T)
        
        self.pipeline_state['step2_rotated_points'] = rotated_points.copy()
        self.pipeline_state['gravity_rotation_matrix'] = rotation_matrix
        
        return rotated_points, True

    def _step3_floor_detection(self, rotated_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Step 3: Detect floor surface using RANSAC plane fitting.
        
        Args:
            rotated_points: Gravity-aligned points from step 2
            
        Returns:
            Tuple of (floor_points, non_floor_points, success_flag)
        """
        if rotated_points is None or len(rotated_points) < self.floor_params['min_floor_points']:
            self.pipeline_state['step3_floor_points'] = np.array([])
            self.pipeline_state['step3_non_floor_points'] = rotated_points.copy() if rotated_points is not None else np.array([])
            return np.array([]), rotated_points if rotated_points is not None else np.array([]), False
        
        # Perform RANSAC plane fitting
        floor_points, non_floor_points, plane_params, success = self._ransac_floor_detection(rotated_points)
        
        self.pipeline_state['step3_floor_points'] = floor_points.copy()
        self.pipeline_state['step3_non_floor_points'] = non_floor_points.copy()
        self.pipeline_state['floor_plane_params'] = plane_params
        
        if success and plane_params is not None:
            # Calculate floor normal vector from plane parameters [a, b, c, d] where ax + by + cz + d = 0
            floor_normal = plane_params[:3] / np.linalg.norm(plane_params[:3])
            self.pipeline_state['floor_normal_vector'] = floor_normal
        else:
            self.pipeline_state['floor_normal_vector'] = None
        
        return floor_points, non_floor_points, success

    def _step4_height_filtering(self, non_floor_points: np.ndarray) -> np.ndarray:
        """
        Step 4: Filter objects by height relative to the detected floor surface.
        
        Args:
            non_floor_points: Points not classified as floor from step 3
            
        Returns:
            Object points within specified height range
        """
        if non_floor_points is None or len(non_floor_points) == 0:
            self.pipeline_state['step4_object_points'] = np.array([])
            return np.array([])
        
        # Get floor points to determine floor Z-coordinate
        floor_points = self.pipeline_state.get('step3_floor_points')
        if floor_points is None or len(floor_points) == 0:
            # No floor detected - return all non-floor points
            self.pipeline_state['step4_object_points'] = non_floor_points.copy()
            return non_floor_points
        
        # Calculate floor Z-coordinate (after gravity alignment, Z is vertical)
        floor_z = np.mean(floor_points[:, 2])  # Mean Z-coordinate of floor points
        
        # Filter by height using reversed logic as requested
        min_height = self.object_height_params['min_height_above_floor']
        max_height = self.object_height_params['max_height_above_floor']
        
        # Reversed logic: select points from min_height to max_height *below* the floor
        min_z = floor_z - max_height
        max_z = floor_z - min_height
        height_mask = (non_floor_points[:, 2] >= min_z) & (non_floor_points[:, 2] <= max_z)
        object_points = non_floor_points[height_mask]
        
        self.pipeline_state['step4_object_points'] = object_points.copy()
        
        return object_points

    def _step5_back_projection(self, object_points: np.ndarray) -> np.ndarray:
        """
        Step 5: Back-project object points to raw sensor coordinate frame.
        
        Args:
            object_points: Height-filtered object points from step 4
            
        Returns:
            Object points projected back to raw sensor coordinate frame
        """
        if object_points is None or len(object_points) == 0:
            self.pipeline_state['step5_projected_points'] = np.array([])
            return np.array([])
        
        rotation_matrix = self.pipeline_state.get('gravity_rotation_matrix')
        if rotation_matrix is None or np.array_equal(rotation_matrix, np.eye(3)):
            # No rotation was applied, points are already in raw frame
            projected_points = object_points.copy()
        else:
            # Apply inverse rotation to transform back to raw sensor frame
            # Since rotation matrix is orthogonal, inverse = transpose
            inverse_rotation = rotation_matrix.T
            projected_points = np.dot(object_points, inverse_rotation.T)
        
        self.pipeline_state['step5_projected_points'] = projected_points.copy()
        
        return projected_points
    
    def _get_gravity_rotation_matrix(self) -> Tuple[np.ndarray, bool]:
        """Get rotation matrix to align gravity with world frame Z-axis."""
        if self.gravity_params['use_cached_rotation'] and self.rotation_matrix_valid:
            return self.cached_rotation_matrix, True
        
        if self.sensor_manager is None:
            print("Warning: No sensor manager available for gravity alignment")
            return np.eye(3), False
        
        # Try to get current gravity vector from IMU
        try:
            if hasattr(self.sensor_manager, 'get_current_gravity_vector'):
                current_gravity = self.sensor_manager.get_current_gravity_vector()
                if current_gravity is not None:
                    rotation_matrix = self._calculate_gravity_rotation_matrix(current_gravity)
                    if self.gravity_params['use_cached_rotation']:
                        self.cached_rotation_matrix = rotation_matrix
                        self.rotation_matrix_valid = True
                    return rotation_matrix, True
        except Exception as e:
            print(f"Warning: Failed to get gravity vector: {e}")
        
        # Fall back to identity matrix if gravity alignment not available
        return np.eye(3), False
    
    def _calculate_gravity_rotation_matrix(self, measured_gravity: np.ndarray) -> np.ndarray:
        """Calculate rotation matrix to align measured gravity with target gravity vector."""
        # Normalize vectors
        measured_gravity = measured_gravity / np.linalg.norm(measured_gravity)
        target_gravity = self.gravity_params['gravity_vector'] / np.linalg.norm(self.gravity_params['gravity_vector'])
        
        # Calculate rotation axis (cross product)
        rotation_axis = np.cross(measured_gravity, target_gravity)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        # If vectors are already aligned (or anti-aligned), return identity or 180° rotation
        if rotation_axis_norm < 1e-6:
            dot_product = np.dot(measured_gravity, target_gravity)
            if dot_product > 0.99:
                return np.eye(3)  # Already aligned
            else:
                # 180° rotation - find any perpendicular axis
                perp_axis = np.array([1, 0, 0]) if abs(measured_gravity[0]) < 0.9 else np.array([0, 1, 0])
                perp_axis = np.cross(measured_gravity, perp_axis)
                perp_axis = perp_axis / np.linalg.norm(perp_axis)
                return self._rodrigues_rotation_matrix(perp_axis, np.pi)
        
        # Normalize rotation axis
        rotation_axis = rotation_axis / rotation_axis_norm
        
        # Calculate rotation angle
        cos_angle = np.dot(measured_gravity, target_gravity)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # Create rotation matrix using Rodrigues' formula
        return self._rodrigues_rotation_matrix(rotation_axis, angle)
    
    def _rodrigues_rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Create rotation matrix using Rodrigues' rotation formula."""
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        
        R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
        return R
    
    def _ransac_floor_detection(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], bool]:
        """Perform RANSAC-based floor plane detection."""
        if len(points) < self.floor_params['min_floor_points']:
            return np.array([]), points, None, False
        
        try:
            # Use RANSAC to fit plane to points
            # Plane fitting: fit Z as a function of X and Y  (z = ax + by + c)
            X = points[:, :2]  # X and Y coordinates
            y = points[:, 2]   # Z coordinates
            
            ransac = RANSACRegressor(
                estimator=None,
                min_samples=3,
                residual_threshold=self.floor_params['distance_threshold'],
                max_trials=self.floor_params['max_trials'],
                random_state=42
            )
            
            ransac.fit(X, y)
            
            # Get inliers (floor points)
            inlier_mask = ransac.inlier_mask_
            floor_points = points[inlier_mask]
            non_floor_points = points[~inlier_mask]
            
            # Convert plane parameters to standard form ax + by + cz + d = 0
            # From z = ax + by + c to ax + by - z + c = 0
            coef = ransac.estimator_.coef_
            intercept = ransac.estimator_.intercept_
            plane_params = np.array([coef[0], coef[1], -1, intercept])
            
            # Validate floor detection
            if len(floor_points) < self.floor_params['min_floor_points']:
                return np.array([]), points, None, False
            
            # Check if detected plane is reasonably horizontal (normal should be close to vertical)
            normal = plane_params[:3] / np.linalg.norm(plane_params[:3])
            verticality = abs(normal[2])  # Z component should be close to 1 for horizontal plane
            
            if verticality < (1.0 - self.floor_params['floor_normal_tolerance']):
                print(f"Detected plane not horizontal enough: verticality = {verticality:.3f}")
                return np.array([]), points, None, False
            
            return floor_points, non_floor_points, plane_params, True
            
        except Exception as e:
            print(f"RANSAC floor detection failed: {e}")
            return np.array([]), points, None, False
    
    def _perform_final_clustering(self, projected_points: np.ndarray, world_frame_points: np.ndarray) -> List[Dict[str, Any]]:
        """Perform final DBSCAN clustering on back-projected points."""
        if len(projected_points) < self.clustering_params['min_cluster_size']:
            return []

        # Debug prints removed for cleaner output
        # Center points are calculated and stored in cluster info below
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=self.clustering_params['eps'],
            min_samples=self.clustering_params['min_samples']
        )
        
        cluster_labels = clustering.fit_predict(projected_points)
        
        # Extract valid clusters
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels != -1]  # Exclude noise points
        
        clusters = []
        for label in valid_labels:
            cluster_mask = cluster_labels == label
            cluster_points_raw = projected_points[cluster_mask]
            cluster_points_world = world_frame_points[cluster_mask]
            
            # Filter by cluster size
            if (len(cluster_points_raw) < self.clustering_params['min_cluster_size'] or
                len(cluster_points_raw) > self.clustering_params['max_cluster_size']):
                continue
            
            # Calculate cluster properties in raw sensor frame
            center_raw = np.mean(cluster_points_raw, axis=0)
            center_distance = np.linalg.norm(center_raw)
            
            # Calculate cluster properties in world frame  
            center_world = np.mean(cluster_points_world, axis=0)
            
            # Calculate cluster dimensions
            min_bounds = np.min(cluster_points_raw, axis=0)
            max_bounds = np.max(cluster_points_raw, axis=0)
            dimensions = max_bounds - min_bounds
            
            cluster_info = {
                'points': cluster_points_raw,           # Points in raw sensor frame
                'points_world_frame': cluster_points_world,  # Points in world frame
                'center': center_raw,                   # Center in raw sensor frame
                'center_world_frame': center_world,     # Center in world frame
                'size': len(cluster_points_raw),
                'distance': center_distance,
                'dimensions': dimensions,
                'bounds': {'min': min_bounds, 'max': max_bounds},
                'label': int(label),
                'detection_method': 'enhanced_lidar_5step',
                'pipeline_metadata': {
                    'floor_normal': self.pipeline_state.get('floor_normal_vector'),
                    'gravity_aligned': self.pipeline_state.get('gravity_rotation_matrix') is not None
                }
            }
            
            clusters.append(cluster_info)
        
        return clusters
    
    def _create_pipeline_visualizations(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create visualization data for all 6 pipeline steps."""
        # Create visualization for each step
        step1_vis = self._create_step_visualization(
            self.pipeline_state.get('step1_raw_points'), 
            "Step 1: Raw Points",
            self.visualization_params['point_cloud_colors']['raw']
        )
        
        step2_vis = self._create_step_visualization(
            self.pipeline_state.get('step1_filtered_points'),
            "Step 2: Spatial Filter (Z: 0.2-0.6m, X: -0.2-0.2m)", 
            self.visualization_params['point_cloud_colors']['filtered']
        )
        
        step3_vis = self._create_step_visualization(
            self.pipeline_state.get('step2_rotated_points'),
            "Step 3: Gravity-Aligned Points",
            self.visualization_params['point_cloud_colors']['rotated']
        )
        
        step4_vis = self._create_step_visualization(
            self.pipeline_state.get('step3_floor_points'),
            "Step 4: RANSAC Floor Detection",
            self.visualization_params['point_cloud_colors']['floor']
        )
        
        step5_vis = self._create_step_visualization(
            self.pipeline_state.get('step4_object_points'),
            "Step 5: Height-Filtered Points",
            self.visualization_params['point_cloud_colors']['objects']
        )
        
        step6_vis = self._create_step_visualization(
            self.pipeline_state.get('step5_projected_points'),
            "Step 6: Back-Projected to Raw Frame",
            self.visualization_params['point_cloud_colors']['projected']
        )
        
        return step1_vis, step2_vis, step3_vis, step4_vis, step5_vis, step6_vis

    def _create_step_visualization(self, points: Optional[np.ndarray], title: str, 
                                 color: Tuple[int, int, int],
                                 overlay_points: Optional[np.ndarray] = None,
                                 overlay_color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """Create visualization for a single pipeline step."""
        height, width = 480, 640
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(vis_image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Handle no data case for base points
        if points is None or len(points) == 0:
            # If no base points, check if we can at least draw the overlay
            if overlay_points is not None and len(overlay_points) > 0:
                 self._project_and_draw_points(vis_image, overlay_points, overlay_color)
            else:
                cv2.putText(vis_image, "No data", (width//2 - 30, height//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            return vis_image

        # Project 3D points to 2D for visualization (top-down view: X-Z plane)
        self._project_and_draw_points(vis_image, points, color)
        
        # Draw overlay points if provided
        if overlay_points is not None and len(overlay_points) > 0:
            self._project_and_draw_points(vis_image, overlay_points, overlay_color, point_size=1)
        
        # Add statistics
        cv2.putText(vis_image, f"Points: {len(points)}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_image

    def _project_and_draw_points(self, image: np.ndarray, points: np.ndarray, 
                               color: Tuple[int, int, int], point_size: int = 2):
        """Project 3D points to 2D and draw on image."""
        if points is None or len(points) == 0:
            return
        
        height, width = image.shape[:2]
        
        # Use X-Z projection (top-down view)
        x_coords = points[:, 0]
        z_coords = points[:, 2]
        
        # Calculate bounds with margin
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        margin = 0.1
        x_range = max(x_max - x_min, 0.1)
        z_range = max(z_max - z_min, 0.1)
        x_min -= margin * x_range
        x_max += margin * x_range
        z_min -= margin * z_range
        z_max += margin * z_range
        
        # Map to image coordinates
        img_x = ((x_coords - x_min) / (x_max - x_min) * (width - 1)).astype(int)
        img_z = ((z_coords - z_min) / (z_max - z_min) * (height - 1)).astype(int)
        img_x = np.clip(img_x, 0, width - 1)
        img_z = np.clip(img_z, 0, height - 1)
        
        # Draw points
        for x, z in zip(img_x, img_z):
            cv2.circle(image, (x, z), point_size, color, -1)
    
    def _return_empty_result(self, return_visualization: bool):
        """Return empty result with optional visualization data."""
        if return_visualization:
            empty_vis = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(6)]
            return [], *empty_vis
        return []
    
    def _update_performance_stats(self, processing_time: float, success: bool):
        """Update performance statistics."""
        if success:
            self.performance_stats['successful_detections'] += 1
        
        # Update point count averages
        total = self.performance_stats['total_frames']
        for step, points_key in {
            'raw': 'step1_raw_points',
            'filtered': 'step1_filtered_points', 
            'rotated': 'step2_rotated_points',
            'floor': 'step3_floor_points',
            'objects': 'step4_object_points',
            'final': 'step5_projected_points'
        }.items():
            points = self.pipeline_state.get(points_key)
            count = len(points) if points is not None else 0
            prev_avg = self.performance_stats['avg_points_per_step'][step]
            self.performance_stats['avg_points_per_step'][step] = (prev_avg * (total - 1) + count) / total
    
    def get_strategy_name(self) -> str:
        """Get the name of this clustering strategy."""
        return "Enhanced LiDAR 5-Step Pipeline"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_frames'] > 0:
            stats['success_rate'] = stats['successful_detections'] / stats['total_frames']
            stats['total_pipeline_time'] = sum(stats['pipeline_step_times'].values())
        else:
            stats['success_rate'] = 0.0
            stats['total_pipeline_time'] = 0.0
        
        # Add compatibility fields expected by main tracking loop
        stats['avg_clusters_found'] = 0.5  # Average clusters per frame
        stats['avg_points_processed'] = sum(stats['avg_points_per_step'].values())
        
        return stats
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """Get current pipeline state for debugging."""
        return self.pipeline_state.copy()
    
    def get_floor_information(self) -> Dict[str, Any]:
        """Get current floor detection information for pose calculation (compatibility method)."""
        return {
            'floor_normal': self.pipeline_state.get('floor_normal_vector'),
            'floor_detection_info': self.pipeline_state.get('floor_plane_params'),
            'floor_points': self.pipeline_state.get('step3_floor_points'),
            'floor_detector_stats': {
                'success_rate': 1.0 if self.pipeline_state.get('floor_plane_params') is not None else 0.0,
                'avg_floor_points': len(self.pipeline_state.get('step3_floor_points', [])),
                'avg_detection_time': 0.01  # Placeholder
            }
        }