"""
LiDAR-based 3D Object Clustering Strategy

This strategy implements markerless object detection using LiDAR point clouds.
It performs floor detection, object clustering, and nearest object selection
for robot tracking applications.

Key Features:
- RANSAC-based floor detection and removal
- 3D DBSCAN clustering for object separation
- Distance filtering for LiDAR effective range (0.25-0.8m)
- Nearest object selection from camera center
- Center point calculation with coordinate transformation
- Temporal stability filtering
- Comprehensive visualization support
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Any, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from module.core.clustering_strategy import ClusteringStrategy
from module.processing.floor_detector import FloorDetector


class LiDARClusteringStrategy(ClusteringStrategy):
    """
    LiDAR-based object clustering using floor detection and 3D clustering.
    
    Pipeline:
    1. Distance filtering (0.25-0.8m LiDAR effective range)
    2. Floor detection and removal using RANSAC
    3. 3D DBSCAN clustering on remaining object points
    4. Cluster size filtering and nearest object selection
    5. Center point calculation for robot coordination
    """
    
    def __init__(self,
                 distance_range: Tuple[float, float] = (0.25, 0.8),
                 floor_params: Optional[Dict] = None,
                 clustering_params: Optional[Dict] = None,
                 selection_params: Optional[Dict] = None,
                 enable_temporal_filtering: bool = True):
        """
        Initialize LiDAR clustering strategy.
        
        Args:
            distance_range: Min/max distance from camera (meters)
            floor_params: Floor detection parameters
            clustering_params: DBSCAN clustering parameters
            selection_params: Object selection parameters
            enable_temporal_filtering: Enable temporal stability
        """
        
        # Distance filtering for LiDAR effective range
        self.distance_range = distance_range
        
        # Floor detection parameters
        self.floor_params = floor_params or {
            'distance_threshold': 0.02,    # 2cm tolerance for floor plane
            'max_trials': 1000,            # RANSAC max iterations
            'min_floor_points': 500,       # Minimum points for valid floor
            'floor_normal_tolerance': 0.3  # Allow slight floor tilt
        }
        
        # 3D clustering parameters
        self.clustering_params = clustering_params or {
            'eps': 0.03,                   # DBSCAN epsilon (3cm)
            'min_samples': 10,             # DBSCAN minimum samples
            'min_cluster_size': 50,        # Minimum points per cluster
            'max_cluster_size': 2000,      # Maximum points per cluster
            'outlier_removal': True        # Enable statistical outlier removal
        }
        
        # Object selection parameters
        self.selection_params = selection_params or {
            'selection_method': 'nearest_center',  # 'nearest_center', 'largest', 'nearest_point'
            'max_selection_distance': 1.0,        # Max distance for selection (meters)
            'height_preference': 0.1               # Prefer objects at certain height
        }
        
        # Initialize floor detector
        self.floor_detector = FloorDetector(
            distance_threshold=self.floor_params['distance_threshold'],
            max_trials=self.floor_params['max_trials'],
            min_floor_points=self.floor_params['min_floor_points'],
            floor_normal_tolerance=self.floor_params['floor_normal_tolerance']
        )
        
        # Temporal filtering for stability
        self.enable_temporal_filtering = enable_temporal_filtering
        self.object_history = []
        self.max_history = 5
        
        # Performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'successful_detections': 0,
            'avg_processing_time': 0.0,
            'avg_clusters_found': 0.0,
            'avg_points_processed': 0
        }
        
        # Last detection results for visualization
        self.last_detection_results = {
            'original_points': None,
            'filtered_points': None,
            'floor_points': None,
            'object_points': None,
            'clusters': None,
            'selected_object': None,
            'floor_normal': None,
            'floor_detection_info': None
        }
    
    def cluster_objects(self, vertices: np.ndarray, color_image: Optional[np.ndarray] = None,
                       return_visualization: bool = False) -> List[Dict[str, Any]]:
        """
        Perform LiDAR-based object clustering with floor detection.
        
        Args:
            vertices: Point cloud vertices as numpy array of shape (N, 3)
            color_image: Optional color image (not used in this strategy)
            return_visualization: Whether to return visualization data
            
        Returns:
            List of cluster dictionaries with 3D center points and metadata
        """
        start_time = time.time()
        
        if vertices is None or len(vertices) == 0:
            return []
        
        self.performance_stats['total_frames'] += 1
        
        # Store original points for visualization
        self.last_detection_results['original_points'] = vertices.copy()
        
        # Step 1: Apply distance filtering for LiDAR effective range
        filtered_points = self._apply_distance_filter(vertices)
        self.last_detection_results['filtered_points'] = filtered_points
        
        if len(filtered_points) < 100:  # Need minimum points for processing
            return []
        
        # Step 2: Detect and remove floor
        object_points, floor_points, floor_debug = self.floor_detector.detect_floor(
            filtered_points, return_debug_info=True
        )
        
        # Get floor information for pose calculation
        floor_normal = self.floor_detector.get_floor_normal()
        floor_detection_info = self.floor_detector.get_floor_detection_info()
        
        self.last_detection_results['floor_points'] = floor_points
        self.last_detection_results['object_points'] = object_points
        self.last_detection_results['floor_normal'] = floor_normal
        self.last_detection_results['floor_detection_info'] = floor_detection_info
        
        if len(object_points) < self.clustering_params['min_cluster_size']:
            return []
        
        # Step 3: Apply 3D clustering to object points
        clusters = self._perform_3d_clustering(object_points, floor_normal, floor_detection_info)
        self.last_detection_results['clusters'] = clusters
        
        if not clusters:
            return []
        
        # Step 4: Select best object based on criteria
        selected_cluster = self._select_target_object(clusters)
        self.last_detection_results['selected_object'] = selected_cluster
        
        # Step 5: Apply temporal filtering for stability
        if self.enable_temporal_filtering and selected_cluster:
            selected_cluster = self._apply_temporal_filtering(selected_cluster)
        
        # Update performance statistics
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time, len(clusters), len(vertices), 
                                     selected_cluster is not None)
        
        # Return results
        result_clusters = [selected_cluster] if selected_cluster else []
        
        if return_visualization:
            # Create visualization data
            lidar_view = self._create_lidar_visualization()
            floor_view = self.floor_detector.create_floor_visualization(
                filtered_points, object_points, floor_points
            )
            cluster_view = self._create_cluster_visualization(object_points, clusters)
            
            return result_clusters, lidar_view, floor_view, cluster_view
        
        return result_clusters
    
    def get_strategy_name(self) -> str:
        """Get the name of this clustering strategy."""
        return "LiDAR 3D Clustering"
    
    def _apply_distance_filter(self, vertices: np.ndarray) -> np.ndarray:
        """Filter points by distance from camera for LiDAR effective range."""
        if vertices is None or len(vertices) == 0:
            return vertices
        
        # Calculate distance from camera (at origin)
        distances = np.linalg.norm(vertices, axis=1)
        
        # Apply distance range filter
        min_dist, max_dist = self.distance_range
        distance_mask = (distances >= min_dist) & (distances <= max_dist)
        
        return vertices[distance_mask]
    
    def _perform_3d_clustering(self, object_points: np.ndarray, 
                             floor_normal: Optional[np.ndarray] = None,
                             floor_detection_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform 3D DBSCAN clustering on object points."""
        if len(object_points) < self.clustering_params['min_samples']:
            return []
        
        # Apply statistical outlier removal if enabled
        if self.clustering_params['outlier_removal']:
            object_points = self._remove_statistical_outliers(object_points)
        
        if len(object_points) < self.clustering_params['min_samples']:
            return []
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=self.clustering_params['eps'],
            min_samples=self.clustering_params['min_samples']
        )
        
        cluster_labels = clustering.fit_predict(object_points)
        
        # Extract valid clusters (exclude noise points with label -1)
        unique_labels = np.unique(cluster_labels)
        valid_labels = unique_labels[unique_labels != -1]
        
        clusters = []
        for label in valid_labels:
            cluster_mask = cluster_labels == label
            cluster_points = object_points[cluster_mask]
            
            # Filter by cluster size
            if (len(cluster_points) < self.clustering_params['min_cluster_size'] or
                len(cluster_points) > self.clustering_params['max_cluster_size']):
                continue
            
            # Calculate cluster properties
            center = np.mean(cluster_points, axis=0)
            center_distance = np.linalg.norm(center)
            
            # Calculate cluster dimensions
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)
            dimensions = max_bounds - min_bounds
            
            # Calculate cluster density
            volume = np.prod(dimensions)
            density = len(cluster_points) / max(volume, 1e-8)
            
            cluster_info = {
                'points': cluster_points,
                'center': center,
                'size': len(cluster_points),
                'distance': center_distance,
                'dimensions': dimensions,
                'bounds': {'min': min_bounds, 'max': max_bounds},
                'density': density,
                'label': int(label),
                # Add floor information for pose calculation
                'floor_normal': floor_normal,
                'floor_detection_info': floor_detection_info
            }
            
            clusters.append(cluster_info)
        
        return clusters
    
    def _remove_statistical_outliers(self, points: np.ndarray, 
                                   k_neighbors: int = 20, 
                                   std_multiplier: float = 1.0) -> np.ndarray:
        """Remove statistical outliers based on distance to neighbors."""
        if len(points) < k_neighbors:
            return points
        
        # Find k nearest neighbors for each point
        k = min(k_neighbors, len(points) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        
        # Calculate mean distance to neighbors (excluding self)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # Remove outliers based on statistical threshold
        distance_mean = np.mean(mean_distances)
        distance_std = np.std(mean_distances)
        threshold = distance_mean + std_multiplier * distance_std
        
        inlier_mask = mean_distances <= threshold
        return points[inlier_mask]
    
    def _select_target_object(self, clusters: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select target object based on selection criteria."""
        if not clusters:
            return None
        
        method = self.selection_params['selection_method']
        max_distance = self.selection_params['max_selection_distance']
        
        # Filter clusters by maximum distance
        nearby_clusters = [c for c in clusters if c['distance'] <= max_distance]
        
        if not nearby_clusters:
            return None
        
        if method == 'nearest_center':
            # Select cluster with center point nearest to camera
            selected = min(nearby_clusters, key=lambda c: c['distance'])
            
        elif method == 'largest':
            # Select largest cluster by point count
            selected = max(nearby_clusters, key=lambda c: c['size'])
            
        elif method == 'nearest_point':
            # Select cluster with any point nearest to camera
            selected = None
            min_point_distance = float('inf')
            
            for cluster in nearby_clusters:
                point_distances = np.linalg.norm(cluster['points'], axis=1)
                closest_point_dist = np.min(point_distances)
                
                if closest_point_dist < min_point_distance:
                    min_point_distance = closest_point_dist
                    selected = cluster
        
        else:
            # Default to nearest center
            selected = min(nearby_clusters, key=lambda c: c['distance'])
        
        # Add selection metadata
        if selected:
            selected['selection_method'] = method
            selected['selection_timestamp'] = time.time()
        
        return selected
    
    def _apply_temporal_filtering(self, current_object: Dict[str, Any]) -> Dict[str, Any]:
        """Apply temporal filtering to reduce noise in object detection."""
        if not current_object:
            return current_object
        
        # Add to history
        self.object_history.append({
            'center': current_object['center'].copy(),
            'timestamp': time.time(),
            'size': current_object['size']
        })
        
        # Keep only recent history
        if len(self.object_history) > self.max_history:
            self.object_history.pop(0)
        
        # If we have enough history, smooth the center position
        if len(self.object_history) >= 3:
            recent_centers = [obj['center'] for obj in self.object_history[-3:]]
            smoothed_center = np.mean(recent_centers, axis=0)
            
            # Update center with smoothed position
            current_object['center'] = smoothed_center
            current_object['temporal_smoothed'] = True
        
        return current_object
    
    def _update_performance_stats(self, processing_time: float, clusters_found: int, 
                                points_processed: int, success: bool):
        """Update performance statistics."""
        if success:
            self.performance_stats['successful_detections'] += 1
        
        # Update running averages
        total = self.performance_stats['total_frames']
        
        # Processing time
        prev_avg_time = self.performance_stats['avg_processing_time']
        self.performance_stats['avg_processing_time'] = (
            (prev_avg_time * (total - 1) + processing_time) / total
        )
        
        # Clusters found
        prev_avg_clusters = self.performance_stats['avg_clusters_found']
        self.performance_stats['avg_clusters_found'] = (
            (prev_avg_clusters * (total - 1) + clusters_found) / total
        )
        
        # Points processed
        prev_avg_points = self.performance_stats['avg_points_processed']
        self.performance_stats['avg_points_processed'] = (
            (prev_avg_points * (total - 1) + points_processed) / total
        )
    
    def _create_lidar_visualization(self) -> np.ndarray:
        """Create overall LiDAR visualization showing processing pipeline."""
        height, width = 480, 640
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        results = self.last_detection_results
        
        if results['filtered_points'] is None or len(results['filtered_points']) == 0:
            cv2.putText(vis_image, "No LiDAR data", (width//2 - 60, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis_image
        
        # Project 3D points to 2D for visualization (side view: X-Y plane)
        points = results['filtered_points']
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Calculate bounds
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        # Add margins
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= margin * x_range
        x_max += margin * x_range  
        y_min -= margin * y_range
        y_max += margin * y_range
        
        # Map to image coordinates
        img_x = ((x_coords - x_min) / (x_max - x_min) * (width - 1)).astype(int)
        img_y = (height - 1 - ((y_coords - y_min) / (y_max - y_min) * (height - 1))).astype(int)
        
        # Clip to bounds
        img_x = np.clip(img_x, 0, width - 1)
        img_y = np.clip(img_y, 0, height - 1)
        
        # Draw filtered points in gray
        for x, y in zip(img_x, img_y):
            cv2.circle(vis_image, (x, y), 1, (128, 128, 128), -1)
        
        # Draw floor points in blue if available
        if results['floor_points'] is not None and len(results['floor_points']) > 0:
            floor_x = ((results['floor_points'][:, 0] - x_min) / (x_max - x_min) * (width - 1)).astype(int)
            floor_y = (height - 1 - ((results['floor_points'][:, 1] - y_min) / (y_max - y_min) * (height - 1))).astype(int)
            floor_x = np.clip(floor_x, 0, width - 1)
            floor_y = np.clip(floor_y, 0, height - 1)
            
            for x, y in zip(floor_x, floor_y):
                cv2.circle(vis_image, (x, y), 1, (255, 0, 0), -1)  # Blue
        
        # Draw object points in green if available
        if results['object_points'] is not None and len(results['object_points']) > 0:
            obj_x = ((results['object_points'][:, 0] - x_min) / (x_max - x_min) * (width - 1)).astype(int)
            obj_y = (height - 1 - ((results['object_points'][:, 1] - y_min) / (y_max - y_min) * (height - 1))).astype(int)
            obj_x = np.clip(obj_x, 0, width - 1)
            obj_y = np.clip(obj_y, 0, height - 1)
            
            for x, y in zip(obj_x, obj_y):
                cv2.circle(vis_image, (x, y), 2, (0, 255, 0), -1)  # Green
        
        # Draw selected object center in red if available
        if results['selected_object'] is not None:
            center = results['selected_object']['center']
            center_x = int((center[0] - x_min) / (x_max - x_min) * (width - 1))
            center_y = int(height - 1 - ((center[1] - y_min) / (y_max - y_min) * (height - 1)))
            center_x = np.clip(center_x, 0, width - 1)
            center_y = np.clip(center_y, 0, height - 1)
            
            cv2.circle(vis_image, (center_x, center_y), 8, (0, 0, 255), -1)  # Red center
            cv2.circle(vis_image, (center_x, center_y), 12, (255, 255, 255), 2)  # White outline
        
        # Add information overlay
        stats = self.get_performance_stats()
        
        cv2.putText(vis_image, f"LiDAR Range: {self.distance_range[0]}-{self.distance_range[1]}m", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(vis_image, f"Points: {len(results['filtered_points'])}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if results['selected_object']:
            obj = results['selected_object']
            cv2.putText(vis_image, f"Object: {obj['size']} pts, {obj['distance']:.2f}m", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(vis_image, f"Success Rate: {stats['success_rate']:.1%}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return vis_image
    
    def _create_cluster_visualization(self, object_points: np.ndarray, 
                                   clusters: List[Dict[str, Any]]) -> np.ndarray:
        """Create visualization of cluster analysis results."""
        height, width = 480, 640
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(object_points) == 0:
            return vis_image
        
        # Use top-down view (X-Z plane) for cluster visualization
        x_coords = object_points[:, 0]
        z_coords = object_points[:, 2]
        
        # Calculate bounds
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        margin = 0.1
        x_range = max(x_max - x_min, 0.1)
        z_range = max(z_max - z_min, 0.1)
        x_min -= margin * x_range
        x_max += margin * x_range
        z_min -= margin * z_range
        z_max += margin * z_range
        
        # Draw all object points in gray first
        img_x = ((x_coords - x_min) / (x_max - x_min) * (width - 1)).astype(int)
        img_z = ((z_coords - z_min) / (z_max - z_min) * (height - 1)).astype(int)
        img_x = np.clip(img_x, 0, width - 1)
        img_z = np.clip(img_z, 0, height - 1)
        
        for x, z in zip(img_x, img_z):
            cv2.circle(vis_image, (x, z), 1, (64, 64, 64), -1)
        
        # Draw clusters in different colors
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 255, 128), 
                 (255, 128, 128), (128, 128, 255), (255, 128, 0), (128, 255, 255)]
        
        for i, cluster in enumerate(clusters):
            color = colors[i % len(colors)]
            points = cluster['points']
            
            # Draw cluster points
            cluster_x = ((points[:, 0] - x_min) / (x_max - x_min) * (width - 1)).astype(int)
            cluster_z = ((points[:, 2] - z_min) / (z_max - z_min) * (height - 1)).astype(int)
            cluster_x = np.clip(cluster_x, 0, width - 1)
            cluster_z = np.clip(cluster_z, 0, height - 1)
            
            for x, z in zip(cluster_x, cluster_z):
                cv2.circle(vis_image, (x, z), 2, color, -1)
            
            # Draw cluster center
            center = cluster['center']
            center_x = int((center[0] - x_min) / (x_max - x_min) * (width - 1))
            center_z = int((center[2] - z_min) / (z_max - z_min) * (height - 1))
            center_x = np.clip(center_x, 0, width - 1)
            center_z = np.clip(center_z, 0, height - 1)
            
            cv2.circle(vis_image, (center_x, center_z), 6, color, -1)
            cv2.circle(vis_image, (center_x, center_z), 8, (255, 255, 255), 2)
            
            # Label cluster
            cv2.putText(vis_image, f"{i}", (center_x + 10, center_z), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add cluster information
        cv2.putText(vis_image, f"Clusters Found: {len(clusters)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if clusters:
            cv2.putText(vis_image, f"Sizes: {[c['size'] for c in clusters]}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.performance_stats.copy()
        
        # Calculate derived metrics
        if stats['total_frames'] > 0:
            stats['success_rate'] = stats['successful_detections'] / stats['total_frames']
        else:
            stats['success_rate'] = 0.0
        
        # Add floor detector stats
        floor_stats = self.floor_detector.get_detection_stats()
        stats.update({f'floor_{k}': v for k, v in floor_stats.items()})
        
        return stats
    
    def get_floor_information(self) -> Dict[str, Any]:
        """Get current floor detection information for pose calculation."""
        return {
            'floor_normal': self.last_detection_results.get('floor_normal'),
            'floor_detection_info': self.last_detection_results.get('floor_detection_info'),
            'floor_points': self.last_detection_results.get('floor_points'),
            'floor_detector_stats': self.floor_detector.get_detection_stats()
        }