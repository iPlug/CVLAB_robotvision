#!/usr/bin/env python3
"""
Point cloud processing utilities for LiDAR-based object detection.
Handles point cloud generation, filtering, and surface detection operations.
"""

import numpy as np
import pyrealsense2 as rs
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN


class PointCloudUtils:
    """Utilities for point cloud processing and filtering operations."""
    
    def __init__(self, config_path="."):
        """Initialize point cloud utilities."""
        self.config_path = config_path
        
        # Default parameters (can be overridden)
        self.surface_distance_threshold = 0.01
        self.clustering_eps = 0.02
        self.clustering_min_samples = 10
        self.viz_max_points = 5000
        self.viz_downsample_factor = 3
        
        # Initialize point cloud object
        self.pc = rs.pointcloud()
        self.points = rs.points()
    
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
    
    def filter_roi(self, point_cloud, roi_bounds):
        """Filter point cloud to region of interest."""
        # Apply ROI filtering
        roi_mask = (
            (point_cloud[:, 0] >= roi_bounds['x_min']) &
            (point_cloud[:, 0] <= roi_bounds['x_max']) &
            (point_cloud[:, 1] >= roi_bounds['y_min']) &
            (point_cloud[:, 1] <= roi_bounds['y_max']) &
            (point_cloud[:, 2] >= roi_bounds['z_min']) &
            (point_cloud[:, 2] <= roi_bounds['z_max'])
        )
        
        return point_cloud[roi_mask]
    
    def detect_surface_plane(self, point_cloud, surface_distance_threshold=None):
        """Detect surface plane using RANSAC."""
        if len(point_cloud) < 3:
            return None
        
        threshold = surface_distance_threshold or self.surface_distance_threshold
        
        # Use only X and Y coordinates to fit a horizontal plane (Z as target)
        X = point_cloud[:, :2]  # X, Y coordinates
        y = point_cloud[:, 2]   # Z coordinates
        
        # RANSAC plane fitting
        ransac = RANSACRegressor(
            residual_threshold=threshold,
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
    
    def find_all_significant_clusters(self, object_points, min_cluster_size=15, clustering_eps=None, clustering_min_samples=None):
        """Find all significant clusters in the object points with color assignments."""
        if len(object_points) < (clustering_min_samples or self.clustering_min_samples):
            return []
        
        eps = clustering_eps or self.clustering_eps
        min_samples = clustering_min_samples or self.clustering_min_samples
        
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
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
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
    
    def downsample_points_for_visualization(self, point_cloud, max_points=None, downsample_factor=None):
        """Downsample point cloud for visualization to improve performance."""
        if point_cloud is None or len(point_cloud) == 0:
            return point_cloud
        
        max_pts = max_points or self.viz_max_points
        factor = downsample_factor or self.viz_downsample_factor
        
        # If point cloud is small enough, return as-is
        if len(point_cloud) <= max_pts:
            return point_cloud
        
        # Method 1: Regular downsampling (every Nth point)
        if len(point_cloud) <= max_pts * factor:
            return point_cloud[::factor]
        
        # Method 2: Random sampling for very dense point clouds
        indices = np.random.choice(len(point_cloud), max_pts, replace=False)
        return point_cloud[indices]
    
    def detect_object_knn_simple(self, point_cloud, intrinsics, knn_radius=0.05, min_neighbors=5, roi_size=50):
        """Detect object using k-nearest neighbors approach."""
        if len(point_cloud) < min_neighbors:
            return {'valid_pose': False, 'center': None, 'confidence': 0.0}
        
        try:
            # Find densest region using KNN
            from sklearn.neighbors import NearestNeighbors
            
            nbrs = NearestNeighbors(radius=knn_radius, algorithm='ball_tree')
            nbrs.fit(point_cloud)
            
            # Find neighbors for each point
            distances, indices = nbrs.radius_neighbors(point_cloud)
            
            # Find point with most neighbors
            neighbor_counts = [len(idx) for idx in indices]
            max_neighbors_idx = np.argmax(neighbor_counts)
            max_neighbors = neighbor_counts[max_neighbors_idx]
            
            if max_neighbors < min_neighbors:
                return {'valid_pose': False, 'center': None, 'confidence': 0.0}
            
            # Get the dense region center
            dense_region_indices = indices[max_neighbors_idx]
            dense_region_points = point_cloud[dense_region_indices]
            center_3d = np.mean(dense_region_points, axis=0)
            
            # Calculate confidence
            confidence = min(1.0, max_neighbors / (min_neighbors * 3))
            
            return {
                'valid_pose': True,
                'center': center_3d,
                'confidence': confidence,
                'point_count': max_neighbors,
                'dense_region_points': dense_region_points
            }
            
        except Exception as e:
            print(f"KNN detection error: {e}")
            return {'valid_pose': False, 'center': None, 'confidence': 0.0}