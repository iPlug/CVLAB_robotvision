#!/usr/bin/env python3
"""
Detection utilities for CharUco pose estimation and object detection applications.
Handles common detection operations, result handling, and object detection algorithms.
"""

import time
import numpy as np
import cv2
from pathlib import Path


class DetectionUtils:
    """Utilities for common detection operations and result handling."""
    
    def __init__(self, config_path="."):
        """Initialize detection utilities."""
        self.config_path = Path(config_path)
        
        # Default detection parameters
        self.object_height_threshold = 0.005
        self.knn_radius = 0.05
        self.knn_min_neighbors = 5
    
    def check_subpixel(self, corners):
        """Check if corners are subpixel accurate."""
        for xy in np.asarray(corners).flatten():
            if not xy.is_integer():
                return True
        return False
    
    def save_results(self, image, results, output_dir_name="realtime_output"):
        """Save detection results to file."""
        timestamp = int(time.time())
        output_dir = self.config_path / f"{output_dir_name}_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(output_dir / "image.png"), image)
        
        if results.get('valid_pose', False):
            if 'rvec' in results and 'tvec' in results:
                rotation_matrix = cv2.Rodrigues(results['rvec'])[0]
                tvec = results['tvec']
                
                charuco_tf = np.concatenate((
                    np.concatenate((rotation_matrix, tvec), axis=1),
                    np.array([[0, 0, 0, 1]])
                ))
                
                camera_position = -np.matrix(rotation_matrix).T * np.matrix(tvec)
                camera_tf = np.concatenate((
                    np.concatenate((rotation_matrix.T, camera_position), axis=1),
                    np.array([[0, 0, 0, 1]])
                ))
                
                np.savetxt(output_dir / 'charuco_tf.txt', charuco_tf)
                np.savetxt(output_dir / 'camera_tf.txt', camera_tf)
            
            print(f"Results saved to {output_dir}")
        else:
            print(f"Image saved to {output_dir} (no pose detected)")
    
    def detect_object_on_surface(self, color_frame, depth_frame, point_cloud, surface_plane, intrinsics):
        """Detect object on surface using point cloud analysis."""
        try:
            if point_cloud is None or len(point_cloud) < 10:
                return self.create_empty_detection_result(point_cloud, surface_plane, point_cloud)
            
            # Filter points above surface
            object_points = self.filter_points_above_surface(point_cloud, surface_plane)
            
            if len(object_points) < 10:
                return self.create_empty_detection_result(point_cloud, surface_plane, point_cloud)
            
            # Find object center
            object_center = np.mean(object_points, axis=0)
            
            # Calculate confidence based on point density
            confidence = min(1.0, len(object_points) / 50.0)
            
            # Project to pixel coordinates
            pixel_position = self.project_point_to_pixel(object_center, intrinsics)
            
            return {
                'valid_pose': True,
                'center': object_center,
                'confidence': confidence,
                'pixel_position': pixel_position,
                'point_count': len(object_points),
                'object_points': object_points,
                'surface_plane': surface_plane,
                'full_point_cloud': point_cloud
            }
            
        except Exception as e:
            print(f"Surface detection error: {e}")
            return self.create_empty_detection_result(point_cloud, surface_plane, point_cloud)
    
    def detect_tabletop_object(self, color_frame, depth_frame, point_cloud, surface_plane, intrinsics):
        """Enhanced tabletop object detection with multiple strategies."""
        try:
            if point_cloud is None or len(point_cloud) < 10:
                return self.create_empty_detection_result(point_cloud, surface_plane, point_cloud)
            
            # Filter points above surface
            object_points = self.filter_points_above_surface(point_cloud, surface_plane)
            
            if len(object_points) < 5:
                return self.create_empty_detection_result(point_cloud, surface_plane, point_cloud)
            
            # Try different detection strategies
            strategies = [
                self._detect_largest_cluster,
                self._detect_densest_region,
                self._detect_highest_point
            ]
            
            best_result = None
            best_confidence = 0.0
            
            for strategy in strategies:
                result = strategy(object_points, intrinsics)
                if result.get('valid_pose', False) and result.get('confidence', 0) > best_confidence:
                    best_result = result
                    best_confidence = result['confidence']
            
            if best_result:
                best_result.update({
                    'object_points': object_points,
                    'surface_plane': surface_plane,
                    'full_point_cloud': point_cloud
                })
                return best_result
            
            return self.create_empty_detection_result(point_cloud, surface_plane, point_cloud)
            
        except Exception as e:
            print(f"Tabletop detection error: {e}")
            return self.create_empty_detection_result(point_cloud, surface_plane, point_cloud)
    
    def _detect_largest_cluster(self, object_points, intrinsics):
        """Detect object by finding the largest cluster."""
        try:
            from sklearn.cluster import DBSCAN
            
            clustering = DBSCAN(eps=0.02, min_samples=5)
            labels = clustering.fit_predict(object_points)
            
            # Find largest cluster
            unique_labels = np.unique(labels)
            largest_cluster_size = 0
            largest_cluster_points = None
            
            for label in unique_labels:
                if label == -1:  # Skip noise
                    continue
                
                cluster_points = object_points[labels == label]
                if len(cluster_points) > largest_cluster_size:
                    largest_cluster_size = len(cluster_points)
                    largest_cluster_points = cluster_points
            
            if largest_cluster_points is None or len(largest_cluster_points) < 5:
                return {'valid_pose': False}
            
            center = np.mean(largest_cluster_points, axis=0)
            confidence = min(1.0, largest_cluster_size / 30.0)
            pixel_position = self.project_point_to_pixel(center, intrinsics)
            
            return {
                'valid_pose': True,
                'center': center,
                'confidence': confidence,
                'pixel_position': pixel_position,
                'point_count': largest_cluster_size,
                'method': 'largest_cluster'
            }
            
        except Exception:
            return {'valid_pose': False}
    
    def _detect_densest_region(self, object_points, intrinsics):
        """Detect object by finding the densest region."""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            nbrs = NearestNeighbors(radius=self.knn_radius)
            nbrs.fit(object_points)
            
            distances, indices = nbrs.radius_neighbors(object_points)
            neighbor_counts = [len(idx) for idx in indices]
            
            if not neighbor_counts or max(neighbor_counts) < self.knn_min_neighbors:
                return {'valid_pose': False}
            
            densest_idx = np.argmax(neighbor_counts)
            densest_neighbors = indices[densest_idx]
            dense_region_points = object_points[densest_neighbors]
            
            center = np.mean(dense_region_points, axis=0)
            confidence = min(1.0, len(dense_region_points) / 20.0)
            pixel_position = self.project_point_to_pixel(center, intrinsics)
            
            return {
                'valid_pose': True,
                'center': center,
                'confidence': confidence,
                'pixel_position': pixel_position,
                'point_count': len(dense_region_points),
                'method': 'densest_region'
            }
            
        except Exception:
            return {'valid_pose': False}
    
    def _detect_highest_point(self, object_points, intrinsics):
        """Detect object by finding the highest point region."""
        try:
            # Find highest 20% of points
            z_values = object_points[:, 2]
            threshold = np.percentile(z_values, 80)
            high_points = object_points[z_values >= threshold]
            
            if len(high_points) < 3:
                return {'valid_pose': False}
            
            center = np.mean(high_points, axis=0)
            confidence = min(1.0, len(high_points) / 15.0)
            pixel_position = self.project_point_to_pixel(center, intrinsics)
            
            return {
                'valid_pose': True,
                'center': center,
                'confidence': confidence,
                'pixel_position': pixel_position,
                'point_count': len(high_points),
                'method': 'highest_point'
            }
            
        except Exception:
            return {'valid_pose': False}
    
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
    
    def filter_points_above_surface(self, point_cloud, surface_plane, min_height=None, max_height=0.1):
        """Filter points that are above the detected surface plane."""
        if surface_plane is None:
            return np.array([])
        
        min_h = min_height or self.object_height_threshold
        a, b, c, d = surface_plane
        
        # Calculate distance from each point to the plane
        norm = np.sqrt(a**2 + b**2 + c**2)
        distances = np.abs(
            a * point_cloud[:, 0] + 
            b * point_cloud[:, 1] + 
            c * point_cloud[:, 2] + 
            d
        ) / norm
        
        # Filter points that are above the surface by the specified height range
        above_surface_mask = (distances >= min_h) & (distances <= max_height)
        
        return point_cloud[above_surface_mask]
    
    def create_empty_detection_result(self, point_cloud=None, surface_plane=None, full_point_cloud=None):
        """Create empty detection result structure."""
        return {
            'valid_pose': False,
            'center': None,
            'confidence': 0.0,
            'pixel_position': None,
            'point_count': 0,
            'object_points': np.array([]),
            'surface_plane': surface_plane,
            'full_point_cloud': full_point_cloud or point_cloud
        }
    
    def find_closest_object(self, detected_objects):
        """Find the closest detected object."""
        if not detected_objects or len(detected_objects) == 0:
            return None
        
        closest_object = None
        min_distance = float('inf')
        
        for obj in detected_objects:
            if obj.get('center') is not None:
                # Calculate distance from camera (assuming camera is at origin)
                distance = np.linalg.norm(obj['center'])
                if distance < min_distance:
                    min_distance = distance
                    closest_object = obj
        
        return closest_object
    
    def extract_object_position(self, obj):
        """Extract position information from an object detection result."""
        if obj is None or not obj.get('valid_pose', False):
            return None
        
        center = obj.get('center')
        if center is None:
            return None
        
        return {
            'position': center,
            'pixel_position': obj.get('pixel_position'),
            'confidence': obj.get('confidence', 0.0),
            'distance': np.linalg.norm(center),
            'method': obj.get('method', 'unknown')
        }