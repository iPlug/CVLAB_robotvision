"""
Point cloud processing utilities shared across different strategies.
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional
from sklearn.neighbors import NearestNeighbors


class PointCloudProcessor:
    """
    Common point cloud processing operations used by different clustering strategies.
    
    Provides utilities for:
    - ROI filtering
    - Statistical outlier removal
    - Coordinate transformations
    - Gravity-based alignment
    - Voxel downsampling
    """
    
    def __init__(self):
        # Rotation matrix for alignment (identity initially)
        self.rotation_matrix = np.eye(3)
        self.gravity_vector = np.array([0, -1, 0])  # Default gravity pointing down
        self.alignment_calibrated = False
    
    def apply_roi_filter(self, vertices: np.ndarray, roi_params: dict) -> np.ndarray:
        """
        Apply ROI filtering to point cloud.
        
        Args:
            vertices: Point cloud vertices as numpy array (N, 3)
            roi_params: Dictionary with ROI parameters containing:
                - x_min, x_max: X-axis bounds
                - y_min, y_max: Y-axis bounds  
                - z_min, z_max: Z-axis bounds
                
        Returns:
            Filtered vertices within ROI
        """
        if vertices is None or len(vertices) == 0:
            return vertices
        
        # Apply ROI filtering
        mask = ((vertices[:, 0] >= roi_params['x_min']) & 
                (vertices[:, 0] <= roi_params['x_max']) &
                (vertices[:, 1] >= roi_params['y_min']) & 
                (vertices[:, 1] <= roi_params['y_max']) &
                (vertices[:, 2] >= roi_params['z_min']) & 
                (vertices[:, 2] <= roi_params['z_max']))
        
        return vertices[mask]
    
    def apply_statistical_outlier_filter(self, vertices: np.ndarray, 
                                       k_neighbors: int = 20, 
                                       std_ratio: float = 2.0) -> np.ndarray:
        """
        Remove statistical outliers based on distance to neighbors.
        
        Args:
            vertices: Point cloud vertices
            k_neighbors: Number of neighbors to consider
            std_ratio: Standard deviation ratio for outlier detection
            
        Returns:
            Filtered vertices with outliers removed
        """
        if vertices is None or len(vertices) < k_neighbors:
            return vertices
        
        # Calculate distances to k nearest neighbors for each point
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
    
    def voxel_downsample(self, vertices: np.ndarray, 
                        voxel_size: float,
                        colors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Downsample point cloud using voxel grid.
        
        Args:
            vertices: Point cloud vertices
            voxel_size: Size of voxel grid
            colors: Optional color information
            
        Returns:
            Tuple of (downsampled_vertices, downsampled_colors)
        """
        if vertices is None or len(vertices) == 0:
            return vertices, colors
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Downsample
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        
        # Extract results
        downsampled_vertices = np.asarray(downsampled_pcd.points)
        downsampled_colors = np.asarray(downsampled_pcd.colors) if downsampled_pcd.has_colors() else None
        
        return downsampled_vertices, downsampled_colors
    
    def calculate_alignment_rotation(self, from_vec: np.ndarray, to_vec: np.ndarray) -> np.ndarray:
        """
        Calculate rotation matrix to align from_vec to to_vec.
        
        Args:
            from_vec: Source vector
            to_vec: Target vector
            
        Returns:
            3x3 rotation matrix
        """
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
    
    def set_gravity_alignment(self, gravity_vector: np.ndarray):
        """
        Set gravity-based alignment from IMU data.
        
        Args:
            gravity_vector: Normalized gravity vector from accelerometer
        """
        # Calculate rotation matrix to align gravity with -Y axis
        target_up = np.array([0, -1, 0])  # Y-up in camera coordinates
        self.rotation_matrix = self.calculate_alignment_rotation(gravity_vector, target_up)
        self.gravity_vector = gravity_vector
        self.alignment_calibrated = True
    
    def apply_alignment(self, vertices: np.ndarray) -> np.ndarray:
        """
        Apply gravity-based alignment to point cloud.
        
        Args:
            vertices: Point cloud vertices
            
        Returns:
            Aligned vertices
        """
        if vertices is None or len(vertices) == 0:
            return vertices
        
        if not self.alignment_calibrated:
            return vertices
        
        # Apply rotation to all points
        rotated_vertices = np.dot(vertices, self.rotation_matrix.T)
        return rotated_vertices
    
    def compute_floor_plane(self, vertices: np.ndarray,
                           distance_threshold: float = 0.01,
                           ransac_n: int = 3,
                           num_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute floor plane using RANSAC.
        
        Args:
            vertices: Point cloud vertices
            distance_threshold: Distance threshold for RANSAC
            ransac_n: Number of points for RANSAC
            num_iterations: Number of RANSAC iterations
            
        Returns:
            Tuple of (plane_model, inlier_indices)
        """
        if vertices is None or len(vertices) < 3:
            return np.array([0, 1, 0, 0]), np.array([])
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        
        # Detect floor using RANSAC
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        
        return np.array(plane_model), np.array(inliers)
    
    def remove_floor_points(self, vertices: np.ndarray,
                          plane_model: np.ndarray,
                          distance_threshold: float = 0.01) -> np.ndarray:
        """
        Remove floor points based on plane model.
        
        Args:
            vertices: Point cloud vertices
            plane_model: Plane equation [a, b, c, d]
            distance_threshold: Distance threshold for floor removal
            
        Returns:
            Vertices with floor points removed
        """
        if vertices is None or len(vertices) == 0:
            return vertices
        
        # Calculate distances to plane
        a, b, c, d = plane_model
        distances = np.abs(a * vertices[:, 0] + b * vertices[:, 1] + c * vertices[:, 2] + d) / np.sqrt(a*a + b*b + c*c)
        
        # Keep points that are far enough from the floor plane
        non_floor_mask = distances > distance_threshold
        
        return vertices[non_floor_mask]
    
    def compute_bounding_box(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute axis-aligned bounding box for point cloud.
        
        Args:
            vertices: Point cloud vertices
            
        Returns:
            Tuple of (min_coords, max_coords, center, dimensions)
        """
        if vertices is None or len(vertices) == 0:
            return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        center = (min_coords + max_coords) / 2
        dimensions = max_coords - min_coords
        
        return min_coords, max_coords, center, dimensions
    
    def reset_alignment(self):
        """Reset alignment to identity matrix."""
        self.rotation_matrix = np.eye(3)
        self.gravity_vector = np.array([0, -1, 0])
        self.alignment_calibrated = False