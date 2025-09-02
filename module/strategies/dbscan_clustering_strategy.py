"""
DBSCAN 3D clustering strategy extracted from 5_folder_recognition.py
"""

import open3d as o3d
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from module.core.clustering_strategy import ClusteringStrategy


class DBSCANClusteringStrategy(ClusteringStrategy):
    """
    3D DBSCAN-based object clustering for point clouds.
    
    This strategy:
    1. Applies floor removal and preprocessing 
    2. Performs noise filtering using statistical outlier removal
    3. Uses DBSCAN clustering to find 3D objects
    4. Filters clusters by size and dimensions
    """
    
    def __init__(self,
                 # Floor parameters
                 floor_plane_model=None,
                 rotation_matrix=None,
                 floor_level=None,
                 # Object detection parameters
                 roi_z_threshold=-0.55,
                 plane_distance_threshold=0.01,
                 plane_ransac_n=3,
                 plane_num_iterations=1000,
                 noise_nb_neighbors=30,
                 noise_std_ratio=0.1,
                 dbscan_eps=0.0001,
                 dbscan_min_points=10,
                 min_object_height=0.05,
                 max_object_height=1.0,
                 # Performance optimization parameters
                 max_points_for_processing=50000,
                 voxel_size_initial=0.003,
                 voxel_size_secondary=0.005):
        """
        Initialize DBSCAN clustering strategy.
        
        Args:
            floor_plane_model: Floor plane equation [a, b, c, d]
            rotation_matrix: Rotation matrix for floor alignment
            floor_level: Y-coordinate of floor level after rotation
            roi_z_threshold: Z threshold for ROI filtering
            plane_distance_threshold: Distance threshold for RANSAC plane detection
            plane_ransac_n: Number of points for RANSAC
            plane_num_iterations: Number of RANSAC iterations
            noise_nb_neighbors: Number of neighbors for noise removal
            noise_std_ratio: Standard deviation ratio for noise removal
            dbscan_eps: DBSCAN epsilon parameter
            dbscan_min_points: DBSCAN minimum points parameter
            min_object_height: Minimum object height to keep
            max_object_height: Maximum object height to keep
            max_points_for_processing: Maximum points before downsampling
            voxel_size_initial: Initial voxel size for downsampling
            voxel_size_secondary: Secondary voxel size for downsampling
        """
        # Floor parameters (can be computed automatically if not provided)
        self.floor_plane_model = floor_plane_model
        self.rotation_matrix = rotation_matrix if rotation_matrix is not None else np.eye(3)
        self.floor_level = floor_level
        self.floor_params_computed = floor_plane_model is not None
        
        # Object detection parameters
        self.roi_z_threshold = roi_z_threshold
        self.plane_distance_threshold = plane_distance_threshold
        self.plane_ransac_n = plane_ransac_n
        self.plane_num_iterations = plane_num_iterations
        self.noise_nb_neighbors = noise_nb_neighbors
        self.noise_std_ratio = noise_std_ratio
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points
        self.min_object_height = min_object_height
        self.max_object_height = max_object_height

        # Performance optimization parameters
        self.max_points_for_processing = max_points_for_processing
        self.voxel_size_initial = voxel_size_initial
        self.voxel_size_secondary = voxel_size_secondary
    
    def cluster_objects(self, vertices: np.ndarray, color_image: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Cluster objects using 3D DBSCAN after floor removal and preprocessing.
        
        Args:
            vertices: Point cloud vertices as numpy array of shape (N, 3)
            color_image: Optional color image (not used in this strategy)
            
        Returns:
            List of cluster dictionaries
        """
        if vertices is None or len(vertices) == 0:
            return []
        
        # Convert to Open3D point cloud for processing
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        
        # Process the point cloud
        clusters, bboxes, _, _ = self.process_single_frame_optimized(pcd)
        
        # Convert Open3D clusters to our standard format
        result_clusters = []
        for i, cluster_pcd in enumerate(clusters):
            points = np.asarray(cluster_pcd.points)
            
            if len(points) > 0:
                # Compute bounding box
                min_coords = np.min(points, axis=0)
                max_coords = np.max(points, axis=0)
                center = (min_coords + max_coords) / 2
                dimensions = max_coords - min_coords
                
                cluster_info = {
                    'id': i,
                    'points': points,
                    'center': center,
                    'min_coords': min_coords,
                    'max_coords': max_coords,
                    'dimensions': dimensions,
                    'num_points': len(points)
                }
                result_clusters.append(cluster_info)
        
        return result_clusters
    
    def get_strategy_name(self) -> str:
        """Get the name of this clustering strategy."""
        return "DBSCAN 3D Clustering"
    
    def compute_floor_parameters_optimized(self, pcd: o3d.geometry.PointCloud):
        """Optimized floor parameter computation using vectorized operations"""
        points = np.asarray(pcd.points)

        # Vectorized ROI filtering
        mask = points[:, 2] >= self.roi_z_threshold
        if not np.any(mask):
            raise ValueError("No points in ROI")

        # Use numpy indexing instead of Open3D selection
        filtered_points = points[mask]

        # Create temporary point cloud only for RANSAC
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # Detect floor using RANSAC
        plane_model, inliers = temp_pcd.segment_plane(
            distance_threshold=self.plane_distance_threshold,
            ransac_n=self.plane_ransac_n,
            num_iterations=self.plane_num_iterations,
        )

        [a, b, c, d] = plane_model
        print(f"Floor plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        # Optimized rotation matrix calculation
        normal_vector = np.array([a, b, c])
        normal_length = np.linalg.norm(normal_vector)
        if normal_length < 1e-6:
            return plane_model, np.identity(3)

        normal_vector /= normal_length
        target_normal = np.array([0, 1, 0])

        rotation_axis = np.cross(normal_vector, target_normal)
        rotation_axis_length = np.linalg.norm(rotation_axis)

        if rotation_axis_length < 1e-6:
            if np.dot(normal_vector, target_normal) < 0:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    np.array([np.pi, 0, 0])
                )
            else:
                R = np.identity(3)
        else:
            rotation_axis /= rotation_axis_length
            cos_angle = np.clip(np.dot(normal_vector, target_normal), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

        return plane_model, R
    
    def process_single_frame_optimized(self, pcd: o3d.geometry.PointCloud):
        """Optimized single frame processing with DBSCAN clustering"""
        if len(pcd.points) == 0:
            print(f"Warning: Empty point cloud")
            return [], [], [], pcd

        # Compute floor parameters if not already available
        if not self.floor_params_computed:
            try:
                self.floor_plane_model, self.rotation_matrix = self.compute_floor_parameters_optimized(pcd)
                
                # Compute floor level for faster floor removal
                points = np.asarray(pcd.points)
                mask = points[:, 2] >= self.roi_z_threshold
                if np.any(mask):
                    filtered_points = points[mask]
                    # Apply rotation to get floor level
                    rotated_points = (self.rotation_matrix @ filtered_points.T).T
                    self.floor_level = np.percentile(rotated_points[:, 1], 10)
                else:
                    self.floor_level = 0.0
                
                self.floor_params_computed = True
            except Exception as e:
                print(f"Warning: Could not compute floor parameters: {e}")
                # Use identity matrix as fallback
                self.rotation_matrix = np.eye(3)
                self.floor_level = 0.0

        # VOXEL DOWNSAMPLING: Use voxel downsampling instead of random subsampling
        if len(pcd.points) > self.max_points_for_processing:
            print(f"Large input cloud ({len(pcd.points)} points), voxel downsampling")
            # Calculate voxel size to achieve target point count
            voxel_size = self.voxel_size_initial  # 3mm voxels as starting point

            # Iteratively adjust voxel size to get close to target
            for _ in range(3):  # Max 3 iterations to find good voxel size
                temp_pcd = pcd.voxel_down_sample(voxel_size)
                if len(temp_pcd.points) <= self.max_points_for_processing:
                    break
                voxel_size *= 1.4  # Increase voxel size to reduce points

            pcd = temp_pcd

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        # Vectorized ROI filtering
        roi_mask = points[:, 2] >= self.roi_z_threshold
        if not np.any(roi_mask):
            print(f"Warning: No points above z={self.roi_z_threshold}")
            return [], [], [], pcd

        filtered_points = points[roi_mask]
        filtered_colors = colors[roi_mask] if colors is not None else None

        # Apply rotation using vectorized operation
        rotated_points = (self.rotation_matrix @ filtered_points.T).T

        # Optimized floor removal using cached floor level
        floor_height_threshold = 0.01
        non_floor_mask = rotated_points[:, 1] > (
            self.floor_level + floor_height_threshold
        )

        if not np.any(non_floor_mask):
            print(f"Warning: No points above floor")
            return [], [], [], pcd

        non_floor_points = rotated_points[non_floor_mask]
        non_floor_colors = (
            filtered_colors[non_floor_mask] if filtered_colors is not None else None
        )

        # SECOND VOXEL DOWNSAMPLING: Additional downsampling after floor removal if still too large
        if len(non_floor_points) > self.max_points_for_processing // 2:
            print(
                f"Large cloud after floor removal ({len(non_floor_points)} points), additional voxel downsampling"
            )
            # Create temporary point cloud for voxel downsampling
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(non_floor_points)
            if non_floor_colors is not None:
                temp_pcd.colors = o3d.utility.Vector3dVector(non_floor_colors)

            # Use smaller voxel size for this stage
            voxel_size = self.voxel_size_secondary  # 5mm voxels
            downsampled_pcd = temp_pcd.voxel_down_sample(voxel_size)

            non_floor_points = np.asarray(downsampled_pcd.points)
            non_floor_colors = (
                np.asarray(downsampled_pcd.colors)
                if downsampled_pcd.has_colors()
                else None
            )
            print(f"Further downsampled to {len(non_floor_points)} points")

        # Create point cloud only when necessary - skip noise removal for very small clouds
        if len(non_floor_points) < self.noise_nb_neighbors:
            print(
                f"Warning: Too few points ({len(non_floor_points)}) for noise removal"
            )
            final_points = non_floor_points
            final_colors = non_floor_colors
        else:
            # OPTIMIZED NOISE REMOVAL
            effective_neighbors = self.noise_nb_neighbors

            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(non_floor_points)
            if non_floor_colors is not None:
                temp_pcd.colors = o3d.utility.Vector3dVector(non_floor_colors)

            # Statistical outlier removal
            clean_pcd, clean_indices = temp_pcd.remove_statistical_outlier(
                nb_neighbors=effective_neighbors,
                std_ratio=self.noise_std_ratio,
            )

            if len(clean_pcd.points) == 0:
                print(f"Warning: No points left after noise removal")
                return [], [], [], pcd

            final_points = np.asarray(clean_pcd.points)
            final_colors = (
                np.asarray(clean_pcd.colors) if clean_pcd.has_colors() else None
            )

        # OPTIMIZED DBSCAN
        temp_pcd_cluster = o3d.geometry.PointCloud()
        temp_pcd_cluster.points = o3d.utility.Vector3dVector(final_points)

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            labels = np.array(
                temp_pcd_cluster.cluster_dbscan(
                    eps=self.dbscan_eps, min_points=self.dbscan_min_points
                )
            )

        if len(labels) == 0 or labels.max() < 0:
            print(f"Warning: No clusters found")
            return [], [], [], pcd

        # Vectorized cluster filtering
        valid_clusters = []
        valid_bboxes = []
        text_labels = []
        max_label = labels.max()

        # Pre-compute all cluster properties
        for i, cluster_id in enumerate(range(max_label + 1)):
            cluster_mask = labels == cluster_id
            if not np.any(cluster_mask):
                continue

            cluster_points = final_points[cluster_mask]

            # Vectorized bounding box calculation
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)

            # Calculate dimensions
            height = max_bounds[1] - min_bounds[1]
            width = max_bounds[0] - min_bounds[0]
            depth = max_bounds[2] - min_bounds[2]

            # Filter by dimensions
            if (
                self.min_object_height <= height <= self.max_object_height
                and width >= 0.01
                and depth >= 0.01
            ):

                # Create cluster point cloud only for valid objects
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                if final_colors is not None:
                    cluster_colors = final_colors[cluster_mask]
                    cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

                # Create bounding box
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bounds, max_bounds)
                bbox.color = (0, 1, 0)

                valid_clusters.append(cluster_pcd)
                valid_bboxes.append(bbox)
                text_labels.append(f"Obj_{len(valid_clusters):02d}")

        return valid_clusters, valid_bboxes, text_labels, pcd
    
    def set_floor_parameters(self, floor_plane_model, rotation_matrix, floor_level):
        """Set floor parameters manually"""
        self.floor_plane_model = floor_plane_model
        self.rotation_matrix = rotation_matrix
        self.floor_level = floor_level
        self.floor_params_computed = True