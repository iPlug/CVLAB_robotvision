"""
BEV (Bird's Eye View) clustering strategy extracted from realsense_bev_cluster.py
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Any, Optional, Tuple
from scipy import ndimage
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from module.core.clustering_strategy import ClusteringStrategy


class BEVClusteringStrategy(ClusteringStrategy):
    """
    BEV-based object clustering using image projection and connected components.
    
    This strategy:
    1. Projects 3D points to Bird's Eye View (BEV) image
    2. Applies noise reduction and morphological operations
    3. Uses connected components to find clusters
    4. Maps image clusters back to 3D point clouds
    """
    
    def __init__(self,
                 # ROI parameters (in meters, after rotation) - for filtering points
                 roi_params=None,
                 # Display ROI parameters (in meters) - for limiting BEV display area  
                 display_roi_params=None,
                 # BEV clustering parameters
                 clustering_params=None,
                 # Temporal filtering
                 enable_temporal_filtering=True):
        """
        Initialize BEV clustering strategy.
        
        Args:
            roi_params: ROI parameters for point filtering
            display_roi_params: Display area parameters for BEV
            clustering_params: Clustering-specific parameters
            enable_temporal_filtering: Whether to use temporal consistency
        """
        
        # ROI parameters (in meters, after rotation) - for filtering points
        # camera on table
        # self.roi_params = roi_params or {
        #     'x_min': -1.0, 'x_max': 1.0,
        #     'y_min': 0.08, 'y_max': .135,  # Height range
        #     'z_min': 0, 'z_max': .5    # Depth range
        # }

        # camera on end effector
        self.roi_params = roi_params or {
            'x_min': -1.0, 'x_max': 1.0,
            'y_min': 0.30, 'y_max': .38,  # Height range
            'z_min': 0, 'z_max': .5    # Depth range
        }
        
        # Display ROI parameters (in meters) - for limiting BEV display area
        self.display_roi_params = display_roi_params or {
            'x_min': -1.5, 'x_max': 1.5,
            'y_min': -0.8, 'y_max': 0.5,  # Height range
            'z_min': 0, 'z_max': 1.0    # Depth range
        }
        
        # BEV clustering parameters
        self.clustering_params = clustering_params or {
            'min_cluster_size': 30,  # Minimum pixels for a valid cluster
            'max_cluster_size': 150,  # Maximum pixels for a valid cluster (filter out large surfaces)
            'morphology_kernel_size': 5,  # Kernel size for morphological operations
            'window_size': 5,  # Number of frames to average for noise reduction
        }
        
        # 3D Statistical outlier filtering parameters
        self.outlier_params = {
            'enabled': True,  # Enable/disable 3D statistical outlier removal
            'k_neighbors': 15,  # Number of neighbors to consider
            'std_ratio': 0.1,   # Standard deviation ratio for outlier detection
            'cluster_std_threshold': 1.5,  # Std threshold for cluster-level filtering
            'min_points': 5     # Minimum points required after filtering
        }
        
        # Frame history for temporal filtering
        self.enable_temporal_filtering = enable_temporal_filtering
        self.frame_history = []
        
    def cluster_objects(self, vertices: np.ndarray, color_image: Optional[np.ndarray] = None, 
                       return_visualization: bool = False) -> List[Dict[str, Any]]:
        """
        Cluster objects using BEV projection and connected components.
        
        Args:
            vertices: Point cloud vertices as numpy array of shape (N, 3)
            color_image: Optional color image (not used in this strategy)
            return_visualization: Whether to return visualization images
            
        Returns:
            List of cluster dictionaries, or tuple of (clusters, bev_image, point_cloud_view) if return_visualization=True
        """
        if vertices is None or len(vertices) == 0:
            return []
        
        # Apply ROI filtering
        filtered_vertices = self.apply_roi_filter(vertices)
        if len(filtered_vertices) == 0:
            return []
        
        # Apply 3D statistical outlier filtering (remove isolated points)
        if self.outlier_params['enabled'] and len(filtered_vertices) > 0:
            filtered_vertices = self.apply_3d_statistical_outlier_filter(filtered_vertices)
        
        # Calculate ranges for consistent coordinate system
        valid_mask = filtered_vertices[:, 2] > 0
        valid_points = filtered_vertices[valid_mask]
        
        if len(valid_points) == 0:
            return []
        
        x_coords = valid_points[:, 0]
        z_coords = valid_points[:, 2]
        
        # Use display ROI bounds for consistent BEV dimensions
        x_range = (self.display_roi_params['x_min'], self.display_roi_params['x_max'])
        z_range = (self.display_roi_params['z_min'], self.display_roi_params['z_max'])
        
        # Create smooth BEV image
        bev_image = self.create_smooth_bev_image(filtered_vertices, x_range, z_range)
        
        # Apply noise reduction and morphological operations
        cleaned_bev = self.apply_realsense_noise_reduction(bev_image)
        
        # Apply temporal filtering to reduce frame-to-frame noise
        if self.enable_temporal_filtering:
            filtered_bev = self.apply_temporal_filtering(cleaned_bev)
        else:
            filtered_bev = cleaned_bev
        
        # Perform clustering on the filtered BEV image
        clusters = self.cluster_from_cleaned_bev(filtered_bev, filtered_vertices, x_range, z_range)
        
        # Apply cluster-level statistical outlier removal
        if self.outlier_params['enabled'] and clusters:
            clusters = self.remove_statistical_outliers_from_clusters(clusters)
        
        # Return visualizations if requested
        if return_visualization:
            # Create BEV visualization with cluster overlays
            bev_visualization = self.create_bev_visualization_with_clusters(filtered_vertices, clusters, x_range, z_range)
            
            # Create point cloud side view
            point_cloud_view = self.create_point_cloud_side_view(filtered_vertices)
            
            return clusters, bev_visualization, point_cloud_view
        
        return clusters
    
    def get_strategy_name(self) -> str:
        """Get the name of this clustering strategy."""
        return "BEV Image Clustering"
    
    def apply_roi_filter(self, vertices: np.ndarray) -> np.ndarray:
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
    
    def apply_3d_statistical_outlier_filter(self, vertices: np.ndarray) -> np.ndarray:
        """Remove 3D statistical outliers based on distance to neighbors"""
        if vertices is None or len(vertices) < self.outlier_params['k_neighbors']:
            return vertices
        
        k_neighbors = min(self.outlier_params['k_neighbors'], len(vertices) - 1)
        if k_neighbors <= 0:
            return vertices
            
        std_ratio = self.outlier_params['std_ratio']
        
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
    
    def create_smooth_bev_image(self, vertices: np.ndarray, x_range: Tuple[float, float], z_range: Tuple[float, float]) -> np.ndarray:
        """Create smooth BEV image using point cloud projection"""
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
            
            # Apply simple colormap (red for high, blue for low)
            colors = np.zeros((len(y_normalized), 3))
            colors[:, 0] = y_normalized * 255  # Red channel
            colors[:, 2] = (1 - y_normalized) * 255  # Blue channel
            colors[:, 1] = 128  # Green channel (constant)
            
            # Draw points
            point_size = max(1, min(3, max_img_size // 200))
            
            for i in range(len(x_img)):
                cv2.circle(bev_image, (x_img[i], img_height - z_img[i] - 1), point_size, 
                          (int(colors[i][2]), int(colors[i][1]), int(colors[i][0])), -1)
        
        return bev_image
    
    def apply_realsense_noise_reduction(self, bev_image: np.ndarray) -> np.ndarray:
        """Apply noise reduction using morphological operations"""
        # Convert to grayscale for binary thresholding
        binary_bev = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold (any non-zero pixel becomes 255)
        _, binary_bev = cv2.threshold(binary_bev, 1, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up the BEV
        kernel_size = self.clustering_params['morphology_kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_bev = cv2.morphologyEx(binary_bev, cv2.MORPH_OPEN, kernel)
        binary_bev = cv2.morphologyEx(binary_bev, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to BGR for consistency
        cleaned_bev = cv2.cvtColor(binary_bev, cv2.COLOR_GRAY2BGR)
        
        return cleaned_bev
    
    def apply_temporal_filtering(self, cleaned_bev: np.ndarray) -> np.ndarray:
        """Apply temporal filtering - only keep pixels that appear in multiple frames"""
        # Add current frame to history
        self.frame_history.append(cleaned_bev.copy())
        
        # Maintain window size
        window_size = self.clustering_params['window_size']
        if len(self.frame_history) > window_size:
            self.frame_history.pop(0)
        
        # If we don't have enough frames yet, return current frame
        if len(self.frame_history) < min(3, window_size):  # Need at least 3 frames
            return cleaned_bev
        
        # Convert all frames to grayscale binary
        binary_frames = []
        for frame in self.frame_history:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            binary_frames.append(binary)
        
        # Stack frames and find pixels that are white in most frames
        frame_stack = np.stack(binary_frames, axis=0)
        
        # Keep pixels that are 255 in at least 70% of frames
        threshold_ratio = 0.7
        min_frames = int(len(binary_frames) * threshold_ratio)
        persistent_pixels = np.sum(frame_stack == 255, axis=0) >= min_frames
        
        # Create output image
        result = np.zeros_like(cleaned_bev)
        result[persistent_pixels] = 255
        
        return result
    
    def cluster_from_cleaned_bev(self, cleaned_bev: np.ndarray, vertices: np.ndarray, 
                                x_range: Tuple[float, float], z_range: Tuple[float, float]) -> List[Dict[str, Any]]:
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
    
    def map_image_pixels_to_3d(self, pixel_mask: np.ndarray, vertices: np.ndarray, 
                              x_range: Tuple[float, float], z_range: Tuple[float, float]) -> np.ndarray:
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
    
    def remove_statistical_outliers_from_clusters(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fast statistical outlier removal: use cluster info → select 3D points → filter → validate"""
        if not clusters:
            return clusters
        
        cleaned_clusters = []
        std_threshold = self.outlier_params['cluster_std_threshold']
        min_points = self.outlier_params['min_points']
        
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
    
    def create_bev_visualization_with_clusters(self, vertices: np.ndarray, clusters: List[Dict[str, Any]], 
                                               x_range: Tuple[float, float], z_range: Tuple[float, float]) -> np.ndarray:
        """Create BEV visualization with cluster overlays"""
        # Create base BEV image
        bev_image = self.create_smooth_bev_image(vertices, x_range, z_range)
        
        # Draw cluster bounding boxes and information
        return self.draw_clusters_on_bev(bev_image, clusters, x_range, z_range)
    
    def draw_clusters_on_bev(self, bev_image: np.ndarray, clusters: List[Dict[str, Any]], 
                            x_range: Tuple[float, float], z_range: Tuple[float, float]) -> np.ndarray:
        """Draw cluster bounding boxes on BEV image"""
        if not clusters:
            return bev_image
        
        height, width = bev_image.shape[:2]
        result_image = bev_image.copy()
        
        # Define colors for clusters
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, cluster in enumerate(clusters):
            color = colors[i % len(colors)]
            
            # Project bounding box corners to image coordinates
            min_coords = cluster['min_coords']
            max_coords = cluster['max_coords']
            
            # Convert 3D coordinates to image coordinates
            x_min_img = int((min_coords[0] - x_range[0]) / (x_range[1] - x_range[0]) * width)
            x_max_img = int((max_coords[0] - x_range[0]) / (x_range[1] - x_range[0]) * width)
            z_min_img = int((min_coords[2] - z_range[0]) / (z_range[1] - z_range[0]) * height)
            z_max_img = int((max_coords[2] - z_range[0]) / (z_range[1] - z_range[0]) * height)
            
            # Flip z coordinates for image display
            z_min_img = height - z_min_img - 1
            z_max_img = height - z_max_img - 1
            
            # Ensure coordinates are within image bounds
            x_min_img = max(0, min(width-1, x_min_img))
            x_max_img = max(0, min(width-1, x_max_img))
            z_min_img = max(0, min(height-1, z_min_img))
            z_max_img = max(0, min(height-1, z_max_img))
            
            # Draw bounding box rectangle
            cv2.rectangle(result_image, (x_min_img, z_max_img), (x_max_img, z_min_img), color, 2)
            
            # Draw center point
            center_x = int((cluster['center'][0] - x_range[0]) / (x_range[1] - x_range[0]) * width)
            center_z = int((cluster['center'][2] - z_range[0]) / (z_range[1] - z_range[0]) * height)
            center_z = height - center_z - 1  # Flip z coordinate
            
            center_x = max(0, min(width-1, center_x))
            center_z = max(0, min(height-1, center_z))
            
            # Draw white dot at center
            cv2.circle(result_image, (center_x, center_z), 3, (255, 255, 255), -1)
            
            # Display 3D coordinates in millimeters (forward, side, height)
            coord_text = f"({cluster['center'][2]*1000:.0f}, {cluster['center'][0]*1000:.0f}, {cluster['center'][1]*1000:.0f})"
            cv2.putText(result_image, coord_text, (center_x + 5, center_z - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Cluster ID removed for cleaner visualization
        
        return result_image
    
    def create_point_cloud_side_view(self, vertices: np.ndarray) -> np.ndarray:
        """Create side view of point cloud"""
        if vertices is None or len(vertices) == 0:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        img_size = 400
        
        # Use X and Y coordinates for side view
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        z_coords = vertices[:, 2]  # For coloring
        
        # Filter valid points
        valid_mask = z_coords > 0
        if not np.any(valid_mask):
            return np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        z_valid = z_coords[valid_mask]
        
        # Calculate bounds with padding
        x_min, x_max = np.min(x_valid), np.max(x_valid)
        y_min, y_max = np.min(y_valid), np.max(y_valid)
        
        x_padding = max(0.1, (x_max - x_min) * 0.1)
        y_padding = max(0.1, (y_max - y_min) * 0.1)
        
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        # Convert to image coordinates
        if x_max > x_min and y_max > y_min:
            x_img = ((x_valid - x_min) / (x_max - x_min) * img_size).astype(int)
            y_img = ((y_valid - y_min) / (y_max - y_min) * img_size).astype(int)
            y_img = img_size - y_img - 1  # Flip Y
            
            # Create image
            pc_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            # Color by depth
            if len(z_valid) > 0:
                z_min, z_max = np.min(z_valid), np.max(z_valid)
                if z_max > z_min:
                    z_normalized = (z_valid - z_min) / (z_max - z_min)
                    # Simple colormap: red for near, blue for far
                    colors = np.zeros((len(z_normalized), 3))
                    colors[:, 0] = z_normalized * 255  # Red channel
                    colors[:, 2] = (1 - z_normalized) * 255  # Blue channel
                else:
                    colors = np.full((len(z_valid), 3), [255, 255, 255], dtype=np.uint8)
                
                # Draw points
                valid_img_mask = (x_img >= 0) & (x_img < img_size) & (y_img >= 0) & (y_img < img_size)
                for i in np.where(valid_img_mask)[0]:
                    cv2.circle(pc_image, (x_img[i], y_img[i]), 1, 
                             (int(colors[i][2]), int(colors[i][1]), int(colors[i][0])), -1)
            
            # Add labels
            cv2.putText(pc_image, "Point Cloud (Side View)", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(pc_image, f"X: [{x_min:.2f}, {x_max:.2f}]m", (10, img_size-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(pc_image, f"Y: [{y_min:.2f}, {y_max:.2f}]m", (10, img_size-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return pc_image
        
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    def reset_temporal_state(self):
        """Reset temporal filtering state (useful when switching scenes)"""
        self.frame_history = []