import open3d as o3d
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

class EdgeAwareGrabPointProposer:
    """Enhanced grab point proposer using edge detection to identify stable surface points"""
    
    def __init__(self, min_surface_area=0.001, max_surface_area=0.1, 
                 min_points_per_surface=20, plane_distance_threshold=0.005,
                 plane_ransac_iterations=100, edge_detection_method='normal_variation',
                 normal_variation_threshold=0.3, boundary_k_neighbors=20,
                 min_distance_from_edge=0.01):
        """
        Initialize edge-aware grab point proposer
        
        Args:
            edge_detection_method: 'normal_variation', 'boundary', 'curvature', or 'combined'
            normal_variation_threshold: Threshold for normal variation edge detection
            boundary_k_neighbors: Number of neighbors for boundary detection
            min_distance_from_edge: Minimum distance grab points should be from edges
        """
        self.min_surface_area = min_surface_area
        self.max_surface_area = max_surface_area
        self.min_points_per_surface = min_points_per_surface
        self.plane_distance_threshold = plane_distance_threshold
        self.plane_ransac_iterations = plane_ransac_iterations
        
        # Edge detection parameters
        self.edge_detection_method = edge_detection_method
        self.normal_variation_threshold = normal_variation_threshold
        self.boundary_k_neighbors = boundary_k_neighbors
        self.min_distance_from_edge = min_distance_from_edge
    
    def detect_edges_normal_variation(self, pcd, k_neighbors=20):
        """
        Detect edges based on normal vector variation
        Points with high normal variation in neighborhood are likely edges
        """
        points = np.asarray(pcd.points)
        
        # Estimate normals if not available
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
        
        normals = np.asarray(pcd.normals)
        
        # Build KD-tree for neighbor search
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        edge_scores = np.zeros(len(points))
        
        for i in range(len(points)):
            # Get normals of neighbors
            neighbor_normals = normals[indices[i]]
            center_normal = normals[i]
            
            # Calculate variation in normal directions
            # Use dot product to measure angular difference
            dot_products = np.abs(np.dot(neighbor_normals, center_normal))
            dot_products = np.clip(dot_products, -1, 1)
            
            # Convert to angles and calculate variation
            angles = np.arccos(dot_products)
            normal_variation = np.std(angles)
            
            edge_scores[i] = normal_variation
        
        # Classify as edge if variation exceeds threshold
        is_edge = edge_scores > self.normal_variation_threshold
        
        return is_edge, edge_scores
    
    def detect_edges_boundary(self, pcd):
        """
        Detect boundary edges - points on the boundary of the object
        Uses Open3D's boundary detection
        """
        points = np.asarray(pcd.points)
        
        # Method 1: Use point density variation
        nbrs = NearestNeighbors(n_neighbors=self.boundary_k_neighbors, algorithm='kd_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        # Calculate local density (inverse of average distance to neighbors)
        local_densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-8)
        
        # Normalize densities
        density_mean = np.mean(local_densities)
        density_std = np.std(local_densities)
        normalized_densities = (local_densities - density_mean) / (density_std + 1e-8)
        
        # Points with significantly lower density are likely on boundaries
        boundary_threshold = -1.0  # Points 1 std below mean density
        is_boundary = normalized_densities < boundary_threshold
        
        return is_boundary, normalized_densities
    
    def detect_edges_curvature(self, pcd):
        """
        Detect edges based on surface curvature
        High curvature points are likely edges
        """
        # Estimate normals if not available
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        # Build KD-tree
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='kd_tree').fit(points)
        distances, indices = nbrs.kneighbors(points)
        
        curvatures = np.zeros(len(points))
        
        for i in range(len(points)):
            neighbors = points[indices[i][1:]]  # Exclude self
            center = points[i]
            center_normal = normals[i]
            
            # Project neighbors onto tangent plane
            relative_pos = neighbors - center
            projected = relative_pos - np.outer(np.dot(relative_pos, center_normal), center_normal)
            
            # Estimate curvature using eigenvalues of covariance matrix
            if len(projected) > 3:
                cov_matrix = np.cov(projected.T)
                eigenvals = np.linalg.eigvals(cov_matrix)
                curvatures[i] = np.max(eigenvals) / (np.sum(eigenvals) + 1e-8)
        
        # High curvature indicates edges
        curvature_threshold = np.percentile(curvatures, 80)  # Top 20% as edges
        is_edge = curvatures > curvature_threshold
        
        return is_edge, curvatures
    
    def detect_edges_combined(self, pcd):
        """
        Combine multiple edge detection methods for robust edge identification
        """
        # Get results from different methods
        edge_normal, scores_normal = self.detect_edges_normal_variation(pcd)
        edge_boundary, scores_boundary = self.detect_edges_boundary(pcd)
        edge_curvature, scores_curvature = self.detect_edges_curvature(pcd)
        
        # Normalize scores to [0, 1]
        scores_normal_norm = (scores_normal - np.min(scores_normal)) / (np.max(scores_normal) - np.min(scores_normal) + 1e-8)
        scores_boundary_norm = (scores_boundary - np.min(scores_boundary)) / (np.max(scores_boundary) - np.min(scores_boundary) + 1e-8)
        scores_curvature_norm = (scores_curvature - np.min(scores_curvature)) / (np.max(scores_curvature) - np.min(scores_curvature) + 1e-8)
        
        # Invert boundary scores (low density = edge)
        scores_boundary_norm = 1.0 - scores_boundary_norm
        
        # Combine scores with weights
        combined_scores = (0.4 * scores_normal_norm + 
                          0.3 * scores_boundary_norm + 
                          0.3 * scores_curvature_norm)
        
        # Threshold for final edge classification
        edge_threshold = 0.6
        is_edge = combined_scores > edge_threshold
        
        return is_edge, combined_scores
    
    def detect_edges(self, pcd):
        """
        Main edge detection function - dispatches to specific method
        """
        if self.edge_detection_method == 'normal_variation':
            return self.detect_edges_normal_variation(pcd)
        elif self.edge_detection_method == 'boundary':
            return self.detect_edges_boundary(pcd)
        elif self.edge_detection_method == 'curvature':
            return self.detect_edges_curvature(pcd)
        elif self.edge_detection_method == 'combined':
            return self.detect_edges_combined(pcd)
        else:
            raise ValueError(f"Unknown edge detection method: {self.edge_detection_method}")
    
    def filter_surface_points(self, pcd, surface_indices, edge_mask):
        """
        Filter surface points to exclude those near edges
        """
        points = np.asarray(pcd.points)
        surface_points = points[surface_indices]
        
        # Get edge points
        edge_points = points[edge_mask]
        
        if len(edge_points) == 0:
            return surface_indices  # No edges detected, return all surface points
        
        # Calculate distances from surface points to nearest edge points
        distances = cdist(surface_points, edge_points)
        min_distances_to_edge = np.min(distances, axis=1)
        
        # Filter out surface points too close to edges
        far_from_edge = min_distances_to_edge > self.min_distance_from_edge
        
        # Return filtered indices
        filtered_indices = surface_indices[far_from_edge]
        
        return filtered_indices
    
    def find_flat_surfaces_edge_aware(self, cluster_pcd):
        """
        Enhanced flat surface detection that avoids edges
        """
        if len(cluster_pcd.points) < self.min_points_per_surface:
            return []
        
        print(f"    Detecting edges in cluster with {len(cluster_pcd.points)} points...")
        
        # Detect edges first
        is_edge, edge_scores = self.detect_edges(cluster_pcd)
        num_edges = np.sum(is_edge)
        print(f"    Detected {num_edges} edge points ({num_edges/len(cluster_pcd.points)*100:.1f}%)")
        
        points = np.asarray(cluster_pcd.points)
        remaining_points = points.copy()
        remaining_indices = np.arange(len(points))
        surfaces = []
        
        # Track edge mask for remaining points
        remaining_edge_mask = is_edge.copy()
        
        # Iteratively find planes, avoiding edge points
        while len(remaining_points) >= self.min_points_per_surface:
            # Create temporary point cloud for RANSAC
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(remaining_points)
            
            try:
                # Find the largest plane using RANSAC
                plane_model, inlier_indices = temp_pcd.segment_plane(
                    distance_threshold=self.plane_distance_threshold,
                    ransac_n=3,
                    num_iterations=self.plane_ransac_iterations
                )
                
                if len(inlier_indices) < self.min_points_per_surface:
                    break
                
                # Map back to original indices
                original_inlier_indices = remaining_indices[inlier_indices]
                
                # Filter out points near edges
                filtered_indices = self.filter_surface_points(
                    cluster_pcd, original_inlier_indices, is_edge
                )
                
                if len(filtered_indices) < self.min_points_per_surface:
                    # Remove these points and continue
                    remaining_indices = remaining_indices[~np.isin(remaining_indices, original_inlier_indices)]
                    if len(remaining_indices) == 0:
                        break
                    remaining_points = points[remaining_indices]
                    remaining_edge_mask = is_edge[remaining_indices]
                    continue
                
                # Extract filtered plane points
                plane_points = points[filtered_indices]
                
                # Calculate surface properties
                surface_area = self._calculate_surface_area(plane_points)
                
                # Filter by surface area
                if self.min_surface_area <= surface_area <= self.max_surface_area:
                    # Calculate surface center and normal
                    center = np.mean(plane_points, axis=0)
                    normal = np.array(plane_model[:3])
                    normal = normal / np.linalg.norm(normal)
                    
                    # Ensure normal points outward
                    object_center = np.mean(points, axis=0)
                    if np.dot(normal, center - object_center) < 0:
                        normal = -normal
                    
                    # Calculate average distance from edges for this surface
                    edge_points = points[is_edge]
                    if len(edge_points) > 0:
                        distances_to_edges = cdist([center], edge_points)
                        avg_distance_from_edge = np.mean(distances_to_edges)
                    else:
                        avg_distance_from_edge = float('inf')
                    
                    surface = {
                        'points': plane_points,
                        'center': center,
                        'normal': normal,
                        'area': surface_area,
                        'plane_model': plane_model,
                        'num_points': len(plane_points),
                        'avg_distance_from_edge': avg_distance_from_edge,
                        'edge_filtered': True
                    }
                    surfaces.append(surface)
                
                # Remove inlier points from remaining points
                remaining_indices = remaining_indices[~np.isin(remaining_indices, original_inlier_indices)]
                if len(remaining_indices) == 0:
                    break
                remaining_points = points[remaining_indices]
                remaining_edge_mask = is_edge[remaining_indices]
                
            except Exception as e:
                print(f"    Error in edge-aware plane detection: {e}")
                break
        
        print(f"    Found {len(surfaces)} edge-filtered surfaces")
        return surfaces
    
    def _calculate_surface_area(self, points):
        """Estimate surface area using convex hull"""
        if len(points) < 3:
            return 0.0
        
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            hull, _ = pcd.compute_convex_hull()
            
            triangles = np.asarray(hull.triangles)
            vertices = np.asarray(hull.vertices)
            
            total_area = 0.0
            for triangle in triangles:
                v0, v1, v2 = vertices[triangle]
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                total_area += area
            
            return total_area
            
        except Exception:
            # Fallback
            min_bounds = np.min(points, axis=0)
            max_bounds = np.max(points, axis=0)
            dimensions = max_bounds - min_bounds
            sorted_dims = np.sort(dimensions)
            return sorted_dims[-1] * sorted_dims[-2]
    
    def propose_grab_points_edge_aware(self, cluster_pcd, filter_orientation=True):
        """
        Propose grab points using edge-aware surface detection
        """
        surfaces = self.find_flat_surfaces_edge_aware(cluster_pcd)
        
        if not surfaces:
            return []
        
        grab_points = []
        
        for i, surface in enumerate(surfaces):
            center = surface['center']
            normal = surface['normal']
            area = surface['area']
            
            # Classify surface orientation
            orientation_type = self._classify_orientation(normal)
            
            # Filter by orientation if requested
            if filter_orientation:
                if orientation_type not in ['horizontal_up', 'horizontal_down', 'vertical']:
                    continue
            
            # Enhanced confidence calculation including edge distance
            confidence = self._calculate_edge_aware_confidence(surface)
            
            grab_point = {
                'position': center,
                'normal': normal,
                'approach_vector': normal,
                'surface_area': area,
                'orientation_type': orientation_type,
                'surface_id': i,
                'confidence': confidence,
                'avg_distance_from_edge': surface['avg_distance_from_edge'],
                'edge_aware': True
            }
            
            grab_points.append(grab_point)
        
        # Sort by confidence (higher is better)
        grab_points.sort(key=lambda x: x['confidence'], reverse=True)
        
        return grab_points
    
    def _classify_orientation(self, normal):
        """Classify surface orientation based on normal vector"""
        vertical_component = abs(normal[1])  # Y-axis is vertical
        
        if vertical_component > 0.9:
            if normal[1] > 0:
                return 'horizontal_up'
            else:
                return 'horizontal_down'
        elif vertical_component < 0.1:
            return 'vertical'
        else:
            return 'angled'
    
    def _calculate_edge_aware_confidence(self, surface):
        """
        Enhanced confidence calculation that includes distance from edges
        """
        # Original factors
        area_score = min(surface['area'] / 0.01, 1.0)
        points_score = min(surface['num_points'] / 100, 1.0)
        
        # Orientation bonus
        normal = surface['normal']
        orientation_type = self._classify_orientation(normal)
        if orientation_type == 'horizontal_up':
            orientation_score = 1.0
        elif orientation_type in ['horizontal_down', 'vertical']:
            orientation_score = 0.8
        else:
            orientation_score = 0.5
        
        # Edge distance bonus - surfaces farther from edges are more stable
        edge_distance = surface['avg_distance_from_edge']
        if edge_distance == float('inf'):
            edge_score = 1.0
        else:
            edge_score = min(edge_distance / 0.05, 1.0)  # Normalize to 5cm
        
        # Combine scores with edge distance having significant weight
        confidence = (area_score * 0.25 + 
                     points_score * 0.25 + 
                     orientation_score * 0.25 + 
                     edge_score * 0.25)
        
        return confidence
    
    def create_edge_visualization(self, pcd, is_edge):
        """
        Create visualization showing edge vs surface points
        """
        points = np.asarray(pcd.points)
        
        # Create colored point cloud
        colors = np.zeros((len(points), 3))
        
        # Surface points in blue
        colors[~is_edge] = [0, 0, 1]  # Blue for surface points
        
        # Edge points in red
        colors[is_edge] = [1, 0, 0]   # Red for edge points
        
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(points)
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return vis_pcd


# Example integration with the main processor
def enhance_processor_with_edge_detection():
    """
    Example showing how to integrate edge-aware grab point detection
    """
    
    # Modify the FrameSequenceProcessor to use edge-aware detection
    def process_with_edge_aware_grab_points(self, file_path):
        # Get original processing results
        clusters, bboxes, text_labels, original_pcd = self.process_single_frame_optimized(file_path)
        
        # Initialize edge-aware grab point proposer
        grab_proposer = EdgeAwareGrabPointProposer(
            min_surface_area=0.0005,  # 5cm²
            max_surface_area=0.05,    # 500cm²
            min_points_per_surface=15,
            edge_detection_method='combined',  # Use combined method for robustness
            min_distance_from_edge=0.015  # 1.5cm from edges
        )
        
        all_grab_points = []
        grab_markers = []
        edge_visualizations = []
        
        for i, cluster in enumerate(clusters):
            print(f"    Processing cluster {i+1} with edge-aware grab point detection...")
            
            # Detect grab points using edge-aware method
            grab_points = grab_proposer.propose_grab_points_edge_aware(
                cluster, filter_orientation=True
            )
            
            if grab_points:
                print(f"      Found {len(grab_points)} edge-aware grab points")
                
                # Store grab points with cluster reference
                for grab_point in grab_points:
                    grab_point['cluster_id'] = i
                    grab_point['object_bbox'] = bboxes[i] if i < len(bboxes) else None
                
                all_grab_points.extend(grab_points)
                
                # Create visualization markers
                markers = grab_proposer.create_grab_point_markers(grab_points)
                grab_markers.extend(markers)
                
                # Create edge visualization for debugging
                is_edge, _ = grab_proposer.detect_edges(cluster)
                edge_vis = grab_proposer.create_edge_visualization(cluster, is_edge)
                edge_visualizations.append(edge_vis)
        
        return clusters, bboxes, text_labels, original_pcd, all_grab_points, grab_markers, edge_visualizations


if __name__ == "__main__":
    # Demo usage
    print("Edge-Aware Grab Point Detection Demo")
    print("This system detects edges and proposes grab points on interior surface areas")
    print("Edge detection methods: normal_variation, boundary, curvature, combined")