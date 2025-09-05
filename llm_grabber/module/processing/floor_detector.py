"""
Floor Detection and Removal System for LiDAR-based Object Tracking

This module implements RANSAC-based floor detection to identify and remove floor/table 
surfaces from point clouds, isolating objects for clustering and tracking.

Key Features:
- RANSAC plane fitting for robust floor detection
- Adaptive floor orientation detection
- Point cloud segmentation (floor vs objects)
- Configurable distance thresholds
- Visualization support for debugging
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any, List
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
import time


class FloorDetector:
    """
    Floor detection system using RANSAC plane fitting.
    
    Detects the dominant horizontal plane (floor/table) in a point cloud
    and separates floor points from object points for further processing.
    """
    
    def __init__(self, 
                 distance_threshold: float = 0.02,
                 max_trials: int = 1000,
                 min_floor_points: int = 500,
                 floor_normal_tolerance: float = 0.3):
        """
        Initialize floor detector with RANSAC parameters.
        
        Args:
            distance_threshold: Max distance from plane for inlier points (meters)
            max_trials: Maximum RANSAC iterations
            min_floor_points: Minimum points required to consider a valid floor
            floor_normal_tolerance: Tolerance for floor normal vector (0=strict horizontal, 1=any angle)
        """
        self.distance_threshold = distance_threshold
        self.max_trials = max_trials
        self.min_floor_points = min_floor_points
        self.floor_normal_tolerance = floor_normal_tolerance
        
        # Floor plane parameters (updated after detection)
        self.floor_plane = None  # [a, b, c, d] coefficients of ax + by + cz + d = 0
        self.floor_normal = None  # Normal vector of the floor plane
        self.floor_center = None  # Center point of detected floor
        self.detection_confidence = 0.0  # Confidence score (0-1)
        
        # Detection history for temporal stability
        self.detection_history = []
        self.max_history = 5
        
        # Performance tracking
        self.last_detection_time = 0.0
        self.detection_stats = {
            'total_detections': 0,
            'successful_detections': 0,
            'avg_detection_time': 0.0,
            'avg_floor_points': 0
        }
    
    def detect_floor(self, vertices: np.ndarray, 
                    return_debug_info: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
        """
        Detect floor plane and separate floor points from object points.
        
        Args:
            vertices: Point cloud as (N, 3) array
            return_debug_info: Whether to return debugging information
            
        Returns:
            Tuple of (object_points, floor_points, debug_info)
            debug_info is None unless return_debug_info=True
        """
        start_time = time.time()
        
        if vertices is None or len(vertices) < self.min_floor_points:
            return vertices, np.array([]), None if not return_debug_info else {}
        
        # Pre-filter points to reasonable height range for floor detection
        # Assume floor is within reasonable range from camera
        height_filtered = self._pre_filter_by_height(vertices)
        
        if len(height_filtered) < self.min_floor_points:
            return vertices, np.array([]), None if not return_debug_info else {}
        
        # Perform RANSAC plane fitting
        floor_points, plane_params, confidence = self._ransac_plane_detection(height_filtered)
        
        # Update detection statistics
        detection_time = time.time() - start_time
        self._update_detection_stats(detection_time, len(floor_points), confidence > 0.5)
        
        if len(floor_points) < self.min_floor_points:
            # No reliable floor detected - return original points
            return vertices, np.array([]), None if not return_debug_info else {}
        
        # Store detected floor parameters
        self.floor_plane = plane_params
        self.floor_normal = plane_params[:3] / np.linalg.norm(plane_params[:3])
        self.floor_center = np.mean(floor_points, axis=0)
        self.detection_confidence = confidence
        
        # Separate all points into floor and object points using detected plane
        all_floor_points, all_object_points = self._separate_points_by_plane(vertices, plane_params)
        
        # Add to detection history for temporal stability
        self._update_detection_history({
            'plane_params': plane_params,
            'confidence': confidence,
            'floor_points_count': len(all_floor_points)
        })
        
        # Prepare debug information
        debug_info = None
        if return_debug_info:
            debug_info = {
                'plane_params': plane_params,
                'floor_normal': self.floor_normal,
                'floor_center': self.floor_center,
                'confidence': confidence,
                'detection_time': detection_time,
                'floor_points_count': len(all_floor_points),
                'object_points_count': len(all_object_points),
                'total_points': len(vertices)
            }
        
        return all_object_points, all_floor_points, debug_info
    
    def _pre_filter_by_height(self, vertices: np.ndarray) -> np.ndarray:
        """Pre-filter points by height to focus on likely floor region."""
        if len(vertices) == 0:
            return vertices
        
        # Assume floor is in the lower portion of the point cloud
        # Use Y-coordinate assuming standard camera orientation
        y_coords = vertices[:, 1]
        
        # Find the range where most points are concentrated (likely floor level)
        y_min, y_max = np.percentile(y_coords, [10, 90])
        y_range = y_max - y_min
        
        # Focus on lower portion for floor detection
        floor_height_threshold = y_min + y_range * 0.3  # Bottom 30% of range
        
        mask = y_coords <= floor_height_threshold
        return vertices[mask]
    
    def _ransac_plane_detection(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform RANSAC-based plane detection.
        
        Returns:
            Tuple of (inlier_points, plane_parameters, confidence_score)
        """
        if len(vertices) < 3:
            return np.array([]), np.array([0, 0, 1, 0]), 0.0
        
        # Prepare data for plane fitting: ax + by + cz + d = 0
        X = vertices[:, [0, 2]]  # X-Z coordinates (assuming Y is height)
        y = vertices[:, 1]       # Y coordinates (height)
        
        try:
            # Use RANSAC to fit plane robustly
            ransac = RANSACRegressor(
                residual_threshold=self.distance_threshold,
                max_trials=self.max_trials,
                random_state=42
            )
            
            # Fit plane model: y = ax + bz + c  =>  ax - y + bz + c = 0
            ransac.fit(X, y)
            
            # Extract plane parameters
            a, b = ransac.estimator_.coef_
            c = -1.0  # Y coefficient
            d = ransac.estimator_.intercept_
            
            plane_params = np.array([a, c, b, d])  # [x_coef, y_coef, z_coef, constant]
            
            # Normalize plane equation
            normal_length = np.linalg.norm(plane_params[:3])
            if normal_length > 1e-8:
                plane_params = plane_params / normal_length
            
            # Check if plane is reasonably horizontal (floor-like)
            normal_vector = plane_params[:3]
            vertical_vector = np.array([0, 1, 0])  # Assuming Y is up
            
            # Calculate angle between plane normal and vertical
            cos_angle = np.abs(np.dot(normal_vector, vertical_vector))
            
            # Confidence based on inlier ratio and horizontal orientation
            inlier_mask = ransac.inlier_mask_
            inlier_ratio = np.sum(inlier_mask) / len(inlier_mask)
            horizontal_score = cos_angle  # Higher when plane is more horizontal
            
            # Adjust confidence based on floor normal tolerance
            if horizontal_score < (1.0 - self.floor_normal_tolerance):
                # Plane is too steep to be a floor
                confidence = 0.0
                inlier_points = np.array([])
            else:
                confidence = (inlier_ratio * 0.7) + (horizontal_score * 0.3)
                inlier_points = vertices[inlier_mask]
            
            return inlier_points, plane_params, confidence
            
        except Exception as e:
            print(f"Warning: RANSAC plane fitting failed: {e}")
            return np.array([]), np.array([0, 1, 0, 0]), 0.0
    
    def _separate_points_by_plane(self, vertices: np.ndarray, 
                                 plane_params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Separate all points into floor and object points using the detected plane."""
        if len(vertices) == 0 or len(plane_params) != 4:
            return np.array([]), vertices
        
        # Calculate distance from each point to the plane
        # Distance = |ax + by + cz + d| / sqrt(a² + b² + c²)
        a, b, c, d = plane_params
        
        # Calculate signed distances
        distances = (a * vertices[:, 0] + 
                    b * vertices[:, 1] + 
                    c * vertices[:, 2] + d)
        
        # Points close to plane are floor points
        floor_mask = np.abs(distances) <= self.distance_threshold
        
        floor_points = vertices[floor_mask]
        object_points = vertices[~floor_mask]
        
        return floor_points, object_points
    
    def _update_detection_history(self, detection_result: Dict):
        """Update detection history for temporal stability."""
        self.detection_history.append(detection_result)
        
        # Keep only recent history
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)
    
    def _update_detection_stats(self, detection_time: float, floor_points_count: int, success: bool):
        """Update performance statistics."""
        self.detection_stats['total_detections'] += 1
        
        if success:
            self.detection_stats['successful_detections'] += 1
        
        # Update running averages
        total = self.detection_stats['total_detections']
        self.detection_stats['avg_detection_time'] = (
            (self.detection_stats['avg_detection_time'] * (total - 1) + detection_time) / total
        )
        
        if success:
            successful = self.detection_stats['successful_detections']
            prev_avg = self.detection_stats['avg_floor_points']
            self.detection_stats['avg_floor_points'] = (
                (prev_avg * (successful - 1) + floor_points_count) / successful
                if successful > 0 else floor_points_count
            )
    
    def get_stable_floor_plane(self) -> Optional[np.ndarray]:
        """
        Get stable floor plane parameters based on recent detection history.
        
        Returns:
            Plane parameters [a, b, c, d] if stable detection available, None otherwise
        """
        if len(self.detection_history) < 3:
            return None
        
        # Get recent detections with good confidence
        good_detections = [d for d in self.detection_history[-3:] if d['confidence'] > 0.5]
        
        if len(good_detections) < 2:
            return None
        
        # Average plane parameters for stability
        plane_params = np.array([d['plane_params'] for d in good_detections])
        stable_plane = np.mean(plane_params, axis=0)
        
        return stable_plane
    
    def is_floor_detection_stable(self) -> bool:
        """Check if floor detection is stable across recent frames."""
        if len(self.detection_history) < 3:
            return False
        
        recent_confidences = [d['confidence'] for d in self.detection_history[-3:]]
        return all(c > 0.5 for c in recent_confidences)
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection performance statistics."""
        stats = self.detection_stats.copy()
        
        # Add derived metrics
        if stats['total_detections'] > 0:
            stats['success_rate'] = stats['successful_detections'] / stats['total_detections']
        else:
            stats['success_rate'] = 0.0
        
        stats['current_confidence'] = self.detection_confidence
        stats['history_length'] = len(self.detection_history)
        stats['is_stable'] = self.is_floor_detection_stable()
        
        return stats
    
    def get_floor_normal(self) -> Optional[np.ndarray]:
        """
        Get the current floor normal vector.
        
        Returns:
            Floor normal vector [nx, ny, nz] if available, None otherwise
        """
        return self.floor_normal.copy() if self.floor_normal is not None else None
    
    def get_floor_plane_parameters(self) -> Optional[np.ndarray]:
        """
        Get current floor plane parameters.
        
        Returns:
            Plane parameters [a, b, c, d] for equation ax + by + cz + d = 0, or None
        """
        return self.floor_plane.copy() if self.floor_plane is not None else None
    
    def get_floor_coordinate_system(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get floor-aligned coordinate system based on detected floor plane.
        
        Returns:
            Dictionary with 'origin', 'x_axis', 'y_axis', 'z_axis' vectors, or None
        """
        if self.floor_normal is None or self.floor_center is None:
            return None
        
        try:
            # Z-axis is the floor normal (pointing up from floor)
            z_axis = self.floor_normal / np.linalg.norm(self.floor_normal)
            
            # Create orthogonal X-axis perpendicular to floor normal
            if abs(z_axis[1]) < 0.9:  # Floor not nearly vertical
                # Use world Y-axis to create perpendicular vector
                x_axis = np.cross([0, 1, 0], z_axis)
            else:  # Floor is nearly vertical, use world X-axis
                x_axis = np.cross([1, 0, 0], z_axis)
            
            # Normalize X-axis
            x_axis = x_axis / np.linalg.norm(x_axis)
            
            # Y-axis completes right-hand coordinate system
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / np.linalg.norm(y_axis)
            
            return {
                'origin': self.floor_center.copy(),
                'x_axis': x_axis,
                'y_axis': y_axis,
                'z_axis': z_axis,
                'transformation_matrix': self._create_floor_transform_matrix(x_axis, y_axis, z_axis)
            }
            
        except Exception as e:
            print(f"Warning: Failed to create floor coordinate system: {e}")
            return None
    
    def _create_floor_transform_matrix(self, x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
        """Create 4x4 transformation matrix for floor coordinate system."""
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
        transform[:3, 3] = self.floor_center
        return transform
    
    def project_point_to_floor(self, point: np.ndarray) -> Optional[np.ndarray]:
        """
        Project a 3D point onto the detected floor plane.
        
        Args:
            point: 3D point [x, y, z]
            
        Returns:
            Projected point on floor plane, or None if no floor detected
        """
        if self.floor_plane is None:
            return None
        
        try:
            # Project point to floor plane
            a, b, c, d = self.floor_plane
            
            # Distance from point to plane
            distance = (a * point[0] + b * point[1] + c * point[2] + d)
            
            # Project point onto plane along normal direction
            normal = np.array([a, b, c])
            projected_point = point - distance * normal
            
            return projected_point
            
        except Exception:
            return None
    
    def get_floor_height_at_position(self, x: float, z: float) -> Optional[float]:
        """
        Get floor height (Y coordinate) at given X, Z position.
        
        Args:
            x: X coordinate
            z: Z coordinate
            
        Returns:
            Y coordinate of floor at (x, z), or None if no floor detected
        """
        if self.floor_plane is None:
            return None
        
        try:
            # Floor plane equation: ax + by + cz + d = 0
            # Solve for y: y = -(ax + cz + d) / b
            a, b, c, d = self.floor_plane
            
            if abs(b) < 1e-8:  # Floor is vertical (shouldn't happen normally)
                return None
            
            y = -(a * x + c * z + d) / b
            return float(y)
            
        except Exception:
            return None
    
    def is_point_on_floor(self, point: np.ndarray, tolerance: Optional[float] = None) -> bool:
        """
        Check if a point lies on the detected floor plane.
        
        Args:
            point: 3D point [x, y, z]
            tolerance: Distance tolerance, uses detection threshold if None
            
        Returns:
            True if point is on floor plane within tolerance
        """
        if self.floor_plane is None:
            return False
        
        tolerance = tolerance or self.distance_threshold
        
        try:
            # Calculate distance from point to floor plane
            a, b, c, d = self.floor_plane
            distance = abs(a * point[0] + b * point[1] + c * point[2] + d)
            
            return distance <= tolerance
            
        except Exception:
            return False
    
    def get_floor_detection_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about current floor detection.
        
        Returns:
            Dictionary with all available floor information
        """
        info = {
            'floor_detected': self.floor_plane is not None,
            'confidence': self.detection_confidence,
            'detection_stable': self.is_floor_detection_stable(),
            'detection_stats': self.get_detection_stats()
        }
        
        if self.floor_plane is not None:
            info.update({
                'floor_plane': self.floor_plane.copy(),
                'floor_normal': self.floor_normal.copy(),
                'floor_center': self.floor_center.copy(),
                'coordinate_system': self.get_floor_coordinate_system()
            })
        
        return info
    
    def create_floor_visualization(self, vertices: np.ndarray, 
                                  object_points: np.ndarray, 
                                  floor_points: np.ndarray,
                                  image_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """
        Create visualization of floor detection results.
        
        Args:
            vertices: Original point cloud
            object_points: Points classified as objects
            floor_points: Points classified as floor
            image_size: Output image size (width, height)
            
        Returns:
            BGR image showing floor detection results
        """
        width, height = image_size
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(vertices) == 0:
            return vis_image
        
        # Project points to 2D for visualization (top-down view)
        x_coords = vertices[:, 0]
        z_coords = vertices[:, 2]
        
        # Calculate bounds for mapping to image coordinates
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        # Add margins
        margin = 0.1
        x_range = x_max - x_min
        z_range = z_max - z_min
        x_min -= margin * x_range
        x_max += margin * x_range
        z_min -= margin * z_range
        z_max += margin * z_range
        
        # Map coordinates to image space
        def to_image_coords(points):
            if len(points) == 0:
                return np.array([])
            
            img_x = ((points[:, 0] - x_min) / (x_max - x_min) * (width - 1)).astype(int)
            img_y = ((points[:, 2] - z_min) / (z_max - z_min) * (height - 1)).astype(int)
            
            # Clip to image bounds
            img_x = np.clip(img_x, 0, width - 1)
            img_y = np.clip(img_y, 0, height - 1)
            
            return np.column_stack([img_x, img_y])
        
        # Draw floor points in blue
        if len(floor_points) > 0:
            floor_coords = to_image_coords(floor_points)
            for x, y in floor_coords:
                cv2.circle(vis_image, (x, y), 1, (255, 0, 0), -1)  # Blue for floor
        
        # Draw object points in green
        if len(object_points) > 0:
            obj_coords = to_image_coords(object_points)
            for x, y in obj_coords:
                cv2.circle(vis_image, (x, y), 2, (0, 255, 0), -1)  # Green for objects
        
        # Draw floor plane line if detected
        if self.floor_plane is not None and len(floor_points) > 0:
            # Get floor line for visualization (simplified)
            floor_center_2d = to_image_coords(self.floor_center.reshape(1, -1))[0]
            cv2.circle(vis_image, tuple(floor_center_2d), 5, (0, 255, 255), -1)  # Yellow center
        
        # Add text information
        cv2.putText(vis_image, f"Floor: {len(floor_points)} pts", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(vis_image, f"Objects: {len(object_points)} pts", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(vis_image, f"Confidence: {self.detection_confidence:.2f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return vis_image