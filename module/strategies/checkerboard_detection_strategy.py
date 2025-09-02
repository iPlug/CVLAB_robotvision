"""
Checkerboard Detection Strategy for Camera Calibration

This strategy replaces point cloud clustering with RGB image-based checkerboard pattern detection
for high-precision camera-robot calibration. Uses OpenCV's findChessboardCorners() with 
sub-pixel refinement to achieve sub-5mm accuracy.

Features:
- Multiple pattern sizes (9x6, 8x6, 7x5)
- Sub-pixel corner refinement
- Pattern quality validation
- 3D pose estimation using solvePnP
- Compatible with existing ClusteringStrategy interface
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import time

from ..core.clustering_strategy import ClusteringStrategy


class CheckerboardDetectionStrategy(ClusteringStrategy):
    """
    Checkerboard pattern detection strategy for camera calibration.
    
    This strategy detects checkerboard patterns in RGB images and returns their
    3D positions and orientations. It's designed for high-precision camera-robot
    calibration with sub-5mm accuracy.
    """
    
    def __init__(self, 
                 pattern_sizes: List[Tuple[int, int]] = None,
                 square_size: float = 25.0,
                 detection_flags: int = None,
                 subpix_criteria: Tuple = None,
                 min_pattern_coverage: float = 0.01,
                 max_pattern_coverage: float = 0.95,
                 quality_threshold: float = 0.1):
        """
        Initialize checkerboard detection strategy.
        
        Args:
            pattern_sizes: List of (width, height) pattern sizes to try [(9,6), (8,6), (7,5)]
            square_size: Size of checkerboard squares in millimeters
            detection_flags: OpenCV flags for findChessboardCorners
            subpix_criteria: Criteria for cornerSubPix refinement
            min_pattern_coverage: Minimum image coverage required (0.0-1.0)
            max_pattern_coverage: Maximum image coverage allowed (0.0-1.0)
            quality_threshold: Minimum quality score for pattern acceptance
        """
        self.pattern_sizes = pattern_sizes or [(9, 6), (8, 6), (7, 5)]
        self.square_size = square_size
        
        # OpenCV detection flags - optimized for longer distance detection
        if detection_flags is None:
            self.detection_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                  cv2.CALIB_CB_NORMALIZE_IMAGE +
                                  cv2.CALIB_CB_FAST_CHECK)
        else:
            self.detection_flags = detection_flags
        
        # Sub-pixel refinement criteria - match standalone test
        if subpix_criteria is None:
            self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        else:
            self.subpix_criteria = subpix_criteria
        
        self.min_pattern_coverage = min_pattern_coverage
        self.max_pattern_coverage = max_pattern_coverage
        self.quality_threshold = quality_threshold
        
        # Camera calibration parameters (will be loaded or estimated)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_calibrated = False
        
        # Pattern object points cache
        self.pattern_object_points = {}
        self._generate_object_points()
        
        # Statistics
        self.detection_stats = {
            'total_attempts': 0,
            'successful_detections': 0,
            'pattern_size_counts': {},
            'avg_detection_time': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _generate_object_points(self):
        """Generate 3D object points for each pattern size."""
        for pattern_size in self.pattern_sizes:
            width, height = pattern_size
            object_points = np.zeros((width * height, 3), np.float32)
            object_points[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
            object_points *= self.square_size
            self.pattern_object_points[pattern_size] = object_points
    
    def cluster_objects(self, vertices: np.ndarray, color_image: Optional[np.ndarray] = None,
                       return_visualization: bool = False) -> List[Dict[str, Any]]:
        """
        Detect checkerboard patterns in the color image.
        
        Args:
            vertices: Point cloud vertices (not used for checkerboard detection)
            color_image: RGB color image for pattern detection
            return_visualization: Whether to return visualization data
            
        Returns:
            List of detected checkerboard patterns with position, orientation, and quality metrics
        """
        start_time = time.time()
        self.detection_stats['total_attempts'] += 1
        
        if color_image is None:
            self.logger.warning("No color image provided for checkerboard detection")
            return [] if not return_visualization else ([], None, None)
        
        # Convert to grayscale for detection
        if len(color_image.shape) == 3:
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = color_image
        
        detected_patterns = []
        visualization_data = None
        
        # Try each pattern size
        for pattern_size in self.pattern_sizes:
            pattern_info = self._detect_single_pattern(gray_image, color_image, pattern_size)
            if pattern_info:
                detected_patterns.append(pattern_info)
                self.detection_stats['successful_detections'] += 1
                
                # Update pattern size statistics
                size_key = f"{pattern_size[0]}x{pattern_size[1]}"
                self.detection_stats['pattern_size_counts'][size_key] = \
                    self.detection_stats['pattern_size_counts'].get(size_key, 0) + 1
                
                # For calibration, we typically want only one pattern
                break
        
        # Update timing statistics
        detection_time = time.time() - start_time
        alpha = 0.1  # Exponential moving average
        self.detection_stats['avg_detection_time'] = \
            alpha * detection_time + (1 - alpha) * self.detection_stats['avg_detection_time']
        
        # Create visualization if requested
        if return_visualization:
            visualization_data = self._create_visualization(color_image, detected_patterns)
            return detected_patterns, visualization_data, None
        
        return detected_patterns
    
    def _detect_single_pattern(self, gray_image: np.ndarray, color_image: np.ndarray,
                              pattern_size: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """
        Detect a single checkerboard pattern of specified size.
        
        Args:
            gray_image: Grayscale image for detection
            color_image: Color image for visualization
            pattern_size: (width, height) of pattern to detect
            
        Returns:
            Pattern information dictionary or None if not detected
        """
        width, height = pattern_size
        
        # Detect checkerboard corners
        ret, corners = cv2.findChessboardCorners(
            gray_image, pattern_size, self.detection_flags
        )
        
        if not ret or corners is None:
            return None
        
        # Refine corner positions with sub-pixel accuracy
        refined_corners = cv2.cornerSubPix(
            gray_image, corners, (11, 11), (-1, -1), self.subpix_criteria
        )
        
        # Validate corner count matches pattern size
        expected_corners = width * height
        if len(refined_corners) != expected_corners:
            print(f"DEBUG: Corner count mismatch: got {len(refined_corners)}, expected {expected_corners}")
            return None
        
        # Validate pattern quality
        quality_score = self._calculate_pattern_quality(refined_corners, gray_image.shape)
        if quality_score < self.quality_threshold:
            return None
        
        # Calculate pattern coverage
        coverage = self._calculate_pattern_coverage(refined_corners, gray_image.shape)
        if coverage < self.min_pattern_coverage or coverage > self.max_pattern_coverage:
            return None
        
        # Get 3D object points for this pattern
        object_points = self.pattern_object_points[pattern_size]
        
        # Calculate pattern center in image coordinates (ensure 2D format)
        corners_2d = refined_corners.reshape(-1, 2)
        center_2d = np.mean(corners_2d, axis=0)
        
        # Estimate 3D pose if camera is calibrated
        pose_data = None
        center_3d_mm = None
        
        if self.camera_calibrated and self.camera_matrix is not None:
            pose_data = self._estimate_3d_pose(object_points, refined_corners)
            if pose_data:
                center_3d_mm = pose_data['translation_mm']
                print(f"SUCCESS: Pattern detected with 3D pose: {center_3d_mm}")
            else:
                print("ERROR: 3D pose estimation failed")
        else:
            print("ERROR: No camera calibration - cannot estimate 3D pose")
        
        # Create pattern information dictionary compatible with clustering interface
        pattern_info = {
            'id': f"checkerboard_{pattern_size[0]}x{pattern_size[1]}",
            'pattern_size': pattern_size,
            'corners_2d': corners_2d,  # Store as (N, 2) format
            'corners_opencv': refined_corners,  # Store original OpenCV format for compatibility
            'corners_count': len(corners_2d),
            'center_2d': center_2d,
            'center_3d': center_3d_mm / 1000.0 if center_3d_mm is not None else None,  # Convert to meters
            'center_mm': center_3d_mm,  # Keep mm version for compatibility
            'quality_score': quality_score,
            'coverage': coverage,
            'object_points': object_points,
            'pose_data': pose_data,
            
            # Compatibility fields for ClusteringStrategy interface
            'points': corners_2d,  # 2D corner points
            'center': center_3d_mm / 1000.0 if center_3d_mm is not None else [center_2d[0]/1000, center_2d[1]/1000, 0],
            'num_points': len(corners_2d),
            'min_coords': [0, 0, 0],  # Placeholder
            'max_coords': [self.square_size * width / 1000, self.square_size * height / 1000, 0],
            'dimensions': [self.square_size * width / 1000, self.square_size * height / 1000, 0]
        }
        
        return pattern_info
    
    def _calculate_pattern_quality(self, corners: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """
        Calculate pattern detection quality score.
        
        Args:
            corners: Detected corner points
            image_shape: Image dimensions (height, width)
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if corners is None or len(corners) < 4:
            return 0.0
        
        corners_2d = corners.reshape(-1, 2)
        
        # Factor 1: Pattern sharpness (based on corner gradients)
        sharpness_score = 0.8  # Placeholder - would need image gradients
        
        # Factor 2: Pattern geometry (rectangularity)
        geometry_score = self._calculate_geometry_score(corners_2d)
        
        # Factor 3: Coverage uniformity
        uniformity_score = self._calculate_uniformity_score(corners_2d, image_shape)
        
        # Factor 4: Edge distance (penalize patterns too close to edges)
        edge_score = self._calculate_edge_distance_score(corners_2d, image_shape)
        
        # Combined quality score
        quality = (sharpness_score * 0.3 + 
                  geometry_score * 0.4 + 
                  uniformity_score * 0.2 + 
                  edge_score * 0.1)
        
        return np.clip(quality, 0.0, 1.0)
    
    def _calculate_geometry_score(self, corners: np.ndarray) -> float:
        """Calculate how well the pattern maintains rectangular geometry."""
        if len(corners) < 4:
            return 0.0
        
        # Calculate corner angles and edge lengths
        # For a perfect checkerboard, internal angles should be close to 90 degrees
        # This is a simplified version - could be enhanced
        
        # Calculate bounding box aspect ratio
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return 0.0
        
        aspect_ratio = min(width/height, height/width)
        
        # Score based on reasonable aspect ratios for checkerboards
        if aspect_ratio > 0.5:  # Not too elongated
            return aspect_ratio
        else:
            return aspect_ratio * 0.5  # Penalize very elongated patterns
    
    def _calculate_uniformity_score(self, corners: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """Calculate how uniformly the pattern is distributed."""
        if len(corners) < 4:
            return 0.0
        
        # Calculate standard deviation of corner distances from center
        center = np.mean(corners, axis=0)
        distances = np.linalg.norm(corners - center, axis=1)
        
        if np.mean(distances) == 0:
            return 0.0
        
        cv = np.std(distances) / np.mean(distances)  # Coefficient of variation
        
        # Lower CV means more uniform distribution
        uniformity = np.exp(-cv * 2)  # Exponential decay
        
        return np.clip(uniformity, 0.0, 1.0)
    
    def _calculate_edge_distance_score(self, corners: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """Calculate score based on distance from image edges."""
        height, width = image_shape[:2]
        
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        
        # Distance from edges
        edge_distances = [
            min_x,           # Left edge
            min_y,           # Top edge
            width - max_x,   # Right edge
            height - max_y   # Bottom edge
        ]
        
        min_edge_distance = min(edge_distances)
        
        # Normalize by image size
        normalized_distance = min_edge_distance / min(width, height)
        
        # Score: prefer patterns not too close to edges
        if normalized_distance > 0.1:  # At least 10% from edge
            return 1.0
        else:
            return normalized_distance / 0.1
    
    def _calculate_pattern_coverage(self, corners: np.ndarray, image_shape: Tuple[int, int]) -> float:
        """Calculate what fraction of the image the pattern covers."""
        if corners is None or len(corners) == 0:
            return 0.0
        
        corners_2d = corners.reshape(-1, 2)
        min_x, min_y = np.min(corners_2d, axis=0)
        max_x, max_y = np.max(corners_2d, axis=0)
        
        pattern_area = (max_x - min_x) * (max_y - min_y)
        image_area = image_shape[0] * image_shape[1]
        
        return pattern_area / image_area if image_area > 0 else 0.0
    
    def _estimate_3d_pose(self, object_points: np.ndarray, image_points: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Estimate 3D pose of the checkerboard pattern using solvePnP.
        
        Args:
            object_points: 3D object points in pattern coordinate system
            image_points: Corresponding 2D image points
            
        Returns:
            Dictionary with pose information or None if estimation fails
        """
        if not self.camera_calibrated or self.camera_matrix is None:
            print("DEBUG: Cannot estimate 3D pose - camera not calibrated")
            return None
        
        try:
            # Solve PnP to get rotation and translation vectors
            success, rvec, tvec = cv2.solvePnP(
                object_points, 
                image_points.reshape(-1, 1, 2),
                self.camera_matrix,
                self.dist_coeffs
            )
            
            if not success:
                return None
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Extract pose information
            translation_mm = tvec.flatten() * 1000  # Convert to mm
            
            # Calculate pattern normal vector
            normal_vector = rotation_matrix[:, 2]  # Z-axis of pattern coordinate system
            
            # Calculate pattern orientation angles
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi
            pitch = np.arctan2(-rotation_matrix[2, 0], 
                              np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2)) * 180 / np.pi
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi
            
            pose_data = {
                'translation_mm': translation_mm,
                'rotation_matrix': rotation_matrix,
                'rotation_vector': rvec.flatten(),
                'normal_vector': normal_vector,
                'orientation_angles': [roll, pitch, yaw],  # [roll, pitch, yaw] in degrees
                'distance_mm': np.linalg.norm(translation_mm)
            }
            
            return pose_data
            
        except Exception as e:
            self.logger.warning(f"Failed to estimate 3D pose: {e}")
            return None
    
    def _create_visualization(self, color_image: np.ndarray, detected_patterns: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create visualization overlay showing detected patterns.
        
        Args:
            color_image: Original color image
            detected_patterns: List of detected patterns
            
        Returns:
            Image with pattern detection overlay
        """
        if color_image is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        vis_image = color_image.copy()
        
        for i, pattern in enumerate(detected_patterns):
            # Try to get corners in OpenCV format first, then fall back to 2D format
            corners_opencv = pattern.get('corners_opencv')
            corners_2d = pattern.get('corners_2d')
            pattern_size = pattern.get('pattern_size', (0, 0))
            quality_score = pattern.get('quality_score', 0.0)
            
            # Use OpenCV format corners if available, otherwise convert from 2D format
            corners_to_draw = corners_opencv if corners_opencv is not None else corners_2d
            
            if corners_to_draw is not None and len(corners_to_draw) > 0:
                try:
                    # Validate that we have the expected number of corners
                    expected_corners = pattern_size[0] * pattern_size[1]
                    if len(corners_to_draw) != expected_corners:
                        print(f"DEBUG: Corner count mismatch: got {len(corners_to_draw)}, expected {expected_corners}")
                        continue
                    
                    # If using 2D format, convert to OpenCV format
                    if corners_opencv is None and corners_2d is not None:
                        # Convert from (N, 2) to (N, 1, 2) format for OpenCV
                        corners_formatted = corners_2d.reshape(-1, 1, 2).astype(np.float32)
                    else:
                        # Use original OpenCV format
                        corners_formatted = corners_opencv.astype(np.float32)
                    
                    print(f"DEBUG: Drawing corners with shape {corners_formatted.shape}, dtype {corners_formatted.dtype}")
                    
                    # Draw checkerboard corners safely
                    cv2.drawChessboardCorners(vis_image, pattern_size, corners_formatted, True)
                    
                except Exception as e:
                    print(f"DEBUG: Error drawing corners for pattern {i}: {e}")
                    print(f"DEBUG: Pattern size: {pattern_size}")
                    print(f"DEBUG: Corners shape: {corners_to_draw.shape if hasattr(corners_to_draw, 'shape') else 'no shape'}")
                    
                    # Fall back to drawing individual points
                    try:
                        if corners_2d is not None:
                            corners_flat = corners_2d if len(corners_2d.shape) == 2 else corners_2d.reshape(-1, 2)
                            for j, corner in enumerate(corners_flat):
                                cv2.circle(vis_image, tuple(corner.astype(int)), 3, (0, 255, 0), -1)
                                if j < 5:  # Only label first few corners to avoid clutter
                                    cv2.putText(vis_image, str(j), tuple((corner + 5).astype(int)), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                        else:
                            print("DEBUG: No valid corner data available for fallback drawing")
                    except Exception as e2:
                        print(f"DEBUG: Fallback corner drawing also failed: {e2}")
                
                # Draw pattern center
                center_2d = pattern.get('center_2d')
                if center_2d is not None:
                    center_int = tuple(center_2d.astype(int))
                    cv2.circle(vis_image, center_int, 5, (0, 255, 0), -1)
                    
                    # Add text information
                    text_lines = [
                        f"Pattern: {pattern_size[0]}x{pattern_size[1]}",
                        f"Quality: {quality_score:.2f}",
                        f"Corners: {len(corners_2d)}"
                    ]
                    
                    if pattern.get('pose_data'):
                        pose = pattern['pose_data']
                        distance = pose.get('distance_mm', 0)
                        text_lines.append(f"Distance: {distance:.0f}mm")
                    
                    # Draw text
                    text_y = center_int[1] - 60
                    for j, line in enumerate(text_lines):
                        cv2.putText(vis_image, line, (center_int[0] - 50, text_y + j * 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return vis_image
    
    def set_camera_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        Set camera calibration parameters for 3D pose estimation.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_calibrated = True
        self.logger.info("Camera calibration parameters set for 3D pose estimation")
    
    def calibrate_camera_from_patterns(self, pattern_detections: List[Dict[str, Any]], 
                                     image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calibrate camera using detected checkerboard patterns.
        
        Args:
            pattern_detections: List of pattern detection results
            image_size: Image size (width, height)
            
        Returns:
            Tuple of (camera_matrix, dist_coeffs, reprojection_error)
        """
        if not pattern_detections:
            raise ValueError("No pattern detections provided for calibration")
        
        # Collect object points and image points from all detections
        all_object_points = []
        all_image_points = []
        
        for detection in pattern_detections:
            pattern_size = detection.get('pattern_size')
            corners_2d = detection.get('corners_2d')
            
            if pattern_size and corners_2d is not None:
                object_points = self.pattern_object_points[pattern_size]
                all_object_points.append(object_points)
                all_image_points.append(corners_2d)
        
        if len(all_object_points) < 3:
            raise ValueError("Need at least 3 pattern detections for camera calibration")
        
        # Perform camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            all_object_points, all_image_points, image_size, None, None
        )
        
        if not ret:
            raise RuntimeError("Camera calibration failed")
        
        # Set calibration parameters
        self.set_camera_calibration(camera_matrix, dist_coeffs)
        
        self.logger.info(f"Camera calibrated with reprojection error: {ret:.4f} pixels")
        
        return camera_matrix, dist_coeffs, ret
    
    def get_strategy_name(self) -> str:
        """Get the name of this detection strategy."""
        return "Checkerboard Detection"
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection performance statistics."""
        success_rate = 0.0
        if self.detection_stats['total_attempts'] > 0:
            success_rate = self.detection_stats['successful_detections'] / self.detection_stats['total_attempts']
        
        return {
            'strategy_name': self.get_strategy_name(),
            'total_attempts': self.detection_stats['total_attempts'],
            'successful_detections': self.detection_stats['successful_detections'],
            'success_rate': success_rate,
            'avg_detection_time_ms': self.detection_stats['avg_detection_time'] * 1000,
            'pattern_size_counts': self.detection_stats['pattern_size_counts'],
            'camera_calibrated': self.camera_calibrated,
            'supported_patterns': [f"{w}x{h}" for w, h in self.pattern_sizes]
        }
    
    def reset_statistics(self):
        """Reset detection statistics."""
        self.detection_stats = {
            'total_attempts': 0,
            'successful_detections': 0,
            'pattern_size_counts': {},
            'avg_detection_time': 0.0
        }
    
    def preprocess_points(self, vertices: np.ndarray) -> np.ndarray:
        """
        Preprocessing not needed for checkerboard detection.
        This method is required by the ClusteringStrategy interface.
        """
        return vertices
    
    def postprocess_clusters(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optional postprocessing for detected patterns.
        This method is required by the ClusteringStrategy interface.
        """
        # Filter patterns by quality score
        filtered_clusters = []
        for cluster in clusters:
            quality = cluster.get('quality_score', 0.0)
            if quality >= self.quality_threshold:
                filtered_clusters.append(cluster)
        
        # Sort by quality score (best first)
        filtered_clusters.sort(key=lambda x: x.get('quality_score', 0.0), reverse=True)
        
        return filtered_clusters