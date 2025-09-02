"""
CharUco Detection Strategy for Camera Calibration

This strategy replaces point cloud clustering with RGB image-based CharUco pattern detection
for high-precision camera-robot calibration. Uses OpenCV's CharUco detector with 
sub-pixel refinement to achieve sub-5mm accuracy.

Features:
- 8x11 CharUco pattern (25mm squares, 20mm markers)
- ArUco marker and ChArUco corner detection
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


class CharucoDetectionStrategy(ClusteringStrategy):
    """
    CharUco pattern detection strategy for camera calibration.
    
    This strategy detects CharUco patterns in RGB images and returns their
    3D positions and orientations. It's designed for high-precision camera-robot
    calibration with sub-5mm accuracy.
    
    Uses the proven 8x11 pattern configuration from test_charuco_detection.py.
    """
    
    def __init__(self, 
                 pattern_size: Tuple[int, int] = (8, 11),
                 square_size_mm: float = 25.0,
                 marker_size_mm: float = 20.0,
                 aruco_dict_type: int = cv2.aruco.DICT_6X6_250,
                 min_pattern_coverage: float = 0.1,
                 max_pattern_coverage: float = 0.9,
                 quality_threshold: float = 0.5,
                 min_corners_for_pose: int = 8,
                 bypass_filtering: bool = False):
        """
        Initialize CharUco detection strategy.
        
        Args:
            pattern_size: CharUco pattern size (squares_x, squares_y)
            square_size_mm: Size of checkerboard squares in millimeters
            marker_size_mm: Size of ArUco markers in millimeters
            aruco_dict_type: ArUco dictionary type (default: DICT_6X6_250)
            min_pattern_coverage: Minimum image coverage required (0.0-1.0)
            max_pattern_coverage: Maximum image coverage allowed (0.0-1.0)
            quality_threshold: Minimum quality score for pattern acceptance
            min_corners_for_pose: Minimum corners required for 3D pose estimation
            bypass_filtering: If True, skip all quality filtering (like test_charuco_detection.py)
        """
        self.pattern_size = pattern_size  # (squares_x, squares_y)
        self.square_size_mm = square_size_mm
        self.marker_size_mm = marker_size_mm
        self.aruco_dict_type = aruco_dict_type
        self.min_pattern_coverage = min_pattern_coverage
        self.max_pattern_coverage = max_pattern_coverage
        self.quality_threshold = quality_threshold
        self.min_corners_for_pose = min_corners_for_pose
        self.bypass_filtering = bypass_filtering
        
        # Create ArUco dictionary and CharUco board
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        # Use millimeter inputs directly (like test_charuco_detection.py)
        self.charuco_board = cv2.aruco.CharucoBoard(
            pattern_size,
            square_size_mm,  # Use mm directly, no conversion needed
            marker_size_mm,  # Use mm directly, no conversion needed
            self.aruco_dict
        )
        
        # Create detectors
        self.detector_params = cv2.aruco.DetectorParameters()
        # Fix: Use CORNER_REFINE_APRILTAG instead of CORNER_REFINE_SUBPIX due to OpenCV bug
        # See: https://github.com/opencv/opencv/issues/24113
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # CharUco detector with parameters (camera matrix will be set when available)
        self.charuco_params = cv2.aruco.CharucoParameters()
        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board, self.charuco_params)
        
        # Camera calibration parameters (will be loaded or estimated)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_calibrated = False
        
        # Statistics
        self.detection_stats = {
            'total_attempts': 0,
            'successful_detections': 0,
            'total_corners': (pattern_size[0] - 1) * (pattern_size[1] - 1),
            'avg_detection_time': 0.0,
            'avg_corners_detected': 0.0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def _check_subpixel_accuracy(self, corners) -> bool:
        """
        Check if corners have sub-pixel accuracy (from reference implementation).
        
        Args:
            corners: Corner points to check
            
        Returns:
            True if corners have sub-pixel accuracy, False otherwise
        """
        if corners is None or len(corners) == 0:
            return False
            
        for corner in corners:
            for xy in np.asarray(corner).flatten():
                if not float(xy).is_integer():
                    return True
        return False
    
    def validate_board_dimensions(self, actual_square_mm: float, actual_marker_mm: float, 
                                tolerance_percent: float = 5.0) -> bool:
        """
        Validate that actual board dimensions match configured parameters.
        Critical for accuracy as emphasized in reference implementation.
        
        Args:
            actual_square_mm: Measured square size in mm
            actual_marker_mm: Measured marker size in mm
            tolerance_percent: Acceptable tolerance percentage (default 5%)
            
        Returns:
            True if dimensions are within tolerance
        """
        square_error = abs(actual_square_mm - self.square_size_mm) / self.square_size_mm * 100
        marker_error = abs(actual_marker_mm - self.marker_size_mm) / self.marker_size_mm * 100
        
        square_valid = square_error <= tolerance_percent
        marker_valid = marker_error <= tolerance_percent
        
        if not square_valid or not marker_valid:
            self.logger.error(f"Board dimension validation failed:")
            self.logger.error(f"  Square: configured={self.square_size_mm}mm, actual={actual_square_mm}mm, error={square_error:.1f}%")
            self.logger.error(f"  Marker: configured={self.marker_size_mm}mm, actual={actual_marker_mm}mm, error={marker_error:.1f}%")
            self.logger.error(f"  Tolerance: {tolerance_percent}%")
            return False
        
        self.logger.info(f"Board dimensions validated within {tolerance_percent}% tolerance")
        return True
    
    def cluster_objects(self, vertices: np.ndarray, color_image: Optional[np.ndarray] = None,
                       return_visualization: bool = False) -> List[Dict[str, Any]]:
        """
        Detect CharUco patterns in the color image.
        
        Args:
            vertices: Point cloud vertices (not used for CharUco detection)
            color_image: RGB color image for pattern detection
            return_visualization: Whether to return visualization data
            
        Returns:
            List of detected CharUco patterns with position, orientation, and quality metrics
        """
        start_time = time.time()
        self.detection_stats['total_attempts'] += 1
        
        if color_image is None:
            self.logger.warning("No color image provided for CharUco detection")
            return [] if not return_visualization else ([], None, None)
        
        # Convert to grayscale for detection
        if len(color_image.shape) == 3:
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = color_image
        
        detected_patterns = []
        visualization_data = None
        
        # Detect CharUco pattern
        pattern_info = self._detect_charuco_pattern(gray_image, color_image)
        if pattern_info:
            detected_patterns.append(pattern_info)
            self.detection_stats['successful_detections'] += 1
            
            # Update corner statistics
            corners_detected = pattern_info['corners_count']
            alpha = 0.1  # Exponential moving average
            self.detection_stats['avg_corners_detected'] = \
                alpha * corners_detected + (1 - alpha) * self.detection_stats['avg_corners_detected']
        
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
    
    def _detect_charuco_pattern(self, gray_image: np.ndarray, color_image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detect CharUco pattern using the proven approach from test_charuco_detection.py.
        
        Args:
            gray_image: Grayscale image for detection
            color_image: Color image for visualization
            
        Returns:
            Pattern information dictionary or None if not detected
        """
        # First detect ArUco markers
        marker_corners, marker_ids, rejected = self.aruco_detector.detectMarkers(gray_image)
        
        if marker_ids is None or len(marker_ids) == 0:
            return None
        
        # Detect CharUco corners - Enhancement: pass pre-detected markers for better accuracy
        # This approach follows the reference implementation for improved detection robustness
        charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(
            gray_image, None, None, marker_corners, marker_ids
        )
        
        if charuco_corners is None or len(charuco_corners) == 0:
            return None
        
        # Validate sub-pixel accuracy (enhancement from reference implementation)
        marker_subpixel = self._check_subpixel_accuracy(marker_corners)
        charuco_subpixel = self._check_subpixel_accuracy(charuco_corners)
        
        if not self.bypass_filtering and (not marker_subpixel or not charuco_subpixel):
            self.logger.warning(f"Sub-pixel accuracy check failed: markers={marker_subpixel}, charuco={charuco_subpixel}")
            # Note: We don't return None here to maintain compatibility, but log the warning
        
        # Skip all filtering if bypass_filtering is enabled (like test_charuco_detection.py)
        if self.bypass_filtering:
            # Use permissive detection - accept any detection with >0 corners
            total_corners = self.detection_stats['total_corners']
            coverage = len(charuco_corners) / total_corners
            quality_score = 1.0  # Set high quality score to bypass downstream checks
        else:
            # Apply normal filtering logic
            # Validate corner count - need minimum for quality detection
            if len(charuco_corners) < 4:
                return None
            
            # Calculate pattern quality metrics
            total_corners = self.detection_stats['total_corners']
            coverage = len(charuco_corners) / total_corners
            
            # Validate pattern coverage
            if coverage < self.min_pattern_coverage or coverage > self.max_pattern_coverage:
                return None
            
            # Calculate quality score
            quality_score = self._calculate_pattern_quality(charuco_corners, marker_ids, gray_image.shape)
            if quality_score < self.quality_threshold:
                return None
        
        # Calculate pattern center in image coordinates
        corners_2d = charuco_corners.reshape(-1, 2)
        center_2d = np.mean(corners_2d, axis=0)
        
        # Estimate 3D pose if camera is calibrated and have enough corners
        pose_data = None
        center_3d_mm = None
        
        # Use more permissive pose estimation when filtering is bypassed
        min_corners_required = 4 if self.bypass_filtering else self.min_corners_for_pose
        
        if (self.camera_calibrated and self.camera_matrix is not None and 
            len(charuco_corners) >= min_corners_required):
            pose_data = self._estimate_3d_pose(charuco_corners, charuco_ids)
            if pose_data:
                center_3d_mm = pose_data['translation_mm']
                # Removed verbose SUCCESS message - pose data available in pose_data
            else:
                if not self.bypass_filtering:  # Only show errors when not bypassing filtering
                    print("ERROR: 3D pose estimation failed")
        else:
            if not self.bypass_filtering:  # Only show errors when not bypassing filtering
                if not self.camera_calibrated:
                    print("ERROR: No camera calibration - cannot estimate 3D pose")
                elif len(charuco_corners) < min_corners_required:
                    print(f"ERROR: Insufficient corners for pose ({len(charuco_corners)} < {min_corners_required})")
        
        # Create pattern information dictionary compatible with clustering interface
        pattern_info = {
            'id': f"charuco_{self.pattern_size[0]}x{self.pattern_size[1]}",
            'pattern_type': 'charuco',
            'pattern_size': self.pattern_size,
            'corners_2d': corners_2d,  # Store as (N, 2) format
            'corners_opencv': charuco_corners,  # Store original OpenCV format for compatibility
            'corner_ids': charuco_ids.flatten() if charuco_ids is not None else None,
            'corners_count': len(corners_2d),
            'total_corners': total_corners,
            'marker_corners': marker_corners,
            'marker_ids': marker_ids.flatten() if marker_ids is not None else None,
            'markers_count': len(marker_ids) if marker_ids is not None else 0,
            'center_2d': center_2d,
            'center_3d': center_3d_mm / 1000.0 if center_3d_mm is not None else None,  # Convert to meters
            'center_mm': center_3d_mm,  # Keep mm version for compatibility
            'quality_score': quality_score,
            'coverage': coverage,
            'pose_data': pose_data,
            
            # Compatibility fields for ClusteringStrategy interface
            'points': corners_2d,  # 2D corner points
            'center': center_3d_mm / 1000.0 if center_3d_mm is not None else [center_2d[0]/1000, center_2d[1]/1000, 0],
            'num_points': len(corners_2d),
            'min_coords': [0, 0, 0],  # Placeholder
            'max_coords': [self.square_size_mm * self.pattern_size[0] / 1000, 
                          self.square_size_mm * self.pattern_size[1] / 1000, 0],
            'dimensions': [self.square_size_mm * self.pattern_size[0] / 1000, 
                          self.square_size_mm * self.pattern_size[1] / 1000, 0]
        }
        
        return pattern_info
    
    def _estimate_3d_pose(self, charuco_corners: np.ndarray, charuco_ids: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Estimate 3D pose of CharUco pattern using solvePnP.
        
        Args:
            charuco_corners: Detected CharUco corner points
            charuco_ids: IDs of detected corners
            
        Returns:
            Pose data dictionary with translation and rotation or None if failed
        """
        try:
            # Get object points for detected corners
            object_points = self.charuco_board.getChessboardCorners()
            selected_obj_points = object_points[charuco_ids.flatten()].copy()
            
            # CRITICAL COORDINATE SYSTEM FIX (from reference implementation):
            # Applies Y-axis flip for "old axis direction" to correct Z-axis orientation.
            # This ensures the coordinate system has:
            # - Origin: Bottom-left corner of CharUco pattern  
            # - X-axis: Right (red arrow)
            # - Y-axis: Up (green arrow)  
            # - Z-axis: Outward from paper (blue arrow)
            # Without this fix, Z-axis points inward (wrong direction for robot applications)
            # Reference: https://github.com/tony2guo/charuco-pose-estimation
            right_bottom = self.charuco_board.getRightBottomCorner()
            for i in range(len(selected_obj_points)):
                selected_obj_points[i][1] = right_bottom[1] - selected_obj_points[i][1]
            
            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                selected_obj_points,
                charuco_corners,
                self.camera_matrix,
                self.dist_coeffs
            )
            
            if not success:
                return None
            
            # Translation is already in millimeters (object points are in mm, result in mm)
            translation_mm = tvec.flatten()  # Already in mm, no conversion needed
            
            # Calculate reprojection error for quality assessment
            reprojected_corners, _ = cv2.projectPoints(
                selected_obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            reprojection_error = cv2.norm(charuco_corners, reprojected_corners, cv2.NORM_L2) / len(charuco_corners)
            
            pose_data = {
                'rvec': rvec.copy(),
                'tvec': tvec.copy(),
                'translation_mm': translation_mm,
                'reprojection_error': reprojection_error,
                'quality_score': max(0, 1.0 - reprojection_error / 10.0)  # Normalize to 0-1
            }
            
            return pose_data
            
        except Exception as e:
            self.logger.error(f"3D pose estimation failed: {e}")
            return None
    
    
    def _calculate_pattern_quality(self, charuco_corners: np.ndarray, marker_ids: np.ndarray, 
                                  image_shape: Tuple[int, int]) -> float:
        """
        Calculate CharUco pattern detection quality score.
        
        Args:
            charuco_corners: Detected CharUco corner points
            marker_ids: Detected ArUco marker IDs
            image_shape: Image dimensions (height, width)
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if charuco_corners is None or len(charuco_corners) < 4:
            return 0.0
        
        corners_2d = charuco_corners.reshape(-1, 2)
        
        # Factor 1: Corner coverage (more corners = better)
        total_corners = self.detection_stats['total_corners']
        coverage_score = min(1.0, len(charuco_corners) / total_corners)
        
        # Factor 2: Marker detection (more markers = better)
        total_expected_markers = self.pattern_size[0] * self.pattern_size[1] // 2  # Approximate
        marker_score = min(1.0, len(marker_ids) / total_expected_markers) if marker_ids is not None else 0.5
        
        # Factor 3: Pattern geometry (based on corner distribution)
        geometry_score = self._calculate_geometry_score(corners_2d)
        
        # Factor 4: Edge distance (penalize patterns too close to edges)
        edge_score = self._calculate_edge_distance_score(corners_2d, image_shape)
        
        # Combined quality score with CharUco-specific weighting
        quality = (coverage_score * 0.4 + 
                  marker_score * 0.3 + 
                  geometry_score * 0.2 + 
                  edge_score * 0.1)
        
        return np.clip(quality, 0.0, 1.0)
    
    def _calculate_geometry_score(self, corners: np.ndarray) -> float:
        """Calculate how well the pattern maintains good geometry."""
        if len(corners) < 4:
            return 0.0
        
        # Calculate bounding box aspect ratio
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width == 0 or height == 0:
            return 0.0
        
        # Calculate aspect ratio based on expected pattern dimensions
        expected_aspect = self.pattern_size[0] / self.pattern_size[1]
        actual_aspect = width / height
        
        # Score based on how close actual aspect ratio is to expected
        aspect_diff = abs(expected_aspect - actual_aspect) / expected_aspect
        aspect_score = max(0.0, 1.0 - aspect_diff * 2)  # Allow some deviation
        
        return aspect_score
    
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
        
        # Score: 1.0 if distance > 10% of image size, decreases towards edges
        if normalized_distance > 0.1:
            return 1.0
        else:
            return normalized_distance / 0.1
    
    def _create_visualization(self, color_image: np.ndarray, detected_patterns: List[Dict[str, Any]]) -> np.ndarray:
        """Create visualization of CharUco detection results."""
        vis_image = color_image.copy()
        
        # Convert RGB to BGR for OpenCV display
        if len(vis_image.shape) == 3:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        
        for pattern in detected_patterns:
            # Draw detected markers
            if 'marker_corners' in pattern and pattern['marker_corners'] is not None:
                cv2.aruco.drawDetectedMarkers(vis_image, pattern['marker_corners'], pattern['marker_ids'])
            
            # Draw CharUco corners
            if 'corners_opencv' in pattern:
                cv2.aruco.drawDetectedCornersCharuco(vis_image, pattern['corners_opencv'], pattern['corner_ids'])
            
            # Draw coordinate axes if pose is available
            if (pattern['pose_data'] is not None and 
                self.camera_calibrated and 
                self.camera_matrix is not None):
                rvec = pattern['pose_data']['rvec']
                tvec = pattern['pose_data']['tvec']
                axis_length = self.square_size_mm * 2
                cv2.drawFrameAxes(vis_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, axis_length, thickness=3)
            
            # Add pattern information text
            center_2d = pattern['center_2d'].astype(int)
            cv2.circle(vis_image, tuple(center_2d), 5, (0, 255, 0), -1)
            
            # Add text overlay
            y_offset = 30
            line_height = 25
            
            cv2.putText(vis_image, f"CharUco {self.pattern_size[0]}x{self.pattern_size[1]} DETECTED", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += line_height
            
            cv2.putText(vis_image, f"Corners: {pattern['corners_count']}/{pattern['total_corners']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
            
            cv2.putText(vis_image, f"Coverage: {pattern['coverage']:.1%}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
            
            cv2.putText(vis_image, f"Quality: {pattern['quality_score']:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += line_height
            
            cv2.putText(vis_image, f"Markers: {pattern['markers_count']}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if pattern['pose_data']:
                y_offset += line_height
                reproj_error = pattern['pose_data']['reprojection_error']
                cv2.putText(vis_image, f"Reproj Error: {reproj_error:.2f}px", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis_image
    
    def set_camera_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        Set camera calibration parameters for 3D pose estimation.
        
        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix.copy()
        self.dist_coeffs = dist_coeffs.copy()
        self.camera_calibrated = True
        
        # Enhancement: Set camera matrix in CharUco detector for improved accuracy
        self.charuco_params.cameraMatrix = camera_matrix.copy()
        self.charuco_detector = cv2.aruco.CharucoDetector(self.charuco_board, self.charuco_params)
        
        self.logger.info("Camera calibration parameters set for CharUco detection")
    
    def calibrate_camera_from_patterns(self, pattern_detections: List[Dict[str, Any]], 
                                     image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calibrate camera using detected CharUco patterns.
        
        Args:
            pattern_detections: List of pattern detection dictionaries
            image_size: Image size (width, height)
            
        Returns:
            Tuple of (camera_matrix, dist_coeffs, reprojection_error)
        """
        if len(pattern_detections) < 3:
            raise ValueError(f"Need at least 3 pattern detections, got {len(pattern_detections)}")
        
        # Extract CharUco corners and IDs from all detections
        all_charuco_corners = []
        all_charuco_ids = []
        
        for detection in pattern_detections:
            if 'corners_opencv' in detection and 'corner_ids' in detection:
                corners = detection['corners_opencv']
                ids = detection['corner_ids']
                
                if corners is not None and ids is not None and len(corners) >= 4:
                    all_charuco_corners.append(corners)
                    all_charuco_ids.append(ids.reshape(-1, 1))
        
        if len(all_charuco_corners) < 3:
            raise ValueError(f"Insufficient valid detections for calibration: {len(all_charuco_corners)}")
        
        # Perform CharUco camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners,
            all_charuco_ids,
            self.charuco_board,
            image_size,
            None, None
        )
        
        if not ret:
            raise RuntimeError("CharUco camera calibration failed")
        
        # Calculate reprojection error
        total_error = 0
        total_points = 0
        
        for i in range(len(all_charuco_corners)):
            # Get object points for detected corners
            object_points = self.charuco_board.getChessboardCorners()
            selected_obj_points = object_points[all_charuco_ids[i].flatten()]
            
            # Project points and calculate error
            projected_corners, _ = cv2.projectPoints(
                selected_obj_points, rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            
            error = cv2.norm(all_charuco_corners[i], projected_corners, cv2.NORM_L2)
            total_error += error
            total_points += len(all_charuco_corners[i])
        
        mean_error = total_error / total_points
        
        # Store calibration
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_calibrated = True
        
        return camera_matrix, dist_coeffs, mean_error
    
    def get_strategy_name(self) -> str:
        """Get the name of this detection strategy."""
        return "CharUco Detection"
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        success_rate = (self.detection_stats['successful_detections'] / 
                       max(1, self.detection_stats['total_attempts'])) * 100
        
        return {
            **self.detection_stats,
            'success_rate': success_rate,
            'pattern_info': f"{self.pattern_size[0]}x{self.pattern_size[1]} CharUco",
            'square_size_mm': self.square_size_mm,
            'marker_size_mm': self.marker_size_mm
        }
    
    def export_transformation_matrices(self, pose_data: Dict[str, Any], output_dir: str = ".") -> bool:
        """
        Export transformation matrices in standard text format (like reference implementation).
        
        Args:
            pose_data: Pose data containing rvec and tvec
            output_dir: Directory to save the files
            
        Returns:
            True if export successful
        """
        try:
            import os
            
            if not pose_data or 'rvec' not in pose_data or 'tvec' not in pose_data:
                self.logger.error("Invalid pose data for export")
                return False
            
            rvec = pose_data['rvec']
            tvec = pose_data['tvec']
            
            # Convert to transformation matrix (same as reference implementation)
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            charuco_tf = np.eye(4)
            charuco_tf[0:3, 0:3] = rotation_matrix
            charuco_tf[0:3, 3] = tvec.flatten()
            
            # Camera transformation matrix (inverse)
            camera_position = -rotation_matrix.T @ tvec.flatten()
            camera_tf = np.eye(4) 
            camera_tf[0:3, 0:3] = rotation_matrix.T
            camera_tf[0:3, 3] = camera_position
            
            # Export files
            charuco_file = os.path.join(output_dir, "charuco_tf.txt")
            camera_file = os.path.join(output_dir, "camera_tf.txt")
            
            np.savetxt(charuco_file, charuco_tf)
            np.savetxt(camera_file, camera_tf)
            
            self.logger.info(f"Transformation matrices exported:")
            self.logger.info(f"  CharUco transform: {charuco_file}")
            self.logger.info(f"  Camera transform: {camera_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export transformation matrices: {e}")
            return False