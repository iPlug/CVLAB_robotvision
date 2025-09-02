"""
Floor-Aligned Object Pose Calculator

This module calculates complete 6DOF poses for objects placed on surfaces using
the detected floor plane as a reference frame. It provides the missing orientation
information for LiDAR-based object tracking.

Key Features:
- Floor-relative coordinate system generation
- Simple pose calculation for surface-placed objects  
- Z-axis aligned with floor normal (objects sit upright)
- Optional yaw rotation for object facing direction
- Integration with existing hand-eye calibration matrices

Design Philosophy:
Objects on surfaces have constrained orientations - they don't tilt randomly.
The floor normal provides the "up" direction, eliminating the need for complex
principal axis analysis.
"""

import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple, List
from scipy.spatial.transform import Rotation
import time


class FloorAlignedPoseCalculator:
    """
    Calculate complete 6DOF object poses using floor plane as orientation reference.
    
    For objects sitting on surfaces, the pose is naturally constrained:
    - Z-axis: Floor normal direction (objects sit upright)
    - X/Y rotations: Zero (objects don't tilt on flat surfaces)
    - Z-rotation (yaw): Optional based on object orientation or default to 0
    """
    
    def __init__(self, 
                 default_yaw: float = 0.0,
                 coordinate_system: str = "standard"):
        """
        Initialize floor-aligned pose calculator.
        
        Args:
            default_yaw: Default yaw rotation in degrees when no specific orientation detected
            coordinate_system: Target coordinate system ("standard", "robotics", "opencv")
        """
        self.default_yaw = np.radians(default_yaw)
        self.coordinate_system = coordinate_system
        
        # Performance tracking
        self.calculation_stats = {
            'total_calculations': 0,
            'successful_calculations': 0,
            'avg_calculation_time': 0.0
        }
    
    def calculate_floor_aligned_pose(self, 
                                   object_position: np.ndarray,
                                   floor_normal: np.ndarray,
                                   object_yaw: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate complete 6DOF pose for an object on a surface.
        
        Args:
            object_position: 3D position of object center [x, y, z] in meters
            floor_normal: Normal vector of the floor plane [nx, ny, nz]
            object_yaw: Optional yaw rotation in radians, uses default if None
            
        Returns:
            Dictionary containing complete pose information
        """
        start_time = time.time()
        
        try:
            print(f"DEBUG: Floor pose calc - position: {object_position}, normal: {floor_normal}")
            # Use default yaw if none specified
            yaw = object_yaw if object_yaw is not None else self.default_yaw
            
            # Create floor-aligned transformation matrix
            transformation_matrix = self._create_floor_aligned_transform(
                object_position, floor_normal, yaw
            )
            print("DEBUG: Floor pose calc - transformation matrix created successfully")
            
            # Extract orientation as Euler angles
            rotation_matrix = transformation_matrix[:3, :3]
            orientation_euler = self._rotation_matrix_to_euler(rotation_matrix)
            
            # Calculate pose metrics
            floor_alignment_score = self._calculate_floor_alignment_score(rotation_matrix, floor_normal)
            
            # Update statistics
            calculation_time = time.time() - start_time
            self._update_calculation_stats(calculation_time, True)
            
            result = {
                'transformation_matrix': transformation_matrix,
                'position': object_position.copy(),
                'orientation_euler': orientation_euler,  # [rx, ry, rz] in radians
                'orientation_degrees': np.degrees(orientation_euler),  # [rx, ry, rz] in degrees
                'rotation_matrix': rotation_matrix,
                'floor_normal': floor_normal.copy(),
                'yaw_rotation': yaw,
                'floor_alignment_score': floor_alignment_score,
                'coordinate_system': self.coordinate_system,
                'calculation_time': calculation_time,
                'is_valid': True
            }
            
            print("DEBUG: Floor pose calc - SUCCESS, returning result")
            return result
            
        except Exception as e:
            print(f"Error calculating floor-aligned pose: {e}")
            self._update_calculation_stats(time.time() - start_time, False)
            
            return {
                'transformation_matrix': np.eye(4),
                'position': object_position.copy() if object_position is not None else np.zeros(3),
                'orientation_euler': np.zeros(3),
                'orientation_degrees': np.zeros(3),
                'rotation_matrix': np.eye(3),
                'floor_normal': floor_normal.copy() if floor_normal is not None else np.array([0, 0, 1]),
                'yaw_rotation': 0.0,
                'floor_alignment_score': 0.0,
                'coordinate_system': self.coordinate_system,
                'calculation_time': time.time() - start_time,
                'is_valid': False,
                'error': str(e)
            }
    
    def _create_floor_aligned_transform(self, 
                                      position: np.ndarray, 
                                      floor_normal: np.ndarray, 
                                      yaw: float = 0.0) -> np.ndarray:
        """
        Create 4x4 transformation matrix for floor-aligned object pose.
        
        Args:
            position: Object position [x, y, z]
            floor_normal: Floor plane normal vector [nx, ny, nz]
            yaw: Yaw rotation in radians around floor normal
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        # Normalize floor normal to ensure unit vector
        z_axis = floor_normal / np.linalg.norm(floor_normal)
        
        # Create orthogonal coordinate system with Z pointing along floor normal
        # Choose X-axis perpendicular to floor normal
        if abs(z_axis[2]) < 0.9:  # Floor not nearly vertical
            # Use world Z-axis to create perpendicular vector
            x_axis = np.cross([0, 0, 1], z_axis)
        else:  # Floor is nearly vertical, use world X-axis
            x_axis = np.cross([1, 0, 0], z_axis)
        
        # Normalize X-axis
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Y-axis completes right-hand coordinate system
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Apply yaw rotation around Z-axis if specified
        if abs(yaw) > 1e-6:
            # Create rotation matrix for yaw around Z-axis
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            yaw_rotation = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw,  cos_yaw, 0],
                [0,        0,       1]
            ])
            
            # Apply yaw rotation to X and Y axes
            base_rotation = np.column_stack([x_axis, y_axis, z_axis])
            final_rotation = base_rotation @ yaw_rotation
        else:
            # No yaw rotation, use base coordinate system
            final_rotation = np.column_stack([x_axis, y_axis, z_axis])
        
        # Create 4x4 transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = final_rotation
        transformation[:3, 3] = position
        
        return transformation
    
    def _rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to Euler angles (XYZ intrinsic order)."""
        try:
            # Use scipy for robust rotation matrix to Euler conversion
            r = Rotation.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=False)  # Radians
            return euler_angles
        except Exception as e:
            print(f"Warning: Euler angle conversion failed: {e}")
            return np.zeros(3)
    
    def _calculate_floor_alignment_score(self, 
                                       rotation_matrix: np.ndarray, 
                                       floor_normal: np.ndarray) -> float:
        """
        Calculate how well the object's Z-axis aligns with the floor normal.
        
        Returns:
            Score from 0.0 (perpendicular) to 1.0 (perfectly aligned)
        """
        try:
            # Extract object's Z-axis from rotation matrix
            object_z_axis = rotation_matrix[:, 2]
            
            # Calculate cosine of angle between object Z-axis and floor normal
            cos_angle = np.dot(object_z_axis, floor_normal)
            
            # Convert to alignment score (0 to 1)
            alignment_score = abs(cos_angle)  # Take absolute value for bidirectional alignment
            
            return float(alignment_score)
            
        except Exception:
            return 0.0
    
    def calculate_multiple_poses(self, 
                               object_positions: List[np.ndarray],
                               floor_normal: np.ndarray,
                               yaw_angles: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Calculate floor-aligned poses for multiple objects.
        
        Args:
            object_positions: List of 3D object positions
            floor_normal: Single floor normal vector for all objects
            yaw_angles: Optional list of yaw angles (one per object)
            
        Returns:
            List of pose dictionaries
        """
        if yaw_angles is None:
            yaw_angles = [None] * len(object_positions)
        
        poses = []
        for i, (position, yaw) in enumerate(zip(object_positions, yaw_angles)):
            pose = self.calculate_floor_aligned_pose(position, floor_normal, yaw)
            pose['object_id'] = i
            poses.append(pose)
        
        return poses
    
    def estimate_object_yaw_from_shape(self, 
                                     object_points: np.ndarray,
                                     floor_normal: np.ndarray) -> float:
        """
        Estimate object yaw rotation based on its point cloud shape.
        
        This is optional - provides a simple way to orient objects based on
        their principal axis if needed.
        
        Args:
            object_points: Point cloud of the object [N, 3]
            floor_normal: Floor normal vector
            
        Returns:
            Estimated yaw angle in radians
        """
        try:
            if len(object_points) < 10:
                return 0.0
            
            # Project points to floor plane for 2D analysis
            floor_normal_unit = floor_normal / np.linalg.norm(floor_normal)
            
            # Create projection matrix to remove floor normal component
            I = np.eye(3)
            projection_matrix = I - np.outer(floor_normal_unit, floor_normal_unit)
            
            # Project points to floor plane
            projected_points = (projection_matrix @ object_points.T).T
            
            # Find principal axis in the floor plane using PCA
            centroid = np.mean(projected_points, axis=0)
            centered_points = projected_points - centroid
            
            # Compute covariance matrix
            cov_matrix = np.cov(centered_points.T)
            
            # Find principal axis (largest eigenvalue)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            principal_axis = eigenvectors[:, -1]  # Eigenvector with largest eigenvalue
            
            # Calculate yaw angle from principal axis
            yaw = np.arctan2(principal_axis[1], principal_axis[0])
            
            return yaw
            
        except Exception as e:
            print(f"Warning: Object yaw estimation failed: {e}")
            return 0.0
    
    def validate_floor_normal(self, floor_normal: np.ndarray) -> bool:
        """
        Validate that floor normal is reasonable for pose calculation.
        
        Args:
            floor_normal: Floor normal vector to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if it's a 3D vector
            if len(floor_normal) != 3:
                return False
            
            # Check if it's not zero vector
            norm = np.linalg.norm(floor_normal)
            if norm < 1e-6:
                return False
            
            # Check if it's pointing roughly upward (optional - depends on setup)
            # This assumes camera is looking down at floor
            normalized_normal = floor_normal / norm
            upward_component = abs(normalized_normal[1])  # Y component
            
            # Allow normals that have some upward component (not purely horizontal)
            return upward_component > 0.3  # At least 30% upward component
            
        except Exception:
            return False
    
    def _update_calculation_stats(self, calculation_time: float, success: bool):
        """Update performance statistics."""
        self.calculation_stats['total_calculations'] += 1
        
        if success:
            self.calculation_stats['successful_calculations'] += 1
        
        # Update running average for calculation time
        total = self.calculation_stats['total_calculations']
        prev_avg = self.calculation_stats['avg_calculation_time']
        self.calculation_stats['avg_calculation_time'] = (
            (prev_avg * (total - 1) + calculation_time) / total
        )
    
    def get_calculation_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.calculation_stats.copy()
        
        if stats['total_calculations'] > 0:
            stats['success_rate'] = stats['successful_calculations'] / stats['total_calculations']
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def create_pose_visualization(self, 
                                pose_result: Dict[str, Any],
                                image_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """
        Create visualization of calculated pose.
        
        Args:
            pose_result: Result from calculate_floor_aligned_pose()
            image_size: Output image size
            
        Returns:
            BGR image showing pose information
        """
        width, height = image_size
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not pose_result.get('is_valid', False):
            cv2.putText(vis_image, "Invalid Pose", (width//2 - 50, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return vis_image
        
        # Display pose information
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        
        # Position
        pos = pose_result['position']
        cv2.putText(vis_image, f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]m", 
                   (10, y_offset), font, font_scale, color, 1)
        y_offset += 30
        
        # Orientation in degrees
        orient = pose_result['orientation_degrees']
        cv2.putText(vis_image, f"Rotation: [{orient[0]:.1f}, {orient[1]:.1f}, {orient[2]:.1f}]°", 
                   (10, y_offset), font, font_scale, color, 1)
        y_offset += 30
        
        # Floor normal
        normal = pose_result['floor_normal']
        cv2.putText(vis_image, f"Floor Normal: [{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]", 
                   (10, y_offset), font, font_scale, color, 1)
        y_offset += 30
        
        # Alignment score
        alignment = pose_result['floor_alignment_score']
        cv2.putText(vis_image, f"Floor Alignment: {alignment:.3f}", 
                   (10, y_offset), font, font_scale, (0, 255, 0) if alignment > 0.8 else (0, 255, 255), 1)
        y_offset += 30
        
        # Yaw rotation
        yaw_deg = np.degrees(pose_result['yaw_rotation'])
        cv2.putText(vis_image, f"Yaw: {yaw_deg:.1f}°", 
                   (10, y_offset), font, font_scale, color, 1)
        
        return vis_image