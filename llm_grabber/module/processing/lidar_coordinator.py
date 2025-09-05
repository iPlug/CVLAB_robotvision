"""
LiDAR Coordinate Transformation and Robot Integration

This module handles the transformation of LiDAR-detected object centers from camera
coordinates to robot coordinates using the existing hand-eye calibration matrix.
It integrates seamlessly with the RGB-based calibration system.

Key Features:
- Center point calculation from 3D clusters
- Integration with existing hand-eye transformation matrix
- Support for multiple coordinate transformation methods
- Robot pose calculation and validation
- Comprehensive error handling and fallback mechanisms
"""

import numpy as np
import cv2
import time
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json

# Import the new floor-aligned pose calculator
from module.processing.floor_aligned_pose_calculator import FloorAlignedPoseCalculator


class LiDARCoordinator:
    """
    Coordinate transformation system for LiDAR-based object tracking.
    
    Integrates with existing hand-eye calibration to transform object centers
    from camera coordinates to robot coordinates for accurate robotic control.
    """
    
    def __init__(self, 
                 transformation_matrix_file: str = "camera_robot_transform.npy",
                 alternative_files: Optional[List[str]] = None,
                 coordinate_system: str = "standard"):
        """
        Initialize LiDAR coordinate transformation system.
        
        Args:
            transformation_matrix_file: Path to hand-eye transformation matrix
            alternative_files: List of alternative transformation matrix files to try
            coordinate_system: Coordinate system convention ("standard", "robotics", "opencv")
        """
        self.transformation_matrix_file = transformation_matrix_file
        self.alternative_files = alternative_files or [
            "eye_in_hand_transform.npy",
            "charuco_eye_in_hand_transform.npy", 
            "camera_robot_transform_work.npy"
        ]
        self.coordinate_system = coordinate_system
        
        # Transformation matrix (4x4 homogeneous)
        self.hand_eye_matrix = None  # This is T_robot_to_depth from calibration
        self.robot_to_color_matrix = None  # This is T_robot_to_color (calculated)
        self.matrix_source = None
        self.matrix_timestamp = None
        
        # Validation parameters (temporarily expanded for debugging)
        self.workspace_bounds = {
            'x_min': -1000, 'x_max': 1000,    # mm (expanded)
            'y_min': -1000, 'y_max': 1000,    # mm (expanded)
            'z_min': -100,   'z_max': 1000   # mm (expanded)
        }
        
        # Robot coordinate system info (myCobot 280)
        self.robot_info = {
            'name': 'myCobot 280',
            'coordinate_system': '+X=forward, +Y=left, +Z=up',
            'units': 'millimeters',
            'working_radius': 280,
            'max_reach': 350
        }
        
        # Performance tracking
        self.transformation_stats = {
            'total_transformations': 0,
            'successful_transformations': 0,
            'out_of_bounds_count': 0,
            'avg_transformation_time': 0.0
        }
        
        # Initialize floor-aligned pose calculator
        self.pose_calculator = FloorAlignedPoseCalculator(
            coordinate_system=coordinate_system
        )
        
        # Load transformation matrix
        self.load_transformation_matrix()
        
        # Initialize sensor manager reference (will be set by parent application)
        self.sensor_manager = None
    
    def set_sensor_manager(self, sensor_manager):
        """Set sensor manager and calculate robot-to-color transformation."""
        self.sensor_manager = sensor_manager
        self._calculate_robot_to_color_transform()
    
    def _calculate_robot_to_color_transform(self):
        """Calculate T_robot_to_color from T_robot_to_depth and camera extrinsics."""
        if self.sensor_manager is None or self.hand_eye_matrix is None:
            print("DEBUG: Cannot calculate robot-to-color transform - missing sensor manager or hand-eye matrix")
            return
        
        try:
            # Get camera extrinsics (T_depth_to_color)
            extrinsics_data = self.sensor_manager.get_factory_extrinsics()
            if extrinsics_data is None:
                print("DEBUG: Failed to get camera extrinsics")
                return
            
            T_depth_to_color = extrinsics_data['transformation_matrix']
            # Invert to get T_color_to_depth  
            T_color_to_depth = np.linalg.inv(T_depth_to_color)
            
            # Calculate T_robot_to_color following CharUco approach:
            # T_robot_to_depth = T_robot_to_color @ T_color_to_depth
            # T_robot_to_color = T_robot_to_depth @ inv(T_color_to_depth)
            self.robot_to_color_matrix = self.hand_eye_matrix @ np.linalg.inv(T_color_to_depth)
            
            print("DEBUG: Calculated robot-to-color transformation matrix")
            print(f"DEBUG: Robot-to-color translation: {self.robot_to_color_matrix[:3, 3] * 1000}mm")
            
        except Exception as e:
            print(f"DEBUG: Error calculating robot-to-color transform: {e}")
            # Fall back to using hand-eye matrix directly
            self.robot_to_color_matrix = self.hand_eye_matrix
    
    def load_transformation_matrix(self) -> bool:
        """
        Load hand-eye transformation matrix from file.
        
        Returns:
            True if matrix loaded successfully, False otherwise
        """
        # Try primary file first
        files_to_try = [self.transformation_matrix_file] + self.alternative_files
        
        for file_path in files_to_try:
            if self._try_load_matrix_from_file(file_path):
                return True
        
        # If no matrix file found, show available files
        self._show_available_transformation_files()
        print("Error: No valid transformation matrix found!")
        print("Please run calibration first (e.g., charuco_eye_in_hand.py)")
        
        return False
    
    def _try_load_matrix_from_file(self, file_path: str) -> bool:
        """Try to load transformation matrix from a specific file."""
        try:
            if not Path(file_path).exists():
                return False
            
            # Load matrix
            matrix = np.load(file_path)
            
            # Validate matrix dimensions and properties
            if not self._validate_transformation_matrix(matrix):
                print(f"Warning: Invalid transformation matrix in {file_path}")
                return False
            
            self.hand_eye_matrix = matrix
            self.matrix_source = file_path
            self.matrix_timestamp = Path(file_path).stat().st_mtime
            
            print(f"Loaded transformation matrix from: {file_path}")
            print(f"Matrix shape: {matrix.shape}")
            print(f"Matrix determinant: {np.linalg.det(matrix[:3, :3]):.6f}")
            
            return True
            
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return False
    
    def _validate_transformation_matrix(self, matrix: np.ndarray) -> bool:
        """Validate transformation matrix properties."""
        if matrix.shape != (4, 4):
            return False
        
        # Check if it's a valid homogeneous transformation matrix
        if not np.allclose(matrix[3, :], [0, 0, 0, 1]):
            return False
        
        # Check if rotation part is orthogonal
        rotation_part = matrix[:3, :3]
        det = np.linalg.det(rotation_part)
        
        if not np.isclose(det, 1.0, atol=0.1):  # Allow some tolerance
            return False
        
        return True
    
    def _show_available_transformation_files(self):
        """Show available transformation matrix files in current directory."""
        print("\nLooking for transformation matrix files...")
        
        # Common patterns for transformation matrices
        patterns = ["*transform*.npy", "*calibration*.npy", "*eye_in_hand*.npy"]
        
        found_files = []
        for pattern in patterns:
            found_files.extend(Path(".").glob(pattern))
        
        if found_files:
            print("Found potential transformation files:")
            for f in found_files:
                print(f"  {f}")
        else:
            print("No transformation matrix files found in current directory.")
    
    def transform_point_to_robot(self, camera_point: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Transform a 3D point from camera coordinates to robot coordinates.
        
        Args:
            camera_point: 3D point in camera coordinates [x, y, z] (meters)
            
        Returns:
            Dictionary with robot coordinates and metadata, or None if transformation fails
        """
        start_time = time.time()
        
        print(f"DEBUG: transform_point_to_robot - input point: {camera_point}")
        
        if self.hand_eye_matrix is None:
            print("DEBUG: transform_point_to_robot - hand_eye_matrix is None")
            return None
        
        # Convert point to homogeneous coordinates
        if len(camera_point) == 3:
            camera_point_homo = np.append(camera_point, 1.0)
        else:
            camera_point_homo = camera_point
        
        try:
            # Use the correct transformation matrix (robot-to-color if available)
            transform_matrix = self.robot_to_color_matrix if self.robot_to_color_matrix is not None else self.hand_eye_matrix
            print(f"DEBUG: Using transform matrix translation: {transform_matrix[:3, 3] * 1000}mm")
            
            # Transform point using hand-eye calibration matrix
            robot_point_homo = transform_matrix @ camera_point_homo
            robot_point = robot_point_homo[:3]
            
            print(f"DEBUG: Transformed point (meters): {robot_point}")
            
            # Convert from meters to millimeters (robot units)
            robot_point_mm = robot_point * 1000.0
            
            print(f"DEBUG: Transformed point (mm): {robot_point_mm}")
            
            # Apply coordinate system conversion if needed
            if self.coordinate_system == "standard":
                # Standard robotics: X=forward, Y=left, Z=up
                robot_coords = robot_point_mm
            elif self.coordinate_system == "opencv":
                # OpenCV: X=right, Y=down, Z=forward -> Robot: X=forward, Y=left, Z=up
                robot_coords = np.array([robot_point_mm[2], -robot_point_mm[0], -robot_point_mm[1]])
            else:
                robot_coords = robot_point_mm
            
            # Validate coordinates are within workspace
            is_valid = self._validate_robot_coordinates(robot_coords)
            
            # Calculate distance from robot base
            distance_from_base = np.linalg.norm(robot_coords)
            
            # Update statistics
            transformation_time = time.time() - start_time
            self._update_transformation_stats(transformation_time, is_valid)
            
            result = {
                'robot_position': robot_coords,  # [x, y, z] in mm
                'camera_position': camera_point,  # [x, y, z] in meters
                'distance_from_base': distance_from_base,
                'is_valid': is_valid,
                'workspace_bounds': self.workspace_bounds,
                'transformation_time': transformation_time,
                'coordinate_system': self.coordinate_system,
                'matrix_source': self.matrix_source
            }
            
            # Add pose suggestion for robot movement
            if is_valid:
                result['suggested_pose'] = self._calculate_robot_pose(robot_coords)
            
            return result
            
        except Exception as e:
            print(f"Error in coordinate transformation: {e}")
            return None
    
    def transform_cluster_center(self, cluster_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform cluster center point to robot coordinates.
        
        Args:
            cluster_info: Cluster dictionary from LiDAR clustering strategy
            
        Returns:
            Transformed coordinates with additional cluster metadata
        """
        if 'center' not in cluster_info:
            print("DEBUG: transform_cluster_center - missing center key")
            return None
        
        # Transform the center point
        result = self.transform_point_to_robot(cluster_info['center'])
        
        if result is None:
            print("DEBUG: transform_cluster_center - transform_point_to_robot failed")
            return None
        else:
            print(f"DEBUG: transform_cluster_center - SUCCESS: {result['robot_position']}")
        
        # Add cluster-specific information
        result.update({
            'cluster_size': cluster_info.get('size', 0),
            'cluster_distance': cluster_info.get('distance', 0.0),
            'cluster_dimensions': cluster_info.get('dimensions', np.array([0, 0, 0])),
            'cluster_density': cluster_info.get('density', 0.0),
            'selection_method': cluster_info.get('selection_method', 'unknown')
        })
        
        return result
    
    def transform_cluster_pose(self, 
                             cluster_info: Dict[str, Any], 
                             floor_normal: Optional[np.ndarray] = None,
                             floor_detection_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Transform complete cluster pose (position + orientation) to robot coordinates.
        
        Args:
            cluster_info: Cluster dictionary from LiDAR clustering strategy
            floor_normal: Floor normal vector in camera coordinates [nx, ny, nz]
            floor_detection_info: Additional floor detection information
            
        Returns:
            Complete 6DOF pose information in robot coordinates
        """
        if 'center' not in cluster_info:
            print("DEBUG: Cluster info missing 'center' key")
            return None
        
        start_time = time.time()
        
        try:
            # Get cluster center in camera coordinates
            camera_position = cluster_info['center']  # [x, y, z] in meters
            print(f"DEBUG: Camera position: {camera_position}")
            print(f"DEBUG: Expected robot position should be around [500, 0, 0]mm")
            
            if floor_normal is not None:
                print(f"DEBUG: Floor normal available: {floor_normal}")
                
                # ONLY USE CAMERA FRAME - eye-in-hand calibration was done with camera frame
                print("DEBUG: Using camera frame only (eye-in-hand calibration coordinate system):")
                
                # Transform camera frame position to robot coordinates
                camera_robot_result = self.transform_point_to_robot(camera_position)
                if camera_robot_result:
                    camera_robot_pos = camera_robot_result['robot_position']
                    print(f"DEBUG: Camera frame → Robot: {camera_robot_pos}")
                else:
                    print("DEBUG: Camera frame transformation failed")
                
                # Calculate floor-aligned pose in camera coordinates
                pose_result = self.pose_calculator.calculate_floor_aligned_pose(
                    camera_position, floor_normal
                )
                
                if not pose_result['is_valid']:
                    print("DEBUG: Floor-aligned pose calculation failed, falling back to position-only")
                    # Fall back to position-only transformation
                    return self.transform_cluster_center(cluster_info)
                
                # Transform complete 4x4 pose matrix to robot coordinates
                camera_pose_matrix = pose_result['transformation_matrix']
                print("DEBUG: About to transform pose matrix to robot coordinates")
                robot_pose_matrix = self._transform_pose_matrix_to_robot(camera_pose_matrix)
                
                if robot_pose_matrix is None:
                    print("DEBUG: Robot pose matrix transformation failed, falling back")
                    return self.transform_cluster_center(cluster_info)
                else:
                    print("DEBUG: Robot pose matrix transformation succeeded")
                
                # Extract position and orientation from robot pose matrix
                robot_position = robot_pose_matrix[:3, 3] * 1000.0  # Convert to mm
                print(f"DEBUG: Extracted robot position: {robot_position}")
                robot_rotation_matrix = robot_pose_matrix[:3, :3]
                
                # Convert rotation matrix to Euler angles
                robot_orientation = self._rotation_matrix_to_euler(robot_rotation_matrix)
                print(f"DEBUG: Calculated robot orientation: {robot_orientation}")
                
                # Transform floor normal to robot coordinates
                floor_normal_robot = self._transform_vector_to_robot(floor_normal)
                print(f"DEBUG: Transformed floor normal: {floor_normal_robot}")
                
            else:
                print("DEBUG: No floor normal available, using position-only transformation")
                # No floor information available - use position-only transformation
                position_result = self.transform_point_to_robot(camera_position)
                if position_result is None:
                    print("DEBUG: Position-only transformation failed")
                    return None
                else:
                    print("DEBUG: Position-only transformation succeeded")
                
                robot_position = position_result['robot_position']
                robot_orientation = np.array([0.0, 0.0, 0.0])  # Default orientation
                robot_pose_matrix = np.eye(4)
                robot_pose_matrix[:3, 3] = robot_position / 1000.0  # Convert back to meters for matrix
                floor_normal_robot = None
            
            # Validate coordinates
            is_valid = self._validate_robot_coordinates(robot_position)
            distance_from_base = np.linalg.norm(robot_position)
            print(f"DEBUG: Coordinate validation result: is_valid={is_valid}, distance={distance_from_base:.1f}mm")
            
            # Update statistics
            transformation_time = time.time() - start_time
            self._update_transformation_stats(transformation_time, is_valid)
            
            # Build result dictionary
            result = {
                # Complete 6DOF pose information
                'robot_position': robot_position,  # [x, y, z] in mm
                'robot_orientation': robot_orientation,  # [rx, ry, rz] in radians
                'robot_orientation_degrees': np.degrees(robot_orientation),  # [rx, ry, rz] in degrees
                'transformation_matrix': robot_pose_matrix,  # 4x4 matrix
                
                # Original information
                'camera_position': camera_position,  # [x, y, z] in meters
                'distance_from_base': distance_from_base,
                'is_valid': is_valid,
                
                # Floor information
                'floor_normal_camera': floor_normal.copy() if floor_normal is not None else None,
                'floor_normal_robot': floor_normal_robot,
                'floor_detection_info': floor_detection_info,
                
                # Cluster information
                'cluster_size': cluster_info.get('size', 0),
                'cluster_distance': cluster_info.get('distance', 0.0),
                'cluster_dimensions': cluster_info.get('dimensions', np.array([0, 0, 0])),
                'cluster_density': cluster_info.get('density', 0.0),
                'selection_method': cluster_info.get('selection_method', 'unknown'),
                
                # Metadata
                'transformation_time': transformation_time,
                'coordinate_system': self.coordinate_system,
                'matrix_source': self.matrix_source,
                'workspace_bounds': self.workspace_bounds,
                'has_orientation': floor_normal is not None
            }
            
            # Add pose suggestion for robot movement
            if is_valid:
                result['suggested_pose'] = self._calculate_robot_pose_6dof(robot_position, robot_orientation)
            
            print(f"DEBUG: About to return result with is_valid={result['is_valid']}")
            return result
            
        except Exception as e:
            print(f"Error in cluster pose transformation: {e}")
            # Fall back to position-only transformation
            return self.transform_cluster_center(cluster_info)
    
    def _transform_pose_matrix_to_robot(self, camera_pose_matrix: np.ndarray) -> Optional[np.ndarray]:
        """Transform 4x4 pose matrix from camera coordinates to robot coordinates."""
        # Use the correct robot-to-color transformation matrix
        transform_matrix = self.robot_to_color_matrix if self.robot_to_color_matrix is not None else self.hand_eye_matrix
        
        if transform_matrix is None:
            print("DEBUG: _transform_pose_matrix_to_robot - no transformation matrix available")
            return None
        
        try:
            print("DEBUG: _transform_pose_matrix_to_robot - performing matrix multiplication")
            print(f"DEBUG: Using transform matrix translation: {transform_matrix[:3, 3] * 1000}mm")  # Convert to mm
            
            # Transform pose using correct robot-to-color calibration: T_robot = T_robot_to_color * T_color_to_object
            robot_pose_matrix = transform_matrix @ camera_pose_matrix
            print("DEBUG: _transform_pose_matrix_to_robot - SUCCESS")
            return robot_pose_matrix
        except Exception as e:
            print(f"DEBUG: _transform_pose_matrix_to_robot - ERROR: {e}")
            return None
    
    def _transform_vector_to_robot(self, camera_vector: np.ndarray) -> Optional[np.ndarray]:
        """Transform a vector from camera coordinates to robot coordinates."""
        if self.hand_eye_matrix is None:
            return None
        
        try:
            # Transform only the rotation part (ignore translation)
            rotation_part = self.hand_eye_matrix[:3, :3]
            robot_vector = rotation_part @ camera_vector
            return robot_vector
        except Exception:
            return None
    
    def _rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to Euler angles (XYZ intrinsic order)."""
        try:
            from scipy.spatial.transform import Rotation
            r = Rotation.from_matrix(rotation_matrix)
            euler_angles = r.as_euler('xyz', degrees=False)  # Radians
            return euler_angles
        except Exception as e:
            print(f"Warning: Euler angle conversion failed: {e}")
            return np.zeros(3)
    
    def _validate_robot_coordinates(self, robot_coords: np.ndarray) -> bool:
        """Validate that robot coordinates are within workspace bounds."""
        x, y, z = robot_coords
        
        bounds = self.workspace_bounds
        
        if not (bounds['x_min'] <= x <= bounds['x_max']):
            print(f"DEBUG: X coordinate {x:.0f}mm outside bounds [{bounds['x_min']}, {bounds['x_max']}]")
            return False
        if not (bounds['y_min'] <= y <= bounds['y_max']):
            print(f"DEBUG: Y coordinate {y:.0f}mm outside bounds [{bounds['y_min']}, {bounds['y_max']}]")
            return False
        if not (bounds['z_min'] <= z <= bounds['z_max']):
            print(f"DEBUG: Z coordinate {z:.0f}mm outside bounds [{bounds['z_min']}, {bounds['z_max']}]")
            return False
        
        # Check if within robot's maximum reach
        distance_xy = np.sqrt(x**2 + y**2)
        if distance_xy > self.robot_info['working_radius']:
            print(f"DEBUG: XY distance {distance_xy:.0f}mm exceeds working radius {self.robot_info['working_radius']}mm")
            return False
        
        print("DEBUG: All coordinate bounds validation passed")
        return True
    
    def _calculate_robot_pose(self, target_position: np.ndarray) -> Dict[str, Any]:
        """
        Calculate suggested robot pose to point at target position.
        
        Args:
            target_position: Target position [x, y, z] in mm
            
        Returns:
            Dictionary with suggested pose parameters
        """
        x, y, z = target_position
        
        # Calculate approach position (slightly back from target for pointing)
        approach_distance = 50  # mm back from target
        distance_xy = np.sqrt(x**2 + y**2)
        
        if distance_xy > 0:
            # Move back along the line from robot base to target
            approach_x = x - (approach_distance * x / distance_xy)
            approach_y = y - (approach_distance * y / distance_xy)
        else:
            approach_x = x
            approach_y = y
        
        # Height should be slightly above target
        approach_z = z + 30  # 3cm above target
        
        # Calculate pointing orientation (look at target from approach position)
        # For myCobot, pointing orientation with gripper facing down
        pointing_rx = 180  # Gripper facing down
        pointing_ry = 0    # No roll
        pointing_rz = np.degrees(np.arctan2(y, x))  # Rotate towards target
        
        return {
            'position': [approach_x, approach_y, approach_z],
            'orientation': [pointing_rx, pointing_ry, pointing_rz],
            'target_position': target_position.tolist(),
            'pointing_distance': approach_distance,
            'description': f"Point at target ({x:.0f}, {y:.0f}, {z:.0f})"
        }
    
    def _calculate_robot_pose_6dof(self, target_position: np.ndarray, target_orientation: np.ndarray) -> Dict[str, Any]:
        """
        Calculate suggested robot pose for 6DOF object interaction.
        
        Args:
            target_position: Target position [x, y, z] in mm
            target_orientation: Target orientation [rx, ry, rz] in radians
            
        Returns:
            Dictionary with complete 6DOF pose parameters
        """
        x, y, z = target_position
        rx, ry, rz = target_orientation
        
        # For floor-aligned objects, we want to approach from above
        # maintaining the floor-aligned orientation
        approach_distance = 50  # mm above target for picking
        
        # Approach position: directly above the object
        approach_x = x
        approach_y = y
        approach_z = z + approach_distance
        
        # For floor-aligned objects, the robot orientation should:
        # - Maintain gripper alignment with floor normal (object Z-axis)
        # - Use object's yaw rotation for proper grasping alignment
        
        # Convert target orientation to degrees
        rx_deg = np.degrees(rx)
        ry_deg = np.degrees(ry)
        rz_deg = np.degrees(rz)
        
        # Adjust for robot gripper orientation
        # Robot gripper faces down when rx = 180
        gripper_rx = 180 + rx_deg  # Align with object's tilt
        gripper_ry = ry_deg        # Match object's roll
        gripper_rz = rz_deg        # Match object's yaw
        
        return {
            'position': [approach_x, approach_y, approach_z],
            'orientation': [gripper_rx, gripper_ry, gripper_rz],
            'target_position': target_position.tolist(),
            'target_orientation': [rx_deg, ry_deg, rz_deg],
            'approach_distance': approach_distance,
            'approach_type': '6DOF_floor_aligned',
            'description': f"6DOF approach to object at ({x:.0f}, {y:.0f}, {z:.0f}) with orientation ({rx_deg:.1f}°, {ry_deg:.1f}°, {rz_deg:.1f}°)"
        }
    
    def _update_transformation_stats(self, transformation_time: float, is_valid: bool):
        """Update transformation performance statistics."""
        self.transformation_stats['total_transformations'] += 1
        
        if is_valid:
            self.transformation_stats['successful_transformations'] += 1
        else:
            self.transformation_stats['out_of_bounds_count'] += 1
        
        # Update running average for transformation time
        total = self.transformation_stats['total_transformations']
        prev_avg = self.transformation_stats['avg_transformation_time']
        self.transformation_stats['avg_transformation_time'] = (
            (prev_avg * (total - 1) + transformation_time) / total
        )
    
    def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about the current transformation setup."""
        info = {
            'matrix_loaded': self.hand_eye_matrix is not None,
            'matrix_source': self.matrix_source,
            'matrix_timestamp': self.matrix_timestamp,
            'coordinate_system': self.coordinate_system,
            'workspace_bounds': self.workspace_bounds,
            'robot_info': self.robot_info,
            'statistics': self.transformation_stats
        }
        
        if self.hand_eye_matrix is not None:
            info['matrix_shape'] = self.hand_eye_matrix.shape
            info['matrix_determinant'] = float(np.linalg.det(self.hand_eye_matrix[:3, :3]))
            
            # Add matrix preview (translation part)
            translation = self.hand_eye_matrix[:3, 3]
            info['translation_preview'] = {
                'x': float(translation[0]),
                'y': float(translation[1]), 
                'z': float(translation[2]),
                'units': 'meters'
            }
        
        return info
    
    def validate_transformation_accuracy(self, test_points: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """
        Validate transformation accuracy using test points.
        
        Args:
            test_points: List of test points in camera coordinates, or None for default tests
            
        Returns:
            Validation results dictionary
        """
        if self.hand_eye_matrix is None:
            return {'error': 'No transformation matrix loaded'}
        
        if test_points is None:
            # Use default test points (known positions in camera frame)
            test_points = [
                np.array([0.0, 0.0, 0.3]),    # Straight ahead at 30cm
                np.array([0.1, 0.0, 0.3]),    # 10cm right at 30cm
                np.array([-0.1, 0.0, 0.3]),   # 10cm left at 30cm
                np.array([0.0, 0.05, 0.25]),  # 5cm up at 25cm
                np.array([0.0, -0.05, 0.25])  # 5cm down at 25cm
            ]
        
        results = {
            'test_count': len(test_points),
            'valid_transformations': 0,
            'invalid_transformations': 0,
            'transformations': []
        }
        
        for i, camera_point in enumerate(test_points):
            result = self.transform_point_to_robot(camera_point)
            
            if result and result['is_valid']:
                results['valid_transformations'] += 1
            else:
                results['invalid_transformations'] += 1
            
            results['transformations'].append({
                'test_id': i,
                'camera_point': camera_point.tolist(),
                'result': result
            })
        
        # Calculate success rate
        results['success_rate'] = results['valid_transformations'] / results['test_count']
        
        return results
    
    def create_transformation_visualization(self, transformations: List[Dict[str, Any]],
                                         image_size: Tuple[int, int] = (640, 480)) -> np.ndarray:
        """
        Create visualization of coordinate transformations.
        
        Args:
            transformations: List of transformation results
            image_size: Output image size (width, height)
            
        Returns:
            BGR image showing transformation results
        """
        width, height = image_size
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if not transformations:
            cv2.putText(vis_image, "No transformations to display", (width//2 - 100, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return vis_image
        
        # Extract valid transformations
        valid_transforms = [t for t in transformations if t and t.get('is_valid', False)]
        
        if not valid_transforms:
            cv2.putText(vis_image, "No valid transformations", (width//2 - 80, height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis_image
        
        # Create top-down view of robot workspace
        robot_positions = [t['robot_position'] for t in valid_transforms]
        robot_x = [pos[0] for pos in robot_positions]  # X coordinates (mm)
        robot_y = [pos[1] for pos in robot_positions]  # Y coordinates (mm)
        
        # Calculate bounds for mapping
        bounds = self.workspace_bounds
        x_min, x_max = bounds['x_min'], bounds['x_max']
        y_min, y_max = bounds['y_min'], bounds['y_max']
        
        # Draw workspace boundary
        workspace_color = (64, 64, 64)
        cv2.rectangle(vis_image, (50, 50), (width-50, height-50), workspace_color, 2)
        
        # Draw robot positions
        for i, (x, y) in enumerate(zip(robot_x, robot_y)):
            # Map to image coordinates
            img_x = int(50 + ((x - x_min) / (x_max - x_min)) * (width - 100))
            img_y = int(height - 50 - ((y - y_min) / (y_max - y_min)) * (height - 100))
            
            # Clip to image bounds
            img_x = np.clip(img_x, 50, width - 50)
            img_y = np.clip(img_y, 50, height - 50)
            
            # Draw point
            cv2.circle(vis_image, (img_x, img_y), 6, (0, 255, 0), -1)
            cv2.circle(vis_image, (img_x, img_y), 8, (255, 255, 255), 2)
            
            # Label point
            cv2.putText(vis_image, f"{i+1}", (img_x + 10, img_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw robot base at center
        center_x = int(50 + 0.5 * (width - 100))
        center_y = int(height - 50 - 0.5 * (height - 100))
        cv2.circle(vis_image, (center_x, center_y), 10, (0, 0, 255), -1)  # Red for robot base
        cv2.putText(vis_image, "Robot", (center_x - 20, center_y - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add information text
        cv2.putText(vis_image, f"Transformations: {len(valid_transforms)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        stats = self.transformation_stats
        success_rate = (stats['successful_transformations'] / max(stats['total_transformations'], 1)) * 100
        cv2.putText(vis_image, f"Success Rate: {success_rate:.1f}%", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(vis_image, f"Matrix: {Path(self.matrix_source).name if self.matrix_source else 'None'}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image