#!/usr/bin/env python3
"""
Camera-to-Robot Transformer

A universal, plug-and-play module for transforming any 3D position from camera
coordinates to robot base coordinates using pre-calibrated eye-in-hand matrices.

Features:
- Universal point transformation (works with any camera-detected position)
- Load transformation matrices from .npy calibration files
- Automatic unit conversion (camera meters ↔ robot mm)
- Validation functions for accuracy testing
- Batch processing support
- Clean, type-agnostic API

Usage:
    transformer = CameraToRobotTransformer('eye_in_hand_transform.npy')
    robot_position = transformer.transform_to_robot(camera_point, robot_pose)
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Union, List, Tuple, Optional
import json


class CameraToRobotTransformer:
    """
    Universal transformer for converting camera coordinates to robot base coordinates.
    
    This class provides a clean, type-agnostic interface for transforming any 3D position
    detected in camera coordinates to robot base coordinates using pre-calibrated
    eye-in-hand transformation matrices.
    
    Coordinate Systems:
    - Camera: x=right, y=down, z=forward (meters)
    - Robot: x=forward, y=left, z=up (millimeters)
    """
    
    def __init__(self, transform_file: Union[str, Path] = None):
        """
        Initialize the camera-to-robot transformer.
        
        Args:
            transform_file: Path to .npy file or directory containing eye-in-hand matrix
        """
        self.eye_in_hand_transform = None
        self.calibration_info = {}
        
        if transform_file:
            self.load_transformation(transform_file)
    
    def load_transformation(self, transform_file: Union[str, Path]) -> bool:
        """
        Load pre-calibrated eye-in-hand transformation matrix.
        
        Args:
            transform_file: Path to .npy file or calibration directory
            
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        transform_path = Path(transform_file)
        
        try:
            if transform_path.is_file() and transform_path.suffix == '.npy':
                # Direct .npy file
                self.eye_in_hand_transform = np.load(transform_path)
                print(f"[OK] Loaded transformation matrix from: {transform_path}")
                
            elif transform_path.is_dir():
                # Calibration directory - look for standard file
                npy_file = transform_path / 'eye_in_hand_transform.npy'
                if npy_file.exists():
                    self.eye_in_hand_transform = np.load(npy_file)
                    print(f"[OK] Loaded transformation matrix from: {npy_file}")
                    
                    # Try to load calibration metadata
                    self._load_calibration_metadata(transform_path)
                else:
                    raise FileNotFoundError(f"No eye_in_hand_transform.npy found in {transform_path}")
            else:
                raise ValueError(f"Invalid transform file: {transform_path}")
            
            # Validate matrix dimensions
            if self.eye_in_hand_transform.shape != (4, 4):
                raise ValueError(f"Invalid transformation matrix shape: {self.eye_in_hand_transform.shape}")
            
            self._print_transformation_info()
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load transformation: {e}")
            self.eye_in_hand_transform = None
            return False
    
    def _load_calibration_metadata(self, calibration_dir: Path):
        """Load calibration metadata if available."""
        metadata_files = ['calibration_data.json', 'README.txt']
        
        for filename in metadata_files:
            metadata_file = calibration_dir / filename
            if metadata_file.exists():
                try:
                    if filename.endswith('.json'):
                        with open(metadata_file, 'r') as f:
                            self.calibration_info = json.load(f)
                        print(f"[INFO] Loaded calibration metadata from {filename}")
                    break
                except Exception as e:
                    print(f"[WARN] Could not load {filename}: {e}")
    
    def _print_transformation_info(self):
        """Print transformation matrix information."""
        print("Eye-in-Hand Transformation Matrix:")
        for i, row in enumerate(self.eye_in_hand_transform):
            print(f"  [{row[0]:8.5f} {row[1]:8.5f} {row[2]:8.5f} {row[3]:8.5f}]")
        
        # Extract translation and rotation info
        translation = self.eye_in_hand_transform[:3, 3]
        rotation_matrix = self.eye_in_hand_transform[:3, :3]
        
        print(f"Translation (camera to gripper): [{translation[0]*1000:.1f}, {translation[1]*1000:.1f}, {translation[2]*1000:.1f}] mm")
        
        if self.calibration_info:
            num_points = self.calibration_info.get('num_points', 'Unknown')
            print(f"Calibration points used: {num_points}")
    
    def is_ready(self) -> bool:
        """Check if transformer is ready to use."""
        return self.eye_in_hand_transform is not None
    
    def pose_to_transform_matrix(self, robot_pose: List[float]) -> np.ndarray:
        """
        Convert robot pose [x, y, z, rx, ry, rz] to 4x4 transformation matrix.
        
        Args:
            robot_pose: Robot pose as [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            
        Returns:
            4x4 transformation matrix (robot base to gripper)
        """
        x, y, z, rx, ry, rz = robot_pose
        
        # Convert to meters and radians
        x_m, y_m, z_m = x/1000.0, y/1000.0, z/1000.0
        rx_rad, ry_rad, rz_rad = np.radians([rx, ry, rz])
        
        # Create rotation matrix from Euler angles (XYZ intrinsic order)
        rotation = Rotation.from_euler('xyz', [rx_rad, ry_rad, rz_rad])
        rot_matrix = rotation.as_matrix()
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = [x_m, y_m, z_m]
        
        return transform
    
    def transform_to_robot(self, 
                          camera_position: Union[List[float], np.ndarray],
                          robot_pose: List[float],
                          input_units: str = 'meters',
                          output_units: str = 'mm') -> np.ndarray:
        """
        Transform any 3D position from camera coordinates to robot base coordinates.
        
        This is the core universal transformation function that works with any type
        of camera-detected position (CharUco corners, object centers, manual points, etc.).
        
        Args:
            camera_position: 3D position in camera coordinates [x, y, z]
                           Camera coords: x=right, y=down, z=forward
            robot_pose: Current robot pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            input_units: Units of camera_position ('meters' or 'mm')
            output_units: Units of output ('meters' or 'mm')
            
        Returns:
            3D position in robot base coordinates [x, y, z]
            Robot coords: x=forward, y=left, z=up
            
        Raises:
            RuntimeError: If transformation matrix not loaded
            ValueError: If invalid input dimensions
        """
        if not self.is_ready():
            raise RuntimeError("Transformation matrix not loaded. Call load_transformation() first.")
        
        # Convert input to numpy array and validate
        position = np.array(camera_position, dtype=float)
        if position.shape != (3,):
            raise ValueError(f"Camera position must be [x, y, z], got shape {position.shape}")
        
        # Convert input units to meters (required for transformation)
        if input_units == 'mm':
            position = position / 1000.0
        elif input_units != 'meters':
            raise ValueError(f"Invalid input_units: {input_units}. Use 'meters' or 'mm'")
        
        # Create homogeneous coordinates for the position
        position_homogeneous = np.array([position[0], position[1], position[2], 1.0])
        
        # Get robot transformation matrix (robot base to gripper)
        robot_transform = self.pose_to_transform_matrix(robot_pose)
        
        # Apply transformation chain (same as calibration validation logic):
        # Position in robot base = Robot_base_to_gripper @ Eye_in_hand @ Camera_to_position
        position_in_robot = robot_transform @ self.eye_in_hand_transform @ position_homogeneous
        
        # Extract 3D position and convert to desired output units
        result_position = position_in_robot[:3]
        if output_units == 'mm':
            result_position = result_position * 1000.0
        elif output_units != 'meters':
            raise ValueError(f"Invalid output_units: {output_units}. Use 'meters' or 'mm'")
        
        return result_position
    
    def transform_multiple_positions(self,
                                   camera_positions: List[Union[List[float], np.ndarray]],
                                   robot_pose: List[float],
                                   input_units: str = 'meters',
                                   output_units: str = 'mm') -> List[np.ndarray]:
        """
        Transform multiple positions from camera to robot coordinates.
        
        Args:
            camera_positions: List of 3D positions in camera coordinates
            robot_pose: Current robot pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
            input_units: Units of camera positions ('meters' or 'mm')
            output_units: Units of output ('meters' or 'mm')
            
        Returns:
            List of 3D positions in robot base coordinates
        """
        results = []
        for i, position in enumerate(camera_positions):
            try:
                result = self.transform_to_robot(position, robot_pose, input_units, output_units)
                results.append(result)
            except Exception as e:
                print(f"Warning: Failed to transform position {i}: {e}")
                results.append(None)
        return results
    
    def validate_transformation(self, 
                              known_camera_position: List[float],
                              known_robot_position: List[float],
                              robot_pose: List[float],
                              tolerance_mm: float = 25.0) -> dict:
        """
        Validate transformation accuracy using known position pairs.
        
        Args:
            known_camera_position: Known position in camera coordinates [x, y, z] meters
            known_robot_position: Known position in robot coordinates [x, y, z] mm
            robot_pose: Robot pose when camera observed the known position
            tolerance_mm: Acceptable error tolerance in mm
            
        Returns:
            Dictionary with validation results
        """
        if not self.is_ready():
            return {'error': 'Transformation matrix not loaded'}
        
        try:
            # Transform camera position to robot coordinates
            predicted_robot = self.transform_to_robot(known_camera_position, robot_pose, 
                                                    input_units='meters', output_units='mm')
            
            # Calculate error
            known_robot = np.array(known_robot_position)
            error_vector = predicted_robot - known_robot
            error_magnitude = np.linalg.norm(error_vector)
            
            # Assess accuracy grade
            if error_magnitude < 5.0:
                accuracy_grade = 'Excellent'
            elif error_magnitude < 10.0:
                accuracy_grade = 'Very Good'
            elif error_magnitude < 20.0:
                accuracy_grade = 'Good'
            elif error_magnitude < 50.0:
                accuracy_grade = 'Fair'
            else:
                accuracy_grade = 'Poor'
            
            return {
                'is_valid': error_magnitude <= tolerance_mm,
                'error_mm': error_magnitude,
                'error_vector_mm': error_vector.tolist(),
                'predicted_robot_mm': predicted_robot.tolist(),
                'known_robot_mm': known_robot.tolist(),
                'tolerance_mm': tolerance_mm,
                'accuracy_grade': accuracy_grade
            }
            
        except Exception as e:
            return {'error': f'Validation failed: {e}'}
    
    def get_transformation_info(self) -> dict:
        """
        Get information about the loaded transformation.
        
        Returns:
            Dictionary with transformation information
        """
        if not self.is_ready():
            return {'status': 'not_loaded', 'message': 'No transformation matrix loaded'}
        
        info = {
            'status': 'loaded',
            'matrix_shape': self.eye_in_hand_transform.shape,
            'translation_mm': (self.eye_in_hand_transform[:3, 3] * 1000).tolist(),
            'has_metadata': bool(self.calibration_info)
        }
        
        # Add calibration metadata if available
        if self.calibration_info:
            info['calibration_points'] = self.calibration_info.get('num_points', 'Unknown')
            info['calibration_timestamp'] = self.calibration_info.get('timestamp', 'Unknown')
        
        return info
    
    def print_usage_examples(self):
        """Print usage examples for different types of camera detections."""
        print("\n" + "="*60)
        print("CAMERA-TO-ROBOT TRANSFORMER - USAGE EXAMPLES")
        print("="*60)
        print("This transformer works with ANY 3D position detected in camera coordinates:")
        print()
        print("# 1. CharUco corner detection")
        print("charuco_corner = [0.1, 0.05, 0.3]  # meters in camera coords")
        print("robot_pos = transformer.transform_to_robot(charuco_corner, robot_pose)")
        print()
        print("# 2. Lidar object center detection")
        print("object_center = [0.08, 0.02, 0.25]  # meters in camera coords")
        print("robot_pos = transformer.transform_to_robot(object_center, robot_pose)")
        print()
        print("# 3. Manual point measurement")
        print("manual_point = [0.12, 0.01, 0.35]  # meters in camera coords")
        print("robot_pos = transformer.transform_to_robot(manual_point, robot_pose)")
        print()
        print("# 4. Batch processing")
        print("multiple_points = [charuco_corner, object_center, manual_point]")
        print("results = transformer.transform_multiple_positions(multiple_points, robot_pose)")
        print()
        print("Coordinate Systems:")
        print("  Camera: x=right, y=down, z=forward (meters)")
        print("  Robot:  x=forward, y=left, z=up (millimeters)")
        print("="*60)


# Convenience functions for quick usage
def create_transformer(transform_file: Union[str, Path]) -> CameraToRobotTransformer:
    """
    Create and load a camera-to-robot transformer.
    
    Args:
        transform_file: Path to transformation matrix file or directory
        
    Returns:
        Ready-to-use CameraToRobotTransformer instance
        
    Raises:
        RuntimeError: If transformation cannot be loaded
    """
    transformer = CameraToRobotTransformer()
    if not transformer.load_transformation(transform_file):
        raise RuntimeError(f"Failed to load transformation from {transform_file}")
    return transformer


def quick_transform(camera_position: List[float],
                   robot_pose: List[float],
                   transform_file: Union[str, Path],
                   input_units: str = 'meters',
                   output_units: str = 'mm') -> np.ndarray:
    """
    Quick one-line transformation without creating persistent transformer.
    
    Args:
        camera_position: 3D position in camera coordinates [x, y, z]
        robot_pose: Current robot pose [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        transform_file: Path to transformation matrix file
        input_units: Units of camera_position ('meters' or 'mm')
        output_units: Units of output ('meters' or 'mm')
        
    Returns:
        3D position in robot base coordinates [x, y, z]
    """
    transformer = create_transformer(transform_file)
    return transformer.transform_to_robot(camera_position, robot_pose, input_units, output_units)


if __name__ == "__main__":
    # Example usage and demonstration
    print("Camera-to-Robot Transformer - Universal Position Transformation")
    print("=" * 70)
    
    # Example transformation file path (adjust for your system)
    example_transform = "charuco-pose-estimation/calibration_result_1754622347/eye_in_hand_transform.npy"
    
    try:
        # Create transformer
        transformer = CameraToRobotTransformer(example_transform)
        
        if transformer.is_ready():
            print("\n[OK] Transformer ready! Testing with example positions...")
            
            # Example robot pose
            robot_pose = [200, 0, 300, 0, 180, 0]  # mm and degrees
            print(f"Robot pose: {robot_pose} [x, y, z, rx, ry, rz]")
            
            # Test different types of camera positions (all in meters)
            test_positions = {
                "CharUco corner": [0.1, 0.05, 0.3],
                "Object center": [0.08, 0.02, 0.25], 
                "Manual point": [0.12, 0.01, 0.35]
            }
            
            print(f"\nTransformation Results:")
            print(f"{'Detection Type':<15} {'Camera (m)':<20} {'Robot (mm)':<25}")
            print("-" * 60)
            
            for detection_type, camera_pos in test_positions.items():
                try:
                    robot_pos = transformer.transform_to_robot(camera_pos, robot_pose)
                    camera_str = f"[{camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f}]"
                    robot_str = f"[{robot_pos[0]:.1f}, {robot_pos[1]:.1f}, {robot_pos[2]:.1f}]"
                    print(f"{detection_type:<15} {camera_str:<20} {robot_str:<25}")
                except Exception as e:
                    print(f"{detection_type:<15} ERROR: {e}")
            
            # Show transformation info
            info = transformer.get_transformation_info()
            print(f"\nTransformation Info:")
            print(f"  Status: {info['status']}")
            print(f"  Matrix shape: {info['matrix_shape']}")
            print(f"  Translation: {info['translation_mm']} mm")
            
            # Print usage examples
            transformer.print_usage_examples()
            
        else:
            print("[ERROR] Transformer not ready. Check transformation file path.")
        
    except Exception as e:
        print(f"[ERROR] Example failed: {e}")
        print("\nTo test with your calibration:")
        print("1. Update 'example_transform' path to your .npy file")
        print("2. Run: python camera_to_robot_transformer.py")
        print("\nOr use in your code:")
        print("transformer = CameraToRobotTransformer('your_transform.npy')")
        print("result = transformer.transform_to_robot(camera_point, robot_pose)")