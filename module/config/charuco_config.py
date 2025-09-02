"""
CharUco Configuration for Camera Calibration

This module provides configuration parameters specifically for CharUco-based
camera-robot calibration, including pattern specifications, detection parameters,
and calibration positions optimized for CharUco pattern visibility.
"""

from typing import Dict, Any, List, Tuple, Optional
import cv2
from .detection_config import DetectionConfig


class CharucoConfig(DetectionConfig):
    """
    Configuration class for CharUco-based camera calibration.
    
    Extends the base DetectionConfig with CharUco-specific parameters
    for pattern detection, camera calibration, and robot positioning.
    Uses the proven 8x11 CharUco pattern from test_charuco_detection.py.
    """
    
    def __init__(self, custom_params: Optional[Dict[str, Any]] = None):
        """
        Initialize CharUco configuration.
        
        Args:
            custom_params: Optional custom parameters to override defaults
        """
        # Initialize base configuration
        super().__init__(custom_params)
        
        # Set CharUco-specific defaults
        self._set_charuco_defaults()
        
        # Apply custom parameters if provided
        if custom_params:
            self.update_config(custom_params)
    
    def _set_charuco_defaults(self):
        """Set default parameters for CharUco detection and calibration."""
        
        # CharUco pattern specifications (proven from test_charuco_detection.py)
        self.config['pattern_params'] = {
            'pattern_size': (8, 11),  # (squares_x, squares_y) - proven configuration
            'square_size_mm': 25.0,   # Size of checkerboard squares in millimeters
            'marker_size_mm': 20.0,   # Size of ArUco markers in millimeters
            'aruco_dict_type': cv2.aruco.DICT_6X6_250,  # ArUco dictionary type
            'aruco_dict_name': 'DICT_6X6_250',  # Human-readable name
            'total_corners': (8-1) * (11-1),  # Total possible CharUco corners (70)
            'total_squares': 8 * 11,          # Total squares in pattern (88)
            'expected_markers': (8 * 11) // 2  # Approximate number of markers (44)
        }
        
        # CharUco detection parameters
        self.config['detection_params'] = {
            'min_corners_for_detection': 4,     # Minimum corners for valid detection
            'min_corners_for_pose': 8,          # Minimum corners for 3D pose estimation
            'detector_params': {
                # ArUco detector parameters - can be customized if needed
                'adaptiveThreshWinSizeMin': 3,
                'adaptiveThreshWinSizeMax': 23,
                'adaptiveThreshWinSizeStep': 10,
                'adaptiveThreshConstant': 7,
                'minMarkerPerimeterRate': 0.03,
                'maxMarkerPerimeterRate': 4.0,
                'polygonalApproxAccuracyRate': 0.03,
                'minCornerDistanceRate': 0.05,
                'minDistanceToBorder': 3,
                'minMarkerDistanceRate': 0.05,
                'cornerRefinementMethod': 'CORNER_REFINE_NONE'  # Can be changed to SUBPIX
            }
        }
        
        # Pattern quality and validation parameters
        self.config['quality_params'] = {
            'min_quality_threshold': 0.5,      # Minimum pattern quality score
            'min_pattern_coverage': 0.1,       # Minimum image coverage (10%)
            'max_pattern_coverage': 0.9,       # Maximum image coverage (90%)
            'min_edge_distance': 0.05,         # Minimum distance from image edges (5%)
            'coverage_weight': 0.4,            # Weight for corner coverage score
            'marker_weight': 0.3,              # Weight for marker detection score
            'geometry_weight': 0.2,            # Weight for geometry score
            'edge_distance_weight': 0.1,       # Weight for edge distance score
            'min_marker_coverage': 0.3         # Minimum ratio of detected markers
        }
        
        # Camera calibration parameters
        self.config['camera_calibration_params'] = {
            'min_calibration_images': 8,        # Minimum images for camera calibration
            'max_reprojection_error': 1.5,     # Maximum allowed reprojection error (pixels)
            'calibration_flags': 0,             # OpenCV calibration flags
            'fix_principal_point': False,
            'fix_aspect_ratio': False,
            'zero_tangent_dist': False,
            'fix_k1': False,
            'fix_k2': False,
            'fix_k3': False,
            'use_charuco_calibration': True     # Use CharUco-specific calibration
        }
        
        # Robot positioning for optimal pattern visibility
        self.config['robot_positioning'] = {
            'optimal_distance_mm': 350,         # Optimal distance from camera to pattern
            'min_distance_mm': 250,             # Minimum safe distance
            'max_distance_mm': 600,             # Maximum effective distance
            'approach_angles': [0, 20, 40, 60], # Angles for multi-view calibration
            'height_offsets': [0, 30, 60],      # Height variations for robustness
            'lighting_positions': ['center', 'left', 'right'],  # Lighting considerations
            'pattern_tilt_angles': [0, 15, 30]  # Pattern tilt for 3D pose diversity
        }
        
        # Specific calibration positions for different modes
        self.config['calibration_modes'] = {
            'fixed_camera': {
                'description': 'Fixed camera calibration with robot-held pattern',
                'num_positions': 15,
                'position_strategy': 'grid_coverage',
                'include_tilted_views': True,
                'min_baseline_ratio': 0.3  # Minimum baseline relative to distance
            },
            'eye_in_hand': {
                'description': 'Eye-in-hand calibration with robot-mounted camera',
                'num_positions': 20,  # Use all recorded positions for better accuracy
                'position_strategy': 'hemisphere_coverage',
                'include_tilted_views': True,
                'fixed_pattern_position': [400, 0, 0]  # Pattern position on table
            },
            'hybrid': {
                'description': 'Combined fixed and eye-in-hand calibration',
                'num_positions': 20,
                'position_strategy': 'comprehensive',
                'include_both_modes': True
            }
        }
        
        # Visualization parameters
        self.config['visualization_params'] = {
            'show_corners': True,
            'show_corner_ids': True,            # CharUco corners have IDs
            'show_markers': True,               # Show detected ArUco markers
            'show_marker_ids': True,            # Show ArUco marker IDs
            'show_coordinate_system': True,
            'show_quality_score': True,
            'show_distance': True,
            'show_coverage': True,
            'corner_color': (0, 255, 0),        # Green for detected corners
            'marker_color': (255, 0, 0),        # Red for ArUco markers
            'center_color': (0, 0, 255),        # Blue for pattern center
            'text_color': (255, 255, 0),        # Yellow for text
            'axes_color': [(0, 0, 255), (0, 255, 0), (255, 0, 0)],  # RGB axes
            'line_thickness': 2,
            'text_scale': 0.6,
            'axes_length_mm': 50                # Length of coordinate axes
        }
        
        # Performance and optimization parameters
        self.config['performance_params'] = {
            'max_detection_time_ms': 150,       # Maximum time per detection (CharUco is slower)
            'enable_gpu_acceleration': False,   # GPU acceleration for detection
            'cache_board_objects': True,        # Cache CharUco board object
            'adaptive_quality_threshold': True, # Adapt quality threshold based on success rate
            'early_termination': True,          # Stop after successful detection
            'parallel_marker_detection': False  # Keep false for stability
        }
        
        # Error handling and robustness
        self.config['robustness_params'] = {
            'max_detection_attempts': 3,        # Maximum attempts per image
            'fallback_detection': False,        # No fallback for CharUco (unlike checkerboard)
            'noise_reduction': True,            # Apply noise reduction preprocessing
            'contrast_enhancement': True,       # Enhance contrast for detection
            'rotation_invariant': True,         # Handle rotated patterns
            'partial_pattern_detection': True,  # Allow detection of partially visible patterns
            'min_markers_for_corners': 4        # Minimum markers needed for corner detection
        }
    
    def get_charuco_board(self):
        """
        Create and return CharUco board object.
        
        Returns:
            OpenCV CharUco board object
        """
        pattern_params = self.config['pattern_params']
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(pattern_params['aruco_dict_type'])
        charuco_board = cv2.aruco.CharucoBoard(
            pattern_params['pattern_size'],
            pattern_params['square_size_mm'],
            pattern_params['marker_size_mm'],
            aruco_dict
        )
        
        return charuco_board
    
    def get_aruco_detector(self):
        """
        Create and return ArUco detector with configured parameters.
        
        Returns:
            OpenCV ArUco detector object
        """
        pattern_params = self.config['pattern_params']
        detection_params = self.config['detection_params']
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(pattern_params['aruco_dict_type'])
        
        # Create detector parameters
        detector_params = cv2.aruco.DetectorParameters()
        
        # Apply custom detector parameters if specified
        custom_params = detection_params.get('detector_params', {})
        for param_name, param_value in custom_params.items():
            if hasattr(detector_params, param_name):
                setattr(detector_params, param_name, param_value)
        
        return cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    
    def get_charuco_detector(self):
        """
        Create and return CharUco detector.
        
        Returns:
            OpenCV CharUco detector object
        """
        charuco_board = self.get_charuco_board()
        return cv2.aruco.CharucoDetector(charuco_board)
    
    def get_optimal_calibration_positions(self, mode: str = 'fixed_camera') -> List[List[float]]:
        """
        Generate optimal calibration positions for the specified mode.
        
        Args:
            mode: Calibration mode ('fixed_camera', 'eye_in_hand', 'hybrid')
            
        Returns:
            List of robot positions [x, y, z, roll, pitch, yaw]
        """
        if mode not in self.config['calibration_modes']:
            raise ValueError(f"Unknown calibration mode: {mode}")
        
        mode_config = self.config['calibration_modes'][mode]
        num_positions = mode_config['num_positions']
        strategy = mode_config['position_strategy']
        
        if strategy == 'grid_coverage':
            return self._generate_grid_positions(num_positions)
        elif strategy == 'hemisphere_coverage':
            return self._generate_hemisphere_positions(num_positions)
        elif strategy == 'comprehensive':
            return self._generate_comprehensive_positions(num_positions)
        else:
            # Default to hemisphere coverage (better for CharUco)
            return self._generate_hemisphere_positions(num_positions)
    
    def _generate_grid_positions(self, num_positions: int) -> List[List[float]]:
        """Generate calibration positions in a grid pattern."""
        positions = []
        
        # Base parameters
        base_x = 300  # mm (slightly farther for CharUco)
        base_z = 80   # mm
        
        # Grid parameters
        x_range = 120  # ±60mm around base_x
        y_range = 240  # ±120mm around center
        
        # Calculate grid dimensions
        grid_size = int(num_positions ** 0.5)
        if grid_size * grid_size < num_positions:
            grid_size += 1
        
        x_step = x_range / max(1, grid_size - 1) if grid_size > 1 else 0
        y_step = y_range / max(1, grid_size - 1) if grid_size > 1 else 0
        
        for i in range(grid_size):
            for j in range(grid_size):
                if len(positions) >= num_positions:
                    break
                
                x = base_x - x_range/2 + i * x_step
                y = -y_range/2 + j * y_step
                z = base_z
                
                # Add some height variation
                if i % 2 == 0:
                    z += 15
                
                # Standard orientation
                roll, pitch, yaw = 0, 180, 0
                
                positions.append([x, y, z, roll, pitch, yaw])
        
        return positions[:num_positions]
    
    def _generate_hemisphere_positions(self, num_positions: int) -> List[List[float]]:
        """Load pre-recorded calibration positions from recorded_coords_280.json."""
        import json
        import os
        
        # Path to recorded positions file
        positions_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'recorded_coords_280.json')
        
        try:
            with open(positions_file, 'r') as f:
                recorded_positions = json.load(f)
            
            print(f"DEBUG: Loaded {len(recorded_positions)} pre-recorded positions from {positions_file}")
            
            # Use recorded positions, limiting to requested number
            positions = recorded_positions[:num_positions]
            
            # If we need more positions than recorded, pad with recorded positions (cycling)
            while len(positions) < num_positions:
                remaining = num_positions - len(positions)
                positions.extend(recorded_positions[:remaining])
            
            print(f"DEBUG: Using {len(positions)} positions for CharUco calibration")
            print(f"DEBUG: First position: {positions[0]}")
            print(f"DEBUG: Last position: {positions[-1]}")
            
            return positions
            
        except FileNotFoundError:
            print(f"WARNING: Could not find recorded positions file: {positions_file}")
            print("Falling back to hemisphere generation...")
            return self._generate_hemisphere_positions_fallback(num_positions)
        except json.JSONDecodeError as e:
            print(f"WARNING: Could not parse recorded positions file: {e}")
            print("Falling back to hemisphere generation...")
            return self._generate_hemisphere_positions_fallback(num_positions)
    
    def _generate_hemisphere_positions_fallback(self, num_positions: int) -> List[List[float]]:
        """Fallback hemisphere generation if recorded positions are not available."""
        positions = []
        
        # Fixed pattern position (center of workspace) - CharUco can be farther
        pattern_pos = self.config['calibration_modes']['eye_in_hand']['fixed_pattern_position']
        print(f"DEBUG: Using pattern position for hemisphere generation: {pattern_pos}")
        
        # Robot workspace parameters (myCobot 280 specs from CLAUDE.md)
        max_robot_reach = 280  # mm - effective radius from robot base (J1)
        min_height = 120   # mm above table (higher for CharUco)
        max_height = 220   # mm above table
        
        import math
        
        # Generate positions around robot base that can view the pattern
        for i in range(num_positions):
            theta = 2 * math.pi * i / num_positions  # Azimuth angle
            phi = math.pi/6 + (math.pi/3) * (i % 3) / 3  # Elevation angle (30-60 degrees)
            
            # Calculate robot position within workspace (distance from robot base J1)
            robot_radius = max_robot_reach * 0.75  # Use 75% of max reach for safety
            
            # Position robot in hemisphere around base
            x = robot_radius * math.cos(phi) * math.cos(theta)
            y = robot_radius * math.cos(phi) * math.sin(theta)
            z = min_height + (max_height - min_height) * math.sin(phi)
            
            # Calculate orientation to look at pattern from this robot position
            robot_pos = [x, y, z]
            target_vector = [pattern_pos[0] - x, pattern_pos[1] - y, pattern_pos[2] - z]
            target_distance = math.sqrt(sum(v**2 for v in target_vector))
            
            if target_distance > 0:
                # Calculate angles to point toward pattern using myCobot convention
                yaw = math.degrees(math.atan2(target_vector[1], target_vector[0]))
                horizontal_dist = math.sqrt(target_vector[0]**2 + target_vector[1]**2)
                
                # Calculate pitch for myCobot: 180° = straight down, 90° = horizontal
                pitch_angle = math.degrees(math.atan2(-target_vector[2], horizontal_dist))
                pitch = 90 + pitch_angle
                
                # Ensure reasonable downward angle but not too steep
                pitch = max(pitch, 110)  # Minimum 110° for downward viewing
                pitch = min(pitch, 150)  # Maximum 150° to avoid too steep angle
                
                # Normalize angles to -180 to 180 range for myCobot
                while pitch > 180:
                    pitch -= 360
                while pitch < -180:
                    pitch += 360
                while yaw > 180:
                    yaw -= 360
                while yaw < -180:
                    yaw += 360
                    
                roll = 0
            else:
                # Fallback orientation
                roll, pitch, yaw = 0, 140, 0
            
            positions.append([x, y, z, roll, pitch, yaw])
        
        return positions
    
    def _generate_comprehensive_positions(self, num_positions: int) -> List[List[float]]:
        """Generate comprehensive positions combining grid and hemisphere strategies."""
        # Split positions between strategies
        grid_positions = self._generate_grid_positions(num_positions // 2)
        hemisphere_positions = self._generate_hemisphere_positions(num_positions - len(grid_positions))
        
        return grid_positions + hemisphere_positions
    
    def get_test_positions(self, mode: str = 'fixed_camera') -> List[List[float]]:
        """
        Generate test positions for validation.
        
        Args:
            mode: Calibration mode
            
        Returns:
            List of test positions for validation
        """
        # Generate fewer test positions than calibration positions
        calibration_positions = self.get_optimal_calibration_positions(mode)
        
        # Select every 3rd position for testing
        test_positions = calibration_positions[::3]
        
        # Ensure we have at least 3 test positions
        if len(test_positions) < 3:
            test_positions = calibration_positions[:3]
        
        return test_positions
    
    def validate_config(self) -> bool:
        """
        Validate CharUco configuration parameters.
        
        Returns:
            True if configuration is valid
        """
        # Call parent validation first
        if not super().validate_config():
            return False
        
        # Validate pattern parameters
        pattern_params = self.config.get('pattern_params', {})
        
        # Validate pattern size
        pattern_size = pattern_params.get('pattern_size', (0, 0))
        if not isinstance(pattern_size, (list, tuple)) or len(pattern_size) != 2:
            self.validation_errors.append(f"Invalid pattern size format: {pattern_size}")
            return False
        
        if pattern_size[0] < 4 or pattern_size[1] < 4:
            self.validation_errors.append(f"CharUco pattern size too small: {pattern_size}")
            return False
        
        # Validate square and marker sizes
        square_size = pattern_params.get('square_size_mm', 0)
        marker_size = pattern_params.get('marker_size_mm', 0)
        
        if square_size <= 0:
            self.validation_errors.append("Square size must be positive")
            return False
        
        if marker_size <= 0:
            self.validation_errors.append("Marker size must be positive")
            return False
        
        if marker_size >= square_size:
            self.validation_errors.append("Marker size must be smaller than square size")
            return False
        
        # Validate ArUco dictionary
        aruco_dict_type = pattern_params.get('aruco_dict_type')
        valid_dicts = [
            cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_4X4_100, cv2.aruco.DICT_4X4_250, cv2.aruco.DICT_4X4_1000,
            cv2.aruco.DICT_5X5_50, cv2.aruco.DICT_5X5_100, cv2.aruco.DICT_5X5_250, cv2.aruco.DICT_5X5_1000,
            cv2.aruco.DICT_6X6_50, cv2.aruco.DICT_6X6_100, cv2.aruco.DICT_6X6_250, cv2.aruco.DICT_6X6_1000,
            cv2.aruco.DICT_7X7_50, cv2.aruco.DICT_7X7_100, cv2.aruco.DICT_7X7_250, cv2.aruco.DICT_7X7_1000
        ]
        
        if aruco_dict_type not in valid_dicts:
            self.validation_errors.append(f"Invalid ArUco dictionary type: {aruco_dict_type}")
            return False
        
        # Validate quality parameters
        quality_params = self.config.get('quality_params', {})
        min_quality = quality_params.get('min_quality_threshold', 0)
        if not (0 <= min_quality <= 1):
            self.validation_errors.append("Quality threshold must be between 0 and 1")
            return False
        
        # Validate coverage parameters
        min_coverage = quality_params.get('min_pattern_coverage', 0)
        max_coverage = quality_params.get('max_pattern_coverage', 1)
        
        if not (0 <= min_coverage <= max_coverage <= 1):
            self.validation_errors.append("Invalid pattern coverage range")
            return False
        
        return True
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the CharUco configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        summary = super().get_config_summary()
        
        # Add CharUco-specific information
        pattern_params = self.config.get('pattern_params', {})
        summary.update({
            'config_type': 'CharUco Configuration',
            'pattern_size': pattern_params.get('pattern_size', (0, 0)),
            'square_size_mm': pattern_params.get('square_size_mm', 0),
            'marker_size_mm': pattern_params.get('marker_size_mm', 0),
            'aruco_dict_name': pattern_params.get('aruco_dict_name', 'Unknown'),
            'total_corners': pattern_params.get('total_corners', 0),
            'expected_markers': pattern_params.get('expected_markers', 0),
            'min_quality_threshold': self.config.get('quality_params', {}).get('min_quality_threshold', 0),
            'camera_calibration_enabled': 'camera_calibration_params' in self.config,
            'supported_modes': list(self.config.get('calibration_modes', {}).keys())
        })
        
        return summary