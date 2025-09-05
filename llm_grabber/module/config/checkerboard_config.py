"""
Checkerboard Configuration for Camera Calibration

This module provides configuration parameters specifically for checkerboard-based
camera-robot calibration, including pattern specifications, detection parameters,
and calibration positions optimized for pattern visibility.
"""

from typing import Dict, Any, List, Tuple, Optional
from .detection_config import DetectionConfig


class CheckerboardConfig(DetectionConfig):
    """
    Configuration class for checkerboard-based camera calibration.
    
    Extends the base DetectionConfig with checkerboard-specific parameters
    for pattern detection, camera calibration, and robot positioning.
    """
    
    def __init__(self, custom_params: Optional[Dict[str, Any]] = None):
        """
        Initialize checkerboard configuration.
        
        Args:
            custom_params: Optional custom parameters to override defaults
        """
        # Initialize base configuration
        super().__init__(custom_params)
        
        # Set checkerboard-specific defaults
        self._set_checkerboard_defaults()
        
        # Apply custom parameters if provided
        if custom_params:
            self.update_config(custom_params)
    
    def _set_checkerboard_defaults(self):
        """Set default parameters for checkerboard detection and calibration."""
        
        # Checkerboard pattern specifications
        self.config['pattern_params'] = {
            'pattern_sizes': [(9, 6), (8, 6), (7, 5)],  # (width, height) in internal corners
            'square_size_mm': 25.0,  # Size of each square in millimeters
            'default_pattern': (9, 6),  # Preferred pattern size
            'pattern_orientation': 'landscape'  # 'landscape' or 'portrait'
        }
        
        # OpenCV detection parameters
        self.config['detection_params'] = {
            'detection_flags': None,  # Will use default flags in strategy
            'subpix_window_size': (11, 11),  # Window size for cornerSubPix
            'subpix_zero_zone': (-1, -1),  # Zero zone for cornerSubPix
            'subpix_criteria': {
                'type': 'EPS_MAX_ITER',  # cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
                'max_iter': 30,
                'epsilon': 0.1
            },
            'adaptive_thresh': True,
            'normalize_image': True,
            'fast_check': True
        }
        
        # Pattern quality and validation parameters
        self.config['quality_params'] = {
            'min_quality_threshold': 0.5,  # Minimum pattern quality score
            'min_pattern_coverage': 0.1,   # Minimum image coverage (10%)
            'max_pattern_coverage': 0.8,   # Maximum image coverage (80%)
            'min_edge_distance': 0.05,     # Minimum distance from image edges (5%)
            'geometry_weight': 0.4,        # Weight for geometry score
            'sharpness_weight': 0.3,       # Weight for sharpness score
            'uniformity_weight': 0.2,      # Weight for uniformity score
            'edge_distance_weight': 0.1    # Weight for edge distance score
        }
        
        # Camera calibration parameters
        self.config['camera_calibration_params'] = {
            'min_calibration_images': 10,   # Minimum images for camera calibration
            'max_reprojection_error': 1.0,  # Maximum allowed reprojection error (pixels)
            'calibration_flags': 0,         # OpenCV calibration flags
            'fix_principal_point': False,
            'fix_aspect_ratio': False,
            'zero_tangent_dist': False,
            'fix_k1': False,
            'fix_k2': False,
            'fix_k3': False
        }
        
        # Robot positioning for optimal pattern visibility
        self.config['robot_positioning'] = {
            'optimal_distance_mm': 300,     # Optimal distance from camera to pattern
            'min_distance_mm': 200,         # Minimum safe distance
            'max_distance_mm': 500,         # Maximum effective distance
            'approach_angles': [0, 15, 30, 45],  # Angles for multi-view calibration
            'height_offsets': [0, 20, 40],  # Height variations for robustness
            'lighting_positions': ['center', 'left', 'right'],  # Lighting considerations
            'pattern_tilt_angles': [0, 10, 20]  # Pattern tilt for 3D pose diversity
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
                'num_positions': 12,
                'position_strategy': 'hemisphere_coverage',
                'include_tilted_views': True,
                'fixed_pattern_position': [200, 0, 0]  # Pattern position on table
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
            'show_corner_ids': False,
            'show_coordinate_system': True,
            'show_quality_score': True,
            'show_distance': True,
            'corner_color': (0, 255, 0),      # Green for detected corners
            'center_color': (255, 0, 0),      # Blue for pattern center
            'text_color': (255, 255, 0),      # Yellow for text
            'line_thickness': 2,
            'text_scale': 0.6
        }
        
        # Performance and optimization parameters
        self.config['performance_params'] = {
            'max_detection_time_ms': 100,     # Maximum time per detection
            'enable_gpu_acceleration': False, # GPU acceleration for detection
            'parallel_pattern_detection': True,  # Detect multiple patterns in parallel
            'cache_object_points': True,      # Cache 3D object points for patterns
            'adaptive_quality_threshold': True,  # Adapt quality threshold based on success rate
            'early_termination': True         # Stop after first successful detection
        }
        
        # Error handling and robustness
        self.config['robustness_params'] = {
            'max_detection_attempts': 3,      # Maximum attempts per image
            'fallback_pattern_sizes': True,   # Try alternative pattern sizes
            'noise_reduction': True,          # Apply noise reduction preprocessing
            'contrast_enhancement': True,     # Enhance contrast for detection
            'rotation_invariant': True,       # Handle rotated patterns
            'partial_pattern_detection': False # Allow detection of partially visible patterns
        }
    
    def get_pattern_object_points(self, pattern_size: Tuple[int, int]) -> List[List[float]]:
        """
        Generate 3D object points for a specific pattern size.
        
        Args:
            pattern_size: (width, height) of pattern in internal corners
            
        Returns:
            List of 3D object points in millimeters
        """
        width, height = pattern_size
        square_size = self.config['pattern_params']['square_size_mm']
        
        object_points = []
        for j in range(height):
            for i in range(width):
                object_points.append([i * square_size, j * square_size, 0.0])
        
        return object_points
    
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
            # Default to grid coverage
            return self._generate_grid_positions(num_positions)
    
    def _generate_grid_positions(self, num_positions: int) -> List[List[float]]:
        """Generate calibration positions in a grid pattern."""
        positions = []
        
        # Base parameters
        base_x = 250  # mm
        base_z = 60   # mm
        
        # Grid parameters
        x_range = 100  # ±50mm around base_x
        y_range = 200  # ±100mm around center
        
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
                    z += 10
                
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
            
            print(f"DEBUG: Using {len(positions)} positions for calibration")
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
        
        # Fixed pattern position (center of workspace)
        pattern_pos = self.config['calibration_modes']['eye_in_hand']['fixed_pattern_position']
        print(f"DEBUG: Using pattern position for hemisphere generation: {pattern_pos}")
        
        # Robot workspace parameters (myCobot 280 specs from CLAUDE.md)
        max_robot_reach = 280  # mm - effective radius from robot base (J1)
        min_height = 100   # mm above table
        max_height = 200   # mm above table
        
        import math
        
        # Use user-specified first position, then generate remaining positions
        for i in range(num_positions):
            if i == 0:
                # Use the specific coordinate provided by user as first position
                x, y, z = 86.2, -62.8, 364.5
                roll, pitch, yaw = -146.85, 30.12, -126.52
                print(f"DEBUG: Using user-specified first position: [{x}, {y}, {z}, {roll}, {pitch}, {yaw}]")
                positions.append([x, y, z, roll, pitch, yaw])
                continue
            
            # Generate remaining positions around robot base that can view the pattern
            pos_index = i - 1
            theta = 2 * math.pi * pos_index / (num_positions - 1)  # Azimuth angle
            phi = math.pi/6 + (math.pi/3) * (pos_index % 3) / 3  # Elevation angle (30-60 degrees)
            
            # Calculate robot position within workspace (distance from robot base J1)
            robot_radius = max_robot_reach * 0.8  # Use 80% of max reach for safety
            
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
                pitch = max(pitch, 120)  # Minimum 120° for downward viewing
                pitch = min(pitch, 140)  # Maximum 140° to avoid too steep angle
                
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
                roll, pitch, yaw = 0, 150, 0
            
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
        Validate checkerboard configuration parameters.
        
        Returns:
            True if configuration is valid
        """
        # Call parent validation first
        if not super().validate_config():
            return False
        
        # Validate pattern parameters
        pattern_params = self.config.get('pattern_params', {})
        pattern_sizes = pattern_params.get('pattern_sizes', [])
        
        if not pattern_sizes:
            self.validation_errors.append("No pattern sizes specified")
            return False
        
        for pattern_size in pattern_sizes:
            if not isinstance(pattern_size, (list, tuple)) or len(pattern_size) != 2:
                self.validation_errors.append(f"Invalid pattern size format: {pattern_size}")
                return False
            
            if pattern_size[0] < 3 or pattern_size[1] < 3:
                self.validation_errors.append(f"Pattern size too small: {pattern_size}")
                return False
        
        # Validate square size
        square_size = pattern_params.get('square_size_mm', 0)
        if square_size <= 0:
            self.validation_errors.append("Square size must be positive")
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
        Get a summary of the checkerboard configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        summary = super().get_config_summary()
        
        # Add checkerboard-specific information
        pattern_params = self.config.get('pattern_params', {})
        summary.update({
            'config_type': 'Checkerboard Configuration',
            'pattern_sizes': pattern_params.get('pattern_sizes', []),
            'square_size_mm': pattern_params.get('square_size_mm', 0),
            'default_pattern': pattern_params.get('default_pattern', (0, 0)),
            'min_quality_threshold': self.config.get('quality_params', {}).get('min_quality_threshold', 0),
            'camera_calibration_enabled': 'camera_calibration_params' in self.config,
            'supported_modes': list(self.config.get('calibration_modes', {}).keys())
        })
        
        return summary