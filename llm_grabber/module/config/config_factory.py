"""
Factory class for creating different configuration types.
"""

from typing import Dict, Any, Optional, Union
from .detection_config import DetectionConfig
from .bev_config import BEVConfig
from .pointnet_config import PointNetConfig
from .checkerboard_config import CheckerboardConfig
from .charuco_config import CharucoConfig


class ConfigFactory:
    """
    Factory class for creating appropriate configuration objects.
    
    Provides convenient methods for:
    - Creating predefined configurations
    - Loading configurations from files
    - Creating hybrid configurations
    - Managing configuration presets
    """
    
    # Registry of available configuration types
    CONFIG_TYPES = {
        'base': DetectionConfig,
        'detection': DetectionConfig,
        'bev': BEVConfig,
        'pointnet': PointNetConfig,
        'checkerboard': CheckerboardConfig,
        'charuco': CharucoConfig,
        'lidar': DetectionConfig  # LiDAR uses base config with custom parameters
    }
    
    # Predefined configuration presets
    PRESETS = {
        'realtime_bev': {
            'type': 'bev',
            'config': {
                'clustering_params': {
                    'min_cluster_size': 20,
                    'max_cluster_size': 100,
                    'morphology_kernel_size': 3,
                    'window_size': 3
                },
                'bev_params': {
                    'image_size': 300
                },
                'temporal_params': {
                    'enabled': True,
                    'persistence_threshold': 0.6
                },
                'performance_params': {
                    'enable_gpu_acceleration': True,
                    'downsample_for_bev': True
                }
            }
        },
        
        'offline_pointnet': {
            'type': 'pointnet',
            'config': {
                'model_params': {
                    'model_dir': 'pointnet_cls',
                    'dataset': 'simple'
                },
                'subspace_params': {
                    'ndim': 50,
                    'use_gds': True
                },
                'pointnet_preprocessing': {
                    'num_points': 1024,
                    'normalize_to_unit_sphere': True,
                    'min_points_threshold': 10
                },
                'performance_params': {
                    'max_points_for_processing': 50000,
                    'enable_voxel_downsampling': True,
                    'voxel_size_initial': 0.003
                }
            }
        },
        
        'high_quality_bev': {
            'type': 'bev',
            'config': {
                'clustering_params': {
                    'min_cluster_size': 15,
                    'max_cluster_size': 200,
                    'morphology_kernel_size': 5,
                    'window_size': 7
                },
                'bev_params': {
                    'image_size': 600
                },
                'temporal_params': {
                    'enabled': True,
                    'persistence_threshold': 0.8,
                    'min_frames_for_filtering': 5
                },
                'performance_params': {
                    'downsample_for_bev': False
                }
            }
        },
        
        'fast_pointnet': {
            'type': 'pointnet',
            'config': {
                'model_params': {
                    'model_dir': 'pointnet_cls'
                },
                'subspace_params': {
                    'ndim': 30,
                    'use_gds': False
                },
                'pointnet_preprocessing': {
                    'num_points': 512,
                    'min_points_threshold': 5
                },
                'performance_params': {
                    'max_points_for_processing': 25000,
                    'enable_voxel_downsampling': True
                },
                'noise_params': {
                    'nb_neighbors': 20
                }
            }
        },
        
        'debug_mode': {
            'type': 'base',
            'config': {
                'debug_mode': True,
                'verbose_logging': True,
                'enable_timing': True,
                'save_results': True
            }
        },
        
        # Object tracking specific presets
        'realtime_tracking': {
            'type': 'bev',
            'config': {
                'clustering_params': {
                    'min_cluster_size': 20,
                    'max_cluster_size': 100,
                    'morphology_kernel_size': 3,
                    'window_size': 3
                },
                'tracking_params': {
                    'update_interval': 1.0,
                    'fast_update_interval': 0.5,
                    'adaptive_timing': True,
                    'max_no_object_time': 5.0
                },
                'robot_params': {
                    'min_height': 100,
                    'max_speed': 90,
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.5
                },
                'performance_params': {
                    'enable_gpu_acceleration': True,
                    'downsample_for_bev': True
                },
                'roi_params': {
                    'x_min': -1.5, 'x_max': 1.5,
                    'y_min': 0.32, 'y_max': 0.38,  # Focus on table height
                    'z_min': 0.1, 'z_max': 0.5     # End-effector view range
                },
                'display_roi_params': {
                    'x_min': -2.0, 'x_max': 2.0,
                    'y_min': -0.5, 'y_max': 0.5,
                    'z_min': 0, 'z_max': 1.0
                },
            }
        },
        
        'precision_tracking': {
            'type': 'bev',
            'config': {
                'clustering_params': {
                    'min_cluster_size': 15,
                    'max_cluster_size': 200,
                    'morphology_kernel_size': 5,
                    'window_size': 7
                },
                'tracking_params': {
                    'update_interval': 2.0,
                    'fast_update_interval': 1.0,
                    'adaptive_timing': True,
                    'max_no_object_time': 10.0
                },
                'robot_params': {
                    'min_height': 120,
                    'max_speed': 50,
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 1.0
                },
                'temporal_params': {
                    'enabled': True,
                    'persistence_threshold': 0.8,
                    'min_frames_for_filtering': 5
                },
                'performance_params': {
                    'downsample_for_bev': False
                }
            }
        },
        
        'adaptive_tracking': {
            'type': 'bev',
            'config': {
                'clustering_params': {
                    'min_cluster_size': 25,
                    'max_cluster_size': 150,
                    'morphology_kernel_size': 4,
                    'window_size': 5
                },
                'tracking_params': {
                    'update_interval': 1.5,
                    'fast_update_interval': 0.3,
                    'adaptive_timing': True,
                    'max_no_object_time': 8.0,
                    'strategy_switching': True
                },
                'robot_params': {
                    'min_height': 110,
                    'max_speed': 70,
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.3
                },
                'performance_params': {
                    'enable_gpu_acceleration': True,
                    'adaptive_quality': True
                }
            }
        },
        
        # Camera-Robot Calibration specific presets
        'calibration_standard': {
            'type': 'bev',
            'config': {
                'clustering_params': {
                    'min_cluster_size': 15,  # Smaller for calibration objects
                    'max_cluster_size': 200,  # Allow larger objects
                    'morphology_kernel_size': 3,  # Finer detail
                    'window_size': 3  # Less temporal filtering for responsiveness
                },
                'roi_params': {
                    'x_min': -1.5, 'x_max': 1.5,
                    'y_min': 0.05, 'y_max': 0.20,  # Focus on table height
                    'z_min': 0.1, 'z_max': 0.8   # Calibration range
                },
                'display_roi_params': {
                    'x_min': -2.0, 'x_max': 2.0,
                    'y_min': -0.5, 'y_max': 0.5,
                    'z_min': 0, 'z_max': 1.0
                },
                'calibration_positions': [
                    [250, 0, 60, 0, 180, 0],     # Center reference
                    [180, -100, 60, 0, 180, 0],  # Left side
                    [180, 100, 60, 0, 180, 0],   # Right side
                    [200, 0, 60, 0, 180, 0],     # Higher position
                    [220, -50, 60, 0, 180, 0],   # Mixed position
                ],
                'test_positions': [
                    [200, -60, 60, 0, 180, 0],   # Test 1: Left-forward
                    [160, 80, 60, 0, 180, 0],    # Test 2: Right-forward
                    [240, 30, 60, 0, 180, 0]     # Test 3: Back-right
                ],
                'robot_params': {
                    'min_height': 50,  # Lower for calibration
                    'max_speed': 40,   # Slower for precision
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.2,
                    'home_position': [0, 0, 0, 0, 0, 0]
                },
                'accuracy_thresholds': {
                    'max_calibration_error': 100.0,  # mm
                    'max_validation_error': 50.0,    # mm
                    'min_calibration_points': 3,
                    'recommended_points': 5
                }
            }
        },
        
        'calibration_precise': {
            'type': 'bev',
            'config': {
                'clustering_params': {
                    'min_cluster_size': 10,  # Very sensitive
                    'max_cluster_size': 300,
                    'morphology_kernel_size': 2,  # Finest detail
                    'window_size': 2  # Minimal temporal filtering
                },
                'roi_params': {
                    'x_min': -1.0, 'x_max': 1.0,
                    'y_min': 0.08, 'y_max': 0.15,  # Tighter height range
                    'z_min': 0.15, 'z_max': 0.6   # Focused calibration range
                },
                'display_roi_params': {
                    'x_min': -1.5, 'x_max': 1.5,
                    'y_min': -0.3, 'y_max': 0.3,
                    'z_min': 0, 'z_max': 0.8
                },
                'calibration_positions': [
                    [220, 0, 70, 0, 180, 0],      # Center reference (higher)
                    [200, -80, 65, 0, 180, 0],   # Left side
                    [200, 80, 65, 0, 180, 0],    # Right side
                    [240, -40, 70, 0, 180, 0],   # Back-left
                    [240, 40, 70, 0, 180, 0],    # Back-right
                    [180, 0, 75, 0, 180, 0],     # Front center (higher)
                    [190, -60, 68, 0, 180, 0],   # Additional precision point
                ],
                'test_positions': [
                    [210, -30, 72, 0, 180, 0],   # Test 1
                    [230, 50, 67, 0, 180, 0],    # Test 2
                    [195, 70, 74, 0, 180, 0]     # Test 3
                ],
                'robot_params': {
                    'min_height': 60,
                    'max_speed': 30,   # Very slow for maximum precision
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.5,
                    'home_position': [0, 0, 0, 0, 0, 0]
                },
                'accuracy_thresholds': {
                    'max_calibration_error': 50.0,   # Stricter
                    'max_validation_error': 25.0,    # Very strict
                    'min_calibration_points': 5,
                    'recommended_points': 7
                }
            }
        },
        
        'calibration_quick': {
            'type': 'bev',
            'config': {
                'clustering_params': {
                    'min_cluster_size': 25,  # Less sensitive for speed
                    'max_cluster_size': 150,
                    'morphology_kernel_size': 5,
                    'window_size': 5
                },
                'roi_params': {
                    'x_min': -1.2, 'x_max': 1.2,
                    'y_min': 0.06, 'y_max': 0.18,
                    'z_min': 0.1, 'z_max': 0.7
                },
                'display_roi_params': {
                    'x_min': -1.8, 'x_max': 1.8,
                    'y_min': -0.4, 'y_max': 0.4,
                    'z_min': 0, 'z_max': 1.0
                },
                'calibration_positions': [
                    [250, 0, 60, 0, 180, 0],      # Center
                    [200, -120, 60, 0, 180, 0],  # Left
                    [200, 120, 60, 0, 180, 0],   # Right
                ],
                'test_positions': [
                    [220, -80, 60, 0, 180, 0],   # Single test point
                ],
                'robot_params': {
                    'min_height': 50,
                    'max_speed': 60,   # Faster for quick calibration
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.1,
                    'home_position': [0, 0, 0, 0, 0, 0]
                },
                'accuracy_thresholds': {
                    'max_calibration_error': 150.0,  # More lenient
                    'max_validation_error': 80.0,
                    'min_calibration_points': 3,
                    'recommended_points': 3
                }
            }
        },
        
        'virtual_fixed_camera': {
            'type': 'bev',
            'config': {
                'clustering_params': {
                    'min_cluster_size': 15,  # Balanced for end-effector camera
                    'max_cluster_size': 200,
                    'morphology_kernel_size': 3,
                    'window_size': 3
                },
                'roi_params': {
                    'x_min': -1.5, 'x_max': 1.5,
                    'y_min': 0.20, 'y_max': 0.26,  # Focus on table height
                    'z_min': 0.1, 'z_max': 0.5     # End-effector view range
                },
                'display_roi_params': {
                    'x_min': -2.0, 'x_max': 2.0,
                    'y_min': -0.5, 'y_max': 0.5,
                    'z_min': 0, 'z_max': 1.0
                },
                'calibration_positions': [
                    [150, 0, 110, 0, 180, 45],      # Center reference
                    [200, 0, 110, 0, 180, 45],  # Left side
                    [250, 0, 110, 0, 180, 45],   # Right side
                    [150, -50, 110, 0, 180, 45],      # Center reference
                    [200, -50, 110, 0, 180, 45],  # Left side
                    [250, -50, 110, 0, 180, 45],   # Right side
                    [150, 50, 110, 0, 180, 45],      # Center reference
                    [200, 50, 110, 0, 180, 45],  # Left side
                    [250, 50, 110, 0, 180, 45],   # Right side
                    [250, 0, 110, 0, 180, 45],      # Center reference
                    [180, -100, 110, 0, 180, 45],  # Left side
                    [180, 100, 110, 0, 180, 45],   # Right side
                    [200, 0, 110, 0, 180, 45],     # Forward position
                    [220, -50, 110, 0, 180, 45],   # Mixed position
                ],
                'test_positions': [
                    [200, -60, 100, 0, 180, 45],   # Test 1: Left-forward
                    [160, 80, 100, 0, 180, 45],    # Test 2: Right-forward
                    [240, 30, 100, 0, 180, 45]     # Test 3: Back-right
                ],
                'robot_params': {
                    'min_height': 50,
                    'max_speed': 40,   # Balanced speed
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.2,
                    'home_method': 'look_at_table'  # Use look_at_table instead of home_position
                },
                'accuracy_thresholds': {
                    'max_calibration_error': 100.0,  # mm
                    'max_validation_error': 50.0,    # mm
                    'min_calibration_points': 3,
                    'recommended_points': 5
                }
            }
        },
        
        # Checkerboard-based calibration presets
        'checkerboard_standard': {
            'type': 'checkerboard',
            'config': {
                'pattern_params': {
                    'pattern_sizes': [(9, 6)],
                    'square_size_mm': 25.0,
                    'default_pattern': (9, 6),
                    'pattern_orientation': 'landscape'
                },
                'detection_params': {
                    'adaptive_thresh': True,
                    'normalize_image': True,
                    'fast_check': True,
                    'subpix_window_size': (11, 11),
                    'subpix_criteria': {
                        'max_iter': 30,
                        'epsilon': 0.1
                    }
                },
                'quality_params': {
                    'min_quality_threshold': 0.5,
                    'min_pattern_coverage': 0.1,
                    'max_pattern_coverage': 0.8,
                    'min_edge_distance': 0.05
                },
                'camera_calibration_params': {
                    'min_calibration_images': 10,
                    'max_reprojection_error': 1.0
                },
                'calibration_modes': {
                    'fixed_camera': {
                        'num_positions': 12,
                        'position_strategy': 'grid_coverage'
                    },
                    'eye_in_hand': {
                        'num_positions': 10,
                        'position_strategy': 'hemisphere_coverage',
                        'fixed_pattern_position': [200, 0, 0]
                    }
                },
                'robot_params': {
                    'min_height': 60,
                    'max_speed': 40,
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.2
                },
                'visualization_params': {
                    'show_corners': True,
                    'show_coordinate_system': True,
                    'show_quality_score': True,
                    'corner_color': (0, 255, 0),
                    'center_color': (255, 0, 0),
                    'text_color': (255, 255, 0)
                }
            }
        },
        
        'checkerboard_precise': {
            'type': 'checkerboard',
            'config': {
                'pattern_params': {
                    'pattern_sizes': [(9, 6), (11, 8)],
                    'square_size_mm': 20.0,
                    'default_pattern': (9, 6),
                    'pattern_orientation': 'landscape'
                },
                'detection_params': {
                    'adaptive_thresh': True,
                    'normalize_image': True,
                    'fast_check': True,
                    'subpix_window_size': (11, 11),
                    'subpix_criteria': {
                        'max_iter': 50,
                        'epsilon': 0.05
                    }
                },
                'quality_params': {
                    'min_quality_threshold': 0.7,
                    'min_pattern_coverage': 0.15,
                    'max_pattern_coverage': 0.75,
                    'min_edge_distance': 0.08
                },
                'camera_calibration_params': {
                    'min_calibration_images': 15,
                    'max_reprojection_error': 0.5
                },
                'calibration_modes': {
                    'fixed_camera': {
                        'num_positions': 18,
                        'position_strategy': 'comprehensive'
                    },
                    'eye_in_hand': {
                        'num_positions': 15,
                        'position_strategy': 'hemisphere_coverage',
                        'fixed_pattern_position': [200, 0, 0]
                    }
                },
                'robot_params': {
                    'min_height': 80,
                    'max_speed': 25,
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.5
                },
                'performance_params': {
                    'max_detection_time_ms': 200,
                    'adaptive_quality_threshold': True
                }
            }
        },
        
        'checkerboard_quick': {
            'type': 'checkerboard',
            'config': {
                'pattern_params': {
                    'pattern_sizes': [(8, 6), (7, 5)],
                    'square_size_mm': 30.0,
                    'default_pattern': (8, 6),
                    'pattern_orientation': 'landscape'
                },
                'detection_params': {
                    'adaptive_thresh': True,
                    'normalize_image': True,
                    'fast_check': True,
                    'subpix_window_size': (9, 9),
                    'subpix_criteria': {
                        'max_iter': 20,
                        'epsilon': 0.2
                    }
                },
                'quality_params': {
                    'min_quality_threshold': 0.3,
                    'min_pattern_coverage': 0.08,
                    'max_pattern_coverage': 0.85,
                    'min_edge_distance': 0.03
                },
                'camera_calibration_params': {
                    'min_calibration_images': 8,
                    'max_reprojection_error': 1.5
                },
                'calibration_modes': {
                    'fixed_camera': {
                        'num_positions': 8,
                        'position_strategy': 'grid_coverage'
                    },
                    'eye_in_hand': {
                        'num_positions': 6,
                        'position_strategy': 'grid_coverage',
                        'fixed_pattern_position': [250, 0, 0]
                    }
                },
                'robot_params': {
                    'min_height': 50,
                    'max_speed': 60,
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.1
                },
                'performance_params': {
                    'max_detection_time_ms': 50,
                    'early_termination': True
                }
            }
        },
        
        'checkerboard_eye_in_hand': {
            'type': 'checkerboard',
            'config': {
                'pattern_params': {
                    'pattern_sizes': [(9, 6)],
                    'square_size_mm': 25.0,
                    'default_pattern': (9, 6),
                    'pattern_orientation': 'landscape'
                },
                'detection_params': {
                    'adaptive_thresh': True,
                    'normalize_image': True,
                    'fast_check': True
                },
                'quality_params': {
                    'min_quality_threshold': 0.6,
                    'min_pattern_coverage': 0.12,
                    'max_pattern_coverage': 0.7
                },
                'camera_calibration_params': {
                    'min_calibration_images': 12,
                    'max_reprojection_error': 0.8
                },
                'calibration_modes': {
                    'eye_in_hand': {
                        'num_positions': 12,
                        'position_strategy': 'hemisphere_coverage',
                        'fixed_pattern_position': [200, 0, 0]
                    }
                },
                'robot_params': {
                    'min_height': 70,
                    'max_speed': 35,
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.3
                },
                'robot_positioning': {
                    'optimal_distance_mm': 300,
                    'min_distance_mm': 200,
                    'max_distance_mm': 450,
                    'approach_angles': [0, 15, 30, 45]
                }
            }
        },
        
        # CharUco calibration presets
        'charuco_standard': {
            'type': 'charuco',
            'config': {
                'pattern_params': {
                    'pattern_size': (8, 11),  # Proven configuration from test_charuco_detection
                    'square_size_mm': 25.0,
                    'marker_size_mm': 20.0,
                    'aruco_dict_type': 'DICT_6X6_250'
                },
                'detection_params': {
                    'min_corners_for_detection': 6,
                    'min_corners_for_pose': 8,
                    'detector_params': {
                        'adaptiveThreshConstant': 7,
                        'minMarkerPerimeterRate': 0.03,
                        'maxMarkerPerimeterRate': 4.0
                    }
                },
                'quality_params': {
                    'min_quality_threshold': 0.5,
                    'min_pattern_coverage': 0.15,
                    'max_pattern_coverage': 0.8,
                    'min_marker_coverage': 0.3
                },
                'camera_calibration_params': {
                    'min_calibration_images': 8,
                    'max_reprojection_error': 1.2,
                    'use_charuco_calibration': True
                },
                'calibration_modes': {
                    'eye_in_hand': {
                        'num_positions': 12,
                        'position_strategy': 'hemisphere_coverage',
                        'fixed_pattern_position': [400, 0, 0]
                    }
                },
                'robot_params': {
                    'min_height': 80,
                    'max_speed': 35,
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.3
                },
                'robot_positioning': {
                    'optimal_distance_mm': 350,
                    'min_distance_mm': 250,
                    'max_distance_mm': 600,
                    'approach_angles': [0, 20, 40, 60]
                }
            }
        },
        
        'charuco_precise': {
            'type': 'charuco',
            'config': {
                'pattern_params': {
                    'pattern_size': (8, 11),
                    'square_size_mm': 25.0,
                    'marker_size_mm': 20.0,
                    'aruco_dict_type': 'DICT_6X6_250'
                },
                'detection_params': {
                    'min_corners_for_detection': 8,
                    'min_corners_for_pose': 12,
                    'detector_params': {
                        'adaptiveThreshConstant': 5,
                        'minMarkerPerimeterRate': 0.02,
                        'maxMarkerPerimeterRate': 5.0,
                        'cornerRefinementMethod': 'CORNER_REFINE_SUBPIX'
                    }
                },
                'quality_params': {
                    'min_quality_threshold': 0.7,
                    'min_pattern_coverage': 0.2,
                    'max_pattern_coverage': 0.85,
                    'min_marker_coverage': 0.4
                },
                'camera_calibration_params': {
                    'min_calibration_images': 12,
                    'max_reprojection_error': 0.8,
                    'use_charuco_calibration': True
                },
                'calibration_modes': {
                    'eye_in_hand': {
                        'num_positions': 15,
                        'position_strategy': 'hemisphere_coverage',
                        'fixed_pattern_position': [400, 0, 0]
                    }
                },
                'robot_params': {
                    'min_height': 90,
                    'max_speed': 25,
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.5
                },
                'robot_positioning': {
                    'optimal_distance_mm': 380,
                    'min_distance_mm': 280,
                    'max_distance_mm': 650,
                    'approach_angles': [0, 15, 30, 45, 60]
                }
            }
        },
        
        'charuco_quick': {
            'type': 'charuco',
            'config': {
                'pattern_params': {
                    'pattern_size': (8, 11),
                    'square_size_mm': 25.0,
                    'marker_size_mm': 20.0,
                    'aruco_dict_type': 'DICT_6X6_250'
                },
                'detection_params': {
                    'min_corners_for_detection': 4,
                    'min_corners_for_pose': 6,
                    'detector_params': {
                        'adaptiveThreshConstant': 10,
                        'minMarkerPerimeterRate': 0.05
                    }
                },
                'quality_params': {
                    'min_quality_threshold': 0.3,
                    'min_pattern_coverage': 0.1,
                    'max_pattern_coverage': 0.9,
                    'min_marker_coverage': 0.2
                },
                'camera_calibration_params': {
                    'min_calibration_images': 6,
                    'max_reprojection_error': 1.5,
                    'use_charuco_calibration': True
                },
                'calibration_modes': {
                    'eye_in_hand': {
                        'num_positions': 8,
                        'position_strategy': 'hemisphere_coverage',
                        'fixed_pattern_position': [400, 0, 0]
                    }
                },
                'robot_params': {
                    'min_height': 70,
                    'max_speed': 45,
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.2
                },
                'robot_positioning': {
                    'optimal_distance_mm': 320,
                    'min_distance_mm': 220,
                    'max_distance_mm': 500,
                    'approach_angles': [0, 30, 60]
                }
            }
        },
        
        'charuco_eye_in_hand': {
            'type': 'charuco',
            'config': {
                'pattern_params': {
                    'pattern_size': (8, 11),
                    'square_size_mm': 25.0,
                    'marker_size_mm': 20.0,
                    'aruco_dict_type': 'DICT_6X6_250'
                },
                'detection_params': {
                    'min_corners_for_detection': 6,
                    'min_corners_for_pose': 8
                },
                'quality_params': {
                    'min_quality_threshold': 0.6,
                    'min_pattern_coverage': 0.12,
                    'max_pattern_coverage': 0.7,
                    'min_marker_coverage': 0.3
                },
                'camera_calibration_params': {
                    'min_calibration_images': 10,
                    'max_reprojection_error': 1.0,
                    'use_charuco_calibration': True
                },
                'calibration_modes': {
                    'eye_in_hand': {
                        'num_positions': 12,
                        'position_strategy': 'hemisphere_coverage',
                        'fixed_pattern_position': [400, 0, 0]
                    }
                },
                'robot_params': {
                    'min_height': 80,
                    'max_speed': 35,
                    'pointing_angles': [0, 180, 0],
                    'min_update_interval': 0.3
                },
                'robot_positioning': {
                    'optimal_distance_mm': 350,
                    'min_distance_mm': 250,
                    'max_distance_mm': 600,
                    'approach_angles': [0, 20, 40, 60]
                }
            }
        },
        
        # LiDAR-based tracking configurations
        'lidar_tracking': {
            'type': 'lidar',
            'config': {
                'distance_range': (0.25, 0.8),  # LiDAR effective range in meters
                'floor_params': {
                    'distance_threshold': 0.02,      # 2cm tolerance for floor plane
                    'max_trials': 1000,              # RANSAC max iterations
                    'min_floor_points': 500,         # Minimum points for valid floor
                    'floor_normal_tolerance': 0.3    # Allow slight floor tilt
                },
                'clustering_params': {
                    'eps': 0.03,                     # DBSCAN epsilon (3cm)
                    'min_samples': 10,               # DBSCAN minimum samples
                    'min_cluster_size': 50,          # Minimum points per cluster
                    'max_cluster_size': 2000,        # Maximum points per cluster
                    'outlier_removal': True          # Enable statistical outlier removal
                },
                'selection_params': {
                    'selection_method': 'nearest_center',  # Object selection method
                    'max_selection_distance': 1.0,         # Max distance for selection (meters)
                    'height_preference': 0.1                # Prefer objects at certain height
                },
                'robot_params': {
                    'update_interval': 1.0,          # Seconds between robot commands
                    'min_confidence': 0.7,           # Minimum confidence for robot commands
                    'safety_bounds': {               # Robot workspace bounds (mm)
                        'x_min': -300, 'x_max': 300,
                        'y_min': -300, 'y_max': 300,
                        'z_min': 50,   'z_max': 350
                    }
                },
                'visualization_params': {
                    'show_lidar_view': True,
                    'show_floor_view': True,
                    'show_cluster_view': True,
                    'show_coordinate_view': True,
                    'show_statistics': True
                }
            }
        },
        
        'lidar_precision': {
            'type': 'lidar',
            'config': {
                'distance_range': (0.3, 0.6),    # More focused range for precision
                'floor_params': {
                    'distance_threshold': 0.015,  # Tighter floor tolerance
                    'max_trials': 1500,           # More RANSAC iterations
                    'min_floor_points': 800,      # Higher floor point requirement
                    'floor_normal_tolerance': 0.2 # Stricter floor orientation
                },
                'clustering_params': {
                    'eps': 0.025,                 # Tighter clustering
                    'min_samples': 15,            # Higher minimum samples
                    'min_cluster_size': 75,       # Larger minimum clusters
                    'max_cluster_size': 1500,     # Smaller maximum clusters
                    'outlier_removal': True
                },
                'selection_params': {
                    'selection_method': 'nearest_center',
                    'max_selection_distance': 0.8,  # Closer objects only
                    'height_preference': 0.1
                },
                'robot_params': {
                    'update_interval': 2.0,       # Slower, more deliberate updates
                    'min_confidence': 0.8,        # Higher confidence threshold
                    'safety_bounds': {
                        'x_min': -250, 'x_max': 250,
                        'y_min': -250, 'y_max': 250,
                        'z_min': 75,   'z_max': 300
                    }
                },
                'visualization_params': {
                    'show_lidar_view': True,
                    'show_floor_view': True,
                    'show_cluster_view': True,
                    'show_coordinate_view': True,
                    'show_statistics': True
                }
            }
        },
        
        'lidar_debug': {
            'type': 'lidar',
            'config': {
                'distance_range': (0.2, 1.0),    # Wide range for debugging
                'floor_params': {
                    'distance_threshold': 0.03,   # Looser floor tolerance
                    'max_trials': 800,            # Fewer iterations for speed
                    'min_floor_points': 300,      # Lower floor requirement
                    'floor_normal_tolerance': 0.5 # Allow tilted floors
                },
                'clustering_params': {
                    'eps': 0.04,                  # Looser clustering
                    'min_samples': 8,             # Lower sample requirement
                    'min_cluster_size': 30,       # Smaller minimum clusters
                    'max_cluster_size': 3000,     # Allow very large clusters
                    'outlier_removal': False      # Disable for debugging raw data
                },
                'selection_params': {
                    'selection_method': 'largest', # Different selection for debugging
                    'max_selection_distance': 1.5, # Accept distant objects
                    'height_preference': 0.0       # No height preference
                },
                'robot_params': {
                    'update_interval': 3.0,       # Very slow for debugging observation
                    'min_confidence': 0.5,        # Lower threshold to see more detections
                    'safety_bounds': {            # Wider bounds for testing
                        'x_min': -400, 'x_max': 400,
                        'y_min': -400, 'y_max': 400,
                        'z_min': 0,    'z_max': 400
                    }
                },
                'visualization_params': {
                    'show_lidar_view': True,
                    'show_floor_view': True,
                    'show_cluster_view': True,
                    'show_coordinate_view': True,
                    'show_statistics': True
                }
            }
        },
        
        # Enhanced 5-Step LiDAR Pipeline Configurations
        'enhanced_lidar_precise': {
            'type': 'lidar',
            'config': {
                'strategy_type': 'enhanced',  # Use EnhancedLiDARStrategy
                'spatial_filter_params': {
                    'z_range': (0.2, 0.6),  # Depth filtering in meters
                    'x_range': (-0.2, 0.2), # Lateral filtering in meters
                    'y_range': None,        # No Y filtering
                    'min_points_after_filter': 100
                },
                'gravity_params': {
                    'gravity_vector': [0, 0, -1],  # Target gravity in world frame (Z-up)
                    'alignment_samples': 10,
                    'alignment_tolerance': 0.1,
                    'use_cached_rotation': True
                },
                'floor_params': {
                    'distance_threshold': 0.015,    # 1.5cm RANSAC tolerance
                    'max_trials': 1500,
                    'min_floor_points': 300,
                    'floor_normal_tolerance': 0.2,  # Stricter horizontal tolerance
                    'min_floor_area': 0.05
                },
                'object_height_params': {
                    'min_height_above_floor': 0.01,  # 1cm minimum above floor
                    'max_height_above_floor': 0.20,  # 20cm maximum above floor
                    'height_tolerance': 0.005,
                    'min_object_points': 30
                },
                'clustering_params': {
                    'eps': 0.025,                    # 2.5cm clustering
                    'min_samples': 8,
                    'min_cluster_size': 50,
                    'max_cluster_size': 1500,
                    'outlier_removal': True
                },
                'visualization_params': {
                    'enable_all_steps': True,
                    'show_6panel_view': True,
                    'point_cloud_colors': {
                        'raw': (128, 128, 128),
                        'filtered': (0, 255, 0),
                        'rotated': (255, 255, 0),
                        'floor': (255, 0, 0),
                        'objects': (0, 0, 255),
                        'projected': (255, 0, 255)
                    }
                }
            }
        },
        
        'enhanced_lidar_demo': {
            'type': 'lidar',
            'config': {
                'strategy_type': 'enhanced',  # Use EnhancedLiDARStrategy
                'spatial_filter_params': {
                    'z_range': (0.2, 0.6),  # Depth filtering in meters
                    'x_range': (-0.2, 0.2), # Lateral filtering in meters  
                    'y_range': None,        # No Y filtering
                    'min_points_after_filter': 80
                },
                'gravity_params': {
                    'gravity_vector': [0, 0, -1],  # Target gravity in world frame (Z-up)
                    'alignment_samples': 5,     # Faster calibration
                    'alignment_tolerance': 0.15, # More tolerant
                    'use_cached_rotation': True
                },
                'floor_params': {
                    'distance_threshold': 0.02,     # 2cm RANSAC tolerance
                    'max_trials': 1000,
                    'min_floor_points': 200,        # Fewer points needed
                    'floor_normal_tolerance': 0.25, # More tolerance
                    'min_floor_area': 0.03
                },
                'object_height_params': {
                    'min_height_above_floor': 0.01,  # 1cm minimum above floor
                    'max_height_above_floor': 0.25,  # 25cm maximum above floor (more lenient)
                    'height_tolerance': 0.01,        # 1cm tolerance
                    'min_object_points': 25          # Fewer points needed
                },
                'clustering_params': {
                    'eps': 0.03,                     # 3cm clustering (more lenient)
                    'min_samples': 6,                # Fewer samples needed
                    'min_cluster_size': 40,          # Smaller clusters allowed
                    'max_cluster_size': 2000,        # Larger clusters allowed
                    'outlier_removal': False         # Disabled for demo
                },
                'visualization_params': {
                    'enable_all_steps': True,
                    'show_6panel_view': True,
                    'point_cloud_colors': {
                        'raw': (128, 128, 128),
                        'filtered': (0, 255, 0),
                        'rotated': (255, 255, 0),
                        'floor': (255, 0, 0),
                        'objects': (0, 0, 255),
                        'projected': (255, 0, 255)
                    }
                }
            }
        },
        
        'enhanced_lidar_debug': {
            'type': 'lidar',
            'config': {
                'strategy_type': 'enhanced',  # Use EnhancedLiDARStrategy  
                'spatial_filter_params': {
                    'z_range': (0.15, 0.8),  # Wider depth range for debugging
                    'x_range': (-0.3, 0.3),  # Wider lateral range
                    'y_range': None,         # No Y filtering
                    'min_points_after_filter': 50
                },
                'gravity_params': {
                    'gravity_vector': [0, 0, -1],  # Target gravity in world frame (Z-up)
                    'alignment_samples': 15,    # More samples for accuracy
                    'alignment_tolerance': 0.2, # Very tolerant
                    'use_cached_rotation': False # Don't cache for debugging
                },
                'floor_params': {
                    'distance_threshold': 0.03,     # 3cm RANSAC tolerance (relaxed)
                    'max_trials': 800,
                    'min_floor_points': 150,        # Fewer points needed
                    'floor_normal_tolerance': 0.4,  # Very tolerant
                    'min_floor_area': 0.02
                },
                'object_height_params': {
                    'min_height_above_floor': 0.005, # 5mm minimum (very low)
                    'max_height_above_floor': 0.30,  # 30cm maximum (high)
                    'height_tolerance': 0.015,       # 1.5cm tolerance
                    'min_object_points': 20          # Very few points needed
                },
                'clustering_params': {
                    'eps': 0.04,                     # 4cm clustering (very lenient)
                    'min_samples': 5,                # Few samples needed
                    'min_cluster_size': 25,          # Small clusters allowed
                    'max_cluster_size': 3000,        # Very large clusters allowed
                    'outlier_removal': False         # Disabled for debugging
                },
                'visualization_params': {
                    'enable_all_steps': True,
                    'show_6panel_view': True,
                    'point_cloud_colors': {
                        'raw': (128, 128, 128),
                        'filtered': (0, 255, 0),
                        'rotated': (255, 255, 0),
                        'floor': (255, 0, 0),
                        'objects': (0, 0, 255),
                        'projected': (255, 0, 255)
                    }
                }
            }
        }
    }
    
    @classmethod
    def create_config(cls, config_type: str, 
                     custom_params: Optional[Dict[str, Any]] = None) -> DetectionConfig:
        """
        Create a configuration of the specified type.
        
        Args:
            config_type: Type of configuration ('base', 'bev', 'pointnet', 'checkerboard', 'charuco')
            custom_params: Optional custom parameters to override defaults
            
        Returns:
            Configuration instance
            
        Raises:
            ValueError: If config_type is not recognized
        """
        if config_type not in cls.CONFIG_TYPES:
            raise ValueError(f"Unknown config type: {config_type}. "
                           f"Available types: {list(cls.CONFIG_TYPES.keys())}")
        
        config_class = cls.CONFIG_TYPES[config_type]
        config = config_class(custom_params)
        
        return config
    
    @classmethod
    def create_preset(cls, preset_name: str, 
                     custom_params: Optional[Dict[str, Any]] = None) -> DetectionConfig:
        """
        Create a configuration from a predefined preset.
        
        Args:
            preset_name: Name of the preset
            custom_params: Optional custom parameters to override preset
            
        Returns:
            Configuration instance
            
        Raises:
            ValueError: If preset_name is not recognized
        """
        if preset_name not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. "
                           f"Available presets: {list(cls.PRESETS.keys())}")
        
        preset = cls.PRESETS[preset_name]
        config_type = preset['type']
        preset_params = preset['config'].copy()
        
        # Merge with custom parameters if provided
        if custom_params:
            preset_params.update(custom_params)
        
        return cls.create_config(config_type, preset_params)
    
    @classmethod
    def load_from_file(cls, file_path: str, 
                      config_type: str = 'base') -> DetectionConfig:
        """
        Load configuration from a JSON file.
        
        Args:
            file_path: Path to configuration file
            config_type: Type of configuration to create
            
        Returns:
            Configuration instance loaded from file
        """
        config = cls.create_config(config_type)
        config.load_from_file(file_path)
        return config
    
    @classmethod
    def create_hybrid_config(cls, 
                           bev_params: Optional[Dict[str, Any]] = None,
                           pointnet_params: Optional[Dict[str, Any]] = None) -> Dict[str, DetectionConfig]:
        """
        Create a hybrid configuration with both BEV and PointNet settings.
        
        Args:
            bev_params: Parameters for BEV configuration
            pointnet_params: Parameters for PointNet configuration
            
        Returns:
            Dictionary with 'bev' and 'pointnet' configurations
        """
        configs = {}
        
        if bev_params is not None:
            configs['bev'] = cls.create_config('bev', bev_params)
        
        if pointnet_params is not None:
            configs['pointnet'] = cls.create_config('pointnet', pointnet_params)
        
        return configs
    
    @classmethod
    def get_available_presets(cls) -> list:
        """
        Get list of available preset names.
        
        Returns:
            List of preset names
        """
        return list(cls.PRESETS.keys())
    
    @classmethod
    def get_available_types(cls) -> list:
        """
        Get list of available configuration types.
        
        Returns:
            List of configuration types
        """
        return list(cls.CONFIG_TYPES.keys())
    
    @classmethod
    def register_config_type(cls, type_name: str, config_class):
        """
        Register a new configuration type.
        
        Args:
            type_name: Name for the new type
            config_class: Configuration class
        """
        cls.CONFIG_TYPES[type_name] = config_class
    
    @classmethod
    def register_preset(cls, preset_name: str, config_type: str, config_dict: Dict[str, Any]):
        """
        Register a new configuration preset.
        
        Args:
            preset_name: Name for the new preset
            config_type: Type of configuration
            config_dict: Configuration parameters
        """
        cls.PRESETS[preset_name] = {
            'type': config_type,
            'config': config_dict
        }
    
    @classmethod
    def create_for_realtime(cls, use_bev: bool = True, 
                           use_pointnet: bool = False) -> Union[DetectionConfig, Dict[str, DetectionConfig]]:
        """
        Create configuration optimized for real-time processing.
        
        Args:
            use_bev: Whether to include BEV configuration
            use_pointnet: Whether to include PointNet configuration
            
        Returns:
            Single configuration or dictionary of configurations
        """
        configs = {}
        
        if use_bev:
            configs['bev'] = cls.create_preset('realtime_bev')
        
        if use_pointnet:
            configs['pointnet'] = cls.create_preset('fast_pointnet')
        
        # Return single config if only one requested
        if len(configs) == 1:
            return list(configs.values())[0]
        
        return configs
    
    @classmethod
    def create_for_offline(cls, use_bev: bool = False, 
                          use_pointnet: bool = True) -> Union[DetectionConfig, Dict[str, DetectionConfig]]:
        """
        Create configuration optimized for offline processing.
        
        Args:
            use_bev: Whether to include BEV configuration
            use_pointnet: Whether to include PointNet configuration
            
        Returns:
            Single configuration or dictionary of configurations
        """
        configs = {}
        
        if use_bev:
            configs['bev'] = cls.create_preset('high_quality_bev')
        
        if use_pointnet:
            configs['pointnet'] = cls.create_preset('offline_pointnet')
        
        # Return single config if only one requested
        if len(configs) == 1:
            return list(configs.values())[0]
        
        return configs
    
    @classmethod
    def validate_config(cls, config: DetectionConfig) -> bool:
        """
        Validate a configuration instance.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        return config.validate_config()
    
    @classmethod
    def get_preset_info(cls, preset_name: str) -> Dict[str, Any]:
        """
        Get information about a specific preset.
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Dictionary with preset information
        """
        if preset_name not in cls.PRESETS:
            return {}
        
        preset = cls.PRESETS[preset_name]
        return {
            'name': preset_name,
            'type': preset['type'],
            'description': f"{preset_name.replace('_', ' ').title()} configuration preset",
            'parameters': preset['config']
        }