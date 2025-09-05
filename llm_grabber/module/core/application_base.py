"""
Base class for detection and tracking applications.

This module provides a common base class that eliminates boilerplate code
and ensures consistent initialization, cleanup, and user interaction patterns
across all applications.
"""

import cv2
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from ..config.config_factory import ConfigFactory
from ..sensors.realsense_manager import RealSenseManager
from ..robot.robot_controller import RobotController
from ..robot.mycobot_controller import MyCobotController
from ..processing.temporal_filter import TemporalFilter
from ..core.visualization_engine import VisualizationEngine
from ..utils.robot_utils import (
    initialize_robot_for_application,
    cleanup_robot_safely,
    get_robot_status_info
)
from ..utils.wait_utils import wait_for_robot_stabilization


class ApplicationBase(ABC):
    """
    Base class for detection and tracking applications.
    
    Provides common functionality for:
    - Configuration loading and management
    - Sensor initialization (RealSense cameras)
    - Robot controller setup and safety
    - Standard cleanup procedures
    - Common keyboard shortcuts
    - Performance monitoring infrastructure
    """
    
    def __init__(self, config_name: str = "default", 
                 transformation_matrix_file: Optional[str] = None,
                 bag_file: Optional[str] = None):
        """
        Initialize application base.
        
        Args:
            config_name: Configuration preset name
            transformation_matrix_file: Path to camera-robot transformation matrix
            bag_file: Optional bag file for replay mode
        """
        self.config_name = config_name
        self.transformation_matrix_file = transformation_matrix_file
        self.bag_file = bag_file
        
        # Load configuration
        self.config = self._load_configuration(config_name)
        
        # Initialize core components (will be set up in initialize())
        self.sensor_manager = None
        self.robot_controller = None
        self.visualizer = VisualizationEngine()
        self.temporal_filter = TemporalFilter(window_size=5)
        
        # Application state
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        
        # Performance monitoring
        self.performance_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'detection_times': [],
            'avg_fps': 0.0
        }
        
        # UI state
        self.show_help = False
        self.show_performance = True
    
    def _load_configuration(self, config_name: str):
        """Load configuration from factory."""
        try:
            return ConfigFactory.create_preset(config_name)
        except Exception as e:
            print(f"Warning: Could not load config '{config_name}': {e}")
            print("Using default configuration")
            return ConfigFactory.create_preset('realtime_tracking')
    
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            True if initialization successful
        """
        print(f"Initializing {self.__class__.__name__}")
        print("=" * 50)
        
        # Initialize sensor manager
        print("Initializing camera sensor...")
        self.sensor_manager = RealSenseManager(self.bag_file, realtime_mode=True, use_imu=True)
        if not self.sensor_manager.initialize():
            print("Error: Failed to initialize camera sensor")
            return False
        
        if not self.sensor_manager.start():
            print("Error: Failed to start camera sensor")
            return False
        
        # Initialize robot controller
        print("Initializing robot controller...")
        self.robot_controller = MyCobotController(self.transformation_matrix_file)
        
        # Connect to robot (optional)
        if not self.robot_controller.connect():
            print("Warning: Failed to connect to robot. Running in visualization-only mode.")
        
        # Configure robot if connected
        if self.robot_controller.is_connected:
            # Get robot parameters from config
            robot_params = self._get_robot_params_from_config()
            
            # Initialize robot with standard settings
            if initialize_robot_for_application(
                self.robot_controller, 
                robot_params,
                enable_async=True,
                max_queue_size=5
            ):
                print("[OK] Robot initialized successfully")
                
                # Perform robot setup sequence if required by subclass
                if self._should_setup_robot_pose():
                    success = self._setup_robot_pose()
                    if not success:
                        print("Warning: Robot pose setup failed")
            else:
                print("Warning: Robot initialization failed")
        
        # Perform sensor calibration if robot is stable
        if self.robot_controller.is_connected and self._should_calibrate_sensors():
            self._calibrate_sensors()
        
        # Initialize application-specific components
        if not self._initialize_application_specific():
            print("Error: Failed to initialize application-specific components")
            return False
        
        print("[OK] All systems initialized successfully")
        return True
    
    def _get_robot_params_from_config(self) -> Dict[str, Any]:
        """Extract robot parameters from configuration."""
        # Try different ways to get robot params from config
        if hasattr(self.config, 'robot_params'):
            return self.config.robot_params
        elif hasattr(self.config, '__dict__') and 'robot_params' in self.config.__dict__:
            return self.config.__dict__['robot_params']
        else:
            # Return default safety parameters
            return {
                'min_height': 100,
                'max_speed': 60,
                'approach_distance': 100,
                'retreat_distance': 200
            }
    
    def cleanup(self):
        """Standard cleanup procedure."""
        print(f"\nShutting down {self.__class__.__name__}...")
        
        # Application-specific cleanup
        self._cleanup_application_specific()
        
        # Return robot to safe pose and cleanup
        if self.robot_controller:
            cleanup_robot_safely(self.robot_controller, return_to_pose="table")
        
        # Stop sensor
        if self.sensor_manager:
            self.sensor_manager.stop()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Print final statistics
        self._print_final_statistics()
        
        print("[OK] Shutdown complete")
    
    def handle_standard_keyboard_input(self, key: int) -> bool:
        """
        Handle standard keyboard shortcuts common to all applications.
        
        Args:
            key: OpenCV key code
            
        Returns:
            True to continue running, False to quit
        """
        if key == ord('q'):
            return False
        elif key == ord('h'):
            self.show_help = not self.show_help
            print(f"Help {'enabled' if self.show_help else 'disabled'}")
        elif key == ord(' '):
            self.is_paused = not self.is_paused
            print(f"{'Paused' if self.is_paused else 'Resumed'}")
        elif key == ord('i'):
            self.show_performance = not self.show_performance
            print(f"Performance info {'enabled' if self.show_performance else 'disabled'}")
        elif key == ord('s'):
            # Stop robot movement
            if self.robot_controller and self.robot_controller.is_connected:
                from ..utils.robot_utils import stop_robot_safely
                stop_robot_safely(self.robot_controller)
        
        return True
    
    def _print_final_statistics(self):
        """Print final performance statistics."""
        stats = self.performance_stats
        
        print("\n" + "=" * 50)
        print("FINAL PERFORMANCE STATISTICS")
        print("=" * 50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        
        if stats['detection_times']:
            import numpy as np
            avg_detection = np.mean(stats['detection_times']) * 1000
            print(f"Average detection time: {avg_detection:.1f}ms")
    
    # Abstract methods that subclasses must implement
    
    @abstractmethod
    def _initialize_application_specific(self) -> bool:
        """
        Initialize application-specific components.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def _cleanup_application_specific(self):
        """Cleanup application-specific resources."""
        pass
    
    @abstractmethod
    def run(self):
        """Main application loop."""
        pass
    
    # Optional methods that subclasses can override
    
    def _should_setup_robot_pose(self) -> bool:
        """
        Whether this application should setup robot pose during initialization.
        
        Returns:
            True if robot pose setup is required
        """
        return False  # Default: no robot pose setup
    
    def _should_calibrate_sensors(self) -> bool:
        """
        Whether this application should calibrate sensors during initialization.
        
        Returns:
            True if sensor calibration is required
        """
        return True  # Default: calibrate sensors
    
    def _setup_robot_pose(self) -> bool:
        """
        Setup robot pose during initialization.
        
        Returns:
            True if setup successful
        """
        # Default implementation: move to observation pose and stabilize
        from ..utils.robot_utils import setup_robot_with_stabilization
        
        return setup_robot_with_stabilization(
            self.robot_controller,
            self.sensor_manager,
            pose_type="table",
            stabilization_time=3.0,
            window_name=f"{self.__class__.__name__} Setup"
        )
    
    def _calibrate_sensors(self):
        """Calibrate sensors after robot stabilization."""
        if self.sensor_manager.use_imu:
            print("Calibrating gravity alignment (robot is stable)...")
            self.sensor_manager.calibrate_gravity_alignment()
    
    def get_application_status(self) -> Dict[str, Any]:
        """
        Get comprehensive application status for display.
        
        Returns:
            Dictionary with status information
        """
        status = {
            'application': self.__class__.__name__,
            'config': self.config_name,
            'running': self.is_running,
            'paused': self.is_paused,
            'frames': self.frame_count,
            'fps': self.temporal_filter.calculate_fps()
        }
        
        # Add sensor status
        if self.sensor_manager:
            status['sensor'] = 'Connected' if self.sensor_manager.is_streaming else 'Disconnected'
        
        # Add robot status
        if self.robot_controller:
            robot_status = get_robot_status_info(self.robot_controller)
            status.update({'robot_' + k: v for k, v in robot_status.items()})
        
        return status
    
    def create_standard_info_overlay(self, image, detected_objects, timing_info, strategy_info=None):
        """
        Create standard information overlay for applications.
        
        Args:
            image: Image to add overlay to
            detected_objects: List of detected objects
            timing_info: Timing information dictionary
            strategy_info: Optional strategy information
            
        Returns:
            Image with overlay added
        """
        status = self.get_application_status()
        
        info_lines = [
            f"{status['application'].upper()}",
            f"Config: {status['config']}",
            f"Objects: {len(detected_objects)}",
            f"FPS: {status['fps']:.1f}",
            f"Robot: {status.get('robot_connection', 'N/A')}",
        ]
        
        # Add strategy info if provided
        if strategy_info:
            info_lines.append(f"Strategy: {strategy_info.get('name', 'Unknown')}")
        
        # Use the extended VisualizationEngine
        return self.visualizer.add_info_overlay(image, info_lines)