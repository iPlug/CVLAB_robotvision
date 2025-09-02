"""
Abstract base class for robot controllers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from .command_queue import RobotCommandQueue, CommandType, CommandStatus


class RobotController(ABC):
    """
    Abstract base class for robot control systems.
    
    Provides a common interface for different robot types while allowing
    for robot-specific implementations of movement, safety constraints,
    and coordinate transformations.
    """
    
    def __init__(self, transformation_matrix_file: Optional[str] = None):
        """
        Initialize robot controller.
        
        Args:
            transformation_matrix_file: Path to camera-robot transformation matrix
        """
        self.is_connected = False
        self.transformation_matrix = None
        self.workspace_bounds = {}
        self.safety_constraints = {}
        self.current_position = None
        
        # Async command system
        self.command_queue = None
        self.async_mode_enabled = False
        self.last_command_time = 0
        
        if transformation_matrix_file:
            self.load_transformation_matrix(transformation_matrix_file)
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the robot.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the robot."""
        pass
    
    @abstractmethod
    def get_current_position(self) -> Optional[List[float]]:
        """
        Get current robot position.
        
        Returns:
            Current position as [x, y, z, rx, ry, rz] or None if unavailable
        """
        pass
    
    @abstractmethod
    def move_to_position(self, position: List[float], speed: int = 50) -> bool:
        """
        Move robot to specified position.
        
        Args:
            position: Target position as [x, y, z, rx, ry, rz]
            speed: Movement speed (robot-specific units)
            
        Returns:
            True if movement command sent successfully
        """
        pass
    
    @abstractmethod
    def move_to_angles(self, angles: List[float], speed: int = 50) -> bool:
        """
        Move robot to specified joint angles.
        
        Args:
            angles: Target joint angles
            speed: Movement speed (robot-specific units)
            
        Returns:
            True if movement command sent successfully
        """
        pass
    
    @abstractmethod
    def stop_movement(self) -> bool:
        """
        Stop robot movement immediately.
        
        Returns:
            True if stop command sent successfully
        """
        pass
    
    @abstractmethod
    def get_robot_info(self) -> Dict[str, Any]:
        """
        Get robot information and status.
        
        Returns:
            Dictionary with robot information
        """
        pass
    
    def load_transformation_matrix(self, filepath: str) -> bool:
        """
        Load camera-robot transformation matrix.
        
        Args:
            filepath: Path to transformation matrix file
            
        Returns:
            True if loaded successfully
        """
        try:
            self.transformation_matrix = np.load(filepath)
            print(f"Transformation matrix loaded: {filepath}")
            return True
        except Exception as e:
            print(f"Error loading transformation matrix: {e}")
            return False
    
    def transform_camera_to_robot(self, camera_position: List[float]) -> Optional[List[float]]:
        """
        Transform camera coordinates to robot coordinates.
        
        Args:
            camera_position: Position in camera frame [x, y, z]
            
        Returns:
            Position in robot frame [x, y, z] or None if transformation fails
        """
        if self.transformation_matrix is None:
            return None
        
        try:
            # Convert to homogeneous coordinates
            camera_homogeneous = np.array([camera_position[0], camera_position[1], camera_position[2], 1])
            
            # Apply transformation
            robot_homogeneous = np.dot(self.transformation_matrix, camera_homogeneous)
            
            # Return 3D coordinates
            return robot_homogeneous[:3].tolist()
        except Exception as e:
            print(f"Error in coordinate transformation: {e}")
            return None
    
    def set_workspace_bounds(self, bounds: Dict[str, Tuple[float, float]]):
        """
        Set workspace bounds for safety validation.
        
        Args:
            bounds: Dictionary with 'x', 'y', 'z' keys and (min, max) tuples
        """
        self.workspace_bounds = bounds
    
    def set_safety_constraints(self, constraints: Dict[str, Any]):
        """
        Set safety constraints.
        
        Args:
            constraints: Dictionary with safety parameters
        """
        self.safety_constraints = constraints
    
    def validate_position_safety(self, position: List[float]) -> Tuple[bool, str]:
        """
        Validate if position is safe for robot movement.
        
        Args:
            position: Target position [x, y, z, rx, ry, rz]
            
        Returns:
            Tuple of (is_safe, reason)
        """
        if len(position) < 3:
            return False, "Invalid position format"
        
        x, y, z = position[:3]
        
        # Check workspace bounds
        if self.workspace_bounds:
            if 'x' in self.workspace_bounds:
                x_min, x_max = self.workspace_bounds['x']
                if not (x_min <= x <= x_max):
                    return False, f"X position {x:.1f} outside bounds [{x_min:.1f}, {x_max:.1f}]"
            
            if 'y' in self.workspace_bounds:
                y_min, y_max = self.workspace_bounds['y']
                if not (y_min <= y <= y_max):
                    return False, f"Y position {y:.1f} outside bounds [{y_min:.1f}, {y_max:.1f}]"
            
            if 'z' in self.workspace_bounds:
                z_min, z_max = self.workspace_bounds['z']
                if not (z_min <= z <= z_max):
                    return False, f"Z position {z:.1f} outside bounds [{z_min:.1f}, {z_max:.1f}]"
        
        # Check minimum height safety constraint
        min_height = self.safety_constraints.get('min_height', 0)
        if z < min_height:
            return False, f"Height {z:.1f} below minimum {min_height:.1f}"
        
        # Check maximum reach constraint
        max_reach = self.safety_constraints.get('max_reach')
        if max_reach:
            distance = np.sqrt(x**2 + y**2 + z**2)
            if distance > max_reach:
                return False, f"Position distance {distance:.1f} exceeds max reach {max_reach:.1f}"
        
        return True, "Position is safe"
    
    def clamp_position_to_workspace(self, position: List[float]) -> Tuple[List[float], List[str]]:
        """
        Clamp position to workspace bounds with safety adjustments.
        
        Args:
            position: Target position [x, y, z, rx, ry, rz]
            
        Returns:
            Tuple of (clamped_position, adjustments_made)
        """
        if len(position) < 3:
            return position, ["Invalid position format"]
        
        clamped = position.copy()
        adjustments = []
        
        # Clamp to workspace bounds
        if self.workspace_bounds:
            if 'x' in self.workspace_bounds:
                x_min, x_max = self.workspace_bounds['x']
                if clamped[0] < x_min:
                    adjustments.append(f"X: {clamped[0]:.1f} → {x_min:.1f}")
                    clamped[0] = x_min
                elif clamped[0] > x_max:
                    adjustments.append(f"X: {clamped[0]:.1f} → {x_max:.1f}")
                    clamped[0] = x_max
            
            if 'y' in self.workspace_bounds:
                y_min, y_max = self.workspace_bounds['y']
                if clamped[1] < y_min:
                    adjustments.append(f"Y: {clamped[1]:.1f} → {y_min:.1f}")
                    clamped[1] = y_min
                elif clamped[1] > y_max:
                    adjustments.append(f"Y: {clamped[1]:.1f} → {y_max:.1f}")
                    clamped[1] = y_max
            
            if 'z' in self.workspace_bounds:
                z_min, z_max = self.workspace_bounds['z']
                if clamped[2] < z_min:
                    adjustments.append(f"Z: {clamped[2]:.1f} → {z_min:.1f}")
                    clamped[2] = z_min
                elif clamped[2] > z_max:
                    adjustments.append(f"Z: {clamped[2]:.1f} → {z_max:.1f}")
                    clamped[2] = z_max
        
        # Apply minimum height constraint
        min_height = self.safety_constraints.get('min_height', 0)
        if clamped[2] < min_height:
            adjustments.append(f"Height safety: {clamped[2]:.1f} → {min_height:.1f}")
            clamped[2] = min_height
        
        return clamped, adjustments
    
    def calculate_pointing_toward_object(self, target_position: List[float], 
                                       pointing_orientation: List[float]) -> Optional[List[float]]:
        """
        Calculate optimal robot position to point toward an unreachable object.
        
        Args:
            target_position: Unreachable object position [x, y, z]
            pointing_orientation: Desired pointing orientation [rx, ry, rz]
            
        Returns:
            Safe pointing position [x, y, z, rx, ry, rz] or None if no solution found
        """
        if len(target_position) < 3:
            return None
        
        # Calculate direction vector from robot base to target
        target_x, target_y, target_z = target_position[:3]
        direction = np.array([target_x, target_y, target_z])
        distance_to_target = np.linalg.norm(direction)
        
        if distance_to_target == 0:
            return None
        
        # Normalize direction vector
        direction_normalized = direction / distance_to_target
        
        # Find maximum safe distance in this direction
        max_safe_distance = self.safety_constraints.get('max_reach', 350)
        min_safe_distance = 50  # Avoid singularities
        
        # Calculate safe position along the direction vector
        # Use 80% of max reach for safety margin
        safe_distance = min(max_safe_distance * 0.8, distance_to_target * 0.9)
        safe_distance = max(safe_distance, min_safe_distance)
        
        pointing_position = direction_normalized * safe_distance
        
        # Ensure Z is within safe bounds
        min_height = self.safety_constraints.get('min_height', 100)
        if pointing_position[2] < min_height:
            pointing_position[2] = min_height
        
        # Create full pose with orientation
        pointing_pose = pointing_position.tolist() + pointing_orientation
        
        # Validate this pointing position
        is_safe, reason = self.validate_position_safety(pointing_pose)
        if is_safe:
            return pointing_pose
        
        # If still not safe, try alternative strategies
        # Strategy 1: Move closer to robot base while maintaining direction
        for scale_factor in [0.7, 0.6, 0.5, 0.4]:
            scaled_distance = max_safe_distance * scale_factor
            scaled_position = direction_normalized * scaled_distance
            scaled_position[2] = max(scaled_position[2], min_height)
            
            scaled_pose = scaled_position.tolist() + pointing_orientation
            is_safe, _ = self.validate_position_safety(scaled_pose)
            if is_safe:
                return scaled_pose
        
        # Strategy 2: Point horizontally toward object (ignore Z difference)
        horizontal_direction = np.array([target_x, target_y, 0])
        horizontal_distance = np.linalg.norm(horizontal_direction)
        
        if horizontal_distance > 0:
            horizontal_normalized = horizontal_direction / horizontal_distance
            horizontal_safe_distance = min(max_safe_distance * 0.8, horizontal_distance * 0.9)
            horizontal_position = horizontal_normalized * horizontal_safe_distance
            horizontal_position[2] = min_height  # Safe height
            
            horizontal_pose = horizontal_position.tolist() + pointing_orientation
            is_safe, _ = self.validate_position_safety(horizontal_pose)
            if is_safe:
                return horizontal_pose
        
        return None
    
    def point_at_object(self, camera_position: List[float], 
                       pointing_orientation: Optional[List[float]] = None) -> bool:
        """
        Point robot at object detected by camera.
        
        Args:
            camera_position: Object position in camera coordinates [x, y, z]
            pointing_orientation: Optional orientation angles [rx, ry, rz]
            
        Returns:
            True if movement successful
        """
        # Transform camera coordinates to robot coordinates
        robot_position = self.transform_camera_to_robot(camera_position)
        if robot_position is None:
            print("Failed to transform camera coordinates to robot coordinates")
            return False
        
        # Use default pointing orientation if not specified
        if pointing_orientation is None:
            pointing_orientation = self.safety_constraints.get('default_pointing_angles', [0, 180, 0])
        
        # Combine position and orientation
        target_pose = robot_position + pointing_orientation
        
        # Validate and clamp position for safety
        is_safe, reason = self.validate_position_safety(target_pose)
        if not is_safe:
            print(f"Object at [{robot_position[0]:.1f}, {robot_position[1]:.1f}, {robot_position[2]:.1f}] is unreachable: {reason}")
            
            # Try to find the best pointing position toward the object
            pointing_pose = self.calculate_pointing_toward_object(robot_position, pointing_orientation)
            if pointing_pose:
                print(f"Using pointing fallback - moving to point toward object")
                target_pose = pointing_pose
            else:
                # Fallback to basic clamping
                clamped_pose, adjustments = self.clamp_position_to_workspace(target_pose)
                if adjustments:
                    print(f"Applied basic safety adjustments: {', '.join(adjustments)}")
                    target_pose = clamped_pose
                else:
                    print("Cannot find safe pointing position, aborting movement")
                    return False
        
        # Execute movement
        print(f"Moving robot to point at object:")
        print(f"  Camera position: [{camera_position[0]:.1f}, {camera_position[1]:.1f}, {camera_position[2]:.1f}]")
        print(f"  Robot position:  [{robot_position[0]:.1f}, {robot_position[1]:.1f}, {robot_position[2]:.1f}]")
        print(f"  Final pose: {[f'{x:.1f}' for x in target_pose]}")
        
        return self.move_to_position(target_pose)
    
    def home_position(self) -> bool:
        """
        Move robot to home position.
        
        Returns:
            True if movement successful
        """
        home_angles = self.safety_constraints.get('home_angles', [0, 0, 0, 0, 0, 0])
        return self.move_to_angles(home_angles)
    
    def look_at_table(self) -> bool:
        """
        Move robot to home position.
        
        Returns:
            True if movement successful
        """
        
        # return self.move_to_angles([0, 0, 0, -45, 0, 45])
        return self.move_to_angles([72.5, 18.54, -3.86, -97.29, -15.11, 116.19])
    
    def look_forward(self) -> bool:
        """
        Move robot to look forward position (horizontal view).
        
        Returns:
            True if movement successful
        """
        
        return self.move_to_angles([0, 0, 0, 0, 0, 45])
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive robot status.
        
        Returns:
            Dictionary with robot status information
        """
        status = {
            'is_connected': self.is_connected,
            'has_transformation_matrix': self.transformation_matrix is not None,
            'current_position': self.current_position,
            'workspace_bounds': self.workspace_bounds,
            'safety_constraints': self.safety_constraints
        }
        
        # Add robot-specific information
        status.update(self.get_robot_info())
        
        # Add async command system status
        if self.async_mode_enabled and self.command_queue:
            status['async_mode'] = True
            status['command_queue_status'] = self.command_queue.get_queue_status()
        else:
            status['async_mode'] = False
        
        return status
    
    # ===== Async Command System Methods =====
    
    def enable_async_mode(self, max_queue_size: int = 10) -> bool:
        """
        Enable asynchronous command execution.
        
        Args:
            max_queue_size: Maximum number of commands in queue
            
        Returns:
            True if async mode enabled successfully
        """
        if self.async_mode_enabled:
            return True
        
        try:
            self.command_queue = RobotCommandQueue(
                robot_controller=self,
                max_queue_size=max_queue_size
            )
            
            if self.command_queue.start():
                self.async_mode_enabled = True
                return True
            else:
                self.command_queue = None
                return False
                
        except Exception as e:
            print(f"Failed to enable async mode: {e}")
            return False
    
    def disable_async_mode(self, timeout: float = 5.0) -> bool:
        """
        Disable asynchronous command execution.
        
        Args:
            timeout: Time to wait for pending commands to complete
            
        Returns:
            True if disabled successfully
        """
        if not self.async_mode_enabled:
            return True
        
        try:
            if self.command_queue:
                success = self.command_queue.stop(timeout=timeout)
                self.command_queue = None
            else:
                success = True
            
            self.async_mode_enabled = False
            return success
            
        except Exception as e:
            print(f"Error disabling async mode: {e}")
            return False
    
    def submit_async_command(self, command_type: CommandType, *args,
                           priority: int = 1, timeout: float = 10.0,
                           callback: Optional[Callable] = None, **kwargs) -> Optional[int]:
        """
        Submit a command for asynchronous execution.
        
        Args:
            command_type: Type of command to execute
            *args: Positional arguments for the command
            priority: Command priority (higher = more important)
            timeout: Command timeout in seconds
            callback: Optional completion callback
            **kwargs: Keyword arguments for the command
            
        Returns:
            Command ID if submitted successfully, None otherwise
        """
        if not self.async_mode_enabled or not self.command_queue:
            print("Error: Async mode not enabled")
            return None
        
        return self.command_queue.submit_command(
            command_type,
            *args,
            priority=priority,
            timeout=timeout,
            callback=callback,
            **kwargs
        )
    
    def point_at_object_async(self, camera_position: List[float],
                            priority: int = 2, timeout: float = 15.0,
                            callback: Optional[Callable] = None) -> Optional[int]:
        """
        Asynchronously point robot at object detected by camera.
        
        Args:
            camera_position: Object position in camera coordinates
            priority: Command priority
            timeout: Command timeout in seconds
            callback: Optional completion callback
            
        Returns:
            Command ID if submitted successfully, None otherwise
        """
        return self.submit_async_command(
            CommandType.CUSTOM_COMMAND,
            'point_at_object',
            camera_position,
            priority=priority,
            timeout=timeout,
            callback=callback
        )
    
    def move_to_position_async(self, position: List[float], speed: int = 50,
                             priority: int = 1, timeout: float = 10.0,
                             callback: Optional[Callable] = None,
                             linear_mode: bool = False) -> Optional[int]:
        """
        Asynchronously move robot to specified position.
        
        Args:
            position: Target position [x, y, z, rx, ry, rz]
            speed: Movement speed
            priority: Command priority
            timeout: Command timeout in seconds
            callback: Optional completion callback
            linear_mode: If True, use linear interpolation (straight path).
                        If False, use joint interpolation (curved path)
            
        Returns:
            Command ID if submitted successfully, None otherwise
        """
        return self.submit_async_command(
            CommandType.MOVE_TO_POSITION,
            position,
            speed,
            linear_mode,  # Add linear_mode as an argument
            priority=priority,
            timeout=timeout,
            callback=callback
        )
    
    def home_position_async(self, priority: int = 1, timeout: float = 10.0,
                          callback: Optional[Callable] = None) -> Optional[int]:
        """
        Asynchronously move robot to home position.
        
        Args:
            priority: Command priority
            timeout: Command timeout in seconds
            callback: Optional completion callback
            
        Returns:
            Command ID if submitted successfully, None otherwise
        """
        return self.submit_async_command(
            CommandType.HOME_POSITION,
            priority=priority,
            timeout=timeout,
            callback=callback
        )
    
    def look_at_table_async(self, priority: int = 1, timeout: float = 10.0,
                           callback: Optional[Callable] = None) -> Optional[int]:
        """
        Asynchronously move robot to look at table position.
        
        Args:
            priority: Command priority
            timeout: Command timeout in seconds
            callback: Optional completion callback
            
        Returns:
            Command ID if submitted successfully, None otherwise
        """
        return self.submit_async_command(
            CommandType.LOOK_AT_TABLE,
            priority=priority,
            timeout=timeout,
            callback=callback
        )
    
    def look_forward_async(self, priority: int = 1, timeout: float = 10.0,
                         callback: Optional[Callable] = None) -> Optional[int]:
        """
        Asynchronously move robot to look forward position.
        
        Args:
            priority: Command priority
            timeout: Command timeout in seconds
            callback: Optional completion callback
            
        Returns:
            Command ID if submitted successfully, None otherwise
        """
        return self.submit_async_command(
            CommandType.LOOK_FORWARD,
            priority=priority,
            timeout=timeout,
            callback=callback
        )
    
    def stop_movement_async(self, priority: int = 3, timeout: float = 5.0,
                          callback: Optional[Callable] = None) -> Optional[int]:
        """
        Asynchronously stop robot movement (high priority).
        
        Args:
            priority: Command priority (default high)
            timeout: Command timeout in seconds
            callback: Optional completion callback
            
        Returns:
            Command ID if submitted successfully, None otherwise
        """
        return self.submit_async_command(
            CommandType.STOP_MOVEMENT,
            priority=priority,
            timeout=timeout,
            callback=callback
        )
    
    def get_command_status(self, command_id: int) -> Optional[CommandStatus]:
        """Get the status of a specific command."""
        if self.command_queue:
            return self.command_queue.get_command_status(command_id)
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current command queue status."""
        if self.command_queue:
            return self.command_queue.get_queue_status()
        return {'error': 'Async mode not enabled'}
    
    def cleanup_async_system(self):
        """Cleanup async command system during shutdown."""
        if self.async_mode_enabled:
            self.disable_async_mode(timeout=3.0)