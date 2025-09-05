"""
myCobot-specific robot controller implementation.
"""

import time
import sys
import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .robot_controller import RobotController

# Add myCobot path for imports (local to llm_grabber)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'myCobot'))


class MyCobotController(RobotController):
    """
    Robot controller implementation for myCobot 280 robot arm.
    
    Provides myCobot-specific implementations for movement, safety constraints,
    and coordinate transformations based on the original track_object.py implementation.
    """
    
    def __init__(self, transformation_matrix_file: Optional[str] = None):
        """
        Initialize myCobot controller.
        
        Args:
            transformation_matrix_file: Path to camera-robot transformation matrix
        """
        super().__init__(transformation_matrix_file)
        
        self.robot = None
        self.robot_type = "myCobot 280"
        
        # Set myCobot 280 M5 safety constraints based on official specifications
        self.set_safety_constraints({
            'min_height': 0,    # Minimum height disabled for maximum grasping flexibility
            'max_reach': 280,   # Maximum reach in mm (280mm effective radius)
            'skip_safety_validation': True,   # Set to False to enable safety checks (disabled by default for grasping)
            'default_pointing_angles': [0, 180, 45],  # Fixed rotation for pointing down (compensates for -45° camera mount)
            'home_angles': [0, 0, 0, 0, 0, 0],  # Home position joint angles
            'max_speed': 90,    # Maximum movement speed
            'min_update_interval': 0.5,  # Minimum time between commands
            'joint_limits': {   # Joint angle limits from myCobot 280 M5 specifications
                'J1': (-165, 165), 'J2': (-165, 165), 'J3': (-165, 165),
                'J4': (-165, 165), 'J5': (-165, 165), 'J6': (-179, 179)
            }
        })
        
        # Set myCobot 280 M5 workspace bounds - 280mm effective radius from J1 (base joint)
        # J1 is 131.56mm above physical base, so workspace extends from J1 level
        self.set_workspace_bounds({
            'x': (50, 350),     # Forward range in mm (280mm from J1 + safety margin)
            'y': (-280, 280),   # Side range in mm (±280mm from J1)
            'z': (50, 420)      # Height range: clearance to J1_height + reach (131.56 + 280 + margin)
        })
    
    def connect(self) -> bool:
        """
        Connect to myCobot robot.
        
        Returns:
            True if connection successful
        """
        try:
            # Import myCobot setup function
            from port_setup import setup
            
            print("Connecting to myCobot...")
            self.robot = setup()
            
            if self.robot:
                self.is_connected = True
                print("myCobot connected successfully!")
                
                # Get current position for status
                try:
                    self.current_position = self.robot.get_coords()
                except:
                    self.current_position = None
                
                return True
            else:
                print("Failed to connect to myCobot")
                return False
                
        except Exception as e:
            print(f"Error connecting to myCobot: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from myCobot robot."""
        if self.is_connected:
            try:
                # No explicit disconnect needed for myCobot
                self.robot = None
                self.is_connected = False
                print("myCobot disconnected")
            except Exception as e:
                print(f"Error disconnecting from myCobot: {e}")
    
    def get_current_position(self) -> Optional[List[float]]:
        """
        Get current myCobot position.
        
        Returns:
            Current position as [x, y, z, rx, ry, rz] or None if unavailable
        """
        if not self.is_connected or not self.robot:
            return None
        
        try:
            position = self.robot.get_coords()
            if position:
                self.current_position = position
                return position
        except Exception as e:
            print(f"Error getting myCobot position: {e}")
        
        return None
    
    def move_to_position(self, position: List[float], speed: int = 50, linear_mode: bool = False) -> bool:
        """
        Move myCobot to specified position with smart reach scaling.
        
        If the position is unreachable, it will be automatically scaled
        to the maximum reach boundary in the target direction.
        
        Args:
            position: Target position as [x, y, z, rx, ry, rz]
            speed: Movement speed (0-100)
            linear_mode: If True, use linear interpolation (straight path). 
                        If False, use joint interpolation (curved path)
            
        Returns:
            True if movement command sent successfully
        """
        if not self.is_connected or not self.robot:
            print("Error: myCobot not connected")
            return False
        
        if len(position) < 6:
            print("Error: Position must have 6 elements [x, y, z, rx, ry, rz]")
            return False
        
        # Check rate limiting
        current_time = time.time()
        min_interval = self.safety_constraints.get('min_update_interval', 0.5)
        if current_time - self.last_command_time < min_interval:
            print(f"Rate limited: {current_time - self.last_command_time:.2f}s < {min_interval}s")
            return False
        
        # Check if safety validation should be skipped
        skip_safety = self.safety_constraints.get('skip_safety_validation', False)
        
        # Apply smart scaling for unreachable positions
        adjusted_position, was_adjusted = self.calculate_reachable_position(position)
        if was_adjusted:
            print(f"Position automatically adjusted to reachable coordinates")
        
        if not skip_safety:
            # Validate safety of the adjusted position
            is_safe, reason = self.validate_position_safety(adjusted_position)
            if not is_safe:
                print(f"Unsafe movement blocked: {reason}")
                return False
        else:
            print("[WARNING] Safety validation skipped - using raw coordinates")
        
        # Clamp speed to safe range (unless safety is completely disabled)
        if not skip_safety:
            speed = max(0, min(self.safety_constraints.get('max_speed', 90), speed))
        # If safety is skipped, use raw speed value
        
        try:
            # Choose movement mode: 1 for linear (straight path), 0 for joint interpolation (curved path)
            movement_mode = 1 if linear_mode else 0
            self.robot.send_coords(adjusted_position, speed, movement_mode)
            self.last_command_time = current_time
            self.current_position = adjusted_position
            
            # Log command for debugging
            mode_desc = "linear" if linear_mode else "joint"
            print(f"myCobot movement command sent: {[f'{x:.1f}' for x in adjusted_position]}, speed: {speed}, mode: {mode_desc}")
            return True
            
        except Exception as e:
            print(f"Error moving myCobot: {e}")
            return False
    
    def validate_joint_angles(self, angles: List[float]) -> Tuple[bool, str]:
        """
        Validate joint angles against myCobot 280 M5 specifications.
        
        Args:
            angles: Joint angles in degrees [J1, J2, J3, J4, J5, J6]
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if len(angles) != 6:
            return False, "Must provide exactly 6 joint angles"
        
        joint_limits = self.safety_constraints.get('joint_limits', {})
        if not joint_limits:
            return True, "No joint limits configured"
        
        joint_names = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']
        
        for i, (joint_name, angle) in enumerate(zip(joint_names, angles)):
            if joint_name in joint_limits:
                min_angle, max_angle = joint_limits[joint_name]
                if not (min_angle <= angle <= max_angle):
                    return False, f"{joint_name} angle {angle:.1f}° outside limits [{min_angle}°, {max_angle}°]"
        
        return True, "Joint angles are valid"

    def calculate_reachable_position(self, target_coords: List[float]) -> Tuple[List[float], bool]:
        """
        Calculate reachable position for target coordinates using J1-based calculation.
        
        If the coordinate is reachable, returns it unchanged.
        If it's unreachable, scales it to the maximum reach boundary
        in the direction from J1 to the target.
        
        Args:
            target_coords: Target coordinates [x, y, z, rx, ry, rz]
            
        Returns:
            Tuple of (adjusted_coords, was_adjusted)
        """
        if len(target_coords) < 6:
            return target_coords, False
        
        # Get maximum reach from safety constraints
        MAX_REACH = self.safety_constraints.get('max_reach', 280.0)
        J1_HEIGHT = 131.56  # Height of J1 above physical base
        
        # Extract the XYZ position and RX, RY, RZ orientation
        x, y, z = target_coords[:3]
        target_orientation = target_coords[3:]
        
        # Calculate the distance from J1 (not physical base)
        distance_from_j1 = np.sqrt(x**2 + y**2 + (z - J1_HEIGHT)**2)
        
        if distance_from_j1 > MAX_REACH:
            print(f"Coordinate is UNREACHABLE. Distance from J1: {distance_from_j1:.2f}mm > Max reach: {MAX_REACH}mm")
            
            # Use pointing behavior like point_to_coordinate.py
            target_position = target_coords[:3]
            default_orientation = self.safety_constraints.get('default_pointing_angles', [0, 180, 0])
            
            # Try to calculate a pointing position toward the target
            pointing_pose = self.calculate_pointing_toward_object(target_position, default_orientation)
            
            if pointing_pose:
                print(f"Using pointing position toward target: {np.round(pointing_pose[:3], 2)}")
                print("Position automatically adjusted to point toward target")
                return pointing_pose, True
            else:
                # Fallback to old scaling method if pointing calculation fails
                print("Pointing calculation failed, using distance scaling fallback")
                
                # Create vector from J1 to target
                target_from_j1 = np.array([x, y, z - J1_HEIGHT])
                
                # Calculate the unit vector (direction from J1)
                direction_vector = target_from_j1 / distance_from_j1
                
                # Scale the vector to the maximum reach (with 90% safety margin)
                scaled_from_j1 = direction_vector * (MAX_REACH * 0.9)
                
                # Convert back to absolute coordinates (add J1 height back)
                scaled_position = np.array([scaled_from_j1[0], scaled_from_j1[1], scaled_from_j1[2] + J1_HEIGHT])
                
                # Combine the new position with the original orientation
                new_coords = scaled_position.tolist() + target_orientation
                
                print(f"Scaled to reachable position: {np.round(new_coords[:3], 2)}")
                return new_coords, True
        else:
            return target_coords, False

    def move_to_angles(self, angles: List[float], speed: int = 50) -> bool:
        """
        Move myCobot to specified joint angles.
        
        Args:
            angles: Target joint angles in degrees [J1, J2, J3, J4, J5, J6]
            speed: Movement speed (0-100)
            
        Returns:
            True if movement command sent successfully
        """
        if not self.is_connected or not self.robot:
            print("Error: myCobot not connected")
            return False
        
        if len(angles) != 6:
            print("Error: Must provide 6 joint angles")
            return False
        
        # Validate joint angles against specifications
        is_valid, reason = self.validate_joint_angles(angles)
        if not is_valid:
            print(f"Invalid joint angles: {reason}")
            return False
        
        # Check rate limiting
        current_time = time.time()
        min_interval = self.safety_constraints.get('min_update_interval', 0.5)
        if current_time - self.last_command_time < min_interval:
            print(f"Rate limited: {current_time - self.last_command_time:.2f}s < {min_interval}s")
            return False
        
        # Clamp speed to safe range
        speed = max(0, min(self.safety_constraints.get('max_speed', 90), speed))
        
        try:
            # Send angle movement command to myCobot
            self.robot.send_angles(angles, speed)
            self.last_command_time = current_time
            print(f"myCobot angle command sent: {[f'{x:.1f}' for x in angles]}, speed: {speed}")
            return True
            
        except Exception as e:
            print(f"Error moving myCobot angles: {e}")
            return False
    
    def stop_movement(self) -> bool:
        """
        Stop myCobot movement immediately.
        
        Returns:
            True if stop command sent successfully
        """
        if not self.is_connected or not self.robot:
            print("Error: myCobot not connected")
            return False
        
        try:
            # Get current position and send it with speed 0 to stop
            current_pos = self.robot.get_coords()
            if current_pos:
                self.robot.send_coords(current_pos, 0, 0)
                print("myCobot movement stopped")
                return True
            else:
                print("Could not get current position to stop movement")
                return False
                
        except Exception as e:
            print(f"Error stopping myCobot: {e}")
            return False
    
    def open_gripper(self) -> bool:
        """
        Open the gripper.
        
        Returns:
            True if successful
        """
        if not self.is_connected:
            print("Robot not connected")
            return False
        
        try:
            # For myCobot 280, gripper control via set_gripper_state
            # State 0 = open, state 1 = closed
            self.robot.set_gripper_state(0, 100)  # Open gripper at speed 100
            print("Gripper opened")
            return True
        except Exception as e:
            print(f"Error opening gripper: {e}")
            return False
    
    def close_gripper(self) -> bool:
        """
        Close the gripper.
        
        Returns:
            True if successful
        """
        if not self.is_connected:
            print("Robot not connected")
            return False
        
        try:
            # For myCobot 280, gripper control via set_gripper_state
            # State 0 = open, state 1 = closed
            self.robot.set_gripper_state(1, 100)  # Close gripper at speed 100
            print("Gripper closed")
            return True
        except Exception as e:
            print(f"Error closing gripper: {e}")
            return False
    
    def get_robot_info(self) -> Dict[str, Any]:
        """
        Get myCobot information and status.
        
        Returns:
            Dictionary with myCobot information
        """
        info = {
            'robot_type': self.robot_type,
            'connection_method': 'USB/Serial',
            'degrees_of_freedom': 6,
            'payload': '250g',
            'reach': '280mm effective radius',
            'repeatability': '0.5mm',
            'joint_limits': 'J1-J5: ±165°, J6: ±179°',
            'supported_commands': ['send_coords', 'send_angles', 'get_coords', 'get_angles']
        }
        
        if self.is_connected and self.robot:
            try:
                # Try to get additional status information
                current_angles = self.robot.get_angles()
                if current_angles:
                    info['current_angles'] = current_angles
                
                # Add connection status
                info['connection_status'] = 'Connected'
                
            except Exception as e:
                info['connection_status'] = f'Connected but error getting status: {e}'
        else:
            info['connection_status'] = 'Disconnected'
        
        return info
    
    def calibrate_home_position(self) -> bool:
        """
        Move myCobot to home position for calibration.
        
        Returns:
            True if successful
        """
        home_angles = self.safety_constraints.get('home_angles', [0, 0, 0, 0, 0, 0])
        print("Moving myCobot to home position for calibration...")
        return self.move_to_angles(home_angles, 30)  # Slow speed for safety
    
    def get_joint_angles(self) -> Optional[List[float]]:
        """
        Get current joint angles.
        
        Returns:
            Current joint angles or None if unavailable
        """
        if not self.is_connected or not self.robot:
            return None
        
        try:
            angles = self.robot.get_angles()
            return angles
        except Exception as e:
            print(f"Error getting myCobot angles: {e}")
            return None
    
    def is_moving(self) -> bool:
        """
        Check if robot is currently moving.
        
        Returns:
            True if robot is moving (estimated based on timing)
        """
        # myCobot doesn't provide direct movement status
        # Estimate based on time since last command
        time_since_command = time.time() - self.last_command_time
        return time_since_command < 3.0  # Assume movement takes max 3 seconds
    
    def set_mycobot_specific_params(self, **kwargs):
        """
        Set myCobot-specific parameters.
        
        Kwargs:
            gripper_enabled: Whether gripper is attached
            tool_offset: Tool offset coordinates [x, y, z]
            joint_limits: Custom joint angle limits
        """
        if 'gripper_enabled' in kwargs:
            self.safety_constraints['gripper_enabled'] = kwargs['gripper_enabled']
        
        if 'tool_offset' in kwargs:
            self.safety_constraints['tool_offset'] = kwargs['tool_offset']
        
        if 'joint_limits' in kwargs:
            self.safety_constraints['joint_limits'] = kwargs['joint_limits']
    
    def set_safety_validation(self, enabled: bool = True):
        """
        Enable or disable safety validation.
        
        Args:
            enabled: True to enable safety validation (default), False to skip all safety checks
        
        Warning:
            Disabling safety validation removes all movement constraints and can be dangerous.
            Use with extreme caution and manual oversight.
        """
        self.safety_constraints['skip_safety_validation'] = not enabled
        status = "ENABLED" if enabled else "DISABLED (WARNING: Use with caution)"
        print(f"Safety validation: {status}")
    
    def emergency_stop(self) -> bool:
        """
        Emergency stop for myCobot.
        
        Returns:
            True if emergency stop successful
        """
        print("EMERGENCY STOP: Stopping myCobot immediately")
        
        # Try multiple stop methods
        success = False
        
        # Method 1: Stop current movement
        if self.stop_movement():
            success = True
        
        # Method 2: Move to safe home position
        try:
            if self.robot:
                self.robot.send_angles([0, 0, 0, 0, 0, 0], 0)  # Zero speed to home
                success = True
        except:
            pass
        
        if success:
            print("Emergency stop completed")
        else:
            print("Emergency stop failed - manual intervention may be required")
        
        return success
    
    def move_towards(self, target_coords: List[float], speed: int = 80) -> bool:
        """
        Convenience method that moves the robot towards a target coordinate.
        
        If the coordinate is reachable, it moves directly to it.
        If it's unreachable, it moves to the maximum reach boundary
        in the direction of the coordinate.
        
        Args:
            target_coords: Target coordinates [x, y, z, rx, ry, rz]
            speed: Movement speed (0-100)
            
        Returns:
            True if movement command sent successfully
        """
        return self.move_to_position(target_coords, speed)
    
    def validate_position_safety(self, position: List[float]) -> Tuple[bool, str]:
        """
        Override base class validation to use myCobot-specific J1-based reach calculation.
        
        Args:
            position: Target position [x, y, z, rx, ry, rz]
            
        Returns:
            Tuple of (is_safe, reason)
        """
        if len(position) < 3:
            return False, "Invalid position format"
        
        x, y, z = position[:3]
        
        # Check workspace bounds (inherited from base class)
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
        
        # myCobot-specific: Check maximum reach constraint from J1 (not base)
        max_reach = self.safety_constraints.get('max_reach')
        if max_reach:
            J1_HEIGHT = 131.56  # Height of J1 above physical base
            # Calculate distance from J1, not from physical base
            distance_from_j1 = np.sqrt(x**2 + y**2 + (z - J1_HEIGHT)**2)
            if distance_from_j1 > max_reach:
                return False, f"Position distance from J1: {distance_from_j1:.1f} exceeds max reach {max_reach:.1f}"
        
        return True, "Position is safe"
    
    def calculate_pointing_toward_object(self, target_position: List[float], 
                                       pointing_orientation: List[float]) -> Optional[List[float]]:
        """
        Override base class to calculate optimal robot position using J1-based calculations.
        
        Args:
            target_position: Unreachable object position [x, y, z]
            pointing_orientation: Desired pointing orientation [rx, ry, rz]
            
        Returns:
            Safe pointing position [x, y, z, rx, ry, rz] or None if no solution found
        """
        if len(target_position) < 3:
            return None
        
        J1_HEIGHT = 131.56  # Height of J1 above physical base
        
        # Calculate direction vector from J1 to target (not from physical base)
        target_x, target_y, target_z = target_position[:3]
        target_from_j1 = np.array([target_x, target_y, target_z - J1_HEIGHT])
        distance_to_target_from_j1 = np.linalg.norm(target_from_j1)
        
        if distance_to_target_from_j1 == 0:
            return None
        
        # Normalize direction vector from J1
        direction_normalized = target_from_j1 / distance_to_target_from_j1
        
        # Find maximum safe distance in this direction from J1
        max_safe_distance = self.safety_constraints.get('max_reach', 280)
        min_safe_distance = 50  # Avoid singularities
        
        # Calculate safe position along the direction vector from J1
        # Use 80% of max reach for safety margin
        safe_distance = min(max_safe_distance * 0.8, distance_to_target_from_j1 * 0.9)
        safe_distance = max(safe_distance, min_safe_distance)
        
        # Calculate pointing position relative to J1, then convert to absolute coordinates
        pointing_from_j1 = direction_normalized * safe_distance
        pointing_position = np.array([pointing_from_j1[0], pointing_from_j1[1], pointing_from_j1[2] + J1_HEIGHT])
        
        # Ensure Z is within safe bounds
        min_height = self.safety_constraints.get('min_height', 100)
        if pointing_position[2] < min_height:
            pointing_position[2] = min_height
        
        # Calculate orientation to point toward target
        # Vector from robot position to target
        robot_to_target = np.array(target_position) - pointing_position
        target_distance = np.linalg.norm(robot_to_target)
        
        if target_distance > 0:
            # Calculate required angles to point toward target
            # Yaw angle (rotation around Z-axis) - test raw calculation
            yaw = np.degrees(np.arctan2(robot_to_target[1], robot_to_target[0])) + 0
            
            # Pitch angle using myCobot convention: 180° = straight down, 90° = horizontal
            horizontal_distance = np.sqrt(robot_to_target[0]**2 + robot_to_target[1]**2)
            pitch_angle = np.degrees(np.arctan2(-robot_to_target[2], horizontal_distance))
            
            # For myCobot: pitch = 90° + angle_down_from_horizontal
            # This gives ~150°-180° for typical downward viewing angles
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
            
            # Roll typically stays at 0 for this application
            roll = 0
            
            calculated_orientation = [roll, pitch, yaw]
            print(f"DEBUG: Calculated pointing orientation toward {target_position}: {[f'{x:.1f}' for x in calculated_orientation]}")
            pointing_pose = pointing_position.tolist() + calculated_orientation
        else:
            # Fallback to default orientation if target is too close
            pointing_pose = pointing_position.tolist() + pointing_orientation
        
        # Validate this pointing position
        is_safe, reason = self.validate_position_safety(pointing_pose)
        if is_safe:
            return pointing_pose
        
        # If still not safe, try alternative strategies
        # Strategy 1: Move closer to J1 while maintaining direction
        for scale_factor in [0.7, 0.6, 0.5, 0.4]:
            scaled_distance = max_safe_distance * scale_factor
            scaled_from_j1 = direction_normalized * scaled_distance
            scaled_position = np.array([scaled_from_j1[0], scaled_from_j1[1], scaled_from_j1[2] + J1_HEIGHT])
            scaled_position[2] = max(scaled_position[2], min_height)
            
            scaled_pose = scaled_position.tolist() + pointing_orientation
            is_safe, _ = self.validate_position_safety(scaled_pose)
            if is_safe:
                return scaled_pose
        
        # Strategy 2: Point horizontally toward object from J1 level
        horizontal_from_j1 = np.array([target_x, target_y, 0])  # Horizontal at J1 level
        horizontal_distance = np.linalg.norm(horizontal_from_j1)
        
        if horizontal_distance > 0:
            horizontal_normalized = horizontal_from_j1 / horizontal_distance
            horizontal_safe_distance = min(max_safe_distance * 0.8, horizontal_distance * 0.9)
            horizontal_position = horizontal_normalized * horizontal_safe_distance
            horizontal_position = np.append(horizontal_position, J1_HEIGHT)  # Set Z to J1 height
            horizontal_position[2] = max(horizontal_position[2], min_height)
            
            horizontal_pose = horizontal_position.tolist() + pointing_orientation
            is_safe, _ = self.validate_position_safety(horizontal_pose)
            if is_safe:
                return horizontal_pose
        
        return None
    
    def validate_mycobot_workspace(self, position: List[float]) -> bool:
        """
        Additional myCobot-specific workspace validation.
        
        Args:
            position: Position to validate [x, y, z, rx, ry, rz]
            
        Returns:
            True if position is within myCobot workspace
        """
        if len(position) < 3:
            return False
        
        x, y, z = position[:3]
        
        # myCobot 280 specific workspace validation
        # 280mm effective radius is measured from J1 (base joint at 131.56mm height)
        J1_HEIGHT = 131.56  # Height of J1 above physical base
        
        # Calculate distance from J1 (not physical base)
        # For horizontal distance, use x,y directly
        horizontal_distance = (x**2 + y**2)**0.5
        
        # For vertical, calculate distance from J1 level
        vertical_distance_from_j1 = abs(z - J1_HEIGHT)
        
        # Check spherical workspace constraint from J1
        distance_from_j1 = (x**2 + y**2 + (z - J1_HEIGHT)**2)**0.5
        if distance_from_j1 > 280:  # 280mm reach from J1
            return False
        
        # Check minimum distance from J1 (avoid singularities)
        if distance_from_j1 < 50:
            return False
        
        # Check Z-axis constraints (avoid floor collision)
        if z < 50:  # Minimum height above physical base
            return False
        
        return True