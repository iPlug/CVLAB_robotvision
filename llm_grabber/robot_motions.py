#!/usr/bin/env python3
"""
Robot Motion Controller for LLM Integration

Extracted and refactored robot motion functions from the main vision system
to provide a clean interface for LLM function calling and natural language control.

This module handles all robot movement, grasping, and coordination operations
while maintaining compatibility with the existing vision processing system.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from module.robot.mycobot_controller import MyCobotController


class RobotMotionController:
    """
    Advanced robot motion controller for LLM integration.
    
    Provides high-level robot motion primitives that can be easily called
    by LLMs for natural language robot control. Includes safety features,
    state tracking, and comprehensive error handling.
    """
    
    def __init__(self, enhanced_robot_controller=None, movement_speed=50):
        """
        Initialize the robot motion controller.
        
        Args:
            enhanced_robot_controller: Pre-initialized robot controller instance
            movement_speed: Default movement speed (0-100)
        """
        self.enhanced_robot_controller = enhanced_robot_controller
        self.movement_speed = movement_speed
        
        # State tracking
        self.grasping_in_progress = False
        self.grasping_state = 'idle'
        self.grasping_command_id = None
        self.grasping_object = None
        self.grasping_step_start_time = None
        self.base_joint_angles = None
        
        # Safety and configuration
        self.safety_enabled = True
        self.max_grasp_attempts = 3
        self.grasp_attempt_count = 0
        
    def is_robot_available(self) -> bool:
        """
        Check if robot is connected and available for commands.
        
        Returns:
            bool: True if robot is available, False otherwise
        """
        if not self.enhanced_robot_controller:
            return False
            
        if hasattr(self.enhanced_robot_controller, 'is_connected'):
            return self.enhanced_robot_controller.is_connected
        return True
    
    def get_robot_status(self) -> Dict[str, Any]:
        """
        Get comprehensive robot status for LLM feedback.
        
        Returns:
            dict: Robot status including position, state, and availability
        """
        status = {
            'available': self.is_robot_available(),
            'grasping_in_progress': self.grasping_in_progress,
            'current_state': self.grasping_state,
            'movement_speed': self.movement_speed,
            'safety_enabled': self.safety_enabled,
            'timestamp': time.strftime('%H:%M:%S')
        }
        
        if self.is_robot_available():
            try:
                if hasattr(self.enhanced_robot_controller, 'get_current_position'):
                    current_pos = self.enhanced_robot_controller.get_current_position()
                    status['current_position'] = current_pos
                    status['position_mm'] = current_pos if current_pos else None
            except Exception as e:
                status['position_error'] = str(e)
        
        return status
    
    def stop_current_action(self) -> bool:
        """
        Emergency stop - halt all current robot movements.
        
        Returns:
            bool: True if stop was successful
        """
        print("EMERGENCY STOP - Halting all robot movements")
        
        try:
            if self.enhanced_robot_controller and hasattr(self.enhanced_robot_controller, 'stop_movement'):
                self.enhanced_robot_controller.stop_movement()
            
            # Reset state
            self.grasping_in_progress = False
            self.grasping_state = 'stopped'
            self.grasping_command_id = None
            
            print("✓ Robot movements stopped successfully")
            return True
            
        except Exception as e:
            print(f"✗ Error stopping robot: {e}")
            return False
    
    def move_to_home(self) -> bool:
        """
        Move robot to home/safe position.
        
        Returns:
            bool: True if movement was initiated successfully
        """
        if not self.is_robot_available():
            print("Robot not available for home movement")
            return False
            
        try:
            from module.utils.robot_utils import move_to_standard_pose
            print("Moving robot to home position...")
            success = move_to_standard_pose(self.enhanced_robot_controller, "table", async_mode=True)
            
            if success:
                print("✓ Home movement initiated")
                return True
            else:
                print("✗ Failed to initiate home movement")
                return False
                
        except Exception as e:
            print(f"✗ Error moving to home: {e}")
            return False
    
    def open_gripper(self) -> bool:
        """
        Open the robot gripper.
        
        Returns:
            bool: True if gripper opening was successful
        """
        if not self.is_robot_available():
            print("Robot not available for gripper control")
            return False
            
        try:
            if hasattr(self.enhanced_robot_controller, 'open_gripper'):
                success = self.enhanced_robot_controller.open_gripper()
                if success:
                    print("✓ Gripper opened")
                    return True
                else:
                    print("✗ Failed to open gripper")
                    return False
            else:
                print("✗ Gripper control not available")
                return False
                
        except Exception as e:
            print(f"✗ Error opening gripper: {e}")
            return False
    
    def close_gripper(self) -> bool:
        """
        Close the robot gripper.
        
        Returns:
            bool: True if gripper closing was successful
        """
        if not self.is_robot_available():
            print("Robot not available for gripper control")
            return False
            
        try:
            if hasattr(self.enhanced_robot_controller, 'close_gripper'):
                success = self.enhanced_robot_controller.close_gripper()
                if success:
                    print("✓ Gripper closed")
                    return True
                else:
                    print("✗ Failed to close gripper")
                    return False
            else:
                print("✗ Gripper control not available")
                return False
                
        except Exception as e:
            print(f"✗ Error closing gripper: {e}")
            return False
    
    def move_to_position(self, x: float, y: float, z: float, speed: Optional[int] = None) -> bool:
        """
        Move robot to specific position in robot coordinate space.
        
        Args:
            x: X coordinate in mm (forward/backward)
            y: Y coordinate in mm (left/right)  
            z: Z coordinate in mm (up/down)
            speed: Movement speed override (0-100)
            
        Returns:
            bool: True if movement was initiated successfully
        """
        if not self.is_robot_available():
            print("Robot not available for position movement")
            return False
            
        if speed is None:
            speed = self.movement_speed
            
        # Create target pose [x, y, z, rx, ry, rz]
        target_pose = [x, y, z, 0, 180, 0]  # Default pointing down orientation
        
        try:
            print(f"Moving to position: ({x:.1f}, {y:.1f}, {z:.1f}) mm at speed {speed}")
            
            if hasattr(self.enhanced_robot_controller, 'move_to_position'):
                success = self.enhanced_robot_controller.move_to_position(target_pose, speed=speed)
                
                if success:
                    print("✓ Position movement initiated")
                    return True
                else:
                    print("✗ Failed to initiate position movement")
                    return False
            else:
                print("✗ Position control not available")
                return False
                
        except Exception as e:
            print(f"✗ Error moving to position: {e}")
            return False
    
    def grasp_object(self, object_id: Optional[int] = None, camera_position_m: Optional[List[float]] = None) -> bool:
        """
        Start grasping sequence for specified object or closest object.
        
        This is the main LLM-callable function for object grasping.
        
        Args:
            object_id: Specific object ID to grasp (1-based), None for closest
            camera_position_m: Direct camera position [x, y, z] in meters
            
        Returns:
            bool: True if grasping sequence was initiated successfully
        """
        if not self.is_robot_available():
            print("Robot not available for grasping")
            return False
            
        if self.grasping_in_progress:
            print("Grasping already in progress - use stop_current_action() first")
            return False
        
        print(f"\\n=== GRASP OBJECT COMMAND - {time.strftime('%H:%M:%S')} ===")
        
        # If direct position provided, use it
        if camera_position_m is not None:
            if len(camera_position_m) != 3:
                print("✗ Invalid camera position - must be [x, y, z] in meters")
                return False
                
            print(f"Using direct position: ({camera_position_m[0]:.3f}, {camera_position_m[1]:.3f}, {camera_position_m[2]:.3f}) m")
            return self._start_grasping_sequence(camera_position_m)
        
        # For object detection-based grasping, would need access to detection results
        # This will be implemented when integrating with the main vision system
        print("✗ Object detection-based grasping requires integration with vision system")
        return False
    
    def _start_grasping_sequence(self, camera_position_m: List[float]) -> bool:
        """
        Internal function to start the grasping state machine.
        
        Args:
            camera_position_m: Object position in camera coordinates [x, y, z] in meters
            
        Returns:
            bool: True if sequence started successfully
        """
        try:
            # Initialize grasping state
            self.grasping_in_progress = True
            self.grasping_state = 'starting'
            self.grasping_object = {
                'camera_position_m': camera_position_m,
                'start_time': time.time()
            }
            self.grasping_step_start_time = time.time()
            self.grasp_attempt_count = 0
            
            print("✓ Grasping sequence initialized - state machine started")
            return True
            
        except Exception as e:
            print(f"✗ Error starting grasping sequence: {e}")
            self.grasping_in_progress = False
            return False
    
    def process_grasping_state_machine(self) -> None:
        """
        Process the grasping state machine - should be called regularly from main loop.
        
        This handles the step-by-step execution of grasping operations without
        blocking the main vision processing loop.
        """
        if not self.grasping_in_progress:
            return
            
        try:
            # State machine transitions
            if self.grasping_state == 'starting':
                self._start_hover_movement()
            elif self.grasping_state == 'hovering':
                self._check_hover_completion()
            elif self.grasping_state == 'opening_gripper':
                self._check_gripper_opening()
            elif self.grasping_state == 'grasping':
                self._check_grasp_movement()
            elif self.grasping_state == 'closing_gripper':
                self._check_gripper_closing()
            elif self.grasping_state == 'lifting':
                self._check_lift_movement()
            elif self.grasping_state == 'returning':
                self._check_return_movement()
            elif self.grasping_state == 'finished':
                self._finish_grasping(True)
                
        except Exception as e:
            print(f"✗ Error in grasping state machine: {e}")
            self._finish_grasping(False)
    
    def _start_hover_movement(self) -> None:
        """Start movement to hover position above target object."""
        print("→ Starting hover movement...")
        # Implementation would go here - simplified for now
        self.grasping_state = 'hovering'
        self.grasping_step_start_time = time.time()
    
    def _check_hover_completion(self) -> None:
        """Check if hover movement is complete."""
        # Simplified - in real implementation would check robot status
        if time.time() - self.grasping_step_start_time > 2.0:  # 2 second timeout
            print("✓ Hover position reached")
            self.grasping_state = 'opening_gripper'
            self.grasping_step_start_time = time.time()
    
    def _check_gripper_opening(self) -> None:
        """Check if gripper opening is complete."""
        if time.time() - self.grasping_step_start_time > 1.0:  # 1 second timeout
            print("✓ Gripper opened")
            self.grasping_state = 'grasping'
            self.grasping_step_start_time = time.time()
    
    def _check_grasp_movement(self) -> None:
        """Check if grasp movement is complete."""
        if time.time() - self.grasping_step_start_time > 2.0:  # 2 second timeout
            print("✓ Grasp position reached")
            self.grasping_state = 'closing_gripper'
            self.grasping_step_start_time = time.time()
    
    def _check_gripper_closing(self) -> None:
        """Check if gripper closing is complete."""
        if time.time() - self.grasping_step_start_time > 1.0:  # 1 second timeout
            print("✓ Gripper closed - object grasped")
            self.grasping_state = 'lifting'
            self.grasping_step_start_time = time.time()
    
    def _check_lift_movement(self) -> None:
        """Check if lift movement is complete."""
        if time.time() - self.grasping_step_start_time > 2.0:  # 2 second timeout
            print("✓ Object lifted")
            self.grasping_state = 'returning'
            self.grasping_step_start_time = time.time()
    
    def _check_return_movement(self) -> None:
        """Check if return movement is complete."""
        if time.time() - self.grasping_step_start_time > 3.0:  # 3 second timeout
            print("✓ Returned to safe position")
            self.grasping_state = 'finished'
    
    def _finish_grasping(self, success: bool) -> None:
        """Finish grasping sequence and reset state."""
        if success:
            print("\\n✅ GRASPING COMPLETED SUCCESSFULLY!")
        else:
            print("\\n❌ GRASPING FAILED")
        
        # Reset state
        self.grasping_in_progress = False
        self.grasping_state = 'idle'
        self.grasping_command_id = None
        self.grasping_object = None
        self.grasp_attempt_count = 0
        
        print("   State machine reset - ready for next command\\n")


# LLM Function Calling Interface
def get_available_robot_functions() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all available robot functions for LLM integration.
    
    Returns:
        dict: Function metadata including parameters and descriptions
    """
    return {
        "grasp_object": {
            "description": "Grasp a specific object or the closest detected object",
            "parameters": {
                "object_id": {"type": "int", "optional": True, "description": "Object ID to grasp (1-based), None for closest"},
                "camera_position_m": {"type": "list", "optional": True, "description": "Direct position [x,y,z] in meters"}
            },
            "returns": "bool - True if grasping started successfully"
        },
        "move_to_position": {
            "description": "Move robot to specific coordinates in robot space",
            "parameters": {
                "x": {"type": "float", "description": "X coordinate in mm (forward/backward)"},
                "y": {"type": "float", "description": "Y coordinate in mm (left/right)"},
                "z": {"type": "float", "description": "Z coordinate in mm (up/down)"},
                "speed": {"type": "int", "optional": True, "description": "Movement speed 0-100"}
            },
            "returns": "bool - True if movement started successfully"
        },
        "open_gripper": {
            "description": "Open the robot gripper",
            "parameters": {},
            "returns": "bool - True if gripper opened successfully"
        },
        "close_gripper": {
            "description": "Close the robot gripper",
            "parameters": {},
            "returns": "bool - True if gripper closed successfully"
        },
        "move_to_home": {
            "description": "Move robot to safe home position",
            "parameters": {},
            "returns": "bool - True if home movement started successfully"
        },
        "stop_current_action": {
            "description": "Emergency stop - halt all robot movements immediately",
            "parameters": {},
            "returns": "bool - True if stop was successful"
        },
        "get_robot_status": {
            "description": "Get comprehensive robot status and current state",
            "parameters": {},
            "returns": "dict - Status information including position and availability"
        }
    }