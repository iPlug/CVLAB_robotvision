"""
Robot operation utilities for standardized setup and movement.

This module provides common utilities for robot initialization, movement,
and coordination with sensors, eliminating duplicated robot handling code
across applications.
"""

from typing import Dict, Any, Optional
from ..robot.robot_controller import RobotController
from ..sensors.sensor_manager import SensorManager
from .wait_utils import wait_with_live_visualization


def initialize_robot_for_application(robot_controller: RobotController,
                                    safety_params: Dict[str, Any],
                                    enable_async: bool = True,
                                    max_queue_size: int = 5) -> bool:
    """
    Initialize robot controller with standard application settings.
    
    Args:
        robot_controller: Robot controller instance
        safety_params: Safety parameters to apply
        enable_async: Whether to enable async command mode
        max_queue_size: Maximum commands in async queue
        
    Returns:
        True if initialization successful
    """
    if not robot_controller.is_connected:
        print("Warning: Robot not connected. Skipping robot initialization.")
        return False
    
    # Apply safety parameters
    robot_controller.safety_constraints.update(safety_params)
    print(f"Applied robot safety parameters: {safety_params}")
    
    # Enable async mode for non-blocking commands
    if enable_async:
        if robot_controller.enable_async_mode(max_queue_size=max_queue_size):
            print("[OK] Robot async mode enabled for smooth operation")
            return True
        else:
            print("[WARNING] Failed to enable robot async mode - commands may block visualization")
            return False
    
    return True


def move_to_standard_pose(robot_controller: RobotController, 
                         pose_type: str = "table",
                         async_mode: bool = True,
                         priority: int = 1) -> Optional[int]:
    """
    Move robot to a standard pose.
    
    Args:
        robot_controller: Robot controller instance
        pose_type: Type of pose ("table", "home", "observe")
        async_mode: Whether to use async movement
        priority: Command priority for async mode
        
    Returns:
        Command ID if async mode, True/False if sync mode, None if failed
    """
    if not robot_controller.is_connected:
        print("Robot not connected - cannot move to pose")
        return None
    
    print(f"Moving robot to {pose_type} pose...")
    
    if pose_type.lower() in ["table", "observe", "observation"]:
        if async_mode and robot_controller.async_mode_enabled:
            command_id = robot_controller.look_at_table_async(priority=priority)
            if command_id:
                print(f"[OK] {pose_type.title()} pose command submitted (ID: {command_id})")
                return command_id
            else:
                print(f"[ERROR] Failed to submit {pose_type} pose command")
                return None
        else:
            success = robot_controller.look_at_table()
            if success:
                print(f"[OK] Robot moved to {pose_type} pose")
                return True
            else:
                print(f"[ERROR] Failed to move to {pose_type} pose")
                return False
    
    elif pose_type.lower() == "home":
        if async_mode and robot_controller.async_mode_enabled:
            command_id = robot_controller.home_position_async(priority=priority)
            if command_id:
                print(f"[OK] Home position command submitted (ID: {command_id})")
                return command_id
            else:
                print(f"[ERROR] Failed to submit home position command")
                return None
        else:
            success = robot_controller.home_position()
            if success:
                print("[OK] Robot moved to home position")
                return True
            else:
                print("[ERROR] Failed to move to home position")
                return False
    
    else:
        print(f"Unknown pose type: {pose_type}")
        return None


def setup_robot_with_stabilization(robot_controller: RobotController,
                                  sensor_manager: SensorManager,
                                  pose_type: str = "table",
                                  stabilization_time: float = 3.0,
                                  window_name: str = "Robot Setup",
                                  home_first: bool = True) -> bool:
    """
    Complete robot setup sequence with safe initialization and stabilization.
    
    This function encapsulates the common pattern used across applications:
    1. (Optional) Move robot to home position [0,0,0,0,0,0] first for safety
    2. Move robot to target pose (observation/table pose)
    3. Wait for robot to stabilize
    4. Ready for sensor calibration (e.g., gravity alignment)
    
    Args:
        robot_controller: Robot controller instance
        sensor_manager: Sensor manager for live visualization during wait
        pose_type: Target pose to move to ("table", "home", "observe")
        stabilization_time: Time to wait for stabilization in seconds
        window_name: Window name for visualization during wait
        home_first: Whether to move to home position first for safety (default: True)
        
    Returns:
        True if setup completed successfully
    """
    if not robot_controller.is_connected:
        print("Robot not connected - skipping robot setup")
        return False
    
    # Step 1: Move to home position first for safety (if requested)
    if home_first and pose_type.lower() != "home":
        print("Step 1: Moving to home position for safety...")
        home_result = move_to_standard_pose(robot_controller, "home", async_mode=True)
        if home_result is None:
            print("Failed to move to home position")
            return False
        
        # Brief pause to allow home movement to complete
        print("Waiting for home position movement to complete...")
        wait_with_live_visualization(
            sensor_manager=sensor_manager,
            duration=1.5,
            message="MOVING TO HOME POSITION",
            window_name=window_name,
            progress_info={"remaining_time": True}
        )
    
    # Step 2: Move to target pose
    if pose_type.lower() != "home" or not home_first:
        step_num = 2 if home_first and pose_type.lower() != "home" else 1
        print(f"Step {step_num}: Moving to {pose_type} pose...")
        result = move_to_standard_pose(robot_controller, pose_type, async_mode=True)
        if result is None:
            print(f"Failed to initiate robot movement to {pose_type}")
            return False
    
    # Step 3: Wait for robot to stabilize with live visualization
    final_step = 3 if home_first and pose_type.lower() != "home" else 2
    print(f"Step {final_step}: Waiting for robot to stabilize in {pose_type} pose...")
    wait_with_live_visualization(
        sensor_manager=sensor_manager,
        duration=stabilization_time,
        message="ROBOT STABILIZING",
        window_name=window_name,
        progress_info={"remaining_time": True}
    )
    
    print(f"[OK] Robot safely initialized and stabilized in {pose_type} pose")
    return True


def point_at_object_safe(robot_controller: RobotController,
                        camera_position: list,
                        async_mode: bool = True,
                        priority: int = 2,
                        timeout: float = 15.0) -> Optional[int]:
    """
    Safely point robot at an object with error handling.
    
    Args:
        robot_controller: Robot controller instance
        camera_position: Object position in camera coordinates [x, y, z]
        async_mode: Whether to use async command
        priority: Command priority
        timeout: Command timeout in seconds
        
    Returns:
        Command ID if async, success boolean if sync, None if failed
    """
    if not robot_controller.is_connected:
        print("Robot not connected - cannot point at object")
        return None
    
    if not camera_position or len(camera_position) < 3:
        print("Invalid camera position - cannot point at object")
        return None
    
    print(f"Pointing at object: ({camera_position[0]:.0f}, {camera_position[1]:.0f}, {camera_position[2]:.0f}) mm")
    
    try:
        if async_mode and robot_controller.async_mode_enabled:
            command_id = robot_controller.point_at_object_async(
                camera_position,
                priority=priority,
                timeout=timeout
            )
            
            if command_id:
                print(f"[OK] Pointing command submitted (ID: {command_id})")
                return command_id
            else:
                print("[ERROR] Failed to submit pointing command")
                return None
        else:
            success = robot_controller.point_at_object(camera_position)
            if success:
                print("[OK] Robot pointing at object")
                return True
            else:
                print("[ERROR] Failed to point at object")
                return False
                
    except Exception as e:
        print(f"Error during pointing command: {e}")
        return None


def stop_robot_safely(robot_controller: RobotController,
                     async_mode: bool = True,
                     priority: int = 3) -> bool:
    """
    Safely stop robot movement with high priority.
    
    Args:
        robot_controller: Robot controller instance
        async_mode: Whether to use async command
        priority: Command priority (default: high)
        
    Returns:
        True if stop command submitted successfully
    """
    if not robot_controller.is_connected:
        print("Robot not connected")
        return False
    
    print("Stopping robot movement...")
    
    try:
        if async_mode and robot_controller.async_mode_enabled:
            command_id = robot_controller.stop_movement_async(priority=priority)
            if command_id:
                print(f"[OK] Stop command submitted (ID: {command_id})")
                return True
            else:
                print("[ERROR] Failed to submit stop command")
                return False
        else:
            success = robot_controller.stop_movement()
            if success:
                print("[OK] Robot movement stopped")
                return True
            else:
                print("[ERROR] Failed to stop robot movement")
                return False
                
    except Exception as e:
        print(f"Error stopping robot: {e}")
        return False


def cleanup_robot_safely(robot_controller: RobotController,
                        return_to_pose: str = "table",
                        timeout: float = 10.0) -> bool:
    """
    Safely cleanup robot controller with return to safe pose.
    
    Args:
        robot_controller: Robot controller instance
        return_to_pose: Pose to return to before cleanup ("table", "home")
        timeout: Timeout for cleanup operations
        
    Returns:
        True if cleanup completed successfully
    """
    if not robot_controller or not robot_controller.is_connected:
        return True
    
    print(f"Cleaning up robot - returning to {return_to_pose} pose...")
    
    try:
        # Return to safe pose before shutdown
        move_to_standard_pose(robot_controller, return_to_pose, async_mode=False)
        
        # Cleanup async system
        robot_controller.cleanup_async_system()
        
        # Disconnect from robot
        robot_controller.disconnect()
        
        print("[OK] Robot cleanup completed")
        return True
        
    except Exception as e:
        print(f"Error during robot cleanup: {e}")
        return False


def get_robot_status_info(robot_controller: RobotController) -> Dict[str, str]:
    """
    Get human-readable robot status information.
    
    Args:
        robot_controller: Robot controller instance
        
    Returns:
        Dictionary with status information for display
    """
    if not robot_controller:
        return {"connection": "No Controller", "mode": "N/A"}
    
    status = {
        "connection": "Connected" if robot_controller.is_connected else "Disconnected",
        "async_mode": "Enabled" if robot_controller.async_mode_enabled else "Disabled"
    }
    
    # Add queue status if async mode is enabled
    if robot_controller.async_mode_enabled and robot_controller.command_queue:
        queue_status = robot_controller.get_queue_status()
        status["queue"] = f"{queue_status.get('pending_commands', 0)} pending"
        
        if queue_status.get('current_command'):
            status["current"] = queue_status['current_command'].get('type', 'Unknown')
        else:
            status["current"] = "Idle"
    
    return status