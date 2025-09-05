"""
Non-blocking wait utilities with live visualization.

This module provides wait functions that maintain live camera visualization
during delays, eliminating the duplicated wait implementations across applications.
"""

import time
import cv2
from typing import Optional, Dict, Any, Callable
from ..sensors.sensor_manager import SensorManager
from ..robot.robot_controller import RobotController


def wait_with_live_visualization(sensor_manager: SensorManager,
                                duration: float,
                                message: str = "WAITING",
                                window_name: str = "Application",
                                progress_info: Optional[Dict[str, Any]] = None,
                                interrupt_key: str = 'q',
                                progress_callback: Optional[Callable] = None) -> bool:
    """
    Wait for specified duration while maintaining live camera visualization.
    
    This replaces the various _wait_for_stabilization and _wait_with_visualization
    methods that were duplicated across applications.
    
    Args:
        sensor_manager: Sensor manager for getting camera frames
        duration: Wait duration in seconds
        message: Status message to display
        window_name: OpenCV window name for display
        progress_info: Optional progress information to display
        interrupt_key: Key to interrupt the wait (default: 'q')
        progress_callback: Optional callback function called each frame
        
    Returns:
        True if wait completed, False if interrupted
    """
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            # Get frame and maintain live visualization during wait
            vertices, color_image, metadata = sensor_manager.get_frame()
            
            if color_image is not None:
                display_image = color_image.copy()
                
                # Add main status message
                cv2.putText(display_image, message.upper(), (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Add remaining time
                remaining_time = duration - (time.time() - start_time)
                if progress_info and progress_info.get('remaining_time', False):
                    cv2.putText(display_image, f"Remaining: {remaining_time:.1f}s", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Add custom progress info
                if progress_info:
                    y_offset = 110
                    
                    if 'position' in progress_info:
                        cv2.putText(display_image, f"Position: {progress_info['position']}", (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        y_offset += 40
                    
                    if 'status' in progress_info:
                        cv2.putText(display_image, progress_info['status'], (10, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_offset += 30
                    
                    if 'custom_lines' in progress_info:
                        for line in progress_info['custom_lines']:
                            cv2.putText(display_image, line, (10, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            y_offset += 25
                
                # Call progress callback if provided
                if progress_callback:
                    try:
                        progress_callback(remaining_time, display_image)
                    except Exception as e:
                        print(f"Progress callback error: {e}")
                
                cv2.imshow(window_name, display_image)
            
            # Check for user interrupt
            key = cv2.waitKey(30) & 0xFF
            if key == ord(interrupt_key):
                print(f"Wait interrupted by user (pressed '{interrupt_key}')")
                return False
                
        except Exception as e:
            print(f"Visualization error during wait: {e}")
            cv2.waitKey(30)  # Keep responsive
    
    return True


def wait_for_command_completion(robot_controller: RobotController,
                               command_id: int,
                               sensor_manager: SensorManager,
                               timeout: float = 30.0,
                               window_name: str = "Robot Operation",
                               interrupt_key: str = 'q') -> bool:
    """
    Wait for robot command completion with live visualization.
    
    Args:
        robot_controller: Robot controller instance
        command_id: Command ID to wait for
        sensor_manager: Sensor manager for live visualization
        timeout: Maximum wait time in seconds
        window_name: OpenCV window name
        interrupt_key: Key to interrupt wait
        
    Returns:
        True if command completed successfully, False if timeout or interrupted
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check command status
            command_status = robot_controller.get_command_status(command_id)
            if command_status is None:
                print(f"Command {command_id} not found")
                return False
            
            # Check if command completed
            if command_status.name in ['COMPLETED', 'FAILED', 'CANCELLED']:
                success = command_status.name == 'COMPLETED'
                print(f"Command {command_id} {command_status.name.lower()}")
                return success
            
            # Get frame and maintain live visualization
            vertices, color_image, metadata = sensor_manager.get_frame()
            
            if color_image is not None:
                display_image = color_image.copy()
                
                # Add status information
                cv2.putText(display_image, "ROBOT MOVING", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                elapsed_time = time.time() - start_time
                cv2.putText(display_image, f"Elapsed: {elapsed_time:.1f}s", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.putText(display_image, f"Command: {command_id} ({command_status.name})", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow(window_name, display_image)
            
            # Check for user interrupt
            key = cv2.waitKey(100) & 0xFF  # Longer delay for robot commands
            if key == ord(interrupt_key):
                print(f"Command wait interrupted by user")
                # Try to stop the command
                robot_controller.stop_movement_async(priority=3)
                return False
                
        except Exception as e:
            print(f"Error waiting for command completion: {e}")
            cv2.waitKey(30)
    
    print(f"Timeout waiting for command {command_id} (waited {timeout}s)")
    return False


def wait_for_robot_stabilization(sensor_manager: SensorManager,
                                duration: float = 3.0,
                                window_name: str = "Robot Stabilization") -> bool:
    """
    Wait for robot to stabilize with standardized message.
    
    This is a convenience wrapper for the common robot stabilization wait.
    
    Args:
        sensor_manager: Sensor manager for live visualization
        duration: Stabilization time in seconds
        window_name: OpenCV window name
        
    Returns:
        True if stabilization completed, False if interrupted
    """
    return wait_with_live_visualization(
        sensor_manager=sensor_manager,
        duration=duration,
        message="ROBOT STABILIZING",
        window_name=window_name,
        progress_info={"remaining_time": True}
    )


def wait_for_user_action(sensor_manager: SensorManager,
                        message: str = "PRESS SPACE TO CONTINUE",
                        window_name: str = "User Action Required",
                        action_key: str = ' ',
                        cancel_key: str = 'q') -> bool:
    """
    Wait for user action with live visualization.
    
    Args:
        sensor_manager: Sensor manager for live visualization
        message: Message to display to user
        window_name: OpenCV window name
        action_key: Key to continue (default: space)
        cancel_key: Key to cancel (default: 'q')
        
    Returns:
        True if action taken, False if cancelled
    """
    print(f"Waiting for user action: {message}")
    
    while True:
        try:
            # Get frame and maintain live visualization
            vertices, color_image, metadata = sensor_manager.get_frame()
            
            if color_image is not None:
                display_image = color_image.copy()
                
                # Add instruction message
                cv2.putText(display_image, message, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Add key instructions
                cv2.putText(display_image, f"'{action_key}' - Continue  |  '{cancel_key}' - Cancel", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow(window_name, display_image)
            
            # Check for user input
            key = cv2.waitKey(30) & 0xFF
            if key == ord(action_key):
                print("User action confirmed")
                return True
            elif key == ord(cancel_key):
                print("User cancelled action")
                return False
                
        except Exception as e:
            print(f"Error during user action wait: {e}")
            cv2.waitKey(30)


def wait_with_countdown(sensor_manager: SensorManager,
                       duration: float,
                       message: str = "STARTING IN",
                       window_name: str = "Countdown",
                       interrupt_key: str = 'q') -> bool:
    """
    Wait with a visible countdown timer.
    
    Args:
        sensor_manager: Sensor manager for live visualization
        duration: Countdown duration in seconds
        message: Countdown message
        window_name: OpenCV window name
        interrupt_key: Key to interrupt countdown
        
    Returns:
        True if countdown completed, False if interrupted
    """
    def countdown_callback(remaining_time: float, display_image):
        # Add large countdown number
        countdown_num = int(remaining_time) + 1
        cv2.putText(display_image, str(countdown_num), (display_image.shape[1]//2 - 50, display_image.shape[0]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 8)
    
    return wait_with_live_visualization(
        sensor_manager=sensor_manager,
        duration=duration,
        message=message,
        window_name=window_name,
        progress_info={"remaining_time": False},  # Don't show text countdown
        interrupt_key=interrupt_key,
        progress_callback=countdown_callback
    )