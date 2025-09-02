#!/usr/bin/env python3
"""
Robot Coordinate Input System

Interactive robot arm control system where users can input coordinates manually.
Control your robot arm by entering coordinates via console input.

Usage:
    python robot_coordinate_input.py [--config CONFIG_NAME] [--no-camera]

Commands:
    Enter coordinates as: x,y,z,rx,ry,rz (e.g., 150,100,200,0,0,0)
    'q' - Quit application
    'h' - Show help
    'home' - Move to home position
    'table' - Move to look at table position
    'forward' - Move to look forward position
    'stop' - Stop all robot movement
    'current' - Show current robot position
"""

import sys
import os
import time
import cv2
import numpy as np
from typing import Optional, Dict, Any, List

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.sensors.realsense_manager import RealSenseManager
from module.robot.mycobot_controller import MyCobotController
from module.config.config_factory import ConfigFactory


class RobotCoordinateInput:
    """
    Interactive robot coordinate input system.
    
    Features:
    - Real-time camera visualization (optional)
    - Manual coordinate input via console
    - Robot position validation and safety
    - Async robot movement for smooth operation
    """
    
    def __init__(self, config_name: str = "realtime_tracking", use_camera: bool = True):
        """Initialize the robot coordinate input system."""
        self.config_name = config_name
        self.config = ConfigFactory.create_preset(config_name)
        self.use_camera = use_camera
        
        # Components
        self.sensor_manager = None
        self.robot_controller = None
        
        # Application state
        self.is_running = False
        self.camera_thread_active = False
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Robot state
        self.current_robot_position = None
        
    def initialize(self) -> bool:
        """Initialize camera and robot systems."""
        print("Initializing Robot Coordinate Input System")
        print("=" * 50)
        
        # Initialize camera (only if enabled)
        if self.use_camera:
            print("Initializing camera...")
            self.sensor_manager = RealSenseManager(None, realtime_mode=True, use_imu=True)
            if not self.sensor_manager.initialize():
                print("Error: Failed to initialize camera")
                return False
            
            if not self.sensor_manager.start():
                print("Error: Failed to start camera")
                return False
            print("[OK] Camera initialized successfully")
        else:
            print("Camera disabled (--no-camera flag)")
            self.sensor_manager = None
        
        # Initialize robot
        print("Initializing robot...")
        self.robot_controller = MyCobotController()
        
        if not self.robot_controller.connect():
            print("Warning: Failed to connect to robot. Camera-only mode.")
        else:
            print("[OK] Robot connected successfully")
            
            # Enable async mode for smooth movement
            if self.robot_controller.enable_async_mode(max_queue_size=5):
                print("[OK] Robot async mode enabled")
            else:
                print("Warning: Failed to enable async mode")
            
            # Get initial position
            self.current_robot_position = self.robot_controller.get_current_position()
            if self.current_robot_position:
                print(f"Initial robot position: {[f'{x:.1f}' for x in self.current_robot_position]}")
        
        print("[OK] Robot Coordinate Input System initialized")
        return True
    
    def run(self):
        """Main execution loop."""
        if not self.initialize():
            print("Failed to initialize system")
            return
        
        print("\nRobot Coordinate Input System Started!")
        print("=" * 50)
        self._print_help()
        
        self.is_running = True
        
        # Start camera display in background thread if enabled
        if self.use_camera and self.sensor_manager:
            import threading
            self.camera_thread_active = True
            camera_thread = threading.Thread(target=self._camera_display_loop, daemon=True)
            camera_thread.start()
        
        try:
            # Main input loop
            while self.is_running:
                try:
                    user_input = input("\nEnter command or coordinates (x,y,z,rx,ry,rz): ").strip()
                    
                    if not self._handle_user_input(user_input):
                        break
                        
                except KeyboardInterrupt:
                    print("\nShutting down...")
                    break
                except EOFError:
                    print("\nEOF received, shutting down...")
                    break
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _camera_display_loop(self):
        """Background thread for camera display."""
        if not self.sensor_manager:
            return
            
        cv2.namedWindow('Robot Coordinate Input - Camera View', cv2.WINDOW_AUTOSIZE)
        
        while self.camera_thread_active and self.is_running:
            try:
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                if color_image is not None:
                    # Create visualization
                    display_image = self._create_visualization(color_image)
                    
                    # Show the image
                    cv2.imshow('Robot Coordinate Input - Camera View', display_image)
                    
                    self.frame_count += 1
                    self._update_fps()
                
                # Check for 'q' key to quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nCamera window quit requested")
                    self.is_running = False
                    break
                    
            except Exception as e:
                print(f"Error in camera display: {e}")
                break
                
        cv2.destroyAllWindows()
        self.camera_thread_active = False
    
    def _create_visualization(self, color_image: np.ndarray) -> np.ndarray:
        """Create visualization with overlay information."""
        display_image = color_image.copy()
        
        # Add info overlay
        self._add_info_overlay(display_image)
        
        return display_image
    
    def _add_info_overlay(self, image: np.ndarray):
        """Add information overlay to the image."""
        # Get robot status
        robot_status = "Disconnected"
        if self.robot_controller and self.robot_controller.is_connected:
            if self.robot_controller.async_mode_enabled:
                queue_status = self.robot_controller.get_queue_status()
                active_cmd = queue_status.get('active_command', {})
                if active_cmd and active_cmd.get('command_id'):
                    robot_status = f"Moving ({active_cmd.get('type', 'unknown')})"
                else:
                    pending_count = queue_status.get('queue_size', 0)
                    if pending_count > 0:
                        robot_status = f"Ready ({pending_count} queued)"
                    else:
                        robot_status = "Ready"
            else:
                robot_status = "Connected"
        
        # Information lines
        info_lines = [
            "ROBOT COORDINATE INPUT",
            f"FPS: {self.current_fps:.1f}",
            f"Frames: {self.frame_count}",
            f"Robot: {robot_status}"
        ]
        
        # Draw background box
        text_height = len(info_lines) * 25 + 10
        cv2.rectangle(image, (5, 5), (350, text_height), (0, 0, 0), -1)
        cv2.rectangle(image, (5, 5), (350, text_height), (255, 255, 255), 1)
        
        # Draw text
        for i, line in enumerate(info_lines):
            color = (0, 255, 0) if i == 0 else (255, 255, 255)
            cv2.putText(image, line, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add robot position if available
        if self.current_robot_position:
            pos_text = f"Position: ({self.current_robot_position[0]:.0f}, {self.current_robot_position[1]:.0f}, {self.current_robot_position[2]:.0f})"
            cv2.putText(image, pos_text, (10, text_height + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Add instruction
        cv2.putText(image, "Press 'q' in this window to quit", (10, image.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    
    def _handle_user_input(self, user_input: str) -> bool:
        """
        Handle user input from console.
        
        Args:
            user_input: User input string
            
        Returns:
            True to continue, False to quit
        """
        if not user_input:
            return True
            
        user_input = user_input.lower()
        
        # Handle quit command
        if user_input == 'q':
            return False
        
        # Handle help command
        elif user_input == 'h':
            self._print_help()
        
        # Handle predefined positions
        elif user_input == 'home':
            self._execute_predefined_command("home", "Moving to home position...")
        
        elif user_input == 'table':
            self._execute_predefined_command("look_at_table", "Moving to look at table...")
        
        elif user_input == 'forward':
            self._execute_predefined_command("look_forward", "Moving to look forward...")
        
        elif user_input == 'stop':
            self._execute_predefined_command("stop", "Stopping all movement...")
        
        elif user_input == 'current':
            self._show_current_position()
        
        # Handle coordinate input
        else:
            self._handle_coordinate_input(user_input)
        
        return True
    
    def _execute_predefined_command(self, command: str, message: str):
        """Execute a predefined robot command."""
        if not self.robot_controller or not self.robot_controller.is_connected:
            print("Error: Robot not connected")
            return
            
        print(message)
        
        if self.robot_controller.async_mode_enabled:
            if command == "home":
                command_id = self.robot_controller.home_position_async(priority=1)
                print(f"Home command queued (ID: {command_id})")
            elif command == "look_at_table":
                command_id = self.robot_controller.look_at_table_async(priority=1)
                print(f"Look at table command queued (ID: {command_id})")
            elif command == "look_forward":
                command_id = self.robot_controller.look_forward_async(priority=1)
                print(f"Look forward command queued (ID: {command_id})")
            elif command == "stop":
                command_id = self.robot_controller.stop_movement_async(priority=3)
                print(f"Stop command queued (ID: {command_id})")
        else:
            if command == "home":
                self.robot_controller.home_position()
            elif command == "look_at_table":
                self.robot_controller.look_at_table()
            elif command == "look_forward":
                self.robot_controller.look_forward()
            elif command == "stop":
                self.robot_controller.stop_movement()
        
        # Update current position after movement
        time.sleep(0.5)  # Give time for command to be processed
        self.current_robot_position = self.robot_controller.get_current_position()
    
    def _handle_coordinate_input(self, coordinate_string: str):
        """Handle coordinate input from user."""
        if not self.robot_controller or not self.robot_controller.is_connected:
            print("Error: Robot not connected")
            return
        
        try:
            # Parse coordinates
            coords = [float(x.strip()) for x in coordinate_string.split(',')]
            
            if len(coords) != 6:
                print("Error: Please provide 6 coordinates: x,y,z,rx,ry,rz")
                print("Example: 150,100,200,0,0,0")
                return
            
            print(f"Parsed coordinates: {[f'{x:.1f}' for x in coords]}")
            
            # Validate position safety
            is_safe, reason = self.robot_controller.validate_position_safety(coords)
            if not is_safe:
                print(f"Warning: Unsafe position - {reason}")
                
                # Try to clamp to safe workspace
                clamped_coords, adjustments = self.robot_controller.clamp_position_to_workspace(coords)
                if adjustments:
                    print(f"Applied safety adjustments: {', '.join(adjustments)}")
                    print(f"Clamped coordinates: {[f'{x:.1f}' for x in clamped_coords]}")
                    coords = clamped_coords
                else:
                    print("Cannot make position safe, movement cancelled")
                    return
            
            # Execute movement
            print("Moving robot to specified coordinates...")
            
            if self.robot_controller.async_mode_enabled:
                from module.robot.command_queue import CommandType
                command_id = self.robot_controller.submit_async_command(
                    CommandType.MOVE_TO_POSITION, coords, speed=50, priority=2)
                if command_id:
                    print(f"Movement command queued (ID: {command_id})")
                else:
                    print("Failed to queue movement command")
            else:
                success = self.robot_controller.move_to_position(coords, speed=50)
                if success:
                    print("Movement command sent successfully")
                else:
                    print("Failed to send movement command")
            
            # Update current position
            time.sleep(0.5)  # Give time for command to be processed
            self.current_robot_position = self.robot_controller.get_current_position()
            
        except ValueError as e:
            print(f"Error parsing coordinates: {e}")
            print("Please enter coordinates as: x,y,z,rx,ry,rz")
            print("Example: 150,100,200,0,0,0")
        except Exception as e:
            print(f"Error executing movement: {e}")
    
    def _show_current_position(self):
        """Show current robot position."""
        if not self.robot_controller or not self.robot_controller.is_connected:
            print("Error: Robot not connected")
            return
        
        position = self.robot_controller.get_current_position()
        if position:
            print(f"Current robot position:")
            print(f"  X: {position[0]:6.1f} mm")
            print(f"  Y: {position[1]:6.1f} mm") 
            print(f"  Z: {position[2]:6.1f} mm")
            print(f"  RX: {position[3]:6.1f} deg")
            print(f"  RY: {position[4]:6.1f} deg")
            print(f"  RZ: {position[5]:6.1f} deg")
            self.current_robot_position = position
        else:
            print("Failed to get current position")
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _print_help(self):
        """Print help information."""
        print("\n" + "=" * 60)
        print("ROBOT COORDINATE INPUT SYSTEM - HELP")
        print("=" * 60)
        print("Commands:")
        print("  q                    - Quit application")
        print("  h                    - Show this help")
        print("  current              - Show current robot position")
        print("  home                 - Move to home position")
        print("  table                - Move to look at table position")
        print("  forward              - Move to look forward position")
        print("  stop                 - Stop all robot movement")
        print()
        print("Coordinate Input:")
        print("  Enter as: x,y,z,rx,ry,rz")
        print("  Example: 150,100,200,0,0,0")
        print("  Units: mm for position, degrees for rotation")
        print()
        print("Safety Features:")
        print("  - Position validation before movement")
        print("  - Automatic workspace clamping")
        print("  - Async movement commands (non-blocking)")
        print()
        if self.use_camera:
            print("Camera Window:")
            print("  - Press 'q' in camera window to quit")
            print("  - Real-time robot status display")
        print("=" * 60)
    
    def _cleanup(self):
        """Cleanup system resources."""
        print("\nShutting down Robot Coordinate Input System...")
        
        # Stop camera thread
        self.camera_thread_active = False
        time.sleep(0.1)
        
        if self.sensor_manager:
            self.sensor_manager.stop()
        
        if self.robot_controller:
            if self.robot_controller.async_mode_enabled:
                self.robot_controller.cleanup_async_system()
            self.robot_controller.disconnect()
        
        cv2.destroyAllWindows()
        print("[OK] Shutdown complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robot Coordinate Input System')
    parser.add_argument('--config', default='realtime_tracking',
                       help='Configuration preset name')
    parser.add_argument('--no-camera', action='store_true',
                       help='Disable camera initialization and use robot-only mode')
    
    args = parser.parse_args()
    
    print("Robot Coordinate Input System")
    print("=" * 40)
    print("Interactive robot arm control via coordinate input")
    if args.no_camera:
        print("Camera disabled (--no-camera)")
    print("=" * 40)
    
    try:
        system = RobotCoordinateInput(config_name=args.config, use_camera=not args.no_camera)
        system.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()