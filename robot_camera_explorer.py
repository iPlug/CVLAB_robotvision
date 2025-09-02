#!/usr/bin/env python3
"""
Robot Camera Explorer

Interactive robot arm camera control system with non-blocking keyboard input.
Control your robot arm to explore and visualize the surroundings through the camera.

Usage:
    python robot_camera_explorer.py [--config CONFIG_NAME] [--no-camera]

Keyboard Controls:
    'q' - Quit application
    'h' - Toggle help display
    'space' - Pause/Resume camera feed
    
    Robot Movement:
    'l' - Look at table (predefined observation position)
    'f' - Look forward (horizontal view)
    'x' - Stop all robot movement
    'r' - Return to home position
    
    Manual Control:
    'a'/'d' - Joint5 turn left/right
    'w'/'s' - Joint4 tilt up/down
    'z'/'c' - Joint1 base rotate left/right
    
    View Modes:
    '1' - Color camera only
    '2' - Point cloud view only
    '3' - Combined view (default)
    
    Speed Control:
    '+'/'=' - Increase movement speed
    '-'/'_' - Decrease movement speed
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


class RobotCameraExplorer:
    """
    Interactive robot camera exploration system.
    
    Features:
    - Real-time camera visualization
    - Non-blocking keyboard controls
    - Manual robot positioning
    - Multiple view modes
    - Async robot movement for smooth operation
    """
    
    def __init__(self, config_name: str = "realtime_tracking", use_camera: bool = True):
        """Initialize the robot camera explorer."""
        self.config_name = config_name
        self.config = ConfigFactory.create_preset(config_name)
        self.use_camera = use_camera
        self.camera_failed = False
        self.consecutive_frame_failures = 0
        self.max_frame_failures = 10  # Switch to black screen after 10 consecutive failures
        
        # Components
        self.sensor_manager = None
        self.robot_controller = None
        
        # Application state
        self.is_running = False
        self.is_paused = False
        self.show_help = False
        self.view_mode = 3  # 1=color, 2=point_cloud, 3=combined
        
        # Movement parameters
        self.movement_speed = 50  # Default speed (1-100)
        self.joint_step = 30  # degrees per keypress for joints
        
        # Joint angles for look_at_table pose (base reference)
        self.base_joint_angles = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Robot state
        self.current_robot_position = None
        self.last_movement_time = 0
        self.movement_cooldown = 0.1  # Minimum time between movements
        
    def initialize(self) -> bool:
        """Initialize camera and robot systems."""
        print("Initializing Robot Camera Explorer")
        print("=" * 40)
        
        # Initialize camera (only if enabled)
        if self.use_camera:
            print("Initializing camera...")
            self.sensor_manager = RealSenseManager(None, realtime_mode=True, use_imu=True)
            if not self.sensor_manager.initialize():
                print("Warning: Failed to initialize camera - continuing with black screen")
                self.sensor_manager = None
                self.camera_failed = True
            elif not self.sensor_manager.start():
                print("Warning: Failed to start camera - continuing with black screen")
                self.sensor_manager.stop()
                self.sensor_manager = None
                self.camera_failed = True
            else:
                print("Camera initialized successfully")
                self.camera_failed = False
        else:
            print("Camera disabled (--no-camera flag)")
            self.sensor_manager = None
            self.camera_failed = False
        
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
            
            # Move to look_at_table pose as starting position
            print("Moving to look_at_table pose (base reference)...")
            self.robot_controller.look_at_table()
            time.sleep(2)  # Wait for movement to complete
            
            # Get joint angles for base reference
            self.base_joint_angles = self.robot_controller.get_joint_angles()
            if self.base_joint_angles:
                print(f"Base joint angles: {[f'{x:.1f}' for x in self.base_joint_angles]}")
            
            # Get position
            self.current_robot_position = self.robot_controller.get_current_position()
            if self.current_robot_position:
                print(f"Initial robot position: {[f'{x:.1f}' for x in self.current_robot_position]}")
        
        print("[OK] Robot Camera Explorer initialized")
        return True
    
    def run(self):
        """Main exploration loop."""
        if not self.initialize():
            print("Failed to initialize system")
            return
        
        print("\nRobot Camera Explorer Started!")
        print("=" * 40)
        self._print_controls()
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Handle pause
                if self.is_paused:
                    cv2.waitKey(30)
                    continue
                
                # Get camera frame (only if camera enabled and not failed)
                if self.use_camera and self.sensor_manager and not self.camera_failed:
                    try:
                        vertices, color_image, metadata = self.sensor_manager.get_frame()
                        
                        if color_image is not None:
                            # Reset failure counter on successful frame
                            self.consecutive_frame_failures = 0
                            
                            # Create visualization based on view mode
                            display_image = self._create_visualization(vertices, color_image, metadata)
                            
                            # Show the image with consistent window properties
                            window_name = 'Robot Camera Explorer'
                            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                            cv2.imshow(window_name, display_image)
                            
                            self.frame_count += 1
                            self._update_fps()
                        else:
                            # No frame received - increment failure counter
                            self.consecutive_frame_failures += 1
                            if self.consecutive_frame_failures >= self.max_frame_failures:
                                print(f"\nWarning: Camera stopped providing frames after {self.max_frame_failures} attempts")
                                print("Switching to black screen mode (camera may be in use by another process)")
                                self.camera_failed = True
                                # Stop the camera to free resources
                                if self.sensor_manager:
                                    self.sensor_manager.stop()
                                    self.sensor_manager = None
                            
                            # Show black screen while trying
                            display_image = self._create_no_camera_display()
                            window_name = 'Robot Camera Explorer'
                            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                            cv2.imshow(window_name, display_image)
                            time.sleep(0.033)  # ~30 FPS
                    
                    except Exception as e:
                        # Camera error - switch to failed mode
                        print(f"\nCamera error: {e}")
                        print("Switching to black screen mode")
                        self.camera_failed = True
                        if self.sensor_manager:
                            self.sensor_manager.stop()
                            self.sensor_manager = None
                        
                        display_image = self._create_no_camera_display()
                        window_name = 'Robot Camera Explorer'
                        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                        cv2.imshow(window_name, display_image)
                        time.sleep(0.033)  # ~30 FPS
                else:
                    # No camera mode - create simple status display
                    display_image = self._create_no_camera_display()
                    window_name = 'Robot Camera Explorer'
                    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                    cv2.imshow(window_name, display_image)
                    time.sleep(0.033)  # ~30 FPS
                
                # Handle user input (non-blocking)
                if not self._handle_user_input():
                    break
                    
        except KeyboardInterrupt:
            print("\nExplorer interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _create_visualization(self, vertices: np.ndarray, color_image: np.ndarray, 
                            metadata: Dict[str, Any]) -> np.ndarray:
        """Create visualization based on current view mode."""
        
        if self.view_mode == 1:
            # Color camera only
            display_image = color_image.copy()
            
        elif self.view_mode == 2:
            # Point cloud view only
            display_image = self._create_point_cloud_view(vertices)
            if display_image is None:
                display_image = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(display_image, "No point cloud data", (200, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                           
        else:  # view_mode == 3
            # Combined view
            color_panel = color_image.copy()
            pc_panel = self._create_point_cloud_view(vertices)
            
            if pc_panel is not None:
                # Resize both panels to same height
                height = 400
                color_aspect = color_panel.shape[1] / color_panel.shape[0]
                color_width = int(height * color_aspect)
                
                color_resized = cv2.resize(color_panel, (color_width, height))
                pc_resized = cv2.resize(pc_panel, (height, height))
                
                # Combine horizontally
                display_image = np.hstack([color_resized, pc_resized])
            else:
                # When point cloud is not available, create a consistent layout
                # by showing color image with a placeholder for point cloud
                color_panel = color_image.copy()
                
                # Create placeholder for missing point cloud with same dimensions as would be used
                height = 400
                placeholder = np.zeros((height, height, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No Point Cloud", (height//2-80, height//2-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                cv2.putText(placeholder, "Data Available", (height//2-90, height//2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
                
                # Resize color panel to match the layout when point cloud is available
                color_aspect = color_panel.shape[1] / color_panel.shape[0]
                color_width = int(height * color_aspect)
                color_resized = cv2.resize(color_panel, (color_width, height))
                
                # Combine with placeholder to maintain consistent window size
                display_image = np.hstack([color_resized, placeholder])
        
        # Add overlay information
        self._add_info_overlay(display_image)
        
        # Add help overlay if enabled
        if self.show_help:
            self._add_help_overlay(display_image)
        
        return display_image
    
    def _create_no_camera_display(self) -> np.ndarray:
        """Create a simple status display when camera is disabled."""
        # Create a simple status window
        display_image = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(display_image, "ROBOT CAMERA EXPLORER", (150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show camera status
        if self.camera_failed:
            cv2.putText(display_image, "Camera Failed - Black Screen Mode", (170, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            if hasattr(self, 'consecutive_frame_failures') and self.consecutive_frame_failures >= self.max_frame_failures:
                cv2.putText(display_image, "Camera may be in use by another process", (140, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        elif not self.use_camera:
            cv2.putText(display_image, "Camera Disabled Mode", (200, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        else:
            cv2.putText(display_image, "No Camera Data", (220, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add robot status
        if self.robot_controller and self.robot_controller.is_connected:
            robot_status = "Robot: Connected"
            if self.robot_controller.async_mode_enabled:
                queue_status = self.robot_controller.get_queue_status()
                active_cmd = queue_status.get('active_command', {})
                if active_cmd and active_cmd.get('command_id'):
                    robot_status = f"Robot: Moving ({active_cmd.get('type', 'unknown')})"
                else:
                    pending_count = queue_status.get('queue_size', 0)
                    if pending_count > 0:
                        robot_status = f"Robot: Ready ({pending_count} queued)"
                    else:
                        robot_status = "Robot: Ready"
            color = (0, 255, 0)
        else:
            robot_status = "Robot: Disconnected"
            color = (0, 0, 255)
        
        cv2.putText(display_image, robot_status, (50, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add movement speed
        cv2.putText(display_image, f"Movement Speed: {self.movement_speed}%", (50, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add robot position if available
        if self.current_robot_position:
            pos_text = f"Position: ({self.current_robot_position[0]:.0f}, {self.current_robot_position[1]:.0f}, {self.current_robot_position[2]:.0f})"
            cv2.putText(display_image, pos_text, (50, 260),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Add controls reminder
        cv2.putText(display_image, "Press 'h' for help", (50, 320),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        cv2.putText(display_image, "Press 'q' to quit", (50, 350),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        # Add help overlay if enabled
        if self.show_help:
            self._add_help_overlay(display_image)
        
        return display_image
    
    def _create_point_cloud_view(self, vertices: np.ndarray) -> Optional[np.ndarray]:
        """Create a point cloud visualization."""
        if vertices is None or len(vertices) == 0:
            return None
        
        try:
            # Remove invalid points
            valid_mask = ~np.any(np.isnan(vertices) | np.isinf(vertices), axis=1)
            valid_vertices = vertices[valid_mask]
            
            if len(valid_vertices) == 0:
                return None
            
            # Create side view projection (X-Z plane, looking from Y axis)
            # Scale and offset for display
            scale = 1000  # Convert meters to mm for better visualization
            offset_x, offset_z = 300, 200
            
            # Project points
            x_proj = (valid_vertices[:, 0] * scale + offset_x).astype(int)
            z_proj = (valid_vertices[:, 2] * scale + offset_z).astype(int)
            
            # Create image
            img_size = 600
            pc_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            # Color points based on height (Y coordinate)
            y_values = valid_vertices[:, 1]
            y_min, y_max = np.percentile(y_values, [5, 95])
            y_normalized = np.clip((y_values - y_min) / (y_max - y_min), 0, 1)
            
            # Draw points
            for i in range(len(valid_vertices)):
                if 0 <= x_proj[i] < img_size and 0 <= z_proj[i] < img_size:
                    color_intensity = int(y_normalized[i] * 255)
                    color = (color_intensity, 255 - color_intensity, 128)
                    cv2.circle(pc_image, (x_proj[i], z_proj[i]), 1, color, -1)
            
            # Add axes
            cv2.line(pc_image, (offset_x, 0), (offset_x, img_size), (100, 100, 100), 1)
            cv2.line(pc_image, (0, offset_z), (img_size, offset_z), (100, 100, 100), 1)
            
            # Add labels
            cv2.putText(pc_image, "Point Cloud (Side View)", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(pc_image, "Forward ->", (img_size - 100, img_size - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return pc_image
            
        except Exception as e:
            print(f"Error creating point cloud view: {e}")
            return None
    
    def _add_info_overlay(self, image: np.ndarray):
        """Add information overlay to the image."""
        # Get robot status
        robot_status = "Disconnected"
        if self.robot_controller.is_connected:
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
            "ROBOT CAMERA EXPLORER",
        ]
        
        if self.use_camera:
            info_lines.extend([
                f"View Mode: {['', 'Color', 'Point Cloud', 'Combined'][self.view_mode]}",
                f"FPS: {self.current_fps:.1f}",
                f"Frames: {self.frame_count}"
            ])
        else:
            info_lines.append("Camera: Disabled")
        
        info_lines.extend([
            f"Robot: {robot_status}",
            f"Speed: {self.movement_speed}%"
        ])
        
        # Draw border only (no black background)
        text_height = len(info_lines) * 25 + 10
        cv2.rectangle(image, (5, 5), (300, text_height), (255, 255, 255), 1)
        
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
    
    def _add_help_overlay(self, image: np.ndarray):
        """Add help overlay showing keyboard controls."""
        help_lines = [
            "KEYBOARD CONTROLS:",
            "'q' - Quit",
            "'h' - Toggle this help",
            "'space' - Pause/Resume",
            "",
            "ROBOT MOVEMENT:",
            "'l' - Look at table",
            "'f' - Look forward",
            "'x' - Stop movement", 
            "'r' - Home position",
            "",
            "MANUAL CONTROL:",
            "'a'/'d' - Joint5 Turn L/R",
            "'w'/'s' - Joint4 Tilt U/D",
            "'z'/'c' - Joint1 Base L/R",
            "",
            "VIEW MODES:",
            "'1' - Color only",
            "'2' - Point cloud only", 
            "'3' - Combined view",
            "",
            "SPEED:",
            "'+'/'-' - Speed up/down"
        ]
        
        # Calculate overlay size
        max_width = max(len(line) for line in help_lines) * 8 + 20
        overlay_height = len(help_lines) * 20 + 20
        
        # Position on right side
        x_start = image.shape[1] - max_width - 10
        y_start = 10
        
        # Draw border only (no black background)
        cv2.rectangle(image, (x_start, y_start), 
                     (x_start + max_width, y_start + overlay_height), 
                     (255, 255, 255), 1)
        
        # Draw text
        for i, line in enumerate(help_lines):
            color = (0, 255, 255) if line.endswith(":") else (255, 255, 255)
            if line == "":
                continue
            cv2.putText(image, line, (x_start + 10, y_start + 20 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _handle_user_input(self) -> bool:
        """Handle keyboard input (non-blocking)."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == 255:  # No key pressed
            return True
        
        current_time = time.time()
        
        # Application controls
        if key == ord('q'):
            return False
        elif key == ord('h'):
            self.show_help = not self.show_help
            print(f"Help {'enabled' if self.show_help else 'disabled'}")
        elif key == ord(' '):
            self.is_paused = not self.is_paused
            print(f"{'Paused' if self.is_paused else 'Resumed'}")
        
        # View mode controls
        elif key == ord('1'):
            self.view_mode = 1
            print("View mode: Color camera only")
        elif key == ord('2'):
            self.view_mode = 2
            print("View mode: Point cloud only")
        elif key == ord('3'):
            self.view_mode = 3
            print("View mode: Combined")
        
        # Speed controls
        elif key in [ord('+'), ord('=')]:
            self.movement_speed = min(100, self.movement_speed + 10)
            print(f"Movement speed: {self.movement_speed}%")
        elif key in [ord('-'), ord('_')]:
            self.movement_speed = max(10, self.movement_speed - 10)
            print(f"Movement speed: {self.movement_speed}%")
        
        # Robot controls (only if robot connected and not in cooldown)
        elif self.robot_controller.is_connected and \
             current_time - self.last_movement_time > self.movement_cooldown:
            
            # Predefined positions
            if key == ord('l'):
                self._execute_robot_command("look_at_table", "Looking at table...")
            elif key == ord('f'):
                self._execute_robot_command("look_forward", "Looking forward...")
            elif key == ord('r'):
                self._execute_robot_command("home", "Returning to home...")
            elif key == ord('x'):
                self._execute_robot_command("stop", "Stopping movement...")
            
            # Joint-based movement
            elif key == ord('a'):
                self._rotate_joint(4, self.joint_step, "Turning left (joint5)")
            elif key == ord('d'):
                self._rotate_joint(4, -self.joint_step, "Turning right (joint5)")
            elif key == ord('w'):
                self._rotate_joint(3, self.joint_step, "Tilting up (joint4)")
            elif key == ord('s'):
                self._rotate_joint(3, -self.joint_step, "Tilting down (joint4)")
            elif key == ord('z'):
                self._rotate_joint(0, self.joint_step, "Base rotating left (joint1)")
            elif key == ord('c'):
                self._rotate_joint(0, -self.joint_step, "Base rotating right (joint1)")
        
        return True
    
    def _execute_robot_command(self, command: str, message: str):
        """Execute a predefined robot command."""
        print(message)
        
        if self.robot_controller.async_mode_enabled:
            if command == "look_at_table":
                command_id = self.robot_controller.look_at_table_async(priority=1)
                print(f"Look at table command queued (ID: {command_id})")
            elif command == "look_forward":
                command_id = self.robot_controller.look_forward_async(priority=1)
                print(f"Look forward command queued (ID: {command_id})")
            elif command == "home":
                command_id = self.robot_controller.home_position_async(priority=1)
                print(f"Home position command queued (ID: {command_id})")
            elif command == "stop":
                command_id = self.robot_controller.stop_movement_async(priority=3)
                print(f"Stop command queued (ID: {command_id})")
        else:
            if command == "look_at_table":
                self.robot_controller.look_at_table()
            elif command == "look_forward":
                self.robot_controller.look_forward()
            elif command == "home":
                self.robot_controller.home_position()
            elif command == "stop":
                self.robot_controller.stop_movement()
        
        self.last_movement_time = time.time()
    
    def _rotate_joint(self, joint_index: int, angle_delta: float, message: str):
        """Rotate a specific joint by the given angle delta."""
        if not self.base_joint_angles:
            print("Cannot move: Base joint angles unknown")
            return
        
        # Get current joint angles
        current_angles = self.robot_controller.get_joint_angles()
        
        # Check if current_angles is valid, otherwise use base angles as fallback
        if not current_angles or not isinstance(current_angles, list) or len(current_angles) < 6:
            print(f"Cannot get valid joint angles (got: {current_angles}), using base angles as fallback")
            current_angles = self.base_joint_angles.copy()
        
        # Calculate new angle for the specified joint
        new_angles = current_angles.copy()
        new_angles[joint_index] += angle_delta
        
        print(f"{message}: Joint {joint_index+1} = {new_angles[joint_index]:.1f}°")
        
        # Execute joint movement
        if self.robot_controller.async_mode_enabled:
            from module.robot.command_queue import CommandType
            command_id = self.robot_controller.submit_async_command(
                CommandType.MOVE_TO_ANGLES, new_angles, speed=self.movement_speed, priority=2)
            if command_id:
                print(f"Joint movement command queued (ID: {command_id})")
        else:
            self.robot_controller.move_to_angles(new_angles, speed=self.movement_speed)
        
        self.last_movement_time = time.time()
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _print_controls(self):
        """Print control instructions."""
        print("Controls:")
        print("  'q' - Quit")
        print("  'h' - Toggle help overlay")
        print("  'space' - Pause/Resume")
        print("  '1/2/3' - Switch view modes")
        print("  '+/-' - Adjust movement speed")
        print("\nRobot Movement:")
        print("  'l' - Look at table")
        print("  'f' - Look forward")
        print("  'r' - Home position")
        print("  'x' - Stop movement")
        print("\nManual Control:")
        print("  'a/d' - Joint5 Turn Left/Right")
        print("  'w/s' - Joint4 Tilt Up/Down")
        print("  'z/c' - Joint1 Base Rotate Left/Right")
        print("-" * 40)
    
    def _cleanup(self):
        """Cleanup system resources."""
        print("\nShutting down Robot Camera Explorer...")
        
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
    
    parser = argparse.ArgumentParser(description='Robot Camera Explorer')
    parser.add_argument('--config', default='realtime_tracking',
                       help='Configuration preset name')
    parser.add_argument('--no-camera', action='store_true',
                       help='Disable camera initialization and use robot-only mode')
    
    args = parser.parse_args()
    
    print("Robot Camera Explorer")
    print("=" * 30)
    print("Interactive robot arm camera control")
    if args.no_camera:
        print("Camera disabled (--no-camera)")
    print("=" * 30)
    
    try:
        explorer = RobotCameraExplorer(config_name=args.config, use_camera=not args.no_camera)
        explorer.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()