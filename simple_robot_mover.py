#!/usr/bin/env python3
"""
Simple Robot Movement Script

Reads robot poses from JSON file and moves the robot through them with keyboard control.
No camera involvement - purely for testing robot movement through predefined poses.

Usage:
    python simple_robot_mover.py [options]

Keyboard Controls:
    'c' - Move to next pose
    'p' - Move to previous pose
    'r' - Reset to first pose
    'h' - Toggle help display
    'q' - Quit program
    
Features:
- Loads poses from recorded_coords_280.json
- Safety validation before each movement
- Non-blocking keyboard interface
- Real-time status display
"""

import sys
import os
import json
import time
import cv2
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.robot.mycobot_controller import MyCobotController


class SimpleRobotMover:
    """Simple robot movement controller for cycling through JSON poses."""
    
    def __init__(self, json_file: str = "recorded_coords_280.json"):
        """
        Initialize robot mover.
        
        Args:
            json_file: Path to JSON file containing poses
        """
        self.json_file = json_file
        self.poses = []
        self.current_pose_index = 0
        self.robot_controller = None
        
        # Application state
        self.is_running = False
        self.show_help = False
        self.last_movement_time = 0
        self.movement_cooldown = 1.0  # seconds between movements
        
    def load_poses(self) -> bool:
        """
        Load poses from JSON file.
        
        Returns:
            True if poses loaded successfully
        """
        try:
            with open(self.json_file, 'r') as f:
                self.poses = json.load(f)
            
            print(f"Loaded {len(self.poses)} poses from {self.json_file}")
            return True
            
        except FileNotFoundError:
            print(f"Error: JSON file '{self.json_file}' not found")
            return False
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in '{self.json_file}': {e}")
            return False
        except Exception as e:
            print(f"Error loading poses: {e}")
            return False
    
    def initialize_robot(self) -> bool:
        """
        Initialize robot controller.
        
        Returns:
            True if robot initialized successfully
        """
        try:
            print("Initializing myCobot controller...")
            self.robot_controller = MyCobotController()
            
            # Try to connect to robot
            if not self.robot_controller.connect():
                print("Warning: Failed to connect to robot hardware")
                print("Continuing in simulation mode for testing")
                return True  # Allow running without hardware for testing
            
            print("Robot connected successfully!")
            return True
            
        except Exception as e:
            print(f"Error initializing robot: {e}")
            return False
    
    def display_pose_info(self, pose_index: int, pose: list):
        """
        Display information about current pose.
        
        Args:
            pose_index: Index of current pose
            pose: Pose coordinates [x, y, z, rx, ry, rz]
        """
        print(f"\n{'='*50}")
        print(f"Pose {pose_index + 1}/{len(self.poses)}")
        print(f"Position: X={pose[0]:6.1f}, Y={pose[1]:6.1f}, Z={pose[2]:6.1f} mm")
        print(f"Rotation: RX={pose[3]:6.1f}, RY={pose[4]:6.1f}, RZ={pose[5]:6.1f} deg")
        print(f"{'='*50}")
    
    def validate_pose_safety(self, pose: list) -> tuple:
        """
        Validate if pose is safe for robot movement.
        
        Args:
            pose: Pose coordinates [x, y, z, rx, ry, rz]
            
        Returns:
            Tuple of (is_safe, reason)
        """
        if not self.robot_controller:
            return False, "Robot controller not initialized"
        
        return self.robot_controller.validate_position_safety(pose)
    
    def move_to_pose(self, pose: list) -> bool:
        """
        Move robot to specified pose.
        
        Args:
            pose: Target pose [x, y, z, rx, ry, rz]
            
        Returns:
            True if movement successful
        """
        if not self.robot_controller:
            print("Error: Robot controller not initialized")
            return False
        
        # Validate pose safety
        is_safe, reason = self.validate_pose_safety(pose)
        if not is_safe:
            print(f"Warning: Unsafe pose - {reason}")
            
            # Try to clamp to safe workspace
            clamped_pose, adjustments = self.robot_controller.clamp_position_to_workspace(pose)
            if adjustments:
                print(f"Applied safety adjustments: {', '.join(adjustments)}")
                print(f"Clamped pose: {[f'{x:.1f}' for x in clamped_pose]}")
                pose = clamped_pose
            else:
                print("Cannot make pose safe, skipping movement")
                return False
        
        # Execute movement
        print("Moving robot to pose...")
        
        if self.robot_controller.is_connected:
            success = self.robot_controller.move_to_position(pose, speed=50)
            if success:
                print("Movement command sent successfully")
                # Wait a moment for movement to start
                time.sleep(0.5)
            else:
                print("Failed to send movement command")
            return success
        else:
            print("Simulated movement (no hardware connected)")
            time.sleep(1)  # Simulate movement time
            return True
    
    def _create_status_display(self) -> np.ndarray:
        """Create a status display window."""
        display_image = np.zeros((500, 700, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(display_image, "SIMPLE ROBOT MOVER", (180, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Current pose info
        if self.poses and 0 <= self.current_pose_index < len(self.poses):
            current_pose = self.poses[self.current_pose_index]
            
            # Pose counter
            cv2.putText(display_image, f"Pose: {self.current_pose_index + 1}/{len(self.poses)}", 
                       (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Position
            cv2.putText(display_image, f"Position:", (50, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_image, f"  X: {current_pose[0]:6.1f} mm", (70, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display_image, f"  Y: {current_pose[1]:6.1f} mm", (70, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display_image, f"  Z: {current_pose[2]:6.1f} mm", (70, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Rotation
            cv2.putText(display_image, f"Rotation:", (350, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_image, f"  RX: {current_pose[3]:6.1f} deg", (370, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display_image, f"  RY: {current_pose[4]:6.1f} deg", (370, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display_image, f"  RZ: {current_pose[5]:6.1f} deg", (370, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Robot status
        if self.robot_controller and self.robot_controller.is_connected:
            robot_status = "Robot: Connected"
            color = (0, 255, 0)
        else:
            robot_status = "Robot: Disconnected (Simulation Mode)"
            color = (0, 255, 255)
        
        cv2.putText(display_image, robot_status, (50, 320),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Controls reminder
        cv2.putText(display_image, "Controls: 'c'=next  'p'=prev  'r'=reset  'h'=help  'q'=quit", 
                   (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        # Add help overlay if enabled
        if self.show_help:
            self._add_help_overlay(display_image)
        
        return display_image
    
    def _add_help_overlay(self, image: np.ndarray):
        """Add help overlay showing keyboard controls."""
        help_lines = [
            "KEYBOARD CONTROLS:",
            "'c' - Move to next pose",
            "'p' - Move to previous pose", 
            "'r' - Reset to first pose",
            "'h' - Toggle this help",
            "'q' - Quit program",
            "",
            "ROBOT COMMANDS:",
            "Robot moves to displayed pose",
            "Validates safety before movement",
            "Cycles through all poses"
        ]
        
        # Calculate overlay size
        max_width = max(len(line) for line in help_lines) * 8 + 20
        overlay_height = len(help_lines) * 20 + 20
        
        # Position on right side
        x_start = image.shape[1] - max_width - 10
        y_start = 10
        
        # Draw border
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
    
    def _handle_keyboard_input(self) -> bool:
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
        
        # Movement controls (with cooldown)
        elif current_time - self.last_movement_time > self.movement_cooldown:
            if key == ord('c'):
                # Move to current pose and advance
                current_pose = self.poses[self.current_pose_index]
                if self.move_to_pose(current_pose):
                    self.current_pose_index = (self.current_pose_index + 1) % len(self.poses)
                    if self.current_pose_index == 0:
                        print("Completed full cycle - starting over")
                self.last_movement_time = current_time
                
            elif key == ord('p'):
                # Move to previous pose
                self.current_pose_index = (self.current_pose_index - 1) % len(self.poses)
                current_pose = self.poses[self.current_pose_index]
                self.move_to_pose(current_pose)
                self.last_movement_time = current_time
                
            elif key == ord('r'):
                # Reset to first pose
                self.current_pose_index = 0
                current_pose = self.poses[self.current_pose_index]
                self.move_to_pose(current_pose)
                print("Reset to first pose")
                self.last_movement_time = current_time
        
        return True
    
    def run(self):
        """Main execution loop."""
        print("Simple Robot Movement Script")
        print("=" * 40)
        
        # Load poses from JSON
        if not self.load_poses():
            return False
        
        if len(self.poses) == 0:
            print("No poses found in JSON file")
            return False
        
        # Initialize robot
        if not self.initialize_robot():
            return False
        
        print(f"\nLoaded {len(self.poses)} poses from {self.json_file}")
        print("\nKeyboard Controls:")
        print("  'c' - Move to next pose")
        print("  'p' - Move to previous pose")
        print("  'r' - Reset to first pose")
        print("  'h' - Toggle help display")
        print("  'q' - Quit program")
        print("\nStarting main loop...")
        
        # Create window
        cv2.namedWindow("Simple Robot Mover", cv2.WINDOW_AUTOSIZE)
        self.is_running = True
        
        try:
            while self.is_running:
                # Create and display status
                display_image = self._create_status_display()
                cv2.imshow("Simple Robot Mover", display_image)
                
                # Handle keyboard input
                if not self._handle_keyboard_input():
                    break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.02)
                
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        finally:
            cv2.destroyAllWindows()
            
            # Cleanup robot
            if self.robot_controller and self.robot_controller.is_connected:
                print("Moving robot to home position...")
                self.robot_controller.home_position()
                self.robot_controller.disconnect()
        
        return True


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple robot movement through JSON poses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Use default recorded_coords_280.json
    python simple_robot_mover.py
    
    # Use custom JSON file
    python simple_robot_mover.py --json-file my_poses.json
        """
    )
    
    parser.add_argument(
        '--json-file',
        type=str,
        default='recorded_coords_280.json',
        help='JSON file containing robot poses (default: recorded_coords_280.json)'
    )
    
    args = parser.parse_args()
    
    try:
        # Create and run robot mover
        mover = SimpleRobotMover(json_file=args.json_file)
        success = mover.run()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)