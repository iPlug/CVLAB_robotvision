#!/usr/bin/env python3
"""
Robot utilities for CharUco pose estimation applications.
Handles robot connection, movement, and pose management.
"""

import sys
import os
import time
import json
from pathlib import Path

# Import robot controller (use relative path within llm_grabber)
try:
    # Add the llm_grabber root to path
    llm_grabber_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, llm_grabber_root)
    from module.robot.mycobot_controller import MyCobotController
    ROBOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Robot controller not available: {e}")
    ROBOT_AVAILABLE = False


class RobotUtils:
    """Utilities for robot connection, movement, and pose management."""
    
    def __init__(self):
        """Initialize robot utilities."""
        self.robot = None
        self.preset_poses = []
        self.current_pose_index = 0
        self.initialize_robot()
        
    def initialize_robot(self):
        """Initialize robot controller if available."""
        if ROBOT_AVAILABLE:
            try:
                self.robot = MyCobotController()
                if self.robot.connect():
                    print("Robot controller initialized and connected successfully")
                else:
                    print("Robot controller initialized but connection failed - using manual input")
            except Exception as e:
                print(f"Failed to initialize robot: {e}")
                self.robot = None
    
    def load_robot_poses(self):
        """Load predefined robot poses from JSON file."""
        poses_file = Path("recorded_coords_280.json")
        try:
            if poses_file.exists():
                with open(poses_file, 'r') as f:
                    self.preset_poses = json.load(f)
                print(f"Loaded {len(self.preset_poses)} preset robot poses")
            else:
                self.preset_poses = [
                    [200, 0, 300, 0, 180, 0],
                    [150, -100, 280, -20, 160, -30],
                    [150, 100, 280, 20, 160, 30],
                    [250, 0, 250, 0, 145, 0],
                    [180, -80, 320, -15, 175, -20],
                    [180, 80, 320, 15, 175, 20],
                ]
                print("Using default calibration poses (recorded_coords_280.json not found)")
        except Exception as e:
            print(f"Error loading poses: {e}")
            self.preset_poses = [
                [200, 0, 300, 0, 180, 0],
                [150, -50, 280, 0, 160, -30],
                [150, 50, 280, 0, 160, 30],
            ]
    
    def move_robot_to_next_pose(self):
        """Move robot to the next preset pose."""
        if not self.robot or not hasattr(self.robot, 'is_connected') or not self.robot.is_connected:
            print("Robot not connected - cannot move automatically")
            return False
        
        if not self.preset_poses:
            print("No preset poses available")
            return False
        
        pose = self.preset_poses[self.current_pose_index]
        self.current_pose_index = (self.current_pose_index + 1) % len(self.preset_poses)
        
        print(f"Moving robot to pose {self.current_pose_index}/{len(self.preset_poses)}: {pose}")
        
        try:
            success = self.robot.move_to_position(pose)
            if success:
                print(f"Robot moved successfully to position {self.current_pose_index}")
                time.sleep(1)
                return True
            else:
                print("Robot movement failed")
                return False
        except Exception as e:
            print(f"Error moving robot: {e}")
            return False
    
    def get_robot_movement_status(self):
        """Get current robot movement status for display."""
        if not self.preset_poses:
            return "No preset poses"
        
        total_poses = len(self.preset_poses)
        current = self.current_pose_index
        next_pose = self.preset_poses[current]
        
        return f"Next pose {current+1}/{total_poses}: [{next_pose[0]:.0f}, {next_pose[1]:.0f}, {next_pose[2]:.0f}]"
    
    def get_robot_pose(self):
        """Get current robot end-effector pose."""
        if self.robot is not None:
            if hasattr(self.robot, 'is_connected') and not self.robot.is_connected:
                pass
            else:
                try:
                    pose = self.robot.get_current_position()
                    if pose and len(pose) >= 6:
                        return pose[:6]
                except Exception:
                    pass
        return None