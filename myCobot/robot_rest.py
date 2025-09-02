#!/usr/bin/env python3
"""
Robot Rest Position Script

This script moves the robot to a safe resting position and optionally powers down servos.
Use this to put the robot in a safe position when not in use.

Usage:
    python robot_rest.py
"""

import sys
import os
import time

from port_setup import setup

# Add parent directory to path for camera preview utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from camera_preview_utils import show_camera_during_robot_sequence

rest_angles = [31.11, -113.02, -64.24, 93.33, 105.11, 79.71]
home_angles = [0, 0, 0, 0, 0, 0]

def robot_rest_sequence(mycobot):
    """Execute robot rest movement sequence."""
    print("Preparing rest position")
    mycobot.send_angles(home_angles, 50)
    time.sleep(3)
    mycobot.send_angles(rest_angles, 30)
    time.sleep(3)
    print("=== Pull out the adapter ===")
    mycobot.release_all_servos()

def test(mycobot):
    show_camera_during_robot_sequence(
        lambda: robot_rest_sequence(mycobot), 
        pre_delay=3, 
        post_delay=3
    )

if __name__ == "__main__":
    mycobot = setup()
    test(mycobot)
