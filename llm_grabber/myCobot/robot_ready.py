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

home_angles = [0, 0, 0, 0, 0, 45]

def robot_ready_sequence(mycobot):
    """Execute robot ready movement sequence."""
    print("Preparing to stand up position")
    mycobot.power_on()
    mycobot.send_angles(home_angles, 50)
    time.sleep(3)

def test(mycobot):
    show_camera_during_robot_sequence(
        lambda: robot_ready_sequence(mycobot), 
        pre_delay=3, 
        post_delay=3
    )

if __name__ == "__main__":
    mycobot = setup()
    test(mycobot)
