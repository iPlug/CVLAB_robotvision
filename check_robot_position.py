#!/usr/bin/env python3
"""
Robot End Effector Position Checker

Since this is an eye-in-hand setup where the camera is mounted on the robot's
end effector, we need to know the robot's current position to properly interpret
the coordinate transformations.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.robot.mycobot_controller import MyCobotController

def main():
    print("=== ROBOT END EFFECTOR POSITION CHECK ===")
    print("Eye-in-hand setup: Camera is mounted on robot end effector")
    print()
    
    # Initialize robot controller
    try:
        print("Connecting to myCobot robot...")
        robot = MyCobotController()
        
        if not robot.connect():
            print("Failed to connect to robot!")
            print("Make sure robot is powered on and connected via USB.")
            return
        
        print("Robot connected successfully!")
        print()
        
        # Get current robot position
        print("Getting current robot end effector position...")
        current_coords = robot.get_current_position()
        
        if current_coords:
            x, y, z, rx, ry, rz = current_coords
            print(f"Current robot end effector position:")
            print(f"  Position: X={x:.1f}mm, Y={y:.1f}mm, Z={z:.1f}mm")
            print(f"  Orientation: RX={rx:.1f}°, RY={ry:.1f}°, RZ={rz:.1f}°")
            print()
            
            # Analyze the position
            print("POSITION ANALYSIS:")
            print(f"  X = {x:.1f}mm ({'FORWARD' if x > 0 else 'BACKWARD'} from robot base)")
            print(f"  Y = {y:.1f}mm ({'LEFT' if y > 0 else 'RIGHT'} from robot base)")
            print(f"  Z = {z:.1f}mm ({'ABOVE' if z > 0 else 'BELOW'} table level)")
            
            # Calculate distance from expected object position [500, 0, 0]
            expected_object_pos = np.array([500.0, 0.0, 0.0])  # mm
            robot_pos = np.array([x, y, z])  # mm
            
            distance_to_expected = np.linalg.norm(robot_pos - expected_object_pos)
            print()
            print("DISTANCE ANALYSIS:")
            print(f"  Distance from robot end effector to expected object position [500, 0, 0]:")
            print(f"  {distance_to_expected:.1f}mm")
            
            # Eye-in-hand transformation explanation
            print()
            print("EYE-IN-HAND COORDINATE INTERPRETATION:")
            print("  Camera is attached to robot end effector")
            print("  Camera-to-robot transformation includes:")
            print("  1. Camera position relative to object")
            print("  2. Robot end effector position relative to robot base")
            print("  3. Combined transformation: camera → object → robot base")
            print()
            print("  If object appears at camera coordinates [-0.2, 0.015, 0.67]m,")
            print("  and robot end effector is at [{}, {}, {}]mm,".format(x, y, z))
            print("  then the final object position relative to robot base depends on")
            print("  the complete transformation chain.")
            
        else:
            print("Failed to get robot coordinates!")
            print("Robot may not be responding or in an error state.")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check robot power and USB connection")
        print("2. Verify COM port in robot controller")
        print("3. Make sure no other applications are using the robot")
    
    finally:
        try:
            if 'robot' in locals():
                robot.disconnect()
                print("\nRobot disconnected.")
        except:
            pass

if __name__ == "__main__":
    main()