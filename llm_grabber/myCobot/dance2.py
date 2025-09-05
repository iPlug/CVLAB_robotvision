import time
import os
import sys
from pymycobot.mycobot280 import MyCobot280
from pymycobot.genre import Angle, Coord

sys.path.append(os.path.dirname(__file__))
from port_setup import setup

# Define safe coordinate positions for smooth movement
home_position = [0, 0, 0, 0, 0, 0]

# Coordinate waypoints for smooth movement (x, y, z, rx, ry, rz)
waypoints = [
    [200, 0, 200, 0, 0, 0],      # Starting position - center front, high
    [150, -100, 150, 0, 45, 0],   # Move left and down, slight pitch
    [150, 100, 150, 0, 45, 0],    # Move right, same height
    [100, 0, 100, 0, 90, 0],      # Move forward, point down for pickup
    [100, 0, 150, 0, 45, 0],      # Lift up slightly
    [150, -80, 120, 0, 60, 45],   # Move to drop position with rotation
    [150, -80, 100, 0, 90, 45],   # Lower for drop
    [200, 0, 200, 0, 0, 0],       # Return to start position
]

def smooth_move_to_coord(mycobot, target_coords, speed=50, mode=1):
    """Move to coordinates with linear interpolation for smoothness"""
    print(f"Moving to coordinates: {target_coords} at speed {speed}")
    mycobot.send_coords(target_coords, speed, mode)  # mode=1 for linear movement
    
    # Wait for movement to complete
    time.sleep(2)
    
    # Verify position reached
    current_coords = mycobot.get_coords()
    if current_coords:
        print(f"Current position: {[round(x, 1) for x in current_coords]}")
    

def smooth_joint_movement(mycobot, joint, target_angle, speed=40):
    """Move a single joint smoothly"""
    current_angles = mycobot.get_angles()
    if current_angles:
        current_angle = current_angles[joint - 1]
        print(f"Moving joint {joint} from {current_angle:.1f}° to {target_angle}° at speed {speed}")
    
    mycobot.send_angle(joint, target_angle, speed)
    time.sleep(1.5)


def test(mycobot):
    print("\n=== Starting Smooth Movement Test ===\n")
    
    # 1. Initialize to home position
    print("1. Moving to home position...")
    mycobot.send_angles(home_position, 60)
    time.sleep(2)
    
    initial_coords = mycobot.get_coords()
    print(f"Home coordinates: {initial_coords}")
    print(f"Home angles: {mycobot.get_angles()}\n")
    
    # 2. Open gripper initially
    print("2. Opening gripper...")
    mycobot.set_gripper_state(0, 80)
    time.sleep(1)
    
    # 3. Execute smooth coordinate movements
    print("3. Executing smooth coordinate sequence...")
    for i, waypoint in enumerate(waypoints):
        print(f"\n--- Waypoint {i+1} ---")
        smooth_move_to_coord(mycobot, waypoint, speed=45, mode=1)
        
        # Special actions at certain waypoints
        if i == 3:  # At pickup position
            print("Closing gripper (picking up object)...")
            mycobot.set_gripper_state(1, 80)
            time.sleep(1.5)
        elif i == 6:  # At drop position
            print("Opening gripper (dropping object)...")
            mycobot.set_gripper_state(0, 80)
            time.sleep(1.5)
    
    # 4. Demonstrate smooth joint movements
    print("\n4. Demonstrating smooth joint rotations...")
    
    # Smooth base rotation sequence
    base_angles = [-30, 30, -15, 15, 0]
    for angle in base_angles:
        smooth_joint_movement(mycobot, Angle.J1.value, angle, speed=35)
    
    # 5. Return to home and close gripper
    print("\n5. Returning to home position...")
    mycobot.send_angles(home_position, 50)
    time.sleep(2)
    
    print("Closing gripper...")
    mycobot.set_gripper_state(1, 80)
    time.sleep(1)
    
    print("\n=== Smooth Movement Test Complete ===\n")


def demonstrate_coordinate_control(mycobot):
    """Additional demonstration of coordinate-based control"""
    print("\n=== Coordinate Control Demonstration ===\n")
    
    # Move to a good starting position
    start_pos = [180, 0, 180, 0, 45, 0]
    smooth_move_to_coord(mycobot, start_pos, speed=50)
    
    # Create a smooth circular motion in the XY plane
    print("Creating circular motion...")
    radius = 50
    center_x, center_y, z = 150, 0, 120
    
    for angle in range(0, 361, 30):  # 30-degree increments for smoothness
        x = center_x + radius * cos(radians(angle))
        y = center_y + radius * sin(radians(angle))
        coords = [x, y, z, 0, 90, 0]  # Point downward
        
        mycobot.send_coords(coords, 40, 1)  # Linear movement
        time.sleep(0.8)
    
    # Return home
    mycobot.send_angles(home_position, 50)
    time.sleep(2)


# Import math functions for circular motion
from math import cos, sin, radians


if __name__ == "__main__":
    print("""
============================================
|     Smooth MyCobot280 Movement Test     |
|                                          |
| Features:                                |
| - Coordinate-based smooth movements      |
| - Linear interpolation between points    |
| - Controlled speeds for fluid motion     |
| - Integrated gripper control            |
| - Safe waypoint navigation              |
============================================
    """)
    
    time.sleep(3)
    
    try:
        mycobot = setup()
        test(mycobot)
        
        # Optional: Run additional coordinate demonstration
        # demonstrate_coordinate_control(mycobot)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Make sure the robot is properly connected and powered on.")