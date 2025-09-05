#!/usr/bin/env python3
"""
Point myCobot to specific coordinates utility.

Simple command-line tool to move the myCobot robot arm to user-specified coordinates.
Supports both absolute positioning and pointing toward coordinates.
"""

import sys
import os
import argparse
import time
from typing import List, Optional

# Add module path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'module'))

from robot.mycobot_controller import MyCobotController


def parse_coordinates(coord_str: str) -> List[float]:
    """
    Parse coordinate string into list of floats.
    
    Args:
        coord_str: Coordinate string like "100,200,300" or "100,200,300,0,180,0"
        
    Returns:
        List of coordinate values
    """
    try:
        coords = [float(x.strip()) for x in coord_str.split(',')]
        if len(coords) < 3:
            raise ValueError("Need at least X,Y,Z coordinates")
        return coords
    except ValueError as e:
        raise ValueError(f"Invalid coordinate format: {e}")


def print_robot_status(robot: MyCobotController):
    """Print current robot status and position."""
    print("\n=== Robot Status ===")
    
    if robot.is_connected:
        position = robot.get_current_position()
        if position:
            print(f"Current Position: [{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}, {position[3]:.1f}, {position[4]:.1f}, {position[5]:.1f}]")
        else:
            print("Current Position: Unable to read")
        
        angles = robot.get_joint_angles()
        if angles:
            print(f"Current Angles: [{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}, {angles[3]:.1f}, {angles[4]:.1f}, {angles[5]:.1f}]")
        else:
            print("Current Angles: Unable to read")
    else:
        print("Robot: Not connected")
    
    print("Workspace bounds:", robot.workspace_bounds)
    print("Max reach:", robot.safety_constraints.get('max_reach', 'Unknown'))


def move_to_coordinate(robot: MyCobotController, coords: List[float], speed: int = 50):
    """
    Move robot to specified coordinates.
    
    Args:
        robot: Robot controller instance
        coords: Target coordinates [x, y, z] or [x, y, z, rx, ry, rz]
        speed: Movement speed (0-100)
    """
    # Ensure we have full 6DOF coordinates
    if len(coords) == 3:
        # Add default orientation (pointing down)
        default_orientation = robot.safety_constraints.get('default_pointing_angles', [0, 180, 0])
        coords = coords + default_orientation
    elif len(coords) != 6:
        raise ValueError("Coordinates must be [x,y,z] or [x,y,z,rx,ry,rz]")
    
    print(f"\nMoving to coordinates: {[f'{x:.1f}' for x in coords]}")
    print(f"Speed: {speed}")
    
    # Validate position safety
    is_safe, reason = robot.validate_position_safety(coords)
    if not is_safe:
        print(f"Warning: Position validation failed: {reason}")
        
        # Try to calculate reachable position
        reachable_coords, was_adjusted = robot.calculate_reachable_position(coords)
        if was_adjusted:
            print(f"Adjusted to reachable position: {[f'{x:.1f}' for x in reachable_coords[:3]]}")
            coords = reachable_coords
        else:
            print("Cannot find safe position. Aborting.")
            return False
    
    # Execute movement
    success = robot.move_to_position(coords, speed)
    if success:
        print("Movement command sent successfully")
        
        # Wait and show final position
        time.sleep(2)
        final_position = robot.get_current_position()
        if final_position:
            print(f"Final position: {[f'{x:.1f}' for x in final_position]}")
    else:
        print("Movement failed")
    
    return success


def calculate_pointing_orientation(robot_pos: List[float], target_pos: List[float]) -> List[float]:
    """
    Calculate orientation angles to point from robot position toward target position.
    
    This function now uses the coordinate utilities module for consistent myCobot calculations.
    
    Args:
        robot_pos: Robot position [forward, side, height]
        target_pos: Target position [forward, side, height]
        
    Returns:
        Orientation angles [rx, ry, rz] in myCobot conventions
    """
    # Import the utility function for consistent calculations
    # Direct import to avoid __init__.py relative import issues
    import sys
    import os
    utils_path = os.path.join(os.path.dirname(__file__), 'module', 'utils')
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    from coordinate_utils import calculate_mycobot_pointing_orientation
    
    return calculate_mycobot_pointing_orientation(robot_pos, target_pos)


def point_toward_coordinate(robot: MyCobotController, coords: List[float], speed: int = 50):
    """
    Point robot toward specified coordinates (for unreachable positions).
    
    Args:
        robot: Robot controller instance
        coords: Target coordinates [x, y, z]
        speed: Movement speed (0-100)
    """
    if len(coords) < 3:
        raise ValueError("Need at least X,Y,Z coordinates for pointing")
    
    print(f"\nPointing toward coordinates: {[f'{x:.1f}' for x in coords[:3]]}")
    
    # First calculate a position along the direction to the target
    # Use default pointing orientation initially to get the position
    default_orientation = robot.safety_constraints.get('default_pointing_angles', [0, 180, 0])
    pointing_pose = robot.calculate_pointing_toward_object(coords[:3], default_orientation)
    
    if pointing_pose:
        robot_position = pointing_pose[:3]  # Extract position
        
        # Now calculate the correct orientation to point toward the target
        pointing_orientation = calculate_pointing_orientation(robot_position, coords[:3])
        
        # Create the final pose with correct pointing orientation
        final_pose = robot_position + pointing_orientation
        
        print(f"Calculated pointing position: {[f'{x:.1f}' for x in robot_position]}")
        print(f"Calculated pointing orientation: {[f'{x:.1f}' for x in pointing_orientation]} (toward target)")
        
        success = robot.move_to_position(final_pose, speed)
        if success:
            print("Pointing movement command sent successfully")
            
            # Wait and show final position
            time.sleep(2)
            final_position = robot.get_current_position()
            if final_position:
                print(f"Final pointing position: {[f'{x:.1f}' for x in final_position]}")
        else:
            print("Pointing movement failed")
        
        return success
    else:
        print("Cannot calculate safe pointing position")
        return False


def interactive_mode(robot: MyCobotController):
    """Run interactive coordinate input mode."""
    print("\n=== Interactive Mode ===")
    print("Enter coordinates as: x,y,z or x,y,z,rx,ry,rz")
    print("Commands:")
    print("  'status' - Show robot status")
    print("  'home' - Move to home position")
    print("  'table' - Move to table viewing position")
    print("  'forward' - Move to forward looking position")
    print("  'stop' - Stop movement")
    print("  'quit' or 'exit' - Exit")
    print()
    
    while True:
        try:
            user_input = input("Enter coordinates or command: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            elif user_input.lower() == 'status':
                print_robot_status(robot)
                continue
            elif user_input.lower() == 'home':
                print("Moving to home position...")
                robot.home_position()
                continue
            elif user_input.lower() == 'table':
                print("Moving to table viewing position...")
                robot.look_at_table()
                continue
            elif user_input.lower() == 'forward':
                print("Moving to forward looking position...")
                robot.look_forward()
                continue
            elif user_input.lower() == 'stop':
                print("Stopping movement...")
                robot.stop_movement()
                continue
            
            # Parse coordinates
            coords = parse_coordinates(user_input)
            
            # Ask for movement type if position might be unreachable
            if len(coords) >= 3:
                distance = (coords[0]**2 + coords[1]**2 + coords[2]**2)**0.5
                max_reach = robot.safety_constraints.get('max_reach', 280)
                
                if distance > max_reach * 0.9:  # Close to max reach
                    move_type = input("Position might be unreachable. Move 'to' position or 'point' toward it? [to/point]: ").strip().lower()
                    if move_type == 'point':
                        point_toward_coordinate(robot, coords)
                        continue
            
            # Default: move to position
            speed_input = input("Enter speed (0-100) [default 50]: ").strip()
            speed = 50
            if speed_input:
                try:
                    speed = max(0, min(100, int(speed_input)))
                except ValueError:
                    print("Invalid speed, using default 50")
            
            move_to_coordinate(robot, coords, speed)
            
        except ValueError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Point myCobot to specific coordinates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python point_to_coordinate.py --coords 200,100,150
  python point_to_coordinate.py --coords 200,100,150,0,180,0 --speed 80
  python point_to_coordinate.py --point-toward 400,200,100
  python point_to_coordinate.py --interactive
  python point_to_coordinate.py --status
  python point_to_coordinate.py --home
        """
    )
    
    # Coordinate input options
    parser.add_argument('--coords', type=str, 
                       help='Target coordinates as "x,y,z" or "x,y,z,rx,ry,rz"')
    parser.add_argument('--point-toward', type=str, default="400,0,0",
                       help='Point toward coordinates (for unreachable positions)')
    parser.add_argument('--speed', type=int, default=50,
                       help='Movement speed (0-100, default: 50)')
    
    # Preset positions
    parser.add_argument('--home', action='store_true',
                       help='Move to home position')
    parser.add_argument('--table', action='store_true',
                       help='Move to table viewing position')
    parser.add_argument('--forward', action='store_true',
                       help='Move to forward looking position')
    
    # Utility options
    parser.add_argument('--status', action='store_true',
                       help='Show robot status and exit')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--transformation-matrix', type=str,
                       help='Path to camera-robot transformation matrix file')
    
    args = parser.parse_args()
    
    # Initialize robot controller
    print("Initializing myCobot controller...")
    robot = MyCobotController(args.transformation_matrix)
    
    # Connect to robot
    if not robot.connect():
        print("Failed to connect to myCobot. Make sure robot is connected and port_setup.py is configured.")
        return 1
    
    try:
        # Show initial status
        print_robot_status(robot)
        
        # Handle different operation modes
        if args.status:
            # Status only mode
            return 0
        
        elif args.interactive:
            # Interactive mode
            interactive_mode(robot)
        
        elif args.home:
            # Home position
            print("Moving to home position...")
            robot.home_position()
        
        elif args.table:
            # Table viewing position
            print("Moving to table viewing position...")
            robot.look_at_table()
        
        elif args.forward:
            # Forward looking position
            print("Moving to forward looking position...")
            robot.look_forward()
        
        elif args.coords:
            # Move to specific coordinates
            coords = parse_coordinates(args.coords)
            move_to_coordinate(robot, coords, args.speed)
        
        elif args.point_toward:
            # Point toward coordinates
            coords = parse_coordinates(args.point_toward)
            point_toward_coordinate(robot, coords, args.speed)
        
        else:
            # No specific action, show help
            parser.print_help()
            print("\nUse --interactive for interactive mode")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        # Cleanup
        robot.disconnect()
        print("Robot disconnected")


if __name__ == "__main__":
    sys.exit(main())