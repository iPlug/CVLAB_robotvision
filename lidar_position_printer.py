#!/usr/bin/env python3
"""
LiDAR Position Printer - Robot Frame Coordinates

This script extends the LiDAR tracking system to print object positions
relative to the robot frame from both camera frame and world frame (gravity-aligned).

Key Features:
- Prints camera frame positions (raw sensor coordinates)
- Prints world frame positions (IMU gravity-aligned coordinates) 
- Shows robot frame coordinates for both
- Displays coordinate transformation details
- Real-time position tracking with enhanced console output
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any, Optional

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lidar_track_object import LiDARObjectTracker
from module.processing.lidar_coordinator import LiDARCoordinator


class LiDARPositionPrinter(LiDARObjectTracker):
    """
    Extended LiDAR tracker that prints detailed position information
    relative to robot frame from both camera and world coordinates.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Enhanced console output settings
        self.print_detailed_positions = True
        self.position_print_interval = 0.5  # Print every 0.5 seconds
        self.last_position_print_time = 0
        
        # Position history for stability analysis
        self.position_history = {
            'camera_frame': [],
            'world_frame': [],
            'robot_frame': []
        }
        self.max_history_size = 10
        
        print("\n" + "="*80)
        print("LIDAR POSITION PRINTER - ROBOT FRAME COORDINATES")
        print("="*80)
        print("This system will print object positions relative to robot frame from:")
        print("  1. Camera Frame (raw sensor coordinates)")
        print("  2. World Frame (gravity-aligned coordinates)")
        print("  3. Robot Frame (transformed coordinates)")
        print("="*80 + "\n")
    
    def _process_frame(self, vertices: np.ndarray, color_image: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Enhanced frame processing with detailed position printing."""
        current_time = time.time()
        
        # Call parent processing method
        detection_result = super()._process_frame(vertices, color_image)
        
        # Print detailed position information if enabled and enough time has passed
        if (self.print_detailed_positions and 
            detection_result and 
            current_time - self.last_position_print_time > self.position_print_interval):
            
            self._print_detailed_positions(detection_result, current_time)
            self.last_position_print_time = current_time
        
        return detection_result
    
    def _print_detailed_positions(self, detection_result: Dict[str, Any], current_time: float):
        """Print detailed position information for the detected object."""
        if not detection_result or 'cluster' not in detection_result:
            return
        
        cluster = detection_result['cluster']
        robot_coords = detection_result['robot_coords']
        
        # Clear console for clean output (optional - can be disabled)
        # os.system('cls' if os.name == 'nt' else 'clear')
        
        print("\n" + "="*80)
        print(f"OBJECT POSITION DETECTION - Frame {self.frame_count}")
        print("="*80)
        
        # 1. Camera Frame Information (Raw Sensor Coordinates)
        print("1. CAMERA FRAME (Raw Sensor Coordinates):")
        camera_center = cluster['center']  # [x, y, z] in meters
        print(f"   Position (m):  [{camera_center[0]:+7.4f}, {camera_center[1]:+7.4f}, {camera_center[2]:+7.4f}]")
        print(f"   Position (mm): [{camera_center[0]*1000:+7.0f}, {camera_center[1]*1000:+7.0f}, {camera_center[2]*1000:+7.0f}]")
        print(f"   Frame: X=right, Y=down, Z=forward (camera perspective)")
        
        # 2. World Frame Information (Gravity-Aligned Coordinates)
        if 'center_world_frame' in cluster:
            world_center = cluster['center_world_frame']  # [x, y, z] in meters (gravity-aligned)
            print("\n2. WORLD FRAME (Gravity-Aligned Coordinates):")
            print(f"   Position (m):  [{world_center[0]:+7.4f}, {world_center[1]:+7.4f}, {world_center[2]:+7.4f}]")
            print(f"   Position (mm): [{world_center[0]*1000:+7.0f}, {world_center[1]*1000:+7.0f}, {world_center[2]*1000:+7.0f}]")
            print(f"   Frame: Gravity-aligned (Z=up, floor detection applied)")
            
            # Print gravity alignment info
            pipeline_metadata = cluster.get('pipeline_metadata', {})
            if pipeline_metadata.get('gravity_aligned', False):
                print(f"   Gravity alignment: ACTIVE")
                floor_normal = pipeline_metadata.get('floor_normal')
                if floor_normal is not None:
                    print(f"   Floor normal: [{floor_normal[0]:+6.3f}, {floor_normal[1]:+6.3f}, {floor_normal[2]:+6.3f}]")
            else:
                print(f"   Gravity alignment: NOT AVAILABLE")
        
        # 3. Robot Frame Information (Transformed Coordinates)
        print("\n3. ROBOT FRAME (Transformed Coordinates):")
        robot_position = robot_coords['robot_position']  # [x, y, z] in mm
        print(f"   Position (mm): [{robot_position[0]:+7.0f}, {robot_position[1]:+7.0f}, {robot_position[2]:+7.0f}]")
        print(f"   Distance from base: {robot_coords['distance_from_base']:6.0f}mm")
        print(f"   Frame: X=forward, Y=left, Z=up (myCobot 280)")
        print(f"   Valid position: {'YES' if robot_coords['is_valid'] else 'NO'}")
        
        # Enhanced 6DOF information if available
        if robot_coords.get('has_orientation', False):
            robot_orientation = robot_coords.get('robot_orientation_degrees', [0, 0, 0])
            print(f"   Orientation (°): [{robot_orientation[0]:+6.1f}, {robot_orientation[1]:+6.1f}, {robot_orientation[2]:+6.1f}]")
            print(f"   6DOF pose: AVAILABLE")
            
            # Floor information in robot frame
            floor_normal_robot = robot_coords.get('floor_normal_robot')
            if floor_normal_robot is not None:
                print(f"   Floor normal (robot): [{floor_normal_robot[0]:+6.3f}, {floor_normal_robot[1]:+6.3f}, {floor_normal_robot[2]:+6.3f}]")
        else:
            print(f"   6DOF pose: NOT AVAILABLE (position-only)")
        
        # 4. Coordinate Transformation Details
        print("\n4. COORDINATE TRANSFORMATION DETAILS:")
        print(f"   Matrix source: {robot_coords.get('matrix_source', 'unknown')}")
        print(f"   Coordinate system: {robot_coords.get('coordinate_system', 'unknown')}")
        print(f"   Transformation time: {robot_coords.get('transformation_time', 0)*1000:.2f}ms")
        
        # 5. Cluster Properties
        print("\n5. CLUSTER PROPERTIES:")
        print(f"   Cluster size: {cluster.get('size', 0)} points")
        print(f"   Detection method: {cluster.get('detection_method', 'unknown')}")
        print(f"   Dimensions (m): [{cluster.get('dimensions', [0,0,0])[0]:.3f}, {cluster.get('dimensions', [0,0,0])[1]:.3f}, {cluster.get('dimensions', [0,0,0])[2]:.3f}]")
        
        # Update position history for stability analysis
        self._update_position_history(camera_center, world_center if 'center_world_frame' in cluster else None, robot_position)
        
        # 6. Position Stability Analysis
        print("\n6. POSITION STABILITY (last 10 detections):")
        self._print_stability_analysis()
        
        # 7. Robot Command Suggestion
        if robot_coords['is_valid'] and 'suggested_pose' in robot_coords:
            pose_suggestion = robot_coords['suggested_pose']
            print("\n7. ROBOT COMMAND SUGGESTION:")
            print(f"   Target: {pose_suggestion.get('description', 'unknown')}")
            approach_pos = pose_suggestion.get('position', [0, 0, 0])
            approach_orient = pose_suggestion.get('orientation', [0, 0, 0])
            print(f"   Approach position: [{approach_pos[0]:+6.0f}, {approach_pos[1]:+6.0f}, {approach_pos[2]:+6.0f}]mm")
            print(f"   Approach orientation: [{approach_orient[0]:+6.1f}, {approach_orient[1]:+6.1f}, {approach_orient[2]:+6.1f}]°")
        
        print("="*80)
        print(f"Press 'q' to quit, 'p' to pause, 'h' for help")
    
    def _update_position_history(self, camera_pos: np.ndarray, world_pos: Optional[np.ndarray], robot_pos: np.ndarray):
        """Update position history for stability analysis."""
        self.position_history['camera_frame'].append(camera_pos.copy())
        self.position_history['robot_frame'].append(robot_pos.copy())
        
        if world_pos is not None:
            self.position_history['world_frame'].append(world_pos.copy())
        
        # Limit history size
        for frame_type in self.position_history:
            if len(self.position_history[frame_type]) > self.max_history_size:
                self.position_history[frame_type].pop(0)
    
    def _print_stability_analysis(self):
        """Print position stability analysis."""
        for frame_type, positions in self.position_history.items():
            if len(positions) < 2:
                continue
            
            # Calculate standard deviation for each axis
            positions_array = np.array(positions)
            std_devs = np.std(positions_array, axis=0)
            
            if frame_type == 'robot_frame':
                # Robot frame is in mm
                print(f"   {frame_type:12}: σ = [{std_devs[0]:5.1f}, {std_devs[1]:5.1f}, {std_devs[2]:5.1f}]mm (n={len(positions)})")
            else:
                # Camera and world frames are in meters
                print(f"   {frame_type:12}: σ = [{std_devs[0]*1000:5.1f}, {std_devs[1]*1000:5.1f}, {std_devs[2]*1000:5.1f}]mm (n={len(positions)})")
    
    def _handle_keyboard_input(self, key: int) -> bool:
        """Enhanced keyboard input handling."""
        # Handle parent keyboard inputs first
        continue_running = super()._handle_keyboard_input(key)
        
        if not continue_running:
            return False
        
        # Additional keyboard commands for position printing
        if key == ord('d'):  # Toggle detailed position printing
            self.print_detailed_positions = not self.print_detailed_positions
            print(f"Detailed position printing {'enabled' if self.print_detailed_positions else 'disabled'}")
        
        elif key == ord('c'):  # Clear position history
            for frame_type in self.position_history:
                self.position_history[frame_type].clear()
            print("Position history cleared")
        
        elif key == ord('t'):  # Toggle print timing
            if self.position_print_interval == 0.5:
                self.position_print_interval = 1.0
                print("Position print interval: 1.0 seconds")
            elif self.position_print_interval == 1.0:
                self.position_print_interval = 0.1
                print("Position print interval: 0.1 seconds (fast)")
            else:
                self.position_print_interval = 0.5
                print("Position print interval: 0.5 seconds (default)")
        
        return True
    
    def start_tracking(self) -> bool:
        """Enhanced start tracking with additional controls."""
        if not super().start_tracking():
            return False
        
        print("\nAdditional Controls for Position Printing:")
        print("  'd' - Toggle detailed position printing")
        print("  'c' - Clear position history")
        print("  't' - Cycle print timing (0.5s/1.0s/0.1s)")
        print()
        
        return True


def main():
    """Main function for position printing demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LiDAR Position Printer - Robot Frame Coordinates")
    
    parser.add_argument('--config', type=str, default='enhanced_lidar_demo',
                       choices=['enhanced_lidar_precise', 'enhanced_lidar_demo', 'enhanced_lidar_debug'],
                       help='Enhanced LiDAR configuration preset')
    
    parser.add_argument('--bag', type=str, default=None,
                       help='RealSense bag file for replay (optional)')
    
    parser.add_argument('--matrix', type=str, default='charuco_eye_in_hand_transform.npy',
                       help='Hand-eye transformation matrix file')
    
    parser.add_argument('--no-robot', action='store_true',
                       help='Disable robot control (detection only)')
    
    parser.add_argument('--print-interval', type=float, default=0.5,
                       help='Position print interval in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    # Create enhanced tracker with position printing
    tracker = LiDARPositionPrinter(
        config_name=args.config,
        bag_file=args.bag,
        transformation_matrix_file=args.matrix,
        enable_robot_control=not args.no_robot
    )
    
    # Set print interval
    tracker.position_print_interval = args.print_interval
    
    print(f"\nStarting LiDAR Position Printer...")
    print(f"Configuration: {args.config}")
    print(f"Position print interval: {args.print_interval}s")
    if args.bag:
        print(f"Bag file: {args.bag}")
    print()
    
    try:
        if not tracker.start_tracking():
            print("Failed to start position tracking system")
            return 1
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())