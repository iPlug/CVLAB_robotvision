#!/usr/bin/env python3
"""
Real-time Object Tracking Script

This script uses the calibrated transformation matrix to track objects detected
by the camera and command the robot to point at them. It runs continuously,
updating every 3 seconds with safety constraints.

Usage:
    python track_object.py

Requirements:
    - Run calibrate_camera.py first to generate camera_robot_transform.npy
    - Robot and camera must be in the same positions as during calibration
"""

import sys
import os
import time
import numpy as np
import cv2

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'myCobot'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'realtime'))

from port_setup import setup
from realsense_bev_cluster import BEVClusteringVisualizer

class ObjectTracker:
    def __init__(self):
        self.robot = None
        self.camera = None
        self.transformation_matrix = None
        self.min_height = 100  # Minimum height in mm for safety
        self.pointing_angles = [0, 180, 0]  # Fixed rotation angles for pointing down
        self.update_interval = 1  # Update every 1 second normally
        self.last_update_time = 0
        self.last_command_had_object = True  # Track if last command found an object
        
    def load_transformation_matrix(self, filename="camera_robot_transform.npy"):
        """Load the calibrated transformation matrix"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        try:
            self.transformation_matrix = np.load(filepath)
            print("Transformation matrix loaded successfully:")
            print(self.transformation_matrix)
            return True
        except FileNotFoundError:
            print(f"Error: Transformation matrix file '{filepath}' not found!")
            print("Please run calibrate_camera.py first to generate the calibration data.")
            return False
        except Exception as e:
            print(f"Error loading transformation matrix: {e}")
            return False
    
    def initialize_robot(self):
        """Initialize robot connection"""
        print("Initializing robot connection...")
        try:
            self.robot = setup()
            print("Robot connected successfully!")
            return True
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            return False
    
    def initialize_camera(self):
        """Initialize camera system"""
        print("Initializing camera system...")
        try:
            self.camera = BEVClusteringVisualizer(realtime_mode=True)
            print("Camera system initialized successfully!")
            return True
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def transform_camera_to_robot(self, camera_position):
        """Transform camera coordinates to robot coordinates"""
        if self.transformation_matrix is None:
            return None
        
        # Convert to homogeneous coordinates
        camera_homogeneous = np.array([camera_position[0], camera_position[1], camera_position[2], 1])
        
        # Apply transformation
        robot_homogeneous = np.dot(self.transformation_matrix, camera_homogeneous)
        
        # Return 3D coordinates
        return robot_homogeneous[:3]
    
    def is_safe_position(self, robot_coords):
        """Check if robot position is safe"""
        # Check height constraint
        if robot_coords[2] < self.min_height:
            return False, f"Height {robot_coords[2]:.0f}mm < minimum {self.min_height}mm"
        
        # Add more safety checks as needed
        # For example, workspace bounds:
        if robot_coords[0] < 100 or robot_coords[0] > 400:  # Forward range
            return False, f"Forward position {robot_coords[0]:.0f}mm out of safe range"
        
        if abs(robot_coords[1]) > 200:  # Side range
            return False, f"Side position {robot_coords[1]:.0f}mm out of safe range"
        
        return True, "Position is safe"
    
    def move_robot_to_object(self, robot_coords):
        """Move robot to point at object with safety constraints"""
        # Clamp coordinates to safe values instead of canceling movement
        safe_coords = robot_coords.copy()
        adjustments_made = []
        
        # Clamp height to safe minimum
        if safe_coords[2] < self.min_height:
            safe_coords[2] = self.min_height
            adjustments_made.append(f"Height: {robot_coords[2]:.0f}mm → {safe_coords[2]:.0f}mm")
        
        # Clamp forward position to workspace bounds
        if safe_coords[0] < 100:
            safe_coords[0] = 100
            adjustments_made.append(f"Forward: {robot_coords[0]:.0f}mm → {safe_coords[0]:.0f}mm")
        elif safe_coords[0] > 400:
            safe_coords[0] = 400
            adjustments_made.append(f"Forward: {robot_coords[0]:.0f}mm → {safe_coords[0]:.0f}mm")
        
        # Clamp side position to workspace bounds
        if safe_coords[1] < -200:
            safe_coords[1] = -200
            adjustments_made.append(f"Side: {robot_coords[1]:.0f}mm → {safe_coords[1]:.0f}mm")
        elif safe_coords[1] > 200:
            safe_coords[1] = 200
            adjustments_made.append(f"Side: {robot_coords[1]:.0f}mm → {safe_coords[1]:.0f}mm")
        
        # Print safety adjustments if any were made
        if adjustments_made:
            print(f"Safety adjustments made: {', '.join(adjustments_made)}")
        
        # Create target coordinates with fixed pointing angles
        target_coords = [
            safe_coords[0],  # Forward
            safe_coords[1],  # Side  
            safe_coords[2],  # Height
            self.pointing_angles[0],  # Roll
            self.pointing_angles[1],  # Pitch (pointing down)
            self.pointing_angles[2]   # Yaw
        ]
        
        try:
            print(f"Moving robot to: {target_coords}")
            self.robot.send_coords(target_coords, 90, 0)  # Speed 50, mode 0
            return True
        except Exception as e:
            print(f"Failed to move robot: {e}")
            return False
    
    
    def update_display(self):
        """Update display every frame for smooth visualization"""
        try:
            # Get camera frames
            if self.camera.bag_file:
                frames_available, frames = self.camera.pipeline.try_wait_for_frames(timeout_ms=10)
                if not frames_available:
                    return []
            else:
                frames = self.camera.pipeline.wait_for_frames()
            
            # Get 3D points and process
            vertices, texcoords, color_image = self.camera.get_3d_points(frames)
            
            if vertices is not None and len(vertices) > 0:
                # Create BEV with clustering
                result = self.camera.project_to_2d_with_clustering(vertices, color_image)
                if len(result) == 3:
                    clustered_bev, clusters, point_cloud_view = result
                else:
                    clustered_bev, clusters = result
                    point_cloud_view = None
                
                # Display tracking info
                if color_image is not None:
                    tracking_text = [
                        "OBJECT TRACKING MODE",
                        f"Update interval: {self.update_interval}s",
                        f"Min height: {self.min_height}mm",
                        f"Objects detected: {len(clusters)}",
                        "Press 'q' to quit"
                    ]
                    
                    for i, text in enumerate(tracking_text):
                        color = (0, 255, 0) if i == 0 else (255, 255, 255)
                        cv2.putText(color_image, text, (10, 30 + i * 25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Show current time until next update (adaptive based on last result)
                    time_since_update = time.time() - self.last_update_time
                    effective_interval = 0.5 if not self.last_command_had_object else self.update_interval
                    time_until_next = max(0, effective_interval - time_since_update)
                    status_text = f"Next update: {time_until_next:.1f}s ({'fast' if not self.last_command_had_object else 'normal'})"
                    cv2.putText(color_image, status_text, 
                              (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # Show object positions
                    for i, cluster in enumerate(clusters):
                        if 'center_mm' in cluster:
                            center_mm = cluster['center_mm']
                            pos_text = f"Obj{i+1}: ({center_mm[2]:.0f}, {center_mm[0]:.0f}, {center_mm[1]:.0f})"
                        else:
                            center = cluster['center']
                            pos_text = f"Obj{i+1}: ({center[2]*1000:.0f}, {center[0]*1000:.0f}, {center[1]*1000:.0f})"
                        
                        cv2.putText(color_image, pos_text, (10, 210 + i * 25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Stack images for display
                if color_image is not None:
                    if point_cloud_view is not None:
                        # Show all three views: Color | BEV | Point Cloud
                        height = color_image.shape[0]
                        clustered_bev_resized = cv2.resize(clustered_bev, (height, height))
                        point_cloud_resized = cv2.resize(point_cloud_view, (height, height))
                        combined = np.hstack((color_image, clustered_bev_resized, point_cloud_resized))
                    else:
                        # Fallback to two views
                        combined = np.hstack((color_image, cv2.resize(clustered_bev, (color_image.shape[1], color_image.shape[0]))))
                else:
                    combined = clustered_bev
                
                # Display
                cv2.namedWindow('Object Tracking', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Object Tracking', combined)
                
                return clusters
            
        except Exception as e:
            print(f"Error updating display: {e}")
        
        return []
    
    def process_robot_movement(self, clusters):
        """Process robot movement with adaptive timing"""
        current_time = time.time()
        
        # Adaptive interval: shorter if no object was found last time
        effective_interval = 0.5 if not self.last_command_had_object else self.update_interval
        
        # Check if it's time for a robot update
        if current_time - self.last_update_time < effective_interval:
            return  # Not time for robot update yet
        
        print(f"\n{'='*60}")
        print(f"Robot movement update - {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        if not clusters:
            print("No objects detected")
            self.last_command_had_object = False  # Mark that no object was found
            self.last_update_time = current_time
            return
        
        print(f"Detected {len(clusters)} objects:")
        
        # Find the closest object (based on distance from camera)
        closest_cluster = None
        min_distance = float('inf')
        
        for i, cluster in enumerate(clusters):
            if 'center_mm' in cluster:
                # center_mm is in format [x, y, z] = [side, height, forward]
                # Convert to [forward, side, height] for consistency
                center_mm = cluster['center_mm']
                camera_pos = [center_mm[2], center_mm[0], center_mm[1]]  # [forward, side, height]
            else:
                # center is in format [x, y, z] = [side, height, forward]
                # Convert to [forward, side, height] for consistency
                center = cluster['center'] * 1000
                camera_pos = [center[2], center[0], center[1]]  # [forward, side, height]
            
            # Calculate distance from camera origin
            distance = np.sqrt(camera_pos[0]**2 + camera_pos[1]**2 + camera_pos[2]**2)
            
            print(f"  Object {i+1}: Camera pos=({camera_pos[0]:.0f}, {camera_pos[1]:.0f}, {camera_pos[2]:.0f})mm, Distance={distance:.0f}mm")
            
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster
                closest_camera_pos = camera_pos
        
        if closest_cluster is not None:
            print(f"\nTargeting closest object at distance {min_distance:.0f}mm")
            
            # Transform camera coordinates to robot coordinates
            robot_coords = self.transform_camera_to_robot(closest_camera_pos)
            
            if robot_coords is not None:
                print(f"Camera coordinates: ({closest_camera_pos[0]:.0f}, {closest_camera_pos[1]:.0f}, {closest_camera_pos[2]:.0f}) mm")
                print(f"Robot coordinates:  ({robot_coords[0]:.0f}, {robot_coords[1]:.0f}, {robot_coords[2]:.0f}) mm")
                
                # Move robot to point at object
                if self.move_robot_to_object(robot_coords):
                    print("Robot movement command sent successfully")
                    self.last_command_had_object = True  # Mark that object was found
                else:
                    print("Failed to move robot")
                    self.last_command_had_object = False
            else:
                print("Failed to transform coordinates")
                self.last_command_had_object = False
        
        # Update timing
        self.last_update_time = current_time
    
    def run_tracking(self):
        """Main tracking process"""
        print("Starting Real-time Object Tracking")
        print("="*50)
        
        # Load transformation matrix
        if not self.load_transformation_matrix():
            return False
        
        # Initialize systems
        if not self.initialize_robot():
            return False
        
        if not self.initialize_camera():
            return False
        
        # Start camera pipeline
        profile = self.camera.start()
        
        # Calibrate gravity alignment if using IMU
        if self.camera.use_imu and not self.camera.alignment_calibrated:
            print("Calibrating camera alignment...")
            self.camera.calibrate_gravity_alignment()
        
        print(f"\nTracking started! Robot will point at objects every {self.update_interval} seconds.")
        print("Press 'q' to quit")
        print("-" * 50)
        
        try:
            while True:
                # Update display every frame for smooth visualization
                clusters = self.update_display()
                
                # Process robot movement only every 3 seconds
                self.process_robot_movement(clusters)
                
                # Handle user input
                key = cv2.waitKey(30) & 0xFF  # Faster response for smooth display
                if key == ord('q'):
                    print("\nTracking stopped by user")
                    break
                elif key == ord('r'):  # Reset robot to home position
                    print("Resetting robot to home position...")
                    try:
                        self.robot.send_angles([0, 0, 0, 0, 0, 0], 50)
                        print("Robot reset to home position")
                    except Exception as e:
                        print(f"Failed to reset robot: {e}")
                elif key == ord('s'):  # Stop robot movement
                    print("Stopping robot movement...")
                    try:
                        current_pos = self.robot.get_coords()
                        if current_pos:
                            self.robot.send_coords(current_pos, 0, 0)  # Stop movement
                        print("Robot stopped")
                    except Exception as e:
                        print(f"Failed to stop robot: {e}")
        
        except KeyboardInterrupt:
            print("\nTracking interrupted by user")
        except Exception as e:
            print(f"Error during tracking: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            try:
                self.camera.pipeline.stop()
                cv2.destroyAllWindows()
                print("Systems shut down successfully")
            except:
                pass
        
        return True

def main():
    tracker = ObjectTracker()
    
    print("Real-time Object Tracking System")
    print("="*40)
    print("Controls:")
    print("  'q' - Quit tracking")
    print("  'r' - Reset robot to home position")
    print("  's' - Stop robot movement")
    print("="*40)
    
    try:
        success = tracker.run_tracking()
        if success:
            print("\nTracking session completed successfully!")
        else:
            print("\nTracking session failed!")
    except KeyboardInterrupt:
        print("\nTracking interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error during tracking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()