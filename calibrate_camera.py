#!/usr/bin/env python3
"""
Camera-Robot Calibration Script

This script helps calibrate the transformation matrix between the RealSense camera
coordinate system and the robot coordinate system. It moves the robot to a known
position, guides the user to place an object below the robot, and calculates
the transformation matrix based on the detected object position.

Usage:
    python calibrate_camera.py
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

class CameraRobotCalibrator:
    def __init__(self):
        self.robot = None
        self.camera = None
        self.camera_pipeline_active = False
        self.transformation_matrix = None
        self.log_file = None
        
        # Multi-point calibration positions (within 280mm radius limit)
        self.calibration_positions = [
            [250, 0, 60, 0, 180, 0],     # Center reference
            [180, -100, 60, 0, 180, 0],  # Left side
            [180, 100, 60, 0, 180, 0],   # Right side  
            [200, 0, 60, 0, 180, 0],     # Higher position
            [220, -50, 60, 0, 180, 0],   # Mixed position
        ]
        
        # Test positions for validation (different from calibration, within radius)
        self.test_positions = [
            [200, -60, 60, 0, 180, 0],   # Test 1: Left-forward
            [160, 80, 60, 0, 180, 0],    # Test 2: Right-forward
            [240, 30, 60, 0, 180, 0]     # Test 3: Back-right
        ]

    def log_message(self, message):
        """Writes a message to the log file and prints to console."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {message}"
        print(log_entry)
        if self.log_file:
            self.log_file.write(log_entry + "\n")
            self.log_file.flush() # Ensure it's written immediately
        
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
    
    def move_robot_to_calibration_position(self):
        """Move robot to calibration position and guide user through object placement"""
        print(f"Moving robot to calibration position: {self.calibration_position}")
        try:
            self.robot.send_coords(self.calibration_position, 50, 0)
            print("Waiting for robot to reach position...")
            time.sleep(5)
            
            # Verify position
            current_pos = self.robot.get_coords()
            print(f"Current robot position: {current_pos}")
            
            # Guide user to place object
            print("\n" + "="*60)
            print("STEP 1: OBJECT PLACEMENT")
            print("="*60)
            print("1. Place your calibration object DIRECTLY BELOW the robot end effector")
            print("2. Make sure the object is clearly visible and well-lit")
            print("3. Ensure the object is on a stable surface")
            print("4. Press ENTER when the object is positioned correctly...")
            input()  # Wait for user confirmation
            
            # Move robot to home position to get it out of camera view
            print("\nMoving robot to home position to clear camera view...")
            self.robot.send_angles([0, 0, 0, 0, 0, 0], 50)
            print("Robot is moving to home position...")
            # Don't wait - camera will be ready when user starts alignment
            
            return True
        except Exception as e:
            print(f"Failed to move robot: {e}")
            return False
    
    def capture_object_position(self):
        """Capture object position from camera"""
        print("\n" + "="*60)
        print("STEP 2: CAMERA ALIGNMENT")
        print("="*60)
        print("1. Position the camera so the calibration object appears clearly in view")
        print("2. Adjust camera angle and distance for best object visibility")
        print("3. Try to center the object in the camera view if possible")
        print("4. Press 'c' to capture the object position when ready")
        print("5. Press 'q' to quit calibration")
        print("="*60)
        
        # Start camera pipeline only if not already started
        if not self.camera_pipeline_active:
            print("Starting camera pipeline...")
            profile = self.camera.start()
            self.camera_pipeline_active = True
        else:
            print("Camera pipeline already running")
        
        # Calibrate gravity alignment if using IMU
        if self.camera.use_imu and not self.camera.alignment_calibrated:
            print("Calibrating camera alignment...")
            self.camera.calibrate_gravity_alignment()
        
        captured_position = None
        
        try:
            while True:
                # Get camera frames
                if self.camera.bag_file:
                    frames_available, frames = self.camera.pipeline.try_wait_for_frames(timeout_ms=100)
                    if not frames_available:
                        continue
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
                    
                    # Display instructions on image
                    if color_image is not None:
                        instruction_text = [
                            "STEP 2: CAMERA ALIGNMENT",
                            "1. Adjust camera position/angle",
                            "2. Ensure object is clearly visible", 
                            "3. Press 'c' to capture position",
                            "4. Press 'q' to quit"
                        ]
                        
                        for i, text in enumerate(instruction_text):
                            color = (0, 255, 0) if i == 0 else (255, 255, 255)
                            cv2.putText(color_image, text, (10, 30 + i * 25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Show detected objects
                        if clusters:
                            cv2.putText(color_image, f"Objects detected: {len(clusters)}", 
                                      (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            
                            # Show positions of detected objects
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
                    cv2.namedWindow('Camera Calibration', cv2.WINDOW_AUTOSIZE)
                    cv2.imshow('Camera Calibration', combined)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Calibration cancelled by user")
                        break
                    elif key == ord('c'):
                        if clusters:
                            # Use the first detected cluster as the calibration object
                            cluster = clusters[0]
                            if 'center_mm' in cluster:
                                # center_mm is in format [x, y, z] = [side, height, forward]
                                # Convert to [forward, side, height] for consistency
                                center_mm = cluster['center_mm']
                                captured_position = [center_mm[2], center_mm[0], center_mm[1]]  # [forward, side, height]
                            else:
                                # center is in format [x, y, z] = [side, height, forward]  
                                # Convert to [forward, side, height] for consistency
                                center = cluster['center'] * 1000
                                captured_position = [center[2], center[0], center[1]]  # [forward, side, height]
                            
                            print(f"\nObject position captured: ({captured_position[0]:.0f}, {captured_position[1]:.0f}, {captured_position[2]:.0f}) mm")
                            print("Camera coordinates: (forward, side, height)")
                            
                            print("\n" + "="*60)
                            print("STEP 3: CALIBRATION COMPLETE")
                            print("="*60)
                            print("Object successfully detected and position recorded!")
                            break
                        else:
                            print("No objects detected! Please ensure object is visible and try again.")
        
        except Exception as e:
            print(f"Error during capture: {e}")
        
        finally:
            # Don't stop pipeline here - we need it for validation tests
            cv2.destroyAllWindows()
        
        return captured_position
    
    def calculate_transformation_matrix_multipoint(self, camera_points, robot_points):
        """Calculate transformation matrix using multiple calibration points with least-squares fitting"""
        if len(camera_points) < 3:
            self.log_message("Error: Need at least 3 calibration points for robust transformation")
            return None
            
        # Convert to numpy arrays
        camera_pts = np.array(camera_points)  # Nx3
        robot_pts = np.array(robot_points)    # Nx3
        
        self.log_message(f"Calculating transformation from {len(camera_points)} calibration points:")
        for i, (cam, rob) in enumerate(zip(camera_points, robot_points)):
            self.log_message(f"  Point {i+1}: Camera {cam} -> Robot {rob}")
        
        # Add homogeneous coordinates for camera points
        camera_homo = np.hstack([camera_pts, np.ones((len(camera_pts), 1))])  # Nx4
        
        # Solve for transformation matrix using least squares
        # robot_pts = camera_homo @ T.T (where T is the transformation matrix)
        # So T.T = pinv(camera_homo) @ robot_pts
        try:
            T_transpose = np.linalg.pinv(camera_homo) @ robot_pts  # 4x3
            T_top = T_transpose.T  # 3x4
            
            # Create full 4x4 homogeneous transformation matrix
            self.transformation_matrix = np.vstack([T_top, [0, 0, 0, 1]])
            
            self.log_message(f"Multi-point transformation matrix calculated:")
            self.log_message(f"Matrix:\n{self.transformation_matrix}")
            
            # Calculate fitting error for each point
            total_error = 0
            for i, (cam_pt, rob_pt) in enumerate(zip(camera_points, robot_points)):
                cam_homo = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1])
                transformed = np.dot(self.transformation_matrix, cam_homo)[:3]
                error = np.linalg.norm(transformed - np.array(rob_pt))
                total_error += error
                self.log_message(f"  Point {i+1} fitting error: {error:.2f} mm")
            
            avg_fitting_error = total_error / len(camera_points)
            self.log_message(f"Average fitting error: {avg_fitting_error:.2f} mm")
            
            print(f"\nMulti-point transformation matrix calculated successfully!")
            print(f"Average fitting error: {avg_fitting_error:.2f} mm")
            print(f"Transformation matrix:")
            print(self.transformation_matrix)
            
            return self.transformation_matrix
            
        except np.linalg.LinAlgError as e:
            self.log_message(f"Error: Failed to calculate transformation matrix: {e}")
            print(f"Error: Failed to calculate transformation matrix: {e}")
            return None
    
    def calculate_transformation_matrix_legacy(self, camera_position, robot_position):
        """Calculate transformation matrix from camera to robot coordinates"""
        # Convert positions to numpy arrays
        camera_pos = np.array(camera_position)  # [forward, side, height] in mm
        robot_pos = np.array(robot_position[:3])  # [forward, side, height] in mm
        
        self.log_message("Calculating transformation matrix with the following data:")
        self.log_message(f"- Robot Position (Expected): {robot_pos.tolist()}")
        self.log_message(f"- Camera Position (Detected): {camera_pos.tolist()}")
        
        print(f"\nCalculating transformation matrix:")
        print(f"Camera position: ({camera_pos[0]:.0f}, {camera_pos[1]:.0f}, {camera_pos[2]:.0f}) mm")
        print(f"Robot position:  ({robot_pos[0]:.0f}, {robot_pos[1]:.0f}, {robot_pos[2]:.0f}) mm")
        
        # Calculate offset (robot - camera)
        offset = robot_pos - camera_pos
        self.log_message(f"- Calculated Offset (Robot - Camera): {offset.tolist()}")
        print(f"Calculated offset: ({offset[0]:.0f}, {offset[1]:.0f}, {offset[2]:.0f}) mm")
        
        # Create transformation matrix
        # Based on the coordinate system described in CLAUDE.md:
        # Robot_forward = Camera_forward + offset[0]
        # Robot_side = -(Camera_side) + offset[1]  # Note the sign flip
        # Robot_height = Camera_height + offset[2]
        
        self.transformation_matrix = np.array([
            [1,  0,  0, offset[0]],  # Forward: direct mapping
            [0, -1,  0, offset[1]],  # Side: sign flip + offset
            [0,  0,  1, offset[2]],  # Height: direct mapping
            [0,  0,  0, 1]           # Homogeneous coordinate
        ])
        
        print(f"\nTransformation matrix (Camera -> Robot):")
        print(self.transformation_matrix)
        self.log_message(f"- Calculated Transformation Matrix:\n{self.transformation_matrix}")
        
        return self.transformation_matrix
    
    def save_transformation_matrix(self, filename="camera_robot_transform.npy"):
        """Save transformation matrix to file"""
        if self.transformation_matrix is not None:
            filepath = os.path.join(os.path.dirname(__file__), filename)
            np.save(filepath, self.transformation_matrix)
            print(f"\nTransformation matrix saved to: {filepath}")
            return True
        return False
    
    def test_transformation(self, camera_position):
        """Test the transformation with the calibration data"""
        if self.transformation_matrix is None:
            return
        
        # Convert camera position to homogeneous coordinates
        camera_homogeneous = np.array([camera_position[0], camera_position[1], camera_position[2], 1])
        
        # Apply transformation
        robot_transformed = np.dot(self.transformation_matrix, camera_homogeneous)
        
        self.log_message("Running initial transformation test:")
        self.log_message(f"- Camera Position (Input): {camera_position}")
        self.log_message(f"- Transformed to Robot Coords: {robot_transformed[:3].tolist()}")
        self.log_message(f"- Expected Robot Coords: {self.calibration_position[:3]}")
        
        print(f"\nTransformation Test:")
        print(f"Camera position: ({camera_position[0]:.0f}, {camera_position[1]:.0f}, {camera_position[2]:.0f}) mm")
        print(f"Transformed to:  ({robot_transformed[0]:.0f}, {robot_transformed[1]:.0f}, {robot_transformed[2]:.0f}) mm")
        print(f"Expected robot:  ({self.calibration_position[0]:.0f}, {self.calibration_position[1]:.0f}, {self.calibration_position[2]:.0f}) mm")
        
        # Calculate error
        expected = np.array(self.calibration_position[:3])
        actual = robot_transformed[:3]
        error = np.linalg.norm(actual - expected)
        print(f"Transformation error: {error:.1f} mm")
    
    def run_multipoint_calibration(self):
        """Run multi-point calibration process"""
        print("\n" + "="*60)
        print("MULTI-POINT CALIBRATION PROCESS")
        print("="*60)
        print(f"This will collect {len(self.calibration_positions)} calibration points")
        print("for accurate coordinate transformation calculation.")
        print("="*60)
        
        camera_points = []
        robot_points = []
        
        for i, cal_pos in enumerate(self.calibration_positions, 1):
            print(f"\n--- CALIBRATION POINT {i}/{len(self.calibration_positions)} ---")
            self.log_message(f"--- Starting Calibration Point {i}/{len(self.calibration_positions)} ---")
            
            # Move robot to calibration position
            print(f"Moving robot to position {i}: {cal_pos}")
            self.log_message(f"Robot position {i}: {cal_pos}")
            
            try:
                self.robot.send_coords(cal_pos, 50, 0)
                print("Waiting for robot to reach position...")
                time.sleep(4)
                
                # Verify position
                current_pos = self.robot.get_coords()
                print(f"Robot reached: {current_pos}")
                self.log_message(f"Robot actual position: {current_pos}")
                
                # Guide user to place object
                print(f"\nSTEP {i}: OBJECT PLACEMENT")
                print("1. Place your calibration object DIRECTLY BELOW the robot end effector")
                print("2. Make sure the object is clearly visible and well-lit")
                print("3. Press ENTER when the object is positioned correctly...")
                input()
                
                # Move robot to home position to clear camera view
                print("Moving robot to home position to clear camera view...")
                self.robot.send_angles([0, 0, 0, 0, 0, 0], 50)
                
                # Capture object position from camera
                print(f"Capturing object position for calibration point {i}...")
                camera_position = self.capture_object_position()
                
                if camera_position is not None:
                    camera_points.append(camera_position)
                    robot_points.append(cal_pos[:3])  # Only XYZ coordinates
                    self.log_message(f"Point {i} captured: Camera {camera_position} -> Robot {cal_pos[:3]}")
                    print(f"Calibration point {i} captured successfully!")
                else:
                    print(f"Failed to capture calibration point {i}")
                    self.log_message(f"Failed to capture calibration point {i}")
                    
                    # Ask user if they want to retry this point
                    retry = input(f"Retry calibration point {i}? [Y/n]: ").strip().lower()
                    if retry in ['', 'y', 'yes']:
                        i -= 1  # Retry this point
                        continue
                    else:
                        print(f"Skipping calibration point {i}")
                        
            except Exception as e:
                print(f"Error at calibration point {i}: {e}")
                self.log_message(f"Error at calibration point {i}: {e}")
                
                # Ask user if they want to retry
                retry = input(f"Retry calibration point {i}? [Y/n]: ").strip().lower()
                if retry in ['', 'y', 'yes']:
                    i -= 1  # Retry this point
                    continue
        
        # Check if we have enough points
        if len(camera_points) < 3:
            print(f"\nInsufficient calibration points! Got {len(camera_points)}, need at least 3.")
            self.log_message(f"Calibration failed: Only {len(camera_points)} points collected, need at least 3")
            return None
        
        print(f"\nCollected {len(camera_points)} calibration points!")
        self.log_message(f"Successfully collected {len(camera_points)} calibration points")
        
        # Calculate transformation matrix using all points
        transformation_matrix = self.calculate_transformation_matrix_multipoint(camera_points, robot_points)
        
        return transformation_matrix
    
    def run_validation_tests(self):
        """Run 3-point validation test to verify calibration accuracy"""
        print("\n" + "="*60)
        print("STEP 4: CALIBRATION VALIDATION")
        print("="*60)
        print("Running 3-point validation test to verify accuracy...")
        print("You will need to move the object to match robot positions.")
        
        validation_errors = []
        
        for i, test_pos in enumerate(self.test_positions, 1):
            print(f"\n--- VALIDATION TEST {i}/3 ---")
            self.log_message(f"--- Starting Validation Test {i}/3 ---")
            
            # Move robot to test position
            print(f"Moving robot to test position {i}: {test_pos}")
            try:
                self.robot.send_coords(test_pos, 50, 0)
                print("Waiting for robot to reach test position...")
                time.sleep(4)
                
                current_pos = self.robot.get_coords()
                print(f"Robot reached: {current_pos}")
                
                # Guide user to move object
                print(f"\nPlease move the calibration object DIRECTLY BELOW the robot end effector")
                print("Press ENTER when the object is positioned correctly...")
                input()
                
                # Move robot away to clear camera view
                print("Moving robot to home position and detecting object...")
                self.robot.send_angles([0, 0, 0, 0, 0, 0], 50)
                # Don't wait - start detection immediately while robot is moving
                
                # Capture object position from camera (interactive)
                # This reuses the same robust capture method as the initial calibration
                camera_position = self.capture_object_position()
                
                if camera_position is not None:
                    self.log_message(f"Validation {i}: Detected camera position: {camera_position}")
                    # Transform camera coordinates to robot coordinates
                    camera_homogeneous = np.array([camera_position[0], camera_position[1], camera_position[2], 1])
                    robot_transformed = np.dot(self.transformation_matrix, camera_homogeneous)
                    
                    # Calculate error
                    expected = np.array(test_pos[:3])  # Only position, not angles
                    actual = robot_transformed[:3]
                    error = np.linalg.norm(actual - expected)
                    validation_errors.append(error)
                    
                    self.log_message(f"Validation {i}: Expected robot position: {expected.tolist()}")
                    self.log_message(f"Validation {i}: Transformed to robot position: {actual.tolist()}")
                    self.log_message(f"Validation {i}: Calculated error: {error:.2f} mm")
                    
                    print(f"Expected robot position: ({expected[0]:.1f}, {expected[1]:.1f}, {expected[2]:.1f}) mm")
                    print(f"Camera detected position: ({camera_position[0]:.1f}, {camera_position[1]:.1f}, {camera_position[2]:.1f}) mm")
                    print(f"Transformed position: ({actual[0]:.1f}, {actual[1]:.1f}, {actual[2]:.1f}) mm")
                    print(f"Validation error: {error:.1f} mm")
                    
                    if error < 50:
                        print("Test PASSED (error < 50mm)")
                    else:
                        print("Test FAILED (error >= 50mm)")
                else:
                    print("Test FAILED (could not detect object)")
                    validation_errors.append(999)  # Large error for failed detection
                    
            except Exception as e:
                print(f"Test FAILED (robot movement error): {e}")
                validation_errors.append(999)
        
        # Summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        passed_tests = sum(1 for error in validation_errors if error < 50)
        avg_error = np.mean([e for e in validation_errors if e < 999])
        
        self.log_message("--- Validation Summary ---")
        self.log_message(f"- Tests Passed: {passed_tests}/3")
        if len([e for e in validation_errors if e < 999]) > 0:
            self.log_message(f"- Average Error: {avg_error:.2f} mm")
        
        print(f"Tests passed: {passed_tests}/3")
        if len([e for e in validation_errors if e < 999]) > 0:
            print(f"Average error: {avg_error:.1f} mm")
        
        if passed_tests >= 2:
            print("CALIBRATION VALIDATION SUCCESSFUL!")
            print("The calibration is accurate enough for tracking.")
            return True
        else:
            print("CALIBRATION VALIDATION FAILED!")
            print("Consider re-running calibration for better accuracy.")
            return False
    def run_calibration(self):
        """Main calibration process"""
        print("Starting Camera-Robot Calibration Process")
        print("="*50)
        
        # Open log file
        log_filepath = os.path.join(os.path.dirname(__file__), 'calibration_report.log')
        self.log_file = open(log_filepath, 'w')
        self.log_message("Calibration process started.")

        try:
            
            self.log_message("Initializing robot connection...")
            # Initialize systems
            if not self.initialize_robot():
                return False
            
            self.log_message("Initializing camera system...")
            if not self.initialize_camera():
                return False
            
            # Run multi-point calibration process
            self.log_message("Starting multi-point calibration process")
            transformation_matrix = self.run_multipoint_calibration()
            
            if transformation_matrix is None:
                print("Multi-point calibration failed!")
                self.log_message("Multi-point calibration failed")
                return False
            
            # Skip single-point transformation test since we used multi-point calibration
            self.log_message("Multi-point calibration completed successfully")
            
            # Save transformation matrix
            if self.save_transformation_matrix():
                print("\n" + "="*50)
                print("CALIBRATION MATRIX SAVED!")
                print("="*50)
                
                # Ask user if they want to run validation tests
                print("\n" + "Would you like to run validation tests? (recommended)")
                print("This will test the calibration accuracy with 3 different positions.")
                response = input("Run validation tests? [Y/n]: ").strip().lower()
                
                if response in ['', 'y', 'yes'] :
                    validation_success = self.run_validation_tests()
                    
                    # Stop camera pipeline after validation
                    self.camera.pipeline.stop()
                    self.camera_pipeline_active = False
                    
                    if validation_success:
                        print("\n" + "="*50)
                        print("CALIBRATION FULLY COMPLETED!")
                        print("You can now use track_object.py for real-time tracking.")
                        print("="*50)
                        return True
                    else:
                        print("\n" + "="*50)
                        print("CALIBRATION COMPLETED WITH WARNINGS!")
                        print("You can still use track_object.py, but accuracy may be limited.")
                        print("Consider re-running calibration if tracking accuracy is poor.")
                        print("="*50)
                        return True
                else:
                    print("\n" + "="*50)
                    print("CALIBRATION COMPLETED!")
                    print("Validation tests skipped. You can now use track_object.py.")
                    print("="*50)
                    # Stop camera pipeline since validation was skipped
                    self.camera.pipeline.stop()
                    self.camera_pipeline_active = False
                    return True
            else:
                print("Failed to save transformation matrix")
                return False
        finally:
            if self.log_file:
                self.log_file.close()

def main():
    calibrator = CameraRobotCalibrator()
    
    try:
        success = calibrator.run_calibration()
        if success:
            print("\nCalibration process completed successfully!")
        else:
            print("\nCalibration process failed!")
    except KeyboardInterrupt:
        print("\nCalibration interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error during calibration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()