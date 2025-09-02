#!/usr/bin/env python3
"""
Depth-Based CharUco Eye-in-Hand Calibration

This system performs eye-in-hand calibration using CharUco pattern detection
in DEPTH camera coordinates instead of RGB coordinates. This eliminates the
systematic coordinate system mismatch that causes errors when using the
calibration matrix with LiDAR-based object detection.

Key Differences from RGB-based calibration:
- CharUco pattern positions extracted from point cloud (depth coordinates)
- Direct compatibility with LiDAR object detection systems
- Maintains 90° Z-rotation correction for robot coordinate alignment
- Generates T_robot_to_depth transformation matrix

Usage:
    python charuco_depth_calibration.py
    
Controls:
    'c' - Move robot to next preset position  
    's' - Collect calibration point (robot + CharUco poses)
    'q' - Calculate eye-in-hand transformation and quit
    ESC - Exit without saving
"""

import sys
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from pathlib import Path
import time
import json
from scipy.spatial.transform import Rotation

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cv2 import aruco

# Import robot controller
try:
    from module.robot.mycobot_controller import MyCobotController
    ROBOT_AVAILABLE = True
except ImportError:
    print("Robot controller not available - will use manual input")
    ROBOT_AVAILABLE = False


class DepthBasedCharUcoCalibrator:
    """Depth-based CharUco eye-in-hand calibration system."""
    
    def __init__(self, config_path="charuco-pose-estimation"):
        """Initialize the depth-based calibration system."""
        self.config_path = Path(config_path)
        
        # CharUco board parameters (measured values)
        self.square_length = 0.039  # 39mm squares
        self.marker_length = 0.029  # 29mm markers  
        self.board_size = (7, 5)   # 7x5 grid
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = np.zeros(4)
        
        # ArUco/CharUco setup
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.board = aruco.CharucoBoard(self.board_size, self.square_length, self.marker_length, self.dictionary)
        
        # Detector setup
        self.setup_detectors()
        
        # RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Point cloud processor
        self.pc = rs.pointcloud()
        
        # Robot controller
        self.robot = None
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
        
        # Calibration data storage
        self.calibration_pairs = []
        
        # Robot movement presets
        self.load_robot_poses()
        self.current_pose_index = 0
        
        # Load camera intrinsics
        self.load_camera_intrinsics()
    
    def setup_detectors(self):
        """Setup ArUco and CharUco detectors."""
        # ArUco detector parameters
        detector_params = aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
        self.aruco_detector = aruco.ArucoDetector(self.dictionary, detector_params)
        
        # CharUco detector parameters
        charuco_params = aruco.CharucoParameters()
        if self.camera_matrix is not None:
            charuco_params.cameraMatrix = self.camera_matrix
        self.charuco_detector = aruco.CharucoDetector(self.board, charuco_params)
    
    def load_camera_intrinsics(self):
        """Load camera intrinsics for RGB detection (will convert to depth coordinates)."""
        # Load RealSense intrinsics from your existing file
        intrinsics_file = Path("realsense_intrinsics_all_resolutions.json")
        if intrinsics_file.exists():
            try:
                with open(intrinsics_file, 'r') as f:
                    intrinsics_data = json.load(f)
                
                # Use 1280x720 resolution intrinsics
                if "1280x720@30" in intrinsics_data:
                    intrinsics = intrinsics_data["1280x720@30"]["intrinsics"]
                    self.camera_matrix = np.array([
                        [intrinsics["fx"], 0.0, intrinsics["ppx"]],
                        [0.0, intrinsics["fy"], intrinsics["ppy"]],
                        [0.0, 0.0, 1.0]
                    ])
                    self.dist_coeffs = np.array(intrinsics["coeffs"][:5])
                    print(f"Loaded RealSense intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}")
                    self.setup_detectors()
                    return
            except Exception as e:
                print(f"Failed to load RealSense intrinsics: {e}")
        
        # Fallback to default values
        print("Using default camera intrinsics")
        self.camera_matrix = np.array([
            [901.46, 0.0, 654.69],
            [0.0, 901.32, 355.19],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.zeros(5)
        self.setup_detectors()
    
    def setup_camera(self):
        """Setup RealSense camera for both RGB and depth streams."""
        try:
            # Configure streams
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Get device info
            device = profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            serial_number = device.get_info(rs.camera_info.serial_number)
            
            print(f"RealSense camera initialized: {device_name} (S/N: {serial_number})")
            print("Streams: RGB 1280x720@30fps, Depth 1024x768@30fps")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize RealSense camera: {e}")
            return False
    
    def get_charuco_position_from_point_cloud(self, image, depth_frame):
        """
        Detect CharUco in RGB and extract its position from depth point cloud.
        
        This is the key innovation: RGB detection for accuracy, depth extraction 
        for coordinate system compatibility with LiDAR detection.
        
        Returns:
            numpy.ndarray: CharUco pattern position in depth camera coordinates [x, y, z] meters
            numpy.ndarray: Rotation vector and translation vector for pose
            bool: Success flag
        """
        # Step 1: Detect CharUco in RGB image
        marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(image)
        
        if marker_ids is None or len(marker_ids) == 0:
            # Only print occasionally to avoid spam
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 1
            
            if self._debug_counter % 30 == 1:  # Print every 30 frames
                print("  No ArUco markers detected in image")
            return None, None, None, False
        
        print(f"  Detected {len(marker_ids)} ArUco markers")
        
        # Detect CharUco corners
        charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(
            image, None, None, marker_corners, marker_ids
        )
        
        if charuco_corners is None or len(charuco_corners) < 6:
            if charuco_corners is not None:
                print(f"  CharUco detected {len(charuco_corners)} corners, need at least 6")
            else:
                print("  No CharUco corners detected")
            return None, None, None, False
        
        print(f"  CharUco detected {len(charuco_corners)} corners (sufficient for pose estimation)")
        
        # Step 2: Get CharUco pose in RGB camera coordinates
        obj_points, img_points = self.match_image_points_corrected(charuco_corners, charuco_ids)
        
        # Ensure we have enough points for solvePnP (minimum 6 required)
        if len(obj_points) < 6 or len(img_points) < 6:
            return None, None, None, False
        
        valid, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, self.camera_matrix, self.dist_coeffs
        )
        
        if not valid:
            return None, None, None, False
        
        # Step 3: Convert RGB-detected pattern center to depth coordinates
        # Get pattern center in image coordinates (average of detected corners)
        center_pixel = np.mean(charuco_corners.reshape(-1, 2), axis=0).astype(int)
        
        # Step 4: Extract 3D position from point cloud at pattern center
        # Generate point cloud from depth frame
        points = self.pc.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        
        # Get depth frame dimensions
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        depth_width = depth_intrinsics.width
        depth_height = depth_intrinsics.height
        
        # Map RGB pixel to depth pixel (assuming aligned frames)
        # For L515, RGB and depth should be reasonably aligned
        depth_pixel_x = int(center_pixel[0] * depth_width / 1280)
        depth_pixel_y = int(center_pixel[1] * depth_height / 720)
        
        # Clamp to valid range
        depth_pixel_x = max(0, min(depth_width - 1, depth_pixel_x))
        depth_pixel_y = max(0, min(depth_height - 1, depth_pixel_y))
        
        # Try to find valid depth in a small area around the detected center
        search_radius = 5
        best_position = None
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                test_x = max(0, min(depth_width - 1, depth_pixel_x + dx))
                test_y = max(0, min(depth_height - 1, depth_pixel_y + dy))
                pixel_index = test_y * depth_width + test_x
                
                if pixel_index < len(vertices):
                    position_3d = vertices[pixel_index]
                    
                    # Check if this is a valid depth measurement
                    if position_3d[2] > 0.1 and position_3d[2] < 2.0:  # Reasonable depth range
                        if best_position is None or abs(position_3d[2] - 0.3) < abs(best_position[2] - 0.3):
                            # Prefer positions around 30cm (typical calibration distance)
                            best_position = position_3d
                            best_pixel = (test_x, test_y)
        
        if best_position is not None:
            print(f"  CharUco detected at depth coordinates: [{best_position[0]:.3f}, {best_position[1]:.3f}, {best_position[2]:.3f}]m")
            print(f"  RGB detection center pixel: [{center_pixel[0]}, {center_pixel[1]}]")
            print(f"  Depth pixel mapping: [{depth_pixel_x}, {depth_pixel_y}] -> [{best_pixel[0]}, {best_pixel[1]}] (search radius: {search_radius})")
            return best_position, rvec, tvec, True
        else:
            print(f"  No valid depth data found in {search_radius}px radius around [{depth_pixel_x}, {depth_pixel_y}]")
            print(f"  CharUco pattern may be too close (<0.1m) or too far (>2.0m) for L515 depth sensor")
            return None, None, None, False
    
    def match_image_points_corrected(self, charuco_corners, charuco_ids):
        """Match image points with corrected coordinate system (bottom-left origin)."""
        objPoints = []
        imgPoints = []
        
        for i in range(len(charuco_ids)):
            index = charuco_ids[i]
            # Get the corner position from the board
            corner_pos = self.board.getChessboardCorners()[index]
            
            # Apply coordinate system correction (bottom-left origin)
            corrected_pos = corner_pos.copy()
            corrected_pos[0][1] = self.board.getRightBottomCorner()[1] - corrected_pos[0][1]
            
            objPoints.append(corrected_pos)
            imgPoints.append(charuco_corners[i])
        
        return np.array(objPoints), np.array(imgPoints)
    
    def get_robot_pose(self):
        """Get current robot end-effector pose."""
        if self.robot is not None and hasattr(self.robot, 'is_connected') and self.robot.is_connected:
            try:
                pose = self.robot.get_current_position()
                if pose and len(pose) >= 6:
                    return pose[:6]  # [x, y, z, rx, ry, rz]
            except Exception as e:
                print(f"Failed to get robot pose: {e}")
        
        # Fallback: manual input
        print("\nRobot pose not available. Enter robot end-effector pose manually:")
        try:
            x = float(input("X (mm): "))
            y = float(input("Y (mm): "))
            z = float(input("Z (mm): "))
            rx = float(input("RX (degrees): "))
            ry = float(input("RY (degrees): "))
            rz = float(input("RZ (degrees): "))
            return [x, y, z, rx, ry, rz]
        except (ValueError, KeyboardInterrupt):
            print("Input cancelled")
            return None
    
    def pose_to_transform_matrix(self, pose):
        """Convert [x, y, z, rx, ry, rz] pose to 4x4 transformation matrix."""
        x, y, z, rx, ry, rz = pose
        
        # Convert to meters and radians
        x_m, y_m, z_m = x/1000.0, y/1000.0, z/1000.0
        rx_rad, ry_rad, rz_rad = np.radians([rx, ry, rz])
        
        # Create rotation matrix from Euler angles
        rotation = Rotation.from_euler('xyz', [rx_rad, ry_rad, rz_rad])
        rot_matrix = rotation.as_matrix()
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = [x_m, y_m, z_m]
        
        return transform
    
    def charuco_depth_position_to_transform_matrix(self, position_3d, rvec, tvec):
        """
        Create transformation matrix from depth-extracted position with coordinate correction.
        
        Args:
            position_3d: 3D position from depth point cloud [x, y, z] in meters
            rvec: Rotation vector from RGB CharUco detection (for orientation reference)
            tvec: Translation vector from RGB detection (unused - we use depth position)
            
        Returns:
            4x4 transformation matrix in depth camera coordinates
        """
        # Use rotation from RGB detection (convert to matrix)
        rot_matrix, _ = cv2.Rodrigues(rvec)
        
        # Apply 90° Z-rotation correction for robot coordinate alignment
        correction_matrix = np.array([
            [0, -1, 0],  # 90° rotation around Z
            [1,  0, 0], 
            [0,  0, 1]
        ])
        
        corrected_rot_matrix = rot_matrix @ correction_matrix
        
        # Create 4x4 transformation matrix using depth-extracted position
        transform = np.eye(4)
        transform[:3, :3] = corrected_rot_matrix
        transform[:3, 3] = position_3d  # Use depth coordinates instead of RGB tvec
        
        return transform
    
    def collect_calibration_point(self, image, depth_frame):
        """Collect robot and CharUco poses for depth-based calibration."""
        # Get CharUco position from depth point cloud
        position_3d, rvec, tvec, success = self.get_charuco_position_from_point_cloud(image, depth_frame)
        
        if not success:
            print("No CharUco pattern detected or depth extraction failed")
            return False
        
        print(f"\nCollecting depth-based calibration point {len(self.calibration_pairs) + 1}...")
        
        # Get robot pose
        robot_pose = self.get_robot_pose()
        if robot_pose is None:
            return False
        
        # Convert to transformation matrices
        robot_transform = self.pose_to_transform_matrix(robot_pose)
        charuco_transform = self.charuco_depth_position_to_transform_matrix(position_3d, rvec, tvec)
        
        # Store calibration pair
        self.calibration_pairs.append({
            'robot_pose': robot_pose,
            'robot_transform': robot_transform,
            'charuco_depth_position': position_3d,
            'charuco_rvec': rvec,
            'charuco_tvec': tvec,  # From RGB detection (reference only)
            'charuco_transform': charuco_transform,
            'timestamp': time.time()
        })
        
        print(f"Calibration point collected:")
        print(f"  Robot: [{robot_pose[0]:.1f}, {robot_pose[1]:.1f}, {robot_pose[2]:.1f}] mm")
        print(f"  CharUco depth: [{position_3d[0]*1000:.1f}, {position_3d[1]*1000:.1f}, {position_3d[2]*1000:.1f}] mm")
        print(f"Total points: {len(self.calibration_pairs)}")
        
        return True
    
    def calculate_eye_in_hand_calibration(self):
        """Calculate eye-in-hand calibration using depth-based CharUco positions."""
        if len(self.calibration_pairs) < 3:
            print(f"Need at least 3 calibration points, have {len(self.calibration_pairs)}")
            return None
        
        print(f"\nCalculating DEPTH-BASED eye-in-hand calibration from {len(self.calibration_pairs)} points...")
        
        # Prepare data for OpenCV calibrateHandEye
        robot_transforms = []
        charuco_transforms = []
        
        for pair in self.calibration_pairs:
            robot_transforms.append(pair['robot_transform'])
            charuco_transforms.append(pair['charuco_transform'])
        
        # Convert to OpenCV format
        R_gripper2base = [T[:3, :3] for T in robot_transforms]
        t_gripper2base = [T[:3, 3].reshape(3, 1) for T in robot_transforms]
        R_target2cam = [T[:3, :3] for T in charuco_transforms]
        t_target2cam = [T[:3, 3].reshape(3, 1) for T in charuco_transforms]
        
        # Calculate hand-eye calibration
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # Create 4x4 transformation matrix
            eye_in_hand_transform = np.eye(4)
            eye_in_hand_transform[:3, :3] = R_cam2gripper
            eye_in_hand_transform[:3, 3] = t_cam2gripper.flatten()
            
            print("DEPTH-BASED eye-in-hand calibration successful!")
            print(f"Transformation matrix (T_robot_to_depth):")
            for row in eye_in_hand_transform:
                print(f"  [{row[0]:8.5f} {row[1]:8.5f} {row[2]:8.5f} {row[3]:8.5f}]")
            
            return eye_in_hand_transform
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return None
    
    def save_calibration_results(self, eye_in_hand_transform):
        """Save depth-based calibration results."""
        timestamp = int(time.time())
        output_dir = self.config_path / f"depth_calibration_result_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        # Save transformation matrix
        np.savetxt(output_dir / 'eye_in_hand_transform_depth.txt', eye_in_hand_transform)
        np.save(output_dir / 'eye_in_hand_transform_depth.npy', eye_in_hand_transform)
        
        # Also save with standard name for easy integration
        np.savetxt(output_dir / 'eye_in_hand_transform.txt', eye_in_hand_transform) 
        np.save(output_dir / 'eye_in_hand_transform.npy', eye_in_hand_transform)
        
        # Create README explaining this is depth-based
        readme_content = f"""# Depth-Based Eye-in-Hand Calibration Results

This calibration was performed using DEPTH camera coordinates instead of RGB.

## Key Features:
- CharUco pattern positions extracted from point cloud (depth coordinates)
- Direct compatibility with LiDAR-based object detection
- Eliminates systematic coordinate system mismatch
- Transformation matrix: T_robot_to_depth

## Usage:
Use eye_in_hand_transform.npy in your LiDAR detection systems for accurate 
robot coordinate transformation.

## Calibration Details:
- Calibration points: {len(self.calibration_pairs)}
- Pattern size: {self.square_length*1000:.1f}mm squares, {self.marker_length*1000:.1f}mm markers
- Board size: {self.board_size[0]}x{self.board_size[1]}
- Coordinate correction: 90° Z-rotation applied
- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_dir / 'README.txt', 'w') as f:
            f.write(readme_content)
        
        print(f"\nDepth-based calibration results saved to: {output_dir}")
        print(f"Main transformation matrix: {output_dir / 'eye_in_hand_transform.npy'}")
        print("This matrix is ready to use with your virtual tracking system!")
        
        return output_dir
    
    def load_robot_poses(self):
        """Load preset robot poses for calibration."""
        poses_file = Path("recorded_coords_280.json")
        try:
            if poses_file.exists():
                with open(poses_file, 'r') as f:
                    self.preset_poses = json.load(f)
                print(f"Loaded {len(self.preset_poses)} preset robot poses")
            else:
                # Default poses for calibration
                self.preset_poses = [
                    [200, 0, 300, 0, 180, 0],       # Center, looking down
                    [150, -100, 280, -20, 160, -30],  # Left side, angled
                    [150, 100, 280, 20, 160, 30],     # Right side, angled
                    [250, 0, 250, 0, 145, 0],         # Forward, tilted
                    [180, -80, 320, -15, 175, -20],   # Left high
                    [180, 80, 320, 15, 175, 20],      # Right high
                    [220, -50, 280, -10, 150, -15],   # Additional poses
                    [220, 50, 280, 10, 150, 15],      # for better coverage
                ]
                print("Using default calibration poses")
        except Exception as e:
            print(f"Error loading poses: {e}")
            self.preset_poses = [[200, 0, 300, 0, 180, 0]]
    
    def move_robot_to_next_pose(self):
        """Move robot to next preset pose."""
        if not self.robot or not hasattr(self.robot, 'is_connected') or not self.robot.is_connected:
            print("Robot not connected - cannot move automatically")
            return False
        
        pose = self.preset_poses[self.current_pose_index]
        self.current_pose_index = (self.current_pose_index + 1) % len(self.preset_poses)
        
        print(f"Moving robot to pose {self.current_pose_index}/{len(self.preset_poses)}: {pose}")
        
        try:
            success = self.robot.move_to_position(pose)
            if success:
                print(f"Robot moved successfully")
                time.sleep(2)  # Give robot time to settle
                return True
            else:
                print("Robot movement failed")
                return False
        except Exception as e:
            print(f"Error moving robot: {e}")
            return False
    
    def draw_calibration_interface(self, image, depth_frame):
        """Draw calibration interface with depth-based information."""
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        # Try to detect CharUco and get depth information
        position_3d, rvec, tvec, success = self.get_charuco_position_from_point_cloud(image, depth_frame)
        
        if success:
            # Draw CharUco detection results
            marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(image)
            if marker_ids is not None:
                vis_image = aruco.drawDetectedMarkers(vis_image, marker_corners, marker_ids)
            
            charuco_corners, charuco_ids, _, _ = self.charuco_detector.detectBoard(
                image, None, None, marker_corners, marker_ids
            )
            if charuco_corners is not None:
                vis_image = aruco.drawDetectedCornersCharuco(vis_image, charuco_corners, charuco_ids)
            
            # Draw axes
            if rvec is not None and tvec is not None:
                cv2.drawFrameAxes(vis_image, self.camera_matrix, self.dist_coeffs,
                                rvec, tvec, self.square_length, 3)
        
        # Create info overlay
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (10, 10), (600, 200), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (600, 200), (255, 255, 255), 2)
        vis_image = cv2.addWeighted(overlay, 0.8, vis_image, 0.2, 0)
        
        # Title
        cv2.putText(vis_image, "Depth-Based CharUco Eye-in-Hand Calibration", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Detection status
        y = 60
        if success and position_3d is not None:
            cv2.putText(vis_image, "Status: CharUco DETECTED in depth coordinates", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20
            cv2.putText(vis_image, f"Depth position: [{position_3d[0]*1000:.1f}, {position_3d[1]*1000:.1f}, {position_3d[2]*1000:.1f}] mm",
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        else:
            cv2.putText(vis_image, "Status: Searching for CharUco pattern...", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Calibration progress
        y += 30
        cv2.putText(vis_image, f"Calibration points collected: {len(self.calibration_pairs)}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        y += 20
        if len(self.calibration_pairs) >= 3:
            cv2.putText(vis_image, "Ready for calibration! Press 'q' to calculate", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            cv2.putText(vis_image, f"Need {3 - len(self.calibration_pairs)} more points for calibration", (20, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        # Controls
        cv2.putText(vis_image, "Controls: 'c'=Move robot, 's'=Collect point, 'q'=Calibrate", 
                   (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_image, "This creates T_robot_to_depth for LiDAR compatibility", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return vis_image
    
    def run(self):
        """Run the depth-based calibration system."""
        if not self.setup_camera():
            print("Failed to setup camera. Exiting.")
            return
        
        print("\n" + "="*60)
        print("DEPTH-BASED CHARUCO EYE-IN-HAND CALIBRATION")
        print("="*60)
        print("This system creates T_robot_to_depth for LiDAR compatibility")
        print("\nControls:")
        print("  'c' - Move robot to next preset position")
        print("  's' - Collect calibration point (robot + CharUco poses)")
        print("  'q' - Calculate eye-in-hand transformation and quit")
        print("  ESC - Exit without saving")
        print("\nCollect at least 3 points, 8-12 recommended for best accuracy")
        print("="*60)
        
        try:
            while True:
                # Get frames from camera
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy array
                image = np.asanyarray(color_frame.get_data())
                
                # Draw interface
                vis_image = self.draw_calibration_interface(image, depth_frame)
                
                # Display
                cv2.imshow('Depth-Based CharUco Calibration', vis_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s'):  # Collect calibration point
                    self.collect_calibration_point(image, depth_frame)
                
                elif key == ord('c'):  # Move robot
                    print("\nMoving robot to next position...")
                    self.move_robot_to_next_pose()
                
                elif key == ord('q'):  # Calculate calibration
                    if len(self.calibration_pairs) >= 3:
                        print("\nCalculating calibration...")
                        eye_in_hand_transform = self.calculate_eye_in_hand_calibration()
                        if eye_in_hand_transform is not None:
                            output_dir = self.save_calibration_results(eye_in_hand_transform)
                            print("\n" + "="*60)
                            print("DEPTH-BASED CALIBRATION COMPLETE!")
                            print("="*60)
                            print(f"Results saved to: {output_dir}")
                            print("Use eye_in_hand_transform.npy in your virtual tracking system")
                            print("This should eliminate the coordinate system mismatch!")
                            break
                    else:
                        print(f"Need at least 3 calibration points, have {len(self.calibration_pairs)}")
                
                elif key == 27:  # ESC - exit
                    print("Exiting without saving...")
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            self.pipeline.stop()
            cv2.destroyAllWindows()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Depth-based CharUco eye-in-hand calibration")
    parser.add_argument('--config-path', type=str, default='charuco-pose-estimation',
                       help='Path to configuration directory')
    
    args = parser.parse_args()
    
    # Create calibration system
    calibrator = DepthBasedCharUcoCalibrator(args.config_path)
    
    # Run the system
    calibrator.run()


if __name__ == "__main__":
    main()