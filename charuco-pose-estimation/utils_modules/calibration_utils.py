#!/usr/bin/env python3
"""
Calibration utilities for CharUco pose estimation applications.
Handles eye-in-hand calibration workflow and validation.
"""

import time
import json
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation


class CalibrationUtils:
    """Utilities for eye-in-hand calibration workflow and validation."""
    
    def __init__(self, config_path="charuco-pose-estimation"):
        """Initialize calibration utilities."""
        self.config_path = Path(config_path)
        self.calibration_pairs = []
    
    def collect_calibration_point(self, results, coordinate_utils, robot_utils):
        """Collect robot and CharUco poses for calibration."""
        if not results['valid_pose']:
            print("No CharUco pose detected - cannot collect calibration point")
            return False
        
        print(f"\nCollecting calibration point {len(self.calibration_pairs) + 1}...")
        
        robot_pose = robot_utils.get_robot_pose()
        if robot_pose is None:
            return False
        
        charuco_rvec = results['rvec']
        charuco_tvec = results['tvec']
        
        robot_transform = coordinate_utils.pose_to_transform_matrix(robot_pose)
        charuco_transform = coordinate_utils.charuco_pose_to_transform_matrix(charuco_rvec, charuco_tvec)
        
        self.calibration_pairs.append({
            'robot_pose': robot_pose,
            'robot_transform': robot_transform,
            'charuco_rvec': charuco_rvec,
            'charuco_tvec': charuco_tvec,
            'charuco_transform': charuco_transform,
            'timestamp': time.time()
        })
        
        print(f"Calibration point collected: Robot=[{robot_pose[0]:.1f}, {robot_pose[1]:.1f}, {robot_pose[2]:.1f}]")
        print(f"Total points: {len(self.calibration_pairs)}")
        
        return True
    
    def calculate_eye_in_hand_calibration(self):
        """Calculate eye-in-hand calibration using collected data points."""
        if len(self.calibration_pairs) < 3:
            print(f"Need at least 3 calibration points, have {len(self.calibration_pairs)}")
            return None
        
        print(f"\nCalculating eye-in-hand calibration from {len(self.calibration_pairs)} points...")
        
        robot_transforms = []
        charuco_transforms = []
        
        for pair in self.calibration_pairs:
            robot_transforms.append(pair['robot_transform'])
            charuco_transforms.append(pair['charuco_transform'])
        
        R_gripper2base = [T[:3, :3] for T in robot_transforms]
        t_gripper2base = [T[:3, 3].reshape(3, 1) for T in robot_transforms]
        R_target2cam = [T[:3, :3] for T in charuco_transforms]
        t_target2cam = [T[:3, 3].reshape(3, 1) for T in charuco_transforms]
        
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            eye_in_hand_transform = np.eye(4)
            eye_in_hand_transform[:3, :3] = R_cam2gripper
            eye_in_hand_transform[:3, 3] = t_cam2gripper.flatten()
            
            print("Eye-in-hand calibration successful!")
            print(f"Transformation matrix:\n{eye_in_hand_transform}")
            
            return eye_in_hand_transform
            
        except Exception as e:
            print(f"Calibration failed: {e}")
            return None
    
    def save_calibration_results(self, eye_in_hand_transform, camera_utils, coordinate_info=None):
        """Save calibration results to file."""
        timestamp = int(time.time())
        output_dir = self.config_path / f"calibration_result_{timestamp}"
        output_dir.mkdir(exist_ok=True)
        
        np.savetxt(output_dir / 'eye_in_hand_transform.txt', eye_in_hand_transform)
        np.save(output_dir / 'eye_in_hand_transform.npy', eye_in_hand_transform)
        np.savetxt(output_dir / 'camera_intrinsics.txt', camera_utils.camera_matrix)
        
        readme_content = f"""CALIBRATION RESULT - {timestamp}

IMPORTANT: This calibration result MUST be used with the specific camera intrinsics.

Files:
- eye_in_hand_transform.npy: Main transformation matrix
- camera_intrinsics.txt: Camera intrinsics used during calibration
- calibration_data.json: Complete calibration data including intrinsics

USAGE:
When using this transformation matrix for prediction, you MUST load and use 
the camera intrinsics from camera_intrinsics.txt, NOT from intrinsic.txt
or RealSense SDK intrinsics.

Camera Matrix Used:
{camera_utils.camera_matrix}

Distortion Coefficients Used:
{camera_utils.dist_coeffs}

Number of calibration points: {len(self.calibration_pairs)}
"""
        
        with open(output_dir / 'README.txt', 'w') as f:
            f.write(readme_content)
        
        calibration_data = {
            'timestamp': timestamp,
            'num_points': len(self.calibration_pairs),
            'camera_intrinsics': {
                'camera_matrix': camera_utils.camera_matrix.tolist(),
                'dist_coeffs': camera_utils.dist_coeffs.tolist(),
                'source': 'intrinsic.txt',
                'note': 'These intrinsics were used during calibration and MUST be used with this transformation matrix'
            },
            'eye_in_hand_transform': eye_in_hand_transform.tolist(),
            'points': []
        }
        
        if coordinate_info:
            calibration_data['coordinate_system'] = coordinate_info
        
        for i, pair in enumerate(self.calibration_pairs):
            point_data = {
                'id': i + 1,
                'robot_pose': pair['robot_pose'],
                'robot_transform': pair['robot_transform'].tolist(),
                'charuco_rvec': pair['charuco_rvec'].tolist(),
                'charuco_tvec': pair['charuco_tvec'].tolist(),
                'charuco_transform': pair['charuco_transform'].tolist(),
                'timestamp': pair['timestamp']
            }
            calibration_data['points'].append(point_data)
        
        with open(output_dir / 'calibration_data.json', 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        for i, pair in enumerate(self.calibration_pairs):
            np.savetxt(output_dir / f'robot_transform_{i+1}.txt', pair['robot_transform'])
            np.savetxt(output_dir / f'charuco_transform_{i+1}.txt', pair['charuco_transform'])
        
        print(f"Calibration results saved to: {output_dir}")
        print(f"Main transformation matrix: {output_dir / 'eye_in_hand_transform.npy'}")
        
        return output_dir
    
    def validate_calibration(self, eye_in_hand_transform):
        """Validate calibration quality using proper hand-eye calibration validation."""
        if len(self.calibration_pairs) < 3:
            return None
        
        print("\nValidating calibration quality with known pattern position...")
        
        validation_results = {
            'reprojection_errors': [],
            'pose_errors': [],
            'mean_reprojection_error': 0.0,
            'max_reprojection_error': 0.0,
            'mean_translation_error': 0.0,
            'mean_rotation_error': 0.0,
            'quality_score': 'Unknown'
        }
        
        pattern_position_robot = np.array([0.3, 0.1, 0.0])
        translation_errors = []
        rotation_errors = []
        
        for i, pair in enumerate(self.calibration_pairs):
            robot_transform = pair['robot_transform']
            charuco_transform = pair['charuco_transform']
            
            expected_pattern_in_base = robot_transform @ eye_in_hand_transform @ charuco_transform
            expected_position = expected_pattern_in_base[:3, 3]
            
            position_error = np.linalg.norm(expected_position - pattern_position_robot)
            translation_errors.append(position_error * 1000)
            
            expected_rotation = np.eye(3)
            predicted_rotation = expected_pattern_in_base[:3, :3]
            
            rotation_error_matrix = predicted_rotation @ expected_rotation.T
            rotation_error_trace = np.trace(rotation_error_matrix)
            rotation_error_angle = np.arccos(np.clip((rotation_error_trace - 1) / 2, -1, 1))
            rotation_error_degrees = np.degrees(rotation_error_angle)
            rotation_errors.append(rotation_error_degrees)
            
            print(f"  Point {i+1}: Expected pattern at [{expected_position[0]*1000:.1f}, {expected_position[1]*1000:.1f}, {expected_position[2]*1000:.1f}]mm")
            print(f"           Known position at [300.0, 100.0, 0.0]mm (forward, left, height)")
            print(f"           Position error = {position_error*1000:.1f}mm")
        
        validation_results['mean_translation_error'] = np.mean(translation_errors)
        validation_results['max_reprojection_error'] = np.max(translation_errors)
        validation_results['mean_rotation_error'] = np.mean(rotation_errors)
        validation_results['reprojection_errors'] = translation_errors
        
        mean_error_mm = validation_results['mean_translation_error']
        
        if mean_error_mm < 10.0:
            validation_results['quality_score'] = 'Excellent'
        elif mean_error_mm < 25.0:
            validation_results['quality_score'] = 'Good'  
        elif mean_error_mm < 50.0:
            validation_results['quality_score'] = 'Fair'
        else:
            validation_results['quality_score'] = 'Poor'
        
        return validation_results
    
    def print_validation_report(self, validation_results):
        """Print a detailed validation report."""
        if validation_results is None:
            print("No validation data available")
            return
        
        print("\n" + "="*50)
        print("CALIBRATION QUALITY REPORT")
        print("="*50)
        print(f"Number of calibration points: {len(self.calibration_pairs)}")
        print(f"Mean translation error: {validation_results['mean_translation_error']:.2f} mm")
        print(f"Mean rotation error: {validation_results['mean_rotation_error']:.2f}°")
        print(f"Max error: {validation_results['max_reprojection_error']:.2f} mm")
        print(f"Quality assessment: {validation_results['quality_score']}")
        
        print("\nPer-point errors:")
        for i, error in enumerate(validation_results['reprojection_errors']):
            print(f"  Point {i+1}: {error:.2f} mm")
        
        print("\nQuality Guidelines:")
        print("  Excellent: <10mm translation, <5° rotation")
        print("  Good:      <25mm translation, <10° rotation")
        print("  Fair:      <50mm translation, <15° rotation")
        print("  Poor:      >50mm translation or >15° rotation")
        
        if validation_results['quality_score'] in ['Poor', 'Fair']:
            print("\nSuggestions for improvement:")
            print("  - Collect more calibration points")
            print("  - Use more diverse robot poses (different angles/distances)")
            print("  - Ensure CharUco pattern is clearly visible in all poses")
            print("  - Check for camera calibration accuracy")
            print("  - Verify robot pose accuracy")
        
        print("="*50)
    
    def interactive_validation_review(self, validation_results):
        """Interactive review of calibration quality with options to continue or recalibrate."""
        if validation_results is None:
            return True
        
        self.print_validation_report(validation_results)
        
        while True:
            print(f"\nCalibration quality: {validation_results['quality_score']}")
            print("Options:")
            print("  'a' - Accept calibration and save results")
            print("  'c' - Continue collecting more points")
            print("  'r' - Reset and start over")
            print("  'q' - Quit without saving")
            
            try:
                choice = input("Your choice (a/c/r/q): ").lower().strip()
            except KeyboardInterrupt:
                return False
            
            if choice == 'a':
                return True
            elif choice == 'c':
                print("Continue collecting calibration points...")
                return False
            elif choice == 'r':
                print("Resetting calibration data...")
                self.calibration_pairs.clear()
                return False
            elif choice == 'q':
                print("Quitting without saving...")
                return None
            else:
                print("Invalid choice. Please enter 'a', 'c', 'r', or 'q'")
    
    def advanced_calibration_metrics(self, eye_in_hand_transform):
        """Calculate advanced calibration metrics."""
        if len(self.calibration_pairs) < 4:
            return None
            
        print("\nCalculating advanced calibration metrics...")
        
        robot_transforms = [pair['robot_transform'] for pair in self.calibration_pairs]
        
        positions = np.array([T[:3, 3] for T in robot_transforms])
        position_span = np.max(positions, axis=0) - np.min(positions, axis=0)
        position_diversity = np.linalg.norm(position_span)
        
        rotations = [Rotation.from_matrix(T[:3, :3]) for T in robot_transforms]
        
        angular_differences = []
        for i in range(len(rotations)):
            for j in range(i+1, len(rotations)):
                angular_diff = (rotations[i] * rotations[j].inv()).magnitude()
                angular_differences.append(np.degrees(angular_diff))
        
        max_angular_span = max(angular_differences) if angular_differences else 0
        
        metrics = {
            'position_diversity_m': position_diversity,
            'max_angular_span_deg': max_angular_span,
            'position_span_xyz': position_span * 1000,
            'num_points': len(self.calibration_pairs)
        }
        
        print(f"  Position diversity: {position_diversity*1000:.1f} mm")
        print(f"  Position span: X={position_span[0]*1000:.1f}mm, Y={position_span[1]*1000:.1f}mm, Z={position_span[2]*1000:.1f}mm") 
        print(f"  Max angular span: {max_angular_span:.1f}°")
        
        if position_diversity < 0.1:
            print("  Warning: Low position diversity - consider more spread-out poses")
        if max_angular_span < 30:
            print("  Warning: Low rotation diversity - consider more varied orientations")
            
        return metrics