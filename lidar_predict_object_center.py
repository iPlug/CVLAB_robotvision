#!/usr/bin/env python3
"""
LiDAR Object Center Prediction

Simple example showing how to predict object center points from L515 LiDAR
sensor relative to robot frame using existing calibration.

Based on your existing lidar_track_object.py but focused on just center point prediction.

Usage:
    python lidar_predict_object_center.py
"""

import sys
import os
import time
import numpy as np
import cv2
from pathlib import Path

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.sensors.realsense_manager import RealSenseManager
from module.strategies.lidar_clustering_strategy import LiDARClusteringStrategy
from module.processing.lidar_coordinator import LiDARCoordinator


class LiDARCenterPredictor:
    """
    Simple LiDAR-based object center point prediction.
    
    This is a simplified version of your existing lidar_track_object.py
    focused specifically on predicting object center points.
    """
    
    def __init__(self, transformation_matrix_file: str = "charuco_eye_in_hand_transform.npy"):
        """Initialize the LiDAR center predictor."""
        self.transformation_matrix_file = transformation_matrix_file
        
        # Initialize components
        self.sensor_manager = RealSenseManager(realtime_mode=True)
        
        # LiDAR clustering for object detection
        self.clustering_strategy = LiDARClusteringStrategy(
            distance_range=(0.25, 0.8),  # L515 effective range
            floor_params={
                'distance_threshold': 0.02,
                'max_trials': 1000,
                'min_floor_points': 500,
                'floor_normal_tolerance': 0.3
            },
            clustering_params={
                'eps': 0.03,
                'min_samples': 10,
                'min_cluster_size': 50,
                'max_cluster_size': 2000,
                'outlier_removal': True
            },
            selection_params={
                'selection_method': 'nearest_center',
                'max_selection_distance': 1.0,
                'height_preference': 0.1
            }
        )
        
        # Coordinate transformation using your existing calibration
        self.coordinator = LiDARCoordinator(
            transformation_matrix_file=self.transformation_matrix_file,
            coordinate_system="standard"
        )
        
        print(f"Initialized LiDAR center predictor with matrix: {transformation_matrix_file}")
    
    def start(self) -> bool:
        """Start the sensor and verify calibration."""
        # Initialize sensor
        if not self.sensor_manager.initialize():
            print("Error: Failed to initialize RealSense L515")
            return False
        
        if not self.sensor_manager.start():
            print("Error: Failed to start RealSense pipeline")
            return False
        
        # Verify transformation matrix
        info = self.coordinator.get_transformation_info()
        if not info['matrix_loaded']:
            print("Error: No transformation matrix loaded!")
            print("Please run CharUco calibration first (charuco_eye_in_hand.py)")
            return False
        
        print(f"Using transformation matrix: {info['matrix_source']}")
        print("L515 LiDAR sensor ready for object center prediction")
        return True
    
    def predict_object_center(self) -> dict:
        """
        Predict the center point of the nearest object in robot coordinates.
        
        Returns:
            dict with:
                - 'success': bool - whether object was detected
                - 'center_robot': [x, y, z] in mm relative to robot base
                - 'center_camera': [x, y, z] in meters relative to camera
                - 'distance_from_base': float in mm
                - 'cluster_size': int - number of points in object
        """
        # Get LiDAR point cloud
        vertices, color_image, metadata = self.sensor_manager.get_lidar_optimized_frame(
            distance_range=(0.25, 0.8),
            enable_filtering=True
        )
        
        result = {
            'success': False,
            'center_robot': None,
            'center_camera': None,
            'distance_from_base': None,
            'cluster_size': None
        }
        
        if vertices is None or len(vertices) == 0:
            return result
        
        # Detect objects using clustering
        clusters = self.clustering_strategy.cluster_objects(vertices, color_image)
        
        if not clusters:
            return result
        
        # Get the first (typically nearest/largest) object
        selected_cluster = clusters[0]
        
        # Transform to robot coordinates
        robot_coords = self.coordinator.transform_cluster_center(selected_cluster)
        
        if not robot_coords or not robot_coords['is_valid']:
            return result
        
        # Extract results
        result.update({
            'success': True,
            'center_robot': robot_coords['robot_position'],  # [x, y, z] in mm
            'center_camera': robot_coords['camera_position'], # [x, y, z] in meters  
            'distance_from_base': robot_coords['distance_from_base'], # mm
            'cluster_size': len(selected_cluster['points'])
        })
        
        return result
    
    def run_continuous_prediction(self):
        """Run continuous object center prediction with live display."""
        print("\nStarting continuous LiDAR object center prediction...")
        print("Press 'q' to quit, 'p' to print current prediction")
        print("Robot coordinates: X=forward(+), Y=left(+), Z=up(+), units in mm")
        print()
        
        try:
            frame_count = 0
            last_detection_time = 0
            
            while True:
                frame_count += 1
                current_time = time.time()
                
                # Predict object center
                prediction = self.predict_object_center()
                
                # Create simple visualization
                vis_image = self._create_visualization(prediction, frame_count, current_time)
                cv2.imshow('LiDAR Object Center Prediction', vis_image)
                
                # Print detection info periodically or when requested
                if prediction['success']:
                    if current_time - last_detection_time > 1.0:  # Every second
                        center = prediction['center_robot']
                        print(f"Object center: ({center[0]:6.1f}, {center[1]:6.1f}, {center[2]:6.1f}) mm "
                              f"| Distance: {prediction['distance_from_base']:6.1f} mm "
                              f"| Points: {prediction['cluster_size']}")
                        last_detection_time = current_time
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p') and prediction['success']:
                    self._print_detailed_prediction(prediction)
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        finally:
            self._cleanup()
    
    def _create_visualization(self, prediction: dict, frame_count: int, current_time: float) -> np.ndarray:
        """Create simple visualization showing prediction results."""
        vis_image = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(vis_image, "LiDAR Object Center Prediction", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Status
        if prediction['success']:
            center = prediction['center_robot']
            
            # Main prediction results
            cv2.putText(vis_image, "OBJECT DETECTED", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(vis_image, f"Center Position (Robot Frame):", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.putText(vis_image, f"  X (forward): {center[0]:7.1f} mm", (20, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.putText(vis_image, f"  Y (left):    {center[1]:7.1f} mm", (20, 165), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.putText(vis_image, f"  Z (up):      {center[2]:7.1f} mm", (20, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            cv2.putText(vis_image, f"Distance from base: {prediction['distance_from_base']:6.1f} mm", 
                       (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.putText(vis_image, f"Cluster size: {prediction['cluster_size']} points", 
                       (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Visual representation (top-down view)
            self._draw_top_down_view(vis_image, center)
            
        else:
            cv2.putText(vis_image, "SEARCHING FOR OBJECTS...", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(vis_image, "No objects detected in range", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Frame info
        cv2.putText(vis_image, f"Frame: {frame_count}", (10, 320), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.putText(vis_image, f"LiDAR Range: 0.25-0.8m", (10, 340), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Controls
        cv2.putText(vis_image, "Controls: 'q'=quit, 'p'=print details", (10, vis_image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return vis_image
    
    def _draw_top_down_view(self, image: np.ndarray, center: list):
        """Draw simple top-down view showing object position relative to robot."""
        # Top-down view area (right side of image)
        view_x = 350
        view_y = 80
        view_size = 200
        
        # Draw coordinate system
        cv2.rectangle(image, (view_x, view_y), (view_x + view_size, view_y + view_size), (100, 100, 100), 1)
        
        # Robot base at center
        robot_x = view_x + view_size // 2
        robot_y = view_y + view_size // 2
        cv2.circle(image, (robot_x, robot_y), 5, (0, 255, 0), -1)
        cv2.putText(image, "Robot", (robot_x - 20, robot_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Object position (scale to fit view)
        scale = 0.2  # 1mm = 0.2 pixels
        obj_x = int(robot_x + center[0] * scale)  # X = forward
        obj_y = int(robot_y - center[1] * scale)  # Y = left (flip for visualization)
        
        # Clamp to view bounds
        obj_x = max(view_x, min(view_x + view_size, obj_x))
        obj_y = max(view_y, min(view_y + view_size, obj_y))
        
        # Draw object
        cv2.circle(image, (obj_x, obj_y), 3, (255, 0, 0), -1)
        cv2.line(image, (robot_x, robot_y), (obj_x, obj_y), (255, 255, 0), 1)
        
        # Labels
        cv2.putText(image, "Top View", (view_x, view_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(image, "Object", (obj_x - 15, obj_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
    
    def _print_detailed_prediction(self, prediction: dict):
        """Print detailed prediction information."""
        print("\n" + "="*50)
        print("DETAILED OBJECT PREDICTION")
        print("="*50)
        
        if prediction['success']:
            center_robot = prediction['center_robot']
            center_camera = prediction['center_camera']
            
            print(f"Object Center (Robot Frame):")
            print(f"  X (forward): {center_robot[0]:8.1f} mm")
            print(f"  Y (left):    {center_robot[1]:8.1f} mm") 
            print(f"  Z (up):      {center_robot[2]:8.1f} mm")
            
            print(f"\nObject Center (Camera Frame):")
            print(f"  X: {center_camera[0]:8.3f} m")
            print(f"  Y: {center_camera[1]:8.3f} m")
            print(f"  Z: {center_camera[2]:8.3f} m")
            
            print(f"\nDistance from robot base: {prediction['distance_from_base']:6.1f} mm")
            print(f"Cluster size: {prediction['cluster_size']} points")
            
            print(f"\nUsing transformation matrix: {self.transformation_matrix_file}")
        else:
            print("No object detected")
        
        print("="*50)
    
    def _cleanup(self):
        """Clean up resources."""
        if self.sensor_manager:
            self.sensor_manager.stop()
        cv2.destroyAllWindows()
        print("LiDAR object center prediction stopped")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LiDAR Object Center Prediction")
    parser.add_argument('--matrix', type=str, default='charuco_eye_in_hand_transform.npy',
                       help='Eye-in-hand transformation matrix file')
    parser.add_argument('--single', action='store_true',
                       help='Make single prediction and exit')
    
    args = parser.parse_args()
    
    print("="*60)
    print("L515 LiDAR OBJECT CENTER PREDICTION")
    print("="*60)
    print(f"Transformation matrix: {args.matrix}")
    print("="*60)
    
    # Create predictor
    predictor = LiDARCenterPredictor(args.matrix)
    
    if not predictor.start():
        print("Failed to start predictor")
        return 1
    
    try:
        if args.single:
            # Single prediction
            prediction = predictor.predict_object_center()
            predictor._print_detailed_prediction(prediction)
        else:
            # Continuous prediction
            predictor.run_continuous_prediction()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())