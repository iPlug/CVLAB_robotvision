"""
Real-time detection application optimized for live camera processing.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List

from module.core.object_detector import ObjectDetector
from module.core.visualization_engine import VisualizationEngine
from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
from module.sensors.realsense_manager import RealSenseManager
from module.processing.temporal_filter import TemporalFilter
from module.config.config_factory import ConfigFactory


class RealTimeDetectionApp:
    """
    Real-time object detection application optimized for live camera processing.
    
    Features:
    - Optimized for real-time performance
    - RealSense camera integration
    - BEV clustering with temporal filtering
    - Interactive parameter adjustment
    - Performance monitoring and optimization
    """
    
    def __init__(self, bag_file: Optional[str] = None, 
                 realtime_mode: bool = False,
                 use_imu: bool = True):
        """
        Initialize real-time detection application.
        
        Args:
            bag_file: Path to bag file for playback (None for live camera)
            realtime_mode: Whether to play bag files at real-time speed
            use_imu: Whether to enable IMU for gravity alignment
        """
        # Create optimized configuration for real-time processing
        self.config = ConfigFactory.create_preset('realtime_bev')
        
        # Initialize RealSense sensor
        self.sensor_manager = RealSenseManager(bag_file, realtime_mode, use_imu)
        
        # Initialize BEV clustering strategy with optimized parameters
        self.clustering_strategy = BEVClusteringStrategy(
            roi_params=self.config.get('roi_params'),
            display_roi_params=self.config.get('display_roi_params'),
            clustering_params=self.config.get('clustering_params'),
            enable_temporal_filtering=self.config.get('temporal_params.enabled', True)
        )
        
        # Initialize object detector (no classification for real-time performance)
        self.detector = ObjectDetector(
            clustering_strategy=self.clustering_strategy,
            classification_strategy=None  # Disabled for real-time performance
        )
        
        # Initialize visualization and temporal filtering
        self.visualizer = VisualizationEngine()
        self.temporal_filter = TemporalFilter(
            window_size=self.config.get('clustering_params.window_size', 5)
        )
        
        # Application state
        self.is_running = False
        self.is_paused = False
        self.frame_count = 0
        self.show_help = False
        self.show_performance = True
        
        # Performance tracking
        self.performance_monitor = {
            'frame_times': [],
            'detection_times': [],
            'target_fps': 30.0,
            'max_frame_time': 1.0 / 30.0,  # 33ms for 30 FPS
            'dropped_frames': 0
        }
        
        # UI parameters that can be adjusted at runtime
        self.ui_params = {
            'show_bounding_boxes': True,
            'show_cluster_ids': True,
            'show_point_cloud': True,
            'show_roi_bounds': True
        }
    
    def start(self) -> bool:
        """
        Start the real-time detection application.
        
        Returns:
            True if started successfully
        """
        # Initialize and start sensor
        print("Initializing RealSense sensor...")
        if not self.sensor_manager.initialize():
            print("Error: Failed to initialize RealSense sensor")
            return False
        
        if not self.sensor_manager.start():
            print("Error: Failed to start RealSense sensor")
            return False
        
        # Calibrate gravity alignment if using IMU
        if hasattr(self.sensor_manager, 'use_imu') and self.sensor_manager.use_imu:
            print("Calibrating gravity alignment...")
            self.sensor_manager.calibrate_gravity_alignment()
        
        self.is_running = True
        print("\nReal-Time Detection Application Started")
        print("=" * 50)
        print("Controls:")
        print("  'q' - Quit")
        print("  'h' - Toggle help")
        print("  'p' - Toggle performance info")
        print("  'b' - Toggle bounding boxes")
        print("  'c' - Toggle cluster IDs")
        print("  'r' - Toggle ROI visualization")
        print("  'Space' - Pause/Resume")
        print("  '+/-' - Adjust clustering sensitivity")
        print("  '[/]' - Adjust temporal window")
        
        return True
    
    def stop(self):
        """Stop the application."""
        self.is_running = False
        self.sensor_manager.stop()
        cv2.destroyAllWindows()
        
        # Print final performance summary
        print("\n" + self.get_performance_summary())
        print("Real-time detection application stopped")
    
    def run(self):
        """Main application loop optimized for real-time performance."""
        if not self.start():
            return
        
        try:
            while self.is_running:
                frame_start = time.time()
                
                # Handle pause
                if self.is_paused:
                    cv2.waitKey(30)
                    continue
                
                # Get frame from sensor
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                # Check for end of stream
                if metadata.get('end_of_stream', False):
                    print("End of stream reached")
                    break
                
                if vertices is not None:
                    # Detect objects with timing
                    detection_start = time.time()
                    detected_objects, timing_info = self.detector.detect_objects(vertices, color_image)
                    detection_time = time.time() - detection_start
                    
                    # Apply temporal filtering for stability
                    if detected_objects:
                        detected_objects = self.temporal_filter.stabilize_object_positions(detected_objects)
                    
                    # Add to temporal filter for FPS calculation
                    self.temporal_filter.add_frame_to_history(
                        objects=detected_objects,
                        timestamp=time.time()
                    )
                    
                    # Create and display visualization
                    self._create_realtime_visualization(vertices, color_image, detected_objects, timing_info, metadata)
                    
                    self.frame_count += 1
                    
                    # Update performance monitoring
                    frame_time = time.time() - frame_start
                    self._update_performance_monitoring(frame_time, detection_time)
                
                # Handle user input
                if not self._handle_realtime_input():
                    break
                
                # Adaptive frame rate control
                self._adaptive_frame_rate_control(frame_start)
                    
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def _create_realtime_visualization(self, vertices: np.ndarray, color_image: Optional[np.ndarray],
                                     detected_objects: List[Dict[str, Any]], timing_info: Dict[str, float],
                                     metadata: Dict[str, Any]):
        """Create optimized real-time visualization."""
        # Get display bounds from config
        bounds = self.config.get_display_bounds()
        x_range = bounds['x_range']
        z_range = bounds['z_range']
        
        # Create BEV visualization
        from module.processing.image_processor import ImageProcessor
        image_processor = ImageProcessor(default_image_size=400)  # Smaller for performance
        
        bev_image = image_processor.create_bev_image(
            vertices, x_range, z_range, 
            height_colormap=True,
            height_range=(self.config.get('roi_params.y_min'), self.config.get('roi_params.y_max'))
        )
        
        # Draw detected objects
        if detected_objects and self.ui_params['show_bounding_boxes']:
            self._draw_objects_on_bev(bev_image, detected_objects, x_range, z_range)
        
        # Add ROI visualization if enabled
        if self.ui_params['show_roi_bounds']:
            self._draw_roi_bounds(bev_image, x_range, z_range)
        
        # Create point cloud side view
        point_cloud_view = None
        if self.ui_params['show_point_cloud']:
            point_cloud_view = self._create_point_cloud_side_view(vertices)
        
        # Combine images
        if color_image is not None:
            # Resize for consistent display
            height = 400
            color_resized = cv2.resize(color_image, (int(color_image.shape[1] * height / color_image.shape[0]), height))
            bev_resized = cv2.resize(bev_image, (height, height))
            
            if point_cloud_view is not None:
                pc_resized = cv2.resize(point_cloud_view, (height, height))
                combined = np.hstack([color_resized, bev_resized, pc_resized])
            else:
                combined = np.hstack([color_resized, bev_resized])
        else:
            combined = bev_image
        
        # Add performance overlay
        if self.show_performance:
            combined = self._add_performance_overlay(combined, timing_info, metadata)
        
        # Add object count and information
        self._add_object_info_overlay(combined, detected_objects)
        
        # Add help overlay if requested
        if self.show_help:
            self._add_help_overlay(combined)
        
        # Display
        cv2.imshow('Real-Time Object Detection', combined)
    
    def _draw_objects_on_bev(self, bev_image: np.ndarray, detected_objects: List[Dict[str, Any]],
                           x_range: tuple, z_range: tuple):
        """Draw detected objects on BEV image."""
        height, width = bev_image.shape[:2]
        
        for i, obj in enumerate(detected_objects):
            center = obj.get('center', [0, 0, 0])
            dimensions = obj.get('dimensions', [0, 0, 0])
            
            # Convert 3D coordinates to 2D image coordinates
            x_img = int((center[0] - x_range[0]) / (x_range[1] - x_range[0]) * width)
            z_img = int((center[2] - z_range[0]) / (z_range[1] - z_range[0]) * height)
            z_img = height - z_img  # Flip Z
            
            # Clamp to image bounds
            x_img = max(0, min(width-1, x_img))
            z_img = max(0, min(height-1, z_img))
            
            # Draw bounding box
            half_width = max(1, int(dimensions[0] / (x_range[1] - x_range[0]) * width / 2))
            half_depth = max(1, int(dimensions[2] / (z_range[1] - z_range[0]) * height / 2))
            
            x1, x2 = max(0, x_img - half_width), min(width-1, x_img + half_width)
            z1, z2 = max(0, z_img - half_depth), min(height-1, z_img + half_depth)
            
            # Color based on object index
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[i % len(colors)]
            
            cv2.rectangle(bev_image, (x1, z1), (x2, z2), color, 2)
            
            # Add center point
            cv2.circle(bev_image, (x_img, z_img), 3, (255, 255, 255), -1)
            
            # Add ID if enabled
            if self.ui_params['show_cluster_ids']:
                cv2.putText(bev_image, f"#{i}", (x_img + 5, z_img - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_roi_bounds(self, bev_image: np.ndarray, x_range: tuple, z_range: tuple):
        """Draw ROI bounds on BEV image."""
        height, width = bev_image.shape[:2]
        
        # Draw border
        cv2.rectangle(bev_image, (0, 0), (width-1, height-1), (128, 128, 128), 1)
        
        # Add coordinate labels
        cv2.putText(bev_image, f"X: [{x_range[0]:.1f}, {x_range[1]:.1f}]m", 
                   (5, height-25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(bev_image, f"Z: [{z_range[0]:.1f}, {z_range[1]:.1f}]m", 
                   (5, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _create_point_cloud_side_view(self, vertices: np.ndarray) -> np.ndarray:
        """Create side view of point cloud."""
        if vertices is None or len(vertices) == 0:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        img_size = 400
        
        # Use X and Y coordinates for side view
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        z_coords = vertices[:, 2]  # For coloring
        
        # Filter valid points
        valid_mask = z_coords > 0
        if not np.any(valid_mask):
            return np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        z_valid = z_coords[valid_mask]
        
        # Calculate bounds with padding
        x_min, x_max = np.min(x_valid), np.max(x_valid)
        y_min, y_max = np.min(y_valid), np.max(y_valid)
        
        x_padding = max(0.1, (x_max - x_min) * 0.1)
        y_padding = max(0.1, (y_max - y_min) * 0.1)
        
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        # Convert to image coordinates
        if x_max > x_min and y_max > y_min:
            x_img = ((x_valid - x_min) / (x_max - x_min) * img_size).astype(int)
            y_img = ((y_valid - y_min) / (y_max - y_min) * img_size).astype(int)
            y_img = img_size - y_img - 1  # Flip Y
            
            # Create image
            pc_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            # Color by depth
            if len(z_valid) > 0:
                z_min, z_max = np.min(z_valid), np.max(z_valid)
                if z_max > z_min:
                    z_normalized = (z_valid - z_min) / (z_max - z_min)
                    colors = (plt.cm.jet(z_normalized)[:, :3] * 255).astype(np.uint8)
                else:
                    colors = np.full((len(z_valid), 3), [255, 255, 255], dtype=np.uint8)
                
                # Draw points
                valid_img_mask = (x_img >= 0) & (x_img < img_size) & (y_img >= 0) & (y_img < img_size)
                for i in np.where(valid_img_mask)[0]:
                    cv2.circle(pc_image, (x_img[i], y_img[i]), 1, 
                             (int(colors[i][2]), int(colors[i][1]), int(colors[i][0])), -1)
            
            # Add labels
            cv2.putText(pc_image, "Side View", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(pc_image, f"X: [{x_min:.2f}, {x_max:.2f}]m", (10, img_size-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(pc_image, f"Y: [{y_min:.2f}, {y_max:.2f}]m", (10, img_size-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return pc_image
        
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    def _add_performance_overlay(self, image: np.ndarray, timing_info: Dict[str, float], 
                               metadata: Dict[str, Any]) -> np.ndarray:
        """Add performance information overlay."""
        if not self.show_performance:
            return image
        
        current_fps = self.temporal_filter.calculate_fps()
        
        perf_text = [
            f"Frame: {self.frame_count}",
            f"FPS: {current_fps:.1f}",
            f"Objects: {timing_info.get('num_objects', 0)}",
            f"Detection: {timing_info.get('total_time', 0)*1000:.1f}ms",
            f"Strategy: BEV Clustering",
            f"Alignment: {'Yes' if metadata.get('alignment_calibrated', False) else 'No'}"
        ]
        
        # Add performance warnings
        if current_fps < self.performance_monitor['target_fps'] * 0.8:
            perf_text.append("⚠ Low FPS")
        
        if self.performance_monitor['dropped_frames'] > 0:
            perf_text.append(f"Dropped: {self.performance_monitor['dropped_frames']}")
        
        # Draw performance info
        y_offset = 30
        for i, text in enumerate(perf_text):
            y_pos = y_offset + i * 25
            
            # Text background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, 
                        (10, y_pos - text_size[1] - 5),
                        (15 + text_size[0], y_pos + 5),
                        (0, 0, 0), -1)
            
            # Text
            cv2.putText(image, text, (15, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return image
    
    def _add_object_info_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]]):
        """Add object information overlay."""
        if not detected_objects:
            return
        
        # Object summary
        y_offset = image.shape[0] - 80
        cv2.putText(image, f"Objects Detected: {len(detected_objects)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Individual object info (first 2 objects)
        for i, obj in enumerate(detected_objects[:2]):
            center = obj.get('center', [0, 0, 0])
            dimensions = obj.get('dimensions', [0, 0, 0])
            
            y_pos = y_offset + 25 + i * 20
            info_text = f"#{i}: Center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})m"
            
            cv2.putText(image, info_text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _add_help_overlay(self, image: np.ndarray):
        """Add help overlay."""
        help_text = [
            "REAL-TIME CONTROLS:",
            "q - Quit",
            "h - Toggle help",
            "p - Toggle performance info",
            "b - Toggle bounding boxes",
            "c - Toggle cluster IDs", 
            "r - Toggle ROI bounds",
            "Space - Pause/Resume",
            "+/- - Adjust clustering sensitivity",
            "[/] - Adjust temporal window"
        ]
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (350, 10 + len(help_text) * 25 + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        for i, text in enumerate(help_text):
            cv2.putText(image, text, (15, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _handle_realtime_input(self) -> bool:
        """Handle real-time keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('h'):
            self.show_help = not self.show_help
        elif key == ord('p'):
            self.show_performance = not self.show_performance
        elif key == ord('b'):
            self.ui_params['show_bounding_boxes'] = not self.ui_params['show_bounding_boxes']
            print(f"Bounding boxes {'enabled' if self.ui_params['show_bounding_boxes'] else 'disabled'}")
        elif key == ord('c'):
            self.ui_params['show_cluster_ids'] = not self.ui_params['show_cluster_ids']
            print(f"Cluster IDs {'enabled' if self.ui_params['show_cluster_ids'] else 'disabled'}")
        elif key == ord('r'):
            self.ui_params['show_roi_bounds'] = not self.ui_params['show_roi_bounds']
            print(f"ROI bounds {'enabled' if self.ui_params['show_roi_bounds'] else 'disabled'}")
        elif key == ord(' '):
            self.is_paused = not self.is_paused
            print(f"{'Paused' if self.is_paused else 'Resumed'}")
        elif key == ord('+') or key == ord('='):
            # Increase clustering sensitivity (decrease min cluster size)
            current_min = self.config.get('clustering_params.min_cluster_size', 30)
            new_min = max(5, current_min - 5)
            self.config.set('clustering_params.min_cluster_size', new_min)
            print(f"Min cluster size: {new_min}")
        elif key == ord('-'):
            # Decrease clustering sensitivity (increase min cluster size)
            current_min = self.config.get('clustering_params.min_cluster_size', 30)
            new_min = min(100, current_min + 5)
            self.config.set('clustering_params.min_cluster_size', new_min)
            print(f"Min cluster size: {new_min}")
        elif key == ord('['):
            # Decrease temporal window
            current_window = self.temporal_filter.window_size
            new_window = max(1, current_window - 1)
            self.temporal_filter.set_window_size(new_window)
            print(f"Temporal window: {new_window}")
        elif key == ord(']'):
            # Increase temporal window
            current_window = self.temporal_filter.window_size
            new_window = min(10, current_window + 1)
            self.temporal_filter.set_window_size(new_window)
            print(f"Temporal window: {new_window}")
        
        return True
    
    def _update_performance_monitoring(self, frame_time: float, detection_time: float):
        """Update performance monitoring statistics."""
        self.performance_monitor['frame_times'].append(frame_time)
        self.performance_monitor['detection_times'].append(detection_time)
        
        # Keep only recent frames for performance calculation
        max_samples = 30
        if len(self.performance_monitor['frame_times']) > max_samples:
            self.performance_monitor['frame_times'].pop(0)
            self.performance_monitor['detection_times'].pop(0)
        
        # Check for dropped frames
        if frame_time > self.performance_monitor['max_frame_time']:
            self.performance_monitor['dropped_frames'] += 1
    
    def _adaptive_frame_rate_control(self, frame_start: float):
        """Implement adaptive frame rate control for consistent performance."""
        frame_time = time.time() - frame_start
        target_frame_time = 1.0 / self.performance_monitor['target_fps']
        
        # If we're running too fast, add a small delay
        if frame_time < target_frame_time:
            sleep_time = target_frame_time - frame_time
            time.sleep(min(sleep_time, 0.01))  # Max 10ms delay
    
    def get_performance_summary(self) -> str:
        """Get performance summary string."""
        if not self.performance_monitor['frame_times']:
            return "No performance data available"
        
        avg_frame_time = np.mean(self.performance_monitor['frame_times'])
        avg_detection_time = np.mean(self.performance_monitor['detection_times'])
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        summary = [
            "=== Real-Time Performance Summary ===",
            f"Total frames processed: {self.frame_count}",
            f"Average FPS: {avg_fps:.1f}",
            f"Average frame time: {avg_frame_time*1000:.1f}ms",
            f"Average detection time: {avg_detection_time*1000:.1f}ms",
            f"Dropped frames: {self.performance_monitor['dropped_frames']}",
            f"Current temporal window: {self.temporal_filter.window_size}",
            f"IMU alignment: {'Enabled' if hasattr(self.sensor_manager, 'alignment_calibrated') and self.sensor_manager.alignment_calibrated else 'Disabled'}"
        ]
        
        return "\n".join(summary)