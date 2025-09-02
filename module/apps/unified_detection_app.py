"""
Unified detection application that allows runtime switching between strategies.
"""

import time
import cv2
import numpy as np
from typing import Dict, Any, Optional, List

from module.core.object_detector import ObjectDetector
from module.core.visualization_engine import VisualizationEngine
from module.strategies.bev_clustering_strategy import BEVClusteringStrategy
from module.strategies.dbscan_clustering_strategy import DBSCANClusteringStrategy
from module.strategies.pointnet_subspace_classifier_strategy import PointNetSubspaceClassifierStrategy
from module.sensors.sensor_manager import SensorManager
from module.processing.temporal_filter import TemporalFilter
from module.config.config_factory import ConfigFactory


class UnifiedDetectionApp:
    """
    Unified object detection application with runtime strategy switching.
    
    Features:
    - Runtime switching between clustering strategies (BEV, DBSCAN)
    - Optional classification with PointNet + Subspace
    - Unified visualization and controls
    - Performance monitoring
    - Configuration management
    """
    
    def __init__(self, sensor_manager: SensorManager, 
                 initial_clustering_strategy: str = 'bev',
                 enable_classification: bool = True):
        """
        Initialize unified detection application.
        
        Args:
            sensor_manager: Sensor manager for input data
            initial_clustering_strategy: 'bev' or 'dbscan'
            enable_classification: Whether to enable object classification
        """
        self.sensor_manager = sensor_manager
        self.enable_classification = enable_classification
        
        # Create configurations
        self.configs = {
            'bev': ConfigFactory.create_preset('realtime_bev'),
            'pointnet': ConfigFactory.create_preset('offline_pointnet')
        }
        
        # Initialize strategies
        self.clustering_strategies = {
            'bev': BEVClusteringStrategy(
                roi_params=self.configs['bev'].get('roi_params'),
                display_roi_params=self.configs['bev'].get('display_roi_params'),
                clustering_params=self.configs['bev'].get('clustering_params')
            ),
            'dbscan': DBSCANClusteringStrategy()
        }
        
        # Initialize classification strategy if enabled
        self.classification_strategy = None
        if enable_classification:
            self.classification_strategy = PointNetSubspaceClassifierStrategy(
                model_dir=self.configs['pointnet'].get('model_params.model_dir', 'pointnet_cls'),
                dataset=self.configs['pointnet'].get('model_params.dataset', 'simple'),
                ndim=self.configs['pointnet'].get('subspace_params.ndim', 50),
                use_gds=self.configs['pointnet'].get('subspace_params.use_gds', True)
            )
            # Load the model
            self.classification_strategy.load_model()
        
        # Current active strategy
        self.current_clustering_strategy = initial_clustering_strategy
        
        # Initialize object detector
        self.detector = ObjectDetector(
            clustering_strategy=self.clustering_strategies[initial_clustering_strategy],
            classification_strategy=self.classification_strategy
        )
        
        # Initialize visualization and temporal filtering
        self.visualizer = VisualizationEngine()
        self.temporal_filter = TemporalFilter()
        
        # Application state
        self.is_running = False
        self.frame_count = 0
        self.show_help = False
        
        # Performance tracking
        self.performance_stats = {
            'total_frames': 0,
            'total_time': 0.0,
            'average_fps': 0.0,
            'strategy_switch_count': 0
        }
    
    def start(self) -> bool:
        """
        Start the unified detection application.
        
        Returns:
            True if started successfully
        """
        # Initialize and start sensor
        if not self.sensor_manager.initialize():
            print("Error: Failed to initialize sensor")
            return False
        
        if not self.sensor_manager.start():
            print("Error: Failed to start sensor")
            return False
        
        self.is_running = True
        print("Unified Detection Application Started")
        print("=" * 50)
        print(f"Initial clustering strategy: {self.current_clustering_strategy}")
        print(f"Classification: {'Enabled' if self.enable_classification else 'Disabled'}")
        print("Press 'h' for help, 'q' to quit")
        
        return True
    
    def stop(self):
        """Stop the application."""
        self.is_running = False
        self.sensor_manager.stop()
        cv2.destroyAllWindows()
        print("Application stopped")
    
    def run(self):
        """Main application loop."""
        if not self.start():
            return
        
        try:
            while self.is_running:
                loop_start = time.time()
                
                # Get frame from sensor
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                # Check for end of stream
                if metadata.get('end_of_stream', False):
                    print("End of stream reached")
                    break
                
                if vertices is not None:
                    # Detect objects
                    detected_objects, timing_info = self.detector.detect_objects(vertices, color_image)
                    
                    # Add to temporal filter
                    self.temporal_filter.add_frame_to_history(
                        objects=detected_objects,
                        timestamp=time.time()
                    )
                    
                    # Create visualization
                    self._create_visualization(vertices, color_image, detected_objects, timing_info, metadata)
                    
                    self.frame_count += 1
                    
                    # Update performance stats
                    loop_time = time.time() - loop_start
                    self._update_performance_stats(loop_time)
                
                # Handle user input
                if not self._handle_input():
                    break
                    
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def _create_visualization(self, vertices: np.ndarray, color_image: Optional[np.ndarray],
                            detected_objects: List[Dict[str, Any]], timing_info: Dict[str, float],
                            metadata: Dict[str, Any]):
        """Create and display visualization."""
        # Get strategy info for display
        strategy_info = self.detector.get_strategy_info()
        
        # Create main visualization based on current strategy
        if self.current_clustering_strategy == 'bev':
            # For BEV, create the BEV view
            main_image = self._create_bev_visualization(vertices, detected_objects)
        else:
            # For DBSCAN, create point cloud view
            main_image = self._create_3d_visualization(vertices, detected_objects)
        
        # Add performance overlay
        if main_image is not None:
            main_image = self.visualizer.add_performance_overlay(
                main_image, timing_info, strategy_info, self.frame_count
            )
            
            # Add object information overlay
            self._add_object_overlay(main_image, detected_objects)
            
            # Add help overlay if requested
            if self.show_help:
                self._add_help_overlay(main_image)
        
        # Combine with color image if available
        if color_image is not None and main_image is not None:
            # Resize images to same height
            height = min(color_image.shape[0], main_image.shape[0])
            color_resized = cv2.resize(color_image, (int(color_image.shape[1] * height / color_image.shape[0]), height))
            main_resized = cv2.resize(main_image, (int(main_image.shape[1] * height / main_image.shape[0]), height))
            
            # Stack horizontally
            combined = np.hstack([color_resized, main_resized])
        else:
            combined = main_image if main_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Display
        cv2.imshow('Unified Object Detection', combined)
    
    def _create_bev_visualization(self, vertices: np.ndarray, detected_objects: List[Dict[str, Any]]) -> np.ndarray:
        """Create BEV visualization."""
        if vertices is None or len(vertices) == 0:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Get BEV bounds from config
        bounds = self.configs['bev'].get_display_bounds()
        x_range = bounds['x_range']
        z_range = bounds['z_range']
        
        # Create BEV image using ImageProcessor
        from module.processing.image_processor import ImageProcessor
        image_processor = ImageProcessor()
        
        bev_image = image_processor.create_bev_image(vertices, x_range, z_range)
        
        # Draw detected objects as bounding boxes
        if detected_objects:
            height, width = bev_image.shape[:2]
            for obj in detected_objects:
                center = obj.get('center', [0, 0, 0])
                dimensions = obj.get('dimensions', [0, 0, 0])
                class_name = obj.get('class_name', 'Unknown')
                
                # Convert 3D coordinates to 2D image coordinates
                x_img = int((center[0] - x_range[0]) / (x_range[1] - x_range[0]) * width)
                z_img = int((center[2] - z_range[0]) / (z_range[1] - z_range[0]) * height)
                z_img = height - z_img  # Flip Z
                
                # Draw bounding box
                half_width = int(dimensions[0] / (x_range[1] - x_range[0]) * width / 2)
                half_depth = int(dimensions[2] / (z_range[1] - z_range[0]) * height / 2)
                
                x1, x2 = max(0, x_img - half_width), min(width-1, x_img + half_width)
                z1, z2 = max(0, z_img - half_depth), min(height-1, z_img + half_depth)
                
                color = self.visualizer.get_color_for_class(class_name)
                cv2.rectangle(bev_image, (x1, z1), (x2, z2), color, 2)
                
                # Add label
                cv2.putText(bev_image, class_name, (x_img + 5, z_img - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return bev_image
    
    def _create_3d_visualization(self, vertices: np.ndarray, detected_objects: List[Dict[str, Any]]) -> np.ndarray:
        """Create 3D point cloud visualization."""
        if vertices is None or len(vertices) == 0:
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Simple 3D to 2D projection for visualization
        # This is a simplified version - could be enhanced with proper 3D rendering
        img_size = 400
        
        # Project points to XY plane (top-down view)
        x_coords = vertices[:, 0]
        y_coords = vertices[:, 1]
        z_coords = vertices[:, 2]
        
        # Calculate bounds
        if len(x_coords) > 0:
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            
            # Add padding
            x_padding = (x_max - x_min) * 0.1
            z_padding = (z_max - z_min) * 0.1
            x_min -= x_padding
            x_max += x_padding
            z_min -= z_padding
            z_max += z_padding
            
            # Convert to image coordinates
            if x_max > x_min and z_max > z_min:
                x_img = ((x_coords - x_min) / (x_max - x_min) * img_size).astype(int)
                z_img = ((z_coords - z_min) / (z_max - z_min) * img_size).astype(int)
                z_img = img_size - z_img  # Flip Z for display
                
                # Create image
                pc_image = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                
                # Draw points
                valid_mask = (x_img >= 0) & (x_img < img_size) & (z_img >= 0) & (z_img < img_size)
                for i in np.where(valid_mask)[0]:
                    cv2.circle(pc_image, (x_img[i], z_img[i]), 1, (255, 255, 255), -1)
                
                # Draw detected objects
                for obj in detected_objects:
                    center = obj.get('center', [0, 0, 0])
                    class_name = obj.get('class_name', 'Unknown')
                    
                    # Convert to image coordinates
                    center_x_img = int((center[0] - x_min) / (x_max - x_min) * img_size)
                    center_z_img = int((center[2] - z_min) / (z_max - z_min) * img_size)
                    center_z_img = img_size - center_z_img
                    
                    color = self.visualizer.get_color_for_class(class_name)
                    cv2.circle(pc_image, (center_x_img, center_z_img), 5, color, -1)
                    cv2.putText(pc_image, class_name, (center_x_img + 8, center_z_img - 8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                return pc_image
        
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    def _add_object_overlay(self, image: np.ndarray, detected_objects: List[Dict[str, Any]]):
        """Add object information overlay."""
        if not detected_objects:
            return
        
        y_offset = image.shape[0] - 100
        cv2.putText(image, f"Detected Objects: {len(detected_objects)}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, obj in enumerate(detected_objects[:3]):  # Show first 3 objects
            class_name = obj.get('class_name', 'Unknown')
            confidence = obj.get('confidence', 0.0)
            y_pos = y_offset + 20 + i * 20
            
            if confidence > 0:
                text = f"{class_name} ({confidence:.2f})"
            else:
                text = class_name
            
            cv2.putText(image, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _add_help_overlay(self, image: np.ndarray):
        """Add help overlay."""
        help_text = [
            "CONTROLS:",
            "q - Quit",
            "h - Toggle help",
            "s - Switch clustering strategy",
            "c - Toggle classification",
            "r - Reset statistics",
            "Space - Pause (if applicable)"
        ]
        
        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (300, 10 + len(help_text) * 25 + 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        for i, text in enumerate(help_text):
            cv2.putText(image, text, (15, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _handle_input(self) -> bool:
        """
        Handle keyboard input.
        
        Returns:
            True to continue, False to quit
        """
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('h'):
            self.show_help = not self.show_help
            print(f"Help {'shown' if self.show_help else 'hidden'}")
        elif key == ord('s'):
            self._switch_clustering_strategy()
        elif key == ord('c'):
            self._toggle_classification()
        elif key == ord('r'):
            self._reset_statistics()
        elif key == ord(' '):
            print("Pause requested (if supported by sensor)")
        
        return True
    
    def _switch_clustering_strategy(self):
        """Switch between clustering strategies."""
        if self.current_clustering_strategy == 'bev':
            self.current_clustering_strategy = 'dbscan'
        else:
            self.current_clustering_strategy = 'bev'
        
        # Update detector strategy
        self.detector.set_clustering_strategy(self.clustering_strategies[self.current_clustering_strategy])
        
        # Reset temporal filter when switching strategies
        self.temporal_filter.reset_history()
        
        self.performance_stats['strategy_switch_count'] += 1
        print(f"Switched to {self.current_clustering_strategy} clustering strategy")
    
    def _toggle_classification(self):
        """Toggle classification on/off."""
        if self.classification_strategy is None:
            print("Classification not available")
            return
        
        if self.detector.classification_strategy is None:
            self.detector.set_classification_strategy(self.classification_strategy)
            print("Classification enabled")
        else:
            self.detector.set_classification_strategy(None)
            print("Classification disabled")
    
    def _reset_statistics(self):
        """Reset performance statistics."""
        self.detector.reset_statistics()
        self.temporal_filter.reset_history()
        self.performance_stats = {
            'total_frames': 0,
            'total_time': 0.0,
            'average_fps': 0.0,
            'strategy_switch_count': 0
        }
        print("Statistics reset")
    
    def _update_performance_stats(self, loop_time: float):
        """Update performance statistics."""
        self.performance_stats['total_frames'] += 1
        self.performance_stats['total_time'] += loop_time
        
        if self.performance_stats['total_time'] > 0:
            self.performance_stats['average_fps'] = self.performance_stats['total_frames'] / self.performance_stats['total_time']
    
    def get_performance_summary(self) -> str:
        """Get performance summary string."""
        detector_stats = self.detector.get_statistics()
        temporal_stats = self.temporal_filter.get_statistics()
        
        summary = [
            "=== Performance Summary ===",
            f"Total frames processed: {self.performance_stats['total_frames']}",
            f"Average FPS: {self.performance_stats['average_fps']:.1f}",
            f"Strategy switches: {self.performance_stats['strategy_switch_count']}",
            f"Current strategy: {self.current_clustering_strategy}",
            f"Classification: {'Enabled' if self.detector.classification_strategy else 'Disabled'}",
            f"Average detection time: {detector_stats.get('average_processing_time', 0)*1000:.1f}ms",
            f"Temporal filter window: {temporal_stats.get('window_size', 0)} frames"
        ]
        
        return "\n".join(summary)