"""
Temporal filtering utilities for frame-to-frame consistency.
"""

import numpy as np
import cv2
from typing import List, Dict, Any, Optional
from collections import deque


class TemporalFilter:
    """
    Temporal filtering for improving detection consistency across frames.
    
    Provides utilities for:
    - Frame-to-frame noise reduction
    - Object position stabilization
    - Persistent object tracking
    - FPS calculation and monitoring
    """
    
    def __init__(self, window_size: int = 5, fps_window_size: int = 30):
        """
        Initialize temporal filter.
        
        Args:
            window_size: Number of frames to consider for filtering
            fps_window_size: Number of frames for FPS calculation
        """
        self.window_size = window_size
        self.fps_window_size = fps_window_size
        
        # Frame history for different types of data
        self.image_history = deque(maxlen=window_size)
        self.object_history = deque(maxlen=window_size)
        
        # FPS tracking
        self.fps_history = deque(maxlen=fps_window_size)
        
        # Object position tracking for stabilization
        self.object_position_history = {}
    
    def add_frame_to_history(self, image: Optional[np.ndarray] = None,
                           objects: Optional[List[Dict[str, Any]]] = None,
                           timestamp: Optional[float] = None):
        """
        Add frame data to temporal history.
        
        Args:
            image: Frame image (for image-based filtering)
            objects: Detected objects (for object tracking)
            timestamp: Frame timestamp for FPS calculation
        """
        if image is not None:
            self.image_history.append(image.copy())
        
        if objects is not None:
            self.object_history.append(objects)
        
        if timestamp is not None:
            self.fps_history.append(timestamp)
    
    def apply_image_temporal_filtering(self, current_image: np.ndarray,
                                     persistence_threshold: float = 0.7) -> np.ndarray:
        """
        Apply temporal filtering to reduce image noise across frames.
        
        Args:
            current_image: Current frame image
            persistence_threshold: Minimum ratio of frames where pixel should be active
            
        Returns:
            Temporally filtered image
        """
        # Add current frame to history
        self.image_history.append(current_image.copy())
        
        # Need at least 3 frames for meaningful filtering
        if len(self.image_history) < min(3, self.window_size):
            return current_image
        
        # Convert all frames to binary if they aren't already
        binary_frames = []
        for frame in self.image_history:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            binary_frames.append(binary)
        
        # Stack frames and find persistent pixels
        frame_stack = np.stack(binary_frames, axis=0)
        
        # Calculate minimum number of frames for persistence
        min_frames = int(len(binary_frames) * persistence_threshold)
        persistent_pixels = np.sum(frame_stack == 255, axis=0) >= min_frames
        
        # Create filtered result
        if len(current_image.shape) == 3:
            result = np.zeros_like(current_image)
            result[persistent_pixels] = 255
        else:
            result = np.zeros_like(current_image)
            result[persistent_pixels] = 255
        
        return result
    
    def stabilize_object_positions(self, current_objects: List[Dict[str, Any]],
                                 max_distance: float = 0.1) -> List[Dict[str, Any]]:
        """
        Stabilize object positions using temporal averaging.
        
        Args:
            current_objects: Current frame detected objects
            max_distance: Maximum distance for object association
            
        Returns:
            Objects with stabilized positions
        """
        stabilized_objects = []
        
        for obj in current_objects:
            obj_id = obj.get('id', 0)
            current_center = obj.get('center', np.array([0, 0, 0]))
            
            # Initialize history for new objects
            if obj_id not in self.object_position_history:
                self.object_position_history[obj_id] = deque(maxlen=self.window_size)
            
            # Add current position to history
            self.object_position_history[obj_id].append(current_center.copy())
            
            # Calculate averaged position
            positions = list(self.object_position_history[obj_id])
            if len(positions) > 1:
                # Weight recent positions more heavily
                weights = np.linspace(0.5, 1.0, len(positions))
                weights = weights / np.sum(weights)
                
                avg_center = np.average(positions, axis=0, weights=weights)
            else:
                avg_center = current_center
            
            # Create stabilized object
            stabilized_obj = obj.copy()
            stabilized_obj['center'] = avg_center
            stabilized_obj['center_mm'] = avg_center * 1000  # Convert to mm
            stabilized_obj['stabilized'] = True
            
            stabilized_objects.append(stabilized_obj)
        
        # Clean up old object histories (objects that no longer exist)
        current_object_ids = {obj.get('id', 0) for obj in current_objects}
        old_ids = set(self.object_position_history.keys()) - current_object_ids
        for old_id in old_ids:
            del self.object_position_history[old_id]
        
        return stabilized_objects
    
    def track_object_consistency(self, current_objects: List[Dict[str, Any]],
                               min_consistency_frames: int = 3) -> List[Dict[str, Any]]:
        """
        Track object consistency across frames.
        
        Args:
            current_objects: Current frame detected objects
            min_consistency_frames: Minimum frames an object should appear
            
        Returns:
            Filtered objects that appear consistently
        """
        # Add current objects to history
        self.object_history.append(current_objects)
        
        # Not enough history for consistency check
        if len(self.object_history) < min_consistency_frames:
            return current_objects
        
        # Count object appearances across recent frames
        object_counts = {}
        
        for frame_objects in list(self.object_history)[-min_consistency_frames:]:
            for obj in frame_objects:
                obj_signature = self._get_object_signature(obj)
                if obj_signature not in object_counts:
                    object_counts[obj_signature] = 0
                object_counts[obj_signature] += 1
        
        # Filter objects that appear in enough frames
        consistent_objects = []
        for obj in current_objects:
            obj_signature = self._get_object_signature(obj)
            if object_counts.get(obj_signature, 0) >= min_consistency_frames:
                obj_copy = obj.copy()
                obj_copy['consistency_score'] = object_counts[obj_signature] / min_consistency_frames
                consistent_objects.append(obj_copy)
        
        return consistent_objects
    
    def _get_object_signature(self, obj: Dict[str, Any]) -> str:
        """
        Create a signature for object matching across frames.
        
        Args:
            obj: Object dictionary
            
        Returns:
            String signature for the object
        """
        center = obj.get('center', [0, 0, 0])
        dimensions = obj.get('dimensions', [0, 0, 0])
        
        # Round to avoid minor floating point differences
        center_rounded = [round(c, 2) for c in center]
        dims_rounded = [round(d, 2) for d in dimensions]
        
        return f"c{center_rounded}_d{dims_rounded}"
    
    def calculate_fps(self) -> float:
        """
        Calculate FPS from recent frame timestamps.
        
        Returns:
            Current FPS, or 0 if insufficient data
        """
        if len(self.fps_history) < 2:
            return 0.0
        
        # Calculate FPS from time span
        time_span = self.fps_history[-1] - self.fps_history[0]
        if time_span > 0:
            fps = (len(self.fps_history) - 1) / time_span
            return fps
        
        return 0.0
    
    def get_average_fps(self) -> float:
        """
        Get average FPS over the entire window.
        
        Returns:
            Average FPS
        """
        return self.calculate_fps()
    
    def reset_history(self):
        """Reset all temporal history."""
        self.image_history.clear()
        self.object_history.clear()
        self.fps_history.clear()
        self.object_position_history.clear()
    
    def set_window_size(self, window_size: int):
        """
        Change temporal window size.
        
        Args:
            window_size: New window size
        """
        self.window_size = window_size
        
        # Update deque max lengths
        self.image_history = deque(self.image_history, maxlen=window_size)
        self.object_history = deque(self.object_history, maxlen=window_size)
        
        # Update object position histories
        for obj_id in self.object_position_history:
            self.object_position_history[obj_id] = deque(
                self.object_position_history[obj_id], 
                maxlen=window_size
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get temporal filtering statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'window_size': self.window_size,
            'frames_in_history': len(self.image_history),
            'objects_in_history': len(self.object_history),
            'tracked_objects': len(self.object_position_history),
            'current_fps': self.calculate_fps(),
            'fps_samples': len(self.fps_history)
        }