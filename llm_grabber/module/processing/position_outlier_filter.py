"""
Position Outlier Filter for LiDAR Object Tracking

This module implements outlier detection and removal for object positions after
coordinate transformation to robot space. It uses multiple criteria to identify
and filter out spurious detections:

1. Distance-based filtering - Remove positions too far from expected ranges
2. Temporal consistency - Remove positions that jump too far from recent history  
3. Workspace validation - Remove positions outside robot workspace
4. Velocity-based filtering - Remove positions with unrealistic movement speeds

Key Features:
- Multi-criteria outlier detection
- Temporal position history tracking
- Configurable thresholds for different scenarios
- Statistical outlier detection using IQR and Z-score methods
- Real-time performance optimization
"""

import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import statistics


class PositionOutlierFilter:
    """
    Filter outliers from detected object positions using multiple criteria.
    
    This filter operates after coordinate transformation and uses robot-space
    coordinates to make intelligent decisions about position validity.
    """
    
    def __init__(self,
                 max_distance_from_base: float = 500.0,        # mm
                 max_velocity: float = 1000.0,                 # mm/s  
                 max_position_jump: float = 100.0,             # mm
                 temporal_window_size: int = 10,
                 enable_statistical_filtering: bool = True,
                 statistical_threshold: float = 2.0,          # Z-score threshold
                 workspace_bounds: Optional[Dict[str, float]] = None):
        """
        Initialize position outlier filter.
        
        Args:
            max_distance_from_base: Maximum allowed distance from robot base (mm)
            max_velocity: Maximum allowed velocity between detections (mm/s)
            max_position_jump: Maximum allowed position change between frames (mm)
            temporal_window_size: Number of recent positions to keep for analysis
            enable_statistical_filtering: Enable statistical outlier detection
            statistical_threshold: Z-score threshold for statistical filtering
            workspace_bounds: Robot workspace bounds for validation
        """
        
        # Distance filtering parameters
        self.max_distance_from_base = max_distance_from_base
        self.max_velocity = max_velocity
        self.max_position_jump = max_position_jump
        
        # Temporal filtering parameters
        self.temporal_window_size = temporal_window_size
        self.position_history = deque(maxlen=temporal_window_size)
        self.timestamp_history = deque(maxlen=temporal_window_size)
        
        # Statistical filtering parameters
        self.enable_statistical_filtering = enable_statistical_filtering
        self.statistical_threshold = statistical_threshold
        
        # Workspace validation
        self.workspace_bounds = workspace_bounds or {
            'x_min': -400, 'x_max': 400,
            'y_min': -400, 'y_max': 400,
            'z_min': 50,   'z_max': 400
        }
        
        # Filter statistics
        self.filter_stats = {
            'total_positions': 0,
            'distance_outliers': 0,
            'velocity_outliers': 0,
            'position_jump_outliers': 0,
            'workspace_outliers': 0,
            'statistical_outliers': 0,
            'total_outliers_removed': 0
        }
        
        # Last valid position for continuity
        self.last_valid_position = None
        self.last_valid_timestamp = None
    
    def filter_position(self, robot_coords: Dict[str, Any], current_time: Optional[float] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Filter a robot position to determine if it's an outlier.
        
        Args:
            robot_coords: Dictionary containing robot position and metadata
            current_time: Current timestamp (uses time.time() if None)
            
        Returns:
            Tuple of (is_valid, filter_details)
        """
        if current_time is None:
            current_time = time.time()
        
        self.filter_stats['total_positions'] += 1
        
        # Extract position
        if not robot_coords or not robot_coords.get('is_valid', False):
            return False, {'reason': 'invalid_input'}
        
        position = robot_coords['robot_position']  # [x, y, z] in mm
        distance_from_base = robot_coords.get('distance_from_base', np.linalg.norm(position))
        
        # Initialize filter details
        filter_details = {
            'position': position.copy(),
            'distance_from_base': distance_from_base,
            'timestamp': current_time,
            'filters_applied': [],
            'filter_results': {}
        }
        
        # 1. Distance-based filtering
        distance_valid = self._check_distance_filter(distance_from_base)
        filter_details['filter_results']['distance'] = distance_valid
        filter_details['filters_applied'].append('distance')
        
        if not distance_valid:
            self.filter_stats['distance_outliers'] += 1
            self.filter_stats['total_outliers_removed'] += 1
            filter_details['reason'] = 'distance_outlier'
            return False, filter_details
        
        # 2. Workspace validation
        workspace_valid = self._check_workspace_bounds(position)
        filter_details['filter_results']['workspace'] = workspace_valid
        filter_details['filters_applied'].append('workspace')
        
        if not workspace_valid:
            self.filter_stats['workspace_outliers'] += 1
            self.filter_stats['total_outliers_removed'] += 1
            filter_details['reason'] = 'workspace_outlier'
            return False, filter_details
        
        # 3. Temporal consistency filtering (if we have previous data)
        if self.last_valid_position is not None and self.last_valid_timestamp is not None:
            
            # Position jump check
            position_jump_valid = self._check_position_jump(position)
            filter_details['filter_results']['position_jump'] = position_jump_valid
            filter_details['filters_applied'].append('position_jump')
            
            if not position_jump_valid:
                self.filter_stats['position_jump_outliers'] += 1
                self.filter_stats['total_outliers_removed'] += 1
                filter_details['reason'] = 'position_jump_outlier'
                return False, filter_details
            
            # Velocity check
            velocity_valid = self._check_velocity_filter(position, current_time)
            filter_details['filter_results']['velocity'] = velocity_valid
            filter_details['filters_applied'].append('velocity')
            
            if not velocity_valid:
                self.filter_stats['velocity_outliers'] += 1
                self.filter_stats['total_outliers_removed'] += 1
                filter_details['reason'] = 'velocity_outlier'
                return False, filter_details
        
        # 4. Statistical filtering (if enabled and we have enough history)
        if self.enable_statistical_filtering and len(self.position_history) >= 3:
            statistical_valid = self._check_statistical_filter(position)
            filter_details['filter_results']['statistical'] = statistical_valid
            filter_details['filters_applied'].append('statistical')
            
            if not statistical_valid:
                self.filter_stats['statistical_outliers'] += 1
                self.filter_stats['total_outliers_removed'] += 1
                filter_details['reason'] = 'statistical_outlier'
                return False, filter_details
        
        # Position passed all filters - add to history and update last valid
        self._add_to_history(position, current_time)
        self.last_valid_position = position.copy()
        self.last_valid_timestamp = current_time
        
        filter_details['reason'] = 'valid'
        return True, filter_details
    
    def _check_distance_filter(self, distance_from_base: float) -> bool:
        """Check if distance from robot base is within acceptable range."""
        return distance_from_base <= self.max_distance_from_base
    
    def _check_workspace_bounds(self, position: np.ndarray) -> bool:
        """Check if position is within robot workspace bounds."""
        x, y, z = position
        bounds = self.workspace_bounds
        
        if not (bounds['x_min'] <= x <= bounds['x_max']):
            return False
        if not (bounds['y_min'] <= y <= bounds['y_max']):
            return False
        if not (bounds['z_min'] <= z <= bounds['z_max']):
            return False
        
        return True
    
    def _check_position_jump(self, position: np.ndarray) -> bool:
        """Check if position change is within acceptable limits."""
        if self.last_valid_position is None:
            return True
        
        position_change = np.linalg.norm(position - self.last_valid_position)
        return position_change <= self.max_position_jump
    
    def _check_velocity_filter(self, position: np.ndarray, current_time: float) -> bool:
        """Check if implied velocity is within acceptable limits."""
        if self.last_valid_position is None or self.last_valid_timestamp is None:
            return True
        
        time_delta = current_time - self.last_valid_timestamp
        if time_delta <= 0:
            return True  # Can't calculate velocity with zero or negative time
        
        position_change = np.linalg.norm(position - self.last_valid_position)
        velocity = position_change / time_delta  # mm/s
        
        return velocity <= self.max_velocity
    
    def _check_statistical_filter(self, position: np.ndarray) -> bool:
        """Check if position is a statistical outlier using Z-score method."""
        if len(self.position_history) < 3:
            return True
        
        # Convert history to array for statistical analysis
        history_array = np.array(list(self.position_history))
        
        # Calculate distances from history mean
        history_distances = [np.linalg.norm(pos - position) for pos in history_array]
        current_distance = np.linalg.norm(position - np.mean(history_array, axis=0))
        
        # Calculate Z-score based on distance from mean
        if len(history_distances) > 1:
            mean_distance = np.mean(history_distances)
            std_distance = np.std(history_distances)
            
            if std_distance > 0:
                z_score = abs((current_distance - mean_distance) / std_distance)
                return z_score <= self.statistical_threshold
        
        return True
    
    def _add_to_history(self, position: np.ndarray, timestamp: float):
        """Add position and timestamp to history."""
        self.position_history.append(position.copy())
        self.timestamp_history.append(timestamp)
    
    def get_smoothed_position(self, method: str = 'moving_average') -> Optional[np.ndarray]:
        """
        Get smoothed position based on recent history.
        
        Args:
            method: Smoothing method ('moving_average', 'weighted_average', 'median')
            
        Returns:
            Smoothed position or None if insufficient history
        """
        if len(self.position_history) < 2:
            return self.last_valid_position
        
        history_array = np.array(list(self.position_history))
        
        if method == 'moving_average':
            return np.mean(history_array, axis=0)
        elif method == 'weighted_average':
            # More recent positions get higher weights
            weights = np.linspace(0.1, 1.0, len(history_array))
            weights = weights / np.sum(weights)
            return np.average(history_array, axis=0, weights=weights)
        elif method == 'median':
            return np.median(history_array, axis=0)
        else:
            return np.mean(history_array, axis=0)
    
    def get_position_confidence(self) -> float:
        """
        Calculate confidence score for position estimates based on consistency.
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if len(self.position_history) < 2:
            return 0.5  # Neutral confidence with limited data
        
        # Calculate position variance as a measure of consistency
        history_array = np.array(list(self.position_history))
        position_variance = np.var(history_array, axis=0)
        avg_variance = np.mean(position_variance)
        
        # Convert variance to confidence (lower variance = higher confidence)
        # Use exponential decay to map variance to 0-1 range
        confidence = np.exp(-avg_variance / 100.0)  # Scale factor of 100mm^2
        
        return min(max(confidence, 0.0), 1.0)
    
    def reset_history(self):
        """Reset all position history and statistics."""
        self.position_history.clear()
        self.timestamp_history.clear()
        self.last_valid_position = None
        self.last_valid_timestamp = None
        
        # Reset statistics
        for key in self.filter_stats:
            self.filter_stats[key] = 0
    
    def update_parameters(self, **kwargs):
        """Update filter parameters dynamically."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filter statistics."""
        stats = self.filter_stats.copy()
        
        if stats['total_positions'] > 0:
            stats['outlier_rate'] = stats['total_outliers_removed'] / stats['total_positions']
            stats['success_rate'] = 1.0 - stats['outlier_rate']
        else:
            stats['outlier_rate'] = 0.0
            stats['success_rate'] = 1.0
        
        stats['history_length'] = len(self.position_history)
        stats['position_confidence'] = self.get_position_confidence()
        
        if self.last_valid_position is not None:
            stats['last_valid_position'] = self.last_valid_position.copy()
            stats['last_valid_timestamp'] = self.last_valid_timestamp
        
        return stats
    
    def create_filter_report(self) -> str:
        """Create human-readable filter performance report."""
        stats = self.get_filter_statistics()
        
        report = f"""
Position Outlier Filter Report
==============================
Total positions processed: {stats['total_positions']}
Valid positions: {stats['total_positions'] - stats['total_outliers_removed']}
Outliers removed: {stats['total_outliers_removed']}
Success rate: {stats['success_rate']:.1%}

Outlier Breakdown:
- Distance outliers: {stats['distance_outliers']}
- Velocity outliers: {stats['velocity_outliers']}
- Position jump outliers: {stats['position_jump_outliers']}
- Workspace outliers: {stats['workspace_outliers']}
- Statistical outliers: {stats['statistical_outliers']}

Current Status:
- History length: {stats['history_length']}
- Position confidence: {stats['position_confidence']:.3f}
"""
        
        if 'last_valid_position' in stats:
            pos = stats['last_valid_position']
            report += f"- Last valid position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm\n"
        
        return report