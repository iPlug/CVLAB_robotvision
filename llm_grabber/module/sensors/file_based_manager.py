"""
File-based sensor manager for offline processing of PLY files.
"""

import open3d as o3d
import numpy as np
import os
import glob
from typing import Tuple, Optional, Dict, Any, List

from .sensor_manager import SensorManager


class FileBasedManager(SensorManager):
    """
    File-based sensor manager for processing PLY files from disk.
    
    Supports:
    - Sequential PLY file processing
    - Automatic file discovery
    - Frame range selection
    - Metadata extraction
    """
    
    def __init__(self, folder_path: str, file_pattern: str = "*.ply"):
        """
        Initialize file-based manager.
        
        Args:
            folder_path: Path to folder containing PLY files
            file_pattern: Glob pattern for file matching
        """
        super().__init__()
        
        self.folder_path = folder_path
        self.file_pattern = file_pattern
        self.file_list = []
        self.current_index = 0
        self.start_frame = 0
        self.end_frame = None
        
    def initialize(self, **kwargs) -> bool:
        """
        Initialize file-based sensor.
        
        Kwargs:
            start_frame: Starting frame index (default: 0)
            end_frame: Ending frame index (default: None for all)
            sort_files: Whether to sort files (default: True)
            
        Returns:
            True if initialization successful
        """
        try:
            # Get parameters
            self.start_frame = kwargs.get('start_frame', 0)
            self.end_frame = kwargs.get('end_frame', None)
            sort_files = kwargs.get('sort_files', True)
            
            # Find all PLY files
            pattern = os.path.join(self.folder_path, self.file_pattern)
            self.file_list = glob.glob(pattern)
            
            if not self.file_list:
                print(f"Error: No files found matching pattern '{pattern}'")
                return False
            
            if sort_files:
                self.file_list = sorted(self.file_list)
            
            # Apply frame range
            if self.end_frame is not None:
                self.file_list = self.file_list[self.start_frame:self.end_frame]
            else:
                self.file_list = self.file_list[self.start_frame:]
            
            self.current_index = 0
            
            print(f"Found {len(self.file_list)} files in {self.folder_path}")
            print(f"Processing frames {self.start_frame} to {self.start_frame + len(self.file_list) - 1}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing file-based sensor: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start file-based processing.
        
        Returns:
            True if started successfully
        """
        if not self.is_initialized:
            print("Error: File-based sensor not initialized")
            return False
        
        if not self.file_list:
            print("Error: No files to process")
            return False
        
        self.current_index = 0
        self.frame_count = 0
        self.is_running = True
        
        print(f"Started file-based processing with {len(self.file_list)} files")
        return True
    
    def stop(self):
        """Stop file-based processing."""
        self.is_running = False
        print("File-based processing stopped")
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """
        Get next frame from file sequence.
        
        Returns:
            Tuple of (vertices, color_image, metadata)
        """
        if not self.is_running:
            return None, None, {'error': 'Sensor not running'}
        
        if self.current_index >= len(self.file_list):
            return None, None, {'end_of_stream': True}
        
        try:
            # Get current file
            file_path = self.file_list[self.current_index]
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(file_path)
            
            if len(pcd.points) == 0:
                print(f"Warning: Empty point cloud in {os.path.basename(file_path)}")
                vertices = np.array([])
            else:
                vertices = np.asarray(pcd.points)
            
            # Extract color information if available
            color_image = None
            if pcd.has_colors():
                # Convert colors to image-like format (not a real image, just color data)
                colors = np.asarray(pcd.colors)
                # This is point cloud color data, not an actual image
                # For compatibility, we'll leave color_image as None
                # Real implementations might look for corresponding image files
            
            # Create metadata
            metadata = {
                'frame_id': self.frame_count,
                'file_index': self.current_index,
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'total_files': len(self.file_list),
                'progress': (self.current_index + 1) / len(self.file_list),
                'has_colors': pcd.has_colors(),
                'num_points': len(pcd.points)
            }
            
            # Add file statistics
            try:
                stat = os.stat(file_path)
                metadata['file_size'] = stat.st_size
                metadata['file_modified'] = stat.st_mtime
            except:
                pass
            
            # Advance to next file
            self.current_index += 1
            self.frame_count += 1
            
            return vertices, color_image, metadata
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            # Skip this file and try the next one
            self.current_index += 1
            return None, None, {'error': str(e), 'skipped_file': file_path}
    
    def seek_to_frame(self, frame_index: int) -> bool:
        """
        Seek to specific frame index.
        
        Args:
            frame_index: Frame index to seek to
            
        Returns:
            True if seek successful
        """
        if 0 <= frame_index < len(self.file_list):
            self.current_index = frame_index
            self.frame_count = frame_index
            return True
        return False
    
    def get_current_file_path(self) -> Optional[str]:
        """
        Get path of current file.
        
        Returns:
            Current file path or None
        """
        if 0 <= self.current_index < len(self.file_list):
            return self.file_list[self.current_index]
        return None
    
    def get_file_list(self) -> List[str]:
        """
        Get list of all files to be processed.
        
        Returns:
            List of file paths
        """
        return self.file_list.copy()
    
    def get_remaining_files(self) -> int:
        """
        Get number of remaining files to process.
        
        Returns:
            Number of remaining files
        """
        return max(0, len(self.file_list) - self.current_index)
    
    def has_more_frames(self) -> bool:
        """
        Check if there are more frames to process.
        
        Returns:
            True if more frames available
        """
        return self.current_index < len(self.file_list)
    
    def reset_to_beginning(self):
        """Reset to the beginning of the file sequence."""
        self.current_index = 0
        self.frame_count = 0
    
    def get_sensor_info(self) -> Dict[str, Any]:
        """Get file-based sensor information"""
        return {
            'sensor_type': 'File-based',
            'folder_path': self.folder_path,
            'file_pattern': self.file_pattern,
            'total_files': len(self.file_list),
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'current_index': self.current_index,
            'current_file': self.get_current_file_path()
        }
    
    def is_available(self) -> bool:
        """Check if files are available"""
        return os.path.exists(self.folder_path) and len(self.file_list) > 0
    
    def get_progress_info(self) -> Dict[str, Any]:
        """
        Get detailed progress information.
        
        Returns:
            Progress information dictionary
        """
        total_files = len(self.file_list)
        
        return {
            'current_index': self.current_index,
            'total_files': total_files,
            'progress_ratio': self.current_index / total_files if total_files > 0 else 0.0,
            'progress_percent': (self.current_index / total_files * 100) if total_files > 0 else 0.0,
            'remaining_files': self.get_remaining_files(),
            'completed_files': self.current_index,
            'current_file': os.path.basename(self.get_current_file_path() or ""),
            'has_more': self.has_more_frames()
        }
    
    def estimate_processing_time(self, avg_time_per_frame: float) -> Dict[str, float]:
        """
        Estimate remaining processing time.
        
        Args:
            avg_time_per_frame: Average processing time per frame in seconds
            
        Returns:
            Time estimates dictionary
        """
        remaining_files = self.get_remaining_files()
        
        estimated_remaining_time = remaining_files * avg_time_per_frame
        estimated_total_time = len(self.file_list) * avg_time_per_frame
        elapsed_time = self.current_index * avg_time_per_frame
        
        return {
            'estimated_remaining_seconds': estimated_remaining_time,
            'estimated_total_seconds': estimated_total_time,
            'elapsed_seconds': elapsed_time,
            'estimated_remaining_minutes': estimated_remaining_time / 60.0,
            'estimated_total_minutes': estimated_total_time / 60.0,
            'elapsed_minutes': elapsed_time / 60.0
        }