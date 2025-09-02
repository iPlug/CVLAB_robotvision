"""
Batch processing application for offline analysis of PLY files.
"""

import os
import time
import numpy as np
from typing import Dict, Any, Optional, List

from module.core.object_detector import ObjectDetector
from module.strategies.dbscan_clustering_strategy import DBSCANClusteringStrategy
from module.strategies.pointnet_subspace_classifier_strategy import PointNetSubspaceClassifierStrategy
from module.sensors.file_based_manager import FileBasedManager
from module.config.config_factory import ConfigFactory


class BatchProcessingApp:
    """
    Batch processing application for offline analysis of point cloud files.
    
    Features:
    - High-quality DBSCAN clustering
    - PointNet + Subspace classification
    - Batch processing of PLY files
    - Progress tracking and reporting
    - Results export and analysis
    """
    
    def __init__(self, folder_path: str, 
                 start_frame: int = 0,
                 end_frame: Optional[int] = None,
                 enable_classification: bool = True,
                 save_results: bool = True):
        """
        Initialize batch processing application.
        
        Args:
            folder_path: Path to folder containing PLY files
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all)
            enable_classification: Whether to enable object classification
            save_results: Whether to save processing results
        """
        self.folder_path = folder_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.enable_classification = enable_classification
        self.save_results = save_results
        
        # Create configuration optimized for accuracy
        self.config = ConfigFactory.create_preset('offline_pointnet')
        
        # Initialize file-based sensor
        self.sensor_manager = FileBasedManager(folder_path)
        
        # Initialize DBSCAN clustering strategy
        self.clustering_strategy = DBSCANClusteringStrategy(
            dbscan_eps=self.config.get('dbscan_params.eps', 0.04),
            dbscan_min_points=self.config.get('dbscan_params.min_points', 10),
            min_object_height=self.config.get('object_params.min_object_height', 0.05),
            max_object_height=self.config.get('object_params.max_object_height', 1.0),
            max_points_for_processing=self.config.get('performance_params.max_points_for_processing', 50000)
        )
        
        # Initialize classification strategy if enabled
        self.classification_strategy = None
        if enable_classification:
            self.classification_strategy = PointNetSubspaceClassifierStrategy(
                model_dir=self.config.get('model_params.model_dir', 'pointnet_cls'),
                dataset=self.config.get('model_params.dataset', 'simple'),
                ndim=self.config.get('subspace_params.ndim', 50),
                use_gds=self.config.get('subspace_params.use_gds', True),
                checkpoint_path=self.config.get('model_params.checkpoint_path')
            )
            
            print("Loading PointNet model...")
            if not self.classification_strategy.load_model():
                print("Warning: Failed to load classification model, proceeding without classification")
                self.classification_strategy = None
                self.enable_classification = False
        
        # Initialize object detector
        self.detector = ObjectDetector(
            clustering_strategy=self.clustering_strategy,
            classification_strategy=self.classification_strategy
        )
        
        # Results storage
        self.results = []
        self.processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'total_objects_detected': 0,
            'total_processing_time': 0.0,
            'classification_accuracy': 0.0,
            'failed_files': []
        }
    
    def run(self) -> bool:
        """
        Run batch processing on all files.
        
        Returns:
            True if processing completed successfully
        """
        # Initialize sensor
        if not self.sensor_manager.initialize(
            start_frame=self.start_frame,
            end_frame=self.end_frame
        ):
            print("Error: Failed to initialize file-based sensor")
            return False
        
        if not self.sensor_manager.start():
            print("Error: Failed to start file-based sensor")
            return False
        
        # Get file list for progress tracking
        file_list = self.sensor_manager.get_file_list()
        self.processing_stats['total_files'] = len(file_list)
        
        print("Batch Processing Started")
        print("=" * 50)
        print(f"Processing {len(file_list)} files from {self.folder_path}")
        print(f"Clustering: DBSCAN")
        print(f"Classification: {'Enabled' if self.enable_classification else 'Disabled'}")
        print(f"Start frame: {self.start_frame}")
        print(f"End frame: {self.end_frame or 'All'}")
        print()
        
        start_time = time.time()
        
        try:
            while self.sensor_manager.has_more_frames():
                # Get frame
                vertices, color_image, metadata = self.sensor_manager.get_frame()
                
                if metadata.get('error'):
                    print(f"Error loading file: {metadata['error']}")
                    self.processing_stats['failed_files'].append(metadata.get('skipped_file', 'unknown'))
                    continue
                
                if vertices is not None:
                    # Process frame
                    frame_result = self._process_single_frame(vertices, metadata)
                    self.results.append(frame_result)
                    
                    self.processing_stats['processed_files'] += 1
                    self.processing_stats['total_objects_detected'] += frame_result['num_objects']
                    
                    # Print progress
                    progress = self.sensor_manager.get_progress_info()
                    self._print_progress(frame_result, progress)
            
            # Calculate final statistics
            total_time = time.time() - start_time
            self.processing_stats['total_processing_time'] = total_time
            
            # Save results if requested
            if self.save_results:
                self._save_results()
            
            # Print summary
            self._print_summary()
            
            return True
            
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            return False
        except Exception as e:
            print(f"Error during batch processing: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.sensor_manager.stop()
    
    def _process_single_frame(self, vertices: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single frame and return results.
        
        Args:
            vertices: Point cloud vertices
            metadata: Frame metadata
            
        Returns:
            Dictionary with processing results
        """
        frame_start = time.time()
        
        # Detect objects
        detected_objects, timing_info = self.detector.detect_objects(vertices)
        
        processing_time = time.time() - frame_start
        
        # Create result record
        result = {
            'frame_id': metadata.get('frame_id', 0),
            'file_path': metadata.get('file_path', ''),
            'file_name': metadata.get('file_name', ''),
            'num_points': metadata.get('num_points', len(vertices)),
            'num_objects': len(detected_objects),
            'objects': detected_objects,
            'processing_time': processing_time,
            'timing_info': timing_info,
            'metadata': metadata
        }
        
        return result
    
    def _print_progress(self, frame_result: Dict[str, Any], progress: Dict[str, Any]):
        """Print processing progress."""
        file_name = frame_result['file_name']
        num_objects = frame_result['num_objects']
        processing_time = frame_result['processing_time']
        
        progress_percent = progress['progress_percent']
        current_index = progress['current_index']
        total_files = progress['total_files']
        
        print(f"[{current_index:4d}/{total_files}] ({progress_percent:5.1f}%) "
              f"{file_name:30s} - {num_objects:2d} objects - {processing_time:.3f}s")
        
        # Print object details if classification is enabled
        if self.enable_classification and num_objects > 0:
            for i, obj in enumerate(frame_result['objects'][:3]):  # Show first 3
                class_name = obj.get('class_name', 'Unknown')
                confidence = obj.get('confidence', 0.0)
                if confidence > 0:
                    print(f"    Object {i+1}: {class_name} (confidence: {confidence:.2f})")
                else:
                    print(f"    Object {i+1}: {class_name}")
    
    def _save_results(self):
        """Save processing results to files."""
        # Create results directory
        results_dir = os.path.join(self.folder_path, 'batch_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results as JSON
        import json
        
        # Prepare data for JSON serialization
        json_results = []
        for result in self.results:
            json_result = result.copy()
            
            # Convert numpy arrays to lists for JSON serialization
            objects = []
            for obj in result['objects']:
                json_obj = {}
                for key, value in obj.items():
                    if isinstance(value, np.ndarray):
                        json_obj[key] = value.tolist()
                    else:
                        json_obj[key] = value
                objects.append(json_obj)
            
            json_result['objects'] = objects
            json_results.append(json_result)
        
        # Save results
        results_file = os.path.join(results_dir, 'detailed_results.json')
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Detailed results saved to: {results_file}")
        
        # Save summary CSV
        import csv
        
        summary_file = os.path.join(results_dir, 'summary.csv')
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            if self.enable_classification:
                writer.writerow(['File', 'Objects', 'Processing_Time_s', 'Classifications'])
            else:
                writer.writerow(['File', 'Objects', 'Processing_Time_s'])
            
            # Data
            for result in self.results:
                row = [
                    result['file_name'],
                    result['num_objects'],
                    f"{result['processing_time']:.3f}"
                ]
                
                if self.enable_classification:
                    classifications = [obj.get('class_name', 'Unknown') for obj in result['objects']]
                    row.append('; '.join(classifications))
                
                writer.writerow(row)
        
        print(f"Summary CSV saved to: {summary_file}")
        
        # Save statistics
        stats_file = os.path.join(results_dir, 'statistics.txt')
        with open(stats_file, 'w') as f:
            f.write(self._get_statistics_report())
        
        print(f"Statistics saved to: {stats_file}")
    
    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 70)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 70)
        print(self._get_statistics_report())
    
    def _get_statistics_report(self) -> str:
        """Get detailed statistics report."""
        stats = self.processing_stats
        
        # Calculate averages
        if stats['processed_files'] > 0:
            avg_objects_per_file = stats['total_objects_detected'] / stats['processed_files']
            avg_time_per_file = stats['total_processing_time'] / stats['processed_files']
        else:
            avg_objects_per_file = 0
            avg_time_per_file = 0
        
        # Object class distribution (if classification enabled)
        class_distribution = {}
        if self.enable_classification:
            for result in self.results:
                for obj in result['objects']:
                    class_name = obj.get('class_name', 'Unknown')
                    class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        
        # Processing time distribution
        processing_times = [result['processing_time'] for result in self.results]
        if processing_times:
            min_time = min(processing_times)
            max_time = max(processing_times)
            avg_time = np.mean(processing_times)
        else:
            min_time = max_time = avg_time = 0
        
        # Create report
        lines = [
            f"Folder: {self.folder_path}",
            f"Total files found: {stats['total_files']}",
            f"Successfully processed: {stats['processed_files']}",
            f"Failed files: {len(stats['failed_files'])}",
            f"Total objects detected: {stats['total_objects_detected']}",
            f"",
            f"Processing Times:",
            f"  Total time: {stats['total_processing_time']:.2f} seconds",
            f"  Average per file: {avg_time:.3f} seconds",
            f"  Min/Max per file: {min_time:.3f}s / {max_time:.3f}s",
            f"",
            f"Detection Statistics:",
            f"  Average objects per file: {avg_objects_per_file:.2f}",
            f"  Clustering method: DBSCAN",
            f"  Classification: {'Enabled' if self.enable_classification else 'Disabled'}",
        ]
        
        if class_distribution:
            lines.extend([
                f"",
                f"Object Class Distribution:"
            ])
            for class_name, count in sorted(class_distribution.items()):
                percentage = count / stats['total_objects_detected'] * 100
                lines.append(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        if stats['failed_files']:
            lines.extend([
                f"",
                f"Failed Files:"
            ])
            for failed_file in stats['failed_files']:
                lines.append(f"  {failed_file}")
        
        # Configuration summary
        lines.extend([
            f"",
            f"Configuration:",
            f"  DBSCAN eps: {self.config.get('dbscan_params.eps')}",
            f"  DBSCAN min_points: {self.config.get('dbscan_params.min_points')}",
            f"  Min object height: {self.config.get('object_params.min_object_height')}m",
            f"  Max object height: {self.config.get('object_params.max_object_height')}m"
        ])
        
        if self.enable_classification:
            lines.extend([
                f"  Model: {self.config.get('model_params.model_dir')}",
                f"  Dataset: {self.config.get('model_params.dataset')}",
                f"  Subspace dim: {self.config.get('subspace_params.ndim')}",
                f"  Use GDS: {self.config.get('subspace_params.use_gds')}"
            ])
        
        return "\n".join(lines)
    
    def get_results(self) -> List[Dict[str, Any]]:
        """
        Get processing results.
        
        Returns:
            List of processing results
        """
        return self.results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.processing_stats.copy()
    
    def export_point_clouds(self, output_dir: str, export_format: str = 'ply'):
        """
        Export detected object point clouds to separate files.
        
        Args:
            output_dir: Output directory for exported files
            export_format: Export format ('ply', 'pcd')
        """
        import open3d as o3d
        
        os.makedirs(output_dir, exist_ok=True)
        
        for result in self.results:
            frame_id = result['frame_id']
            
            for obj_id, obj in enumerate(result['objects']):
                if 'points' in obj and len(obj['points']) > 0:
                    # Create Open3D point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(obj['points'])
                    
                    # Generate filename
                    class_name = obj.get('class_name', 'unknown')
                    filename = f"frame_{frame_id:04d}_obj_{obj_id:02d}_{class_name}.{export_format}"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save
                    if export_format == 'ply':
                        o3d.io.write_point_cloud(filepath, pcd)
                    elif export_format == 'pcd':
                        o3d.io.write_point_cloud(filepath, pcd)
                    
        print(f"Object point clouds exported to: {output_dir}")