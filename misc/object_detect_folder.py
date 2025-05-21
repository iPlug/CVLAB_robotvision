import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import pickle
from concurrent.futures import ThreadPoolExecutor
import time


class FrameSequenceProcessor:
    def __init__(self, folder_path, visualize=False, cache_floor_params=True):
        self.folder_path = folder_path
        self.visualize = visualize
        self.cache_floor_params = cache_floor_params

        # Floor parameters (computed once and reused)
        self.floor_plane_model = None
        self.rotation_matrix = None
        self.floor_params_computed = False
        self.floor_level = None  # Cache floor level

        # Object detection parameters
        self.roi_z_threshold = -0.55
        self.plane_distance_threshold = 0.01
        self.plane_ransac_n = 3
        self.plane_num_iterations = 1000
        self.noise_nb_neighbors = 20
        self.noise_std_ratio = 2.0
        self.dbscan_eps = 0.05
        self.dbscan_min_points = 10
        self.min_object_height = 0.05
        self.max_object_height = 1.0

        # Performance optimization parameters
        self.max_points_for_processing = 50000  # Max points before voxel downsampling
        self.voxel_size_initial = 0.005  # Initial voxel size (2cm)
        self.voxel_size_secondary = 0.005  # Secondary voxel size (1.5cm)

        # Cache file for floor parameters
        self.cache_file = os.path.join(folder_path, "floor_params_cache.pkl")

    def get_ply_files(self):
        """Get all PLY files in the folder, sorted by name"""
        pattern = os.path.join(self.folder_path, "*.ply")
        ply_files = glob.glob(pattern)
        return sorted(ply_files)

    def create_3d_text_label(self, text, position, scale=0.005, color=[1, 1, 1]):
        """Helper function to create flat 3D text"""
        try:
            # Create flat text (depth=0 makes it a flat surface)
            text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0)
            text_mesh_legacy = text_mesh.to_legacy()

            # Manually create double-sided by duplicating vertices and faces
            vertices = np.asarray(text_mesh_legacy.vertices)
            triangles = np.asarray(text_mesh_legacy.triangles)

            # Create back faces by flipping triangle orientation
            back_triangles = triangles[:, [0, 2, 1]]  # Flip winding order

            # Combine front and back triangles
            all_triangles = np.vstack([triangles, back_triangles])
            text_mesh_legacy.triangles = o3d.utility.Vector3iVector(all_triangles)

            # Rest of your code...
            bbox = text_mesh_legacy.get_axis_aligned_bounding_box()
            text_center = bbox.get_center()
            text_mesh_legacy.paint_uniform_color(color)

            # Apply transformations...
            scale_matrix = np.array(
                [
                    [scale, 0, 0, 0],
                    [0, scale, 0, 0],
                    [0, 0, scale, 0],
                    [0, 0, 0, 1],
                ]
            )
            text_mesh_legacy.transform(scale_matrix)

            scaled_center = text_center * scale
            translation_matrix = np.array(
                [
                    [1, 0, 0, position[0] - scaled_center[0]],
                    [0, 1, 0, position[1] - scaled_center[1]],
                    [0, 0, 1, position[2] - scaled_center[2]],
                    [0, 0, 0, 1],
                ]
            )
            text_mesh_legacy.transform(translation_matrix)

            return text_mesh_legacy
        except Exception as e:
            print(f"    Warning: Could not create 3D text: {e}")
            return None

    def compute_floor_parameters_optimized(self, pcd):
        """Optimized floor parameter computation using vectorized operations"""
        points = np.asarray(pcd.points)

        # Vectorized ROI filtering
        mask = points[:, 2] >= self.roi_z_threshold
        if not np.any(mask):
            raise ValueError("No points in ROI")

        # Use numpy indexing instead of Open3D selection
        filtered_points = points[mask]

        # Create temporary point cloud only for RANSAC
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # Detect floor using RANSAC
        plane_model, inliers = temp_pcd.segment_plane(
            distance_threshold=self.plane_distance_threshold,
            ransac_n=self.plane_ransac_n,
            num_iterations=self.plane_num_iterations,
        )

        [a, b, c, d] = plane_model
        print(f"Floor plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        # Optimized rotation matrix calculation
        normal_vector = np.array([a, b, c])
        normal_length = np.linalg.norm(normal_vector)
        if normal_length < 1e-6:
            return plane_model, np.identity(3)

        normal_vector /= normal_length
        target_normal = np.array([0, 1, 0])

        rotation_axis = np.cross(normal_vector, target_normal)
        rotation_axis_length = np.linalg.norm(rotation_axis)

        if rotation_axis_length < 1e-6:
            if np.dot(normal_vector, target_normal) < 0:
                R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    np.array([np.pi, 0, 0])
                )
            else:
                R = np.identity(3)
        else:
            rotation_axis /= rotation_axis_length
            cos_angle = np.clip(np.dot(normal_vector, target_normal), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

        return plane_model, R

    def save_floor_parameters(self):
        """Save floor parameters to cache file"""
        if self.cache_floor_params and self.floor_params_computed:
            with open(self.cache_file, "wb") as f:
                pickle.dump(
                    {
                        "plane_model": self.floor_plane_model,
                        "rotation_matrix": self.rotation_matrix,
                        "floor_level": self.floor_level,
                    },
                    f,
                )
            print(f"Floor parameters saved to {self.cache_file}")

    def load_floor_parameters(self):
        """Load floor parameters from cache file"""
        if self.cache_floor_params and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    params = pickle.load(f)
                self.floor_plane_model = params["plane_model"]
                self.rotation_matrix = params["rotation_matrix"]
                self.floor_level = params.get("floor_level", None)
                self.floor_params_computed = True
                print("Floor parameters loaded from cache")
                return True
            except Exception as e:
                print(f"Error loading cached floor parameters: {e}")
                return False
        return False

    def initialize_floor_parameters(self, pcd_files):
        """Initialize floor parameters using only the first frame (optimized)"""
        # Try to load from cache first
        if self.load_floor_parameters():
            # Check if floor_level is missing from cache
            if self.floor_level is None:
                print("Floor level not in cache, computing...")
                # Load first frame to compute floor level
                pcd = o3d.io.read_point_cloud(pcd_files[0])
                points = np.asarray(pcd.points)
                mask = points[:, 2] >= self.roi_z_threshold
                if np.any(mask):
                    filtered_points = points[mask]
                    # Apply rotation to get floor level
                    rotated_points = (self.rotation_matrix @ filtered_points.T).T
                    self.floor_level = np.percentile(rotated_points[:, 1], 10)
                    # Save updated parameters
                    self.save_floor_parameters()
                else:
                    self.floor_level = 0.0
            return

        print("Computing floor parameters...")

        # Use only the first frame for faster initialization
        pcd = o3d.io.read_point_cloud(pcd_files[0])
        self.floor_plane_model, self.rotation_matrix = (
            self.compute_floor_parameters_optimized(pcd)
        )

        # Compute floor level for faster floor removal
        points = np.asarray(pcd.points)
        mask = points[:, 2] >= self.roi_z_threshold
        if np.any(mask):
            filtered_points = points[mask]
            # Apply rotation to get floor level
            rotated_points = (self.rotation_matrix @ filtered_points.T).T
            self.floor_level = np.percentile(rotated_points[:, 1], 10)
        else:
            self.floor_level = 0.0

        self.floor_params_computed = True
        self.save_floor_parameters()

    def process_single_frame_optimized(self, file_path):
        """Optimized single frame processing with voxel downsampling for large point clouds"""
        pcd = o3d.io.read_point_cloud(file_path)

        if len(pcd.points) == 0:
            print(f"  Warning: Empty point cloud in {os.path.basename(file_path)}")
            return [], [], [], pcd

        # **VOXEL DOWNSAMPLING**: Use voxel downsampling instead of random subsampling
        if len(pcd.points) > self.max_points_for_processing:
            print(f"  Large input cloud ({len(pcd.points)} points), voxel downsampling")
            # Calculate voxel size to achieve target point count
            # Start with a reasonable voxel size and adjust
            voxel_size = self.voxel_size_initial  # 2cm voxels as starting point

            # Iteratively adjust voxel size to get close to target
            for _ in range(3):  # Max 3 iterations to find good voxel size
                temp_pcd = pcd.voxel_down_sample(voxel_size)
                if len(temp_pcd.points) <= self.max_points_for_processing:
                    break
                voxel_size *= 1.4  # Increase voxel size to reduce points

            pcd = temp_pcd
            print(
                f"    Downsampled to {len(pcd.points)} points with voxel size {voxel_size:.3f}m"
            )

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        # Vectorized ROI filtering
        roi_mask = points[:, 2] >= self.roi_z_threshold
        if not np.any(roi_mask):
            print(
                f"  Warning: No points above z={self.roi_z_threshold} in {os.path.basename(file_path)}"
            )
            return [], [], [], pcd

        filtered_points = points[roi_mask]
        filtered_colors = colors[roi_mask] if colors is not None else None

        # Apply rotation using vectorized operation
        rotated_points = (self.rotation_matrix @ filtered_points.T).T

        # Optimized floor removal using cached floor level
        floor_height_threshold = 0.01
        non_floor_mask = rotated_points[:, 1] > (
            self.floor_level + floor_height_threshold
        )

        if not np.any(non_floor_mask):
            print(f"  Warning: No points above floor in {os.path.basename(file_path)}")
            return [], [], [], pcd

        non_floor_points = rotated_points[non_floor_mask]
        non_floor_colors = (
            filtered_colors[non_floor_mask] if filtered_colors is not None else None
        )

        # **SECOND VOXEL DOWNSAMPLING**: Additional downsampling after floor removal if still too large
        if len(non_floor_points) > self.max_points_for_processing // 2:
            print(
                f"  Large cloud after floor removal ({len(non_floor_points)} points), additional voxel downsampling"
            )
            # Create temporary point cloud for voxel downsampling
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(non_floor_points)
            if non_floor_colors is not None:
                temp_pcd.colors = o3d.utility.Vector3dVector(non_floor_colors)

            # Use smaller voxel size for this stage
            voxel_size = self.voxel_size_secondary  # 1.5cm voxels
            downsampled_pcd = temp_pcd.voxel_down_sample(voxel_size)

            non_floor_points = np.asarray(downsampled_pcd.points)
            non_floor_colors = (
                np.asarray(downsampled_pcd.colors)
                if downsampled_pcd.has_colors()
                else None
            )
            print(f"    Further downsampled to {len(non_floor_points)} points")

        # Create point cloud only when necessary - skip noise removal for very small clouds
        if len(non_floor_points) < self.noise_nb_neighbors:
            print(
                f"  Warning: Too few points ({len(non_floor_points)}) for noise removal in {os.path.basename(file_path)}"
            )
            final_points = non_floor_points
            final_colors = non_floor_colors
        else:
            # **OPTIMIZED NOISE REMOVAL**: More aggressive optimization for large clouds
            if len(non_floor_points) > 20000:
                # For very large clouds, skip noise removal entirely to save time
                print(
                    f"  Skipping noise removal for large cloud ({len(non_floor_points)} points)"
                )
                final_points = non_floor_points
                final_colors = non_floor_colors
            else:
                # Reduce noise removal parameters further for medium-large point clouds
                effective_neighbors = min(10, max(5, len(non_floor_points) // 2000))

                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(non_floor_points)
                if non_floor_colors is not None:
                    temp_pcd.colors = o3d.utility.Vector3dVector(non_floor_colors)

                # More relaxed noise removal for faster processing
                clean_pcd, clean_indices = temp_pcd.remove_statistical_outlier(
                    nb_neighbors=effective_neighbors,
                    std_ratio=self.noise_std_ratio * 2.0,  # Much more permissive
                )

                if len(clean_pcd.points) == 0:
                    print(
                        f"  Warning: No points left after noise removal in {os.path.basename(file_path)}"
                    )
                    return [], [], [], pcd

                final_points = np.asarray(clean_pcd.points)
                final_colors = (
                    np.asarray(clean_pcd.colors) if clean_pcd.has_colors() else None
                )

        # **OPTIMIZED DBSCAN**: Even more aggressive parameter adjustment
        temp_pcd_cluster = o3d.geometry.PointCloud()
        temp_pcd_cluster.points = o3d.utility.Vector3dVector(final_points)

        # Scale DBSCAN parameters more aggressively based on point cloud size
        if len(final_points) > 15000:
            # For large point clouds, use much larger eps and higher min_points
            adaptive_eps = self.dbscan_eps * 2.5
            adaptive_min_points = max(
                self.dbscan_min_points * 2, len(final_points) // 800
            )
        elif len(final_points) > 5000:
            adaptive_eps = self.dbscan_eps * 1.8
            adaptive_min_points = max(self.dbscan_min_points, len(final_points) // 1000)
        else:
            adaptive_eps = self.dbscan_eps
            adaptive_min_points = self.dbscan_min_points

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            labels = np.array(
                temp_pcd_cluster.cluster_dbscan(
                    eps=adaptive_eps, min_points=adaptive_min_points
                )
            )

        if len(labels) == 0 or labels.max() < 0:
            print(f"  Warning: No clusters found in {os.path.basename(file_path)}")
            return [], [], [], pcd

        # Vectorized cluster filtering with text labels
        valid_clusters = []
        valid_bboxes = []
        text_labels = []  # New list for text labels
        max_label = labels.max()

        # Pre-compute all cluster properties
        for i, cluster_id in enumerate(range(max_label + 1)):
            cluster_mask = labels == cluster_id
            if not np.any(cluster_mask):
                continue

            cluster_points = final_points[cluster_mask]

            # Vectorized bounding box calculation
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)

            # Calculate dimensions
            height = max_bounds[1] - min_bounds[1]
            width = max_bounds[0] - min_bounds[0]
            depth = max_bounds[2] - min_bounds[2]

            # Filter by dimensions
            if (
                self.min_object_height <= height <= self.max_object_height
                and width >= 0.01
                and depth >= 0.01
            ):

                # Create cluster point cloud only for valid objects
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                if final_colors is not None:
                    cluster_colors = final_colors[cluster_mask]
                    cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

                # Create bounding box
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bounds, max_bounds)
                bbox.color = (0, 1, 0)

                # Create text label above the bounding box
                # Position text above the center of the top face of the bbox
                text_position = [
                    (min_bounds[0] + max_bounds[0]) / 2,  # Center X
                    max_bounds[1] + 0.1,  # 10cm above the top
                    (min_bounds[2] + max_bounds[2]) / 2,  # Center Z
                ]

                # Create placeholder text with object ID and dimensions
                label_text = f"Obj_{len(valid_clusters):02d}"
                # Uncomment next line to include dimensions in the label
                # label_text = f"Obj_{len(valid_clusters):02d}\n{height:.2f}m"

                text_label = self.create_3d_text_label(
                    label_text,
                    text_position,
                    scale=0.003,  # Much smaller scale (3mm text height)
                    color=[1, 1, 0],  # Yellow text
                )

                valid_clusters.append(cluster_pcd)
                valid_bboxes.append(bbox)
                if text_label is not None:
                    text_labels.append(text_label)

        return valid_clusters, valid_bboxes, text_labels, pcd

    def process_all_frames(self):
        """Process all PLY files in the folder with optimizations"""
        # Get all PLY files
        ply_files = self.get_ply_files()
        if not ply_files:
            print(f"No PLY files found in {self.folder_path}")
            return

        print(f"Found {len(ply_files)} PLY files")

        # Initialize floor parameters (now using only first frame)
        self.initialize_floor_parameters(ply_files)

        # Create visualization window if requested
        vis = None
        vis_pcd = None
        bboxes_geom = []
        text_geom = []  # Track text geometries for removal

        if self.visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name="LiDAR Object Detection - Frame Sequence",
                width=1280,
                height=720,
            )

            # Create point cloud object for visualization
            vis_pcd = None

            # Set up camera view
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.1, 0.1, 0.1])
            opt.point_size = 2.0

        # Process each frame with timing
        results = []
        start_time = time.time()

        for i, file_path in enumerate(ply_files):
            frame_start = time.time()
            print(
                f"Processing frame {i+1}/{len(ply_files)}: {os.path.basename(file_path)}"
            )

            clusters, bboxes, text_labels, original_pcd = (
                self.process_single_frame_optimized(file_path)
            )

            frame_time = time.time() - frame_start
            print(f"  Found {len(clusters)} objects in {frame_time:.3f}s")

            results.append(
                {
                    "file_path": file_path,
                    "clusters": clusters,
                    "bboxes": bboxes,
                    "text_labels": text_labels,
                    "original_pcd": original_pcd,
                    "num_objects": len(clusters),
                    "processing_time": frame_time,
                }
            )

            # Update visualization if requested (with text labels)
            if self.visualize and vis is not None:
                # Remove old bounding boxes and text labels
                for bbox in bboxes_geom:
                    vis.remove_geometry(bbox, reset_bounding_box=False)
                for text in text_geom:
                    vis.remove_geometry(text, reset_bounding_box=False)
                bboxes_geom.clear()
                text_geom.clear()

                # Prepare filtered point cloud for visualization
                points = np.asarray(original_pcd.points)
                colors = (
                    np.asarray(original_pcd.colors)
                    if original_pcd.has_colors()
                    else None
                )

                # Apply ROI filter
                mask = points[:, 2] >= self.roi_z_threshold
                filtered_points = points[mask]
                filtered_colors = colors[mask] if colors is not None else None

                # Apply rotation to match processing
                if len(filtered_points) > 0:
                    filtered_points_rotated = (
                        self.rotation_matrix @ filtered_points.T
                    ).T

                    if vis_pcd is None:
                        # Create a new point cloud for visualization
                        vis_pcd = o3d.geometry.PointCloud()
                        vis_pcd.points = o3d.utility.Vector3dVector(
                            filtered_points_rotated
                        )
                        vis_pcd.colors = o3d.utility.Vector3dVector(
                            filtered_colors
                            if filtered_colors is not None
                            else np.tile(
                                [0.3, 0.6, 1.0], (len(filtered_points_rotated), 1)
                            )
                        )
                        vis.add_geometry(vis_pcd)
                    else:
                        # Update existing point cloud
                        vis_pcd.points = o3d.utility.Vector3dVector(
                            filtered_points_rotated
                        )
                        if filtered_colors is not None:
                            vis_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
                        else:
                            vis_pcd.colors = o3d.utility.Vector3dVector(
                                np.tile(
                                    [0.3, 0.6, 1.0], (len(filtered_points_rotated), 1)
                                )
                            )

                    # Update geometry
                    vis.update_geometry(vis_pcd)

                    # Add bounding boxes
                    for bbox in bboxes:
                        vis.add_geometry(bbox, reset_bounding_box=False)
                        bboxes_geom.append(bbox)

                    # Add text labels
                    for text_label in text_labels:
                        vis.add_geometry(text_label, reset_bounding_box=False)
                        text_geom.append(text_label)

                # Update view
                vis.poll_events()
                vis.update_renderer()

                # Small delay to see each frame
                time.sleep(0.05)  # 50ms delay

        # Close visualization window
        if vis is not None:
            vis.destroy_window()

        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.3f}s")
        print(f"Average time per frame: {total_time/len(ply_files):.3f}s")

        return results

    def save_results_summary(self, results):
        """Save a summary of detection results with timing information"""
        summary_file = os.path.join(self.folder_path, "detection_summary.txt")
        with open(summary_file, "w") as f:
            f.write("Object Detection Summary\n")
            f.write("=" * 60 + "\n\n")

            total_objects = 0
            total_time = 0
            for i, result in enumerate(results):
                frame_name = os.path.basename(result["file_path"])
                num_objects = result["num_objects"]
                frame_time = result.get("processing_time", 0)
                total_objects += num_objects
                total_time += frame_time
                f.write(
                    f"Frame {i+1:4d}: {frame_name:40s} - {num_objects:3d} objects - {frame_time:.3f}s\n"
                )

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Total frames processed: {len(results)}\n")
            f.write(f"Total processing time: {total_time:.3f}s\n")
            f.write(f"Average time per frame: {total_time/len(results):.3f}s\n")
            f.write(f"Average objects per frame: {total_objects / len(results):.2f}\n")

        print(f"Summary saved to {summary_file}")


# Usage example
if __name__ == "__main__":
    # Set your folder path here
    folder_path = r"F:\LiDAR Videos\objects_detection\ply"

    # Create processor instance
    processor = FrameSequenceProcessor(
        folder_path=folder_path,
        visualize=True,  # Set to True if you want to visualize each frame
        cache_floor_params=True,
    )

    # Adjust parameters if needed
    processor.roi_z_threshold = -0.55
    processor.min_object_height = 0.05
    processor.max_object_height = 1.0
    processor.dbscan_eps = 0.05
    processor.dbscan_min_points = 10

    # Process all frames
    results = processor.process_all_frames()

    # Save summary
    processor.save_results_summary(results)

    # Print statistics
    if results:
        num_objects_per_frame = [r["num_objects"] for r in results]
        processing_times = [r.get("processing_time", 0) for r in results]
        print(f"\nDetection Statistics:")
        print(f"Total frames: {len(results)}")
        print(f"Min objects per frame: {min(num_objects_per_frame)}")
        print(f"Max objects per frame: {max(num_objects_per_frame)}")
        print(f"Average objects per frame: {np.mean(num_objects_per_frame):.2f}")
        print(f"Average processing time: {np.mean(processing_times):.3f}s")
        print(
            f"Min/Max processing time: {min(processing_times):.3f}s / {max(processing_times):.3f}s"
        )
