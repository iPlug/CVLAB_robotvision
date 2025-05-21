import open3d as o3d
import numpy as np
import os
import glob
import pickle
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import the shape subspace module (you'll need to ensure this is available)
import utils.shapesubspace as shs


# Define the GeometricOrderingNet model (from second code)
class GeometricOrderingNet(nn.Module):
    """Network that learns to produce consistent ordering for different samplings of the same shape"""

    def __init__(self, num_points=1024):
        super(GeometricOrderingNet, self).__init__()
        self.num_points = num_points

        # Feature extraction with strong geometric awareness
        self.point_features = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        # Global feature extraction
        self.global_features = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        # Canonical space generator
        self.canonical_generator = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_points * 3),
        )

    def forward(self, x):
        batch_size, _, _ = x.size()
        x = x.transpose(1, 2)  # B×3×N
        point_features = self.point_features(x)
        global_feature = self.global_features(point_features)
        global_feature = global_feature.view(batch_size, -1)
        canonical_points = self.canonical_generator(global_feature)
        canonical_points = canonical_points.view(batch_size, self.num_points, 3)
        return canonical_points


class FrameSequenceProcessor:
    def __init__(
        self,
        folder_path,
        visualize=False,
        cache_floor_params=True,
        enable_recognition=True,
        model_path=None,
        num_canonical_points=1024,
    ):
        self.folder_path = folder_path
        self.visualize = visualize
        self.cache_floor_params = cache_floor_params
        self.enable_recognition = enable_recognition

        # Floor parameters (computed once and reused)
        self.floor_plane_model = None
        self.rotation_matrix = None
        self.floor_params_computed = False
        self.floor_level = None

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
        self.max_points_for_processing = 50000
        self.voxel_size_initial = 0.005
        self.voxel_size_secondary = 0.005

        # Cache file for floor parameters
        self.cache_file = os.path.join(folder_path, "floor_params_cache.pkl")

        # Recognition system initialization
        self.recognition_model = None
        self.class_subspaces = []
        self.class_names = ["bottle", "box", "can", "hand", "unknown"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_canonical_points = num_canonical_points

        if self.enable_recognition:
            self._load_recognition_model(model_path)
            self._load_class_subspaces()

    def _load_recognition_model(self, model_path):
        """Load the GeometricOrderingNet model for canonical transformation"""
        if model_path and os.path.exists(model_path):
            print("Loading GeometricOrderingNet model...")
            self.recognition_model = GeometricOrderingNet(
                num_points=self.num_canonical_points
            )
            try:
                self.recognition_model.load_state_dict(
                    torch.load(model_path, map_location=self.device)
                )
                self.recognition_model.to(self.device)
                self.recognition_model.eval()
                print("GeometricOrderingNet loaded successfully!")
            except Exception as e:
                print(f"Error loading recognition model: {e}")
                self.enable_recognition = False
        else:
            print(
                "Recognition model path not provided or doesn't exist. Recognition disabled."
            )
            self.enable_recognition = False

    def _load_class_subspaces(self):
        """Load pre-computed class subspaces from file or create empty lists"""
        subspace_file = os.path.join(self.folder_path, "class_subspaces.pkl")
        if os.path.exists(subspace_file):
            try:
                with open(subspace_file, "rb") as f:
                    self.class_subspaces = pickle.load(f)
                print(
                    f"Loaded pre-computed class subspaces with {len(self.class_subspaces)} classes"
                )
                # Print number of subspaces per class
                for i, class_subspaces in enumerate(self.class_subspaces):
                    if i < len(self.class_names) - 1:  # Exclude 'unknown'
                        print(
                            f"  {self.class_names[i]}: {len(class_subspaces)} subspaces"
                        )
                return
            except Exception as e:
                print(f"Error loading class subspaces: {e}")

        print(
            "No pre-computed class subspaces found. Please provide training data to compute them."
        )
        self._create_empty_class_subspaces()

    def _create_empty_class_subspaces(self):
        """Create empty lists for class subspaces (except 'unknown')"""
        self.class_subspaces = [[] for _ in range(len(self.class_names) - 1)]

    def compute_class_subspaces_from_training_data(
        self, training_data_path, batch_size=16
    ):
        """
        Compute class subspaces from training data

        Args:
            training_data_path: Path to training data directory
            batch_size: Batch size for processing training data
        """
        print("Computing class subspaces from training data...")

        # Import the data loader (you may need to adjust this import)
        try:
            from SimpleDataLoader import SimpleDataLoader
        except ImportError:
            print(
                "SimpleDataLoader not found. Please ensure it's available in your Python path."
            )
            return False

        # Load training dataset
        try:
            train_dataset = SimpleDataLoader(root=training_data_path, split="train")
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=False
            )
            print(f"Loaded training dataset with {len(train_dataset)} samples")
        except Exception as e:
            print(f"Error loading training data: {e}")
            return False

        # Initialize class subspaces
        self.class_subspaces = [[] for _ in range(len(self.class_names) - 1)]

        # Process training data
        for points, labels in tqdm(train_loader, desc="Computing class subspaces"):
            # Process each point cloud in the batch
            for i in range(points.shape[0]):
                point_set = points[i].numpy()
                label = labels[i].item()

                # Only use xyz coordinates if normals are present
                if point_set.shape[1] > 3:
                    point_set = point_set[:, 0:3]

                # Transform to canonical representation first
                canonical_points = self.transform_to_canonical(point_set)
                if canonical_points is None:
                    continue

                # Compute shape subspace from canonical points
                try:
                    sp_pcd = shs.get_shapesubspace(canonical_points)
                    self.class_subspaces[label].append(sp_pcd)
                except Exception as e:
                    print(f"Error computing subspace for label {label}: {e}")
                    continue

        # Print statistics
        print("Class subspace computation completed:")
        total_subspaces = 0
        for i, class_subspaces in enumerate(self.class_subspaces):
            print(f"  {self.class_names[i]}: {len(class_subspaces)} subspaces")
            total_subspaces += len(class_subspaces)

        if total_subspaces == 0:
            print(
                "Warning: No subspaces were computed. Check your training data and model."
            )
            return False

        # Save computed subspaces
        self._save_class_subspaces()
        return True

    def _save_class_subspaces(self):
        """Save computed class subspaces to file"""
        subspace_file = os.path.join(self.folder_path, "class_subspaces.pkl")
        try:
            with open(subspace_file, "wb") as f:
                pickle.dump(self.class_subspaces, f)
            print(f"Class subspaces saved to {subspace_file}")
        except Exception as e:
            print(f"Error saving class subspaces: {e}")

    def normalize_point_cloud(self, points):
        """Center and normalize a point cloud to unit sphere"""
        # Center
        centroid = np.mean(points, axis=0)
        points = points - centroid

        # Scale to unit sphere
        max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if max_dist > 0:
            points = points / max_dist

        return points

    def transform_to_canonical(self, point_cloud):
        """Transform a point cloud to its canonical representation"""
        if self.recognition_model is None:
            return None

        # Normalize the point cloud first
        normalized_points = self.normalize_point_cloud(point_cloud.copy())

        # Resample to fixed number of points for the network
        if len(normalized_points) > self.num_canonical_points:
            # Random subsample
            indices = np.random.choice(
                len(normalized_points), self.num_canonical_points, replace=False
            )
            normalized_points = normalized_points[indices]
        elif len(normalized_points) < self.num_canonical_points:
            # Upsample by repeating points
            repeats = self.num_canonical_points // len(normalized_points) + 1
            repeated_points = np.tile(normalized_points, (repeats, 1))
            normalized_points = repeated_points[: self.num_canonical_points]

        # Convert to tensor and add batch dimension
        point_tensor = (
            torch.tensor(normalized_points, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        # Get canonical representation
        with torch.no_grad():
            canonical_points = self.recognition_model(point_tensor)
            canonical_points = canonical_points.squeeze(0).cpu().numpy()

        return canonical_points

    def classify_object(self, point_cloud):
        """Classify an object using subspace analysis"""
        if not self.enable_recognition or self.recognition_model is None:
            return "unknown"

        try:
            # Transform to canonical representation
            canonical_points = self.transform_to_canonical(point_cloud)
            if canonical_points is None:
                return "unknown"

            # Compute subspace from canonical points
            input_subspace = shs.get_shapesubspace(canonical_points)

            # Classify using subspace comparison
            best_cosine = 0
            best_class = len(self.class_names) - 1  # Default to "unknown"

            for i, class_subspaces in enumerate(self.class_subspaces):
                if len(class_subspaces) == 0:
                    continue

                class_sim = []
                for class_subspace in class_subspaces:
                    sim = shs.compute_similarity_between_subspaces(
                        class_subspace, input_subspace
                    )
                    class_sim.append(sim)

                class_sim = np.array(class_sim)
                if len(class_sim) > 0:
                    max_cosine = np.max(class_sim)
                    if max_cosine > best_cosine:
                        best_cosine = max_cosine
                        best_class = i

            # Only return recognized class if similarity is above threshold
            if best_cosine > 0.5:  # Threshold for recognition confidence
                return self.class_names[best_class]
            else:
                return "unknown"

        except Exception as e:
            print(f"Error in object classification: {e}")
            return "unknown"

    def get_ply_files(self):
        """Get all PLY files in the folder, sorted by name"""
        pattern = os.path.join(self.folder_path, "*.ply")
        ply_files = glob.glob(pattern)
        return sorted(ply_files)

    def create_3d_text_label(self, text, position, scale=0.005, color=[1, 1, 1]):
        """Helper function to create flat 3D text"""
        try:
            text_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0)
            text_mesh_legacy = text_mesh.to_legacy()

            vertices = np.asarray(text_mesh_legacy.vertices)
            triangles = np.asarray(text_mesh_legacy.triangles)
            back_triangles = triangles[:, [0, 2, 1]]
            all_triangles = np.vstack([triangles, back_triangles])
            text_mesh_legacy.triangles = o3d.utility.Vector3iVector(all_triangles)

            bbox = text_mesh_legacy.get_axis_aligned_bounding_box()
            text_center = bbox.get_center()
            text_mesh_legacy.paint_uniform_color(color)

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

        mask = points[:, 2] >= self.roi_z_threshold
        if not np.any(mask):
            raise ValueError("No points in ROI")

        filtered_points = points[mask]
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        plane_model, inliers = temp_pcd.segment_plane(
            distance_threshold=self.plane_distance_threshold,
            ransac_n=self.plane_ransac_n,
            num_iterations=self.plane_num_iterations,
        )

        [a, b, c, d] = plane_model
        print(f"Floor plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

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
        """Initialize floor parameters using only the first frame"""
        if self.load_floor_parameters():
            if self.floor_level is None:
                print("Floor level not in cache, computing...")
                pcd = o3d.io.read_point_cloud(pcd_files[0])
                points = np.asarray(pcd.points)
                mask = points[:, 2] >= self.roi_z_threshold
                if np.any(mask):
                    filtered_points = points[mask]
                    rotated_points = (self.rotation_matrix @ filtered_points.T).T
                    self.floor_level = np.percentile(rotated_points[:, 1], 10)
                    self.save_floor_parameters()
                else:
                    self.floor_level = 0.0
            return

        print("Computing floor parameters...")
        pcd = o3d.io.read_point_cloud(pcd_files[0])
        self.floor_plane_model, self.rotation_matrix = (
            self.compute_floor_parameters_optimized(pcd)
        )

        points = np.asarray(pcd.points)
        mask = points[:, 2] >= self.roi_z_threshold
        if np.any(mask):
            filtered_points = points[mask]
            rotated_points = (self.rotation_matrix @ filtered_points.T).T
            self.floor_level = np.percentile(rotated_points[:, 1], 10)
        else:
            self.floor_level = 0.0

        self.floor_params_computed = True
        self.save_floor_parameters()

    def process_single_frame_optimized(self, file_path):
        """Optimized single frame processing with voxel downsampling and recognition"""
        pcd = o3d.io.read_point_cloud(file_path)

        if len(pcd.points) == 0:
            print(f"  Warning: Empty point cloud in {os.path.basename(file_path)}")
            return [], [], [], pcd

        # Voxel downsampling for large point clouds
        if len(pcd.points) > self.max_points_for_processing:
            print(f"  Large input cloud ({len(pcd.points)} points), voxel downsampling")
            voxel_size = self.voxel_size_initial

            for _ in range(3):
                temp_pcd = pcd.voxel_down_sample(voxel_size)
                if len(temp_pcd.points) <= self.max_points_for_processing:
                    break
                voxel_size *= 1.4

            pcd = temp_pcd
            print(
                f"    Downsampled to {len(pcd.points)} points with voxel size {voxel_size:.3f}m"
            )

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        # ROI filtering
        roi_mask = points[:, 2] >= self.roi_z_threshold
        if not np.any(roi_mask):
            print(
                f"  Warning: No points above z={self.roi_z_threshold} in {os.path.basename(file_path)}"
            )
            return [], [], [], pcd

        filtered_points = points[roi_mask]
        filtered_colors = colors[roi_mask] if colors is not None else None

        # Apply rotation
        rotated_points = (self.rotation_matrix @ filtered_points.T).T

        # Floor removal
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

        # Second voxel downsampling if needed
        if len(non_floor_points) > self.max_points_for_processing // 2:
            print(
                f"  Large cloud after floor removal ({len(non_floor_points)} points), additional voxel downsampling"
            )
            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(non_floor_points)
            if non_floor_colors is not None:
                temp_pcd.colors = o3d.utility.Vector3dVector(non_floor_colors)

            voxel_size = self.voxel_size_secondary
            downsampled_pcd = temp_pcd.voxel_down_sample(voxel_size)

            non_floor_points = np.asarray(downsampled_pcd.points)
            non_floor_colors = (
                np.asarray(downsampled_pcd.colors)
                if downsampled_pcd.has_colors()
                else None
            )
            print(f"    Further downsampled to {len(non_floor_points)} points")

        # Noise removal (optimized for large clouds)
        if len(non_floor_points) < self.noise_nb_neighbors:
            print(
                f"  Warning: Too few points ({len(non_floor_points)}) for noise removal"
            )
            final_points = non_floor_points
            final_colors = non_floor_colors
        else:
            if len(non_floor_points) > 20000:
                print(
                    f"  Skipping noise removal for large cloud ({len(non_floor_points)} points)"
                )
                final_points = non_floor_points
                final_colors = non_floor_colors
            else:
                effective_neighbors = min(10, max(5, len(non_floor_points) // 2000))

                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(non_floor_points)
                if non_floor_colors is not None:
                    temp_pcd.colors = o3d.utility.Vector3dVector(non_floor_colors)

                clean_pcd, clean_indices = temp_pcd.remove_statistical_outlier(
                    nb_neighbors=effective_neighbors,
                    std_ratio=self.noise_std_ratio * 2.0,
                )

                if len(clean_pcd.points) == 0:
                    print(f"  Warning: No points left after noise removal")
                    return [], [], [], pcd

                final_points = np.asarray(clean_pcd.points)
                final_colors = (
                    np.asarray(clean_pcd.colors) if clean_pcd.has_colors() else None
                )

        # DBSCAN clustering
        temp_pcd_cluster = o3d.geometry.PointCloud()
        temp_pcd_cluster.points = o3d.utility.Vector3dVector(final_points)

        # Adaptive DBSCAN parameters
        if len(final_points) > 15000:
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

        # Process clusters with recognition
        valid_clusters = []
        valid_bboxes = []
        text_labels = []
        max_label = labels.max()

        for cluster_id in range(max_label + 1):
            cluster_mask = labels == cluster_id
            if not np.any(cluster_mask):
                continue

            cluster_points = final_points[cluster_mask]

            # Calculate dimensions
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)
            height = max_bounds[1] - min_bounds[1]
            width = max_bounds[0] - min_bounds[0]
            depth = max_bounds[2] - min_bounds[2]

            # Filter by dimensions
            if (
                self.min_object_height <= height <= self.max_object_height
                and width >= 0.01
                and depth >= 0.01
            ):

                # Create cluster point cloud
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                if final_colors is not None:
                    cluster_colors = final_colors[cluster_mask]
                    cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

                # Classify the object
                if self.enable_recognition:
                    object_class = self.classify_object(cluster_points)
                else:
                    object_class = "unknown"

                # Create bounding box
                bbox = o3d.geometry.AxisAlignedBoundingBox(min_bounds, max_bounds)
                bbox.color = (0, 1, 0)

                # Create text label with classification result
                text_position = [
                    (min_bounds[0] + max_bounds[0]) / 2,
                    max_bounds[1] + 0.1,
                    (min_bounds[2] + max_bounds[2]) / 2,
                ]

                # Create label with class name and object ID
                label_text = f"{object_class}_{len(valid_clusters):02d}"

                text_label = self.create_3d_text_label(
                    label_text,
                    text_position,
                    scale=0.003,
                    color=[1, 1, 0],
                )

                valid_clusters.append(cluster_pcd)
                valid_bboxes.append(bbox)
                if text_label is not None:
                    text_labels.append(text_label)

        return valid_clusters, valid_bboxes, text_labels, pcd

    def process_all_frames(self):
        """Process all PLY files in the folder with optimizations and recognition"""
        ply_files = self.get_ply_files()
        if not ply_files:
            print(f"No PLY files found in {self.folder_path}")
            return

        print(f"Found {len(ply_files)} PLY files")

        # Initialize floor parameters
        self.initialize_floor_parameters(ply_files)

        # Create visualization window if requested
        vis = None
        vis_pcd = None
        bboxes_geom = []
        text_geom = []

        if self.visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name="LiDAR Object Detection with Recognition",
                width=1280,
                height=720,
            )

            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.1, 0.1, 0.1])
            opt.point_size = 2.0

        # Process each frame
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

            # Update visualization
            if self.visualize and vis is not None:
                # Remove old geometries
                for bbox in bboxes_geom:
                    vis.remove_geometry(bbox, reset_bounding_box=False)
                for text in text_geom:
                    vis.remove_geometry(text, reset_bounding_box=False)
                bboxes_geom.clear()
                text_geom.clear()

                # Prepare point cloud for visualization
                points = np.asarray(original_pcd.points)
                colors = (
                    np.asarray(original_pcd.colors)
                    if original_pcd.has_colors()
                    else None
                )

                # Apply filters
                mask = points[:, 2] >= self.roi_z_threshold
                filtered_points = points[mask]
                filtered_colors = colors[mask] if colors is not None else None

                if len(filtered_points) > 0:
                    filtered_points_rotated = (
                        self.rotation_matrix @ filtered_points.T
                    ).T

                    if vis_pcd is None:
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

                    vis.update_geometry(vis_pcd)

                    # Add bounding boxes and text labels
                    for bbox in bboxes:
                        vis.add_geometry(bbox, reset_bounding_box=False)
                        bboxes_geom.append(bbox)

                    for text_label in text_labels:
                        vis.add_geometry(text_label, reset_bounding_box=False)
                        text_geom.append(text_label)

                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.05)

        # Close visualization
        if vis is not None:
            vis.destroy_window()

        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.3f}s")
        print(f"Average time per frame: {total_time/len(ply_files):.3f}s")

        return results

    def save_results_summary(self, results):
        """Save a summary of detection results with recognition information"""
        summary_file = os.path.join(
            self.folder_path, "detection_recognition_summary.txt"
        )

        # Collect classification statistics
        class_counts = {name: 0 for name in self.class_names}

        with open(summary_file, "w") as f:
            f.write("Object Detection and Recognition Summary\n")
            f.write("=" * 60 + "\n\n")

            total_objects = 0
            total_time = 0

            for i, result in enumerate(results):
                frame_name = os.path.basename(result["file_path"])
                num_objects = result["num_objects"]
                frame_time = result.get("processing_time", 0)
                total_objects += num_objects
                total_time += frame_time

                # Count object classes in this frame
                frame_classes = []
                for text_label in result.get("text_labels", []):
                    # Extract class name from label text
                    if hasattr(text_label, "triangles"):
                        # This is a bit hacky since we can't directly get text from mesh
                        # You might want to store classification results separately
                        pass

                f.write(
                    f"Frame {i+1:4d}: {frame_name:40s} - {num_objects:3d} objects - {frame_time:.3f}s\n"
                )

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Total frames processed: {len(results)}\n")
            f.write(f"Total processing time: {total_time:.3f}s\n")
            f.write(f"Average time per frame: {total_time/len(results):.3f}s\n")
            f.write(f"Average objects per frame: {total_objects / len(results):.2f}\n")
            f.write(f"Recognition enabled: {self.enable_recognition}\n")

        print(f"Summary saved to {summary_file}")


# Usage example
if __name__ == "__main__":
    # Set your folder path here
    folder_path = r"F:\LiDAR Videos\objects_detection\ply"

    # Path to the trained GeometricOrderingNet model
    model_path = "log/geometric_ordering_net.pth"  # Update this path

    # Path to training data (required for computing class subspaces)
    training_data_path = r"D:\LAB\DATASETS\robot_simple"  # Update this path

    # Create processor instance with recognition enabled
    processor = FrameSequenceProcessor(
        folder_path=folder_path,
        visualize=True,
        cache_floor_params=True,
        enable_recognition=True,  # Enable recognition
        model_path=model_path,  # Path to trained model
        num_canonical_points=1024,  # Number of points for canonical transformation
    )

    # IMPORTANT: Compute class subspaces from training data if not already available
    # This step is crucial for classification to work!
    if processor.enable_recognition:
        # Check if class subspaces are empty and need to be computed
        if all(len(subspaces) == 0 for subspaces in processor.class_subspaces):
            print("Class subspaces not found. Computing from training data...")
            success = processor.compute_class_subspaces_from_training_data(
                training_data_path=training_data_path, batch_size=16
            )
            if not success:
                print("Failed to compute class subspaces. Disabling recognition.")
                processor.enable_recognition = False

    # Adjust detection parameters if needed
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
        print(f"Recognition enabled: {processor.enable_recognition}")

        # Print class subspace statistics if recognition is enabled
        if processor.enable_recognition:
            print(f"\nRecognition Statistics:")
            for i, class_subspaces in enumerate(processor.class_subspaces):
                if i < len(processor.class_names) - 1:
                    print(
                        f"  {processor.class_names[i]}: {len(class_subspaces)} reference subspaces"
                    )
