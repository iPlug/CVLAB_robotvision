import open3d as o3d
import numpy as np
import os
import glob
import pickle
import time
import torch
import sys
from PIL import Image
from SimpleDataLoader import farthest_point_sample

# Import PointNet models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

# Import PointNet models
from models import pointnet2_cls_ssg, pointnet_cls

# Import subspace utilities
from utils.subspace_utils import (
    get_subspace, 
    build_class_subspaces, 
    subspace_classify,
    create_generalized_difference_subspace, 
    project_subspaces,
    sm_similarity,
    sm_gds_similarity
)

class LiDARObjectRecognizer:
    """Object recognition system using PointNet features + subspace classification with GDS support"""

    def __init__(self, model_dir="pointnet_cls", dataset="simple", ndim=50, 
                use_gds=True, n_components_to_remove=1, checkpoint_path=None, use_uniform_sample=False):
        self.model_dir = model_dir
        self.dataset = dataset
        self.ndim = ndim
        self.checkpoint_path = checkpoint_path
        
        # GDS parameters
        self.use_gds = use_gds
        self.n_components_to_remove = n_components_to_remove
        
        # Model parameters
        self.V = None  # Original subspace bases
        self.G = None  # GDS basis
        self.U = None  # Projected subspaces
        self.class_names = None
        self.class_indices = None
        self.feature_dim = None
        self.loaded = False
        self.use_uniform_sample = use_uniform_sample
        
        # PointNet model for feature extraction
        self.pointnet_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_point = 1024  # Standard PointNet input size


    def load_pointnet_model(self):
        """Load the trained PointNet model for feature extraction"""
        try:
            # Load model based on model_dir
            if self.model_dir == "pointnet2_ssg_wo_normals":
                self.pointnet_model = pointnet2_cls_ssg.get_model(
                    num_class=40, normal_channel=False
                )
                default_checkpoint = "log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
            else:  # default to pointnet
                self.pointnet_model = pointnet_cls.get_model(k=40, normal_channel=False)
                default_checkpoint = "log/classification/pointnet_cls/checkpoints/best_model.pth"

            # Use provided checkpoint path or default
            checkpoint_path = self.checkpoint_path or default_checkpoint
            
            print(f"Loading PointNet checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                self.pointnet_model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.pointnet_model.load_state_dict(checkpoint)

            # Move model to device and set to eval mode
            self.pointnet_model = self.pointnet_model.to(self.device)
            self.pointnet_model.eval()
            
            print(f"PointNet model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading PointNet model: {e}")
            return False

    def load_model(self, features_dir="features"):
        """Load the trained subspace model and PointNet with optional GDS creation"""
        try:
            # First load PointNet model
            if not self.load_pointnet_model():
                return False
            
            if self.use_uniform_sample:
                add_name = "_fps_"
            else:
                add_name = "_"

            # Load trained features
            f_dir = os.path.join(features_dir, self.model_dir)
            train_features = np.load(
                os.path.join(f_dir, f"{self.dataset}{add_name}train_features_full.npy"),
                allow_pickle=True,
            ).item()

            # Build class subspaces using subspace_utils
            print(f"Building class subspaces from {self.dataset}{add_name}train_features_full.npy")
            self.V, self.class_indices = build_class_subspaces(train_features, self.ndim)
            
            # Get feature dimension from V
            self.feature_dim = self.V.shape[0]
            
            # Create GDS if requested
            if self.use_gds:
                print(f"Creating Generalized Difference Subspace (GDS)...")
                self.G, evals = create_generalized_difference_subspace(
                    self.V, n_components_to_remove=self.n_components_to_remove
                )
                
                # Project subspaces onto GDS
                print(f"Projecting subspaces onto GDS...")
                self.U = project_subspaces(self.V, self.G)
                
                print(f"GDS created with shape {self.G.shape}, projected subspaces shape {self.U.shape}")
            
            # Load or create class names mapping
            self.class_names = self._load_class_names()

            self.loaded = True
            print(f"Model loaded successfully with {len(self.class_indices)} classes")
            print(f"Using subspace dimension: {self.ndim}")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            self.loaded = False
            return False
        
    def _load_class_names(self):
        """Load class names mapping. Customize this based on your dataset"""
        # Default to generic class names
        class_names = {
            0: "bottle",
            1: "box",
            2: "can",
            3: "hand",
        }  # Add more as needed

        return class_names

    def preprocess_pointcloud_for_pointnet(self, point_cloud):
        """Preprocess point cloud for PointNet input"""
        points = np.asarray(point_cloud.points)
        
        # If point cloud has too few points, duplicate some points
        if len(points) < 3:
            print(f"Warning: Point cloud has only {len(points)} points, skipping...")
            return None
            
        # Resample to fixed number of points
        if len(points) >= self.num_point:
            # Randomly sample num_point points
            indices = np.random.choice(len(points), self.num_point, replace=False)
            sampled_points = points[indices]
        else:
            # Upsample by random sampling with replacement
            indices = np.random.choice(len(points), self.num_point, replace=True)
            sampled_points = points[indices]
        
        # Normalize point cloud to unit sphere (common PointNet preprocessing)
        # Center the points
        centroid = np.mean(sampled_points, axis=0)
        sampled_points = sampled_points - centroid
        
        # Scale to unit sphere
        furthest_distance = np.max(np.sqrt(np.sum(sampled_points ** 2, axis=1)))
        if furthest_distance > 0:
            sampled_points = sampled_points / furthest_distance
        
        # Convert to torch tensor and reshape for PointNet [1, 3, N]
        points_tensor = torch.from_numpy(sampled_points.T).float().unsqueeze(0)  # [1, 3, N]
        
        return points_tensor

    def extract_features(self, point_cloud):
        """Extract features from a point cloud cluster using PointNet"""
        if self.pointnet_model is None:
            print("PointNet model not loaded!")
            return None
            
        # Preprocess point cloud for PointNet
        points_tensor = self.preprocess_pointcloud_for_pointnet(point_cloud)
        if points_tensor is None:
            return None
            
        # Move to device
        points_tensor = points_tensor.to(self.device)
        
        # Extract features using PointNet
        with torch.no_grad():
            try:
                features = self.pointnet_model.extract_features(points_tensor, feature_level="global")["global_features"]
                # Convert to numpy and reshape to match the format expected by subspace_classify
                features_np = features.cpu().numpy()
                return features_np
            except Exception as e:
                print(f"Error during PointNet inference: {e}")
                return None

    def classify_object(self, point_cloud):
        """Classify a single object point cloud using subspace or GDS method"""
        if not self.loaded:
            return None, 0.0

        # Extract features using PointNet
        features = self.extract_features(point_cloud)
        if features is None:
            return None, 0.0

        # Ensure features are the right shape (N, feature_dim)
        if features.ndim == 1:
            features = features.reshape(1, -1)  # Add batch dimension if missing
        
        # Check feature dimensions for safety
        if features.shape[1] != self.feature_dim:
            print(f"Warning: Feature dimension mismatch: {features.shape[1]} vs expected {self.feature_dim}")
            if features.shape[1] > self.feature_dim:
                features = features[:, :self.feature_dim]  # Truncate
            else:
                return None, 0.0  # Can't proceed with insufficient dimensions
        
        features_t = features.T  # Convert to (feature_dim, N) for matrix multiply
        
        if self.use_gds:
            # Use sm_gds_similarity from subspace_utils to compute similarity scores
            sim_matrix = sm_gds_similarity(features_t, self.U, self.G)
            
            # Get prediction (class with highest similarity)
            class_idx = np.argmax(sim_matrix, axis=0)[0]  # First (and only) prediction
            predicted_class_id = self.class_indices[class_idx]
            
            # Get normalized confidence
            confidence = sim_matrix[class_idx, 0] / np.sum(sim_matrix[:, 0])
        else:
            # Use subspace_classify for standard subspace method
            predicted_class_id = subspace_classify(features, self.V, self.class_indices)[0]
            
            # Compute confidence using sm_similarity
            sim_matrix = sm_similarity(features_t, self.V)
            class_idx = np.where(np.array(self.class_indices) == predicted_class_id)[0][0]
            confidence = sim_matrix[class_idx, 0] / np.sum(sim_matrix[:, 0])

        # Get class name
        class_name = self.class_names.get(predicted_class_id, f"Unknown_{predicted_class_id}")

        return class_name, confidence

class FrameSequenceProcessor:
    def __init__(
        self,
        folder_path,
        visualize=False,
        cache_floor_params=True,
        enable_recognition=True,
        model_dir="pointnet_cls",
        dataset="simple",
        checkpoint_path=None,
        subspace_dim = 50,
        use_gds=True,
        n_components_to_remove=1,
        use_uniform_sample=False,
    ):
        self.folder_path = folder_path
        self.visualize = visualize
        self.cache_floor_params = cache_floor_params
        self.enable_recognition = enable_recognition

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
        self.noise_nb_neighbors = 30
        self.noise_std_ratio = .1
        self.dbscan_eps = 0.0001
        self.dbscan_min_points = 10
        self.min_object_height = 0.05
        self.max_object_height = 1.0

        # Performance optimization parameters
        self.max_points_for_processing = 50000  # Max points before voxel downsampling
        self.voxel_size_initial = 0.003  # Initial voxel size (2cm)
        self.voxel_size_secondary = 0.005  # Secondary voxel size (1.5cm)


        # Cache file for floor parameters
        self.cache_file = os.path.join(folder_path, "floor_params_cache.pkl")

        # Initialize object recognizer
        if self.enable_recognition:
            self.recognizer = LiDARObjectRecognizer(
                model_dir=model_dir, dataset=dataset, checkpoint_path=checkpoint_path, ndim=subspace_dim, use_gds=use_gds, n_components_to_remove=n_components_to_remove, use_uniform_sample=use_uniform_sample
            )
            self.recognizer.load_model()
        else:
            self.recognizer = None

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
        """Optimized single frame processing with PointNet-based object recognition"""
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
            f"Point cloud with {len(pcd.points)} points."
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
            
            # Reduce noise removal parameters further for medium-large point clouds
            effective_neighbors = self.noise_nb_neighbors

            temp_pcd = o3d.geometry.PointCloud()
            temp_pcd.points = o3d.utility.Vector3dVector(non_floor_points)
            if non_floor_colors is not None:
                temp_pcd.colors = o3d.utility.Vector3dVector(non_floor_colors)

            # More relaxed noise removal for faster processing
            clean_pcd, clean_indices = temp_pcd.remove_statistical_outlier(
                nb_neighbors=effective_neighbors,
                std_ratio=self.noise_std_ratio,
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

        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            labels = np.array(
                temp_pcd_cluster.cluster_dbscan(
                    eps=self.dbscan_eps, min_points=self.dbscan_min_points
                )
            )

        if len(labels) == 0 or labels.max() < 0:
            print(f"  Warning: No clusters found in {os.path.basename(file_path)}")
            return [], [], [], pcd

        # Vectorized cluster filtering with PointNet-based recognition
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

                # **POINTNET-BASED OBJECT RECOGNITION**
                class_name = f"Obj_{len(valid_clusters):02d}"
                confidence = 0.0

                if (
                    self.enable_recognition
                    and self.recognizer
                    and self.recognizer.loaded
                ):
                    try:
                        if self.recognizer.use_uniform_sample:
                            if len(cluster_points) < 1024:
                                print(
                                    f"  Warning: Cluster {i} has only {len(cluster_points)} points, using FPS to sample 1024 points"
                                )
                            else:
                                fps_points = farthest_point_sample(cluster_points, 1024)
                                cluster_pcd.points = o3d.utility.Vector3dVector(fps_points)

                        predicted_class, confidence = self.recognizer.classify_object(
                            cluster_pcd
                        )
                        
                        if predicted_class is not None:
                            class_name = predicted_class
                            print(
                                f"    Detected: {class_name} (confidence: {confidence:.2f})"
                            )
                    except Exception as e:
                        print(f"    Recognition error: {e}")

                # Create text label above the bounding box
                # Position text above the center of the top face of the bbox
                text_position = [
                    (min_bounds[0] + max_bounds[0]) / 2,  # Center X
                    max_bounds[1] + 0.1,  # 10cm above the top
                    (min_bounds[2] + max_bounds[2]) / 2,  # Center Z
                ]

                # Create text with class name and optionally confidence
                label_text = class_name

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

    def process_all_frames(self, start_frame=0, end_frame=None):
        """Process all PLY files and save separate GIFs for segments with clusters > 0"""
        # Get all PLY files
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
        clusters_geom = []

        if self.visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name="LiDAR Object Detection with PointNet Recognition",
                width=1280,
                height=720,
            )

            # Set up camera view
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0.1, 0.1, 0.1])
            opt.point_size = 2.0

        # Process each frame with timing
        results = []
        start_time = time.time()
        
        # Initialize variables for tracking segments
        current_segment_imgs = []
        segment_index = 0
        in_segment = False

        for i, file_path in enumerate(ply_files):
            if end_frame is not None and i >= end_frame:
                break
            if i < start_frame:
                continue

            frame_start = time.time()
            print(f"Processing frame {i+1}/{len(ply_files)}: {os.path.basename(file_path)}")

            clusters, bboxes, text_labels, original_pcd = self.process_single_frame_optimized(file_path)

            frame_time = time.time() - frame_start
            print(f"  Found {len(clusters)} objects in {frame_time:.3f}s")

            results.append({
                "file_path": file_path,
                "clusters": clusters,
                "bboxes": bboxes,
                "text_labels": text_labels,
                "original_pcd": original_pcd,
                "num_objects": len(clusters),
                "processing_time": frame_time,
            })

            # Only visualize frames with clusters > 0
            if self.visualize and vis is not None:
                if len(clusters) > 0:
                    # If we're not currently in a segment, start a new one
                    if not in_segment:
                        in_segment = True
                        current_segment_imgs = []
                        print(f"Starting new segment {segment_index + 1} at frame {i+1}")
                    
                # Remove old geometries
                for bbox in bboxes_geom:
                    vis.remove_geometry(bbox, reset_bounding_box=False)
                for text in text_geom:
                    vis.remove_geometry(text, reset_bounding_box=False)
                for cluster in clusters_geom:
                    vis.remove_geometry(cluster, reset_bounding_box=False)

                bboxes_geom.clear()
                text_geom.clear()
                clusters_geom.clear()
                
                # Prepare filtered point cloud for visualization
                points = np.asarray(original_pcd.points)
                colors = np.asarray(original_pcd.colors) if original_pcd.has_colors() else None

                # Apply ROI filter
                mask = points[:, 2] >= self.roi_z_threshold
                filtered_points = points[mask]
                filtered_colors = colors[mask] if colors is not None else None

                # Apply rotation to match processing
                if len(filtered_points) > 0:
                    filtered_points_rotated = (self.rotation_matrix @ filtered_points.T).T

                    if vis_pcd is None:
                        # Create a new point cloud for visualization
                        vis_pcd = o3d.geometry.PointCloud()
                        vis_pcd.points = o3d.utility.Vector3dVector(filtered_points_rotated)
                        vis_pcd.colors = o3d.utility.Vector3dVector(
                            filtered_colors if filtered_colors is not None 
                            else np.tile([0.3, 0.6, 1.0], (len(filtered_points_rotated), 1))
                        )
                        vis.add_geometry(vis_pcd)
                    else:
                        # Update existing point cloud
                        vis_pcd.points = o3d.utility.Vector3dVector(filtered_points_rotated)
                        if filtered_colors is not None:
                            vis_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
                        else:
                            vis_pcd.colors = o3d.utility.Vector3dVector(
                                np.tile([0.3, 0.6, 1.0], (len(filtered_points_rotated), 1))
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

                    # Capture image for current segment GIF
                    image = vis.capture_screen_float_buffer(do_render=True)
                    image_np = np.asarray(image)
                    image_np = (image_np * 255).astype(np.uint8)
                    img = Image.fromarray(image_np)
                    current_segment_imgs.append(img)
                
                # If we were in a segment but this frame has 0 clusters, end the segment and save the GIF
                elif in_segment and len(clusters) == 0:
                    in_segment = False
                    
                    if current_segment_imgs:
                        # Create log directory if it doesn't exist
                        os.makedirs("log", exist_ok=True)
                        
                        # Save the GIF for this segment
                        gif_path = f"log/lidar_detection_segment_{segment_index:03d}.gif"
                        current_segment_imgs[0].save(
                            gif_path,
                            save_all=True,
                            append_images=current_segment_imgs[1:],
                            optimize=False,
                            duration=1000/60,  # Duration in milliseconds
                            loop=0,
                        )
                        print(f"Segment {segment_index + 1} GIF saved to {gif_path} with {len(current_segment_imgs)} frames")
                        segment_index += 1

        # Save the last segment GIF if we're still in a segment at the end
        if self.visualize and in_segment and current_segment_imgs:
            os.makedirs("log", exist_ok=True)
            gif_path = f"log/lidar_detection_segment_{segment_index:03d}.gif"
            current_segment_imgs[0].save(
                gif_path,
                save_all=True,
                append_images=current_segment_imgs[1:],
                optimize=False,
                duration=1000/60,  # Duration in milliseconds
                loop=0,
            )
            print(f"Final segment {segment_index + 1} GIF saved to {gif_path} with {len(current_segment_imgs)} frames")

        # Close visualization window
        if vis is not None:
            vis.destroy_window()
            print("Visualization closed")

        total_time = time.time() - start_time
        print(f"\nTotal processing time: {total_time:.3f}s")
        print(f"Average time per frame: {total_time/len(results):.3f}s")
        print(f"Total segments with objects: {segment_index + (1 if in_segment else 0)}")

        return results

    def save_results_summary(self, results):
        """Save a summary of detection results with timing information"""
        summary_file = os.path.join(self.folder_path, "detection_summary.txt")
        with open(summary_file, "w") as f:
            f.write("Object Detection and Recognition Summary (PointNet-based)\n")
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
            f.write(f"Recognition enabled: {self.enable_recognition}\n")
            f.write(f"Recognition method: PointNet + Subspace Classification\n")

        print(f"Summary saved to {summary_file}")


# Usage example
if __name__ == "__main__":
    # Set your folder path here
    folder_path = r"F:\LiDAR Videos\objects_detection\ply"

    # Create processor instance with PointNet-based recognition
    processor = FrameSequenceProcessor(
        folder_path=folder_path,
        visualize=True,  # Set to True if you want to visualize each frame
        cache_floor_params=False,
        enable_recognition=True,  # Enable object recognition
        model_dir="pointnet_cls",  # Path to your trained model
        # model_dir="pointnet2_ssg_wo_normals",  # Path to your trained model
        dataset="simple",  # Dataset used for training
        checkpoint_path=None,  # Path to PointNet checkpoint (optional, uses default if None)
        subspace_dim=300,  # Subspace dimension for PointNet
        use_gds=False,  # Use GDS method for recognition
        n_components_to_remove=5,  # Number of components to remove for GDS
        use_uniform_sample=False,  # Use uniform sampling for GDS
    )

    start_from = 50
    end_frame = None  # Set to None to process all frames


    processor.roi_z_threshold = -0.55
    processor.min_object_height = 0.05
    processor.max_object_height = 1.0
    processor.dbscan_eps = 0.04
    processor.dbscan_min_points = 10

    # Process all frames
    results = processor.process_all_frames(start_frame=start_from, end_frame=end_frame)

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