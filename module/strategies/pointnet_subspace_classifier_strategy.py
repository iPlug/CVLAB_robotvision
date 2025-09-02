"""
PointNet + Subspace classification strategy extracted from 5_folder_recognition.py
"""

import open3d as o3d
import numpy as np
import os
import sys
import torch
from typing import Tuple, Optional, Any

# Add the project root to path for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))

from module.core.classification_strategy import ClassificationStrategy

# Import PointNet models
try:
    from models import pointnet2_cls_ssg, pointnet_cls
except ImportError:
    print("Warning: PointNet models not found. Please ensure models/ directory is in the path.")
    pointnet2_cls_ssg = None
    pointnet_cls = None

# Import subspace utilities
try:
    from utils.subspace_utils import (
        build_class_subspaces, 
        subspace_classify,
        create_generalized_difference_subspace, 
        project_subspaces,
        sm_similarity,
        sm_gds_similarity
    )
except ImportError:
    print("Warning: Subspace utilities not found. Please ensure utils/ directory is in the path.")


class PointNetSubspaceClassifierStrategy(ClassificationStrategy):
    """
    Object classification using PointNet features + subspace classification with GDS support.
    
    This is the exact LiDARObjectRecognizer implementation from 5_folder_recognition.py,
    adapted to work as a classification strategy.
    """

    def __init__(self, model_dir="pointnet_cls", dataset="simple", ndim=50, 
                use_gds=True, n_components_to_remove=1, checkpoint_path=None, use_uniform_sample=False):
        """
        Initialize the PointNet + Subspace classifier.
        
        Args:
            model_dir: Model directory name
            dataset: Dataset name used for training
            ndim: Subspace dimension
            use_gds: Whether to use Generalized Difference Subspace
            n_components_to_remove: Number of components to remove for GDS
            checkpoint_path: Path to PointNet checkpoint file
            use_uniform_sample: Whether to use uniform sampling for preprocessing
        """
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
        if pointnet2_cls_ssg is None or pointnet_cls is None:
            print("Error: PointNet models not available")
            return False
            
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
        if hasattr(point_cloud, 'points'):
            # Open3D PointCloud object
            points = np.asarray(point_cloud.points)
        else:
            # Assume it's already a numpy array
            points = point_cloud
        
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

    def classify_object(self, point_cloud: Any) -> Tuple[Optional[str], float]:
        """
        Classify a single object point cloud using subspace or GDS method.
        
        Args:
            point_cloud: Point cloud data (numpy array, Open3D PointCloud, or cluster dict)
            
        Returns:
            Tuple of (class_name, confidence)
        """
        if not self.loaded:
            return None, 0.0

        # Handle different input formats
        if isinstance(point_cloud, dict) and 'points' in point_cloud:
            # Cluster dictionary from clustering strategy
            actual_point_cloud = point_cloud['points']
        else:
            # Direct point cloud input
            actual_point_cloud = point_cloud

        # Extract features using PointNet
        features = self.extract_features(actual_point_cloud)
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

    def get_strategy_name(self) -> str:
        """Get the name of this classification strategy."""
        gds_suffix = " + GDS" if self.use_gds else ""
        return f"PointNet Subspace{gds_suffix}"

    def is_loaded(self) -> bool:
        """Check if the classification model is properly loaded."""
        return self.loaded

    def get_supported_classes(self) -> Optional[list]:
        """Get list of classes that this strategy can classify."""
        if self.class_names:
            return list(self.class_names.values())
        return None

    def get_model_info(self) -> dict:
        """Get information about the underlying model."""
        return {
            "strategy_name": self.get_strategy_name(),
            "loaded": self.is_loaded(),
            "model_dir": self.model_dir,
            "dataset": self.dataset,
            "subspace_dim": self.ndim,
            "use_gds": self.use_gds,
            "device": str(self.device),
            "num_classes": len(self.class_indices) if self.class_indices else 0
        }