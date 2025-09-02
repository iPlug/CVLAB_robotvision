"""
Configuration for PointNet + Subspace classification strategy.
"""

from typing import Dict, Any, Optional
from .detection_config import DetectionConfig


class PointNetConfig(DetectionConfig):
    """
    Configuration class for PointNet + Subspace classification strategy.
    
    Contains all parameters needed for PointNet-based object classification:
    - Model parameters
    - Subspace classification settings
    - DBSCAN clustering parameters
    - Preprocessing settings
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize PointNet configuration with defaults.
        
        Args:
            config_dict: Optional dictionary with custom parameters
        """
        # Initialize base configuration
        super().__init__()
        
        # PointNet-specific default configuration
        pointnet_defaults = {
            # Model parameters
            'model_params': {
                'model_dir': 'pointnet_cls',
                'dataset': 'simple',
                'checkpoint_path': None,  # Uses default if None
                'num_classes': 40,
                'normal_channel': False,
                'device': 'auto'  # 'cuda', 'cpu', or 'auto'
            },
            
            # Subspace classification parameters
            'subspace_params': {
                'ndim': 50,                    # Subspace dimension
                'use_gds': True,               # Use Generalized Difference Subspace
                'n_components_to_remove': 1,   # Components to remove for GDS
                'features_dir': 'features'     # Directory containing trained features
            },
            
            # PointNet preprocessing parameters
            'pointnet_preprocessing': {
                'num_points': 1024,           # Standard PointNet input size
                'normalize_to_unit_sphere': True,
                'center_points': True,
                'use_uniform_sample': False,  # Use uniform sampling vs random
                'min_points_threshold': 3     # Minimum points to process
            },
            
            # DBSCAN clustering parameters
            'dbscan_params': {
                'eps': 0.0001,
                'min_points': 10,
                'max_distance': 0.1           # For object association
            },
            
            # Floor detection and removal parameters
            'floor_params': {
                'roi_z_threshold': -0.55,
                'plane_distance_threshold': 0.01,
                'plane_ransac_n': 3,
                'plane_num_iterations': 1000,
                'floor_height_threshold': 0.01,
                'cache_floor_params': True
            },
            
            # Noise filtering parameters
            'noise_params': {
                'nb_neighbors': 30,
                'std_ratio': 0.1,
                'enable_statistical_outlier_removal': True
            },
            
            # Object filtering parameters
            'object_params': {
                'min_object_height': 0.05,
                'max_object_height': 1.0,
                'min_object_width': 0.01,
                'min_object_depth': 0.01
            },
            
            # Performance optimization parameters
            'performance_params': {
                'max_points_for_processing': 50000,
                'voxel_size_initial': 0.003,     # 3mm voxels
                'voxel_size_secondary': 0.005,   # 5mm voxels
                'enable_voxel_downsampling': True,
                'max_iterations_for_voxel_adjustment': 3
            },
            
            # Class name mapping
            'class_names': {
                0: "bottle",
                1: "box", 
                2: "can",
                3: "hand"
            },
            
            # Recognition confidence parameters
            'confidence_params': {
                'min_confidence_threshold': 0.1,
                'confidence_smoothing': True,
                'smoothing_window_size': 3
            }
        }
        
        # Update with defaults
        self.update_config(pointnet_defaults)
        
        # Update with provided configuration
        if config_dict:
            self.update_config(config_dict)
    
    def validate_config(self) -> bool:
        """
        Validate PointNet-specific configuration parameters.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Validate model parameters
            model = self.get('model_params', {})
            if not model.get('model_dir'):
                print("Error: model_dir cannot be empty")
                return False
            
            # Validate subspace parameters
            subspace = self.get('subspace_params', {})
            if subspace.get('ndim', 0) <= 0:
                print("Error: subspace ndim must be positive")
                return False
            if subspace.get('n_components_to_remove', 0) < 0:
                print("Error: n_components_to_remove cannot be negative")
                return False
            
            # Validate PointNet preprocessing
            preprocessing = self.get('pointnet_preprocessing', {})
            if preprocessing.get('num_points', 0) <= 0:
                print("Error: num_points must be positive")
                return False
            if preprocessing.get('min_points_threshold', 0) <= 0:
                print("Error: min_points_threshold must be positive")
                return False
            
            # Validate DBSCAN parameters
            dbscan = self.get('dbscan_params', {})
            if dbscan.get('eps', 0) <= 0:
                print("Error: DBSCAN eps must be positive")
                return False
            if dbscan.get('min_points', 0) <= 0:
                print("Error: DBSCAN min_points must be positive")
                return False
            
            # Validate object parameters
            obj_params = self.get('object_params', {})
            min_height = obj_params.get('min_object_height', 0)
            max_height = obj_params.get('max_object_height', 1)
            if min_height >= max_height:
                print("Error: min_object_height must be less than max_object_height")
                return False
            
            # Validate performance parameters
            perf = self.get('performance_params', {})
            if perf.get('max_points_for_processing', 0) <= 0:
                print("Error: max_points_for_processing must be positive")
                return False
            if perf.get('voxel_size_initial', 0) <= 0:
                print("Error: voxel_size_initial must be positive")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error validating PointNet configuration: {e}")
            return False
    
    def reset_to_defaults(self):
        """Reset configuration to PointNet defaults."""
        self.__init__()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information in a convenient format.
        
        Returns:
            Dictionary with model information
        """
        model_params = self.get('model_params', {})
        subspace_params = self.get('subspace_params', {})
        
        return {
            'model_dir': model_params.get('model_dir'),
            'dataset': model_params.get('dataset'),
            'checkpoint_path': model_params.get('checkpoint_path'),
            'subspace_dim': subspace_params.get('ndim'),
            'use_gds': subspace_params.get('use_gds'),
            'device': model_params.get('device')
        }
    
    def get_preprocessing_settings(self) -> Dict[str, Any]:
        """
        Get preprocessing settings in a convenient format.
        
        Returns:
            Dictionary with preprocessing settings
        """
        return self.get('pointnet_preprocessing', {})
    
    def get_clustering_settings(self) -> Dict[str, Any]:
        """
        Get DBSCAN clustering settings.
        
        Returns:
            Dictionary with clustering settings
        """
        return self.get('dbscan_params', {})
    
    def get_class_mapping(self) -> Dict[int, str]:
        """
        Get class ID to name mapping.
        
        Returns:
            Dictionary mapping class IDs to names
        """
        return self.get('class_names', {})
    
    def add_class(self, class_id: int, class_name: str):
        """
        Add a new class to the mapping.
        
        Args:
            class_id: Numeric class ID
            class_name: Human-readable class name
        """
        class_names = self.get('class_names', {})
        class_names[class_id] = class_name
        self.set('class_names', class_names)
    
    def set_model_path(self, model_dir: str, checkpoint_path: Optional[str] = None):
        """
        Set model directory and optional checkpoint path.
        
        Args:
            model_dir: Model directory name
            checkpoint_path: Optional path to specific checkpoint
        """
        self.set('model_params.model_dir', model_dir)
        if checkpoint_path:
            self.set('model_params.checkpoint_path', checkpoint_path)
    
    def set_subspace_parameters(self, ndim: int, use_gds: bool = True, n_components_to_remove: int = 1):
        """
        Set subspace classification parameters.
        
        Args:
            ndim: Subspace dimension
            use_gds: Whether to use Generalized Difference Subspace
            n_components_to_remove: Number of components to remove for GDS
        """
        self.set('subspace_params.ndim', ndim)
        self.set('subspace_params.use_gds', use_gds)
        self.set('subspace_params.n_components_to_remove', n_components_to_remove)
    
    def create_fast_config(self) -> 'PointNetConfig':
        """
        Create a configuration optimized for speed.
        
        Returns:
            New PointNetConfig instance with fast settings
        """
        fast = self.copy()
        
        # Optimize for speed
        fast.set('pointnet_preprocessing.num_points', 512)  # Fewer points
        fast.set('subspace_params.ndim', 30)                # Smaller subspace
        fast.set('performance_params.max_points_for_processing', 25000)
        fast.set('performance_params.enable_voxel_downsampling', True)
        fast.set('noise_params.nb_neighbors', 20)           # Fewer neighbors
        fast.set('dbscan_params.min_points', 5)             # Less strict clustering
        fast.set('confidence_params.confidence_smoothing', False)  # No smoothing
        
        return fast
    
    def create_accurate_config(self) -> 'PointNetConfig':
        """
        Create a configuration optimized for accuracy.
        
        Returns:
            New PointNetConfig instance with accurate settings
        """
        accurate = self.copy()
        
        # Optimize for accuracy
        accurate.set('pointnet_preprocessing.num_points', 2048)  # More points
        accurate.set('subspace_params.ndim', 100)               # Larger subspace
        accurate.set('performance_params.max_points_for_processing', 100000)
        accurate.set('performance_params.enable_voxel_downsampling', False)
        accurate.set('noise_params.nb_neighbors', 50)           # More neighbors
        accurate.set('noise_params.std_ratio', 0.05)            # Stricter filtering
        accurate.set('dbscan_params.min_points', 15)            # Stricter clustering
        accurate.set('confidence_params.confidence_smoothing', True)
        accurate.set('confidence_params.smoothing_window_size', 5)
        
        return accurate
    
    def create_pointnet2_config(self) -> 'PointNetConfig':
        """
        Create a configuration for PointNet++ model.
        
        Returns:
            New PointNetConfig instance configured for PointNet++
        """
        pointnet2 = self.copy()
        
        # Configure for PointNet++
        pointnet2.set('model_params.model_dir', 'pointnet2_ssg_wo_normals')
        pointnet2.set('pointnet_preprocessing.num_points', 1024)
        pointnet2.set('subspace_params.ndim', 75)  # Different dimension for PointNet++
        
        return pointnet2