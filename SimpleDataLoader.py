"""
@author: Modified for simple folder structure
@file: SimpleDataLoader.py
"""

import os
import numpy as np
import warnings
import glob
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import open3d as o3d

warnings.filterwarnings("ignore")


def pc_normalize(pc):
    """Normalize point cloud to unit sphere"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def load_ply(file_path, seed=99):
    """Load point cloud from PLY file (LiDAR scan from Intel RealSense L515)"""
    try:
        # Load point cloud directly (not mesh) for LiDAR data
        pcd = o3d.io.read_point_cloud(file_path)
        
        if not pcd.has_points():
            raise Exception("Point cloud has no points")

        # Set the random seed for deterministic operations
        o3d.utility.random.seed(seed)
        np.random.seed(seed)

        # Get points
        points = np.asarray(pcd.points)
        
        # LiDAR data might already have normals, check if they exist
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            point_cloud = np.concatenate([points, normals], axis=1)
        else:
            # If no normals in the PLY file, estimate them
            # This is useful for surface normal information
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30
                )
            )
            if pcd.has_normals():
                normals = np.asarray(pcd.normals)
                point_cloud = np.concatenate([points, normals], axis=1)
            else:
                # If still no normals, just use points
                point_cloud = points

        return point_cloud.astype(np.float32)
    except Exception as e:
        print(f"Error loading PLY file {file_path}: {e}")
        return None

class SimpleDataLoader(Dataset):
    def __init__(
        self,
        root,
        args=None,
        split="train",
        max_files=None,
        samples_per_file=1,
        partial_pc=False,
        cut_type="half",
    ):
        """
        Simple DataLoader for directory structure:
        root/
        ├── train/
        │   ├── class1/
        │   │   ├── file1.ply
        │   │   └── file2.ply
        │   └── class2/
        │       ├── file3.ply
        │       └── file4.ply
        └── test/
            ├── class1/
            │   ├── file5.ply
            │   └── file6.ply
            └── class2/
                ├── file7.ply
                └── file8.ply
        """
        self.root = root
        self.npoints = args.num_point if args is not None else 1024
        self.uniform = args.use_uniform_sample if args is not None else True
        self.use_normals = args.use_normals if args is not None else False
        self.max_files = max_files
        self.samples_per_file = samples_per_file
        self.split = split
        self.partial_pc = partial_pc
        self.cut_type = cut_type

        # Define the path to the split folder
        self.split_path = os.path.join(self.root, split)
        
        if not os.path.exists(self.split_path):
            raise ValueError(f"Split directory {self.split_path} does not exist")

        # Get all class directories in the split folder
        class_dirs = [d for d in os.listdir(self.split_path) 
                     if os.path.isdir(os.path.join(self.split_path, d))]
        class_dirs.sort()

        # Create class-to-index mapping
        self.classes = dict(zip(class_dirs, range(len(class_dirs))))
        self.cat = class_dirs

        # Collect all PLY files with their classes
        self.base_datapath = []
        for class_name in class_dirs:
            class_path = os.path.join(self.split_path, class_name)
            ply_files = glob.glob(os.path.join(class_path, "*.ply"))
            
            for ply_file in ply_files:
                self.base_datapath.append((class_name, ply_file))

        # Apply max_files limit if specified
        if self.max_files is not None:
            self.base_datapath = self.base_datapath[:self.max_files]

        # Implementation of multiple samples per file
        # For each file path, create multiple entries with a sample index
        self.datapath = []
        for shape_name, file_path in self.base_datapath:
            for sample_idx in range(self.samples_per_file):
                self.datapath.append((shape_name, file_path, sample_idx))

        print(f"The size of {split} data is {len(self.datapath)} "
              f"(from {len(self.base_datapath)} files with {self.samples_per_file} samples per file)")
        print(f"Found classes: {self.cat}")

    def __len__(self):
        return len(self.datapath)

    def _cut_point_cloud_half(self, point_set):
        """Cut the point cloud in half along the x-axis at the center of mass."""
        # Calculate center of mass
        center = np.mean(point_set[:, 0:3], axis=0)

        # Create a cutting plane through the center (using x-axis by default)
        # Keep only points where x > center_x
        mask = point_set[:, 0] > center[0]

        # Apply mask to keep only points on one side
        partial_points = point_set[mask]

        # If we filtered out too many points, use a different cutting plane
        if len(partial_points) < 50:  # Ensure we have a minimum number of points
            # Try cutting along y-axis instead
            mask = point_set[:, 1] > center[1]
            partial_points = point_set[mask]

            # If still too few points, try z-axis
            if len(partial_points) < 50:
                mask = point_set[:, 2] > center[2]
                partial_points = point_set[mask]

        return partial_points

    def _cut_point_cloud_random_plane(self, point_set):
        """Cut the point cloud using a random plane through the center of mass."""
        # Calculate center of mass
        center = np.mean(point_set[:, 0:3], axis=0)

        # Generate random normal vector for cutting plane
        normal = np.random.randn(3)
        normal = normal / np.linalg.norm(normal)  # Normalize to unit vector

        # Compute the signed distance from each point to the plane
        # d = dot(p - c, normal)
        distances = np.dot(point_set[:, 0:3] - center, normal)

        # Keep points on the positive side of the plane
        mask = distances > 0
        partial_points = point_set[mask]

        # If we filtered out too many points, flip the direction
        if len(partial_points) < 50:
            mask = distances < 0
            partial_points = point_set[mask]

        return partial_points

    def _cut_point_cloud_camera_facing(self, point_set, camera_position=None):
        """
        Cut the point cloud to keep only points potentially visible from a camera.
        """
        # Set default camera position if not provided (origin looking forward)
        if camera_position is None:
            camera_position = np.array([0, 0, 0])

        # Calculate center of mass
        center = np.mean(point_set[:, 0:3], axis=0)

        # Calculate vector from camera to center of point cloud
        view_vector = center - camera_position

        # Normalize the view vector
        view_vector = view_vector / np.linalg.norm(view_vector)

        # For each point, calculate if it's on the camera-facing side of the center plane
        # This creates a plane perpendicular to the view vector at the center point
        # We keep points that are on the camera side of this plane

        # For each point, calculate vector from center to point
        point_vectors = point_set[:, 0:3] - center

        # Calculate dot product with view vector
        # Negative dot product means the point is on the camera side of the plane
        dots = np.sum(point_vectors * (-view_vector), axis=1)

        # Create mask for points that are on the camera side (dot product < 0)
        # or close to the dividing plane
        mask = dots < 0

        # Apply mask to keep only camera-facing points
        camera_facing_points = point_set[mask]

        # If we filtered out too many points, fall back to the original method
        if len(camera_facing_points) < 50:  # Ensure we have a minimum number of points
            return self._cut_point_cloud_half(point_set)

        return camera_facing_points

    def _cut_point_cloud_pca_half(self, point_set):
        """
        Cut the point cloud in half along the principal component with
        the largest variance, keeping the half with the most points.
        """
        from sklearn.decomposition import PCA

        # Get the point positions (first 3 dimensions)
        positions = point_set[:, 0:3]

        # Calculate center of mass
        center = np.mean(positions, axis=0)

        # Center the points
        centered_points = positions - center

        # Perform PCA to find the principal components
        pca = PCA(n_components=3)
        pca.fit(centered_points)

        # Get the first principal component (direction of maximum variance)
        first_pc = pca.components_[0]

        # Project the points onto the first principal component
        projections = np.dot(centered_points, first_pc)

        # Try both sides of the cut and keep the one with more points
        # First, try keeping points with positive projection
        mask_positive = projections > 0
        positive_points = point_set[mask_positive]

        # Then, try keeping points with negative projection
        mask_negative = projections <= 0
        negative_points = point_set[mask_negative]

        # Return the half with more points
        if len(positive_points) >= len(negative_points):
            partial_points = positive_points
        else:
            partial_points = negative_points

        # If we filtered out too many points, try the second principal component
        if len(partial_points) < 50:
            second_pc = pca.components_[1]
            projections = np.dot(centered_points, second_pc)

            # Try both sides of the cut for the second component
            mask_positive = projections > 0
            positive_points = point_set[mask_positive]

            mask_negative = projections <= 0
            negative_points = point_set[mask_negative]

            # Return the half with more points
            if len(positive_points) >= len(negative_points):
                partial_points = positive_points
            else:
                partial_points = negative_points

            # If still too few points, try the third principal component
            if len(partial_points) < 50:
                third_pc = pca.components_[2]
                projections = np.dot(centered_points, third_pc)

                mask_positive = projections > 0
                partial_points = point_set[mask_positive]

                if len(partial_points) < 50:
                    mask_negative = projections <= 0
                    partial_points = point_set[mask_negative]

        return partial_points

    def __getitem__(self, index):
        shape_name, file_path, sample_idx = self.datapath[index]
        cls = self.classes[shape_name]
        label = np.array([cls]).astype(np.int32)

        # Load PLY file fresh for each sample
        if self.samples_per_file > 1 and self.split == "train":
            point_set = load_ply(file_path, seed=index)
        else:
            point_set = load_ply(file_path)

        if point_set is None or len(point_set) < 10:
            print(f"Error or too few points in {file_path}, creating empty point set")
            point_set = np.zeros((self.npoints, 6 if self.use_normals else 3)).astype(
                np.float32
            )

        # Apply partial point cloud transformation if requested (MOVED HERE - before sampling)
        if self.partial_pc:
            if self.cut_type == "half":
                point_set = self._cut_point_cloud_half(point_set)
            elif self.cut_type == "random_plane":
                point_set = self._cut_point_cloud_random_plane(point_set)
            elif self.cut_type == "camera_facing":
                point_set = self._cut_point_cloud_camera_facing(point_set)
            elif self.cut_type == "pca_half":
                point_set = self._cut_point_cloud_pca_half(point_set)

        # Sample or truncate point cloud - with random sampling for each sample
        if len(point_set) > self.npoints:
            if self.uniform:
                # For FPS, perform deterministic sampling
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                # For each sample_idx, perform different random sampling
                # Use the sample_idx to seed the random sampling differently
                np.random.seed(index)  # Use index to ensure reproducibility
                indices = np.random.choice(len(point_set), self.npoints, replace=False)
                point_set = point_set[indices]
                np.random.seed(None)  # Reset the seed
        elif len(point_set) < self.npoints:
            # If not enough points, duplicate some
            indices = np.random.choice(len(point_set), self.npoints - len(point_set))
            extra_points = point_set[indices]
            point_set = np.vstack((point_set, extra_points))
            print(f"Warning: Duplicating points in {file_path} to reach {self.npoints}")

        # PointNet-style normalization: center to mean and scale to unit sphere
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # Keep only XYZ or XYZ+normals based on use_normals flag
        if self.use_normals and point_set.shape[1] > 3:
            # Ensure normals are also normalized (optional)
            # Some implementations normalize normals, others keep them as-is
            normal_lengths = np.sqrt(
                np.sum(point_set[:, 3:6] ** 2, axis=1, keepdims=True)
            )
            normal_lengths[normal_lengths == 0] = 1  # Avoid division by zero
            point_set[:, 3:6] = point_set[:, 3:6] / normal_lengths

            # Return XYZ+normals
            return point_set[:, 0:6].astype(np.float32), label[0]
        else:
            # Return only XYZ
            return point_set[:, 0:3].astype(np.float32), label[0]


if __name__ == "__main__":
    import torch
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser("Simple DataLoader Test")
    parser.add_argument(
        "--root",
        type=str,
        default="D:\LAB\DATASETS\\robot_simple",
        help="Dataset root (should contain train/ and test/ folders)",
    )
    parser.add_argument("--num_point", type=int, default=1024, help="Point Number")
    parser.add_argument(
        "--use_uniform_sample", action="store_true", help="Use uniform sampling"
    )
    parser.add_argument("--use_normals", action="store_true", help="Use normals")
    parser.add_argument(
        "--partial", action="store_true", help="Use partial point clouds"
    )
    parser.add_argument(
        "--cut_type", 
        type=str, 
        default="half", 
        choices=["half", "random_plane", "camera_facing", "pca_half"],
        help="Cut type for partial point clouds"
    )
    parser.add_argument(
        "--samples_per_file", type=int, default=1, help="Number of samples per file"
    )

    args = parser.parse_args()

    print("TESTING SIMPLE DATALOADER WITH ARGS:", args)

    # Test train split
    train_data = SimpleDataLoader(
        args.root,
        args=args,
        split="train",
        partial_pc=args.partial,
        cut_type=args.cut_type,
        samples_per_file=args.samples_per_file,
    )

    print(f"Train data: {len(train_data)} samples")
    print(f"Classes found: {train_data.cat}")

    # Test test split
    test_data = SimpleDataLoader(
        args.root,
        args=args,
        split="test",
        partial_pc=args.partial,
        cut_type=args.cut_type,
        samples_per_file=args.samples_per_file,
    )

    print(f"Test data: {len(test_data)} samples")

    # Test loading some samples
    print("\nTesting sample loading...")
    for i in range(min(3, len(train_data))):
        point, label = train_data[i]
        print(f"Sample {i}: Point cloud shape: {point.shape}, Label: {label}")

    # Test with DataLoader
    DataLoader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
    
    print("\nTesting PyTorch DataLoader...")
    for i, (points, labels) in enumerate(DataLoader):
        print(f"Batch {i}: Points shape: {points.shape}, Labels: {labels}")

        # visualize the first point cloud in the batch
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[0].numpy())
        pcd.paint_uniform_color([1, 0.706, 0])  # Gold color
        o3d.visualization.draw_geometries([pcd], window_name=f"Batch {i} - Sample 0")
        if i >= 2:  # Just test a few batches
            break