"""
Subspace-based classification utilities with Generalized Difference Subspace (GDS) support
"""

import numpy as np
import torch


def get_subspace(matrix, n_dim=10):
    """
    Get the subspace of the matrix

    Parameters:
    -----------
    matrix : numpy array
        Input data matrix where each column is a feature vector
    n_dim : int, optional
        Dimension of the subspace (default: 10)

    Returns:
    --------
    subspace : numpy array
        Orthonormal basis vectors of the subspace
    """
    # SVD returns U with columns as the left singular vectors
    # These form an orthonormal basis for the column space of the matrix
    U, S, V = np.linalg.svd(matrix, full_matrices=False)
    return U[
        :, : min(n_dim, U.shape[1])
    ]  # Ensure we don't request more dims than available


def get_subspace_by_t(matrix, threshold=0.95):
    """
    Get the subspace of the matrix with dimensionality determined by
    cumulative energy threshold

    Parameters:
    -----------
    matrix : numpy array
        Input data matrix where each column is a feature vector
    threshold : float, optional
        Energy threshold (default: 0.95)

    Returns:
    --------
    subspace : numpy array
        Orthonormal basis vectors of the subspace
    """
    # SVD returns U with columns as the left singular vectors
    # and S as singular values
    U, S, V = np.linalg.svd(matrix, full_matrices=False)

    # Calculate the squared singular values (energy)
    squared_s = S**2

    # Calculate the total energy
    total_energy = np.sum(squared_s)

    # Calculate the cumulative energy ratio
    cumulative_energy_ratio = np.cumsum(squared_s) / total_energy

    # Find the minimum k_c that satisfies the threshold
    k_c = np.argmax(cumulative_energy_ratio >= threshold) + 1

    print(
        f"Selected dimensionality k_c = {k_c} with energy ratio {cumulative_energy_ratio[k_c-1]:.4f}"
    )

    # Return the first k_c columns of U
    return U[:, :k_c]


def determine_global_dimensionality(train_features, threshold=0.99):
    """
    Determine a single dimensionality that satisfies the energy threshold
    across all classes

    Parameters:
    -----------
    train_features : dict
        Dictionary of training feature matrices by class
    threshold : float
        Energy threshold (default: 0.99)

    Returns:
    --------
    global_dim : int
        Dimensionality to use for all class subspaces
    """
    max_dim_needed = 0

    # Check each class
    for class_idx, features in train_features.items():
        features_t = features.T
        U, S, V = np.linalg.svd(features_t, full_matrices=False)

        # Calculate energy ratio
        squared_s = S**2
        total_energy = np.sum(squared_s)
        cumulative_energy_ratio = np.cumsum(squared_s) / total_energy

        # Find k_c that meets threshold
        k_c = np.argmax(cumulative_energy_ratio >= threshold) + 1
        max_dim_needed = max(max_dim_needed, k_c)

    print(
        f"Global dimensionality needed for {threshold} energy threshold: {max_dim_needed}"
    )
    return max_dim_needed


def create_generalized_difference_subspace(V, n_components_to_remove=1):
    """
    Create a generalized difference subspace (GDS).

    Parameters:
    -----------
    V : numpy array
        Tensor of subspace basis vectors organized by class
    n_components_to_remove : int, optional
        Number of principal components to remove (default: 1)

    Returns:
    --------
    G : numpy array
        Orthonormal basis vectors of the GDS
    evals : numpy array
        Eigenvalues of the sum projection matrix
    """
    n_dim, n_subdim, n_class = V.shape

    # Create sum projection matrix P
    P = np.zeros((n_dim, n_dim))
    for i in range(n_class):
        P += V[:, :, i] @ V[:, :, i].T

    # Eigendecomposition of P
    evals, evecs = np.linalg.eig(P)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Determine rank of P (count significant eigenvalues)
    rank_P = np.sum(evals > 1e-10)
    rank_P = int(rank_P)

    # Limit n_components_to_remove to avoid removing all
    n_components_to_remove = min(n_components_to_remove, rank_P - 1)

    # Create GDS by removing principal components
    G = evecs[:, n_components_to_remove:rank_P].T

    return G, evals


def project_subspaces(V, G):
    """
    Project subspaces onto the GDS.

    Parameters:
    -----------
    V : numpy array
        Tensor of subspace basis vectors organized by class
    G : numpy array
        GDS basis vectors

    Returns:
    --------
    U : numpy array
        Projected subspace basis vectors
    """
    n_dim, n_subdim, n_class = V.shape
    U = np.zeros((G.shape[0], n_subdim, n_class))

    for i in range(n_class):
        # Project subspace onto GDS and orthonormalize
        projected = G @ V[:, :, i]
        q, r = np.linalg.qr(projected)
        U[:, :, i] = q[
            :, : min(n_subdim, q.shape[1])
        ]  # Handle case where q has fewer columns

    return U


def sm_similarity(Y, V):
    """
    Compute similarity scores for standard Subspace Method (SM).

    Parameters:
    -----------
    Y : numpy array
        Feature vectors (each column is a feature vector)
    V : numpy array
        Subspace basis vectors organized by class

    Returns:
    --------
    SIM : numpy array
        Similarity matrix (n_class × n_samples)
    """
    n_dim, n_subdim, n_class = V.shape
    SIM = np.zeros((n_class, Y.shape[1]))

    for i in range(n_class):
        # Project features onto class subspace basis, square, and sum
        SIM[i, :] = np.sum((V[:, :, i].T @ Y) ** 2, axis=0)

    return SIM


def sm_gds_similarity(Y, U, G):
    """
    Compute similarity scores for Subspace Method with GDS.

    Parameters:
    -----------
    Y : numpy array
        Feature vectors (each column is a feature vector)
    U : numpy array
        Projected subspace basis vectors
    G : numpy array
        GDS basis vectors

    Returns:
    --------
    SIM : numpy array
        Similarity matrix (n_class × n_samples)
    """
    n_gdim, n_subdim, n_class = U.shape
    n_samples = Y.shape[1]
    SIM = np.zeros((n_class, n_samples))

    # Project all feature vectors onto GDS at once
    Y_projected = G @ Y

    for i in range(n_class):
        # Project projected features onto projected class subspace basis, square, and sum
        SIM[i, :] = np.sum((U[:, :, i].T @ Y_projected) ** 2, axis=0)

    return SIM


def build_class_subspaces(train_features, subspace_dim=50):
    """
    Build subspace representations for each class from training features

    Parameters:
    -----------
    train_features : dict
        Dictionary mapping class indices to feature matrices
    subspace_dim : int, optional
        Dimension of the subspace (default: 50)

    Returns:
    --------
    V : numpy array
        Subspace bases organized by class
    class_indices : list
        List of class indices in the order they appear in V
    """
    # Get the number of classes and feature dimension
    n_classes = len(train_features)
    sample_class = list(train_features.keys())[0]
    feature_dim = train_features[sample_class].shape[1]  # Assuming features are rows

    # Initialize the subspace tensor
    V = np.zeros((feature_dim, subspace_dim, n_classes))
    class_indices = sorted(train_features.keys())

    # Compute subspace for each class
    for i, class_idx in enumerate(class_indices):
        # Transpose so that columns are feature vectors
        features_t = train_features[class_idx].T

        # Compute subspace
        V[:, :, i] = get_subspace(features_t, subspace_dim)

    return V, class_indices


def extract_features_batch(model, points, device):
    """
    Extract features for a batch of point clouds using the model's existing extract_features method

    Parameters:
    -----------
    model : torch.nn.Module
        Trained model with feature extraction capability
    points : torch.Tensor
        Batch of point clouds in format [B, N, 3]
    device : str
        Device to run inference on

    Returns:
    --------
    features : torch.Tensor
        Extracted features [B, feature_dim]
    """
    model.eval()
    with torch.no_grad():
        # Move to device and transpose to [B, 3, N] format
        points = points.to(device)
        points = points.transpose(2, 1)

        # Extract features using the model's existing method
        features = model.extract_features(points, feature_level="global")[
            "global_features"
        ]

        return features.cpu()


def subspace_classify(features, V, class_indices):
    """
    Classify features using standard subspace similarity

    Parameters:
    -----------
    features : numpy array
        Feature vectors to classify [N, feature_dim]
    V : numpy array
        Subspace bases organized by class
    class_indices : list
        List of class indices in the order they appear in V

    Returns:
    --------
    pred_classes : numpy array
        Predicted class indices
    """
    # Convert features to [feature_dim, N] for sm_similarity
    features_t = features.T

    # Compute similarity scores
    SIM = sm_similarity(features_t, V)

    # Get predictions (class with highest similarity)
    predictions = np.argmax(SIM, axis=0)

    # Convert predictions to class indices
    pred_classes = np.array([class_indices[p] for p in predictions])

    return pred_classes


def create_uniform_rotations(n_viewpoints=30):
    """
    Create a set of rotation matrices that uniformly sample viewpoints on a sphere

    Args:
        n_viewpoints: approximate number of viewpoints to generate

    Returns:
        list of rotation matrices
    """
    # Determine number of viewpoints in each dimension based on total count
    # Use Fibonacci sphere method for uniform distribution on a sphere
    rotations = []
    golden_ratio = (1 + 5**0.5) / 2

    for i in range(n_viewpoints):
        # Fibonacci lattice algorithm for uniform sphere sampling
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n_viewpoints)

        # Convert spherical to cartesian coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Create view vector (camera position)
        view_vector = np.array([x, y, z])

        # We need to create a rotation matrix that rotates [0,0,1] to this point
        # First, compute the rotation axis (cross product with [0,0,1])
        rotation_axis = np.cross(np.array([0, 0, 1]), view_vector)

        if np.linalg.norm(rotation_axis) < 1e-6:
            # Special case: view vector is parallel to [0,0,1]
            if view_vector[2] > 0:  # Same direction
                rotation_matrix = np.eye(3)
            else:  # Opposite direction
                rotation_matrix = np.diag([1, 1, -1])
        else:
            # Normalize rotation axis
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            # Calculate rotation angle
            cos_angle = np.dot(np.array([0, 0, 1]), view_vector)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

            # Create rotation matrix using Rodrigues' rotation formula
            K = np.array(
                [
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0],
                ]
            )
            rotation_matrix = (
                np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            )

        rotations.append(rotation_matrix)

    return rotations


def rotate_point_cloud_with_matrix(point_cloud, rotation_matrix):
    """
    Apply a specific rotation matrix to a point cloud

    Args:
        point_cloud: numpy array of shape (N, 3) for N points
        rotation_matrix: 3x3 rotation matrix

    Returns:
        rotated point cloud of the same shape
    """
    return np.dot(point_cloud, rotation_matrix.T)


def compute_similarity_msm(V1, V2):
    """
    Compute similarity between two subspaces using canonical angles
    following the Mutual Subspace Method.

    Parameters:
    -----------
    V1 : numpy array
        First subspace basis vectors
    V2 : numpy array
        Second subspace basis vectors

    Returns:
    --------
    similarity : float
        Similarity based on canonical angles
    """
    # Compute cosines of canonical angles using SVD
    COS_C_ANGLES = np.linalg.svd(V1.T @ V2, compute_uv=False)

    # Similarity is the mean of squared cosines (like in MSM)
    similarity = np.mean(COS_C_ANGLES**2)

    return similarity
