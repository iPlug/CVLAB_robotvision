import numpy as np
import open3d as o3d
import scipy.linalg as scla


def get_pcd_from_mesh(path, number_of_points=100, fps=True, normalize=True):
    o3d.utility.random.seed(30)
    np.random.seed(30)
    mesh = o3d.io.read_triangle_mesh(path)

    if fps:
        pcd = mesh.sample_points_uniformly(number_of_points=10000)
        fps_points = farthest_point_sample(np.array(pcd.points), number_of_points)

        if normalize:
            fps_points = pc_normalize(fps_points)

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(fps_points)
    else:
        new_pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)

        if normalize:
            new_pcd.points = o3d.utility.Vector3dVector(
                pc_normalize(np.array(new_pcd.points))
            )

    # Visualize the point cloud (uncomment for visualization during processing)
    # pcd.translate([1, 0, 0])
    # pcd_2 = mesh.sample_points_uniformly(number_of_points=number_of_points)
    # pcd_2.translate([2, 0, 0])
    # o3d.visualization.draw_geometries([pcd, pcd_2, new_pcd])

    return new_pcd


def pc_normalize(pc):
    """Normalize point cloud to unit sphere"""
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint, fixed_seed=True):
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
    if fixed_seed:
        # Always start with the first point instead of random
        farthest = 0
    else:
        # Original random initialization
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


def batch_farthest_point_sample(xyz, npoint, fixed_seed=True):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
        fixed_seed: if True, always use the same starting point
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = np.zeros(B, npoint, dtype=np.long)
    distance = np.ones(B, N) * 1e10

    if fixed_seed:
        # Always start with the first point instead of random
        farthest = np.zeros(B, dtype=np.long)
    else:
        # Original random initialization
        farthest = np.randint(0, N, (B,), dtype=np.long)

    batch_indices = np.arange(B, dtype=np.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.max(distance, -1)[1]
    return centroids


def get_shapesubspace(points, center=True, dim=3):
    if center:
        centered_points = points - np.mean(points, axis=0)
    else:
        centered_points = points

    # U, _, _ = scla.svd(centered_points)
    U, _, _ = np.linalg.svd(centered_points)
    return U[:, :dim]


def compute_similarity_between_subspaces(subspace1, subspace2, qr=True):
    """
    Compute the similarity between two subspaces based on their canonical angles.

    Args:
    subspace1 (ndarray): First subspace of shape (n_features, n_dim1)
    subspace2 (ndarray): Second subspace of shape (n_features, n_dim2)

    Returns:
    float: Similarity based on the cosine of the smallest canonical angle
    """
    # Ensure the subspaces are orthonormal
    if qr:
        subspace1, _ = np.linalg.qr(subspace1)
        subspace2, _ = np.linalg.qr(subspace2)

    # Compute the dot product between the subspaces
    dot_product = np.dot(subspace1.T, subspace2)

    # Compute singular values (which are the cosines of the canonical angles)
    singular_values = np.linalg.svd(dot_product, compute_uv=False)

    # The largest singular value corresponds to the cosine of the smallest angle
    similarity = singular_values[0]

    return similarity


def compute_similarities(reference_subspaces, input_subspaces):
    """
    Compute the canonical angles between two sets of subspaces.

    Args:
    reference_subspaces (ndarray): Set of reference subspaces
    input_subspaces (ndarray): Set of input subspaces

    Returns:
    tuple: (max_cosines, mean_cosines) where max_cosines contains the cosines of the first
           canonical angles and mean_cosines the average of cosines for each pair of subspaces
    """
    # Reshape inputs to 3D arrays
    reference_subspaces = reference_subspaces.reshape(
        reference_subspaces.shape[0], reference_subspaces.shape[1], -1, order="F"
    )
    input_subspaces = input_subspaces.reshape(
        input_subspaces.shape[0], input_subspaces.shape[1], -1, order="F"
    )

    n_features, n_ref_dims, n_ref_subspaces = reference_subspaces.shape
    _, n_input_dims, n_input_subspaces = input_subspaces.shape

    # Compute dot products between all basis vectors of all subspaces
    ref_subspaces_flat = reference_subspaces.reshape(n_features, -1, order="F").T
    input_subspaces_flat = input_subspaces.reshape(n_features, -1, order="F")
    all_dot_products = np.dot(ref_subspaces_flat, input_subspaces_flat)

    # Reshape to separate dot products for each pair of subspaces
    dot_products_per_pair = np.reshape(
        all_dot_products,
        (n_ref_dims, n_ref_subspaces, n_input_dims, n_input_subspaces),
        order="F",
    )
    dot_products_per_pair = np.transpose(dot_products_per_pair, (0, 2, 1, 3))

    max_cosines = np.zeros((n_ref_subspaces, n_input_subspaces, n_input_dims))
    mean_cosines = np.zeros((n_ref_subspaces, n_input_subspaces, n_input_dims))

    # Compute canonical angles for each pair of subspaces
    for ref_idx in range(n_ref_subspaces):
        for input_idx in range(n_input_subspaces):
            # Compute Gram matrix for this pair of subspaces
            gram_matrix = np.dot(
                dot_products_per_pair[:, :, ref_idx, input_idx].T,
                dot_products_per_pair[:, :, ref_idx, input_idx],
            )

            # Compute canonical angles up to the minimum dimension of the two subspaces
            for dim in range(n_input_dims):
                # Eigenvalues of the Gram matrix are squared cosines of canonical angles
                cosines_squared = np.linalg.eigvalsh(gram_matrix[: dim + 1, : dim + 1])

                # Store the maximum and mean of cosines
                max_cosines[ref_idx, input_idx, dim] = np.sqrt(np.max(cosines_squared))
                mean_cosines[ref_idx, input_idx, dim] = np.mean(
                    np.sqrt(cosines_squared)
                )

    return max_cosines, mean_cosines


## REORDERING SHAPE SUBSPACE
def get_ordered_similarity(input_centered, target_centered):
    Q_input = get_orthogonal_projection_matrix(input_centered)
    Q_target = get_orthogonal_projection_matrix(target_centered)
    correspondence = orthogonal_matching(Q_input, Q_target)
    # Sort the rows based on the first column
    sorted_indices = np.argsort(correspondence[:, 0])
    sorted_correspondence = correspondence[sorted_indices]
    new_order = sorted_correspondence[:, 1]
    sp_input = np.array(input_centered)
    sp_target = np.array(target_centered)[new_order]

    sim = compute_similarity_between_subspaces(sp_input, sp_target)
    return sim


def get_orthogonal_projection_matrix(V):
    """
    Compute the orthogonal projection matrix of a matrix.

    Args:
    matrix (ndarray): Matrix to project

    Returns:
    ndarray: Orthogonal projection matrix
    """
    V = np.array(V)
    return V @ V.T


def sort_rows(matrix):
    """Sort the unmasked elements of each row in the given matrix."""
    sorted_matrix = np.sort(matrix, axis=1)
    return sorted_matrix


def l1_norm_distance(u, v, t, x_t, y_t):
    """Calculate the L1 norm distance between two rows with optional masking."""
    distance = np.sum(np.abs(u - v))
    if t >= 1:
        distance += np.sum(np.abs(x_t - y_t))
    return distance


def orthogonal_matching(X_A, X_B):
    N = X_A.shape[0]
    C = np.zeros((N, 2), dtype=int)

    Q_A = X_A.copy()
    Q_B = X_B.copy()

    LARGE_VALUE = 1e6

    for t in range(N):
        Q_A_t = sort_rows(Q_A)
        Q_B_t = sort_rows(Q_B)

        min_distance = float("inf")
        min_pair = None

        for i in range(N):
            if Q_A_t[i, 0] >= LARGE_VALUE:
                continue
            for j in range(N):
                if Q_B_t[j, 0] >= LARGE_VALUE:
                    continue
                distance = l1_norm_distance(Q_A_t[i], Q_B_t[j], t, X_A[i], X_B[j])
                if distance < min_distance:
                    min_distance = distance
                    min_pair = (i, j)

        if min_pair is None:
            raise ValueError("No valid pair found. Check the algorithm and data.")

        r_A, r_B = min_pair
        C[t] = [r_A, r_B]

        Q_A[r_A, :] = LARGE_VALUE
        Q_A[:, r_A] = LARGE_VALUE
        Q_B[r_B, :] = LARGE_VALUE
        Q_B[:, r_B] = LARGE_VALUE

    return C


## GMSM


def generate_shape_subspace(points, dim=3):
    if len(points) < 3:
        return None  # Not enough points to generate a subspace
    centered_points = points - np.mean(points, axis=0)
    U, _, _ = scla.svd(centered_points)
    return U[:, :dim]


def grassmann_distance(subspace1, subspace2):
    """Compute the projection metric on Grassmann manifold."""
    Y = subspace1.T @ subspace2
    _, S, _ = scla.svd(Y)
    return np.sqrt(subspace1.shape[1] - np.sum(S**2))


def normalized_projection_kernel(subspace1, subspace2):
    similarity = np.sum((subspace1.T @ subspace2) ** 2)
    return similarity / (subspace1.shape[1] * subspace2.shape[1])


def projection_kernel(subspace1, subspace2):
    return np.sum((subspace1.T @ subspace2) ** 2)


def canonical_angles_kernel(subspace1, subspace2):
    Y = subspace1.T @ subspace2
    _, S, _ = np.linalg.svd(Y)
    cos_theta_squared = S**2
    return np.mean(cos_theta_squared)


def grassmann_kernel(subspace1, subspace2, method="canonical_angles"):
    """Compute the projection kernel on Grassmann manifold."""
    return canonical_angles_kernel(subspace1, subspace2)
    # if method == 'canonical_angles':
    #     return canonical_angles_kernel(subspace1, subspace2)
    # elif method == 'projection':
    #     return projection_kernel(subspace1, subspace2)
    # else:
    #     return normalized_projection_kernel(subspace1, subspace2)


def gmsm(reference_subspaces, input_subspace, n_dim):
    # Compute kernel matrix
    K = np.zeros((len(reference_subspaces), len(reference_subspaces)))
    for i in range(len(reference_subspaces)):
        for j in range(len(reference_subspaces)):
            K[i, j] = grassmann_kernel(reference_subspaces[i], reference_subspaces[j])

    # Compute eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Project reference subspaces
    projected_references = []
    for subspace in reference_subspaces:
        proj = np.zeros(n_dim)
        for i in range(n_dim):
            for j in range(len(reference_subspaces)):
                proj[i] += eigenvectors[j, i] * grassmann_kernel(
                    subspace, reference_subspaces[j]
                )
        projected_references.append(proj / np.sqrt(eigenvalues[:n_dim]))

    # Project input subspace
    proj_input = np.zeros(n_dim)
    for i in range(n_dim):
        for j in range(len(reference_subspaces)):
            proj_input[i] += eigenvectors[j, i] * grassmann_kernel(
                input_subspace, reference_subspaces[j]
            )
    proj_input /= np.sqrt(eigenvalues[:n_dim])

    # Compute similarity
    similarities = [np.sum(proj_input * proj_ref) for proj_ref in projected_references]
    max_similarity = np.max(similarities)
    max_index = np.argmax(similarities)
    sum_similarity = np.sum(similarities)

    return float(sum_similarity), max_index
