import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

file = "F:\LiDAR Videos\objects_detection\ply\_1294775.16500000003725.ply"  # 2 objects hand and jar
# file = "F:\LiDAR Videos\objects_detection\ply\_1329392.46399999991991.ply"  # 3 objects with pet bottle

visualize = False

# Read the point cloud
pcd = o3d.io.read_point_cloud(file)
points = np.asarray(pcd.points)

# Filter points within ROI
mask = points[:, 2] >= -0.55
filtered_pcd = pcd.select_by_index(np.where(mask)[0])

# Visualize the filtered point cloud
if visualize:
    print(f"Filtered point cloud has {len(filtered_pcd.points)} points")
    o3d.visualization.draw_geometries([filtered_pcd])

# Detect floor using RANSAC
plane_model, inliers = filtered_pcd.segment_plane(
    distance_threshold=0.01, ransac_n=3, num_iterations=1000
)
[a, b, c, d] = plane_model

if visualize:
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# Extract floor as a separate point cloud
floor = filtered_pcd.select_by_index(inliers)
floor.paint_uniform_color([1.0, 0, 0])  # Red color for visualization

# Get points that are not part of the floor
non_floor = filtered_pcd.select_by_index(inliers, invert=True)

# Calculate the rotation to make the floor horizontal
# The normal vector of the floor plane is [a, b, c]
normal_vector = np.array([a, b, c])
normal_vector = normal_vector / np.linalg.norm(normal_vector)  # Normalize
target_normal = np.array([0, 1, 0])  # We want floor normal to point up

# Calculate rotation axis and angle
rotation_axis = np.cross(normal_vector, target_normal)
if np.linalg.norm(rotation_axis) < 1e-6:
    # Normal vectors are already aligned or opposite
    if np.dot(normal_vector, target_normal) < 0:
        # They're opposite, rotate 180 degrees around x-axis
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0, 0]))
    else:
        # They're already aligned, no rotation needed
        R = np.identity(3)
else:
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    cos_angle = np.dot(normal_vector, target_normal)
    angle = np.arccos(cos_angle)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

# Apply rotation to both point clouds
filtered_pcd.rotate(R, center=(0, 0, 0))
floor.rotate(R, center=(0, 0, 0))
non_floor.rotate(R, center=(0, 0, 0))

# Visualize the result
if visualize:
    o3d.visualization.draw_geometries([non_floor])

# Remove noise using statistical outlier removal
print("Performing statistical outlier removal...")
non_floor_clean, ind = non_floor.remove_statistical_outlier(
    nb_neighbors=20,  # Number of neighbors to consider
    std_ratio=2.0,  # Standard deviation threshold
)

print(f"Removed {len(non_floor.points) - len(non_floor_clean.points)} points as noise")

# Visualize the result after noise removal

non_floor_clean.paint_uniform_color([0, 1, 0])  # Green color for visualization
if visualize:
    print("Displaying clean point cloud without noise:")
    print(f"Cleaned point cloud has {len(non_floor_clean.points)} points")
    o3d.visualization.draw_geometries([non_floor_clean])

# Cluster objects using DBSCAN
print("Clustering objects using DBSCAN...")
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(non_floor_clean.cluster_dbscan(eps=0.05, min_points=10))

print(f"Found {labels.max() + 1} clusters")

# Visualize the clusters with different colors
max_label = labels.max()
print(f"Point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# Set color for noise points (labeled as -1)
colors[labels < 0] = [0, 0, 0, 1]  # Black color for noise points

non_floor_clean.colors = o3d.utility.Vector3dVector(colors[:, :3])

if visualize:
    o3d.visualization.draw_geometries([non_floor_clean])

# Create individual point clouds for each cluster
clusters = []
for i in range(max_label + 1):
    cluster = non_floor_clean.select_by_index(np.where(labels == i)[0])
    clusters.append(cluster)
    print(f"Cluster {i} has {len(cluster.points)} points")

    # Calculate and draw bounding box for each cluster
    bbox = cluster.get_axis_aligned_bounding_box()
    bbox.color = (1, 0, 1)  # Purple color for bounding box
    clusters.append(bbox)

# Visualize all clusters with their bounding boxes
if visualize:
    o3d.visualization.draw_geometries(clusters)


# Filter objects based on their height
print("Filtering objects based on height...")
filtered_clusters = []
filtered_bboxes = []

# Define height thresholds (adjust these values based on your needs)
MIN_HEIGHT = 0.05  # Minimum object height in meters
MAX_HEIGHT = 1.0  # Maximum object height in meters

for i in range(max_label + 1):
    cluster = non_floor_clean.select_by_index(np.where(labels == i)[0])
    bbox = cluster.get_axis_aligned_bounding_box()

    # Get the min and max points of the bounding box
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound

    # Calculate height (z-axis difference)
    height = max_bound[2] - min_bound[2]

    if MIN_HEIGHT <= height <= MAX_HEIGHT:
        print(f"Cluster {i}: Height = {height:.3f}m - Kept")
        filtered_clusters.append(cluster)

        # Add colored bounding box
        bbox.color = (0, 1, 0)  # Green for valid objects
        filtered_bboxes.append(bbox)
    else:
        print(f"Cluster {i}: Height = {height:.3f}m - Filtered out")

print(f"Kept {len(filtered_clusters)} objects after height filtering")

# Visualize the filtered objects with their bounding boxes
o3d.visualization.draw_geometries(filtered_clusters + filtered_bboxes)


pcd.rotate(R, center=(0, 0, 0))
o3d.visualization.draw_geometries([pcd] + filtered_bboxes)
