import os
import sys
import glob
import copy
import time
from pathlib import Path
import numpy as np
import open3d as o3d
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import PIL.Image as Image
from utils.load_env import load_env, get_env




def process_point_clouds_with_alignment(
    ply_folder: str,
    output_folder: str,
    reference_point: np.ndarray = np.array([-0.02, 0.1, -0.54]),
    roi_size: np.ndarray = np.array([0.3, 0.5, 0.3]),
    visualization: bool = False,
) -> str:
    """
    Enhanced point cloud processing pipeline with single loop:
    Processing and visualization happen in a single loop.

    Args:
        ply_folder: Folder containing input PLY files
        output_folder: Folder to save output PLY files
        reference_point: Center point of the manual ROI
        roi_size: Size of the ROI box [x, y, z] in meters
        visualization: Whether to visualize results

    Returns:
        str: Status of the processing ('success' or error message)

    Raises:
        FileNotFoundError: If ply_folder doesn't exist
        ValueError: If no PLY files are found in the folder
    """
    # Convert string paths to Path objects
    ply_folder = Path(ply_folder)
    output_folder = Path(output_folder)

    # Validate input folder
    if not ply_folder.exists():
        raise FileNotFoundError(f"Input folder not found: {ply_folder}")
        
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    # gif_folder is parent of ply_folder
    gif_folder = os.path.dirname(ply_folder)

    # 1. Load all PLY files
    ply_files = sorted(glob.glob(os.path.join(ply_folder, "*.ply")))
    print(f"Found {len(ply_files)} PLY files")

    # 2. Calculate manual ROI bounds
    min_bound = reference_point - roi_size / 2
    max_bound = reference_point + roi_size / 2
    print(f"Manual ROI: Min {min_bound}, Max {max_bound}")

    # Setup for visualization
    imgs = []
    display_pcd = None

    if visualization:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=400, height=300)

    # Single loop to process and visualize all point clouds
    print("Processing point clouds...")
    for i, file in enumerate(tqdm(ply_files, desc="Processing point clouds")):
        # Load point cloud
        pcd = o3d.io.read_point_cloud(file)

        # 1. Apply ROI filter
        points = np.asarray(pcd.points)
        # TODO: make this a parameter
        mask = np.all((points >= (-0.2, -0.5, -.5)) & (points <= (0.2, 0.5, 0)), axis=1)
        filtered_pcd = pcd.select_by_index(np.where(mask)[0])

        # Skip further processing if point cloud is empty
        if len(np.asarray(filtered_pcd.points)) == 0:
            continue

        # 2. Apply RANSAC for floor plane detection and alignment
        try:
            # Use RANSAC to detect floor plane
            plane_model, inliers = filtered_pcd.segment_plane(
                distance_threshold=0.01, ransac_n=3, num_iterations=1000
            )

            # Extract floor normal from plane model
            a, b, c, d = plane_model
            floor_normal = np.array([a, b, c])
            floor_normal = floor_normal / np.linalg.norm(floor_normal)  # Normalize

            # Calculate rotation to align floor normal with Y-axis (0,1,0)
            up_axis = np.array([0, 1, 0])
            rotation_axis = np.cross(floor_normal, up_axis)

            # Camera position as rotation center
            camera_point = np.array([0, 0, 0])

            # Apply rotation if needed
            if np.linalg.norm(rotation_axis) >= 1e-6:
                # Normalize rotation axis and calculate angle
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                angle = np.arccos(np.dot(floor_normal, up_axis))

                # Create rotation matrix
                try:
                    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                        rotation_axis * angle
                    )
                except AttributeError:
                    rotation_matrix = (
                        o3d.utility.Vector3dVector.get_rotation_matrix_from_axis_angle(
                            rotation_axis * angle
                        )
                    )

                # Apply rotation around the camera point
                filtered_pcd.rotate(rotation_matrix, center=camera_point)
        except Exception as e:
            print(f"RANSAC failed for frame {i}: {e}")

        
        # 3. Apply height threshold filtering (y > -0.09)
        # TODO: make this a parameter
        mask = np.asarray(filtered_pcd.points)[:, 1] > -0.094
        filtered_pcd = filtered_pcd.select_by_index(np.where(mask)[0])

        # 4. Remove statistical outliers
        if len(np.asarray(filtered_pcd.points)) > 20:
            filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=0.1
            )

        bbox = filtered_pcd.get_axis_aligned_bounding_box()
        print(
            f"Frame {i}: Filtered point cloud has {len(filtered_pcd.points)} points, "
            f"Bounding box: {bbox.get_max_bound()-bbox.get_min_bound()}"            
        )

        # Save the processed point cloud
        output_file = os.path.join(output_folder, os.path.basename(file))
        o3d.io.write_point_cloud(output_file, filtered_pcd)

        # Store for next iteration (ICP reference)
        previous_pcd = copy.deepcopy(filtered_pcd)

        # Visualization
        if visualization:
            if i == 0:
                # For the first frame, create a point cloud and add it to the visualizer
                display_pcd = o3d.geometry.PointCloud()
                display_pcd.points = o3d.utility.Vector3dVector(
                    np.asarray(filtered_pcd.points)
                )
                display_pcd.colors = o3d.utility.Vector3dVector(
                    np.asarray(filtered_pcd.colors)
                )
                vis.add_geometry(display_pcd)
            else:
                # For subsequent frames, update the existing point cloud's points and colors
                display_pcd.points = o3d.utility.Vector3dVector(
                    np.asarray(filtered_pcd.points)
                )
                display_pcd.colors = o3d.utility.Vector3dVector(
                    np.asarray(filtered_pcd.colors)
                )

                vis.update_geometry(display_pcd)

            # Update visualization
            vis.poll_events()
            vis.update_renderer()

            # Capture image for GIF
            image = vis.capture_screen_float_buffer(do_render=True)
            image_np = np.asarray(image)
            image_np = (image_np * 255).astype(np.uint8)
            img = Image.fromarray(image_np)
            imgs.append(img)

    # Clean up visualization
    if visualization:
        vis.destroy_window()

        # Save images as a GIF
        if len(imgs) > 0:
            gif_path = os.path.join(gif_folder, "roi_filtered.gif")
            imgs[0].save(
                gif_path,
                save_all=True,
                append_images=imgs[1 : min(len(imgs), 200)],  # Limit to 200 frames
                optimize=False,
                duration=30,  # Duration in milliseconds
                loop=0,
            )
            print(f"Saved ROI filtered images as GIF: {gif_path}")

    print("Processing complete!")
    return "success"


# Example usage
if __name__ == "__main__":
    # Load env
    load_env('local')

    objects = [
        "ball_1",
        # "box_1",
        # "bottle_1",
        # "hand_1",
        # "can_1",
        # "box_2",
        # "box_3",
        # "can_2",
        # "can_3",
        # "hand_2",
        # "hand_3",
        # "bottle_2",
        # "bottle_3",
    ]

    # Load and validate environment
    try:
        load_env('local')
    except (FileNotFoundError, ValueError) as e:
        print(f"Environment configuration error: {e}")
        sys.exit(1)

    root_folder = get_env("BAG_FILE_ROOT_FOLDER")
    if not os.path.exists(root_folder):
        print(f"Error: BAG_FILE_ROOT_FOLDER '{root_folder}' does not exist")
        sys.exit(1)

    for object_3d in objects:
        object_folder_path = Path(root_folder) / object_3d
        print(f"Processing object {object_3d} in {object_folder_path}")
        
        # Verify object folder exists
        if not object_folder_path.exists():
            print(f"Warning: Object folder {object_folder_path} not found, skipping...")
            continue

        result = process_point_clouds_with_alignment(
            ply_folder=str(object_folder_path / "ply"),
            output_folder=str(object_folder_path / "filtered_ply"),
            reference_point=np.array([0, 0, -0.25]),
            roi_size=np.array([0.4, 1, 0.5]),
            visualization=True,
        )
