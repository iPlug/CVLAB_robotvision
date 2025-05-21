#!/usr/bin/env python3
"""
PLY to GIF Converter

A script that converts a folder of PLY files into an animated GIF.
Input: Folder path containing PLY files
Output: GIF animation of the PLY files

Requirements:
- open3d
- PIL (Pillow)
- numpy
- imageio
"""

import os
import glob
import argparse
import numpy as np
import open3d as o3d
from PIL import Image
import imageio
import tempfile
import copy
from tqdm import tqdm


def ply_to_gif(input_folder, output_gif, framerate=10, width=800, height=600):
    """
    Convert a folder of PLY files to an animated GIF
    Using a single display point cloud and updating it for each frame

    Args:
        input_folder: Path to folder containing PLY files
        output_gif: Path to save the output GIF
        framerate: Frame rate of the GIF (frames per second)
        width: Image width
        height: Image height
    """
    # Get all PLY files in the input folder
    ply_files = sorted(glob.glob(os.path.join(input_folder, "*.ply")))

    if not ply_files:
        print(f"No PLY files found in {input_folder}")
        return

    print(f"Found {len(ply_files)} PLY files")

    # Create temporary directory to store rendered images
    with tempfile.TemporaryDirectory() as temp_dir:
        image_files = []

        # Create visualizer once
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True, width=width, height=height)

        # Load the first point cloud to initialize
        first_pcd = o3d.io.read_point_cloud(ply_files[0])
        points = np.asarray(first_pcd.points)

        # Filter points within ROI
        mask = np.all((points >= (-0.2, -0.5, -1)) & (points <= (0.2, 0.5, 1)), axis=1)
        first_pcd = first_pcd.select_by_index(np.where(mask)[0])

        # Ensure point cloud has normals, otherwise visualization looks flat
        if not first_pcd.has_normals():
            first_pcd.estimate_normals()

        # Create a display point cloud that will be updated for each frame
        display_pcd = copy.deepcopy(first_pcd)
        vis.add_geometry(display_pcd)

        # Set rendering options
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])  # White background
        opt.point_size = 2.0  # Adjust point size

        # Auto-position the camera for a good view
        # vis.get_view_control().rotate(0.0, -180.0)  # Rotate around vertical axis
        # Set a reasonable viewpoint distance and field of view
        view_control = vis.get_view_control()
        view_control.set_zoom(0.5)

        # Render each PLY file to an image
        for i, ply_file in enumerate(
            tqdm(ply_files, desc="Rendering frames", unit="frame")
        ):

            # Create output image path
            output_image = os.path.join(temp_dir, f"frame_{i:04d}.png")

            # Load the current point cloud
            current_pcd = o3d.io.read_point_cloud(ply_file)
            points = np.asarray(current_pcd.points)

            # Filter points within ROI
            mask = np.all(
                (points >= (-0.2, -0.5, -1)) & (points <= (0.2, 0.5, 1)), axis=1
            )
            current_pcd = current_pcd.select_by_index(np.where(mask)[0])

            # Ensure point cloud has normals
            if not current_pcd.has_normals():
                current_pcd.estimate_normals()

            # Update the display point cloud with new points and colors
            display_pcd.points = current_pcd.points
            display_pcd.colors = current_pcd.colors
            display_pcd.normals = current_pcd.normals

            # Update the geometry in the visualizer
            vis.update_geometry(display_pcd)
            vis.poll_events()
            vis.update_renderer()

            # Capture image and save
            image = vis.capture_screen_float_buffer(do_render=True)
            image_np = np.asarray(image)
            image_np = (image_np * 255).astype(np.uint8)

            # Convert from RGB float array to PIL Image and save
            img = Image.fromarray(image_np)
            img.save(output_image)

            # Add to list of images
            image_files.append(output_image)

        # Clean up visualizer
        vis.destroy_window()

        # Create GIF from images
        print(f"Creating GIF animation...")

        # Calculate duration between frames in milliseconds
        duration = int(1000 / framerate)

        # Read all images and create GIF
        images = []
        for image_file in image_files:
            images.append(imageio.imread(image_file))

        # Save as GIF
        imageio.mimsave(output_gif, images, duration=duration, loop=0)

        print(f"GIF animation saved to {output_gif}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert PLY files to GIF animation")
    parser.add_argument(
        "--i",
        type=str,
        help="Folder containing PLY files",
        default="F:\LiDAR Videos\RobotCV\\hand_3\\ply",
    )
    parser.add_argument(
        "--o",
        type=str,
        help="Output GIF file",
        default="F:\LiDAR Videos\RobotCV\\hand_3\\raw.gif",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=30,
        help="Frame rate of the GIF (default: 10 fps)",
    )
    parser.add_argument(
        "--width", type=int, default=400, help="Width of the output GIF (default: 800)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=300,
        help="Height of the output GIF (default: 600)",
    )

    args = parser.parse_args()

    # Convert PLY to GIF
    ply_to_gif(args.i, args.o, args.framerate, args.width, args.height)


if __name__ == "__main__":
    main()
