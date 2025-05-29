import os
import subprocess
import glob
from tqdm import tqdm
from utils.load_env import load_env

#Load env
load_env('local')

# Configure these variables as needed
root_folder = os.getenv("BAG_FILE_ROOT_FOLDER")  # Root folder containing bag files
output_parent_folder = root_folder  # Where to create the output folders (can be different from root_folder)

# Find all .bag files in the root folder
os.chdir(root_folder)
bag_files = glob.glob("*.bag")

# Use tqdm for a progress bar
for bag_file in tqdm(bag_files, desc="Processing bag files"):
    # Extract the filename without extension to use as folder name
    file_name = os.path.splitext(bag_file)[0]

    # Create the directory structure
    output_dir = os.path.join(output_parent_folder, file_name)
    ply_dir = os.path.join(output_dir, "ply")
    png_dir = os.path.join(output_dir, "png")

    # Check if the directories already exist and have content
    if os.path.exists(ply_dir) and os.path.exists(png_dir):
        # Check if there are files in both directories
        ply_files = os.listdir(ply_dir)
        png_files = os.listdir(png_dir)

        if ply_files and png_files:
            tqdm.write(
                f"Skipping {bag_file} - Output directories already exist with content"
            )
            continue

    # Create directories if they don't exist
    os.makedirs(ply_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    # Build the command
    command = [
        "rs-convert.exe",
        "-i",
        os.path.join(root_folder, bag_file),
        "-l",
        f"{ply_dir}\\",
        "-p",
        f"{png_dir}\\",
    ]

    # Execute the command
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("All bag files have been processed!")
