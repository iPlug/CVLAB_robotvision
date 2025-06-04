import os
import sys
import subprocess
import glob
from tqdm import tqdm
from utils.load_env import load_env, get_env

# Load and validate environment variables
try:
    load_env('local')
except (FileNotFoundError, ValueError) as e:
    print(f"Environment configuration error: {e}")
    sys.exit(1)

# Get root folder from environment
root_folder = get_env("BAG_FILE_ROOT_FOLDER")
if not os.path.exists(root_folder):
    print(f"Error: BAG_FILE_ROOT_FOLDER '{root_folder}' does not exist")
    sys.exit(1)

# Change to root directory
os.chdir(root_folder)
bag_files = glob.glob("*.bag")

# Use tqdm for a progress bar
for bag_file in tqdm(bag_files, desc="Processing bag files"):
    # Extract the filename without extension to use as folder name
    file_name = os.path.splitext(bag_file)[0]

    # Create the directory structure
    output_dir = os.path.join(root_folder, file_name)
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
