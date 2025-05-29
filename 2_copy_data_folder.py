import os
import shutil
import glob
from utils.load_env import load_env

load_env('local')

# Source folders list
objects = [
    "box_1",
    "bottle_1",
    "hand_1",
    "can_1",
    "box_2",
    "box_3",
    "can_2",
    "can_3",
    "hand_2",
    "hand_3",
    "bottle_2",
    "bottle_3",
]

# Destination root folder
dest_root = os.getenv('DEST_COPY_ROOT_FOLDER')

# Maximum files per source folder
MAX_FILES_PER_SOURCE = 450

# Create destination root if it doesn't exist
if not os.path.exists(dest_root):
    os.makedirs(dest_root)
    print(f"Created destination root: {dest_root}")

# Counter for statistics
total_files = 0
total_copied = 0

# Process each folder
for object_3d in objects:
    # Create object path according to env
    object_3d_path = os.path.join(os.getenv('BAG_FILE_ROOT_FOLDER'), object_3d)

    # Source path with filtered_pcd
    source_path = os.path.join(object_3d_path, "filtered_ply")

    # Destination path without filtered_pcd
    dest_path = os.path.join(dest_root, object_3d)

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
        print(f"Created directory: {dest_path}")

    # Counter for files copied from this source folder
    files_copied_from_source = 0

    # Check if source path exists
    if os.path.exists(source_path):
        # Find all .ply files in the source path
        ply_files = glob.glob(os.path.join(source_path, "*.ply"))
        
        # Count total files found before applying limit
        total_files += len(ply_files)
        
        # Apply limit of MAX_FILES_PER_SOURCE
        if len(ply_files) > MAX_FILES_PER_SOURCE:
            print(f"Found {len(ply_files)} files in {source_path}, limiting to {MAX_FILES_PER_SOURCE}")
            ply_files = ply_files[:MAX_FILES_PER_SOURCE]

        # Copy each file (now limited to MAX_FILES_PER_SOURCE)
        for file_path in ply_files:
            file_name = os.path.basename(file_path)
            dest_file = os.path.join(dest_path, file_name)

            try:
                shutil.copy2(file_path, dest_file)
                print(f"Copied: {file_path} -> {dest_file}")
                total_copied += 1
                files_copied_from_source += 1
            except Exception as e:
                print(f"Error copying {file_path}: {e}")
                
        print(f"Copied {files_copied_from_source} files from {source_path}")
    else:
        print(f"Warning: Source path not found: {source_path}")

# Print summary
print("\nCopy operation completed!")
print(f"Total files found: {total_files}")
print(f"Total files copied: {total_copied}")


# Function to organize files into train/test structure
def organize_train_test_split(source_folders, target_root, max_files_per_source=450):
    # Create train and test directories
    train_dir = os.path.join(target_root, "train")
    test_dir = os.path.join(target_root, "test")

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Statistics
    train_files_count = 0
    test_files_count = 0

    # Process each folder
    for folder in source_folders:
        # Get folder name and extract class name (e.g., "box" from "box_1")
        folder_name = os.path.basename(os.path.normpath(folder))
        parts = folder_name.split("_")
        class_name = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        # Determine if it's train or test
        if suffix == "3":
            # Test data
            target_class_dir = os.path.join(test_dir, class_name)
            is_train = False
        else:
            # Train data (_1 and _2)
            target_class_dir = os.path.join(train_dir, class_name)
            is_train = True

        # Create class directory if it doesn't exist
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

        # Source path with filtered_ply
        source_path = os.path.join(folder, "filtered_ply")

        # Files copied from this source counter
        files_copied_from_source = 0

        # Copy files if source path exists
        if os.path.exists(source_path):
            # Find all .ply files
            ply_files = glob.glob(os.path.join(source_path, "*.ply"))
            
            # Apply limit of max_files_per_source
            if len(ply_files) > max_files_per_source:
                print(f"Found {len(ply_files)} files in {source_path}, limiting to {max_files_per_source}")
                ply_files = ply_files[:max_files_per_source]

            # Copy each file (now limited)
            for file_path in ply_files:
                file_name = os.path.basename(file_path)
                target_file = os.path.join(target_class_dir, file_name)

                try:
                    shutil.copy2(file_path, target_file)
                    files_copied_from_source += 1
                    
                    if is_train:
                        train_files_count += 1
                    else:
                        test_files_count += 1
                except Exception as e:
                    print(f"Error copying {file_path}: {e}")
            
            print(f"Copied {files_copied_from_source} files from {source_path} to {target_class_dir}")

    # Return statistics
    return train_files_count, test_files_count


target_folder = os.getenv('DATASET_ROOT_FOLDER')

# Create target folder if it doesn't exist
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
    print(f"Created target directory: {target_folder}")

# Organize files with maximum files per source limit
print(f"\nOrganizing files into train/test structure at: {target_folder}")
train_count, test_count = organize_train_test_split(objects, target_folder, MAX_FILES_PER_SOURCE)

# Print summary
print("\nTrain/Test split completed!")
print(f"Train files copied: {train_count}")
print(f"Test files copied: {test_count}")
print(f"Total files in new structure: {train_count + test_count}")