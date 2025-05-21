import torch
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import os
import sys
import argparse

# Import paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))

# Import PointNet models
from models import pointnet2_cls_ssg, pointnet_cls


def extract_features_by_class(
    model, dataset, batch_size=32, device="cpu", pc_type="full"
):
    """
    Extract features for all point clouds and organize them by class

    Args:
        model: Trained PointNet/PointNet++ model
        dataset: Dataset loader instance
        batch_size: Batch size for processing
        device: Device to run inference on ('cuda' or 'cpu')
        pc_type: Point cloud type ("full", "partial", or "combined")

    Returns:
        dict: Dictionary mapping class indices to feature matrices
    """
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Dictionary to store features by class
    features_by_class = defaultdict(list)

    # Track processed samples
    processed_samples = 0

    with torch.no_grad():
        for batch_points, batch_labels in tqdm(
            dataloader, desc=f"Extracting {pc_type} features"
        ):
            # Prepare points for model
            points = batch_points.float().to(device)
            points = points.transpose(2, 1)  # Change to [B, 3, N] format

            # Extract features
            features = model.extract_features(points, feature_level="global")[
                "global_features"
            ]

            # Store features by class
            for feat, label in zip(features, batch_labels):
                features_by_class[label.item()].append(feat.cpu().numpy())

            processed_samples += len(batch_points)

        print(
            f"Processed {processed_samples} samples from {len(dataset)} total samples"
        )

    # Convert lists to matrices
    feature_matrices = {}
    for class_idx, feature_list in features_by_class.items():
        feature_matrices[class_idx] = np.vstack(feature_list)

    return feature_matrices

def load_dataset(args, partial=False):
    """
    Load the appropriate dataset based on the dataset type and partial flag

    Args:
        args: Command line arguments
        partial: Whether to load partial point clouds

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Create dataset arguments
    dataset_args = argparse.Namespace(
        num_point=args.num_point,
        use_uniform_sample=args.use_uniform_sample,
        use_normals=args.use_normals,
        num_category=args.num_category,
    )

    train_dataset, test_dataset = None, None

    try:
        from SimpleDataLoader import SimpleDataLoader

        if args.extract_train:
            train_dataset = SimpleDataLoader(
                root=args.data_path,
                args=dataset_args,
                split="train",
                partial_pc=partial,
                cut_type=args.cut_type if partial else None,
                samples_per_file=args.samples_per_file,
            )

        if args.extract_test:
            test_dataset = SimpleDataLoader(
                root=args.data_path,
                args=dataset_args,
                split="test",
                partial_pc=partial,
                cut_type=args.cut_type if partial else None,
                samples_per_file=1,
            )

    except ImportError:
        print("Error: Error")
        sys.exit(1)

    return train_dataset, test_dataset


def main(args):
    """
    Main function for feature extraction
    """
    # Load datasets for full point clouds
    print("Loading datasets for full point clouds...")
    train_dataset_full, test_dataset_full = load_dataset(args, partial=False)

    # Load model based on model_dir argument
    if args.model_dir == "pointnet2_ssg_wo_normals":
        model = pointnet2_cls_ssg.get_model(
            num_class=40, normal_channel=args.use_normals
        )
        checkpoint_path = (
            args.checkpoint
            or "log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth"
        )
    else:  # default to pointnet
        model = pointnet_cls.get_model(k=40, normal_channel=args.use_normals)
        checkpoint_path = (
            args.checkpoint
            or "log/classification/pointnet_cls/checkpoints/best_model.pth"
        )

    print(f"Loading model checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Try direct loading
        model.load_state_dict(checkpoint)

    # Set device for extraction
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Extracting features on: {device}")

    # Extract features for full point clouds
    train_feature_matrices_full, test_feature_matrices_full = None, None

    if args.use_uniform_sample:
        add_name = "_fps_"
    else:
        add_name = "_"
    # Process train data if requested
    if args.extract_train:
        # Extract features from full point clouds
        if train_dataset_full is not None:
            print(f"Extracting features from {args.dataset} full training set...")
            train_feature_matrices_full = extract_features_by_class(
                model,
                train_dataset_full,
                batch_size=args.batch_size,
                device=device,
                pc_type="full",
            )

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(
                args.train_output_full
                or f"features/{args.model_dir}/{args.dataset}{add_name}train_features_full.npy"
            )
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            output_path = (
                args.train_output_full
                or f"features/{args.model_dir}/{args.dataset}{add_name}train_features_full.npy"
            )
            np.save(output_path, train_feature_matrices_full)
            print(f"Saved full train features to: {output_path}")

            # Print feature matrix shapes for each class
            for class_idx, features in train_feature_matrices_full.items():
                class_name = (
                    train_dataset_full.cat[class_idx]
                    if hasattr(train_dataset_full, "cat")
                    else f"Class {class_idx}"
                )
                print(
                    f"Full - Class {class_idx} ({class_name}): Feature matrix shape {features.shape}"
                )

    # Process test data if requested
    if args.extract_test:
        # Extract features from full point clouds
        if test_dataset_full is not None:
            print(f"Extracting features from {args.dataset} full test set...")
            test_feature_matrices_full = extract_features_by_class(
                model,
                test_dataset_full,
                batch_size=args.batch_size,
                device=device,
                pc_type="full",
            )

            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(
                args.test_output_full
                or f"features/{args.model_dir}/{args.dataset}{add_name}test_features_full.npy"
            )
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            output_path = (
                args.test_output_full
                or f"features/{args.model_dir}/{args.dataset}{add_name}test_features_full.npy"
            )
            np.save(output_path, test_feature_matrices_full)
            print(f"Saved full test features to: {output_path}")

            # Print feature matrix shapes for each class
            for class_idx, features in test_feature_matrices_full.items():
                class_name = (
                    test_dataset_full.cat[class_idx]
                    if hasattr(test_dataset_full, "cat")
                    else f"Class {class_idx}"
                )
                print(
                    f"Full - Class {class_idx} ({class_name}): Feature matrix shape {features.shape}"
                )


    return {
        "train": {
            "full": train_feature_matrices_full,
        },
        "test": {
            "full": test_feature_matrices_full,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Point Cloud Feature Extraction")

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="simple",
        help="Type of dataset to process",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to dataset",
    )

    # General dataset parameters
    parser.add_argument(
        "--num_point", type=int, default=1024, help="Number of points per sample"
    )
    parser.add_argument(
        "--use_uniform_sample", action="store_true", help="Use uniform point sampling"
    )
    parser.add_argument(
        "--use_normals", action="store_true", help="Use normal features"
    )
    parser.add_argument(
        "--samples_per_file", type=int, default=1, help="Number of samples per file"
    )
    parser.add_argument(
        "--num_category",
        type=int,
        default=None,
        help="Number of categories (40 for ModelNet40, 10 for 3DNet Cat10, 10 for FAUST poses)",
    )

    # Extraction parameters
    parser.add_argument(
        "--extract_train",
        action="store_true",
        help="Extract features from training set",
    )
    parser.add_argument(
        "--extract_test", action="store_true", help="Extract features from test set"
    )
   
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda or cpu)",
    )

    # Model parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pointnet_cls",
        help="Type of model to use",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to model checkpoint"
    )

    # Output parameters for full features
    parser.add_argument(
        "--train_output_full",
        type=str,
        default=None,
        help="Output path for full train features",
    )
    parser.add_argument(
        "--test_output_full",
        type=str,
        default=None,
        help="Output path for full test features",
    )

    # Output parameters for partial features
    parser.add_argument(
        "--train_output_partial",
        type=str,
        default=None,
        help="Output path for partial train features",
    )
    parser.add_argument(
        "--test_output_partial",
        type=str,
        default=None,
        help="Output path for partial test features",
    )

    # Output parameters for combined features
    parser.add_argument(
        "--train_output_combined",
        type=str,
        default=None,
        help="Output path for combined train features",
    )
    parser.add_argument(
        "--test_output_combined",
        type=str,
        default=None,
        help="Output path for combined test features",
    )

    args = parser.parse_args()

    # Set defaults for extraction if none specified
    if not args.extract_train and not args.extract_test:
        args.extract_test = True
        args.extract_train = True

    # Set default num_category based on dataset if not specified
    
    args.num_category = 4
    args.data_path = "D:\LAB\DATASETS\\robot_simple"

    results = main(args)
    print("Feature extraction completed.")
