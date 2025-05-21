"""
Modified test script that integrates subspace-based classification with flexible dataset support
"""

import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib

# Import subspace utilities
from utils.subspace_utils import (
    build_class_subspaces,
    extract_features_batch,
    subspace_classify,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser("Testing with Subspace Classification")
    parser.add_argument(
        "--use_cpu", action="store_true", default=False, help="use cpu mode"
    )
    parser.add_argument("--gpu", type=str, default="0", help="specify gpu device")
    parser.add_argument(
        "--batch_size", type=int, default=24, help="batch size in testing"
    )
    
    parser.add_argument("--num_point", type=int, default=1024, help="Point Number")
    parser.add_argument("--log_dir", type=str, default="pointnet_cls", help="Experiment root")
    parser.add_argument(
        "--dataset",
        type=str,
        default="simple",
        choices=["simple"],
        help="Dataset name",
    )

    parser.add_argument(
        "--use_normals", action="store_true", default=False, help="use normals"
    )

    parser.add_argument(
        "--use_uniform_sample",
        action="store_true",
        default=False,
        help="use uniform sampiling",
    )

    parser.add_argument(
        "--train_features_path",
        type=str,
        default=None,
        help="Path to pre-extracted training features",
    )

    parser.add_argument(
        "--subspace_dim",
        type=int,
        default=50,
        help="Dimension of the subspace for each class",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="subspace",
        choices=["subspace"],
        help="Classification method: subspace only",
    )
    
    parser.add_argument(
        "--partial",
        action="store_true",
        default=False,
        help="Use partial point clouds",
    )

    parser.add_argument(
        "--rotate",
        action="store_true",
        default=False,
        help="Use random rotation point clouds",
    )
    
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs to average results over",
    )
    return parser.parse_args()


def load_dataset(args):
    """Load the appropriate dataset based on the dataset type"""
    # Create dataset arguments to maintain compatibility
    args.data_path = "D:\LAB\DATASETS\\robot_simple"
    args.num_category = 4

    dataset_args = argparse.Namespace(
        num_point=args.num_point,
        use_uniform_sample=args.use_uniform_sample,
        use_normals=args.use_normals,
        num_category=args.num_category,
    )

    test_dataset = None

    try:
        from SimpleDataLoader import SimpleDataLoader

        test_dataset = SimpleDataLoader(
            root=args.data_path,
            args=dataset_args,
            split="test",
            samples_per_file=1,
        )

    except ImportError:
        print("Error: Error")
        sys.exit(1)

    return test_dataset


def test_subspace(model, loader, V, class_indices, device="cuda"):
    """Standard Subspace-based classification testing"""
    all_preds = []
    all_targets = []

    for batch_points, batch_targets in tqdm(loader, desc="Testing Subspace"):
        # Extract features for this batch
        features = extract_features_batch(model, batch_points, device)
        features_np = features.numpy()

        # Classify using subspace method
        batch_preds = subspace_classify(features_np, V, class_indices)

        # Save predictions and targets for accuracy calculation
        all_preds.append(batch_preds)
        all_targets.append(batch_targets.numpy())

    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate accuracy
    instance_acc = np.mean(all_preds == all_targets)

    # Calculate class-specific accuracy
    class_accs = []
    for class_idx in np.unique(all_targets):
        class_mask = all_targets == class_idx
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_preds[class_mask] == all_targets[class_mask])
            class_accs.append(class_acc)

    class_acc = np.mean(class_accs)

    return instance_acc, class_acc, all_preds, all_targets


def get_default_train_features_path(args):
    """Get default path for training features based on dataset and model"""
    model_dir = os.path.basename(args.log_dir.strip("/"))
    if args.partial:
        return f"features/{model_dir}/{args.dataset}_train_features_combined.npy"
    else:
        return f"features/{model_dir}/{args.dataset}_train_features_full.npy"


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    """HYPER PARAMETER"""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    """CREATE DIR"""
    experiment_dir = "log/classification/" + args.log_dir

    """LOG"""
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("%s/eval_subspace.txt" % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string("PARAMETER ...")
    log_string(str(args))

    """DATA LOADING"""
    log_string("Load dataset ...")
    test_dataset = load_dataset(args)
    log_string(f"Data path: {args.data_path}")

    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    log_string("Dataset loaded.")

    """MODEL LOADING"""
    log_string("Load model ...")
    num_class = 40
    model_path = os.path.join(experiment_dir + "/logs")
    model_name = os.listdir(model_path)[0].split(".")[0]
    log_string(f"Model name: {model_name}, Path: {model_path}")
    model = importlib.import_module(model_name)
    log_string("Import module done.")
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    log_string("Model loaded.")

    # Determine device
    device = "cpu" if args.use_cpu else "cuda"
    classifier = classifier.to(device)
    if not args.use_cpu:
        log_string("Model moved to GPU.")

    checkpoint_path = str(experiment_dir) + "/checkpoints/best_model.pth"
    log_string(f"Checkpoint path: {checkpoint_path}")
    # Load checkpoint to appropriate device
    if args.use_cpu:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        classifier.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Try direct loading
        classifier.load_state_dict(checkpoint)

    log_string("Checkpoint loaded.")

    """LOAD TRAINING FEATURES FOR SUBSPACE"""
    # Use default path if none provided
    if args.train_features_path is None:
        args.train_features_path = get_default_train_features_path(args)

    log_string(f"Loading training features from {args.train_features_path}...")
    train_features = np.load(args.train_features_path, allow_pickle=True).item()

    log_string("Building class subspaces...")
    V, class_indices = build_class_subspaces(train_features, args.subspace_dim)
    log_string(f"Built subspaces with dimension {args.subspace_dim}")

    """TESTING"""
    results = {}
    all_methods = ["subspace"]

    with torch.no_grad():
        log_string("Start testing ...")

        # Standard Subspace testing
        all_methods.append("subspace")
        log_string("Testing with Standard Subspace method...")
        instance_acc, class_acc, all_preds, all_targets = test_subspace(
            classifier.eval(), testDataLoader, V, class_indices, device=device
        )

        log_string(
            "Subspace Test Instance Accuracy: %f, Class Accuracy: %f"
            % (instance_acc, class_acc)
        )

        results["subspace"] = {
            "instance_acc": instance_acc,
            "class_acc": class_acc,
            "predictions": all_preds,
            "targets": all_targets,
        }

        log_string("Testing finished.")

        return results


if __name__ == "__main__":
    args = parse_args()

    all_runs_results = {
        "subspace": {"instance_acc": [], "class_acc": []},
    }

    # Run multiple times and collect results
    for i in range(args.runs):
        print(f"Run {i+1}/{args.runs}")
        run_results = main(args)

        # Collect results from each run
        for method, results in run_results.items():
            if method in all_runs_results:
                all_runs_results[method]["instance_acc"].append(results["instance_acc"])
                all_runs_results[method]["class_acc"].append(results["class_acc"])

    # Calculate and display average results
    print("\n===== AVERAGE RESULTS ACROSS ALL RUNS =====")
    for method, results in all_runs_results.items():
        if results["instance_acc"]:  # Check if method was used
            avg_instance_acc = np.mean(results["instance_acc"])
            avg_class_acc = np.mean(results["class_acc"])
            std_instance_acc = np.std(results["instance_acc"])
            std_class_acc = np.std(results["class_acc"])

            print(f"{method.capitalize()} ({args.runs} runs):")
            print(
                f"  Instance Accuracy: {avg_instance_acc:.4f} ± {std_instance_acc:.4f}"
            )
            print(f"  Class Accuracy: {avg_class_acc:.4f} ± {std_class_acc:.4f}")