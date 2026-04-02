# Filename: infer.py
# Author: AK Wash
# Created: 2026-03-10

# Description: inference pipeline for running the trained model on
# new LiDAR point clouds. Performs following steps:
# 1. Load ply/csv point cloud file, or accept (N,3) numpy XYZ
# 2. Preprocess point cloud into model input format
# 3. Construct model
# 4. Load trained model weights
# 5. Run forward inference
# 6. Produce predicted labels for each point
# 7. Visualize predicted labels in Open3D

# Outputs:
#   points -> input point coords
#   true_labels -> ground truth labels (if available)
#   pred_labels -> predicted semantic classes

from pathlib import Path
from typing import Any

import numpy as np
import torch

from rover_navigation.preprocessing.load_ply import load_cloudcompare_ply
from rover_navigation.util.config_loader import load_yaml
from rover_navigation.perception.randlanet_model import RandLANet
from rover_navigation.perception.dataset import (
    _normalize_features,
    _normalize_points,
    _knn_indices,
)

from rover_navigation.rover.csv_loader import load_csv_point_cloud

# build paths relative to this file so code works no matter where it is run from
ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "config" / "dataset.yaml"
TRAINING_CONFIG = ROOT / "config" / "training.yaml"

def load_point_cloud_for_inference(
        input_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    either load in the ply or csv
    - ply is the annotated training point clouds
    - csv is the in situ collected point cloud

    :param input_path: the path to the file
    :return points: (N,3)
    :return features: (N,F)
    :return labels: (N,) <- there is a dummy label of -1 for the csv
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    suffix = input_path.suffix.lower()

    if suffix == ".ply":
        points, features, _ = load_cloudcompare_ply(input_path)
        labels = np.full(points.shape[0],-1, dtype=np.int64)
        return points, features, labels
    
    if suffix == '.csv':
        points, features = load_csv_point_cloud(input_path)
        labels = np.full(points.shape[0],-1, dtype=np.int64)
        return points, features, labels
    
    raise ValueError(
        f"Unsupported input file type: {input_path.suffix}."
        f"Expected .ply or .csv"
    )

def build_inference_batch_from_xyz(
    points_xyz: np.ndarray,
    features: np.ndarray | None = None,
    dataset_config_path: str | Path = DATASET_CONFIG,
    log_label: str = "numpy xyz",
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Build a RandLA-Net inference batch from in-memory point coordinates.

    :param points_xyz: (N, 3) array of XYZ in meters (same units as file-based loads).
    :param features: optional (N, F) per-point features. If None, uses empty features (N, 0)
        like CSV inference.
    :param dataset_config_path: dataset YAML for preprocessing/sampling.
    :param log_label: string printed when building the batch (for debugging).
    :return: batch dict, sampled_idx, vis_points (subsampled raw xyz for visualization).
    """
    points = np.asarray(points_xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N, 3), got shape {points.shape}")

    if features is None:
        features = np.zeros((points.shape[0], 0), dtype=np.float32)
    else:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != points.shape[0]:
            raise ValueError(
                f"features must be (N, F) with N={points.shape[0]}, got {features.shape}"
            )

    labels = np.full(points.shape[0], -1, dtype=np.int64)

    return _build_inference_batch_from_arrays(
        points=points,
        features=features,
        labels=labels,
        dataset_config_path=dataset_config_path,
        log_label=log_label,
    )


def _build_inference_batch_from_arrays(
    points: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    dataset_config_path: str | Path,
    log_label: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Shared batch construction from aligned points, features, and labels arrays.
    """
    dataset_cfg = load_yaml(dataset_config_path)

    prep_cfg = dataset_cfg["preprocessing"]
    sampling_cfg = dataset_cfg["sampling"]

    # processing param from config
    num_points = int(prep_cfg["num_points"])
    center_cloud = bool(prep_cfg["center_cloud"])
    normalize_xyz = bool(prep_cfg["normalize_xyz"])
    normalize_features = bool(prep_cfg["normalize_features"])

    # sampling parameters
    k_n = int(sampling_cfg["k_n"])
    num_layers = int(sampling_cfg["num_layers"])
    sub_sampling_ratio = list(sampling_cfg["sub_sampling_ratio"])

    assert points.ndim == 2 and points.shape[1] == 3, "Points must be (N, 3)"
    assert features.ndim == 2, "Features must be (N, F)"
    assert labels.ndim == 1 and labels.shape[0] == points.shape[0], \
        "Labels must be (N,)"

    # keep a copy of sampled points before normalization for nicer visualization
    if points.shape[0] >= num_points:
        sampled_idx = np.random.choice(points.shape[0], num_points, replace=False)
    else:
        extra = np.random.choice(points.shape[0], num_points - points.shape[0], replace=True)
        sampled_idx = np.concatenate([np.arange(points.shape[0]), extra], axis=0)

    vis_points = points[sampled_idx].astype(np.float32)

    # subsample points, features, and labels
    points = points[sampled_idx].astype(np.float32)
    features = features[sampled_idx].astype(np.float32)
    labels = labels[sampled_idx].astype(np.int64)

    # center and normalize as needed
    if center_cloud:
        points = points - np.mean(points, axis=0, keepdims=True)

    if normalize_xyz:
        points = _normalize_points(points)

    if features.shape[1] > 0:
        if normalize_features:
            features = _normalize_features(features)
        else:
            if np.max(features) > 1.0:
                features = features / 255.0

    # input matrix shape (N, 3) -> xyz
    input_features = np.concatenate([points, features], axis=1).astype(np.float32)

    # one list per layer (hierarchical RandLA-Net structure)
    xyz_layers = []
    neigh_idx_layers = []
    sub_idx_layers = []
    interp_idx_layers = []

    # initialize
    current_xyz = points.copy()

    # go over model layers for neighbor and subsampling indices
    for layer_idx in range(num_layers):
        xyz_layers.append(torch.tensor(current_xyz[None, ...], dtype=torch.float32))

        neigh_idx = _knn_indices(current_xyz, current_xyz, k_n)
        neigh_idx_layers.append(torch.tensor(neigh_idx[None, ...], dtype=torch.long))

        # downsampling
        ratio = sub_sampling_ratio[layer_idx]
        num_sub = max(1, current_xyz.shape[0] // ratio)

        sampled_indices = np.random.choice(current_xyz.shape[0], num_sub, replace=False)
        sampled_xyz = current_xyz[sampled_indices]

        sub_idx = neigh_idx[sampled_indices]
        sub_idx_layers.append(torch.tensor(sub_idx[None, ...], dtype=torch.long))

        # interpolation for upsampling
        interp_idx = _knn_indices(current_xyz, sampled_xyz, 1)
        interp_idx_layers.append(torch.tensor(interp_idx[None, ...], dtype=torch.long))

        current_xyz = sampled_xyz

    print("Running inference on:", log_label)

    # build batch dictionary
    batch = {
        "features": torch.tensor(input_features[None, ...], dtype=torch.float32),
        "labels": torch.tensor(labels[None, ...], dtype=torch.long),
        "xyz": xyz_layers,
        "neigh_idx": neigh_idx_layers,
        "sub_idx": sub_idx_layers,
        "interp_idx": interp_idx_layers,
    }

    return batch, sampled_idx, vis_points


# build input batch for inference
# input: ply file path, dataset config
# output: dictionary with: features, labels, xyz, neigh_idx, sub_idx, interp_idx
def build_inference_batch(
    ply_path: str,
    dataset_config_path: str | Path = DATASET_CONFIG,
):
    points, features, labels = load_point_cloud_for_inference(ply_path)
    return _build_inference_batch_from_arrays(
        points=points,
        features=features,
        labels=labels,
        dataset_config_path=dataset_config_path,
        log_label=str(ply_path),
    )


# helper function for moving the batch tensors to the correct device
def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, list):
            moved[key] = [v.to(device) for v in value]
        else:
            moved[key] = value.to(device)
    return moved


def _run_inference_from_batch(
    batch: dict,
    vis_points: np.ndarray,
    model_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load weights, run forward pass, return vis_points, true_labels, pred_labels."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_labels = batch["labels"].squeeze(0).cpu().numpy()

    model = RandLANet(
        dataset_config_path=DATASET_CONFIG,
        training_config_path=TRAINING_CONFIG,
    ).to(device)

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            f"Train the model first or pass the correct checkpoint path."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        logits = model(batch)
        pred = torch.argmax(logits, dim=-1)

    pred = pred.squeeze(0).cpu().numpy()
    return vis_points, true_labels, pred


# function to run inference on a ply file using the trained model
def run_inference(
    ply_path: str,  # path to input ply file
    model_path: str | Path = Path("checkpoints") / "randlanet.pt",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch, _sampled_idx, vis_points = build_inference_batch(ply_path)
    return _run_inference_from_batch(batch, vis_points, model_path)


def run_inference_from_xyz(
    points_xyz: np.ndarray,
    features: np.ndarray | None = None,
    model_path: str | Path = Path("checkpoints") / "randlanet.pt",
    dataset_config_path: str | Path = DATASET_CONFIG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run semantic segmentation on an in-memory point cloud (N, 3) XYZ.

    Same preprocessing and model as :func:`run_inference`, without reading a file.
    Optional ``features`` is (N, F); if omitted, empty features (N, 0) are used like CSV loads.
    """
    batch, _sampled_idx, vis_points = build_inference_batch_from_xyz(
        points_xyz,
        features=features,
        dataset_config_path=dataset_config_path,
    )
    return _run_inference_from_batch(batch, vis_points, model_path)


# helper function to create a colored Open3D point cloud
def create_colored_point_cloud(
    points: np.ndarray,
    labels: np.ndarray,
) -> Any:
    """
    Build an Open3D point cloud and color it by class label.

    class 0 -> light gray
    class 1 -> red
    unknown -> blue
    """
    import open3d as o3d  # optional dependency for visualization only

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {points.shape}")

    if labels.ndim != 1:
        raise ValueError(f"Expected labels shape (N,), got {labels.shape}")

    if len(points) != len(labels):
        raise ValueError(f"Points/labels length mismatch: {len(points)} vs {len(labels)}")

    colors = np.zeros((points.shape[0], 3), dtype=np.float32)

    # default colors
    colors[labels == 0] = np.array([0.75, 0.75, 0.75], dtype=np.float32)  # non-obstacle
    colors[labels == 1] = np.array([1.0, 0.0, 0.0], dtype=np.float32)     # obstacle
    colors[labels < 0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)      # unknown

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


# visualize predicted labels
def visualize_predictions(
    points: np.ndarray,
    pred_labels: np.ndarray,
    window_name: str = "Predicted Labels",
) -> None:
    import open3d as o3d

    pcd = create_colored_point_cloud(points, pred_labels)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


# visualize ground truth labels
def visualize_ground_truth(
    points: np.ndarray,
    true_labels: np.ndarray,
    window_name: str = "Ground Truth Labels",
) -> None:
    import open3d as o3d

    pcd = create_colored_point_cloud(points, true_labels)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)


# usage example
if __name__ == "__main__":
    dataset_cfg = load_yaml(DATASET_CONFIG)
    ply_path = dataset_cfg["paths"]["test_file"]

    points, labels, pred = run_inference(ply_path)
    print("Unique true classes:", np.unique(labels))
    print("True label counts:", np.unique(labels, return_counts=True))
    print("Unique predicted classes:", np.unique(pred))
    print("Pred label counts:", np.unique(pred, return_counts=True))

    print("Points shape:", points.shape)
    print("True labels shape:", labels.shape)
    print("Pred labels shape:", pred.shape)
    print("Unique predicted classes:", np.unique(pred))

    # show prediction result
    visualize_predictions(points, pred)

    # optionally also show ground truth
    visualize_ground_truth(points, labels)