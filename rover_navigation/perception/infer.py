# Filename: infer.py
# Author: AK Wash
# Created: 2026-03-10

# Description: inference pipeline for running the trained model on 
# new LiDAR point clouds. Performs following steps:
# 1. Load ply point cloud file
# 2. Preprocess point cloud into model input format
# 3. Construct model
# 4. Load trained model weights
# 5. Run forward inference
# 6. Produce predicted labels for each point

# Outputs:
#   points -> input point coords
#   true_labels -> ground truth labels (if available)
#   pred_labels -> predicted semantic classes


import numpy as np
import torch

from rover_navigation.preprocessing.load_ply import load_cloudcompare_ply
from rover_navigation.util.config_loader import load_yaml
from rover_navigation.perception.randlanet_model import RandLANet
from rover_navigation.perception.dataset import _normalize_features, _normalize_points, _knn_indices

# build input batch for inference
# input: ply file path, dataset config
# output: dictionary with: features, labels, xyz, neigh_idx, sub_idx, interp_idx
def build_inference_batch(
    ply_path: str,
    dataset_config_path: str = "config/dataset.yaml",
) -> dict:
    dataset_cfg = load_yaml(dataset_config_path)

    prep_cfg = dataset_cfg["preprocessing"]
    sampling_cfg = dataset_cfg["sampling"]
    label_col = dataset_cfg["point_cloud"]["label_scalar_index"]

    # processing param from config
    num_points = int(prep_cfg["num_points"])
    center_cloud = bool(prep_cfg["center_cloud"])
    normalize_xyz = bool(prep_cfg["normalize_xyz"])
    normalize_features = bool(prep_cfg["normalize_features"])

    # sampling parameters 
    k_n = int(sampling_cfg["k_n"])
    num_layers = int(sampling_cfg["num_layers"])
    sub_sampling_ratio = list(sampling_cfg["sub_sampling_ratio"])
    
    # load point cloud data
    points, scalars = load_cloudcompare_ply(ply_path)

    # seperate labels and features
    if scalars.size == 0 or label_col >= scalars.shape[1]:
        labels = np.full(points.shape[0],-1,dtype=np.int64)
        features = scalars.astype(np.float32) if scalars.size > 0 else np.zeros((points.shape[0],0), dtype=np.float32)
    else:
        labels = scalars[:,label_col].astype(np.int64)
        feature_cols = [i for i in range(scalars.shape[1]) if i != label_col]
        features = scalars[:,feature_cols].astype(np.float32)

    # sample points to a fixed number
    if points.shape[0] >= num_points:
        sampled_idx = np.random.choice(points.shape[0], num_points, replace=False)
    else:
        extra = np.random.choice(points.shape[0], num_points - points.shape[0], replace=True)
        sampled_idx = np.concatenate([np.arange(points.shape[0]), extra], axis=0)

    # subsample points, features, and labels
    points = points[sampled_idx].astype(np.float32)
    features = features[sampled_idx].astype(np.float32)
    labels = labels[sampled_idx].astype(np.int64)

    # center and normalize as needed
    if center_cloud:
        points = points - np.mean(points, axis=0, keepdims=True)

    if normalize_xyz:
        points = _normalize_points(points)

    if normalize_features:
        features = _normalize_features(features)

    # input matrix shape (N, 3+F) <- F is number of features per point
    input_features = np.concatenate([points, features], axis=1).astype(np.float32)

    # one list per layer (hierarchical RandLA-Net structure)
    xyz_layers = []
    neigh_idx_layers = []
    sub_idx_layers = []
    interp_idx_layers = []

    # intialize
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

    # build batch directory
    batch = {
        "features": torch.tensor(input_features[None, ...], dtype=torch.float32),
        "labels": torch.tensor(labels[None, ...], dtype=torch.long),
        "xyz": xyz_layers,
        "neigh_idx": neigh_idx_layers,
        "sub_idx": sub_idx_layers,
        "interp_idx": interp_idx_layers,
    }
    return batch, sampled_idx, points

# helper function for moving the batch tensors to the correct device
def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, list):
            moved[key] = [v.to(device) for v in value]
        else:
            moved[key] = value.to(device)
    return moved

# function to run inference on a ply file using the trained model
def run_inference(
    ply_path: str, # path to input ply file
    model_path: str = "checkpoints/randlanet.pt",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the batch
    batch = build_inference_batch(ply_path)
    points = batch["xyz"][0].squeeze(0).cpu().numpy()
    true_labels = batch["labels"].squeeze(0).cpu().numpy()

    # load the model
    model = RandLANet().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # move the batch to the device
    batch = move_batch_to_device(batch, device)

    # run inference 
    with torch.no_grad():
        logits = model(batch)                   # (1, N, num_classes)
        pred = torch.argmax(logits, dim=-1)     # (1, N)

    # convert the predictions to numpy
    pred = pred.squeeze(0).cpu().numpy()
    return points, true_labels, pred

# usage example
if __name__ == "__main__":
    dataset_cfg = load_yaml("config/dataset.yaml")
    ply_path = dataset_cfg["paths"]["test_file"]

    points, labels, pred = run_inference(ply_path)

    print("Points shape:", points.shape)
    print("True labels shape:", labels.shape)
    print("Pred labels shape:", pred.shape)
    print("Unique predicted classes:", np.unique(pred))