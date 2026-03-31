# Filename: dataset.py
# Author: AK Wash
# Created: 2026-03-10

# Description: PyTorch dataset for training the RandLa-Net model.
# Loads preprocessed point clouds stored as .npz files and prepares
# them for use by the nueral network.

# for each sample:
# 1. random point sampling
# 2. normalization and centering
# 3. feature reconstruction
# 4. nerighbohood computation
# 5. subsampling for RandLa-Net enconder

# features -> input feature tensor
# labels -> ground truth class labels
# xyz -> point coords
# neigh_idx -> neighborhood indices
# sub_idx -> subsampling index
# interp_idx -> interpolation idx

# used by: perception/train.py, perception/infer.py

import os
from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from rover_navigation.util.config_loader import load_yaml


# helper function to normalize point coords
# input: (N,3) point cloud
# output: (N,3) normalized point cloud
def _normalize_points(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0, keepdims=True)  # center the point cloud
    # scale the point cloud (furthest point has norm 1)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered = centered / scale
    return centered.astype(np.float32)


# helper function to normalize feature values
# input: (N,F) feature matrix
# output: (N,F) normalzied feature matrix
def _normalize_features(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=0, keepdims=True)  # mean for each feature column
    std = np.std(features, axis=0, keepdims=True)    # std for each feature column
    std[std < 1e-6] = 1.0
    return ((features - mean) / std).astype(np.float32)

# function for sample without replacement if enough points exist, otherwise sample all points
# and then duplicate to reach target. (Fixed size batches for training)
# input: total number of points, target number of points
# output: array of indices for sampling
def _random_sample_indices(num_points_total: int, num_points_target: int) -> np.ndarray:
    if num_points_total >= num_points_target:
        return np.random.choice(num_points_total, num_points_target, replace=False)
    extra = np.random.choice(
        num_points_total,
        num_points_target - num_points_total,
        replace=True,
    )
    base = np.arange(num_points_total)
    return np.concatenate([base, extra], axis=0)


# knn search to find the neighborhood indices for each point. Used for building the RandLa-Net hierarchy.
# input: query points (N,3), support points (M,3), number of neighbors (k)
# output: (N,k) array of neighbor indices for each query point
def _knn_indices(query_pts: np.ndarray, support_pts: np.ndarray, k: int) -> np.ndarray:
    if support_pts.shape[0] == 0:
        raise ValueError("support_pts is empty")

    k_eff = min(k, support_pts.shape[0])
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto")
    nbrs.fit(support_pts)
    _, indices = nbrs.kneighbors(query_pts)

    # pad if fewer than k neighbors exist at deep layers
    if k_eff < k:
        pad = np.repeat(indices[:, -1:], k - k_eff, axis=1)
        indices = np.concatenate([indices, pad], axis=1)

    return indices.astype(np.int64)


# initalize classdataset, loads .npz files and prepares batches for RandLa-Net training
class RandLANetDataset(Dataset):
    # store directory and load dataset config from the yaml file
    def __init__(self, data_dir: str, dataset_config_path: str = "rover_navigation/config/dataset.yaml"):
        self.data_dir = data_dir
        self.dataset_cfg = load_yaml(dataset_config_path)
        self.training_cfg = load_yaml("rover_navigation/config/training.yaml")

        # extract the preprocessing and sampling parameters
        prep_cfg = self.dataset_cfg["preprocessing"]
        sampling_cfg = self.dataset_cfg["sampling"]

        self.num_points = int(prep_cfg["num_points"])  # number of points to sample
        self.center_cloud = bool(prep_cfg["center_cloud"])  # decide if point cloud is centered at the origin
        self.normalize_xyz = bool(prep_cfg["normalize_xyz"])  # decide if points are normalized
        self.normalize_features = bool(prep_cfg["normalize_features"])  # decide if features are normalized

        self.k_n = int(sampling_cfg["k_n"])  # number of neighbors for the search
        self.num_layers = int(sampling_cfg["num_layers"])  # number of layers in RandLA-Net heirarchy
        self.sub_sampling_ratio = list(sampling_cfg["sub_sampling_ratio"])  # subsampling ratio of layers

        self.expected_input_dim = int(self.training_cfg["model"]["input_dim"])

        # collect all .npz files in directory
        self.files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        )

        # check if dataset is empty
        if not self.files:
            raise ValueError(f"No .npz files found in {data_dir}")

    # return number of samples in the dataset
    def __len__(self) -> int:
        return len(self.files)

    # build the RandLa-Net hierarchy by neighborhood indices, subsampling, and interpolation indices for the layers
    def _build_hierarchy(
        self, xyz: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        # lists for per-layer data
        xyz_layers: list[np.ndarray] = []
        neigh_idx_layers: list[np.ndarray] = []
        sub_idx_layers: list[np.ndarray] = []
        interp_idx_layers: list[np.ndarray] = []

        # starting with full point cloud
        current_xyz = xyz.copy()

        # loop over layers
        for layer_idx in range(self.num_layers):
            xyz_layers.append(current_xyz)  # store current layer coords

            # knn for this layer
            neigh_idx = _knn_indices(current_xyz, current_xyz, self.k_n)
            neigh_idx_layers.append(neigh_idx)

            # how many points to keep after subsampling
            ratio = self.sub_sampling_ratio[layer_idx]
            num_sub = max(1, current_xyz.shape[0] // ratio)

            # randomly subsample points
            sampled_indices = np.random.choice(current_xyz.shape[0], num_sub, replace=False)
            sampled_xyz = current_xyz[sampled_indices]

            # subsampling indices
            sub_idx = neigh_idx[sampled_indices]
            sub_idx_layers.append(sub_idx.astype(np.int64))

            # interpolation index (for upsampling)
            interp_idx = _knn_indices(current_xyz, sampled_xyz, 1)
            interp_idx_layers.append(interp_idx.astype(np.int64))

            current_xyz = sampled_xyz  # move to the next layer

        return xyz_layers, neigh_idx_layers, sub_idx_layers, interp_idx_layers

    # get sample from dataset, apply preprocessing, prepare for RandLa-Net use
    # input: index of sample
    # output: dictionary with features, labels, and hierarchy data for the sample
    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = np.load(self.files[idx])

        required_keys = {"points", "features", "labels"}
        missing = required_keys - set(sample.files)
        if missing:
            raise KeyError(f"{self.files[idx]} is missing keys: {sorted(missing)}")

        # extract arrays
        points = sample["points"].astype(np.float32)      # (N, 3) -> x, y, z
        features = sample["features"]                     # (N, 3) -> red, green, blue
        labels = sample["labels"].astype(np.int64)        # (N,)   -> scalar_label

        expected_feature_dim = self.expected_input_dim - 3

        if features.ndim != 2 or features.shape[1] != expected_feature_dim:
            raise ValueError(
                f"Expected features shape (N, {expected_feature_dim}), got {features.shape}"
            )

        if labels.ndim != 1:
            raise ValueError(f"Expected labels shape (N,), got {labels.shape}")

        # make sure labels are binary for obstacle / non-obstacle
        unique_labels = np.unique(labels)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                f"Expected binary labels in {{0,1}}, got {unique_labels.tolist()}"
            )

        # make sure there is a fixed number of points per sample
        sampled_idx = _random_sample_indices(points.shape[0], self.num_points)
        points = points[sampled_idx]
        features = features[sampled_idx]
        labels = labels[sampled_idx]

        # center and normalize
        if self.center_cloud:
            points = points - np.mean(points, axis=0, keepdims=True)

        if self.normalize_xyz:
            points = _normalize_points(points)

        features = features.astype(np.float32)

        if features.shape[1] > 0:
            if self.normalize_features:
                features = _normalize_features(features)

            input_features = np.concatenate([points, features], axis=1).astype(np.float32)
        else:
            input_features = points.astype(np.float32)
        if self.normalize_features:
            features = _normalize_features(features)
        else:
            # even without normalization, keep RGB on a reasonable scale
            if np.max(features) > 1.0:
                features = features / 255.0

        if input_features.shape[1] != self.expected_input_dim:
            raise ValueError(
                f"Expected input_dim={self.expected_input_dim}, got {input_features.shape[1]} "
                f"from points({points.shape[1]}) + features({features.shape[1]})"
            )

        xyz_layers, neigh_idx_layers, sub_idx_layers, interp_idx_layers = self._build_hierarchy(points)

        # convert to tensors and return batch dictionary
        # dictionary contains: input features, labels, layer coords, layer neighbor indices,
        # subsampling indices, interpolation indices
        batch = {
            "features": torch.tensor(input_features, dtype=torch.float32),   # (N, C)
            "labels": torch.tensor(labels, dtype=torch.long),                # (N,)
            "xyz": [torch.tensor(arr, dtype=torch.float32) for arr in xyz_layers],
            "neigh_idx": [torch.tensor(arr, dtype=torch.long) for arr in neigh_idx_layers],
            "sub_idx": [torch.tensor(arr, dtype=torch.long) for arr in sub_idx_layers],
            "interp_idx": [torch.tensor(arr, dtype=torch.long) for arr in interp_idx_layers],
        }
        return batch