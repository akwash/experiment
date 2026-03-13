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
import sys

print("Hello World!")

import os
from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset

from util.config_loader import load_yaml


def _normalize_points(points: np.ndarray) -> np.ndarray:
    centered = points - np.mean(points, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(centered, axis=1))
    if scale > 0:
        centered = centered / scale
    return centered.astype(np.float32)


def _normalize_features(features: np.ndarray) -> np.ndarray:
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((features - mean) / std).astype(np.float32)


def _random_sample_indices(num_points_total: int, num_points_target: int) -> np.ndarray:
    if num_points_total >= num_points_target:
        return np.random.choice(num_points_total, num_points_target, replace=False)
    extra = np.random.choice(num_points_total, num_points_target - num_points_total, replace=True)
    base = np.arange(num_points_total)
    return np.concatenate([base, extra], axis=0)


def _knn_indices(query_pts: np.ndarray, support_pts: np.ndarray, k: int) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nbrs.fit(support_pts)
    _, indices = nbrs.kneighbors(query_pts)
    return indices.astype(np.int64)


class RandLANetDataset(Dataset):
    def __init__(self, data_dir: str, dataset_config_path: str = "config/dataset.yaml"):
        self.data_dir = data_dir
        self.dataset_cfg = load_yaml(dataset_config_path)

        prep_cfg = self.dataset_cfg["preprocessing"]
        sampling_cfg = self.dataset_cfg["sampling"]

        self.num_points = int(prep_cfg["num_points"])
        self.center_cloud = bool(prep_cfg["center_cloud"])
        self.normalize_xyz = bool(prep_cfg["normalize_xyz"])
        self.normalize_features = bool(prep_cfg["normalize_features"])

        self.k_n = int(sampling_cfg["k_n"])
        self.num_layers = int(sampling_cfg["num_layers"])
        self.sub_sampling_ratio = list(sampling_cfg["sub_sampling_ratio"])

        self.files = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        )

        if not self.files:
            raise ValueError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _build_hierarchy(self, xyz: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        xyz_layers: list[np.ndarray] = []
        neigh_idx_layers: list[np.ndarray] = []
        sub_idx_layers: list[np.ndarray] = []
        interp_idx_layers: list[np.ndarray] = []

        current_xyz = xyz.copy()

        for layer_idx in range(self.num_layers):
            xyz_layers.append(current_xyz)

            neigh_idx = _knn_indices(current_xyz, current_xyz, self.k_n)
            neigh_idx_layers.append(neigh_idx)

            ratio = self.sub_sampling_ratio[layer_idx]
            num_sub = max(1, current_xyz.shape[0] // ratio)

            sampled_indices = np.random.choice(current_xyz.shape[0], num_sub, replace=False)
            sampled_xyz = current_xyz[sampled_indices]

            sub_idx = neigh_idx[sampled_indices]
            sub_idx_layers.append(sub_idx.astype(np.int64))

            interp_idx = _knn_indices(current_xyz, sampled_xyz, 1)
            interp_idx_layers.append(interp_idx.astype(np.int64))

            current_xyz = sampled_xyz

        return xyz_layers, neigh_idx_layers, sub_idx_layers, interp_idx_layers

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = np.load(self.files[idx])

        points = sample["points"].astype(np.float32)      # (N, 3)
        features = sample["features"].astype(np.float32)  # (N, F)
        labels = sample["labels"].astype(np.int64)        # (N,)

        sampled_idx = _random_sample_indices(points.shape[0], self.num_points)
        points = points[sampled_idx]
        features = features[sampled_idx]
        labels = labels[sampled_idx]

        if self.center_cloud:
            points = points - np.mean(points, axis=0, keepdims=True)

        if self.normalize_xyz:
            points = _normalize_points(points)

        if self.normalize_features:
            features = _normalize_features(features)

        input_features = np.concatenate([points, features], axis=1).astype(np.float32)

        xyz_layers, neigh_idx_layers, sub_idx_layers, interp_idx_layers = self._build_hierarchy(points)

        batch = {
            "features": torch.tensor(input_features, dtype=torch.float32),   # (N, C)
            "labels": torch.tensor(labels, dtype=torch.long),                # (N,)
            "xyz": [torch.tensor(arr, dtype=torch.float32) for arr in xyz_layers],
            "neigh_idx": [torch.tensor(arr, dtype=torch.long) for arr in neigh_idx_layers],
            "sub_idx": [torch.tensor(arr, dtype=torch.long) for arr in sub_idx_layers],
            "interp_idx": [torch.tensor(arr, dtype=torch.long) for arr in interp_idx_layers],
        }
        return batch