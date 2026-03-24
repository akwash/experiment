# Filename: randlasnet_blocks.py
# Author: AK Wash
# Created: 2026-03-10

# Description: nueral network building blocks for the RandLA-Net
# architecture. Implements compoenents for the point cloud feature
# learning

# SharedMLP1d / SharedMLP2d
      #  Shared multilayer perceptron layers used throughout the network.

   # LocalSpatialEncoding
       # Encodes spatial relationships between neighboring points.

   # AttentivePooling
       # Aggregates neighborhood features using attention-based weighting.

   # DilatedResidualBlock
       # The primary feature extraction block used in the encoder.

# used in: perception/randlanet_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# util function for moving the batch to the device
# input: batch dictionary and device
def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B, N, C)
    idx:    (B, M, K) or (B, M, 1)
    returns: (B, M, K, C)
    """
    if points.ndim != 3:
        raise ValueError(f"points must have shape (B, N, C), got {points.shape}")
    if idx.ndim != 3:
        raise ValueError(f"idx must have shape (B, M, K), got {idx.shape}")

    b, n, c = points.shape
    if idx.min() < 0 or idx.max() >= n:
        raise IndexError(f"Neighbor index out of bounds for N={n}")

    _, m, k = idx.shape
    idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, c)
    points_expanded = points.unsqueeze(1).expand(-1, m, -1, -1)
    return torch.gather(points_expanded, 2, idx_expanded)


class SharedMLP2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn: bool = True, activation: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation:
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedMLP1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bn: bool = True, activation: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=not bn)]
        if bn:
            layers.append(nn.BatchNorm1d(out_channels))
        if activation:
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalSpatialEncoding(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.mlp = SharedMLP2d(10, out_channels)

    def forward(
        self,
        xyz: torch.Tensor,         # (B, N, 3)
        neigh_idx: torch.Tensor,   # (B, N, K)
        neigh_feat: torch.Tensor,  # (B, C, N, K)
    ) -> torch.Tensor:
        neighbor_xyz = index_points(xyz, neigh_idx)  # (B, N, K, 3)
        xyz_tile = xyz.unsqueeze(2).expand_as(neighbor_xyz)

        relative_xyz = xyz_tile - neighbor_xyz
        relative_dist = torch.norm(relative_xyz, dim=-1, keepdim=True)

        spatial_encoding = torch.cat(
            [xyz_tile, neighbor_xyz, relative_xyz, relative_dist],
            dim=-1,
        )  # (B, N, K, 10)

        spatial_encoding = spatial_encoding.permute(0, 3, 1, 2)  # (B, 10, N, K)
        spatial_encoding = self.mlp(spatial_encoding)

        return torch.cat([spatial_encoding, neigh_feat], dim=1)


class AttentivePooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Softmax(dim=-1),
        )
        self.mlp = SharedMLP2d(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score_fn(x)
        features = torch.sum(scores * x, dim=-1, keepdim=True)
        return self.mlp(features)


class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        mid_channels = out_channels // 2

        self.mlp1 = SharedMLP1d(in_channels, mid_channels)
        self.lse1 = LocalSpatialEncoding(mid_channels)
        self.pool1 = AttentivePooling(mid_channels * 2, mid_channels)

        self.lse2 = LocalSpatialEncoding(mid_channels)
        self.pool2 = AttentivePooling(mid_channels * 2, out_channels)

        self.shortcut = SharedMLP1d(in_channels, out_channels, activation=False)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(
        self,
        features: torch.Tensor,    # (B, C, N)
        xyz: torch.Tensor,         # (B, N, 3)
        neigh_idx: torch.Tensor,   # (B, N, K)
    ) -> torch.Tensor:
        x = self.mlp1(features)  # (B, mid, N)

        neigh_feat = index_points(x.permute(0, 2, 1), neigh_idx)     # (B, N, K, mid)
        neigh_feat = neigh_feat.permute(0, 3, 1, 2)                  # (B, mid, N, K)

        x = self.lse1(xyz, neigh_idx, neigh_feat)
        x = self.pool1(x).squeeze(-1)                                # (B, mid, N)

        neigh_feat = index_points(x.permute(0, 2, 1), neigh_idx)     # (B, N, K, mid)
        neigh_feat = neigh_feat.permute(0, 3, 1, 2)                  # (B, mid, N, K)

        x = self.lse2(xyz, neigh_idx, neigh_feat)
        x = self.pool2(x).squeeze(-1)                                # (B, out, N)

        shortcut = self.shortcut(features)
        return self.activation(x + shortcut)