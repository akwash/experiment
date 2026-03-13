# Filename: randlanet_model.py
# Author: AK Wash
# Created: 2026-03-10

# Description: implementation of RandLA-Net neural network for 
# the semantic segmentation of point clouds 

# combines:
# - random point sampling
# - feature learning
# - local spatial encloding
# - attentive feature aggregation

# network architecture:
# - encoder: Multiple dilated residual blocks with hierarchical subsampling.
# - bottleneck: Feature compression layer.
# - decoder: Feature compression layer.
#- classifier: Final layers producing per-point semantic class predictions.

# input: batch dictionary produced by dataset.py
# output: tensor of shape (B,N,num_classes) representing the predicted
# semantic labels for each point

import torch
import torch.nn as nn
import torch.nn.functional as F

from perception.randlanet_blocks import DilatedResidualBlock, SharedMLP1d, index_points
from util.config_loader import load_yaml


def random_sample(features: torch.Tensor, pool_idx: torch.Tensor) -> torch.Tensor:
    """
    features: (B, C, N)
    pool_idx: (B, M, K)
    returns:  (B, C, M)
    """
    neighbor_features = index_points(features.permute(0, 2, 1), pool_idx)  # (B, M, K, C)
    neighbor_features = neighbor_features.max(dim=2)[0]                     # (B, M, C)
    return neighbor_features.permute(0, 2, 1)                              # (B, C, M)


def nearest_interpolation(features: torch.Tensor, interp_idx: torch.Tensor) -> torch.Tensor:
    """
    features:  (B, C, M)
    interp_idx: (B, N, 1)
    returns:   (B, C, N)
    """
    interpolated = index_points(features.permute(0, 2, 1), interp_idx)     # (B, N, 1, C)
    interpolated = interpolated.squeeze(2)                                  # (B, N, C)
    return interpolated.permute(0, 2, 1)                                    # (B, C, N)


class RandLANet(nn.Module):
    def __init__(self, dataset_config_path: str = "config/dataset.yaml", training_config_path: str = "config/training.yaml"):
        super().__init__()

        dataset_cfg = load_yaml(dataset_config_path)
        training_cfg = load_yaml(training_config_path)

        sampling_cfg = dataset_cfg["sampling"]
        model_cfg = training_cfg["model"]

        self.num_layers = int(sampling_cfg["num_layers"])
        self.d_out = list(sampling_cfg["d_out"])
        self.input_dim = int(model_cfg["input_dim"])
        self.num_classes = int(model_cfg["num_classes"])

        self.fc_start = SharedMLP1d(self.input_dim, 8)

        encoder_channels = [8] + self.d_out
        self.encoder = nn.ModuleList(
            [
                DilatedResidualBlock(encoder_channels[i], encoder_channels[i + 1])
                for i in range(self.num_layers)
            ]
        )

        self.bottleneck = SharedMLP1d(self.d_out[-1], self.d_out[-1])

        decoder_in_channels = [256, 128, 64, 32]
        decoder_skip_channels = [128, 64, 16, 8]
        decoder_out_channels = [128, 64, 32, 32]

        self.decoder = nn.ModuleList(
            [
                SharedMLP1d(decoder_in_channels[i] + decoder_skip_channels[i], decoder_out_channels[i])
                for i in range(self.num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            SharedMLP1d(32, 32),
            nn.Dropout(p=0.5),
            nn.Conv1d(32, self.num_classes, kernel_size=1),
        )

    def forward(self, batch: dict[str, torch.Tensor | list[torch.Tensor]]) -> torch.Tensor:
        features = batch["features"]  # (B, N, C)
        xyz_list = batch["xyz"]
        neigh_idx_list = batch["neigh_idx"]
        sub_idx_list = batch["sub_idx"]
        interp_idx_list = batch["interp_idx"]

        if features.ndim != 3:
            raise ValueError(f"Expected features shape (B, N, C), got {features.shape}")

        x = features.permute(0, 2, 1)  # (B, C, N)
        x = self.fc_start(x)

        skip_features = []

        for i in range(self.num_layers):
            xyz = xyz_list[i]
            neigh_idx = neigh_idx_list[i]
            sub_idx = sub_idx_list[i]

            x = self.encoder[i](x, xyz, neigh_idx)
            skip_features.append(x)
            x = random_sample(x, sub_idx)

        x = self.bottleneck(x)

        for i in range(self.num_layers - 1, -1, -1):
            interp_idx = interp_idx_list[i]
            x = nearest_interpolation(x, interp_idx)
            x = torch.cat([x, skip_features[i]], dim=1)
            dec_idx = self.num_layers - 1 - i
            x = self.decoder[dec_idx](x)

        logits = self.classifier(x)      # (B, num_classes, N)
        logits = logits.permute(0, 2, 1) # (B, N, num_classes)
        return logits