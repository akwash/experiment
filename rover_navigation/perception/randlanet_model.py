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
#
# network architecture:
# - encoder: Multiple dilated residual blocks with hierarchical subsampling.
# - bottleneck: Feature compression layer.
# - decoder: Feature compression layer.
# - classifier: Final layers producing per-point semantic class predictions.
#
# input: batch dictionary produced by dataset.py
# output: tensor of shape (B,N,num_classes) representing the predicted
# semantic labels for each point

from pathlib import Path

import torch
import torch.nn as nn

from rover_navigation.perception.randlanet_blocks import (
    DilatedResidualBlock,
    SharedMLP1d,
    index_points,
)
from rover_navigation.util.config_loader import load_yaml


# build paths relative to this file so code works no matter where it is run from
ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "config" / "dataset.yaml"
TRAINING_CONFIG = ROOT / "config" / "training.yaml"


# random sampling and nearest neighbor interpolation for encoder/decoder
def random_sample(features: torch.Tensor, pool_idx: torch.Tensor) -> torch.Tensor:
    """
    Random-sampling style pooling used in the encoder.

    features: (B, C, N)
    pool_idx: (B, M, K)
    returns:  (B, C, M)
    """
    if features.ndim != 3:
        raise ValueError(f"Expected features shape (B, C, N), got {features.shape}")

    if pool_idx.ndim != 3:
        raise ValueError(f"Expected pool_idx shape (B, M, K), got {pool_idx.shape}")

    # index_points expects (B, N, C), so permute first
    neighbor_features = index_points(features.permute(0, 2, 1), pool_idx)  # (B, M, K, C)
    neighbor_features = neighbor_features.max(dim=2)[0]  # (B, M, C)

    return neighbor_features.permute(0, 2, 1)  # (B, C, M)


def nearest_interpolation(features: torch.Tensor, interp_idx: torch.Tensor) -> torch.Tensor:
    """
    Nearest-neighbor interpolation used in the decoder.

    features:   (B, C, M)
    interp_idx: (B, N, 1)
    returns:    (B, C, N)
    """
    if features.ndim != 3:
        raise ValueError(f"Expected features shape (B, C, M), got {features.shape}")

    if interp_idx.ndim != 3:
        raise ValueError(f"Expected interp_idx shape (B, N, 1), got {interp_idx.shape}")

    interpolated = index_points(features.permute(0, 2, 1), interp_idx)  # (B, N, 1, C)
    interpolated = interpolated.squeeze(2)  # (B, N, C)

    return interpolated.permute(0, 2, 1)  # (B, C, N)


# main RandLA-Net model class
# input: dataset and training config paths for model parameters
class RandLANet(nn.Module):
    def __init__(
        self,
        dataset_config_path: str | Path = DATASET_CONFIG,
        training_config_path: str | Path = TRAINING_CONFIG,
    ):
        super().__init__()

        # load conig param
        dataset_cfg = load_yaml(dataset_config_path)
        training_cfg = load_yaml(training_config_path)

        sampling_cfg = dataset_cfg["sampling"]
        model_cfg = training_cfg["model"]

        # extract model param
        self.num_layers = int(sampling_cfg["num_layers"])
        self.d_out = list(sampling_cfg["d_out"])
        self.input_dim = int(model_cfg["input_dim"])
        self.num_classes = int(model_cfg["num_classes"])

        # initial feature projection
        # input is expected to be (xyz + additional per-point features)
        self.fc_start = SharedMLP1d(self.input_dim, 8)

        # define encoder channels
        # first layer is eight channels, then channels defined by d_out list in config
        encoder_channels = [8] + self.d_out
        self.encoder = nn.ModuleList(
            [
                DilatedResidualBlock(encoder_channels[i], encoder_channels[i + 1])
                for i in range(self.num_layers)
            ]
        )

        # bottleneck layer with same output channels as last encoder layer
        self.bottleneck = SharedMLP1d(self.d_out[-1], self.d_out[-1])

        # decoder uses skip connections from encoder outputs
        skip_channels = self.d_out[:]  # one skip feature map per encoder layer
        decoder_out_channels = list(reversed(self.d_out[:-1])) + [32]

        decoder_layers = []
        current_in_channels = self.d_out[-1]

        for i in range(self.num_layers):
            skip_ch = skip_channels[-1 - i]
            out_ch = decoder_out_channels[i]
            decoder_layers.append(
                SharedMLP1d(current_in_channels + skip_ch, out_ch)
            )
            current_in_channels = out_ch

        self.decoder = nn.ModuleList(decoder_layers)

        # final classifier layers
        self.classifier = nn.Sequential(
            SharedMLP1d(current_in_channels, 32),
            nn.Dropout(p=0.5),
            nn.Conv1d(32, self.num_classes, kernel_size=1),
        )

    # forward pass through network
    def forward(self, batch: dict[str, torch.Tensor | list[torch.Tensor]]) -> torch.Tensor:
        features = batch["features"]            # (B, N, C) input features (xyz + other features)
        xyz_list = batch["xyz"]                 # xyz coords for each layer
        neigh_idx_list = batch["neigh_idx"]     # neighbor indices for each layer
        sub_idx_list = batch["sub_idx"]         # subsampling indices for each layer
        interp_idx_list = batch["interp_idx"]   # interpol indices for each layer

        # check the input dimensions
        if features.ndim != 3:
            raise ValueError(f"Expected features shape (B, N, C), got {features.shape}")

        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} input features per point, got {features.shape[-1]}"
            )

        if len(xyz_list) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} xyz layers, got {len(xyz_list)}"
            )

        if len(neigh_idx_list) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} neigh_idx layers, got {len(neigh_idx_list)}"
            )

        if len(sub_idx_list) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} sub_idx layers, got {len(sub_idx_list)}"
            )

        if len(interp_idx_list) != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} interp_idx layers, got {len(interp_idx_list)}"
            )

        # intial feature transformation
        x = features.permute(0, 2, 1)  # (B, C, N)
        x = self.fc_start(x)

        skip_features = []  # store features for skip connection in decoder

        # encoder with random sampling and feature learning
        for i in range(self.num_layers):
            xyz = xyz_list[i]
            neigh_idx = neigh_idx_list[i]
            sub_idx = sub_idx_list[i]

            x = self.encoder[i](x, xyz, neigh_idx)
            skip_features.append(x)
            x = random_sample(x, sub_idx)

        # bottleneck
        x = self.bottleneck(x)

        # decoder with nearest neighbor interpolation and skip connections
        for i in range(self.num_layers - 1, -1, -1):
            interp_idx = interp_idx_list[i]
            x = nearest_interpolation(x, interp_idx)

            x = torch.cat([x, skip_features[i]], dim=1)

            dec_idx = self.num_layers - 1 - i
            x = self.decoder[dec_idx](x)

        # final classifer for per-point semantic prediction
        logits = self.classifier(x)      # (B, num_classes, N)
        logits = logits.permute(0, 2, 1) # (B, N, num_classes)

        return logits