# Filename: model.py
# Author: AK Wash
# Created: 2026-03-16
# Description: provides a wrapper around RandLA-Net arhcitecture
# to clearly integrate into training and inference pipelines. Keeps
# external interface simple

"""
RandLa-Net expects input tensors of (B,N,d_in):
- B -> batch size
- N -> number of points in point cloud
- d_in number of input features per point (1st three are XYZ coords)

Outputs segmentation logits of (B, num_classes, N).
Training loop reshapes logits and computes per-point loss

"""

from rover_navigation.perception.randlanet_model import RandLANet
import torch.nn as nn

__all__ = ["RandLANet"]

class MyRandLANet(nn.Module):
    """
    Wrappr module that gives RandLA-Net in simplfied interface
    """
    def __init__(self,d_in,num_classes,device):
        """
        :param d_in: number of input features per point. The first three
        features are XYZ coordinates
        :param num_classes: number of semantic classes to predict
        :param device: which device the model is allocated on
        """
        super().__init__()
        self.model = RandLANet(
            d_in = d_in,
            num_classes = num_classes,
            num_neighbors = 16,
            decimation = 4,
            device = device
        )
    def forward(self, x):
        """
        forward pass through RandLA-Net
        :param x: input tensor (B,N,d_in) that contains point clouds
        coordiantes and per-point features
        :return: tensor, segmentation logits of shape (B,num_classes,N)
        """
        return self.model(x)