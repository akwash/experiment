import os
import torch
from perception.dataset import PointCloudDataset


def test_pointcloud_dataset_loads_npz():
    data_dir = os.path.join("data", "processed")
    dataset = PointCloudDataset(data_dir)

    x, y = dataset[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    assert x.ndim == 2
    assert y.ndim == 1

    assert x.shape[1] == 8
    assert y.shape[0] == x.shape[0]