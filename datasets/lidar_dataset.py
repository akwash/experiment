import torch
import numpy as np
from torch.utils.data import Dataset


class LidarDataset(Dataset):

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = np.load(self.files[idx], allow_pickle=True).item()

        points = torch.tensor(data["points"]).float()
        features = torch.tensor(data["features"]).float()
        labels = torch.tensor(data["labels"]).long()

        return points, features, labels