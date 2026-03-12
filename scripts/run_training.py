import glob
import torch
from torch.utils.data import DataLoader

from datasets.lidar_dataset import LidarDataset
from models.randlanet_model import create_model
from training.train_model import train
from utils.device import get_device


device = get_device()

dataset_files = glob.glob("data/processed/*.npy")

dataset = LidarDataset(dataset_files)

dataloader = DataLoader(dataset, batch_size=2)

model = create_model(num_classes=2, device=device)

train(model, dataloader, device)

torch.save(model.state_dict(), "randlanet_model.pth")