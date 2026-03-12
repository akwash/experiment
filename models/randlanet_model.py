import torch
from RandLANet import RandLANet


def create_model(num_classes, device):

    model = RandLANet(num_classes)

    model = model.to(device)

    return model