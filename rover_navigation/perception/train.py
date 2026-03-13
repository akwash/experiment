# Filename: train.py
# Author: AK Wash
# Created: 2026-03-10

# Description: training pipeline for model. Performs following
# tasks:
# 1. Load dataset and training config files
# 2. Construct RandLa-Net neural network
# 3. Create PyTorch dataloader for training data
# 4. train model using corss-entropy loss
# 5. Save trained model checkpoints

# output: trained model weights saved to the checkpoints directory
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from perception.dataset import RandLANetDataset
from perception.randlanet_model import RandLANet
from util.config_loader import load_yaml


def collate_fn(batch: list[dict]) -> dict:
    out = {}

    keys = batch[0].keys()
    for key in keys:
        if isinstance(batch[0][key], list):
            out[key] = []
            for i in range(len(batch[0][key])):
                out[key].append(torch.stack([sample[key][i] for sample in batch], dim=0))
        else:
            out[key] = torch.stack([sample[key] for sample in batch], dim=0)

    return out


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, list):
            moved[key] = [v.to(device) for v in value]
        else:
            moved[key] = value.to(device)
    return moved


def train() -> None:
    dataset_cfg = load_yaml("config/dataset.yaml")
    training_cfg = load_yaml("config/training.yaml")

    processed_dir = dataset_cfg["paths"]["processed_dir"]
    train_cfg = training_cfg["training"]
    ckpt_cfg = training_cfg["checkpoints"]
    loss_cfg = training_cfg["loss"]
    model_cfg = training_cfg["model"]

    if train_cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg["device"])

    dataset = RandLANetDataset(processed_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_fn,
    )

    model = RandLANet().to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=int(loss_cfg["ignore_index"]))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    epochs = int(train_cfg["epochs"])
    num_classes = int(model_cfg["num_classes"])

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            batch = move_batch_to_device(batch, device)

            logits = model(batch)                  # (B, N, num_classes)
            labels = batch["labels"]               # (B, N)

            logits = logits.reshape(-1, num_classes)
            labels = labels.reshape(-1)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    save_dir = Path(ckpt_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / ckpt_cfg["save_name"]

    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    train()