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
#
# output: trained model weights saved to the checkpoints directory
# adapted from: https://github.com/aRI0U/RandLA-Net-pytorch

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rover_navigation.perception.dataset import RandLANetDataset
from rover_navigation.perception.randlanet_model import RandLANet
from rover_navigation.util.config_loader import load_yaml


# build paths relative to this file so code works no matter where it is run from
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
DATASET_CONFIG = ROOT / "config" / "dataset.yaml"
TRAINING_CONFIG = ROOT / "config" / "training.yaml"


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for batches that contain both tensors and lists of tensors.

    Each sample from RandLANetDataset returns:
    - features: tensor (N, C)
    - labels: tensor (N,)
    - xyz, neigh_idx, sub_idx, interp_idx: lists of tensors for each layer

    This stacks plain tensors directly and stacks each element of list fields
    layer-by-layer across the batch dimension.
    """
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
    """
    Move a collated batch to CPU or GPU.
    Handles both normal tensors and list[tensor] hierarchy fields.
    """
    moved = {}

    for key, value in batch.items():
        if isinstance(value, list):
            moved[key] = [v.to(device) for v in value]
        else:
            moved[key] = value.to(device)

    return moved


def train() -> None:
    dataset_cfg = load_yaml(DATASET_CONFIG)
    training_cfg = load_yaml(TRAINING_CONFIG)

    paths_cfg = dataset_cfg["paths"]
    train_cfg = training_cfg["training"]
    ckpt_cfg = training_cfg["checkpoints"]
    loss_cfg = training_cfg["loss"]

    processed_dir = PROJECT_ROOT / paths_cfg["processed_dir"]

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")

    if train_cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg["device"])

    dataset = RandLANetDataset(str(processed_dir), dataset_config_path=DATASET_CONFIG)

    if len(dataset) == 0:
        raise ValueError(f"No training samples found in {processed_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        collate_fn=collate_fn,
    )

    # use internal plain-PyTorch RandLA-Net implementation
    model = RandLANet(
        dataset_config_path=DATASET_CONFIG,
        training_config_path=TRAINING_CONFIG,
    ).to(device)

    class_weights = torch.tensor([1.0, 50.0], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    epochs = int(train_cfg["epochs"])
    num_classes = model.num_classes

    print(f"Training on device: {device}")
    print(f"Processed data directory: {processed_dir}")
    print(f"Number of training samples: {len(dataset)}")
    print(f"Model input dim: {model.input_dim}")
    print(f"Model num classes: {model.num_classes}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader:
            batch = move_batch_to_device(batch, device)

            logits = model(batch)      # (B, N, num_classes)
            labels = batch["labels"]   # (B, N)

            if logits.ndim != 3:
                raise ValueError(f"Expected logits shape (B, N, C), got {logits.shape}")

            if labels.ndim != 2:
                raise ValueError(f"Expected labels shape (B, N), got {labels.shape}")

            if logits.shape[:2] != labels.shape:
                raise ValueError(
                    f"Logits/labels shape mismatch: logits {logits.shape}, labels {labels.shape}"
                )

            loss = criterion(
                logits.reshape(-1, num_classes),
                labels.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    save_dir = PROJECT_ROOT / ckpt_cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / ckpt_cfg["save_name"]
    torch.save(model.state_dict(), save_path)

    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    train()