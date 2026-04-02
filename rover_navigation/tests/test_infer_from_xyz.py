"""Tests for in-memory XYZ inference batch building."""

import numpy as np
import pytest

from rover_navigation.perception.infer import build_inference_batch_from_xyz


def test_build_inference_batch_from_xyz_shapes():
    rng = np.random.default_rng(0)
    n = 9000
    xyz = rng.standard_normal((n, 3), dtype=np.float64).astype(np.float32)

    batch, sampled_idx, vis_points = build_inference_batch_from_xyz(
        xyz,
        features=None,
        log_label="test array",
    )

    assert vis_points.ndim == 2 and vis_points.shape[1] == 3
    assert sampled_idx.shape[0] == vis_points.shape[0]
    assert "features" in batch and "xyz" in batch
    assert batch["features"].shape[0] == 1
    # Model input_dim in training.yaml is 3 (XYZ only); empty extra features keeps C=3.
    assert batch["features"].shape[-1] == 3


def test_build_inference_batch_from_xyz_rejects_bad_shape():
    with pytest.raises(ValueError, match="points_xyz must be"):
        build_inference_batch_from_xyz(np.zeros((10, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="features must be"):
        build_inference_batch_from_xyz(
            np.zeros((10, 3), dtype=np.float32),
            features=np.zeros((5, 1), dtype=np.float32),
        )
