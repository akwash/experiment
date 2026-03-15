import os
from preprocessing.load_ply import load_cloudcompare_ply


def test_load_cloudcompare_ply_shapes():
    test_path = os.path.join("data", "test_cloud.ply")
    points, scalars = load_cloudcompare_ply(test_path)

    assert points.ndim == 2
    assert points.shape[1] == 3

    assert scalars.ndim == 2
    assert scalars.shape[0] == points.shape[0]
    assert scalars.shape[1] == 6