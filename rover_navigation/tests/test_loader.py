import os
from rover_navigation.preprocessing.load_ply import load_cloudcompare_ply
from pathlib import Path


def print_ply_header(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("utf-8", errors="ignore").strip()
            print(line)
            if line == "end_header":
                break
def test_load_cloudcompare_ply_shapes():
    test_path =  Path(__file__).resolve().parents[1] / "data" / "raw" / "test_cloud.ply"

    print("\n ---PLY HEADER ---")
    print_ply_header(test_path)
    print("---END HEADER---")

    points, features, labels = load_cloudcompare_ply(test_path)

    # points:
    assert points.ndim == 2
    assert points.shape[1] ==3

    # features:
    assert features is not None
    assert features.ndim ==2
    assert features.shape[0] == points.shape[0]
    assert features.shape[1] ==0

    # labels
    assert labels is not None
    assert labels.ndim == 1
    assert labels.shape[0] == points.shape[0]

    # label vals:
    unique_labels = set(labels.tolist())
    assert unique_labels.issubset({0,1})

    print("Unique labels:", unique_labels)