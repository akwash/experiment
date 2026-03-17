import os
import numpy as np
from rover_navigation.preprocessing.load_ply import load_cloudcompare_ply


LABEL_COLUMN = 5


def prepare_one_file(input_path: str, output_path: str, label_column: int = LABEL_COLUMN) -> None:
    points, scalars = load_cloudcompare_ply(input_path)

    if scalars.ndim != 2:
        raise ValueError(f"Expected scalars to be 2D, got shape {scalars.shape}")

    if label_column < 0 or label_column >= scalars.shape[1]:
        raise ValueError(
            f"Label column {label_column} is out of bounds for scalars with shape {scalars.shape}"
        )

    labels = scalars[:, label_column].astype(np.int64)

    feature_cols = [i for i in range(scalars.shape[1]) if i != label_column]
    features = scalars[:, feature_cols].astype(np.float32)

    np.savez_compressed(
        output_path,
        points=points.astype(np.float32),
        features=features,
        labels=labels,
    )


if __name__ == "__main__":
    input_path = os.path.join("data", "test_cloud.ply")
    output_path = os.path.join("data", "processed", "test_cloud.npz")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prepare_one_file(input_path, output_path)

    print(f"Saved processed dataset to {output_path}")