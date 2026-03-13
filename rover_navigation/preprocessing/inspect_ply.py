import numpy as np


def load_cloudcompare_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        header = []

        while True:
            line = f.readline().decode("utf-8").strip()
            header.append(line)
            if line == "end_header":
                break

        vertex_count = None
        property_count = 0
        in_vertex_block = False

        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
                in_vertex_block = True
                continue

            if in_vertex_block and line.startswith("property"):
                property_count += 1
                continue

            if in_vertex_block and line.startswith("element"):
                in_vertex_block = False

        if vertex_count is None:
            raise ValueError("Could not find vertex count in PLY header.")

        data = np.fromfile(f, dtype=np.float32)

    expected_size = vertex_count * property_count
    if data.size != expected_size:
        raise ValueError(
            f"Unexpected data size. Got {data.size} floats, expected {expected_size}."
        )

    data = data.reshape(vertex_count, property_count)

    points = data[:, :3]
    scalars = data[:, 3:]

    return points, scalars