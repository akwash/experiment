import numpy as np

def load_cloudcompare_ply(path):

    with open(path, "rb") as f:

        header = []
        while True:
            line = f.readline().decode("utf-8").strip()
            header.append(line)
            if line == "end_header":
                break

        # find number of vertices
        vertex_count = None
        for line in header:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])

        data = np.fromfile(f, dtype=np.float32)

    data = data.reshape(vertex_count, 9)

    points = data[:, :3]
    scalars = data[:, 3:]

    return points, scalars