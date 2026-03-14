import numpy as np

# input: path to cloudcomapare plyy file
# output: tuple of two numpy arrays
def load_cloudcompare_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f: # open in binary mode
        header = [] # intialize list for header names

        # read file line by line, convert bytes to string, remove whitespace
        # newline characters, append each line to header
        while True:
            line = f.readline().decode("utf-8").strip()
            header.append(line)
            if line == "end_header":
                break

        vertex_count = None # number of vertices
        property_count = 0 # number of properies per vertex
        in_vertex_block = False # if in element vertex of header

        # find vertex block in header, then count number of properties until next element
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

    # check that the data size matches the expected size (from header)
    expected_size = vertex_count * property_count
    if data.size != expected_size:
        raise ValueError(
            f"Unexpected data size. Got {data.size} floats, expected {expected_size}."
        )

    # reshape data into a mtraix
    data = data.reshape(vertex_count, property_count)

    # split the data into points and scalars
    points = data[:, :3]
    scalars = data[:, 3:]

    # return the points and scalars as numpy arrays
    return points, scalars