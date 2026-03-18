import os
from rover_navigation.preprocessing.load_ply import load_cloudcompare_ply

def print_ply_header(path):
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("utf-8", errors="ignore").strip()
            print(line)
            if line == "end_header":
                break

# def test_load_cloudcompare_ply_shapes():
#     test_path = os.path.join("data", "test_cloud.ply")

#     print("\n--- PLY HEADER ---")
#     print_ply_header(test_path)
#     print("--- END HEADER ---\n")

#     points, scalars = load_cloudcompare_ply(test_path)

#     assert points.ndim == 2
#     assert points.shape[1] == 3

#     assert scalars.ndim == 2
#     assert scalars.shape[0] == points.shape[0]
#     assert scalars.shape[1] == 6

def test_load_cloudcompare_ply_shapes_2():
    test_path = os.path.join("data", "test_cloud.ply")
    points, scalars = load_cloudcompare_ply(test_path)

    print("\n--- PLY HEADER ---")
    print_ply_header(test_path)
    print("--- END HEADER ---\n")

    assert points.ndim == 2
    assert points.shape[1] == 3

    assert scalars.ndim == 2
    assert scalars.shape[0] == points.shape[0]
    assert scalars.shape[1] == 6