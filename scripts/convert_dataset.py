import os
from preprocessing.bin_to_ply import convert_bin_to_ply


BIN_DIR = "data/raw_bin"
PLY_DIR = "data/ply"

os.makedirs(PLY_DIR, exist_ok=True)

for file in os.listdir(BIN_DIR):

    if file.endswith(".bin"):

        bin_path = os.path.join(BIN_DIR, file)
        ply_path = os.path.join(PLY_DIR, file.replace(".bin", ".ply"))

        convert_bin_to_ply(bin_path, ply_path)