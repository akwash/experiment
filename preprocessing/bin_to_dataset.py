import open3d as o3d
from utils.io_utils import load_bin


def convert_bin_to_ply(bin_file, ply_file):

    xyz, intensity = load_bin(bin_file)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    o3d.io.write_point_cloud(ply_file, pcd)

    print("Saved:", ply_file)