import numpy as np
import open3d as o3d


def ply_to_numpy(ply_file, label_file=None):

    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)

    features = points

    if label_file:
        labels = np.loadtxt(label_file)
    else:
        labels = np.zeros(len(points))

    return {
        "points": points,
        "features": features,
        "labels": labels
    }