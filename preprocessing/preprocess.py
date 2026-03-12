import numpy as np

def save_dataset(points,labels, output_fole = "cloud.npy")
    data = {
        "points": points,
        "features": points,
        "labels": labels
    }

    np.save(output_file, data)