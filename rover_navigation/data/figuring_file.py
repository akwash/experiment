from plyfile import PlyData
import numpy as np

file_path = r"C:\Users\ak\OneDrive\Desktop\experiment\rover_navigation\data\raw\test_cloud.ply"

ply = PlyData.read(file_path)
data = ply["vertex"].data

print("=== FIELD NAMES ===")
print(data.dtype.names)

print("\n=== FIELD STATS ===")
for name in data.dtype.names:
    vals = np.asarray(data[name])
    
    print(f"\n{name}")
    print("dtype:", vals.dtype)
    print("min:", np.min(vals))
    print("max:", np.max(vals))
    
    unique_vals = np.unique(vals)
    print("num unique:", len(unique_vals))
    print("unique sample:", unique_vals[:10])