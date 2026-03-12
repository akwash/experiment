from preprocessing.load_ply import load_cloudcompare_ply

points, scalars = load_cloudcompare_ply("data/test_cloud.ply")

print("points:", points.shape)
print("scalar fields:", scalars.shape)

print(points[:5])
print(scalars[:5])