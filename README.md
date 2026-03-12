# experiment
Senior Design Experiement

# LiDAR obstacle detection and path planning by D* Lite

1. Convert annoted point clouds '.bin' (CloudCompare export) to '.ply'
2. Train PyTorch to classify obstacle points
3. Run in-situ obstacle detection from LiDAR scans,
4. Score a gridded traversal map based on obstacle location
5. Plan around obstacles by D* Lite

## Project Layout
- `src/pipeline/convert.py` – conversion helper using CloudCompare CLI.
- `src/pipeline/dataset.py` – point cloud dataset loader for `.ply` + labels.
- `src/pipeline/model.py` – lightweight RandLA-Net-inspired point classifier.
- `src/pipeline/grid_mapping.py` – map predicted obstacle probabilities to a traversal grid.
- `src/pipeline/dstar_lite.py` – D* Lite planner for dynamic replanning.
- `src/pipeline/inference.py` – end-to-end single-scan inference + planning demo.