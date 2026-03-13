# experiment
Senior Design Experiement

# LiDAR obstacle detection and path planning by D* Lite
Implements LiDAR based obstacle detection and pathplanning. uses emantic segmentation of point clouds to identify obstacles and generate a traversal map for D* lite path planning.

# System Overview
Pipeline: LiDAR scan --> segmentation --> traversability map --> D* Lite path planning --> rover path

Components:
1. Point Cloud processing
2. semantic segmenatation through RandLA-Net
3. Traversability mapping
4. D* lite path planning

# Architecture
Lidar Scan ('.ply')
 |
 RandLa-Net
 |
 Obstacle Classification
 | 
 Traversability Map
 |
 D* Lite
 |
 Rover path


# Author

Anne-Kinsey Wash  