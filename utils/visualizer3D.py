import open3d as o3d
import numpy as np

# Load the PLY file
point_cloud = o3d.io.read_point_cloud("colored_gaussian_centers.ply")

# Check if the point cloud is empty
if point_cloud.is_empty():
    print("Point cloud is empty!")
else:
    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])
