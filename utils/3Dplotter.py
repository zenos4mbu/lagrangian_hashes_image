from plyfile import PlyData, PlyElement
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

def read_obj(filename):
    vertices = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
    return np.array(vertices)

# Your Gaussian centers and original object points
original_object_points = read_obj("chair.obj")
gaussian_centers = read_obj("gaussian_positions_lv07_epoch_00251_5_05.obj")

# Compute the distance matrix between Gaussian centers and the original object vertices
distances = distance_matrix(gaussian_centers, original_object_points)

# Get the minimum distance to the original object for each Gaussian center
min_distances = np.min(distances, axis=1)

# # Normalize distances for coloring
min_distance = np.min(min_distances)
max_distance = np.max(min_distances)
# normalized_distances = (min_distances - min_distance) / (max_distance - min_distance)

# Add a small constant to ensure there are no zero distances
adjusted_distances = min_distances + 0.001

# Apply a logarithmic scale. The constant can be adjusted to control the scaling.
log_distances = np.log(adjusted_distances)

# Normalize the log-scaled distances
log_min_distance = np.min(log_distances)
log_max_distance = np.max(log_distances)
log_normalized_distances = (log_distances - log_min_distance) / (log_max_distance - log_min_distance)

# Convert normalized distances to RGB colors (let's use the Viridis colormap)
# colors = (plt.cm.viridis(normalized_distances) * 255).astype(np.uint8)
colors = (plt.cm.hot(1 - log_normalized_distances) * 255).astype(np.uint8)

# Prepare data for PlyElement
vertex_data = np.zeros(gaussian_centers.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

vertex_data['x'] = gaussian_centers[:, 0]
vertex_data['y'] = gaussian_centers[:, 1]
vertex_data['z'] = gaussian_centers[:, 2]
vertex_data['red'] = colors[:, 0]
vertex_data['green'] = colors[:, 1]
vertex_data['blue'] = colors[:, 2]

# Create PlyElement and PlyData
vertex_element = PlyElement.describe(vertex_data, 'vertex')
ply_data = PlyData([vertex_element], text=True)

# Save to file
ply_data.write('colored_gaussian_centers.ply')
