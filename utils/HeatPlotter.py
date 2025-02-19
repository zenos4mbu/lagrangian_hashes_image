from plyfile import PlyData, PlyElement
import numpy as np
from scipy.stats import gaussian_kde
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
gaussian_centers = read_obj("gaussian_positions_lv07_epoch_00201_ficus.obj")

# Calculate the density at each Gaussian center using KDE
values = gaussian_centers.T  # Transpose to fit gaussian_kde's input format
kde = gaussian_kde(values)
density = kde(values)

# Normalize the density estimates for coloring
min_density = np.min(density)
max_density = np.max(density)
normalized_density = (density - min_density) / (max_density - min_density)

# Convert normalized density to RGB colors (let's use the hot colormap for "heat" visualization)
# colors = (plt.cm.hot(1 - normalized_density) * 255).astype(np.uint8)
colors = (plt.cm.viridis( 1 - normalized_density) * 255).astype(np.uint8)

# Prepare data for PlyElement (similar to your existing code)
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
ply_data.write('FICUS_200.ply')