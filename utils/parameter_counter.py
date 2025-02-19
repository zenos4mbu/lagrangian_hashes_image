import numpy as np
# Given values
max_grid_res = 2048
min_grid_res = 16
num_lods = 16
max_entries_per_lod = 2**19  # Maximum number of entries for each LOD
feature_vector_dim = 2  # Dimension of feature vectors


# Calculating the base for the geometric progression
b = np.exp((np.log(max_grid_res) - np.log(min_grid_res)) / (num_lods - 1))

# Calculating the resolutions for each LOD
resolutions = [int(np.floor(min_grid_res * (b ** l))) for l in range(num_lods)]

# Calculate the number of parameters for each LOD
total_parameters = 0
for res in resolutions:
    num_entries = min(res**3, max_entries_per_lod)
    parameters_at_lod = num_entries * feature_vector_dim
    total_parameters += parameters_at_lod

print(total_parameters, resolutions)