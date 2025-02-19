import torch

def hash_index(coords, resolution, codebook_size):
    prime = [1, 2654435761, 805459861]

    if resolution < codebook_size and pow(resolution, 2) < codebook_size and pow(resolution, 3) < codebook_size:
        index = coords[..., 0] + coords[..., 1] * resolution + coords[..., 2] * pow(resolution, 2)
    else:
        index = ((coords[..., 0] * prime[0]) ^ (coords[..., 1] * prime[1]) ^ (coords[..., 2] * prime[2])) % codebook_size

    return index

def hash_index_2d(coords, resolution, codebook_size):
    prime = [1, 2654435761]
    
    if resolution < codebook_size and pow(resolution, 2) < codebook_size:
        # Using simple linear indexing if the resolution and codebook_size allow for it
        index = coords[..., 0] + coords[..., 1] * resolution
    else:
        # Using a hash function when linear indexing is not sufficient
        index = ((coords[..., 0] * prime[0]) ^ (coords[..., 1] * prime[1])) % codebook_size

    return index