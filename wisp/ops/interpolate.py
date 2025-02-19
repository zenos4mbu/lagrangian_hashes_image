import torch
import torch.nn.functional as F
import math

from .hash import hash_index, hash_index_2d

def get_corners(coords, codebook_size, resolution):
    num_coords, coord_dim = coords.shape
    assert coord_dim == 2, "Coordinates should be 2D"

    x = torch.clamp(resolution * (coords * 0.5 + 0.5), 0.0, float(resolution - 1 - 1e-4))
    pos = torch.floor(x).long()
    x_ = x - pos
    _x = 1.0 - x_

    coeffs = torch.empty([num_coords, 4], device=coords.device)
    coeffs[:, 0] = _x[:, 0] * _x[:, 1]
    coeffs[:, 1] = _x[:, 0] * x_[:, 1]
    coeffs[:, 2] = x_[:, 0] * _x[:, 1]
    coeffs[:, 3] = x_[:, 0] * x_[:, 1]

    corners = torch.empty([num_coords, 4, coord_dim], device=coords.device).long()
    for k in range(4):
        corners[:, k, 0] = pos[:, 0] + ((k & 2) >> 1)
        corners[:, k, 1] = pos[:, 1] + ((k & 1) >> 0)

    corner_idx = hash_index_2d(corners, resolution, codebook_size)
    return corners, corner_idx, coeffs

def get_corners_3d(coords, codebook_size, resolution):

    num_coords, coord_dim = coords.shape
    
    x = torch.clamp(resolution * (coords * 0.5 + 0.5), 0.0, float(resolution-1-1e-5))
    pos = torch.floor(x).long()
    x_ = x - pos
    _x = 1.0 - x_

    coeffs = torch.empty([num_coords, 8], device=coords.device)
    coeffs[:, 0] = _x[:, 0] * _x[:, 1] * _x[:, 2]
    coeffs[:, 1] = _x[:, 0] * _x[:, 1] * x_[:, 2]
    coeffs[:, 2] = _x[:, 0] * x_[:, 1] * _x[:, 2]
    coeffs[:, 3] = _x[:, 0] * x_[:, 1] * x_[:, 2]
    coeffs[:, 4] = x_[:, 0] * _x[:, 1] * _x[:, 2]
    coeffs[:, 5] = x_[:, 0] * _x[:, 1] * x_[:, 2]
    coeffs[:, 6] = x_[:, 0] * x_[:, 1] * _x[:, 2]
    coeffs[:, 7] = x_[:, 0] * x_[:, 1] * x_[:, 2]

    corners = torch.empty([num_coords, 8, coord_dim], device=coords.device).long()
    for k in range(8):
        corners[:, k, 0] = pos[:, 0] + ((k & 4) >> 2)
        corners[:, k, 1] = pos[:, 1] + ((k & 2) >> 1)
        corners[:, k, 2] = pos[:, 2] + ((k & 1) >> 0)
    
    corner_idx = hash_index(corners, resolution, codebook_size)
    return corners, corner_idx, coeffs

def pytorch_interpolate(coords, codebook, means, stds, first_idx, first_idx_pos, resolutions, codebook_bitwidth, num_splashes, normalize):
    num_coords, coord_dim = coords.shape
    feature_dim = codebook.shape[-1]
    codebook_size = pow(2, codebook_bitwidth)
    num_lods = len(resolutions)
    feats = torch.empty([num_coords, feature_dim*num_lods], device=coords.device)
    gmms = torch.zeros([num_coords, int((num_splashes>0).sum())], device=coords.device)
    j = 0

    # codebook_divided = []
    # meanstd_divided = []

    for i in range(num_lods):
        # For codebook
        start_idx_codebook = first_idx[i]
        end_idx_codebook = first_idx[i + 1]
        sub_codebook = codebook[start_idx_codebook:end_idx_codebook]
        # codebook_divided.append(sub_codebook)
        
        # For mean_stds
        start_idx_meanstd = first_idx_pos[i]
        end_idx_meanstd = first_idx_pos[i + 1]
        sub_means = means[start_idx_meanstd:end_idx_meanstd]
        sub_stds = stds[start_idx_meanstd:end_idx_meanstd]
        # meanstd_divided.append(sub_meanstd)

        resolution = int(resolutions[i])
        _, corner_idx, coeffs = get_corners(coords, codebook_size, resolution)
        corner_idx = corner_idx.view(-1)# + first_idx[i]                                 # [num_coords*4]
        corner_idx_pos = corner_idx.view(-1)# + first_idx_pos[i]                         # [num_coords*4]
        coeffs = coeffs.view(num_coords, 4, 1, 1)                                       # [num_coords, 4, 1, 1]
        num_splash = int(num_splashes[i])  

        if num_splash:
            sub_means = sub_means.view(-1, num_splash, 2)                                         # [codebook_size, num_splashes - 1, 2]
            sub_stds = sub_stds.view(-1, num_splash, 1)                                           # [codebook_size, num_splashes - 1, 3]
            sub_codebook = sub_codebook.view(-1, num_splash, feature_dim)                             # [codebook_size, num_splashes, feature_dim]
            mean_out = torch.index_select(sub_means, dim=0, index=corner_idx_pos)                       # [num_coords*4, num_splashes - 1, 2]
            std_out = torch.index_select(sub_stds, dim=0, index=corner_idx_pos)                         # [num_coords*4, num_splashes - 1, 3]
            mean = mean_out.view(num_coords, 4, num_splash,2)                                     # [num_coords, 4, num_splashes, 2]
            std = std_out.view(num_coords, 4, num_splash, 1)                                      # [num_coords, 4, num_splashes, 3]
            std = torch.abs(std)                                                                        # [num_coords, 4, num_splashes, 3]
            
            coords_mod = coords.view(num_coords, 1, 1, coord_dim)
            diff = coords_mod - mean
            sq_dist = torch.div(torch.pow(diff, 2), 2*torch.pow(std, 2) + 1e-7)             # [num_coords, 8, num_splash, 3]
            sq_dist = torch.sum(sq_dist, dim=-1, keepdim=True)                              # [num_coords, 8, num_splash, 1]
            gau_weights = torch.exp(-1 * sq_dist)                                           # [num_coords, 8, num_splash, 1]
            gau_norm = math.sqrt(2 * math.pi) * std
            gau_weights = torch.div(gau_weights, gau_norm + 1e-7)

            norm = 2.0
            dist_weights = torch.pow(torch.abs(diff/std), norm)
            dist_weights = torch.sum(dist_weights, dim=-1, keepdim=True)
            gmm, _ = torch.min((dist_weights * coeffs).view(num_coords, 4*num_splash, 1), dim=-2)
            gmms[:, j] = gmm.squeeze()                                                       # [num_coords, 8, num_splash, 1]
            j += 1
        
            feat = torch.index_select(sub_codebook, dim=0, index=corner_idx)                    # [num_coords*4, num_splashes, feature_dim]
            feat = feat.view(num_coords, 4, num_splash, feature_dim)                      # [num_coords, 4, num_splashes, feature_dim]

            feat_comp = feat * gau_weights * coeffs   
            feat_comp = torch.sum(feat_comp, dim=[1, 2])                                  # [num_coords, feature_dim]
        else:
            feat = torch.index_select(sub_codebook, dim=0, index=corner_idx)                    # [num_coords*4, feature_dim]
            feat = feat.view(num_coords, 4, feature_dim)
            feat_comp = torch.sum(feat * coeffs.squeeze(-1), dim=1)                                    # [num_coords, feature_dim]

        feats[:, feature_dim*i:feature_dim*(i+1)] = feat_comp

    return feats, gmms

def pytorch_interpolate_3d(coords, codebook, mean_stds, first_idx, first_idx_pos, resolutions, codebook_bitwidth, num_splashes, normalize):
    num_coords, coord_dim = coords.shape
    feature_dim = codebook.shape[-1]
    codebook_size = pow(2, codebook_bitwidth)
    num_lods = len(resolutions)
    feats = torch.empty([num_coords, feature_dim*num_lods], device=coords.device)

    # codebook_divided = []
    # meanstd_divided = []

    for i in range(num_lods):
        # For codebook
        start_idx_codebook = first_idx[i]
        end_idx_codebook = first_idx[i + 1]
        sub_codebook = codebook[start_idx_codebook:end_idx_codebook]
        # codebook_divided.append(sub_codebook)
        
        # For mean_stds
        start_idx_meanstd = first_idx_pos[i]
        end_idx_meanstd = first_idx_pos[i + 1]
        sub_meanstd = mean_stds[start_idx_meanstd:end_idx_meanstd]
        # meanstd_divided.append(sub_meanstd)

    # for i in range(num_lods):
        resolution = int(resolutions[i])
        _, corner_idx, coeffs = get_corners_3d(coords, codebook_size, resolution)
        corner_idx = corner_idx.view(-1)# + first_idx[i]                                 # [num_coords*4]
        corner_idx_pos = corner_idx.view(-1)# + first_idx_pos[i]                         # [num_coords*4]
        coeffs = coeffs.view(num_coords, 8, 1, 1)                                       # [num_coords, 8, 1, 1]
        if resolution < codebook_size * num_splashes and pow(resolution, 2) and pow(resolution, 3) < codebook_size * num_splashes:
            # codebook = sub_codebook                      # [codebook_size, num_splashes, feature_dim]
            feat = torch.index_select(sub_codebook, dim=0, index=corner_idx)                    # [num_coords*4, feature_dim]
            feat = feat.view(num_coords, 8, feature_dim)
            feat_comp = torch.sum(feat * coeffs.squeeze(-1), dim=1)                                    # [num_coords, feature_dim]
        else:
            sub_meanstd = sub_meanstd.view(-1, num_splashes - 1, 4)                                     # [codebook_size, num_splashes - 1, 4]
            sub_codebook = sub_codebook.view(-1, num_splashes, feature_dim)                            # [codebook_size, num_splashes - 1, feature_dim]
            mean_std_out = torch.index_select(sub_meanstd, dim=0, index=corner_idx_pos)                    # [num_coords*4, num_splashes, 4]
            mean_std_reshaped = mean_std_out.view(num_coords, 8, num_splashes - 1, 4)                        # [num_coords, 4, num_splashes, 4]
            mean = mean_std_reshaped[..., 1:]                                                       # [num_coords, 4, num_splashes, 3]
            std = mean_std_reshaped[..., 0].unsqueeze(-1)                                           # [num_coords, 4, num_splashes, 1]
            std = torch.abs(std)
            coords_mod = coords.view(num_coords, 1, 1, coord_dim)
            sq_dist = torch.sum(torch.pow(coords_mod - mean, 2), dim=-1, keepdim=True)      # [num_coords, 4, num_splashes-1, 1]
            gau_weights = torch.exp(torch.div(-1 * sq_dist, 2 * torch.pow(std, 2)))  # [num_coords, 4, num_splashes -1, 1]
            epsilon = 1e-7

            feat = torch.index_select(sub_codebook, dim=0, index=corner_idx)                    # [num_coords*4, num_splashes, feature_dim]
            feat = feat.view(num_coords, 8, num_splashes, feature_dim)                      # [num_coords, 4, num_splashes, feature_dim]

            if normalize:
                norm_factor = torch.sum(gau_weights, dim=-2, keepdim=True)                      # [num_coords, 4, 1, 1]
                gau_weights_norm = torch.div(gau_weights, norm_factor + epsilon)                        # [num_coords, 4, num_splashes - 1, 1]
                sum_gau = torch.sum(gau_weights_norm, dim=-2)                                 # [num_coords, 4, feature_dim]
                unif_weight = 1.0 - sum_gau
                sum_gau = torch.sum(sum_gau, dim = -2)
                gaunif_weight = torch.cat([gau_weights_norm, unif_weight.unsqueeze(-1)], dim=-2)   # [num_coords, 4, num_splashes, 1]
                feat_comp = torch.sum(feat * gaunif_weight * coeffs , dim=[1,2])                                    # [num_coords, num_splashes, feature_dim]
            else:
                feat_comp = feat[:,:,:-1] * gau_weights * coeffs                                         # [num_coords, 4, num_splashes, feature_dim]
                sum_gau = torch.sum(gau_weights, dim=[-1,-2])
                feat_uniform = feat[:,:,-1].unsqueeze(-2) * epsilon * coeffs  
                feat_comp = torch.cat([feat_comp, feat_uniform], dim=-2)                           # [num_coords, 4, num_splashes+1, feature_dim]
                feat_comp = torch.sum(feat_comp, dim=[1, 2])                                    # [num_coords, feature_dim]
        feats[:, feature_dim*i:feature_dim*(i+1)] = feat_comp

    return feats