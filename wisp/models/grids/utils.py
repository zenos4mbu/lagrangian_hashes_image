# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
from typing import Tuple, Optional

class MultiTable(nn.Module):
    """Class that holds multiresolution grid tables.
    """

    def __init__(
        self, 
        resolutions : Tuple[int, ...], 
        coord_dim   : int, 
        feature_dim : int, 
        std         : float             = 0.01, 
        max_feats   : Optional[int]     = None, 
    ):
        """
        Args:
            resolutions (List[int, ...]): The resolutions in the multiresolution hierarchy.
            coord_dim (int): The coordinate dimension for the grid.
            feature_dim (int): The feature dimension for the grid.
            std (float): The standard deviation for the features.
            max_feats (Optional[int]): The max number of features (when in use for hash grids, for example)
        """
        super().__init__()

        self.num_lods = len(resolutions)
        self.max_feats = max_feats
        
        self.register_buffer("begin_idxes", torch.zeros(self.num_lods+1, dtype=torch.int64))
        self.register_buffer("num_feats", torch.zeros(self.num_lods, dtype=torch.int64))

        self.coord_dim = coord_dim
        self.feature_dim = feature_dim

        self.resolutions = torch.zeros([self.num_lods, 1], dtype=torch.int64)
        for i in range(len(resolutions)):
            self.resolutions[i] = resolutions[i]
        
        num_so_far = 0
        for i in range(self.num_lods):
            resolution = self.resolutions[i]
            num_feats_level = resolution[0] ** self.coord_dim
            
            if self.max_feats:
                num_feats_level = min(self.max_feats, num_feats_level)
            
            self.begin_idxes[i] = num_so_far
            self.num_feats[i] = num_feats_level
            num_so_far += num_feats_level

        self.begin_idxes[self.num_lods] = num_so_far

        self.total_feats = sum(self.num_feats)
        self.feats = nn.Parameter(torch.randn(self.total_feats, self.feature_dim) * std)

    def get_level(self, idx):
        """Gets the features for a specific level.

        Args:
            idx (int): The level of the multiresolution grid to get.
        """
        return self.feats[self.begin_idxes[idx]:self.begin_idxes[idx+1]]
    
class SplashTable(nn.Module):
    """Class that holds multiresolution grid tables.
    """

    def __init__(
        self, 
        resolutions : Tuple[int, ...], 
        coord_dim   : int, 
        feature_dim : int, 
        std         : float             = 0.01, 
        max_feats   : Optional[int]     = None,
        num_splashes: int               = 1, 
        init_std_factor: int            = 1,
        final_std_factor: int           = 1,

    ):
        """
        Args:
            resolutions (List[int, ...]): The resolutions in the multiresolution hierarchy.
            coord_dim (int): The coordinate dimension for the grid.
            feature_dim (int): The feature dimension for the grid.
            std (float): The standard deviation for the features.
            max_feats (Optional[int]): The max number of features (when in use for hash grids, for example)
        """
        super().__init__()

        self.num_lods = len(resolutions)
        self.num_splashes = num_splashes
        self.max_feats = max_feats
        self.init_std_factor = init_std_factor
        self.final_std_factor = final_std_factor

        self.std_decay_factor = (self.final_std_factor / self.init_std_factor) ** (1/500)
        
        self.register_buffer("begin_idxes", torch.zeros(self.num_lods+1, dtype=torch.int64))
        self.register_buffer("begin_idxes_pos", torch.zeros(self.num_lods+1, dtype=torch.int64))
        self.register_buffer("num_feats", torch.zeros(self.num_lods, dtype=torch.int64))
        self.register_buffer("num_pos", torch.zeros(self.num_lods, dtype=torch.int64))

        self.coord_dim = coord_dim
        self.feature_dim = feature_dim

        self.resolutions = torch.zeros([self.num_lods, 1], dtype=torch.int64)
        self.num_splashes_lod = torch.zeros([self.num_lods, 1], dtype=torch.int64)

        # for i in range(len(resolutions)):
        #     self.resolutions[i] = resolutions[i]

        for i in range(self.num_lods):
            self.resolutions[i] = resolutions[i]
            if i == self.num_lods-1:
                self.num_splashes_lod[i] = num_splashes
            elif i >= 0.875*self.num_lods:
                self.num_splashes_lod[i] = num_splashes // 2
            # elif i >= 0.75*self.num_lods:
            #     self.num_splashes_lod[i] = num_splashes // 4
            else:
                self.num_splashes_lod[i] = 0
        print("Resolution of LoDs:", self.resolutions)


        num_so_far = 0
        pos_so_far = 0
        for i in range(self.num_lods):
            resolution = self.resolutions[i]
            num_feats_level = resolution[0] ** self.coord_dim
            if self.max_feats:
                num_index_level = min(self.max_feats, num_feats_level)
                pos_per_level = int(num_index_level * self.num_splashes_lod[i])
                feats_per_level = int(num_index_level * max(self.num_splashes_lod[i], 1))
                self.begin_idxes[i] = num_so_far
                self.begin_idxes_pos[i] = pos_so_far
                self.num_feats[i] = feats_per_level
                self.num_pos[i] = pos_per_level
                num_so_far += feats_per_level
                pos_so_far += pos_per_level

        self.begin_idxes[self.num_lods] = num_so_far
        self.begin_idxes_pos[self.num_lods] = pos_so_far

        self.total_feats = sum(self.num_feats)
        self.total_pos = sum(self.num_pos)
        self.var = torch.ones(self.total_pos, 1).cuda()
        self.init_std(init_std_factor)

    def init_std(self, std):
        for lod in range(self.num_lods):
            if self.num_splashes_lod[lod]:
                gau_size = std * 2 / self.resolutions[lod].to('cuda')
                self.var[self.begin_idxes_pos[lod]:self.begin_idxes_pos[lod+1]] *= gau_size

    def update_stds(self):
        self.vas = self.var * self.std_decay_factor

    def get_level(self, idx):
        """Gets the features for a specific level.

        Args:
            idx (int): The level of the multiresolution grid to get.
        """
        return self.feats[self.begin_idxes[idx]:self.begin_idxes[idx+1]]

