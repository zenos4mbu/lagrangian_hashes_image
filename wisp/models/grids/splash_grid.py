# The MIT License (MIT)
# 
# Copyright (c) 2023, Towaki Takikawa.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from typing import Dict, Set, Any, Type, List, Tuple
import torch
import torch.nn as nn
import numpy as np
import kaolin.ops.spc as spc_ops
from wisp.accelstructs import BaseAS
from wisp.models.grids import HashGrid
from wisp.models.grids.utils import MultiTable, SplashTable

import wisp.ops.grid as grid_ops

class SplashGrid(HashGrid):
    def __init__(self,
        blas               : BaseAS,
        feature_dim        : int, # doesn't do anything right now
        resolutions        : List[int],
        multiscale_type    : str = 'cat',  # options: 'cat', 'sum'
        feature_std        : float = 0.0,
        feature_bias       : float = 0.0,
        codebook_bitwidth  : int   = 8,
        coord_dim          : int   = 3,  # options: 2, 3
        num_splashes       : int   = 1, # probably broken
        use_cuda           : bool  = True,
        init_std_factor    : int = 1,
        final_std_factor    : int = 1,
        normalize          : bool  = True,
        base_lod           : int   = 0
    ):
        # Occupancy Structure
        super().__init__(
            blas=blas, 
            feature_dim=feature_dim, 
            resolutions=resolutions, 
            multiscale_type=multiscale_type, 
            feature_std=feature_std, 
            feature_bias=feature_bias, 
            codebook_bitwidth=codebook_bitwidth,
            coord_dim=coord_dim)
        
        del self.codebook
        self.use_cuda = use_cuda
        self.normalize = normalize
        self.base_lod = base_lod
        self.coord_dim = coord_dim
        self.num_splashes = num_splashes
        self.init_std_factor = init_std_factor
        self.final_std_factor = final_std_factor
        print("Num Splashes:", num_splashes)
        
        # if self.use_cuda:
        self.codebook = SplashTable(resolutions, self.coord_dim, self.feature_dim, self.feature_std, self.codebook_size, self.num_splashes, self.init_std_factor, self.final_std_factor)
        
        init_params = torch.randn(self.codebook.total_feats, self.feature_dim) * feature_std
        init_params_pos = torch.rand(self.codebook.total_pos, 2) * 2.0 - 1.0
        
        self.codebook.feats = nn.Parameter(init_params)
        self.codebook.pos = nn.Parameter(init_params_pos)

    def interpolate(self, coords, lod_idx):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3] or [batch, 3]
                For some grid implementations, specifying num_samples may allow for slightly faster trilinear
                interpolation. HashGrid doesn't use this optimization, but allows this input type for compatability.
            lod_idx  (int): int specifying the index to ``active_lods``

        Returns:
            (torch.FloatTensor): interpolated features of shape
             [batch, num_samples, feature_dim] or [batch, feature_dim]
        """
        # Remember desired output shape
        output_shape = coords.shape[:-1]
        if coords.ndim == 3:    # flatten num_samples dim with batch for cuda call
            batch, num_samples, coords_dim = coords.shape  # batch x num_samples
            coords = coords.reshape(batch * num_samples, coords_dim)

        # if self.use_cuda:
        feats = grid_ops.splashgrid(coords, self.codebook_bitwidth, self.codebook.num_splashes_lod, lod_idx, self.codebook, self.normalize)
        # else:
        # feats, gmms = grid_ops.interpolate(coords, self.codebook_bitwidth, self.num_splashes, lod_idx, self.use_cuda, self.codebook, self.normalize)
        # feats = grid_ops.splashgrid(coords, self.codebook_bitwidth, self.num_splashes, lod_idx, self.codebook)

        if self.multiscale_type == 'cat':
            feats = feats.reshape(*output_shape, feats.shape[-1])
            return feats#, gmms
        elif self.multiscale_type == 'sum':
            return feats.reshape(*output_shape, len(self.resolutions), feats.shape[-1] // len(self.resolutions)).sum(-2), gmms
        else:
            raise NotImplementedError

    @classmethod
    def from_geometric(cls,
                       blas               : BaseAS,
                       feature_dim        : int,
                       num_lods           : int,
                       multiscale_type    : str = 'sum',    # options: 'cat', 'sum'
                       feature_std        : float = 0.0,
                       feature_bias       : float = 0.0,
                       codebook_bitwidth  : int   = 8,
                       coord_dim          : int   = 2,      # options: 2, 3
                       min_grid_res       : int   = 16,
                       max_grid_res       : int   = None,
                       init_std_factor       : int   = 11,
                       final_std_factor       : int   = 1,
                       num_splashes       : int   = None,
                       use_cuda          : bool  = True,
                       normalize          : bool  = True,
                       base_lod           : int   = 0):
        b = np.exp((np.log(max_grid_res) - np.log(min_grid_res)) / (num_lods-1))
        resolutions = [int(1 + np.floor(min_grid_res*(b**l))) for l in range(num_lods)]
        # resolutions = [2**L for L in range(base_lod, base_lod + num_lods)]
        return cls(blas=blas, feature_dim=feature_dim, resolutions=resolutions, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, coord_dim=coord_dim, codebook_bitwidth=codebook_bitwidth,
                   num_splashes=num_splashes, use_cuda=use_cuda, normalize = normalize, init_std_factor = init_std_factor, final_std_factor = final_std_factor)


