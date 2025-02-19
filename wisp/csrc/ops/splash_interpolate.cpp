/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2023, Towaki Takikawa.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include <ATen/ATen.h>
#include <vector>
#include <iostream>

namespace wisp {

void hashgrid_interpolate_cuda_impl(
    int64_t num_coords, 
    int32_t codebook_size,
    int64_t feature_dim,
    at::Tensor resolution,
    int32_t lod_idx,
    int32_t num_lods,
    int32_t coord_dim,
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor codebook_first_idx,
    at::Tensor feats);

void hashgrid_interpolate_backward_cuda_impl(
    int64_t num_coords, 
    int32_t codebook_size,
    int64_t feature_dim,
    at::Tensor resolution,
    int32_t lod_idx,
    int32_t num_lods,
    int32_t coord_dim,
    bool require_grad_coords,
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor codebook_first_idx,
    at::Tensor grad_output,
    at::Tensor grad_codebook,
    at::Tensor grad_coords);

void splash_interpolate_cuda_impl(
    int64_t num_coords, 
    int32_t codebook_size,
    int32_t num_splashes,
    int64_t feature_dim,
    at::Tensor resolution,
    int32_t lod_idx,
    int32_t num_lods,
    int32_t coord_dim,
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor positions,
    at::Tensor variances,
    at::Tensor codebook_first_idx,
    at::Tensor codebook_first_idx_pos,
    at::Tensor feats);

void splash_interpolate_backward_cuda_impl(
    int64_t num_coords, 
    int32_t codebook_size,
    int32_t num_splashes,
    int64_t feature_dim,
    at::Tensor resolution,
    int32_t lod_idx,
    int32_t num_lods,
    int32_t coord_dim,
    bool require_grad_coords,
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor positions,
    at::Tensor variances,
    at::Tensor codebook_first_idx,
    at::Tensor codebook_first_idx_pos,
    at::Tensor grad_output,
    at::Tensor grad_codebook,
    at::Tensor grad_positions,
    at::Tensor grad_coords);

at::Tensor splash_interpolate_cuda(
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor positions,
    at::Tensor variances,
    at::Tensor codebook_first_idx,
    at::Tensor codebook_first_idx_pos,
    at::Tensor resolution,
    int32_t codebook_bitwidth,
    at::Tensor num_splashes,
    bool normalize) {
#ifdef WITH_CUDA
    int64_t num_coords = coords.size(0);  
    int64_t feature_dim = codebook.size(1);
    int32_t num_lods = resolution.size(0);
    int32_t coord_dim = coords.size(1);
    at::Tensor feats = at::empty({num_coords, feature_dim * int(resolution.size(0))}, codebook.options());

    int32_t codebook_size = pow(2, codebook_bitwidth);

    for (int32_t i=0; i < num_lods; ++i) {
        if (num_splashes[i].item<int32_t>() == 0) {
            hashgrid_interpolate_cuda_impl(num_coords, codebook_size, feature_dim, resolution[i], i, num_lods, coord_dim, coords, 
                                       codebook, codebook_first_idx, feats);
        }
        else
        {
            splash_interpolate_cuda_impl(num_coords, codebook_size, num_splashes[i].item<int32_t>(), feature_dim, resolution[i], i, num_lods, coord_dim, coords, 
                                    codebook, positions, variances, codebook_first_idx, codebook_first_idx_pos, feats);
        }
    }
    return feats;
#else
    AT_ERROR(__func__);
#endif  // WITH_CUDA
}

std::vector<at::Tensor> splash_interpolate_backward_cuda(
    at::Tensor coords,
    at::Tensor grad_output,
    at::Tensor codebook,
    at::Tensor positions,
    at::Tensor variances,
    at::Tensor codebook_first_idx,
    at::Tensor codebook_first_idx_pos,
    at::Tensor resolution,
    int32_t codebook_bitwidth,
    at::Tensor num_splashes,
    bool normalize,
    bool require_grad_coords) {
#ifdef WITH_CUDA
    int64_t num_coords = coords.size(0);  
    int32_t num_lods = resolution.size(0);
    int32_t coord_dim = coords.size(1);
    int32_t feature_dim = codebook.size(1);

    at::Tensor grad_codebook = at::zeros_like(codebook);
    at::Tensor grad_positions = at::zeros_like(positions);
    int32_t codebook_size = pow(2, codebook_bitwidth);

    at::Tensor grad_coords;
    if (require_grad_coords) {
        grad_coords = at::zeros({num_coords, 3}, coords.options());
    } else {
        grad_coords = at::empty({0}, coords.options());
    }
    for (int32_t i=0; i < num_lods; ++i) {
        if (num_splashes[i].item<int32_t>() == 0) {
            hashgrid_interpolate_backward_cuda_impl(num_coords, codebook_size, feature_dim, 
                resolution[i], i, num_lods, coord_dim, require_grad_coords,
                coords, codebook, codebook_first_idx, grad_output, grad_codebook, grad_coords);
        }
        else
        {
            splash_interpolate_backward_cuda_impl(num_coords, codebook_size, num_splashes[i].item<int32_t>(), feature_dim, 
                resolution[i], i, num_lods, coord_dim, require_grad_coords,
                coords, codebook, positions, variances, codebook_first_idx, codebook_first_idx_pos, grad_output, grad_codebook ,grad_positions, grad_coords);
        }
    }
    return {grad_coords, grad_codebook, grad_positions};
#else
    AT_ERROR(__func__);
#endif  // WITH_CUDA
}


}

