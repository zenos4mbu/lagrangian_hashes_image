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

#pragma once

#include <ATen/ATen.h>
#include <vector>

namespace wisp {

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
    bool normalize);

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
    bool require_grad_coords);

at::Tensor hashgrid_interrpolate_cuda(
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor codebook_first_idx,
    at::Tensor resolution,
    int32_t codebook_bitwidth);

std::vector<at::Tensor> hashgrid_interpolate_backward_cuda(
    at::Tensor coords,
    at::Tensor grad_output,
    at::Tensor codebook,
    at::Tensor codebook_first_idx,
    at::Tensor resolution,
    int32_t codebook_bitwidth,
    int32_t feature_dim,
    bool require_grad_coords);

}

