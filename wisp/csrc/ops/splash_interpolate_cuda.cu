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

#include <iostream>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include "hash_utils.cuh"
#include <cmath>

namespace wisp {
typedef unsigned int uint;

template<typename scalar_t>
__global__ void
splash_interpolate_3d_cuda_kernel(
    const int64_t num_coords,
    const int32_t codebook_size,
    const int32_t num_splashes,
    const int64_t feature_dim,
    const int32_t resolution,
    const int32_t lod_idx,
    const int32_t num_lods,
    const float* __restrict__ coords,
    const scalar_t* __restrict__ codebook,
    const scalar_t* __restrict__ positions,
    const int64_t *codebook_first_idx,
    const int64_t *codebook_first_idx_pos,
    scalar_t* __restrict__ feats
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;

    codebook = codebook + codebook_first_idx[lod_idx] * 2;// * num_gaussian;
    positions = positions + codebook_first_idx_pos[lod_idx] * 4;// * num_gaussian;

    for (int64_t i=tidx; i<num_coords; i+=stride) { 
        
        float3 x = make_float3(clamp((resolution-1) * (coords[i*3+0] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp((resolution-1) * (coords[i*3+1] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp((resolution-1) * (coords[i*3+2] * 0.5 + 0.5), 0, resolution-1-1e-5));
        int3 pos = make_int3(floor(x.x), floor(x.y), floor(x.z));
        float3 x_ = make_float3(x.x - static_cast<float>(pos.x), 
                                x.y - static_cast<float>(pos.y), 
                                x.z - static_cast<float>(pos.z));
        float3 _x = make_float3(1.0 - x_.x, 1.0 - x_.y, 1.0 - x_.z);

        float coeffs[8];
        coeffs[0] = _x.x * _x.y * _x.z;
        coeffs[1] = _x.x * _x.y * x_.z;
        coeffs[2] = _x.x * x_.y * _x.z;
        coeffs[3] = _x.x * x_.y * x_.z;
        coeffs[4] = x_.x * _x.y * _x.z;
        coeffs[5] = x_.x * _x.y * x_.z;
        coeffs[6] = x_.x * x_.y * _x.z;
        coeffs[7] = x_.x * x_.y * x_.z;
        
        int3 corners[8];
        int32_t corner_idx[8];
#       pragma unroll
        for (int j=0; j<8; ++j) {
            corners[j].x = pos.x + ((j & 4) >> 2);
            corners[j].y = pos.y + ((j & 2) >> 1);
            corners[j].z = pos.z + ((j & 1) >> 0);
            corner_idx[j] = hash_index_3d(corners[j], resolution, codebook_size);
        }
        
        float feat0 = 0;
        float feat1 = 0;
        for (int k=0; k<8; ++k) {
#           pragma unroll
            for (int j=0; j<(num_splashes - 1); ++j) {
                // new initialization
                float mu0 = static_cast<float>(codebook[(corner_idx[k] * num_splashes + j) * 2 + 0]);
                float mu1 = static_cast<float>(codebook[(corner_idx[k] * num_splashes + j) * 2 + 1]);
                float var = static_cast<float>(positions[(corner_idx[k] * (num_splashes - 1) + j) * 4 + 0]);
                float gx  = static_cast<float>(positions[(corner_idx[k] * (num_splashes - 1) + j) * 4 + 1]);
                float gy  = static_cast<float>(positions[(corner_idx[k] * (num_splashes - 1) + j) * 4 + 2]);
                float gz  = static_cast<float>(positions[(corner_idx[k] * (num_splashes - 1) + j) * 4 + 3]);
                // take the absolute value of var
                var = abs(var);

                // check this math maybe
                float dist = pow(((static_cast<float>(x.x) / (resolution-1)) * 2.0) - 1.0 - gx, 2.0) +
                             pow(((static_cast<float>(x.y) / (resolution-1)) * 2.0) - 1.0 - gy, 2.0) +
                             pow(((static_cast<float>(x.z) / (resolution-1)) * 2.0) - 1.0 - gz, 2.0); 

                // compute Gaussian kernel
                float exp_dist = exp(- dist / (2 * var * var));

                feat0 += coeffs[k] * mu0 * exp_dist;
                feat1 += coeffs[k] * mu1 * exp_dist;
            }
            // last feature
            float mu0 = static_cast<float>(codebook[(corner_idx[k] * num_splashes + num_splashes - 1) * 2 + 0]);
            float mu1 = static_cast<float>(codebook[(corner_idx[k] * num_splashes + num_splashes - 1) * 2 + 1]);

            feat0 += coeffs[k] * mu0 * 1e-7;
            feat1 += coeffs[k] * mu1 * 1e-7;
        }
        feats[num_lods*i*2+2*lod_idx + 0] = static_cast<scalar_t>(feat0);
        feats[num_lods*i*2+2*lod_idx + 1] = static_cast<scalar_t>(feat1);
    }
} 


template<typename scalar_t>
__global__ void
splash_interpolate_3d_backward_cuda_kernel(
    const int64_t num_coords,
    const int32_t codebook_size,
    const int32_t num_splashes,
    const int64_t feature_dim,
    const int32_t resolution,
    const int32_t lod_idx,
    const int32_t num_lods,
    const bool require_grad_coords,
    const float* __restrict__ coords,
    const scalar_t* __restrict__ codebook,
    const scalar_t* __restrict__ positions,
    const int64_t *__restrict__ codebook_first_idx,
    const int64_t *__restrict__ codebook_first_idx_pos,
    const scalar_t* __restrict__ grad_output, // N, feature_dim*num_lods
    scalar_t* __restrict__ grad_codebook, // codebook_size, feature_dim
    scalar_t* __restrict__ grad_positions, // codebook_size, feature_dim
    float* __restrict__ grad_coords // N, 3
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;

    grad_codebook = grad_codebook + codebook_first_idx[lod_idx] * 2;// * num_gaussian;
    grad_positions = grad_positions + codebook_first_idx_pos[lod_idx] * 4;// * num_gaussian;
    codebook = codebook + codebook_first_idx[lod_idx] * 2;// * num_gaussian;
    positions = positions + codebook_first_idx_pos[lod_idx] * 4;// * num_gaussian;

    for (int64_t i=tidx; i<num_coords; i+=stride) { 
        
        float3 x = make_float3(clamp((resolution-1) * (coords[i*3+0] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp((resolution-1) * (coords[i*3+1] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp((resolution-1) * (coords[i*3+2] * 0.5 + 0.5), 0, resolution-1-1e-5));
        int3 pos = make_int3(floor(x.x), floor(x.y), floor(x.z));
        float3 x_ = make_float3(x.x - static_cast<float>(pos.x), 
                                x.y - static_cast<float>(pos.y), 
                                x.z - static_cast<float>(pos.z));
        float3 _x = make_float3(1.0 - x_.x, 1.0 - x_.y, 1.0 - x_.z);

        float coeffs[8];
        coeffs[0] = _x.x * _x.y * _x.z;
        coeffs[1] = _x.x * _x.y * x_.z;
        coeffs[2] = _x.x * x_.y * _x.z;
        coeffs[3] = _x.x * x_.y * x_.z;
        coeffs[4] = x_.x * _x.y * _x.z;
        coeffs[5] = x_.x * _x.y * x_.z;
        coeffs[6] = x_.x * x_.y * _x.z;
        coeffs[7] = x_.x * x_.y * x_.z;

        int3 corners[8];
        int32_t corner_idx[8];
#       pragma unroll
        for (int j=0; j<8; ++j) {
            corners[j].x = pos.x + ((j & 4) >> 2);
            corners[j].y = pos.y + ((j & 2) >> 1);
            corners[j].z = pos.z + ((j & 1) >> 0);
            corner_idx[j] = hash_index_3d(corners[j], resolution, codebook_size);
        }
#       pragma unroll
        for (int l=0; l<8; ++l) {
            for(int j=0; j<(num_splashes-1); j++){
                // make sure this indexing is correct
                float mu0 = static_cast<float>(codebook[(corner_idx[l] * num_splashes + j) * 2 + 0]);
                float mu1 = static_cast<float>(codebook[(corner_idx[l] * num_splashes + j) * 2 + 1]);
                float var = static_cast<float>(positions[(corner_idx[l] * (num_splashes - 1) + j) * 4 + 0]);
                float gx  = static_cast<float>(positions[(corner_idx[l] * (num_splashes - 1) + j) * 4 + 1]);
                float gy  = static_cast<float>(positions[(corner_idx[l] * (num_splashes - 1) + j) * 4 + 2]);
                float gz  = static_cast<float>(positions[(corner_idx[l] * (num_splashes - 1) + j) * 4 + 3]);
                // take the absolute value of var
                var = abs(var);

                // check this math maybe
                float distx = ((static_cast<float>(x.x) / (resolution-1)) * 2.0) - 1.0 - gx;
                float disty = ((static_cast<float>(x.y) / (resolution-1)) * 2.0) - 1.0 - gy;
                float distz = ((static_cast<float>(x.y) / (resolution-1)) * 2.0) - 1.0 - gz;

                float neg_dist = -0.5 * (pow(distx, 2) + pow(disty, 2) + pow(distz, 2));
                float exp_dist = exp(neg_dist / (var * var));

                // this is correct only with feature_dim = 2
                uint64_t _idx_mu0 = i * num_lods * 2 + lod_idx * 2 + 0;
                uint64_t _idx_mu1 = i * num_lods * 2 + lod_idx * 2 + 1;

                // Compute the gradiens for mu0 and mu1
                float grad_mu0 = grad_output[_idx_mu0] * coeffs[l];
                float grad_mu1 = grad_output[_idx_mu1] * coeffs[l];

                // Add gradients for Mu0 and Mu1
                atomicAdd((float*)(grad_codebook + (corner_idx[l] * num_splashes * 2 + j * 2 + 0)), grad_mu0 * exp_dist);
                atomicAdd((float*)(grad_codebook + (corner_idx[l] * num_splashes * 2 + j * 2 + 1)), grad_mu1 * exp_dist);

                // Gradient computation for x and y
                float grad_x = distx * exp_dist / (var * var);
                float grad_y = disty * exp_dist / (var * var);
                float grad_z = distz * exp_dist / (var * var);

                // Gradient computation for var
                float grad_var = - 2.0 * (neg_dist) * exp_dist / (var * var * var);

                // Complete gradients
                float grad_x_tot = grad_x * grad_mu0 * mu0 + grad_x * grad_mu1 * mu1;
                float grad_y_tot = grad_y * grad_mu0 * mu0 + grad_y * grad_mu1 * mu1;
                float grad_z_tot = grad_z * grad_mu0 * mu0 + grad_z * grad_mu1 * mu1;
                float grad_var_tot = grad_var * grad_mu0 * mu0 + grad_var * grad_mu1 * mu1;

                // Add gradients for x, y and var
                atomicAdd((float*)(grad_positions + (corner_idx[l] * (num_splashes - 1) * 4 + j * 4 + 0)), grad_var_tot);
                atomicAdd((float*)(grad_positions + (corner_idx[l] * (num_splashes - 1) * 4 + j * 4 + 1)), grad_x_tot);
                atomicAdd((float*)(grad_positions + (corner_idx[l] * (num_splashes - 1) * 4 + j * 4 + 2)), grad_y_tot);
                atomicAdd((float*)(grad_positions + (corner_idx[l] * (num_splashes - 1) * 4 + j * 4 + 3)), grad_z_tot);
            }
            // last splash with only feature
            float exp_dist = 1e-7;
            float mu0 = static_cast<float>(codebook[(corner_idx[l] * num_splashes + num_splashes - 1) * 2 + 0]);
            float mu1 = static_cast<float>(codebook[(corner_idx[l] * num_splashes + num_splashes - 1) * 2 + 1]);
            
            uint64_t _idx_mu0 = i * num_lods * 2 + lod_idx * 2 + 0;
            uint64_t _idx_mu1 = i * num_lods * 2 + lod_idx * 2 + 1;
            float grad_mu0 = grad_output[_idx_mu0] * coeffs[l];
            float grad_mu1 = grad_output[_idx_mu1] * coeffs[l];

            // Add gradients for Mu0 and Mu1
            atomicAdd((float*)(grad_codebook + (corner_idx[l] * num_splashes * 2 + (num_splashes - 1) * 2 + 0)), grad_mu0 * exp_dist);
            atomicAdd((float*)(grad_codebook + (corner_idx[l] * num_splashes * 2 + (num_splashes - 1) * 2 + 1)), grad_mu1 * exp_dist);
        }
    }
}

template<typename scalar_t>
__global__ void
splash_interpolate_2d_cuda_kernel(
    const int64_t num_coords,
    const int32_t codebook_size,
    const int32_t num_splashes,
    const int64_t feature_dim,
    const int32_t resolution,
    const int32_t lod_idx,
    const int32_t num_lods,
    const float* __restrict__ coords,
    const scalar_t* __restrict__ codebook,
    const scalar_t* __restrict__ positions,
    const scalar_t* __restrict__ variances,
    const int64_t *codebook_first_idx,
    const int64_t *codebook_first_idx_pos,
    scalar_t* __restrict__ feats
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    
    codebook = codebook + codebook_first_idx[lod_idx] * 2;// * num_gaussian;
    positions = positions + codebook_first_idx_pos[lod_idx] * 2;// * num_gaussian;
    variances = variances + codebook_first_idx_pos[lod_idx];// * num_gaussian;
    
    for (int64_t i=tidx; i<num_coords; i+=stride) { 
        
        float2 x = make_float2(clamp(resolution * (coords[i*2+0] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp(resolution * (coords[i*2+1] * 0.5 + 0.5), 0, resolution-1-1e-5));
        int2 pos = make_int2(floor(x.x), floor(x.y));
        float2 x_ = make_float2(x.x - static_cast<float>(pos.x), 
                                x.y - static_cast<float>(pos.y));
        float2 _x = make_float2(1.0 - x_.x, 1.0 - x_.y);

        float coeffs[4];
        coeffs[0] = _x.x * _x.y;
        coeffs[1] = _x.x * x_.y;
        coeffs[2] = x_.x * _x.y;
        coeffs[3] = x_.x * x_.y;
        
        int2 corners[4];
        int32_t corner_idx[4];
#       pragma unroll
        for (uint32_t j=0; j<4; ++j) {
            corners[j].x = pos.x + ((j & 2) >> 1);
            corners[j].y = pos.y + ((j & 1) >> 0);
            corner_idx[j] = hash_2d(corners[j], resolution, resolution, codebook_size);
        }
        
        float feat0 = 0;
        float feat1 = 0;
        for (int k=0; k<4; ++k) {
#           pragma unroll
            for (int j=0; j<num_splashes; ++j) {
                // new initialization
                float mu0 = static_cast<float>(codebook[(corner_idx[k] * num_splashes + j) * 2 + 0]);
                float mu1 = static_cast<float>(codebook[(corner_idx[k] * num_splashes + j) * 2 + 1]);
                float var = static_cast<float>(variances[(corner_idx[k] * num_splashes + j)]);
                float gx  = static_cast<float>(positions[(corner_idx[k] * num_splashes + j) * 2 + 0]);
                float gy  = static_cast<float>(positions[(corner_idx[k] * num_splashes + j) * 2 + 1]);
                // take the absolute value of var
                var = abs(var);

                float dist = pow(((static_cast<float>(x.x) / (resolution-1)) * 2.0) - 1.0 - gx, 2.0) +
                             pow(((static_cast<float>(x.y) / (resolution-1)) * 2.0) - 1.0 - gy, 2.0); 

                // compute Gaussian kernel
                float exp_dist = exp(- dist / (2 * var * var));

                // compute norm
                float gau_norm = sqrt(2 * M_PI * var);

                // final_weight
                float final_weight = exp_dist / gau_norm;

                feat0 += coeffs[k] * mu0 * final_weight;
                feat1 += coeffs[k] * mu1 * final_weight;
            }
        }
        feats[num_lods*i*2+2*lod_idx + 0] = static_cast<scalar_t>(feat0);
        feats[num_lods*i*2+2*lod_idx + 1] = static_cast<scalar_t>(feat1);
    }
} 


template<typename scalar_t>
__global__ void
splash_interpolate_2d_backward_cuda_kernel(
    const int64_t num_coords,
    const int32_t codebook_size,
    const int32_t num_splashes,
    const int64_t feature_dim,
    const int32_t resolution,
    const int32_t lod_idx,
    const int32_t num_lods,
    const bool require_grad_coords,
    const float* __restrict__ coords,
    const scalar_t* __restrict__ codebook,
    const scalar_t* __restrict__ positions,
    const scalar_t* __restrict__ variances,
    const int64_t *codebook_first_idx,
    const int64_t *codebook_first_idx_pos,
    const scalar_t* __restrict__ grad_output, // N, feature_dim*num_lods
    scalar_t* __restrict__ grad_codebook, // codebook_size, feature_dim
    scalar_t* __restrict__ grad_positions, // codebook_size, feature_dim
    float* __restrict__ grad_coords // N, 3
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    
    grad_codebook = grad_codebook + codebook_first_idx[lod_idx] * 2;// * num_gaussian;
    grad_positions = grad_positions + codebook_first_idx_pos[lod_idx] * 2;// * num_gaussian;
    codebook = codebook + codebook_first_idx[lod_idx] * 2;// * num_gaussian;
    positions = positions + codebook_first_idx_pos[lod_idx] *2;// * num_gaussian;
    variances = variances + codebook_first_idx_pos[lod_idx];// * num_gaussian;
    
    for (int64_t i=tidx; i<num_coords; i+=stride) { 
        float2 x = make_float2(clamp(resolution * (coords[i*2+0] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp(resolution * (coords[i*2+1] * 0.5 + 0.5), 0, resolution-1-1e-5));
        int2 pos = make_int2(floor(x.x), floor(x.y));
        float2 x_ = make_float2(x.x - static_cast<float>(pos.x), 
                                x.y - static_cast<float>(pos.y));
        float2 _x = make_float2(1.0 - x_.x, 1.0 - x_.y);

        float coeffs[4];
        coeffs[0] = _x.x * _x.y;
        coeffs[1] = _x.x * x_.y;
        coeffs[2] = x_.x * _x.y;
        coeffs[3] = x_.x * x_.y;
        
        int2 corners[4];
        int32_t corner_idx[4];
#       pragma unroll
        for (uint32_t j=0; j<4; ++j) {
            corners[j].x = pos.x + ((j & 2) >> 1);
            corners[j].y = pos.y + ((j & 1) >> 0);
            corner_idx[j] = hash_2d(corners[j], resolution, resolution, codebook_size);
        }
#       pragma unroll
        for (int l=0; l<4; ++l) {
            for(int j=0; j<num_splashes; j++){
                // make sure this indexing is correct
                float mu0 = static_cast<float>(codebook[(corner_idx[l] * num_splashes + j) * 2 + 0]);
                float mu1 = static_cast<float>(codebook[(corner_idx[l] * num_splashes + j) * 2 + 1]);
                float var = static_cast<float>(variances[(corner_idx[l] * num_splashes + j)]);
                float gx  = static_cast<float>(positions[(corner_idx[l] * num_splashes + j) * 2 + 0]);
                float gy  = static_cast<float>(positions[(corner_idx[l] * num_splashes + j) * 2 + 1]);

                // take the absolute value of var
                var = abs(var);

                // check this math maybe
                float distx = ((static_cast<float>(x.x) / (resolution-1)) * 2.0) - 1.0 - gx;
                float disty = ((static_cast<float>(x.y) / (resolution-1)) * 2.0) - 1.0 - gy;

                float neg_dist = -0.5 * (pow(distx, 2) + pow(disty, 2));
                float exp_dist = exp(neg_dist / (var * var));

                // compute norm
                float gau_norm = sqrt(2 * M_PI * var);

                // final_weight
                float final_weight = exp_dist / gau_norm;

                // this is correct only with feature_dim = 2
                uint64_t _idx_mu0 = i * num_lods * 2 + lod_idx * 2 + 0;
                uint64_t _idx_mu1 = i * num_lods * 2 + lod_idx * 2 + 1;

                // Compute the gradiens for mu0 and mu1
                float grad_mu0 = grad_output[_idx_mu0] * coeffs[l];
                float grad_mu1 = grad_output[_idx_mu1] * coeffs[l];

                // Add gradients for Mu0 and Mu1
                atomicAdd((float*)(grad_codebook + (corner_idx[l] * num_splashes * 2 + j * 2 + 0)), grad_mu0 * final_weight);
                atomicAdd((float*)(grad_codebook + (corner_idx[l] * num_splashes * 2 + j * 2 + 1)), grad_mu1 * final_weight);

                // Gradient computation for x and y
                float grad_x = distx * final_weight / (var * var);
                float grad_y = disty * final_weight / (var * var);

                // Complete gradients
                float grad_x_tot = grad_x * grad_mu0 * mu0 + grad_x * grad_mu1 * mu1;
                float grad_y_tot = grad_y * grad_mu0 * mu0 + grad_y * grad_mu1 * mu1;

                // Add gradients for x, y and var
                atomicAdd((float*)(grad_positions + (corner_idx[l] * num_splashes * 2 + j * 2 + 0)), grad_x_tot);
                atomicAdd((float*)(grad_positions + (corner_idx[l] * num_splashes * 2 + j * 2 + 1)), grad_y_tot);
            }
        }
    }
}

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
    at::Tensor feats){

    int num_threads = 512;
    
    if (coord_dim == 3) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats.type(), "splash_interpolate_3d_cuda", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats));
            auto stream = at::cuda::getCurrentCUDAStream();
            splash_interpolate_3d_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
                num_coords,
                codebook_size,
                num_splashes,
                feature_dim,
                resolution[0].item<int>(),
                lod_idx,
                num_lods,
                coords.data_ptr<float>(),
                codebook.data_ptr<scalar_t>(),
                positions.data_ptr<scalar_t>(),
                codebook_first_idx.data_ptr<int64_t>(),
                codebook_first_idx_pos.data_ptr<int64_t>(),
                feats.data_ptr<scalar_t>()
            );
        }));
    } else if (coord_dim == 2) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats.type(), "splash_interpolate_2d_cuda", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats));
            auto stream = at::cuda::getCurrentCUDAStream();
            splash_interpolate_2d_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
                num_coords,
                codebook_size,
                num_splashes,
                feature_dim,
                resolution[0].item<int>(),
                lod_idx,
                num_lods,
                coords.data_ptr<float>(),
                codebook.data_ptr<scalar_t>(),
                positions.data_ptr<scalar_t>(),
                variances.data_ptr<scalar_t>(),
                codebook_first_idx.data_ptr<int64_t>(),
                codebook_first_idx_pos.data_ptr<int64_t>(),
                feats.data_ptr<scalar_t>()
            );
        }));
    }
}

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
    at::Tensor grad_coords){

    int num_threads = 512;

    if (coord_dim == 3) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.type(), "splash_interpolate_3d_backward_cuda", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_codebook));
            auto stream = at::cuda::getCurrentCUDAStream();
            splash_interpolate_3d_backward_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
                num_coords,
                codebook_size,
                num_splashes,
                feature_dim,
                resolution[0].item<int>(),
                lod_idx,
                num_lods,
                require_grad_coords,
                coords.data_ptr<float>(),
                codebook.data_ptr<scalar_t>(),
                positions.data_ptr<scalar_t>(),
                codebook_first_idx.data_ptr<int64_t>(),
                codebook_first_idx_pos.data_ptr<int64_t>(),
                grad_output.data_ptr<scalar_t>(),
                grad_codebook.data_ptr<scalar_t>(),
                grad_positions.data_ptr<scalar_t>(),
                grad_coords.data_ptr<float>()
            );
        }));
    } else if (coord_dim == 2) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.type(), "splash_interpolate_2d_backward_cuda", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_codebook));
            auto stream = at::cuda::getCurrentCUDAStream();
            splash_interpolate_2d_backward_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
                num_coords,
                codebook_size,
                num_splashes,
                feature_dim,
                resolution[0].item<int>(),
                lod_idx,
                num_lods,
                require_grad_coords,
                coords.data_ptr<float>(),
                codebook.data_ptr<scalar_t>(),
                positions.data_ptr<scalar_t>(),
                variances.data_ptr<scalar_t>(),
                codebook_first_idx.data_ptr<int64_t>(),
                codebook_first_idx_pos.data_ptr<int64_t>(),
                grad_output.data_ptr<scalar_t>(),
                grad_codebook.data_ptr<scalar_t>(),
                grad_positions.data_ptr<scalar_t>(),
                grad_coords.data_ptr<float>()
            );
        }));
    }
}

} // namespace wisp