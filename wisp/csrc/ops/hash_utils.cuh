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
#ifndef _HASH_UTILS_CUH_
#define _HASH_UTILS_CUH_

typedef unsigned int uint;
namespace wisp {

static __inline__ __device__ int32_t 
hash_index_3d(
    const int3 pos,
    const int32_t resolution,
    const int32_t codebook_size
){
    int32_t index = 0;

    constexpr uint32_t primes[3] = { 1u, 2654435761u, 805459861u };

    if (resolution < codebook_size && 
        resolution * resolution < codebook_size && 
        resolution * resolution * resolution < codebook_size) {
        index = pos.x + 
                pos.y * resolution + 
                pos.z * resolution * resolution;
    } else {
        index = ((pos.x * primes[0]) ^
                 (pos.y * primes[1]) ^
                 (pos.z * primes[2])) % codebook_size;
    }
    return index;
}

static __inline__ __device__ int32_t 
hash_index_3d_alt(
    const int3 pos,
    const int32_t resolution,
    const int32_t codebook_size
){
    int32_t index = 0;

    constexpr uint32_t primes[3] = { 3674653429u, 2097192037u, 1434869437u };

    if (resolution < codebook_size && 
        resolution * resolution < codebook_size && 
        resolution * resolution * resolution < codebook_size) {
        index = pos.x + 
                pos.y * resolution + 
                pos.z * resolution * resolution;
    } else {
        index = ((pos.x * primes[0]) ^
                 (pos.y * primes[1]) ^
                 (pos.z * primes[2])) % codebook_size;
    }
    return index;
}

static __inline__ __device__ int32_t 
    hash_index_2d(
        const int2 pos,
        const int32_t resolution_x,
        const int32_t resolution_y,
        const int32_t codebook_size
    )
    {
        int32_t index = 0;
        constexpr uint32_t primes[3] = { 1u, 2654435761u };

        if (resolution_x < codebook_size && resolution_x * resolution_y < codebook_size)
            index = pos.x + pos.y * resolution_x;
        else
            index = ((pos.x * primes[0]) ^ (pos.y * primes[1])) % codebook_size;
        return index;
    }

static __inline__ __device__ int32_t 
    hash_2d(
        const int2 pos,
        const int32_t resolution_x,
        const int32_t resolution_y,
        const int32_t codebook_size
    )
    {
        int32_t index = 0;
        constexpr uint32_t primes[3] = { 1u, 2654435761u };
        index = ((pos.x * primes[0]) ^ (pos.y * primes[1])) % codebook_size;
        return index;
    }

static __inline__ __device__ int32_t 
hash_index_2d_alt(
    const int2 pos,
    const int32_t resolution_x,
    const int32_t resolution_y,
    const int32_t codebook_size
){
    int32_t index = 0;

    constexpr uint32_t primes[3] = { 3674653429u, 2097192037u };
    index = ((pos.x * primes[0]) ^
                (pos.y * primes[1])) % codebook_size;
    return index;
}


static __inline__ __device__ float 
clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

}
#endif
