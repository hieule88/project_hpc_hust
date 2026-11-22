#pragma once
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define DIVUP(x, y) (((x) + (y) - 1) / (y))

// Threads per block for element-wise / one-thread-per-output kernels.
#define NAIVE_BLOCK 256
// Threads per block for kernels that reduce a whole vector (rmsnorm, softmax).
// Must be a power of two: block_reduce_* below is a plain shared-memory tree.
#define REDUCE_BLOCK 256

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t status_ = (call);                                             \
        if (status_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error (%s:%d): %s: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorName(status_), cudaGetErrorString(status_));      \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Check a kernel launch (configuration errors show up here; execution errors show
// up at the next synchronizing call).
#define CHECK_LAUNCH() CHECK_CUDA(cudaGetLastError())

// Naive block reduction: every thread writes its value to shared memory, then the
// active half of the block adds the other half until one value is left.
// `sdata` must hold blockDim.x floats. All threads get the result.
__device__ inline float block_reduce_sum(float val, float *sdata) {
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float result = sdata[0];
    __syncthreads();  // sdata may be reused right after
    return result;
}

__device__ inline float block_reduce_max(float val, float *sdata) {
    int tid = threadIdx.x;
    sdata[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float result = sdata[0];
    __syncthreads();
    return result;
}
