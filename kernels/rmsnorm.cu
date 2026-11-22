#include "rmsnorm.h"

#include <cuda_runtime.h>
#include <math.h>

#include "utils.h"

__global__ void rmsnorm_kernel(float *__restrict__ o, const float *__restrict__  x, const float *__restrict__ weight, int size) {
    const int thread_id = threadIdx.x;
    const int block_size = blockDim.x;
    const int req_id = blockIdx.x;

    float ss = 0.0F;
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
        ss += x[req_id * size + elem_id] * x[req_id * size + elem_id];

    ss = blockReduceSum(ss);

    // serialization point to calculate normalization factor
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // normalize and scale
    for (int i = thread_id; i < size; i += block_size) {
        int elem_id = req_id * size + i;
        o[elem_id] = weight[i] * (ss * x[elem_id]);
    }
}

void rmsnorm_gpu(float *o, float *x, float *weight, int size, int batch_size, cudaStream_t stream) {
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(batch_size);
    rmsnorm_kernel<<<gridDim, blockDim, 0, stream>>>(o, x, weight, size);
    CHECK_CUDA(cudaGetLastError());
}


void rmsnorm_cpu(float *o, float *x, float *weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}
