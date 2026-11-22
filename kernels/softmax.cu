#include "softmax.h"

#include <float.h>
#include <cuda_runtime.h>
#include <math.h>

#include "utils.h"

__global__ void softmax_kernel_wrapper(float *x, int size) {
    softmax_kernel(x, size);
}

void softmax_gpu(float *x, int size) {
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(1);
    softmax_kernel_wrapper<<<gridDim, blockDim, 0, 0>>>(x, size);
    CHECK_CUDA(cudaGetLastError());
}


void softmax_cpu(float *x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}
