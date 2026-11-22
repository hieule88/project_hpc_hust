#include "softmax.h"

#include <float.h>
#include <hip/hip_runtime.h>
#include <math.h>

#include "utils.h"

__global__ void softmax_kernel_wrapper(float *x, int size) {
    softmax_kernel(x, size);
}

void softmax_gpu(float *x, int size) {
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(1);
    hipLaunchKernelGGL(softmax_kernel_wrapper, gridDim, blockDim, 0, 0, x, size);
    CHECK_HIP(hipGetLastError());
}

void softmax_gpu_test(float *x, int size) {
    float *x_d;

    CHECK_HIP(hipMalloc(&x_d, sizeof(float) * size));

    CHECK_HIP(hipMemcpyHtoD(x_d, x, sizeof(float) * size));

    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(1);
    hipLaunchKernelGGL(softmax_kernel_wrapper, gridDim, blockDim, 0, 0, x_d, size);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipMemcpyDtoH(x, x_d, sizeof(float) * size));
    CHECK_HIP(hipFree(x_d));
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
