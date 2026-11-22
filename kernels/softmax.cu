#include <math.h>

#include "softmax.h"
#include "utils.h"

// One block per row: max (tree reduction), exp, sum (tree reduction), normalize.
__global__ void softmax_kernel(float *x, int size, int stride) {
    __shared__ float sdata[REDUCE_BLOCK];
    float *row = x + (size_t)blockIdx.x * stride;

    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < size; i += blockDim.x) max_val = fmaxf(max_val, row[i]);
    max_val = block_reduce_max(max_val, sdata);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        row[i] = expf(row[i] - max_val);
        sum += row[i];
    }
    sum = block_reduce_sum(sum, sdata);

    for (int i = threadIdx.x; i < size; i += blockDim.x) row[i] /= sum;
}

void softmax_gpu(float *x, int rows, int size, int stride) {
    softmax_kernel<<<rows, REDUCE_BLOCK>>>(x, size, stride);
    CHECK_LAUNCH();
}

// CPU reference (run.c)
void softmax_cpu(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}
