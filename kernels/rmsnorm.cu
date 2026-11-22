#include <math.h>

#include "rmsnorm.h"
#include "utils.h"

// One block for the whole vector: each thread sums the squares of a strided slice,
// then a shared-memory tree reduction gives the total.
__global__ void rmsnorm_kernel(float *o, const float *x, const float *weight, int size) {
    __shared__ float sdata[REDUCE_BLOCK];
    float ss = 0.0f;
    for (int j = threadIdx.x; j < size; j += blockDim.x) {
        ss += x[j] * x[j];
    }
    ss = block_reduce_sum(ss, sdata);  // all reads of x happen before this barrier
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = threadIdx.x; j < size; j += blockDim.x) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void rmsnorm_gpu(float *o, const float *x, const float *weight, int size) {
    rmsnorm_kernel<<<1, REDUCE_BLOCK>>>(o, x, weight, size);
    CHECK_LAUNCH();
}

// CPU reference (run.c)
void rmsnorm_cpu(float *o, const float *x, const float *weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}
