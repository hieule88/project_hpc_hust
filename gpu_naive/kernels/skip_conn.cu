#include "skip_conn.h"
#include "utils.h"

// One thread per element.
__global__ void residual_add_kernel(float *x, const float *y, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) x[i] += y[i];
}

void residual_add_gpu(float *x, const float *y, int size) {
    residual_add_kernel<<<DIVUP(size, NAIVE_BLOCK), NAIVE_BLOCK>>>(x, y, size);
    CHECK_LAUNCH();
}

// CPU reference (the residual loops inside forward() in run.c)
void residual_add_cpu(float *x, const float *y, int size) {
    for (int i = 0; i < size; i++) x[i] += y[i];
}
