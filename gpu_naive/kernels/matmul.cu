#include "matmul.h"
#include "utils.h"

// Naive GEMV: one thread computes one output row, looping over the row like the CPU code.
// Adjacent threads read different rows (stride n), so global loads are not coalesced;
// with d = 768 there are only 3 blocks of 256 threads to spread over the whole GPU.
__global__ void matmul_kernel(float *xout, const float *x, const float *w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    float val = 0.0f;
    for (int j = 0; j < n; j++) {
        val += w[(size_t)i * n + j] * x[j];
    }
    xout[i] = val;
}

void matmul_gpu(float *xout, const float *x, const float *w, int n, int d) {
    matmul_kernel<<<DIVUP(d, NAIVE_BLOCK), NAIVE_BLOCK>>>(xout, x, w, n, d);
    CHECK_LAUNCH();
}

// CPU reference (run.c, without OpenMP)
void matmul_cpu(float *xout, const float *x, const float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[(size_t)i * n + j] * x[j];
        }
        xout[i] = val;
    }
}
