#include <math.h>

#include "swiglu.h"
#include "utils.h"

// One thread per element.
__global__ void swiglu_kernel(float *hb, const float *hb2, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float val = hb[i];
    val *= (1.0f / (1.0f + expf(-val)));  // silu(x) = x * sigmoid(x)
    val *= hb2[i];
    hb[i] = val;
}

void swiglu_gpu(float *hb, const float *hb2, int size) {
    swiglu_kernel<<<DIVUP(size, NAIVE_BLOCK), NAIVE_BLOCK>>>(hb, hb2, size);
    CHECK_LAUNCH();
}

// CPU reference (the SwiGLU loop inside forward() in run.c)
void swiglu_cpu(float *hb, const float *hb2, int size) {
    for (int i = 0; i < size; i++) {
        float val = hb[i];
        val *= (1.0f / (1.0f + expf(-val)));
        val *= hb2[i];
        hb[i] = val;
    }
}
