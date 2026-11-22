#include <math.h>

#include "rope.h"
#include "utils.h"

// One thread per (even, odd) pair of q; pairs with i < kv_dim also rotate k.
__global__ void rotary_embedding_kernel(float *q, float *k, int pos, int dim, int kv_dim, int head_size) {
    int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= dim) return;
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1;  // 2 = q & k, 1 = q only
    for (int v = 0; v < rotn; v++) {
        float *vec = v == 0 ? q : k;
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

void rotary_embedding_gpu(float *q, float *k, int pos, int dim, int kv_dim, int head_size) {
    rotary_embedding_kernel<<<DIVUP(dim / 2, NAIVE_BLOCK), NAIVE_BLOCK>>>(q, k, pos, dim, kv_dim, head_size);
    CHECK_LAUNCH();
}

// CPU reference (the RoPE loop inside forward() in run.c)
void rotary_embedding_cpu(float *q, float *k, int pos, int dim, int kv_dim, int head_size) {
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1;
        for (int v = 0; v < rotn; v++) {
            float *vec = v == 0 ? q : k;
            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}
