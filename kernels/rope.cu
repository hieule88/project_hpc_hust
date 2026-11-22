#include "rope.h"

#include <cuda_runtime.h>
#include <math.h>

#include "types.h"
#include "utils.h"

__global__ void rotary_embedding_kernel(int pos, int dim, float *sq, float *sk, int kv_dim, int head_size) {
    for (int j = threadIdx.x; j < dim / 2; j += blockDim.x) {
        int i = j * 2;
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);  // accurate: angles grow with pos
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
            float *vec = v == 0 ? sq : sk;  // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

void rotary_embedding_gpu(RunState *s, int pos, int dim, int kv_dim, int head_size, cudaStream_t stream) {
    dim3 blockDim(MIN(dim / 2, BLOCK_SIZE * BLOCK_SIZE));
    dim3 gridDim(1);
    rotary_embedding_kernel<<<gridDim, blockDim, 0, stream>>>(pos, dim, s->q, s->k, kv_dim, head_size);
    CHECK_CUDA(cudaGetLastError());
}


void rotary_embedding_cpu(RunState *s, int pos, int dim, int kv_dim, int head_size) {
    for (int i = 0; i < dim; i += 2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        int rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
        for (int v = 0; v < rotn; v++) {
            float *vec = v == 0 ? s->q : s->k;  // the vector to rotate (query or key)
            float v0 = vec[i];
            float v1 = vec[i + 1];
            vec[i] = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}
