#include "swiglu.h"

#include <cuda_runtime.h>
#include <math.h>

#include "types.h"
#include "utils.h"

// F.silu(self.w1(x)) * self.w3(x)
__global__ void batch_mv_swiglu_kernel(float *__restrict__ out, const float *__restrict__ vec,
                                       const float *__restrict__ w1, const float *__restrict__ w3,
                                       int n, int d, int numSerialLoads, int batch_size) {
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= d)
        return;

    w1 += idx * n;
    w3 += idx * n;

    float x1[MAX_BATCH_SIZE] = {0};
    float x3[MAX_BATCH_SIZE] = {0};
    float4 ww1, ww3;
    float4 inp;

    for (int i = 0; i < numSerialLoads; ++i) {
        int j = (i * WARP_SIZE + threadIdx.x) * 4;
        if (j < n) {
            ww1 = *((float4 *)(&w1[j]));
            ww3 = *((float4 *)(&w3[j]));

            for (int b = 0; b < batch_size; ++b) {
                inp = *((float4 *)(&vec[b * n + j]));
                x1[b] += ww1.x * inp.x + ww1.y * inp.y + ww1.z * inp.z + ww1.w * inp.w;
                x3[b] += ww3.x * inp.x + ww3.y * inp.y + ww3.z * inp.z + ww3.w * inp.w;
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        x1[b] = warpReduceSum(x1[b]);
        x3[b] = warpReduceSum(x3[b]);
        if (threadIdx.x == 0)
            out[b * d + idx] = (x1[b] / (1.0f + expf(-x1[b]))) * x3[b];
    }
}

void matmul_swiglu_gpu(float *xout, float *x, float *w1, float *w3, int n, int d, int batch_size, cudaStream_t stream) {
    // printf("%d %d %d\n", n, d, batch_size);
    int serialElements = DIVUP(n, WARP_SIZE);
    int serialLoads = DIVUP(serialElements, 4);
    dim3 blockDim(WARP_SIZE, 4);
    dim3 gridDim(DIVUP(d, 4));
    batch_mv_swiglu_kernel<<<gridDim, blockDim, 0, stream>>>(xout, x, w1, w3, n, d, serialLoads, batch_size);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void swiglu_kernel(float *shb, float *shb2, int hidden_dim) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (j < hidden_dim) {
        float val = shb[i * hidden_dim + j];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= shb2[i * hidden_dim + j];
        shb[i * hidden_dim + j] = val;
    }
}

void swiglu_gpu(RunState *s, int hidden_dim, int batch_size, cudaStream_t stream) {
    // SwiGLU non-linearity
    dim3 blockDim(64);
    dim3 gridDim(DIVUP(hidden_dim, 64), batch_size);
    swiglu_kernel<<<gridDim, blockDim, 0, stream>>>(s->hb, s->hb2, hidden_dim);
    CHECK_CUDA(cudaGetLastError());
}


void swiglu_cpu(RunState *s, int hidden_dim) {
    // SwiGLU non-linearity
    for (int i = 0; i < hidden_dim; i++) {
        float val = s->hb[i];
        // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
        val *= (1.0f / (1.0f + expf(-val)));
        // elementwise multiply with w3(x)
        val *= s->hb2[i];
        s->hb[i] = val;
    }
}
