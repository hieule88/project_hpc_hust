#include "skip_conn.h"

#include <cuda_runtime.h>
#include <math.h>

#include "types.h"
#include "utils.h"

__global__ void batch_mv_add_kernel(float *__restrict__ out, const float *__restrict__ vec, const float *__restrict__ mat,
                                    int n, int d, int numSerialLoads, int batch_size) {
    int idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (idx >= d)
        return;

    mat += idx * n;

    float sum[MAX_BATCH_SIZE] = {0};
    float4 w;
    float4 inp;

    for (int i = 0; i < numSerialLoads; ++i) {
        int j = (i * WARP_SIZE + threadIdx.x) * 4;
        if (j < n) {
            w = *((float4 *)(&mat[j]));

            for (int b = 0; b < batch_size; ++b) {
                inp = *((float4 *)(&vec[b * n + j]));
                sum[b] += w.x * inp.x + w.y * inp.y + w.z * inp.z + w.w * inp.w;
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        sum[b] = warpReduceSum(sum[b]);
        if (threadIdx.x == 0)
            out[b * d + idx] += sum[b];
    }
}

void matmul_skipadd_gpu(float *xout, float *x, float *w, int n, int d, int batch_size, cudaStream_t stream) {
    // printf("%d %d %d\n", n, d, batch_size);
    int serialElements = DIVUP(n, WARP_SIZE);
    int serialLoads = DIVUP(serialElements, 4);
    dim3 blockDim(WARP_SIZE, 4);
    dim3 gridDim(DIVUP(d, 4));
    batch_mv_add_kernel<<<gridDim, blockDim, 0, stream>>>(xout, x, w, n, d, serialLoads, batch_size);
    CHECK_CUDA(cudaGetLastError());
}

__global__ void skip_conn_kernel(float *x, float *y, int dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (j < dim) {
        x[i * dim + j] += y[i * dim + j];
    }
}

void skip_conn_gpu(float *x, float *y, int dim, int batch_size, cudaStream_t stream) {
    dim3 blockDim(64);
    dim3 gridDim(DIVUP(dim, 64), batch_size);
    skip_conn_kernel<<<gridDim, blockDim, 0, stream>>>(x, y, dim);
    CHECK_CUDA(cudaGetLastError());
}


void skip_conn_cpu(float *x, float *y, int dim) {
    for (int i = 0; i < dim; i++) {
        x[i] += y[i];
    }
}
