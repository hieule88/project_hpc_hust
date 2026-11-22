#include "skip_conn.h"

#include <hip/hip_runtime.h>
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

void matmul_skipadd_gpu(float *xout, float *x, float *w, int n, int d, int batch_size, hipStream_t stream) {
    // printf("%d %d %d\n", n, d, batch_size);
    int serialElements = DIVUP(n, WARP_SIZE);
    int serialLoads = DIVUP(serialElements, 4);
    dim3 blockDim(WARP_SIZE, 4);
    dim3 gridDim(DIVUP(d, 4));
    hipLaunchKernelGGL(batch_mv_add_kernel, gridDim, blockDim, 0, stream,
                       xout, x, w, n, d, serialLoads, batch_size);
    CHECK_HIP(hipGetLastError());
}

__global__ void skip_conn_kernel(float *x, float *y, int dim) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (j < dim) {
        x[i * dim + j] += y[i * dim + j];
    }
}

void skip_conn_gpu(float *x, float *y, int dim, int batch_size, hipStream_t stream) {
    dim3 blockDim(64);
    dim3 gridDim(DIVUP(dim, 64), batch_size);
    hipLaunchKernelGGL(skip_conn_kernel, gridDim, blockDim, 0, stream, x, y, dim);
    CHECK_HIP(hipGetLastError());
}

void skip_conn_gpu_test(float *x, float *y, int dim) {
    float *x_d, *y_d;

    CHECK_HIP(hipMalloc(&x_d, sizeof(float) * dim));
    CHECK_HIP(hipMalloc(&y_d, sizeof(float) * dim));

    CHECK_HIP(hipMemcpyHtoD(x_d, x, sizeof(float) * dim));
    CHECK_HIP(hipMemcpyHtoD(y_d, y, sizeof(float) * dim));

    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(DIVUP(dim, BLOCK_SIZE * BLOCK_SIZE));
    hipLaunchKernelGGL(skip_conn_kernel, gridDim, blockDim, 0, 0, x_d, y_d, dim);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipMemcpyDtoH(x, x_d, sizeof(float) * dim));
    CHECK_HIP(hipMemcpyDtoH(y, y_d, sizeof(float) * dim));

    CHECK_HIP(hipFree(x_d));
    CHECK_HIP(hipFree(y_d));
}

void skip_conn_cpu(float *x, float *y, int dim) {
    for (int i = 0; i < dim; i++) {
        x[i] += y[i];
    }
}
