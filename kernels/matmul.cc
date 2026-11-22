#include "matmul.h"

#include <hip/hip_runtime.h>
#include <sys/time.h>

#include "types.h"
#include "utils.h"

__global__ void mat_vec_kernel(float *C, const float *__restrict__ B, const float *__restrict__ A,
                               int n, int d, int numSerialLoads) {
    int index = blockIdx.x * blockDim.y + threadIdx.y;
    if (index >= d)
        return;

    A += index * n;
    B += blockIdx.y * n;
    C += blockIdx.y * d;

    float sum = 0;
    float4 w;
    float4 inp;

#pragma unroll
    for (int i = 0; i < numSerialLoads; i++) {
        int j = (i * WARP_SIZE + threadIdx.x) * 4;
        if (j < n) {
            w = *((float4 *)(&A[j]));
            inp = *((float4 *)(&B[j]));
            sum += w.x * inp.x + w.y * inp.y + w.z * inp.z + w.w * inp.w;
        }
    }

    sum = warpReduceSum(sum);
    if (threadIdx.x == 0)
        C[index] = sum;
}

void matmul_gpu_v1(float *xout, float *x, float *w, int n, int d, int batch_size, hipStream_t stream) {
    int serialElements = DIVUP(n, WARP_SIZE);
    int serialLoads = DIVUP(serialElements, 4);
    dim3 blockDim(WARP_SIZE, 4);
    dim3 gridDim(DIVUP(d, 4), batch_size);
    hipLaunchKernelGGL(mat_vec_kernel, gridDim, blockDim, 0, stream, xout, x, w, n, d, serialLoads);
    CHECK_HIP(hipGetLastError());
}

__global__ void batch_mat_vec_kernel(float *__restrict__ out, const float *__restrict__ vec, const float *__restrict__ mat,
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
            out[b * d + idx] = sum[b];
    }
}

void matmul_gpu_v2(float *xout, float *x, float *w, int n, int d, int batch_size, hipStream_t stream) {
    // printf("%d %d %d\n", n, d, batch_size);
    int serialElements = DIVUP(n, WARP_SIZE);
    int serialLoads = DIVUP(serialElements, 4);
    dim3 blockDim(WARP_SIZE, 4);
    dim3 gridDim(DIVUP(d, 4));
    hipLaunchKernelGGL(batch_mat_vec_kernel, gridDim, blockDim, 0, stream,
                       xout, x, w, n, d, serialLoads, batch_size);
    CHECK_HIP(hipGetLastError());
}

__global__ void mat_kv_kernel(float *__restrict__ out, const float *__restrict__ vec, const float *__restrict__ mat,
                              int n, int d, int numSerialLoads,
                              const int *__restrict__ p_pos, int kv_cache_off, int kv_dim, int batch_size) {
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
        if (threadIdx.x == 0) {
            float *o = out + b * kv_cache_off + p_pos[b] * kv_dim;
            o[idx] = sum[b];
        }
    }
}

void matmul_gpu_v3(float *xout, float *x, float *w, int n, int d,
                   Config *p, int *p_pos, int kv_dim, int loff, int kv_cache_off,
                   int batch_size, hipStream_t stream) {
    int serialElements = DIVUP(n, WARP_SIZE);
    int serialLoads = DIVUP(serialElements, 4);
    dim3 blockDim(WARP_SIZE, 4);
    dim3 gridDim(DIVUP(d, 4));

    hipLaunchKernelGGL(mat_kv_kernel, gridDim, blockDim, 0, stream,
                       xout + loff, x, w, n, d, serialLoads,
                       p_pos, kv_cache_off, kv_dim, batch_size);
    CHECK_HIP(hipGetLastError());
}

double get_mtime() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void matmul_gpu_test(float *xout, float *x, float *w, int n, int d, int batch_size) {
    float *xoutGPU, *xGPU, *wGPU;
    CHECK_HIP(hipMalloc(&xoutGPU, sizeof(float) * d * batch_size));
    CHECK_HIP(hipMalloc(&xGPU, sizeof(float) * n * batch_size));
    CHECK_HIP(hipMalloc(&wGPU, sizeof(float) * d * n));
    CHECK_HIP(hipMemcpy(xGPU, x, sizeof(float) * n * batch_size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(wGPU, w, sizeof(float) * d * n, hipMemcpyHostToDevice));
    double elapsed_time = 0.0;
    for (int i = 0; i <= 10; ++i) {
        double start_time = get_mtime();
        matmul_gpu_v2(xoutGPU, xGPU, wGPU, n, d, batch_size);
        CHECK_HIP(hipDeviceSynchronize());
        if (i > 0) elapsed_time += get_mtime() - start_time;
    }
    printf("%f sec, GFLOPS: %f \n", elapsed_time, 2ll * n * d * batch_size / (elapsed_time / 10) / 1e9);

    CHECK_HIP(hipMemcpy(xout, xoutGPU, sizeof(float) * d * batch_size, hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(xoutGPU));
    CHECK_HIP(hipFree(xGPU));
    CHECK_HIP(hipFree(wGPU));
}

void matmul_cpu(float *xout, float *x, float *w, int n, int d, int batch_size) {
    for (int req_id = 0; req_id < batch_size; req_id++) {
        for (int i = 0; i < d; i++) {
            float val = 0.0f;
            for (int j = 0; j < n; j++) {
                val += w[i * n + j] * x[req_id * n + j];
            }
            xout[req_id * d + i] = val;
        }
    }
}

__global__ void mat_kv_kernel_v2(float *__restrict__ out_k, float *__restrict__ out_v,
                                 const float *__restrict__ vec,
                                 const float *__restrict__ mat_k, const float *__restrict__ mat_v,
                                 int n, int d, int numSerialLoads,
                                 const int *__restrict__ p_pos, size_t kv_cache_off, int kv_dim, int batch_size) {
    long long idx = blockIdx.x * blockDim.y + threadIdx.y;

    if (idx >= d)
        return;

    mat_k += idx * n;
    mat_v += idx * n;

    float sum_k[MAX_BATCH_SIZE] = {0};
    float sum_v[MAX_BATCH_SIZE] = {0};
    float4 w_k, w_v;
    float4 inp;

    for (int i = 0; i < numSerialLoads; ++i) {
        int j = (i * WARP_SIZE + threadIdx.x) * 4;
        if (j < n) {
            w_k = *((float4 *)(&mat_k[j]));
            w_v = *((float4 *)(&mat_v[j]));
            for (int b = 0; b < batch_size; ++b) {
                inp = *((float4 *)(&vec[b * n + j]));
                sum_k[b] += w_k.x * inp.x + w_k.y * inp.y + w_k.z * inp.z + w_k.w * inp.w;
                sum_v[b] += w_v.x * inp.x + w_v.y * inp.y + w_v.z * inp.z + w_v.w * inp.w;
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        size_t offset = b * kv_cache_off + p_pos[b] * kv_dim;
        float *o_k = out_k + offset;
        float *o_v = out_v + offset;
        sum_k[b] = warpReduceSum(sum_k[b]);
        sum_v[b] = warpReduceSum(sum_v[b]);
        if (threadIdx.x == 0) {
            o_k[idx] = sum_k[b];
            o_v[idx] = sum_v[b];
        }
    }
}

void matmul_gpu_v4(float *xout_k, float *xout_v, float *x, float *w_k, float *w_v, int n, int d,
                   Config *p, int *p_pos, int kv_dim, int loff, size_t kv_cache_off,
                   int batch_size, hipStream_t stream) {
    int serialElements = DIVUP(n, WARP_SIZE);
    int serialLoads = DIVUP(serialElements, 4);
    dim3 blockDim(WARP_SIZE, 4);
    dim3 gridDim(DIVUP(d, 4));

    hipLaunchKernelGGL(mat_kv_kernel_v2, gridDim, blockDim, 0, stream,
                       xout_k + loff, xout_v + loff, x, w_k, w_v, n, d, serialLoads,
                       p_pos, kv_cache_off, kv_dim, batch_size);
    CHECK_HIP(hipGetLastError());
}
