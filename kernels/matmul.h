#include <hip/hip_runtime.h>

#include "types.h"

void matmul_gpu_v1(float *xout, float *x, float *w, int n, int d, int batch_size, hipStream_t stream = 0);
void matmul_gpu_v2(float *xout, float *x, float *w, int n, int d, int batch_size, hipStream_t stream = 0);
void matmul_gpu_v3(float *xout, float *x, float *w, int n, int d,
                   Config *p, int *p_pos, int kv_dim, int loff, int kv_cache_off, int batch_size, hipStream_t stream = 0);
void matmul_gpu_v4(float *xout_k, float *xout_v, float *x, float *w_k, float *w_v, int n, int d,
                   Config *p, int *p_pos, int kv_dim, int loff, size_t kv_cache_off, int batch_size, hipStream_t stream = 0);
void matmul_gpu_test(float *xout, float *x, float *w, int n, int d, int batch_size);

void matmul_cpu(float *xout, float *x, float *w, int n, int d, int batch_size);
