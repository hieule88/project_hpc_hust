#include <hip/hip_runtime.h>

#include "types.h"

void matmul_swiglu_gpu(float *xout, float *x, float *w1, float *w3, int n, int d, int batch_size, hipStream_t stream = 0);
void swiglu_gpu(RunState *s, int hidden_dim, int batch_size, hipStream_t stream = 0);
void swiglu_gpu_test(RunState *s, int hidden_dim);

void swiglu_cpu(RunState *s, int hidden_dim);
