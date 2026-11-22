#include <cuda_runtime.h>

#include "types.h"

void matmul_swiglu_gpu(float *xout, float *x, float *w1, float *w3, int n, int d, int batch_size, cudaStream_t stream = 0);
void swiglu_gpu(RunState *s, int hidden_dim, int batch_size, cudaStream_t stream = 0);

void swiglu_cpu(RunState *s, int hidden_dim);
