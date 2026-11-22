#include <cuda_runtime.h>

void rmsnorm_gpu(float *o, float *x, float *weight, int size, int batch_size, cudaStream_t stream = 0);

void rmsnorm_cpu(float *o, float *x, float *weight, int size);
