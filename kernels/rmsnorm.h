#include <hip/hip_runtime.h>

void rmsnorm_gpu(float *o, float *x, float *weight, int size, int batch_size, hipStream_t stream = 0);
void rmsnorm_gpu_test(float *o, float *x, float *weight, int size);

void rmsnorm_cpu(float *o, float *x, float *weight, int size);
