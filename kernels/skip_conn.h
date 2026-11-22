#include <cuda_runtime.h>


void matmul_skipadd_gpu(float *xout, float *x, float *w, int n, int d, int batch_size, cudaStream_t stream = 0);
void skip_conn_gpu(float *x, float *y, int dim, int batch_size, cudaStream_t stream = 0);

void skip_conn_cpu(float *x, float *y, int dim);
