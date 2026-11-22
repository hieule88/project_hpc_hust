#include <hip/hip_runtime.h>


void matmul_skipadd_gpu(float *xout, float *x, float *w, int n, int d, int batch_size, hipStream_t stream = 0);
void skip_conn_gpu(float *x, float *y, int dim, int batch_size, hipStream_t stream = 0);
void skip_conn_gpu_test(float *x, float *y, int dim);

void skip_conn_cpu(float *x, float *y, int dim);
