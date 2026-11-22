#pragma once
// W (d,n) @ x (n,) -> xout (d,)
void matmul_gpu_v1(float *xout, const float *x, const float *w, int n, int d);
void matmul_cpu(float *xout, const float *x, const float *w, int n, int d);
