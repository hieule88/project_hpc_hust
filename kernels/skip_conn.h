#pragma once
// Residual connection: x[i] += y[i]
void skip_conn_gpu(float *x, const float *y, int size);
void skip_conn_cpu(float *x, const float *y, int size);
