#pragma once
// Residual connection: x[i] += y[i]
void residual_add_gpu(float *x, const float *y, int size);
void residual_add_cpu(float *x, const float *y, int size);
