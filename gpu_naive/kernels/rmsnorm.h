#pragma once
// o = weight * x / sqrt(mean(x^2) + 1e-5). o may equal x.
void rmsnorm_gpu(float *o, const float *x, const float *weight, int size);
void rmsnorm_cpu(float *o, const float *x, const float *weight, int size);
