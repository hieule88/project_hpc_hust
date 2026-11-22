#pragma once
#include <hip/hip_runtime.h>

void softmax_gpu(float *x, int size);
void softmax_gpu_test(float *x, int size);

void softmax_cpu(float *x, int size);
