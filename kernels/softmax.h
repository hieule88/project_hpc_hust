#pragma once
#include <cuda_runtime.h>

void softmax_gpu(float *x, int size);

void softmax_cpu(float *x, int size);
