#pragma once
// In-place softmax over `rows` rows of `size` elements; row r starts at x + r * stride.
void softmax_gpu(float *x, int rows, int size, int stride);
void softmax_cpu(float *x, int size);
