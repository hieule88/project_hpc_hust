#pragma once
// hb[i] = silu(hb[i]) * hb2[i]
void swiglu_gpu(float *hb, const float *hb2, int size);
void swiglu_cpu(float *hb, const float *hb2, int size);
