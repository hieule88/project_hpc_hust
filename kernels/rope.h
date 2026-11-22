#pragma once
// Rotary position embedding: rotate q (dim,) and k (kv_dim,) in place for position pos.
void rotary_embedding_gpu(float *q, float *k, int pos, int dim, int kv_dim, int head_size);
void rotary_embedding_cpu(float *q, float *k, int pos, int dim, int kv_dim, int head_size);
