#include <cuda_runtime.h>

#include "types.h"

void batch_multihead_attention_gpu(Config *p, RunState *s, int kv_dim, int kv_mul, int head_size,
                                   int loff, size_t kv_cache_off, int batch_size, cudaStream_t stream = 0);

void multihead_attention_gpu(Config *p, RunState *s, int pos, int kv_dim, int kv_mul, int head_size,
                             int loff, cudaStream_t stream = 0);


void multihead_attention_cpu(Config *p, RunState *s, int pos, int kv_dim, int kv_mul, int head_size, int loff);
