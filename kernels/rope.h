#include <cuda_runtime.h>

#include "types.h"

void rotary_embedding_gpu(RunState *s, int pos, int dim, int kv_dim, int head_size, cudaStream_t stream = 0);

void rotary_embedding_cpu(RunState *s, int pos, int dim, int kv_dim, int head_size);
