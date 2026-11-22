#include <hip/hip_runtime.h>

#include "types.h"

void rotary_embedding_gpu(RunState *s, int pos, int dim, int kv_dim, int head_size, hipStream_t stream = 0);
void rotary_embedding_gpu_test(RunState *s, int pos, int dim, int kv_dim, int head_size);

void rotary_embedding_cpu(RunState *s, int pos, int dim, int kv_dim, int head_size);
