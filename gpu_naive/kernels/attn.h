#pragma once
// Multi-head attention for one token at position pos, over cache positions 0..pos.
//   xb          (n_heads * head_size,)  output
//   q           (n_heads * head_size,)
//   att         (n_heads, seq_len)      scratch for scores / weights
//   key_cache, value_cache: full cache; loff = offset of this layer
// kv_mul = n_heads / n_kv_heads (grouped-query attention).
void attention_gpu(float *xb, const float *q, float *att, const float *key_cache,
                   const float *value_cache, int pos, int seq_len, int n_heads, int head_size,
                   int kv_dim, int kv_mul, long long loff);
void attention_cpu(float *xb, const float *q, float *att, const float *key_cache,
                   const float *value_cache, int pos, int seq_len, int n_heads, int head_size,
                   int kv_dim, int kv_mul, long long loff);
