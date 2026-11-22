#include <math.h>
#include <string.h>

#include "attn.h"
#include "softmax.h"
#include "utils.h"

// The CPU loop over heads is split into three naive kernels:
//   1. scores:       one thread per (head, t)  -> att[h][t] = q_h . k_t / sqrt(head_size)
//   2. softmax:      one block per head        (softmax_gpu)
//   3. weighted sum: one thread per (head, i)  -> xb[h][i] = sum_t att[h][t] * v_t[i]

__global__ void attn_scores_kernel(float *att, const float *q, const float *key_cache, int pos,
                                   int seq_len, int n_heads, int head_size, int kv_dim,
                                   int kv_mul, long long loff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_t = pos + 1;
    if (idx >= n_heads * n_t) return;
    int h = idx / n_t;
    int t = idx % n_t;
    const float *qh = q + h * head_size;
    const float *k = key_cache + loff + (long long)t * kv_dim + (h / kv_mul) * head_size;
    float score = 0.0f;
    for (int i = 0; i < head_size; i++) score += qh[i] * k[i];
    score /= sqrtf(head_size);
    att[h * seq_len + t] = score;
}

__global__ void attn_weighted_sum_kernel(float *xb, const float *att, const float *value_cache,
                                         int pos, int seq_len, int n_heads, int head_size,
                                         int kv_dim, int kv_mul, long long loff) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads * head_size) return;
    int h = idx / head_size;
    int i = idx % head_size;
    const float *a = att + h * seq_len;
    const float *v = value_cache + loff + (h / kv_mul) * head_size + i;
    float val = 0.0f;
    for (int t = 0; t <= pos; t++) val += a[t] * v[(long long)t * kv_dim];
    xb[idx] = val;
}

void multihead_attention_gpu(float *xb, const float *q, float *att, const float *key_cache,
                   const float *value_cache, int pos, int seq_len, int n_heads, int head_size,
                   int kv_dim, int kv_mul, long long loff) {
    int n_scores = n_heads * (pos + 1);
    attn_scores_kernel<<<DIVUP(n_scores, NAIVE_BLOCK), NAIVE_BLOCK>>>(
        att, q, key_cache, pos, seq_len, n_heads, head_size, kv_dim, kv_mul, loff);
    CHECK_LAUNCH();
    softmax_gpu(att, n_heads, pos + 1, seq_len);
    int n_out = n_heads * head_size;
    attn_weighted_sum_kernel<<<DIVUP(n_out, NAIVE_BLOCK), NAIVE_BLOCK>>>(
        xb, att, value_cache, pos, seq_len, n_heads, head_size, kv_dim, kv_mul, loff);
    CHECK_LAUNCH();
}

// CPU reference (the attention loop inside forward() in run.c, without OpenMP)
void multihead_attention_cpu(float *xb, const float *q, float *att, const float *key_cache,
                   const float *value_cache, int pos, int seq_len, int n_heads, int head_size,
                   int kv_dim, int kv_mul, long long loff) {
    for (int h = 0; h < n_heads; h++) {
        const float *qh = q + h * head_size;
        float *atth = att + h * seq_len;
        for (int t = 0; t <= pos; t++) {
            const float *k = key_cache + loff + (long long)t * kv_dim + (h / kv_mul) * head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) score += qh[i] * k[i];
            score /= sqrtf(head_size);
            atth[t] = score;
        }
        softmax_cpu(atth, pos + 1);
        float *xbh = xb + h * head_size;
        memset(xbh, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float *v = value_cache + loff + (long long)t * kv_dim + (h / kv_mul) * head_size;
            float a = atth[t];
            for (int i = 0; i < head_size; i++) xbh[i] += a * v[i];
        }
    }
}
