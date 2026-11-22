#include "attn.h"

#include <hip/hip_runtime.h>
#include <math.h>

#include "softmax.h"
#include "types.h"
#include "utils.h"

__global__ void batch_fused_rope_mha(const int *__restrict__ p_pos, float *__restrict__ sq, float *__restrict__ sxb,
                                     float *__restrict__ s_key_cache, float *__restrict__ s_value_cache,
                                     int dim, int kv_dim, int kv_mul, int head_size,
                                     int loff, int kv_cache_off, float scale) {
    int batch_id = blockIdx.y;
    int h = blockIdx.x;
    __shared__ int pos;
    if (threadIdx.x==0)
        pos = p_pos[batch_id];
    __syncthreads();
    // get the query kv_cache vector for this head
    kv_cache_off = batch_id * kv_cache_off;
    const float *q = sq + batch_id * dim + h * head_size;
    float *key_cache = s_key_cache + kv_cache_off + loff + (h / kv_mul) * head_size;
    float *value_cache = s_value_cache + kv_cache_off + loff + (h / kv_mul) * head_size;

    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];
    __shared__ float q_smem[MAX_SEQ_LEN + 1];
    __shared__ float max_val;
    __shared__ float inv_d_total;



    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    float *k = key_cache + pos * kv_dim;
    for (int j = threadIdx.x; j < head_size >> 1; j += blockDim.x) {
        int i = j << 1;
        float val = (float)pos / __powf(10000.0f, i / (float)head_size);
        float fcr = __cosf(val), fci = __sinf(val);
        // calculate q and store to shared memory
        float q0 = q[i], q1 = q[i + 1];
        q_smem[i] = q0 * fcr - q1 * fci;
        q_smem[i + 1] = q0 * fci + q1 * fcr;
        // calculate k
        float k0 = k[i], k1 = k[i + 1];
        k[i] = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }

    __syncthreads();

    // iterate over all timesteps, including the current one
    // In CUDA, each thread does a small portion of the calc
    // attention scores for this head
    float4 a, b;
    float score;
    float m_partial = -FLT_MAX;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        k = key_cache + t * kv_dim;
        // calculate the attention score as the dot product of q and k
        score = 0.0f;
#pragma unroll
        for (int i = 0; i < head_size; i += 4) {
            a = *((float4 *)(&q_smem[i]));
            b = *((float4 *)(&k[i]));
            score += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        // save the score to the attention buffer
        att[t] = score * scale;
        m_partial = fmaxf(m_partial, att[t]);
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    if (threadIdx.x == 0)
        max_val = m_partial;
    __syncthreads();

    float d_partial = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        att[t] = __expf(att[t] - max_val);
        d_partial += att[t];
    }

    d_partial = blockReduceSum(d_partial);
    if (threadIdx.x == 0)
        inv_d_total = 1.0f / d_partial;
    __syncthreads();

    // weighted sum of the values, store back into xb
    // NOTE: by swapping the order of the for loops (vs. C) a simpler
    // version of the code accomplishes the same task and fits more
    // naturally with the CUDA way of subdividing the problem.
    float *xb = sxb + batch_id * dim + h * head_size;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
#pragma unroll
        for (int t = 0; t <= pos; ++t) {
            // get the value vector for this head and at this timestep
            float *v = value_cache + t * kv_dim;
            // get the attention weight for this timestep
            val += att[t] * v[i] * inv_d_total;
        }
        xb[i] = val;
    }
}

void batch_multihead_attention_gpu(Config *p, RunState *batch_s,
                                   int kv_dim, int kv_mul, int head_size, int loff, int kv_cache_off, int batch_size, hipStream_t stream) {
    int n_heads = p->n_heads;
    int dim = p->dim;
    dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridDim(n_heads, batch_size);

    hipLaunchKernelGGL(batch_fused_rope_mha, gridDim, blockDim, 0, stream,
                       batch_s->p_pos, batch_s->q, batch_s->xb, batch_s->key_cache, batch_s->value_cache,
                       dim, kv_dim, kv_mul, head_size, loff, kv_cache_off, 1.0 / sqrt(head_size));
    CHECK_HIP(hipGetLastError());
}

__global__ void multihead_attention_kernel(int pos, float *sq, float *sxb, float *key_cache, float *value_cache,
                                           int kv_dim, int kv_mul, int head_size,
                                           int loff, float scale) {
    int h = blockIdx.x;
    // get the query vector for this head
    const float *q = sq + h * head_size;
    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];
    __shared__ float max_val;
    __shared__ float inv_d_total;
    float4 a, b;
    // float *att = satt + h * seq_len;
    // iterate over all timesteps, including the current one
    // In CUDA, each thread does a small portion of the calc
    // attention scores for this head
    float m_partial = -FLT_MAX;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        // get the key vector for this head and at this timestep
        float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < head_size; i += 4) {
            a = *((float4 *)(&q[i]));
            b = *((float4 *)(&k[i]));
            score += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        // save the score to the attention buffer
        att[t] = score * scale;
        m_partial = fmaxf(m_partial, att[t]);
    }

    // softmax the scores to get attention weights, from 0..pos inclusively
    if (threadIdx.x == 0)
        max_val = m_partial;
    __syncthreads();

    float d_partial = 0.0f;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        att[t] = __expf(att[t] - max_val);
        d_partial += att[t];
    }

    d_partial = blockReduceSum(d_partial);
    if (threadIdx.x == 0)
        inv_d_total = 1.0f / d_partial;
    __syncthreads();

    // weighted sum of the values, store back into xb
    // NOTE: by swapping the order of the for loops (vs. C) a simpler
    // version of the code accomplishes the same task and fits more
    // naturally with the CUDA way of subdividing the problem.
    float *xb = sxb + h * head_size;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
#pragma unroll
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            val += att[t] * v[i] * inv_d_total;
        }
        xb[i] = val;
    }
}

void multihead_attention_gpu(Config *p, RunState *s,
                             int pos, int kv_dim, int kv_mul, int head_size, int loff, hipStream_t stream) {
    int n_heads = p->n_heads;
    int n_layers = p->n_layers;
    int dim = p->dim;
    // int smem = (pos + 1) * sizeof(float);
    hipLaunchKernelGGL(multihead_attention_kernel, n_heads, BLOCK_SIZE * BLOCK_SIZE, 0, stream,
                       pos, s->q, s->xb, s->key_cache, s->value_cache, kv_dim, kv_mul, head_size, loff, 1.0 / sqrt(head_size));
    CHECK_HIP(hipGetLastError());
}

void multihead_attention_gpu_test(Config *p, RunState *s,
                                  int pos, int kv_dim, int kv_mul, int head_size, int loff, int batch_size) {
    int n_heads = p->n_heads;
    int n_layers = p->n_layers;
    int seq_len = p->seq_len;
    int dim = p->dim;
    int kv_cache_off = p->n_layers * p->seq_len * kv_dim;

    // float *q_d, *att_d, *xb_d, *key_cache_d, *value_cache_d;
    RunState *s_d;
    CHECK_HIP(hipMalloc(&s_d, sizeof(float) * dim));
    CHECK_HIP(hipMalloc(&s_d->q, sizeof(float) * dim));
    CHECK_HIP(hipMalloc(&s_d->att, sizeof(float) * n_heads * seq_len));
    CHECK_HIP(hipMalloc(&s_d->xb, sizeof(float) * dim));
    CHECK_HIP(hipMalloc(&s_d->key_cache, sizeof(float) * n_layers * seq_len * kv_dim));
    CHECK_HIP(hipMalloc(&s_d->value_cache, sizeof(float) * n_layers * seq_len * kv_dim));
    CHECK_HIP(hipHostMalloc(&s_d->p_pos, sizeof(int *) * batch_size, hipHostMallocNonCoherent));

    CHECK_HIP(hipMemcpyHtoD(s_d->q, s->q, sizeof(float) * dim));
    CHECK_HIP(hipMemcpyHtoD(s_d->att, s->att, sizeof(float) * n_heads * seq_len));
    CHECK_HIP(hipMemcpyHtoD(s_d->xb, s->xb, sizeof(float) * dim));
    CHECK_HIP(hipMemcpyHtoD(s_d->key_cache, s->key_cache, sizeof(float) * n_layers * seq_len * kv_dim));
    CHECK_HIP(hipMemcpyHtoD(s_d->value_cache, s->value_cache, sizeof(float) * n_layers * seq_len * kv_dim));

    for (int i = 0; i < batch_size; ++i) {
        s_d->p_pos[i] = pos;
    }

    double elapsed_time = 0.0;
    for (int i = 0; i < 10; ++i) {
        // double start_time = get_mtime();
        for (int i = 0; i < batch_size; ++i) {
            // multihead_attention_gpu(p, s_d, pos, kv_dim, kv_mul, head_size, loff);
            batch_multihead_attention_gpu(p, s_d, kv_dim, kv_mul, head_size, loff, kv_cache_off, batch_size);
        }
        CHECK_HIP(hipDeviceSynchronize());
        // elapsed_time += get_mtime() - start_time;
    }
    // printf("Stream MHA version - Time Avg: %f sec\n", (elapsed_time / 10) / 1e9);

    CHECK_HIP(hipMemcpyDtoH(s->xb, s_d->xb, sizeof(float) * dim));

    CHECK_HIP(hipFree(s_d->q));
    CHECK_HIP(hipFree(s_d->att));
    CHECK_HIP(hipFree(s_d->xb));
    CHECK_HIP(hipFree(s_d->key_cache));
    CHECK_HIP(hipFree(s_d->value_cache));
}

void multihead_attention_cpu(Config *p, RunState *s, int pos, int kv_dim, int kv_mul, int head_size, int loff) {
    int h;
    for (h = 0; h < p->n_heads; h++) {
        // get the query vector for this head
        float *q = s->q + h * head_size;
        // attention scores for this head
        float *att = s->att + h * p->seq_len;
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q[i] * k[i];
            }
            score /= sqrtf(head_size);
            // save the score to the attention buffer
            att[t] = score;
        }

        // softmax the scores to get attention weights, from 0..pos inclusively
        softmax_cpu(att, pos + 1);

        // weighted sum of the values, store back into xb
        float *xb = s->xb + h * head_size;
        memset(xb, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < head_size; i++) {
                xb[i] += a * v[i];
            }
        }
    }
}