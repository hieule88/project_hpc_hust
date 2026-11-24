/*
 * run.cu + PagedAttention with vLLM-style preemption.
 *
 * Same model code, batched kernels, continuous batching, stop rule and output format as
 * run.cu. Only the KV cache management changes:
 *
 *   run.cu        : every slot owns n_layers * seq_len * kv_dim floats of K and V, reserved
 *                   up front, so the number of slots is bounded by seq_len-sized reservations.
 *   run_pgattn.cu : one pool of KV blocks per GPU (PAGE_SIZE token positions of every layer
 *                   per block) fills the free VRAM. A request gets blocks only as its position
 *                   grows; each slot has a block table (logical block -> physical block) that
 *                   the KV-write and attention kernels follow. A request may still use the full
 *                   seq_len; memory is held only for the positions it actually reached.
 *
 * Scheduling (as vLLM's default FCFS scheduler, with recompute preemption):
 *   - admission: the oldest waiting request starts when a slot is free and the pool can give
 *     it the blocks of its prompt while keeping a watermark (1% of the pool) free. Admission
 *     stops at the first request that does not fit (no skipping ahead).
 *   - growth: before every forward step each running request gets the block for its current
 *     position, oldest request first. When the pool is empty the most recently admitted
 *     request is preempted: its blocks are freed, its output and sampler are reset, and it
 *     goes back to the FRONT of the waiting queue, to be recomputed from position 0 later.
 *     Recomputation gives the same tokens (same seed, same arithmetic), so outputs match run.cu.
 *
 * The two kernels that touch the KV cache are copies of mat_kv_kernel_v2 and
 * batch_fused_rope_mha with block-table addressing; their arithmetic is unchanged.
 *
 * Usage: runpgattn <model> -m test -f <input> -o <output> [-b batch] [-n steps] [-t temp]
 *                  [-p topp] [-s seed] [-z tokenizer] [-k page_size] [-c max_blocks]
 * Prints the harness RESULT line, then a PAGED line with pool and preemption statistics.
 */

#include <ctype.h>
#include <fcntl.h>
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>
#include <vector>
// ----------------------------------------------------------------------------
// Transformer model
#include "kernels/attn.h"
#include "kernels/matmul.h"
#include "kernels/rmsnorm.h"
#include "kernels/rope.h"
#include "kernels/skip_conn.h"
#include "kernels/softmax.h"
#include "kernels/swiglu.h"
#include "kernels/types.h"
#include "kernels/utils.h"  // warpReduceSum, blockReduceMax/Sum, MAX_SEQ_LEN for the paged kernels

const int MAX_GPUS = 4;
int NUM_GPUS = 1;
int BATCH_SIZE = 1;
int PAGE_SIZE = 16;     // token positions per KV block (-k)
long MAX_BLOCKS = 0;    // cap on KV blocks per GPU, 0 = as many as fit in free VRAM (-c)

float *weights_ptr[MAX_GPUS];
TransformerWeights weights_d[MAX_GPUS];
RunState batch_run_state_d[MAX_GPUS];

#ifndef DIVUP
#define DIVUP(x, y) ((x + y - 1) / y)
#endif

#undef CHECK_CUDA
#define CHECK_CUDA(call)                                                       \
    do {                                                                      \
        cudaError_t status_ = call;                                            \
        if (status_ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
                    cudaGetErrorName(status_), cudaGetErrorString(status_));    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Activations only: the KV cache lives in the per-GPU block pool (KVPool below).
void malloc_run_state(RunState *s, Config *p) {
    CHECK_CUDA(cudaMalloc((void **)&s->x, p->dim * sizeof(float) * BATCH_SIZE));
    CHECK_CUDA(cudaMalloc((void **)&s->xb, p->dim * sizeof(float) * BATCH_SIZE));
    CHECK_CUDA(cudaMalloc((void **)&s->hb, p->hidden_dim * sizeof(float) * BATCH_SIZE));
    CHECK_CUDA(cudaMalloc((void **)&s->q, p->dim * sizeof(float) * BATCH_SIZE));
    s->key_cache = NULL;
    s->value_cache = NULL;

    CHECK_CUDA(cudaMalloc(&s->p_pos, sizeof(int) * BATCH_SIZE));
    CHECK_CUDA(cudaHostAlloc(&s->logits, p->vocab_size * sizeof(float) * BATCH_SIZE, cudaHostAllocDefault));
    CHECK_CUDA(cudaDeviceSynchronize());
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->hb || !s->q || !s->logits) {
        fprintf(stderr, "cudaMalloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void batch_map_run_state(RunState *batch_s, RunState *s, Config *p, int batch_id) {
    s->x = batch_s->x + batch_id * p->dim;
    s->xb = batch_s->xb + batch_id * p->dim;
    s->hb = batch_s->hb + batch_id * p->hidden_dim;
    s->q = batch_s->q + batch_id * p->dim;
    s->key_cache = NULL;  // paged: reached through the block table
    s->value_cache = NULL;
    s->p_pos = batch_s->p_pos + batch_id;
    s->logits = batch_s->logits + batch_id * p->vocab_size;
}

void free_run_state(RunState *s) {
    CHECK_CUDA(cudaFree(s->x));
    CHECK_CUDA(cudaFree(s->xb));
    CHECK_CUDA(cudaFree(s->hb));
    CHECK_CUDA(cudaFree(s->q));
    CHECK_CUDA(cudaFree(s->p_pos));
    CHECK_CUDA(cudaFreeHost(s->logits));
}

// ----------------------------------------------------------------------------
// Paged KV cache
//
// Physical layout of each pool (K and V separately):
//   [num_blocks][n_layers][PAGE_SIZE][kv_dim]
// so position t of a request, layer l, element i lives at
//   pool + ((block_table[t / PAGE_SIZE] * n_layers + l) * PAGE_SIZE + t % PAGE_SIZE) * kv_dim + i
// A block holds PAGE_SIZE consecutive positions of every layer, so one block-table entry
// covers the whole forward pass of those positions.

struct KVPool {
    float *k = NULL, *v = NULL;  // device pools
    long num_blocks = 0;
    int max_blocks_per_seq = 0;  // DIVUP(seq_len, PAGE_SIZE) = block-table row length
    int *d_table = NULL;         // device block tables, [BATCH_SIZE][max_blocks_per_seq]
    std::vector<int> h_table;    // host copy of the block tables
    std::vector<int> free_list;  // free physical blocks (used as a stack)
    long peak_used = 0;
};
KVPool kv_pool[MAX_GPUS];

static size_t kv_block_floats(Config *p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    return (size_t)p->n_layers * PAGE_SIZE * kv_dim;
}

// Called after the weights and activations are on the GPU: the pool takes the free VRAM
// minus a margin for the CUDA runtime (or MAX_BLOCKS blocks if -c is given and smaller).
void alloc_kv_pool(KVPool *pool, Config *p) {
    const size_t margin = (size_t)512 << 20;
    size_t free_b, total_b;
    CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
    size_t block_bytes = kv_block_floats(p) * sizeof(float);  // one of K or V
    long fit = free_b > margin ? (long)((free_b - margin) / (2 * block_bytes)) : 0;
    pool->num_blocks = (MAX_BLOCKS > 0 && MAX_BLOCKS < fit) ? MAX_BLOCKS : fit;
    if (pool->num_blocks <= 0) {
        fprintf(stderr, "no GPU memory left for the KV pool (free %zu MB)\n", free_b >> 20);
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA(cudaMalloc((void **)&pool->k, pool->num_blocks * block_bytes));
    CHECK_CUDA(cudaMalloc((void **)&pool->v, pool->num_blocks * block_bytes));

    pool->max_blocks_per_seq = DIVUP(p->seq_len, PAGE_SIZE);
    pool->h_table.assign((size_t)BATCH_SIZE * pool->max_blocks_per_seq, 0);
    CHECK_CUDA(cudaMalloc((void **)&pool->d_table, pool->h_table.size() * sizeof(int)));
    pool->free_list.clear();
    for (long b = pool->num_blocks - 1; b >= 0; --b) pool->free_list.push_back((int)b);
    pool->peak_used = 0;
}

void free_kv_pool(KVPool *pool) {
    CHECK_CUDA(cudaFree(pool->k));
    CHECK_CUDA(cudaFree(pool->v));
    CHECK_CUDA(cudaFree(pool->d_table));
}

// ----------------------------------------------------------------------------
// Paged copies of the two kernels that touch the KV cache. Only the address of the KV
// entries changes; the arithmetic (and so the result) is that of mat_kv_kernel_v2 and
// batch_fused_rope_mha. A row whose position is negative is an empty slot inside the
// batch window: it owns no blocks, so it writes and reads nothing.

__device__ __forceinline__ size_t paged_offset(const int *table, int t, int layer, int n_layers,
                                               int page_size, int kv_dim) {
    int blk = table[t / page_size];
    return (((size_t)blk * n_layers + layer) * page_size + t % page_size) * kv_dim;
}

__global__ void paged_mat_kv_kernel(float *__restrict__ pool_k, float *__restrict__ pool_v,
                                    const float *__restrict__ vec,
                                    const float *__restrict__ mat_k, const float *__restrict__ mat_v,
                                    int n, int d, int numSerialLoads,
                                    const int *__restrict__ p_pos, const int *__restrict__ block_table,
                                    int max_blocks_per_seq, int layer, int n_layers, int page_size,
                                    int kv_dim, int batch_size) {
    long long idx = blockIdx.x * blockDim.y + threadIdx.y;

    if (idx >= d)
        return;

    mat_k += idx * n;
    mat_v += idx * n;

    float sum_k[MAX_BATCH_SIZE] = {0};
    float sum_v[MAX_BATCH_SIZE] = {0};
    float4 w_k, w_v;
    float4 inp;

    for (int i = 0; i < numSerialLoads; ++i) {
        int j = (i * WARP_SIZE + threadIdx.x) * 4;
        if (j < n) {
            w_k = *((float4 *)(&mat_k[j]));
            w_v = *((float4 *)(&mat_v[j]));
            for (int b = 0; b < batch_size; ++b) {
                inp = *((float4 *)(&vec[b * n + j]));
                sum_k[b] += w_k.x * inp.x + w_k.y * inp.y + w_k.z * inp.z + w_k.w * inp.w;
                sum_v[b] += w_v.x * inp.x + w_v.y * inp.y + w_v.z * inp.z + w_v.w * inp.w;
            }
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        sum_k[b] = warpReduceSum(sum_k[b]);
        sum_v[b] = warpReduceSum(sum_v[b]);
        int pos = p_pos[b];
        if (threadIdx.x == 0 && pos >= 0) {
            size_t offset = paged_offset(block_table + (size_t)b * max_blocks_per_seq, pos, layer,
                                         n_layers, page_size, kv_dim);
            pool_k[offset + idx] = sum_k[b];
            pool_v[offset + idx] = sum_v[b];
        }
    }
}

__global__ void paged_fused_rope_mha(const int *__restrict__ p_pos, float *__restrict__ sq, float *__restrict__ sxb,
                                     float *__restrict__ pool_k, float *__restrict__ pool_v,
                                     const int *__restrict__ block_table, int max_blocks_per_seq,
                                     int layer, int n_layers, int page_size,
                                     int dim, int kv_dim, int kv_mul, int head_size, float scale) {
    int batch_id = blockIdx.y;
    int h = blockIdx.x;
    __shared__ int pos;
    if (threadIdx.x == 0)
        pos = p_pos[batch_id];
    __syncthreads();
    if (pos < 0)
        return;  // empty slot: the whole block leaves together

    const int *table = block_table + (size_t)batch_id * max_blocks_per_seq;
    const float *q = sq + batch_id * dim + h * head_size;
    size_t head_off = (size_t)(h / kv_mul) * head_size;

    // attention scores for this head
    __shared__ float att[MAX_SEQ_LEN];
    // read as float4 below: CUDA requires 16-byte alignment
    __shared__ __align__(16) float q_smem[MAX_SEQ_LEN + 1];
    __shared__ float max_val;
    __shared__ float inv_d_total;

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    float *k = pool_k + paged_offset(table, pos, layer, n_layers, page_size, kv_dim) + head_off;
    for (int j = threadIdx.x; j < head_size >> 1; j += blockDim.x) {
        int i = j << 1;
        float freq = 1.0f / powf(10000.0f, i / (float)head_size);
        float val = pos * freq;
        float fcr = cosf(val), fci = sinf(val);
        float q0 = q[i], q1 = q[i + 1];
        q_smem[i] = q0 * fcr - q1 * fci;
        q_smem[i + 1] = q0 * fci + q1 * fcr;
        float k0 = k[i], k1 = k[i + 1];
        k[i] = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }

    __syncthreads();

    float4 a, b;
    float score;
    float m_partial = -FLT_MAX;
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        k = pool_k + paged_offset(table, t, layer, n_layers, page_size, kv_dim) + head_off;
        score = 0.0f;
#pragma unroll
        for (int i = 0; i < head_size; i += 4) {
            a = *((float4 *)(&q_smem[i]));
            b = *((float4 *)(&k[i]));
            score += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        att[t] = score * scale;
        m_partial = fmaxf(m_partial, att[t]);
    }

    m_partial = blockReduceMax(m_partial);
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

    float *xb = sxb + batch_id * dim + h * head_size;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
#pragma unroll
        for (int t = 0; t <= pos; ++t) {
            float *v = pool_v + paged_offset(table, t, layer, n_layers, page_size, kv_dim) + head_off;
            val += att[t] * v[i] * inv_d_total;
        }
        xb[i] = val;
    }
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->seq_len * head_size / 2;  // skip what used to be freq_cis_real (for RoPE)
    ptr += p->seq_len * head_size / 2;  // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights,
                     int *fd, float **data, ssize_t *file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) {
        exit(EXIT_FAILURE);
    }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END);  // move file pointer to end of file
    *file_size = ftell(file);  // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY);  // open in read only mode
    if (*fd == -1) {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = (float *)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    size_t weights_size = *file_size - sizeof(Config);

#pragma omp parallel for num_threads(NUM_GPUS)
    for (int i = 0; i < NUM_GPUS; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&weights_ptr[i], weights_size));
        CHECK_CUDA(cudaMemcpy(weights_ptr[i], *data + sizeof(Config) / sizeof(float), weights_size, cudaMemcpyHostToDevice));
        memory_map_weights(&weights_d[i], config, weights_ptr[i], shared_weights);
    }

    memory_map_weights(weights, config, weights_ptr[0], shared_weights);
}

void print_transformer(Transformer *t) {
    printf("---------Model Information----------\n");
    printf("dim: %d\n", t->config.dim);
    printf("hidden_dim: %d\n", t->config.hidden_dim);
    printf("n_layers: %d\n", t->config.n_layers);
    printf("n_heads: %d\n", t->config.n_heads);
    printf("n_kv_heads: %d\n", t->config.n_kv_heads);
    printf("vocab_size: %d\n", t->config.vocab_size);
    printf("seq_len: %d\n", t->config.seq_len);
    printf("weights_size: %lu MB\n", (t->file_size - sizeof(Config)) / (1024L * 1024L));
    printf("------------------------------------\n");
}

void build_transformer(Transformer *t, char *checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    // malloc_run_state(&t->state, &t->config);
    print_transformer(t);
}

void free_transformer(Transformer *t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) {
        munmap(t->data, t->file_size);
    }
    if (t->fd != -1) {
        close(t->fd);
    }
    // we cudaMalloc a region of memory, then hand the address to
    // the token_embedding_table field.  Free it here.
    CHECK_CUDA(cudaFree(t->weights.token_embedding_table));
}

// Same as run.cu's forward(), with the KV write and the attention going through the block
// tables of slots [first, first + batch_size) of `pool`.
void forward(Transformer *transformer, HandlingReq *handling_reqs, RunState *batch_s, TransformerWeights *w,
             int batch_size, KVPool *pool, int first) {
    Config *p = &transformer->config;
    int *h_pos = (int *)malloc(batch_size * sizeof(int));
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;  // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    for (int i = 0; i < batch_size; ++i) {
        // an empty slot inside the window owns no KV blocks: position -1 tells the paged
        // kernels to skip it (run.cu lets it write at position 0 of its own reserved slot)
        h_pos[i] = handling_reqs[i].is_done ? -1 : handling_reqs[i].pos;
        if (handling_reqs[i].is_done) {
            continue;
        }
        int token = handling_reqs[i].token;
        RunState *s = &handling_reqs[i].run_state;
        // copy the token embedding into s->x
        float *content_row = w->token_embedding_table + (size_t)token * dim;

        // both pointers are device memory: copy on the GPU (a CPU memcpy would segfault)
        CHECK_CUDA(cudaMemcpy(s->x, content_row, dim * sizeof(float), cudaMemcpyDeviceToDevice));
    }
    CHECK_CUDA(cudaMemcpy(batch_s->p_pos, h_pos, batch_size * sizeof(int), cudaMemcpyHostToDevice));
    free(h_pos);

    // block tables of the window's slots (rows of empty slots are never read)
    size_t row = pool->max_blocks_per_seq;
    int *d_table = pool->d_table + (size_t)first * row;
    CHECK_CUDA(cudaMemcpy(d_table, pool->h_table.data() + (size_t)first * row,
                          (size_t)batch_size * row * sizeof(int), cudaMemcpyHostToDevice));

    // launch shapes of matmul_gpu_v4 and batch_multihead_attention_gpu
    int kv_serial_loads = DIVUP(DIVUP(dim, WARP_SIZE), 4);
    dim3 kv_block(WARP_SIZE, 4), kv_grid(DIVUP(kv_dim, 4));
    dim3 att_block(BLOCK_SIZE * BLOCK_SIZE), att_grid(p->n_heads, batch_size);
    float att_scale = 1.0 / sqrt(head_size);

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        rmsnorm_gpu(batch_s->xb, batch_s->x, w->rms_att_weight + l * dim, dim, batch_size);

        matmul_gpu_v2(batch_s->q, batch_s->xb, w->wq + l * dim * dim, dim, dim, batch_size);

        // K and V of the current positions, written into each slot's block for that position
        paged_mat_kv_kernel<<<kv_grid, kv_block>>>(pool->k, pool->v, batch_s->xb,
                                                   w->wk + l * dim * kv_dim, w->wv + l * dim * kv_dim,
                                                   dim, kv_dim, kv_serial_loads, batch_s->p_pos, d_table,
                                                   (int)row, (int)l, p->n_layers, PAGE_SIZE, kv_dim, batch_size);
        CHECK_CUDA(cudaGetLastError());

        // Fused RoPE & multihead attention over the slot's blocks
        paged_fused_rope_mha<<<att_grid, att_block>>>(batch_s->p_pos, batch_s->q, batch_s->xb, pool->k, pool->v,
                                                      d_table, (int)row, (int)l, p->n_layers, PAGE_SIZE,
                                                      dim, kv_dim, kv_mul, head_size, att_scale);
        CHECK_CUDA(cudaGetLastError());

        // final matmul to get the output of the attention
        // residual connection back into x
        matmul_skipadd_gpu(batch_s->x, batch_s->xb, w->wo + l * dim * dim, dim, dim, batch_size);

        // ffn rmsnorm
        rmsnorm_gpu(batch_s->xb, batch_s->x, w->rms_ffn_weight + l * dim, dim, batch_size);

        // SwiGLU non-linearity
        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul_swiglu_gpu(batch_s->hb, batch_s->xb,
                          w->w1 + l * dim * hidden_dim,
                          w->w3 + l * dim * hidden_dim,
                          dim, hidden_dim, batch_size);

        // final matmul to get the output of the ffn
        // residual connection
        matmul_skipadd_gpu(batch_s->x, batch_s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim, batch_size);
    }

    rmsnorm_gpu(batch_s->x, batch_s->x, w->rms_final_weight, dim, batch_size);
    matmul_gpu_v2(batch_s->logits, batch_s->x, w->wcls, p->dim, p->vocab_size, batch_size);
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct
{
    char *str;
    int id;
} TokenIndex;

typedef struct
{
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];  // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;  // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0';  // add the string terminating token
    }
    // lazily malloc and sort the vocabulary
    t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);

    fclose(file);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) {
        free(t->vocab[i]);
    }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) {
        return;
    }
    if (piece[0] == '\0') {
        return;
    }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;  // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

void append_str(char *piece, std::string &str) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) {
        return;
    }
    if (piece[0] == '\0') {
        return;
    }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;  // bad byte, don't print it
        }
    }
    // printf("%s", piece);
    str += piece;
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = {.str = str};  // acts as the key to search for
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = (char *)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos)
        tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char *)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {
        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c;  // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;  // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;  // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--;  // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos)
        tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct
{
    float prob;
    int index;
} ProbIndex;  // struct used when sorting probabilities during top-p sampling

typedef struct
{
    int vocab_size;
    ProbIndex *probindex;  // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
    printf("Sample mult\n");
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;  // in case of rounding errors
}

int compare(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *)a;
    ProbIndex *b_ = (ProbIndex *)b;
    if (a_->prob > b_->prob)
        return -1;
    if (a_->prob < b_->prob)
        return 1;
    return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;  // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;  // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;  // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) {  // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q = 0; q < sampler->vocab_size; q++) {
            logits[q] /= sampler->temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax_cpu(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

int sample_greedy(Sampler *sampler, float *logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits, sampler->vocab_size);
    return next;
}

int sample_determin(const Sampler *sampler, float *logits, unsigned long long *rng_states, int idx) {
    // sample the token given the logits and some hyperparameters
    int next;
    float temperature = 1.0f;
    // apply the temperature to the logits
    for (int q = 0; q < sampler->vocab_size; q++) {
        logits[q] /= temperature;
    }
    // apply softmax to the logits to get the probabilities for next token
    softmax_cpu(logits, sampler->vocab_size);
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&rng_states[idx]);

    next = sample_mult(logits, sampler->vocab_size, coin);
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';  // strip newline
        }
    }
}

void printHandlingReq(HandlingReq *req) {
    printf("HandlingReq: is_done: %d, pos: %d, token: %d, num_prompt_tokens: %d, next: %d, req_id: %d\n", req->is_done, req->pos, req->token, req->num_prompt_tokens, req->next, req->req_id);
    fflush(stdout);
}

// ----------------------------------------------------------------------------
// You should parallelize and optimize from this function exploiting multiple GPUs
//
// ----------------------------------------------------------------------------
// test mode: continuous batching over all requests of the input file
#define HARNESS_NO_RUN_TEST
#include "harness.h"

typedef struct {
    long fwd_tokens, gen_tokens;  // useful work only: preempted (recomputed) steps are not counted
    long preemptions, recomputed_tokens;
} Counters;

// Per-slot bookkeeping that HandlingReq (kernels/types.h, shared with run.cu) does not have.
struct SlotInfo {
    std::vector<int> blocks;  // physical KV blocks of the request, in logical order
    long admit_seq = 0;       // admission order: the most recent is preempted first
    long gen = 0;             // tokens sampled so far, undone if the request is preempted
};

// Same stop rule and output format as harness.h run_test() and run.cu's test(), with the KV
// cache in a block pool and vLLM-style FCFS scheduling with recompute preemption.
void test(Transformer *transformer, Tokenizer *tokenizer, const char *model_path,
          const char *input_path, const char *output_path, int steps, float temperature,
          float topp, unsigned long long seed) {
    Requests reqs;
    read_requests(input_path, &reqs);
    int num_reqs = reqs.num_reqs;
    Sampler *samplers = (Sampler *)malloc(num_reqs * sizeof(Sampler));
    StrBuf *outs = (StrBuf *)calloc(num_reqs, sizeof(StrBuf));
    for (int r = 0; r < num_reqs; r++)
        build_sampler(&samplers[r], transformer->config.vocab_size, temperature, topp, seed);

    // shared waiting queue (FCFS); preempted requests go back to the front
    std::deque<int> waiting;
    for (int r = 0; r < num_reqs; r++) waiting.push_back(r);
    Counters total = {0, 0, 0, 0};
    long total_blocks = 0, total_peak = 0;
    omp_set_max_active_levels(1);

    double t0 = harness_now();
#pragma omp parallel num_threads(NUM_GPUS)
    {
        int gpu_id = omp_get_thread_num();
        CHECK_CUDA(cudaSetDevice(gpu_id));
        Config *p = &transformer->config;
        malloc_run_state(&batch_run_state_d[gpu_id], p);
        KVPool *pool = &kv_pool[gpu_id];
        alloc_kv_pool(pool, p);
        const size_t row = pool->max_blocks_per_seq;
        // vLLM keeps 1% of the blocks free at admission so that running requests can grow
        // for a while before anything has to be preempted
        const long watermark = pool->num_blocks / 100 > 1 ? pool->num_blocks / 100 : 1;
        // one request alone must always fit, otherwise it would preempt itself forever
        if (pool->num_blocks < DIVUP(steps, PAGE_SIZE)) {
            fprintf(stderr, "KV pool of %ld blocks cannot hold one request of %d positions (%d blocks)\n",
                    pool->num_blocks, steps, DIVUP(steps, PAGE_SIZE));
            exit(EXIT_FAILURE);
        }

        HandlingReq *slots = new HandlingReq[BATCH_SIZE];
        std::vector<SlotInfo> info(BATCH_SIZE);
        for (int i = 0; i < BATCH_SIZE; ++i) {
            slots[i].reset();
            slots[i].prompt_tokens = NULL;
            batch_map_run_state(&batch_run_state_d[gpu_id], &slots[i].run_state, p, i);
        }
        Counters mine = {0, 0, 0, 0};
        long admit_counter = 0;

        auto take_block = [&](int i) {
            int blk = pool->free_list.back();
            pool->free_list.pop_back();
            pool->h_table[i * row + info[i].blocks.size()] = blk;
            info[i].blocks.push_back(blk);
        };
        auto release_blocks = [&](int i) {
            for (int blk : info[i].blocks) pool->free_list.push_back(blk);
            info[i].blocks.clear();
        };
        // recompute preemption: drop everything the request computed and queue it again
        auto preempt = [&](int i) {
            int req = slots[i].req_id;
            mine.preemptions++;
            mine.recomputed_tokens += slots[i].pos;
            mine.gen_tokens -= info[i].gen;
            release_blocks(i);
            free(slots[i].prompt_tokens);
            slots[i].prompt_tokens = NULL;
            slots[i].reset();
            outs[req].len = 0;  // restart the output from scratch...
            if (outs[req].s) outs[req].s[0] = '\0';
            samplers[req].rng_state = seed;  // ...and the sampler from its seed: same tokens again
#pragma omp critical(queue)
            waiting.push_front(req);
        };

        while (true) {
            // 1. admission: refill free slots from the front of the queue while the pool can
            //    hold the request's prompt and keep the watermark free
            for (int i = 0; i < BATCH_SIZE; ++i) {
                if (!slots[i].is_done) continue;
                int req_id = -1;
#pragma omp critical(queue)
                {
                    if (!waiting.empty()) {
                        req_id = waiting.front();
                        waiting.pop_front();
                    }
                }
                if (req_id < 0) break;
                char *prompt = reqs.prompts[req_id];
                int *tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
                int num_tokens;
                encode(tokenizer, prompt, 1, 0, tokens, &num_tokens);

                long need = DIVUP(num_tokens < steps ? num_tokens : steps, PAGE_SIZE);
                long avail = (long)pool->free_list.size();
                bool idle = true;
                for (int j = 0; j < BATCH_SIZE; ++j) idle = idle && slots[j].is_done;
                // an idle GPU admits without the watermark, so a request always gets started
                if (avail - need < (idle ? 0 : watermark)) {
                    free(tokens);
#pragma omp critical(queue)
                    waiting.push_front(req_id);
                    break;  // FCFS: no later request may overtake this one
                }
                slots[i].prompt_tokens = tokens;
                slots[i].num_prompt_tokens = num_tokens;
                slots[i].is_done = false;
                slots[i].pos = 0;
                slots[i].token = tokens[0];
                slots[i].req_id = req_id;
                info[i].admit_seq = ++admit_counter;
                info[i].gen = 0;
                for (long b = 0; b < need; ++b) take_block(i);
            }

            // 2. growth: every running request needs the block of its current position.
            //    Oldest first; when the pool is empty the newest running request is preempted
            //    (possibly the one asking, which then simply stops for now).
            std::vector<int> order;
            for (int i = 0; i < BATCH_SIZE; ++i)
                if (!slots[i].is_done) order.push_back(i);
            std::sort(order.begin(), order.end(),
                      [&](int a, int b) { return info[a].admit_seq < info[b].admit_seq; });
            for (int i : order) {
                if (slots[i].is_done) continue;  // preempted earlier in this loop
                size_t need = slots[i].pos / PAGE_SIZE + 1;
                while (info[i].blocks.size() < need) {
                    if (!pool->free_list.empty()) {
                        take_block(i);
                        continue;
                    }
                    int victim = -1;
                    for (int k = (int)order.size() - 1; k >= 0 && victim < 0; --k)
                        if (!slots[order[k]].is_done) victim = order[k];
                    preempt(victim);
                    if (victim == i) break;
                }
            }
            long used = pool->num_blocks - (long)pool->free_list.size();
            if (used > pool->peak_used) pool->peak_used = used;

            // 3. run the contiguous window [first, last] of busy slots (empty slots inside the
            //    window get position -1 and are skipped by the paged kernels)
            int first = 0, last = BATCH_SIZE - 1;
            while (first < BATCH_SIZE && slots[first].is_done) first++;
            while (last >= 0 && slots[last].is_done) last--;
            if (first > last) {
                bool more;
#pragma omp critical(queue)
                more = !waiting.empty();
                if (more) continue;  // everything here was preempted: admit again
                break;               // queue empty and nothing running on this GPU
            }
            RunState window = batch_run_state_d[gpu_id];
            batch_map_run_state(&batch_run_state_d[gpu_id], &window, p, first);
            forward(transformer, &slots[first], &window, &weights_d[gpu_id], last - first + 1, pool, first);

            // 4. advance every busy slot by one token
            for (int i = first; i <= last; ++i) {
                HandlingReq *r = &slots[i];
                if (r->is_done) continue;
                if (r->pos < r->num_prompt_tokens - 1) {
                    r->next = r->prompt_tokens[r->pos + 1];
                } else {
                    r->next = sample(&samplers[r->req_id], r->run_state.logits);
                    mine.gen_tokens++;
                    info[i].gen++;
                }
                r->pos++;
                bool stop = r->next == 1 || r->next == 2;
                if (!stop) {
                    sb_append_piece(&outs[r->req_id], decode(tokenizer, r->token, r->next));
                    r->token = r->next;
                }
                if (stop || r->pos >= steps) {
                    mine.fwd_tokens += r->pos;
                    release_blocks(i);
                    free(r->prompt_tokens);
                    r->prompt_tokens = NULL;
                    r->reset();
                }
            }
        }

#pragma omp critical(counters)
        {
            total.fwd_tokens += mine.fwd_tokens;
            total.gen_tokens += mine.gen_tokens;
            total.preemptions += mine.preemptions;
            total.recomputed_tokens += mine.recomputed_tokens;
            total_blocks += pool->num_blocks;
            total_peak += pool->peak_used;
        }

        delete[] slots;
        free_kv_pool(pool);
        free_run_state(&batch_run_state_d[gpu_id]);
    }
    double elapsed = harness_now() - t0;

    for (int r = 0; r < num_reqs; r++) {
        reqs.outputs[r] = outs[r].s ? outs[r].s : strdup("");
        free_sampler(&samplers[r]);
    }
    write_outputs(output_path, &reqs);
    char impl[32];
    snprintf(impl, sizeof impl, "gpu_paged_b%d", BATCH_SIZE);
    print_result(impl, model_path, num_reqs, steps, total.fwd_tokens, total.gen_tokens, elapsed);
    printf("PAGED page_size=%d kv_blocks=%ld kv_tokens=%ld peak_blocks=%ld preemptions=%ld recomputed_tokens=%ld\n",
           PAGE_SIZE, total_blocks, total_blocks * PAGE_SIZE, total_peak, total.preemptions,
           total.recomputed_tokens);
    free_requests(&reqs);
    free(samplers);
    free(outs);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   runpgattn <checkpoint> -m test -f <input> -o <output> [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -b <int>    batch size per GPU, 1..%d, default 1\n", MAX_BATCH_SIZE);
    fprintf(stderr, "  -n <int>    max positions per request, default 256 (0 = seq_len)\n");
    fprintf(stderr, "  -t <float>  temperature, default 1.0 (0 = greedy)\n");
    fprintf(stderr, "  -p <float>  top-p, default 0.9\n");
    fprintf(stderr, "  -s <int>    sampler seed (same for every request), default 314028\n");
    fprintf(stderr, "  -z <string> tokenizer path, default tokenizer.bin\n");
    fprintf(stderr, "  -m <string> mode: only 'test' is supported\n");
    fprintf(stderr, "  -k <int>    KV block (page) size in token positions, default 16\n");
    fprintf(stderr, "  -c <int>    max KV blocks per GPU, default 0 = all free VRAM (small values force preemption)\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;
    char default_tokenizer[] = "tokenizer.bin";
    char *tokenizer_path = default_tokenizer;
    float temperature = 1.0f, topp = 0.9f;
    int steps = 256;
    unsigned long long seed = 314028;
    const char *mode = "test";
    char *input_path = NULL, *output_path = NULL;

    if (argc < 2) error_usage();
    checkpoint_path = argv[1];
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2) error_usage();
        switch (argv[i][1]) {
            case 't': temperature = atof(argv[i + 1]); break;
            case 'p': topp = atof(argv[i + 1]); break;
            case 's': seed = strtoull(argv[i + 1], NULL, 10); break;
            case 'n': steps = atoi(argv[i + 1]); break;
            case 'z': tokenizer_path = argv[i + 1]; break;
            case 'm': mode = argv[i + 1]; break;
            case 'f': input_path = argv[i + 1]; break;
            case 'o': output_path = argv[i + 1]; break;
            case 'b': BATCH_SIZE = atoi(argv[i + 1]); break;
            case 'k': PAGE_SIZE = atoi(argv[i + 1]); break;
            case 'c': MAX_BLOCKS = atol(argv[i + 1]); break;
            default: error_usage();
        }
    }
    if (strcmp(mode, "test") != 0 || !input_path || !output_path) error_usage();
    if (PAGE_SIZE < 1 || MAX_BLOCKS < 0) error_usage();
    if (BATCH_SIZE < 1 || BATCH_SIZE > MAX_BATCH_SIZE) {
        fprintf(stderr, "-b must be in 1..%d (kernels keep one register per request, see types.h)\n", MAX_BATCH_SIZE);
        exit(EXIT_FAILURE);
    }
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;

    CHECK_CUDA(cudaGetDeviceCount(&NUM_GPUS));
    if (NUM_GPUS > MAX_GPUS) NUM_GPUS = MAX_GPUS;
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    Config *p = &transformer.config;
    if (p->n_kv_heads != p->n_heads) {
        fprintf(stderr, "grouped-query attention (n_kv_heads=%d < n_heads=%d) is not supported: "
                        "the fused RoPE+attention kernel would rotate shared keys several times\n",
                p->n_kv_heads, p->n_heads);
        exit(EXIT_FAILURE);
    }
    if (steps <= 0 || steps > p->seq_len) steps = p->seq_len;
    if (steps > MAX_SEQ_LEN) {  // the attention kernel keeps the scores of one head in shared memory
        fprintf(stderr, "-n %d exceeds MAX_SEQ_LEN %d of the attention kernel\n", steps, MAX_SEQ_LEN);
        exit(EXIT_FAILURE);
    }
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, p->vocab_size);

    test(&transformer, &tokenizer, checkpoint_path, input_path, output_path, steps, temperature, topp, seed);

    free_tokenizer(&tokenizer);
    for (int i = 1; i < NUM_GPUS; ++i) CHECK_CUDA(cudaFree(weights_ptr[i]));  // [0] freed below
    free_transformer(&transformer);
    return 0;
}
#endif
