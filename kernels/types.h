#pragma once
#include <sys/types.h>

typedef struct {
    int dim;         // transformer dimension
    int hidden_dim;  // for ffn layers
    int n_layers;    // number of layers
    int n_heads;     // number of query heads
    int n_kv_heads;  // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;  // vocabulary size, usually 256 (byte-level)
    int seq_len;     // max sequence length
} Config;

// Same layout as run.c. In the GPU build every pointer is a device pointer.
typedef struct {
    float *token_embedding_table;  // (vocab_size, dim)
    float *rms_att_weight;         // (layer, dim)
    float *rms_ffn_weight;         // (layer, dim)
    float *wq;                     // (layer, dim, n_heads * head_size)
    float *wk;                     // (layer, dim, n_kv_heads * head_size)
    float *wv;                     // (layer, dim, n_kv_heads * head_size)
    float *wo;                     // (layer, n_heads * head_size, dim)
    float *w1;                     // (layer, hidden_dim, dim)
    float *w2;                     // (layer, dim, hidden_dim)
    float *w3;                     // (layer, hidden_dim, dim)
    float *rms_final_weight;       // (dim,)
    float *wcls;                   // (vocab_size, dim)
} TransformerWeights;

// Activation buffers. Everything lives on the GPU except `logits`, which is a host
// buffer that receives a copy of `logits_gpu` after every forward pass (sampling runs
// on the CPU, as in run.c).
typedef struct {
    float *x;            // (dim,)
    float *xb;           // (dim,)
    float *xb2;          // (dim,)
    float *hb;           // (hidden_dim,)
    float *hb2;          // (hidden_dim,)
    float *q;            // (dim,)
    float *k;            // points into key_cache at the current position
    float *v;            // points into value_cache at the current position
    float *att;          // (n_heads, seq_len)
    float *logits_gpu;   // (vocab_size,) device
    float *logits;       // (vocab_size,) host
    float *key_cache;    // (layer, seq_len, kv_dim)
    float *value_cache;  // (layer, seq_len, kv_dim)
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;  // device pointers
    RunState state;
    float *weights_gpu;          // one device allocation holding all weights
    int shared_weights;          // wcls == token_embedding_table
    int fd;                      // checkpoint file descriptor (kept for unit tests)
    float *data;                 // host mmap of the checkpoint (kept for unit tests)
    ssize_t file_size;
} Transformer;
