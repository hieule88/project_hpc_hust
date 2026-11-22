#pragma once

// Upper bound for the batch size: the batched GEMV kernels keep one partial sum per
// request in registers (float sum[MAX_BATCH_SIZE]). run.cu / run_pp.cu reject larger -b.
#define MAX_BATCH_SIZE 10
#include <stdio.h>
#include <unistd.h>

#include <string>

typedef struct {
    int dim;         // transformer dimension
    int hidden_dim;  // for ffn layers
    int n_layers;    // number of layers
    int n_heads;     // number of query heads
    int n_kv_heads;  // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;  // vocabulary size, usually 256 (byte-level)
    int seq_len;     // max sequence length
} Config;

typedef struct {
    // which GPU to store
    int gpu_id;
    // weights for rmsnorms
    float *rms_att_weight;  // (dim,) rmsnorm weights
    float *rms_ffn_weight;  // (dim,)
    // weights for matmuls. note dim == n_heads * head_size
    float *wq;  // (dim, n_heads * head_size)
    float *wk;  // (dim, n_kv_heads * head_size)
    float *wv;  // (dim, n_kv_heads * head_size)
    float *wo;  // (n_heads * head_size, dim)
    // weights for ffn
    float *w1;  // (hidden_dim, dim)
    float *w2;  // (dim, hidden_dim)
    float *w3;  // (hidden_dim, dim)
} PerLayerWeight;

typedef struct {
    // token embedding table
    float *token_embedding_table;  // (vocab_size, dim)
    // weights for rmsnorms
    float *rms_att_weight;  // (layer, dim) rmsnorm weights
    float *rms_ffn_weight;  // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float *wq;  // (layer, dim, n_heads * head_size)
    float *wk;  // (layer, dim, n_kv_heads * head_size)
    float *wv;  // (layer, dim, n_kv_heads * head_size)
    float *wo;  // (layer, n_heads * head_size, dim)
    // weights for ffn
    float *w1;  // (layer, hidden_dim, dim)
    float *w2;  // (layer, dim, hidden_dim)
    float *w3;  // (layer, hidden_dim, dim)
    // final rmsnorm
    float *rms_final_weight;  // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float *wcls;
    // weights for each layer
    PerLayerWeight *layers;
} TransformerWeights;

typedef struct {
    int *p_pos;
    // current wave of activations
    float *x;       // activation at current time stamp (dim,)
    float *xb;      // same, but inside a residual branch (dim,)
    float *xb2;     // an additional buffer just for convenience (dim,)
    float *hb;      // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2;     // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q;       // query (dim,)
    float *k;       // key (dim,)
    float *v;       // value (dim,)
    float *att;     // buffer for scores/attention values (n_heads, seq_len)
    float *logits;  // output logits in CPU
    // kv cache
    float *key_cache;    // (layer, seq_len, dim)
    float *value_cache;  // (layer, seq_len, dim)
    // float *kv_cache;     // (layer, seq_len, dim, 2)
} RunState;

typedef struct {
    Config config;               // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights;  // the weights of the model
    RunState state;              // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd;             // file descriptor for memory mapping
    float *data;        // memory mapped data pointer
    ssize_t file_size;  // size of the checkpoint file in bytes
} Transformer;

struct HandlingReq {
    int *prompt_tokens;
    std::string gen_str;
    bool is_done;
    int pos;
    int token;
    int num_prompt_tokens;
    int next;
    int req_id;
    RunState run_state;

    void reset() {
        is_done = true;
        gen_str = "";
        pos = 0;
        token = 0;
        num_prompt_tokens = 0;
        next = 0;
        req_id = 0;
    }
};
