/*
 * Naive CUDA port of llama2.c (run.c).
 *
 * Every step of run.c's forward() becomes one or more simple kernels (see kernels/):
 * one thread per output element, shared-memory tree reductions, no fusion, no batching,
 * no vectorized loads. The goal is a straightforward baseline to compare against the CPU
 * code and against an optimized GPU version later.
 *
 * Weights, activations and the KV cache live on the GPU. After each forward pass the
 * logits are copied back to the host and sampled on the CPU, exactly as in run.c.
 *
 * Modes:
 *   generate / chat : same behaviour and CLI as run.c
 *   test            : run every prompt of an input file (common/harness.h) and print a
 *                     RESULT line with the throughput
 * Set LLAMA_PROFILE=1 to print a per-kernel-group time breakdown at the end.
 */
#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#include "kernels/attn.h"
#include "kernels/matmul.h"
#include "kernels/rmsnorm.h"
#include "kernels/rope.h"
#include "kernels/skip_conn.h"
#include "kernels/softmax.h"
#include "kernels/swiglu.h"
#include "kernels/types.h"
#include "kernels/utils.h"

// ----------------------------------------------------------------------------
// Optional profiling: time each group of kernels with CUDA events.
// Each measured group is synchronized, so profiling slows the run down a little.

enum {
    P_EMBED, P_RMSNORM, P_QKV, P_ROPE, P_ATTN, P_WO, P_RESIDUAL,
    P_W1W3, P_SWIGLU, P_W2, P_CLASSIFIER, P_LOGITS_D2H, P_COUNT
};
static const char *prof_names[P_COUNT] = {
    "embedding copy", "rmsnorm", "matmul q/k/v", "rope", "attention", "matmul wo",
    "residual add", "matmul w1/w3", "swiglu", "matmul w2", "matmul classifier", "logits D2H"};
static int g_profile = 0;
static double prof_ms[P_COUNT];
static long prof_tokens = 0;
static cudaEvent_t prof_ev0, prof_ev1;

#define PROF(cat, stmt)                                                   \
    do {                                                                  \
        if (g_profile) {                                                  \
            CHECK_CUDA(cudaEventRecord(prof_ev0));                        \
            stmt;                                                         \
            CHECK_CUDA(cudaEventRecord(prof_ev1));                        \
            CHECK_CUDA(cudaEventSynchronize(prof_ev1));                   \
            float ms_;                                                    \
            CHECK_CUDA(cudaEventElapsedTime(&ms_, prof_ev0, prof_ev1));   \
            prof_ms[cat] += ms_;                                          \
        } else {                                                          \
            stmt;                                                         \
        }                                                                 \
    } while (0)

static void init_profile(void) {
    const char *env = getenv("LLAMA_PROFILE");
    g_profile = env && env[0] == '1';
    if (g_profile) {
        CHECK_CUDA(cudaEventCreate(&prof_ev0));
        CHECK_CUDA(cudaEventCreate(&prof_ev1));
    }
}

static void print_profile(void) {
    if (!g_profile || prof_tokens == 0) return;
    double total = 0;
    for (int i = 0; i < P_COUNT; i++) total += prof_ms[i];
    printf("\nPROFILE (GPU time per kernel group, %ld forward passes)\n", prof_tokens);
    printf("| group | total ms | %% | us / token |\n|---|---|---|---|\n");
    for (int i = 0; i < P_COUNT; i++) {
        printf("| %s | %.1f | %.1f | %.1f |\n", prof_names[i], prof_ms[i],
               100.0 * prof_ms[i] / total, 1000.0 * prof_ms[i] / prof_tokens);
    }
    printf("| total | %.1f | 100.0 | %.1f |\n", total, 1000.0 * total / prof_tokens);
}

// ----------------------------------------------------------------------------
// Transformer model

void malloc_run_state(RunState *s, Config *p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    size_t cache = (size_t)p->n_layers * p->seq_len * kv_dim * sizeof(float);
    CHECK_CUDA(cudaMalloc((void **)&s->x, p->dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&s->xb, p->dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&s->xb2, p->dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&s->hb, p->hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&s->hb2, p->hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&s->q, p->dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&s->key_cache, cache));
    CHECK_CUDA(cudaMalloc((void **)&s->value_cache, cache));
    CHECK_CUDA(cudaMemset(s->key_cache, 0, cache));
    CHECK_CUDA(cudaMemset(s->value_cache, 0, cache));
    CHECK_CUDA(cudaMalloc((void **)&s->att, (size_t)p->n_heads * p->seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&s->logits_gpu, p->vocab_size * sizeof(float)));
    s->logits = (float *)calloc(p->vocab_size, sizeof(float));
    if (!s->logits) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
}

void free_run_state(RunState *s) {
    CHECK_CUDA(cudaFree(s->x));
    CHECK_CUDA(cudaFree(s->xb));
    CHECK_CUDA(cudaFree(s->xb2));
    CHECK_CUDA(cudaFree(s->hb));
    CHECK_CUDA(cudaFree(s->hb2));
    CHECK_CUDA(cudaFree(s->q));
    CHECK_CUDA(cudaFree(s->att));
    CHECK_CUDA(cudaFree(s->logits_gpu));
    CHECK_CUDA(cudaFree(s->key_cache));
    CHECK_CUDA(cudaFree(s->value_cache));
    free(s->logits);
}

// Same as run.c; works for host or device pointers (it only does pointer arithmetic).
void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;  // 64-bit products for large models
    w->token_embedding_table = ptr;
    ptr += (size_t)p->vocab_size * p->dim;
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

// mmap the checkpoint, then copy all weights to the GPU in one allocation.
void read_checkpoint(const char *checkpoint, Transformer *t) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    if (fread(&t->config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    t->shared_weights = t->config.vocab_size > 0 ? 1 : 0;
    t->config.vocab_size = abs(t->config.vocab_size);
    fseek(file, 0, SEEK_END);
    t->file_size = ftell(file);
    fclose(file);
    t->fd = open(checkpoint, O_RDONLY);
    if (t->fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    t->data = (float *)mmap(NULL, t->file_size, PROT_READ, MAP_PRIVATE, t->fd, 0);
    if (t->data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

    size_t weights_bytes = t->file_size - sizeof(Config);
    CHECK_CUDA(cudaMalloc((void **)&t->weights_gpu, weights_bytes));
    CHECK_CUDA(cudaMemcpy(t->weights_gpu, t->data + sizeof(Config) / sizeof(float),
                          weights_bytes, cudaMemcpyHostToDevice));
    memory_map_weights(&t->weights, &t->config, t->weights_gpu, t->shared_weights);
}

void build_transformer(Transformer *t, const char *checkpoint_path) {
    read_checkpoint(checkpoint_path, t);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    CHECK_CUDA(cudaFree(t->weights_gpu));
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// Forward pass: same sequence of operations as run.c, each one on the GPU.

float *forward(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    TransformerWeights *w = &transformer->weights;
    RunState *s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x (device to device)
    PROF(P_EMBED, CHECK_CUDA(cudaMemcpy(x, w->token_embedding_table + (size_t)token * dim,
                                        dim * sizeof(float), cudaMemcpyDeviceToDevice)));

    for (unsigned long long l = 0; l < (unsigned long long)p->n_layers; l++) {
        // attention rmsnorm
        PROF(P_RMSNORM, rmsnorm_gpu(s->xb, x, w->rms_att_weight + l * dim, dim));

        // key and value point to the kv cache
        long long loff = (long long)l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + (long long)pos * kv_dim;
        s->v = s->value_cache + loff + (long long)pos * kv_dim;

        // qkv matmuls for this position
        PROF(P_QKV, {
            matmul_gpu(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
            matmul_gpu(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
            matmul_gpu(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);
        });

        // RoPE relative positional encoding
        PROF(P_ROPE, rope_gpu(s->q, s->k, pos, dim, kv_dim, head_size));

        // multihead attention over positions 0..pos
        PROF(P_ATTN, attention_gpu(s->xb, s->q, s->att, s->key_cache, s->value_cache, pos,
                                   p->seq_len, p->n_heads, head_size, kv_dim, kv_mul, loff));

        // output projection, then residual connection
        PROF(P_WO, matmul_gpu(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim));
        PROF(P_RESIDUAL, residual_add_gpu(x, s->xb2, dim));

        // ffn rmsnorm
        PROF(P_RMSNORM, rmsnorm_gpu(s->xb, x, w->rms_ffn_weight + l * dim, dim));

        // FFN: w2(silu(w1(x)) * w3(x))
        PROF(P_W1W3, {
            matmul_gpu(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
            matmul_gpu(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        });
        PROF(P_SWIGLU, swiglu_gpu(s->hb, s->hb2, hidden_dim));
        PROF(P_W2, matmul_gpu(s->xb, s->hb, w->w2 + l * dim * hidden_dim, hidden_dim, dim));

        // residual connection
        PROF(P_RESIDUAL, residual_add_gpu(x, s->xb, dim));
    }

    // final rmsnorm and classifier into logits
    PROF(P_RMSNORM, rmsnorm_gpu(x, x, w->rms_final_weight, dim));
    PROF(P_CLASSIFIER, matmul_gpu(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size));

    // copy logits back to the host for sampling (this also waits for all kernels)
    PROF(P_LOGITS_D2H, CHECK_CUDA(cudaMemcpy(s->logits, s->logits_gpu,
                                             p->vocab_size * sizeof(float),
                                             cudaMemcpyDeviceToHost)));
    prof_tokens++;
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer (unchanged from run.c, except C++ casts)

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
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

void build_tokenizer(Tokenizer *t, const char *tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;  // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // raw byte tokens look like '<0x01>'
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) { return; }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok;
    tok.str = str;
    tok.id = 0;
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char *str_buffer = (char *)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    if (text[0] != '\0') {
        char space[] = " ";
        int dummy_prefix = str_lookup(space, t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) { str_len = 0; }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) { continue; }
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: +3 because the first 3 vocab elements are <unk>, <s>, </s>
            for (size_t i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) { break; }
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) { tokens[i] = tokens[i + 1]; }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler (unchanged from run.c, except C++ casts). Runs on the CPU.

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;  // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float *probabilities, int n) {
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
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) { return i; }
    }
    return n - 1;
}

int compare(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *)a;
    ProbIndex *b_ = (ProbIndex *)b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) { return probindex[i].index; }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = (ProbIndex *)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) { free(sampler->probindex); }

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { return (random_u32(state) >> 8) / 16777216.0f; }

int sample(Sampler *sampler, float *logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax_cpu(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// generation loop (unchanged from run.c)

long time_in_ms() {
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char empty_prompt[] = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        float *logits = forward(transformer, token, pos);
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;
        if (next == 1) { break; }
        char *piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }
    free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') { buffer[len - 1] = '\0'; }
    }
}

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int *prompt_tokens = (int *)malloc(1152 * sizeof(int));
    int user_idx = 0;

    int8_t user_turn = 1;
    int next = 0;
    int token;
    int pos = 0;
    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                if (cli_system_prompt == NULL) {
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            if (pos == 0 && cli_user_prompt != NULL) {
                strcpy(user_prompt, cli_user_prompt);
            } else {
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0;
            user_turn = 0;
            printf("Assistant: ");
        }
        if (user_idx < num_prompt_tokens) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        if (token == 2) { user_turn = 1; }
        float *logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;
        if (user_idx >= num_prompt_tokens && next != 2) {
            char *piece = decode(tokenizer, token, next);
            safe_printf(piece);
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}

// test mode: shared request loop (same code as cpu/runtest.c)
#include "../common/harness.h"

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   runcu <checkpoint> [options]\n");
    fprintf(stderr, "Example: runcu model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Example: runcu model.bin -m test -f input/12.txt -o output/out.txt\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL) (test mode: 314028)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat|test, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    fprintf(stderr, "  -f <string> (test mode) input file\n");
    fprintf(stderr, "  -o <string> (test mode) output file\n");
    fprintf(stderr, "Env:     LLAMA_PROFILE=1 prints a per-kernel-group time breakdown\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL;
    const char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;
    float topp = 0.9f;
    int steps = 256;
    char *prompt = NULL;
    unsigned long long rng_seed = 0;
    const char *mode = "generate";
    char *system_prompt = NULL;
    char *input_path = NULL;
    char *output_path = NULL;

    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = strtoull(argv[i + 1], NULL, 10); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else if (argv[i][1] == 'f') { input_path = argv[i + 1]; }
        else if (argv[i][1] == 'o') { output_path = argv[i + 1]; }
        else { error_usage(); }
    }
    int test_mode = strcmp(mode, "test") == 0;
    if (rng_seed <= 0) rng_seed = test_mode ? 314028 : (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    init_profile();
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    if (test_mode) {
        if (input_path == NULL || output_path == NULL) { error_usage(); }
        run_test(&transformer, &tokenizer, "gpu_naive", checkpoint_path, input_path, output_path,
                 steps, temperature, topp, rng_seed);
    } else {
        Sampler sampler;
        build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
        if (strcmp(mode, "generate") == 0) {
            generate(&transformer, &tokenizer, &sampler, prompt, steps);
        } else if (strcmp(mode, "chat") == 0) {
            chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
        } else {
            fprintf(stderr, "unknown mode: %s\n", mode);
            error_usage();
        }
        free_sampler(&sampler);
    }
    print_profile();

    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
