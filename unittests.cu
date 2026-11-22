/*
 * Unit tests for the optimized (batched, fused) CUDA kernels.
 *
 *   ./bin/unit-tests [model.bin] [--bench]
 *
 * 1. Kernel tests: every GPU function of kernels/ against the CPU reference functions
 *    (copied from run.c), for batch sizes 1, 3 and MAX_BATCH_SIZE, with a different
 *    position per slot where the kernel reads p_pos. A value passes when
 *    |gpu - cpu| <= atol + rtol * |cpu|.
 * 2. Forward test (needs a model): run.cu's batched forward() against a CPU forward per
 *    request. Three slots run different prompts, one of them joins 5 steps late, so the
 *    slots sit at different positions (as with continuous batching). Compares the logits
 *    of every busy slot at every step and the argmax token.
 * 3. --bench: time of the batched kernels at the model's shapes, batch 1 vs MAX_BATCH_SIZE.
 * Exit code is non-zero when any test fails.
 */
#define TESTING
#include "run.cu"

#include <vector>

static int g_failed = 0, g_passed = 0;
static unsigned int g_rng = 12345;

static float frand(void) {  // uniform in [-1, 1)
    g_rng = g_rng * 1664525u + 1013904223u;
    return (g_rng >> 8) / 8388608.0f - 1.0f;
}
static std::vector<float> rand_vec(size_t n, float scale = 1.0f) {
    std::vector<float> v(n);
    for (auto &x : v) x = frand() * scale;
    return v;
}
template <typename T>
static T *to_gpu(const std::vector<T> &v) {
    T *d;
    CHECK_CUDA(cudaMalloc((void **)&d, v.size() * sizeof(T)));
    CHECK_CUDA(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
static std::vector<float> from_gpu(const float *d, size_t n) {
    std::vector<float> v(n);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(v.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

static bool check_close(const char *name, const std::vector<float> &gpu,
                        const std::vector<float> &cpu, float atol, float rtol) {
    double max_abs = 0, max_ref = 0;
    size_t bad = 0, first_bad = 0;
    for (size_t i = 0; i < cpu.size(); i++) {
        double diff = fabs((double)gpu[i] - cpu[i]);
        max_abs = fmax(max_abs, diff);
        max_ref = fmax(max_ref, fabs(cpu[i]));
        if (!(diff <= atol + rtol * fabs(cpu[i]))) {  // also catches NaN
            if (bad == 0) first_bad = i;
            bad++;
        }
    }
    bool ok = bad == 0;
    printf("  [%s] %-58s max|diff|=%.2e  max|ref|=%.2e", ok ? "PASS" : "FAIL", name, max_abs, max_ref);
    if (!ok) printf("  %zu bad, first i=%zu gpu=%g cpu=%g", bad, first_bad, gpu[first_bad], cpu[first_bad]);
    printf("\n");
    ok ? g_passed++ : g_failed++;
    return ok;
}

static const int BATCHES[] = {1, 3, MAX_BATCH_SIZE};

// ----------------------------------------------------------------------------
// 1. kernel tests

// matmul_gpu_v1 (one block row per request), v2 (weights read once for the batch),
// skipadd (out += W x)
static void test_matmul(int n, int d, int B) {
    auto x = rand_vec((size_t)B * n), w = rand_vec((size_t)n * d), base = rand_vec((size_t)B * d);
    std::vector<float> ref((size_t)B * d);
    matmul_cpu(ref.data(), x.data(), w.data(), n, d, B);
    float *dx = to_gpu(x), *dw = to_gpu(w), *dout;
    CHECK_CUDA(cudaMalloc((void **)&dout, ref.size() * sizeof(float)));
    float atol = 1e-6f * n + 1e-6f, rtol = 1e-5f;  // float dot products of length n
    char name[96];

    matmul_gpu_v1(dout, dx, dw, n, d, B);
    snprintf(name, sizeof name, "matmul_gpu_v1 n=%d d=%d B=%d", n, d, B);
    check_close(name, from_gpu(dout, ref.size()), ref, atol, rtol);

    matmul_gpu_v2(dout, dx, dw, n, d, B);
    snprintf(name, sizeof name, "matmul_gpu_v2 n=%d d=%d B=%d", n, d, B);
    check_close(name, from_gpu(dout, ref.size()), ref, atol, rtol);

    CHECK_CUDA(cudaMemcpy(dout, base.data(), base.size() * sizeof(float), cudaMemcpyHostToDevice));
    matmul_skipadd_gpu(dout, dx, dw, n, d, B);
    auto ref_add = base;
    for (size_t i = 0; i < ref_add.size(); i++) ref_add[i] += ref[i];
    snprintf(name, sizeof name, "matmul_skipadd_gpu n=%d d=%d B=%d", n, d, B);
    check_close(name, from_gpu(dout, ref.size()), ref_add, atol, rtol);
    cudaFree(dx); cudaFree(dw); cudaFree(dout);
}

// matmul_swiglu_gpu: silu(W1 x) * (W3 x)
static void test_matmul_swiglu(int n, int d, int B) {
    auto x = rand_vec((size_t)B * n), w1 = rand_vec((size_t)n * d), w3 = rand_vec((size_t)n * d);
    std::vector<float> h1((size_t)B * d), h3((size_t)B * d);
    matmul_cpu(h1.data(), x.data(), w1.data(), n, d, B);
    matmul_cpu(h3.data(), x.data(), w3.data(), n, d, B);
    RunState s;
    s.hb = h1.data(); s.hb2 = h3.data();
    for (int b = 0; b < B; b++) {
        s.hb = h1.data() + (size_t)b * d; s.hb2 = h3.data() + (size_t)b * d;
        swiglu_cpu(&s, d);
    }
    float *dx = to_gpu(x), *dw1 = to_gpu(w1), *dw3 = to_gpu(w3), *dout;
    CHECK_CUDA(cudaMalloc((void **)&dout, h1.size() * sizeof(float)));
    matmul_swiglu_gpu(dout, dx, dw1, dw3, n, d, B);
    char name[96];
    snprintf(name, sizeof name, "matmul_swiglu_gpu n=%d d=%d B=%d", n, d, B);
    // silu(a)*b amplifies the dot-product rounding of a and b by up to |a|,|b| ~ 30
    check_close(name, from_gpu(dout, h1.size()), h1, 1e-4f * n, 1e-4f);
    cudaFree(dx); cudaFree(dw1); cudaFree(dw3); cudaFree(dout);
}

// matmul_gpu_v3 (K only) and v4 (K and V): write into the KV cache of slot b at p_pos[b]
static void test_matmul_kv(int dim, int kv_dim, int seq_len, int B) {
    int n_layers = 2;
    long long loff = (long long)1 * seq_len * kv_dim;  // layer 1
    size_t slot = (size_t)n_layers * seq_len * kv_dim;
    auto x = rand_vec((size_t)B * dim), wk = rand_vec((size_t)dim * kv_dim), wv = rand_vec((size_t)dim * kv_dim);
    auto kc = rand_vec(slot * B), vc = rand_vec(slot * B);
    std::vector<int> pos(B);
    for (int b = 0; b < B; b++) pos[b] = (b * 37 + 5) % seq_len;
    auto ref_k = kc, ref_v = vc;
    for (int b = 0; b < B; b++) {
        float *k = ref_k.data() + b * slot + loff + (size_t)pos[b] * kv_dim;
        float *v = ref_v.data() + b * slot + loff + (size_t)pos[b] * kv_dim;
        matmul_cpu(k, x.data() + (size_t)b * dim, wk.data(), dim, kv_dim, 1);
        matmul_cpu(v, x.data() + (size_t)b * dim, wv.data(), dim, kv_dim, 1);
    }
    float *dx = to_gpu(x), *dwk = to_gpu(wk), *dwv = to_gpu(wv), *dkc = to_gpu(kc), *dvc = to_gpu(vc);
    int *dpos = to_gpu(pos);
    Config p;
    float atol = 1e-6f * dim + 1e-6f;
    char name[96];

    matmul_gpu_v4(dkc, dvc, dx, dwk, dwv, dim, kv_dim, &p, dpos, kv_dim, loff, slot, B);
    snprintf(name, sizeof name, "matmul_gpu_v4 K cache dim=%d kv_dim=%d B=%d", dim, kv_dim, B);
    check_close(name, from_gpu(dkc, kc.size()), ref_k, atol, 1e-5f);  // whole cache: untouched parts too
    snprintf(name, sizeof name, "matmul_gpu_v4 V cache dim=%d kv_dim=%d B=%d", dim, kv_dim, B);
    check_close(name, from_gpu(dvc, vc.size()), ref_v, atol, 1e-5f);

    CHECK_CUDA(cudaMemcpy(dkc, kc.data(), kc.size() * sizeof(float), cudaMemcpyHostToDevice));
    matmul_gpu_v3(dkc, dx, dwk, dim, kv_dim, &p, dpos, kv_dim, loff, slot, B);
    snprintf(name, sizeof name, "matmul_gpu_v3 K cache dim=%d kv_dim=%d B=%d", dim, kv_dim, B);
    check_close(name, from_gpu(dkc, kc.size()), ref_k, atol, 1e-5f);
    cudaFree(dx); cudaFree(dwk); cudaFree(dwv); cudaFree(dkc); cudaFree(dvc); cudaFree(dpos);
}

static void test_rmsnorm(int size, int B, bool in_place) {
    auto x = rand_vec((size_t)B * size, 3.0f), w = rand_vec(size);
    std::vector<float> ref(x.size());
    for (int b = 0; b < B; b++) rmsnorm_cpu(ref.data() + (size_t)b * size, x.data() + (size_t)b * size, w.data(), size);
    float *dx = to_gpu(x), *dw = to_gpu(w), *dout = dx;
    if (!in_place) CHECK_CUDA(cudaMalloc((void **)&dout, x.size() * sizeof(float)));
    rmsnorm_gpu(dout, dx, dw, size, B);
    char name[96];
    snprintf(name, sizeof name, "rmsnorm_gpu size=%d B=%d%s", size, B, in_place ? " (in place)" : "");
    check_close(name, from_gpu(dout, x.size()), ref, 1e-5f, 1e-5f);
    cudaFree(dx); cudaFree(dw);
    if (!in_place) cudaFree(dout);
}

static void test_softmax(int size) {
    auto x = rand_vec(size, 10.0f);
    auto ref = x;
    softmax_cpu(ref.data(), size);
    float *dx = to_gpu(x);
    softmax_gpu(dx, size);
    char name[96];
    snprintf(name, sizeof name, "softmax_gpu size=%d", size);
    check_close(name, from_gpu(dx, size), ref, 1e-6f, 1e-4f);  // __expf
    cudaFree(dx);
}

static void test_rope(int dim, int head_size, int pos) {
    auto q = rand_vec(dim), k = rand_vec(dim);
    auto rq = q, rk = k;
    RunState hs;
    hs.q = rq.data(); hs.k = rk.data();
    rotary_embedding_cpu(&hs, pos, dim, dim, head_size);
    RunState ds;
    ds.q = to_gpu(q); ds.k = to_gpu(k);
    rotary_embedding_gpu(&ds, pos, dim, dim, head_size);
    char name[96];
    snprintf(name, sizeof name, "rotary_embedding_gpu dim=%d pos=%d (q)", dim, pos);
    check_close(name, from_gpu(ds.q, dim), rq, 1e-5f, 1e-3f);  // __powf/__cosf/__sinf
    snprintf(name, sizeof name, "rotary_embedding_gpu dim=%d pos=%d (k)", dim, pos);
    check_close(name, from_gpu(ds.k, dim), rk, 1e-5f, 1e-3f);
    cudaFree(ds.q); cudaFree(ds.k);
}

static void test_skip_conn_swiglu(int size, int B) {
    auto a = rand_vec((size_t)B * size, 5.0f), b = rand_vec((size_t)B * size);
    auto ref_add = a, ref_glu = a;
    for (int i = 0; i < B; i++) skip_conn_cpu(ref_add.data() + (size_t)i * size, b.data() + (size_t)i * size, size);
    RunState hs;
    for (int i = 0; i < B; i++) {
        hs.hb = ref_glu.data() + (size_t)i * size; hs.hb2 = b.data() + (size_t)i * size;
        swiglu_cpu(&hs, size);
    }
    float *da = to_gpu(a), *db = to_gpu(b);
    skip_conn_gpu(da, db, size, B);
    char name[96];
    snprintf(name, sizeof name, "skip_conn_gpu size=%d B=%d", size, B);
    check_close(name, from_gpu(da, a.size()), ref_add, 0.0f, 0.0f);  // exact
    CHECK_CUDA(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    RunState ds;
    ds.hb = da; ds.hb2 = db;
    swiglu_gpu(&ds, size, B);
    snprintf(name, sizeof name, "swiglu_gpu size=%d B=%d", size, B);
    check_close(name, from_gpu(da, a.size()), ref_glu, 1e-6f, 1e-5f);
    cudaFree(da); cudaFree(db);
}

// multihead_attention_gpu: one request, plain attention (q and k already rotated)
static void test_attention(int n_heads, int head_size, int seq_len, int pos) {
    int dim = n_heads * head_size, n_layers = 2;
    int loff = 1 * seq_len * dim;
    Config p;
    p.dim = dim; p.n_heads = n_heads; p.n_kv_heads = n_heads; p.seq_len = seq_len; p.n_layers = n_layers;
    auto q = rand_vec(dim), kc = rand_vec((size_t)n_layers * seq_len * dim), vc = rand_vec((size_t)n_layers * seq_len * dim);
    std::vector<float> att((size_t)n_heads * seq_len), ref(dim);
    RunState hs;
    hs.q = q.data(); hs.att = att.data(); hs.xb = ref.data(); hs.key_cache = kc.data(); hs.value_cache = vc.data();
    multihead_attention_cpu(&p, &hs, pos, dim, 1, head_size, loff);
    RunState ds;
    ds.q = to_gpu(q); ds.key_cache = to_gpu(kc); ds.value_cache = to_gpu(vc);
    CHECK_CUDA(cudaMalloc((void **)&ds.xb, dim * sizeof(float)));
    multihead_attention_gpu(&p, &ds, pos, dim, 1, head_size, loff);
    char name[96];
    snprintf(name, sizeof name, "multihead_attention_gpu heads=%d hs=%d pos=%d", n_heads, head_size, pos);
    check_close(name, from_gpu(ds.xb, dim), ref, 1e-5f, 1e-4f);
    cudaFree(ds.q); cudaFree(ds.key_cache); cudaFree(ds.value_cache); cudaFree(ds.xb);
}

// batch_multihead_attention_gpu: RoPE on q and on the cached k at p_pos[b], then attention
// over 0..p_pos[b], for every slot b. The kernel leaves q untouched (rotated copy in shared
// memory) and rotates k inside the cache.
static void test_batch_attention(int n_heads, int head_size, int seq_len, int B) {
    int dim = n_heads * head_size, n_layers = 2;
    int loff = 1 * seq_len * dim;
    size_t slot = (size_t)n_layers * seq_len * dim;
    Config p;
    p.dim = dim; p.n_heads = n_heads; p.n_kv_heads = n_heads; p.seq_len = seq_len; p.n_layers = n_layers;
    auto q = rand_vec((size_t)B * dim), kc = rand_vec(slot * B), vc = rand_vec(slot * B);
    std::vector<int> pos(B);
    for (int b = 0; b < B; b++) pos[b] = b == 0 ? 0 : (b * 113 + 7) % seq_len;  // slot 0 at position 0
    if (B > 1) pos[B - 1] = seq_len - 1;                                       // last slot at the end

    auto ref_k = kc;
    std::vector<float> ref((size_t)B * dim), att((size_t)n_heads * seq_len);
    for (int b = 0; b < B; b++) {
        std::vector<float> qb(q.begin() + (size_t)b * dim, q.begin() + (size_t)(b + 1) * dim);
        RunState hs;
        hs.q = qb.data(); hs.k = ref_k.data() + b * slot + loff + (size_t)pos[b] * dim;
        rotary_embedding_cpu(&hs, pos[b], dim, dim, head_size);
        hs.att = att.data(); hs.xb = ref.data() + (size_t)b * dim;
        hs.key_cache = ref_k.data() + b * slot; hs.value_cache = vc.data() + b * slot;
        multihead_attention_cpu(&p, &hs, pos[b], dim, 1, head_size, loff);
    }
    RunState ds;
    ds.q = to_gpu(q); ds.key_cache = to_gpu(kc); ds.value_cache = to_gpu(vc); ds.p_pos = to_gpu(pos);
    CHECK_CUDA(cudaMalloc((void **)&ds.xb, ref.size() * sizeof(float)));
    batch_multihead_attention_gpu(&p, &ds, dim, 1, head_size, loff, slot, B);
    char name[96];
    snprintf(name, sizeof name, "batch_multihead_attention_gpu hs=%d B=%d (out)", head_size, B);
    check_close(name, from_gpu(ds.xb, ref.size()), ref, 1e-5f, 1e-3f);
    snprintf(name, sizeof name, "batch_multihead_attention_gpu hs=%d B=%d (k cache)", head_size, B);
    check_close(name, from_gpu(ds.key_cache, kc.size()), ref_k, 1e-5f, 1e-3f);
    snprintf(name, sizeof name, "batch_multihead_attention_gpu hs=%d B=%d (q kept)", head_size, B);
    check_close(name, from_gpu(ds.q, q.size()), q, 0.0f, 0.0f);
    cudaFree(ds.q); cudaFree(ds.key_cache); cudaFree(ds.value_cache); cudaFree(ds.p_pos); cudaFree(ds.xb);
}

static void run_kernel_tests(void) {
    printf("== kernel tests (GPU vs CPU reference) ==\n");
    for (int B : BATCHES) {
        test_matmul(768, 768, B);     // stories110M wq/wo
        test_matmul(2048, 768, B);    // stories110M w2
        test_matmul(768, 32000, B);   // classifier
        test_matmul(288, 260, B);     // stories15M dim, odd d
        test_matmul_swiglu(768, 2048, B);
        test_matmul_kv(768, 768, 1024, B);
        test_rmsnorm(768, B, false);
        test_rmsnorm(288, B, true);
        test_skip_conn_swiglu(2048, B);
        test_batch_attention(12, 64, 1024, B);  // stories110M
        test_batch_attention(6, 48, 256, B);    // stories15M
    }
    test_matmul(4096, 4096, 1);
    test_rmsnorm(4096, 1, false);
    for (int s : {1, 7, 1000, 32000}) test_softmax(s);
    for (int pos : {0, 1, 255, 1023}) test_rope(768, 64, pos);
    for (int pos : {0, 1, 100, 1023}) test_attention(12, 64, 1024, pos);
}

// ----------------------------------------------------------------------------
// 2. forward test: run.cu's batched forward() vs a CPU forward per request

typedef struct {
    std::vector<float> x, xb, xb2, hb, hb2, q, att, logits, key_cache, value_cache;
} CpuState;

static float *forward_cpu(Config *p, TransformerWeights *w, CpuState *s, int token, int pos) {
    int dim = p->dim, hidden_dim = p->hidden_dim, head_size = dim / p->n_heads;
    int kv_dim = dim;  // GQA is rejected by run.cu
    float *x = s->x.data();
    memcpy(x, w->token_embedding_table + (size_t)token * dim, dim * sizeof(float));
    for (unsigned long long l = 0; l < (unsigned long long)p->n_layers; l++) {
        rmsnorm_cpu(s->xb.data(), x, w->rms_att_weight + l * dim, dim);
        int loff = l * p->seq_len * kv_dim;
        RunState rs;
        rs.q = s->q.data(); rs.k = s->key_cache.data() + loff + (size_t)pos * kv_dim;
        float *v = s->value_cache.data() + loff + (size_t)pos * kv_dim;
        matmul_cpu(rs.q, s->xb.data(), w->wq + l * dim * dim, dim, dim, 1);
        matmul_cpu(rs.k, s->xb.data(), w->wk + l * dim * kv_dim, dim, kv_dim, 1);
        matmul_cpu(v, s->xb.data(), w->wv + l * dim * kv_dim, dim, kv_dim, 1);
        rotary_embedding_cpu(&rs, pos, dim, kv_dim, head_size);
        rs.att = s->att.data(); rs.xb = s->xb.data();
        rs.key_cache = s->key_cache.data(); rs.value_cache = s->value_cache.data();
        multihead_attention_cpu(p, &rs, pos, kv_dim, 1, head_size, loff);
        matmul_cpu(s->xb2.data(), s->xb.data(), w->wo + l * dim * dim, dim, dim, 1);
        skip_conn_cpu(x, s->xb2.data(), dim);
        rmsnorm_cpu(s->xb.data(), x, w->rms_ffn_weight + l * dim, dim);
        matmul_cpu(s->hb.data(), s->xb.data(), w->w1 + l * dim * hidden_dim, dim, hidden_dim, 1);
        matmul_cpu(s->hb2.data(), s->xb.data(), w->w3 + l * dim * hidden_dim, dim, hidden_dim, 1);
        rs.hb = s->hb.data(); rs.hb2 = s->hb2.data();
        swiglu_cpu(&rs, hidden_dim);
        matmul_cpu(s->xb.data(), s->hb.data(), w->w2 + l * dim * hidden_dim, hidden_dim, dim, 1);
        skip_conn_cpu(x, s->xb.data(), dim);
    }
    rmsnorm_cpu(x, x, w->rms_final_weight, dim);
    matmul_cpu(s->logits.data(), x, w->wcls, p->dim, p->vocab_size, 1);
    return s->logits.data();
}

static void run_forward_test(Transformer *t, const char *model_path, int n_steps) {
    Config *p = &t->config;
    const int B = 3, LATE_SLOT = 1, LATE_START = 5;
    printf("\n== forward test (batched GPU forward vs CPU forward, B=%d, %d steps) ==\n", B, n_steps);
    printf("  model: dim=%d hidden=%d layers=%d heads=%d vocab=%d seq_len=%d\n", p->dim,
           p->hidden_dim, p->n_layers, p->n_heads, p->vocab_size, p->seq_len);

    // host view of the weights (unshared classifier is signalled by a negative vocab size)
    Config header;
    FILE *f = fopen(model_path, "rb");
    if (!f || fread(&header, sizeof(Config), 1, f) != 1) { fprintf(stderr, "cannot read %s\n", model_path); exit(1); }
    fclose(f);
    TransformerWeights hw;
    memory_map_weights(&hw, p, t->data + sizeof(Config) / sizeof(float), header.vocab_size > 0);

    BATCH_SIZE = B;
    CHECK_CUDA(cudaSetDevice(0));
    malloc_run_state(&batch_run_state_d[0], p);
    std::vector<HandlingReq> slots(B);
    std::vector<CpuState> cpu(B);
    Tokenizer tok;
    build_tokenizer(&tok, (char *)"tokenizer.bin", p->vocab_size);
    const char *prompts[B] = {"Once upon a time, there was a little", "The cat sat on the",
                              "Lily and Tom went to the park and"};
    std::vector<std::vector<int>> ptoks(B);
    for (int b = 0; b < B; b++) {
        slots[b].reset();
        batch_map_run_state(&batch_run_state_d[0], &slots[b].run_state, p, b);
        CpuState &s = cpu[b];
        s.x.resize(p->dim); s.xb.resize(p->dim); s.xb2.resize(p->dim); s.q.resize(p->dim);
        s.hb.resize(p->hidden_dim); s.hb2.resize(p->hidden_dim);
        s.att.resize((size_t)p->n_heads * p->seq_len); s.logits.resize(p->vocab_size);
        s.key_cache.assign((size_t)p->n_layers * p->seq_len * p->dim, 0.0f);
        s.value_cache.assign((size_t)p->n_layers * p->seq_len * p->dim, 0.0f);
        ptoks[b].resize(strlen(prompts[b]) + 3);
        int n = 0;
        encode(&tok, (char *)prompts[b], 1, 0, ptoks[b].data(), &n);
        ptoks[b].resize(n);
    }

    double worst_abs = 0;
    int compared = 0, argmax_match = 0, bad = 0;
    for (int step = 0; step < n_steps; step++) {
        for (int b = 0; b < B; b++)  // the late slot joins at LATE_START (continuous batching)
            if (slots[b].is_done && (b != LATE_SLOT || step == LATE_START) && slots[b].pos == 0) {
                slots[b].is_done = false;
                slots[b].token = ptoks[b][0];
            }
        forward(t, slots.data(), &batch_run_state_d[0], &weights_d[0], B);
        for (int b = 0; b < B; b++) {
            HandlingReq &r = slots[b];
            if (r.is_done) continue;
            float *gl = r.run_state.logits;
            float *cl = forward_cpu(p, &hw, &cpu[b], r.token, r.pos);
            double max_abs = 0;
            bool ok = true;
            for (int i = 0; i < p->vocab_size; i++) {
                double d = fabs((double)gl[i] - cl[i]);
                max_abs = fmax(max_abs, d);
                if (!(d <= 1e-3 + 1e-3 * fabs(cl[i]))) ok = false;
            }
            int ca = sample_argmax(cl, p->vocab_size);
            argmax_match += sample_argmax(gl, p->vocab_size) == ca;
            bad += !ok;
            compared++;
            worst_abs = fmax(worst_abs, max_abs);
            r.pos++;
            r.token = r.pos < (int)ptoks[b].size() ? ptoks[b][r.pos] : ca;  // same token on both sides
        }
    }
    printf("  slot positions at the end: %d, %d, %d (slot %d joined at step %d)\n", slots[0].pos,
           slots[1].pos, slots[2].pos, LATE_SLOT, LATE_START);
    printf("  worst max|diff| of the logits = %.2e, argmax equal at %d / %d (slot, step) pairs\n",
           worst_abs, argmax_match, compared);
    bool ok = bad == 0 && argmax_match == compared;
    printf("  [%s] forward logits within atol=1e-3, rtol=1e-3 for every slot and step\n", ok ? "PASS" : "FAIL");
    ok ? g_passed++ : g_failed++;
    free_run_state(&batch_run_state_d[0]);
    free_tokenizer(&tok);
}

// ----------------------------------------------------------------------------
// 3. kernel benchmarks

template <typename F>
static float time_gpu_us(F f, int iters = 200) {
    for (int i = 0; i < 10; i++) f();
    cudaEvent_t a, b;
    CHECK_CUDA(cudaEventCreate(&a));
    CHECK_CUDA(cudaEventCreate(&b));
    CHECK_CUDA(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) f();
    CHECK_CUDA(cudaEventRecord(b));
    CHECK_CUDA(cudaEventSynchronize(b));
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return 1000.0f * ms / iters;
}

static void run_benchmarks(Config *p) {
    int dim = p->dim, hidden = p->hidden_dim, vocab = p->vocab_size, seq_len = p->seq_len;
    int n_heads = p->n_heads, head_size = dim / n_heads;
    const int BMAX = MAX_BATCH_SIZE;
    printf("\n== kernel benchmarks (model shapes: dim=%d hidden=%d vocab=%d seq_len=%d) ==\n", dim, hidden, vocab, seq_len);
    printf("Average over 200 launches (CUDA events). Weight GB/s = weight bytes / time (weights are read once per launch).\n\n");
    printf("| kernel | B=1 us | B=1 weight GB/s | B=%d us | B=%d weight GB/s | us per request, B=%d |\n|---|---|---|---|---|---|\n", BMAX, BMAX, BMAX);

    size_t big = (size_t)vocab * dim > (size_t)hidden * dim ? (size_t)vocab * dim : (size_t)hidden * dim;
    auto w = rand_vec(big), w3 = rand_vec((size_t)hidden * dim);
    auto x = rand_vec((size_t)BMAX * (hidden > dim ? hidden : dim));
    std::vector<float> out((size_t)BMAX * (vocab > hidden ? vocab : hidden));
    float *dw = to_gpu(w), *dw3 = to_gpu(w3), *dx = to_gpu(x), *dout = to_gpu(out);
    auto row = [&](const char *name, double weight_bytes, auto launch) {
        float t1 = time_gpu_us([&] { launch(1); });
        float tb = time_gpu_us([&] { launch(BMAX); });
        printf("| %s | %.1f | %.1f | %.1f | %.1f | %.1f |\n", name, t1, weight_bytes / (t1 * 1e3),
               tb, weight_bytes / (tb * 1e3), tb / BMAX);
    };
    row("matmul_gpu_v2 dim->dim", 4.0 * dim * dim, [&](int B) { matmul_gpu_v2(dout, dx, dw, dim, dim, B); });
    row("matmul_swiglu_gpu dim->hidden (W1+W3)", 8.0 * dim * hidden, [&](int B) { matmul_swiglu_gpu(dout, dx, dw, dw3, dim, hidden, B); });
    row("matmul_skipadd_gpu hidden->dim", 4.0 * dim * hidden, [&](int B) { matmul_skipadd_gpu(dout, dx, dw, hidden, dim, B); });
    row("matmul_gpu_v2 classifier", 4.0 * dim * vocab, [&](int B) { matmul_gpu_v2(dout, dx, dw, dim, vocab, B); });
    {
        size_t slot = (size_t)seq_len * dim;
        auto kc = rand_vec(slot * BMAX);
        float *dkc = to_gpu(kc), *dvc = to_gpu(kc);
        std::vector<int> pos(BMAX, seq_len - 1);
        RunState ds;
        ds.q = dx; ds.xb = dout; ds.key_cache = dkc; ds.value_cache = dvc; ds.p_pos = to_gpu(pos);
        Config c = *p;
        c.n_layers = 1;
        row("batch_multihead_attention_gpu, pos=seq_len-1", 0.0, [&](int B) {
            batch_multihead_attention_gpu(&c, &ds, dim, 1, head_size, 0, slot, B);
        });
        cudaFree(dkc); cudaFree(dvc); cudaFree(ds.p_pos);
    }
    cudaFree(dw); cudaFree(dw3); cudaFree(dx); cudaFree(dout);
}

int main(int argc, char **argv) {
    const char *model = NULL;
    bool bench = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) bench = true;
        else model = argv[i];
    }
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs)\n\n", prop.name, prop.multiProcessorCount);

    if (!bench) run_kernel_tests();
    if (model) {
        NUM_GPUS = 1;
        Transformer t;
        build_transformer(&t, (char *)model);
        if (bench) run_benchmarks(&t.config);
        else run_forward_test(&t, model, 48);
        free_transformer(&t);
    } else {
        printf("\n(no model given: skipping forward test)\n");
    }
    if (!bench) {
        printf("\n%d passed, %d failed\n", g_passed, g_failed);
        return g_failed ? 1 : 0;
    }
    return 0;
}
