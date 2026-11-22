/*
 * Unit tests for the naive CUDA kernels.
 *
 *   ./bin/unittests [model.bin] [--bench]
 *
 * 1. Kernel tests: every GPU kernel against its CPU reference (the code from run.c) on
 *    random inputs, at the real model shapes and at odd sizes that are not multiples of
 *    the block size. A value passes when |gpu - cpu| <= atol + rtol * |cpu|.
 * 2. Forward test (needs a model): the full GPU forward pass against a CPU forward pass
 *    built from the same reference functions, over a prompt plus greedy continuation.
 *    Compares the logits at every position and the argmax token.
 * 3. --bench: average time of every kernel at the shapes of the given model (CUDA events),
 *    plus the single-thread CPU reference time for the same call.
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
static float *to_gpu(const std::vector<float> &v) {
    float *d;
    CHECK_CUDA(cudaMalloc((void **)&d, v.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
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
    printf("  [%s] %-46s max|diff|=%.2e  max|ref|=%.2e", ok ? "PASS" : "FAIL", name, max_abs, max_ref);
    if (!ok) printf("  %zu bad, first i=%zu gpu=%g cpu=%g", bad, first_bad, gpu[first_bad], cpu[first_bad]);
    printf("\n");
    ok ? g_passed++ : g_failed++;
    return ok;
}

// ----------------------------------------------------------------------------
// 1. kernel tests

static void test_matmul(int n, int d) {
    auto x = rand_vec(n), w = rand_vec((size_t)n * d);
    std::vector<float> ref(d);
    matmul_cpu(ref.data(), x.data(), w.data(), n, d);
    float *dx = to_gpu(x), *dw = to_gpu(w), *dout;
    CHECK_CUDA(cudaMalloc((void **)&dout, d * sizeof(float)));
    matmul_gpu_v1(dout, dx, dw, n, d);
    char name[96];
    snprintf(name, sizeof name, "matmul n=%d d=%d", n, d);
    // a float dot product of length n can differ by ~n*eps in the last bits (FMA vs mul+add)
    check_close(name, from_gpu(dout, d), ref, 1e-6f * n + 1e-6f, 1e-5f);
    cudaFree(dx); cudaFree(dw); cudaFree(dout);
}

static void test_rmsnorm(int size, bool in_place) {
    auto x = rand_vec(size, 3.0f), w = rand_vec(size);
    std::vector<float> ref(size);
    rmsnorm_cpu(ref.data(), x.data(), w.data(), size);
    float *dx = to_gpu(x), *dw = to_gpu(w), *dout = dx;
    if (!in_place) CHECK_CUDA(cudaMalloc((void **)&dout, size * sizeof(float)));
    rmsnorm_gpu(dout, dx, dw, size);
    char name[96];
    snprintf(name, sizeof name, "rmsnorm size=%d%s", size, in_place ? " (in place)" : "");
    check_close(name, from_gpu(dout, size), ref, 1e-5f, 1e-5f);
    cudaFree(dx); cudaFree(dw);
    if (!in_place) cudaFree(dout);
}

static void test_softmax(int rows, int size, int stride) {
    auto x = rand_vec((size_t)rows * stride, 10.0f);
    std::vector<float> ref = x;
    for (int r = 0; r < rows; r++) softmax_cpu(ref.data() + (size_t)r * stride, size);
    float *dx = to_gpu(x);
    softmax_gpu(dx, rows, size, stride);
    char name[96];
    snprintf(name, sizeof name, "softmax rows=%d size=%d stride=%d", rows, size, stride);
    check_close(name, from_gpu(dx, x.size()), ref, 1e-6f, 1e-5f);
    cudaFree(dx);
}

static void test_rope(int dim, int kv_dim, int head_size, int pos) {
    auto q = rand_vec(dim), k = rand_vec(kv_dim);
    auto rq = q, rk = k;
    rotary_embedding_cpu(rq.data(), rk.data(), pos, dim, kv_dim, head_size);
    float *dq = to_gpu(q), *dk = to_gpu(k);
    rotary_embedding_gpu(dq, dk, pos, dim, kv_dim, head_size);
    char name[96];
    snprintf(name, sizeof name, "rope dim=%d kv_dim=%d pos=%d (q)", dim, kv_dim, pos);
    check_close(name, from_gpu(dq, dim), rq, 1e-5f, 1e-4f);
    snprintf(name, sizeof name, "rope dim=%d kv_dim=%d pos=%d (k)", dim, kv_dim, pos);
    check_close(name, from_gpu(dk, kv_dim), rk, 1e-5f, 1e-4f);
    cudaFree(dq); cudaFree(dk);
}

// Cache with 2 layers; the test runs on layer 1 so that loff != 0 is exercised.
static void test_attn(int n_heads, int n_kv_heads, int head_size, int seq_len, int pos) {
    int dim = n_heads * head_size, kv_dim = n_kv_heads * head_size, kv_mul = n_heads / n_kv_heads;
    int n_layers = 2;
    long long loff = (long long)1 * seq_len * kv_dim;
    auto q = rand_vec(dim);
    auto kc = rand_vec((size_t)n_layers * seq_len * kv_dim);
    auto vc = rand_vec((size_t)n_layers * seq_len * kv_dim);
    std::vector<float> ref(dim), att((size_t)n_heads * seq_len);
    multihead_attention_cpu(ref.data(), q.data(), att.data(), kc.data(), vc.data(), pos, seq_len, n_heads,
                  head_size, kv_dim, kv_mul, loff);
    float *dq = to_gpu(q), *dkc = to_gpu(kc), *dvc = to_gpu(vc), *datt, *dout;
    CHECK_CUDA(cudaMalloc((void **)&datt, att.size() * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&dout, dim * sizeof(float)));
    multihead_attention_gpu(dout, dq, datt, dkc, dvc, pos, seq_len, n_heads, head_size, kv_dim, kv_mul, loff);
    char name[96];
    snprintf(name, sizeof name, "attention heads=%d kv_heads=%d hs=%d pos=%d", n_heads,
             n_kv_heads, head_size, pos);
    check_close(name, from_gpu(dout, dim), ref, 1e-5f, 1e-4f);
    cudaFree(dq); cudaFree(dkc); cudaFree(dvc); cudaFree(datt); cudaFree(dout);
}

static void test_swiglu(int size) {
    auto a = rand_vec(size, 5.0f), b = rand_vec(size);
    auto ref = a;
    swiglu_cpu(ref.data(), b.data(), size);
    float *da = to_gpu(a), *db = to_gpu(b);
    swiglu_gpu(da, db, size);
    char name[96];
    snprintf(name, sizeof name, "swiglu size=%d", size);
    check_close(name, from_gpu(da, size), ref, 1e-6f, 1e-5f);
    cudaFree(da); cudaFree(db);
}

static void test_skip_conn(int size) {
    auto a = rand_vec(size), b = rand_vec(size);
    auto ref = a;
    skip_conn_cpu(ref.data(), b.data(), size);
    float *da = to_gpu(a), *db = to_gpu(b);
    skip_conn_gpu(da, db, size);
    char name[96];
    snprintf(name, sizeof name, "skip_conn size=%d", size);
    check_close(name, from_gpu(da, size), ref, 0.0f, 0.0f);  // exact
    cudaFree(da); cudaFree(db);
}

static void run_kernel_tests(void) {
    printf("== kernel tests (GPU vs CPU reference) ==\n");
    int mm[][2] = {{1, 1}, {13, 7}, {257, 1000}, {768, 768}, {768, 2048}, {2048, 768},
                   {768, 32000}, {4096, 4096}};
    for (auto &s : mm) test_matmul(s[0], s[1]);
    for (int s : {1, 7, 257, 768, 4096}) test_rmsnorm(s, false);
    test_rmsnorm(768, true);
    test_softmax(1, 1, 1);
    test_softmax(1, 7, 7);
    test_softmax(1, 1000, 1000);
    test_softmax(1, 32000, 32000);
    test_softmax(12, 300, 1024);  // the attention case: rows with a stride
    for (int pos : {0, 1, 255, 1023}) test_rope(768, 768, 64, pos);
    test_rope(768, 256, 64, 77);  // grouped-query: only part of q has a matching k
    for (int pos : {0, 1, 100, 1023}) test_attn(12, 12, 64, 1024, pos);
    test_attn(12, 4, 64, 1024, 500);  // grouped-query attention, kv_mul = 3
    test_attn(6, 6, 48, 256, 255);    // stories15M shape
    for (int s : {1, 257, 2048, 11008}) test_swiglu(s);
    for (int s : {1, 257, 768, 4096}) test_skip_conn(s);
}

// ----------------------------------------------------------------------------
// 2. forward test: GPU forward() vs a CPU forward built from the reference functions

typedef struct {
    std::vector<float> x, xb, xb2, hb, hb2, q, att, logits, key_cache, value_cache;
} CpuState;

static float *forward_cpu(Config *p, TransformerWeights *w, CpuState *s, int token, int pos) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;
    float *x = s->x.data();
    memcpy(x, w->token_embedding_table + (size_t)token * dim, dim * sizeof(float));
    for (unsigned long long l = 0; l < (unsigned long long)p->n_layers; l++) {
        rmsnorm_cpu(s->xb.data(), x, w->rms_att_weight + l * dim, dim);
        long long loff = (long long)l * p->seq_len * kv_dim;
        float *k = s->key_cache.data() + loff + (long long)pos * kv_dim;
        float *v = s->value_cache.data() + loff + (long long)pos * kv_dim;
        matmul_cpu(s->q.data(), s->xb.data(), w->wq + l * dim * dim, dim, dim);
        matmul_cpu(k, s->xb.data(), w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul_cpu(v, s->xb.data(), w->wv + l * dim * kv_dim, dim, kv_dim);
        rotary_embedding_cpu(s->q.data(), k, pos, dim, kv_dim, head_size);
        multihead_attention_cpu(s->xb.data(), s->q.data(), s->att.data(), s->key_cache.data(),
                      s->value_cache.data(), pos, p->seq_len, p->n_heads, head_size, kv_dim,
                      kv_mul, loff);
        matmul_cpu(s->xb2.data(), s->xb.data(), w->wo + l * dim * dim, dim, dim);
        skip_conn_cpu(x, s->xb2.data(), dim);
        rmsnorm_cpu(s->xb.data(), x, w->rms_ffn_weight + l * dim, dim);
        matmul_cpu(s->hb.data(), s->xb.data(), w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul_cpu(s->hb2.data(), s->xb.data(), w->w3 + l * dim * hidden_dim, dim, hidden_dim);
        swiglu_cpu(s->hb.data(), s->hb2.data(), hidden_dim);
        matmul_cpu(s->xb.data(), s->hb.data(), w->w2 + l * dim * hidden_dim, hidden_dim, dim);
        skip_conn_cpu(x, s->xb.data(), dim);
    }
    rmsnorm_cpu(x, x, w->rms_final_weight, dim);
    matmul_cpu(s->logits.data(), x, w->wcls, p->dim, p->vocab_size);
    return s->logits.data();
}

static void run_forward_test(Transformer *t, const char *tokenizer_path, int n_positions) {
    Config *p = &t->config;
    printf("\n== forward test (GPU forward vs CPU forward, %d positions) ==\n", n_positions);
    printf("  model: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d\n",
           p->dim, p->hidden_dim, p->n_layers, p->n_heads, p->n_kv_heads, p->vocab_size, p->seq_len);

    TransformerWeights hw;  // host view of the same weights (mmap)
    memory_map_weights(&hw, p, t->data + sizeof(Config) / sizeof(float), t->shared_weights);
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    CpuState s;
    s.x.resize(p->dim); s.xb.resize(p->dim); s.xb2.resize(p->dim); s.q.resize(p->dim);
    s.hb.resize(p->hidden_dim); s.hb2.resize(p->hidden_dim);
    s.att.resize((size_t)p->n_heads * p->seq_len);
    s.logits.resize(p->vocab_size);
    s.key_cache.assign((size_t)p->n_layers * p->seq_len * kv_dim, 0.0f);
    s.value_cache.assign((size_t)p->n_layers * p->seq_len * kv_dim, 0.0f);

    Tokenizer tok;
    build_tokenizer(&tok, tokenizer_path, p->vocab_size);
    char prompt[] = "Once upon a time, there was a little";
    std::vector<int> tokens(strlen(prompt) + 3);
    int n_prompt = 0;
    encode(&tok, prompt, 1, 0, tokens.data(), &n_prompt);

    // Both sides are fed the same token at every position: prompt tokens first, then the
    // CPU's argmax, so a small numerical difference cannot make the sequences diverge.
    double worst_abs = 0, worst_rel = 0;
    int argmax_match = 0, bad_positions = 0;
    int token = tokens[0];
    for (int pos = 0; pos < n_positions && pos < p->seq_len; pos++) {
        float *gl = forward(t, token, pos);
        float *cl = forward_cpu(p, &hw, &s, token, pos);
        double max_abs = 0, max_ref = 0;
        bool bad = false;
        for (int i = 0; i < p->vocab_size; i++) {
            double d = fabs((double)gl[i] - cl[i]);
            max_abs = fmax(max_abs, d);
            max_ref = fmax(max_ref, fabs(cl[i]));
            if (!(d <= 1e-3 + 1e-3 * fabs(cl[i]))) bad = true;
        }
        int ga = sample_argmax(gl, p->vocab_size), ca = sample_argmax(cl, p->vocab_size);
        argmax_match += ga == ca;
        bad_positions += bad;
        worst_abs = fmax(worst_abs, max_abs);
        worst_rel = fmax(worst_rel, max_abs / (max_ref + 1e-12));
        token = pos + 1 < n_prompt ? tokens[pos + 1] : ca;
    }
    printf("  worst max|diff| over positions = %.2e (relative to max|logit|: %.2e)\n", worst_abs, worst_rel);
    printf("  argmax token equal at %d / %d positions\n", argmax_match, n_positions);
    bool ok = bad_positions == 0 && argmax_match == n_positions;
    printf("  [%s] forward logits within atol=1e-3, rtol=1e-3 at every position%s\n",
           ok ? "PASS" : "FAIL", bad_positions ? " (some positions out of tolerance)" : "");
    ok ? g_passed++ : g_failed++;
    free_tokenizer(&tok);
}

// ----------------------------------------------------------------------------
// 3. kernel benchmarks at the model's shapes

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

template <typename F>
static float time_cpu_us(F f, int iters) {
    f();
    double t0 = harness_now();
    for (int i = 0; i < iters; i++) f();
    return (float)(1e6 * (harness_now() - t0) / iters);
}

static void bench_row(const char *name, float gpu_us, float cpu_us, double bytes) {
    printf("| %s | %.1f | %.1f | %.1fx | %.1f |\n", name, gpu_us, cpu_us, cpu_us / gpu_us,
           bytes / (gpu_us * 1e-6) / 1e9);
}

static void run_benchmarks(Config *p) {
    int dim = p->dim, hidden = p->hidden_dim, vocab = p->vocab_size, seq_len = p->seq_len;
    int n_heads = p->n_heads, head_size = dim / n_heads;
    int kv_dim = (dim * p->n_kv_heads) / n_heads, kv_mul = n_heads / p->n_kv_heads;
    printf("\n== kernel benchmarks (model shapes: dim=%d hidden=%d vocab=%d seq_len=%d) ==\n",
           dim, hidden, vocab, seq_len);
    printf("GPU: average over 200 launches (CUDA events). CPU: single-thread reference from run.c.\n");
    printf("GB/s = bytes the kernel must read+write at least / GPU time.\n\n");
    printf("| kernel | GPU us | CPU us | CPU/GPU | GPU GB/s |\n|---|---|---|---|---|\n");

    size_t big = (size_t)vocab * dim;
    if ((size_t)hidden * dim > big) big = (size_t)hidden * dim;
    auto w = rand_vec(big), x = rand_vec(hidden > dim ? hidden : dim), y = rand_vec(hidden);
    std::vector<float> out(vocab > hidden ? vocab : hidden);
    float *dw = to_gpu(w), *dx = to_gpu(x), *dy = to_gpu(y), *dout = to_gpu(out);

    struct { const char *name; int n, d; } mms[] = {
        {"matmul wq/wo (dim->dim)", dim, dim},
        {"matmul wk/wv (dim->kv_dim)", dim, kv_dim},
        {"matmul w1/w3 (dim->hidden)", dim, hidden},
        {"matmul w2 (hidden->dim)", hidden, dim},
        {"matmul classifier (dim->vocab)", dim, vocab}};
    for (auto &m : mms) {
        float g = time_gpu_us([&] { matmul_gpu_v1(dout, dx, dw, m.n, m.d); });
        float c = time_cpu_us([&] { matmul_cpu(out.data(), x.data(), w.data(), m.n, m.d); }, 5);
        bench_row(m.name, g, c, 4.0 * ((double)m.n * m.d + m.n + m.d));
    }
    {
        float g = time_gpu_us([&] { rmsnorm_gpu(dout, dx, dy, dim); });
        float c = time_cpu_us([&] { rmsnorm_cpu(out.data(), x.data(), y.data(), dim); }, 1000);
        bench_row("rmsnorm (dim)", g, c, 4.0 * 3 * dim);
    }
    {
        float g = time_gpu_us([&] { rotary_embedding_gpu(dx, dy, seq_len - 1, dim, kv_dim, head_size); });
        float c = time_cpu_us([&] { rotary_embedding_cpu(x.data(), y.data(), seq_len - 1, dim, kv_dim, head_size); }, 1000);
        bench_row("rope", g, c, 4.0 * 2 * (dim + kv_dim));
    }
    {
        auto kc = rand_vec((size_t)seq_len * kv_dim), vc = rand_vec((size_t)seq_len * kv_dim);
        std::vector<float> att((size_t)n_heads * seq_len), xb(dim);
        float *dkc = to_gpu(kc), *dvc = to_gpu(vc), *datt = to_gpu(att);
        std::vector<int> positions;
        for (int pos : {63, 255, seq_len - 1})
            if (pos < seq_len && (positions.empty() || pos != positions.back())) positions.push_back(pos);
        for (int pos : positions) {
            float g = time_gpu_us([&] {
                multihead_attention_gpu(dout, dx, datt, dkc, dvc, pos, seq_len, n_heads, head_size, kv_dim, kv_mul, 0);
            });
            float c = time_cpu_us([&] {
                multihead_attention_cpu(xb.data(), x.data(), att.data(), kc.data(), vc.data(), pos, seq_len,
                              n_heads, head_size, kv_dim, kv_mul, 0);
            }, 20);
            char name[64];
            snprintf(name, sizeof name, "attention (pos=%d)", pos);
            bench_row(name, g, c, 4.0 * 2 * (double)(pos + 1) * kv_dim);
        }
        cudaFree(dkc); cudaFree(dvc); cudaFree(datt);
    }
    {
        float g = time_gpu_us([&] { softmax_gpu(dout, 1, vocab, vocab); });
        float c = time_cpu_us([&] { softmax_cpu(out.data(), vocab); }, 100);
        bench_row("softmax (vocab)", g, c, 4.0 * 2 * vocab);
    }
    {
        float g = time_gpu_us([&] { swiglu_gpu(dout, dy, hidden); });
        float c = time_cpu_us([&] { swiglu_cpu(out.data(), y.data(), hidden); }, 1000);
        bench_row("swiglu (hidden)", g, c, 4.0 * 3 * hidden);
    }
    {
        float g = time_gpu_us([&] { skip_conn_gpu(dout, dy, dim); });
        float c = time_cpu_us([&] { skip_conn_cpu(out.data(), y.data(), dim); }, 1000);
        bench_row("residual add (dim)", g, c, 4.0 * 3 * dim);
    }
    cudaFree(dw); cudaFree(dx); cudaFree(dy); cudaFree(dout);
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
        Transformer t;
        build_transformer(&t, model);
        if (bench) {
            run_benchmarks(&t.config);
        } else {
            run_forward_test(&t, "tokenizer.bin", 48);
        }
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
