#include <hip/hip_runtime.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include <cstdio>
#include <cstdlib>

#include "kernels/attn.h"
#include "kernels/matmul.h"
#include "kernels/rmsnorm.h"
#include "kernels/rope.h"
#include "kernels/skip_conn.h"
#include "kernels/softmax.h"
#include "kernels/swiglu.h"
#include "kernels/types.h"

const int MAX_DIM = 10000;
const int MIN_DIM = 1000;

#define CHECK_HIP(call)                                                       \
    do {                                                                      \
        hipError_t status_ = call;                                            \
        if (status_ != hipSuccess) {                                          \
            fprintf(stderr, "HIP error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
                    hipGetErrorName(status_), hipGetErrorString(status_));    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

float *alloc_mat(int R, int C) {
    float *m;
    CHECK_HIP(hipHostMalloc(&m, sizeof(float) * R * C, hipHostMallocNonCoherent));
    return m;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void rand_mat(float *m, int R, int C) {
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            m[i * C + j] = (float)rand() / (float)RAND_MAX - 0.5;
        }
    }
}

void rand_vec(float *v, int size) {
    for (int i = 0; i < size; i++) {
        v[i] = (float)rand() / (float)RAND_MAX - 0.5;
    }
}

int rand_int(int maxI, int minI) {
    return rand() % (maxI - minI + 1) + minI;
}

void print_mat(float *m, int R, int C) {
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            printf("%+.3f ", m[i * C + j]);
        }
        printf("\n");
    }
}

void vec_diff_check(float *original, float *changed, int size, bool verbose) {
    float max_diff = 0.0F;
    double total_diff = 0.0F;
    if (verbose) printf("size: %d\n", size);
    for (int i = 0; i < size; ++i) {
        if (i > size - 10 && verbose)
            printf("%d CPU: %f, GPU: %f\n", i, original[i], changed[i]);
        float diff = fabsf(original[i] - changed[i]);
        max_diff = fmaxf(max_diff, diff);
        total_diff += diff;
    }
    printf("Comparing CPU and GPU: Max diff %f, Avg diff: %f\n", max_diff, total_diff / size);
}

void test_matmul(bool verbose) {
    int n, d;
    n = 4096;
    d = 11008;
    // n = 4096;
    // d = 4096;
    // int n = 11008;
    // int d = 4096;
    int batch_size = 8;
    printf("\n\nMatmul validating with n = %d, d = %d, batch_size = %d...\n", n, d, batch_size);

    float *A = alloc_mat(d, n);
    float *B = alloc_mat(batch_size, n);
    float *C_cpu = alloc_mat(batch_size, d);
    float *C_gpu = alloc_mat(batch_size, d);
    rand_mat(A, d, n);
    rand_mat(B, batch_size, n);

    matmul_gpu_test(C_gpu, B, A, n, d, batch_size);

    matmul_cpu(C_cpu, B, A, n, d, batch_size);

    vec_diff_check(C_cpu, C_gpu, d * batch_size, verbose);
    // print_mat(A, d, n);
    // print_mat(B, batch_size, n);
    // print_mat(C_gpu, batch_size, d);
    // printf("\n");
    // print_mat(C_cpu, batch_size, d);
}

void test_rsmnorm(bool verbose) {
    printf("\n\nRmsnorm validating...\n");

    int size = rand_int(MAX_DIM, MIN_DIM);
    float *o, *o_d;
    float *x, *x_d;
    float *weight, *weight_d;

    CHECK_HIP(hipHostMalloc(&o, size * sizeof(float)));
    CHECK_HIP(hipHostMalloc(&o_d, size * sizeof(float)));
    CHECK_HIP(hipHostMalloc(&x, size * sizeof(float)));
    CHECK_HIP(hipHostMalloc(&x_d, size * sizeof(float)));
    CHECK_HIP(hipHostMalloc(&weight, size * sizeof(float)));
    CHECK_HIP(hipHostMalloc(&weight_d, size * sizeof(float)));

    for (int i = 0; i < size; i++) {
        x[i] = ((float)rand() / (float)RAND_MAX);
        weight[i] = ((float)rand() / (float)RAND_MAX);
        x_d[i] = x[i];
        weight_d[i] = weight[i];
    }

    rmsnorm_cpu(o, x, weight, size);
    rmsnorm_gpu_test(o_d, x_d, weight_d, size);
    vec_diff_check(o, o_d, size, verbose);
}

void test_softmax(bool verbose) {
    printf("\n\nSoftmax validating...\n");
    int size = rand_int(MAX_DIM, MIN_DIM);
    float *x, *x_d;

    CHECK_HIP(hipHostMalloc(&x, size * sizeof(float)));
    CHECK_HIP(hipHostMalloc(&x_d, size * sizeof(float)));

    for (int i = 0; i < size; i++) {
        x[i] = ((float)rand() / (float)RAND_MAX);
        x_d[i] = x[i];
    }

    softmax_cpu(x, size);
    softmax_gpu_test(x_d, size);
    vec_diff_check(x, x_d, size, verbose);
}

void test_swiglu(bool verbose) {
    printf("\n\nSwiGLU validating...\n");
    int hidden_dim = rand_int(MAX_DIM, MIN_DIM);
    RunState *s = new RunState, *s_d = new RunState;
    CHECK_HIP(hipHostMalloc(&s->hb, sizeof(float) * hidden_dim));
    CHECK_HIP(hipHostMalloc(&s->hb2, sizeof(float) * hidden_dim));

    CHECK_HIP(hipHostMalloc(&s_d->hb, sizeof(float) * hidden_dim));
    CHECK_HIP(hipHostMalloc(&s_d->hb2, sizeof(float) * hidden_dim));

    rand_vec(s->hb, hidden_dim);
    rand_vec(s->hb2, hidden_dim);

    CHECK_HIP(hipMemcpy(s_d->hb, s->hb, sizeof(float) * hidden_dim, hipMemcpyHostToHost));
    CHECK_HIP(hipMemcpy(s_d->hb2, s->hb2, sizeof(float) * hidden_dim, hipMemcpyHostToHost));

    swiglu_cpu(s, hidden_dim);
    swiglu_gpu_test(s_d, hidden_dim);

    vec_diff_check(s->hb, s_d->hb, hidden_dim, verbose);
    vec_diff_check(s->hb2, s_d->hb2, hidden_dim, verbose);

    CHECK_HIP(hipHostFree(s->hb));
    CHECK_HIP(hipHostFree(s->hb2));
    CHECK_HIP(hipHostFree(s_d->hb));
    CHECK_HIP(hipHostFree(s_d->hb2));

    delete s;
    delete s_d;
}

void test_rope(bool verbose) {
    printf("\n\nRoPE validating...\n");

    int seq_len = 1024;
    int dim = 4096;
    int n_kv_heads = 32;
    int n_heads = 32;
    int kv_dim = (dim * n_kv_heads) / n_heads;
    int head_size = dim / n_heads;

    int pos = rand_int(seq_len, 0);

    RunState *s = new RunState, *s_d = new RunState;
    CHECK_HIP(hipHostMalloc(&s->q, sizeof(float) * dim));
    CHECK_HIP(hipHostMalloc(&s->k, sizeof(float) * kv_dim));

    CHECK_HIP(hipHostMalloc(&s_d->q, sizeof(float) * dim));
    CHECK_HIP(hipHostMalloc(&s_d->k, sizeof(float) * kv_dim));

    rand_vec(s->q, dim);
    rand_vec(s->k, kv_dim);

    CHECK_HIP(hipMemcpy(s_d->q, s->q, sizeof(float) * dim, hipMemcpyHostToHost));
    CHECK_HIP(hipMemcpy(s_d->k, s->k, sizeof(float) * kv_dim, hipMemcpyHostToHost));

    rotary_embedding_cpu(s, pos, dim, kv_dim, head_size);
    double s_time, elapsed;
    s_time = get_time();
    rotary_embedding_gpu_test(s_d, pos, dim, kv_dim, head_size);
    elapsed = get_time() - s_time;
    printf("%f sec\n", elapsed);

    vec_diff_check(s->q, s_d->q, dim, verbose);
    vec_diff_check(s->k, s_d->k, kv_dim, verbose);

    CHECK_HIP(hipHostFree(s->q));
    CHECK_HIP(hipHostFree(s->k));
    CHECK_HIP(hipHostFree(s_d->q));
    CHECK_HIP(hipHostFree(s_d->k));

    delete s;
    delete s_d;
}

void test_skip_conn(bool verbose) {
    printf("\n\nSkip connection validating...\n");

    int size = rand_int(MAX_DIM, MIN_DIM);
    float *x, *y;
    float *x_d, *y_d;

    CHECK_HIP(hipHostMalloc(&x, sizeof(float) * size));
    CHECK_HIP(hipHostMalloc(&y, sizeof(float) * size));

    CHECK_HIP(hipHostMalloc(&x_d, sizeof(float) * size));
    CHECK_HIP(hipHostMalloc(&y_d, sizeof(float) * size));

    rand_vec(x, size);
    rand_vec(y, size);

    CHECK_HIP(hipMemcpy(x_d, x, sizeof(float) * size, hipMemcpyHostToHost));
    CHECK_HIP(hipMemcpy(y_d, y, sizeof(float) * size, hipMemcpyHostToHost));

    skip_conn_cpu(x, y, size);
    skip_conn_gpu_test(x_d, y_d, size);

    vec_diff_check(x, x_d, size, verbose);
    vec_diff_check(y, y_d, size, verbose);

    CHECK_HIP(hipHostFree(x));
    CHECK_HIP(hipHostFree(y));
    CHECK_HIP(hipHostFree(x_d));
    CHECK_HIP(hipHostFree(y_d));
}

void test_attn(bool verbose) {
    printf("\n\nMultihead attention validating...\n");

    Config *p = new Config;
    p->dim = 768;
    p->hidden_dim = 2048;
    p->n_layers = 12;
    p->n_heads = 12;
    p->n_kv_heads = 12;
    p->seq_len = 1024;
    p->vocab_size = 32000;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int l = rand_int(p->n_layers, 0);
    int pos = rand_int(p->seq_len, 0);

    RunState *s = new RunState, *s_d = new RunState;
    CHECK_HIP(hipHostMalloc(&s->q, sizeof(float) * p->dim));
    CHECK_HIP(hipHostMalloc(&s->att, sizeof(float) * p->n_heads * p->seq_len));
    CHECK_HIP(hipHostMalloc(&s->xb, sizeof(float) * p->dim));
    CHECK_HIP(hipHostMalloc(&s->key_cache, sizeof(float) * p->n_layers * p->seq_len * kv_dim));
    CHECK_HIP(hipHostMalloc(&s->value_cache, sizeof(float) * p->n_layers * p->seq_len * kv_dim));

    CHECK_HIP(hipHostMalloc(&s_d->q, sizeof(float) * p->dim));
    CHECK_HIP(hipHostMalloc(&s_d->att, sizeof(float) * p->n_heads * p->seq_len));
    CHECK_HIP(hipHostMalloc(&s_d->xb, sizeof(float) * p->dim));
    CHECK_HIP(hipHostMalloc(&s_d->key_cache, sizeof(float) * p->n_layers * p->seq_len * kv_dim));
    CHECK_HIP(hipHostMalloc(&s_d->value_cache, sizeof(float) * p->n_layers * p->seq_len * kv_dim));

    int kv_mul = p->n_heads / p->n_kv_heads;  // integer multiplier of the kv sharing in multiquery
    int hidden_dim = p->hidden_dim;
    int head_size = p->dim / p->n_heads;
    int loff = l * p->seq_len * kv_dim;

    rand_vec(s->q, p->dim);
    rand_vec(s->att, p->n_heads * p->seq_len);
    rand_vec(s->xb, p->dim);
    rand_vec(s->key_cache, p->n_layers * p->seq_len * kv_dim);
    rand_vec(s->value_cache, p->n_layers * p->seq_len * kv_dim);

    CHECK_HIP(hipMemcpy(s_d->q, s->q, sizeof(float) * p->dim, hipMemcpyHostToHost));
    CHECK_HIP(hipMemcpy(s_d->att, s->att, sizeof(float) * p->n_heads * p->seq_len, hipMemcpyHostToHost));
    CHECK_HIP(hipMemcpy(s_d->xb, s->xb, sizeof(float) * p->dim, hipMemcpyHostToHost));
    CHECK_HIP(hipMemcpy(s_d->key_cache, s->key_cache, sizeof(float) * p->n_layers * p->seq_len * kv_dim, hipMemcpyHostToHost));
    CHECK_HIP(hipMemcpy(s_d->value_cache, s->value_cache, sizeof(float) * p->n_layers * p->seq_len * kv_dim, hipMemcpyHostToHost));

    multihead_attention_cpu(p, s, pos, kv_dim, kv_mul, head_size, loff);
    multihead_attention_gpu_test(p, s_d, pos, kv_dim, kv_mul, head_size, loff, 1);

    vec_diff_check(s->xb, s_d->xb, p->dim, verbose);

    CHECK_HIP(hipHostFree(s->q));
    CHECK_HIP(hipHostFree(s->att));
    CHECK_HIP(hipHostFree(s->xb));
    CHECK_HIP(hipHostFree(s->key_cache));
    CHECK_HIP(hipHostFree(s->value_cache));

    CHECK_HIP(hipHostFree(s_d->q));
    CHECK_HIP(hipHostFree(s_d->att));
    CHECK_HIP(hipHostFree(s_d->xb));
    CHECK_HIP(hipHostFree(s_d->key_cache));
    CHECK_HIP(hipHostFree(s_d->value_cache));
    delete p;
    delete s;
    delete s_d;
}

int main(int argc, char **argv) {
    // Seed the random number generator with the current time
    bool verbose = atoi(argv[0]);
    srand(time(NULL));
    test_matmul(verbose);
    // test_matmul_2(verbose);
    // test_rope(verbose);
    // test_skip_conn(verbose);
    // test_rsmnorm(verbose);
    // test_softmax(verbose);
    // test_attn(verbose);
    // test_swiglu(verbose);
}
