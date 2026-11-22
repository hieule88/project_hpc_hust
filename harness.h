/*
 * Test-mode harness shared by every implementation (CPU runtest.c, GPU run.cu, later optimized versions).
 *
 * It runs the same request loop for each implementation, so throughput and outputs are
 * directly comparable:
 *   - input file : first line = number of requests, then one prompt per line
 *   - output file: first line = number of requests, then one generated text per line
 *                  (newlines inside the text are escaped as "\n")
 *   - stdout     : one machine-readable "RESULT ..." line (parsed by bench/bench.sh)
 *
 * Requests are processed one after another (no batching). Each request gets a fresh
 * sampler with the same seed, so a run is deterministic.
 *
 * Include this file AFTER the implementation defines: Transformer, Tokenizer, Sampler,
 * forward(), encode(), decode(), build_sampler(), free_sampler(), sample().
 * The file is valid C and C++ (gcc compiles it into runtest.c, nvcc into run.cu).
 */
#ifndef HARNESS_H
#define HARNESS_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int num_reqs;
    char **prompts;
    char **outputs;
} Requests;

typedef struct {
    char *s;
    size_t len, cap;
} StrBuf;

static double harness_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void sb_putc(StrBuf *b, char c) {
    if (b->len + 2 > b->cap) {
        b->cap = b->cap ? b->cap * 2 : 256;
        b->s = (char *)realloc(b->s, b->cap);
    }
    b->s[b->len++] = c;
    b->s[b->len] = '\0';
}

// Same filtering as safe_printf() in run.c, plus newline escaping so one request = one line.
static void sb_append_piece(StrBuf *b, const char *piece) {
    if (piece == NULL || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) return;
    }
    for (const char *c = piece; *c; c++) {
        if (*c == '\n') { sb_putc(b, '\\'); sb_putc(b, 'n'); }
        else if (*c != '\r') sb_putc(b, *c);
    }
}

static void read_requests(const char *path, Requests *r) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open input file: %s\n", path); exit(EXIT_FAILURE); }
    char line[8192];
    if (!fgets(line, sizeof line, f)) { fprintf(stderr, "empty input file: %s\n", path); exit(EXIT_FAILURE); }
    r->num_reqs = atoi(line);
    if (r->num_reqs <= 0) { fprintf(stderr, "bad request count in %s\n", path); exit(EXIT_FAILURE); }
    r->prompts = (char **)calloc(r->num_reqs, sizeof(char *));
    r->outputs = (char **)calloc(r->num_reqs, sizeof(char *));
    for (int i = 0; i < r->num_reqs; i++) {
        if (!fgets(line, sizeof line, f)) {
            fprintf(stderr, "%s declares %d requests but has only %d\n", path, r->num_reqs, i);
            exit(EXIT_FAILURE);
        }
        line[strcspn(line, "\r\n")] = '\0';
        r->prompts[i] = strdup(line);
    }
    fclose(f);
}

static void write_outputs(const char *path, const Requests *r) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "cannot write output file: %s\n", path); exit(EXIT_FAILURE); }
    fprintf(f, "%d\n", r->num_reqs);
    for (int i = 0; i < r->num_reqs; i++) fprintf(f, "%s\n", r->outputs[i]);
    fclose(f);
}

static void free_requests(Requests *r) {
    for (int i = 0; i < r->num_reqs; i++) { free(r->prompts[i]); free(r->outputs[i]); }
    free(r->prompts);
    free(r->outputs);
}

/*
 * Runs every request in `input_path` and reports throughput.
 *   fwd_tokens: number of forward() calls (prompt tokens + generated tokens) = work done
 *   gen_tokens: number of sampled tokens
 * A request stops at BOS(1)/EOS(2) or after `steps` positions.
 * Timing covers only the request loop (model loading is excluded).
 */
static void run_test(Transformer *transformer, Tokenizer *tokenizer, const char *impl,
                     const char *model_path, const char *input_path, const char *output_path,
                     int steps, float temperature, float topp, unsigned long long seed) {
    Requests reqs;
    read_requests(input_path, &reqs);
    long fwd_tokens = 0, gen_tokens = 0;

    double t0 = harness_now();
    for (int r = 0; r < reqs.num_reqs; r++) {
        Sampler sampler;
        build_sampler(&sampler, transformer->config.vocab_size, temperature, topp, seed);

        char *prompt = reqs.prompts[r];
        int *prompt_tokens = (int *)malloc((strlen(prompt) + 3) * sizeof(int));
        int num_prompt_tokens = 0;
        encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);

        StrBuf out = {NULL, 0, 0};
        int token = prompt_tokens[0], next, pos = 0;
        while (pos < steps) {
            float *logits = forward(transformer, token, pos);
            if (pos < num_prompt_tokens - 1) {
                next = prompt_tokens[pos + 1];
            } else {
                next = sample(&sampler, logits);
                gen_tokens++;
            }
            pos++;
            if (next == 1 || next == 2) break;
            sb_append_piece(&out, decode(tokenizer, token, next));
            token = next;
        }
        fwd_tokens += pos;
        reqs.outputs[r] = out.s ? out.s : strdup("");
        free(prompt_tokens);
        free_sampler(&sampler);
    }
    double elapsed = harness_now() - t0;

    write_outputs(output_path, &reqs);
    const char *model_name = strrchr(model_path, '/');
    model_name = model_name ? model_name + 1 : model_path;
    printf("RESULT impl=%s model=%s reqs=%d steps=%d fwd_tokens=%ld gen_tokens=%ld "
           "elapsed_s=%.3f fwd_tok_s=%.2f gen_tok_s=%.2f\n",
           impl, model_name, reqs.num_reqs, steps, fwd_tokens, gen_tokens, elapsed,
           fwd_tokens / elapsed, gen_tokens / elapsed);
    free_requests(&reqs);
}

#endif  // HARNESS_H
