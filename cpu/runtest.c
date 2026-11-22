/*
 * Test-mode entry point for the CPU reference.
 * run.c is reused unchanged: TESTING hides its main(), and common/harness.h adds the
 * shared request loop, so the CPU and GPU versions run exactly the same workload.
 *
 * Usage: runtest <checkpoint> -f <input> -o <output> [-n steps] [-t temp] [-p topp] [-s seed] [-z tokenizer]
 */
#define TESTING
#include "run.c"
#include "../common/harness.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static void usage(void) {
    fprintf(stderr, "Usage: runtest <checkpoint> -f <input> -o <output> [options]\n");
    fprintf(stderr, "  -n <int>    max positions per request, default 256 (0 = seq_len)\n");
    fprintf(stderr, "  -t <float>  temperature, default 1.0 (0 = greedy)\n");
    fprintf(stderr, "  -p <float>  top-p, default 0.9\n");
    fprintf(stderr, "  -s <int>    sampler seed (same for every request), default 314028\n");
    fprintf(stderr, "  -z <string> tokenizer path, default tokenizer.bin\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    char *checkpoint_path = NULL, *input_path = NULL, *output_path = NULL;
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f, topp = 0.9f;
    int steps = 256;
    unsigned long long seed = 314028;

    if (argc < 2) usage();
    checkpoint_path = argv[1];
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc || argv[i][0] != '-' || strlen(argv[i]) != 2) usage();
        switch (argv[i][1]) {
            case 'f': input_path = argv[i + 1]; break;
            case 'o': output_path = argv[i + 1]; break;
            case 'n': steps = atoi(argv[i + 1]); break;
            case 't': temperature = atof(argv[i + 1]); break;
            case 'p': topp = atof(argv[i + 1]); break;
            case 's': seed = strtoull(argv[i + 1], NULL, 10); break;
            case 'z': tokenizer_path = argv[i + 1]; break;
            default: usage();
        }
    }
    if (!input_path || !output_path) usage();

    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps <= 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len;
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    char impl[64];
#ifdef _OPENMP
    snprintf(impl, sizeof impl, "cpu_%dt", omp_get_max_threads());
#else
    snprintf(impl, sizeof impl, "cpu_1t");
#endif
    run_test(&transformer, &tokenizer, impl, checkpoint_path, input_path, output_path,
             steps, temperature, topp, seed);

    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
