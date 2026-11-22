# Benchmark on vmo-B760M-DS3H-DDR4

- Date: 2026-11-22
- CPU: Intel(R) Core(TM) i5-14400, cpu_omp uses 6 threads
- GPU: NVIDIA GeForce RTX 5060 Ti, 595.99.02
- Workload: `input/12.txt`, up to 256 positions per request, temperature 1.0, top-p 0.9, seed 314028
- tok/s = forward passes (prompt + generated tokens) per second, model loading excluded

| model | cpu_1t tok/s | cpu_omp tok/s | gpu_naive tok/s | gpu_naive vs cpu_1t | gpu_naive vs cpu_omp |
|---|---|---|---|---|---|
| stories15M | 176.6 | 681.0 | 457.6 | 2.59x | 0.67x |
| stories42M | 59.3 | 245.2 | 222.0 | 3.74x | 0.91x |
| stories110M | 22.2 | 92.5 | 108.6 | 4.89x | 1.17x |

Correctness (greedy decoding, generated text compared with cpu_1t):

- cpu_omp on stories110M: 12/12 identical
- cpu_omp on stories15M: 12/12 identical
- cpu_omp on stories42M: 12/12 identical
- gpu_naive on stories110M: 12/12 identical
- gpu_naive on stories15M: 12/12 identical
- gpu_naive on stories42M: 12/12 identical
