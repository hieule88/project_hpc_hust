# Benchmark on vmo-B760M-DS3H-DDR4

- Date: 2026-11-24
- CPU: Intel(R) Core(TM) i5-14400, cpu_omp uses 6 threads
- GPU: NVIDIA GeForce RTX 5060 Ti, 595.99.02
- Workload: `input/3.txt`, up to 64 positions per request, temperature 1.0, top-p 0.9, seed 314028
- tok/s = forward passes (prompt + generated tokens) per second, model loading excluded; 1 run per cell

| model | gpu_paged_b3 | gpu_optim_b3 vs cpu_omp | gpu_optim_b3 vs gpu_naive |
|---|---|---|---|
| sheared1.3B | 172.2 | n/a | n/a |

Correctness (greedy decoding, generated text compared with cpu_1t):

- gpu_paged_b3 on sheared1.3B: 3/3 identical
