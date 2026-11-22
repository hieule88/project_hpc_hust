# Benchmark on vmo-B760M-DS3H-DDR4

- Date: 2026-11-22
- CPU: Intel(R) Core(TM) i5-14400, cpu_omp uses 6 threads
- GPU: NVIDIA GeForce RTX 5060 Ti, 595.99.02
- Workload: `input/128.txt`, up to 256 positions per request, temperature 1.0, top-p 0.9, seed 314028
- tok/s = forward passes (prompt + generated tokens) per second, model loading excluded; 1 run per cell

| model | gpu_naive | gpu_optim_b1 | gpu_optim_b10 | gpu_pp_4x10 | gpu_optim_b10 vs cpu_omp | gpu_optim_b10 vs gpu_naive |
|---|---|---|---|---|---|---|
| stories15M | 457.8 | 1793.9 | 2903.2 | 5082.9 | n/a | 6.3x |
| stories42M | 222.1 | 1115.0 | 2489.4 | 4101.0 | n/a | 11.2x |
| stories110M | 108.4 | 595.8 | 2115.9 | 2690.0 | n/a | 19.5x |

Correctness (greedy decoding, generated text compared with cpu_1t):

- gpu_naive on stories110M: 128/128 identical
- gpu_naive on stories15M: 128/128 identical
- gpu_naive on stories42M: 128/128 identical
- gpu_optim_b1 on stories110M: 128/128 identical
- gpu_optim_b1 on stories15M: 128/128 identical
- gpu_optim_b1 on stories42M: 128/128 identical
- gpu_optim_b10 on stories110M: 128/128 identical
- gpu_optim_b10 on stories15M: 128/128 identical
- gpu_optim_b10 on stories42M: 128/128 identical
- gpu_pp_4x10 on stories110M: 128/128 identical
- gpu_pp_4x10 on stories15M: 128/128 identical
- gpu_pp_4x10 on stories42M: 128/128 identical
