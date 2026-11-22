# Benchmark on vmo-B760M-DS3H-DDR4

- Date: 2026-11-22
- CPU: Intel(R) Core(TM) i5-14400, cpu_omp uses 6 threads
- GPU: NVIDIA GeForce RTX 5060 Ti, 595.99.02
- Workload: `input/12.txt`, up to 256 positions per request, temperature 1.0, top-p 0.9, seed 314028
- tok/s = forward passes (prompt + generated tokens) per second, model loading excluded; 1 run per cell

| model | cpu_1t | cpu_omp | gpu_naive | gpu_optim_b1 | gpu_optim_b10 | gpu_pp_4x10 | gpu_optim_b10 vs cpu_omp | gpu_optim_b10 vs gpu_naive |
|---|---|---|---|---|---|---|---|---|
| stories15M | 175.4 | 693.8 | 457.4 | 1797.8 | 2619.0 | 2195.0 | 3.8x | 5.7x |
| stories42M | 59.3 | 246.1 | 222.2 | 1100.2 | 2269.0 | 2721.4 | 9.2x | 10.2x |
| stories110M | 22.2 | 95.0 | 108.6 | 591.1 | 1814.3 | 1472.1 | 19.1x | 16.7x |

Correctness (greedy decoding, generated text compared with cpu_1t):

- cpu_omp on stories110M: 12/12 identical
- cpu_omp on stories15M: 12/12 identical
- cpu_omp on stories42M: 12/12 identical
- gpu_naive on stories110M: 12/12 identical
- gpu_naive on stories15M: 12/12 identical
- gpu_naive on stories42M: 12/12 identical
- gpu_optim_b1 on stories110M: 12/12 identical
- gpu_optim_b1 on stories15M: 12/12 identical
- gpu_optim_b1 on stories42M: 12/12 identical
- gpu_optim_b10 on stories110M: 12/12 identical
- gpu_optim_b10 on stories15M: 12/12 identical
- gpu_optim_b10 on stories42M: 12/12 identical
- gpu_pp_4x10 on stories110M: 12/12 identical
- gpu_pp_4x10 on stories15M: 12/12 identical
- gpu_pp_4x10 on stories42M: 12/12 identical
