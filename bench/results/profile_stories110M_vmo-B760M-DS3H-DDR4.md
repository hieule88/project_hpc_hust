RESULT impl=gpu_naive model=stories110M.bin reqs=12 steps=256 fwd_tokens=2652 gen_tokens=2488 elapsed_s=27.251 fwd_tok_s=97.32 gen_tok_s=91.30

PROFILE (GPU time per kernel group, 2652 forward passes)
| group | total ms | % | us / token |
|---|---|---|---|
| embedding copy | 9.8 | 0.0 | 3.7 |
| rmsnorm | 283.0 | 1.1 | 106.7 |
| matmul q/k/v | 7649.1 | 30.2 | 2884.3 |
| rope | 115.9 | 0.5 | 43.7 |
| attention | 577.9 | 2.3 | 217.9 |
| matmul wo | 2599.1 | 10.3 | 980.0 |
| residual add | 142.3 | 0.6 | 53.7 |
| matmul w1/w3 | 5136.5 | 20.3 | 1936.8 |
| swiglu | 90.6 | 0.4 | 34.2 |
| matmul w2 | 6715.3 | 26.5 | 2532.2 |
| matmul classifier | 1967.4 | 7.8 | 741.9 |
| logits D2H | 62.8 | 0.2 | 23.7 |
| total | 25349.6 | 100.0 | 9558.7 |
