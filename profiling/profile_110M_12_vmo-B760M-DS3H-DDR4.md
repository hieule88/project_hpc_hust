RESULT impl=gpu_naive model=stories110M.bin reqs=12 steps=256 fwd_tokens=2652 gen_tokens=2488 elapsed_s=27.261 fwd_tok_s=97.28 gen_tok_s=91.27

PROFILE (GPU time per kernel group, 2652 forward passes)
| group | total ms | % | us / token |
|---|---|---|---|
| embedding copy | 9.9 | 0.0 | 3.7 |
| rmsnorm | 282.2 | 1.1 | 106.4 |
| matmul q/k/v | 7646.6 | 30.2 | 2883.3 |
| rope | 115.8 | 0.5 | 43.7 |
| attention | 577.4 | 2.3 | 217.7 |
| matmul wo | 2599.1 | 10.3 | 980.0 |
| residual add | 143.0 | 0.6 | 53.9 |
| matmul w1/w3 | 5135.6 | 20.3 | 1936.5 |
| swiglu | 93.1 | 0.4 | 35.1 |
| matmul w2 | 6714.8 | 26.5 | 2532.0 |
| matmul classifier | 1965.7 | 7.8 | 741.2 |
| logits D2H | 68.3 | 0.3 | 25.7 |
| total | 25351.4 | 100.0 | 9559.3 |
