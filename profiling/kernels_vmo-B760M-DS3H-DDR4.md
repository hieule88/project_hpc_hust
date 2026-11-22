---------Model Information----------
dim: 288
hidden_dim: 768
n_layers: 6
n_heads: 6
n_kv_heads: 6
vocab_size: 32000
seq_len: 256
weights_size: 57 MB
------------------------------------

== kernel benchmarks (model shapes: dim=288 hidden=768 vocab=32000 seq_len=256) ==
Average over 200 launches (CUDA events). Weight GB/s = weight bytes / time (weights are read once per launch).

| kernel | B=1 us | B=1 weight GB/s | B=10 us | B=10 weight GB/s | us per request, B=10 |
|---|---|---|---|---|---|
| matmul_gpu_v2 dim->dim | 1.8 | 188.1 | 3.9 | 85.5 | 0.4 |
| matmul_swiglu_gpu dim->hidden (W1+W3) | 3.8 | 466.4 | 12.0 | 147.2 | 1.2 |
| matmul_skipadd_gpu hidden->dim | 2.6 | 335.5 | 7.3 | 121.9 | 0.7 |
| matmul_gpu_v2 classifier | 89.0 | 414.1 | 113.3 | 325.3 | 11.3 |
| batch_multihead_attention_gpu, pos=seq_len-1 | 21.6 | 0.0 | 40.8 | 0.0 | 4.1 |

---------Model Information----------
dim: 512
hidden_dim: 1376
n_layers: 8
n_heads: 8
n_kv_heads: 8
vocab_size: 32000
seq_len: 1024
weights_size: 159 MB
------------------------------------

== kernel benchmarks (model shapes: dim=512 hidden=1376 vocab=32000 seq_len=1024) ==
Average over 200 launches (CUDA events). Weight GB/s = weight bytes / time (weights are read once per launch).

| kernel | B=1 us | B=1 weight GB/s | B=10 us | B=10 weight GB/s | us per request, B=10 |
|---|---|---|---|---|---|
| matmul_gpu_v2 dim->dim | 2.3 | 456.1 | 5.3 | 196.2 | 0.5 |
| matmul_swiglu_gpu dim->hidden (W1+W3) | 6.5 | 865.3 | 25.2 | 223.7 | 2.5 |
| matmul_skipadd_gpu hidden->dim | 4.5 | 626.9 | 12.1 | 233.9 | 1.2 |
| matmul_gpu_v2 classifier | 155.9 | 420.4 | 161.6 | 405.5 | 16.2 |
| batch_multihead_attention_gpu, pos=seq_len-1 | 83.1 | 0.0 | 586.4 | 0.0 | 58.6 |

---------Model Information----------
dim: 768
hidden_dim: 2048
n_layers: 12
n_heads: 12
n_kv_heads: 12
vocab_size: 32000
seq_len: 1024
weights_size: 418 MB
------------------------------------

== kernel benchmarks (model shapes: dim=768 hidden=2048 vocab=32000 seq_len=1024) ==
Average over 200 launches (CUDA events). Weight GB/s = weight bytes / time (weights are read once per launch).

| kernel | B=1 us | B=1 weight GB/s | B=10 us | B=10 weight GB/s | us per request, B=10 |
|---|---|---|---|---|---|
| matmul_gpu_v2 dim->dim | 3.2 | 733.8 | 7.8 | 301.2 | 0.8 |
| matmul_swiglu_gpu dim->hidden (W1+W3) | 10.8 | 1160.1 | 48.3 | 260.7 | 4.8 |
| matmul_skipadd_gpu hidden->dim | 6.8 | 928.1 | 16.8 | 374.0 | 1.7 |
| matmul_gpu_v2 classifier | 233.1 | 421.7 | 239.4 | 410.6 | 23.9 |
| batch_multihead_attention_gpu, pos=seq_len-1 | 82.2 | 0.0 | 791.7 | 0.0 | 79.2 |

