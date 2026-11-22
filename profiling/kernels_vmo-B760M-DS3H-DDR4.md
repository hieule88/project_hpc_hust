
== kernel benchmarks (model shapes: dim=288 hidden=768 vocab=32000 seq_len=256) ==
GPU: average over 200 launches (CUDA events). CPU: single-thread reference from run.c.
GB/s = bytes the kernel must read+write at least / GPU time.

| kernel | GPU us | CPU us | CPU/GPU | GPU GB/s |
|---|---|---|---|---|
| matmul wq/wo (dim->dim) | 31.1 | 25.6 | 0.8x | 10.8 |
| matmul wk/wv (dim->kv_dim) | 31.1 | 25.6 | 0.8x | 10.8 |
| matmul w1/w3 (dim->hidden) | 31.2 | 71.5 | 2.3x | 28.5 |
| matmul w2 (hidden->dim) | 81.7 | 84.8 | 1.0x | 10.9 |
| matmul classifier (dim->vocab) | 225.4 | 3042.4 | 13.5x | 164.2 |
| rmsnorm (dim) | 1.7 | 0.1 | 0.1x | 2.0 |
| rope | 1.4 | 1.7 | 1.2x | 3.2 |
| attention (pos=63) | 10.0 | 7.4 | 0.7x | 14.7 |
| attention (pos=255) | 14.2 | 26.4 | 1.9x | 41.7 |
| softmax (vocab) | 60.9 | 91.0 | 1.5x | 4.2 |
| swiglu (hidden) | 1.0 | 2.3 | 2.2x | 8.8 |
| residual add (dim) | 1.0 | 0.0 | 0.0x | 3.3 |


== kernel benchmarks (model shapes: dim=512 hidden=1376 vocab=32000 seq_len=1024) ==
GPU: average over 200 launches (CUDA events). CPU: single-thread reference from run.c.
GB/s = bytes the kernel must read+write at least / GPU time.

| kernel | GPU us | CPU us | CPU/GPU | GPU GB/s |
|---|---|---|---|---|
| matmul wq/wo (dim->dim) | 53.3 | 95.6 | 1.8x | 19.7 |
| matmul wk/wv (dim->kv_dim) | 53.4 | 94.6 | 1.8x | 19.7 |
| matmul w1/w3 (dim->hidden) | 53.7 | 256.9 | 4.8x | 52.7 |
| matmul w2 (hidden->dim) | 144.4 | 290.7 | 2.0x | 19.6 |
| matmul classifier (dim->vocab) | 493.1 | 6106.8 | 12.4x | 133.2 |
| rmsnorm (dim) | 1.8 | 0.3 | 0.2x | 3.5 |
| rope | 1.5 | 3.3 | 2.2x | 5.6 |
| attention (pos=63) | 11.7 | 11.9 | 1.0x | 22.4 |
| attention (pos=255) | 15.8 | 47.7 | 3.0x | 66.2 |
| attention (pos=1023) | 32.9 | 280.1 | 8.5x | 127.3 |
| softmax (vocab) | 61.0 | 90.7 | 1.5x | 4.2 |
| swiglu (hidden) | 1.1 | 3.9 | 3.7x | 15.6 |
| residual add (dim) | 1.0 | 0.0 | 0.0x | 5.9 |


== kernel benchmarks (model shapes: dim=768 hidden=2048 vocab=32000 seq_len=1024) ==
GPU: average over 200 launches (CUDA events). CPU: single-thread reference from run.c.
GB/s = bytes the kernel must read+write at least / GPU time.

| kernel | GPU us | CPU us | CPU/GPU | GPU GB/s |
|---|---|---|---|---|
| matmul wq/wo (dim->dim) | 82.4 | 227.8 | 2.8x | 28.7 |
| matmul wk/wv (dim->kv_dim) | 82.4 | 226.6 | 2.8x | 28.7 |
| matmul w1/w3 (dim->hidden) | 82.8 | 609.4 | 7.4x | 76.1 |
| matmul w2 (hidden->dim) | 211.5 | 650.0 | 3.1x | 29.8 |
| matmul classifier (dim->vocab) | 733.6 | 9801.9 | 13.4x | 134.2 |
| rmsnorm (dim) | 2.0 | 0.4 | 0.2x | 4.5 |
| rope | 1.5 | 4.9 | 3.4x | 8.4 |
| attention (pos=63) | 11.7 | 20.0 | 1.7x | 33.7 |
| attention (pos=255) | 15.9 | 75.6 | 4.8x | 99.0 |
| attention (pos=1023) | 39.5 | 432.6 | 10.9x | 159.2 |
| softmax (vocab) | 60.5 | 91.3 | 1.5x | 4.2 |
| swiglu (hidden) | 1.1 | 5.9 | 5.6x | 23.1 |
| residual add (dim) | 1.0 | 0.1 | 0.1x | 9.0 |

