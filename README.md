# Llama-2 inference: GPU optimized (CUDA port of llama2.hip-master)

Nhánh `code_optim_cuda`: bản GPU **optimized** là code của `llama2.hip-master` (viết cho GPU AMD, HIP) chuyển sang CUDA, đặt trong đúng cấu trúc của hip-master. Bản CPU (`run.c`) và bộ test/benchmark giữ nguyên từ nhánh `main`; bản **GPU naive** nằm ở nhánh `main` và được `bench.sh` tự build lại (qua `git worktree`) để so trên cùng máy.

**Kết luận chính (RTX 5060 Ti, i5-14400):**
- **Độ chính xác:** mọi bản GPU cho output greedy **giống hệt** CPU: 12/12 request (`input/12.txt`) và 128/128 (`input/128.txt`) ở cả 3 model. 99/99 unit test pass.
- **Tốc độ (stories110M, 128 prompt):** optimized `-b 10` đạt **2116 tok/s**, gấp **19.5×** bản naive (108 tok/s). Với 12 prompt, nó gấp **19.1×** CPU 6 luồng (1814 so với 95 tok/s).
- **Nguồn gốc của mức tăng:** riêng việc viết lại kernel (cùng batch 1) đã tăng **5.5×**; batching đến 10 request tăng thêm **3.6×** nữa (stories110M, 128 prompt).
- **Kernel classifier đạt 94% băng thông VRAM** (422 / 448 GB/s), nên ở batch 1 phần nhân ma trận đã gần chạm giới hạn phần cứng.

## 1. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| request | 1 prompt trong file input |
| ô (slot) | 1 chỗ trong batch, giữ 1 request đang chạy cùng KV cache của nó |
| forward pass | 1 lần chạy model cho 1 token của 1 request |
| tok/s | số forward pass mỗi giây, cộng mọi request, không tính thời gian nạp model |
| cpu_1t / cpu_omp | `bin/runtest` (`-O3`, 1 luồng, chuẩn số học) / `bin/runtest_omp` (`-Ofast -fopenmp`, 6 luồng) |
| gpu_naive | `bin/runcu` của nhánh `main` (1 thread / hàng output, không batch, không gộp kernel) |
| gpu_optim_bB | `bin/runcu -b B` của nhánh này (`run.cu`) |
| gpu_pp_4xB | `bin/runpp -b B` (`run_pp.cu`): 4 micro-batch × B ô |

## 2. Cấu trúc (giống llama2.hip-master)

| File | Vai trò | Trong llama2.hip-master |
|---|---|---|
| `run.cu` | Mỗi GPU giữ cả model; continuous batching tối đa 10 ô mỗi GPU | `run.cc` |
| `run_pp.cu` | Chia layer cho các GPU; 4 luồng OpenMP, mỗi luồng 1 micro-batch | `run_pp.cc` |
| `kernels/*.cu, *.h` | `attn`, `matmul`, `rmsnorm`, `rope`, `skip_conn`, `softmax`, `swiglu`, `types.h`, `utils.h`: bản GPU + bản CPU tham chiếu | `kernels/*.cc, *.h` |
| `unittests.cu` | Test mọi hàm GPU với batch 1/3/10, test forward batch so với CPU, benchmark kernel | `unittests.cc` |
| `Makefile` | `unit-tests`, `test-<M>`, `test-pp-<M>`, `cpu-<M>`, `check-diff`, `bench-kernels`, `bench` | `Makefile` |
| `input/`, `output/` | Prompt; output tham chiếu của CPU (`110M_12.txt`) và GPU (`_gpu`, `_gpu_pp`) | như nhau |
| `profiling/` | Kết quả đo của nhánh này | `profiling/` |
| `run.c`, `runtest.c`, `harness.h`, `bench.sh`, `check_diff.sh` | CPU gốc, chế độ test dùng chung, benchmark, so output | — |

## 3. Các kỹ thuật tối ưu (giữ nguyên từ hip-master)

Xếp theo mức đóng góp, theo số đo ở mục 5:
1. **GEMV theo warp:** mỗi warp tính 1 hàng output, mỗi lane đọc `float4`, rút gọn bằng `__shfl_xor_sync`. Đọc bộ nhớ gộp được và có đủ thread. Đây là phần cho mức tăng 5.5× ở batch 1.
2. **Batching, đọc trọng số 1 lần cho B request** (`matmul_gpu_v2`): mỗi lane giữ `sum[B]` trong thanh ghi.
3. **Continuous batching** (`test()` trong `run.cu`): sau mỗi bước sinh token, ô xong được nạp request mới; cửa sổ batch co lại khi hàng đợi hết.
4. **Gộp kernel:** K+V ghi thẳng vào KV cache (`matmul_gpu_v4`), RoPE+attention (`batch_multihead_attention_gpu`), W1+W3+SwiGLU (`matmul_swiglu_gpu`), matmul+residual (`matmul_skipadd_gpu`). Còn 8 lần khởi chạy kernel mỗi layer, bản naive là 16.
5. **Logits ghi thẳng vào bộ nhớ CPU pinned** (`cudaHostAlloc`), không cần lệnh chép từ GPU về.
6. **Nhiều GPU:** `run.cu` chép model lên mỗi GPU; `run_pp.cu` chia layer ra các GPU.

## 4. Những thay đổi khi chuyển HIP → CUDA

| Thay đổi | Lý do | Bằng chứng |
|---|---|---|
| `WARP_SIZE` 64 → 32 | Warp NVIDIA có 32 lane; giữ 64 thì phép rút gọn chỉ cộng nửa hàng | đặt lại 64: 72/98 unit test FAIL |
| `__align__(16)` cho `q_smem` | đọc `float4` từ shared memory phải căn 16 byte trên CUDA | trước khi sửa: `cudaErrorMisalignedAddress` |
| Chép embedding bằng `cudaMemcpy` | CPU không ghi được vào bộ nhớ `cudaMalloc` (`run.cu`), hoặc ghi vào bộ nhớ GPU từ bảng pinned (`run_pp.cu`) | trước khi sửa: segfault |
| Softmax lấy max của cả block (`blockReduceMax`) | bản gốc chỉ lấy max của thread 0, có thể tràn `inf` | lỗi đã ghi nhận khi đọc code |
| RoPE dùng `sinf/cosf/powf` chính xác, cùng công thức `run.c` | `__sinf/__cosf` lệch tới 1e-4 ở vị trí 1023, và lệch tăng theo vị trí | unit test RoPE FAIL trước khi sửa |
| Offset KV cache 64-bit | `batch_id × n_layers × seq_len × kv_dim` tràn `int` với 7B, `b ≥ 8` | tính toán |
| Chặn `-b > 10`; báo lỗi khi model dùng GQA | kernel giữ `sum[10]`; kernel RoPE gộp sẽ xoay K dùng chung nhiều lần | — |
| `run_pp.cu` đếm số GPU lúc chạy (tối đa 4, phải chia hết số layer) | bản gốc gán cứng 4 GPU | chạy được trên máy 1 GPU |
| Bỏ các hàm `*_gpu_test` | ghi qua con trỏ GPU từ CPU; thay bằng `unittests.cu` | — |
| Nhận `-n/-t/-p/-s`, output và dòng `RESULT` theo `harness.h` | để so sánh được với CPU và bản naive | — |

## 5. Kết quả

Máy: Intel Core i5-14400 (RAM DDR4), NVIDIA RTX 5060 Ti 16 GB (36 SM, 448 GB/s theo thông số), CUDA 12.9. Mỗi số là 1 lần chạy. Trên nhánh `main`, chạy lại lần 2 thì mọi ô lệch không quá 1.6%.

### 5.1 Unit test: 99/99 pass

- **Kernel:** 98 test, gồm mọi hàm GPU trong `kernels/`, với batch 1, 3 và 10. Mỗi ô ở một vị trí khác nhau, có cả vị trí 0 và `seq_len - 1`. Kiểm tra cả việc kernel chỉ ghi đúng chỗ trong KV cache và không đụng tới `q`.
- **Forward:** 3 ô chạy 3 prompt khác nhau; một ô vào muộn 5 bước, giống continuous batching. So logits với CPU ở 139 cặp (ô, bước): lệch tối đa 3.6e-5, argmax trùng 139/139.

File: [profiling/unittests_vmo-B760M-DS3H-DDR4.txt](profiling/unittests_vmo-B760M-DS3H-DDR4.txt).

### 5.2 End-to-end, 12 prompt (có CPU)

`input/12.txt`, 256 bước, temperature 1.0, top-p 0.9:

| model | cpu_1t | cpu_omp | gpu_naive | gpu_optim_b1 | gpu_optim_b10 | gpu_pp_4x10 | b10 / cpu_omp | b10 / naive |
|---|---|---|---|---|---|---|---|---|
| stories15M | 175.4 | 693.8 | 457.4 | 1797.8 | 2619.0 | 2195.0 | 3.8× | 5.7× |
| stories42M | 59.3 | 246.1 | 222.2 | 1100.2 | 2269.0 | 2721.4 | 9.2× | 10.2× |
| stories110M | 22.2 | 95.0 | 108.6 | 591.1 | 1814.3 | 1472.1 | 19.1× | 16.7× |

### 5.3 End-to-end, 128 prompt (chỉ GPU, đủ việc để lấp đầy batch)

| model | gpu_naive | gpu_optim_b1 | gpu_optim_b10 | gpu_pp_4x10 | b1 / naive | b10 / b1 |
|---|---|---|---|---|---|---|
| stories15M | 457.8 | 1793.9 | 2903.2 | 5082.9 | 3.9× | 1.6× |
| stories42M | 222.1 | 1115.0 | 2489.4 | 4101.0 | 5.0× | 2.2× |
| stories110M | 108.4 | 595.8 | 2115.9 | 2690.0 | 5.5× | 3.6× |

Cột `b1 / naive` là phần lợi nhờ viết lại kernel; cột `b10 / b1` là phần lợi nhờ batching. Batching lợi nhiều hơn ở model lớn, vì model càng lớn thì càng bị giới hạn bởi tốc độ đọc trọng số.

**`gpu_pp_4x10` nhanh nhất trên 1 GPU** dù không có pipeline thật, vì chỉ có 1 GPU. Giả thuyết, chưa đo: nó có 4 luồng CPU, nên trong lúc luồng này lấy mẫu token trên CPU (softmax + top-p trên 32000 logits cho mỗi ô), luồng khác đã đưa micro-batch tiếp theo vào GPU. Còn `run.cu` chạy tuần tự forward → lấy mẫu → forward. Để kiểm chứng: so `gpu_optim_b10` với `-t 0` (greedy, lấy mẫu rất rẻ) và `-t 1`.

Giải mã greedy cho output giống hệt CPU ở mọi bản, mọi model, cả 12 lẫn 128 prompt. File: [profiling/bench_vmo-B760M-DS3H-DDR4_in12.md](profiling/bench_vmo-B760M-DS3H-DDR4_in12.md), [profiling/bench_vmo-B760M-DS3H-DDR4_in128.md](profiling/bench_vmo-B760M-DS3H-DDR4_in128.md).

### 5.4 Từng kernel (stories110M, `make bench-kernels`)

| kernel | B=1 µs | B=10 µs | µs / request ở B=10 | băng thông đọc trọng số, B=1 |
|---|---|---|---|---|
| matmul dim→dim (768×768) | 3.2 | 7.8 | 0.8 | 734 GB/s* |
| matmul W1+W3+SwiGLU (768→2048) | 10.8 | 48.3 | 4.8 | 1160 GB/s* |
| matmul W2 + residual (2048→768) | 6.8 | 16.8 | 1.7 | 928 GB/s* |
| matmul classifier (768→32000, 98 MB) | 233.1 | 239.4 | 23.9 | **422 GB/s** |
| attention, pos = 1023 | 82.2 | 791.7 | 79.2 | — |

\* Vượt băng thông VRAM vì ma trận nhỏ (2.4–12.6 MB) nằm trong L2 cache 32 MB khi chạy lặp 200 lần; trong lúc chạy model thật thì không như vậy. Chỉ dòng classifier phản ánh đúng việc đọc từ VRAM.

So với bản naive (nhánh `main`, cùng kích thước): matmul dim→dim mất 82.4 µs, classifier mất 732.4 µs (134 GB/s).

Hai điểm rút ra:
- **Classifier:** đi từ B=1 lên B=10, thời gian gần như không đổi (233 → 239 µs). Trọng số được đọc 1 lần cho 10 request, nên thời gian mỗi request giảm gần 10 lần.
- **Attention:** không có gì để dùng chung giữa các request, nên thời gian tăng tuyến tính theo B. Ở vị trí cuối (1023), attention của 10 request (792 µs) chiếm nhiều hơn mọi phép nhân ma trận của một layer cộng lại. Đây là chỗ cần tối ưu tiếp, ví dụ chia chiều thời gian cho nhiều block (split-K, kiểu flash-decoding).

## 6. Build và chạy

```bash
source ~/miniconda3/bin/activate cuda129
make download-models && make
make unit-tests                       # 99 test
make check-diff                       # greedy GPU (run.cu, run_pp.cu) vs CPU, 3 model
make test-110M BATCH=10               # -> output/110M_12_gpu.txt
make bench                            # 12 prompt, mọi bản, gồm cả naive build từ nhánh main
ONLY='^gpu' INPUT=input/128.txt ./bench.sh
./bin/runcu model/stories110M.bin -m test -f input/128.txt -o out.txt -b 10
```
