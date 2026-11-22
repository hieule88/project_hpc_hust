# Llama-2 inference: CPU vs GPU naive (vs GPU optimized sau này)

Repo này có cấu trúc giống `llama2.hip-master`: code GPU (`run.cu`, `kernels/`, `unittests.cu`) nằm ở thư mục gốc cùng code CPU llama2.c (`run.c`). Bản GPU hiện tại là **bản naive**, chuyển trực tiếp từ `run.c` sang CUDA; bản optimized sau này sẽ thay dần các kernel trong `kernels/` (`matmul_gpu_v1` → `v2`, …) và giữ nguyên giao diện chạy, test, benchmark.

**Kết luận chính (RTX 5060 Ti, i5-14400, 12 prompt, 256 bước):**
- **Độ chính xác:** GPU naive cho output **giống hệt** CPU ở cả 3 model (`make check-diff`, giải mã greedy).
- **So với CPU 1 luồng:** GPU naive nhanh hơn **2.6× đến 4.9×**.
- **So với CPU 6 luồng:** GPU naive chỉ ngang ngửa (**0.67× đến 1.17×**).
- **Nút thắt:** nhân ma trận chiếm 95% thời gian GPU. Kernel matmul naive chỉ dùng được 6–30% băng thông bộ nhớ của GPU, nên đây là chỗ cần tối ưu đầu tiên.

## 1. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| request | 1 prompt trong file input; chương trình sinh tiếp văn bản cho prompt đó |
| forward pass | 1 lần chạy model cho 1 token (prompt hoặc token sinh ra) |
| tok/s | số forward pass mỗi giây, cộng chung cho mọi request, không tính thời gian nạp model |
| cpu_1t | `bin/runtest`: `-O3 -march=native`, 1 luồng, không fast-math; dùng làm chuẩn số học |
| cpu_omp | `bin/runtest_omp`: `-Ofast -fopenmp -march=native`, 6 luồng; cấu hình CPU nhanh nhất |
| gpu_naive | `bin/runcu`: bản CUDA naive (`run.cu` + `kernels/`) |

## 2. Cấu trúc repo (giống llama2.hip-master)

| File / thư mục | Vai trò | Tương ứng trong llama2.hip-master |
|---|---|---|
| `run.cu` | GPU: nạp model, `forward()`, tokenizer, sampler, mode `generate` / `chat` / `test` | `run.cc` |
| `kernels/*.cu, *.h` | Mỗi phép toán 1 file: bản GPU + bản CPU tham chiếu (chép từ `run.c`): `matmul`, `rmsnorm`, `softmax`, `rope`, `attn`, `swiglu`, `skip_conn`, `types.h`, `utils.h` | `kernels/*.cc, *.h` |
| `unittests.cu` | Test từng kernel, test forward so với CPU, benchmark từng kernel | `unittests.cc` |
| `Makefile` | build, `unit-tests`, `test-110M`, `check-diff`, `profile-110M`, `bench` | `Makefile` |
| `input/` | File prompt (dòng đầu = số request), giống hệt hip-master | `input/` |
| `output/` | Output tham chiếu cho `check-diff`: `110M_12.txt` (CPU), `110M_12_gpu.txt` (GPU) | `output/` |
| `profiling/` | Kết quả đo: unit test, từng kernel, profile, benchmark end-to-end | `profiling/` (hip-master không commit) |
| `run.c` | CPU llama2.c gốc, **không sửa** | — |
| `runtest.c`, `harness.h` | Chế độ test cho CPU; `harness.h` là vòng lặp test dùng chung cho CPU và GPU nên số liệu so sánh được | phần `test()` trong `run.cc` |
| `bench.sh` | Benchmark end-to-end CPU vs GPU + kiểm tra output | target `test-*` trong Makefile |
| `model/` | Checkpoint, không commit; tải bằng `make download-models` | `model/` (trong `.gitignore`) |
| `*.py`, `test.c`, `test_all.py`, `win.c`, `tokenizer.*` | File gốc của llama2.c | như nhau |

## 3. "Naive" ở đây nghĩa là gì

Mỗi bước trong `forward()` của `run.c` được chuyển thành kernel đơn giản nhất chạy đúng:

| Phép toán | Hàm | Cách chia thread |
|---|---|---|
| matmul (GEMV) | `matmul_gpu_v1` | 1 thread tính 1 hàng output, lặp qua cả hàng như code CPU |
| rmsnorm | `rmsnorm_gpu` | 1 block cho cả vector, cộng dồn bằng cây rút gọn trong shared memory |
| attention | `multihead_attention_gpu` | 3 kernel: score (1 thread / (head, t)), softmax (1 block / head), tổng có trọng số (1 thread / (head, i)) |
| rope | `rotary_embedding_gpu` | 1 thread / cặp phần tử |
| swiglu, residual | `swiglu_gpu`, `skip_conn_gpu` | 1 thread / phần tử |
| sampling | `sample` (CPU) | trên CPU sau khi chép logits về, giống `run.c` |

Không có: batching, gộp kernel, đọc vector (`float4`), warp shuffle, tiling shared memory, FP16. Mỗi token tốn 16 lần khởi chạy kernel cho mỗi layer.

## 4. Build và chạy

Yêu cầu: `gcc`, CUDA Toolkit có `nvcc` (máy này: CUDA 12.9 trong conda env `cuda129`), GPU NVIDIA. Mặc định build cho `sm_120` (RTX 50xx); GPU khác thì thêm `ARCH=sm_86` (hoặc tương ứng).

```bash
source ~/miniconda3/bin/activate cuda129    # để có nvcc
make download-models                        # model/stories{15M,42M,110M}.bin
make                                        # bin/runcu, bin/unit-tests, bin/runc, bin/runtest, bin/runtest_omp

make unit-tests                             # 43 test kernel + test forward (stories110M)
make check-diff                             # output greedy của GPU == output CPU trong output/
make test-110M                              # GPU, input/12.txt -> output/110M_12_gpu.txt
make cpu-110M                               # CPU 1 luồng -> output/110M_12.txt
make cpu-omp-110M NUM_THREADS=6             # CPU OpenMP
make profile-110M                           # thời gian GPU theo nhóm kernel -> profiling/
make bench-kernels                          # thời gian từng kernel
make bench                                  # CPU vs GPU end-to-end -> profiling/bench_<host>.{csv,md}
make test-110M INPUT=input/32.txt STEPS=128 # đổi workload

./bin/runcu model/stories110M.bin -i "Once upon a time" -n 256    # sinh văn bản như run.c
```

Chế độ test (`bin/runtest`, `bin/runcu -m test`): chạy lần lượt từng request; mỗi request dùng sampler mới với seed 314028, dừng ở BOS/EOS hoặc sau `-n` vị trí. Các target `test-*`, `cpu-*` dùng greedy (`-t 0`) để output so sánh được; `bench.sh` đo tốc độ với temperature 1.0, top-p 0.9. Output ghi mỗi request một dòng; chương trình in dòng `RESULT impl=... fwd_tok_s=... gen_tok_s=...`.

## 5. Kết quả trên máy dev

Máy: Intel Core i5-14400 (6 P-core + 4 E-core, RAM DDR4), NVIDIA RTX 5060 Ti 16 GB (36 SM, 448 GB/s theo thông số), driver 595.99, CUDA 12.9. File gốc trong [profiling/](profiling/).

### 5.1 Unit test: 44/44 pass

- **43 test kernel:** so mỗi kernel GPU với bản CPU trên dữ liệu ngẫu nhiên, ở kích thước thật của model và ở kích thước lẻ (1, 7, 13, 257, …). Có cả case GQA (`kv_heads < heads`) và layer thứ 2 trong KV cache (`loff ≠ 0`).
- **1 test forward:** chạy toàn bộ model stories110M trên GPU và CPU qua 48 vị trí. Logits lệch tối đa **2.0e-5** (tương đối 1.6e-6 so với logit lớn nhất), token argmax trùng **48/48**.
- **Test có bắt được lỗi không:** đã kiểm tra bằng cách cố ý cài lỗi chỉ số GQA vào kernel attention; test GQA báo FAIL như mong đợi.

Chi tiết: [profiling/unittests_vmo-B760M-DS3H-DDR4.txt](profiling/unittests_vmo-B760M-DS3H-DDR4.txt).

### 5.2 End-to-end: GPU naive chỉ thắng CPU đa luồng ở model lớn nhất

Mỗi số là 1 lần chạy với `input/12.txt`, 256 bước, khoảng 2400–2800 forward pass mỗi lần (`make bench`). Chạy lại lần 2 thì mọi ô lệch không quá 1.6% so với lần 1.

| model | cpu_1t tok/s | cpu_omp (6t) tok/s | gpu_naive tok/s | gpu_naive / cpu_1t | gpu_naive / cpu_omp |
|---|---|---|---|---|---|
| stories15M | 176.6 | 681.0 | 457.6 | 2.59× | 0.67× |
| stories42M | 59.3 | 245.2 | 222.0 | 3.74× | 0.91× |
| stories110M | 22.2 | 92.5 | 108.6 | 4.89× | 1.17× |

Giải mã greedy (`-t 0`): output của `cpu_omp` và `gpu_naive` **giống hệt** `cpu_1t` ở 12/12 request cho mọi model. File: [profiling/bench_vmo-B760M-DS3H-DDR4.md](profiling/bench_vmo-B760M-DS3H-DDR4.md).

Chọn 6 luồng cho `cpu_omp` vì nhanh nhất khi thử trên stories110M (3 prompt, 128 bước, 1 lần chạy mỗi mức): 4t = 91.7, 6t = 95.0, 10t = 85.2, 12t = 87.2, 16t = 88.1 tok/s.

### 5.3 Thời gian GPU của stories110M: matmul chiếm 95%

Đo bằng `make profile-110M` (`input/12.txt`, 2652 forward pass). Profile đồng bộ GPU sau mỗi nhóm kernel nên chạy chậm hơn bản thường khoảng 11% (97 so với 109 tok/s).

| nhóm kernel | µs / token | % |
|---|---|---|
| matmul q/k/v | 2884 | 30.2 |
| matmul w2 | 2532 | 26.5 |
| matmul w1/w3 | 1937 | 20.3 |
| matmul wo | 980 | 10.3 |
| matmul classifier | 742 | 7.8 |
| attention | 218 | 2.3 |
| rmsnorm, rope, swiglu, residual, embedding, logits D2H | 266 | 2.8 |
| **tổng** | **9559** | 100 |

### 5.4 Matmul naive dùng rất ít băng thông

Đo bằng `make bench-kernels` ở kích thước stories110M, trung bình 200 lần chạy:

| kernel | GPU µs | băng thông đạt được |
|---|---|---|
| matmul dim→dim (768×768) | 82.4 | 28.7 GB/s |
| matmul dim→hidden (768→2048) | 83.5 | 75.5 GB/s |
| matmul hidden→dim (2048→768) | 211.7 | 29.8 GB/s |
| matmul classifier (768→32000) | 732.4 | 134.4 GB/s |

Hai lý do, xếp theo mức ảnh hưởng:
1. **Quá ít thread:** với `d = 768`, cả GPU 36 SM chỉ có 3 block × 256 thread chạy.
2. **Đọc không gộp được:** các thread cạnh nhau đọc các hàng khác nhau của W (cách nhau `n` phần tử).

Ma trận classifier có `d = 32000` hàng nên có đủ thread, vì vậy đạt băng thông cao nhất. Bảng đầy đủ cho cả 3 model: [profiling/kernels_vmo-B760M-DS3H-DDR4.md](profiling/kernels_vmo-B760M-DS3H-DDR4.md).

## 6. Hướng tối ưu cho bản GPU optimized

Xếp theo mức đóng góp dự kiến, dựa trên profile ở mục 5.3:
1. **Matmul** (`matmul_gpu_v2`): mỗi warp tính 1 hàng, đọc `float4`, rút gọn bằng warp shuffle, để có đủ thread và đọc được gộp (như `batch_mat_vec_kernel` của hip-master).
2. **Batching nhiều request (continuous batching):** đọc trọng số 1 lần cho cả B request (như `test()` trong `run.cc` của hip-master).
3. **FP16 trọng số và KV cache:** giảm một nửa lượng bộ nhớ phải đọc.
4. **Gộp kernel:** K+V, W1+W3+SwiGLU, matmul+residual, RoPE+attention; giảm 16 lần khởi chạy kernel mỗi layer.

Giữ nguyên khi tối ưu để còn so sánh được với số liệu ở trên:
- **Giao diện test mode:** `bin/runcu <model> -m test -f <input> -o <output> -n <steps> -t <temp>`, in dòng `RESULT` như `harness.h`.
- **Kiểm tra đúng sai:** `make unit-tests` (thêm test cho kernel mới, so với bản `*_cpu`) và `make check-diff`.
- **Benchmark:** nếu muốn giữ bản naive làm mốc, thêm binary mới vào mảng `IMPLS` trong `bench.sh`.
