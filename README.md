# Llama-2 inference: CPU vs GPU naive (vs GPU optimized sau này)

Repo này chứa bản CPU gốc (llama2.c), một bản **GPU naive** chuyển trực tiếp từ code CPU sang CUDA, và bộ test/benchmark dùng chung để so sánh với bản **GPU optimized** sau này.

**Kết luận chính (RTX 5060 Ti, i5-14400, 12 prompt, 256 bước):**
- **Độ chính xác:** GPU naive cho output **giống hệt** CPU ở cả 3 model (36/36 request, giải mã greedy).
- **So với CPU 1 luồng:** GPU naive nhanh hơn **2.6× đến 4.9×**.
- **So với CPU 6 luồng:** GPU naive chỉ ngang ngửa (**0.66× đến 1.18×**).
- **Nút thắt:** nhân ma trận chiếm 95% thời gian GPU. Kernel matmul naive chỉ dùng được 6–30% băng thông bộ nhớ của GPU, nên đây là chỗ cần tối ưu đầu tiên.

## 1. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| request | 1 prompt trong file input; chương trình sinh tiếp văn bản cho prompt đó |
| forward pass | 1 lần chạy model cho 1 token (prompt hoặc token sinh ra) |
| tok/s | số forward pass mỗi giây, cộng chung cho mọi request, không tính thời gian nạp model |
| cpu_1t | `cpu/runtest`: `-O3 -march=native`, 1 luồng, không fast-math; dùng làm chuẩn số học |
| cpu_omp | `cpu/runtest_omp`: `-Ofast -fopenmp -march=native`, 6 luồng; cấu hình CPU nhanh nhất |
| gpu_naive | `gpu_naive/bin/runcu`: bản CUDA naive trong repo này |

## 2. Cấu trúc repo

```
cpu/            llama2.c gốc, run.c KHÔNG sửa; thêm runtest.c (chế độ test) + target `make runtest`
gpu_naive/      bản CUDA naive, cấu trúc giống llama2.hip-master
  run.cu          nạp model lên GPU, forward(), tokenizer, sampler, mode generate|chat|test
  kernels/        mỗi phép toán 1 file: bản GPU naive + bản CPU tham chiếu (chép từ run.c)
    matmul  rmsnorm  softmax  rope  attn  swiglu  skip_conn  (+ types.h, utils.h)
  unittests.cu    unit test kernel, test forward so với CPU, benchmark từng kernel
  input/          file prompt (dòng đầu = số request)
  Makefile
common/harness.h  vòng lặp chế độ test dùng chung cho mọi bản -> số liệu so sánh được
bench/bench.sh    benchmark end-to-end + kiểm tra output, ghi bench/results/
models/           checkpoint (không commit), tải bằng models/download.sh
```

## 3. "Naive" ở đây nghĩa là gì

Mỗi bước trong `forward()` của `run.c` được chuyển thành kernel đơn giản nhất chạy đúng:

| Phép toán | Cách chia thread |
|---|---|
| matmul (GEMV) | 1 thread tính 1 hàng output, lặp qua cả hàng như code CPU |
| rmsnorm | 1 block cho cả vector, cộng dồn bằng cây rút gọn trong shared memory |
| attention | 3 kernel: score (1 thread / (head, t)), softmax (1 block / head), tổng có trọng số (1 thread / (head, i)) |
| rope, swiglu, residual | 1 thread / phần tử |
| sampling | trên CPU sau khi chép logits về, giống `run.c` |

Không có: batching, gộp kernel, đọc vector (`float4`), warp shuffle, tiling shared memory, FP16. Mỗi token tốn 16 lần khởi chạy kernel cho mỗi layer.

## 4. Build và chạy

Yêu cầu: `gcc`, CUDA Toolkit có `nvcc` (máy này: CUDA 12.9 trong conda env `cuda129`), GPU NVIDIA. Mặc định build cho `sm_120` (RTX 50xx); GPU khác thì thêm `ARCH=sm_86` (hoặc tương ứng).

```bash
models/download.sh                          # stories15M / 42M / 110M
source ~/miniconda3/bin/activate cuda129    # để có nvcc

cd gpu_naive
make                                        # bin/runcu, bin/unittests
make test                                   # unit test + forward test (stories110M)
make bench-kernels                          # thời gian từng kernel
./bin/runcu ../models/stories110M.bin -i "Once upon a time" -n 256          # như run.c
./bin/runcu ../models/stories110M.bin -m test -f input/12.txt -o output/out.txt
LLAMA_PROFILE=1 ./bin/runcu ...             # thêm bảng thời gian theo nhóm kernel

cd ..
bench/bench.sh                              # CPU vs GPU, ghi bench/results/<host>.{csv,md}
MODELS=stories110M STEPS=128 INPUT=gpu_naive/input/32.txt CPU_THREADS=6 bench/bench.sh
```

Chế độ test (`cpu/runtest`, `runcu -m test`): chạy lần lượt từng request; mỗi request dùng sampler mới với seed 314028 (temperature 1.0, top-p 0.9 nếu không đổi bằng `-t`/`-p`), dừng ở BOS/EOS hoặc sau `-n` vị trí. Output ghi mỗi request một dòng; in ra dòng `RESULT impl=... fwd_tok_s=... gen_tok_s=...`.

## 5. Kết quả trên máy dev

Máy: Intel Core i5-14400 (6 P-core + 4 E-core, RAM DDR4), NVIDIA RTX 5060 Ti 16 GB (36 SM, 448 GB/s theo thông số), driver 595.99, CUDA 12.9. File gốc: [bench/results/](bench/results/).

### 5.1 Unit test: 44/44 pass

- **43 test kernel:** so mỗi kernel GPU với bản CPU trên dữ liệu ngẫu nhiên, ở kích thước thật của model và ở kích thước lẻ (1, 7, 13, 257, …). Có cả case GQA (`kv_heads < heads`) và layer thứ 2 trong KV cache (`loff ≠ 0`).
- **1 test forward:** chạy toàn bộ model stories110M trên GPU và CPU qua 48 vị trí. Logits lệch tối đa **2.0e-5** (tương đối 1.6e-6 so với logit lớn nhất), token argmax trùng **48/48**.
- **Test có bắt được lỗi không:** đã kiểm tra bằng cách cố ý cài lỗi chỉ số GQA vào kernel attention; test GQA báo FAIL như mong đợi.

Chi tiết: [bench/results/unittests_vmo-B760M-DS3H-DDR4.txt](bench/results/unittests_vmo-B760M-DS3H-DDR4.txt).

### 5.2 End-to-end: GPU naive chỉ thắng CPU đa luồng ở model lớn nhất

Mỗi số là 1 lần chạy với `input/12.txt`, 256 bước, khoảng 2400–2800 forward pass mỗi lần:

| model | cpu_1t tok/s | cpu_omp (6t) tok/s | gpu_naive tok/s | gpu_naive / cpu_1t | gpu_naive / cpu_omp |
|---|---|---|---|---|---|
| stories15M | 176.8 | 692.2 | 457.6 | 2.59× | 0.66× |
| stories42M | 59.2 | 246.6 | 221.9 | 3.75× | 0.90× |
| stories110M | 22.4 | 92.2 | 108.8 | 4.87× | 1.18× |

Giải mã greedy (`-t 0`): output của `cpu_omp` và `gpu_naive` **giống hệt** `cpu_1t` ở 12/12 request cho mọi model.

Chọn 6 luồng cho `cpu_omp` vì nhanh nhất khi thử trên stories110M (3 prompt, 128 bước, 1 lần chạy mỗi mức): 4t = 91.7, 6t = 95.0, 10t = 85.2, 12t = 87.2, 16t = 88.1 tok/s.

### 5.3 Thời gian GPU của stories110M: matmul chiếm 95%

Đo bằng `LLAMA_PROFILE=1`, `input/12.txt`, 2652 forward pass. Profile đồng bộ GPU sau mỗi nhóm kernel nên chạy chậm hơn bản thường khoảng 11% (97 so với 109 tok/s).

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

Đo từng kernel ở kích thước stories110M, trung bình 200 lần chạy:

| kernel | GPU µs | băng thông đạt được |
|---|---|---|
| matmul dim→dim (768×768) | 82.4 | 28.7 GB/s |
| matmul dim→hidden (768→2048) | 83.5 | 75.5 GB/s |
| matmul hidden→dim (2048→768) | 211.7 | 29.8 GB/s |
| matmul classifier (768→32000) | 732.4 | 134.4 GB/s |

Hai lý do, xếp theo mức ảnh hưởng:
1. **Quá ít thread:** với `d = 768`, cả GPU 36 SM chỉ có 3 block × 256 thread chạy.
2. **Đọc không gộp được:** các thread cạnh nhau đọc các hàng khác nhau của W (cách nhau `n` phần tử).

Ma trận classifier có `d = 32000` hàng nên có đủ thread, vì vậy đạt băng thông cao nhất. Bảng đầy đủ cho cả 3 model: [bench/results/kernels_vmo-B760M-DS3H-DDR4.md](bench/results/kernels_vmo-B760M-DS3H-DDR4.md).

## 6. Hướng tối ưu cho bản GPU optimized

Xếp theo mức đóng góp dự kiến, dựa trên profile ở mục 5.3:
1. **Matmul:** mỗi warp tính 1 hàng, đọc `float4`, rút gọn bằng warp shuffle, để có đủ thread và đọc được gộp (xem `llama2.hip-master`, `cullama`).
2. **Batching nhiều request (continuous batching):** đọc trọng số 1 lần cho cả B request.
3. **FP16 trọng số và KV cache:** giảm một nửa lượng bộ nhớ phải đọc.
4. **Gộp kernel:** K+V, W1+W3+SwiGLU, matmul+residual, RoPE+attention; giảm 16 lần khởi chạy kernel mỗi layer.

Để so sánh được với số liệu ở trên, bản optimized cần:
- **Giao diện:** nhận `<model> -m test -f <input> -o <output> -n <steps> -t <temp> -z <tokenizer>` và in dòng `RESULT` giống `common/harness.h`.
- **Đăng ký vào bench:** thêm một dòng vào mảng `IMPLS` trong `bench/bench.sh`.
