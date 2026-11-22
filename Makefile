# Compiler and flags
CC        := gcc
NVCC      ?= nvcc
# RTX 50xx; e.g. `make ARCH=sm_86` for RTX 30xx
ARCH      ?= sm_120
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xcompiler -fopenmp
NVCCLIBS  := -lgomp

# Directories
SRC_DIR   := kernels
BUILD_DIR := build
MODEL_DIR := model

# Source files
SRC_FILES  := $(wildcard $(SRC_DIR)/*.cu)
HEADERS    := $(wildcard $(SRC_DIR)/*.h) harness.h
OBJ_FILES  := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRC_FILES))

# Executables
# GPU, data parallel + continuous batching (run.cu)
EXEC        := bin/runcu
# GPU, layers split over the GPUs + micro-batches (run_pp.cu)
PP_EXEC     := bin/runpp
# GPU kernels vs CPU reference (unittests.cu)
UNIT_TESTS  := bin/unit-tests
# CPU llama2.c, generate/chat (run.c)
CPU_EXEC    := bin/runc
# CPU test mode, -O3, 1 thread: numerical reference
CPU_TEST    := bin/runtest
# CPU test mode, -Ofast + OpenMP: fastest CPU config
CPU_TEST_MP := bin/runtest_omp

# Workload for the test-*/cpu-*/check-diff targets
# batch size per GPU (run.cu) / micro-batch size (run_pp.cu), 1..10
BATCH       ?= 10
INPUT       ?= input/12.txt
STEPS       ?= 256
# fastest OpenMP setting on the i5-14400 (see README)
NUM_THREADS ?= 6
TAG         := $(basename $(notdir $(INPUT)))

# Default target
all: $(EXEC) $(PP_EXEC) $(UNIT_TESTS) $(CPU_EXEC) $(CPU_TEST) $(CPU_TEST_MP)

# Rule to compile kernel sources into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(EXEC): run.cu $(OBJ_FILES) $(HEADERS)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) run.cu $(OBJ_FILES) $(NVCCLIBS) -o $@

$(PP_EXEC): run_pp.cu $(OBJ_FILES) $(HEADERS)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) run_pp.cu $(OBJ_FILES) $(NVCCLIBS) -o $@

$(UNIT_TESTS): unittests.cu run.cu $(OBJ_FILES) $(HEADERS)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) unittests.cu $(OBJ_FILES) $(NVCCLIBS) -o $@

$(CPU_EXEC): run.c
	@mkdir -p bin
	$(CC) -O3 -march=native -o $@ run.c -lm

$(CPU_TEST): runtest.c run.c harness.h
	@mkdir -p bin
	$(CC) -O3 -march=native -o $@ runtest.c -lm

$(CPU_TEST_MP): runtest.c run.c harness.h
	@mkdir -p bin
	$(CC) -Ofast -fopenmp -march=native -o $@ runtest.c -lm

# ---------------------------------------------------------------------------
# Tests

# kernel unit tests + full forward pass vs CPU
unit-tests: $(UNIT_TESTS)
	./$(UNIT_TESTS) $(MODEL_DIR)/stories110M.bin

# per-kernel timing at the model's shapes
bench-kernels: $(UNIT_TESTS)
	./$(UNIT_TESTS) $(MODEL_DIR)/stories110M.bin --bench

# GPU test mode, greedy, e.g. `make test-110M` -> output/110M_12_gpu.txt
test-%: $(EXEC)
	@mkdir -p output
	./$(EXEC) $(MODEL_DIR)/stories$*.bin -m test -f $(INPUT) -o output/$*_$(TAG)_gpu.txt -n $(STEPS) -t 0 -b $(BATCH)

# pipeline version, greedy, e.g. `make test-pp-110M` -> output/110M_12_gpu_pp.txt
test-pp-%: $(PP_EXEC)
	@mkdir -p output
	./$(PP_EXEC) $(MODEL_DIR)/stories$*.bin -m test -f $(INPUT) -o output/$*_$(TAG)_gpu_pp.txt -n $(STEPS) -t 0 -b $(BATCH)

# CPU reference, greedy, e.g. `make cpu-110M` -> output/110M_12.txt
cpu-%: $(CPU_TEST)
	@mkdir -p output
	./$(CPU_TEST) $(MODEL_DIR)/stories$*.bin -f $(INPUT) -o output/$*_$(TAG).txt -n $(STEPS) -t 0

# CPU OpenMP, same workload
cpu-omp-%: $(CPU_TEST_MP)
	@mkdir -p output
	OMP_NUM_THREADS=$(NUM_THREADS) ./$(CPU_TEST_MP) $(MODEL_DIR)/stories$*.bin -f $(INPUT) -o output/$*_$(TAG)_omp.txt -n $(STEPS) -t 0

# GPU greedy output must equal the committed CPU reference output
check-diff: test-15M test-42M test-110M test-pp-15M test-pp-42M test-pp-110M
	./check_diff.sh $(TAG)

# end-to-end CPU vs GPU throughput + greedy output check -> profiling/
bench: all
	NUM_THREADS=$(NUM_THREADS) INPUT=$(INPUT) STEPS=$(STEPS) BATCH=$(BATCH) ./bench.sh

download-models:
	@mkdir -p $(MODEL_DIR)
	for m in stories15M stories42M stories110M; do \
	  [ -f $(MODEL_DIR)/$$m.bin ] || wget -q --show-progress -O $(MODEL_DIR)/$$m.bin \
	    https://huggingface.co/karpathy/tinyllamas/resolve/main/$$m.bin; done

model-size:
	ls -lah $(MODEL_DIR)/

# Clean rule
clean:
	rm -rf $(BUILD_DIR) bin

.PHONY: all clean unit-tests bench-kernels check-diff bench download-models model-size
.PRECIOUS: $(BUILD_DIR)/%.o
