# Naive CUDA port of llama2.c
#   make              build bin/runcu and bin/unittests
#   make test         run unit tests + forward check against the CPU reference
#   make bench-kernels  per-kernel timing at model shapes
#   make run          generate a story with $(MODEL)
# Needs nvcc on PATH (here: `source ~/miniconda3/bin/activate cuda129`).
# ARCH defaults to sm_120 (RTX 50xx); override e.g. `make ARCH=sm_86`.

NVCC  ?= nvcc
ARCH  ?= sm_120
MODEL ?= ../models/stories110M.bin
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo

KERNEL_SRCS := $(wildcard kernels/*.cu)
KERNEL_OBJS := $(patsubst kernels/%.cu,build/%.o,$(KERNEL_SRCS))
HEADERS     := $(wildcard kernels/*.h) ../common/harness.h

all: bin/runcu bin/unittests

build/%.o: kernels/%.cu $(HEADERS)
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

bin/runcu: run.cu $(KERNEL_OBJS) $(HEADERS)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) run.cu $(KERNEL_OBJS) -o $@

bin/unittests: unittests.cu run.cu $(KERNEL_OBJS) $(HEADERS)
	@mkdir -p bin
	$(NVCC) $(NVCCFLAGS) unittests.cu $(KERNEL_OBJS) -o $@

test: bin/unittests
	./bin/unittests $(MODEL)

bench-kernels: bin/unittests
	./bin/unittests $(MODEL) --bench

run: bin/runcu
	./bin/runcu $(MODEL) -i "Once upon a time" -n 256

clean:
	rm -rf build bin output

.PHONY: all test bench-kernels run clean
