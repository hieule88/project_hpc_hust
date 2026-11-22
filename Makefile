# Compiler and flags
CXX := hipcc
CXXFLAGS := -fopenmp --offload-arch=gfx908 -O2#-ggdb -O0

# Directories
SRC_DIR := kernels
BUILD_DIR := build

# Source files
SRC_FILES := $(wildcard $(SRC_DIR)/*.cc)
MAIN_SRC := run.cc
MAIN_PP_SRC := run_pp.cc
UNIT_TESTS_SRC := unittests.cc

# Object files
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(SRC_FILES))
MAIN_OBJ := $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(MAIN_SRC))
MAIN_PP_OBJ := $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(MAIN_PP_SRC))
UNIT_TESTS_OBJ := $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(UNIT_TESTS_SRC))

# Executable name
EXEC := bin/runcc
PP_EXEC := bin/runpp
UNIT_TESTS := bin/unit-tests
ROCPROF := rocprof --hip-trace -o ./profiling/llama2-profile-7B-pipeline-6.csv
ROCPROF2 := rocprof --hip-trace -o ./profiling/unit-tests.csv
ROCPROF13B := rocprof --hip-trace -o ./profiling/llama2-profile-13B-2.csv
OMNIPERF := omniperf profile -n 110M3 --
MATMUL_OMNIPERF := omniperf profile -n matmul --
NUMA := numactl -N 0 -C 0-32,48-79 -l
NUM_GPUS := 4

# Default target
all: $(EXEC) $(PP_EXEC)
unit-tests: $(UNIT_TESTS)
	srun -p EM --gres=gpu:1 ./bin/unit-tests

gprof:
	g++ -o ./bin/profile -O0 -g -pg run.cc $(KERNEL_FILES) && \
	./bin/profile /shared/erc/getpTA/main/modelbin/stories110M.bin -n 100 && \
	gprof ./bin/profile gmon.out > ./profiling/gprof.out

# Main target
$(EXEC): $(OBJ_FILES) $(MAIN_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(PP_EXEC): $(OBJ_FILES) $(MAIN_PP_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(UNIT_TESTS): $(OBJ_FILES) $(UNIT_TESTS_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Rule to compile source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -rf $(BUILD_DIR) $(EXEC) $(PP_EXEC)

.PHONY: all clean

test-110M:
	srun --exclusive -p EM --gres=gpu:${NUM_GPUS} ./bin/runcc /shared/erc/getpTA/main/modelbin/stories110M.bin -m test -f input/128.txt -o output/110M_128_gpu.txt -b 10
test-4B:
	srun --exclusive -p EM --gres=gpu:${NUM_GPUS} ./bin/runcc /shared/erc/getpTA/main/modelbin/llama2-4b.bin -m test -f input/512.txt -o output/4B_512_gpu.txt -b 8
test-7B-pp:
	srun --exclusive -p EM --gres=gpu:4 ./bin/runpp /shared/erc/getpTA/main/modelbin/llama2-7b.bin -m test -f input/512.txt -o output/7B_512_gpu.txt -b 10
test-13B:
	srun --exclusive -p EM --gres=gpu:4 ./bin/runpp /shared/erc/getpTA/main/modelbin/llama2-13b.bin -m test -f input/128.txt -o output/13B_128_gpu.txt -b 6
debug:
	srun -p EM --gres=gpu:${NUM_GPUS} --pty rocgdb --args  ./bin/runcc /shared/erc/getpTA/main/modelbin/llama2-4b.bin -m test -f input/12.txt -o output/4B_12_gpu.txt -b 8
matmul-tests:
	hipcc -o ./bin/matmul-tests -O2 --offload-arch=gfx908 unittests.cc ./kernels/utils.h ./kernels/matmul.cc &&\
	srun -p EM --gres=gpu:1 ${MATMUL_OMNIPERF} ./bin/matmul-tests
check-diff:
	# diff output/110M_1_gpu.txt output/110M_1.txt && \
	# diff output/110M_5_gpu.txt output/110M_5.txt && \
	diff output/110M_12_gpu.txt output/110M_12.txt && \
	echo "Done"
model-size:
	ls -lah /shared/erc/getpTA/main/modelbin/
perfetto-trace:
	./trace_processor --httpd --http-port 9091 ./profiling/llama2-profile-4B-1gpu-3batch.json
