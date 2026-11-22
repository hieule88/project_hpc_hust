#include <float.h>
#include <hip/hip_runtime.h>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define DIVUP(x, y) ((x + y - 1) / y)
#define BLOCK_SIZE 32
#define WARP_SIZE 64
#define MAX_SEQ_LEN 4096
#define MAX_BATCH_SIZE 10

#define CHECK_HIP(call)                                                       \
    do {                                                                      \
        hipError_t status_ = call;                                            \
        if (status_ != hipSuccess) {                                          \
            fprintf(stderr, "HIP error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
                    hipGetErrorName(status_), hipGetErrorString(status_));    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__device__ __forceinline__ float warpReduceSum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[WARP_SIZE];  // Shared mem for 32 partial sums
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceSum(val);  // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val;  // Write reduced value to shared memory

    __syncthreads();  // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceSum(val);  // Final reduce within first warp

    return val;
}

__device__ __forceinline__ float warpReduceMax(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor(val, offset));
    return val;
}

__device__ __forceinline__ float blockReduceMax(float val) {
    static __shared__ float shared[WARP_SIZE];  // Shared mem for 32 partial sums
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduceMax(val);  // Each warp performs partial reduction

    if (lane == 0)
        shared[wid] = val;  // Write reduced value to shared memory

    __syncthreads();  // Wait for all partial reductions

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if (wid == 0)
        val = warpReduceMax(val);  // Final reduce within first warp

    return val;
}

// TODO: move to softmax file
__device__ __forceinline__ void softmax_kernel(float *x, int size) {
    const int thread_id = threadIdx.x;
    const int block_size = blockDim.x;
    const int block_id = blockIdx.x;

    __shared__ float max_val;
    __shared__ float d_total;

    float m_partial = -FLT_MAX;
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
        m_partial = fmaxf(m_partial, x[elem_id]);

    m_partial = blockReduceMax(m_partial);

    if (thread_id == 0)
        max_val = m_partial;
    __syncthreads();

    float d_partial = 0.0f;
    for (int elem_id = thread_id; elem_id < size; elem_id += block_size) {
        x[elem_id] = __expf(x[elem_id] - max_val);
        d_partial += x[elem_id];
    }

    d_partial = blockReduceSum(d_partial);
    if (thread_id == 0)
        d_total = d_partial;
    __syncthreads();

    for (int elem_id = thread_id; elem_id < size; elem_id += block_size)
        x[elem_id] /= d_total;
}
