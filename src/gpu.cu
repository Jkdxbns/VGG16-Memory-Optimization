#include <cfloat>
#include "gpu_common.cuh"

// Tile/block configuration for coalesced kernel
#define BLOCK_DIM_X (16)
#define BLOCK_DIM_Y (16)
#define THREADS (256)

/**
 * @brief Global constant memory for weight tensors.
 * 
 * Used only in the coalesced kernel for small filter sizes (e.g., 3x3).
 * Accessing constant memory is faster when the same value is read across threads.
 */
__constant__ float const_weights[MAX_PARAM_CONSTANT];

/**
 * @brief Uncoalesced Conv2D kernel — intentionally inefficient.
 * 
 * Memory accesses are scattered and break coalescing. Used as a baseline for performance comparison.
 * 
 * Threads: 1D grid [N*K*P*Q], 1 thread per output pixel.
 * 
 * @param input  Global pointer to [N, C, H, W]
 * @param weight Global pointer to [K, C, R, S]
 * @param bias   Global pointer to [K]
 * @param output Global pointer to [N, K, P, Q]
 */
__global__ void conv2d_uncoalesced_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K * P * Q) return;

    // Compute 4D output index: [n, k, p, q]
    int q = idx % Q;
    int p = (idx / Q) % P;
    int k = (idx / (P * Q)) % K;
    int n = idx / (K * P * Q);

    float acc = bias[k];  // Initialize with bias

    // Iterate over all input channels and kernel region
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int h_in = p + r - R / 2;
                int w_in = q + s - S / 2;

                // Check for valid input pixel
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    // BAD access pattern: breaks coalescing (strides over N)
                    int input_idx = (((c * H + h_in) * W + w_in) * N + n);
                    int weight_idx = ((k * C + c) * R + r) * S + s;
                    acc += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((n * K + k) * P + p) * Q + q;
    output[output_idx] = acc;
}


/**
 * @brief Optimized memory-coalesced Conv2D kernel.
 * 
 * Threads are assigned to output pixels with 2D tiling. Global memory access is coalesced along the W dimension.
 * 
 * @param use_const_mem If true, kernel loads weights from constant memory instead of global memory.
 */
__global__ void conv2d_naive_kernel(
    const float* __restrict__ input, 
    const float* __restrict__ weight, 
    const float* __restrict__ bias, 
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int P, int Q,
    bool use_const_mem)
{
    // Compute output coordinates
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z;
    int n = idx / K;
    int k = idx % K;

    // Skip threads outside output bounds
    if (h_out >= P || w_out >= Q) return;

    float acc = bias[k];

    // Compute output[n, k, h_out, w_out]
    for (int c = 0; c < C; ++c) {
        int input_base = ((n * C + c) * H);           // Faster than recomputing each access
        int weight_base = (k * C + c) * R * S;

        // Loop over 3x3 filter region
        #pragma unroll
        for (int r = 0; r < 3; ++r) {
            int h_in = h_out + r - 1;
            if (h_in < 0 || h_in >= H) continue;

            #pragma unroll
            for (int s = 0; s < 3; ++s) {
                int w_in = w_out + s - 1;
                if (w_in < 0 || w_in >= W) continue;

                int input_idx = (input_base + h_in) * W + w_in;
                int weight_idx = weight_base + r * 3 + s;

                float w_val = use_const_mem ? const_weights[weight_idx] : weight[weight_idx];
                acc += input[input_idx] * w_val;
            }
        }
    }

    // Coalesced write to output
    int output_idx = ((n * K + k) * P + h_out) * Q + w_out;
    output[output_idx] = acc;
}

/**
 * @brief Launch wrapper for the optimized (coalesced) CUDA kernel.
 * 
 * Sets up 3D grid with 2D spatial blocks and batch/output channel index in Z.
 * 
 * @param stream CUDA stream to run the kernel asynchronously
 */
void launch_conv2d_naive(float* input, float* weight, float* bias, float* output,
                         int N, int C, int H, int W,
                         int K, int R, int S, int P, int Q,
                         bool use_const_mem, cudaStream_t stream)
{
    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((Q + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                 (P + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y,
                 N * K);

    conv2d_naive_kernel<<<gridDim, blockDim, 0, stream>>>(
        input, weight, bias, output,
        N, C, H, W, K, R, S, P, Q, use_const_mem);

    cudaDeviceSynchronize();
}

/**
 * @brief Launch wrapper for the uncoalesced (slow baseline) CUDA kernel.
 * 
 * Assigns 1D thread grid where each thread computes one output pixel.
 * 
 * @param stream CUDA stream to run the kernel
 */
void launch_conv2d_uncoalesced(float* input, float* weight, float* bias, float* output,
                               int N, int C, int H, int W,
                               int K, int R, int S, int P, int Q,
                               cudaStream_t stream)
{
    int total_threads = N * K * P * Q;
    int blockSize = 128;
    int gridSize = (total_threads + blockSize - 1) / blockSize;

    conv2d_uncoalesced_kernel<<<gridSize, blockSize, 0, stream>>>(
        input, weight, bias, output, N, C, H, W, K, R, S, P, Q);

    cudaStreamSynchronize(stream);
}

