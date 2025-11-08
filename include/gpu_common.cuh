#pragma once
#include <cuda_runtime.h> 

#define MAX_PARAM_CONSTANT 16384


/**
 * @brief Constant memory buffer for filter weights (used in optimized kernel).
 * 
 * Only used when weight size is small enough to fit in constant memory.
 */
extern float const_weights[];

/**
 * @brief Launches the memory-coalesced Conv2D CUDA kernel.
 * 
 * @param input        Device pointer to input data
 * @param weight       Device pointer to weights
 * @param bias         Device pointer to bias
 * @param output       Device pointer to output buffer
 * @param N, C, H, W   Input dimensions
 * @param K, R, S      Output channels and filter size
 * @param P, Q         Output height and width
 * @param use_const_mem Whether to use constant memory for weights
 * @param stream       CUDA stream to launch the kernel
 */
void launch_conv2d_naive(float* input, float* weight, float* bias, float* output,
                         int N, int C, int H, int W,
                         int K, int R, int S, int P, int Q,
                         bool use_const_mem, cudaStream_t stream);

/**
 * @brief Launches the uncoalesced Conv2D CUDA kernel.
 * 
 * @param input        Device pointer to input data
 * @param weight       Device pointer to weights
 * @param bias         Device pointer to bias
 * @param output       Device pointer to output buffer
 * @param N, C, H, W   Input dimensions
 * @param K, R, S      Output channels and filter size
 * @param P, Q         Output height and width
 * @param stream       CUDA stream to launch the kernel
 */
void launch_conv2d_uncoalesced(float* input, float* weight, float* bias, float* output,
                                int N, int C, int H, int W,
                                int K, int R, int S, int P, int Q,
                                cudaStream_t stream);

