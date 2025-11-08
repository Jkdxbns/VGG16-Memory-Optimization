#pragma once
#include <torch/torch.h>

/**
 * @brief Device memory buffers for coalesced memory access kernel.
 * These buffers are allocated once and reused across all Conv2D layers.
 */
extern float* global_d_input;
extern float* global_d_weight;
extern float* global_d_bias;
extern float* global_d_output;

/**
 * @brief Launches the optimized memory-coalesced Conv2D CUDA kernel.
 * 
 * @param input   Input tensor of shape [N, C, H, W]
 * @param weight  Filter weights of shape [K, C, R, S]
 * @param bias    Bias tensor of shape [K]
 * @param output  Output tensor of shape [N, K, P, Q] (written by kernel)
 * @param duration Output parameter to store kernel execution time in milliseconds
 */
void run_conv_gpu(const torch::Tensor& input, const torch::Tensor& weight,
                  const torch::Tensor& bias, torch::Tensor& output, float& duration);

/**
 * @brief Launches the uncoalesced baseline Conv2D CUDA kernel.
 * 
 * This version is intentionally inefficient (no coalescing, no buffer reuse).
 * Useful as a performance baseline to evaluate memory optimization benefits.
 * 
 * @param input   Input tensor of shape [N, C, H, W]
 * @param weight  Filter weights of shape [K, C, R, S]
 * @param bias    Bias tensor of shape [K]
 * @param output  Output tensor of shape [N, K, P, Q] (written by kernel)
 * @param duration Output parameter to store kernel execution time in milliseconds
 */
void run_convnc_gpu(const torch::Tensor& input, const torch::Tensor& weight,
                    const torch::Tensor& bias, torch::Tensor& output, float& duration);

