#include "gpu_common.cuh"
#include "gpu.h"
#include <cuda_runtime.h>
#include <chrono>

// Global device buffers reused by coalesced kernel
float *global_d_input = nullptr;
float *global_d_weight = nullptr;
float *global_d_bias = nullptr;
float *global_d_output = nullptr;

/**
 * @brief Runs the optimized (memory-coalesced) CUDA Conv2D kernel.
 * 
 * Allocates and reuses shared global memory buffers for performance. Optionally uses constant
 * memory if the filter size is small enough. Runs the kernel on a CUDA stream and records execution time.
 * 
 * @param input     Input tensor of shape [N, C, H, W]
 * @param weight    Weights tensor of shape [K, C, R, S]
 * @param bias      Bias tensor of shape [K]
 * @param output    Output tensor to store result [N, K, P, Q]
 * @param duration  (Output) Measured kernel execution time in milliseconds
 */
void run_conv_gpu(const torch::Tensor& input, const torch::Tensor& weight,
                  const torch::Tensor& bias, torch::Tensor& output, float& duration)
{
    // Extract tensor dimensions
    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int K = weight.size(0), R = weight.size(2), S = weight.size(3);
    int P = output.size(2), Q = output.size(3);

    // Calculate memory sizes
    size_t input_size = N * C * H * W * sizeof(float);
    size_t weight_size = K * C * R * S * sizeof(float);
    size_t bias_size = K * sizeof(float);
    size_t output_size = N * K * P * Q * sizeof(float);
    
    // Launch using a non-blocking CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Copy input, weights, and bias to device
    cudaMemcpyAsync(global_d_input, input.data_ptr<float>(), input_size, cudaMemcpyHostToDevice, stream);

    // Use constant memory for weights if small enough (faster, cached reads)
    bool use_const_mem = (weight.numel() <= MAX_PARAM_CONSTANT);
    if (use_const_mem) {
        cudaMemcpyToSymbolAsync(const_weights, weight.data_ptr<float>(), weight_size,
                                0, cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpyAsync(global_d_weight, weight.data_ptr<float>(), weight_size, cudaMemcpyHostToDevice, stream);
    }

    cudaMemcpyAsync(global_d_bias, bias.data_ptr<float>(), bias_size, cudaMemcpyHostToDevice, stream);

    // Measure kernel execution time
    auto start = std::chrono::high_resolution_clock::now();
    launch_conv2d_naive(global_d_input, global_d_weight, global_d_bias, global_d_output,
                        N, C, H, W, K, R, S, P, Q, use_const_mem, stream);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(end - start).count();

    // Copy result back to host
    cudaMemcpyAsync(output.data_ptr<float>(), global_d_output, output_size, cudaMemcpyDeviceToHost, stream);

    // Clean up stream
    cudaStreamDestroy(stream);
}

/**
 * @brief Runs the uncoalesced (baseline) CUDA Conv2D kernel.
 * 
 * Allocates and frees device memory for every layer (no reuse).
 * This function is intentionally inefficient for performance benchmarking comparison.
 * 
 * @param input     Input tensor of shape [N, C, H, W]
 * @param weight    Weights tensor of shape [K, C, R, S]
 * @param bias      Bias tensor of shape [K]
 * @param output    Output tensor to store result [N, K, P, Q]
 * @param duration  (Output) Measured kernel execution time in milliseconds
 */
void run_convnc_gpu(const torch::Tensor& input, const torch::Tensor& weight,
                    const torch::Tensor& bias, torch::Tensor& output, float& duration)
{
    // Extract tensor dimensions
    int N = input.size(0), C = input.size(1), H = input.size(2), W = input.size(3);
    int K = weight.size(0), R = weight.size(2), S = weight.size(3);
    int P = output.size(2), Q = output.size(3);

    // Allocate temporary device memory (freed after kernel)
    size_t input_size = N * C * H * W * sizeof(float);
    size_t weight_size = K * C * R * S * sizeof(float);
    size_t bias_size = K * sizeof(float);
    size_t output_size = N * K * P * Q * sizeof(float);

    float *d_input, *d_weight, *d_bias, *d_output;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_weight, weight_size);
    cudaMalloc(&d_bias, bias_size);
    cudaMalloc(&d_output, output_size);

    // Copy all data to device
    cudaMemcpy(d_input, input.data_ptr<float>(), input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight.data_ptr<float>(), weight_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias.data_ptr<float>(), bias_size, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Launch the uncoalesced (inefficient) kernel
    auto start = std::chrono::high_resolution_clock::now();
    launch_conv2d_uncoalesced(d_input, d_weight, d_bias, d_output,
                              N, C, H, W, K, R, S, P, Q, stream);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<float, std::milli>(end - start).count();

    // Copy output back to host
    cudaMemcpy(output.data_ptr<float>(), d_output, output_size, cudaMemcpyDeviceToHost);

    // Free memory (not reused for benchmarking purposes)
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}

