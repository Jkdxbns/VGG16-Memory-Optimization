#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <fstream>
#include "gpu.h"
#include "utils.h"
#include <cuda_runtime.h>

const std::string MODEL_PATH = "./models/vgg16_scripted.pt";
const std::string LABELS_PATH = "./images/labels.txt";

/**
 * @brief Runs end-to-end inference for VGG16 with CPU, coalesced GPU, and uncoalesced GPU paths.
 * 
 * For each Conv2D layer:
 * - CPU uses TorchScript forward pass
 * - GPU uses a custom coalesced CUDA kernel
 * - GPU (NC) uses an uncoalesced, intentionally inefficient kernel
 * 
 * All other layers use TorchScript forward. Measures runtime and L2 norm error for both kernels.
 * 
 * @param image_path Path to a single input image (preprocessed using ImageNet standards)
 */
void run_vgg16_layers(const std::string& image_path) {
    std::cout << "Starting VGG16 Inference:\n";

    // Preprocess the image
    torch::Tensor input = preprocess_image(image_path);

    // Load the scripted VGG16 model
    torch::jit::script::Module model = torch::jit::load(MODEL_PATH);
    model.eval();

    // Clone tensors for CPU, GPU (optimized), and GPU (non-coalesced)
    torch::Tensor cpu_tensor = input.clone();
    torch::Tensor gpu_tensor = input.clone();
    torch::Tensor gpunc_tensor = input.clone();

    // Track total time for each path
    float total_cpu_time = 0.0f;
    float total_gpu_time = 0.0f;
    float totalnc_gpu_time = 0.0f;

    // Preallocate GPU memory buffers (for optimized GPU path)
    size_t max_input_bytes  = 1 * 512 * 224 * 224 * sizeof(float);
    size_t max_weight_bytes = 512 * 512 * 3 * 3 * sizeof(float);
    size_t max_bias_bytes   = 512 * sizeof(float);
    size_t max_output_bytes = 1 * 512 * 224 * 224 * sizeof(float);

    cudaMalloc(&global_d_input, max_input_bytes);
    cudaMalloc(&global_d_weight, max_weight_bytes);
    cudaMalloc(&global_d_bias, max_bias_bytes);
    cudaMalloc(&global_d_output, max_output_bytes);

    // Lambda to process a block (features/classifier)
    auto process_block = [&](torch::jit::script::Module block, std::string block_name, int& layer_idx) {
        for (const auto& layer : block.named_children()) {
            const auto& name = layer.name;
            auto submod = layer.value;
            std::string type = submod.type()->name()->name();

            if (type == "Conv2d") {
                // Extract weights and bias
                auto weight = submod.attr("weight").toTensor();
                auto bias = submod.attr("bias").toTensor();

                // --- CPU (TorchScript forward) ---
                auto start_cpu = std::chrono::high_resolution_clock::now();
                cpu_tensor = submod.forward({cpu_tensor}).toTensor();
                auto end_cpu = std::chrono::high_resolution_clock::now();
                total_cpu_time += std::chrono::duration<float, std::milli>(end_cpu - start_cpu).count();

                // --- GPU Coalesced (custom kernel) ---
                torch::Tensor gpu_out = torch::zeros({gpu_tensor.size(0), weight.size(0), gpu_tensor.size(2), gpu_tensor.size(3)});
                float time_coal;
                run_conv_gpu(gpu_tensor, weight, bias, gpu_out, time_coal);
                total_gpu_time += time_coal;
                gpu_tensor = gpu_out;

                // --- GPU Non-Coalesced (baseline kernel) ---
                torch::Tensor gpunc_out = torch::zeros_like(gpu_tensor);
                float time_uncoal;
                run_convnc_gpu(gpunc_tensor, weight, bias, gpunc_out, time_uncoal);
                totalnc_gpu_time += time_uncoal;
                gpunc_tensor = gpunc_out;
            } else {
                // Other layers (e.g., ReLU, MaxPool, Linear) — run normally
                cpu_tensor = submod.forward({cpu_tensor}).toTensor();
                gpu_tensor = submod.forward({gpu_tensor}).toTensor();
                gpunc_tensor = submod.forward({gpunc_tensor}).toTensor();
            }

            // L2 error between CPU and each GPU variant
            float l2 = torch::norm(cpu_tensor - gpu_tensor).item<float>();
            float l2nc = torch::norm(cpu_tensor - gpunc_tensor).item<float>();
            //std::cout << "L2Norm (coalesced): " << l2 << "\n";
            //std::cout << "L2Norm (uncoalesced): " << l2nc << "\n";

            layer_idx++;
        }
    };

    int layer_idx = 0;
    process_block(model.attr("features").toModule(), "features", layer_idx);

    // --- AdaptiveAvgPool2d ---
    auto avgpool = model.attr("avgpool").toModule();
    cpu_tensor = avgpool.forward({cpu_tensor}).toTensor();
    gpu_tensor = avgpool.forward({gpu_tensor}).toTensor();
    gpunc_tensor = avgpool.forward({gpunc_tensor}).toTensor();

    // Flatten tensors before classifier
    cpu_tensor = cpu_tensor.view({cpu_tensor.size(0), -1});
    gpu_tensor = gpu_tensor.view({gpu_tensor.size(0), -1});
    gpunc_tensor = gpunc_tensor.view({gpunc_tensor.size(0), -1});

    process_block(model.attr("classifier").toModule(), "classifier", layer_idx);

    // --- Summary Results ---
    float final_l2 = torch::norm(cpu_tensor - gpu_tensor).item<float>();
    float final_l2nc = torch::norm(cpu_tensor - gpunc_tensor).item<float>();

    std::cout << "\n--- Summary ---\n";
    std::cout << "Total CPU Time: " << total_cpu_time << " ms\n";
    std::cout << "Total GPU Time: " << total_gpu_time << " ms\n";
    std::cout << "Total Non-Coalesced GPU Time: " << totalnc_gpu_time << " ms\n";
    //std::cout << "Final L2Norm (coalesced): " << final_l2 << "\n";
    //std::cout << "Final L2Norm (uncoalesced): " << final_l2nc << "\n";

    // --- Final classification results ---
    std::vector<std::string> labels = load_labels(LABELS_PATH);
    int predicted_cpu = cpu_tensor.argmax(1).item<int>();
    int predicted_gpu = gpu_tensor.argmax(1).item<int>();
    int predicted_gpunc = gpunc_tensor.argmax(1).item<int>();

    std::cout << "\n--- Classification Result ---\n";
    std::cout << "CPU Prediction:        class " << predicted_cpu << " → " << labels[predicted_cpu] << "\n";
    std::cout << "GPU (Coalesced) Pred:  class " << predicted_gpu << " → " << labels[predicted_gpu] << "\n";
    std::cout << "GPU (Uncoalesced) Pred: class " << predicted_gpunc << " → " << labels[predicted_gpunc] << "\n";

    // Free device buffers
    cudaFree(global_d_input);
    cudaFree(global_d_weight);
    cudaFree(global_d_bias);
    cudaFree(global_d_output);
}

/**
 * @brief Entry point. Expects the image path as the first command-line argument.
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./conv_test <image_path>\n";
        return -1;
    }
    std::string image_path = argv[1];
    run_vgg16_layers(image_path);
    return 0;
}

