#include "utils.h"
#include <torch/script.h>
#include <iostream>
#include <filesystem>  

#define IMG_H 224
#define IMG_W 224
namespace fs = std::filesystem;



/**
 * @brief Loads labels from a file, one per line.
 * 
 * This function is used to map model outputs (class indices) to human-readable labels.
 * 
 * @param path Path to the labels.txt file.
 * @return Vector of strings representing class labels.
 */
std::vector<std::string> load_labels(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(file, line)) labels.push_back(line);
    return labels;
}


/**
 * @brief Preprocesses an image for VGG16-like models.
 * 
 * Steps:
 *  - Load image using OpenCV
 *  - Convert from BGR to RGB (OpenCV default is BGR)
 *  - Resize to 224x224 (VGG16 input requirement)
 *  - Normalize to [0,1]
 *  - Apply ImageNet mean and std normalization
 *  - Convert to Torch tensor in [N, C, H, W] format
 * 
 * @param image_path Full path to the image file
 * @return A Torch tensor of shape [1, 3, 224, 224] ready for inference
 */
torch::Tensor preprocess_image(const std::string& image_path) {
    static const std::vector<float> MEAN = {0.485, 0.456, 0.406}; // ImageNet mean (RGB)
    static const std::vector<float> STD = {0.229, 0.224, 0.225};  // ImageNet std (RGB)
    
    // Load the image
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("Failed to load image: " + image_path);
    if (img.channels() != 3) throw std::runtime_error("Image must have 3 channels: " + image_path);
    
    // Convert BGR to RGB (VGG expects RGB input) but openCV imports in BGR format
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // Resize since our images are 1024x1024 but model expects 224x224
    cv::resize(img, img, cv::Size(IMG_W, IMG_H), 0, 0, cv::INTER_LINEAR);   
    // Convert to float and normalize to [0, 1]
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);
    
    // Normalize per channel (ImageNet normalization)
    std::vector<cv::Mat> channels(3);
    cv::split(img, channels);
    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - MEAN[c]) / STD[c];
    }
    cv::merge(channels, img);
    
    // Convert to Torch tensor and permute to [C, H, W]
    torch::Tensor tensor = torch::from_blob(img.data, {IMG_H, IMG_W, 3}, torch::kFloat32);
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0);     // Add batch dimension → [1, C, H, W]
    return tensor.contiguous();   // Ensure the tensor is memory-contiguous
}

/**
 * @brief Recursively prints a hierarchical summary of a TorchScript model's structure.
 * 
 * This function is useful for debugging or inspecting TorchScript models. It traverses
 * all named child modules and displays:
 *   - The hierarchical layer name (e.g., "features.3")
 *   - The layer type (e.g., "torch.nn.modules.conv.Conv2d")
 *   - The shape of any associated weight and bias tensors
 * 
 */
void print_model_summary(const torch::jit::script::Module& model, const std::string& prefix) {
    for (const auto& child : model.named_children()) {
        std::string name = prefix.empty() ? child.name : prefix + "." + child.name;
        std::cout << name << " -> " << child.value.type()->str() << "\n";

        // Safely try to access parameters
        if (child.value.hasattr("weight")) {
            auto weight = child.value.attr("weight").toTensor();
            std::cout << "  weight shape: " << weight.sizes() << "\n";
        }

        if (child.value.hasattr("bias")) {
            auto bias = child.value.attr("bias").toTensor();
            std::cout << "  bias shape: " << bias.sizes() << "\n";
        }

        print_model_summary(child.value, name);
    }
}

/**
 * @brief Loads a batch of images from a directory for inference.
 * 
 * - Applies the same preprocessing as preprocess_image()
 * - Stops once `batch_size` images are loaded
 * - Returns both the input tensor and filenames for identification
 * 
 * @param dir_path Path to the folder containing .jpg or .png images
 * @param batch_size Number of images to load
 * @return Tuple of:
 *         - Tensor of shape [N, 3, 224, 224]
 *         - Corresponding filenames
 */
 
std::tuple<torch::Tensor, std::vector<std::string>>
load_batch_from_dir(const std::string& dir_path, int batch_size) {
    std::vector<torch::Tensor> batch;
    std::vector<std::string> filenames;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            batch.push_back(preprocess_image(entry.path().string()));
            filenames.push_back(entry.path().filename().string());
            // Stop when batch is full
            if (batch.size() == batch_size) break;
        }
    }
    // Stack into a single batch tensor
    if (batch.empty()) throw std::runtime_error("No images loaded from: " + dir_path);
    return {torch::cat(batch, 0), filenames};  // [N, 3, 224, 224]
}


