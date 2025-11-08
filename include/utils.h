#pragma once
#include <tuple>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

/**
 * @brief Loads class labels from a text file.
 * 
 * Assumes each line of the file contains one label.
 * 
 * @param path Path to the labels.txt file.
 * @return Vector of label strings.
 */
std::vector<std::string> load_labels(const std::string& path);

/**
 * @brief Preprocesses an image for VGG input.
 * 
 * Steps include:
 * - BGR to RGB conversion
 * - Resize to 224x224
 * - Normalize to [0,1] range
 * - Convert to torch::Tensor in [N, C, H, W] format
 * 
 * @param image_path Path to the input image.
 * @return A Torch tensor of shape [1, 3, 224, 224], ready for inference.
 */
torch::Tensor preprocess_image(const std::string& image_path);

/**
 * @brief Prints a hierarchical summary of a TorchScript model's structure.
 * 
 * Recursively prints child module names, types, and structure depth.
 * 
 * @param model TorchScript model to summarize.
 * @param prefix Optional string to prefix each layer name (used internally for nesting).
 */
void print_model_summary(const torch::jit::script::Module& model, const std::string& prefix = "");

/**
 * @brief Loads a batch of images from a directory.
 * 
 * - Images are resized and normalized like `preprocess_image()`
 * - Assumes .png or .jpg images
 * - Uses OpenCV for loading and preprocessing
 * 
 * @param dir_path Path to directory containing images.
 * @param batch_size Number of images to load.
 * @return A tuple of:
 *         - [B, 3, 224, 224] input tensor
 *         - Corresponding image file names or labels
 */
std::tuple<torch::Tensor, std::vector<std::string>> load_batch_from_dir(const std::string& dir_path, int batch_size);

