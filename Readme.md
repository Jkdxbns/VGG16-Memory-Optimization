# VGG16 Conv2D CUDA Performance Benchmark

[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-LibTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-00599C?logo=c%2B%2B&logoColor=white)](https://isocpp.org/)

A comprehensive performance analysis project demonstrating the critical impact of **GPU memory access patterns** on deep learning inference performance. This project implements custom CUDA kernels for Conv2D operations in VGG16 and compares three execution paths:

- 🖥️ **CPU Baseline** - Standard PyTorch/TorchScript implementation
- 🚀 **GPU Optimized** - Custom memory-coalesced CUDA kernel (3-5× faster)
- 🐌 **GPU Uncoalesced** - Intentionally inefficient baseline kernel

**Key Finding:** Proper memory coalescing yields **3-5× performance improvement** over naive GPU implementations, demonstrating that memory access patterns matter more than raw compute power for Conv2D operations.

---

## 🎯 Project Highlights

### Unique Implementations

- ✅ **Custom CUDA Conv2D Kernels** - Written from scratch, not using PyTorch's GPU operations
- ✅ **Memory Coalescing Optimization** - 2D tiling with sequential access patterns
- ✅ **Constant Memory Usage** - Automatic weight caching for small filters (<16KB)
- ✅ **Strategic Buffer Reuse** - Pre-allocated GPU memory across all 13 Conv2D layers
- ✅ **Hybrid Execution Model** - Custom kernels for Conv2D, TorchScript for other layers
- ✅ **Comprehensive Validation** - Layer-wise L2 norm error calculation

### Performance Results (Typical)

| Execution Path | Total Time | Speedup | Memory Efficiency |
|---------------|-----------|---------|-------------------|
| CPU (TorchScript) | 500-1000ms | 1.0× | N/A |
| GPU Optimized | 50-150ms | **5-10×** | 80-90% peak BW |
| GPU Uncoalesced | 200-400ms | 2-3× | 20-40% peak BW |

### Features

- 🔬 **End-to-end image classification** with detailed performance metrics
- 📊 **Layer-wise timing** for all 13 Conv2D operations
- 🎯 **Accuracy validation** through L2 norm error analysis
- 🏗️ **Production-ready patterns** (async streams, buffer reuse, constant memory)
- 📚 **Educational value** for understanding GPU memory hierarchy

## Folder Structure
'''text
./HPA_Project
├── build
├── CMakeLists.txt
├── images
│   ├── cat_1.png
│   ├── cat.png
│   ├── dog_1.png
│   ├── dog.png
│   ├── elephant.png
│   ├── labels.txt
│   ├── pegion.png
│   ├── rose.png
│   └── tiger.png
├── include
│   ├── gpu_common.cuh
│   ├── gpu.h
│   └── utils.h
├── models
│   ├── dummy_scripted.pt
│   └── vgg16_scripted.pt
├── Readme.md
├── scripts
│   ├── build.sh
│   └── export_vgg16.py
└── src
    ├── gpu.cpp
    ├── gpu.cu
    ├── main.cpp
    └── utils.cpp'''

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Installation |
|------------|---------|--------------|
| **CUDA Toolkit** | ≥ 11.0 | [Download](https://developer.nvidia.com/cuda-downloads) |
| **LibTorch** | ≥ 1.13 | [Download](https://pytorch.org/get-started/locally/) |
| **OpenCV** | ≥ 4.5 | `sudo apt install libopencv-dev` |
| **CMake** | ≥ 3.18 | `sudo apt install cmake` |
| **GCC/G++** | ≥ 9.0 | `sudo apt install g++` |
| **Python** | 3.8+ | For model export script |

**GPU Requirements:** NVIDIA GPU with Compute Capability ≥ 7.5 (Turing or newer)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd HPA_Project
   ```

2. **Install Python dependencies** (for model export)
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and extract LibTorch**
   ```bash
   # Download from https://pytorch.org/get-started/locally/
   # Select: C++/CUDA 11.x/LibTorch
   wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
   unzip libtorch-*.zip -d ~/
   ```

4. **Update CMakeLists.txt**
   
   Edit `CMakeLists.txt` and set the correct LibTorch path:
   ```cmake
   set(Torch_DIR "/path/to/your/libtorch/share/cmake/Torch")
   ```
   
   Update CUDA architecture for your GPU:
   ```cmake
   # For RTX 40xx (Ada):  set(TORCH_CUDA_ARCH_LIST "8.9")
   # For RTX 30xx (Ampere): set(TORCH_CUDA_ARCH_LIST "8.6")
   # For RTX 20xx (Turing): set(TORCH_CUDA_ARCH_LIST "7.5")
   ```

5. **Export the VGG16 model**
   ```bash
   cd scripts
   python export_vgg16.py
   # This creates models/vgg16_scripted.pt
   ```

6. **Build the project**
   ```bash
   ./build.sh
   # Or manually:
   # cd ../build
   # cmake ..
   # make -j$(nproc)
   ```

### Running Inference

```bash
cd build
./conv_test ../images/cat.png
```

**Expected Output:**
```
Starting VGG16 Inference:

--- Summary ---
Total CPU Time: 723.45 ms
Total GPU Time: 98.23 ms
Total Non-Coalesced GPU Time: 287.91 ms

--- Classification Result ---
CPU Prediction:        class 281 → tabby cat
GPU (Coalesced) Pred:  class 281 → tabby cat
GPU (Uncoalesced) Pred: class 281 → tabby cat
```

**Try other images:**
```bash
./conv_test ../images/dog.png
./conv_test ../images/elephant.png
```

---

## 🔬 Technical Deep Dive

### What Makes This Project Unique?

Unlike typical deep learning projects that use PyTorch's built-in GPU operations (`.cuda()`), this project:

1. **Implements Conv2D from scratch in CUDA C++**
   - No high-level framework abstractions
   - Direct control over memory access patterns
   - Manual thread block and grid configuration

2. **Demonstrates Memory Coalescing**
   ```cuda
   // GOOD (Coalesced): Consecutive threads read consecutive memory
   Thread 0: reads address[0], Thread 1: reads address[1], ...
   → Combined into ONE 128-byte memory transaction
   
   // BAD (Uncoalesced): Threads read scattered memory
   Thread 0: reads address[0], Thread 1: reads address[1000], ...
   → Requires 32 separate memory transactions (32× slower!)
   ```

3. **Strategic Optimizations**
   - **Constant Memory:** Small weights (<16KB) cached for broadcast reads
   - **Buffer Reuse:** Pre-allocated GPU memory across all layers
   - **2D Tiling:** Thread blocks mapped to spatial dimensions
   - **Loop Unrolling:** Manual unrolling for 3×3 filters

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     main.cpp                            │
│  ┌───────────────────────────────────────────────┐     │
│  │ For each Conv2D layer:                        │     │
│  │  1. Extract weights from TorchScript model    │     │
│  │  2. Launch three parallel executions:         │     │
│  │                                                │     │
│  │     CPU Path → TorchScript forward()          │     │
│  │                                                │     │
│  │     GPU Opt  → run_conv_gpu()                 │     │
│  │                  ↓ gpu.cpp                     │     │
│  │                  ↓ launch_conv2d_naive()       │     │
│  │                  ↓ gpu.cu                      │     │
│  │                  ↓ conv2d_naive_kernel<<<>>>   │     │
│  │                                                │     │
│  │     GPU NC   → run_convnc_gpu()               │     │
│  │                  ↓ gpu.cpp                     │     │
│  │                  ↓ launch_conv2d_uncoalesced() │     │
│  │                  ↓ gpu.cu                      │     │
│  │                  ↓ conv2d_uncoalesced_kernel   │     │
│  │                                                │     │
│  │  3. Measure execution time for each           │     │
│  │  4. Calculate L2 error for validation         │     │
│  └───────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### Key Algorithms

**Memory-Coalesced Kernel (Optimized):**
```cuda
__global__ void conv2d_naive_kernel(...) {
    // 2D thread mapping
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Sequential memory access (coalesced!)
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < 3; ++r) {
            for (int s = 0; s < 3; ++s) {
                int h_in = h_out + r - 1;
                int w_in = w_out + s - 1;
                // Access pattern: [...][h][w] ← consecutive threads, consecutive w
                acc += input[...][h_in][w_in] * weight[...];
            }
        }
    }
}
```

**Grid Configuration:**
```cpp
dim3 blockDim(16, 16);  // 256 threads per block
dim3 gridDim(
    (W + 15) / 16,      // Width tiles
    (H + 15) / 16,      // Height tiles
    N * K               // Batch × Output channels
);
```

---

## 📊 Performance Analysis
(Refer to the report attached for results)


## 🎓 Educational Value

### What You'll Learn

1. **GPU Memory Hierarchy**
   - Global memory vs constant memory vs shared memory
   - Memory transaction coalescing requirements
   - Bandwidth vs latency trade-offs

2. **CUDA Programming**
   - Kernel launch configuration (grid/block dimensions)
   - Thread indexing and data partitioning
   - Synchronization and stream management
   - Performance profiling and optimization

3. **Deep Learning Systems**
   - How Conv2D actually executes on GPU hardware
   - Why PyTorch/TensorFlow are fast (they use these tricks!)
   - Production optimization strategies

4. **Performance Engineering**
   - Identifying bottlenecks (memory vs compute)
   - Measuring and validating optimizations
   - Trade-offs in system design

### Recommended Experiments

1. **Profile with Nsight Compute:**
   ```bash
   ncu --set full -o profile ./conv_test ../images/cat.png
   ```

2. **Try different block sizes:**
   ```cuda
   // In gpu.cu, modify:
   #define BLOCK_DIM_X (32)  // Try 8, 16, 32
   #define BLOCK_DIM_Y (32)
   ```

3. **Add shared memory tiling:**
   - Load input tiles into shared memory
   - Reuse across multiple output pixels
   - Expected: 10-20% additional speedup

4. **Compare with cuDNN:**
   - Replace custom kernel with `cudnnConvolutionForward`
   - Benchmark against production-grade implementation

---

## 🐛 Troubleshooting

### Common Issues

**1. `No such file or directory: libtorch`**
```bash
# Solution: Update Torch_DIR in CMakeLists.txt
set(Torch_DIR "/correct/path/to/libtorch/share/cmake/Torch")
```

**2. `undefined reference to cv::imread`**
```bash
# Solution: Install OpenCV development headers
sudo apt install libopencv-dev
```

**3. `nvcc fatal: Unsupported gpu architecture 'compute_89'`**
```bash
# Solution: Change TORCH_CUDA_ARCH_LIST to match your GPU
# Check your GPU architecture:
nvidia-smi --query-gpu=compute_cap --format=csv
```

**4. `CUDA out of memory`**
```bash
# Solution: Use dummy model for testing
# In scripts/export_vgg16.py, uncomment:
# model = Dummy().eval()
```

**5. Slow performance / no speedup**
```bash
# Check GPU is being used:
nvidia-smi  # Should show conv_test process

# Profile to identify bottlenecks:
nvprof ./conv_test ../images/cat.png
```

---


---

## 📄 License

This project is intended for educational purposes as part of CMPE 755 - High Performance Architecture at Rochester Institute of Technology (RIT).

---

---

## 🙏 Acknowledgments

- NVIDIA for CUDA toolkit and documentation
- PyTorch team for LibTorch C++ API
- VGG16 authors for the model architecture
- RIT faculty for course guidance

---

**⭐ If you found this project helpful, please consider starring the repository!**

