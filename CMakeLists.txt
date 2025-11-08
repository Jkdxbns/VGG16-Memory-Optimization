cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0146 NEW)

project(VGG16_Conv0_Benchmark LANGUAGES CXX CUDA)

# Explicit CUDA compiler (optional if already default)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA compiler flags
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --std=c++17 -diag-suppress=20042"
)

# DO NOT try to drive architectures via CMAKE_CUDA_ARCHITECTURES;
# PyTorch ignores it and wants TORCH_CUDA_ARCH_LIST instead.
# set(CMAKE_CUDA_ARCHITECTURES 89)

# Torch path (POINT TO THE CMAKE CONFIG FOLDER)
set(Torch_DIR "/home/jkdxbns/general/lib/python3.10/site-packages/torch/share/cmake/Torch")

# Help CMake find Torch
set(CMAKE_PREFIX_PATH
    "${CMAKE_PREFIX_PATH};/home/jkdxbns/general/lib/python3.10/site-packages/torch"
)

# Tell PyTorch which GPU archs to build for.
# If your GPU is Ada (RTX 40xx, sm_89):
set(TORCH_CUDA_ARCH_LIST "8.9")
# If you have a different GPU, use e.g. "8.6" (30xx), "8.0" (A100), etc.

# Find packages
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

# Include paths
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${TORCH_INCLUDE_DIRS})

# Source files
file(GLOB SRC_CPP src/*.cpp)
file(GLOB SRC_CU src/*.cu)

# Build executable
add_executable(conv_test ${SRC_CPP} ${SRC_CU})

# Link libraries
target_link_libraries(conv_test
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    cuda
    cudart
)

# CUDA build properties
set_property(TARGET conv_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

