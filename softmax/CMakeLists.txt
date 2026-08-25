cmake_minimum_required(VERSION 3.18)
project(cuda_softmax LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

include_directories(include)

add_executable(softmax_bench
    src/main.cu
    src/softmax_baseline.cu
    src/softmax_v0.cu
    src/softmax_v1.cu
    src/softmax_v2.cu
    src/softmax_v3.cu
    src/softmax_v4.cu
    src/softmax_cudnn.cu
    src/softmax_v4_2.cu
    src/softmax_v4_3.cu
    src/softmax_v4_4.cu
    src/softmax_cublas.cu
)

target_link_libraries(softmax_bench PRIVATE cublas)

# 标准库基准 cuDNN:头文件取自 venv 内 nvidia-cudnn 包,运行库为系统 libcudnn.so.9
target_include_directories(softmax_bench PRIVATE
    /root/venvs/v0.25.1/lib/python3.12/site-packages/nvidia/cudnn/include)
# 系统只提供带版本号的 .so(无 libcudnn.so 符号链),直接按全路径链接
find_library(CUDNN_LIB NAMES libcudnn.so.9 cudnn
             PATHS /usr/lib/x86_64-linux-gnu REQUIRED)
target_link_libraries(softmax_bench PRIVATE ${CUDNN_LIB})
