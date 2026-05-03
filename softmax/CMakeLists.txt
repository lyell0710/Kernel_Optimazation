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
)
