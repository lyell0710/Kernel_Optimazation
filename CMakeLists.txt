cmake_minimum_required(VERSION 3.18)
project(cuda_reduce LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

add_executable(reduce_bench
    src/main.cu
    src/reduce_baseline.cu
    src/reduce_naive_v0.cu
    src/reduce_v1.cu
    src/reduce_v2.cu
    src/reduce_reference.cpp
)
