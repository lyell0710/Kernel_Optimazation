#pragma once
#include <cuda_fp16.h>
// 行主序:A[M,K] B[K,N] C[M,N],fp16 存储、fp32 累加。
void gemm_v0(const half* A, const half* B, half* C, int M, int N, int K);
void gemm_v1(const half* A, const half* B, half* C, int M, int N, int K);
void gemm_v2(const half* A, const half* B, half* C, int M, int N, int K);
void gemm_v3(const half* A, const half* B, half* C, int M, int N, int K);
void gemm_v4(const half* A, const half* B, half* C, int M, int N, int K);
void gemm_cublas(const half* A, const half* B, half* C, int M, int N, int K);
