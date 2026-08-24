#include <cublas_v2.h>
#include "gemm_common.h"
// 真 cuBLAS 对照(cublasGemmEx, fp16 输入 fp32 计算)。
// 行主序技巧:C_row = A_row·B_row 等价于列主序的 C^T = B^T·A^T,
// 直接以 (N,M,K) 调用并交换 A/B 指针。
static cublasHandle_t handle = nullptr;
void gemm_cublas(const half* A, const half* B, half* C, int M, int N, int K) {
    if (!handle) cublasCreate(&handle);
    const float alpha = 1.f, beta = 0.f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                 &alpha, B, CUDA_R_16F, N, A, CUDA_R_16F, K,
                 &beta, C, CUDA_R_16F, N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
