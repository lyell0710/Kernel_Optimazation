// GEMM 版本梯 bench:v0 naive → v1 tile → v2 wmma → v3 双缓冲 → v4 大tile
// 对照 = 真 cuBLAS(cublasGemmEx)。正确性 = 与 cuBLAS 输出的最大相对误差。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <iomanip>
#include <vector>
#include <ctime>
#include <cuda_runtime.h>
#include "gemm_common.h"


// 真实内核驱动版本(/proc/driver/nvidia/version)。cudaDriverGetVersion 返回的是
// CUDA driver-API 版本,曾被误填进 provenance 的 driver= 字段(勘误见
// project-proof/data/manifest.txt);现 driver=真实驱动,CUDA 版本另立 cuda= 字段。
static const char* nvidia_driver_version() {
    static char buf[64] = "unknown";
    std::ifstream f("/proc/driver/nvidia/version");
    std::string tok;
    while (f >> tok) {
        if (tok.find('.') == std::string::npos) continue;
        bool ok = true;
        for (char c : tok)
            if (!(c >= '0' && c <= '9') && c != '.') { ok = false; break; }
        if (ok && tok.size() < sizeof(buf)) {
            snprintf(buf, sizeof buf, "%s", tok.c_str());
            break;
        }
    }
    return buf;
}

using Fn = void (*)(const half*, const half*, half*, int, int, int);

int main() {
    const int M = 4096, N = 4096, K = 4096;
    int iters = 50;
    if (const char* e = std::getenv("BENCH_ITERS")) iters = atoi(e);

    std::vector<half> hA(size_t(M) * K), hB(size_t(K) * N);
    srand(42);
    for (auto& x : hA) x = __float2half((rand() / float(RAND_MAX) - 0.5f) * 2);
    for (auto& x : hB) x = __float2half((rand() / float(RAND_MAX) - 0.5f) * 2);

    half *A, *B, *C, *Ref;
    cudaMalloc(&A, hA.size() * 2); cudaMalloc(&B, hB.size() * 2);
    cudaMalloc(&C, size_t(M) * N * 2); cudaMalloc(&Ref, size_t(M) * N * 2);
    cudaMemcpy(A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(B, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice);

    gemm_cublas(A, B, Ref, M, N, K); cudaDeviceSynchronize();
    std::vector<half> href(size_t(M) * N);
    cudaMemcpy(href.data(), Ref, href.size() * 2, cudaMemcpyDeviceToHost);
    float ref_absmax = 0;
    for (size_t i = 0; i < href.size(); i += 997)
        ref_absmax = fmaxf(ref_absmax, fabsf(__half2float(href[i])));

    struct { const char* name; Fn fn; } vs[] = {
        {"v0", gemm_v0}, {"v1", gemm_v1}, {"v2_wmma", gemm_v2},
        {"v3_dbuf", gemm_v3}, {"v4_bigtile", gemm_v4}, {"cublas", gemm_cublas}};

    const char* outp = std::getenv("BENCH_OUT");          // CORE 铁律5:UTC 前缀新文件
    std::ofstream csv(outp ? outp : "project-proof/data/benchmark_results.csv",
                      std::ios::trunc);
    {   // CORE 铁律4:首行 provenance
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
        int drv = 0; cudaDriverGetVersion(&drv);
        char ts[32]; time_t t = time(nullptr);
        strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t));
        const char* sha = std::getenv("GIT_SHA");
        csv << "# provenance: env=4090-container sha=" << (sha ? sha : "unknown")
            << " cmd=\"BENCH_ITERS=" << iters << " gemm_bench\" date=" << ts
            << " gpu=\"" << prop.name << "\" driver=" << nvidia_driver_version()
            << " cuda=" << drv / 1000 << "." << drv % 1000 / 10 << "\n";
    }
    csv << "version,m,n,k,latency_ms,tflops,speedup_vs_v0,max_rel_err,correctness_pass\n";
    const double flops = 2.0 * M * N * K;
    float v0_ms = 0;
    std::vector<half> hout(size_t(M) * N);

    for (auto& v : vs) {
        v.fn(A, B, C, M, N, K); cudaDeviceSynchronize();       // 预热+出数
        cudaMemcpy(hout.data(), C, hout.size() * 2, cudaMemcpyDeviceToHost);
        float maxrel = 0;
        for (size_t i = 0; i < hout.size(); i += 97) {
            float d = fabsf(__half2float(hout[i]) - __half2float(href[i]));
            maxrel = fmaxf(maxrel, d / ref_absmax);
        }
        bool pass = maxrel < 2e-2f;
        int it = (strcmp(v.name, "v0") == 0 || strcmp(v.name, "v1") == 0)
                     ? std::max(3, iters / 10) : iters;         // 慢版少跑
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int w = 0; w < 3; ++w) v.fn(A, B, C, M, N, K);
        cudaEventRecord(e0);
        for (int i = 0; i < it; ++i) v.fn(A, B, C, M, N, K);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= it;
        if (strcmp(v.name, "v0") == 0) v0_ms = ms;
        double tf = flops / (ms / 1e3) / 1e12;
        printf("%-10s %9.4f ms  %7.1f TFLOPS  maxrel=%.2e  %s\n",
               v.name, ms, tf, maxrel, pass ? "PASS" : "FAIL");
        csv << v.name << "," << M << "," << N << "," << K << ","
            << std::fixed << std::setprecision(4) << ms << ","
            << std::setprecision(1) << tf << ","
            << std::setprecision(2) << (v0_ms / ms) << ","
            << std::scientific << maxrel << ","
            << (pass ? "true" : "false") << "\n";
    }
    return 0;
}
