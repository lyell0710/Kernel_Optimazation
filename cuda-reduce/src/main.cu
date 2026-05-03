#include "reduce_common.h"
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>

int main()
{
    const int N = 1 << 24;
    const int kBenchmarkIters = 100;
    const char* kCsvPath = "project-proof/data/benchmark_results.csv";

    // host
    std::vector<float> h_in(N, 1.0f);

    // device
    float* d_in = nullptr;
    float* d_out = nullptr;

    // allocate GPU memory
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    // h2d
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // cpu compute
    float cpu = 0.0f;
    for (int i = 0; i < N; i++)
    {
        cpu += h_in[i];
    }

    // create cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto bench_reduce = [&](const std::string& name, void (*fn)(const float*, float*, int), float& gpu_out, float& mean_ms_out) {
        // warmup
        fn(d_in, d_out, N);
        cudaDeviceSynchronize();

        float total_ms = 0.0f;
        for (int i = 0; i < kBenchmarkIters; ++i)
        {
            cudaEventRecord(start);
            fn(d_in, d_out, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float iter_ms = 0.0f;
            cudaEventElapsedTime(&iter_ms, start, stop);
            total_ms += iter_ms;
        }
        mean_ms_out = total_ms / static_cast<float>(kBenchmarkIters);
        cudaMemcpy(&gpu_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

        std::cout << "[mean over " << kBenchmarkIters << " iters] " << name << ": " << mean_ms_out << " ms" << std::endl;
    };

    float baseline_gpu = 0.0f, baseline_ms = 0.0f;
    float v0_gpu = 0.0f, v0_ms = 0.0f;
    float v1_gpu = 0.0f, v1_ms = 0.0f;
    float v2_gpu = 0.0f, v2_ms = 0.0f;
    float v3_gpu = 0.0f, v3_ms = 0.0f;
    float v4_gpu = 0.0f, v4_ms = 0.0f;
    float v5_gpu = 0.0f, v5_ms = 0.0f;
    float v6_gpu = 0.0f, v6_ms = 0.0f;
    float v7_gpu = 0.0f, v7_ms = 0.0f;

    bench_reduce("baseline", reduce_baseline, baseline_gpu, baseline_ms);
    bench_reduce("v0", reduce_v0, v0_gpu, v0_ms);
    bench_reduce("v1", reduce_v1, v1_gpu, v1_ms);
    bench_reduce("v2", reduce_v2, v2_gpu, v2_ms);
    bench_reduce("v3", reduce_v3, v3_gpu, v3_ms);
    bench_reduce("v4", reduce_v4, v4_gpu, v4_ms);
    bench_reduce("v5", reduce_v5, v5_gpu, v5_ms);
    bench_reduce("v6", reduce_v6, v6_gpu, v6_ms);
    bench_reduce("v7", reduce_v7, v7_gpu, v7_ms);

    // overwrite CSV with the latest averaged benchmark results.
    std::ofstream csv_out(kCsvPath, std::ios::trunc);
    if (!csv_out.is_open())
    {
        std::cerr << "Failed to open CSV for writing: " << kCsvPath << std::endl;
    }
    else
    {
        csv_out << "version,cpu_result,gpu_result,diff,latency_ms,speedup,correctness_pass\n";

        const float baseline_diff = std::fabs(cpu - baseline_gpu);
        const float v0_diff = std::fabs(cpu - v0_gpu);
        const float v1_diff = std::fabs(cpu - v1_gpu);
        const float v2_diff = std::fabs(cpu - v2_gpu);
        const float v3_diff = std::fabs(cpu - v3_gpu);
        const float v4_diff = std::fabs(cpu - v4_gpu);
        const float v5_diff = std::fabs(cpu - v5_gpu);
        const float v6_diff = std::fabs(cpu - v6_gpu);
        const float v7_diff = std::fabs(cpu - v7_gpu);

        auto write_row = [&](const char* version, float gpu_value, float diff, float latency_ms) {
            const float speedup = baseline_ms / latency_ms;
            const bool correct = diff < 1e-4f;
            csv_out << std::scientific << std::setprecision(5) << version << "," << cpu << "," << gpu_value << ",";
            csv_out << std::defaultfloat << diff << ",";
            csv_out << std::fixed << std::setprecision(6) << latency_ms << ",";
            csv_out << std::fixed << std::setprecision(2) << speedup << ",";
            csv_out << (correct ? "true" : "false") << "\n";
        };

        write_row("baseline", baseline_gpu, baseline_diff, baseline_ms);
        write_row("v0", v0_gpu, v0_diff, v0_ms);
        write_row("v1", v1_gpu, v1_diff, v1_ms);
        write_row("v2", v2_gpu, v2_diff, v2_ms);
        write_row("v3", v3_gpu, v3_diff, v3_ms);
        write_row("v4", v4_gpu, v4_diff, v4_ms);
        write_row("v5", v5_gpu, v5_diff, v5_ms);
        write_row("v6", v6_gpu, v6_diff, v6_ms);
        write_row("v7", v7_gpu, v7_diff, v7_ms);

        std::cout << "Updated CSV: " << kCsvPath << std::endl;
    }

    // print
    std::cout << "CPU: " << cpu << std::endl;

    std::cout << "baseline GPU: " << baseline_gpu << std::endl;
    std::cout << "baseline Diff: " << std::fabs(cpu - baseline_gpu) << std::endl;
    std::cout << "[baseline] " << baseline_ms << " ms" << std::endl;

    std::cout << "v0 GPU: " << v0_gpu << std::endl;
    std::cout << "v0 Diff: " << std::fabs(cpu - v0_gpu) << std::endl;
    std::cout << "[v0] " << v0_ms << " ms" << std::endl;

    std::cout << "v1 GPU: " << v1_gpu << std::endl;
    std::cout << "v1 Diff: " << std::fabs(cpu - v1_gpu) << std::endl;
    std::cout << "[v1] " << v1_ms << " ms" << std::endl;

    std::cout << "v2 GPU: " << v2_gpu << std::endl;
    std::cout << "v2 Diff: " << std::fabs(cpu - v2_gpu) << std::endl;
    std::cout << "[v2] " << v2_ms << " ms" << std::endl;

    std::cout << "v3 GPU: " << v3_gpu << std::endl;
    std::cout << "v3 Diff: " << std::fabs(cpu - v3_gpu) << std::endl;
    std::cout << "[v3] " << v3_ms << " ms" << std::endl;

    std::cout << "v4 GPU: " << v4_gpu << std::endl;
    std::cout << "v4 Diff: " << std::fabs(cpu - v4_gpu) << std::endl;
    std::cout << "[v4] " << v4_ms << " ms" << std::endl;

    std::cout << "v5 GPU: " << v5_gpu << std::endl;
    std::cout << "v5 Diff: " << std::fabs(cpu - v5_gpu) << std::endl;
    std::cout << "[v5] " << v5_ms << " ms" << std::endl;

    std::cout << "v6 GPU: " << v6_gpu << std::endl;
    std::cout << "v6 Diff: " << std::fabs(cpu - v6_gpu) << std::endl;
    std::cout << "[v6] " << v6_ms << " ms" << std::endl;

    std::cout << "v7 GPU: " << v7_gpu << std::endl;
    std::cout << "v7 Diff: " << std::fabs(cpu - v7_gpu) << std::endl;
    std::cout << "[v7] " << v7_ms << " ms" << std::endl;

    // free GPU memory
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}