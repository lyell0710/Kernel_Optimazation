#pragma once

void softmax_baseline(const float* input, float* output, int rows, int cols);
void softmax_v0(const float* input, float* output, int rows, int cols);
void softmax_v1(const float* input, float* output, int rows, int cols);
void softmax_v2(const float* input, float* output, int rows, int cols);
void softmax_v3(const float* input, float* output, int rows, int cols);
void softmax_v4(const float* input, float* output, int rows, int cols);

void cpu_softmax(const float* input, float* output, int rows, int cols);
