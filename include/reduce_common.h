#pragma once

void reduce_baseline(const float* data, float* output, int n);
void reduce_v0(const float* data, float* output, int n);
void reduce_v1(const float* data, float* output, int n);
void reduce_v2(const float* data, float* output, int n);

float cpu_reduce(const float* data, int n);