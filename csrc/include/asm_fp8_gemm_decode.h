#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Custom FP8 GEMM kernel optimized for LLaMA 70B decode workloads
// Target: MI355X (gfx950) and MI300X (gfx942)
// Optimized for M=128 (decode batch), memory-bound scenarios

#include <torch/extension.h>

namespace custom_fp8_gemm {

// Main decode GEMM function
// Optimized for shapes like (128, 1280, 8192) - QKV projection
// 
// A: [M, K] FP8 (row-major)
// B: [N, K] FP8 (transposed weight, column-major)
// A_scale: [M] FP32 per-token scale
// B_scale: [N] FP32 per-channel scale
// out: [M, N] BF16 (row-major)
torch::Tensor fp8_gemm_decode_asm(
    torch::Tensor& A,           // [M, K] FP8
    torch::Tensor& B,           // [N, K] FP8 (transposed)
    torch::Tensor& A_scale,     // [M] FP32
    torch::Tensor& B_scale,     // [N] FP32
    torch::Tensor& out,         // [M, N] BF16
    int split_k = 1             // Split-K factor for better CU utilization
);

// Check if the custom kernel is available for current GPU
bool is_fp8_gemm_decode_available();

// Get recommended split_k for given shape
int get_recommended_split_k(int M, int N, int K);

} // namespace custom_fp8_gemm

