// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Custom FP8 GEMM kernel optimized for LLaMA 70B decode workloads
// Target shapes: M=128 (decode batch), variable N and K
// Optimized for memory-bound scenarios on MI355X and MI300X

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include "hip_float8.h"

namespace custom_fp8_gemm {

// Kernel arguments structure - must match ASM kernel
struct __attribute__((packed)) Fp8GemmDecodeArgs {
    void* ptr_C;        // Output: [M, N] BF16
    p2 _pad0;
    void* ptr_A;        // Input A: [M, K] FP8
    p2 _pad1;
    void* ptr_B;        // Input B: [N, K] FP8 (transposed)
    p2 _pad2;
    void* ptr_A_scale;  // A scale: [M] FP32
    p2 _pad3;
    void* ptr_B_scale;  // B scale: [N] FP32
    p2 _pad4;
    void* ptr_bias;     // Optional bias: [N] FP32 (can be nullptr)
    p2 _pad5;
    unsigned int M;
    p3 _pad6;
    unsigned int N;
    p3 _pad7;
    unsigned int K;
    p3 _pad8;
    unsigned int lda;   // Leading dimension of A (= K)
    p3 _pad9;
    unsigned int ldb;   // Leading dimension of B (= K)
    p3 _pad10;
    unsigned int ldc;   // Leading dimension of C (= N * 2 for BF16)
    p3 _pad11;
    unsigned int split_k;
    p3 _pad12;
};

// Tile configuration
constexpr int TILE_M = 128;
constexpr int TILE_N = 128;
constexpr int TILE_K = 128;
constexpr int BLOCK_SIZE = 256;

// Check if kernel is available for current GPU
bool is_fp8_gemm_decode_available() {
    std::string arch = get_gpu_arch();
    return (arch.find("gfx950") != std::string::npos || 
            arch.find("gfx942") != std::string::npos);
}

// Get recommended split_k for given shape
int get_recommended_split_k(int M, int N, int K) {
    // For memory-bound GEMMs with small M, split-K helps utilize more CUs
    uint32_t num_cu = get_num_cu_func();
    
    int grid_m = (M + TILE_M - 1) / TILE_M;
    int grid_n = (N + TILE_N - 1) / TILE_N;
    int base_tiles = grid_m * grid_n;
    
    // Target: at least 2 tiles per CU for good occupancy
    int target_tiles = num_cu * 2;
    
    if (base_tiles >= target_tiles) {
        return 1;  // No split-K needed
    }
    
    // Calculate split_k to achieve target occupancy
    int split_k = (target_tiles + base_tiles - 1) / base_tiles;
    
    // Ensure K is divisible by split_k * TILE_K
    while (split_k > 1 && (K / split_k) < TILE_K) {
        split_k--;
    }
    
    return std::min(split_k, 8);  // Cap at 8
}

// Kernel selector based on GPU architecture and shape
static AiterAsmKernel* get_kernel_for_shape(int M, int N, int K) {
    std::string arch = get_gpu_arch();
    
    // Select .co file based on architecture
    // Files are in fp8_gemm_decode/ subdirectory under AITER_ASM_DIR
    std::string co_file;
    if (arch.find("gfx950") != std::string::npos) {
        co_file = "fp8_gemm_decode/fp8_gemm_decode_128x128_gfx950.co";
    } else if (arch.find("gfx942") != std::string::npos) {
        co_file = "fp8_gemm_decode/fp8_gemm_decode_128x128_gfx942.co";
    } else {
        TORCH_CHECK(false, "fp8_gemm_decode_asm: Unsupported GPU architecture: " + arch);
    }
    
    // Static kernel instances (loaded once)
    static std::unique_ptr<AiterAsmKernel> kernel_gfx950;
    static std::unique_ptr<AiterAsmKernel> kernel_gfx942;
    
    if (arch.find("gfx950") != std::string::npos) {
        if (!kernel_gfx950) {
            kernel_gfx950 = std::make_unique<AiterAsmKernel>(
                "fp8_gemm_decode_128x128", 
                "fp8_gemm_decode/fp8_gemm_decode_128x128_gfx950.co"
            );
        }
        return kernel_gfx950.get();
    } else {
        if (!kernel_gfx942) {
            kernel_gfx942 = std::make_unique<AiterAsmKernel>(
                "fp8_gemm_decode_128x128", 
                "fp8_gemm_decode/fp8_gemm_decode_128x128_gfx942.co"
            );
        }
        return kernel_gfx942.get();
    }
}

torch::Tensor fp8_gemm_decode_asm(
    torch::Tensor& A,           // [M, K] FP8
    torch::Tensor& B,           // [N, K] FP8 (transposed)
    torch::Tensor& A_scale,     // [M] FP32
    torch::Tensor& B_scale,     // [N] FP32
    torch::Tensor& out,         // [M, N] BF16
    int split_k
) {
    // Input validation
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "Output must be contiguous");
    TORCH_CHECK(A_scale.is_contiguous(), "A_scale must be contiguous");
    TORCH_CHECK(B_scale.is_contiguous(), "B_scale must be contiguous");
    
    TORCH_CHECK(out.dtype() == torch::ScalarType::BFloat16,
                "fp8_gemm_decode_asm only supports BFloat16 output");
    TORCH_CHECK(A.dtype() == torch_fp8, "A must be FP8");
    TORCH_CHECK(B.dtype() == torch_fp8, "B must be FP8");
    TORCH_CHECK(A_scale.dtype() == torch::kFloat32, "A_scale must be FP32");
    TORCH_CHECK(B_scale.dtype() == torch::kFloat32, "B_scale must be FP32");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    
    TORCH_CHECK(B.size(1) == K, "K dimension mismatch between A and B");
    TORCH_CHECK(out.size(0) == M && out.size(1) == N, "Output shape mismatch");
    TORCH_CHECK(A_scale.size(0) == M, "A_scale must have M elements");
    TORCH_CHECK(B_scale.size(0) == N, "B_scale must have N elements");
    
    // Shape constraints for the kernel
    TORCH_CHECK(N % TILE_N == 0, "N must be divisible by ", TILE_N);
    TORCH_CHECK(K % TILE_K == 0, "K must be divisible by ", TILE_K);
    TORCH_CHECK(M >= 16, "M must be at least 16");
    TORCH_CHECK(K >= 128, "K must be at least 128");
    
    // Auto-select split_k if not specified
    if (split_k <= 0) {
        split_k = get_recommended_split_k(M, N, K);
    }
    
    // Validate split_k
    int k_per_split = (K + split_k - 1) / split_k;
    int k_per_split_aligned = ((k_per_split + TILE_K - 1) / TILE_K) * TILE_K;
    int actual_split_k = (K + k_per_split_aligned - 1) / k_per_split_aligned;
    if (actual_split_k != split_k) {
        split_k = actual_split_k;
    }
    
    // Zero output for split-K accumulation
    if (split_k > 1) {
        out.zero_();
    }
    
    // Build kernel arguments
    Fp8GemmDecodeArgs args;
    args.ptr_C = out.data_ptr();
    args.ptr_A = A.data_ptr();
    args.ptr_B = B.data_ptr();
    args.ptr_A_scale = A_scale.data_ptr();
    args.ptr_B_scale = B_scale.data_ptr();
    args.ptr_bias = nullptr;  // No bias support yet
    args.M = M;
    args.N = N;
    args.K = K;
    args.lda = K;
    args.ldb = K;
    args.ldc = N * 2;  // BF16 = 2 bytes
    args.split_k = split_k;
    
    size_t args_size = sizeof(args);
    
    // Calculate grid dimensions
    int grid_m = (M + TILE_M - 1) / TILE_M;
    int grid_n = (N + TILE_N - 1) / TILE_N;
    int grid_x = grid_n * split_k;
    int grid_y = grid_m;
    
    // Get stream and device guard
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(A));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Get and launch kernel
    AiterAsmKernel* kernel = get_kernel_for_shape(M, N, K);
    
    kernel->launch_kernel({
        &args,
        &args_size,
        grid_x,     // gdx
        grid_y,     // gdy
        1,          // gdz
        BLOCK_SIZE, // bdx
        1,          // bdy
        1,          // bdz
        stream
    });
    
    return out;
}

} // namespace custom_fp8_gemm

// Export functions for the header
using namespace custom_fp8_gemm;

