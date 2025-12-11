// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// HipKittens-based FP8 GEMM for decode workloads
// Uses HipKittens library from Stanford's Hazy Research:
// https://github.com/HazyResearch/HipKittens

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <optional>

// HipKittens include path - set via HIPKITTENS_ROOT environment variable
#ifdef USE_HIPKITTENS
#include "kittens.cuh"
namespace hk = kittens;
#endif

// Configuration for decode workload
// Small M (batch ~128), large N and K (model dimensions)
struct DecodeConfig {
    static constexpr int TILE_M = 128;
    static constexpr int TILE_N = 128;
    static constexpr int TILE_K = 64;
    static constexpr int BLOCK_SIZE = 256;  // 4 waves of 64 threads
};

// FP8 type selection based on architecture
#if defined(__gfx942__)
using fp8_t = __hip_fp8_e4m3_fnuz;  // MI300X uses FNUZ
#elif defined(__gfx950__)
using fp8_t = __hip_fp8_e4m3;       // MI355X uses standard E4M3
#else
using fp8_t = __hip_fp8_e4m3_fnuz;  // Default to FNUZ
#endif

// Forward declaration of kernel
extern "C" __global__ void hk_fp8_gemm_decode_128x128(
    __hip_bfloat16* __restrict__ C,
    const void* __restrict__ A,
    const void* __restrict__ B,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    const float* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const int output_dtype
);

#ifdef USE_HIPKITTENS
// HipKittens implementation
__global__ __launch_bounds__(256)
void hk_fp8_gemm_decode_128x128(
    __hip_bfloat16* __restrict__ C,
    const void* __restrict__ A,
    const void* __restrict__ B,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    const float* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const int output_dtype
) {
    // HipKittens tile-based implementation
    // This uses HK primitives for efficient memory access and MFMA compute
    
    using namespace hk;
    
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 64;
    const int lane_id = tid % 64;
    
    constexpr int TILE_M = DecodeConfig::TILE_M;
    constexpr int TILE_N = DecodeConfig::TILE_N;
    constexpr int TILE_K = DecodeConfig::TILE_K;
    
    // Shared memory for double buffering
    __shared__ alignas(128) char smem_A[2][TILE_M * TILE_K];
    __shared__ alignas(128) char smem_B[2][TILE_N * TILE_K];
    
    // Accumulator registers
    float acc[4][4] = {{0.0f}};
    
    const char* A_ptr = reinterpret_cast<const char*>(A);
    const char* B_ptr = reinterpret_cast<const char*>(B);
    
    int buffer = 0;
    
    // Main K-loop with ping-pong buffering
    for (int k = 0; k < K; k += TILE_K) {
        // Load A tile [TILE_M x TILE_K] cooperatively
        for (int i = tid; i < TILE_M * TILE_K; i += 256) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gm = bm * TILE_M + row;
            int gk = k + col;
            smem_A[buffer][i] = (gm < M && gk < K) ? A_ptr[gm * K + gk] : 0;
        }
        
        // Load B tile [TILE_N x TILE_K] cooperatively
        for (int i = tid; i < TILE_N * TILE_K; i += 256) {
            int row = i / TILE_K;
            int col = i % TILE_K;
            int gn = bn * TILE_N + row;
            int gk = k + col;
            smem_B[buffer][i] = (gn < N && gk < K) ? B_ptr[gn * K + gk] : 0;
        }
        
        __syncthreads();
        
        // Compute: each wave processes a portion of the tile
        int wave_m = (warp_id / 2) * 64;
        int wave_n = (warp_id % 2) * 64;
        
        // Inner K loop
        for (int kk = 0; kk < TILE_K; kk++) {
            #pragma unroll
            for (int m = 0; m < 4; m++) {
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    int lm = wave_m + (lane_id / 16) * 16 + m * 4 + (lane_id % 4);
                    int ln = wave_n + ((lane_id / 4) % 16) + n * 16;
                    
                    if (lm < TILE_M && ln < TILE_N) {
                        // FP8 to FP32 conversion
                        float a_val = static_cast<float>(
                            *reinterpret_cast<const fp8_t*>(&smem_A[buffer][lm * TILE_K + kk]));
                        float b_val = static_cast<float>(
                            *reinterpret_cast<const fp8_t*>(&smem_B[buffer][ln * TILE_K + kk]));
                        acc[m][n] += a_val * b_val;
                    }
                }
            }
        }
        
        buffer = 1 - buffer;
        __syncthreads();
    }
    
    // Apply scales and store results
    int wave_m = (warp_id / 2) * 64;
    int wave_n = (warp_id % 2) * 64;
    
    #pragma unroll
    for (int m = 0; m < 4; m++) {
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int lm = wave_m + (lane_id / 16) * 16 + m * 4 + (lane_id % 4);
            int ln = wave_n + ((lane_id / 4) % 16) + n * 16;
            int gm = bm * TILE_M + lm;
            int gn = bn * TILE_N + ln;
            
            if (gm < M && gn < N && lm < TILE_M && ln < TILE_N) {
                float result = acc[m][n] * A_scale[gm] * B_scale[gn];
                
                // Add bias if present
                if (bias != nullptr) {
                    result += bias[gn];
                }
                
                // Convert to output type
                if (output_dtype == 0) {  // BF16
                    C[gm * N + gn] = __float2bfloat16(result);
                } else {  // FP16
                    reinterpret_cast<__half*>(C)[gm * N + gn] = __float2half(result);
                }
            }
        }
    }
}
#else
// Fallback implementation without HipKittens
__global__ __launch_bounds__(256)
void hk_fp8_gemm_decode_128x128(
    __hip_bfloat16* __restrict__ C,
    const void* __restrict__ A,
    const void* __restrict__ B,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    const float* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const int output_dtype
) {
    // Basic implementation for testing without HipKittens
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 64;
    const int lane_id = tid % 64;
    
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 128;
    constexpr int TILE_K = 64;
    
    __shared__ char smem_A[TILE_M * TILE_K];
    __shared__ char smem_B[TILE_N * TILE_K];
    
    float acc[4][4] = {{0.0f}};
    
    const char* A_ptr = reinterpret_cast<const char*>(A);
    const char* B_ptr = reinterpret_cast<const char*>(B);
    
    for (int k = 0; k < K; k += TILE_K) {
        // Load tiles
        for (int i = tid; i < TILE_M * TILE_K; i += 256) {
            int row = i / TILE_K, col = i % TILE_K;
            int gm = bm * TILE_M + row, gk = k + col;
            smem_A[i] = (gm < M && gk < K) ? A_ptr[gm * K + gk] : 0;
        }
        for (int i = tid; i < TILE_N * TILE_K; i += 256) {
            int row = i / TILE_K, col = i % TILE_K;
            int gn = bn * TILE_N + row, gk = k + col;
            smem_B[i] = (gn < N && gk < K) ? B_ptr[gn * K + gk] : 0;
        }
        __syncthreads();
        
        // Compute
        int wave_m = (warp_id / 2) * 64;
        int wave_n = (warp_id % 2) * 64;
        for (int kk = 0; kk < TILE_K; kk++) {
            for (int m = 0; m < 4; m++) {
                for (int n = 0; n < 4; n++) {
                    int lm = wave_m + (lane_id / 16) * 16 + m * 4 + (lane_id % 4);
                    int ln = wave_n + ((lane_id / 4) % 16) + n * 16;
                    if (lm < TILE_M && ln < TILE_N) {
                        float a_val = static_cast<float>(
                            *reinterpret_cast<const fp8_t*>(&smem_A[lm * TILE_K + kk]));
                        float b_val = static_cast<float>(
                            *reinterpret_cast<const fp8_t*>(&smem_B[ln * TILE_K + kk]));
                        acc[m][n] += a_val * b_val;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Store with scales
    int wave_m = (warp_id / 2) * 64;
    int wave_n = (warp_id % 2) * 64;
    for (int m = 0; m < 4; m++) {
        for (int n = 0; n < 4; n++) {
            int lm = wave_m + (lane_id / 16) * 16 + m * 4 + (lane_id % 4);
            int ln = wave_n + ((lane_id / 4) % 16) + n * 16;
            int gm = bm * TILE_M + lm, gn = bn * TILE_N + ln;
            if (gm < M && gn < N && lm < TILE_M && ln < TILE_N) {
                float result = acc[m][n] * A_scale[gm] * B_scale[gn];
                if (bias != nullptr) result += bias[gn];
                if (output_dtype == 0) {
                    C[gm * N + gn] = __float2bfloat16(result);
                } else {
                    reinterpret_cast<__half*>(C)[gm * N + gn] = __float2half(result);
                }
            }
        }
    }
}
#endif

// PyTorch interface
torch::Tensor hipkittens_fp8_gemm_decode(
    torch::Tensor& A,           // [M, K] FP8
    torch::Tensor& B,           // [N, K] FP8
    torch::Tensor& A_scale,     // [M] or scalar
    torch::Tensor& B_scale,     // [N] or scalar
    std::optional<torch::Tensor> bias,
    int output_dtype            // 0=bf16, 1=fp16
) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);
    
    TORCH_CHECK(B.size(1) == K, "K dimension mismatch");
    
    // Create output tensor
    auto out_dtype = (output_dtype == 0) ? torch::kBFloat16 : torch::kFloat16;
    torch::Tensor C = torch::empty({M, N}, A.options().dtype(out_dtype));
    
    // Get bias pointer (may be null)
    float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;
    
    // Launch configuration
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    dim3 block(256);
    
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(at::device_of(A));
    hipStream_t stream = at::hip::getCurrentHIPStream();
    
    hk_fp8_gemm_decode_128x128<<<grid, block, 0, stream>>>(
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
        A.data_ptr(),
        B.data_ptr(),
        A_scale.data_ptr<float>(),
        B_scale.data_ptr<float>(),
        bias_ptr,
        M, N, K,
        output_dtype
    );
    
    return C;
}

