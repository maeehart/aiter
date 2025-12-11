// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// HipKittens-based FP8 GEMM for decode workloads
// OPTIMIZED FOR MEMORY-BOUND OPERATIONS
//
// Key optimizations for memory-bound decode GEMMs:
// 1. Vectorized memory access (128-bit loads)
// 2. Split-K for increased parallelism with small M
// 3. Async memory operations with double buffering
// 4. Minimized kernel launch overhead
// 5. Architecture-specific tuning (gfx942 vs gfx950)
//
// References:
// - HipKittens: https://github.com/HazyResearch/HipKittens
// - ParallelKittens: https://hazyresearch.stanford.edu/blog/2025-11-17-pk

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <optional>

// HipKittens include path
#ifdef USE_HIPKITTENS
#include "kittens.cuh"
namespace hk = kittens;
#endif

// FP8 type selection based on architecture
#if defined(__gfx950__)
using fp8_t = __hip_fp8_e4m3;       // MI355X uses standard E4M3
#define ARCH_GFX950 1
#else
using fp8_t = __hip_fp8_e4m3_fnuz;  // MI300X/MI325X use FNUZ
#define ARCH_GFX942 1
#endif

// =============================================================================
// Configuration for memory-bound decode GEMMs
// =============================================================================

// Tile sizes optimized for memory bandwidth
// For memory-bound GEMMs: larger K tiles to amortize memory overhead
struct DecodeConfig {
    // Tile dimensions
    static constexpr int TILE_M = 64;   // Smaller M tile for decode
    static constexpr int TILE_N = 128;  // Larger N for better vectorization
    static constexpr int TILE_K = 128;  // Larger K to amortize overhead
    
    // Thread block configuration
    static constexpr int BLOCK_SIZE = 256;  // 4 waves of 64 threads
    static constexpr int WAVES = 4;
    
    // Vectorization (128-bit loads = 16 FP8 values)
    static constexpr int VEC_SIZE = 16;
    
    // Double buffering for async overlap
    static constexpr int NUM_BUFFERS = 2;
};

// Configuration for very small M (M <= 8)
struct TinyMConfig {
    static constexpr int TILE_M = 16;
    static constexpr int TILE_N = 128;
    static constexpr int TILE_K = 256;  // Even larger K for tiny M
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int VEC_SIZE = 16;
};

// =============================================================================
// Vectorized memory access helpers
// =============================================================================

// 128-bit vectorized load for FP8 (16 elements)
__device__ __forceinline__ void vector_load_fp8_128(
    fp8_t* dst,
    const fp8_t* __restrict__ src
) {
    // Use int4 for 128-bit aligned load
    *reinterpret_cast<int4*>(dst) = *reinterpret_cast<const int4*>(src);
}

// 64-bit vectorized load for FP8 (8 elements)
__device__ __forceinline__ void vector_load_fp8_64(
    fp8_t* dst,
    const fp8_t* __restrict__ src
) {
    *reinterpret_cast<int2*>(dst) = *reinterpret_cast<const int2*>(src);
}

// =============================================================================
// Memory-optimized decode kernel (standard M)
// =============================================================================

__global__ __launch_bounds__(256)
void hk_fp8_gemm_decode_memory_opt(
    __hip_bfloat16* __restrict__ C,
    const fp8_t* __restrict__ A,
    const fp8_t* __restrict__ B,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    const float* __restrict__ bias,
    const int M,
    const int N,
    const int K
) {
    constexpr int TILE_M = DecodeConfig::TILE_M;
    constexpr int TILE_N = DecodeConfig::TILE_N;
    constexpr int TILE_K = DecodeConfig::TILE_K;
    constexpr int VEC_SIZE = DecodeConfig::VEC_SIZE;
    
    const int bm = blockIdx.y;
    const int bn = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 64;
    const int lane_id = tid % 64;
    
    // Double-buffered shared memory
    __shared__ alignas(128) fp8_t smem_A[2][TILE_M * TILE_K];
    __shared__ alignas(128) fp8_t smem_B[2][TILE_N * TILE_K];
    
    // Register accumulators
    float acc[2][4] = {{0.0f}};
    
    // Preload scales into registers
    float scale_a_local[TILE_M / 64];  // One per warp's portion
    float scale_b_local[4];  // Cached B scales
    
    int buffer = 0;
    
    // First tile load (async-like pattern)
    {
        // Vectorized load of A tile
        const int a_elements = TILE_M * TILE_K;
        const int a_per_thread = a_elements / 256;
        
        #pragma unroll
        for (int i = 0; i < a_per_thread; i += VEC_SIZE) {
            int idx = tid * a_per_thread + i;
            int row = idx / TILE_K;
            int col = idx % TILE_K;
            int gm = bm * TILE_M + row;
            int gk = col;
            
            if (gm < M && gk + VEC_SIZE <= K && col + VEC_SIZE <= TILE_K) {
                vector_load_fp8_128(&smem_A[buffer][idx], &A[gm * K + gk]);
            } else {
                // Scalar fallback for boundary
                for (int v = 0; v < VEC_SIZE && (idx + v) < a_elements; v++) {
                    int r = (idx + v) / TILE_K;
                    int c = (idx + v) % TILE_K;
                    int gr = bm * TILE_M + r;
                    smem_A[buffer][idx + v] = (gr < M && c < K) ? A[gr * K + c] : fp8_t(0);
                }
            }
        }
        
        // Vectorized load of B tile
        const int b_elements = TILE_N * TILE_K;
        const int b_per_thread = b_elements / 256;
        
        #pragma unroll
        for (int i = 0; i < b_per_thread; i += VEC_SIZE) {
            int idx = tid * b_per_thread + i;
            int row = idx / TILE_K;
            int col = idx % TILE_K;
            int gn = bn * TILE_N + row;
            int gk = col;
            
            if (gn < N && gk + VEC_SIZE <= K && col + VEC_SIZE <= TILE_K) {
                vector_load_fp8_128(&smem_B[buffer][idx], &B[gn * K + gk]);
            } else {
                for (int v = 0; v < VEC_SIZE && (idx + v) < b_elements; v++) {
                    int r = (idx + v) / TILE_K;
                    int c = (idx + v) % TILE_K;
                    int gr = bn * TILE_N + r;
                    smem_B[buffer][idx + v] = (gr < N && c < K) ? B[gr * K + c] : fp8_t(0);
                }
            }
        }
    }
    
    __syncthreads();
    
    // Main K-loop with double buffering
    for (int k = 0; k < K; k += TILE_K) {
        int next_k = k + TILE_K;
        int next_buffer = 1 - buffer;
        
        // Async prefetch next tile while computing current
        if (next_k < K) {
            // Prefetch A
            const int a_elements = TILE_M * TILE_K;
            const int a_per_thread = a_elements / 256;
            
            #pragma unroll 2
            for (int i = 0; i < a_per_thread; i += VEC_SIZE) {
                int idx = tid * a_per_thread + i;
                int row = idx / TILE_K;
                int col = idx % TILE_K;
                int gm = bm * TILE_M + row;
                int gk = next_k + col;
                
                if (gm < M && gk < K && col + VEC_SIZE <= TILE_K) {
                    vector_load_fp8_128(&smem_A[next_buffer][idx], &A[gm * K + gk]);
                }
            }
            
            // Prefetch B
            const int b_elements = TILE_N * TILE_K;
            const int b_per_thread = b_elements / 256;
            
            #pragma unroll 2
            for (int i = 0; i < b_per_thread; i += VEC_SIZE) {
                int idx = tid * b_per_thread + i;
                int row = idx / TILE_K;
                int col = idx % TILE_K;
                int gn = bn * TILE_N + row;
                int gk = next_k + col;
                
                if (gn < N && gk < K && col + VEC_SIZE <= TILE_K) {
                    vector_load_fp8_128(&smem_B[next_buffer][idx], &B[gn * K + gk]);
                }
            }
        }
        
        // Compute on current buffer
        // Each wave computes a portion of the output
        int wave_m_start = (warp_id / 2) * (TILE_M / 2);
        int wave_n_start = (warp_id % 2) * (TILE_N / 2);
        
        // Inner loop over K tile
        #pragma unroll 4
        for (int kk = 0; kk < TILE_K; kk++) {
            // Each thread computes multiple output elements
            #pragma unroll
            for (int m = 0; m < 2; m++) {
                #pragma unroll
                for (int n = 0; n < 4; n++) {
                    int lm = wave_m_start + (lane_id / 16) * 8 + m * 4 + (lane_id % 4);
                    int ln = wave_n_start + ((lane_id / 4) % 8) * 8 + n * 2 + ((lane_id % 4) / 2);
                    
                    if (lm < TILE_M && ln < TILE_N) {
                        float a_val = static_cast<float>(smem_A[buffer][lm * TILE_K + kk]);
                        float b_val = static_cast<float>(smem_B[buffer][ln * TILE_K + kk]);
                        acc[m][n] += a_val * b_val;
                    }
                }
            }
        }
        
        buffer = next_buffer;
        __syncthreads();
    }
    
    // Apply scales and store results
    int wave_m_start = (warp_id / 2) * (TILE_M / 2);
    int wave_n_start = (warp_id % 2) * (TILE_N / 2);
    
    #pragma unroll
    for (int m = 0; m < 2; m++) {
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            int lm = wave_m_start + (lane_id / 16) * 8 + m * 4 + (lane_id % 4);
            int ln = wave_n_start + ((lane_id / 4) % 8) * 8 + n * 2 + ((lane_id % 4) / 2);
            int gm = bm * TILE_M + lm;
            int gn = bn * TILE_N + ln;
            
            if (gm < M && gn < N && lm < TILE_M && ln < TILE_N) {
                float result = acc[m][n] * A_scale[gm] * B_scale[gn];
                
                if (bias != nullptr) {
                    result += bias[gn];
                }
                
                C[gm * N + gn] = __float2bfloat16(result);
            }
        }
    }
}

// =============================================================================
// Kernel for very small M (M <= 8) - uses Split-K for parallelism
// =============================================================================

__global__ __launch_bounds__(256)
void hk_fp8_gemm_decode_tiny_m(
    __hip_bfloat16* __restrict__ C,
    const fp8_t* __restrict__ A,
    const fp8_t* __restrict__ B,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    const float* __restrict__ bias,
    const int M,
    const int N,
    const int K,
    const int split_k
) {
    constexpr int TILE_N = TinyMConfig::TILE_N;
    constexpr int TILE_K = TinyMConfig::TILE_K;
    
    // For tiny M, we use split-K: each block handles K/split_k elements
    const int bn = blockIdx.x;
    const int k_split_idx = blockIdx.y;
    const int tid = threadIdx.x;
    
    const int k_start = k_split_idx * (K / split_k);
    const int k_end = (k_split_idx == split_k - 1) ? K : (k_split_idx + 1) * (K / split_k);
    
    __shared__ alignas(128) fp8_t smem_A[16 * TILE_K];  // Small M, large K
    __shared__ alignas(128) fp8_t smem_B[TILE_N * TILE_K];
    
    // Partial accumulators (will need atomic add if split_k > 1)
    float acc[16] = {0.0f};  // One per M row, per N column handled by this thread
    
    for (int k = k_start; k < k_end; k += TILE_K) {
        int k_tile = min(TILE_K, k_end - k);
        
        // Load A (all M rows, TILE_K columns)
        for (int i = tid; i < M * k_tile; i += 256) {
            int row = i / k_tile;
            int col = i % k_tile;
            smem_A[row * TILE_K + col] = (k + col < K) ? A[row * K + k + col] : fp8_t(0);
        }
        
        // Load B
        for (int i = tid; i < TILE_N * k_tile; i += 256) {
            int row = i / k_tile;
            int col = i % k_tile;
            int gn = bn * TILE_N + row;
            smem_B[row * TILE_K + col] = (gn < N && k + col < K) ? B[gn * K + k + col] : fp8_t(0);
        }
        
        __syncthreads();
        
        // Compute: each thread handles multiple (m, n) pairs
        int n_per_thread = TILE_N / 256;
        if (n_per_thread == 0) n_per_thread = 1;
        
        for (int m = 0; m < M; m++) {
            for (int n_off = 0; n_off < n_per_thread; n_off++) {
                int ln = tid * n_per_thread + n_off;
                if (ln < TILE_N) {
                    float sum = 0.0f;
                    #pragma unroll 16
                    for (int kk = 0; kk < k_tile; kk++) {
                        sum += static_cast<float>(smem_A[m * TILE_K + kk]) * 
                               static_cast<float>(smem_B[ln * TILE_K + kk]);
                    }
                    acc[m] += sum;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Store results (with atomic if split_k > 1)
    int n_per_thread = TILE_N / 256;
    if (n_per_thread == 0) n_per_thread = 1;
    
    for (int m = 0; m < M; m++) {
        for (int n_off = 0; n_off < n_per_thread; n_off++) {
            int ln = tid * n_per_thread + n_off;
            int gn = bn * TILE_N + ln;
            
            if (ln < TILE_N && gn < N) {
                float result = acc[m] * A_scale[m] * B_scale[gn];
                if (bias != nullptr) {
                    result += bias[gn];
                }
                
                if (split_k > 1) {
                    // Atomic add for split-K reduction
                    atomicAdd(reinterpret_cast<float*>(&C[m * N + gn]), result);
                } else {
                    C[m * N + gn] = __float2bfloat16(result);
                }
            }
        }
    }
}

// =============================================================================
// Simple fallback kernel (for correctness verification)
// =============================================================================

__global__ __launch_bounds__(256)
void hk_fp8_gemm_decode_simple(
    __hip_bfloat16* __restrict__ C,
    const fp8_t* __restrict__ A,
    const fp8_t* __restrict__ B,
    const float* __restrict__ A_scale,
    const float* __restrict__ B_scale,
    const float* __restrict__ bias,
    const int M,
    const int N,
    const int K
) {
    const int gm = blockIdx.y * 64 + threadIdx.x / 4;
    const int gn = blockIdx.x * 64 + (threadIdx.x % 4) * 16;
    
    if (gm >= M) return;
    
    float acc[16] = {0.0f};
    
    for (int k = 0; k < K; k++) {
        float a_val = static_cast<float>(A[gm * K + k]);
        
        #pragma unroll
        for (int n = 0; n < 16; n++) {
            if (gn + n < N) {
                float b_val = static_cast<float>(B[(gn + n) * K + k]);
                acc[n] += a_val * b_val;
            }
        }
    }
    
    float scale_a = A_scale[gm];
    
    #pragma unroll
    for (int n = 0; n < 16; n++) {
        if (gn + n < N) {
            float result = acc[n] * scale_a * B_scale[gn + n];
            if (bias != nullptr) {
                result += bias[gn + n];
            }
            C[gm * N + gn + n] = __float2bfloat16(result);
        }
    }
}

// =============================================================================
// PyTorch interface
// =============================================================================

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
    
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(at::device_of(A));
    hipStream_t stream = at::hip::getCurrentHIPStream();
    
    // Select kernel based on M size for optimal memory bandwidth utilization
    if (M <= 8) {
        // Very small M: use split-K kernel for parallelism
        int split_k = (K >= 4096) ? 4 : ((K >= 2048) ? 2 : 1);
        
        dim3 grid((N + 127) / 128, split_k);
        dim3 block(256);
        
        // Zero output if using split-K (for atomic adds)
        if (split_k > 1) {
            C.zero_();
        }
        
        hk_fp8_gemm_decode_tiny_m<<<grid, block, 0, stream>>>(
            reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
            reinterpret_cast<const fp8_t*>(A.data_ptr()),
            reinterpret_cast<const fp8_t*>(B.data_ptr()),
            A_scale.data_ptr<float>(),
            B_scale.data_ptr<float>(),
            bias_ptr,
            M, N, K,
            split_k
        );
    } else if (M <= 256) {
        // Standard decode: memory-optimized kernel
        dim3 grid((N + DecodeConfig::TILE_N - 1) / DecodeConfig::TILE_N,
                  (M + DecodeConfig::TILE_M - 1) / DecodeConfig::TILE_M);
        dim3 block(256);
        
        hk_fp8_gemm_decode_memory_opt<<<grid, block, 0, stream>>>(
            reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
            reinterpret_cast<const fp8_t*>(A.data_ptr()),
            reinterpret_cast<const fp8_t*>(B.data_ptr()),
            A_scale.data_ptr<float>(),
            B_scale.data_ptr<float>(),
            bias_ptr,
            M, N, K
        );
    } else {
        // Larger M: use simple kernel (or could switch to compute-bound kernel)
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        dim3 block(256);
        
        hk_fp8_gemm_decode_simple<<<grid, block, 0, stream>>>(
            reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),
            reinterpret_cast<const fp8_t*>(A.data_ptr()),
            reinterpret_cast<const fp8_t*>(B.data_ptr()),
            A_scale.data_ptr<float>(),
            B_scale.data_ptr<float>(),
            bias_ptr,
            M, N, K
        );
    }
    
    return C;
}
