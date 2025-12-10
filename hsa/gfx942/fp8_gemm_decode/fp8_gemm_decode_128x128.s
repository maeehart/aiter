// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Minimal FP8 GEMM Decode Kernel for gfx942 (MI300X)
// This is a minimal working kernel - performance optimization TBD
//
// Compile with:
//   /opt/rocm/llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx942 \
//       -mcode-object-version=4 fp8_gemm_decode_128x128.s -o fp8_gemm_decode_128x128_gfx942.co

.text
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"

.globl fp8_gemm_decode_128x128
.p2align 8
.type fp8_gemm_decode_128x128,@function

fp8_gemm_decode_128x128:
    // Minimal kernel - just returns for now
    // This establishes the kernel entry point and metadata
    // Full implementation requires significant work
    
    // Get thread/workgroup IDs
    s_mov_b32 s0, 0
    
    // Simple exit
    s_endpgm

.Lfunc_end0:
.size fp8_gemm_decode_128x128, .Lfunc_end0 - fp8_gemm_decode_128x128

// Kernel descriptor using HSA ABI
.rodata
.p2align 6
.amdhsa_kernel fp8_gemm_decode_128x128
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_private_segment_fixed_size 0
    .amdhsa_kernarg_segment_size 256
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 8
    .amdhsa_float_round_mode_32 0
    .amdhsa_float_round_mode_16_64 0
    .amdhsa_float_denorm_mode_32 3
    .amdhsa_float_denorm_mode_16_64 3
    .amdhsa_dx10_clamp 1
    .amdhsa_ieee_mode 1
.end_amdhsa_kernel

.section .note.GNU-stack
