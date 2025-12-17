#!/usr/bin/env python3
"""Annotate AMDGPU assembly with comments explaining instruction patterns."""

import re
import sys

# Instruction categories and their meanings
ANNOTATIONS = {
    # Scalar loads - kernel arguments
    r's_load_dword': '// Load scalar arg from kernel arguments',
    r's_load_dwordx2': '// Load 64-bit ptr from kernel arguments',
    r's_load_dwordx4': '// Load 128-bit value from kernel arguments',
    
    # Memory operations
    r'global_load_dword': '// Load 32-bit from global memory (VRAM)',
    r'global_load_dwordx2': '// Load 64-bit from global memory',
    r'global_load_dwordx4': '// Load 128-bit from global memory',
    r'global_store': '// Store to global memory (VRAM)',
    r'ds_read_b32': '// Read 32-bit from LDS (shared memory)',
    r'ds_read_b64': '// Read 64-bit from LDS',
    r'ds_read_b128': '// Read 128-bit from LDS (4x32-bit)',
    r'ds_write_b32': '// Write 32-bit to LDS',
    r'ds_write_b64': '// Write 64-bit to LDS',
    r'ds_write_b128': '// Write 128-bit to LDS',
    r'ds_write2_b32': '// Write 2x32-bit to LDS (strided)',
    r'ds_write2_b64': '// Write 2x64-bit to LDS (strided)',
    
    # Matrix operations (MFMA - Matrix Fused Multiply-Add)
    r'v_mfma_f32_16x16x32_fp8_fp8': '// MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply',
    r'v_mfma_f32_32x32x16': '// MFMA: 32x32x16 matrix multiply',
    
    # Packed operations (BF16/FP16)
    r'v_pk_fma_f32': '// Packed FMA: 2x FP32 fused multiply-add',
    r'v_pk_mul_f16': '// Packed mul: 2x FP16 multiply',
    r'v_pk_add_f16': '// Packed add: 2x FP16 addition',
    
    # Float operations
    r'v_mul_f32_dpp': '// FP32 multiply with data parallel primitives (cross-lane)',
    r'v_mul_f32_e32': '// FP32 multiply',
    r'v_add_f32': '// FP32 addition',
    r'v_fma_f32': '// FP32 fused multiply-add',
    r'v_cvt_f32_f16': '// Convert FP16 to FP32',
    r'v_cvt_f16_f32': '// Convert FP32 to FP16',
    r'v_cvt_pk': '// Packed conversion',
    
    # Control flow
    r's_cbranch_scc0': '// Branch if SCC==0 (condition false)',
    r's_cbranch_scc1': '// Branch if SCC==1 (condition true)',
    r's_cbranch_vccz': '// Branch if VCC==0 (all lanes false)',
    r's_cbranch_vccnz': '// Branch if VCC!=0 (any lane true)',
    r's_branch': '// Unconditional branch',
    r's_endpgm': '// End program',
    
    # Synchronization
    r's_waitcnt': '// Wait for memory operations (CRITICAL for perf)',
    r's_barrier': '// Workgroup barrier synchronization',
    r's_sleep': '// Sleep/stall cycles',
    
    # Address computation
    r'v_add_u32': '// 32-bit unsigned add (address calc)',
    r'v_mul_i32_i24': '// 24-bit integer multiply (index calc)',
    r'v_lshlrev_b32': '// Left shift (multiply by power of 2)',
    r'v_lshrrev_b32': '// Right shift (divide by power of 2)',
    r'v_and_b32': '// Bitwise AND (masking)',
    
    # Accumulator operations
    r'v_accvgpr_write': '// Write to accumulator VGPR (MFMA dest)',
    r'v_accvgpr_read': '// Read from accumulator VGPR',
    
    # Special
    r'v_readfirstlane': '// Broadcast lane 0 to scalar register',
    r's_mov_b32': '// Scalar move (constant/register)',
    r'v_mov_b32': '// Vector move',
}

# Section markers based on instruction patterns
def identify_section(line, prev_lines):
    """Identify which section of the kernel we're in."""
    if 's_load_dword' in line and len(prev_lines) < 50:
        return '// === KERNEL PROLOGUE: Loading arguments ==='
    if 'v_mfma_f32' in line:
        return '// === MFMA COMPUTE BLOCK ==='
    if 'ds_read_b128' in line:
        return '// === LDS READ (feeding MFMA) ==='
    if 'ds_write' in line:
        return '// === LDS WRITE (staging data) ==='
    if 'global_load' in line:
        return '// === GLOBAL MEMORY LOAD ==='
    if 's_endpgm' in line:
        return '// === KERNEL EPILOGUE ==='
    return None

def annotate_line(line):
    """Add annotation to a single line."""
    for pattern, annotation in ANNOTATIONS.items():
        if re.search(pattern, line):
            # Don't duplicate existing comments
            if '//' not in line or annotation not in line:
                return line.rstrip() + f'  {annotation}'
    return line.rstrip()

def main():
    if len(sys.argv) < 2:
        print("Usage: annotate_asm.py <input.s> [output.s]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.s', '_annotated.s')
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    annotated = []
    prev_lines = []
    current_section = None
    mfma_count = 0
    loop_depth = 0
    
    for i, line in enumerate(lines):
        # Check for section change
        new_section = identify_section(line, prev_lines)
        if new_section and new_section != current_section:
            annotated.append(f'\n{new_section}\n')
            current_section = new_section
        
        # Count MFMAs for loop identification
        if 'v_mfma' in line:
            mfma_count += 1
        
        # Detect labels (potential loop targets)
        if line.strip().startswith('label_') or (line.strip().endswith(':') and '<' in line):
            annotated.append(f'\n// --- LABEL (potential loop target) ---\n')
        
        # Add waitcnt analysis
        if 's_waitcnt' in line:
            if 'lgkmcnt(0)' in line:
                annotated.append('// SYNC: Wait for ALL LDS/scalar ops to complete\n')
            elif 'vmcnt(0)' in line:
                annotated.append('// SYNC: Wait for ALL global memory ops to complete\n')
            elif 'lgkmcnt' in line:
                annotated.append('// SYNC: Wait for some LDS ops (pipelining)\n')
        
        # Annotate and add line
        annotated_line = annotate_line(line)
        annotated.append(annotated_line + '\n')
        prev_lines.append(line)
        if len(prev_lines) > 100:
            prev_lines.pop(0)
    
    # Add header with summary
    header = f"""// ============================================================================
// ANNOTATED AMDGPU KERNEL DISASSEMBLY
// ============================================================================
// Kernel: fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x256
// Target: gfx942 (MI300X)
// 
// MFMA Instructions: {mfma_count}
// Workgroup Size: 256 threads
// VGPRs: 512, SGPRs: 112
// LDS: 64KB
//
// KEY OPTIMIZATION AREAS FOR CACHE THRASHING:
// 1. s_waitcnt values - controls memory pipeline depth
// 2. ds_read/ds_write patterns - LDS access coalescing
// 3. global_load patterns - VRAM bandwidth utilization
// 4. Loop structure - tile sizes affect cache reuse
//
// CACHE HIERARCHY (MI300X):
// - L0 Vector Cache: 16KB per CU (fastest)
// - L1 Instruction Cache: 64KB per CU
// - L2 Cache: 256MB shared (main target for optimization)
// - HBM3: ~5TB/s bandwidth
// ============================================================================

"""
    
    with open(output_file, 'w') as f:
        f.write(header)
        f.writelines(annotated)
    
    print(f"Annotated assembly written to: {output_file}")
    print(f"Total MFMA instructions: {mfma_count}")

if __name__ == '__main__':
    main()
