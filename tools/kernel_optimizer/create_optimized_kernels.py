#!/usr/bin/env python3
"""
Create optimized kernel variants by applying binary patches.
Saves the optimized kernels alongside the originals.
"""

import os
import shutil
import struct
from pathlib import Path

# Paths
AITER_PATH = "/workspace/dev/aiter_long_context2"
KERNEL_DIR = f"{AITER_PATH}/hsa/gfx942/fmoe/silu"
ORIGINAL_KERNEL = "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x256.co"

def decode_waitcnt(imm16: int):
    vmcnt_lo = imm16 & 0xF
    vmcnt_hi = (imm16 >> 14) & 0x3
    vmcnt = vmcnt_lo | (vmcnt_hi << 4)
    lgkmcnt = (imm16 >> 8) & 0xF
    expcnt = (imm16 >> 4) & 0x7
    return vmcnt, lgkmcnt, expcnt

def encode_waitcnt(vmcnt: int, lgkmcnt: int, expcnt: int) -> int:
    vmcnt_lo = vmcnt & 0xF
    vmcnt_hi = (vmcnt >> 4) & 0x3
    imm16 = vmcnt_lo | (expcnt << 4) | (lgkmcnt << 8) | (vmcnt_hi << 14)
    return imm16

def find_waitcnt_locations(kernel_data: bytes):
    locations = []
    i = 0
    while i < len(kernel_data) - 4:
        if kernel_data[i+2:i+4] == b'\x8c\xbf':
            imm16 = struct.unpack('<H', kernel_data[i:i+2])[0]
            vmcnt, lgkmcnt, expcnt = decode_waitcnt(imm16)
            locations.append({
                'offset': i,
                'vmcnt': vmcnt,
                'lgkmcnt': lgkmcnt,
                'expcnt': expcnt,
            })
            i += 4
        else:
            i += 1
    return locations

def apply_patch(kernel_data: bytes, offset: int, vmcnt: int, lgkmcnt: int, expcnt: int) -> bytes:
    imm16 = encode_waitcnt(vmcnt, lgkmcnt, expcnt)
    patched = bytearray(kernel_data)
    patched[offset:offset+2] = struct.pack('<H', imm16)
    return bytes(patched)

def create_kernel_variant(kernel_data: bytes, locations: list, strategy_name: str, transform_fn) -> bytes:
    """Apply a transformation to create a kernel variant."""
    patched = kernel_data
    for loc in locations:
        if loc['vmcnt'] > 0:  # Only patch non-zero vmcnt
            new_vmcnt, new_lgkmcnt, new_expcnt = transform_fn(
                loc['vmcnt'], loc['lgkmcnt'], loc['expcnt']
            )
            if (new_vmcnt, new_lgkmcnt, new_expcnt) != (loc['vmcnt'], loc['lgkmcnt'], loc['expcnt']):
                patched = apply_patch(patched, loc['offset'], new_vmcnt, new_lgkmcnt, new_expcnt)
    return patched

def main():
    # Load original kernel
    original_path = os.path.join(KERNEL_DIR, ORIGINAL_KERNEL)
    print(f"Loading original kernel: {original_path}")
    
    with open(original_path, 'rb') as f:
        kernel_data = f.read()
    
    locations = find_waitcnt_locations(kernel_data)
    print(f"Found {len(locations)} waitcnt instructions")
    
    # Define optimization strategies
    strategies = {
        # Best for 8K batch size
        "vmcnt_zero": lambda v, l, e: (0, l, e),
        # Best for 24K batch size  
        "vmcnt_reduce25": lambda v, l, e: (max(0, v - v // 4), l, e),
        # Good general purpose
        "vmcnt_cap4": lambda v, l, e: (min(v, 4), l, e),
        # For very high batches
        "vmcnt_cap8": lambda v, l, e: (min(v, 8), l, e),
    }
    
    created_kernels = []
    
    for strategy_name, transform in strategies.items():
        # Create output filename
        base_name = ORIGINAL_KERNEL.replace('.co', '')
        output_name = f"{base_name}_opt_{strategy_name}.co"
        output_path = os.path.join(KERNEL_DIR, output_name)
        
        # Create variant
        print(f"\nCreating variant: {strategy_name}")
        patched = create_kernel_variant(kernel_data, locations, strategy_name, transform)
        
        # Save
        with open(output_path, 'wb') as f:
            f.write(patched)
        os.chmod(output_path, 0o755)
        
        print(f"  Saved: {output_path}")
        created_kernels.append({
            'name': strategy_name,
            'filename': output_name,
            'path': output_path,
        })
    
    # Also create the CSV mapping file for AITER
    csv_path = os.path.join(KERNEL_DIR, "optimized_kernels.csv")
    with open(csv_path, 'w') as f:
        f.write("strategy,filename,description\n")
        f.write(f"original,{ORIGINAL_KERNEL},Original unmodified kernel\n")
        for k in created_kernels:
            desc = {
                'vmcnt_zero': "All vmcnt=0, best for 8K batch",
                'vmcnt_reduce25': "vmcnt reduced by 25%, best for 24K batch",
                'vmcnt_cap4': "vmcnt capped at 4, good general purpose",
                'vmcnt_cap8': "vmcnt capped at 8, moderate optimization",
            }.get(k['name'], k['name'])
            f.write(f"{k['name']},{k['filename']},{desc}\n")
    
    print(f"\n=== Created {len(created_kernels)} optimized kernel variants ===")
    print(f"CSV index saved to: {csv_path}")
    
    return created_kernels

if __name__ == '__main__':
    main()
