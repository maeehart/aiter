#!/usr/bin/env python3
"""
Create new kernel optimization variants.

Strategies to try:
1. Targeted vmcnt reduction (only in specific patterns)
2. Different vmcnt cap values
3. lgkmcnt-only modifications
4. Combined strategies with different ratios
"""

import os
import shutil
import struct
from typing import List, Tuple, Dict

KERNEL_DIR = "/workspace/dev/aiter_long_context2/hsa/gfx942/fmoe/silu"
BASE_KERNELS = [
    "fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co",
]


def decode_waitcnt(value: int) -> Tuple[int, int, int]:
    """Decode s_waitcnt value into (vmcnt, expcnt, lgkmcnt)."""
    vmcnt_lo = value & 0xF
    vmcnt_hi = (value >> 14) & 0x3
    vmcnt = vmcnt_lo | (vmcnt_hi << 4)
    expcnt = (value >> 4) & 0x7
    lgkmcnt = (value >> 8) & 0x3F
    return vmcnt, expcnt, lgkmcnt


def encode_waitcnt(vmcnt: int, expcnt: int, lgkmcnt: int) -> int:
    """Encode waitcnt values back to instruction format."""
    vmcnt = min(vmcnt, 63)
    expcnt = min(expcnt, 7)
    lgkmcnt = min(lgkmcnt, 63)
    
    vmcnt_lo = vmcnt & 0xF
    vmcnt_hi = (vmcnt >> 4) & 0x3
    
    return vmcnt_lo | (expcnt << 4) | (lgkmcnt << 8) | (vmcnt_hi << 14)


def find_waitcnt_locations(data: bytes) -> List[Dict]:
    """Find all s_waitcnt instructions and their values."""
    locations = []
    i = 0
    
    while i < len(data) - 4:
        instr = struct.unpack('<I', data[i:i+4])[0]
        
        # s_waitcnt: BF8Cxxxx
        if (instr >> 16) == 0xBF8C:
            value = instr & 0xFFFF
            vmcnt, expcnt, lgkmcnt = decode_waitcnt(value)
            
            locations.append({
                'offset': i,
                'instr': instr,
                'value': value,
                'vmcnt': vmcnt,
                'expcnt': expcnt,
                'lgkmcnt': lgkmcnt,
            })
        
        i += 4
    
    return locations


def apply_strategy(data: bytes, locations: List[Dict], strategy: str) -> bytes:
    """Apply optimization strategy to kernel data."""
    patched = bytearray(data)
    
    for loc in locations:
        vmcnt = loc['vmcnt']
        expcnt = loc['expcnt']
        lgkmcnt = loc['lgkmcnt']
        new_vmcnt = vmcnt
        new_lgkmcnt = lgkmcnt
        
        if strategy == 'vmcnt_cap2':
            new_vmcnt = min(vmcnt, 2)
        
        elif strategy == 'vmcnt_cap6':
            new_vmcnt = min(vmcnt, 6)
        
        elif strategy == 'vmcnt_cap12':
            new_vmcnt = min(vmcnt, 12)
        
        elif strategy == 'vmcnt_reduce50':
            new_vmcnt = max(0, vmcnt // 2)
        
        elif strategy == 'vmcnt_reduce75':
            new_vmcnt = max(0, vmcnt // 4)
        
        elif strategy == 'lgkmcnt_zero':
            new_lgkmcnt = 0
        
        elif strategy == 'lgkmcnt_reduce50':
            new_lgkmcnt = max(0, lgkmcnt // 2)
        
        elif strategy == 'aggressive':
            # Most aggressive: cap vmcnt at 2, lgkmcnt at 0
            new_vmcnt = min(vmcnt, 2)
            new_lgkmcnt = 0
        
        elif strategy == 'high_vmcnt_only':
            # Only reduce high vmcnt values (>16)
            if vmcnt > 16:
                new_vmcnt = vmcnt // 2
        
        elif strategy == 'medium_vmcnt_only':
            # Only reduce medium vmcnt values (8-16)
            if 8 <= vmcnt <= 16:
                new_vmcnt = vmcnt - 4
        
        # Encode and patch
        new_value = encode_waitcnt(new_vmcnt, expcnt, new_lgkmcnt)
        new_instr = 0xBF8C0000 | new_value
        patched[loc['offset']:loc['offset']+4] = struct.pack('<I', new_instr)
    
    return bytes(patched)


def create_variant(base_kernel: str, strategy: str) -> str:
    """Create a new kernel variant with the given strategy."""
    base_path = f"{KERNEL_DIR}/{base_kernel}"
    
    # Check for backup
    backup_path = f"{base_path}.backup"
    if os.path.exists(backup_path):
        source_path = backup_path
    else:
        source_path = base_path
        shutil.copy(base_path, backup_path)
    
    # Load kernel
    with open(source_path, 'rb') as f:
        data = f.read()
    
    # Find waitcnt locations
    locations = find_waitcnt_locations(data)
    
    # Apply strategy
    patched = apply_strategy(data, locations, strategy)
    
    # Save variant
    variant_name = base_kernel.replace('.co', f'_opt_{strategy}.co')
    variant_path = f"{KERNEL_DIR}/{variant_name}"
    
    with open(variant_path, 'wb') as f:
        f.write(patched)
    
    os.chmod(variant_path, 0o755)
    
    return variant_path


def main():
    print("=" * 70)
    print("CREATING NEW KERNEL VARIANTS")
    print("=" * 70)
    
    # Strategies to create
    strategies = [
        'vmcnt_cap2',
        'vmcnt_cap6',
        'vmcnt_cap12',
        'vmcnt_reduce50',
        'vmcnt_reduce75',
        'lgkmcnt_zero',
        'lgkmcnt_reduce50',
        'aggressive',
        'high_vmcnt_only',
        'medium_vmcnt_only',
    ]
    
    created = []
    
    for base_kernel in BASE_KERNELS:
        print(f"\nBase kernel: {base_kernel}")
        
        for strategy in strategies:
            path = create_variant(base_kernel, strategy)
            created.append((strategy, path))
            print(f"  Created: {strategy}")
    
    print(f"\n✓ Created {len(created)} new variants")
    
    return created


if __name__ == '__main__':
    main()

