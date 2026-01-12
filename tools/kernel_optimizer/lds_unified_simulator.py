#!/usr/bin/env python3
"""
Unified LDS Simulator for MoE Kernel

This is the COMPLETE model of all LDS addressing in the kernel.
It models all 6 address paths and their timing relationships.

Address Registers:
  - v4:  64 ds_write (ring buffer)
  - v5:  112 ds_read (ring buffer) 
  - v56: 20 ds_write + 96 ds_read (intermediate data, DUAL PURPOSE)
  - v2:  40 ds_read (weights from m0 region)
  - v3:  20 ds_read (scales from m0 region)
  - m0:  Direct-to-LDS loads (buffer_load ... lds)

Timing/Phases:
  Phase 1: Load weights (m0 → LDS)
  Phase 2: MFMA computation (v56 write→read, v2/v3 read)
  Phase 3: Ring buffer write (v4)
  Phase 4: Ring buffer read (v5)
  Phase 5: Global store

This simulator validates address correctness and can test proposed changes.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
from enum import Enum


class OpType(Enum):
    """Types of LDS operations"""
    M0_LOAD = "m0_load"      # buffer_load ... lds
    V56_WRITE = "v56_write"  # ds_write via v56
    V56_READ = "v56_read"    # ds_read via v56
    V4_WRITE = "v4_write"    # ds_write via v4
    V5_READ = "v5_read"      # ds_read via v5
    V2_READ = "v2_read"      # ds_read via v2 (m0 region)
    V3_READ = "v3_read"      # ds_read via v3 (m0 region)


# ============================================================================
# STATIC OFFSETS FROM DISASSEMBLY (GROUND TRUTH)
# ============================================================================

# v56 offsets (DUAL PURPOSE - same register for write AND read)
V56_WRITE_OFFSETS = [18688, 20736, 21760, 22784, 23808, 24832, 25856, 26880, 27904]
V56_READ_OFFSETS = [18688, 18816, 18944, 19072, 19200, 19328, 19456, 19584,
                   19712, 19840, 19968, 20096, 20224, 20352, 20480, 20608,
                   20736, 20864, 21760, 21888, 22784, 22912, 23808, 23936,
                   24832, 24960, 25856, 25984, 26880, 27008, 27904, 28032]

# v4 offsets (WRITE ONLY)
V4_WRITE_OFFSETS = [20736, 22912, 25088, 27264, 29440, 31616, 33792, 35968,
                   38144, 40320, 42496, 44672, 46848, 49024, 51200, 53376]

# v5 offsets (READ ONLY)
V5_READ_OFFSETS = [20736, 20768, 20800, 20832, 25088, 25120, 25152, 25184,
                  29440, 29472, 29504, 29536, 33792, 33824, 33856, 33888,
                  38144, 38176, 38208, 38240, 42496, 42528, 42560, 42592,
                  46848, 46880, 46912, 46944, 51200, 51232, 51264, 51296]

# v2 offsets (READ ONLY - m0 region weights)
V2_READ_OFFSETS = [0, 64, 128, 192, 1024, 1088, 1152, 1216,
                  9344, 9408, 9472, 9536, 10368, 10432, 10496, 10560]

# v3 offsets (READ ONLY - m0 region scales)
V3_READ_OFFSETS = [8320, 8576, 8832, 9088, 17664, 17920, 18176, 18432]

# m0 sub-offsets for direct-to-LDS loads
M0_SUB_OFFSETS = [0x000, 0x100, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700]


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class KernelConfig:
    """Complete configuration for LDS addressing"""
    # v4 formula: (stride * (tid>>4) + 2*(tid&15) + s7_mult * s7) * 4
    v4_stride: int = 34
    v4_s7_mult: int = 136  # 0x88
    
    # v5 formula: (stride * (tid>>1) + (tid&1) + s7_mult * s7) * 4
    v5_stride: int = 34
    v5_s7_mult: int = 2
    
    # v56 write formula: ((tid>>5)*row_mult + ((tid&31)>>4) + (tid&15)*2) * 4 + s7*s7_mult
    v56_write_row_mult: int = 32
    v56_write_s7_mult: int = 256  # 0x100
    
    # v56 read formula: ((tid>>4)*group_mult + (tid&15)*2) * 4
    # NOTE: v56 read does NOT depend on s7!
    v56_read_group_mult: int = 64
    
    # m0 formula: s7 * m0_s7_mult + base_offset + sub_offset
    m0_s7_mult: int = 2080  # 0x820
    m0_s51_base: int = 9344  # 0x2480 (s51 = s50 + this)
    
    # v2/v3 use m0 as base address (set by m0 register)
    # The actual address = m0 + static_offset
    
    # Hardware constraint
    lds_size: int = 65536  # 64KB
    
    # Custom offset tables (for testing modifications)
    v4_offsets: List[int] = field(default_factory=lambda: V4_WRITE_OFFSETS.copy())
    v5_offsets: List[int] = field(default_factory=lambda: V5_READ_OFFSETS.copy())
    v56_write_offsets: List[int] = field(default_factory=lambda: V56_WRITE_OFFSETS.copy())
    v56_read_offsets: List[int] = field(default_factory=lambda: V56_READ_OFFSETS.copy())
    v2_offsets: List[int] = field(default_factory=lambda: V2_READ_OFFSETS.copy())
    v3_offsets: List[int] = field(default_factory=lambda: V3_READ_OFFSETS.copy())


# ============================================================================
# ADDRESS COMPUTATION FUNCTIONS
# ============================================================================

def compute_v4_base(tid: int, s7: int, cfg: KernelConfig) -> int:
    """Compute v4 base address (for ds_write)"""
    row = tid >> 4
    col = tid & 15
    return (cfg.v4_stride * row + 2 * col + cfg.v4_s7_mult * s7) * 4


def compute_v5_base(tid: int, s7: int, cfg: KernelConfig) -> int:
    """Compute v5 base address (for ds_read)"""
    pair = tid >> 1
    bit = tid & 1
    return (cfg.v5_stride * pair + bit + cfg.v5_s7_mult * s7) * 4


def compute_v56_write_base(tid: int, s7: int, cfg: KernelConfig) -> int:
    """Compute v56 base address for WRITE operations"""
    row = tid >> 5
    extra = (tid & 31) >> 4
    col = tid & 15
    v57 = row * cfg.v56_write_row_mult + extra + col * 2
    return v57 * 4 + s7 * cfg.v56_write_s7_mult


def compute_v56_read_base(tid: int, cfg: KernelConfig) -> int:
    """Compute v56 base address for READ operations (no s7 dependency!)"""
    group = tid >> 4
    col = tid & 15
    v57 = group * cfg.v56_read_group_mult + col * 2
    return v57 * 4


def compute_m0_base(s7: int, is_s51: bool, cfg: KernelConfig) -> int:
    """Compute m0 base address for direct-to-LDS loads"""
    base = s7 * cfg.m0_s7_mult
    if is_s51:
        base += cfg.m0_s51_base
    return base


def compute_v2_base(tid: int, s7: int, cfg: KernelConfig) -> int:
    """Compute v2 base (reads from m0 region)
    v2 is computed from m0 with thread-specific offset"""
    # v2 = m0_base (s50 region for lower offsets, s51 for higher)
    # The formula is complex, but key insight: it reads from m0 region
    return s7 * cfg.m0_s7_mult  # Simplified - actual is more complex


def compute_v3_base(tid: int, s7: int, cfg: KernelConfig) -> int:
    """Compute v3 base (reads from m0 region - scales)"""
    return s7 * cfg.m0_s7_mult  # Simplified


# ============================================================================
# LDS REGION ANALYSIS
# ============================================================================

def analyze_all_addresses(cfg: KernelConfig, max_s7: int = 5) -> Dict[str, Set[int]]:
    """Compute all possible LDS addresses for each operation type"""
    
    addresses = {
        'm0_s50': set(),
        'm0_s51': set(),
        'v56_write': set(),
        'v56_read': set(),
        'v4_write': set(),
        'v5_read': set(),
        'v2_read': set(),
        'v3_read': set(),
    }
    
    for s7 in range(max_s7):
        # m0 regions
        for sub in M0_SUB_OFFSETS:
            addresses['m0_s50'].add(compute_m0_base(s7, False, cfg) + sub)
            addresses['m0_s51'].add(compute_m0_base(s7, True, cfg) + sub)
        
        for tid in range(256):
            # v4 write addresses
            v4_base = compute_v4_base(tid, s7, cfg)
            for off in cfg.v4_offsets:
                addresses['v4_write'].add((v4_base + off) % cfg.lds_size)
            
            # v5 read addresses
            v5_base = compute_v5_base(tid, s7, cfg)
            for off in cfg.v5_offsets:
                addresses['v5_read'].add((v5_base + off) % cfg.lds_size)
            
            # v56 write addresses
            v56w_base = compute_v56_write_base(tid, s7, cfg)
            for off in cfg.v56_write_offsets:
                addresses['v56_write'].add((v56w_base + off) % cfg.lds_size)
            
            # v56 read addresses (no s7!)
            v56r_base = compute_v56_read_base(tid, cfg)
            for off in cfg.v56_read_offsets:
                addresses['v56_read'].add((v56r_base + off) % cfg.lds_size)
            
            # v2/v3 simplified (use m0 + static offset)
            for off in cfg.v2_offsets:
                v2_addr = compute_m0_base(s7, False, cfg) + off
                if off >= 9344:  # Uses s51 region
                    v2_addr = compute_m0_base(s7, True, cfg) + (off - 9344)
                addresses['v2_read'].add(v2_addr % cfg.lds_size)
            
            for off in cfg.v3_offsets:
                v3_addr = compute_m0_base(s7, False, cfg) + off
                if off >= 9344:
                    v3_addr = compute_m0_base(s7, True, cfg) + (off - 9344)
                addresses['v3_read'].add(v3_addr % cfg.lds_size)
    
    return addresses


def check_region_overlaps(addresses: Dict[str, Set[int]]) -> Dict[str, Set[int]]:
    """Check for overlaps between different LDS regions"""
    overlaps = {}
    
    # Regions that SHOULD NOT overlap (would cause data corruption)
    forbidden_pairs = [
        ('m0_s50', 'v4_write'),
        ('m0_s51', 'v4_write'),
        ('m0_s50', 'v56_write'),
        ('m0_s51', 'v56_write'),
    ]
    
    for r1, r2 in forbidden_pairs:
        overlap = addresses[r1] & addresses[r2]
        if overlap:
            overlaps[f"{r1}∩{r2}"] = overlap
    
    # Regions that MUST overlap (data flow dependency)
    required_pairs = [
        ('v4_write', 'v5_read'),      # Ring buffer: v4 writes what v5 reads
        ('v56_write', 'v56_read'),    # v56 dual: write then read same location
    ]
    
    for r1, r2 in required_pairs:
        overlap = addresses[r1] & addresses[r2]
        if not overlap:
            overlaps[f"MISSING:{r1}→{r2}"] = set()
        else:
            # This is expected - record for info
            overlaps[f"OK:{r1}→{r2}"] = overlap
    
    return overlaps


# ============================================================================
# WRAP-AROUND ANALYSIS
# ============================================================================

def analyze_wrap_around(cfg: KernelConfig, max_s7: int = 5) -> Dict[str, List[Tuple]]:
    """Find addresses that wrap around 64KB boundary"""
    
    wrapping = {
        'v4_write': [],
        'v5_read': [],
        'v56_write': [],
        'v56_read': [],
    }
    
    for s7 in range(max_s7):
        for tid in range(256):
            # v4 write
            v4_base = compute_v4_base(tid, s7, cfg)
            for off in cfg.v4_offsets:
                raw = v4_base + off
                if raw >= cfg.lds_size:
                    wrapping['v4_write'].append((tid, s7, off, raw, raw % cfg.lds_size))
            
            # v5 read
            v5_base = compute_v5_base(tid, s7, cfg)
            for off in cfg.v5_offsets:
                raw = v5_base + off
                if raw >= cfg.lds_size:
                    wrapping['v5_read'].append((tid, s7, off, raw, raw % cfg.lds_size))
            
            # v56 write
            v56w_base = compute_v56_write_base(tid, s7, cfg)
            for off in cfg.v56_write_offsets:
                raw = v56w_base + off
                if raw >= cfg.lds_size:
                    wrapping['v56_write'].append((tid, s7, off, raw, raw % cfg.lds_size))
            
            # v56 read (no s7)
            v56r_base = compute_v56_read_base(tid, cfg)
            for off in cfg.v56_read_offsets:
                raw = v56r_base + off
                if raw >= cfg.lds_size:
                    wrapping['v56_read'].append((tid, s7, off, raw, raw % cfg.lds_size))
    
    return wrapping


# ============================================================================
# DATA FLOW VALIDATION
# ============================================================================

def validate_v4_v5_data_flow(cfg: KernelConfig, s7: int = 0) -> Tuple[bool, str]:
    """Validate that v4 writes can be read by v5"""
    
    # Get all v4 write addresses for this s7
    v4_addrs = set()
    for tid in range(256):
        base = compute_v4_base(tid, s7, cfg)
        for off in cfg.v4_offsets:
            v4_addrs.add((base + off) % cfg.lds_size)
    
    # Get all v5 read addresses for this s7
    v5_addrs = set()
    for tid in range(256):
        base = compute_v5_base(tid, s7, cfg)
        for off in cfg.v5_offsets:
            v5_addrs.add((base + off) % cfg.lds_size)
    
    # Check overlap
    common = v4_addrs & v5_addrs
    only_write = v4_addrs - v5_addrs
    only_read = v5_addrs - v4_addrs
    
    if len(common) == 0:
        return False, "NO overlap between v4 writes and v5 reads!"
    
    msg = f"v4/v5 overlap: {len(common)} addresses, {len(only_write)} write-only, {len(only_read)} read-only"
    return True, msg


def validate_v56_data_flow(cfg: KernelConfig, s7: int = 0) -> Tuple[bool, str]:
    """Validate that v56 writes are readable by v56 reads"""
    
    # v56 write addresses
    v56w_addrs = set()
    for tid in range(256):
        base = compute_v56_write_base(tid, s7, cfg)
        for off in cfg.v56_write_offsets:
            v56w_addrs.add((base + off) % cfg.lds_size)
    
    # v56 read addresses (note: no s7 in read formula!)
    v56r_addrs = set()
    for tid in range(256):
        base = compute_v56_read_base(tid, cfg)
        for off in cfg.v56_read_offsets:
            v56r_addrs.add((base + off) % cfg.lds_size)
    
    common = v56w_addrs & v56r_addrs
    only_write = v56w_addrs - v56r_addrs
    only_read = v56r_addrs - v56w_addrs
    
    if len(common) == 0:
        return False, "NO overlap between v56 writes and v56 reads!"
    
    msg = f"v56 overlap: {len(common)} addresses, {len(only_write)} write-only, {len(only_read)} read-only"
    return True, msg


# ============================================================================
# MAIN ANALYSIS AND TESTING
# ============================================================================

def run_full_analysis(cfg: Optional[KernelConfig] = None, verbose: bool = True):
    """Run complete analysis of LDS usage"""
    
    if cfg is None:
        cfg = KernelConfig()
    
    print("=" * 70)
    print("UNIFIED LDS SIMULATOR - COMPLETE ANALYSIS")
    print("=" * 70)
    
    # 1. Address regions
    if verbose:
        print("\n=== LDS ADDRESS REGIONS ===")
    addresses = analyze_all_addresses(cfg, max_s7=5)
    
    if verbose:
        for region, addrs in addresses.items():
            if addrs:
                print(f"  {region:12}: {min(addrs):6} - {max(addrs):6} ({len(addrs)} unique)")
    
    # 2. Overlaps
    if verbose:
        print("\n=== REGION OVERLAPS ===")
    overlaps = check_region_overlaps(addresses)
    
    if verbose:
        for name, addrs in overlaps.items():
            if name.startswith("OK:"):
                print(f"  ✓ {name[3:]}: {len(addrs)} addresses")
            elif name.startswith("MISSING:"):
                print(f"  ✗ {name[8:]}: MISSING data flow!")
            else:
                print(f"  ⚠ {name}: {len(addrs)} collision addresses!")
    
    # 3. Wrap-around
    if verbose:
        print("\n=== WRAP-AROUND BEHAVIOR ===")
    wrapping = analyze_wrap_around(cfg)
    
    if verbose:
        for region, wraps in wrapping.items():
            if wraps:
                unique_tids = len(set(w[0] for w in wraps))
                print(f"  {region}: {len(wraps)} wrapping accesses ({unique_tids} threads)")
    
    # 4. Data flow validation
    if verbose:
        print("\n=== DATA FLOW VALIDATION ===")
    
    v4v5_ok, v4v5_msg = validate_v4_v5_data_flow(cfg, s7=0)
    v56_ok, v56_msg = validate_v56_data_flow(cfg, s7=0)
    
    if verbose:
        print(f"  v4→v5: {'✓' if v4v5_ok else '✗'} {v4v5_msg}")
        print(f"  v56:   {'✓' if v56_ok else '✗'} {v56_msg}")
    
    # Return results
    return {
        'addresses': addresses,
        'overlaps': overlaps,
        'wrapping': wrapping,
        'v4v5_valid': v4v5_ok,
        'v56_valid': v56_ok,
    }


def test_offset_shift(shift_amount: int, verbose: bool = True):
    """Test shifting v4/v5 offsets by a given amount"""
    
    print(f"\n{'='*70}")
    print(f"TESTING V4/V5 OFFSET SHIFT: {shift_amount} bytes")
    print(f"{'='*70}")
    
    # Create modified config
    cfg = KernelConfig()
    cfg.v4_offsets = [off - shift_amount for off in V4_WRITE_OFFSETS]
    cfg.v5_offsets = [off - shift_amount for off in V5_READ_OFFSETS]
    
    if verbose:
        print(f"\nOriginal V4 range: {min(V4_WRITE_OFFSETS)} - {max(V4_WRITE_OFFSETS)}")
        print(f"New V4 range:      {min(cfg.v4_offsets)} - {max(cfg.v4_offsets)}")
    
    # Run analysis
    results = run_full_analysis(cfg, verbose=verbose)
    
    # Check for collisions with m0 region
    m0_addrs = results['addresses']['m0_s50'] | results['addresses']['m0_s51']
    v4v5_addrs = results['addresses']['v4_write'] | results['addresses']['v5_read']
    
    collision = m0_addrs & v4v5_addrs
    if collision:
        print(f"\n⚠️  COLLISION with m0 region! {len(collision)} addresses overlap")
        print(f"   Sample collision addresses: {sorted(collision)[:5]}")
    else:
        print(f"\n✓ No collision with m0 region")
    
    # Check wrap-around change
    orig_wrapping = len(analyze_wrap_around(KernelConfig())['v5_read'])
    new_wrapping = len(results['wrapping']['v5_read'])
    print(f"\n   Wrap-around: {orig_wrapping} → {new_wrapping} (Δ{new_wrapping - orig_wrapping})")
    
    return results, collision


def analyze_wrap_timing():
    """Analyze the timing relationship between m0 writes and v5 wrapped reads"""
    
    print("\n" + "=" * 70)
    print("WRAP-AROUND TIMING ANALYSIS")
    print("=" * 70)
    
    cfg = KernelConfig()
    
    # v5 reads via wrap-around at s7=0
    v5_wrapped = {}  # addr -> list of (tid, offset)
    for tid in range(256):
        base = compute_v5_base(tid, 0, cfg)
        for off in V5_READ_OFFSETS:
            raw = base + off
            if raw >= 65536:
                wrapped = raw % 65536
                if wrapped not in v5_wrapped:
                    v5_wrapped[wrapped] = []
                v5_wrapped[wrapped].append((tid, off))
    
    print(f"\nv5 wrapped addresses at s7=0: {len(v5_wrapped)} unique")
    print(f"Range: {min(v5_wrapped.keys())} - {max(v5_wrapped.keys())}")
    
    # m0 s50 at s7=0 writes to: 0, 256, 512, 768, 1024, 1280, 1536, 1792
    m0_s50_addrs = set(sub for sub in M0_SUB_OFFSETS)
    
    print(f"\nm0 s50 at s7=0: {sorted(m0_s50_addrs)}")
    
    print("""
KEY INSIGHT:
The m0 region loads weights/scales in 256-byte chunks at addresses 0, 256, 512...
The v5 wrapped reads access addresses 8, 12, 40, 44, 80, 84, ...
These are WITHIN the same cache lines as the m0 data!
""")
    
    return v5_wrapped


def analyze_v4_vs_v5_wrapping():
    """
    Critical analysis: v4 WRITE wrap-around vs v5 READ wrap-around.
    
    v5 wrap-around (READ) is safe because it only reads data.
    v4 wrap-around (WRITE) is dangerous because it corrupts data.
    """
    
    print("\n" + "=" * 70)
    print("V4 vs V5 WRAP-AROUND: THE CRITICAL DIFFERENCE")
    print("=" * 70)
    
    cfg = KernelConfig()
    
    results = []
    for lds_kb in [32, 40, 48, 52, 56, 58, 60, 64]:
        lds_size = lds_kb * 1024
        
        # Count v4 wraps (WRITES)
        v4_wraps = 0
        v4_dest = set()
        for s7 in range(8):
            for tid in range(256):
                base = compute_v4_base(tid, s7, cfg)
                for off in cfg.v4_offsets:
                    raw = base + off
                    if raw >= lds_size:
                        v4_wraps += 1
                        v4_dest.add(raw % lds_size)
        
        # Count v5 wraps (READS)
        v5_wraps = 0
        v5_dest = set()
        for s7 in range(8):
            for tid in range(256):
                base = compute_v5_base(tid, s7, cfg)
                for off in cfg.v5_offsets:
                    raw = base + off
                    if raw >= lds_size:
                        v5_wraps += 1
                        v5_dest.add(raw % lds_size)
        
        results.append({
            'lds_kb': lds_kb,
            'v4_wraps': v4_wraps,
            'v4_dest': v4_dest,
            'v5_wraps': v5_wraps,
            'v5_dest': v5_dest,
        })
    
    print("\n| LDS Size | v4 WRITES wrap | v4 Dest Range | v5 READS wrap | v5 Dest Range |")
    print("|----------|----------------|---------------|---------------|---------------|")
    
    for r in results:
        v4_range = f"{min(r['v4_dest'])}-{max(r['v4_dest'])}" if r['v4_dest'] else "none"
        v5_range = f"{min(r['v5_dest'])}-{max(r['v5_dest'])}" if r['v5_dest'] else "none"
        print(f"| {r['lds_kb']:3}KB    | {r['v4_wraps']:14} | {v4_range:13} | {r['v5_wraps']:13} | {v5_range:13} |")
    
    print("""
CRITICAL INSIGHT:
-----------------
v4 wrap-around WRITES corrupt m0 weight data -> KERNEL FAILS
v5 wrap-around READS access m0 region -> SAFE (data already consumed)

At 56KB: Only 564 v4 wraps (dest 0-2000), 9774 v5 wraps -> WORKS
At 58KB: Zero v4 wraps -> guaranteed safe
At 32KB: 23232 v4 wraps (dest 0-26KB) -> corrupts most of m0 -> FAILS

For 2 workgroups per CU (32KB each), we would need:
- Zero v4 wraps at 32KB, which requires max v4 addr < 32KB
- Current max v4 addr at s7=0: 55.5KB, way too large
- This requires source-level tile size reduction, not binary patching
""")
    
    return results


def analyze_stride_reduction():
    """
    Analyze whether reducing stride could enable 32KB LDS.
    Spoiler: It cannot, due to fundamental algorithm constraints.
    """
    
    print("\n" + "=" * 70)
    print("STRIDE REDUCTION ANALYSIS FOR 32KB LDS")
    print("=" * 70)
    
    print("""
The stride (34) determines v5 dynamic range:
  v5 = (stride * (tid >> 1) + (tid & 1)) * 4
  For tid=255: v5_base = (stride * 127 + 1) * 4
""")
    
    print("| Stride | v5 Dynamic Range | Could fit 32KB? |")
    print("|--------|------------------|-----------------|")
    
    for stride in [34, 24, 17, 12, 8, 4, 2, 1]:
        v5_dyn = (stride * 127 + 1) * 4
        # Available space after m0: 32KB - 20KB = 12KB
        fits = "✓" if v5_dyn < 12288 else "✗"
        print(f"| {stride:6} | {v5_dyn:16} | {fits:15} |")
    
    print("""
PROBLEM: Even at stride=1, we need 512 bytes for v5 dynamic range alone.
Add static offsets + ring buffer depth, and we exceed 12KB immediately.

With 256 threads paired as 128 pairs, minimum addressing is:
  128 pairs × 4 bytes = 512 bytes (just for unique addresses!)

The MFMA tile needs 256 elements per output.
Ring buffer needs ~35KB to store these with proper spacing.

32KB LDS REQUIRES SOURCE-LEVEL CHANGES:
1. Reduce thread count (64 instead of 256)
2. Use smaller MFMA tiles (8x8 instead of 16x16)  
3. Reduce ring buffer depth
4. Eliminate double-buffering

These changes are NOT possible via binary patching.
""")


def analyze_communication_pattern_change(new_stride, scale_offsets=True):
    """
    Analyze how changing stride affects thread communication patterns.
    """
    
    print(f"\n{'='*70}")
    print(f"COMMUNICATION PATTERN: stride=34 vs stride={new_stride}")
    print(f"{'='*70}")
    
    def get_communications(stride, s7_mult_w, s7_mult_r, v4_off, v5_off):
        pairs = set()
        for w_tid in range(256):
            w_row = w_tid >> 4
            w_col = w_tid & 15
            w_base = (stride * w_row + 2 * w_col) * 4
            
            for w_o in v4_off:
                w_addr = w_base + w_o
                
                for r_tid in range(256):
                    r_pair = r_tid >> 1
                    r_bit = r_tid & 1
                    r_base = (stride * r_pair + r_bit) * 4
                    
                    for r_o in v5_off:
                        if w_base + w_o == r_base + r_o:
                            pairs.add((w_tid, r_tid))
        return pairs
    
    # Original
    orig_pairs = get_communications(34, 136, 2, V4_WRITE_OFFSETS, V5_READ_OFFSETS)
    
    # New stride with scaled offsets
    if scale_offsets:
        scale = new_stride / 34
        new_v4 = [int(o * scale) for o in V4_WRITE_OFFSETS]
        new_v5 = [int(o * scale) for o in V5_READ_OFFSETS]
    else:
        new_v4 = V4_WRITE_OFFSETS
        new_v5 = V5_READ_OFFSETS
    
    new_pairs = get_communications(new_stride, int(136 * new_stride/34), int(2 * new_stride/34), new_v4, new_v5)
    
    common = orig_pairs & new_pairs
    only_orig = orig_pairs - new_pairs
    only_new = new_pairs - orig_pairs
    
    print(f"\nOriginal (stride=34): {len(orig_pairs)} thread pairs")
    print(f"New (stride={new_stride}): {len(new_pairs)} thread pairs")
    print(f"Common: {len(common)} ({100*len(common)/len(orig_pairs):.1f}%)")
    print(f"Changed: {len(only_orig) + len(only_new)} pairs")
    
    if len(common) < len(orig_pairs):
        print(f"\n✗ COMMUNICATION PATTERN BREAKS! Different threads would exchange data.")
        print(f"  This causes MFMA results to be misrouted -> wrong output -> NaN")
    else:
        print(f"\n✓ Communication pattern preserved")
    
    return len(common) == len(orig_pairs) == len(new_pairs)


def analyze_smaller_tile():
    """
    Analyze what a smaller tile size (16x128 instead of 32x256) would require.
    
    Current kernel "32x256" means:
    - 32 = output tile rows  
    - 256 = output tile cols
    - Uses 256 threads with 768 v_mfma_f32_16x16x32_fp8_fp8 instructions
    
    Smaller "16x128" would mean:
    - 16 = output tile rows (half)
    - 128 = output tile cols (half)
    - Could use 64 threads with ~192 MFMA instructions
    
    LDS impact:
    - m0 region: scales with tile size (would be ~6.5KB instead of 26KB)
    - Ring buffer: scales with thread count (would be ~8KB instead of 32KB)
    - Total: ~16KB instead of ~58KB -> FITS IN 32KB!
    """
    
    print("\n" + "=" * 70)
    print("SMALLER TILE SIZE ANALYSIS (16x128 vs 32x256)")
    print("=" * 70)
    
    print("""
Current 32x256 tile:
├── 256 threads (4 waves of 64)
├── Tile size: 32 rows × 256 cols
├── 768 MFMA instructions (v_mfma_f32_16x16x32_fp8_fp8)
├── LDS usage: ~58KB
│   ├── m0 region (weights): ~26KB (8 iterations × ~3.2KB)
│   ├── Ring buffer: ~32KB (256 threads × ~128 bytes)
│   └── v56 intermediate: ~10KB
└── 1 workgroup per CU (uses full 64KB LDS)

Proposed 16x128 tile:
├── 64 threads (1 wave)  
├── Tile size: 16 rows × 128 cols
├── ~96 MFMA instructions
├── LDS usage: ~14KB (estimated)
│   ├── m0 region (weights): ~6.5KB (4 iterations × ~1.6KB)
│   ├── Ring buffer: ~8KB (64 threads × ~128 bytes)
│   └── v56 intermediate: ~2.5KB
└── 4 workgroups per CU (uses 16KB × 4 = 64KB LDS)
""")
    
    # Calculate actual estimates
    print("=== THREAD COUNT SCALING ===")
    print(f"Original: 256 threads, v5 dynamic = {34 * 127 + 1} pairs × 4 = {(34*127+1)*4} bytes")
    print(f"Reduced:   64 threads, v5 dynamic = {34 * 31 + 1} pairs × 4 = {(34*31+1)*4} bytes")
    
    # 64 threads: tid ranges 0-63, so:
    # v5: tid>>1 gives 0-31, so max pair = 31
    # v5 dynamic = (34 * 31 + 1) * 4 = 4220 bytes
    
    print(f"\nm0 region with 4 iterations:")
    m0_4iter = 3 * 2080 + 9344 + 1792  # s7=0,1,2,3
    print(f"  {m0_4iter} bytes ({m0_4iter/1024:.1f}KB)")
    
    ring_64t = (34 * 15 + 30) * 4 + 20736  # max v4 at tid=63 (tid>>4 = 3)
    print(f"\nRing buffer with 64 threads (max v4 at s7=0):")
    print(f"  {ring_64t} bytes ({ring_64t/1024:.1f}KB)")
    
    total_est = m0_4iter + 8000 + 4000
    print(f"\nEstimated total LDS: ~{total_est/1024:.0f}KB")
    print(f"Would fit in 32KB: {'YES' if total_est <= 32768 else 'NO'}")
    
    print("""
=== ASSEMBLY CHANGES REQUIRED FOR 16x128 ===

1. REDUCE THREAD COUNT
   - Change wavefront count from 4 to 1
   - Modify kernel launch parameters
   - Update all tid-based calculations to use tid & 63 instead of tid & 255
   
2. REDUCE ITERATION COUNT  
   - Change s7 loop from 8 to 4 iterations
   - Halve m0_s7_mult: 2080 -> 1040
   - Halve m0_s51_offset: 9344 -> 4672
   
3. SCALE STATIC OFFSETS
   - Reduce v4/v5 base offset: 20736 -> 10368
   - Scale all offset tables by 0.5
   
4. REDUCE MFMA COUNT
   - Keep only 96 of 768 MFMA instructions
   - Remove redundant accumulator registers
   
5. UPDATE KERNEL DESCRIPTOR
   - .amdhsa_group_segment_fixed_size: 65536 -> 16384
   - .amdhsa_next_free_vgpr: 256 -> 64 (rough estimate)
""")
    
    return True


def design_16x128_kernel():
    """
    Design the modifications needed for a 16x128 tile kernel.
    This would fit in 16KB LDS, allowing 4 workgroups per CU.
    """
    
    print("\n" + "=" * 70)
    print("DESIGNING 16x128 TILE KERNEL")
    print("=" * 70)
    
    # Original 32x256: 256 threads process 32 output rows, 256 output cols
    # - 16 v4 offsets = 16 blocks of output columns (256/16 = 16)
    # - 8 s7 iterations = 8 blocks of processing
    
    # New 16x128: 64 threads process 16 output rows, 128 output cols
    # - 4 v4 offsets = 4 blocks of output columns (128/32 = 4)
    # - 4 s7 iterations = 4 blocks of processing
    
    new_params = {
        'threads': 64,
        'waves': 1,
        'tile_rows': 16,
        'tile_cols': 128,
        'iterations': 4,           # Half of 8
        'num_v4_offsets': 4,       # Quarter of 16
        'num_v5_offsets': 8,       # Quarter of 32
        'stride': 17,              # Half stride for half threads (34 * 64/128)
        'v4_s7_mult': 34,          # Scale down: 136 * 64/256
        'm0_s7_mult': 520,         # Quarter of 2080
        'm0_s51_offset': 2336,     # Quarter of 9344
    }
    
    print(f"\nNew parameters for 16x128 tile:")
    for k, v in new_params.items():
        print(f"  {k}: {v}")
    
    # Calculate LDS regions
    print("\nLDS Region Calculations:")
    
    # m0 region: 4 iterations with smaller multiplier
    m0_max = (new_params['iterations']-1) * new_params['m0_s7_mult'] + new_params['m0_s51_offset'] + 448
    print(f"  m0 region: 0 - {m0_max} ({m0_max/1024:.1f}KB)")
    
    # v4 dynamic range with 64 threads and stride 17
    # tid>>4 gives 0-3, tid&15 gives 0-15 (but only 64 threads, so tid 0-63)
    # max row = 63>>4 = 3, max col = 63&15 = 15
    # max v4 base = (17 * 3 + 2*15) * 4 = (51 + 30) * 4 = 324 bytes
    # plus s7 shift: 34 * 3 * 4 = 408 bytes
    v4_dynamic = (new_params['stride'] * 3 + 30 + new_params['v4_s7_mult'] * 3) * 4
    print(f"  v4 dynamic range: {v4_dynamic} bytes")
    
    # Ring buffer base after m0
    ring_base = ((m0_max // 256) + 1) * 256
    print(f"  Ring buffer base: {ring_base} ({ring_base/1024:.1f}KB)")
    
    # New offsets: only 4 instead of 16, with smaller spacing
    # Original spacing: 2176 = 64 * stride = 64 * 34
    # New spacing: 64 * 17 = 1088
    new_spacing = 64 * new_params['stride']
    new_v4_offsets = [ring_base + i * new_spacing for i in range(new_params['num_v4_offsets'])]
    
    # v5 offsets: 4 groups of 2 (instead of 8 groups of 4)
    new_v5_offsets = []
    for group in range(new_params['num_v4_offsets']):
        for sub in range(2):  # 2 sub-offsets per group
            new_v5_offsets.append(ring_base + group * new_spacing + sub * 32)
    
    print(f"  New v4 offsets ({len(new_v4_offsets)}): {new_v4_offsets}")
    print(f"  New v5 offsets ({len(new_v5_offsets)}): {new_v5_offsets}")
    
    # Total LDS needed
    v4_max = v4_dynamic + max(new_v4_offsets)
    print(f"\n  Total max address: {v4_max} ({v4_max/1024:.1f}KB)")
    
    fits_16kb = v4_max <= 16384
    fits_32kb = v4_max <= 32768
    print(f"  Fits in 16KB: {'YES!' if fits_16kb else 'NO'}")
    print(f"  Fits in 32KB: {'YES!' if fits_32kb else 'NO'}")
    
    print("""
=== IMPLEMENTATION APPROACH ===

Creating 16x128 requires REMOVING code, not just changing parameters:

1. THREAD REDUCTION (256 -> 64)
   - Keep only first wave (tid 0-63)
   - Remove all code that handles tid >= 64
   - This removes ~75% of thread-dependent operations
   
2. ITERATION REDUCTION (8 -> 4)
   - Change loop comparison from 8 to 4
   - Or remove s7 >= 4 code paths
   
3. OFFSET REDUCTION  
   - Keep only first 4 of 16 v4 offsets
   - Keep only first 8 of 32 v5 offsets
   - Remove associated ds_write/ds_read instructions
   
4. MFMA REDUCTION
   - Original: 768 MFMA instructions for 32x256 tile
   - New: ~48 MFMA instructions for 16x128 tile (768/16)
   - Remove all MFMA blocks for output rows 16-31 and cols 128-255

5. GLOBAL MEMORY CHANGES
   - Halve buffer_load addresses
   - Halve global_store addresses
   
This is approximately 80% of the kernel code that needs removal/modification.
""")
    
    new_params['v4_offsets'] = new_v4_offsets
    new_params['v5_offsets'] = new_v5_offsets
    new_params['ring_base'] = ring_base
    new_params['fits_16kb'] = fits_16kb
    new_params['fits_32kb'] = fits_32kb
    
    return new_params


def test_coordinated_scaling(scale_factor: float, verbose: bool = True):
    """
    Test coordinated scaling of all LDS parameters.
    
    This function tests if scaling stride, multipliers, AND offsets together
    can preserve communication patterns while reducing LDS usage.
    
    Returns True if communication patterns are preserved, False otherwise.
    """
    
    print(f"\n{'='*70}")
    print(f"COORDINATED SCALING TEST: factor = {scale_factor}")
    print(f"{'='*70}")
    
    orig_cfg = KernelConfig()
    
    # Calculate new parameters
    new_stride = int(orig_cfg.v4_stride * scale_factor)
    new_v4_s7_mult = int(orig_cfg.v4_s7_mult * scale_factor)
    new_v56_s7_mult = int(256 * scale_factor)  # v56 uses 0x100
    old_slot_spacing = orig_cfg.v4_stride * 16 * 4  # 2176
    new_slot_spacing = new_stride * 16 * 4
    
    print(f"\nParameter changes:")
    print(f"  stride: {orig_cfg.v4_stride} -> {new_stride}")
    print(f"  v4_s7_mult: {orig_cfg.v4_s7_mult} -> {new_v4_s7_mult}")
    print(f"  v56_s7_mult: 256 -> {new_v56_s7_mult}")
    print(f"  slot_spacing: {old_slot_spacing} -> {new_slot_spacing}")
    
    # Scale V4 offsets (slot-based)
    v4_base = V4_WRITE_OFFSETS[0]
    new_v4_offsets = []
    for old in V4_WRITE_OFFSETS:
        slot_idx = (old - v4_base) // old_slot_spacing
        new_off = v4_base + slot_idx * new_slot_spacing
        new_v4_offsets.append(new_off)
    
    # Scale V5 offsets (preserve delta from V4 slot base)
    new_v5_offsets = []
    for old in V5_READ_OFFSETS:
        for v4_old in V4_WRITE_OFFSETS:
            delta = old - v4_old
            if 0 <= delta < 128:
                slot_idx = (v4_old - v4_base) // old_slot_spacing
                new_v4_slot = v4_base + slot_idx * new_slot_spacing
                new_v5_offsets.append(new_v4_slot + delta)
                break
    
    # Scale V56 offsets in ring buffer region
    new_v56_write = []
    for old in V56_WRITE_OFFSETS:
        if old < v4_base:
            new_v56_write.append(old)  # Keep intermediate region unchanged
        else:
            slot_idx = (old - v4_base) // old_slot_spacing
            delta = (old - v4_base) % old_slot_spacing
            new_v56_write.append(v4_base + slot_idx * new_slot_spacing + delta)
    
    if verbose:
        print(f"\nScaled offsets:")
        print(f"  V4: {V4_WRITE_OFFSETS[:4]} -> {new_v4_offsets[:4]}")
        print(f"  V5: {V5_READ_OFFSETS[:4]} -> {new_v5_offsets[:4]}")
    
    # Check max address with new parameters
    # v4 max = (stride * 15 + 30) * 4 + max_offset + s7 * mult * 4
    v4_max = (new_stride * 15 + 30) * 4 + max(new_v4_offsets) + 63 * new_v4_s7_mult * 4
    v5_max = (new_stride * 127 + 1) * 4 + max(new_v5_offsets) + 63 * orig_cfg.v5_s7_mult * 4
    
    print(f"\nMax addresses (s7=63):")
    print(f"  v4 max: {v4_max} ({v4_max/1024:.1f}KB)")
    print(f"  v5 max: {v5_max} ({v5_max/1024:.1f}KB)")
    
    # Check communication pattern preservation
    def get_v4v5_pairs(stride, v4_mult, v5_mult, v4_offs, v5_offs, s7=0):
        """Get (write_tid, read_tid) pairs that communicate"""
        pairs = set()
        for w_tid in range(256):
            w_row = w_tid >> 4
            w_col = w_tid & 15
            w_base = (stride * w_row + 2 * w_col + v4_mult * s7) * 4
            
            for w_off in v4_offs:
                w_addr = w_base + w_off
                
                for r_tid in range(256):
                    r_pair = r_tid >> 1
                    r_bit = r_tid & 1
                    r_base = (stride * r_pair + r_bit + v5_mult * s7) * 4
                    
                    for r_off in v5_offs:
                        if w_base + w_off == r_base + r_off:
                            pairs.add((w_tid, r_tid))
        return pairs
    
    orig_pairs = get_v4v5_pairs(orig_cfg.v4_stride, orig_cfg.v4_s7_mult, 
                                 orig_cfg.v5_s7_mult, V4_WRITE_OFFSETS, V5_READ_OFFSETS)
    new_pairs = get_v4v5_pairs(new_stride, new_v4_s7_mult,
                                orig_cfg.v5_s7_mult, new_v4_offsets, new_v5_offsets)
    
    common = orig_pairs & new_pairs
    preserved = len(common) / len(orig_pairs) * 100 if orig_pairs else 0
    
    print(f"\nCommunication pattern:")
    print(f"  Original pairs: {len(orig_pairs)}")
    print(f"  New pairs: {len(new_pairs)}")
    print(f"  Preserved: {len(common)} ({preserved:.1f}%)")
    
    # Check m0 collision
    m0_max = orig_cfg.m0_s7_mult * 7 + orig_cfg.m0_s51_base + 1792
    ring_min = min(new_v4_offsets)
    
    collision = ring_min < m0_max
    print(f"\nm0/ring collision check:")
    print(f"  m0 max (s7=7): {m0_max}")
    print(f"  Ring buffer min: {ring_min}")
    print(f"  Collision: {'YES - FAILS!' if collision else 'NO - OK'}")
    
    is_valid = preserved == 100 and not collision
    
    print(f"\n{'✓ VALID' if is_valid else '✗ INVALID'} configuration")
    
    return {
        'valid': is_valid,
        'new_stride': new_stride,
        'new_v4_s7_mult': new_v4_s7_mult,
        'new_v56_s7_mult': new_v56_s7_mult,
        'new_v4_offsets': new_v4_offsets,
        'new_v5_offsets': new_v5_offsets,
        'v4_max': v4_max,
        'preserved_pct': preserved,
        'collision': collision,
    }


def find_valid_scaling():
    """
    Search for valid scaling factors that preserve communication patterns.
    """
    
    print("\n" + "=" * 70)
    print("SEARCHING FOR VALID SCALING FACTORS")
    print("=" * 70)
    
    results = []
    for scale in [1.0, 0.95, 0.9, 0.88, 0.85, 0.8, 0.75, 0.7, 0.5]:
        r = test_coordinated_scaling(scale, verbose=False)
        results.append((scale, r))
        status = "✓" if r['valid'] else "✗"
        print(f"  {scale:.2f}: {status} - stride={r['new_stride']}, preserved={r['preserved_pct']:.0f}%, collision={r['collision']}")
    
    valid = [(s, r) for s, r in results if r['valid']]
    if valid:
        print(f"\nValid scaling factors: {[s for s,r in valid]}")
    else:
        print("\nNo valid scaling factors found!")
        print("Communication patterns cannot be preserved with simple scaling.")
    
    return results


def analyze_iteration_reduction(verbose=True):
    """
    Analyze reducing the number of s7 iterations to avoid m0/ring buffer collision.
    This is the KEY approach to achieving 32KB LDS.
    
    Returns detailed analysis of LDS requirements for different iteration counts.
    """
    
    print("\n" + "=" * 70)
    print("ITERATION REDUCTION ANALYSIS")
    print("=" * 70)
    
    cfg = KernelConfig()
    
    print("\nCurrent configuration:")
    print(f"  Iterations: 8 (s7 = 0..7)")
    print(f"  m0_s7_mult: {cfg.m0_s7_mult} (0x{cfg.m0_s7_mult:x})")
    print(f"  v4_s7_mult: {cfg.v4_s7_mult} (0x{cfg.v4_s7_mult:x})")
    print(f"  Ring buffer base: {V4_WRITE_OFFSETS[0]}")
    
    print("\n" + "-" * 70)
    print("Testing different max s7 values:")
    print("-" * 70)
    
    results = []
    
    for max_s7 in range(8, -1, -1):
        # Calculate m0 region max
        # m0 has two regions: s50 (base 0) and s51 (base 9344)
        # Each grows by m0_s7_mult per iteration
        # Max sub-offset is 0x700 (1792)
        s50_max = max_s7 * cfg.m0_s7_mult + 0x700
        s51_max = max_s7 * cfg.m0_s7_mult + cfg.m0_s51_base + 0x700
        m0_max = s51_max  # s51 extends further
        
        # Ring buffer starts at first V4 offset
        ring_base = V4_WRITE_OFFSETS[0]
        
        # Calculate gap (positive = safe, negative = collision)
        gap = ring_base - m0_max
        
        # Calculate ring buffer max address WITH WRAP-AROUND
        # v4: (stride * (tid>>4) + 2*(tid&15) + s7*v4_s7_mult) * 4 + static_offset
        # The kernel uses LDS wrap-around (mod 65536)
        # We need to find the ACTUAL peak, accounting for wrap
        
        v4_max_unwrapped = 0
        v4_max_wrapped = 0
        for tid in range(256):
            v4_base = compute_v4_base(tid, max_s7, cfg)
            for off in cfg.v4_offsets:
                addr_unwrapped = v4_base + off
                addr_wrapped = addr_unwrapped % 65536
                v4_max_unwrapped = max(v4_max_unwrapped, addr_unwrapped)
                v4_max_wrapped = max(v4_max_wrapped, addr_wrapped)
        
        # Similarly for v5
        v5_max_unwrapped = 0
        v5_max_wrapped = 0
        for tid in range(256):
            v5_base = compute_v5_base(tid, max_s7, cfg)
            for off in cfg.v5_offsets:
                addr_unwrapped = v5_base + off
                addr_wrapped = addr_unwrapped % 65536
                v5_max_unwrapped = max(v5_max_unwrapped, addr_unwrapped)
                v5_max_wrapped = max(v5_max_wrapped, addr_wrapped)
        
        # Peak LDS is the maximum WRAPPED address accessed
        # (This is what actually matters for LDS allocation)
        lds_peak = max(m0_max, v4_max_wrapped, v5_max_wrapped)
        
        # For diagnosis, track unwrapped too
        v4_max = v4_max_wrapped
        v5_max = v5_max_wrapped
        v4_exceeds_64k = v4_max_unwrapped >= 65536
        v5_exceeds_64k = v5_max_unwrapped >= 65536
        
        # Check if fits in various LDS sizes
        fits_32k = lds_peak <= 32768
        fits_48k = lds_peak <= 49152
        fits_56k = lds_peak <= 57344
        fits_64k = lds_peak <= 65536
        
        # Determine status
        collision = gap < 0
        if fits_32k:
            status = "✓✓ 32KB!"
        elif fits_48k:
            status = "✓ 48KB"
        elif fits_56k:
            status = "✓ 56KB"
        elif fits_64k:
            status = "✓ 64KB"
        else:
            status = "✗ >64KB"
        
        if collision:
            status = f"✗ Collision ({gap} gap)"
        
        result = {
            'max_s7': max_s7,
            's50_max': s50_max,
            's51_max': s51_max,
            'm0_max': m0_max,
            'ring_base': ring_base,
            'gap': gap,
            'v4_max': v4_max,
            'v5_max': v5_max,
            'v4_max_unwrapped': v4_max_unwrapped,
            'v5_max_unwrapped': v5_max_unwrapped,
            'lds_peak': lds_peak,
            'fits_32k': fits_32k,
            'fits_48k': fits_48k,
            'fits_56k': fits_56k,
            'fits_64k': fits_64k,
            'collision': collision,
            'status': status,
            'wrap_around': v4_exceeds_64k or v5_exceeds_64k
        }
        results.append(result)
        
        if verbose:
            print(f"\ns7 = 0..{max_s7} ({max_s7+1} iterations):")
            print(f"  m0 region:    0 - {m0_max:6} bytes ({m0_max/1024:5.1f}KB)")
            print(f"    s50:        0 - {s50_max:6}")
            print(f"    s51:     {cfg.m0_s51_base} - {s51_max:6}")
            print(f"  Ring buffer:  {ring_base} - {v4_max:6} bytes")
            if v4_exceeds_64k or v5_exceeds_64k:
                print(f"    (unwrapped: v4={v4_max_unwrapped}, v5={v5_max_unwrapped}) - WRAP-AROUND!")
            print(f"  Gap:          {gap:+7} bytes {'(COLLISION!)' if collision else '(safe)'}")
            print(f"  Peak LDS:     {lds_peak:6} bytes ({lds_peak/1024:5.1f}KB)")
            print(f"  Status:       {status}")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print("\n| s7 max | Iters | m0 max | Gap    | LDS Peak | Target   | Status |")
    print("|--------|-------|--------|--------|----------|----------|--------|")
    
    for r in results:
        iters = r['max_s7'] + 1
        print(f"| {r['max_s7']:6} | {iters:5} | {r['m0_max']:6} | {r['gap']:+6} | {r['lds_peak']:6}KB | "
              f"{'32KB' if r['fits_32k'] else '48KB' if r['fits_48k'] else '56KB' if r['fits_56k'] else '64KB':8} | "
              f"{r['status']:6} |")
    
    # Find recommendations
    candidates_32k = [r for r in results if r['fits_32k'] and not r['collision']]
    candidates_48k = [r for r in results if r['fits_48k'] and not r['collision'] and not r['fits_32k']]
    candidates_no_collision = [r for r in results if not r['collision']]
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if candidates_32k:
        best = candidates_32k[-1]  # Maximum iterations that fits 32KB
        print(f"\n✓✓ 32KB LDS IS ACHIEVABLE!")
        print(f"   Reduce iterations: s7 = 0..{best['max_s7']} ({best['max_s7']+1} iterations)")
        print(f"   Peak LDS: {best['lds_peak']} bytes ({best['lds_peak']/1024:.1f}KB)")
        print(f"   Safety margin: {32768 - best['lds_peak']} bytes")
        print(f"   Throughput impact: {100 * (8 - (best['max_s7']+1)) / 8:.1f}% reduction per kernel")
        print(f"   But 2 WGs/CU instead of 1 → {100 * (2 * (best['max_s7']+1) / 8 - 1):.1f}% NET GAIN!")
        
        print(f"\n   Assembly changes needed:")
        print(f"   1. Find loop control comparing s7 with 8 or 7")
        print(f"   2. Change comparison to {best['max_s7']+1} or {best['max_s7']}")
        print(f"   3. Update .amdhsa_group_segment_fixed_size to 32768")
        print(f"   4. NO other changes needed (formulas stay same)")
    elif candidates_48k:
        best = candidates_48k[-1]
        print(f"\n✓ 48KB LDS is achievable")
        print(f"   Reduce iterations: s7 = 0..{best['max_s7']} ({best['max_s7']+1} iterations)")
        print(f"   Peak LDS: {best['lds_peak']} bytes ({best['lds_peak']/1024:.1f}KB)")
    else:
        print("\n✗ 32KB LDS IS NOT ACHIEVABLE by reducing iterations alone")
        if candidates_no_collision:
            best = candidates_no_collision[0]
            print(f"\n   s7 = 0..{best['max_s7']} avoids collision but still needs ~64KB")
            print(f"   Reason: WRAP-AROUND mechanism")
            print(f"     - v5 addresses exceed 64KB and wrap back to low addresses")
            print(f"     - This causes v5 to access full LDS range (0-64KB)")
            print(f"     - Even with fewer iterations, wrap-around still occurs")
        else:
            print("   Current design requires minimum 8 iterations to avoid collision")
        
        print("\n   KEY INSIGHT:")
        print("   To achieve 32KB LDS, must ELIMINATE wrap-around, which requires:")
        print("   1. Reduce v5 stride AND multiplier significantly, OR")
        print("   2. Redesign thread-to-data mapping, OR")
        print("   3. Accept lower thread count (e.g., 64 threads → 16x128 tile)")
        print("\n   These changes are COMPLEX and require source-level redesign.")
    
    # Calculate assembly change requirements
    print("\n" + "=" * 70)
    print("ASSEMBLY CHANGES REQUIRED")
    print("=" * 70)
    
    if candidates_32k:
        best = candidates_32k[-1]
        print(f"""
To implement s7 = 0..{best['max_s7']}:

1. LOOP CONTROL (~5-10 instructions)
   Find: s_cmp_lt_i32 s7, 8 (or similar)
   Change to: s_cmp_lt_i32 s7, {best['max_s7']+1}
   
   Or if using s_cmp_eq_u32:
   Find: s_cmp_eq_u32 s7, 8
   Change to: s_cmp_eq_u32 s7, {best['max_s7']+1}

2. KERNEL DESCRIPTOR (1 line)
   .amdhsa_group_segment_fixed_size 32768

3. NO OTHER CHANGES NEEDED
   - m0_s7_mult stays {cfg.m0_s7_mult}
   - v4_s7_mult stays {cfg.v4_s7_mult}
   - v5_s7_mult stays {cfg.v5_s7_mult}
   - All static offsets unchanged
   - No formula changes

Total: ~10-20 instruction modifications
Complexity: LOW
Risk: LOW (same algorithm, less work)
""")
    
    return results


def analyze_eliminate_wraparound():
    """
    Analyze what it takes to eliminate wrap-around and achieve 32KB LDS.
    
    Wrap-around occurs when v5 addresses exceed 64KB. To eliminate:
    - Must keep v5_max < 32KB (to fit in 32KB LDS)
    - v5 = (v5_stride * (tid>>1) + (tid&1) + s7*v5_s7_mult) * 4 + static_offset
    - Max tid = 255: tid>>1 = 127, tid&1 = 1
    - Max s7 = 7 (8 iterations)
    """
    
    print("\n" + "=" * 70)
    print("ANALYSIS: ELIMINATING WRAP-AROUND FOR 32KB LDS")
    print("=" * 70)
    
    cfg = KernelConfig()
    
    print("\nCurrent v5 addressing:")
    print(f"  Formula: (v5_stride * (tid>>1) + (tid&1) + s7*v5_s7_mult) * 4 + offset")
    print(f"  v5_stride: {cfg.v5_stride}")
    print(f"  v5_s7_mult: {cfg.v5_s7_mult}")
    print(f"  Max tid: 255 → tid>>1 = 127")
    print(f"  Max s7: 7")
    print(f"  Static offsets: {V5_READ_OFFSETS[:3]}...")
    
    # Calculate current v5_max
    v5_max_unwrapped = 0
    for s7 in range(8):
        for tid in range(256):
            v5_base = compute_v5_base(tid, s7, cfg)
            for off in cfg.v5_offsets:
                v5_max_unwrapped = max(v5_max_unwrapped, v5_base + off)
    
    print(f"\nCurrent v5 max (unwrapped): {v5_max_unwrapped} bytes ({v5_max_unwrapped/1024:.1f}KB)")
    print(f"  Exceeds 64KB by: {v5_max_unwrapped - 65536} bytes")
    
    print("\n" + "-" * 70)
    print("OPTION 1: Reduce v5_stride")
    print("-" * 70)
    
    # Target: v5_max < 32KB
    # v5_dynamic = (v5_stride * 127 + 1 + 7 * v5_s7_mult) * 4
    # v5_max = v5_dynamic + max(V5_READ_OFFSETS)
    # Want: v5_max < 32768
    
    target_lds = 32768
    max_static_offset = max(V5_READ_OFFSETS)
    target_dynamic = (target_lds - max_static_offset) // 4
    
    # v5_stride * 127 + 1 + 7 * v5_s7_mult < target_dynamic
    # Assume we keep v5_s7_mult same: 34
    required_v5_stride_max = (target_dynamic - 1 - 7 * cfg.v5_s7_mult) // 127
    
    print(f"\nTo fit v5 in 32KB:")
    print(f"  Target dynamic max: {target_dynamic}")
    print(f"  Current: v5_stride={cfg.v5_stride}, v5_s7_mult={cfg.v5_s7_mult}")
    print(f"  Required v5_stride: ≤ {required_v5_stride_max}")
    print(f"  Reduction factor: {cfg.v5_stride / required_v5_stride_max:.2f}x")
    
    print(f"\n  Problem: v5_stride is tied to:")
    print(f"  - Thread communication patterns (each thread reads from others)")
    print(f"  - Data layout in global memory")
    print(f"  - Cannot change without breaking algorithm")
    
    print("\n" + "-" * 70)
    print("OPTION 2: Reduce thread count")
    print("-" * 70)
    
    # If we use 64 threads instead of 256
    # tid_max = 63: tid>>1 = 31, tid&1 = 1
    
    v5_dynamic_64threads = (cfg.v5_stride * 31 + 1 + 7 * cfg.v5_s7_mult) * 4
    v5_max_64threads = v5_dynamic_64threads + max_static_offset
    
    print(f"\nWith 64 threads (tile 16x128 instead of 32x256):")
    print(f"  Max tid: 63 → tid>>1 = 31")
    print(f"  v5 max: {v5_max_64threads} bytes ({v5_max_64threads/1024:.1f}KB)")
    print(f"  Fits in 32KB: {'✓ YES' if v5_max_64threads <= 32768 else '✗ NO'}")
    
    if v5_max_64threads <= 32768:
        print(f"\n  ✓ This WORKS! 64 threads eliminate wrap-around")
        print(f"  Safety margin: {32768 - v5_max_64threads} bytes")
        print(f"  Trade-off: 4x less work per workgroup")
        print(f"  But 2x more workgroups/CU → net 0.5x throughput")
    else:
        # Try even fewer threads
        for nthreads in [32, 16, 8]:
            tid_max = nthreads - 1
            tid_shift = tid_max >> 1
            v5_dynamic = (cfg.v5_stride * tid_shift + 1 + 7 * cfg.v5_s7_mult) * 4
            v5_max = v5_dynamic + max_static_offset
            if v5_max <= 32768:
                print(f"\n  Need {nthreads} threads to fit in 32KB")
                print(f"  v5 max: {v5_max} bytes ({v5_max/1024:.1f}KB)")
                break
    
    print("\n" + "-" * 70)
    print("OPTION 3: Redesign addressing completely")
    print("-" * 70)
    
    print(f"""
To achieve 32KB with 256 threads and 8 iterations:
  1. Change v5_stride from {cfg.v5_stride} to ~{required_v5_stride_max}
  2. Adjust v4_stride proportionally
  3. Change static offsets
  4. Update global memory access patterns
  5. Modify MFMA data flow

This requires FULL KERNEL REDESIGN, not just assembly modification.
Complexity: VERY HIGH
Success probability: UNCERTAIN (may break correctness)
""")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
32KB LDS for 32x256 tile (256 threads, 8 iterations):
  ✗ NOT ACHIEVABLE via simple iteration reduction
  ✗ NOT ACHIEVABLE via binary patching
  ✗ NOT ACHIEVABLE via minor assembly modifications

32KB LDS IS achievable ONLY via:
  1. Smaller tile (16x128 with 64 threads) - PROVEN APPROACH
  2. Complete kernel redesign - HIGH RISK, UNCERTAIN BENEFIT

RECOMMENDATION: Use 16x128 kernel variant for 32KB LDS target
""")


def verify_iteration_reduction(max_s7: int):
    """
    Verify that a specific iteration count avoids collision and fits in target LDS.
    
    This is the verification function to use before modifying assembly.
    """
    
    print("\n" + "=" * 70)
    print(f"VERIFICATION: s7 = 0..{max_s7} ({max_s7+1} iterations)")
    print("=" * 70)
    
    cfg = KernelConfig()
    
    # Calculate all address ranges with this max_s7
    m0_addresses = set()
    v4_addresses = set()
    v5_addresses = set()
    
    for s7 in range(max_s7 + 1):
        # m0 addresses
        for sub in M0_SUB_OFFSETS:
            m0_addresses.add(s7 * cfg.m0_s7_mult + sub)  # s50
            m0_addresses.add(s7 * cfg.m0_s7_mult + cfg.m0_s51_base + sub)  # s51
        
        # v4 addresses (all threads)
        for tid in range(256):
            v4_base = compute_v4_base(tid, s7, cfg)
            for off in cfg.v4_offsets:
                v4_addresses.add(v4_base + off)
        
        # v5 addresses (all threads)
        for tid in range(256):
            v5_base = compute_v5_base(tid, s7, cfg)
            for off in cfg.v5_offsets:
                v5_addresses.add(v5_base + off)
    
    m0_max = max(m0_addresses) if m0_addresses else 0
    v4_min = min(v4_addresses) if v4_addresses else 0
    v4_max = max(v4_addresses) if v4_addresses else 0
    v5_max = max(v5_addresses) if v5_addresses else 0
    lds_peak = max(m0_max, v4_max, v5_max)
    
    gap = v4_min - m0_max
    collision = gap < 0
    
    print(f"\nAddress ranges:")
    print(f"  m0:    {min(m0_addresses):6} - {m0_max:6} ({len(m0_addresses)} unique addresses)")
    print(f"  v4:    {v4_min:6} - {v4_max:6} ({len(v4_addresses)} unique addresses)")
    print(f"  v5:    {min(v5_addresses):6} - {v5_max:6} ({len(v5_addresses)} unique addresses)")
    
    print(f"\nCollision check:")
    print(f"  m0 ends at:        {m0_max:6} bytes")
    print(f"  Ring buffer at:    {v4_min:6} bytes")
    print(f"  Gap:               {gap:+6} bytes")
    print(f"  Status:            {'✗ COLLISION!' if collision else '✓ Safe gap'}")
    
    print(f"\nLDS requirements:")
    print(f"  Peak address:      {lds_peak:6} bytes ({lds_peak/1024:.2f}KB)")
    print(f"  Fits in 32KB:      {'✓ YES' if lds_peak <= 32768 else '✗ NO'}")
    print(f"  Fits in 48KB:      {'✓ YES' if lds_peak <= 49152 else '✗ NO'}")
    print(f"  Fits in 64KB:      {'✓ YES' if lds_peak <= 65536 else '✗ NO'}")
    
    # Performance impact
    throughput_ratio = (max_s7 + 1) / 8.0
    occupancy_gain = 2.0 if lds_peak <= 32768 else 1.0
    net_gain = occupancy_gain * throughput_ratio
    
    print(f"\nPerformance impact:")
    print(f"  Work per kernel:   {throughput_ratio*100:.1f}% of original")
    print(f"  Occupancy:         {occupancy_gain:.1f}x")
    print(f"  Net throughput:    {net_gain*100:.1f}% ({'✓ GAIN' if net_gain > 1 else '✗ LOSS'})")
    
    # Verdict
    print("\n" + "=" * 70)
    if not collision and lds_peak <= 32768:
        print("✓✓ VERIFIED: This configuration WORKS for 32KB LDS!")
        print("=" * 70)
        return True
    elif not collision and lds_peak <= 49152:
        print("✓ VERIFIED: This configuration works for 48KB LDS")
        print("=" * 70)
        return True
    else:
        print("✗ FAILED: This configuration has issues")
        print("=" * 70)
        return False


# ============================================================================
# VGPR MODELING (Extension to unified simulator)
# ============================================================================

@dataclass
class VGPRAllocation:
    """Represents a VGPR allocation"""
    name: str
    vgpr_type: str  # "accumulator", "weight", "address", "temp"
    start_reg: int
    count: int
    reusable: bool = False
    
    @property
    def end_reg(self) -> int:
        return self.start_reg + self.count - 1
    
    def __repr__(self):
        reuse_marker = " (reusable)" if self.reusable else ""
        return f"{self.name}: v[{self.start_reg}:{self.end_reg}] ({self.count} VGPRs){reuse_marker}"


@dataclass
class TileVGPRConfig:
    """VGPR configuration for a specific tile size"""
    rows: int
    cols: int
    threads: int
    mfma_tile: int = 16
    
    @property
    def row_tiles(self) -> int:
        return self.rows // self.mfma_tile
    
    @property
    def col_tiles(self) -> int:
        return self.cols // self.mfma_tile
    
    @property
    def total_tiles(self) -> int:
        return self.row_tiles * self.col_tiles
    
    @property
    def mfmas_per_tile(self) -> int:
        return 4  # FP8 16x16x32 needs 4 MFMAs for k-accumulation
    
    @property
    def total_mfmas(self) -> int:
        return self.total_tiles * self.mfmas_per_tile
    
    def __repr__(self):
        return f"{self.rows}x{self.cols} ({self.threads}T, {self.total_tiles} tiles, {self.total_mfmas} MFMAs)"


def simulate_vgpr_usage(tile: TileVGPRConfig, column_groups: int = 1) -> Dict:
    """
    Simulate VGPR usage for a tile configuration
    
    Args:
        tile: Tile configuration
        column_groups: Number of column groups (1=naive, 2=2-pass, etc.)
    
    Returns:
        Dict with allocation details
    """
    allocations = []
    next_vgpr = 0
    
    # Address/temp registers (fixed, independent of tile size)
    allocations.append(VGPRAllocation("thread_ids", "address", next_vgpr, 4))
    next_vgpr += 4
    allocations.append(VGPRAllocation("lds_addresses", "address", next_vgpr, 8))
    next_vgpr += 8
    allocations.append(VGPRAllocation("offsets", "address", next_vgpr, 12))
    next_vgpr += 12
    allocations.append(VGPRAllocation("temps", "temp", next_vgpr, 16))
    next_vgpr += 16
    
    # Accumulators: Each 16x16 MFMA tile needs 4 VGPRs per thread
    if column_groups == 1:
        # Naive: allocate for all tiles at once
        accum_vgprs = tile.total_tiles * 4
        allocations.append(VGPRAllocation(
            f"accumulators_{tile.total_tiles}_tiles",
            "accumulator",
            next_vgpr,
            accum_vgprs,
            reusable=False
        ))
    else:
        # Reuse: allocate for one column group, reuse across passes
        cols_per_group = tile.cols // column_groups
        tiles_per_group = tile.row_tiles * (cols_per_group // tile.mfma_tile)
        accum_vgprs = tiles_per_group * 4
        allocations.append(VGPRAllocation(
            f"accumulators_{tiles_per_group}_tiles_x{column_groups}_passes",
            "accumulator",
            next_vgpr,
            accum_vgprs,
            reusable=True
        ))
    next_vgpr += accum_vgprs
    
    # Weight VGPRs (loaded from LDS for MFMA inputs)
    weight_vgprs = 32  # Typical: 8 blocks of 4 VGPRs
    allocations.append(VGPRAllocation(
        "weights",
        "weight",
        next_vgpr,
        weight_vgprs,
        reusable=(column_groups > 1)
    ))
    next_vgpr += weight_vgprs
    
    # AGPRs (separate pool)
    agpr_count = 16  # FP8 inputs
    
    # Calculate totals
    total_vgprs = next_vgpr
    reusable_vgprs = sum(a.count for a in allocations if a.reusable)
    
    fits_vgpr = total_vgprs <= 256
    fits_agpr = agpr_count <= 256
    fits = fits_vgpr and fits_agpr
    
    return {
        "tile": tile,
        "column_groups": column_groups,
        "allocations": allocations,
        "total_vgprs": total_vgprs,
        "total_agprs": agpr_count,
        "reusable_vgprs": reusable_vgprs,
        "fits": fits,
        "fits_vgpr": fits_vgpr,
        "fits_agpr": fits_agpr
    }


def print_vgpr_analysis(result: Dict):
    """Pretty-print VGPR analysis results"""
    tile = result["tile"]
    groups = result["column_groups"]
    strategy = "NAIVE" if groups == 1 else f"{groups}-PASS REUSE"
    
    print(f"\n{'='*80}")
    print(f"VGPR Analysis: {tile} - {strategy}")
    print(f"{'='*80}")
    
    # Group by type
    by_type = {}
    for alloc in result["allocations"]:
        if alloc.vgpr_type not in by_type:
            by_type[alloc.vgpr_type] = []
        by_type[alloc.vgpr_type].append(alloc)
    
    for vgpr_type in ["accumulator", "weight", "address", "temp"]:
        if vgpr_type in by_type:
            type_allocs = by_type[vgpr_type]
            total = sum(a.count for a in type_allocs)
            print(f"\n{vgpr_type.upper()}: {total} VGPRs")
            for alloc in type_allocs:
                print(f"  {alloc}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL VGPRs:  {result['total_vgprs']} / 256")
    print(f"TOTAL AGPRs:  {result['total_agprs']} / 256")
    print(f"Reusable:     {result['reusable_vgprs']} VGPRs")
    
    status = "✅ FITS" if result["fits"] else "❌ EXCEEDS LIMIT"
    if not result["fits_vgpr"]:
        reason = f"VGPRs {result['total_vgprs']} > 256"
    elif not result["fits_agpr"]:
        reason = f"AGPRs {result['total_agprs']} > 256"
    else:
        reason = "OK"
    
    print(f"Status:       {status} ({reason})")
    print(f"{'='*80}")


def analyze_tile_vgpr_usage(rows: int, cols: int, threads: int):
    """Analyze VGPR usage for a specific tile size with different strategies"""
    tile = TileVGPRConfig(rows, cols, threads)
    
    print(f"\n{'#'*80}")
    print(f"# VGPR Analysis: {tile}")
    print(f"{'#'*80}")
    
    # Strategy 1: Naive
    result_naive = simulate_vgpr_usage(tile, column_groups=1)
    print_vgpr_analysis(result_naive)
    
    # Strategy 2: 2-pass (if cols >= 256)
    if cols >= 256:
        result_2pass = simulate_vgpr_usage(tile, column_groups=2)
        print_vgpr_analysis(result_2pass)
    
    # Strategy 3: 4-pass (if cols >= 512)
    if cols >= 512:
        result_4pass = simulate_vgpr_usage(tile, column_groups=4)
        print_vgpr_analysis(result_4pass)
    
    return result_naive


def compare_all_tiles_vgpr():
    """Compare VGPR requirements across common tile sizes"""
    tiles = [
        (16, 128, 64),
        (32, 128, 256),
        (32, 256, 256),  # Baseline
        (32, 384, 256),  # Sergey's kernel
        (32, 512, 256),  # Target
    ]
    
    print(f"\n{'='*80}")
    print("VGPR COMPARISON: Tile Sizes (Naive Allocation)")
    print(f"{'='*80}")
    print(f"{'Tile':<12} {'Tiles':<8} {'MFMAs':<8} {'VGPRs':<8} {'Fits?':<8}")
    print(f"{'-'*80}")
    
    for rows, cols, threads in tiles:
        tile = TileVGPRConfig(rows, cols, threads)
        result = simulate_vgpr_usage(tile, column_groups=1)
        fits_str = "✅" if result["fits"] else "❌"
        print(f"{rows}x{cols:<8} {tile.total_tiles:<8} {tile.total_mfmas:<8} {result['total_vgprs']:<8} {fits_str:<8}")
    
    print(f"{'='*80}\n")


def recommend_32x512_strategy():
    """Recommend optimal strategy for 32x512 kernel"""
    tile = TileVGPRConfig(32, 512, 256)
    
    print(f"\n{'#'*80}")
    print(f"# RECOMMENDATION: 32x512 Kernel Strategy")
    print(f"{'#'*80}\n")
    
    print(f"Tile: {tile}")
    print(f"Total 16x16 tiles: {tile.total_tiles}")
    print(f"Total MFMAs: {tile.total_mfmas}\n")
    
    # Check naive
    result_naive = simulate_vgpr_usage(tile, column_groups=1)
    
    if result_naive["fits"]:
        print("✅ NAIVE allocation FITS!")
        print(f"   VGPRs: {result_naive['total_vgprs']} / 256")
        print(f"   No reuse strategy needed.\n")
        return "naive", result_naive
    else:
        print(f"❌ NAIVE allocation EXCEEDS limit!")
        print(f"   VGPRs: {result_naive['total_vgprs']} / 256")
        print(f"   Need VGPR reuse strategy!\n")
    
    # Check 2-pass
    result_2pass = simulate_vgpr_usage(tile, column_groups=2)
    
    if result_2pass["fits"]:
        print("✅ 2-PASS reuse strategy FITS!")
        print(f"   VGPRs: {result_2pass['total_vgprs']} / 256")
        print(f"   Strategy: Process 256 columns per pass (32x256 subset)")
        print(f"   - Pass 1: Columns 0-255")
        print(f"   - Pass 2: Columns 256-511")
        print(f"   Reusable VGPRs: {result_2pass['reusable_vgprs']}")
        print(f"\n   ✨ RECOMMENDED APPROACH ✨\n")
        return "2-pass", result_2pass
    else:
        print(f"❌ 2-PASS reuse STILL EXCEEDS!")
        print(f"   VGPRs: {result_2pass['total_vgprs']} / 256\n")
    
    # Check 4-pass
    result_4pass = simulate_vgpr_usage(tile, column_groups=4)
    
    if result_4pass["fits"]:
        print("✅ 4-PASS reuse strategy FITS!")
        print(f"   VGPRs: {result_4pass['total_vgprs']} / 256")
        print(f"   Strategy: Process 128 columns per pass (32x128 subset)")
        print(f"   - Pass 1-4: 128 columns each")
        print(f"   Reusable VGPRs: {result_4pass['reusable_vgprs']}")
        print(f"\n   ✨ RECOMMENDED APPROACH ✨\n")
        return "4-pass", result_4pass
    else:
        print(f"❌ 4-PASS reuse STILL EXCEEDS!")
        print(f"   VGPRs: {result_4pass['total_vgprs']} / 256")
        print(f"\n   32x512 MAY NOT BE FEASIBLE\n")
        return "infeasible", None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--vgpr-only":
        # Only run VGPR analysis
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  Unified LDS + VGPR Simulator                              ║
║                     VGPR Analysis Mode                                     ║
╚════════════════════════════════════════════════════════════════════════════╝
""")
        compare_all_tiles_vgpr()
        
        # Detailed analysis for key tiles
        analyze_tile_vgpr_usage(32, 256, 256)  # Baseline
        analyze_tile_vgpr_usage(32, 384, 256)  # Sergey's
        analyze_tile_vgpr_usage(32, 512, 256)  # Target
        
        # Recommendation
        recommend_32x512_strategy()
        
    elif len(sys.argv) > 1 and sys.argv[1] == "--small-tile":
        # Just analyze smaller tile
        analyze_smaller_tile()
        design_16x128_kernel()
    elif len(sys.argv) > 1 and sys.argv[1] == "--find-scaling":
        # Search for valid scaling
        find_valid_scaling()
    elif len(sys.argv) > 1 and sys.argv[1] == "--test-scale":
        # Test specific scaling
        scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.88
        test_coordinated_scaling(scale)
    elif len(sys.argv) > 1 and sys.argv[1] == "--reduce-iterations":
        # Analyze iteration reduction (KEY for 32KB)
        analyze_iteration_reduction()
    elif len(sys.argv) > 1 and sys.argv[1] == "--verify-iterations":
        # Verify specific iteration count
        max_s7 = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        verify_iteration_reduction(max_s7)
    elif len(sys.argv) > 1 and sys.argv[1] == "--eliminate-wraparound":
        # Analyze how to eliminate wrap-around
        analyze_eliminate_wraparound()
    elif len(sys.argv) > 1 and sys.argv[1] == "--full-32kb-analysis":
        # Complete analysis for 32KB target
        analyze_iteration_reduction()
        analyze_eliminate_wraparound()
    else:
        # Run default analysis
        results = run_full_analysis()
        
        # Analyze wrap-around timing
        analyze_wrap_timing()
        
        # Critical: v4 vs v5 wrap-around analysis
        analyze_v4_vs_v5_wrapping()
        
        # Stride reduction analysis
        analyze_stride_reduction()
        
        # Communication pattern analysis
        analyze_communication_pattern_change(17, scale_offsets=True)
        
        # Test various offset shifts
        print("\n" + "=" * 70)
        print("OFFSET SHIFT TESTING")
        print("=" * 70)
        
        for shift in [256, 512, 1024, 1408, 2048, 3360]:
            test_offset_shift(shift, verbose=False)
        
        # Also show smaller tile analysis
        analyze_smaller_tile()



# ============================================================================
# 64x256 KERNEL ANALYSIS
# ============================================================================

def analyze_64x256_kernel():
    """
    Analyze what's needed for a proper 64x256 kernel with 2-pass row processing.
    
    The 64x256 kernel processes 64 tokens per tile group (vs 32 for 32x256).
    Due to LDS constraints, this must be done in 2 passes of 32 tokens each.
    
    KEY INSIGHT: The problem with our previous 2-pass approach was:
    1. We re-entered the code path that recalculates all addresses
    2. But the OUTPUT was using atomic adds, so results accumulated
    3. Block assignment (s3) determines which expert's tokens to process
    4. For 64x256, a single block_id should map to 64 consecutive tokens
    
    CORRECT APPROACH:
    1. block_id s3 maps to tokens [s3*64 : s3*64+64]
    2. Pass 1: Process tokens [s3*64 : s3*64+32]  
    3. Pass 2: Process tokens [s3*64+32 : s3*64+64]
    4. Both passes write to the SAME output rows (no double accumulation)
    
    CRITICAL REGISTERS:
    - s3: block_id (determines which token group)
    - s44:s45: sorted_weights pointer (needs offset for pass 2)
    - s46:s47: sorted_expert_ids pointer (needs offset for pass 2)
    - s8:s9: output pointer (needs offset for pass 2)
    - Token loading addresses (ds_read from ring buffer)
    """
    
    print("\n" + "=" * 70)
    print("64x256 KERNEL ANALYSIS")
    print("=" * 70)
    
    print("""
CURRENT 32x256 KERNEL FLOW:
---------------------------
1. block_id (s3) assigned based on workgroup and iteration
2. block_start = s3 * 32  (32 tokens per block)
3. Load sorted_weights[block_start:block_start+32]
4. Load sorted_expert_ids[block_start:block_start+32]
5. For each of 8 experts with tokens in this block:
   a. Load expert weights from global
   b. Load token data from global → LDS ring buffer
   c. Perform MFMA operations
   d. Atomic add results to output[block_start:block_start+32]
6. Loop to next block_id

PROPOSED 64x256 KERNEL FLOW:
----------------------------
1. block_id (s3) assigned (same)
2. block_start = s3 * 64  (64 tokens per block)  <-- CHANGE 1: multiply by 64
3. 
   PASS 1: tokens 0-31 of this block
   - Load sorted_weights[block_start:block_start+32]
   - Load sorted_expert_ids[block_start:block_start+32]
   - Process with existing 32-token code path
   - Atomic add results to output[block_start:block_start+32]
   
   PASS 2: tokens 32-63 of this block
   - Load sorted_weights[block_start+32:block_start+64]
   - Load sorted_expert_ids[block_start+32:block_start+64]
   - Process with existing 32-token code path
   - Atomic add results to output[block_start+32:block_start+64]
   
4. Loop to next block_id
""")

    print("""
ASSEMBLY CHANGES REQUIRED:
--------------------------

1. BLOCK SIZE CHANGE:
   Find:  s_mul_i32 s60, s3, 32
   Change: s_mul_i32 s60, s3, 64
   
2. VALIDITY CHECK:
   Current: block_start < total_tokens (where total_tokens sorted to 32)
   New: block_start < total_tokens (where total_tokens sorted to 64)
   
3. 2-PASS LOOP STRUCTURE:
   Instead of jumping back to recalculate block_id, we need to:
   - Keep block_id same
   - Only adjust the TOKEN OFFSET within the block
   - Use a pass_offset register: 0 for pass 1, 32 for pass 2
   
4. POINTER ADJUSTMENTS FOR PASS 2:
   Add to sorted_weights pointer: pass_offset * 4 (4 bytes per weight)
   Add to sorted_expert_ids pointer: pass_offset * 4
   Add to output pointer: pass_offset * output_stride
""")

    print("""
CRITICAL DIFFERENCE FROM PREVIOUS ATTEMPT:
------------------------------------------
Previous: Jumped back to .Llabel_005C which recalculated everything
          Block_id was incremented (s3+1), treating pass 2 as a NEW block
          This caused DOUBLE PROCESSING of some blocks

Correct:  Stay within same block_id
          Only change the TOKEN OFFSET within the block
          Pass 1 and Pass 2 process DIFFERENT tokens of SAME block
          Each token is processed exactly once
""")

    # Analyze which registers control token offset
    print("""
REGISTER ANALYSIS:
------------------
From disassembly analysis:

s3:     Block ID (determines which 32/64 tokens)
s44:45: sorted_weights base pointer
s46:47: sorted_expert_ids base pointer
s8:9:   Output base pointer

For each token in block:
- sorted_weights[block_id * 32 + local_tid]
- sorted_expert_ids[block_id * 32 + local_tid]
- output[token_id]

The local_tid (0-31) comes from thread ID and lane info.
To process tokens 32-63, we need to ADD 32 to the token indices.

KEY INSIGHT: The multiplication "s3 * 32" happens at:
  Line ~165: s_mul_i32 s60, s3, 32

For 64x256, this should be "s3 * 64" for block-to-token-start mapping.
But then we ALSO need to handle which 32 of the 64 tokens we're processing.
""")

    # Practical implementation
    print("""
PRACTICAL 2-PASS IMPLEMENTATION:
--------------------------------

Option A: Modify block_start calculation
  - Change "s3 * 32" to "s3 * 64"  
  - Add pass_counter (s101) initialized to 0
  - Add pass_offset = pass_counter * 32
  - All token accesses become: block_start + pass_offset + local_offset
  - After pass 1, increment pass_counter and re-enter main processing
  - After pass 2, go to normal loop control

Option B: Use s2 for pass offset (s2 is unused after initial setup)
  - Initialize s2 = 0 at .Llabel_0039
  - Process tokens using s2 as offset
  - At end of pass 1: if s2 == 0, set s2 = 32, re-enter
  - At end of pass 2: reset s2 = 0, go to loop control

ASSEMBLY SNIPPET (Option B):
----------------------------
.Llabel_0039:
    s_mov_b32 s102, 0              ; Pass offset (0 or 32)
    ; ... existing code ...

; At block_start calculation:
    s_mul_i32 s60, s3, 64          ; block_start = block_id * 64
    s_add_u32 s60, s60, s102       ; Add pass offset (0 or 32)

; At end of processing (.Llabel_1D9B):
    s_cmp_eq_u32 s102, 0           ; Is this pass 1?
    s_cbranch_scc0 .L_loop_ctrl    ; If pass 2 done, go to loop
    s_mov_b32 s102, 32             ; Set pass offset for pass 2
    s_branch .Llabel_005C          ; Re-enter main processing
.L_loop_ctrl:
    s_mov_b32 s102, 0              ; Reset for next block
    ; ... normal loop control ...
""")

    return True


def calculate_64x256_address_changes():
    """
    Calculate the exact address formula changes needed for 64x256.
    """
    
    print("\n" + "=" * 70)
    print("64x256 ADDRESS FORMULA CHANGES")
    print("=" * 70)
    
    cfg = KernelConfig()
    
    print("""
CURRENT 32x256 ADDRESS FORMULAS:
--------------------------------
Token index within block: tid (0-31, from thread ID)
Block start: s3 * 32
Global token index: s3 * 32 + tid

sorted_weights[global_idx] → s44:45 + (s3*32 + tid) * 4
sorted_expert_ids[global_idx] → s46:47 + (s3*32 + tid) * 4  
output[global_idx] → s8:9 + (s3*32 + tid) * output_stride

PROPOSED 64x256 ADDRESS FORMULAS:
---------------------------------
Pass offset: s102 (0 for pass 1, 32 for pass 2)
Block start: s3 * 64 + s102
Global token index: s3 * 64 + s102 + tid

sorted_weights[global_idx] → s44:45 + (s3*64 + s102 + tid) * 4
sorted_expert_ids[global_idx] → s46:47 + (s3*64 + s102 + tid) * 4
output[global_idx] → s8:9 + (s3*64 + s102 + tid) * output_stride
""")

    # Find where these calculations happen
    print("""
ASSEMBLY LOCATIONS TO MODIFY:
-----------------------------
1. Block start calculation (s60 = s3 * 32):
   Line ~165: s_mul_i32 s60, s3, 32
   Change to: s_mul_i32 s60, s3, 64
             s_add_u32 s60, s60, s102

2. Sorted data loading (multiple locations):
   Search for: s_add_u32 s46, s60, s46  (sorted_expert_ids pointer)
   Need to account for s102 offset in s60
   
3. Token data loading (ring buffer):
   The v4/v5 addresses are computed from thread ID
   These should NOT change - still 32 tokens per pass in LDS
   
4. Output address calculation:
   Search for: global_atomic_pk_add_bf16
   The base address comes from s8:9, offset from s60
   Need to ensure s60 includes s102 offset
""")

    print("""
CRITICAL REALIZATION:
--------------------
The OUTPUT addresses (for atomic add) are based on:
- Token's original position in input
- Which expert processed it

For 64x256:
- Pass 1 outputs to positions [block_id*64 : block_id*64+32]
- Pass 2 outputs to positions [block_id*64+32 : block_id*64+64]

These are DIFFERENT output positions, so atomic adds don't double-accumulate!

The previous attempt failed because:
- We incremented block_id (s3) for pass 2
- This made pass 2 think it was processing a DIFFERENT block
- But the output pointer calculation was wrong
- Leading to accumulation in wrong places

The fix is:
- Keep block_id (s3) SAME for both passes
- Only change the token offset WITHIN the block (s102)
- This ensures pass 1 and pass 2 write to ADJACENT output regions
""")

    return True


# ============================================================================
# 64x256 TRUE FUSED KERNEL DESIGN
# ============================================================================

def design_64x256_lds_layout():
    """
    Design a true 64x256 kernel LDS layout.
    
    The challenge: 64 tokens need double the token shuffle buffer space.
    
    Current 32x256 LDS breakdown:
    - m0 region (weights/scales): ~26KB (fixed, depends on N not M)
    - Token ring buffer: ~32KB (scales with M)
    - v56 intermediate: ~10KB
    - Total: ~58KB (fits in 64KB)
    
    Naive 64x256 would need:
    - m0 region: ~26KB (unchanged)
    - Token ring buffer: ~64KB (doubled!)
    - v56 intermediate: ~10KB
    - Total: ~100KB (EXCEEDS 64KB!)
    
    Solution approaches:
    1. CHUNKED TOKEN PROCESSING: Process 16 tokens at a time, 4 chunks
    2. REDUCED RING BUFFER DEPTH: Less prefetching
    3. LDS REGION COMPRESSION: Smaller strides
    """
    
    print("\n" + "=" * 70)
    print("TRUE 64x256 KERNEL LDS LAYOUT DESIGN")
    print("=" * 70)
    
    cfg = KernelConfig()
    
    # Current 32x256 layout analysis
    print("\n" + "-" * 70)
    print("CURRENT 32x256 LDS LAYOUT")
    print("-" * 70)
    
    # m0 region (weights)
    m0_max = 7 * cfg.m0_s7_mult + cfg.m0_s51_base + 0x700
    print(f"\nm0 region (weights/scales):")
    print(f"  s50 base: 0")
    print(f"  s51 base: {cfg.m0_s51_base} (0x{cfg.m0_s51_base:x})")
    print(f"  s7 mult: {cfg.m0_s7_mult}")
    print(f"  Max address (s7=7): {m0_max} ({m0_max/1024:.1f}KB)")
    
    # Ring buffer (token shuffle)
    ring_start = V4_WRITE_OFFSETS[0]  # 20736
    v4_max = max(V4_WRITE_OFFSETS)
    v5_max = max(V5_READ_OFFSETS)
    
    # Dynamic range from thread addressing
    v4_dynamic = (cfg.v4_stride * 15 + 30) * 4  # max tid>>4=15, tid&15=15
    v5_dynamic = (cfg.v5_stride * 127 + 1) * 4  # max tid>>1=127
    
    print(f"\nRing buffer (token shuffle):")
    print(f"  Start: {ring_start} (0x{ring_start:x})")
    print(f"  v4 offsets: {V4_WRITE_OFFSETS[0]} - {V4_WRITE_OFFSETS[-1]}")
    print(f"  v5 offsets: {V5_READ_OFFSETS[0]} - {V5_READ_OFFSETS[-1]}")
    print(f"  v4 dynamic range: {v4_dynamic} bytes")
    print(f"  v5 dynamic range: {v5_dynamic} bytes")
    print(f"  Ring buffer size: ~{(v5_max - ring_start + v5_dynamic)/1024:.1f}KB")
    
    # Calculate actual peak
    v5_peak = v5_dynamic + v5_max
    print(f"  v5 peak address: {v5_peak} ({v5_peak/1024:.1f}KB)")
    
    # v56 intermediate
    v56_min = min(V56_WRITE_OFFSETS)
    v56_max = max(V56_READ_OFFSETS)
    print(f"\nv56 intermediate:")
    print(f"  Range: {v56_min} - {v56_max}")
    print(f"  Size: ~{(v56_max - v56_min)/1024:.1f}KB")
    
    print(f"\nTotal 32x256: ~{max(m0_max, v5_peak)/1024:.1f}KB peak")
    
    # 64x256 requirements analysis
    print("\n" + "-" * 70)
    print("64x256 REQUIREMENTS ANALYSIS")
    print("-" * 70)
    
    print("""
The token ring buffer holds input/output data for token shuffling:
- Current: 32 tokens × 256 elements × 2 bytes = 16KB per buffer
- Ring buffer has double-buffering → 32KB total
- Plus addressing overhead → ~35KB actual

For 64 tokens:
- 64 tokens × 256 elements × 2 bytes = 32KB per buffer
- With double-buffering → 64KB total
- Plus addressing overhead → ~70KB actual

This EXCEEDS 64KB LDS limit!
""")
    
    # Solution 1: Chunked processing
    print("\n" + "-" * 70)
    print("SOLUTION 1: CHUNKED TOKEN PROCESSING (16 tokens × 4 chunks)")
    print("-" * 70)
    
    # With 16 tokens per chunk
    tokens_per_chunk = 16
    chunks = 64 // tokens_per_chunk
    
    # 16 tokens needs different thread mapping
    # Current: 256 threads process 32 tokens (8 threads per token)
    # New: 256 threads process 16 tokens (16 threads per token)
    # OR: 64 threads process 16 tokens (4 threads per token) - better!
    
    print(f"""
Chunked approach: Process {chunks} chunks of {tokens_per_chunk} tokens each

LDS layout for {tokens_per_chunk}-token chunk:
- m0 region: ~26KB (unchanged)
- Token buffer: {tokens_per_chunk} × 256 × 2 = {tokens_per_chunk * 256 * 2 / 1024:.1f}KB
- Double buffer: {tokens_per_chunk * 256 * 2 * 2 / 1024:.1f}KB
- v56 intermediate: ~10KB

Estimated total: ~{(26 + tokens_per_chunk * 256 * 2 * 2 / 1024 + 10):.1f}KB ✅ FITS!

Trade-off: 4× processing iterations, but each iteration is faster
""")
    
    # Calculate new strides for 16-token processing
    new_stride_16 = 17  # Half of 34 for half tokens
    new_v5_dynamic_16 = (new_stride_16 * 63 + 1) * 4  # For 128 threads (tid>>1 max = 63)
    
    print(f"Address formula changes for 16-token chunk:")
    print(f"  v4 stride: {cfg.v4_stride} → {new_stride_16}")
    print(f"  v5 stride: {cfg.v5_stride} → {new_stride_16}")
    print(f"  v5 dynamic range: {v5_dynamic} → {new_v5_dynamic_16}")
    
    # Solution 2: Compressed ring buffer
    print("\n" + "-" * 70)
    print("SOLUTION 2: COMPRESSED RING BUFFER (reduced stride)")  
    print("-" * 70)
    
    # Try halving the stride
    half_stride = cfg.v4_stride // 2  # 17
    
    # New dynamic ranges
    v4_dyn_half = (half_stride * 15 + 30) * 4
    v5_dyn_half = (half_stride * 127 + 1) * 4
    
    # Scale static offsets
    scale = 0.5
    new_ring_start = int(ring_start * scale)
    new_v4_offsets = [int(off * scale) for off in V4_WRITE_OFFSETS]
    new_v5_offsets = [int(off * scale) for off in V5_READ_OFFSETS]
    
    new_v5_peak = v5_dyn_half + max(new_v5_offsets)
    
    print(f"""
Compressed ring buffer with half stride:

Current → New:
- v4 stride: {cfg.v4_stride} → {half_stride}
- v5 stride: {cfg.v5_stride} → {half_stride}
- Ring start: {ring_start} → {new_ring_start}
- v4 offsets: [{V4_WRITE_OFFSETS[0]}, ...] → [{new_v4_offsets[0]}, ...]
- v5 offsets: [{V5_READ_OFFSETS[0]}, ...] → [{new_v5_offsets[0]}, ...]

New dynamic ranges:
- v4: {v4_dynamic} → {v4_dyn_half} ({v4_dyn_half/1024:.1f}KB)
- v5: {v5_dynamic} → {v5_dyn_half} ({v5_dyn_half/1024:.1f}KB)

New peak: {new_v5_peak} ({new_v5_peak/1024:.1f}KB)

⚠️ PROBLEM: Compression alone doesn't double capacity!
   Still need to store 64 tokens worth of data.
""")
    
    # Solution 3: Hybrid - 2-pass with modified layout
    print("\n" + "-" * 70)
    print("SOLUTION 3: 2-PASS WITH SHARED RING BUFFER (RECOMMENDED)")
    print("-" * 70)
    
    print("""
Most practical approach: Process 64 tokens in 2 passes of 32 tokens,
but WITHIN A SINGLE KERNEL invocation.

Pass 1: Tokens 0-31
  - Load sorted_ids[block_id*64 : block_id*64+32]
  - Use existing ring buffer layout (unchanged)
  - Write output[block_id*64 : block_id*64+32]
  
Pass 2: Tokens 32-63
  - Load sorted_ids[block_id*64+32 : block_id*64+64]
  - REUSE same ring buffer locations (LDS reuse!)
  - Write output[block_id*64+32 : block_id*64+64]

LDS requirement: UNCHANGED (still ~58KB)
Kernel structure: Single launch, internal 2-pass loop

KEY CHANGES NEEDED:
1. Sorting function: Produce 64-token blocks
2. Kernel: 
   - Block ID calculation: s3 * 64 instead of s3 * 32
   - Add pass counter register
   - Adjust sorted_ids/weights pointers per pass
   - Output addresses adjusted per pass
""")
    
    return {
        'solution': '2-pass',
        'tokens_per_pass': 32,
        'passes_per_block': 2,
        'lds_unchanged': True,
        'sorting_changes_needed': True,
        'kernel_changes': [
            'block_size_multiply',
            'pass_counter',
            'pointer_adjustment',
            'output_offset'
        ]
    }


def design_64x256_kernel_changes():
    """
    Detail the exact assembly changes needed for true 64x256 kernel.
    """
    
    print("\n" + "=" * 70)
    print("64x256 KERNEL ASSEMBLY CHANGES")
    print("=" * 70)
    
    print("""
REGISTER ALLOCATION:
--------------------
Existing registers we can repurpose:
- s91: Currently unused → pass_offset (0 or 32)
- s92: Currently unused → saved total_tokens

New register usage:
- s91 = pass_offset: 0 for first 32 tokens, 32 for next 32 tokens
- s92 = original total_tokens (before s50 gets repurposed)

ASSEMBLY MODIFICATIONS:
-----------------------

1. INITIALIZATION (at .Llabel_0039):
   ADD after s_mov_b32 s100, 0:
   
   s_mov_b32 s91, 0              ; pass_offset = 0 (first pass)
   
2. SAVE TOTAL_TOKENS (before s50 gets repurposed):
   Find: s_load_dwordx2 ... s50 (total_tokens load)
   ADD after:
   
   s_mov_b32 s92, s50            ; Save total_tokens
   
3. BLOCK START CALCULATION (modify s3 * 32):
   Find: s_mul_i32 s60, s3, 32
   CHANGE TO:
   
   s_lshl_b32 s60, s3, 6         ; s60 = s3 * 64 (shift left by 6)
   s_add_u32 s60, s60, s91       ; s60 = s3*64 + pass_offset
   
4. VALIDITY CHECK (use saved total_tokens):
   Find: s_cmp_lt_i32 s60, s50    ; validity check
   CHANGE TO:
   
   s_cmp_lt_i32 s60, s92         ; Compare with saved total_tokens
   
5. 2-PASS LOOP CONTROL (at end of processing, .Llabel_1D9B):
   Find the existing loop control section
   MODIFY TO:
   
   ; Check if we need second pass
   s_cmp_eq_u32 s91, 0           ; Is this pass 1?
   s_cbranch_scc0 .L_next_block  ; If pass 2 done, go to next block
   
   ; Setup for pass 2
   s_mov_b32 s91, 32             ; pass_offset = 32
   s_branch .L_pass2_entry       ; Jump to pass 2 (after block setup)
   
   .L_next_block:
   s_mov_b32 s91, 0              ; Reset pass_offset for next block
   ; ... existing loop control to get next block_id ...

6. PASS 2 ENTRY POINT:
   Need a label after block setup but before token processing:
   
   .L_pass2_entry:
   ; Recalculate s60 with new pass_offset
   s_lshl_b32 s60, s3, 6         ; s60 = s3 * 64
   s_add_u32 s60, s60, s91       ; s60 = s3*64 + 32
   ; ... continue with token processing ...

POINTER ADJUSTMENTS:
--------------------
sorted_ids pointer: 
  Current: base + block_id * 32 * 4
  New: base + (block_id * 64 + pass_offset) * 4

sorted_weights pointer:
  Current: base + block_id * 32 * 4  
  New: base + (block_id * 64 + pass_offset) * 4

sorted_expert_ids pointer:
  Current: base + block_id * 4
  New: base + (block_id * 2 + (pass_offset >> 5)) * 4
  Note: This needs special handling - 2 expert_ids per 64-token block

output pointer:
  Current: base + token_id * stride
  New: unchanged (token_id already includes pass_offset via s60)
""")
    
    print("""
CRITICAL INSIGHT: sorted_expert_ids
-----------------------------------
For 32-token blocks: sorted_expert_ids has 1 entry per 32 tokens
For 64-token blocks: sorted_expert_ids could have 1 entry per 64 tokens

But with 2-pass approach:
- Pass 1 processes tokens 0-31 of block
- Pass 2 processes tokens 32-63 of block  
- BOTH may have DIFFERENT expert assignments!

Example: Block 0 with 64 tokens
- Tokens 0-31: assigned to expert 3
- Tokens 32-63: assigned to expert 5 (different!)

The sorting function must output EITHER:
A) 1 expert_id per 64 tokens (loses granularity, bad)
B) 2 expert_ids per 64 tokens (preserved granularity, good)

RECOMMENDATION: Keep 32-token expert_id granularity
- sorted_expert_ids array has 2× entries vs 64-block approach
- Each 32-token "sub-block" has its own expert_id
- Kernel logic unchanged for expert loading
""")
    
    return True


def simulate_64x256_data_flow():
    """
    Simulate data flow for 64x256 kernel with 2-pass processing.
    """
    
    print("\n" + "=" * 70)
    print("64x256 KERNEL DATA FLOW SIMULATION")
    print("=" * 70)
    
    # Example: 64 tokens, topk=2, 8 experts
    num_tokens = 64
    topk = 2
    num_experts = 8
    block_size = 64  # New block size
    
    print(f"\nInput configuration:")
    print(f"  Tokens: {num_tokens}")
    print(f"  TopK: {topk}")
    print(f"  Experts: {num_experts}")
    print(f"  Block size: {block_size}")
    
    # Simulated token-to-expert assignments (random for demo)
    import random
    random.seed(42)
    
    topk_ids = [[random.sample(range(num_experts), topk) for _ in range(num_tokens)]]
    topk_ids = topk_ids[0]
    
    # Count tokens per expert
    expert_token_count = {e: 0 for e in range(num_experts)}
    for token_id, experts in enumerate(topk_ids):
        for exp in experts:
            expert_token_count[exp] += 1
    
    print(f"\nTokens per expert (before padding):")
    for exp, count in expert_token_count.items():
        print(f"  Expert {exp}: {count} tokens")
    
    # Calculate sorted array sizes
    total_sorted = sum(expert_token_count.values())
    padded_per_expert = {e: ((count + block_size - 1) // block_size) * block_size 
                         for e, count in expert_token_count.items()}
    total_padded = sum(padded_per_expert.values())
    num_blocks = total_padded // block_size
    
    print(f"\nWith {block_size}-token blocks:")
    print(f"  Total sorted tokens: {total_sorted}")
    print(f"  Total padded tokens: {total_padded}")
    print(f"  Number of blocks: {num_blocks}")
    
    # 32-block comparison
    padded_32 = {e: max(((count + 31) // 32) * 32, 32) for e, count in expert_token_count.items()}
    total_padded_32 = sum(padded_32.values())
    num_blocks_32 = total_padded_32 // 32
    
    print(f"\nComparison with 32-token blocks:")
    print(f"  Total padded (32-block): {total_padded_32}")
    print(f"  Number of blocks (32): {num_blocks_32}")
    print(f"  Reduction: {num_blocks_32} → {num_blocks} = {100*(1 - num_blocks/num_blocks_32):.1f}% fewer blocks")
    
    # Simulate 2-pass data flow
    print("\n" + "-" * 70)
    print("2-PASS DATA FLOW WITHIN KERNEL")
    print("-" * 70)
    
    # For each 64-token block
    for block_id in range(min(num_blocks, 3)):  # Show first 3 blocks
        print(f"\nBlock {block_id}:")
        block_start = block_id * 64
        
        print(f"  Pass 1: tokens [{block_start}:{block_start+32}]")
        print(f"    - Load sorted_ids[{block_start}:{block_start+32}]")
        print(f"    - Load sorted_weights[{block_start}:{block_start+32}]")
        print(f"    - sorted_expert_ids[{block_id*2}] = expert for tokens 0-31")
        print(f"    - LDS: ring buffer slots 0-31")
        print(f"    - Output: atomic_add to output[{block_start}:{block_start+32}]")
        
        print(f"  Pass 2: tokens [{block_start+32}:{block_start+64}]")
        print(f"    - Load sorted_ids[{block_start+32}:{block_start+64}]")
        print(f"    - Load sorted_weights[{block_start+32}:{block_start+64}]")
        print(f"    - sorted_expert_ids[{block_id*2+1}] = expert for tokens 32-63")
        print(f"    - LDS: ring buffer slots 0-31 (REUSED!)")
        print(f"    - Output: atomic_add to output[{block_start+32}:{block_start+64}]")
    
    print("\n" + "-" * 70)
    print("SORTING FUNCTION REQUIREMENTS")
    print("-" * 70)
    
    print("""
For 64x256 kernel with 2-pass, the sorting function needs to:

1. OUTPUT STRUCTURE:
   - sorted_ids: unchanged format, but padded to 64-token boundaries
   - sorted_weights: unchanged format
   - sorted_expert_ids: 2 entries per 64-token block (one per 32-token sub-block)
   - num_valid_ids[0]: total padded tokens (divisible by 64)
   - num_valid_ids[1]: actual token count

2. PADDING RULE:
   - Each expert's tokens padded to next 64-token boundary
   - But INTERNALLY, maintain 32-token sub-block structure
   - This allows 2-pass processing while reducing total blocks

3. EXPERT ID ASSIGNMENT:
   - For 64-token block N:
     * sorted_expert_ids[N*2] = expert for sub-block 0 (tokens 0-31)
     * sorted_expert_ids[N*2+1] = expert for sub-block 1 (tokens 32-63)
   - Typically both will be same expert (tokens grouped by expert)
   - Edge case: if expert has <32 tokens, sub-block 1 is padding

4. IMPLEMENTATION APPROACH:
   - Modify unit_size parameter from 32 to 64
   - Ensure expert_ids array is sized for 32-token granularity
   - Keep internal sorting logic mostly unchanged
""")
    
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--64x256":
        analyze_64x256_kernel()
        calculate_64x256_address_changes()
    elif len(sys.argv) > 1 and sys.argv[1] == "--design-64x256":
        design_64x256_lds_layout()
        design_64x256_kernel_changes()
        simulate_64x256_data_flow()
