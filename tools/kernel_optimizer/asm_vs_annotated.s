// ============================================================================
// ANNOTATED AMDGPU KERNEL DISASSEMBLY
// ============================================================================
// Kernel: fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x256
// Target: gfx942 (MI300X)
// 
// MFMA Instructions: 768
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


/workspace/dev/aiter_long_context2/hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x256.co:	file format elf64-amdgpu

Disassembly of section .text:


// --- LABEL (potential loop target) ---
0000000000002e00 <_ZN5aiter47fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x256E>:
	s_and_b32 s1, s1, 0xffff                                   // 000000002E00: 8601FF01 0000FFFF

// === KERNEL PROLOGUE: Loading arguments ===
	s_load_dwordx2 s[8:9], s[0:1], 0x0                         // 000000002E08: C0060200 00000000  // Load scalar arg from kernel arguments
	s_load_dwordx2 s[20:21], s[0:1], 0x10                      // 000000002E10: C0060500 00000010  // Load scalar arg from kernel arguments
	s_load_dwordx2 s[24:25], s[0:1], 0x20                      // 000000002E18: C0060600 00000020  // Load scalar arg from kernel arguments
	s_load_dwordx2 s[50:51], s[0:1], 0x30                      // 000000002E20: C0060C80 00000030  // Load scalar arg from kernel arguments
	s_load_dwordx2 s[12:13], s[0:1], 0x40                      // 000000002E28: C0060300 00000040  // Load scalar arg from kernel arguments
	s_load_dwordx2 s[28:29], s[0:1], 0x50                      // 000000002E30: C0060700 00000050  // Load scalar arg from kernel arguments
	s_load_dwordx2 s[32:33], s[0:1], 0x60                      // 000000002E38: C0060800 00000060  // Load scalar arg from kernel arguments
	s_load_dwordx2 s[16:17], s[0:1], 0x70                      // 000000002E40: C0060400 00000070  // Load scalar arg from kernel arguments
	s_load_dwordx2 s[36:37], s[0:1], 0x80                      // 000000002E48: C0060900 00000080  // Load scalar arg from kernel arguments
	s_load_dwordx2 s[44:45], s[0:1], 0x90                      // 000000002E50: C0060B00 00000090  // Load scalar arg from kernel arguments
	s_load_dwordx2 s[40:41], s[0:1], 0xa0                      // 000000002E58: C0060A00 000000A0  // Load scalar arg from kernel arguments
	s_load_dwordx2 s[46:47], s[0:1], 0xb0                      // 000000002E60: C0060B80 000000B0  // Load scalar arg from kernel arguments
	s_load_dword s64, s[0:1], 0xc0                             // 000000002E68: C0021000 000000C0  // Load scalar arg from kernel arguments
	s_load_dword s65, s[0:1], 0xd0                             // 000000002E70: C0021040 000000D0  // Load scalar arg from kernel arguments
	s_load_dword s66, s[0:1], 0xe0                             // 000000002E78: C0021080 000000E0  // Load scalar arg from kernel arguments
	s_load_dword s67, s[0:1], 0xf0                             // 000000002E80: C00210C0 000000F0  // Load scalar arg from kernel arguments
	s_load_dword s68, s[0:1], 0x100                            // 000000002E88: C0021100 00000100  // Load scalar arg from kernel arguments
	s_load_dword s69, s[0:1], 0x110                            // 000000002E90: C0021140 00000110  // Load scalar arg from kernel arguments
	s_load_dword s70, s[0:1], 0x120                            // 000000002E98: C0021180 00000120  // Load scalar arg from kernel arguments
	s_load_dword s71, s[0:1], 0x130                            // 000000002EA0: C00211C0 00000130  // Load scalar arg from kernel arguments
	s_load_dword s72, s[0:1], 0x140                            // 000000002EA8: C0021200 00000140  // Load scalar arg from kernel arguments
	s_load_dword s73, s[0:1], 0x150                            // 000000002EB0: C0021240 00000150  // Load scalar arg from kernel arguments
	s_load_dword s74, s[0:1], 0x160                            // 000000002EB8: C0021280 00000160  // Load scalar arg from kernel arguments
	s_load_dword s75, s[0:1], 0x170                            // 000000002EC0: C00212C0 00000170  // Load scalar arg from kernel arguments
	s_load_dword s76, s[0:1], 0x180                            // 000000002EC8: C0021300 00000180  // Load scalar arg from kernel arguments
	v_lshrrev_b32_e32 v1, 10, v0                               // 000000002ED0: 2002008A  // Right shift (divide by power of 2)
	v_lshrrev_b32_e32 v2, 10, v1                               // 000000002ED4: 2004028A  // Right shift (divide by power of 2)
	v_and_b32_e32 v2, 0x3ff, v2                                // 000000002ED8: 260404FF 000003FF  // Bitwise AND (masking)
	v_and_b32_e32 v1, 0x3ff, v1                                // 000000002EE0: 260202FF 000003FF  // Bitwise AND (masking)
	v_and_b32_e32 v0, 0x3ff, v0                                // 000000002EE8: 260000FF 000003FF  // Bitwise AND (masking)
	v_lshrrev_b32_e32 v3, 6, v0                                // 000000002EF0: 20060086  // Right shift (divide by power of 2)
	v_and_b32_e32 v0, 63, v0                                   // 000000002EF4: 260000BF  // Bitwise AND (masking)
	s_mov_b32 s2, s2                                           // 000000002EF8: BE820002  // Scalar move (constant/register)
	s_mov_b32 s3, s3                                           // 000000002EFC: BE830003  // Scalar move (constant/register)
	s_mov_b32 s4, s4                                           // 000000002F00: BE840004  // Scalar move (constant/register)
	v_readfirstlane_b32 s7, v3                                 // 000000002F04: 7E0E0503  // Broadcast lane 0 to scalar register
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 000000002F08: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_and_b32 s51, s51, 0xffff                                 // 000000002F0C: 8633FF33 0000FFFF
	s_load_dword s50, s[50:51], 0x0                            // 000000002F14: C0020C99 00000000  // Load scalar arg from kernel arguments
	s_and_b32 s45, s45, 0xffff                                 // 000000002F1C: 862DFF2D 0000FFFF
	s_and_b32 s47, s47, 0xffff                                 // 000000002F24: 862FFF2F 0000FFFF
	s_and_b32 s9, s9, 0xffff                                   // 000000002F2C: 8609FF09 0000FFFF
	s_mul_i32 s60, s66, s68                                    // 000000002F34: 923C4442
	s_mul_i32 s61, s66, 4                                      // 000000002F38: 923D8442
	s_mov_b32 s22, s60                                         // 000000002F3C: BE96003C  // Scalar move (constant/register)
	s_mov_b32 s26, -16                                         // 000000002F40: BE9A00D0  // Scalar move (constant/register)
	s_mov_b32 s14, -16                                         // 000000002F44: BE8E00D0  // Scalar move (constant/register)
	s_mov_b32 s42, -16                                         // 000000002F48: BEAA00D0  // Scalar move (constant/register)
	s_mov_b32 s30, -16                                         // 000000002F4C: BE9E00D0  // Scalar move (constant/register)
	s_mov_b32 s34, -16                                         // 000000002F50: BEA200D0  // Scalar move (constant/register)
	s_mov_b32 s38, -16                                         // 000000002F54: BEA600D0  // Scalar move (constant/register)
	s_mov_b32 s18, -16                                         // 000000002F58: BE9200D0  // Scalar move (constant/register)
	s_mov_b32 s23, 0x20000                                     // 000000002F5C: BE9700FF 00020000  // Scalar move (constant/register)
	s_mov_b32 s27, 0x20000                                     // 000000002F64: BE9B00FF 00020000  // Scalar move (constant/register)
	s_mov_b32 s15, 0x20000                                     // 000000002F6C: BE8F00FF 00020000  // Scalar move (constant/register)
	s_mov_b32 s43, 0x20000                                     // 000000002F74: BEAB00FF 00020000  // Scalar move (constant/register)
	s_mov_b32 s31, 0x20000                                     // 000000002F7C: BE9F00FF 00020000  // Scalar move (constant/register)
	s_mov_b32 s35, 0x20000                                     // 000000002F84: BEA300FF 00020000  // Scalar move (constant/register)
	s_mov_b32 s39, 0x20000                                     // 000000002F8C: BEA700FF 00020000  // Scalar move (constant/register)
	s_mov_b32 s19, 0x20000                                     // 000000002F94: BE9300FF 00020000  // Scalar move (constant/register)
	s_and_b32 s21, s21, 0xffff                                 // 000000002F9C: 8615FF15 0000FFFF
	s_and_b32 s25, s25, 0xffff                                 // 000000002FA4: 8619FF19 0000FFFF
	s_and_b32 s13, s13, 0xffff                                 // 000000002FAC: 860DFF0D 0000FFFF
	s_and_b32 s41, s41, 0xffff                                 // 000000002FB4: 8629FF29 0000FFFF
	s_and_b32 s29, s29, 0xffff                                 // 000000002FBC: 861DFF1D 0000FFFF
	s_and_b32 s33, s33, 0xffff                                 // 000000002FC4: 8621FF21 0000FFFF
	s_and_b32 s37, s37, 0xffff                                 // 000000002FCC: 8625FF25 0000FFFF
	s_and_b32 s17, s17, 0xffff                                 // 000000002FD4: 8611FF11 0000FFFF
	s_or_b32 s21, s21, 0x40000                                 // 000000002FDC: 8715FF15 00040000
	s_or_b32 s25, s25, 0x40000                                 // 000000002FE4: 8719FF19 00040000
	s_or_b32 s13, s13, 0x40000                                 // 000000002FEC: 870DFF0D 00040000
	s_or_b32 s41, s41, 0x40000                                 // 000000002FF4: 8729FF29 00040000
	s_or_b32 s29, s29, 0x40000                                 // 000000002FFC: 871DFF1D 00040000
	s_or_b32 s33, s33, 0x40000                                 // 000000003004: 8721FF21 00040000
	s_or_b32 s37, s37, 0x40000                                 // 00000000300C: 8725FF25 00040000
	s_or_b32 s17, s17, 0x40000                                 // 000000003014: 8711FF11 00040000
	v_accvgpr_write_b32 a127, 0                                // 00000000301C: D3D9407F 18000080  // Write to accumulator VGPR (MFMA dest)
	v_mov_b32_e32 v255, 0                                      // 000000003024: 7FFE0280  // Vector move
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 000000003028: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_mul_i32 s60, s3, 32                                      // 00000000302C: 923CA003
	s_cmp_lt_i32 s60, s50                                      // 000000003030: BF04323C
	s_cbranch_scc0 label_1D49                                  // 000000003034: BF841CBB  // Branch if SCC==0 (condition false)
	s_mov_b32 s80, 0                                           // 000000003038: BED00080  // Scalar move (constant/register)
	s_mov_b32 s81, s64                                         // 00000000303C: BED10040  // Scalar move (constant/register)
	s_mul_i32 s60, s3, 4                                       // 000000003040: 923C8403
	s_add_u32 s46, s60, s46                                    // 000000003044: 802E2E3C
	s_addc_u32 s47, 0, s47                                     // 000000003048: 822F2F80
	s_load_dword s5, s[46:47], 0x0                             // 00000000304C: C0020157 00000000  // Load scalar arg from kernel arguments
	s_mul_i32 s60, s3, 32                                      // 000000003054: 923CA003
	s_mul_i32 s60, 4, s60                                      // 000000003058: 923C3C84
	v_and_b32_e32 v56, 15, v0                                  // 00000000305C: 2670008F  // Bitwise AND (masking)
	v_lshlrev_b32_e32 v56, 2, v56                              // 000000003060: 24707082  // Left shift (multiply by power of 2)
	v_add_u32_e32 v56, s60, v56                                // 000000003064: 6870703C  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v57, 0                                       // 000000003068: 7E720280  // Vector move

// === GLOBAL MEMORY LOAD ===
	global_load_dword v7, v56, s[44:45]                        // 00000000306C: DC508000 072C0038  // Load 32-bit from global memory (VRAM)
	v_add_u32_e32 v56, 64, v56                                 // 000000003074: 687070C0  // 32-bit unsigned add (address calc)
	global_load_dword v8, v56, s[44:45]                        // 000000003078: DC508000 082C0038  // Load 32-bit from global memory (VRAM)
	s_mul_i32 s60, s3, 32                                      // 000000003080: 923CA003
	s_add_u32 s60, s7, s60                                     // 000000003084: 803C3C07
	s_mul_i32 s60, 4, s60                                      // 000000003088: 923C3C84
	s_add_u32 s44, s60, s44                                    // 00000000308C: 802C2C3C
	s_addc_u32 s45, 0, s45                                     // 000000003090: 822D2D80
	s_load_dword s82, s[44:45], 0x0                            // 000000003094: C0021496 00000000  // Load scalar arg from kernel arguments
	s_load_dword s83, s[44:45], 0x10                           // 00000000309C: C00214D6 00000010  // Load scalar arg from kernel arguments
	s_load_dword s84, s[44:45], 0x20                           // 0000000030A4: C0021516 00000020  // Load scalar arg from kernel arguments
	s_load_dword s85, s[44:45], 0x30                           // 0000000030AC: C0021556 00000030  // Load scalar arg from kernel arguments
	s_load_dword s86, s[44:45], 0x40                           // 0000000030B4: C0021596 00000040  // Load scalar arg from kernel arguments
	s_load_dword s87, s[44:45], 0x50                           // 0000000030BC: C00215D6 00000050  // Load scalar arg from kernel arguments
	s_load_dword s88, s[44:45], 0x60                           // 0000000030C4: C0021616 00000060  // Load scalar arg from kernel arguments
	s_load_dword s89, s[44:45], 0x70                           // 0000000030CC: C0021656 00000070  // Load scalar arg from kernel arguments
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 0000000030D4: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	v_lshlrev_b32_e32 v56, 2, v0                               // 0000000030D8: 24700082  // Left shift (multiply by power of 2)
	s_and_b32 s82, s82, 0xffffff                               // 0000000030DC: 8652FF52 00FFFFFF
	s_mul_i32 s60, s82, s68                                    // 0000000030E4: 923C4452
	v_add_u32_e64 v36, v56, s60                                // 0000000030E8: D1340024 00007938  // 32-bit unsigned add (address calc)
	s_and_b32 s83, s83, 0xffffff                               // 0000000030F0: 8653FF53 00FFFFFF
	s_mul_i32 s60, s83, s68                                    // 0000000030F8: 923C4453
	v_add_u32_e64 v37, v56, s60                                // 0000000030FC: D1340025 00007938  // 32-bit unsigned add (address calc)
	s_and_b32 s84, s84, 0xffffff                               // 000000003104: 8654FF54 00FFFFFF
	s_mul_i32 s60, s84, s68                                    // 00000000310C: 923C4454
	v_add_u32_e64 v38, v56, s60                                // 000000003110: D1340026 00007938  // 32-bit unsigned add (address calc)
	s_and_b32 s85, s85, 0xffffff                               // 000000003118: 8655FF55 00FFFFFF
	s_mul_i32 s60, s85, s68                                    // 000000003120: 923C4455
	v_add_u32_e64 v39, v56, s60                                // 000000003124: D1340027 00007938  // 32-bit unsigned add (address calc)
	s_and_b32 s86, s86, 0xffffff                               // 00000000312C: 8656FF56 00FFFFFF
	s_mul_i32 s60, s86, s68                                    // 000000003134: 923C4456
	v_add_u32_e64 v40, v56, s60                                // 000000003138: D1340028 00007938  // 32-bit unsigned add (address calc)
	s_and_b32 s87, s87, 0xffffff                               // 000000003140: 8657FF57 00FFFFFF
	s_mul_i32 s60, s87, s68                                    // 000000003148: 923C4457
	v_add_u32_e64 v41, v56, s60                                // 00000000314C: D1340029 00007938  // 32-bit unsigned add (address calc)
	s_and_b32 s88, s88, 0xffffff                               // 000000003154: 8658FF58 00FFFFFF
	s_mul_i32 s60, s88, s68                                    // 00000000315C: 923C4458
	v_add_u32_e64 v42, v56, s60                                // 000000003160: D134002A 00007938  // 32-bit unsigned add (address calc)
	s_and_b32 s89, s89, 0xffffff                               // 000000003168: 8659FF59 00FFFFFF
	s_mul_i32 s60, s89, s68                                    // 000000003170: 923C4459
	v_add_u32_e64 v43, v56, s60                                // 000000003174: D134002B 00007938  // 32-bit unsigned add (address calc)
	v_lshlrev_b32_e32 v56, 2, v0                               // 00000000317C: 24700082  // Left shift (multiply by power of 2)
	s_mul_i32 s60, s82, s71                                    // 000000003180: 923C4752
	v_add_u32_e64 v80, v56, s60                                // 000000003184: D1340050 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v81, 0                                       // 00000000318C: 7EA20280  // Vector move
	s_mul_i32 s60, s83, s71                                    // 000000003190: 923C4753
	v_add_u32_e64 v82, v56, s60                                // 000000003194: D1340052 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v83, 0                                       // 00000000319C: 7EA60280  // Vector move
	s_mul_i32 s60, s84, s71                                    // 0000000031A0: 923C4754
	v_add_u32_e64 v84, v56, s60                                // 0000000031A4: D1340054 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v85, 0                                       // 0000000031AC: 7EAA0280  // Vector move
	s_mul_i32 s60, s85, s71                                    // 0000000031B0: 923C4755
	v_add_u32_e64 v86, v56, s60                                // 0000000031B4: D1340056 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v87, 0                                       // 0000000031BC: 7EAE0280  // Vector move
	s_mul_i32 s60, s86, s71                                    // 0000000031C0: 923C4756
	v_add_u32_e64 v88, v56, s60                                // 0000000031C4: D1340058 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v89, 0                                       // 0000000031CC: 7EB20280  // Vector move
	s_mul_i32 s60, s87, s71                                    // 0000000031D0: 923C4757
	v_add_u32_e64 v90, v56, s60                                // 0000000031D4: D134005A 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v91, 0                                       // 0000000031DC: 7EB60280  // Vector move
	s_mul_i32 s60, s88, s71                                    // 0000000031E0: 923C4758
	v_add_u32_e64 v92, v56, s60                                // 0000000031E4: D134005C 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v93, 0                                       // 0000000031EC: 7EBA0280  // Vector move
	s_mul_i32 s60, s89, s71                                    // 0000000031F0: 923C4759
	v_add_u32_e64 v94, v56, s60                                // 0000000031F4: D134005E 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v95, 0                                       // 0000000031FC: 7EBE0280  // Vector move
	s_mul_i32 s60, s7, 0x820                                   // 000000003200: 923CFF07 00000820
	s_add_u32 s50, 0, s60                                      // 000000003208: 80323C80
	s_add_u32 s51, 0x2480, s50                                 // 00000000320C: 803332FF 00002480
	v_lshrrev_b32_e32 v56, 4, v0                               // 000000003214: 20700084  // Right shift (divide by power of 2)
	v_lshlrev_b32_e32 v57, 2, v56                              // 000000003218: 24727082  // Left shift (multiply by power of 2)
	v_and_b32_e32 v56, 15, v0                                  // 00000000321C: 2670008F  // Bitwise AND (masking)
	v_lshrrev_b32_e32 v58, 2, v56                              // 000000003220: 20747082  // Right shift (divide by power of 2)
	v_lshlrev_b32_e32 v58, 6, v58                              // 000000003224: 24747486  // Left shift (multiply by power of 2)
	v_add_u32_e32 v57, v58, v57                                // 000000003228: 6872733A  // 32-bit unsigned add (address calc)
	v_and_b32_e32 v56, 3, v0                                   // 00000000322C: 26700083  // Bitwise AND (masking)
	v_mul_i32_i24_e32 v58, 0x208, v56                          // 000000003230: 0C7470FF 00000208  // 24-bit integer multiply (index calc)
	v_add_u32_e32 v57, v58, v57                                // 000000003238: 6872733A  // 32-bit unsigned add (address calc)
	v_lshlrev_b32_e32 v2, 2, v57                               // 00000000323C: 24047282  // Left shift (multiply by power of 2)
	s_mul_i32 s60, s2, 0x100                                   // 000000003240: 923CFF02 00000100
	s_mul_i32 s60, s60, s69                                    // 000000003248: 923C453C
	s_mul_i32 s61, s5, s72                                     // 00000000324C: 923D4805
	s_add_u32 s60, s61, s60                                    // 000000003250: 803C3C3D
	s_add_u32 s24, s60, s24                                    // 000000003254: 8018183C
	s_addc_u32 s25, 0, s25                                     // 000000003258: 82191980
	s_mul_i32 s60, s7, 16                                      // 00000000325C: 923C9007
	s_mul_i32 s60, s60, s69                                    // 000000003260: 923C453C
	v_lshlrev_b32_e32 v44, 4, v0                               // 000000003264: 24580084  // Left shift (multiply by power of 2)
	v_add_u32_e32 v44, s60, v44                                // 000000003268: 6858583C  // 32-bit unsigned add (address calc)
	s_mul_i32 s60, 64, s69                                     // 00000000326C: 923C45C0
	v_add_u32_e32 v45, s60, v44                                // 000000003270: 685A583C  // 32-bit unsigned add (address calc)
	v_add_u32_e32 v46, s60, v45                                // 000000003274: 685C5A3C  // 32-bit unsigned add (address calc)
	v_add_u32_e32 v47, s60, v46                                // 000000003278: 685E5C3C  // 32-bit unsigned add (address calc)
	s_mov_b32 s92, s24                                         // 00000000327C: BEDC0018  // Scalar move (constant/register)
	s_mov_b32 s93, s25                                         // 000000003280: BEDD0019  // Scalar move (constant/register)
	s_mov_b32 s94, s26                                         // 000000003284: BEDE001A  // Scalar move (constant/register)
	s_mov_b32 s95, s27                                         // 000000003288: BEDF001B  // Scalar move (constant/register)
	s_mul_i32 s60, s69, s65                                    // 00000000328C: 923C4145
	s_add_u32 s92, s60, s92                                    // 000000003290: 805C5C3C
	s_addc_u32 s93, 0, s93                                     // 000000003294: 825D5D80
	s_mul_i32 s60, s2, 0x1000                                  // 000000003298: 923CFF02 00001000
	s_mul_i32 s61, s5, s73                                     // 0000000032A0: 923D4905
	s_add_u32 s60, s61, s60                                    // 0000000032A4: 803C3C3D
	s_add_u32 s12, s60, s12                                    // 0000000032A8: 800C0C3C
	s_addc_u32 s13, 0, s13                                     // 0000000032AC: 820D0D80
	s_mul_i32 s60, s7, 16                                      // 0000000032B0: 923C9007
	s_mul_i32 s60, s60, s70                                    // 0000000032B4: 923C463C
	v_lshlrev_b32_e32 v48, 4, v0                               // 0000000032B8: 24600084  // Left shift (multiply by power of 2)
	v_add_u32_e32 v48, s60, v48                                // 0000000032BC: 6860603C  // 32-bit unsigned add (address calc)
	s_mul_i32 s60, 64, s70                                     // 0000000032C0: 923C46C0
	v_add_u32_e32 v49, s60, v48                                // 0000000032C4: 6862603C  // 32-bit unsigned add (address calc)
	v_add_u32_e32 v50, s60, v49                                // 0000000032C8: 6864623C  // 32-bit unsigned add (address calc)
	v_add_u32_e32 v51, s60, v50                                // 0000000032CC: 6866643C  // 32-bit unsigned add (address calc)
	s_mul_i32 s60, s70, 0x100                                  // 0000000032D0: 923CFF46 00000100
	s_mov_b32 s78, 0x400                                       // 0000000032D8: BECE00FF 00000400  // Scalar move (constant/register)
	s_mul_i32 s61, s78, 3                                      // 0000000032E0: 923D834E
	s_sub_u32 s56, s60, s61                                    // 0000000032E4: 80B83D3C
	s_mul_i32 s60, s3, 32                                      // 0000000032E8: 923CA003
	s_mul_i32 s60, 4, s60                                      // 0000000032EC: 923C3C84
	s_add_u32 s40, s60, s40                                    // 0000000032F0: 8028283C
	s_addc_u32 s41, 0, s41                                     // 0000000032F4: 82292980
	v_and_b32_e32 v56, 15, v0                                  // 0000000032F8: 2670008F  // Bitwise AND (masking)
	v_lshlrev_b32_e32 v9, 2, v56                               // 0000000032FC: 24127082  // Left shift (multiply by power of 2)
	v_add_u32_e32 v10, 64, v9                                  // 000000003300: 681412C0  // 32-bit unsigned add (address calc)
	s_lshr_b32 s60, s64, 7                                     // 000000003304: 8F3C8740
	s_mul_i32 s61, s60, 4                                      // 000000003308: 923D843C
	v_and_b32_e64 v11, v0, 1                                   // 00000000330C: D113000B 00010300  // Bitwise AND (masking)
	v_mul_i32_i24_e64 v11, v11, s61                            // 000000003314: D106000B 00007B0B  // 24-bit integer multiply (index calc)
	v_and_b32_e64 v56, v0, 3                                   // 00000000331C: D1130038 00010700  // Bitwise AND (masking)
	v_lshrrev_b32_e32 v56, 1, v56                              // 000000003324: 20707081  // Right shift (divide by power of 2)
	v_mul_i32_i24_e32 v56, 4, v56                              // 000000003328: 0C707084  // 24-bit integer multiply (index calc)
	v_add_u32_e32 v11, v11, v56                                // 00000000332C: 6816710B  // 32-bit unsigned add (address calc)
	s_lshr_b32 s60, s65, 7                                     // 000000003330: 8F3C8741
	s_mul_i32 s60, s60, s61                                    // 000000003334: 923C3D3C
	v_add_u32_e64 v13, v11, s60                                // 000000003338: D134000D 0000790B  // 32-bit unsigned add (address calc)
	s_mov_b32 s4, 8                                            // 000000003340: BE840088  // Scalar move (constant/register)
	s_mul_i32 s60, s2, 2                                       // 000000003344: 923C8202
	s_mul_i32 s60, s60, s61                                    // 000000003348: 923C3D3C
	s_mul_i32 s61, s5, s74                                     // 00000000334C: 923D4A05
	s_add_u32 s61, s61, s60                                    // 000000003350: 803D3C3D
	s_add_u32 s32, s61, s32                                    // 000000003354: 8020203D
	s_addc_u32 s33, 0, s33                                     // 000000003358: 82212180
	s_lshr_b32 s60, s65, 7                                     // 00000000335C: 8F3C8741
	s_mul_i32 s61, s60, 4                                      // 000000003360: 923D843C
	s_mul_i32 s60, s2, 2                                       // 000000003364: 923C8202
	s_mul_i32 s60, s60, 4                                      // 000000003368: 923C843C
	v_and_b32_e64 v6, v0, 1                                    // 00000000336C: D1130006 00010300  // Bitwise AND (masking)
	v_mul_i32_i24_e64 v6, v6, s61                              // 000000003374: D1060006 00007B06  // 24-bit integer multiply (index calc)
	v_and_b32_e64 v56, v0, 3                                   // 00000000337C: D1130038 00010700  // Bitwise AND (masking)
	v_lshrrev_b32_e32 v56, 1, v56                              // 000000003384: 20707081  // Right shift (divide by power of 2)
	v_mul_i32_i24_e32 v56, 4, v56                              // 000000003388: 0C707084  // 24-bit integer multiply (index calc)
	v_add_i32 v6, v6, v56                                      // 00000000338C: D29C0006 00027106
	v_add_i32 v6, v6, s60                                      // 000000003394: D29C0006 00007906
	s_mul_i32 s60, s5, s75                                     // 00000000339C: 923C4B05
	s_add_u32 s16, s60, s16                                    // 0000000033A0: 8010103C
	s_addc_u32 s17, 0, s17                                     // 0000000033A4: 82111180
	s_mov_b32 s57, 0x100                                       // 0000000033A8: BEB900FF 00000100  // Scalar move (constant/register)
	s_mov_b32 s58, 0x1000                                      // 0000000033B0: BEBA00FF 00001000  // Scalar move (constant/register)
	s_mul_i32 s79, 2, s61                                      // 0000000033B8: 924F3D82
	s_mov_b32 s59, 0                                           // 0000000033BC: BEBB0080  // Scalar move (constant/register)
	s_mov_b32 s90, s58                                         // 0000000033C0: BEDA003A  // Scalar move (constant/register)
	s_mov_b32 s52, 0x7060302                                   // 0000000033C4: BEB400FF 07060302  // Scalar move (constant/register)
	s_mov_b32 s53, 0x400                                       // 0000000033CC: BEB500FF 00000400  // Scalar move (constant/register)
	s_mov_b32 s54, 0x40100                                     // 0000000033D4: BEB600FF 00040100  // Scalar move (constant/register)
	s_mov_b32 s55, 0x4020100                                   // 0000000033DC: BEB700FF 04020100  // Scalar move (constant/register)
	s_mov_b32 s6, 0x3fb8aa3b                                   // 0000000033E4: BE8600FF 3FB8AA3B  // Scalar move (constant/register)
	s_mov_b32 s77, 0xbd92220c                                  // 0000000033EC: BECD00FF BD92220C  // Scalar move (constant/register)
	s_mov_b32 m0, s50                                          // 0000000033F4: BEFC0032  // Scalar move (constant/register)
	v_mov_b32_e32 v1, 0xbfcc4231                               // 0000000033F8: 7E0202FF BFCC4231  // Vector move
	v_mov_b32_e32 v53, 0xffff0000                              // 000000003400: 7E6A02FF FFFF0000  // Vector move
	v_mov_b32_e32 v54, 0x7fff0000                              // 000000003408: 7E6C02FF 7FFF0000  // Vector move
	v_mov_b32_e32 v55, 0x7fff                                  // 000000003410: 7E6E02FF 00007FFF  // Vector move
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 000000003418: BF8C0000  // Wait for memory operations (CRITICAL for perf)
	v_and_b32_e32 v7, 0xffffff, v7                             // 00000000341C: 260E0EFF 00FFFFFF  // Bitwise AND (masking)
	v_and_b32_e32 v8, 0xffffff, v8                             // 000000003424: 261010FF 00FFFFFF  // Bitwise AND (masking)
	v_lshlrev_b32_e32 v7, 2, v7                                // 00000000342C: 240E0E82  // Left shift (multiply by power of 2)
	v_lshlrev_b32_e32 v8, 2, v8                                // 000000003430: 24101082  // Left shift (multiply by power of 2)
	s_lshr_b32 s60, s7, 1                                      // 000000003434: 8F3C8107
	s_lshl_b32 s3, s66, 2                                      // 000000003438: 8E038242
	s_mul_i32 s60, s60, s3                                     // 00000000343C: 923C033C
	s_add_u32 s28, s28, s60                                    // 000000003440: 801C3C1C
	s_addc_u32 s29, 0, s29                                     // 000000003444: 821D1D80
	s_mov_b32 s30, s3                                          // 000000003448: BE9E0003  // Scalar move (constant/register)
	s_lshl_b32 s3, s3, 1                                       // 00000000344C: 8E038103
	s_and_b32 s61, s7, 1                                       // 000000003450: 863D8107
	s_cmp_eq_u32 s61, 1                                        // 000000003454: BF06813D
	s_cselect_b32 s60, 0, 1                                    // 000000003458: 853C8180
	v_mul_i32_i24_e64 v56, v7, s60                             // 00000000345C: D1060038 00007907  // 24-bit integer multiply (index calc)
	v_mul_i32_i24_e64 v57, v8, s61                             // 000000003464: D1060039 00007B08  // 24-bit integer multiply (index calc)
	v_add_u32_e32 v56, v56, v57                                // 00000000346C: 68707338  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v7, v56                                      // 000000003470: 7E0E0338  // Vector move
	s_mul_i32 s60, s7, 0x100                                   // 000000003474: 923CFF07 00000100
	s_sub_u32 s61, 4, s7                                       // 00000000347C: 80BD0784
	s_mul_i32 s61, s61, 0x820                                  // 000000003480: 923DFF3D 00000820
	s_add_u32 s76, s60, s61                                    // 000000003488: 804C3D3C
	v_lshlrev_b32_e32 v3, 2, v0                                // 00000000348C: 24060082  // Left shift (multiply by power of 2)
	buffer_load_dword v23, v11, s[32:35], 0 offen              // 000000003490: E0501000 8008170B
	buffer_load_dword v25, v9, s[40:43], 0 offen               // 000000003498: E0501000 800A1909
	buffer_load_dword v26, v10, s[40:43], 0 offen              // 0000000034A0: E0501000 800A1A0A
	buffer_load_dword v36, s[20:23], 0 offen lds               // 0000000034A8: E0511000 80050024
	s_add_u32 m0, 0x100, s50                                   // 0000000034B0: 807C32FF 00000100
	buffer_load_dword v37, s[20:23], 0 offen lds               // 0000000034B8: E0511000 80050025
	s_add_u32 m0, 0x200, s50                                   // 0000000034C0: 807C32FF 00000200
	buffer_load_dword v38, s[20:23], 0 offen lds               // 0000000034C8: E0511000 80050026
	s_add_u32 m0, 0x300, s50                                   // 0000000034D0: 807C32FF 00000300
	buffer_load_dword v39, s[20:23], 0 offen lds               // 0000000034D8: E0511000 80050027
	s_add_u32 m0, 0x400, s50                                   // 0000000034E0: 807C32FF 00000400
	buffer_load_dword v40, s[20:23], 0 offen lds               // 0000000034E8: E0511000 80050028
	s_add_u32 m0, 0x500, s50                                   // 0000000034F0: 807C32FF 00000500
	buffer_load_dword v41, s[20:23], 0 offen lds               // 0000000034F8: E0511000 80050029
	s_add_u32 m0, 0x600, s50                                   // 000000003500: 807C32FF 00000600
	buffer_load_dword v42, s[20:23], 0 offen lds               // 000000003508: E0511000 8005002A
	s_add_u32 m0, 0x700, s50                                   // 000000003510: 807C32FF 00000700
	buffer_load_dword v43, s[20:23], 0 offen lds               // 000000003518: E0511000 8005002B
	s_add_u32 m0, s50, s76                                     // 000000003520: 807C4C32
	buffer_load_dword v7, s[28:31], 0 offen lds                // 000000003524: E0511000 80070007
	s_add_u32 m0, 0, s51                                       // 00000000352C: 807C3380
	s_add_u32 s20, s57, s20                                    // 000000003530: 80141439
	s_addc_u32 s21, 0, s21                                     // 000000003534: 82151580
	s_add_u32 s28, s3, s28                                     // 000000003538: 801C1C03
	s_addc_u32 s29, 0, s29                                     // 00000000353C: 821D1D80
	buffer_load_dwordx4 a[0:3], v44, s[24:27], 0 offen         // 000000003540: E05C1000 8086002C
	buffer_load_dwordx4 a[4:7], v44, s[24:27], 0 offen offset:1024// 000000003548: E05C1400 8086042C
	buffer_load_dwordx4 a[16:19], v45, s[24:27], 0 offen       // 000000003550: E05C1000 8086102D
	buffer_load_dwordx4 a[20:23], v45, s[24:27], 0 offen offset:1024// 000000003558: E05C1400 8086142D
	buffer_load_dwordx4 a[32:35], v46, s[24:27], 0 offen       // 000000003560: E05C1000 8086202E
	buffer_load_dwordx4 a[36:39], v46, s[24:27], 0 offen offset:1024// 000000003568: E05C1400 8086242E
	buffer_load_dwordx4 a[48:51], v47, s[24:27], 0 offen       // 000000003570: E05C1000 8086302F
	buffer_load_dwordx4 a[52:55], v47, s[24:27], 0 offen offset:1024// 000000003578: E05C1400 8086342F
	buffer_load_dwordx4 a[8:11], v44, s[24:27], 0 offen offset:2048// 000000003580: E05C1800 8086082C
	buffer_load_dwordx4 a[12:15], v44, s[24:27], 0 offen offset:3072// 000000003588: E05C1C00 80860C2C
	buffer_load_dwordx4 a[24:27], v45, s[24:27], 0 offen offset:2048// 000000003590: E05C1800 8086182D
	buffer_load_dwordx4 a[28:31], v45, s[24:27], 0 offen offset:3072// 000000003598: E05C1C00 80861C2D
	buffer_load_dwordx4 a[40:43], v46, s[24:27], 0 offen offset:2048// 0000000035A0: E05C1800 8086282E
	buffer_load_dwordx4 a[44:47], v46, s[24:27], 0 offen offset:3072// 0000000035A8: E05C1C00 80862C2E
	buffer_load_dwordx4 a[56:59], v47, s[24:27], 0 offen offset:2048// 0000000035B0: E05C1800 8086382F
	buffer_load_dwordx4 a[60:63], v47, s[24:27], 0 offen offset:3072// 0000000035B8: E05C1C00 80863C2F
	s_add_u32 s24, s58, s24                                    // 0000000035C0: 8018183A
	s_addc_u32 s25, 0, s25                                     // 0000000035C4: 82191980
	v_mov_b32_e32 v128, 0                                      // 0000000035C8: 7F000280  // Vector move
	v_mov_b32_e32 v64, 0                                       // 0000000035CC: 7E800280  // Vector move
	v_mov_b32_e32 v129, 0                                      // 0000000035D0: 7F020280  // Vector move
	v_mov_b32_e32 v65, 0                                       // 0000000035D4: 7E820280  // Vector move
	v_mov_b32_e32 v130, 0                                      // 0000000035D8: 7F040280  // Vector move
	v_mov_b32_e32 v66, 0                                       // 0000000035DC: 7E840280  // Vector move
	v_mov_b32_e32 v131, 0                                      // 0000000035E0: 7F060280  // Vector move
	v_mov_b32_e32 v67, 0                                       // 0000000035E4: 7E860280  // Vector move
	v_mov_b32_e32 v132, 0                                      // 0000000035E8: 7F080280  // Vector move
	v_mov_b32_e32 v68, 0                                       // 0000000035EC: 7E880280  // Vector move
	v_mov_b32_e32 v133, 0                                      // 0000000035F0: 7F0A0280  // Vector move
	v_mov_b32_e32 v69, 0                                       // 0000000035F4: 7E8A0280  // Vector move
	v_mov_b32_e32 v134, 0                                      // 0000000035F8: 7F0C0280  // Vector move
	v_mov_b32_e32 v70, 0                                       // 0000000035FC: 7E8C0280  // Vector move
	v_mov_b32_e32 v135, 0                                      // 000000003600: 7F0E0280  // Vector move
	v_mov_b32_e32 v71, 0                                       // 000000003604: 7E8E0280  // Vector move
	v_mov_b32_e32 v136, 0                                      // 000000003608: 7F100280  // Vector move
	v_mov_b32_e32 v72, 0                                       // 00000000360C: 7E900280  // Vector move
	v_mov_b32_e32 v137, 0                                      // 000000003610: 7F120280  // Vector move
	v_mov_b32_e32 v73, 0                                       // 000000003614: 7E920280  // Vector move
	v_mov_b32_e32 v138, 0                                      // 000000003618: 7F140280  // Vector move
	v_mov_b32_e32 v74, 0                                       // 00000000361C: 7E940280  // Vector move
	v_mov_b32_e32 v139, 0                                      // 000000003620: 7F160280  // Vector move
	v_mov_b32_e32 v75, 0                                       // 000000003624: 7E960280  // Vector move
	v_mov_b32_e32 v140, 0                                      // 000000003628: 7F180280  // Vector move
	v_mov_b32_e32 v76, 0                                       // 00000000362C: 7E980280  // Vector move
	v_mov_b32_e32 v141, 0                                      // 000000003630: 7F1A0280  // Vector move
	v_mov_b32_e32 v77, 0                                       // 000000003634: 7E9A0280  // Vector move
	v_mov_b32_e32 v142, 0                                      // 000000003638: 7F1C0280  // Vector move
	v_mov_b32_e32 v78, 0                                       // 00000000363C: 7E9C0280  // Vector move
	v_mov_b32_e32 v143, 0                                      // 000000003640: 7F1E0280  // Vector move
	v_mov_b32_e32 v79, 0                                       // 000000003644: 7E9E0280  // Vector move
	v_mov_b32_e32 v144, 0                                      // 000000003648: 7F200280  // Vector move
	v_mov_b32_e32 v80, 0                                       // 00000000364C: 7EA00280  // Vector move
	v_mov_b32_e32 v145, 0                                      // 000000003650: 7F220280  // Vector move
	v_mov_b32_e32 v81, 0                                       // 000000003654: 7EA20280  // Vector move
	v_mov_b32_e32 v146, 0                                      // 000000003658: 7F240280  // Vector move
	v_mov_b32_e32 v82, 0                                       // 00000000365C: 7EA40280  // Vector move
	v_mov_b32_e32 v147, 0                                      // 000000003660: 7F260280  // Vector move
	v_mov_b32_e32 v83, 0                                       // 000000003664: 7EA60280  // Vector move
	v_mov_b32_e32 v148, 0                                      // 000000003668: 7F280280  // Vector move
	v_mov_b32_e32 v84, 0                                       // 00000000366C: 7EA80280  // Vector move
	v_mov_b32_e32 v149, 0                                      // 000000003670: 7F2A0280  // Vector move
	v_mov_b32_e32 v85, 0                                       // 000000003674: 7EAA0280  // Vector move
	v_mov_b32_e32 v150, 0                                      // 000000003678: 7F2C0280  // Vector move
	v_mov_b32_e32 v86, 0                                       // 00000000367C: 7EAC0280  // Vector move
	v_mov_b32_e32 v151, 0                                      // 000000003680: 7F2E0280  // Vector move
	v_mov_b32_e32 v87, 0                                       // 000000003684: 7EAE0280  // Vector move
	v_mov_b32_e32 v152, 0                                      // 000000003688: 7F300280  // Vector move
	v_mov_b32_e32 v88, 0                                       // 00000000368C: 7EB00280  // Vector move
	v_mov_b32_e32 v153, 0                                      // 000000003690: 7F320280  // Vector move
	v_mov_b32_e32 v89, 0                                       // 000000003694: 7EB20280  // Vector move
	v_mov_b32_e32 v154, 0                                      // 000000003698: 7F340280  // Vector move
	v_mov_b32_e32 v90, 0                                       // 00000000369C: 7EB40280  // Vector move
	v_mov_b32_e32 v155, 0                                      // 0000000036A0: 7F360280  // Vector move
	v_mov_b32_e32 v91, 0                                       // 0000000036A4: 7EB60280  // Vector move
	v_mov_b32_e32 v156, 0                                      // 0000000036A8: 7F380280  // Vector move
	v_mov_b32_e32 v92, 0                                       // 0000000036AC: 7EB80280  // Vector move
	v_mov_b32_e32 v157, 0                                      // 0000000036B0: 7F3A0280  // Vector move
	v_mov_b32_e32 v93, 0                                       // 0000000036B4: 7EBA0280  // Vector move
	v_mov_b32_e32 v158, 0                                      // 0000000036B8: 7F3C0280  // Vector move
	v_mov_b32_e32 v94, 0                                       // 0000000036BC: 7EBC0280  // Vector move
	v_mov_b32_e32 v159, 0                                      // 0000000036C0: 7F3E0280  // Vector move
	v_mov_b32_e32 v95, 0                                       // 0000000036C4: 7EBE0280  // Vector move
	v_mov_b32_e32 v160, 0                                      // 0000000036C8: 7F400280  // Vector move
	v_mov_b32_e32 v96, 0                                       // 0000000036CC: 7EC00280  // Vector move
	v_mov_b32_e32 v161, 0                                      // 0000000036D0: 7F420280  // Vector move
	v_mov_b32_e32 v97, 0                                       // 0000000036D4: 7EC20280  // Vector move
	v_mov_b32_e32 v162, 0                                      // 0000000036D8: 7F440280  // Vector move
	v_mov_b32_e32 v98, 0                                       // 0000000036DC: 7EC40280  // Vector move
	v_mov_b32_e32 v163, 0                                      // 0000000036E0: 7F460280  // Vector move
	v_mov_b32_e32 v99, 0                                       // 0000000036E4: 7EC60280  // Vector move
	v_mov_b32_e32 v164, 0                                      // 0000000036E8: 7F480280  // Vector move
	v_mov_b32_e32 v100, 0                                      // 0000000036EC: 7EC80280  // Vector move
	v_mov_b32_e32 v165, 0                                      // 0000000036F0: 7F4A0280  // Vector move
	v_mov_b32_e32 v101, 0                                      // 0000000036F4: 7ECA0280  // Vector move
	v_mov_b32_e32 v166, 0                                      // 0000000036F8: 7F4C0280  // Vector move
	v_mov_b32_e32 v102, 0                                      // 0000000036FC: 7ECC0280  // Vector move
	v_mov_b32_e32 v167, 0                                      // 000000003700: 7F4E0280  // Vector move
	v_mov_b32_e32 v103, 0                                      // 000000003704: 7ECE0280  // Vector move
	v_mov_b32_e32 v168, 0                                      // 000000003708: 7F500280  // Vector move
	v_mov_b32_e32 v104, 0                                      // 00000000370C: 7ED00280  // Vector move
	v_mov_b32_e32 v169, 0                                      // 000000003710: 7F520280  // Vector move
	v_mov_b32_e32 v105, 0                                      // 000000003714: 7ED20280  // Vector move
	v_mov_b32_e32 v170, 0                                      // 000000003718: 7F540280  // Vector move
	v_mov_b32_e32 v106, 0                                      // 00000000371C: 7ED40280  // Vector move
	v_mov_b32_e32 v171, 0                                      // 000000003720: 7F560280  // Vector move
	v_mov_b32_e32 v107, 0                                      // 000000003724: 7ED60280  // Vector move
	v_mov_b32_e32 v172, 0                                      // 000000003728: 7F580280  // Vector move
	v_mov_b32_e32 v108, 0                                      // 00000000372C: 7ED80280  // Vector move
	v_mov_b32_e32 v173, 0                                      // 000000003730: 7F5A0280  // Vector move
	v_mov_b32_e32 v109, 0                                      // 000000003734: 7EDA0280  // Vector move
	v_mov_b32_e32 v174, 0                                      // 000000003738: 7F5C0280  // Vector move
	v_mov_b32_e32 v110, 0                                      // 00000000373C: 7EDC0280  // Vector move
	v_mov_b32_e32 v175, 0                                      // 000000003740: 7F5E0280  // Vector move
	v_mov_b32_e32 v111, 0                                      // 000000003744: 7EDE0280  // Vector move
	v_mov_b32_e32 v176, 0                                      // 000000003748: 7F600280  // Vector move
	v_mov_b32_e32 v112, 0                                      // 00000000374C: 7EE00280  // Vector move
	v_mov_b32_e32 v177, 0                                      // 000000003750: 7F620280  // Vector move
	v_mov_b32_e32 v113, 0                                      // 000000003754: 7EE20280  // Vector move
	v_mov_b32_e32 v178, 0                                      // 000000003758: 7F640280  // Vector move
	v_mov_b32_e32 v114, 0                                      // 00000000375C: 7EE40280  // Vector move
	v_mov_b32_e32 v179, 0                                      // 000000003760: 7F660280  // Vector move
	v_mov_b32_e32 v115, 0                                      // 000000003764: 7EE60280  // Vector move
	v_mov_b32_e32 v180, 0                                      // 000000003768: 7F680280  // Vector move
	v_mov_b32_e32 v116, 0                                      // 00000000376C: 7EE80280  // Vector move
	v_mov_b32_e32 v181, 0                                      // 000000003770: 7F6A0280  // Vector move
	v_mov_b32_e32 v117, 0                                      // 000000003774: 7EEA0280  // Vector move
	v_mov_b32_e32 v182, 0                                      // 000000003778: 7F6C0280  // Vector move
	v_mov_b32_e32 v118, 0                                      // 00000000377C: 7EEC0280  // Vector move
	v_mov_b32_e32 v183, 0                                      // 000000003780: 7F6E0280  // Vector move
	v_mov_b32_e32 v119, 0                                      // 000000003784: 7EEE0280  // Vector move
	v_mov_b32_e32 v184, 0                                      // 000000003788: 7F700280  // Vector move
	v_mov_b32_e32 v120, 0                                      // 00000000378C: 7EF00280  // Vector move
	v_mov_b32_e32 v185, 0                                      // 000000003790: 7F720280  // Vector move
	v_mov_b32_e32 v121, 0                                      // 000000003794: 7EF20280  // Vector move
	v_mov_b32_e32 v186, 0                                      // 000000003798: 7F740280  // Vector move
	v_mov_b32_e32 v122, 0                                      // 00000000379C: 7EF40280  // Vector move
	v_mov_b32_e32 v187, 0                                      // 0000000037A0: 7F760280  // Vector move
	v_mov_b32_e32 v123, 0                                      // 0000000037A4: 7EF60280  // Vector move
	v_mov_b32_e32 v188, 0                                      // 0000000037A8: 7F780280  // Vector move
	v_mov_b32_e32 v124, 0                                      // 0000000037AC: 7EF80280  // Vector move
	v_mov_b32_e32 v189, 0                                      // 0000000037B0: 7F7A0280  // Vector move
	v_mov_b32_e32 v125, 0                                      // 0000000037B4: 7EFA0280  // Vector move
	v_mov_b32_e32 v190, 0                                      // 0000000037B8: 7F7C0280  // Vector move
	v_mov_b32_e32 v126, 0                                      // 0000000037BC: 7EFC0280  // Vector move
	v_mov_b32_e32 v191, 0                                      // 0000000037C0: 7F7E0280  // Vector move
	v_mov_b32_e32 v127, 0                                      // 0000000037C4: 7EFE0280  // Vector move
	v_lshrrev_b32_e32 v56, 4, v0                               // 0000000037C8: 20700084  // Right shift (divide by power of 2)
	v_mul_i32_i24_e32 v4, 34, v56                              // 0000000037CC: 0C0870A2  // 24-bit integer multiply (index calc)
	v_and_b32_e32 v56, 15, v0                                  // 0000000037D0: 2670008F  // Bitwise AND (masking)
	v_mul_i32_i24_e32 v57, 2, v56                              // 0000000037D4: 0C727082  // 24-bit integer multiply (index calc)
	v_add_u32_e32 v4, v57, v4                                  // 0000000037D8: 68080939  // 32-bit unsigned add (address calc)
	s_mul_i32 s60, s7, 0x88                                    // 0000000037DC: 923CFF07 00000088
	v_add_u32_e32 v4, s60, v4                                  // 0000000037E4: 6808083C  // 32-bit unsigned add (address calc)
	v_lshlrev_b32_e32 v4, 2, v4                                // 0000000037E8: 24080882  // Left shift (multiply by power of 2)
	v_lshrrev_b32_e32 v56, 1, v0                               // 0000000037EC: 20700081  // Right shift (divide by power of 2)
	v_mul_i32_i24_e32 v5, 34, v56                              // 0000000037F0: 0C0A70A2  // 24-bit integer multiply (index calc)
	v_and_b32_e32 v57, 1, v0                                   // 0000000037F4: 26720081  // Bitwise AND (masking)
	v_add_u32_e32 v5, v57, v5                                  // 0000000037F8: 680A0B39  // 32-bit unsigned add (address calc)
	s_mul_i32 s60, s7, 2                                       // 0000000037FC: 923C8207
	v_add_u32_e32 v5, s60, v5                                  // 000000003800: 680A0A3C  // 32-bit unsigned add (address calc)
	v_lshlrev_b32_e32 v5, 2, v5                                // 000000003804: 240A0A82  // Left shift (multiply by power of 2)
	s_waitcnt vmcnt(16)                                        // 000000003808: BF8C4F70  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 00000000380C: BF8A0000  // Workgroup barrier synchronization

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[192:195], v2                                // 000000003810: D9FE0000 C0000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b128 v[196:199], v2 offset:64                      // 000000003818: D9FE0040 C4000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b128 v[200:203], v2 offset:128                     // 000000003820: D9FE0080 C8000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b128 v[204:207], v2 offset:192                     // 000000003828: D9FE00C0 CC000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b128 v[208:211], v2 offset:1024                    // 000000003830: D9FE0400 D0000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b128 v[212:215], v2 offset:1088                    // 000000003838: D9FE0440 D4000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b128 v[216:219], v2 offset:1152                    // 000000003840: D9FE0480 D8000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b128 v[220:223], v2 offset:1216                    // 000000003848: D9FE04C0 DC000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v15, v3 offset:8320                            // 000000003850: D86C2080 0F000003  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v16, v3 offset:8576                            // 000000003858: D86C2180 10000003  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v17, v3 offset:8832                            // 000000003860: D86C2280 11000003  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v18, v3 offset:9088                            // 000000003868: D86C2380 12000003  // Read 32-bit from LDS (shared memory)
	s_cmp_lt_i32 s7, 2                                         // 000000003870: BF048207
	s_cbranch_scc0 label_0F4C                                  // 000000003874: BF840CAE  // Branch if SCC==0 (condition false)


// --- LABEL (potential loop target) ---
0000000000003878 <label_029E>:
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(8) lgkmcnt(0)                              // 000000003878: BF8C0078  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 00000000387C: BF8A0000  // Workgroup barrier synchronization

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[0:1], v[192:193], 0// 000000003880: D3F300A0 0A038100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[2:3], v[194:195], v[160:163]// 000000003888: D3F300A0 0E838502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[64:67], v44, s[92:95], 0 offen       // 000000003890: E05C1000 8097402C
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[4:5], v[196:197], v[160:163]// 000000003898: D3F300A0 0E838904  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[6:7], v[198:199], v[160:163]// 0000000038A0: D3F300A0 0E838D06  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v36, s[20:23], 0 offen lds               // 0000000038A8: E0511000 80050024
	s_add_u32 m0, 0x100, s51                                   // 0000000038B0: 807C33FF 00000100
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[0:1], v[208:209], 0// 0000000038B8: D3F300A4 0A03A100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[2:3], v[210:211], v[164:167]// 0000000038C0: D3F300A4 0E93A502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[68:71], v44, s[92:95], 0 offen offset:1024// 0000000038C8: E05C1400 8097442C
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[4:5], v[212:213], v[164:167]// 0000000038D0: D3F300A4 0E93A904  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[6:7], v[214:215], v[164:167]// 0000000038D8: D3F300A4 0E93AD06  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v37, s[20:23], 0 offen lds               // 0000000038E0: E0511000 80050025
	s_add_u32 m0, 0x200, s51                                   // 0000000038E8: 807C33FF 00000200
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[16:17], v[192:193], 0// 0000000038F0: D3F300A8 0A038110  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[18:19], v[194:195], v[168:171]// 0000000038F8: D3F300A8 0EA38512  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[80:83], v45, s[92:95], 0 offen       // 000000003900: E05C1000 8097502D
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[20:21], v[196:197], v[168:171]// 000000003908: D3F300A8 0EA38914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[22:23], v[198:199], v[168:171]// 000000003910: D3F300A8 0EA38D16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v38, s[20:23], 0 offen lds               // 000000003918: E0511000 80050026
	s_add_u32 m0, 0x300, s51                                   // 000000003920: 807C33FF 00000300
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[16:17], v[208:209], 0// 000000003928: D3F300AC 0A03A110  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[18:19], v[210:211], v[172:175]// 000000003930: D3F300AC 0EB3A512  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[84:87], v45, s[92:95], 0 offen offset:1024// 000000003938: E05C1400 8097542D
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[20:21], v[212:213], v[172:175]// 000000003940: D3F300AC 0EB3A914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[22:23], v[214:215], v[172:175]// 000000003948: D3F300AC 0EB3AD16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v39, s[20:23], 0 offen lds               // 000000003950: E0511000 80050027
	s_add_u32 m0, 0x400, s51                                   // 000000003958: 807C33FF 00000400
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[32:33], v[192:193], 0// 000000003960: D3F300B0 0A038120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[34:35], v[194:195], v[176:179]// 000000003968: D3F300B0 0EC38522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[96:99], v46, s[92:95], 0 offen       // 000000003970: E05C1000 8097602E
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[36:37], v[196:197], v[176:179]// 000000003978: D3F300B0 0EC38924  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[38:39], v[198:199], v[176:179]// 000000003980: D3F300B0 0EC38D26  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v40, s[20:23], 0 offen lds               // 000000003988: E0511000 80050028
	s_add_u32 m0, 0x500, s51                                   // 000000003990: 807C33FF 00000500
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[32:33], v[208:209], 0// 000000003998: D3F300B4 0A03A120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[34:35], v[210:211], v[180:183]// 0000000039A0: D3F300B4 0ED3A522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[100:103], v46, s[92:95], 0 offen offset:1024// 0000000039A8: E05C1400 8097642E
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[36:37], v[212:213], v[180:183]// 0000000039B0: D3F300B4 0ED3A924  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[38:39], v[214:215], v[180:183]// 0000000039B8: D3F300B4 0ED3AD26  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v41, s[20:23], 0 offen lds               // 0000000039C0: E0511000 80050029
	s_add_u32 m0, 0x600, s51                                   // 0000000039C8: 807C33FF 00000600
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[48:49], v[192:193], 0// 0000000039D0: D3F300B8 0A038130  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[50:51], v[194:195], v[184:187]// 0000000039D8: D3F300B8 0EE38532  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[112:115], v47, s[92:95], 0 offen     // 0000000039E0: E05C1000 8097702F
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[52:53], v[196:197], v[184:187]// 0000000039E8: D3F300B8 0EE38934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[54:55], v[198:199], v[184:187]// 0000000039F0: D3F300B8 0EE38D36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v42, s[20:23], 0 offen lds               // 0000000039F8: E0511000 8005002A
	s_add_u32 m0, 0x700, s51                                   // 000000003A00: 807C33FF 00000700
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[48:49], v[208:209], 0// 000000003A08: D3F300BC 0A03A130  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[50:51], v[210:211], v[188:191]// 000000003A10: D3F300BC 0EF3A532  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[116:119], v47, s[92:95], 0 offen offset:1024// 000000003A18: E05C1400 8097742F
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[52:53], v[212:213], v[188:191]// 000000003A20: D3F300BC 0EF3A934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[54:55], v[214:215], v[188:191]// 000000003A28: D3F300BC 0EF3AD36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v43, s[20:23], 0 offen lds               // 000000003A30: E0511000 8005002B
	s_add_u32 m0, s51, s76                                     // 000000003A38: 807C4C33
	buffer_load_dword v7, s[28:31], 0 offen lds                // 000000003A3C: E0511000 80070007
	s_add_u32 m0, 0, s50                                       // 000000003A44: 807C3280
	buffer_load_dword v24, v13, s[32:35], 0 offen              // 000000003A48: E0501000 8008180D
	v_mul_f32_dpp v56, v23, v15 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000003A50: 0A701EFA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003A58: 7E720338  // Vector move
	v_pk_fma_f32 v[128:129], v[160:161], v[56:57], v[128:129]  // 000000003A5C: D3B04080 1E0271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[130:131], v[162:163], v[56:57], v[130:131]  // 000000003A64: D3B04082 1E0A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[136:137], v[168:169], v[56:57], v[136:137]  // 000000003A6C: D3B04088 1E2271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[138:139], v[170:171], v[56:57], v[138:139]  // 000000003A74: D3B0408A 1E2A71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v15 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000003A7C: 0A701EFA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003A84: 7E720338  // Vector move
	v_pk_fma_f32 v[144:145], v[176:177], v[56:57], v[144:145]  // 000000003A88: D3B04090 1E4271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[146:147], v[178:179], v[56:57], v[146:147]  // 000000003A90: D3B04092 1E4A71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[152:153], v[184:185], v[56:57], v[152:153]  // 000000003A98: D3B04098 1E6271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[154:155], v[186:187], v[56:57], v[154:155]  // 000000003AA0: D3B0409A 1E6A71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v16 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000003AA8: 0A7020FA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003AB0: 7E720338  // Vector move
	v_pk_fma_f32 v[132:133], v[164:165], v[56:57], v[132:133]  // 000000003AB4: D3B04084 1E1271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[134:135], v[166:167], v[56:57], v[134:135]  // 000000003ABC: D3B04086 1E1A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[140:141], v[172:173], v[56:57], v[140:141]  // 000000003AC4: D3B0408C 1E3271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[142:143], v[174:175], v[56:57], v[142:143]  // 000000003ACC: D3B0408E 1E3A71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v16 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000003AD4: 0A7020FA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003ADC: 7E720338  // Vector move
	v_pk_fma_f32 v[148:149], v[180:181], v[56:57], v[148:149]  // 000000003AE0: D3B04094 1E5271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[150:151], v[182:183], v[56:57], v[150:151]  // 000000003AE8: D3B04096 1E5A71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[156:157], v[188:189], v[56:57], v[156:157]  // 000000003AF0: D3B0409C 1E7271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[158:159], v[190:191], v[56:57], v[158:159]  // 000000003AF8: D3B0409E 1E7A71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_waitcnt vmcnt(22)                                        // 000000003B00: BF8C4F76  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[8:9], v[200:201], 0// 000000003B04: D3F300A0 0A039108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[10:11], v[202:203], v[160:163]// 000000003B0C: D3F300A0 0E83950A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[72:75], v44, s[92:95], 0 offen offset:2048// 000000003B14: E05C1800 8097482C
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[12:13], v[204:205], v[160:163]// 000000003B1C: D3F300A0 0E83990C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[14:15], v[206:207], v[160:163]// 000000003B24: D3F300A0 0E839D0E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[8:9], v[216:217], 0// 000000003B2C: D3F300A4 0A03B108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[10:11], v[218:219], v[164:167]// 000000003B34: D3F300A4 0E93B50A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[76:79], v44, s[92:95], 0 offen offset:3072// 000000003B3C: E05C1C00 80974C2C
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[12:13], v[220:221], v[164:167]// 000000003B44: D3F300A4 0E93B90C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[14:15], v[222:223], v[164:167]// 000000003B4C: D3F300A4 0E93BD0E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[24:25], v[200:201], 0// 000000003B54: D3F300A8 0A039118  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[26:27], v[202:203], v[168:171]// 000000003B5C: D3F300A8 0EA3951A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[88:91], v45, s[92:95], 0 offen offset:2048// 000000003B64: E05C1800 8097582D
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[28:29], v[204:205], v[168:171]// 000000003B6C: D3F300A8 0EA3991C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[30:31], v[206:207], v[168:171]// 000000003B74: D3F300A8 0EA39D1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[24:25], v[216:217], 0// 000000003B7C: D3F300AC 0A03B118  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[26:27], v[218:219], v[172:175]// 000000003B84: D3F300AC 0EB3B51A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[92:95], v45, s[92:95], 0 offen offset:3072// 000000003B8C: E05C1C00 80975C2D
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[28:29], v[220:221], v[172:175]// 000000003B94: D3F300AC 0EB3B91C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[30:31], v[222:223], v[172:175]// 000000003B9C: D3F300AC 0EB3BD1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	s_waitcnt vmcnt(22)                                        // 000000003BA4: BF8C4F76  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[40:41], v[200:201], 0// 000000003BA8: D3F300B0 0A039128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[42:43], v[202:203], v[176:179]// 000000003BB0: D3F300B0 0EC3952A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[104:107], v46, s[92:95], 0 offen offset:2048// 000000003BB8: E05C1800 8097682E
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[44:45], v[204:205], v[176:179]// 000000003BC0: D3F300B0 0EC3992C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[46:47], v[206:207], v[176:179]// 000000003BC8: D3F300B0 0EC39D2E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[40:41], v[216:217], 0// 000000003BD0: D3F300B4 0A03B128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[42:43], v[218:219], v[180:183]// 000000003BD8: D3F300B4 0ED3B52A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[108:111], v46, s[92:95], 0 offen offset:3072// 000000003BE0: E05C1C00 80976C2E
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[44:45], v[220:221], v[180:183]// 000000003BE8: D3F300B4 0ED3B92C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[46:47], v[222:223], v[180:183]// 000000003BF0: D3F300B4 0ED3BD2E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[56:57], v[200:201], 0// 000000003BF8: D3F300B8 0A039138  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[58:59], v[202:203], v[184:187]// 000000003C00: D3F300B8 0EE3953A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[120:123], v47, s[92:95], 0 offen offset:2048// 000000003C08: E05C1800 8097782F
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[60:61], v[204:205], v[184:187]// 000000003C10: D3F300B8 0EE3993C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[62:63], v[206:207], v[184:187]// 000000003C18: D3F300B8 0EE39D3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[56:57], v[216:217], 0// 000000003C20: D3F300BC 0A03B138  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[58:59], v[218:219], v[188:191]// 000000003C28: D3F300BC 0EF3B53A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[124:127], v47, s[92:95], 0 offen offset:3072// 000000003C30: E05C1C00 80977C2F
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[60:61], v[220:221], v[188:191]// 000000003C38: D3F300BC 0EF3B93C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[62:63], v[222:223], v[188:191]// 000000003C40: D3F300BC 0EF3BD3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v23, v17 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000003C48: 0A7022FA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003C50: 7E720338  // Vector move
	v_pk_fma_f32 v[128:129], v[160:161], v[56:57], v[128:129]  // 000000003C54: D3B04080 1E0271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[130:131], v[162:163], v[56:57], v[130:131]  // 000000003C5C: D3B04082 1E0A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[136:137], v[168:169], v[56:57], v[136:137]  // 000000003C64: D3B04088 1E2271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[138:139], v[170:171], v[56:57], v[138:139]  // 000000003C6C: D3B0408A 1E2A71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v17 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000003C74: 0A7022FA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003C7C: 7E720338  // Vector move
	v_pk_fma_f32 v[144:145], v[176:177], v[56:57], v[144:145]  // 000000003C80: D3B04090 1E4271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[146:147], v[178:179], v[56:57], v[146:147]  // 000000003C88: D3B04092 1E4A71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[152:153], v[184:185], v[56:57], v[152:153]  // 000000003C90: D3B04098 1E6271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[154:155], v[186:187], v[56:57], v[154:155]  // 000000003C98: D3B0409A 1E6A71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v18 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000003CA0: 0A7024FA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003CA8: 7E720338  // Vector move
	v_pk_fma_f32 v[132:133], v[164:165], v[56:57], v[132:133]  // 000000003CAC: D3B04084 1E1271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[134:135], v[166:167], v[56:57], v[134:135]  // 000000003CB4: D3B04086 1E1A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[140:141], v[172:173], v[56:57], v[140:141]  // 000000003CBC: D3B0408C 1E3271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[142:143], v[174:175], v[56:57], v[142:143]  // 000000003CC4: D3B0408E 1E3A71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v18 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000003CCC: 0A7024FA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003CD4: 7E720338  // Vector move
	v_pk_fma_f32 v[148:149], v[180:181], v[56:57], v[148:149]  // 000000003CD8: D3B04094 1E5271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[150:151], v[182:183], v[56:57], v[150:151]  // 000000003CE0: D3B04096 1E5A71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[156:157], v[188:189], v[56:57], v[156:157]  // 000000003CE8: D3B0409C 1E7271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[158:159], v[190:191], v[56:57], v[158:159]  // 000000003CF0: D3B0409E 1E7A71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x100, s80                                  // 000000003CF8: 803C50FF 00000100
	s_cmp_lt_u32 s60, s81                                      // 000000003D00: BF0A513C
	s_cselect_b32 s4, s4, 0                                    // 000000003D04: 85048004
	s_add_u32 s32, s4, s32                                     // 000000003D08: 80202004
	s_addc_u32 s33, 0, s33                                     // 000000003D0C: 82212180
	s_waitcnt vmcnt(8)                                         // 000000003D10: BF8C0F78  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 000000003D14: BF8A0000  // Workgroup barrier synchronization
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[64:65], v[192:193], 0// 000000003D18: D3F30060 0A038140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[66:67], v[194:195], v[96:99]// 000000003D20: D3F30060 0D838542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[0:3], v44, s[24:27], 0 offen         // 000000003D28: E05C1000 8086002C
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[68:69], v[196:197], v[96:99]// 000000003D30: D3F30060 0D838944  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[70:71], v[198:199], v[96:99]// 000000003D38: D3F30060 0D838D46  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v23, v11, s[32:35], 0 offen              // 000000003D40: E0501000 8008170B
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[64:65], v[208:209], 0// 000000003D48: D3F30064 0A03A140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[66:67], v[210:211], v[100:103]// 000000003D50: D3F30064 0D93A542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[4:7], v44, s[24:27], 0 offen offset:1024// 000000003D58: E05C1400 8086042C
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[68:69], v[212:213], v[100:103]// 000000003D60: D3F30064 0D93A944  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[70:71], v[214:215], v[100:103]// 000000003D68: D3F30064 0D93AD46  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[80:81], v[192:193], 0// 000000003D70: D3F30068 0A038150  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[82:83], v[194:195], v[104:107]// 000000003D78: D3F30068 0DA38552  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[16:19], v45, s[24:27], 0 offen       // 000000003D80: E05C1000 8086102D
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[84:85], v[196:197], v[104:107]// 000000003D88: D3F30068 0DA38954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[86:87], v[198:199], v[104:107]// 000000003D90: D3F30068 0DA38D56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[80:81], v[208:209], 0// 000000003D98: D3F3006C 0A03A150  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[82:83], v[210:211], v[108:111]// 000000003DA0: D3F3006C 0DB3A552  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[20:23], v45, s[24:27], 0 offen offset:1024// 000000003DA8: E05C1400 8086142D
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[84:85], v[212:213], v[108:111]// 000000003DB0: D3F3006C 0DB3A954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[86:87], v[214:215], v[108:111]// 000000003DB8: D3F3006C 0DB3AD56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[96:97], v[192:193], 0// 000000003DC0: D3F30070 0A038160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[98:99], v[194:195], v[112:115]// 000000003DC8: D3F30070 0DC38562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[32:35], v46, s[24:27], 0 offen       // 000000003DD0: E05C1000 8086202E
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[100:101], v[196:197], v[112:115]// 000000003DD8: D3F30070 0DC38964  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[102:103], v[198:199], v[112:115]// 000000003DE0: D3F30070 0DC38D66  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[96:97], v[208:209], 0// 000000003DE8: D3F30074 0A03A160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[98:99], v[210:211], v[116:119]// 000000003DF0: D3F30074 0DD3A562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[36:39], v46, s[24:27], 0 offen offset:1024// 000000003DF8: E05C1400 8086242E
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[100:101], v[212:213], v[116:119]// 000000003E00: D3F30074 0DD3A964  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[102:103], v[214:215], v[116:119]// 000000003E08: D3F30074 0DD3AD66  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[112:113], v[192:193], 0// 000000003E10: D3F30078 0A038170  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[114:115], v[194:195], v[120:123]// 000000003E18: D3F30078 0DE38572  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[48:51], v47, s[24:27], 0 offen       // 000000003E20: E05C1000 8086302F
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[116:117], v[196:197], v[120:123]// 000000003E28: D3F30078 0DE38974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[118:119], v[198:199], v[120:123]// 000000003E30: D3F30078 0DE38D76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[112:113], v[208:209], 0// 000000003E38: D3F3007C 0A03A170  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[114:115], v[210:211], v[124:127]// 000000003E40: D3F3007C 0DF3A572  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[52:55], v47, s[24:27], 0 offen offset:1024// 000000003E48: E05C1400 8086342F
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[116:117], v[212:213], v[124:127]// 000000003E50: D3F3007C 0DF3A974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[118:119], v[214:215], v[124:127]// 000000003E58: D3F3007C 0DF3AD76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v24, v15 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000003E60: 0A701EFA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003E68: 7E720338  // Vector move
	v_pk_fma_f32 v[64:65], v[96:97], v[56:57], v[64:65]        // 000000003E6C: D3B04040 1D027160  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[66:67], v[98:99], v[56:57], v[66:67]        // 000000003E74: D3B04042 1D0A7162  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[72:73], v[104:105], v[56:57], v[72:73]      // 000000003E7C: D3B04048 1D227168  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[74:75], v[106:107], v[56:57], v[74:75]      // 000000003E84: D3B0404A 1D2A716A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v15 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000003E8C: 0A701EFA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003E94: 7E720338  // Vector move
	v_pk_fma_f32 v[80:81], v[112:113], v[56:57], v[80:81]      // 000000003E98: D3B04050 1D427170  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[82:83], v[114:115], v[56:57], v[82:83]      // 000000003EA0: D3B04052 1D4A7172  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[88:89], v[120:121], v[56:57], v[88:89]      // 000000003EA8: D3B04058 1D627178  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[90:91], v[122:123], v[56:57], v[90:91]      // 000000003EB0: D3B0405A 1D6A717A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v16 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000003EB8: 0A7020FA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003EC0: 7E720338  // Vector move
	v_pk_fma_f32 v[68:69], v[100:101], v[56:57], v[68:69]      // 000000003EC4: D3B04044 1D127164  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[70:71], v[102:103], v[56:57], v[70:71]      // 000000003ECC: D3B04046 1D1A7166  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[76:77], v[108:109], v[56:57], v[76:77]      // 000000003ED4: D3B0404C 1D32716C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[78:79], v[110:111], v[56:57], v[78:79]      // 000000003EDC: D3B0404E 1D3A716E  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v16 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000003EE4: 0A7020FA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000003EEC: 7E720338  // Vector move
	v_pk_fma_f32 v[84:85], v[116:117], v[56:57], v[84:85]      // 000000003EF0: D3B04054 1D527174  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[86:87], v[118:119], v[56:57], v[86:87]      // 000000003EF8: D3B04056 1D5A7176  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[92:93], v[124:125], v[56:57], v[92:93]      // 000000003F00: D3B0405C 1D72717C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[94:95], v[126:127], v[56:57], v[94:95]      // 000000003F08: D3B0405E 1D7A717E  // Packed FMA: 2x FP32 fused multiply-add
	s_waitcnt vmcnt(13)                                        // 000000003F10: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[72:73], v[200:201], 0// 000000003F14: D3F30060 0A039148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[74:75], v[202:203], v[96:99]// 000000003F1C: D3F30060 0D83954A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[8:11], v44, s[24:27], 0 offen offset:2048// 000000003F24: E05C1800 8086082C
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[76:77], v[204:205], v[96:99]// 000000003F2C: D3F30060 0D83994C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[78:79], v[206:207], v[96:99]// 000000003F34: D3F30060 0D839D4E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[224:227], v2 offset:9344                    // 000000003F3C: D9FE2480 E0000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v19, v3 offset:17664                           // 000000003F44: D86C4500 13000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[72:73], v[216:217], 0// 000000003F4C: D3F30064 0A03B148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[74:75], v[218:219], v[100:103]// 000000003F54: D3F30064 0D93B54A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[12:15], v44, s[24:27], 0 offen offset:3072// 000000003F5C: E05C1C00 80860C2C
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[76:77], v[220:221], v[100:103]// 000000003F64: D3F30064 0D93B94C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[78:79], v[222:223], v[100:103]// 000000003F6C: D3F30064 0D93BD4E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[228:231], v2 offset:9408                    // 000000003F74: D9FE24C0 E4000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v20, v3 offset:17920                           // 000000003F7C: D86C4600 14000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[88:89], v[200:201], 0// 000000003F84: D3F30068 0A039158  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[90:91], v[202:203], v[104:107]// 000000003F8C: D3F30068 0DA3955A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[24:27], v45, s[24:27], 0 offen offset:2048// 000000003F94: E05C1800 8086182D
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[92:93], v[204:205], v[104:107]// 000000003F9C: D3F30068 0DA3995C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[94:95], v[206:207], v[104:107]// 000000003FA4: D3F30068 0DA39D5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[232:235], v2 offset:9472                    // 000000003FAC: D9FE2500 E8000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v21, v3 offset:18176                           // 000000003FB4: D86C4700 15000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[88:89], v[216:217], 0// 000000003FBC: D3F3006C 0A03B158  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[90:91], v[218:219], v[108:111]// 000000003FC4: D3F3006C 0DB3B55A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[28:31], v45, s[24:27], 0 offen offset:3072// 000000003FCC: E05C1C00 80861C2D
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[92:93], v[220:221], v[108:111]// 000000003FD4: D3F3006C 0DB3B95C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[94:95], v[222:223], v[108:111]// 000000003FDC: D3F3006C 0DB3BD5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[236:239], v2 offset:9536                    // 000000003FE4: D9FE2540 EC000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v22, v3 offset:18432                           // 000000003FEC: D86C4800 16000003  // Read 32-bit from LDS (shared memory)
	s_waitcnt vmcnt(13)                                        // 000000003FF4: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[104:105], v[200:201], 0// 000000003FF8: D3F30070 0A039168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[106:107], v[202:203], v[112:115]// 000000004000: D3F30070 0DC3956A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[40:43], v46, s[24:27], 0 offen offset:2048// 000000004008: E05C1800 8086282E
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[108:109], v[204:205], v[112:115]// 000000004010: D3F30070 0DC3996C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[110:111], v[206:207], v[112:115]// 000000004018: D3F30070 0DC39D6E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[240:243], v2 offset:10368                   // 000000004020: D9FE2880 F0000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[104:105], v[216:217], 0// 000000004028: D3F30074 0A03B168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[106:107], v[218:219], v[116:119]// 000000004030: D3F30074 0DD3B56A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[44:47], v46, s[24:27], 0 offen offset:3072// 000000004038: E05C1C00 80862C2E
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[108:109], v[220:221], v[116:119]// 000000004040: D3F30074 0DD3B96C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[110:111], v[222:223], v[116:119]// 000000004048: D3F30074 0DD3BD6E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[244:247], v2 offset:10432                   // 000000004050: D9FE28C0 F4000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[120:121], v[200:201], 0// 000000004058: D3F30078 0A039178  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[122:123], v[202:203], v[120:123]// 000000004060: D3F30078 0DE3957A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[56:59], v47, s[24:27], 0 offen offset:2048// 000000004068: E05C1800 8086382F
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[124:125], v[204:205], v[120:123]// 000000004070: D3F30078 0DE3997C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[126:127], v[206:207], v[120:123]// 000000004078: D3F30078 0DE39D7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[248:251], v2 offset:10496                   // 000000004080: D9FE2900 F8000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[120:121], v[216:217], 0// 000000004088: D3F3007C 0A03B178  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[122:123], v[218:219], v[124:127]// 000000004090: D3F3007C 0DF3B57A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[60:63], v47, s[24:27], 0 offen offset:3072// 000000004098: E05C1C00 80863C2F
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[124:125], v[220:221], v[124:127]// 0000000040A0: D3F3007C 0DF3B97C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[126:127], v[222:223], v[124:127]// 0000000040A8: D3F3007C 0DF3BD7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[252:255], v2 offset:10560                   // 0000000040B0: D9FE2940 FC000002  // Read 128-bit from LDS (4x32-bit)
	v_mul_f32_dpp v56, v24, v17 row_newbcast:2 row_mask:0xf bank_mask:0xf// 0000000040B8: 0A7022FA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000040C0: 7E720338  // Vector move
	v_pk_fma_f32 v[64:65], v[96:97], v[56:57], v[64:65]        // 0000000040C4: D3B04040 1D027160  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[66:67], v[98:99], v[56:57], v[66:67]        // 0000000040CC: D3B04042 1D0A7162  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[72:73], v[104:105], v[56:57], v[72:73]      // 0000000040D4: D3B04048 1D227168  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[74:75], v[106:107], v[56:57], v[74:75]      // 0000000040DC: D3B0404A 1D2A716A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v17 row_newbcast:3 row_mask:0xf bank_mask:0xf// 0000000040E4: 0A7022FA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000040EC: 7E720338  // Vector move
	v_pk_fma_f32 v[80:81], v[112:113], v[56:57], v[80:81]      // 0000000040F0: D3B04050 1D427170  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[82:83], v[114:115], v[56:57], v[82:83]      // 0000000040F8: D3B04052 1D4A7172  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[88:89], v[120:121], v[56:57], v[88:89]      // 000000004100: D3B04058 1D627178  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[90:91], v[122:123], v[56:57], v[90:91]      // 000000004108: D3B0405A 1D6A717A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v18 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000004110: 0A7024FA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000004118: 7E720338  // Vector move
	v_pk_fma_f32 v[68:69], v[100:101], v[56:57], v[68:69]      // 00000000411C: D3B04044 1D127164  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[70:71], v[102:103], v[56:57], v[70:71]      // 000000004124: D3B04046 1D1A7166  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[76:77], v[108:109], v[56:57], v[76:77]      // 00000000412C: D3B0404C 1D32716C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[78:79], v[110:111], v[56:57], v[78:79]      // 000000004134: D3B0404E 1D3A716E  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v18 row_newbcast:3 row_mask:0xf bank_mask:0xf// 00000000413C: 0A7024FA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000004144: 7E720338  // Vector move
	v_pk_fma_f32 v[84:85], v[116:117], v[56:57], v[84:85]      // 000000004148: D3B04054 1D527174  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[86:87], v[118:119], v[56:57], v[86:87]      // 000000004150: D3B04056 1D5A7176  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[92:93], v[124:125], v[56:57], v[92:93]      // 000000004158: D3B0405C 1D72717C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[94:95], v[126:127], v[56:57], v[94:95]      // 000000004160: D3B0405E 1D7A717E  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x200, s80                                  // 000000004168: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 000000004170: BF0A513C
	s_cselect_b32 s57, s57, 0                                  // 000000004174: 85398039
	s_cselect_b32 s3, s3, 0                                    // 000000004178: 85038003
	s_add_u32 s60, 0x200, s80                                  // 00000000417C: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 000000004184: BF0A513C
	s_cselect_b32 s58, s58, 0                                  // 000000004188: 853A803A
	s_add_u32 s20, s57, s20                                    // 00000000418C: 80141439
	s_addc_u32 s21, 0, s21                                     // 000000004190: 82151580
	s_add_u32 s28, s3, s28                                     // 000000004194: 801C1C03
	s_addc_u32 s29, 0, s29                                     // 000000004198: 821D1D80
	s_add_u32 s24, s58, s24                                    // 00000000419C: 8018183A
	s_addc_u32 s25, 0, s25                                     // 0000000041A0: 82191980
	s_add_u32 s92, s90, s92                                    // 0000000041A4: 805C5C5A
	s_addc_u32 s93, 0, s93                                     // 0000000041A8: 825D5D80
	s_addk_i32 s80, 0x100                                      // 0000000041AC: B7500100
	s_cmp_lt_i32 s80, s81                                      // 0000000041B0: BF045150
	s_cbranch_scc0 label_073F                                  // 0000000041B4: BF840251  // Branch if SCC==0 (condition false)
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(8) lgkmcnt(0)                              // 0000000041B8: BF8C0078  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 0000000041BC: BF8A0000  // Workgroup barrier synchronization

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[0:1], v[224:225], 0// 0000000041C0: D3F300A0 0A03C100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[2:3], v[226:227], v[160:163]// 0000000041C8: D3F300A0 0E83C502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[64:67], v44, s[92:95], 0 offen       // 0000000041D0: E05C1000 8097402C
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[4:5], v[228:229], v[160:163]// 0000000041D8: D3F300A0 0E83C904  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[6:7], v[230:231], v[160:163]// 0000000041E0: D3F300A0 0E83CD06  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v36, s[20:23], 0 offen lds               // 0000000041E8: E0511000 80050024
	s_add_u32 m0, 0x100, s50                                   // 0000000041F0: 807C32FF 00000100
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[0:1], v[240:241], 0// 0000000041F8: D3F300A4 0A03E100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[2:3], v[242:243], v[164:167]// 000000004200: D3F300A4 0E93E502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[68:71], v44, s[92:95], 0 offen offset:1024// 000000004208: E05C1400 8097442C
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[4:5], v[244:245], v[164:167]// 000000004210: D3F300A4 0E93E904  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[6:7], v[246:247], v[164:167]// 000000004218: D3F300A4 0E93ED06  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v37, s[20:23], 0 offen lds               // 000000004220: E0511000 80050025
	s_add_u32 m0, 0x200, s50                                   // 000000004228: 807C32FF 00000200
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[16:17], v[224:225], 0// 000000004230: D3F300A8 0A03C110  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[18:19], v[226:227], v[168:171]// 000000004238: D3F300A8 0EA3C512  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[80:83], v45, s[92:95], 0 offen       // 000000004240: E05C1000 8097502D
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[20:21], v[228:229], v[168:171]// 000000004248: D3F300A8 0EA3C914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[22:23], v[230:231], v[168:171]// 000000004250: D3F300A8 0EA3CD16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v38, s[20:23], 0 offen lds               // 000000004258: E0511000 80050026
	s_add_u32 m0, 0x300, s50                                   // 000000004260: 807C32FF 00000300
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[16:17], v[240:241], 0// 000000004268: D3F300AC 0A03E110  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[18:19], v[242:243], v[172:175]// 000000004270: D3F300AC 0EB3E512  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[84:87], v45, s[92:95], 0 offen offset:1024// 000000004278: E05C1400 8097542D
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[20:21], v[244:245], v[172:175]// 000000004280: D3F300AC 0EB3E914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[22:23], v[246:247], v[172:175]// 000000004288: D3F300AC 0EB3ED16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v39, s[20:23], 0 offen lds               // 000000004290: E0511000 80050027
	s_add_u32 m0, 0x400, s50                                   // 000000004298: 807C32FF 00000400
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[32:33], v[224:225], 0// 0000000042A0: D3F300B0 0A03C120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[34:35], v[226:227], v[176:179]// 0000000042A8: D3F300B0 0EC3C522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[96:99], v46, s[92:95], 0 offen       // 0000000042B0: E05C1000 8097602E
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[36:37], v[228:229], v[176:179]// 0000000042B8: D3F300B0 0EC3C924  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[38:39], v[230:231], v[176:179]// 0000000042C0: D3F300B0 0EC3CD26  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v40, s[20:23], 0 offen lds               // 0000000042C8: E0511000 80050028
	s_add_u32 m0, 0x500, s50                                   // 0000000042D0: 807C32FF 00000500
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[32:33], v[240:241], 0// 0000000042D8: D3F300B4 0A03E120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[34:35], v[242:243], v[180:183]// 0000000042E0: D3F300B4 0ED3E522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[100:103], v46, s[92:95], 0 offen offset:1024// 0000000042E8: E05C1400 8097642E
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[36:37], v[244:245], v[180:183]// 0000000042F0: D3F300B4 0ED3E924  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[38:39], v[246:247], v[180:183]// 0000000042F8: D3F300B4 0ED3ED26  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v41, s[20:23], 0 offen lds               // 000000004300: E0511000 80050029
	s_add_u32 m0, 0x600, s50                                   // 000000004308: 807C32FF 00000600
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[48:49], v[224:225], 0// 000000004310: D3F300B8 0A03C130  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[50:51], v[226:227], v[184:187]// 000000004318: D3F300B8 0EE3C532  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[112:115], v47, s[92:95], 0 offen     // 000000004320: E05C1000 8097702F
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[52:53], v[228:229], v[184:187]// 000000004328: D3F300B8 0EE3C934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[54:55], v[230:231], v[184:187]// 000000004330: D3F300B8 0EE3CD36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v42, s[20:23], 0 offen lds               // 000000004338: E0511000 8005002A
	s_add_u32 m0, 0x700, s50                                   // 000000004340: 807C32FF 00000700
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[48:49], v[240:241], 0// 000000004348: D3F300BC 0A03E130  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[50:51], v[242:243], v[188:191]// 000000004350: D3F300BC 0EF3E532  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[116:119], v47, s[92:95], 0 offen offset:1024// 000000004358: E05C1400 8097742F
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[52:53], v[244:245], v[188:191]// 000000004360: D3F300BC 0EF3E934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[54:55], v[246:247], v[188:191]// 000000004368: D3F300BC 0EF3ED36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v43, s[20:23], 0 offen lds               // 000000004370: E0511000 8005002B
	s_add_u32 m0, s50, s76                                     // 000000004378: 807C4C32
	buffer_load_dword v7, s[28:31], 0 offen lds                // 00000000437C: E0511000 80070007
	s_add_u32 m0, 0, s51                                       // 000000004384: 807C3380
	buffer_load_dword v24, v13, s[32:35], 0 offen              // 000000004388: E0501000 8008180D
	v_mul_f32_dpp v56, v23, v19 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000004390: 0A7026FA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000004398: 7E720338  // Vector move
	v_pk_fma_f32 v[128:129], v[160:161], v[56:57], v[128:129]  // 00000000439C: D3B04080 1E0271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[130:131], v[162:163], v[56:57], v[130:131]  // 0000000043A4: D3B04082 1E0A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[136:137], v[168:169], v[56:57], v[136:137]  // 0000000043AC: D3B04088 1E2271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[138:139], v[170:171], v[56:57], v[138:139]  // 0000000043B4: D3B0408A 1E2A71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v19 row_newbcast:1 row_mask:0xf bank_mask:0xf// 0000000043BC: 0A7026FA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000043C4: 7E720338  // Vector move
	v_pk_fma_f32 v[144:145], v[176:177], v[56:57], v[144:145]  // 0000000043C8: D3B04090 1E4271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[146:147], v[178:179], v[56:57], v[146:147]  // 0000000043D0: D3B04092 1E4A71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[152:153], v[184:185], v[56:57], v[152:153]  // 0000000043D8: D3B04098 1E6271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[154:155], v[186:187], v[56:57], v[154:155]  // 0000000043E0: D3B0409A 1E6A71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v20 row_newbcast:0 row_mask:0xf bank_mask:0xf// 0000000043E8: 0A7028FA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000043F0: 7E720338  // Vector move
	v_pk_fma_f32 v[132:133], v[164:165], v[56:57], v[132:133]  // 0000000043F4: D3B04084 1E1271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[134:135], v[166:167], v[56:57], v[134:135]  // 0000000043FC: D3B04086 1E1A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[140:141], v[172:173], v[56:57], v[140:141]  // 000000004404: D3B0408C 1E3271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[142:143], v[174:175], v[56:57], v[142:143]  // 00000000440C: D3B0408E 1E3A71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v20 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000004414: 0A7028FA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 00000000441C: 7E720338  // Vector move
	v_pk_fma_f32 v[148:149], v[180:181], v[56:57], v[148:149]  // 000000004420: D3B04094 1E5271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[150:151], v[182:183], v[56:57], v[150:151]  // 000000004428: D3B04096 1E5A71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[156:157], v[188:189], v[56:57], v[156:157]  // 000000004430: D3B0409C 1E7271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[158:159], v[190:191], v[56:57], v[158:159]  // 000000004438: D3B0409E 1E7A71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_waitcnt vmcnt(22)                                        // 000000004440: BF8C4F76  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[8:9], v[232:233], 0// 000000004444: D3F300A0 0A03D108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[10:11], v[234:235], v[160:163]// 00000000444C: D3F300A0 0E83D50A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[72:75], v44, s[92:95], 0 offen offset:2048// 000000004454: E05C1800 8097482C
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[12:13], v[236:237], v[160:163]// 00000000445C: D3F300A0 0E83D90C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[14:15], v[238:239], v[160:163]// 000000004464: D3F300A0 0E83DD0E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[8:9], v[248:249], 0// 00000000446C: D3F300A4 0A03F108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[10:11], v[250:251], v[164:167]// 000000004474: D3F300A4 0E93F50A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[76:79], v44, s[92:95], 0 offen offset:3072// 00000000447C: E05C1C00 80974C2C
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[12:13], v[252:253], v[164:167]// 000000004484: D3F300A4 0E93F90C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[14:15], v[254:255], v[164:167]// 00000000448C: D3F300A4 0E93FD0E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[24:25], v[232:233], 0// 000000004494: D3F300A8 0A03D118  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[26:27], v[234:235], v[168:171]// 00000000449C: D3F300A8 0EA3D51A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[88:91], v45, s[92:95], 0 offen offset:2048// 0000000044A4: E05C1800 8097582D
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[28:29], v[236:237], v[168:171]// 0000000044AC: D3F300A8 0EA3D91C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[30:31], v[238:239], v[168:171]// 0000000044B4: D3F300A8 0EA3DD1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[24:25], v[248:249], 0// 0000000044BC: D3F300AC 0A03F118  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[26:27], v[250:251], v[172:175]// 0000000044C4: D3F300AC 0EB3F51A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[92:95], v45, s[92:95], 0 offen offset:3072// 0000000044CC: E05C1C00 80975C2D
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[28:29], v[252:253], v[172:175]// 0000000044D4: D3F300AC 0EB3F91C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[30:31], v[254:255], v[172:175]// 0000000044DC: D3F300AC 0EB3FD1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	s_waitcnt vmcnt(22)                                        // 0000000044E4: BF8C4F76  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[40:41], v[232:233], 0// 0000000044E8: D3F300B0 0A03D128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[42:43], v[234:235], v[176:179]// 0000000044F0: D3F300B0 0EC3D52A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[104:107], v46, s[92:95], 0 offen offset:2048// 0000000044F8: E05C1800 8097682E
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[44:45], v[236:237], v[176:179]// 000000004500: D3F300B0 0EC3D92C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[46:47], v[238:239], v[176:179]// 000000004508: D3F300B0 0EC3DD2E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[40:41], v[248:249], 0// 000000004510: D3F300B4 0A03F128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[42:43], v[250:251], v[180:183]// 000000004518: D3F300B4 0ED3F52A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[108:111], v46, s[92:95], 0 offen offset:3072// 000000004520: E05C1C00 80976C2E
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[44:45], v[252:253], v[180:183]// 000000004528: D3F300B4 0ED3F92C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[46:47], v[254:255], v[180:183]// 000000004530: D3F300B4 0ED3FD2E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[56:57], v[232:233], 0// 000000004538: D3F300B8 0A03D138  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[58:59], v[234:235], v[184:187]// 000000004540: D3F300B8 0EE3D53A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[120:123], v47, s[92:95], 0 offen offset:2048// 000000004548: E05C1800 8097782F
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[60:61], v[236:237], v[184:187]// 000000004550: D3F300B8 0EE3D93C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[62:63], v[238:239], v[184:187]// 000000004558: D3F300B8 0EE3DD3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[56:57], v[248:249], 0// 000000004560: D3F300BC 0A03F138  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[58:59], v[250:251], v[188:191]// 000000004568: D3F300BC 0EF3F53A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[124:127], v47, s[92:95], 0 offen offset:3072// 000000004570: E05C1C00 80977C2F
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[60:61], v[252:253], v[188:191]// 000000004578: D3F300BC 0EF3F93C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[62:63], v[254:255], v[188:191]// 000000004580: D3F300BC 0EF3FD3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v23, v21 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000004588: 0A702AFA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000004590: 7E720338  // Vector move
	v_pk_fma_f32 v[128:129], v[160:161], v[56:57], v[128:129]  // 000000004594: D3B04080 1E0271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[130:131], v[162:163], v[56:57], v[130:131]  // 00000000459C: D3B04082 1E0A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[136:137], v[168:169], v[56:57], v[136:137]  // 0000000045A4: D3B04088 1E2271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[138:139], v[170:171], v[56:57], v[138:139]  // 0000000045AC: D3B0408A 1E2A71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v21 row_newbcast:3 row_mask:0xf bank_mask:0xf// 0000000045B4: 0A702AFA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000045BC: 7E720338  // Vector move
	v_pk_fma_f32 v[144:145], v[176:177], v[56:57], v[144:145]  // 0000000045C0: D3B04090 1E4271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[146:147], v[178:179], v[56:57], v[146:147]  // 0000000045C8: D3B04092 1E4A71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[152:153], v[184:185], v[56:57], v[152:153]  // 0000000045D0: D3B04098 1E6271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[154:155], v[186:187], v[56:57], v[154:155]  // 0000000045D8: D3B0409A 1E6A71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v22 row_newbcast:2 row_mask:0xf bank_mask:0xf// 0000000045E0: 0A702CFA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000045E8: 7E720338  // Vector move
	v_pk_fma_f32 v[132:133], v[164:165], v[56:57], v[132:133]  // 0000000045EC: D3B04084 1E1271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[134:135], v[166:167], v[56:57], v[134:135]  // 0000000045F4: D3B04086 1E1A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[140:141], v[172:173], v[56:57], v[140:141]  // 0000000045FC: D3B0408C 1E3271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[142:143], v[174:175], v[56:57], v[142:143]  // 000000004604: D3B0408E 1E3A71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v22 row_newbcast:3 row_mask:0xf bank_mask:0xf// 00000000460C: 0A702CFA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000004614: 7E720338  // Vector move
	v_pk_fma_f32 v[148:149], v[180:181], v[56:57], v[148:149]  // 000000004618: D3B04094 1E5271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[150:151], v[182:183], v[56:57], v[150:151]  // 000000004620: D3B04096 1E5A71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[156:157], v[188:189], v[56:57], v[156:157]  // 000000004628: D3B0409C 1E7271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[158:159], v[190:191], v[56:57], v[158:159]  // 000000004630: D3B0409E 1E7A71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x100, s80                                  // 000000004638: 803C50FF 00000100
	s_cmp_lt_u32 s60, s81                                      // 000000004640: BF0A513C
	s_cselect_b32 s4, s4, 0                                    // 000000004644: 85048004
	s_add_u32 s32, s4, s32                                     // 000000004648: 80202004
	s_addc_u32 s33, 0, s33                                     // 00000000464C: 82212180
	s_waitcnt vmcnt(8)                                         // 000000004650: BF8C0F78  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 000000004654: BF8A0000  // Workgroup barrier synchronization
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[64:65], v[224:225], 0// 000000004658: D3F30060 0A03C140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[66:67], v[226:227], v[96:99]// 000000004660: D3F30060 0D83C542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[0:3], v44, s[24:27], 0 offen         // 000000004668: E05C1000 8086002C
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[68:69], v[228:229], v[96:99]// 000000004670: D3F30060 0D83C944  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[70:71], v[230:231], v[96:99]// 000000004678: D3F30060 0D83CD46  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v23, v11, s[32:35], 0 offen              // 000000004680: E0501000 8008170B
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[64:65], v[240:241], 0// 000000004688: D3F30064 0A03E140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[66:67], v[242:243], v[100:103]// 000000004690: D3F30064 0D93E542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[4:7], v44, s[24:27], 0 offen offset:1024// 000000004698: E05C1400 8086042C
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[68:69], v[244:245], v[100:103]// 0000000046A0: D3F30064 0D93E944  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[70:71], v[246:247], v[100:103]// 0000000046A8: D3F30064 0D93ED46  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[80:81], v[224:225], 0// 0000000046B0: D3F30068 0A03C150  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[82:83], v[226:227], v[104:107]// 0000000046B8: D3F30068 0DA3C552  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[16:19], v45, s[24:27], 0 offen       // 0000000046C0: E05C1000 8086102D
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[84:85], v[228:229], v[104:107]// 0000000046C8: D3F30068 0DA3C954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[86:87], v[230:231], v[104:107]// 0000000046D0: D3F30068 0DA3CD56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[80:81], v[240:241], 0// 0000000046D8: D3F3006C 0A03E150  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[82:83], v[242:243], v[108:111]// 0000000046E0: D3F3006C 0DB3E552  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[20:23], v45, s[24:27], 0 offen offset:1024// 0000000046E8: E05C1400 8086142D
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[84:85], v[244:245], v[108:111]// 0000000046F0: D3F3006C 0DB3E954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[86:87], v[246:247], v[108:111]// 0000000046F8: D3F3006C 0DB3ED56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[96:97], v[224:225], 0// 000000004700: D3F30070 0A03C160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[98:99], v[226:227], v[112:115]// 000000004708: D3F30070 0DC3C562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[32:35], v46, s[24:27], 0 offen       // 000000004710: E05C1000 8086202E
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[100:101], v[228:229], v[112:115]// 000000004718: D3F30070 0DC3C964  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[102:103], v[230:231], v[112:115]// 000000004720: D3F30070 0DC3CD66  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[96:97], v[240:241], 0// 000000004728: D3F30074 0A03E160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[98:99], v[242:243], v[116:119]// 000000004730: D3F30074 0DD3E562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[36:39], v46, s[24:27], 0 offen offset:1024// 000000004738: E05C1400 8086242E
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[100:101], v[244:245], v[116:119]// 000000004740: D3F30074 0DD3E964  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[102:103], v[246:247], v[116:119]// 000000004748: D3F30074 0DD3ED66  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[112:113], v[224:225], 0// 000000004750: D3F30078 0A03C170  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[114:115], v[226:227], v[120:123]// 000000004758: D3F30078 0DE3C572  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[48:51], v47, s[24:27], 0 offen       // 000000004760: E05C1000 8086302F
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[116:117], v[228:229], v[120:123]// 000000004768: D3F30078 0DE3C974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[118:119], v[230:231], v[120:123]// 000000004770: D3F30078 0DE3CD76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[112:113], v[240:241], 0// 000000004778: D3F3007C 0A03E170  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[114:115], v[242:243], v[124:127]// 000000004780: D3F3007C 0DF3E572  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[52:55], v47, s[24:27], 0 offen offset:1024// 000000004788: E05C1400 8086342F
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[116:117], v[244:245], v[124:127]// 000000004790: D3F3007C 0DF3E974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[118:119], v[246:247], v[124:127]// 000000004798: D3F3007C 0DF3ED76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v24, v19 row_newbcast:0 row_mask:0xf bank_mask:0xf// 0000000047A0: 0A7026FA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000047A8: 7E720338  // Vector move
	v_pk_fma_f32 v[64:65], v[96:97], v[56:57], v[64:65]        // 0000000047AC: D3B04040 1D027160  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[66:67], v[98:99], v[56:57], v[66:67]        // 0000000047B4: D3B04042 1D0A7162  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[72:73], v[104:105], v[56:57], v[72:73]      // 0000000047BC: D3B04048 1D227168  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[74:75], v[106:107], v[56:57], v[74:75]      // 0000000047C4: D3B0404A 1D2A716A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v19 row_newbcast:1 row_mask:0xf bank_mask:0xf// 0000000047CC: 0A7026FA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000047D4: 7E720338  // Vector move
	v_pk_fma_f32 v[80:81], v[112:113], v[56:57], v[80:81]      // 0000000047D8: D3B04050 1D427170  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[82:83], v[114:115], v[56:57], v[82:83]      // 0000000047E0: D3B04052 1D4A7172  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[88:89], v[120:121], v[56:57], v[88:89]      // 0000000047E8: D3B04058 1D627178  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[90:91], v[122:123], v[56:57], v[90:91]      // 0000000047F0: D3B0405A 1D6A717A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v20 row_newbcast:0 row_mask:0xf bank_mask:0xf// 0000000047F8: 0A7028FA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000004800: 7E720338  // Vector move
	v_pk_fma_f32 v[68:69], v[100:101], v[56:57], v[68:69]      // 000000004804: D3B04044 1D127164  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[70:71], v[102:103], v[56:57], v[70:71]      // 00000000480C: D3B04046 1D1A7166  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[76:77], v[108:109], v[56:57], v[76:77]      // 000000004814: D3B0404C 1D32716C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[78:79], v[110:111], v[56:57], v[78:79]      // 00000000481C: D3B0404E 1D3A716E  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v20 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000004824: 0A7028FA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 00000000482C: 7E720338  // Vector move
	v_pk_fma_f32 v[84:85], v[116:117], v[56:57], v[84:85]      // 000000004830: D3B04054 1D527174  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[86:87], v[118:119], v[56:57], v[86:87]      // 000000004838: D3B04056 1D5A7176  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[92:93], v[124:125], v[56:57], v[92:93]      // 000000004840: D3B0405C 1D72717C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[94:95], v[126:127], v[56:57], v[94:95]      // 000000004848: D3B0405E 1D7A717E  // Packed FMA: 2x FP32 fused multiply-add
	s_waitcnt vmcnt(13)                                        // 000000004850: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[72:73], v[232:233], 0// 000000004854: D3F30060 0A03D148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[74:75], v[234:235], v[96:99]// 00000000485C: D3F30060 0D83D54A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[8:11], v44, s[24:27], 0 offen offset:2048// 000000004864: E05C1800 8086082C
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[76:77], v[236:237], v[96:99]// 00000000486C: D3F30060 0D83D94C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[78:79], v[238:239], v[96:99]// 000000004874: D3F30060 0D83DD4E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[192:195], v2                                // 00000000487C: D9FE0000 C0000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v15, v3 offset:8320                            // 000000004884: D86C2080 0F000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[72:73], v[248:249], 0// 00000000488C: D3F30064 0A03F148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[74:75], v[250:251], v[100:103]// 000000004894: D3F30064 0D93F54A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[12:15], v44, s[24:27], 0 offen offset:3072// 00000000489C: E05C1C00 80860C2C
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[76:77], v[252:253], v[100:103]// 0000000048A4: D3F30064 0D93F94C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[78:79], v[254:255], v[100:103]// 0000000048AC: D3F30064 0D93FD4E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[196:199], v2 offset:64                      // 0000000048B4: D9FE0040 C4000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v16, v3 offset:8576                            // 0000000048BC: D86C2180 10000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[88:89], v[232:233], 0// 0000000048C4: D3F30068 0A03D158  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[90:91], v[234:235], v[104:107]// 0000000048CC: D3F30068 0DA3D55A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[24:27], v45, s[24:27], 0 offen offset:2048// 0000000048D4: E05C1800 8086182D
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[92:93], v[236:237], v[104:107]// 0000000048DC: D3F30068 0DA3D95C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[94:95], v[238:239], v[104:107]// 0000000048E4: D3F30068 0DA3DD5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[200:203], v2 offset:128                     // 0000000048EC: D9FE0080 C8000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v17, v3 offset:8832                            // 0000000048F4: D86C2280 11000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[88:89], v[248:249], 0// 0000000048FC: D3F3006C 0A03F158  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[90:91], v[250:251], v[108:111]// 000000004904: D3F3006C 0DB3F55A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[28:31], v45, s[24:27], 0 offen offset:3072// 00000000490C: E05C1C00 80861C2D
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[92:93], v[252:253], v[108:111]// 000000004914: D3F3006C 0DB3F95C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[94:95], v[254:255], v[108:111]// 00000000491C: D3F3006C 0DB3FD5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[204:207], v2 offset:192                     // 000000004924: D9FE00C0 CC000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v18, v3 offset:9088                            // 00000000492C: D86C2380 12000003  // Read 32-bit from LDS (shared memory)
	s_waitcnt vmcnt(13)                                        // 000000004934: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[104:105], v[232:233], 0// 000000004938: D3F30070 0A03D168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[106:107], v[234:235], v[112:115]// 000000004940: D3F30070 0DC3D56A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[40:43], v46, s[24:27], 0 offen offset:2048// 000000004948: E05C1800 8086282E
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[108:109], v[236:237], v[112:115]// 000000004950: D3F30070 0DC3D96C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[110:111], v[238:239], v[112:115]// 000000004958: D3F30070 0DC3DD6E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[208:211], v2 offset:1024                    // 000000004960: D9FE0400 D0000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[104:105], v[248:249], 0// 000000004968: D3F30074 0A03F168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[106:107], v[250:251], v[116:119]// 000000004970: D3F30074 0DD3F56A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[44:47], v46, s[24:27], 0 offen offset:3072// 000000004978: E05C1C00 80862C2E
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[108:109], v[252:253], v[116:119]// 000000004980: D3F30074 0DD3F96C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[110:111], v[254:255], v[116:119]// 000000004988: D3F30074 0DD3FD6E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[212:215], v2 offset:1088                    // 000000004990: D9FE0440 D4000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[120:121], v[232:233], 0// 000000004998: D3F30078 0A03D178  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[122:123], v[234:235], v[120:123]// 0000000049A0: D3F30078 0DE3D57A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[56:59], v47, s[24:27], 0 offen offset:2048// 0000000049A8: E05C1800 8086382F
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[124:125], v[236:237], v[120:123]// 0000000049B0: D3F30078 0DE3D97C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[126:127], v[238:239], v[120:123]// 0000000049B8: D3F30078 0DE3DD7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[216:219], v2 offset:1152                    // 0000000049C0: D9FE0480 D8000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[120:121], v[248:249], 0// 0000000049C8: D3F3007C 0A03F178  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[122:123], v[250:251], v[124:127]// 0000000049D0: D3F3007C 0DF3F57A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[60:63], v47, s[24:27], 0 offen offset:3072// 0000000049D8: E05C1C00 80863C2F
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[124:125], v[252:253], v[124:127]// 0000000049E0: D3F3007C 0DF3F97C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[126:127], v[254:255], v[124:127]// 0000000049E8: D3F3007C 0DF3FD7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[220:223], v2 offset:1216                    // 0000000049F0: D9FE04C0 DC000002  // Read 128-bit from LDS (4x32-bit)
	v_mul_f32_dpp v56, v24, v21 row_newbcast:2 row_mask:0xf bank_mask:0xf// 0000000049F8: 0A702AFA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000004A00: 7E720338  // Vector move
	v_pk_fma_f32 v[64:65], v[96:97], v[56:57], v[64:65]        // 000000004A04: D3B04040 1D027160  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[66:67], v[98:99], v[56:57], v[66:67]        // 000000004A0C: D3B04042 1D0A7162  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[72:73], v[104:105], v[56:57], v[72:73]      // 000000004A14: D3B04048 1D227168  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[74:75], v[106:107], v[56:57], v[74:75]      // 000000004A1C: D3B0404A 1D2A716A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v21 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000004A24: 0A702AFA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000004A2C: 7E720338  // Vector move
	v_pk_fma_f32 v[80:81], v[112:113], v[56:57], v[80:81]      // 000000004A30: D3B04050 1D427170  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[82:83], v[114:115], v[56:57], v[82:83]      // 000000004A38: D3B04052 1D4A7172  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[88:89], v[120:121], v[56:57], v[88:89]      // 000000004A40: D3B04058 1D627178  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[90:91], v[122:123], v[56:57], v[90:91]      // 000000004A48: D3B0405A 1D6A717A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v22 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000004A50: 0A702CFA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000004A58: 7E720338  // Vector move
	v_pk_fma_f32 v[68:69], v[100:101], v[56:57], v[68:69]      // 000000004A5C: D3B04044 1D127164  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[70:71], v[102:103], v[56:57], v[70:71]      // 000000004A64: D3B04046 1D1A7166  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[76:77], v[108:109], v[56:57], v[76:77]      // 000000004A6C: D3B0404C 1D32716C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[78:79], v[110:111], v[56:57], v[78:79]      // 000000004A74: D3B0404E 1D3A716E  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v22 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000004A7C: 0A702CFA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000004A84: 7E720338  // Vector move
	v_pk_fma_f32 v[84:85], v[116:117], v[56:57], v[84:85]      // 000000004A88: D3B04054 1D527174  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[86:87], v[118:119], v[56:57], v[86:87]      // 000000004A90: D3B04056 1D5A7176  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[92:93], v[124:125], v[56:57], v[92:93]      // 000000004A98: D3B0405C 1D72717C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[94:95], v[126:127], v[56:57], v[94:95]      // 000000004AA0: D3B0405E 1D7A717E  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x200, s80                                  // 000000004AA8: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 000000004AB0: BF0A513C
	s_cselect_b32 s57, s57, 0                                  // 000000004AB4: 85398039
	s_cselect_b32 s3, s3, 0                                    // 000000004AB8: 85038003
	s_add_u32 s60, 0x200, s80                                  // 000000004ABC: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 000000004AC4: BF0A513C
	s_cselect_b32 s58, s58, 0                                  // 000000004AC8: 853A803A
	s_add_u32 s20, s57, s20                                    // 000000004ACC: 80141439
	s_addc_u32 s21, 0, s21                                     // 000000004AD0: 82151580
	s_add_u32 s28, s3, s28                                     // 000000004AD4: 801C1C03
	s_addc_u32 s29, 0, s29                                     // 000000004AD8: 821D1D80
	s_add_u32 s24, s58, s24                                    // 000000004ADC: 8018183A
	s_addc_u32 s25, 0, s25                                     // 000000004AE0: 82191980
	s_add_u32 s92, s90, s92                                    // 000000004AE4: 805C5C5A
	s_addc_u32 s93, 0, s93                                     // 000000004AE8: 825D5D80
	s_addk_i32 s80, 0x100                                      // 000000004AEC: B7500100
	s_cmp_lt_i32 s80, s81                                      // 000000004AF0: BF045150
	s_cbranch_scc0 label_073F                                  // 000000004AF4: BF840001  // Branch if SCC==0 (condition false)
	s_branch label_029E                                        // 000000004AF8: BF82FB5F  // Unconditional branch


// --- LABEL (potential loop target) ---
0000000000004afc <label_073F>:
	s_mov_b32 s20, 0                                           // 000000004AFC: BE940080  // Scalar move (constant/register)
	s_cmp_lt_u32 s89, s66                                      // 000000004B00: BF0A4259
	s_cselect_b32 s60, 0, 1                                    // 000000004B04: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000004B08: 97143C14
	s_cmp_lt_u32 s88, s66                                      // 000000004B0C: BF0A4258
	s_cselect_b32 s60, 0, 1                                    // 000000004B10: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000004B14: 97143C14
	s_cmp_lt_u32 s87, s66                                      // 000000004B18: BF0A4257
	s_cselect_b32 s60, 0, 1                                    // 000000004B1C: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000004B20: 97143C14
	s_cmp_lt_u32 s86, s66                                      // 000000004B24: BF0A4256
	s_cselect_b32 s60, 0, 1                                    // 000000004B28: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000004B2C: 97143C14
	s_cmp_lt_u32 s85, s66                                      // 000000004B30: BF0A4255
	s_cselect_b32 s60, 0, 1                                    // 000000004B34: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000004B38: 97143C14
	s_cmp_lt_u32 s84, s66                                      // 000000004B3C: BF0A4254
	s_cselect_b32 s60, 0, 1                                    // 000000004B40: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000004B44: 97143C14
	s_cmp_lt_u32 s83, s66                                      // 000000004B48: BF0A4253
	s_cselect_b32 s60, 0, 1                                    // 000000004B4C: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000004B50: 97143C14
	s_cmp_lt_u32 s82, s66                                      // 000000004B54: BF0A4252
	s_cselect_b32 s60, 0, 1                                    // 000000004B58: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000004B5C: 97143C14
	s_waitcnt vmcnt(12)                                        // 000000004B60: BF8C0F7C  // Wait for memory operations (CRITICAL for perf)
	buffer_load_dwordx4 a[0:3], v48, s[12:15], 0 offen         // 000000004B64: E05C1000 80830030
	v_mul_f32_e64 v56, -v128, s6                               // 000000004B6C: D1050038 20000D80
	v_mul_f32_e64 v57, -v129, s6                               // 000000004B74: D1050039 20000D81
	v_mul_f32_e64 v58, -v130, s6                               // 000000004B7C: D105003A 20000D82
	v_mul_f32_e64 v59, -v131, s6                               // 000000004B84: D105003B 20000D83
	v_exp_f32_e32 v56, v56                                     // 000000004B8C: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000004B90: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000004B94: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000004B98: 7E76413B
	buffer_load_dwordx4 a[4:7], v49, s[12:15], 0 offen         // 000000004B9C: E05C1000 80830431
	v_add_f32_e64 v56, v56, 1.0                                // 000000004BA4: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000004BAC: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000004BB4: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000004BBC: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000004BC4: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000004BC8: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000004BCC: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000004BD0: 7E76453B
	v_mul_f32_e32 v128, v128, v56                              // 000000004BD4: 0B007180  // FP32 multiply
	v_mul_f32_e32 v129, v129, v57                              // 000000004BD8: 0B027381  // FP32 multiply
	v_mul_f32_e32 v130, v130, v58                              // 000000004BDC: 0B047582  // FP32 multiply
	v_mul_f32_e32 v131, v131, v59                              // 000000004BE0: 0B067783  // FP32 multiply
	v_mul_f32_e32 v128, v128, v64                              // 000000004BE4: 0B008180  // FP32 multiply
	v_mul_f32_e32 v129, v129, v65                              // 000000004BE8: 0B028381  // FP32 multiply
	v_mul_f32_e32 v130, v130, v66                              // 000000004BEC: 0B048582  // FP32 multiply
	v_mul_f32_e32 v131, v131, v67                              // 000000004BF0: 0B068783  // FP32 multiply
	buffer_load_dwordx4 a[8:11], v50, s[12:15], 0 offen        // 000000004BF4: E05C1000 80830832
	v_mul_f32_e64 v56, -v132, s6                               // 000000004BFC: D1050038 20000D84
	v_mul_f32_e64 v57, -v133, s6                               // 000000004C04: D1050039 20000D85
	v_mul_f32_e64 v58, -v134, s6                               // 000000004C0C: D105003A 20000D86
	v_mul_f32_e64 v59, -v135, s6                               // 000000004C14: D105003B 20000D87
	v_exp_f32_e32 v56, v56                                     // 000000004C1C: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000004C20: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000004C24: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000004C28: 7E76413B
	buffer_load_dwordx4 a[12:15], v51, s[12:15], 0 offen       // 000000004C2C: E05C1000 80830C33
	s_add_u32 s12, s78, s12                                    // 000000004C34: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000004C38: 820D0D80
	v_add_f32_e64 v56, v56, 1.0                                // 000000004C3C: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000004C44: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000004C4C: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000004C54: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000004C5C: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000004C60: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000004C64: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000004C68: 7E76453B
	v_mul_f32_e32 v132, v132, v56                              // 000000004C6C: 0B087184  // FP32 multiply
	v_mul_f32_e32 v133, v133, v57                              // 000000004C70: 0B0A7385  // FP32 multiply
	v_mul_f32_e32 v134, v134, v58                              // 000000004C74: 0B0C7586  // FP32 multiply
	v_mul_f32_e32 v135, v135, v59                              // 000000004C78: 0B0E7787  // FP32 multiply
	v_mul_f32_e32 v132, v132, v68                              // 000000004C7C: 0B088984  // FP32 multiply
	v_mul_f32_e32 v133, v133, v69                              // 000000004C80: 0B0A8B85  // FP32 multiply
	v_mul_f32_e32 v134, v134, v70                              // 000000004C84: 0B0C8D86  // FP32 multiply
	v_mul_f32_e32 v135, v135, v71                              // 000000004C88: 0B0E8F87  // FP32 multiply
	s_waitcnt vmcnt(12)                                        // 000000004C8C: BF8C0F7C  // Wait for memory operations (CRITICAL for perf)
	buffer_load_dwordx4 a[16:19], v48, s[12:15], 0 offen       // 000000004C90: E05C1000 80831030
	v_mul_f32_e64 v56, -v136, s6                               // 000000004C98: D1050038 20000D88
	v_mul_f32_e64 v57, -v137, s6                               // 000000004CA0: D1050039 20000D89
	v_mul_f32_e64 v58, -v138, s6                               // 000000004CA8: D105003A 20000D8A
	v_mul_f32_e64 v59, -v139, s6                               // 000000004CB0: D105003B 20000D8B
	v_exp_f32_e32 v56, v56                                     // 000000004CB8: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000004CBC: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000004CC0: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000004CC4: 7E76413B
	buffer_load_dwordx4 a[20:23], v49, s[12:15], 0 offen       // 000000004CC8: E05C1000 80831431
	v_add_f32_e64 v56, v56, 1.0                                // 000000004CD0: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000004CD8: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000004CE0: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000004CE8: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000004CF0: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000004CF4: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000004CF8: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000004CFC: 7E76453B
	v_mul_f32_e32 v136, v136, v56                              // 000000004D00: 0B107188  // FP32 multiply
	v_mul_f32_e32 v137, v137, v57                              // 000000004D04: 0B127389  // FP32 multiply
	v_mul_f32_e32 v138, v138, v58                              // 000000004D08: 0B14758A  // FP32 multiply
	v_mul_f32_e32 v139, v139, v59                              // 000000004D0C: 0B16778B  // FP32 multiply
	v_mul_f32_e32 v136, v136, v72                              // 000000004D10: 0B109188  // FP32 multiply
	v_mul_f32_e32 v137, v137, v73                              // 000000004D14: 0B129389  // FP32 multiply
	v_mul_f32_e32 v138, v138, v74                              // 000000004D18: 0B14958A  // FP32 multiply
	v_mul_f32_e32 v139, v139, v75                              // 000000004D1C: 0B16978B  // FP32 multiply
	buffer_load_dwordx4 a[24:27], v50, s[12:15], 0 offen       // 000000004D20: E05C1000 80831832
	v_mul_f32_e64 v56, -v140, s6                               // 000000004D28: D1050038 20000D8C
	v_mul_f32_e64 v57, -v141, s6                               // 000000004D30: D1050039 20000D8D
	v_mul_f32_e64 v58, -v142, s6                               // 000000004D38: D105003A 20000D8E
	v_mul_f32_e64 v59, -v143, s6                               // 000000004D40: D105003B 20000D8F
	v_exp_f32_e32 v56, v56                                     // 000000004D48: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000004D4C: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000004D50: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000004D54: 7E76413B
	buffer_load_dwordx4 a[28:31], v51, s[12:15], 0 offen       // 000000004D58: E05C1000 80831C33
	s_add_u32 s12, s78, s12                                    // 000000004D60: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000004D64: 820D0D80
	v_add_f32_e64 v56, v56, 1.0                                // 000000004D68: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000004D70: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000004D78: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000004D80: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000004D88: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000004D8C: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000004D90: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000004D94: 7E76453B
	v_mul_f32_e32 v140, v140, v56                              // 000000004D98: 0B18718C  // FP32 multiply
	v_mul_f32_e32 v141, v141, v57                              // 000000004D9C: 0B1A738D  // FP32 multiply
	v_mul_f32_e32 v142, v142, v58                              // 000000004DA0: 0B1C758E  // FP32 multiply
	v_mul_f32_e32 v143, v143, v59                              // 000000004DA4: 0B1E778F  // FP32 multiply
	v_mul_f32_e32 v140, v140, v76                              // 000000004DA8: 0B18998C  // FP32 multiply
	v_mul_f32_e32 v141, v141, v77                              // 000000004DAC: 0B1A9B8D  // FP32 multiply
	v_mul_f32_e32 v142, v142, v78                              // 000000004DB0: 0B1C9D8E  // FP32 multiply
	v_mul_f32_e32 v143, v143, v79                              // 000000004DB4: 0B1E9F8F  // FP32 multiply
	s_waitcnt vmcnt(12)                                        // 000000004DB8: BF8C0F7C  // Wait for memory operations (CRITICAL for perf)
	buffer_load_dwordx4 a[32:35], v48, s[12:15], 0 offen       // 000000004DBC: E05C1000 80832030
	v_mul_f32_e64 v56, -v144, s6                               // 000000004DC4: D1050038 20000D90
	v_mul_f32_e64 v57, -v145, s6                               // 000000004DCC: D1050039 20000D91
	v_mul_f32_e64 v58, -v146, s6                               // 000000004DD4: D105003A 20000D92
	v_mul_f32_e64 v59, -v147, s6                               // 000000004DDC: D105003B 20000D93
	v_exp_f32_e32 v56, v56                                     // 000000004DE4: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000004DE8: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000004DEC: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000004DF0: 7E76413B
	buffer_load_dwordx4 a[36:39], v49, s[12:15], 0 offen       // 000000004DF4: E05C1000 80832431
	v_add_f32_e64 v56, v56, 1.0                                // 000000004DFC: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000004E04: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000004E0C: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000004E14: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000004E1C: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000004E20: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000004E24: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000004E28: 7E76453B
	v_mul_f32_e32 v144, v144, v56                              // 000000004E2C: 0B207190  // FP32 multiply
	v_mul_f32_e32 v145, v145, v57                              // 000000004E30: 0B227391  // FP32 multiply
	v_mul_f32_e32 v146, v146, v58                              // 000000004E34: 0B247592  // FP32 multiply
	v_mul_f32_e32 v147, v147, v59                              // 000000004E38: 0B267793  // FP32 multiply
	v_mul_f32_e32 v144, v144, v80                              // 000000004E3C: 0B20A190  // FP32 multiply
	v_mul_f32_e32 v145, v145, v81                              // 000000004E40: 0B22A391  // FP32 multiply
	v_mul_f32_e32 v146, v146, v82                              // 000000004E44: 0B24A592  // FP32 multiply
	v_mul_f32_e32 v147, v147, v83                              // 000000004E48: 0B26A793  // FP32 multiply
	buffer_load_dwordx4 a[40:43], v50, s[12:15], 0 offen       // 000000004E4C: E05C1000 80832832
	v_mul_f32_e64 v56, -v148, s6                               // 000000004E54: D1050038 20000D94
	v_mul_f32_e64 v57, -v149, s6                               // 000000004E5C: D1050039 20000D95
	v_mul_f32_e64 v58, -v150, s6                               // 000000004E64: D105003A 20000D96
	v_mul_f32_e64 v59, -v151, s6                               // 000000004E6C: D105003B 20000D97
	v_exp_f32_e32 v56, v56                                     // 000000004E74: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000004E78: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000004E7C: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000004E80: 7E76413B
	buffer_load_dwordx4 a[44:47], v51, s[12:15], 0 offen       // 000000004E84: E05C1000 80832C33
	s_add_u32 s12, s78, s12                                    // 000000004E8C: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000004E90: 820D0D80
	v_add_f32_e64 v56, v56, 1.0                                // 000000004E94: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000004E9C: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000004EA4: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000004EAC: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000004EB4: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000004EB8: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000004EBC: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000004EC0: 7E76453B
	v_mul_f32_e32 v148, v148, v56                              // 000000004EC4: 0B287194  // FP32 multiply
	v_mul_f32_e32 v149, v149, v57                              // 000000004EC8: 0B2A7395  // FP32 multiply
	v_mul_f32_e32 v150, v150, v58                              // 000000004ECC: 0B2C7596  // FP32 multiply
	v_mul_f32_e32 v151, v151, v59                              // 000000004ED0: 0B2E7797  // FP32 multiply
	v_mul_f32_e32 v148, v148, v84                              // 000000004ED4: 0B28A994  // FP32 multiply
	v_mul_f32_e32 v149, v149, v85                              // 000000004ED8: 0B2AAB95  // FP32 multiply
	v_mul_f32_e32 v150, v150, v86                              // 000000004EDC: 0B2CAD96  // FP32 multiply
	v_mul_f32_e32 v151, v151, v87                              // 000000004EE0: 0B2EAF97  // FP32 multiply
	s_waitcnt vmcnt(12)                                        // 000000004EE4: BF8C0F7C  // Wait for memory operations (CRITICAL for perf)
	buffer_load_dwordx4 a[48:51], v48, s[12:15], 0 offen       // 000000004EE8: E05C1000 80833030
	v_mul_f32_e64 v56, -v152, s6                               // 000000004EF0: D1050038 20000D98
	v_mul_f32_e64 v57, -v153, s6                               // 000000004EF8: D1050039 20000D99
	v_mul_f32_e64 v58, -v154, s6                               // 000000004F00: D105003A 20000D9A
	v_mul_f32_e64 v59, -v155, s6                               // 000000004F08: D105003B 20000D9B
	v_exp_f32_e32 v56, v56                                     // 000000004F10: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000004F14: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000004F18: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000004F1C: 7E76413B
	buffer_load_dwordx4 a[52:55], v49, s[12:15], 0 offen       // 000000004F20: E05C1000 80833431
	v_add_f32_e64 v56, v56, 1.0                                // 000000004F28: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000004F30: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000004F38: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000004F40: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000004F48: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000004F4C: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000004F50: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000004F54: 7E76453B
	v_mul_f32_e32 v152, v152, v56                              // 000000004F58: 0B307198  // FP32 multiply
	v_mul_f32_e32 v153, v153, v57                              // 000000004F5C: 0B327399  // FP32 multiply
	v_mul_f32_e32 v154, v154, v58                              // 000000004F60: 0B34759A  // FP32 multiply
	v_mul_f32_e32 v155, v155, v59                              // 000000004F64: 0B36779B  // FP32 multiply
	v_mul_f32_e32 v152, v152, v88                              // 000000004F68: 0B30B198  // FP32 multiply
	v_mul_f32_e32 v153, v153, v89                              // 000000004F6C: 0B32B399  // FP32 multiply
	v_mul_f32_e32 v154, v154, v90                              // 000000004F70: 0B34B59A  // FP32 multiply
	v_mul_f32_e32 v155, v155, v91                              // 000000004F74: 0B36B79B  // FP32 multiply
	buffer_load_dwordx4 a[56:59], v50, s[12:15], 0 offen       // 000000004F78: E05C1000 80833832
	v_mul_f32_e64 v56, -v156, s6                               // 000000004F80: D1050038 20000D9C
	v_mul_f32_e64 v57, -v157, s6                               // 000000004F88: D1050039 20000D9D
	v_mul_f32_e64 v58, -v158, s6                               // 000000004F90: D105003A 20000D9E
	v_mul_f32_e64 v59, -v159, s6                               // 000000004F98: D105003B 20000D9F
	v_exp_f32_e32 v56, v56                                     // 000000004FA0: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000004FA4: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000004FA8: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000004FAC: 7E76413B
	buffer_load_dwordx4 a[60:63], v51, s[12:15], 0 offen       // 000000004FB0: E05C1000 80833C33
	v_add_f32_e64 v56, v56, 1.0                                // 000000004FB8: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000004FC0: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000004FC8: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000004FD0: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000004FD8: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000004FDC: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000004FE0: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000004FE4: 7E76453B
	v_mul_f32_e32 v156, v156, v56                              // 000000004FE8: 0B38719C  // FP32 multiply
	v_mul_f32_e32 v157, v157, v57                              // 000000004FEC: 0B3A739D  // FP32 multiply
	v_mul_f32_e32 v158, v158, v58                              // 000000004FF0: 0B3C759E  // FP32 multiply
	v_mul_f32_e32 v159, v159, v59                              // 000000004FF4: 0B3E779F  // FP32 multiply
	v_mul_f32_e32 v156, v156, v92                              // 000000004FF8: 0B38B99C  // FP32 multiply
	v_mul_f32_e32 v157, v157, v93                              // 000000004FFC: 0B3ABB9D  // FP32 multiply
	v_mul_f32_e32 v158, v158, v94                              // 000000005000: 0B3CBD9E  // FP32 multiply
	v_mul_f32_e32 v159, v159, v95                              // 000000005004: 0B3EBF9F  // FP32 multiply
	v_lshlrev_b32_e32 v56, 2, v0                               // 000000005008: 24700082  // Left shift (multiply by power of 2)
	s_mul_i32 s60, s82, s71                                    // 00000000500C: 923C4752
	v_add_u32_e64 v80, v56, s60                                // 000000005010: D1340050 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v81, 0                                       // 000000005018: 7EA20280  // Vector move
	s_mul_i32 s60, s83, s71                                    // 00000000501C: 923C4753
	v_add_u32_e64 v82, v56, s60                                // 000000005020: D1340052 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v83, 0                                       // 000000005028: 7EA60280  // Vector move
	s_mul_i32 s60, s84, s71                                    // 00000000502C: 923C4754
	v_add_u32_e64 v84, v56, s60                                // 000000005030: D1340054 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v85, 0                                       // 000000005038: 7EAA0280  // Vector move
	s_mul_i32 s60, s85, s71                                    // 00000000503C: 923C4755
	v_add_u32_e64 v86, v56, s60                                // 000000005040: D1340056 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v87, 0                                       // 000000005048: 7EAE0280  // Vector move
	s_mul_i32 s60, s86, s71                                    // 00000000504C: 923C4756
	v_add_u32_e64 v88, v56, s60                                // 000000005050: D1340058 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v89, 0                                       // 000000005058: 7EB20280  // Vector move
	s_mul_i32 s60, s87, s71                                    // 00000000505C: 923C4757
	v_add_u32_e64 v90, v56, s60                                // 000000005060: D134005A 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v91, 0                                       // 000000005068: 7EB60280  // Vector move
	s_mul_i32 s60, s88, s71                                    // 00000000506C: 923C4758
	v_add_u32_e64 v92, v56, s60                                // 000000005070: D134005C 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v93, 0                                       // 000000005078: 7EBA0280  // Vector move
	s_mul_i32 s60, s89, s71                                    // 00000000507C: 923C4759
	v_add_u32_e64 v94, v56, s60                                // 000000005080: D134005E 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v95, 0                                       // 000000005088: 7EBE0280  // Vector move
	buffer_load_dword v23, v6, s[16:19], 0 offen               // 00000000508C: E0501000 80041706
	v_mov_b32_e32 v28, 0x358637bd                              // 000000005094: 7E3802FF 358637BD  // Vector move
	v_mov_b32_e32 v29, 0x358637bd                              // 00000000509C: 7E3A02FF 358637BD  // Vector move
	v_max3_f32 v28, |v128|, |v129|, v28                        // 0000000050A4: D1D3031C 04730380
	v_max3_f32 v28, |v130|, |v131|, v28                        // 0000000050AC: D1D3031C 04730782
	v_max3_f32 v29, |v132|, |v133|, v29                        // 0000000050B4: D1D3031D 04770B84
	v_max3_f32 v29, |v134|, |v135|, v29                        // 0000000050BC: D1D3031D 04770F86
	v_max3_f32 v28, |v136|, |v137|, v28                        // 0000000050C4: D1D3031C 04731388
	v_max3_f32 v28, |v138|, |v139|, v28                        // 0000000050CC: D1D3031C 0473178A
	v_max3_f32 v29, |v140|, |v141|, v29                        // 0000000050D4: D1D3031D 04771B8C
	v_max3_f32 v29, |v142|, |v143|, v29                        // 0000000050DC: D1D3031D 04771F8E
	v_lshlrev_b32_e32 v56, 3, v0                               // 0000000050E4: 24700083  // Left shift (multiply by power of 2)
	s_mul_i32 s60, 0x200, s7                                   // 0000000050E8: 923C07FF 00000200
	v_add_u32_e32 v56, s60, v56                                // 0000000050F0: 6870703C  // 32-bit unsigned add (address calc)

// === LDS WRITE (staging data) ===
	ds_write_b64 v56, v[28:29] offset:18688                    // 0000000050F4: D89A4900 00001C38  // Write 64-bit to LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 0000000050FC: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 000000005100: BF8A0000  // Workgroup barrier synchronization
	v_and_b32_e32 v56, 15, v0                                  // 000000005104: 2670008F  // Bitwise AND (masking)
	v_lshlrev_b32_e32 v56, 3, v56                              // 000000005108: 24707083  // Left shift (multiply by power of 2)
	ds_read_b64 v[96:97], v56 offset:18688                     // 00000000510C: D8EC4900 60000038  // Read 64-bit from LDS
	ds_read_b64 v[98:99], v56 offset:18816                     // 000000005114: D8EC4980 62000038  // Read 64-bit from LDS
	ds_read_b64 v[100:101], v56 offset:18944                   // 00000000511C: D8EC4A00 64000038  // Read 64-bit from LDS
	ds_read_b64 v[102:103], v56 offset:19072                   // 000000005124: D8EC4A80 66000038  // Read 64-bit from LDS
	ds_read_b64 v[104:105], v56 offset:19200                   // 00000000512C: D8EC4B00 68000038  // Read 64-bit from LDS
	ds_read_b64 v[106:107], v56 offset:19328                   // 000000005134: D8EC4B80 6A000038  // Read 64-bit from LDS
	ds_read_b64 v[108:109], v56 offset:19456                   // 00000000513C: D8EC4C00 6C000038  // Read 64-bit from LDS
	ds_read_b64 v[110:111], v56 offset:19584                   // 000000005144: D8EC4C80 6E000038  // Read 64-bit from LDS
	ds_read_b64 v[112:113], v56 offset:19712                   // 00000000514C: D8EC4D00 70000038  // Read 64-bit from LDS
	ds_read_b64 v[114:115], v56 offset:19840                   // 000000005154: D8EC4D80 72000038  // Read 64-bit from LDS
	ds_read_b64 v[116:117], v56 offset:19968                   // 00000000515C: D8EC4E00 74000038  // Read 64-bit from LDS
	ds_read_b64 v[118:119], v56 offset:20096                   // 000000005164: D8EC4E80 76000038  // Read 64-bit from LDS
	ds_read_b64 v[120:121], v56 offset:20224                   // 00000000516C: D8EC4F00 78000038  // Read 64-bit from LDS
	ds_read_b64 v[122:123], v56 offset:20352                   // 000000005174: D8EC4F80 7A000038  // Read 64-bit from LDS
	ds_read_b64 v[124:125], v56 offset:20480                   // 00000000517C: D8EC5000 7C000038  // Read 64-bit from LDS
	ds_read_b64 v[126:127], v56 offset:20608                   // 000000005184: D8EC5080 7E000038  // Read 64-bit from LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 00000000518C: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	v_max3_f32 v28, |v96|, |v98|, v28                          // 000000005190: D1D3031C 0472C560
	v_max3_f32 v29, |v97|, |v99|, v29                          // 000000005198: D1D3031D 0476C761
	v_max3_f32 v28, |v100|, |v102|, v28                        // 0000000051A0: D1D3031C 0472CD64
	v_max3_f32 v29, |v101|, |v103|, v29                        // 0000000051A8: D1D3031D 0476CF65
	v_max3_f32 v28, |v104|, |v106|, v28                        // 0000000051B0: D1D3031C 0472D568
	v_max3_f32 v29, |v105|, |v107|, v29                        // 0000000051B8: D1D3031D 0476D769
	v_max3_f32 v28, |v108|, |v110|, v28                        // 0000000051C0: D1D3031C 0472DD6C
	v_max3_f32 v29, |v109|, |v111|, v29                        // 0000000051C8: D1D3031D 0476DF6D
	v_max3_f32 v28, |v112|, |v114|, v28                        // 0000000051D0: D1D3031C 0472E570
	v_max3_f32 v29, |v113|, |v115|, v29                        // 0000000051D8: D1D3031D 0476E771
	v_max3_f32 v28, |v116|, |v118|, v28                        // 0000000051E0: D1D3031C 0472ED74
	v_max3_f32 v29, |v117|, |v119|, v29                        // 0000000051E8: D1D3031D 0476EF75
	v_max3_f32 v28, |v120|, |v122|, v28                        // 0000000051F0: D1D3031C 0472F578
	v_max3_f32 v29, |v121|, |v123|, v29                        // 0000000051F8: D1D3031D 0476F779
	v_max3_f32 v28, |v124|, |v126|, v28                        // 000000005200: D1D3031C 0472FD7C
	v_max3_f32 v29, |v125|, |v127|, v29                        // 000000005208: D1D3031D 0476FF7D
	v_rcp_f32_e32 v28, v28                                     // 000000005210: 7E38451C
	v_rcp_f32_e32 v29, v29                                     // 000000005214: 7E3A451D
	v_mov_b32_e32 v56, 0x43700000                              // 000000005218: 7E7002FF 43700000  // Vector move
	v_mul_f32_e32 v28, v56, v28                                // 000000005220: 0A383938  // FP32 multiply
	v_mul_f32_e32 v29, v56, v29                                // 000000005224: 0A3A3B38  // FP32 multiply
	v_mul_f32_e32 v128, v28, v128                              // 000000005228: 0B01011C  // FP32 multiply
	v_mul_f32_e32 v129, v28, v129                              // 00000000522C: 0B03031C  // FP32 multiply
	v_mul_f32_e32 v130, v28, v130                              // 000000005230: 0B05051C  // FP32 multiply
	v_mul_f32_e32 v131, v28, v131                              // 000000005234: 0B07071C  // FP32 multiply
	v_cvt_pk_fp8_f32 v128, v128, v129                          // 000000005238: D2A20080 00030380  // Packed conversion
	v_cvt_pk_fp8_f32 v128, v130, v131 op_sel:[0,0,1]           // 000000005240: D2A24080 00030782  // Packed conversion
	v_mul_f32_e32 v132, v29, v132                              // 000000005248: 0B09091D  // FP32 multiply
	v_mul_f32_e32 v133, v29, v133                              // 00000000524C: 0B0B0B1D  // FP32 multiply
	v_mul_f32_e32 v134, v29, v134                              // 000000005250: 0B0D0D1D  // FP32 multiply
	v_mul_f32_e32 v135, v29, v135                              // 000000005254: 0B0F0F1D  // FP32 multiply
	v_cvt_pk_fp8_f32 v129, v132, v133                          // 000000005258: D2A20081 00030B84  // Packed conversion
	v_cvt_pk_fp8_f32 v129, v134, v135 op_sel:[0,0,1]           // 000000005260: D2A24081 00030F86  // Packed conversion
	v_mul_f32_e32 v136, v28, v136                              // 000000005268: 0B11111C  // FP32 multiply
	v_mul_f32_e32 v137, v28, v137                              // 00000000526C: 0B13131C  // FP32 multiply
	v_mul_f32_e32 v138, v28, v138                              // 000000005270: 0B15151C  // FP32 multiply
	v_mul_f32_e32 v139, v28, v139                              // 000000005274: 0B17171C  // FP32 multiply
	v_cvt_pk_fp8_f32 v130, v136, v137                          // 000000005278: D2A20082 00031388  // Packed conversion
	v_cvt_pk_fp8_f32 v130, v138, v139 op_sel:[0,0,1]           // 000000005280: D2A24082 0003178A  // Packed conversion
	v_mul_f32_e32 v140, v29, v140                              // 000000005288: 0B19191D  // FP32 multiply
	v_mul_f32_e32 v141, v29, v141                              // 00000000528C: 0B1B1B1D  // FP32 multiply
	v_mul_f32_e32 v142, v29, v142                              // 000000005290: 0B1D1D1D  // FP32 multiply
	v_mul_f32_e32 v143, v29, v143                              // 000000005294: 0B1F1F1D  // FP32 multiply
	v_cvt_pk_fp8_f32 v131, v140, v141                          // 000000005298: D2A20083 00031B8C  // Packed conversion
	v_cvt_pk_fp8_f32 v131, v142, v143 op_sel:[0,0,1]           // 0000000052A0: D2A24083 00031F8E  // Packed conversion
	v_rcp_f32_e32 v32, v28                                     // 0000000052A8: 7E40451C
	v_rcp_f32_e32 v33, v29                                     // 0000000052AC: 7E42451D
	v_mov_b32_e32 v30, 0x358637bd                              // 0000000052B0: 7E3C02FF 358637BD  // Vector move
	v_mov_b32_e32 v31, 0x358637bd                              // 0000000052B8: 7E3E02FF 358637BD  // Vector move
	v_max3_f32 v30, |v144|, |v145|, v30                        // 0000000052C0: D1D3031E 047B2390
	v_max3_f32 v30, |v146|, |v147|, v30                        // 0000000052C8: D1D3031E 047B2792
	v_max3_f32 v31, |v148|, |v149|, v31                        // 0000000052D0: D1D3031F 047F2B94
	v_max3_f32 v31, |v150|, |v151|, v31                        // 0000000052D8: D1D3031F 047F2F96
	v_max3_f32 v30, |v152|, |v153|, v30                        // 0000000052E0: D1D3031E 047B3398
	v_max3_f32 v30, |v154|, |v155|, v30                        // 0000000052E8: D1D3031E 047B379A
	v_max3_f32 v31, |v156|, |v157|, v31                        // 0000000052F0: D1D3031F 047F3B9C
	v_max3_f32 v31, |v158|, |v159|, v31                        // 0000000052F8: D1D3031F 047F3F9E
	v_lshlrev_b32_e32 v56, 3, v0                               // 000000005300: 24700083  // Left shift (multiply by power of 2)
	s_mul_i32 s60, 0x200, s7                                   // 000000005304: 923C07FF 00000200
	v_add_u32_e32 v56, s60, v56                                // 00000000530C: 6870703C  // 32-bit unsigned add (address calc)
	ds_write_b64 v56, v[30:31] offset:18688                    // 000000005310: D89A4900 00001E38  // Write 64-bit to LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 000000005318: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 00000000531C: BF8A0000  // Workgroup barrier synchronization
	v_and_b32_e32 v56, 15, v0                                  // 000000005320: 2670008F  // Bitwise AND (masking)
	v_lshlrev_b32_e32 v56, 3, v56                              // 000000005324: 24707083  // Left shift (multiply by power of 2)
	ds_read_b64 v[96:97], v56 offset:18688                     // 000000005328: D8EC4900 60000038  // Read 64-bit from LDS
	ds_read_b64 v[98:99], v56 offset:18816                     // 000000005330: D8EC4980 62000038  // Read 64-bit from LDS
	ds_read_b64 v[100:101], v56 offset:18944                   // 000000005338: D8EC4A00 64000038  // Read 64-bit from LDS
	ds_read_b64 v[102:103], v56 offset:19072                   // 000000005340: D8EC4A80 66000038  // Read 64-bit from LDS
	ds_read_b64 v[104:105], v56 offset:19200                   // 000000005348: D8EC4B00 68000038  // Read 64-bit from LDS
	ds_read_b64 v[106:107], v56 offset:19328                   // 000000005350: D8EC4B80 6A000038  // Read 64-bit from LDS
	ds_read_b64 v[108:109], v56 offset:19456                   // 000000005358: D8EC4C00 6C000038  // Read 64-bit from LDS
	ds_read_b64 v[110:111], v56 offset:19584                   // 000000005360: D8EC4C80 6E000038  // Read 64-bit from LDS
	ds_read_b64 v[112:113], v56 offset:19712                   // 000000005368: D8EC4D00 70000038  // Read 64-bit from LDS
	ds_read_b64 v[114:115], v56 offset:19840                   // 000000005370: D8EC4D80 72000038  // Read 64-bit from LDS
	ds_read_b64 v[116:117], v56 offset:19968                   // 000000005378: D8EC4E00 74000038  // Read 64-bit from LDS
	ds_read_b64 v[118:119], v56 offset:20096                   // 000000005380: D8EC4E80 76000038  // Read 64-bit from LDS
	ds_read_b64 v[120:121], v56 offset:20224                   // 000000005388: D8EC4F00 78000038  // Read 64-bit from LDS
	ds_read_b64 v[122:123], v56 offset:20352                   // 000000005390: D8EC4F80 7A000038  // Read 64-bit from LDS
	ds_read_b64 v[124:125], v56 offset:20480                   // 000000005398: D8EC5000 7C000038  // Read 64-bit from LDS
	ds_read_b64 v[126:127], v56 offset:20608                   // 0000000053A0: D8EC5080 7E000038  // Read 64-bit from LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 0000000053A8: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	v_max3_f32 v30, |v96|, |v98|, v30                          // 0000000053AC: D1D3031E 047AC560
	v_max3_f32 v31, |v97|, |v99|, v31                          // 0000000053B4: D1D3031F 047EC761
	v_max3_f32 v30, |v100|, |v102|, v30                        // 0000000053BC: D1D3031E 047ACD64
	v_max3_f32 v31, |v101|, |v103|, v31                        // 0000000053C4: D1D3031F 047ECF65
	v_max3_f32 v30, |v104|, |v106|, v30                        // 0000000053CC: D1D3031E 047AD568
	v_max3_f32 v31, |v105|, |v107|, v31                        // 0000000053D4: D1D3031F 047ED769
	v_max3_f32 v30, |v108|, |v110|, v30                        // 0000000053DC: D1D3031E 047ADD6C
	v_max3_f32 v31, |v109|, |v111|, v31                        // 0000000053E4: D1D3031F 047EDF6D
	v_max3_f32 v30, |v112|, |v114|, v30                        // 0000000053EC: D1D3031E 047AE570
	v_max3_f32 v31, |v113|, |v115|, v31                        // 0000000053F4: D1D3031F 047EE771
	v_max3_f32 v30, |v116|, |v118|, v30                        // 0000000053FC: D1D3031E 047AED74
	v_max3_f32 v31, |v117|, |v119|, v31                        // 000000005404: D1D3031F 047EEF75
	v_max3_f32 v30, |v120|, |v122|, v30                        // 00000000540C: D1D3031E 047AF578
	v_max3_f32 v31, |v121|, |v123|, v31                        // 000000005414: D1D3031F 047EF779
	v_max3_f32 v30, |v124|, |v126|, v30                        // 00000000541C: D1D3031E 047AFD7C
	v_max3_f32 v31, |v125|, |v127|, v31                        // 000000005424: D1D3031F 047EFF7D
	v_rcp_f32_e32 v30, v30                                     // 00000000542C: 7E3C451E
	v_rcp_f32_e32 v31, v31                                     // 000000005430: 7E3E451F
	v_mov_b32_e32 v56, 0x43700000                              // 000000005434: 7E7002FF 43700000  // Vector move
	v_mul_f32_e32 v30, v56, v30                                // 00000000543C: 0A3C3D38  // FP32 multiply
	v_mul_f32_e32 v31, v56, v31                                // 000000005440: 0A3E3F38  // FP32 multiply
	v_mul_f32_e32 v144, v30, v144                              // 000000005444: 0B21211E  // FP32 multiply
	v_mul_f32_e32 v145, v30, v145                              // 000000005448: 0B23231E  // FP32 multiply
	v_mul_f32_e32 v146, v30, v146                              // 00000000544C: 0B25251E  // FP32 multiply
	v_mul_f32_e32 v147, v30, v147                              // 000000005450: 0B27271E  // FP32 multiply
	v_cvt_pk_fp8_f32 v132, v144, v145                          // 000000005454: D2A20084 00032390  // Packed conversion
	v_cvt_pk_fp8_f32 v132, v146, v147 op_sel:[0,0,1]           // 00000000545C: D2A24084 00032792  // Packed conversion
	v_mul_f32_e32 v148, v31, v148                              // 000000005464: 0B29291F  // FP32 multiply
	v_mul_f32_e32 v149, v31, v149                              // 000000005468: 0B2B2B1F  // FP32 multiply
	v_mul_f32_e32 v150, v31, v150                              // 00000000546C: 0B2D2D1F  // FP32 multiply
	v_mul_f32_e32 v151, v31, v151                              // 000000005470: 0B2F2F1F  // FP32 multiply
	v_cvt_pk_fp8_f32 v133, v148, v149                          // 000000005474: D2A20085 00032B94  // Packed conversion
	v_cvt_pk_fp8_f32 v133, v150, v151 op_sel:[0,0,1]           // 00000000547C: D2A24085 00032F96  // Packed conversion
	v_mul_f32_e32 v152, v30, v152                              // 000000005484: 0B31311E  // FP32 multiply
	v_mul_f32_e32 v153, v30, v153                              // 000000005488: 0B33331E  // FP32 multiply
	v_mul_f32_e32 v154, v30, v154                              // 00000000548C: 0B35351E  // FP32 multiply
	v_mul_f32_e32 v155, v30, v155                              // 000000005490: 0B37371E  // FP32 multiply
	v_cvt_pk_fp8_f32 v134, v152, v153                          // 000000005494: D2A20086 00033398  // Packed conversion
	v_cvt_pk_fp8_f32 v134, v154, v155 op_sel:[0,0,1]           // 00000000549C: D2A24086 0003379A  // Packed conversion
	v_mul_f32_e32 v156, v31, v156                              // 0000000054A4: 0B39391F  // FP32 multiply
	v_mul_f32_e32 v157, v31, v157                              // 0000000054A8: 0B3B3B1F  // FP32 multiply
	v_mul_f32_e32 v158, v31, v158                              // 0000000054AC: 0B3D3D1F  // FP32 multiply
	v_mul_f32_e32 v159, v31, v159                              // 0000000054B0: 0B3F3F1F  // FP32 multiply
	v_cvt_pk_fp8_f32 v135, v156, v157                          // 0000000054B4: D2A20087 00033B9C  // Packed conversion
	v_cvt_pk_fp8_f32 v135, v158, v159 op_sel:[0,0,1]           // 0000000054BC: D2A24087 00033F9E  // Packed conversion
	v_rcp_f32_e32 v34, v30                                     // 0000000054C4: 7E44451E
	v_rcp_f32_e32 v35, v31                                     // 0000000054C8: 7E46451F
	v_lshrrev_b32_e32 v56, 5, v0                               // 0000000054CC: 20700085  // Right shift (divide by power of 2)
	v_lshlrev_b32_e32 v57, 5, v56                              // 0000000054D0: 24727085  // Left shift (multiply by power of 2)
	v_and_b32_e32 v56, 31, v0                                  // 0000000054D4: 2670009F  // Bitwise AND (masking)
	v_lshrrev_b32_e32 v58, 4, v56                              // 0000000054D8: 20747084  // Right shift (divide by power of 2)
	v_add_u32_e32 v57, v58, v57                                // 0000000054DC: 6872733A  // 32-bit unsigned add (address calc)
	v_and_b32_e32 v56, 15, v0                                  // 0000000054E0: 2670008F  // Bitwise AND (masking)
	v_lshlrev_b32_e32 v56, 1, v56                              // 0000000054E4: 24707081  // Left shift (multiply by power of 2)
	v_add_u32_e32 v57, v56, v57                                // 0000000054E8: 68727338  // 32-bit unsigned add (address calc)
	v_lshlrev_b32_e32 v56, 2, v57                              // 0000000054EC: 24707282  // Left shift (multiply by power of 2)
	s_mul_i32 s60, 0x100, s7                                   // 0000000054F0: 923C07FF 00000100
	v_add_u32_e64 v56, v56, s60                                // 0000000054F8: D1340038 00007938  // 32-bit unsigned add (address calc)
	ds_write_b32 v56, v128 offset:20736                        // 000000005500: D81A5100 00008038  // Write 32-bit to LDS
	ds_write_b32 v56, v129 offset:24832                        // 000000005508: D81A6100 00008138  // Write 32-bit to LDS
	ds_write_b32 v56, v130 offset:21760                        // 000000005510: D81A5500 00008238  // Write 32-bit to LDS
	ds_write_b32 v56, v131 offset:25856                        // 000000005518: D81A6500 00008338  // Write 32-bit to LDS
	ds_write_b32 v56, v132 offset:22784                        // 000000005520: D81A5900 00008438  // Write 32-bit to LDS
	ds_write_b32 v56, v133 offset:26880                        // 000000005528: D81A6900 00008538  // Write 32-bit to LDS
	ds_write_b32 v56, v134 offset:23808                        // 000000005530: D81A5D00 00008638  // Write 32-bit to LDS
	ds_write_b32 v56, v135 offset:27904                        // 000000005538: D81A6D00 00008738  // Write 32-bit to LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 000000005540: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 000000005544: BF8A0000  // Workgroup barrier synchronization
	v_lshrrev_b32_e32 v56, 4, v0                               // 000000005548: 20700084  // Right shift (divide by power of 2)
	v_lshlrev_b32_e32 v57, 6, v56                              // 00000000554C: 24727086  // Left shift (multiply by power of 2)
	v_and_b32_e32 v56, 15, v0                                  // 000000005550: 2670008F  // Bitwise AND (masking)
	v_lshlrev_b32_e32 v56, 1, v56                              // 000000005554: 24707081  // Left shift (multiply by power of 2)
	v_add_u32_e32 v57, v56, v57                                // 000000005558: 68727338  // 32-bit unsigned add (address calc)
	v_lshlrev_b32_e32 v56, 2, v57                              // 00000000555C: 24707282  // Left shift (multiply by power of 2)
	ds_read_b64 v[128:129], v56 offset:20736                   // 000000005560: D8EC5100 80000038  // Read 64-bit from LDS
	ds_read_b64 v[130:131], v56 offset:20864                   // 000000005568: D8EC5180 82000038  // Read 64-bit from LDS
	ds_read_b64 v[132:133], v56 offset:21760                   // 000000005570: D8EC5500 84000038  // Read 64-bit from LDS
	ds_read_b64 v[134:135], v56 offset:21888                   // 000000005578: D8EC5580 86000038  // Read 64-bit from LDS
	ds_read_b64 v[136:137], v56 offset:22784                   // 000000005580: D8EC5900 88000038  // Read 64-bit from LDS
	ds_read_b64 v[138:139], v56 offset:22912                   // 000000005588: D8EC5980 8A000038  // Read 64-bit from LDS
	ds_read_b64 v[140:141], v56 offset:23808                   // 000000005590: D8EC5D00 8C000038  // Read 64-bit from LDS
	ds_read_b64 v[142:143], v56 offset:23936                   // 000000005598: D8EC5D80 8E000038  // Read 64-bit from LDS
	ds_read_b64 v[144:145], v56 offset:24832                   // 0000000055A0: D8EC6100 90000038  // Read 64-bit from LDS
	ds_read_b64 v[146:147], v56 offset:24960                   // 0000000055A8: D8EC6180 92000038  // Read 64-bit from LDS
	ds_read_b64 v[148:149], v56 offset:25856                   // 0000000055B0: D8EC6500 94000038  // Read 64-bit from LDS
	ds_read_b64 v[150:151], v56 offset:25984                   // 0000000055B8: D8EC6580 96000038  // Read 64-bit from LDS
	ds_read_b64 v[152:153], v56 offset:26880                   // 0000000055C0: D8EC6900 98000038  // Read 64-bit from LDS
	ds_read_b64 v[154:155], v56 offset:27008                   // 0000000055C8: D8EC6980 9A000038  // Read 64-bit from LDS
	ds_read_b64 v[156:157], v56 offset:27904                   // 0000000055D0: D8EC6D00 9C000038  // Read 64-bit from LDS
	ds_read_b64 v[158:159], v56 offset:28032                   // 0000000055D8: D8EC6D80 9E000038  // Read 64-bit from LDS
	s_add_u32 s12, s56, s12                                    // 0000000055E0: 800C0C38
	s_addc_u32 s13, 0, s13                                     // 0000000055E4: 820D0D80
	s_add_u32 s16, s79, s16                                    // 0000000055E8: 8010104F
	s_addc_u32 s17, 0, s17                                     // 0000000055EC: 82111180
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 0000000055F0: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 0000000055F4: BF8A0000  // Workgroup barrier synchronization
	v_mov_b32_e32 v192, 0                                      // 0000000055F8: 7F800280  // Vector move
	v_mov_b32_e32 v224, 0                                      // 0000000055FC: 7FC00280  // Vector move
	v_mov_b32_e32 v193, 0                                      // 000000005600: 7F820280  // Vector move
	v_mov_b32_e32 v225, 0                                      // 000000005604: 7FC20280  // Vector move
	v_mov_b32_e32 v194, 0                                      // 000000005608: 7F840280  // Vector move
	v_mov_b32_e32 v226, 0                                      // 00000000560C: 7FC40280  // Vector move
	v_mov_b32_e32 v195, 0                                      // 000000005610: 7F860280  // Vector move
	v_mov_b32_e32 v227, 0                                      // 000000005614: 7FC60280  // Vector move
	v_mov_b32_e32 v196, 0                                      // 000000005618: 7F880280  // Vector move
	v_mov_b32_e32 v228, 0                                      // 00000000561C: 7FC80280  // Vector move
	v_mov_b32_e32 v197, 0                                      // 000000005620: 7F8A0280  // Vector move
	v_mov_b32_e32 v229, 0                                      // 000000005624: 7FCA0280  // Vector move
	v_mov_b32_e32 v198, 0                                      // 000000005628: 7F8C0280  // Vector move
	v_mov_b32_e32 v230, 0                                      // 00000000562C: 7FCC0280  // Vector move
	v_mov_b32_e32 v199, 0                                      // 000000005630: 7F8E0280  // Vector move
	v_mov_b32_e32 v231, 0                                      // 000000005634: 7FCE0280  // Vector move
	v_mov_b32_e32 v200, 0                                      // 000000005638: 7F900280  // Vector move
	v_mov_b32_e32 v232, 0                                      // 00000000563C: 7FD00280  // Vector move
	v_mov_b32_e32 v201, 0                                      // 000000005640: 7F920280  // Vector move
	v_mov_b32_e32 v233, 0                                      // 000000005644: 7FD20280  // Vector move
	v_mov_b32_e32 v202, 0                                      // 000000005648: 7F940280  // Vector move
	v_mov_b32_e32 v234, 0                                      // 00000000564C: 7FD40280  // Vector move
	v_mov_b32_e32 v203, 0                                      // 000000005650: 7F960280  // Vector move
	v_mov_b32_e32 v235, 0                                      // 000000005654: 7FD60280  // Vector move
	v_mov_b32_e32 v204, 0                                      // 000000005658: 7F980280  // Vector move
	v_mov_b32_e32 v236, 0                                      // 00000000565C: 7FD80280  // Vector move
	v_mov_b32_e32 v205, 0                                      // 000000005660: 7F9A0280  // Vector move
	v_mov_b32_e32 v237, 0                                      // 000000005664: 7FDA0280  // Vector move
	v_mov_b32_e32 v206, 0                                      // 000000005668: 7F9C0280  // Vector move
	v_mov_b32_e32 v238, 0                                      // 00000000566C: 7FDC0280  // Vector move
	v_mov_b32_e32 v207, 0                                      // 000000005670: 7F9E0280  // Vector move
	v_mov_b32_e32 v239, 0                                      // 000000005674: 7FDE0280  // Vector move
	ds_write_b64 v4, v[192:193] offset:20736                   // 000000005678: D89A5100 0000C004  // Write 64-bit to LDS
	ds_write_b64 v4, v[194:195] offset:29440                   // 000000005680: D89A7300 0000C204  // Write 64-bit to LDS
	ds_write_b64 v4, v[196:197] offset:22912                   // 000000005688: D89A5980 0000C404  // Write 64-bit to LDS
	ds_write_b64 v4, v[198:199] offset:31616                   // 000000005690: D89A7B80 0000C604  // Write 64-bit to LDS
	ds_write_b64 v4, v[200:201] offset:25088                   // 000000005698: D89A6200 0000C804  // Write 64-bit to LDS
	ds_write_b64 v4, v[202:203] offset:33792                   // 0000000056A0: D89A8400 0000CA04  // Write 64-bit to LDS
	ds_write_b64 v4, v[204:205] offset:27264                   // 0000000056A8: D89A6A80 0000CC04  // Write 64-bit to LDS
	ds_write_b64 v4, v[206:207] offset:35968                   // 0000000056B0: D89A8C80 0000CE04  // Write 64-bit to LDS
	s_mov_b32 s80, 0                                           // 0000000056B8: BED00080  // Scalar move (constant/register)
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 0000000056BC: BF8C0000  // Wait for memory operations (CRITICAL for perf)


// --- LABEL (potential loop target) ---
00000000000056c0 <label_0A30>:
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(12) lgkmcnt(0)                             // 0000000056C0: BF8C007C  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 0000000056C4: BF8A0000  // Workgroup barrier synchronization

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[192:195], a[0:1], v[128:129], 0// 0000000056C8: D3F300C0 0A030100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v64, v5 offset:20736                           // 0000000056D0: D86C5100 40000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v65, v5 offset:25088                           // 0000000056D8: D86C6200 41000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[192:195], a[2:3], v[130:131], v[192:195]// 0000000056E0: D3F300C0 0F030502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[64:67], v48, s[12:15], 0 offen       // 0000000056E8: E05C1000 80834030
	v_mfma_f32_16x16x32_fp8_fp8 v[196:199], a[0:1], v[144:145], 0// 0000000056F0: D3F300C4 0A032100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v66, v5 offset:20768                           // 0000000056F8: D86C5120 42000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v67, v5 offset:25120                           // 000000005700: D86C6220 43000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[196:199], a[2:3], v[146:147], v[196:199]// 000000005708: D3F300C4 0F132502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v24, v6, s[16:19], 0 offen               // 000000005710: E0501000 80041806
	v_mfma_f32_16x16x32_fp8_fp8 v[200:203], a[4:5], v[128:129], 0// 000000005718: D3F300C8 0A030104  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v68, v5 offset:20800                           // 000000005720: D86C5140 44000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v69, v5 offset:25152                           // 000000005728: D86C6240 45000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[200:203], a[6:7], v[130:131], v[200:203]// 000000005730: D3F300C8 0F230506  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[68:71], v49, s[12:15], 0 offen       // 000000005738: E05C1000 80834431
	v_mfma_f32_16x16x32_fp8_fp8 v[204:207], a[4:5], v[144:145], 0// 000000005740: D3F300CC 0A032104  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v70, v5 offset:20832                           // 000000005748: D86C5160 46000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v71, v5 offset:25184                           // 000000005750: D86C6260 47000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[204:207], a[6:7], v[146:147], v[204:207]// 000000005758: D3F300CC 0F332506  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[208:211], a[8:9], v[128:129], 0// 000000005760: D3F300D0 0A030108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v72, v5 offset:29440                           // 000000005768: D86C7300 48000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v73, v5 offset:33792                           // 000000005770: D86C8400 49000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[208:211], a[10:11], v[130:131], v[208:211]// 000000005778: D3F300D0 0F43050A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[72:75], v50, s[12:15], 0 offen       // 000000005780: E05C1000 80834832
	v_mfma_f32_16x16x32_fp8_fp8 v[212:215], a[8:9], v[144:145], 0// 000000005788: D3F300D4 0A032108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v74, v5 offset:29472                           // 000000005790: D86C7320 4A000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v75, v5 offset:33824                           // 000000005798: D86C8420 4B000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[212:215], a[10:11], v[146:147], v[212:215]// 0000000057A0: D3F300D4 0F53250A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[216:219], a[12:13], v[128:129], 0// 0000000057A8: D3F300D8 0A03010C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v76, v5 offset:29504                           // 0000000057B0: D86C7340 4C000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v77, v5 offset:33856                           // 0000000057B8: D86C8440 4D000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[216:219], a[14:15], v[130:131], v[216:219]// 0000000057C0: D3F300D8 0F63050E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[76:79], v51, s[12:15], 0 offen       // 0000000057C8: E05C1000 80834C33
	s_add_u32 s12, s78, s12                                    // 0000000057D0: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 0000000057D4: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[220:223], a[12:13], v[144:145], 0// 0000000057D8: D3F300DC 0A03210C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v78, v5 offset:29536                           // 0000000057E0: D86C7360 4E000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v79, v5 offset:33888                           // 0000000057E8: D86C8460 4F000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[220:223], a[14:15], v[146:147], v[220:223]// 0000000057F0: D3F300DC 0F73250E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	s_waitcnt vmcnt(13)                                        // 0000000057F8: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[192:195], a[16:17], v[132:133], v[192:195]// 0000000057FC: D3F300C0 0F030910  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[192:195], a[18:19], v[134:135], v[192:195]// 000000005804: D3F300C0 0F030D12  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[80:83], v48, s[12:15], 0 offen       // 00000000580C: E05C1000 80835030
	v_mfma_f32_16x16x32_fp8_fp8 v[196:199], a[16:17], v[148:149], v[196:199]// 000000005814: D3F300C4 0F132910  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[196:199], a[18:19], v[150:151], v[196:199]// 00000000581C: D3F300C4 0F132D12  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[200:203], a[20:21], v[132:133], v[200:203]// 000000005824: D3F300C8 0F230914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[200:203], a[22:23], v[134:135], v[200:203]// 00000000582C: D3F300C8 0F230D16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[84:87], v49, s[12:15], 0 offen       // 000000005834: E05C1000 80835431
	v_mfma_f32_16x16x32_fp8_fp8 v[204:207], a[20:21], v[148:149], v[204:207]// 00000000583C: D3F300CC 0F332914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[204:207], a[22:23], v[150:151], v[204:207]// 000000005844: D3F300CC 0F332D16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[208:211], a[24:25], v[132:133], v[208:211]// 00000000584C: D3F300D0 0F430918  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[208:211], a[26:27], v[134:135], v[208:211]// 000000005854: D3F300D0 0F430D1A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[88:91], v50, s[12:15], 0 offen       // 00000000585C: E05C1000 80835832
	v_mfma_f32_16x16x32_fp8_fp8 v[212:215], a[24:25], v[148:149], v[212:215]// 000000005864: D3F300D4 0F532918  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[212:215], a[26:27], v[150:151], v[212:215]// 00000000586C: D3F300D4 0F532D1A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[216:219], a[28:29], v[132:133], v[216:219]// 000000005874: D3F300D8 0F63091C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[216:219], a[30:31], v[134:135], v[216:219]// 00000000587C: D3F300D8 0F630D1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[92:95], v51, s[12:15], 0 offen       // 000000005884: E05C1000 80835C33
	s_add_u32 s12, s78, s12                                    // 00000000588C: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000005890: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[220:223], a[28:29], v[148:149], v[220:223]// 000000005894: D3F300DC 0F73291C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[220:223], a[30:31], v[150:151], v[220:223]// 00000000589C: D3F300DC 0F732D1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v23, v32 row_newbcast:0 row_mask:0xf bank_mask:0xf// 0000000058A4: 0A7040FA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000058AC: 7E720338  // Vector move
	v_pk_mul_f32 v[192:193], v[56:57], v[192:193]              // 0000000058B0: D3B140C0 18038138
	v_pk_mul_f32 v[194:195], v[56:57], v[194:195]              // 0000000058B8: D3B140C2 18038538
	v_pk_mul_f32 v[200:201], v[56:57], v[200:201]              // 0000000058C0: D3B140C8 18039138
	v_pk_mul_f32 v[202:203], v[56:57], v[202:203]              // 0000000058C8: D3B140CA 18039538
	v_mul_f32_dpp v56, v23, v32 row_newbcast:1 row_mask:0xf bank_mask:0xf// 0000000058D0: 0A7040FA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000058D8: 7E720338  // Vector move
	v_pk_mul_f32 v[208:209], v[56:57], v[208:209]              // 0000000058DC: D3B140D0 1803A138
	v_pk_mul_f32 v[210:211], v[56:57], v[210:211]              // 0000000058E4: D3B140D2 1803A538
	v_pk_mul_f32 v[216:217], v[56:57], v[216:217]              // 0000000058EC: D3B140D8 1803B138
	v_pk_mul_f32 v[218:219], v[56:57], v[218:219]              // 0000000058F4: D3B140DA 1803B538
	v_mul_f32_dpp v56, v23, v33 row_newbcast:0 row_mask:0xf bank_mask:0xf// 0000000058FC: 0A7042FA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000005904: 7E720338  // Vector move
	v_pk_mul_f32 v[196:197], v[56:57], v[196:197]              // 000000005908: D3B140C4 18038938
	v_pk_mul_f32 v[198:199], v[56:57], v[198:199]              // 000000005910: D3B140C6 18038D38
	v_pk_mul_f32 v[204:205], v[56:57], v[204:205]              // 000000005918: D3B140CC 18039938
	v_pk_mul_f32 v[206:207], v[56:57], v[206:207]              // 000000005920: D3B140CE 18039D38
	v_mul_f32_dpp v56, v23, v33 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000005928: 0A7042FA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000005930: 7E720338  // Vector move
	v_pk_mul_f32 v[212:213], v[56:57], v[212:213]              // 000000005934: D3B140D4 1803A938
	v_pk_mul_f32 v[214:215], v[56:57], v[214:215]              // 00000000593C: D3B140D6 1803AD38
	v_pk_mul_f32 v[220:221], v[56:57], v[220:221]              // 000000005944: D3B140DC 1803B938
	v_pk_mul_f32 v[222:223], v[56:57], v[222:223]              // 00000000594C: D3B140DE 1803BD38
	s_waitcnt vmcnt(13)                                        // 000000005954: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[32:33], v[136:137], 0// 000000005958: D3F300A0 0A031120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[224:225] offset:38144                   // 000000005960: D89A9500 0000E004  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[34:35], v[138:139], v[160:163]// 000000005968: D3F300A0 0E831522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[96:99], v48, s[12:15], 0 offen       // 000000005970: E05C1000 80836030
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[32:33], v[152:153], 0// 000000005978: D3F300A4 0A033120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[226:227] offset:46848                   // 000000005980: D89AB700 0000E204  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[34:35], v[154:155], v[164:167]// 000000005988: D3F300A4 0E933522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[36:37], v[136:137], 0// 000000005990: D3F300A8 0A031124  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[228:229] offset:40320                   // 000000005998: D89A9D80 0000E404  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[38:39], v[138:139], v[168:171]// 0000000059A0: D3F300A8 0EA31526  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[100:103], v49, s[12:15], 0 offen     // 0000000059A8: E05C1000 80836431
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[36:37], v[152:153], 0// 0000000059B0: D3F300AC 0A033124  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[230:231] offset:49024                   // 0000000059B8: D89ABF80 0000E604  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[38:39], v[154:155], v[172:175]// 0000000059C0: D3F300AC 0EB33526  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[40:41], v[136:137], 0// 0000000059C8: D3F300B0 0A031128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[232:233] offset:42496                   // 0000000059D0: D89AA600 0000E804  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[42:43], v[138:139], v[176:179]// 0000000059D8: D3F300B0 0EC3152A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[104:107], v50, s[12:15], 0 offen     // 0000000059E0: E05C1000 80836832
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[40:41], v[152:153], 0// 0000000059E8: D3F300B4 0A033128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[234:235] offset:51200                   // 0000000059F0: D89AC800 0000EA04  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[42:43], v[154:155], v[180:183]// 0000000059F8: D3F300B4 0ED3352A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[44:45], v[136:137], 0// 000000005A00: D3F300B8 0A03112C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[236:237] offset:44672                   // 000000005A08: D89AAE80 0000EC04  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[46:47], v[138:139], v[184:187]// 000000005A10: D3F300B8 0EE3152E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[108:111], v51, s[12:15], 0 offen     // 000000005A18: E05C1000 80836C33
	s_add_u32 s12, s78, s12                                    // 000000005A20: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000005A24: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[44:45], v[152:153], 0// 000000005A28: D3F300BC 0A03312C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[238:239] offset:53376                   // 000000005A30: D89AD080 0000EE04  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[46:47], v[154:155], v[188:191]// 000000005A38: D3F300BC 0EF3352E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	s_waitcnt vmcnt(13)                                        // 000000005A40: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[48:49], v[140:141], v[160:163]// 000000005A44: D3F300A0 0E831930  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[50:51], v[142:143], v[160:163]// 000000005A4C: D3F300A0 0E831D32  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[112:115], v48, s[12:15], 0 offen     // 000000005A54: E05C1000 80837030
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[48:49], v[156:157], v[164:167]// 000000005A5C: D3F300A4 0E933930  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[50:51], v[158:159], v[164:167]// 000000005A64: D3F300A4 0E933D32  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[52:53], v[140:141], v[168:171]// 000000005A6C: D3F300A8 0EA31934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[54:55], v[142:143], v[168:171]// 000000005A74: D3F300A8 0EA31D36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[116:119], v49, s[12:15], 0 offen     // 000000005A7C: E05C1000 80837431
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[52:53], v[156:157], v[172:175]// 000000005A84: D3F300AC 0EB33934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[54:55], v[158:159], v[172:175]// 000000005A8C: D3F300AC 0EB33D36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[56:57], v[140:141], v[176:179]// 000000005A94: D3F300B0 0EC31938  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[58:59], v[142:143], v[176:179]// 000000005A9C: D3F300B0 0EC31D3A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[120:123], v50, s[12:15], 0 offen     // 000000005AA4: E05C1000 80837832
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[56:57], v[156:157], v[180:183]// 000000005AAC: D3F300B4 0ED33938  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[58:59], v[158:159], v[180:183]// 000000005AB4: D3F300B4 0ED33D3A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[60:61], v[140:141], v[184:187]// 000000005ABC: D3F300B8 0EE3193C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[62:63], v[142:143], v[184:187]// 000000005AC4: D3F300B8 0EE31D3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[124:127], v51, s[12:15], 0 offen     // 000000005ACC: E05C1000 80837C33
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[60:61], v[156:157], v[188:191]// 000000005AD4: D3F300BC 0EF3393C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[62:63], v[158:159], v[188:191]// 000000005ADC: D3F300BC 0EF33D3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v23, v34 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000005AE4: 0A7044FA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000005AEC: 7E720338  // Vector move
	v_pk_fma_f32 v[192:193], v[160:161], v[56:57], v[192:193]  // 000000005AF0: D3B040C0 1F0271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[194:195], v[162:163], v[56:57], v[194:195]  // 000000005AF8: D3B040C2 1F0A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[200:201], v[168:169], v[56:57], v[200:201]  // 000000005B00: D3B040C8 1F2271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[202:203], v[170:171], v[56:57], v[202:203]  // 000000005B08: D3B040CA 1F2A71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v34 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000005B10: 0A7044FA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000005B18: 7E720338  // Vector move
	v_pk_fma_f32 v[208:209], v[176:177], v[56:57], v[208:209]  // 000000005B1C: D3B040D0 1F4271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[210:211], v[178:179], v[56:57], v[210:211]  // 000000005B24: D3B040D2 1F4A71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[216:217], v[184:185], v[56:57], v[216:217]  // 000000005B2C: D3B040D8 1F6271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[218:219], v[186:187], v[56:57], v[218:219]  // 000000005B34: D3B040DA 1F6A71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v35 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000005B3C: 0A7046FA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000005B44: 7E720338  // Vector move
	v_pk_fma_f32 v[196:197], v[164:165], v[56:57], v[196:197]  // 000000005B48: D3B040C4 1F1271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[198:199], v[166:167], v[56:57], v[198:199]  // 000000005B50: D3B040C6 1F1A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[204:205], v[172:173], v[56:57], v[204:205]  // 000000005B58: D3B040CC 1F3271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[206:207], v[174:175], v[56:57], v[206:207]  // 000000005B60: D3B040CE 1F3A71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v35 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000005B68: 0A7046FA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000005B70: 7E720338  // Vector move
	v_pk_fma_f32 v[212:213], v[180:181], v[56:57], v[212:213]  // 000000005B74: D3B040D4 1F5271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[214:215], v[182:183], v[56:57], v[214:215]  // 000000005B7C: D3B040D6 1F5A71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[220:221], v[188:189], v[56:57], v[220:221]  // 000000005B84: D3B040DC 1F7271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[222:223], v[190:191], v[56:57], v[222:223]  // 000000005B8C: D3B040DE 1F7A71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x200, s80                                  // 000000005B94: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 000000005B9C: BF0A513C
	s_cselect_b32 s56, s56, 0                                  // 000000005BA0: 85388038
	s_cselect_b32 s78, s78, 0                                  // 000000005BA4: 854E804E
	s_cselect_b32 s79, s79, 0                                  // 000000005BA8: 854F804F
	s_add_u32 s12, s56, s12                                    // 000000005BAC: 800C0C38
	s_addc_u32 s13, 0, s13                                     // 000000005BB0: 820D0D80
	s_add_u32 s16, s79, s16                                    // 000000005BB4: 8010104F
	s_addc_u32 s17, 0, s17                                     // 000000005BB8: 82111180
	v_mov_b32_e32 v56, v25                                     // 000000005BBC: 7E700319  // Vector move
	v_mov_b32_e32 v57, v25                                     // 000000005BC0: 7E720319  // Vector move
	v_pk_mul_f32 v[192:193], v[56:57], v[192:193]              // 000000005BC4: D3B140C0 18038138
	v_pk_mul_f32 v[194:195], v[56:57], v[194:195]              // 000000005BCC: D3B140C2 18038538
	v_pk_mul_f32 v[200:201], v[56:57], v[200:201]              // 000000005BD4: D3B140C8 18039138
	v_pk_mul_f32 v[202:203], v[56:57], v[202:203]              // 000000005BDC: D3B140CA 18039538
	v_pk_mul_f32 v[208:209], v[56:57], v[208:209]              // 000000005BE4: D3B140D0 1803A138
	v_pk_mul_f32 v[210:211], v[56:57], v[210:211]              // 000000005BEC: D3B140D2 1803A538
	v_pk_mul_f32 v[216:217], v[56:57], v[216:217]              // 000000005BF4: D3B140D8 1803B138
	v_pk_mul_f32 v[218:219], v[56:57], v[218:219]              // 000000005BFC: D3B140DA 1803B538
	v_mov_b32_e32 v56, v26                                     // 000000005C04: 7E70031A  // Vector move
	v_mov_b32_e32 v57, v26                                     // 000000005C08: 7E72031A  // Vector move
	v_pk_mul_f32 v[196:197], v[56:57], v[196:197]              // 000000005C0C: D3B140C4 18038938
	v_pk_mul_f32 v[198:199], v[56:57], v[198:199]              // 000000005C14: D3B140C6 18038D38
	v_pk_mul_f32 v[204:205], v[56:57], v[204:205]              // 000000005C1C: D3B140CC 18039938
	v_pk_mul_f32 v[206:207], v[56:57], v[206:207]              // 000000005C24: D3B140CE 18039D38
	v_pk_mul_f32 v[212:213], v[56:57], v[212:213]              // 000000005C2C: D3B140D4 1803A938
	v_pk_mul_f32 v[214:215], v[56:57], v[214:215]              // 000000005C34: D3B140D6 1803AD38
	v_pk_mul_f32 v[220:221], v[56:57], v[220:221]              // 000000005C3C: D3B140DC 1803B938
	v_pk_mul_f32 v[222:223], v[56:57], v[222:223]              // 000000005C44: D3B140DE 1803BD38
	v_cmp_u_f32_e64 s[48:49], v192, v192                       // 000000005C4C: D0480030 000381C0
	v_add3_u32 v52, v192, v55, 1                               // 000000005C54: D1FF0034 02066FC0
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005C5C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v193, v193                       // 000000005C64: D0480030 000383C1
	v_add3_u32 v52, v193, v55, 1                               // 000000005C6C: D1FF0034 02066FC1
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005C74: D1000039 00C26D34
	v_perm_b32 v192, v57, v56, s52                             // 000000005C7C: D1ED00C0 00D27139
	v_cmp_u_f32_e64 s[48:49], v194, v194                       // 000000005C84: D0480030 000385C2
	v_add3_u32 v52, v194, v55, 1                               // 000000005C8C: D1FF0034 02066FC2
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005C94: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v195, v195                       // 000000005C9C: D0480030 000387C3
	v_add3_u32 v52, v195, v55, 1                               // 000000005CA4: D1FF0034 02066FC3
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005CAC: D1000039 00C26D34
	v_perm_b32 v193, v57, v56, s52                             // 000000005CB4: D1ED00C1 00D27139
	v_cmp_u_f32_e64 s[48:49], v196, v196                       // 000000005CBC: D0480030 000389C4
	v_add3_u32 v52, v196, v55, 1                               // 000000005CC4: D1FF0034 02066FC4
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005CCC: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v197, v197                       // 000000005CD4: D0480030 00038BC5
	v_add3_u32 v52, v197, v55, 1                               // 000000005CDC: D1FF0034 02066FC5
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005CE4: D1000039 00C26D34
	v_perm_b32 v194, v57, v56, s52                             // 000000005CEC: D1ED00C2 00D27139
	v_cmp_u_f32_e64 s[48:49], v198, v198                       // 000000005CF4: D0480030 00038DC6
	v_add3_u32 v52, v198, v55, 1                               // 000000005CFC: D1FF0034 02066FC6
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005D04: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v199, v199                       // 000000005D0C: D0480030 00038FC7
	v_add3_u32 v52, v199, v55, 1                               // 000000005D14: D1FF0034 02066FC7
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005D1C: D1000039 00C26D34
	v_perm_b32 v195, v57, v56, s52                             // 000000005D24: D1ED00C3 00D27139
	v_cmp_u_f32_e64 s[48:49], v200, v200                       // 000000005D2C: D0480030 000391C8
	v_add3_u32 v52, v200, v55, 1                               // 000000005D34: D1FF0034 02066FC8
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005D3C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v201, v201                       // 000000005D44: D0480030 000393C9
	v_add3_u32 v52, v201, v55, 1                               // 000000005D4C: D1FF0034 02066FC9
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005D54: D1000039 00C26D34
	v_perm_b32 v196, v57, v56, s52                             // 000000005D5C: D1ED00C4 00D27139
	v_cmp_u_f32_e64 s[48:49], v202, v202                       // 000000005D64: D0480030 000395CA
	v_add3_u32 v52, v202, v55, 1                               // 000000005D6C: D1FF0034 02066FCA
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005D74: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v203, v203                       // 000000005D7C: D0480030 000397CB
	v_add3_u32 v52, v203, v55, 1                               // 000000005D84: D1FF0034 02066FCB
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005D8C: D1000039 00C26D34
	v_perm_b32 v197, v57, v56, s52                             // 000000005D94: D1ED00C5 00D27139
	v_cmp_u_f32_e64 s[48:49], v204, v204                       // 000000005D9C: D0480030 000399CC
	v_add3_u32 v52, v204, v55, 1                               // 000000005DA4: D1FF0034 02066FCC
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005DAC: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v205, v205                       // 000000005DB4: D0480030 00039BCD
	v_add3_u32 v52, v205, v55, 1                               // 000000005DBC: D1FF0034 02066FCD
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005DC4: D1000039 00C26D34
	v_perm_b32 v198, v57, v56, s52                             // 000000005DCC: D1ED00C6 00D27139
	v_cmp_u_f32_e64 s[48:49], v206, v206                       // 000000005DD4: D0480030 00039DCE
	v_add3_u32 v52, v206, v55, 1                               // 000000005DDC: D1FF0034 02066FCE
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005DE4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v207, v207                       // 000000005DEC: D0480030 00039FCF
	v_add3_u32 v52, v207, v55, 1                               // 000000005DF4: D1FF0034 02066FCF
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005DFC: D1000039 00C26D34
	v_perm_b32 v199, v57, v56, s52                             // 000000005E04: D1ED00C7 00D27139
	v_cmp_u_f32_e64 s[48:49], v208, v208                       // 000000005E0C: D0480030 0003A1D0
	v_add3_u32 v52, v208, v55, 1                               // 000000005E14: D1FF0034 02066FD0
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005E1C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v209, v209                       // 000000005E24: D0480030 0003A3D1
	v_add3_u32 v52, v209, v55, 1                               // 000000005E2C: D1FF0034 02066FD1
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005E34: D1000039 00C26D34
	v_perm_b32 v200, v57, v56, s52                             // 000000005E3C: D1ED00C8 00D27139
	v_cmp_u_f32_e64 s[48:49], v210, v210                       // 000000005E44: D0480030 0003A5D2
	v_add3_u32 v52, v210, v55, 1                               // 000000005E4C: D1FF0034 02066FD2
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005E54: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v211, v211                       // 000000005E5C: D0480030 0003A7D3
	v_add3_u32 v52, v211, v55, 1                               // 000000005E64: D1FF0034 02066FD3
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005E6C: D1000039 00C26D34
	v_perm_b32 v201, v57, v56, s52                             // 000000005E74: D1ED00C9 00D27139
	v_cmp_u_f32_e64 s[48:49], v212, v212                       // 000000005E7C: D0480030 0003A9D4
	v_add3_u32 v52, v212, v55, 1                               // 000000005E84: D1FF0034 02066FD4
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005E8C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v213, v213                       // 000000005E94: D0480030 0003ABD5
	v_add3_u32 v52, v213, v55, 1                               // 000000005E9C: D1FF0034 02066FD5
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005EA4: D1000039 00C26D34
	v_perm_b32 v202, v57, v56, s52                             // 000000005EAC: D1ED00CA 00D27139
	v_cmp_u_f32_e64 s[48:49], v214, v214                       // 000000005EB4: D0480030 0003ADD6
	v_add3_u32 v52, v214, v55, 1                               // 000000005EBC: D1FF0034 02066FD6
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005EC4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v215, v215                       // 000000005ECC: D0480030 0003AFD7
	v_add3_u32 v52, v215, v55, 1                               // 000000005ED4: D1FF0034 02066FD7
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005EDC: D1000039 00C26D34
	v_perm_b32 v203, v57, v56, s52                             // 000000005EE4: D1ED00CB 00D27139
	v_cmp_u_f32_e64 s[48:49], v216, v216                       // 000000005EEC: D0480030 0003B1D8
	v_add3_u32 v52, v216, v55, 1                               // 000000005EF4: D1FF0034 02066FD8
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005EFC: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v217, v217                       // 000000005F04: D0480030 0003B3D9
	v_add3_u32 v52, v217, v55, 1                               // 000000005F0C: D1FF0034 02066FD9
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005F14: D1000039 00C26D34
	v_perm_b32 v204, v57, v56, s52                             // 000000005F1C: D1ED00CC 00D27139
	v_cmp_u_f32_e64 s[48:49], v218, v218                       // 000000005F24: D0480030 0003B5DA
	v_add3_u32 v52, v218, v55, 1                               // 000000005F2C: D1FF0034 02066FDA
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005F34: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v219, v219                       // 000000005F3C: D0480030 0003B7DB
	v_add3_u32 v52, v219, v55, 1                               // 000000005F44: D1FF0034 02066FDB
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005F4C: D1000039 00C26D34
	v_perm_b32 v205, v57, v56, s52                             // 000000005F54: D1ED00CD 00D27139
	v_cmp_u_f32_e64 s[48:49], v220, v220                       // 000000005F5C: D0480030 0003B9DC
	v_add3_u32 v52, v220, v55, 1                               // 000000005F64: D1FF0034 02066FDC
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005F6C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v221, v221                       // 000000005F74: D0480030 0003BBDD
	v_add3_u32 v52, v221, v55, 1                               // 000000005F7C: D1FF0034 02066FDD
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005F84: D1000039 00C26D34
	v_perm_b32 v206, v57, v56, s52                             // 000000005F8C: D1ED00CE 00D27139
	v_cmp_u_f32_e64 s[48:49], v222, v222                       // 000000005F94: D0480030 0003BDDE
	v_add3_u32 v52, v222, v55, 1                               // 000000005F9C: D1FF0034 02066FDE
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000005FA4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v223, v223                       // 000000005FAC: D0480030 0003BFDF
	v_add3_u32 v52, v223, v55, 1                               // 000000005FB4: D1FF0034 02066FDF
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000005FBC: D1000039 00C26D34
	v_perm_b32 v207, v57, v56, s52                             // 000000005FC4: D1ED00CF 00D27139
	s_cmp_ge_u32 s80, 0x200                                    // 000000005FCC: BF09FF50 00000200
	s_cselect_b32 s59, 0x200, s59                              // 000000005FD4: 853B3BFF 00000200
	s_setvskip s20, 0                                          // 000000005FDC: BF108014
	global_atomic_pk_add_bf16 v80, v64, s[8:9]                 // 000000005FE0: DD488000 00084050
	s_setvskip 0, 0                                            // 000000005FE8: BF108080
	s_setvskip s20, 0                                          // 000000005FEC: BF108014
	global_atomic_pk_add_bf16 v80, v65, s[8:9] offset:256      // 000000005FF0: DD488100 00084150
	s_setvskip 0, 0                                            // 000000005FF8: BF108080
	s_setvskip s20, 1                                          // 000000005FFC: BF108114
	global_atomic_pk_add_bf16 v82, v66, s[8:9]                 // 000000006000: DD488000 00084252
	s_setvskip 0, 0                                            // 000000006008: BF108080
	s_setvskip s20, 1                                          // 00000000600C: BF108114
	global_atomic_pk_add_bf16 v82, v67, s[8:9] offset:256      // 000000006010: DD488100 00084352
	s_setvskip 0, 0                                            // 000000006018: BF108080
	s_setvskip s20, 2                                          // 00000000601C: BF108214
	global_atomic_pk_add_bf16 v84, v68, s[8:9]                 // 000000006020: DD488000 00084454
	s_setvskip 0, 0                                            // 000000006028: BF108080
	s_setvskip s20, 2                                          // 00000000602C: BF108214
	global_atomic_pk_add_bf16 v84, v69, s[8:9] offset:256      // 000000006030: DD488100 00084554
	s_setvskip 0, 0                                            // 000000006038: BF108080
	s_setvskip s20, 3                                          // 00000000603C: BF108314
	global_atomic_pk_add_bf16 v86, v70, s[8:9]                 // 000000006040: DD488000 00084656
	s_setvskip 0, 0                                            // 000000006048: BF108080
	s_setvskip s20, 3                                          // 00000000604C: BF108314
	global_atomic_pk_add_bf16 v86, v71, s[8:9] offset:256      // 000000006050: DD488100 00084756
	s_setvskip 0, 0                                            // 000000006058: BF108080
	s_setvskip s20, 4                                          // 00000000605C: BF108414
	global_atomic_pk_add_bf16 v88, v72, s[8:9]                 // 000000006060: DD488000 00084858
	s_setvskip 0, 0                                            // 000000006068: BF108080
	s_setvskip s20, 4                                          // 00000000606C: BF108414
	global_atomic_pk_add_bf16 v88, v73, s[8:9] offset:256      // 000000006070: DD488100 00084958
	s_setvskip 0, 0                                            // 000000006078: BF108080
	s_setvskip s20, 5                                          // 00000000607C: BF108514
	global_atomic_pk_add_bf16 v90, v74, s[8:9]                 // 000000006080: DD488000 00084A5A
	s_setvskip 0, 0                                            // 000000006088: BF108080
	s_setvskip s20, 5                                          // 00000000608C: BF108514
	global_atomic_pk_add_bf16 v90, v75, s[8:9] offset:256      // 000000006090: DD488100 00084B5A
	s_setvskip 0, 0                                            // 000000006098: BF108080
	s_setvskip s20, 6                                          // 00000000609C: BF108614
	global_atomic_pk_add_bf16 v92, v76, s[8:9]                 // 0000000060A0: DD488000 00084C5C
	s_setvskip 0, 0                                            // 0000000060A8: BF108080
	s_setvskip s20, 6                                          // 0000000060AC: BF108614
	global_atomic_pk_add_bf16 v92, v77, s[8:9] offset:256      // 0000000060B0: DD488100 00084D5C
	s_setvskip 0, 0                                            // 0000000060B8: BF108080
	s_setvskip s20, 7                                          // 0000000060BC: BF108714
	global_atomic_pk_add_bf16 v94, v78, s[8:9]                 // 0000000060C0: DD488000 00084E5E
	s_setvskip 0, 0                                            // 0000000060C8: BF108080
	s_setvskip s20, 7                                          // 0000000060CC: BF108714
	global_atomic_pk_add_bf16 v94, v79, s[8:9] offset:256      // 0000000060D0: DD488100 00084F5E
	s_setvskip 0, 0                                            // 0000000060D8: BF108080
	s_add_u32 s8, s59, s8                                      // 0000000060DC: 8008083B
	s_addc_u32 s9, 0, s9                                       // 0000000060E0: 82090980
	s_addk_i32 s80, 0x100                                      // 0000000060E4: B7500100
	s_cmp_lt_i32 s80, s81                                      // 0000000060E8: BF045150
	s_cbranch_scc0 label_0F49                                  // 0000000060EC: BF84028D  // Branch if SCC==0 (condition false)
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(12) lgkmcnt(0)                             // 0000000060F0: BF8C007C  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 0000000060F4: BF8A0000  // Workgroup barrier synchronization
	v_mfma_f32_16x16x32_fp8_fp8 v[224:227], a[64:65], v[128:129], 0// 0000000060F8: D3F300E0 0A030140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v64, v5 offset:38144                           // 000000006100: D86C9500 40000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v65, v5 offset:42496                           // 000000006108: D86CA600 41000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[224:227], a[66:67], v[130:131], v[224:227]// 000000006110: D3F300E0 0F830542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[0:3], v48, s[12:15], 0 offen         // 000000006118: E05C1000 80830030
	v_mfma_f32_16x16x32_fp8_fp8 v[228:231], a[64:65], v[144:145], 0// 000000006120: D3F300E4 0A032140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v66, v5 offset:38176                           // 000000006128: D86C9520 42000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v67, v5 offset:42528                           // 000000006130: D86CA620 43000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[228:231], a[66:67], v[146:147], v[228:231]// 000000006138: D3F300E4 0F932542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v23, v6, s[16:19], 0 offen               // 000000006140: E0501000 80041706
	v_mfma_f32_16x16x32_fp8_fp8 v[232:235], a[68:69], v[128:129], 0// 000000006148: D3F300E8 0A030144  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v68, v5 offset:38208                           // 000000006150: D86C9540 44000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v69, v5 offset:42560                           // 000000006158: D86CA640 45000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[232:235], a[70:71], v[130:131], v[232:235]// 000000006160: D3F300E8 0FA30546  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[4:7], v49, s[12:15], 0 offen         // 000000006168: E05C1000 80830431
	v_mfma_f32_16x16x32_fp8_fp8 v[236:239], a[68:69], v[144:145], 0// 000000006170: D3F300EC 0A032144  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v70, v5 offset:38240                           // 000000006178: D86C9560 46000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v71, v5 offset:42592                           // 000000006180: D86CA660 47000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[236:239], a[70:71], v[146:147], v[236:239]// 000000006188: D3F300EC 0FB32546  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[240:243], a[72:73], v[128:129], 0// 000000006190: D3F300F0 0A030148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v72, v5 offset:46848                           // 000000006198: D86CB700 48000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v73, v5 offset:51200                           // 0000000061A0: D86CC800 49000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[240:243], a[74:75], v[130:131], v[240:243]// 0000000061A8: D3F300F0 0FC3054A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[8:11], v50, s[12:15], 0 offen        // 0000000061B0: E05C1000 80830832
	v_mfma_f32_16x16x32_fp8_fp8 v[244:247], a[72:73], v[144:145], 0// 0000000061B8: D3F300F4 0A032148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v74, v5 offset:46880                           // 0000000061C0: D86CB720 4A000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v75, v5 offset:51232                           // 0000000061C8: D86CC820 4B000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[244:247], a[74:75], v[146:147], v[244:247]// 0000000061D0: D3F300F4 0FD3254A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[248:251], a[76:77], v[128:129], 0// 0000000061D8: D3F300F8 0A03014C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v76, v5 offset:46912                           // 0000000061E0: D86CB740 4C000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v77, v5 offset:51264                           // 0000000061E8: D86CC840 4D000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[248:251], a[78:79], v[130:131], v[248:251]// 0000000061F0: D3F300F8 0FE3054E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[12:15], v51, s[12:15], 0 offen       // 0000000061F8: E05C1000 80830C33
	s_add_u32 s12, s78, s12                                    // 000000006200: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000006204: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[252:255], a[76:77], v[144:145], 0// 000000006208: D3F300FC 0A03214C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v78, v5 offset:46944                           // 000000006210: D86CB760 4E000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v79, v5 offset:51296                           // 000000006218: D86CC860 4F000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[252:255], a[78:79], v[146:147], v[252:255]// 000000006220: D3F300FC 0FF3254E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	s_waitcnt vmcnt(13)                                        // 000000006228: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[224:227], a[80:81], v[132:133], v[224:227]// 00000000622C: D3F300E0 0F830950  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[224:227], a[82:83], v[134:135], v[224:227]// 000000006234: D3F300E0 0F830D52  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[16:19], v48, s[12:15], 0 offen       // 00000000623C: E05C1000 80831030
	v_mfma_f32_16x16x32_fp8_fp8 v[228:231], a[80:81], v[148:149], v[228:231]// 000000006244: D3F300E4 0F932950  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[228:231], a[82:83], v[150:151], v[228:231]// 00000000624C: D3F300E4 0F932D52  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[232:235], a[84:85], v[132:133], v[232:235]// 000000006254: D3F300E8 0FA30954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[232:235], a[86:87], v[134:135], v[232:235]// 00000000625C: D3F300E8 0FA30D56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[20:23], v49, s[12:15], 0 offen       // 000000006264: E05C1000 80831431
	v_mfma_f32_16x16x32_fp8_fp8 v[236:239], a[84:85], v[148:149], v[236:239]// 00000000626C: D3F300EC 0FB32954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[236:239], a[86:87], v[150:151], v[236:239]// 000000006274: D3F300EC 0FB32D56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[240:243], a[88:89], v[132:133], v[240:243]// 00000000627C: D3F300F0 0FC30958  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[240:243], a[90:91], v[134:135], v[240:243]// 000000006284: D3F300F0 0FC30D5A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[24:27], v50, s[12:15], 0 offen       // 00000000628C: E05C1000 80831832
	v_mfma_f32_16x16x32_fp8_fp8 v[244:247], a[88:89], v[148:149], v[244:247]// 000000006294: D3F300F4 0FD32958  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[244:247], a[90:91], v[150:151], v[244:247]// 00000000629C: D3F300F4 0FD32D5A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[248:251], a[92:93], v[132:133], v[248:251]// 0000000062A4: D3F300F8 0FE3095C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[248:251], a[94:95], v[134:135], v[248:251]// 0000000062AC: D3F300F8 0FE30D5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[28:31], v51, s[12:15], 0 offen       // 0000000062B4: E05C1000 80831C33
	s_add_u32 s12, s78, s12                                    // 0000000062BC: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 0000000062C0: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[252:255], a[92:93], v[148:149], v[252:255]// 0000000062C4: D3F300FC 0FF3295C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[252:255], a[94:95], v[150:151], v[252:255]// 0000000062CC: D3F300FC 0FF32D5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v24, v32 row_newbcast:0 row_mask:0xf bank_mask:0xf// 0000000062D4: 0A7040FA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000062DC: 7E720338  // Vector move
	v_pk_mul_f32 v[224:225], v[56:57], v[224:225]              // 0000000062E0: D3B140E0 1803C138
	v_pk_mul_f32 v[226:227], v[56:57], v[226:227]              // 0000000062E8: D3B140E2 1803C538
	v_pk_mul_f32 v[232:233], v[56:57], v[232:233]              // 0000000062F0: D3B140E8 1803D138
	v_pk_mul_f32 v[234:235], v[56:57], v[234:235]              // 0000000062F8: D3B140EA 1803D538
	v_mul_f32_dpp v56, v24, v32 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000006300: 0A7040FA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006308: 7E720338  // Vector move
	v_pk_mul_f32 v[240:241], v[56:57], v[240:241]              // 00000000630C: D3B140F0 1803E138
	v_pk_mul_f32 v[242:243], v[56:57], v[242:243]              // 000000006314: D3B140F2 1803E538
	v_pk_mul_f32 v[248:249], v[56:57], v[248:249]              // 00000000631C: D3B140F8 1803F138
	v_pk_mul_f32 v[250:251], v[56:57], v[250:251]              // 000000006324: D3B140FA 1803F538
	v_mul_f32_dpp v56, v24, v33 row_newbcast:0 row_mask:0xf bank_mask:0xf// 00000000632C: 0A7042FA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006334: 7E720338  // Vector move
	v_pk_mul_f32 v[228:229], v[56:57], v[228:229]              // 000000006338: D3B140E4 1803C938
	v_pk_mul_f32 v[230:231], v[56:57], v[230:231]              // 000000006340: D3B140E6 1803CD38
	v_pk_mul_f32 v[236:237], v[56:57], v[236:237]              // 000000006348: D3B140EC 1803D938
	v_pk_mul_f32 v[238:239], v[56:57], v[238:239]              // 000000006350: D3B140EE 1803DD38
	v_mul_f32_dpp v56, v24, v33 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000006358: 0A7042FA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006360: 7E720338  // Vector move
	v_pk_mul_f32 v[244:245], v[56:57], v[244:245]              // 000000006364: D3B140F4 1803E938
	v_pk_mul_f32 v[246:247], v[56:57], v[246:247]              // 00000000636C: D3B140F6 1803ED38
	v_pk_mul_f32 v[252:253], v[56:57], v[252:253]              // 000000006374: D3B140FC 1803F938
	v_pk_mul_f32 v[254:255], v[56:57], v[254:255]              // 00000000637C: D3B140FE 1803FD38
	s_waitcnt vmcnt(13)                                        // 000000006384: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[96:97], v[136:137], 0// 000000006388: D3F300A0 0A031160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[192:193] offset:20736                   // 000000006390: D89A5100 0000C004  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[98:99], v[138:139], v[160:163]// 000000006398: D3F300A0 0E831562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[32:35], v48, s[12:15], 0 offen       // 0000000063A0: E05C1000 80832030
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[96:97], v[152:153], 0// 0000000063A8: D3F300A4 0A033160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[194:195] offset:29440                   // 0000000063B0: D89A7300 0000C204  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[98:99], v[154:155], v[164:167]// 0000000063B8: D3F300A4 0E933562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[100:101], v[136:137], 0// 0000000063C0: D3F300A8 0A031164  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[196:197] offset:22912                   // 0000000063C8: D89A5980 0000C404  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[102:103], v[138:139], v[168:171]// 0000000063D0: D3F300A8 0EA31566  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[36:39], v49, s[12:15], 0 offen       // 0000000063D8: E05C1000 80832431
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[100:101], v[152:153], 0// 0000000063E0: D3F300AC 0A033164  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[198:199] offset:31616                   // 0000000063E8: D89A7B80 0000C604  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[102:103], v[154:155], v[172:175]// 0000000063F0: D3F300AC 0EB33566  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[104:105], v[136:137], 0// 0000000063F8: D3F300B0 0A031168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[200:201] offset:25088                   // 000000006400: D89A6200 0000C804  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[106:107], v[138:139], v[176:179]// 000000006408: D3F300B0 0EC3156A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[40:43], v50, s[12:15], 0 offen       // 000000006410: E05C1000 80832832
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[104:105], v[152:153], 0// 000000006418: D3F300B4 0A033168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[202:203] offset:33792                   // 000000006420: D89A8400 0000CA04  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[106:107], v[154:155], v[180:183]// 000000006428: D3F300B4 0ED3356A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[108:109], v[136:137], 0// 000000006430: D3F300B8 0A03116C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[204:205] offset:27264                   // 000000006438: D89A6A80 0000CC04  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[110:111], v[138:139], v[184:187]// 000000006440: D3F300B8 0EE3156E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[44:47], v51, s[12:15], 0 offen       // 000000006448: E05C1000 80832C33
	s_add_u32 s12, s78, s12                                    // 000000006450: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000006454: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[108:109], v[152:153], 0// 000000006458: D3F300BC 0A03316C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[206:207] offset:35968                   // 000000006460: D89A8C80 0000CE04  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[110:111], v[154:155], v[188:191]// 000000006468: D3F300BC 0EF3356E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	s_waitcnt vmcnt(13)                                        // 000000006470: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[112:113], v[140:141], v[160:163]// 000000006474: D3F300A0 0E831970  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[114:115], v[142:143], v[160:163]// 00000000647C: D3F300A0 0E831D72  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[48:51], v48, s[12:15], 0 offen       // 000000006484: E05C1000 80833030
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[112:113], v[156:157], v[164:167]// 00000000648C: D3F300A4 0E933970  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[114:115], v[158:159], v[164:167]// 000000006494: D3F300A4 0E933D72  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[116:117], v[140:141], v[168:171]// 00000000649C: D3F300A8 0EA31974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[118:119], v[142:143], v[168:171]// 0000000064A4: D3F300A8 0EA31D76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[52:55], v49, s[12:15], 0 offen       // 0000000064AC: E05C1000 80833431
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[116:117], v[156:157], v[172:175]// 0000000064B4: D3F300AC 0EB33974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[118:119], v[158:159], v[172:175]// 0000000064BC: D3F300AC 0EB33D76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[120:121], v[140:141], v[176:179]// 0000000064C4: D3F300B0 0EC31978  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[122:123], v[142:143], v[176:179]// 0000000064CC: D3F300B0 0EC31D7A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[56:59], v50, s[12:15], 0 offen       // 0000000064D4: E05C1000 80833832
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[120:121], v[156:157], v[180:183]// 0000000064DC: D3F300B4 0ED33978  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[122:123], v[158:159], v[180:183]// 0000000064E4: D3F300B4 0ED33D7A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[124:125], v[140:141], v[184:187]// 0000000064EC: D3F300B8 0EE3197C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[126:127], v[142:143], v[184:187]// 0000000064F4: D3F300B8 0EE31D7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[60:63], v51, s[12:15], 0 offen       // 0000000064FC: E05C1000 80833C33
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[124:125], v[156:157], v[188:191]// 000000006504: D3F300BC 0EF3397C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[126:127], v[158:159], v[188:191]// 00000000650C: D3F300BC 0EF33D7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v24, v34 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000006514: 0A7044FA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 00000000651C: 7E720338  // Vector move
	v_pk_fma_f32 v[224:225], v[160:161], v[56:57], v[224:225]  // 000000006520: D3B040E0 1F8271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[226:227], v[162:163], v[56:57], v[226:227]  // 000000006528: D3B040E2 1F8A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[232:233], v[168:169], v[56:57], v[232:233]  // 000000006530: D3B040E8 1FA271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[234:235], v[170:171], v[56:57], v[234:235]  // 000000006538: D3B040EA 1FAA71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v34 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000006540: 0A7044FA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006548: 7E720338  // Vector move
	v_pk_fma_f32 v[240:241], v[176:177], v[56:57], v[240:241]  // 00000000654C: D3B040F0 1FC271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[242:243], v[178:179], v[56:57], v[242:243]  // 000000006554: D3B040F2 1FCA71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[248:249], v[184:185], v[56:57], v[248:249]  // 00000000655C: D3B040F8 1FE271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[250:251], v[186:187], v[56:57], v[250:251]  // 000000006564: D3B040FA 1FEA71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v35 row_newbcast:2 row_mask:0xf bank_mask:0xf// 00000000656C: 0A7046FA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006574: 7E720338  // Vector move
	v_pk_fma_f32 v[228:229], v[164:165], v[56:57], v[228:229]  // 000000006578: D3B040E4 1F9271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[230:231], v[166:167], v[56:57], v[230:231]  // 000000006580: D3B040E6 1F9A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[236:237], v[172:173], v[56:57], v[236:237]  // 000000006588: D3B040EC 1FB271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[238:239], v[174:175], v[56:57], v[238:239]  // 000000006590: D3B040EE 1FBA71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v35 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000006598: 0A7046FA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000065A0: 7E720338  // Vector move
	v_pk_fma_f32 v[244:245], v[180:181], v[56:57], v[244:245]  // 0000000065A4: D3B040F4 1FD271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[246:247], v[182:183], v[56:57], v[246:247]  // 0000000065AC: D3B040F6 1FDA71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[252:253], v[188:189], v[56:57], v[252:253]  // 0000000065B4: D3B040FC 1FF271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[254:255], v[190:191], v[56:57], v[254:255]  // 0000000065BC: D3B040FE 1FFA71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x200, s80                                  // 0000000065C4: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 0000000065CC: BF0A513C
	s_cselect_b32 s56, s56, 0                                  // 0000000065D0: 85388038
	s_cselect_b32 s78, s78, 0                                  // 0000000065D4: 854E804E
	s_cselect_b32 s79, s79, 0                                  // 0000000065D8: 854F804F
	s_add_u32 s12, s56, s12                                    // 0000000065DC: 800C0C38
	s_addc_u32 s13, 0, s13                                     // 0000000065E0: 820D0D80
	s_add_u32 s16, s79, s16                                    // 0000000065E4: 8010104F
	s_addc_u32 s17, 0, s17                                     // 0000000065E8: 82111180
	v_mov_b32_e32 v56, v25                                     // 0000000065EC: 7E700319  // Vector move
	v_mov_b32_e32 v57, v25                                     // 0000000065F0: 7E720319  // Vector move
	v_pk_mul_f32 v[224:225], v[56:57], v[224:225]              // 0000000065F4: D3B140E0 1803C138
	v_pk_mul_f32 v[226:227], v[56:57], v[226:227]              // 0000000065FC: D3B140E2 1803C538
	v_pk_mul_f32 v[232:233], v[56:57], v[232:233]              // 000000006604: D3B140E8 1803D138
	v_pk_mul_f32 v[234:235], v[56:57], v[234:235]              // 00000000660C: D3B140EA 1803D538
	v_pk_mul_f32 v[240:241], v[56:57], v[240:241]              // 000000006614: D3B140F0 1803E138
	v_pk_mul_f32 v[242:243], v[56:57], v[242:243]              // 00000000661C: D3B140F2 1803E538
	v_pk_mul_f32 v[248:249], v[56:57], v[248:249]              // 000000006624: D3B140F8 1803F138
	v_pk_mul_f32 v[250:251], v[56:57], v[250:251]              // 00000000662C: D3B140FA 1803F538
	v_mov_b32_e32 v56, v26                                     // 000000006634: 7E70031A  // Vector move
	v_mov_b32_e32 v57, v26                                     // 000000006638: 7E72031A  // Vector move
	v_pk_mul_f32 v[228:229], v[56:57], v[228:229]              // 00000000663C: D3B140E4 1803C938
	v_pk_mul_f32 v[230:231], v[56:57], v[230:231]              // 000000006644: D3B140E6 1803CD38
	v_pk_mul_f32 v[236:237], v[56:57], v[236:237]              // 00000000664C: D3B140EC 1803D938
	v_pk_mul_f32 v[238:239], v[56:57], v[238:239]              // 000000006654: D3B140EE 1803DD38
	v_pk_mul_f32 v[244:245], v[56:57], v[244:245]              // 00000000665C: D3B140F4 1803E938
	v_pk_mul_f32 v[246:247], v[56:57], v[246:247]              // 000000006664: D3B140F6 1803ED38
	v_pk_mul_f32 v[252:253], v[56:57], v[252:253]              // 00000000666C: D3B140FC 1803F938
	v_pk_mul_f32 v[254:255], v[56:57], v[254:255]              // 000000006674: D3B140FE 1803FD38
	v_cmp_u_f32_e64 s[48:49], v224, v224                       // 00000000667C: D0480030 0003C1E0
	v_add3_u32 v52, v224, v55, 1                               // 000000006684: D1FF0034 02066FE0
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 00000000668C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v225, v225                       // 000000006694: D0480030 0003C3E1
	v_add3_u32 v52, v225, v55, 1                               // 00000000669C: D1FF0034 02066FE1
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 0000000066A4: D1000039 00C26D34
	v_perm_b32 v224, v57, v56, s52                             // 0000000066AC: D1ED00E0 00D27139
	v_cmp_u_f32_e64 s[48:49], v226, v226                       // 0000000066B4: D0480030 0003C5E2
	v_add3_u32 v52, v226, v55, 1                               // 0000000066BC: D1FF0034 02066FE2
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000066C4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v227, v227                       // 0000000066CC: D0480030 0003C7E3
	v_add3_u32 v52, v227, v55, 1                               // 0000000066D4: D1FF0034 02066FE3
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 0000000066DC: D1000039 00C26D34
	v_perm_b32 v225, v57, v56, s52                             // 0000000066E4: D1ED00E1 00D27139
	v_cmp_u_f32_e64 s[48:49], v228, v228                       // 0000000066EC: D0480030 0003C9E4
	v_add3_u32 v52, v228, v55, 1                               // 0000000066F4: D1FF0034 02066FE4
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000066FC: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v229, v229                       // 000000006704: D0480030 0003CBE5
	v_add3_u32 v52, v229, v55, 1                               // 00000000670C: D1FF0034 02066FE5
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000006714: D1000039 00C26D34
	v_perm_b32 v226, v57, v56, s52                             // 00000000671C: D1ED00E2 00D27139
	v_cmp_u_f32_e64 s[48:49], v230, v230                       // 000000006724: D0480030 0003CDE6
	v_add3_u32 v52, v230, v55, 1                               // 00000000672C: D1FF0034 02066FE6
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000006734: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v231, v231                       // 00000000673C: D0480030 0003CFE7
	v_add3_u32 v52, v231, v55, 1                               // 000000006744: D1FF0034 02066FE7
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 00000000674C: D1000039 00C26D34
	v_perm_b32 v227, v57, v56, s52                             // 000000006754: D1ED00E3 00D27139
	v_cmp_u_f32_e64 s[48:49], v232, v232                       // 00000000675C: D0480030 0003D1E8
	v_add3_u32 v52, v232, v55, 1                               // 000000006764: D1FF0034 02066FE8
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 00000000676C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v233, v233                       // 000000006774: D0480030 0003D3E9
	v_add3_u32 v52, v233, v55, 1                               // 00000000677C: D1FF0034 02066FE9
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000006784: D1000039 00C26D34
	v_perm_b32 v228, v57, v56, s52                             // 00000000678C: D1ED00E4 00D27139
	v_cmp_u_f32_e64 s[48:49], v234, v234                       // 000000006794: D0480030 0003D5EA
	v_add3_u32 v52, v234, v55, 1                               // 00000000679C: D1FF0034 02066FEA
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000067A4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v235, v235                       // 0000000067AC: D0480030 0003D7EB
	v_add3_u32 v52, v235, v55, 1                               // 0000000067B4: D1FF0034 02066FEB
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 0000000067BC: D1000039 00C26D34
	v_perm_b32 v229, v57, v56, s52                             // 0000000067C4: D1ED00E5 00D27139
	v_cmp_u_f32_e64 s[48:49], v236, v236                       // 0000000067CC: D0480030 0003D9EC
	v_add3_u32 v52, v236, v55, 1                               // 0000000067D4: D1FF0034 02066FEC
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000067DC: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v237, v237                       // 0000000067E4: D0480030 0003DBED
	v_add3_u32 v52, v237, v55, 1                               // 0000000067EC: D1FF0034 02066FED
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 0000000067F4: D1000039 00C26D34
	v_perm_b32 v230, v57, v56, s52                             // 0000000067FC: D1ED00E6 00D27139
	v_cmp_u_f32_e64 s[48:49], v238, v238                       // 000000006804: D0480030 0003DDEE
	v_add3_u32 v52, v238, v55, 1                               // 00000000680C: D1FF0034 02066FEE
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000006814: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v239, v239                       // 00000000681C: D0480030 0003DFEF
	v_add3_u32 v52, v239, v55, 1                               // 000000006824: D1FF0034 02066FEF
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 00000000682C: D1000039 00C26D34
	v_perm_b32 v231, v57, v56, s52                             // 000000006834: D1ED00E7 00D27139
	v_cmp_u_f32_e64 s[48:49], v240, v240                       // 00000000683C: D0480030 0003E1F0
	v_add3_u32 v52, v240, v55, 1                               // 000000006844: D1FF0034 02066FF0
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 00000000684C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v241, v241                       // 000000006854: D0480030 0003E3F1
	v_add3_u32 v52, v241, v55, 1                               // 00000000685C: D1FF0034 02066FF1
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000006864: D1000039 00C26D34
	v_perm_b32 v232, v57, v56, s52                             // 00000000686C: D1ED00E8 00D27139
	v_cmp_u_f32_e64 s[48:49], v242, v242                       // 000000006874: D0480030 0003E5F2
	v_add3_u32 v52, v242, v55, 1                               // 00000000687C: D1FF0034 02066FF2
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000006884: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v243, v243                       // 00000000688C: D0480030 0003E7F3
	v_add3_u32 v52, v243, v55, 1                               // 000000006894: D1FF0034 02066FF3
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 00000000689C: D1000039 00C26D34
	v_perm_b32 v233, v57, v56, s52                             // 0000000068A4: D1ED00E9 00D27139
	v_cmp_u_f32_e64 s[48:49], v244, v244                       // 0000000068AC: D0480030 0003E9F4
	v_add3_u32 v52, v244, v55, 1                               // 0000000068B4: D1FF0034 02066FF4
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000068BC: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v245, v245                       // 0000000068C4: D0480030 0003EBF5
	v_add3_u32 v52, v245, v55, 1                               // 0000000068CC: D1FF0034 02066FF5
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 0000000068D4: D1000039 00C26D34
	v_perm_b32 v234, v57, v56, s52                             // 0000000068DC: D1ED00EA 00D27139
	v_cmp_u_f32_e64 s[48:49], v246, v246                       // 0000000068E4: D0480030 0003EDF6
	v_add3_u32 v52, v246, v55, 1                               // 0000000068EC: D1FF0034 02066FF6
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000068F4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v247, v247                       // 0000000068FC: D0480030 0003EFF7
	v_add3_u32 v52, v247, v55, 1                               // 000000006904: D1FF0034 02066FF7
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 00000000690C: D1000039 00C26D34
	v_perm_b32 v235, v57, v56, s52                             // 000000006914: D1ED00EB 00D27139
	v_cmp_u_f32_e64 s[48:49], v248, v248                       // 00000000691C: D0480030 0003F1F8
	v_add3_u32 v52, v248, v55, 1                               // 000000006924: D1FF0034 02066FF8
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 00000000692C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v249, v249                       // 000000006934: D0480030 0003F3F9
	v_add3_u32 v52, v249, v55, 1                               // 00000000693C: D1FF0034 02066FF9
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000006944: D1000039 00C26D34
	v_perm_b32 v236, v57, v56, s52                             // 00000000694C: D1ED00EC 00D27139
	v_cmp_u_f32_e64 s[48:49], v250, v250                       // 000000006954: D0480030 0003F5FA
	v_add3_u32 v52, v250, v55, 1                               // 00000000695C: D1FF0034 02066FFA
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000006964: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v251, v251                       // 00000000696C: D0480030 0003F7FB
	v_add3_u32 v52, v251, v55, 1                               // 000000006974: D1FF0034 02066FFB
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 00000000697C: D1000039 00C26D34
	v_perm_b32 v237, v57, v56, s52                             // 000000006984: D1ED00ED 00D27139
	v_cmp_u_f32_e64 s[48:49], v252, v252                       // 00000000698C: D0480030 0003F9FC
	v_add3_u32 v52, v252, v55, 1                               // 000000006994: D1FF0034 02066FFC
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 00000000699C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v253, v253                       // 0000000069A4: D0480030 0003FBFD
	v_add3_u32 v52, v253, v55, 1                               // 0000000069AC: D1FF0034 02066FFD
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 0000000069B4: D1000039 00C26D34
	v_perm_b32 v238, v57, v56, s52                             // 0000000069BC: D1ED00EE 00D27139
	v_cmp_u_f32_e64 s[48:49], v254, v254                       // 0000000069C4: D0480030 0003FDFE
	v_add3_u32 v52, v254, v55, 1                               // 0000000069CC: D1FF0034 02066FFE
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000069D4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v255, v255                       // 0000000069DC: D0480030 0003FFFF
	v_add3_u32 v52, v255, v55, 1                               // 0000000069E4: D1FF0034 02066FFF
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 0000000069EC: D1000039 00C26D34
	v_perm_b32 v239, v57, v56, s52                             // 0000000069F4: D1ED00EF 00D27139
	s_cmp_ge_u32 s80, 0x200                                    // 0000000069FC: BF09FF50 00000200
	s_cselect_b32 s59, 0x200, s59                              // 000000006A04: 853B3BFF 00000200
	s_setvskip s20, 0                                          // 000000006A0C: BF108014
	global_atomic_pk_add_bf16 v80, v64, s[8:9]                 // 000000006A10: DD488000 00084050
	s_setvskip 0, 0                                            // 000000006A18: BF108080
	s_setvskip s20, 0                                          // 000000006A1C: BF108014
	global_atomic_pk_add_bf16 v80, v65, s[8:9] offset:256      // 000000006A20: DD488100 00084150
	s_setvskip 0, 0                                            // 000000006A28: BF108080
	s_setvskip s20, 1                                          // 000000006A2C: BF108114
	global_atomic_pk_add_bf16 v82, v66, s[8:9]                 // 000000006A30: DD488000 00084252
	s_setvskip 0, 0                                            // 000000006A38: BF108080
	s_setvskip s20, 1                                          // 000000006A3C: BF108114
	global_atomic_pk_add_bf16 v82, v67, s[8:9] offset:256      // 000000006A40: DD488100 00084352
	s_setvskip 0, 0                                            // 000000006A48: BF108080
	s_setvskip s20, 2                                          // 000000006A4C: BF108214
	global_atomic_pk_add_bf16 v84, v68, s[8:9]                 // 000000006A50: DD488000 00084454
	s_setvskip 0, 0                                            // 000000006A58: BF108080
	s_setvskip s20, 2                                          // 000000006A5C: BF108214
	global_atomic_pk_add_bf16 v84, v69, s[8:9] offset:256      // 000000006A60: DD488100 00084554
	s_setvskip 0, 0                                            // 000000006A68: BF108080
	s_setvskip s20, 3                                          // 000000006A6C: BF108314
	global_atomic_pk_add_bf16 v86, v70, s[8:9]                 // 000000006A70: DD488000 00084656
	s_setvskip 0, 0                                            // 000000006A78: BF108080
	s_setvskip s20, 3                                          // 000000006A7C: BF108314
	global_atomic_pk_add_bf16 v86, v71, s[8:9] offset:256      // 000000006A80: DD488100 00084756
	s_setvskip 0, 0                                            // 000000006A88: BF108080
	s_setvskip s20, 4                                          // 000000006A8C: BF108414
	global_atomic_pk_add_bf16 v88, v72, s[8:9]                 // 000000006A90: DD488000 00084858
	s_setvskip 0, 0                                            // 000000006A98: BF108080
	s_setvskip s20, 4                                          // 000000006A9C: BF108414
	global_atomic_pk_add_bf16 v88, v73, s[8:9] offset:256      // 000000006AA0: DD488100 00084958
	s_setvskip 0, 0                                            // 000000006AA8: BF108080
	s_setvskip s20, 5                                          // 000000006AAC: BF108514
	global_atomic_pk_add_bf16 v90, v74, s[8:9]                 // 000000006AB0: DD488000 00084A5A
	s_setvskip 0, 0                                            // 000000006AB8: BF108080
	s_setvskip s20, 5                                          // 000000006ABC: BF108514
	global_atomic_pk_add_bf16 v90, v75, s[8:9] offset:256      // 000000006AC0: DD488100 00084B5A
	s_setvskip 0, 0                                            // 000000006AC8: BF108080
	s_setvskip s20, 6                                          // 000000006ACC: BF108614
	global_atomic_pk_add_bf16 v92, v76, s[8:9]                 // 000000006AD0: DD488000 00084C5C
	s_setvskip 0, 0                                            // 000000006AD8: BF108080
	s_setvskip s20, 6                                          // 000000006ADC: BF108614
	global_atomic_pk_add_bf16 v92, v77, s[8:9] offset:256      // 000000006AE0: DD488100 00084D5C
	s_setvskip 0, 0                                            // 000000006AE8: BF108080
	s_setvskip s20, 7                                          // 000000006AEC: BF108714
	global_atomic_pk_add_bf16 v94, v78, s[8:9]                 // 000000006AF0: DD488000 00084E5E
	s_setvskip 0, 0                                            // 000000006AF8: BF108080
	s_setvskip s20, 7                                          // 000000006AFC: BF108714
	global_atomic_pk_add_bf16 v94, v79, s[8:9] offset:256      // 000000006B00: DD488100 00084F5E
	s_setvskip 0, 0                                            // 000000006B08: BF108080
	s_add_u32 s8, s59, s8                                      // 000000006B0C: 8008083B
	s_addc_u32 s9, 0, s9                                       // 000000006B10: 82090980
	s_addk_i32 s80, 0x100                                      // 000000006B14: B7500100
	s_cmp_lt_i32 s80, s81                                      // 000000006B18: BF045150
	s_cbranch_scc0 label_0F49                                  // 000000006B1C: BF840001  // Branch if SCC==0 (condition false)
	s_branch label_0A30                                        // 000000006B20: BF82FAE7  // Unconditional branch


// --- LABEL (potential loop target) ---
0000000000006b24 <label_0F49>:
	s_nop 0                                                    // 000000006B24: BF800000
	s_nop 0                                                    // 000000006B28: BF800000
	s_branch label_1BF7                                        // 000000006B2C: BF820CAB  // Unconditional branch


// --- LABEL (potential loop target) ---
0000000000006b30 <label_0F4C>:
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(8) lgkmcnt(0)                              // 000000006B30: BF8C0078  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 000000006B34: BF8A0000  // Workgroup barrier synchronization
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[0:1], v[192:193], 0// 000000006B38: D3F300A0 0A038100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[64:67], v44, s[92:95], 0 offen       // 000000006B40: E05C1000 8097402C
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[2:3], v[194:195], v[160:163]// 000000006B48: D3F300A0 0E838502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[4:5], v[196:197], v[160:163]// 000000006B50: D3F300A0 0E838904  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v36, s[20:23], 0 offen lds               // 000000006B58: E0511000 80050024
	s_add_u32 m0, 0x100, s51                                   // 000000006B60: 807C33FF 00000100
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[6:7], v[198:199], v[160:163]// 000000006B68: D3F300A0 0E838D06  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[0:1], v[208:209], 0// 000000006B70: D3F300A4 0A03A100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[68:71], v44, s[92:95], 0 offen offset:1024// 000000006B78: E05C1400 8097442C
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[2:3], v[210:211], v[164:167]// 000000006B80: D3F300A4 0E93A502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[4:5], v[212:213], v[164:167]// 000000006B88: D3F300A4 0E93A904  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v37, s[20:23], 0 offen lds               // 000000006B90: E0511000 80050025
	s_add_u32 m0, 0x200, s51                                   // 000000006B98: 807C33FF 00000200
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[6:7], v[214:215], v[164:167]// 000000006BA0: D3F300A4 0E93AD06  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[16:17], v[192:193], 0// 000000006BA8: D3F300A8 0A038110  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[80:83], v45, s[92:95], 0 offen       // 000000006BB0: E05C1000 8097502D
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[18:19], v[194:195], v[168:171]// 000000006BB8: D3F300A8 0EA38512  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[20:21], v[196:197], v[168:171]// 000000006BC0: D3F300A8 0EA38914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v38, s[20:23], 0 offen lds               // 000000006BC8: E0511000 80050026
	s_add_u32 m0, 0x300, s51                                   // 000000006BD0: 807C33FF 00000300
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[22:23], v[198:199], v[168:171]// 000000006BD8: D3F300A8 0EA38D16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[16:17], v[208:209], 0// 000000006BE0: D3F300AC 0A03A110  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[84:87], v45, s[92:95], 0 offen offset:1024// 000000006BE8: E05C1400 8097542D
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[18:19], v[210:211], v[172:175]// 000000006BF0: D3F300AC 0EB3A512  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[20:21], v[212:213], v[172:175]// 000000006BF8: D3F300AC 0EB3A914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v39, s[20:23], 0 offen lds               // 000000006C00: E0511000 80050027
	s_add_u32 m0, 0x400, s51                                   // 000000006C08: 807C33FF 00000400
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[22:23], v[214:215], v[172:175]// 000000006C10: D3F300AC 0EB3AD16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[32:33], v[192:193], 0// 000000006C18: D3F300B0 0A038120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[96:99], v46, s[92:95], 0 offen       // 000000006C20: E05C1000 8097602E
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[34:35], v[194:195], v[176:179]// 000000006C28: D3F300B0 0EC38522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[36:37], v[196:197], v[176:179]// 000000006C30: D3F300B0 0EC38924  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v40, s[20:23], 0 offen lds               // 000000006C38: E0511000 80050028
	s_add_u32 m0, 0x500, s51                                   // 000000006C40: 807C33FF 00000500
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[38:39], v[198:199], v[176:179]// 000000006C48: D3F300B0 0EC38D26  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[32:33], v[208:209], 0// 000000006C50: D3F300B4 0A03A120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[100:103], v46, s[92:95], 0 offen offset:1024// 000000006C58: E05C1400 8097642E
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[34:35], v[210:211], v[180:183]// 000000006C60: D3F300B4 0ED3A522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[36:37], v[212:213], v[180:183]// 000000006C68: D3F300B4 0ED3A924  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v41, s[20:23], 0 offen lds               // 000000006C70: E0511000 80050029
	s_add_u32 m0, 0x600, s51                                   // 000000006C78: 807C33FF 00000600
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[38:39], v[214:215], v[180:183]// 000000006C80: D3F300B4 0ED3AD26  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[48:49], v[192:193], 0// 000000006C88: D3F300B8 0A038130  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[112:115], v47, s[92:95], 0 offen     // 000000006C90: E05C1000 8097702F
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[50:51], v[194:195], v[184:187]// 000000006C98: D3F300B8 0EE38532  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[52:53], v[196:197], v[184:187]// 000000006CA0: D3F300B8 0EE38934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v42, s[20:23], 0 offen lds               // 000000006CA8: E0511000 8005002A
	s_add_u32 m0, 0x700, s51                                   // 000000006CB0: 807C33FF 00000700
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[54:55], v[198:199], v[184:187]// 000000006CB8: D3F300B8 0EE38D36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[48:49], v[208:209], 0// 000000006CC0: D3F300BC 0A03A130  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[116:119], v47, s[92:95], 0 offen offset:1024// 000000006CC8: E05C1400 8097742F
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[50:51], v[210:211], v[188:191]// 000000006CD0: D3F300BC 0EF3A532  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[52:53], v[212:213], v[188:191]// 000000006CD8: D3F300BC 0EF3A934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v43, s[20:23], 0 offen lds               // 000000006CE0: E0511000 8005002B
	s_add_u32 m0, s51, s76                                     // 000000006CE8: 807C4C33
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[54:55], v[214:215], v[188:191]// 000000006CEC: D3F300BC 0EF3AD36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v7, s[28:31], 0 offen lds                // 000000006CF4: E0511000 80070007
	s_add_u32 m0, 0, s50                                       // 000000006CFC: 807C3280
	buffer_load_dword v24, v13, s[32:35], 0 offen              // 000000006D00: E0501000 8008180D
	v_mul_f32_dpp v56, v23, v15 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000006D08: 0A701EFA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006D10: 7E720338  // Vector move
	v_pk_fma_f32 v[128:129], v[160:161], v[56:57], v[128:129]  // 000000006D14: D3B04080 1E0271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[130:131], v[162:163], v[56:57], v[130:131]  // 000000006D1C: D3B04082 1E0A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[136:137], v[168:169], v[56:57], v[136:137]  // 000000006D24: D3B04088 1E2271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[138:139], v[170:171], v[56:57], v[138:139]  // 000000006D2C: D3B0408A 1E2A71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v15 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000006D34: 0A701EFA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006D3C: 7E720338  // Vector move
	v_pk_fma_f32 v[144:145], v[176:177], v[56:57], v[144:145]  // 000000006D40: D3B04090 1E4271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[146:147], v[178:179], v[56:57], v[146:147]  // 000000006D48: D3B04092 1E4A71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[152:153], v[184:185], v[56:57], v[152:153]  // 000000006D50: D3B04098 1E6271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[154:155], v[186:187], v[56:57], v[154:155]  // 000000006D58: D3B0409A 1E6A71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v16 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000006D60: 0A7020FA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006D68: 7E720338  // Vector move
	v_pk_fma_f32 v[132:133], v[164:165], v[56:57], v[132:133]  // 000000006D6C: D3B04084 1E1271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[134:135], v[166:167], v[56:57], v[134:135]  // 000000006D74: D3B04086 1E1A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[140:141], v[172:173], v[56:57], v[140:141]  // 000000006D7C: D3B0408C 1E3271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[142:143], v[174:175], v[56:57], v[142:143]  // 000000006D84: D3B0408E 1E3A71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v16 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000006D8C: 0A7020FA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006D94: 7E720338  // Vector move
	v_pk_fma_f32 v[148:149], v[180:181], v[56:57], v[148:149]  // 000000006D98: D3B04094 1E5271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[150:151], v[182:183], v[56:57], v[150:151]  // 000000006DA0: D3B04096 1E5A71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[156:157], v[188:189], v[56:57], v[156:157]  // 000000006DA8: D3B0409C 1E7271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[158:159], v[190:191], v[56:57], v[158:159]  // 000000006DB0: D3B0409E 1E7A71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_waitcnt vmcnt(22)                                        // 000000006DB8: BF8C4F76  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[8:9], v[200:201], 0// 000000006DBC: D3F300A0 0A039108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[72:75], v44, s[92:95], 0 offen offset:2048// 000000006DC4: E05C1800 8097482C
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[10:11], v[202:203], v[160:163]// 000000006DCC: D3F300A0 0E83950A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[12:13], v[204:205], v[160:163]// 000000006DD4: D3F300A0 0E83990C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[14:15], v[206:207], v[160:163]// 000000006DDC: D3F300A0 0E839D0E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[8:9], v[216:217], 0// 000000006DE4: D3F300A4 0A03B108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[76:79], v44, s[92:95], 0 offen offset:3072// 000000006DEC: E05C1C00 80974C2C
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[10:11], v[218:219], v[164:167]// 000000006DF4: D3F300A4 0E93B50A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[12:13], v[220:221], v[164:167]// 000000006DFC: D3F300A4 0E93B90C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[14:15], v[222:223], v[164:167]// 000000006E04: D3F300A4 0E93BD0E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[24:25], v[200:201], 0// 000000006E0C: D3F300A8 0A039118  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[88:91], v45, s[92:95], 0 offen offset:2048// 000000006E14: E05C1800 8097582D
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[26:27], v[202:203], v[168:171]// 000000006E1C: D3F300A8 0EA3951A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[28:29], v[204:205], v[168:171]// 000000006E24: D3F300A8 0EA3991C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[30:31], v[206:207], v[168:171]// 000000006E2C: D3F300A8 0EA39D1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[24:25], v[216:217], 0// 000000006E34: D3F300AC 0A03B118  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[92:95], v45, s[92:95], 0 offen offset:3072// 000000006E3C: E05C1C00 80975C2D
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[26:27], v[218:219], v[172:175]// 000000006E44: D3F300AC 0EB3B51A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[28:29], v[220:221], v[172:175]// 000000006E4C: D3F300AC 0EB3B91C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[30:31], v[222:223], v[172:175]// 000000006E54: D3F300AC 0EB3BD1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	s_waitcnt vmcnt(22)                                        // 000000006E5C: BF8C4F76  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[40:41], v[200:201], 0// 000000006E60: D3F300B0 0A039128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[104:107], v46, s[92:95], 0 offen offset:2048// 000000006E68: E05C1800 8097682E
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[42:43], v[202:203], v[176:179]// 000000006E70: D3F300B0 0EC3952A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[44:45], v[204:205], v[176:179]// 000000006E78: D3F300B0 0EC3992C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[46:47], v[206:207], v[176:179]// 000000006E80: D3F300B0 0EC39D2E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[40:41], v[216:217], 0// 000000006E88: D3F300B4 0A03B128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[108:111], v46, s[92:95], 0 offen offset:3072// 000000006E90: E05C1C00 80976C2E
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[42:43], v[218:219], v[180:183]// 000000006E98: D3F300B4 0ED3B52A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[44:45], v[220:221], v[180:183]// 000000006EA0: D3F300B4 0ED3B92C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[46:47], v[222:223], v[180:183]// 000000006EA8: D3F300B4 0ED3BD2E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[56:57], v[200:201], 0// 000000006EB0: D3F300B8 0A039138  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[120:123], v47, s[92:95], 0 offen offset:2048// 000000006EB8: E05C1800 8097782F
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[58:59], v[202:203], v[184:187]// 000000006EC0: D3F300B8 0EE3953A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[60:61], v[204:205], v[184:187]// 000000006EC8: D3F300B8 0EE3993C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[62:63], v[206:207], v[184:187]// 000000006ED0: D3F300B8 0EE39D3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[56:57], v[216:217], 0// 000000006ED8: D3F300BC 0A03B138  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[124:127], v47, s[92:95], 0 offen offset:3072// 000000006EE0: E05C1C00 80977C2F
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[58:59], v[218:219], v[188:191]// 000000006EE8: D3F300BC 0EF3B53A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[60:61], v[220:221], v[188:191]// 000000006EF0: D3F300BC 0EF3B93C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[62:63], v[222:223], v[188:191]// 000000006EF8: D3F300BC 0EF3BD3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v23, v17 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000006F00: 0A7022FA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006F08: 7E720338  // Vector move
	v_pk_fma_f32 v[128:129], v[160:161], v[56:57], v[128:129]  // 000000006F0C: D3B04080 1E0271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[130:131], v[162:163], v[56:57], v[130:131]  // 000000006F14: D3B04082 1E0A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[136:137], v[168:169], v[56:57], v[136:137]  // 000000006F1C: D3B04088 1E2271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[138:139], v[170:171], v[56:57], v[138:139]  // 000000006F24: D3B0408A 1E2A71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v17 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000006F2C: 0A7022FA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006F34: 7E720338  // Vector move
	v_pk_fma_f32 v[144:145], v[176:177], v[56:57], v[144:145]  // 000000006F38: D3B04090 1E4271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[146:147], v[178:179], v[56:57], v[146:147]  // 000000006F40: D3B04092 1E4A71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[152:153], v[184:185], v[56:57], v[152:153]  // 000000006F48: D3B04098 1E6271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[154:155], v[186:187], v[56:57], v[154:155]  // 000000006F50: D3B0409A 1E6A71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v18 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000006F58: 0A7024FA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006F60: 7E720338  // Vector move
	v_pk_fma_f32 v[132:133], v[164:165], v[56:57], v[132:133]  // 000000006F64: D3B04084 1E1271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[134:135], v[166:167], v[56:57], v[134:135]  // 000000006F6C: D3B04086 1E1A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[140:141], v[172:173], v[56:57], v[140:141]  // 000000006F74: D3B0408C 1E3271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[142:143], v[174:175], v[56:57], v[142:143]  // 000000006F7C: D3B0408E 1E3A71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v18 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000006F84: 0A7024FA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000006F8C: 7E720338  // Vector move
	v_pk_fma_f32 v[148:149], v[180:181], v[56:57], v[148:149]  // 000000006F90: D3B04094 1E5271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[150:151], v[182:183], v[56:57], v[150:151]  // 000000006F98: D3B04096 1E5A71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[156:157], v[188:189], v[56:57], v[156:157]  // 000000006FA0: D3B0409C 1E7271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[158:159], v[190:191], v[56:57], v[158:159]  // 000000006FA8: D3B0409E 1E7A71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x100, s80                                  // 000000006FB0: 803C50FF 00000100
	s_cmp_lt_u32 s60, s81                                      // 000000006FB8: BF0A513C
	s_cselect_b32 s4, s4, 0                                    // 000000006FBC: 85048004
	s_add_u32 s32, s4, s32                                     // 000000006FC0: 80202004
	s_addc_u32 s33, 0, s33                                     // 000000006FC4: 82212180
	s_waitcnt vmcnt(8)                                         // 000000006FC8: BF8C0F78  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 000000006FCC: BF8A0000  // Workgroup barrier synchronization
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[64:65], v[192:193], 0// 000000006FD0: D3F30060 0A038140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[0:3], v44, s[24:27], 0 offen         // 000000006FD8: E05C1000 8086002C
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[66:67], v[194:195], v[96:99]// 000000006FE0: D3F30060 0D838542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[68:69], v[196:197], v[96:99]// 000000006FE8: D3F30060 0D838944  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v23, v11, s[32:35], 0 offen              // 000000006FF0: E0501000 8008170B
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[70:71], v[198:199], v[96:99]// 000000006FF8: D3F30060 0D838D46  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[64:65], v[208:209], 0// 000000007000: D3F30064 0A03A140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[4:7], v44, s[24:27], 0 offen offset:1024// 000000007008: E05C1400 8086042C
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[66:67], v[210:211], v[100:103]// 000000007010: D3F30064 0D93A542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[68:69], v[212:213], v[100:103]// 000000007018: D3F30064 0D93A944  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[70:71], v[214:215], v[100:103]// 000000007020: D3F30064 0D93AD46  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[80:81], v[192:193], 0// 000000007028: D3F30068 0A038150  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[16:19], v45, s[24:27], 0 offen       // 000000007030: E05C1000 8086102D
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[82:83], v[194:195], v[104:107]// 000000007038: D3F30068 0DA38552  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[84:85], v[196:197], v[104:107]// 000000007040: D3F30068 0DA38954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[86:87], v[198:199], v[104:107]// 000000007048: D3F30068 0DA38D56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[80:81], v[208:209], 0// 000000007050: D3F3006C 0A03A150  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[20:23], v45, s[24:27], 0 offen offset:1024// 000000007058: E05C1400 8086142D
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[82:83], v[210:211], v[108:111]// 000000007060: D3F3006C 0DB3A552  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[84:85], v[212:213], v[108:111]// 000000007068: D3F3006C 0DB3A954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[86:87], v[214:215], v[108:111]// 000000007070: D3F3006C 0DB3AD56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[96:97], v[192:193], 0// 000000007078: D3F30070 0A038160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[32:35], v46, s[24:27], 0 offen       // 000000007080: E05C1000 8086202E
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[98:99], v[194:195], v[112:115]// 000000007088: D3F30070 0DC38562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[100:101], v[196:197], v[112:115]// 000000007090: D3F30070 0DC38964  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[102:103], v[198:199], v[112:115]// 000000007098: D3F30070 0DC38D66  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[96:97], v[208:209], 0// 0000000070A0: D3F30074 0A03A160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[36:39], v46, s[24:27], 0 offen offset:1024// 0000000070A8: E05C1400 8086242E
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[98:99], v[210:211], v[116:119]// 0000000070B0: D3F30074 0DD3A562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[100:101], v[212:213], v[116:119]// 0000000070B8: D3F30074 0DD3A964  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[102:103], v[214:215], v[116:119]// 0000000070C0: D3F30074 0DD3AD66  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[112:113], v[192:193], 0// 0000000070C8: D3F30078 0A038170  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[48:51], v47, s[24:27], 0 offen       // 0000000070D0: E05C1000 8086302F
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[114:115], v[194:195], v[120:123]// 0000000070D8: D3F30078 0DE38572  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[116:117], v[196:197], v[120:123]// 0000000070E0: D3F30078 0DE38974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[118:119], v[198:199], v[120:123]// 0000000070E8: D3F30078 0DE38D76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[112:113], v[208:209], 0// 0000000070F0: D3F3007C 0A03A170  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[52:55], v47, s[24:27], 0 offen offset:1024// 0000000070F8: E05C1400 8086342F
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[114:115], v[210:211], v[124:127]// 000000007100: D3F3007C 0DF3A572  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[116:117], v[212:213], v[124:127]// 000000007108: D3F3007C 0DF3A974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[118:119], v[214:215], v[124:127]// 000000007110: D3F3007C 0DF3AD76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v24, v15 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000007118: 0A701EFA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007120: 7E720338  // Vector move
	v_pk_fma_f32 v[64:65], v[96:97], v[56:57], v[64:65]        // 000000007124: D3B04040 1D027160  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[66:67], v[98:99], v[56:57], v[66:67]        // 00000000712C: D3B04042 1D0A7162  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[72:73], v[104:105], v[56:57], v[72:73]      // 000000007134: D3B04048 1D227168  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[74:75], v[106:107], v[56:57], v[74:75]      // 00000000713C: D3B0404A 1D2A716A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v15 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000007144: 0A701EFA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 00000000714C: 7E720338  // Vector move
	v_pk_fma_f32 v[80:81], v[112:113], v[56:57], v[80:81]      // 000000007150: D3B04050 1D427170  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[82:83], v[114:115], v[56:57], v[82:83]      // 000000007158: D3B04052 1D4A7172  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[88:89], v[120:121], v[56:57], v[88:89]      // 000000007160: D3B04058 1D627178  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[90:91], v[122:123], v[56:57], v[90:91]      // 000000007168: D3B0405A 1D6A717A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v16 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000007170: 0A7020FA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007178: 7E720338  // Vector move
	v_pk_fma_f32 v[68:69], v[100:101], v[56:57], v[68:69]      // 00000000717C: D3B04044 1D127164  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[70:71], v[102:103], v[56:57], v[70:71]      // 000000007184: D3B04046 1D1A7166  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[76:77], v[108:109], v[56:57], v[76:77]      // 00000000718C: D3B0404C 1D32716C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[78:79], v[110:111], v[56:57], v[78:79]      // 000000007194: D3B0404E 1D3A716E  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v16 row_newbcast:1 row_mask:0xf bank_mask:0xf// 00000000719C: 0A7020FA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000071A4: 7E720338  // Vector move
	v_pk_fma_f32 v[84:85], v[116:117], v[56:57], v[84:85]      // 0000000071A8: D3B04054 1D527174  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[86:87], v[118:119], v[56:57], v[86:87]      // 0000000071B0: D3B04056 1D5A7176  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[92:93], v[124:125], v[56:57], v[92:93]      // 0000000071B8: D3B0405C 1D72717C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[94:95], v[126:127], v[56:57], v[94:95]      // 0000000071C0: D3B0405E 1D7A717E  // Packed FMA: 2x FP32 fused multiply-add
	s_waitcnt vmcnt(13)                                        // 0000000071C8: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[72:73], v[200:201], 0// 0000000071CC: D3F30060 0A039148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[8:11], v44, s[24:27], 0 offen offset:2048// 0000000071D4: E05C1800 8086082C
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[74:75], v[202:203], v[96:99]// 0000000071DC: D3F30060 0D83954A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[76:77], v[204:205], v[96:99]// 0000000071E4: D3F30060 0D83994C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[224:227], v2 offset:9344                    // 0000000071EC: D9FE2480 E0000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v19, v3 offset:17664                           // 0000000071F4: D86C4500 13000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[78:79], v[206:207], v[96:99]// 0000000071FC: D3F30060 0D839D4E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[72:73], v[216:217], 0// 000000007204: D3F30064 0A03B148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[12:15], v44, s[24:27], 0 offen offset:3072// 00000000720C: E05C1C00 80860C2C
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[74:75], v[218:219], v[100:103]// 000000007214: D3F30064 0D93B54A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[76:77], v[220:221], v[100:103]// 00000000721C: D3F30064 0D93B94C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[228:231], v2 offset:9408                    // 000000007224: D9FE24C0 E4000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v20, v3 offset:17920                           // 00000000722C: D86C4600 14000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[78:79], v[222:223], v[100:103]// 000000007234: D3F30064 0D93BD4E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[88:89], v[200:201], 0// 00000000723C: D3F30068 0A039158  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[24:27], v45, s[24:27], 0 offen offset:2048// 000000007244: E05C1800 8086182D
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[90:91], v[202:203], v[104:107]// 00000000724C: D3F30068 0DA3955A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[92:93], v[204:205], v[104:107]// 000000007254: D3F30068 0DA3995C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[232:235], v2 offset:9472                    // 00000000725C: D9FE2500 E8000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v21, v3 offset:18176                           // 000000007264: D86C4700 15000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[94:95], v[206:207], v[104:107]// 00000000726C: D3F30068 0DA39D5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[88:89], v[216:217], 0// 000000007274: D3F3006C 0A03B158  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[28:31], v45, s[24:27], 0 offen offset:3072// 00000000727C: E05C1C00 80861C2D
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[90:91], v[218:219], v[108:111]// 000000007284: D3F3006C 0DB3B55A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[92:93], v[220:221], v[108:111]// 00000000728C: D3F3006C 0DB3B95C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[236:239], v2 offset:9536                    // 000000007294: D9FE2540 EC000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v22, v3 offset:18432                           // 00000000729C: D86C4800 16000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[94:95], v[222:223], v[108:111]// 0000000072A4: D3F3006C 0DB3BD5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	s_waitcnt vmcnt(13)                                        // 0000000072AC: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[104:105], v[200:201], 0// 0000000072B0: D3F30070 0A039168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[40:43], v46, s[24:27], 0 offen offset:2048// 0000000072B8: E05C1800 8086282E
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[106:107], v[202:203], v[112:115]// 0000000072C0: D3F30070 0DC3956A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[108:109], v[204:205], v[112:115]// 0000000072C8: D3F30070 0DC3996C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[240:243], v2 offset:10368                   // 0000000072D0: D9FE2880 F0000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[110:111], v[206:207], v[112:115]// 0000000072D8: D3F30070 0DC39D6E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[104:105], v[216:217], 0// 0000000072E0: D3F30074 0A03B168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[44:47], v46, s[24:27], 0 offen offset:3072// 0000000072E8: E05C1C00 80862C2E
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[106:107], v[218:219], v[116:119]// 0000000072F0: D3F30074 0DD3B56A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[108:109], v[220:221], v[116:119]// 0000000072F8: D3F30074 0DD3B96C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[244:247], v2 offset:10432                   // 000000007300: D9FE28C0 F4000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[110:111], v[222:223], v[116:119]// 000000007308: D3F30074 0DD3BD6E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[120:121], v[200:201], 0// 000000007310: D3F30078 0A039178  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[56:59], v47, s[24:27], 0 offen offset:2048// 000000007318: E05C1800 8086382F
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[122:123], v[202:203], v[120:123]// 000000007320: D3F30078 0DE3957A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[124:125], v[204:205], v[120:123]// 000000007328: D3F30078 0DE3997C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[248:251], v2 offset:10496                   // 000000007330: D9FE2900 F8000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[126:127], v[206:207], v[120:123]// 000000007338: D3F30078 0DE39D7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[120:121], v[216:217], 0// 000000007340: D3F3007C 0A03B178  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[60:63], v47, s[24:27], 0 offen offset:3072// 000000007348: E05C1C00 80863C2F
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[122:123], v[218:219], v[124:127]// 000000007350: D3F3007C 0DF3B57A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[124:125], v[220:221], v[124:127]// 000000007358: D3F3007C 0DF3B97C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[252:255], v2 offset:10560                   // 000000007360: D9FE2940 FC000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[126:127], v[222:223], v[124:127]// 000000007368: D3F3007C 0DF3BD7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v24, v17 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000007370: 0A7022FA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007378: 7E720338  // Vector move
	v_pk_fma_f32 v[64:65], v[96:97], v[56:57], v[64:65]        // 00000000737C: D3B04040 1D027160  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[66:67], v[98:99], v[56:57], v[66:67]        // 000000007384: D3B04042 1D0A7162  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[72:73], v[104:105], v[56:57], v[72:73]      // 00000000738C: D3B04048 1D227168  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[74:75], v[106:107], v[56:57], v[74:75]      // 000000007394: D3B0404A 1D2A716A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v17 row_newbcast:3 row_mask:0xf bank_mask:0xf// 00000000739C: 0A7022FA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000073A4: 7E720338  // Vector move
	v_pk_fma_f32 v[80:81], v[112:113], v[56:57], v[80:81]      // 0000000073A8: D3B04050 1D427170  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[82:83], v[114:115], v[56:57], v[82:83]      // 0000000073B0: D3B04052 1D4A7172  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[88:89], v[120:121], v[56:57], v[88:89]      // 0000000073B8: D3B04058 1D627178  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[90:91], v[122:123], v[56:57], v[90:91]      // 0000000073C0: D3B0405A 1D6A717A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v18 row_newbcast:2 row_mask:0xf bank_mask:0xf// 0000000073C8: 0A7024FA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000073D0: 7E720338  // Vector move
	v_pk_fma_f32 v[68:69], v[100:101], v[56:57], v[68:69]      // 0000000073D4: D3B04044 1D127164  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[70:71], v[102:103], v[56:57], v[70:71]      // 0000000073DC: D3B04046 1D1A7166  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[76:77], v[108:109], v[56:57], v[76:77]      // 0000000073E4: D3B0404C 1D32716C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[78:79], v[110:111], v[56:57], v[78:79]      // 0000000073EC: D3B0404E 1D3A716E  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v18 row_newbcast:3 row_mask:0xf bank_mask:0xf// 0000000073F4: 0A7024FA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000073FC: 7E720338  // Vector move
	v_pk_fma_f32 v[84:85], v[116:117], v[56:57], v[84:85]      // 000000007400: D3B04054 1D527174  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[86:87], v[118:119], v[56:57], v[86:87]      // 000000007408: D3B04056 1D5A7176  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[92:93], v[124:125], v[56:57], v[92:93]      // 000000007410: D3B0405C 1D72717C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[94:95], v[126:127], v[56:57], v[94:95]      // 000000007418: D3B0405E 1D7A717E  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x200, s80                                  // 000000007420: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 000000007428: BF0A513C
	s_cselect_b32 s57, s57, 0                                  // 00000000742C: 85398039
	s_cselect_b32 s3, s3, 0                                    // 000000007430: 85038003
	s_add_u32 s60, 0x200, s80                                  // 000000007434: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 00000000743C: BF0A513C
	s_cselect_b32 s58, s58, 0                                  // 000000007440: 853A803A
	s_add_u32 s20, s57, s20                                    // 000000007444: 80141439
	s_addc_u32 s21, 0, s21                                     // 000000007448: 82151580
	s_add_u32 s28, s3, s28                                     // 00000000744C: 801C1C03
	s_addc_u32 s29, 0, s29                                     // 000000007450: 821D1D80
	s_add_u32 s24, s58, s24                                    // 000000007454: 8018183A
	s_addc_u32 s25, 0, s25                                     // 000000007458: 82191980
	s_add_u32 s92, s90, s92                                    // 00000000745C: 805C5C5A
	s_addc_u32 s93, 0, s93                                     // 000000007460: 825D5D80
	s_addk_i32 s80, 0x100                                      // 000000007464: B7500100
	s_cmp_lt_i32 s80, s81                                      // 000000007468: BF045150
	s_cbranch_scc0 label_13ED                                  // 00000000746C: BF840251  // Branch if SCC==0 (condition false)
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(8) lgkmcnt(0)                              // 000000007470: BF8C0078  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 000000007474: BF8A0000  // Workgroup barrier synchronization
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[0:1], v[224:225], 0// 000000007478: D3F300A0 0A03C100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[64:67], v44, s[92:95], 0 offen       // 000000007480: E05C1000 8097402C
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[2:3], v[226:227], v[160:163]// 000000007488: D3F300A0 0E83C502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[4:5], v[228:229], v[160:163]// 000000007490: D3F300A0 0E83C904  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v36, s[20:23], 0 offen lds               // 000000007498: E0511000 80050024
	s_add_u32 m0, 0x100, s50                                   // 0000000074A0: 807C32FF 00000100
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[6:7], v[230:231], v[160:163]// 0000000074A8: D3F300A0 0E83CD06  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[0:1], v[240:241], 0// 0000000074B0: D3F300A4 0A03E100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[68:71], v44, s[92:95], 0 offen offset:1024// 0000000074B8: E05C1400 8097442C
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[2:3], v[242:243], v[164:167]// 0000000074C0: D3F300A4 0E93E502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[4:5], v[244:245], v[164:167]// 0000000074C8: D3F300A4 0E93E904  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v37, s[20:23], 0 offen lds               // 0000000074D0: E0511000 80050025
	s_add_u32 m0, 0x200, s50                                   // 0000000074D8: 807C32FF 00000200
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[6:7], v[246:247], v[164:167]// 0000000074E0: D3F300A4 0E93ED06  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[16:17], v[224:225], 0// 0000000074E8: D3F300A8 0A03C110  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[80:83], v45, s[92:95], 0 offen       // 0000000074F0: E05C1000 8097502D
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[18:19], v[226:227], v[168:171]// 0000000074F8: D3F300A8 0EA3C512  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[20:21], v[228:229], v[168:171]// 000000007500: D3F300A8 0EA3C914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v38, s[20:23], 0 offen lds               // 000000007508: E0511000 80050026
	s_add_u32 m0, 0x300, s50                                   // 000000007510: 807C32FF 00000300
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[22:23], v[230:231], v[168:171]// 000000007518: D3F300A8 0EA3CD16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[16:17], v[240:241], 0// 000000007520: D3F300AC 0A03E110  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[84:87], v45, s[92:95], 0 offen offset:1024// 000000007528: E05C1400 8097542D
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[18:19], v[242:243], v[172:175]// 000000007530: D3F300AC 0EB3E512  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[20:21], v[244:245], v[172:175]// 000000007538: D3F300AC 0EB3E914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v39, s[20:23], 0 offen lds               // 000000007540: E0511000 80050027
	s_add_u32 m0, 0x400, s50                                   // 000000007548: 807C32FF 00000400
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[22:23], v[246:247], v[172:175]// 000000007550: D3F300AC 0EB3ED16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[32:33], v[224:225], 0// 000000007558: D3F300B0 0A03C120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[96:99], v46, s[92:95], 0 offen       // 000000007560: E05C1000 8097602E
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[34:35], v[226:227], v[176:179]// 000000007568: D3F300B0 0EC3C522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[36:37], v[228:229], v[176:179]// 000000007570: D3F300B0 0EC3C924  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v40, s[20:23], 0 offen lds               // 000000007578: E0511000 80050028
	s_add_u32 m0, 0x500, s50                                   // 000000007580: 807C32FF 00000500
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[38:39], v[230:231], v[176:179]// 000000007588: D3F300B0 0EC3CD26  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[32:33], v[240:241], 0// 000000007590: D3F300B4 0A03E120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[100:103], v46, s[92:95], 0 offen offset:1024// 000000007598: E05C1400 8097642E
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[34:35], v[242:243], v[180:183]// 0000000075A0: D3F300B4 0ED3E522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[36:37], v[244:245], v[180:183]// 0000000075A8: D3F300B4 0ED3E924  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v41, s[20:23], 0 offen lds               // 0000000075B0: E0511000 80050029
	s_add_u32 m0, 0x600, s50                                   // 0000000075B8: 807C32FF 00000600
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[38:39], v[246:247], v[180:183]// 0000000075C0: D3F300B4 0ED3ED26  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[48:49], v[224:225], 0// 0000000075C8: D3F300B8 0A03C130  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[112:115], v47, s[92:95], 0 offen     // 0000000075D0: E05C1000 8097702F
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[50:51], v[226:227], v[184:187]// 0000000075D8: D3F300B8 0EE3C532  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[52:53], v[228:229], v[184:187]// 0000000075E0: D3F300B8 0EE3C934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v42, s[20:23], 0 offen lds               // 0000000075E8: E0511000 8005002A
	s_add_u32 m0, 0x700, s50                                   // 0000000075F0: 807C32FF 00000700
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[54:55], v[230:231], v[184:187]// 0000000075F8: D3F300B8 0EE3CD36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[48:49], v[240:241], 0// 000000007600: D3F300BC 0A03E130  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[116:119], v47, s[92:95], 0 offen offset:1024// 000000007608: E05C1400 8097742F
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[50:51], v[242:243], v[188:191]// 000000007610: D3F300BC 0EF3E532  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[52:53], v[244:245], v[188:191]// 000000007618: D3F300BC 0EF3E934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v43, s[20:23], 0 offen lds               // 000000007620: E0511000 8005002B
	s_add_u32 m0, s50, s76                                     // 000000007628: 807C4C32
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[54:55], v[246:247], v[188:191]// 00000000762C: D3F300BC 0EF3ED36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v7, s[28:31], 0 offen lds                // 000000007634: E0511000 80070007
	s_add_u32 m0, 0, s51                                       // 00000000763C: 807C3380
	buffer_load_dword v24, v13, s[32:35], 0 offen              // 000000007640: E0501000 8008180D
	v_mul_f32_dpp v56, v23, v19 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000007648: 0A7026FA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007650: 7E720338  // Vector move
	v_pk_fma_f32 v[128:129], v[160:161], v[56:57], v[128:129]  // 000000007654: D3B04080 1E0271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[130:131], v[162:163], v[56:57], v[130:131]  // 00000000765C: D3B04082 1E0A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[136:137], v[168:169], v[56:57], v[136:137]  // 000000007664: D3B04088 1E2271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[138:139], v[170:171], v[56:57], v[138:139]  // 00000000766C: D3B0408A 1E2A71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v19 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000007674: 0A7026FA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 00000000767C: 7E720338  // Vector move
	v_pk_fma_f32 v[144:145], v[176:177], v[56:57], v[144:145]  // 000000007680: D3B04090 1E4271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[146:147], v[178:179], v[56:57], v[146:147]  // 000000007688: D3B04092 1E4A71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[152:153], v[184:185], v[56:57], v[152:153]  // 000000007690: D3B04098 1E6271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[154:155], v[186:187], v[56:57], v[154:155]  // 000000007698: D3B0409A 1E6A71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v20 row_newbcast:0 row_mask:0xf bank_mask:0xf// 0000000076A0: 0A7028FA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000076A8: 7E720338  // Vector move
	v_pk_fma_f32 v[132:133], v[164:165], v[56:57], v[132:133]  // 0000000076AC: D3B04084 1E1271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[134:135], v[166:167], v[56:57], v[134:135]  // 0000000076B4: D3B04086 1E1A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[140:141], v[172:173], v[56:57], v[140:141]  // 0000000076BC: D3B0408C 1E3271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[142:143], v[174:175], v[56:57], v[142:143]  // 0000000076C4: D3B0408E 1E3A71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v20 row_newbcast:1 row_mask:0xf bank_mask:0xf// 0000000076CC: 0A7028FA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000076D4: 7E720338  // Vector move
	v_pk_fma_f32 v[148:149], v[180:181], v[56:57], v[148:149]  // 0000000076D8: D3B04094 1E5271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[150:151], v[182:183], v[56:57], v[150:151]  // 0000000076E0: D3B04096 1E5A71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[156:157], v[188:189], v[56:57], v[156:157]  // 0000000076E8: D3B0409C 1E7271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[158:159], v[190:191], v[56:57], v[158:159]  // 0000000076F0: D3B0409E 1E7A71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_waitcnt vmcnt(22)                                        // 0000000076F8: BF8C4F76  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[8:9], v[232:233], 0// 0000000076FC: D3F300A0 0A03D108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[72:75], v44, s[92:95], 0 offen offset:2048// 000000007704: E05C1800 8097482C
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[10:11], v[234:235], v[160:163]// 00000000770C: D3F300A0 0E83D50A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[12:13], v[236:237], v[160:163]// 000000007714: D3F300A0 0E83D90C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[14:15], v[238:239], v[160:163]// 00000000771C: D3F300A0 0E83DD0E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[8:9], v[248:249], 0// 000000007724: D3F300A4 0A03F108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[76:79], v44, s[92:95], 0 offen offset:3072// 00000000772C: E05C1C00 80974C2C
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[10:11], v[250:251], v[164:167]// 000000007734: D3F300A4 0E93F50A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[12:13], v[252:253], v[164:167]// 00000000773C: D3F300A4 0E93F90C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[14:15], v[254:255], v[164:167]// 000000007744: D3F300A4 0E93FD0E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[24:25], v[232:233], 0// 00000000774C: D3F300A8 0A03D118  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[88:91], v45, s[92:95], 0 offen offset:2048// 000000007754: E05C1800 8097582D
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[26:27], v[234:235], v[168:171]// 00000000775C: D3F300A8 0EA3D51A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[28:29], v[236:237], v[168:171]// 000000007764: D3F300A8 0EA3D91C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[30:31], v[238:239], v[168:171]// 00000000776C: D3F300A8 0EA3DD1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[24:25], v[248:249], 0// 000000007774: D3F300AC 0A03F118  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[92:95], v45, s[92:95], 0 offen offset:3072// 00000000777C: E05C1C00 80975C2D
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[26:27], v[250:251], v[172:175]// 000000007784: D3F300AC 0EB3F51A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[28:29], v[252:253], v[172:175]// 00000000778C: D3F300AC 0EB3F91C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[30:31], v[254:255], v[172:175]// 000000007794: D3F300AC 0EB3FD1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	s_waitcnt vmcnt(22)                                        // 00000000779C: BF8C4F76  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[40:41], v[232:233], 0// 0000000077A0: D3F300B0 0A03D128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[104:107], v46, s[92:95], 0 offen offset:2048// 0000000077A8: E05C1800 8097682E
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[42:43], v[234:235], v[176:179]// 0000000077B0: D3F300B0 0EC3D52A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[44:45], v[236:237], v[176:179]// 0000000077B8: D3F300B0 0EC3D92C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[46:47], v[238:239], v[176:179]// 0000000077C0: D3F300B0 0EC3DD2E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[40:41], v[248:249], 0// 0000000077C8: D3F300B4 0A03F128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[108:111], v46, s[92:95], 0 offen offset:3072// 0000000077D0: E05C1C00 80976C2E
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[42:43], v[250:251], v[180:183]// 0000000077D8: D3F300B4 0ED3F52A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[44:45], v[252:253], v[180:183]// 0000000077E0: D3F300B4 0ED3F92C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[46:47], v[254:255], v[180:183]// 0000000077E8: D3F300B4 0ED3FD2E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[56:57], v[232:233], 0// 0000000077F0: D3F300B8 0A03D138  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[120:123], v47, s[92:95], 0 offen offset:2048// 0000000077F8: E05C1800 8097782F
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[58:59], v[234:235], v[184:187]// 000000007800: D3F300B8 0EE3D53A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[60:61], v[236:237], v[184:187]// 000000007808: D3F300B8 0EE3D93C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[62:63], v[238:239], v[184:187]// 000000007810: D3F300B8 0EE3DD3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[56:57], v[248:249], 0// 000000007818: D3F300BC 0A03F138  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[124:127], v47, s[92:95], 0 offen offset:3072// 000000007820: E05C1C00 80977C2F
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[58:59], v[250:251], v[188:191]// 000000007828: D3F300BC 0EF3F53A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[60:61], v[252:253], v[188:191]// 000000007830: D3F300BC 0EF3F93C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[62:63], v[254:255], v[188:191]// 000000007838: D3F300BC 0EF3FD3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v23, v21 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000007840: 0A702AFA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007848: 7E720338  // Vector move
	v_pk_fma_f32 v[128:129], v[160:161], v[56:57], v[128:129]  // 00000000784C: D3B04080 1E0271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[130:131], v[162:163], v[56:57], v[130:131]  // 000000007854: D3B04082 1E0A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[136:137], v[168:169], v[56:57], v[136:137]  // 00000000785C: D3B04088 1E2271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[138:139], v[170:171], v[56:57], v[138:139]  // 000000007864: D3B0408A 1E2A71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v21 row_newbcast:3 row_mask:0xf bank_mask:0xf// 00000000786C: 0A702AFA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007874: 7E720338  // Vector move
	v_pk_fma_f32 v[144:145], v[176:177], v[56:57], v[144:145]  // 000000007878: D3B04090 1E4271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[146:147], v[178:179], v[56:57], v[146:147]  // 000000007880: D3B04092 1E4A71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[152:153], v[184:185], v[56:57], v[152:153]  // 000000007888: D3B04098 1E6271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[154:155], v[186:187], v[56:57], v[154:155]  // 000000007890: D3B0409A 1E6A71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v22 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000007898: 0A702CFA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000078A0: 7E720338  // Vector move
	v_pk_fma_f32 v[132:133], v[164:165], v[56:57], v[132:133]  // 0000000078A4: D3B04084 1E1271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[134:135], v[166:167], v[56:57], v[134:135]  // 0000000078AC: D3B04086 1E1A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[140:141], v[172:173], v[56:57], v[140:141]  // 0000000078B4: D3B0408C 1E3271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[142:143], v[174:175], v[56:57], v[142:143]  // 0000000078BC: D3B0408E 1E3A71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v22 row_newbcast:3 row_mask:0xf bank_mask:0xf// 0000000078C4: 0A702CFA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000078CC: 7E720338  // Vector move
	v_pk_fma_f32 v[148:149], v[180:181], v[56:57], v[148:149]  // 0000000078D0: D3B04094 1E5271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[150:151], v[182:183], v[56:57], v[150:151]  // 0000000078D8: D3B04096 1E5A71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[156:157], v[188:189], v[56:57], v[156:157]  // 0000000078E0: D3B0409C 1E7271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[158:159], v[190:191], v[56:57], v[158:159]  // 0000000078E8: D3B0409E 1E7A71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x100, s80                                  // 0000000078F0: 803C50FF 00000100
	s_cmp_lt_u32 s60, s81                                      // 0000000078F8: BF0A513C
	s_cselect_b32 s4, s4, 0                                    // 0000000078FC: 85048004
	s_add_u32 s32, s4, s32                                     // 000000007900: 80202004
	s_addc_u32 s33, 0, s33                                     // 000000007904: 82212180
	s_waitcnt vmcnt(8)                                         // 000000007908: BF8C0F78  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 00000000790C: BF8A0000  // Workgroup barrier synchronization
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[64:65], v[224:225], 0// 000000007910: D3F30060 0A03C140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[0:3], v44, s[24:27], 0 offen         // 000000007918: E05C1000 8086002C
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[66:67], v[226:227], v[96:99]// 000000007920: D3F30060 0D83C542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[68:69], v[228:229], v[96:99]// 000000007928: D3F30060 0D83C944  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v23, v11, s[32:35], 0 offen              // 000000007930: E0501000 8008170B
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[70:71], v[230:231], v[96:99]// 000000007938: D3F30060 0D83CD46  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[64:65], v[240:241], 0// 000000007940: D3F30064 0A03E140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[4:7], v44, s[24:27], 0 offen offset:1024// 000000007948: E05C1400 8086042C
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[66:67], v[242:243], v[100:103]// 000000007950: D3F30064 0D93E542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[68:69], v[244:245], v[100:103]// 000000007958: D3F30064 0D93E944  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[70:71], v[246:247], v[100:103]// 000000007960: D3F30064 0D93ED46  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[80:81], v[224:225], 0// 000000007968: D3F30068 0A03C150  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[16:19], v45, s[24:27], 0 offen       // 000000007970: E05C1000 8086102D
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[82:83], v[226:227], v[104:107]// 000000007978: D3F30068 0DA3C552  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[84:85], v[228:229], v[104:107]// 000000007980: D3F30068 0DA3C954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[86:87], v[230:231], v[104:107]// 000000007988: D3F30068 0DA3CD56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[80:81], v[240:241], 0// 000000007990: D3F3006C 0A03E150  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[20:23], v45, s[24:27], 0 offen offset:1024// 000000007998: E05C1400 8086142D
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[82:83], v[242:243], v[108:111]// 0000000079A0: D3F3006C 0DB3E552  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[84:85], v[244:245], v[108:111]// 0000000079A8: D3F3006C 0DB3E954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[86:87], v[246:247], v[108:111]// 0000000079B0: D3F3006C 0DB3ED56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[96:97], v[224:225], 0// 0000000079B8: D3F30070 0A03C160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[32:35], v46, s[24:27], 0 offen       // 0000000079C0: E05C1000 8086202E
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[98:99], v[226:227], v[112:115]// 0000000079C8: D3F30070 0DC3C562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[100:101], v[228:229], v[112:115]// 0000000079D0: D3F30070 0DC3C964  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[102:103], v[230:231], v[112:115]// 0000000079D8: D3F30070 0DC3CD66  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[96:97], v[240:241], 0// 0000000079E0: D3F30074 0A03E160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[36:39], v46, s[24:27], 0 offen offset:1024// 0000000079E8: E05C1400 8086242E
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[98:99], v[242:243], v[116:119]// 0000000079F0: D3F30074 0DD3E562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[100:101], v[244:245], v[116:119]// 0000000079F8: D3F30074 0DD3E964  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[102:103], v[246:247], v[116:119]// 000000007A00: D3F30074 0DD3ED66  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[112:113], v[224:225], 0// 000000007A08: D3F30078 0A03C170  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[48:51], v47, s[24:27], 0 offen       // 000000007A10: E05C1000 8086302F
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[114:115], v[226:227], v[120:123]// 000000007A18: D3F30078 0DE3C572  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[116:117], v[228:229], v[120:123]// 000000007A20: D3F30078 0DE3C974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[118:119], v[230:231], v[120:123]// 000000007A28: D3F30078 0DE3CD76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[112:113], v[240:241], 0// 000000007A30: D3F3007C 0A03E170  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[52:55], v47, s[24:27], 0 offen offset:1024// 000000007A38: E05C1400 8086342F
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[114:115], v[242:243], v[124:127]// 000000007A40: D3F3007C 0DF3E572  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[116:117], v[244:245], v[124:127]// 000000007A48: D3F3007C 0DF3E974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[118:119], v[246:247], v[124:127]// 000000007A50: D3F3007C 0DF3ED76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v24, v19 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000007A58: 0A7026FA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007A60: 7E720338  // Vector move
	v_pk_fma_f32 v[64:65], v[96:97], v[56:57], v[64:65]        // 000000007A64: D3B04040 1D027160  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[66:67], v[98:99], v[56:57], v[66:67]        // 000000007A6C: D3B04042 1D0A7162  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[72:73], v[104:105], v[56:57], v[72:73]      // 000000007A74: D3B04048 1D227168  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[74:75], v[106:107], v[56:57], v[74:75]      // 000000007A7C: D3B0404A 1D2A716A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v19 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000007A84: 0A7026FA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007A8C: 7E720338  // Vector move
	v_pk_fma_f32 v[80:81], v[112:113], v[56:57], v[80:81]      // 000000007A90: D3B04050 1D427170  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[82:83], v[114:115], v[56:57], v[82:83]      // 000000007A98: D3B04052 1D4A7172  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[88:89], v[120:121], v[56:57], v[88:89]      // 000000007AA0: D3B04058 1D627178  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[90:91], v[122:123], v[56:57], v[90:91]      // 000000007AA8: D3B0405A 1D6A717A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v20 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000007AB0: 0A7028FA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007AB8: 7E720338  // Vector move
	v_pk_fma_f32 v[68:69], v[100:101], v[56:57], v[68:69]      // 000000007ABC: D3B04044 1D127164  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[70:71], v[102:103], v[56:57], v[70:71]      // 000000007AC4: D3B04046 1D1A7166  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[76:77], v[108:109], v[56:57], v[76:77]      // 000000007ACC: D3B0404C 1D32716C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[78:79], v[110:111], v[56:57], v[78:79]      // 000000007AD4: D3B0404E 1D3A716E  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v20 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000007ADC: 0A7028FA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007AE4: 7E720338  // Vector move
	v_pk_fma_f32 v[84:85], v[116:117], v[56:57], v[84:85]      // 000000007AE8: D3B04054 1D527174  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[86:87], v[118:119], v[56:57], v[86:87]      // 000000007AF0: D3B04056 1D5A7176  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[92:93], v[124:125], v[56:57], v[92:93]      // 000000007AF8: D3B0405C 1D72717C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[94:95], v[126:127], v[56:57], v[94:95]      // 000000007B00: D3B0405E 1D7A717E  // Packed FMA: 2x FP32 fused multiply-add
	s_waitcnt vmcnt(13)                                        // 000000007B08: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[72:73], v[232:233], 0// 000000007B0C: D3F30060 0A03D148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[8:11], v44, s[24:27], 0 offen offset:2048// 000000007B14: E05C1800 8086082C
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[74:75], v[234:235], v[96:99]// 000000007B1C: D3F30060 0D83D54A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[76:77], v[236:237], v[96:99]// 000000007B24: D3F30060 0D83D94C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[192:195], v2                                // 000000007B2C: D9FE0000 C0000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v15, v3 offset:8320                            // 000000007B34: D86C2080 0F000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], a[78:79], v[238:239], v[96:99]// 000000007B3C: D3F30060 0D83DD4E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[72:73], v[248:249], 0// 000000007B44: D3F30064 0A03F148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[12:15], v44, s[24:27], 0 offen offset:3072// 000000007B4C: E05C1C00 80860C2C
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[74:75], v[250:251], v[100:103]// 000000007B54: D3F30064 0D93F54A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[76:77], v[252:253], v[100:103]// 000000007B5C: D3F30064 0D93F94C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[196:199], v2 offset:64                      // 000000007B64: D9FE0040 C4000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v16, v3 offset:8576                            // 000000007B6C: D86C2180 10000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[100:103], a[78:79], v[254:255], v[100:103]// 000000007B74: D3F30064 0D93FD4E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[88:89], v[232:233], 0// 000000007B7C: D3F30068 0A03D158  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[24:27], v45, s[24:27], 0 offen offset:2048// 000000007B84: E05C1800 8086182D
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[90:91], v[234:235], v[104:107]// 000000007B8C: D3F30068 0DA3D55A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[92:93], v[236:237], v[104:107]// 000000007B94: D3F30068 0DA3D95C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[200:203], v2 offset:128                     // 000000007B9C: D9FE0080 C8000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v17, v3 offset:8832                            // 000000007BA4: D86C2280 11000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[104:107], a[94:95], v[238:239], v[104:107]// 000000007BAC: D3F30068 0DA3DD5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[88:89], v[248:249], 0// 000000007BB4: D3F3006C 0A03F158  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[28:31], v45, s[24:27], 0 offen offset:3072// 000000007BBC: E05C1C00 80861C2D
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[90:91], v[250:251], v[108:111]// 000000007BC4: D3F3006C 0DB3F55A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[92:93], v[252:253], v[108:111]// 000000007BCC: D3F3006C 0DB3F95C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[204:207], v2 offset:192                     // 000000007BD4: D9FE00C0 CC000002  // Read 128-bit from LDS (4x32-bit)
	ds_read_b32 v18, v3 offset:9088                            // 000000007BDC: D86C2380 12000003  // Read 32-bit from LDS (shared memory)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[108:111], a[94:95], v[254:255], v[108:111]// 000000007BE4: D3F3006C 0DB3FD5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	s_waitcnt vmcnt(13)                                        // 000000007BEC: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[104:105], v[232:233], 0// 000000007BF0: D3F30070 0A03D168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[40:43], v46, s[24:27], 0 offen offset:2048// 000000007BF8: E05C1800 8086282E
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[106:107], v[234:235], v[112:115]// 000000007C00: D3F30070 0DC3D56A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[108:109], v[236:237], v[112:115]// 000000007C08: D3F30070 0DC3D96C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[208:211], v2 offset:1024                    // 000000007C10: D9FE0400 D0000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[112:115], a[110:111], v[238:239], v[112:115]// 000000007C18: D3F30070 0DC3DD6E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[104:105], v[248:249], 0// 000000007C20: D3F30074 0A03F168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[44:47], v46, s[24:27], 0 offen offset:3072// 000000007C28: E05C1C00 80862C2E
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[106:107], v[250:251], v[116:119]// 000000007C30: D3F30074 0DD3F56A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[108:109], v[252:253], v[116:119]// 000000007C38: D3F30074 0DD3F96C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[212:215], v2 offset:1088                    // 000000007C40: D9FE0440 D4000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[116:119], a[110:111], v[254:255], v[116:119]// 000000007C48: D3F30074 0DD3FD6E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[120:121], v[232:233], 0// 000000007C50: D3F30078 0A03D178  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[56:59], v47, s[24:27], 0 offen offset:2048// 000000007C58: E05C1800 8086382F
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[122:123], v[234:235], v[120:123]// 000000007C60: D3F30078 0DE3D57A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[124:125], v[236:237], v[120:123]// 000000007C68: D3F30078 0DE3D97C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[216:219], v2 offset:1152                    // 000000007C70: D9FE0480 D8000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[120:123], a[126:127], v[238:239], v[120:123]// 000000007C78: D3F30078 0DE3DD7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[120:121], v[248:249], 0// 000000007C80: D3F3007C 0A03F178  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[60:63], v47, s[24:27], 0 offen offset:3072// 000000007C88: E05C1C00 80863C2F
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[122:123], v[250:251], v[124:127]// 000000007C90: D3F3007C 0DF3F57A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[124:125], v[252:253], v[124:127]// 000000007C98: D3F3007C 0DF3F97C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS READ (feeding MFMA) ===
	ds_read_b128 v[220:223], v2 offset:1216                    // 000000007CA0: D9FE04C0 DC000002  // Read 128-bit from LDS (4x32-bit)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], a[126:127], v[254:255], v[124:127]// 000000007CA8: D3F3007C 0DF3FD7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v24, v21 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000007CB0: 0A702AFA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007CB8: 7E720338  // Vector move
	v_pk_fma_f32 v[64:65], v[96:97], v[56:57], v[64:65]        // 000000007CBC: D3B04040 1D027160  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[66:67], v[98:99], v[56:57], v[66:67]        // 000000007CC4: D3B04042 1D0A7162  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[72:73], v[104:105], v[56:57], v[72:73]      // 000000007CCC: D3B04048 1D227168  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[74:75], v[106:107], v[56:57], v[74:75]      // 000000007CD4: D3B0404A 1D2A716A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v21 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000007CDC: 0A702AFA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007CE4: 7E720338  // Vector move
	v_pk_fma_f32 v[80:81], v[112:113], v[56:57], v[80:81]      // 000000007CE8: D3B04050 1D427170  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[82:83], v[114:115], v[56:57], v[82:83]      // 000000007CF0: D3B04052 1D4A7172  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[88:89], v[120:121], v[56:57], v[88:89]      // 000000007CF8: D3B04058 1D627178  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[90:91], v[122:123], v[56:57], v[90:91]      // 000000007D00: D3B0405A 1D6A717A  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v22 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000007D08: 0A702CFA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007D10: 7E720338  // Vector move
	v_pk_fma_f32 v[68:69], v[100:101], v[56:57], v[68:69]      // 000000007D14: D3B04044 1D127164  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[70:71], v[102:103], v[56:57], v[70:71]      // 000000007D1C: D3B04046 1D1A7166  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[76:77], v[108:109], v[56:57], v[76:77]      // 000000007D24: D3B0404C 1D32716C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[78:79], v[110:111], v[56:57], v[78:79]      // 000000007D2C: D3B0404E 1D3A716E  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v22 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000007D34: 0A702CFA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000007D3C: 7E720338  // Vector move
	v_pk_fma_f32 v[84:85], v[116:117], v[56:57], v[84:85]      // 000000007D40: D3B04054 1D527174  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[86:87], v[118:119], v[56:57], v[86:87]      // 000000007D48: D3B04056 1D5A7176  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[92:93], v[124:125], v[56:57], v[92:93]      // 000000007D50: D3B0405C 1D72717C  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[94:95], v[126:127], v[56:57], v[94:95]      // 000000007D58: D3B0405E 1D7A717E  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x200, s80                                  // 000000007D60: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 000000007D68: BF0A513C
	s_cselect_b32 s57, s57, 0                                  // 000000007D6C: 85398039
	s_cselect_b32 s3, s3, 0                                    // 000000007D70: 85038003
	s_add_u32 s60, 0x200, s80                                  // 000000007D74: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 000000007D7C: BF0A513C
	s_cselect_b32 s58, s58, 0                                  // 000000007D80: 853A803A
	s_add_u32 s20, s57, s20                                    // 000000007D84: 80141439
	s_addc_u32 s21, 0, s21                                     // 000000007D88: 82151580
	s_add_u32 s28, s3, s28                                     // 000000007D8C: 801C1C03
	s_addc_u32 s29, 0, s29                                     // 000000007D90: 821D1D80
	s_add_u32 s24, s58, s24                                    // 000000007D94: 8018183A
	s_addc_u32 s25, 0, s25                                     // 000000007D98: 82191980
	s_add_u32 s92, s90, s92                                    // 000000007D9C: 805C5C5A
	s_addc_u32 s93, 0, s93                                     // 000000007DA0: 825D5D80
	s_addk_i32 s80, 0x100                                      // 000000007DA4: B7500100
	s_cmp_lt_i32 s80, s81                                      // 000000007DA8: BF045150
	s_cbranch_scc0 label_13ED                                  // 000000007DAC: BF840001  // Branch if SCC==0 (condition false)
	s_branch label_0F4C                                        // 000000007DB0: BF82FB5F  // Unconditional branch


// --- LABEL (potential loop target) ---
0000000000007db4 <label_13ED>:
	s_mov_b32 s20, 0                                           // 000000007DB4: BE940080  // Scalar move (constant/register)
	s_cmp_lt_u32 s89, s66                                      // 000000007DB8: BF0A4259
	s_cselect_b32 s60, 0, 1                                    // 000000007DBC: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000007DC0: 97143C14
	s_cmp_lt_u32 s88, s66                                      // 000000007DC4: BF0A4258
	s_cselect_b32 s60, 0, 1                                    // 000000007DC8: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000007DCC: 97143C14
	s_cmp_lt_u32 s87, s66                                      // 000000007DD0: BF0A4257
	s_cselect_b32 s60, 0, 1                                    // 000000007DD4: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000007DD8: 97143C14
	s_cmp_lt_u32 s86, s66                                      // 000000007DDC: BF0A4256
	s_cselect_b32 s60, 0, 1                                    // 000000007DE0: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000007DE4: 97143C14
	s_cmp_lt_u32 s85, s66                                      // 000000007DE8: BF0A4255
	s_cselect_b32 s60, 0, 1                                    // 000000007DEC: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000007DF0: 97143C14
	s_cmp_lt_u32 s84, s66                                      // 000000007DF4: BF0A4254
	s_cselect_b32 s60, 0, 1                                    // 000000007DF8: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000007DFC: 97143C14
	s_cmp_lt_u32 s83, s66                                      // 000000007E00: BF0A4253
	s_cselect_b32 s60, 0, 1                                    // 000000007E04: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000007E08: 97143C14
	s_cmp_lt_u32 s82, s66                                      // 000000007E0C: BF0A4252
	s_cselect_b32 s60, 0, 1                                    // 000000007E10: 853C8180
	s_lshl1_add_u32 s20, s20, s60                              // 000000007E14: 97143C14
	s_waitcnt vmcnt(12)                                        // 000000007E18: BF8C0F7C  // Wait for memory operations (CRITICAL for perf)
	buffer_load_dwordx4 a[0:3], v48, s[12:15], 0 offen         // 000000007E1C: E05C1000 80830030
	v_mul_f32_e64 v56, -v128, s6                               // 000000007E24: D1050038 20000D80
	v_mul_f32_e64 v57, -v129, s6                               // 000000007E2C: D1050039 20000D81
	v_mul_f32_e64 v58, -v130, s6                               // 000000007E34: D105003A 20000D82
	v_mul_f32_e64 v59, -v131, s6                               // 000000007E3C: D105003B 20000D83
	v_exp_f32_e32 v56, v56                                     // 000000007E44: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000007E48: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000007E4C: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000007E50: 7E76413B
	buffer_load_dwordx4 a[4:7], v49, s[12:15], 0 offen         // 000000007E54: E05C1000 80830431
	v_add_f32_e64 v56, v56, 1.0                                // 000000007E5C: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000007E64: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000007E6C: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000007E74: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000007E7C: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000007E80: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000007E84: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000007E88: 7E76453B
	v_mul_f32_e32 v128, v128, v56                              // 000000007E8C: 0B007180  // FP32 multiply
	v_mul_f32_e32 v129, v129, v57                              // 000000007E90: 0B027381  // FP32 multiply
	v_mul_f32_e32 v130, v130, v58                              // 000000007E94: 0B047582  // FP32 multiply
	v_mul_f32_e32 v131, v131, v59                              // 000000007E98: 0B067783  // FP32 multiply
	v_mul_f32_e32 v128, v128, v64                              // 000000007E9C: 0B008180  // FP32 multiply
	v_mul_f32_e32 v129, v129, v65                              // 000000007EA0: 0B028381  // FP32 multiply
	v_mul_f32_e32 v130, v130, v66                              // 000000007EA4: 0B048582  // FP32 multiply
	v_mul_f32_e32 v131, v131, v67                              // 000000007EA8: 0B068783  // FP32 multiply
	buffer_load_dwordx4 a[8:11], v50, s[12:15], 0 offen        // 000000007EAC: E05C1000 80830832
	v_mul_f32_e64 v56, -v132, s6                               // 000000007EB4: D1050038 20000D84
	v_mul_f32_e64 v57, -v133, s6                               // 000000007EBC: D1050039 20000D85
	v_mul_f32_e64 v58, -v134, s6                               // 000000007EC4: D105003A 20000D86
	v_mul_f32_e64 v59, -v135, s6                               // 000000007ECC: D105003B 20000D87
	v_exp_f32_e32 v56, v56                                     // 000000007ED4: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000007ED8: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000007EDC: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000007EE0: 7E76413B
	buffer_load_dwordx4 a[12:15], v51, s[12:15], 0 offen       // 000000007EE4: E05C1000 80830C33
	s_add_u32 s12, s78, s12                                    // 000000007EEC: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000007EF0: 820D0D80
	v_add_f32_e64 v56, v56, 1.0                                // 000000007EF4: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000007EFC: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000007F04: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000007F0C: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000007F14: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000007F18: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000007F1C: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000007F20: 7E76453B
	v_mul_f32_e32 v132, v132, v56                              // 000000007F24: 0B087184  // FP32 multiply
	v_mul_f32_e32 v133, v133, v57                              // 000000007F28: 0B0A7385  // FP32 multiply
	v_mul_f32_e32 v134, v134, v58                              // 000000007F2C: 0B0C7586  // FP32 multiply
	v_mul_f32_e32 v135, v135, v59                              // 000000007F30: 0B0E7787  // FP32 multiply
	v_mul_f32_e32 v132, v132, v68                              // 000000007F34: 0B088984  // FP32 multiply
	v_mul_f32_e32 v133, v133, v69                              // 000000007F38: 0B0A8B85  // FP32 multiply
	v_mul_f32_e32 v134, v134, v70                              // 000000007F3C: 0B0C8D86  // FP32 multiply
	v_mul_f32_e32 v135, v135, v71                              // 000000007F40: 0B0E8F87  // FP32 multiply
	s_waitcnt vmcnt(12)                                        // 000000007F44: BF8C0F7C  // Wait for memory operations (CRITICAL for perf)
	buffer_load_dwordx4 a[16:19], v48, s[12:15], 0 offen       // 000000007F48: E05C1000 80831030
	v_mul_f32_e64 v56, -v136, s6                               // 000000007F50: D1050038 20000D88
	v_mul_f32_e64 v57, -v137, s6                               // 000000007F58: D1050039 20000D89
	v_mul_f32_e64 v58, -v138, s6                               // 000000007F60: D105003A 20000D8A
	v_mul_f32_e64 v59, -v139, s6                               // 000000007F68: D105003B 20000D8B
	v_exp_f32_e32 v56, v56                                     // 000000007F70: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000007F74: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000007F78: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000007F7C: 7E76413B
	buffer_load_dwordx4 a[20:23], v49, s[12:15], 0 offen       // 000000007F80: E05C1000 80831431
	v_add_f32_e64 v56, v56, 1.0                                // 000000007F88: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000007F90: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000007F98: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000007FA0: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000007FA8: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000007FAC: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000007FB0: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000007FB4: 7E76453B
	v_mul_f32_e32 v136, v136, v56                              // 000000007FB8: 0B107188  // FP32 multiply
	v_mul_f32_e32 v137, v137, v57                              // 000000007FBC: 0B127389  // FP32 multiply
	v_mul_f32_e32 v138, v138, v58                              // 000000007FC0: 0B14758A  // FP32 multiply
	v_mul_f32_e32 v139, v139, v59                              // 000000007FC4: 0B16778B  // FP32 multiply
	v_mul_f32_e32 v136, v136, v72                              // 000000007FC8: 0B109188  // FP32 multiply
	v_mul_f32_e32 v137, v137, v73                              // 000000007FCC: 0B129389  // FP32 multiply
	v_mul_f32_e32 v138, v138, v74                              // 000000007FD0: 0B14958A  // FP32 multiply
	v_mul_f32_e32 v139, v139, v75                              // 000000007FD4: 0B16978B  // FP32 multiply
	buffer_load_dwordx4 a[24:27], v50, s[12:15], 0 offen       // 000000007FD8: E05C1000 80831832
	v_mul_f32_e64 v56, -v140, s6                               // 000000007FE0: D1050038 20000D8C
	v_mul_f32_e64 v57, -v141, s6                               // 000000007FE8: D1050039 20000D8D
	v_mul_f32_e64 v58, -v142, s6                               // 000000007FF0: D105003A 20000D8E
	v_mul_f32_e64 v59, -v143, s6                               // 000000007FF8: D105003B 20000D8F
	v_exp_f32_e32 v56, v56                                     // 000000008000: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000008004: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000008008: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 00000000800C: 7E76413B
	buffer_load_dwordx4 a[28:31], v51, s[12:15], 0 offen       // 000000008010: E05C1000 80831C33
	s_add_u32 s12, s78, s12                                    // 000000008018: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 00000000801C: 820D0D80
	v_add_f32_e64 v56, v56, 1.0                                // 000000008020: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000008028: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000008030: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000008038: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000008040: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000008044: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000008048: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 00000000804C: 7E76453B
	v_mul_f32_e32 v140, v140, v56                              // 000000008050: 0B18718C  // FP32 multiply
	v_mul_f32_e32 v141, v141, v57                              // 000000008054: 0B1A738D  // FP32 multiply
	v_mul_f32_e32 v142, v142, v58                              // 000000008058: 0B1C758E  // FP32 multiply
	v_mul_f32_e32 v143, v143, v59                              // 00000000805C: 0B1E778F  // FP32 multiply
	v_mul_f32_e32 v140, v140, v76                              // 000000008060: 0B18998C  // FP32 multiply
	v_mul_f32_e32 v141, v141, v77                              // 000000008064: 0B1A9B8D  // FP32 multiply
	v_mul_f32_e32 v142, v142, v78                              // 000000008068: 0B1C9D8E  // FP32 multiply
	v_mul_f32_e32 v143, v143, v79                              // 00000000806C: 0B1E9F8F  // FP32 multiply
	s_waitcnt vmcnt(12)                                        // 000000008070: BF8C0F7C  // Wait for memory operations (CRITICAL for perf)
	buffer_load_dwordx4 a[32:35], v48, s[12:15], 0 offen       // 000000008074: E05C1000 80832030
	v_mul_f32_e64 v56, -v144, s6                               // 00000000807C: D1050038 20000D90
	v_mul_f32_e64 v57, -v145, s6                               // 000000008084: D1050039 20000D91
	v_mul_f32_e64 v58, -v146, s6                               // 00000000808C: D105003A 20000D92
	v_mul_f32_e64 v59, -v147, s6                               // 000000008094: D105003B 20000D93
	v_exp_f32_e32 v56, v56                                     // 00000000809C: 7E704138
	v_exp_f32_e32 v57, v57                                     // 0000000080A0: 7E724139
	v_exp_f32_e32 v58, v58                                     // 0000000080A4: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 0000000080A8: 7E76413B
	buffer_load_dwordx4 a[36:39], v49, s[12:15], 0 offen       // 0000000080AC: E05C1000 80832431
	v_add_f32_e64 v56, v56, 1.0                                // 0000000080B4: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 0000000080BC: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 0000000080C4: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 0000000080CC: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 0000000080D4: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 0000000080D8: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 0000000080DC: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 0000000080E0: 7E76453B
	v_mul_f32_e32 v144, v144, v56                              // 0000000080E4: 0B207190  // FP32 multiply
	v_mul_f32_e32 v145, v145, v57                              // 0000000080E8: 0B227391  // FP32 multiply
	v_mul_f32_e32 v146, v146, v58                              // 0000000080EC: 0B247592  // FP32 multiply
	v_mul_f32_e32 v147, v147, v59                              // 0000000080F0: 0B267793  // FP32 multiply
	v_mul_f32_e32 v144, v144, v80                              // 0000000080F4: 0B20A190  // FP32 multiply
	v_mul_f32_e32 v145, v145, v81                              // 0000000080F8: 0B22A391  // FP32 multiply
	v_mul_f32_e32 v146, v146, v82                              // 0000000080FC: 0B24A592  // FP32 multiply
	v_mul_f32_e32 v147, v147, v83                              // 000000008100: 0B26A793  // FP32 multiply
	buffer_load_dwordx4 a[40:43], v50, s[12:15], 0 offen       // 000000008104: E05C1000 80832832
	v_mul_f32_e64 v56, -v148, s6                               // 00000000810C: D1050038 20000D94
	v_mul_f32_e64 v57, -v149, s6                               // 000000008114: D1050039 20000D95
	v_mul_f32_e64 v58, -v150, s6                               // 00000000811C: D105003A 20000D96
	v_mul_f32_e64 v59, -v151, s6                               // 000000008124: D105003B 20000D97
	v_exp_f32_e32 v56, v56                                     // 00000000812C: 7E704138
	v_exp_f32_e32 v57, v57                                     // 000000008130: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000008134: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000008138: 7E76413B
	buffer_load_dwordx4 a[44:47], v51, s[12:15], 0 offen       // 00000000813C: E05C1000 80832C33
	s_add_u32 s12, s78, s12                                    // 000000008144: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000008148: 820D0D80
	v_add_f32_e64 v56, v56, 1.0                                // 00000000814C: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000008154: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 00000000815C: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000008164: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 00000000816C: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000008170: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000008174: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 000000008178: 7E76453B
	v_mul_f32_e32 v148, v148, v56                              // 00000000817C: 0B287194  // FP32 multiply
	v_mul_f32_e32 v149, v149, v57                              // 000000008180: 0B2A7395  // FP32 multiply
	v_mul_f32_e32 v150, v150, v58                              // 000000008184: 0B2C7596  // FP32 multiply
	v_mul_f32_e32 v151, v151, v59                              // 000000008188: 0B2E7797  // FP32 multiply
	v_mul_f32_e32 v148, v148, v84                              // 00000000818C: 0B28A994  // FP32 multiply
	v_mul_f32_e32 v149, v149, v85                              // 000000008190: 0B2AAB95  // FP32 multiply
	v_mul_f32_e32 v150, v150, v86                              // 000000008194: 0B2CAD96  // FP32 multiply
	v_mul_f32_e32 v151, v151, v87                              // 000000008198: 0B2EAF97  // FP32 multiply
	s_waitcnt vmcnt(12)                                        // 00000000819C: BF8C0F7C  // Wait for memory operations (CRITICAL for perf)
	buffer_load_dwordx4 a[48:51], v48, s[12:15], 0 offen       // 0000000081A0: E05C1000 80833030
	v_mul_f32_e64 v56, -v152, s6                               // 0000000081A8: D1050038 20000D98
	v_mul_f32_e64 v57, -v153, s6                               // 0000000081B0: D1050039 20000D99
	v_mul_f32_e64 v58, -v154, s6                               // 0000000081B8: D105003A 20000D9A
	v_mul_f32_e64 v59, -v155, s6                               // 0000000081C0: D105003B 20000D9B
	v_exp_f32_e32 v56, v56                                     // 0000000081C8: 7E704138
	v_exp_f32_e32 v57, v57                                     // 0000000081CC: 7E724139
	v_exp_f32_e32 v58, v58                                     // 0000000081D0: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 0000000081D4: 7E76413B
	buffer_load_dwordx4 a[52:55], v49, s[12:15], 0 offen       // 0000000081D8: E05C1000 80833431
	v_add_f32_e64 v56, v56, 1.0                                // 0000000081E0: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 0000000081E8: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 0000000081F0: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 0000000081F8: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000008200: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000008204: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000008208: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 00000000820C: 7E76453B
	v_mul_f32_e32 v152, v152, v56                              // 000000008210: 0B307198  // FP32 multiply
	v_mul_f32_e32 v153, v153, v57                              // 000000008214: 0B327399  // FP32 multiply
	v_mul_f32_e32 v154, v154, v58                              // 000000008218: 0B34759A  // FP32 multiply
	v_mul_f32_e32 v155, v155, v59                              // 00000000821C: 0B36779B  // FP32 multiply
	v_mul_f32_e32 v152, v152, v88                              // 000000008220: 0B30B198  // FP32 multiply
	v_mul_f32_e32 v153, v153, v89                              // 000000008224: 0B32B399  // FP32 multiply
	v_mul_f32_e32 v154, v154, v90                              // 000000008228: 0B34B59A  // FP32 multiply
	v_mul_f32_e32 v155, v155, v91                              // 00000000822C: 0B36B79B  // FP32 multiply
	buffer_load_dwordx4 a[56:59], v50, s[12:15], 0 offen       // 000000008230: E05C1000 80833832
	v_mul_f32_e64 v56, -v156, s6                               // 000000008238: D1050038 20000D9C
	v_mul_f32_e64 v57, -v157, s6                               // 000000008240: D1050039 20000D9D
	v_mul_f32_e64 v58, -v158, s6                               // 000000008248: D105003A 20000D9E
	v_mul_f32_e64 v59, -v159, s6                               // 000000008250: D105003B 20000D9F
	v_exp_f32_e32 v56, v56                                     // 000000008258: 7E704138
	v_exp_f32_e32 v57, v57                                     // 00000000825C: 7E724139
	v_exp_f32_e32 v58, v58                                     // 000000008260: 7E74413A
	v_exp_f32_e32 v59, v59                                     // 000000008264: 7E76413B
	buffer_load_dwordx4 a[60:63], v51, s[12:15], 0 offen       // 000000008268: E05C1000 80833C33
	v_add_f32_e64 v56, v56, 1.0                                // 000000008270: D1010038 0001E538  // FP32 addition
	v_add_f32_e64 v57, v57, 1.0                                // 000000008278: D1010039 0001E539  // FP32 addition
	v_add_f32_e64 v58, v58, 1.0                                // 000000008280: D101003A 0001E53A  // FP32 addition
	v_add_f32_e64 v59, v59, 1.0                                // 000000008288: D101003B 0001E53B  // FP32 addition
	v_rcp_f32_e32 v56, v56                                     // 000000008290: 7E704538
	v_rcp_f32_e32 v57, v57                                     // 000000008294: 7E724539
	v_rcp_f32_e32 v58, v58                                     // 000000008298: 7E74453A
	v_rcp_f32_e32 v59, v59                                     // 00000000829C: 7E76453B
	v_mul_f32_e32 v156, v156, v56                              // 0000000082A0: 0B38719C  // FP32 multiply
	v_mul_f32_e32 v157, v157, v57                              // 0000000082A4: 0B3A739D  // FP32 multiply
	v_mul_f32_e32 v158, v158, v58                              // 0000000082A8: 0B3C759E  // FP32 multiply
	v_mul_f32_e32 v159, v159, v59                              // 0000000082AC: 0B3E779F  // FP32 multiply
	v_mul_f32_e32 v156, v156, v92                              // 0000000082B0: 0B38B99C  // FP32 multiply
	v_mul_f32_e32 v157, v157, v93                              // 0000000082B4: 0B3ABB9D  // FP32 multiply
	v_mul_f32_e32 v158, v158, v94                              // 0000000082B8: 0B3CBD9E  // FP32 multiply
	v_mul_f32_e32 v159, v159, v95                              // 0000000082BC: 0B3EBF9F  // FP32 multiply
	v_lshlrev_b32_e32 v56, 2, v0                               // 0000000082C0: 24700082  // Left shift (multiply by power of 2)
	s_mul_i32 s60, s82, s71                                    // 0000000082C4: 923C4752
	v_add_u32_e64 v80, v56, s60                                // 0000000082C8: D1340050 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v81, 0                                       // 0000000082D0: 7EA20280  // Vector move
	s_mul_i32 s60, s83, s71                                    // 0000000082D4: 923C4753
	v_add_u32_e64 v82, v56, s60                                // 0000000082D8: D1340052 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v83, 0                                       // 0000000082E0: 7EA60280  // Vector move
	s_mul_i32 s60, s84, s71                                    // 0000000082E4: 923C4754
	v_add_u32_e64 v84, v56, s60                                // 0000000082E8: D1340054 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v85, 0                                       // 0000000082F0: 7EAA0280  // Vector move
	s_mul_i32 s60, s85, s71                                    // 0000000082F4: 923C4755
	v_add_u32_e64 v86, v56, s60                                // 0000000082F8: D1340056 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v87, 0                                       // 000000008300: 7EAE0280  // Vector move
	s_mul_i32 s60, s86, s71                                    // 000000008304: 923C4756
	v_add_u32_e64 v88, v56, s60                                // 000000008308: D1340058 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v89, 0                                       // 000000008310: 7EB20280  // Vector move
	s_mul_i32 s60, s87, s71                                    // 000000008314: 923C4757
	v_add_u32_e64 v90, v56, s60                                // 000000008318: D134005A 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v91, 0                                       // 000000008320: 7EB60280  // Vector move
	s_mul_i32 s60, s88, s71                                    // 000000008324: 923C4758
	v_add_u32_e64 v92, v56, s60                                // 000000008328: D134005C 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v93, 0                                       // 000000008330: 7EBA0280  // Vector move
	s_mul_i32 s60, s89, s71                                    // 000000008334: 923C4759
	v_add_u32_e64 v94, v56, s60                                // 000000008338: D134005E 00007938  // 32-bit unsigned add (address calc)
	v_mov_b32_e32 v95, 0                                       // 000000008340: 7EBE0280  // Vector move
	buffer_load_dword v23, v6, s[16:19], 0 offen               // 000000008344: E0501000 80041706
	v_mov_b32_e32 v28, 0x358637bd                              // 00000000834C: 7E3802FF 358637BD  // Vector move
	v_mov_b32_e32 v29, 0x358637bd                              // 000000008354: 7E3A02FF 358637BD  // Vector move
	v_max3_f32 v28, |v128|, |v129|, v28                        // 00000000835C: D1D3031C 04730380
	v_max3_f32 v28, |v130|, |v131|, v28                        // 000000008364: D1D3031C 04730782
	v_max3_f32 v29, |v132|, |v133|, v29                        // 00000000836C: D1D3031D 04770B84
	v_max3_f32 v29, |v134|, |v135|, v29                        // 000000008374: D1D3031D 04770F86
	v_max3_f32 v28, |v136|, |v137|, v28                        // 00000000837C: D1D3031C 04731388
	v_max3_f32 v28, |v138|, |v139|, v28                        // 000000008384: D1D3031C 0473178A
	v_max3_f32 v29, |v140|, |v141|, v29                        // 00000000838C: D1D3031D 04771B8C
	v_max3_f32 v29, |v142|, |v143|, v29                        // 000000008394: D1D3031D 04771F8E
	v_lshlrev_b32_e32 v56, 3, v0                               // 00000000839C: 24700083  // Left shift (multiply by power of 2)
	s_mul_i32 s60, 0x200, s7                                   // 0000000083A0: 923C07FF 00000200
	v_add_u32_e32 v56, s60, v56                                // 0000000083A8: 6870703C  // 32-bit unsigned add (address calc)

// === LDS WRITE (staging data) ===
	ds_write_b64 v56, v[28:29] offset:18688                    // 0000000083AC: D89A4900 00001C38  // Write 64-bit to LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 0000000083B4: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 0000000083B8: BF8A0000  // Workgroup barrier synchronization
	v_and_b32_e32 v56, 15, v0                                  // 0000000083BC: 2670008F  // Bitwise AND (masking)
	v_lshlrev_b32_e32 v56, 3, v56                              // 0000000083C0: 24707083  // Left shift (multiply by power of 2)
	ds_read_b64 v[96:97], v56 offset:18688                     // 0000000083C4: D8EC4900 60000038  // Read 64-bit from LDS
	ds_read_b64 v[98:99], v56 offset:18816                     // 0000000083CC: D8EC4980 62000038  // Read 64-bit from LDS
	ds_read_b64 v[100:101], v56 offset:18944                   // 0000000083D4: D8EC4A00 64000038  // Read 64-bit from LDS
	ds_read_b64 v[102:103], v56 offset:19072                   // 0000000083DC: D8EC4A80 66000038  // Read 64-bit from LDS
	ds_read_b64 v[104:105], v56 offset:19200                   // 0000000083E4: D8EC4B00 68000038  // Read 64-bit from LDS
	ds_read_b64 v[106:107], v56 offset:19328                   // 0000000083EC: D8EC4B80 6A000038  // Read 64-bit from LDS
	ds_read_b64 v[108:109], v56 offset:19456                   // 0000000083F4: D8EC4C00 6C000038  // Read 64-bit from LDS
	ds_read_b64 v[110:111], v56 offset:19584                   // 0000000083FC: D8EC4C80 6E000038  // Read 64-bit from LDS
	ds_read_b64 v[112:113], v56 offset:19712                   // 000000008404: D8EC4D00 70000038  // Read 64-bit from LDS
	ds_read_b64 v[114:115], v56 offset:19840                   // 00000000840C: D8EC4D80 72000038  // Read 64-bit from LDS
	ds_read_b64 v[116:117], v56 offset:19968                   // 000000008414: D8EC4E00 74000038  // Read 64-bit from LDS
	ds_read_b64 v[118:119], v56 offset:20096                   // 00000000841C: D8EC4E80 76000038  // Read 64-bit from LDS
	ds_read_b64 v[120:121], v56 offset:20224                   // 000000008424: D8EC4F00 78000038  // Read 64-bit from LDS
	ds_read_b64 v[122:123], v56 offset:20352                   // 00000000842C: D8EC4F80 7A000038  // Read 64-bit from LDS
	ds_read_b64 v[124:125], v56 offset:20480                   // 000000008434: D8EC5000 7C000038  // Read 64-bit from LDS
	ds_read_b64 v[126:127], v56 offset:20608                   // 00000000843C: D8EC5080 7E000038  // Read 64-bit from LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 000000008444: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	v_max3_f32 v28, |v96|, |v98|, v28                          // 000000008448: D1D3031C 0472C560
	v_max3_f32 v29, |v97|, |v99|, v29                          // 000000008450: D1D3031D 0476C761
	v_max3_f32 v28, |v100|, |v102|, v28                        // 000000008458: D1D3031C 0472CD64
	v_max3_f32 v29, |v101|, |v103|, v29                        // 000000008460: D1D3031D 0476CF65
	v_max3_f32 v28, |v104|, |v106|, v28                        // 000000008468: D1D3031C 0472D568
	v_max3_f32 v29, |v105|, |v107|, v29                        // 000000008470: D1D3031D 0476D769
	v_max3_f32 v28, |v108|, |v110|, v28                        // 000000008478: D1D3031C 0472DD6C
	v_max3_f32 v29, |v109|, |v111|, v29                        // 000000008480: D1D3031D 0476DF6D
	v_max3_f32 v28, |v112|, |v114|, v28                        // 000000008488: D1D3031C 0472E570
	v_max3_f32 v29, |v113|, |v115|, v29                        // 000000008490: D1D3031D 0476E771
	v_max3_f32 v28, |v116|, |v118|, v28                        // 000000008498: D1D3031C 0472ED74
	v_max3_f32 v29, |v117|, |v119|, v29                        // 0000000084A0: D1D3031D 0476EF75
	v_max3_f32 v28, |v120|, |v122|, v28                        // 0000000084A8: D1D3031C 0472F578
	v_max3_f32 v29, |v121|, |v123|, v29                        // 0000000084B0: D1D3031D 0476F779
	v_max3_f32 v28, |v124|, |v126|, v28                        // 0000000084B8: D1D3031C 0472FD7C
	v_max3_f32 v29, |v125|, |v127|, v29                        // 0000000084C0: D1D3031D 0476FF7D
	v_rcp_f32_e32 v28, v28                                     // 0000000084C8: 7E38451C
	v_rcp_f32_e32 v29, v29                                     // 0000000084CC: 7E3A451D
	v_mov_b32_e32 v56, 0x43700000                              // 0000000084D0: 7E7002FF 43700000  // Vector move
	v_mul_f32_e32 v28, v56, v28                                // 0000000084D8: 0A383938  // FP32 multiply
	v_mul_f32_e32 v29, v56, v29                                // 0000000084DC: 0A3A3B38  // FP32 multiply
	v_mul_f32_e32 v128, v28, v128                              // 0000000084E0: 0B01011C  // FP32 multiply
	v_mul_f32_e32 v129, v28, v129                              // 0000000084E4: 0B03031C  // FP32 multiply
	v_mul_f32_e32 v130, v28, v130                              // 0000000084E8: 0B05051C  // FP32 multiply
	v_mul_f32_e32 v131, v28, v131                              // 0000000084EC: 0B07071C  // FP32 multiply
	v_cvt_pk_fp8_f32 v128, v128, v129                          // 0000000084F0: D2A20080 00030380  // Packed conversion
	v_cvt_pk_fp8_f32 v128, v130, v131 op_sel:[0,0,1]           // 0000000084F8: D2A24080 00030782  // Packed conversion
	v_mul_f32_e32 v132, v29, v132                              // 000000008500: 0B09091D  // FP32 multiply
	v_mul_f32_e32 v133, v29, v133                              // 000000008504: 0B0B0B1D  // FP32 multiply
	v_mul_f32_e32 v134, v29, v134                              // 000000008508: 0B0D0D1D  // FP32 multiply
	v_mul_f32_e32 v135, v29, v135                              // 00000000850C: 0B0F0F1D  // FP32 multiply
	v_cvt_pk_fp8_f32 v129, v132, v133                          // 000000008510: D2A20081 00030B84  // Packed conversion
	v_cvt_pk_fp8_f32 v129, v134, v135 op_sel:[0,0,1]           // 000000008518: D2A24081 00030F86  // Packed conversion
	v_mul_f32_e32 v136, v28, v136                              // 000000008520: 0B11111C  // FP32 multiply
	v_mul_f32_e32 v137, v28, v137                              // 000000008524: 0B13131C  // FP32 multiply
	v_mul_f32_e32 v138, v28, v138                              // 000000008528: 0B15151C  // FP32 multiply
	v_mul_f32_e32 v139, v28, v139                              // 00000000852C: 0B17171C  // FP32 multiply
	v_cvt_pk_fp8_f32 v130, v136, v137                          // 000000008530: D2A20082 00031388  // Packed conversion
	v_cvt_pk_fp8_f32 v130, v138, v139 op_sel:[0,0,1]           // 000000008538: D2A24082 0003178A  // Packed conversion
	v_mul_f32_e32 v140, v29, v140                              // 000000008540: 0B19191D  // FP32 multiply
	v_mul_f32_e32 v141, v29, v141                              // 000000008544: 0B1B1B1D  // FP32 multiply
	v_mul_f32_e32 v142, v29, v142                              // 000000008548: 0B1D1D1D  // FP32 multiply
	v_mul_f32_e32 v143, v29, v143                              // 00000000854C: 0B1F1F1D  // FP32 multiply
	v_cvt_pk_fp8_f32 v131, v140, v141                          // 000000008550: D2A20083 00031B8C  // Packed conversion
	v_cvt_pk_fp8_f32 v131, v142, v143 op_sel:[0,0,1]           // 000000008558: D2A24083 00031F8E  // Packed conversion
	v_rcp_f32_e32 v32, v28                                     // 000000008560: 7E40451C
	v_rcp_f32_e32 v33, v29                                     // 000000008564: 7E42451D
	v_mov_b32_e32 v30, 0x358637bd                              // 000000008568: 7E3C02FF 358637BD  // Vector move
	v_mov_b32_e32 v31, 0x358637bd                              // 000000008570: 7E3E02FF 358637BD  // Vector move
	v_max3_f32 v30, |v144|, |v145|, v30                        // 000000008578: D1D3031E 047B2390
	v_max3_f32 v30, |v146|, |v147|, v30                        // 000000008580: D1D3031E 047B2792
	v_max3_f32 v31, |v148|, |v149|, v31                        // 000000008588: D1D3031F 047F2B94
	v_max3_f32 v31, |v150|, |v151|, v31                        // 000000008590: D1D3031F 047F2F96
	v_max3_f32 v30, |v152|, |v153|, v30                        // 000000008598: D1D3031E 047B3398
	v_max3_f32 v30, |v154|, |v155|, v30                        // 0000000085A0: D1D3031E 047B379A
	v_max3_f32 v31, |v156|, |v157|, v31                        // 0000000085A8: D1D3031F 047F3B9C
	v_max3_f32 v31, |v158|, |v159|, v31                        // 0000000085B0: D1D3031F 047F3F9E
	v_lshlrev_b32_e32 v56, 3, v0                               // 0000000085B8: 24700083  // Left shift (multiply by power of 2)
	s_mul_i32 s60, 0x200, s7                                   // 0000000085BC: 923C07FF 00000200
	v_add_u32_e32 v56, s60, v56                                // 0000000085C4: 6870703C  // 32-bit unsigned add (address calc)
	ds_write_b64 v56, v[30:31] offset:18688                    // 0000000085C8: D89A4900 00001E38  // Write 64-bit to LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 0000000085D0: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 0000000085D4: BF8A0000  // Workgroup barrier synchronization
	v_and_b32_e32 v56, 15, v0                                  // 0000000085D8: 2670008F  // Bitwise AND (masking)
	v_lshlrev_b32_e32 v56, 3, v56                              // 0000000085DC: 24707083  // Left shift (multiply by power of 2)
	ds_read_b64 v[96:97], v56 offset:18688                     // 0000000085E0: D8EC4900 60000038  // Read 64-bit from LDS
	ds_read_b64 v[98:99], v56 offset:18816                     // 0000000085E8: D8EC4980 62000038  // Read 64-bit from LDS
	ds_read_b64 v[100:101], v56 offset:18944                   // 0000000085F0: D8EC4A00 64000038  // Read 64-bit from LDS
	ds_read_b64 v[102:103], v56 offset:19072                   // 0000000085F8: D8EC4A80 66000038  // Read 64-bit from LDS
	ds_read_b64 v[104:105], v56 offset:19200                   // 000000008600: D8EC4B00 68000038  // Read 64-bit from LDS
	ds_read_b64 v[106:107], v56 offset:19328                   // 000000008608: D8EC4B80 6A000038  // Read 64-bit from LDS
	ds_read_b64 v[108:109], v56 offset:19456                   // 000000008610: D8EC4C00 6C000038  // Read 64-bit from LDS
	ds_read_b64 v[110:111], v56 offset:19584                   // 000000008618: D8EC4C80 6E000038  // Read 64-bit from LDS
	ds_read_b64 v[112:113], v56 offset:19712                   // 000000008620: D8EC4D00 70000038  // Read 64-bit from LDS
	ds_read_b64 v[114:115], v56 offset:19840                   // 000000008628: D8EC4D80 72000038  // Read 64-bit from LDS
	ds_read_b64 v[116:117], v56 offset:19968                   // 000000008630: D8EC4E00 74000038  // Read 64-bit from LDS
	ds_read_b64 v[118:119], v56 offset:20096                   // 000000008638: D8EC4E80 76000038  // Read 64-bit from LDS
	ds_read_b64 v[120:121], v56 offset:20224                   // 000000008640: D8EC4F00 78000038  // Read 64-bit from LDS
	ds_read_b64 v[122:123], v56 offset:20352                   // 000000008648: D8EC4F80 7A000038  // Read 64-bit from LDS
	ds_read_b64 v[124:125], v56 offset:20480                   // 000000008650: D8EC5000 7C000038  // Read 64-bit from LDS
	ds_read_b64 v[126:127], v56 offset:20608                   // 000000008658: D8EC5080 7E000038  // Read 64-bit from LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 000000008660: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	v_max3_f32 v30, |v96|, |v98|, v30                          // 000000008664: D1D3031E 047AC560
	v_max3_f32 v31, |v97|, |v99|, v31                          // 00000000866C: D1D3031F 047EC761
	v_max3_f32 v30, |v100|, |v102|, v30                        // 000000008674: D1D3031E 047ACD64
	v_max3_f32 v31, |v101|, |v103|, v31                        // 00000000867C: D1D3031F 047ECF65
	v_max3_f32 v30, |v104|, |v106|, v30                        // 000000008684: D1D3031E 047AD568
	v_max3_f32 v31, |v105|, |v107|, v31                        // 00000000868C: D1D3031F 047ED769
	v_max3_f32 v30, |v108|, |v110|, v30                        // 000000008694: D1D3031E 047ADD6C
	v_max3_f32 v31, |v109|, |v111|, v31                        // 00000000869C: D1D3031F 047EDF6D
	v_max3_f32 v30, |v112|, |v114|, v30                        // 0000000086A4: D1D3031E 047AE570
	v_max3_f32 v31, |v113|, |v115|, v31                        // 0000000086AC: D1D3031F 047EE771
	v_max3_f32 v30, |v116|, |v118|, v30                        // 0000000086B4: D1D3031E 047AED74
	v_max3_f32 v31, |v117|, |v119|, v31                        // 0000000086BC: D1D3031F 047EEF75
	v_max3_f32 v30, |v120|, |v122|, v30                        // 0000000086C4: D1D3031E 047AF578
	v_max3_f32 v31, |v121|, |v123|, v31                        // 0000000086CC: D1D3031F 047EF779
	v_max3_f32 v30, |v124|, |v126|, v30                        // 0000000086D4: D1D3031E 047AFD7C
	v_max3_f32 v31, |v125|, |v127|, v31                        // 0000000086DC: D1D3031F 047EFF7D
	v_rcp_f32_e32 v30, v30                                     // 0000000086E4: 7E3C451E
	v_rcp_f32_e32 v31, v31                                     // 0000000086E8: 7E3E451F
	v_mov_b32_e32 v56, 0x43700000                              // 0000000086EC: 7E7002FF 43700000  // Vector move
	v_mul_f32_e32 v30, v56, v30                                // 0000000086F4: 0A3C3D38  // FP32 multiply
	v_mul_f32_e32 v31, v56, v31                                // 0000000086F8: 0A3E3F38  // FP32 multiply
	v_mul_f32_e32 v144, v30, v144                              // 0000000086FC: 0B21211E  // FP32 multiply
	v_mul_f32_e32 v145, v30, v145                              // 000000008700: 0B23231E  // FP32 multiply
	v_mul_f32_e32 v146, v30, v146                              // 000000008704: 0B25251E  // FP32 multiply
	v_mul_f32_e32 v147, v30, v147                              // 000000008708: 0B27271E  // FP32 multiply
	v_cvt_pk_fp8_f32 v132, v144, v145                          // 00000000870C: D2A20084 00032390  // Packed conversion
	v_cvt_pk_fp8_f32 v132, v146, v147 op_sel:[0,0,1]           // 000000008714: D2A24084 00032792  // Packed conversion
	v_mul_f32_e32 v148, v31, v148                              // 00000000871C: 0B29291F  // FP32 multiply
	v_mul_f32_e32 v149, v31, v149                              // 000000008720: 0B2B2B1F  // FP32 multiply
	v_mul_f32_e32 v150, v31, v150                              // 000000008724: 0B2D2D1F  // FP32 multiply
	v_mul_f32_e32 v151, v31, v151                              // 000000008728: 0B2F2F1F  // FP32 multiply
	v_cvt_pk_fp8_f32 v133, v148, v149                          // 00000000872C: D2A20085 00032B94  // Packed conversion
	v_cvt_pk_fp8_f32 v133, v150, v151 op_sel:[0,0,1]           // 000000008734: D2A24085 00032F96  // Packed conversion
	v_mul_f32_e32 v152, v30, v152                              // 00000000873C: 0B31311E  // FP32 multiply
	v_mul_f32_e32 v153, v30, v153                              // 000000008740: 0B33331E  // FP32 multiply
	v_mul_f32_e32 v154, v30, v154                              // 000000008744: 0B35351E  // FP32 multiply
	v_mul_f32_e32 v155, v30, v155                              // 000000008748: 0B37371E  // FP32 multiply
	v_cvt_pk_fp8_f32 v134, v152, v153                          // 00000000874C: D2A20086 00033398  // Packed conversion
	v_cvt_pk_fp8_f32 v134, v154, v155 op_sel:[0,0,1]           // 000000008754: D2A24086 0003379A  // Packed conversion
	v_mul_f32_e32 v156, v31, v156                              // 00000000875C: 0B39391F  // FP32 multiply
	v_mul_f32_e32 v157, v31, v157                              // 000000008760: 0B3B3B1F  // FP32 multiply
	v_mul_f32_e32 v158, v31, v158                              // 000000008764: 0B3D3D1F  // FP32 multiply
	v_mul_f32_e32 v159, v31, v159                              // 000000008768: 0B3F3F1F  // FP32 multiply
	v_cvt_pk_fp8_f32 v135, v156, v157                          // 00000000876C: D2A20087 00033B9C  // Packed conversion
	v_cvt_pk_fp8_f32 v135, v158, v159 op_sel:[0,0,1]           // 000000008774: D2A24087 00033F9E  // Packed conversion
	v_rcp_f32_e32 v34, v30                                     // 00000000877C: 7E44451E
	v_rcp_f32_e32 v35, v31                                     // 000000008780: 7E46451F
	v_lshrrev_b32_e32 v56, 5, v0                               // 000000008784: 20700085  // Right shift (divide by power of 2)
	v_lshlrev_b32_e32 v57, 5, v56                              // 000000008788: 24727085  // Left shift (multiply by power of 2)
	v_and_b32_e32 v56, 31, v0                                  // 00000000878C: 2670009F  // Bitwise AND (masking)
	v_lshrrev_b32_e32 v58, 4, v56                              // 000000008790: 20747084  // Right shift (divide by power of 2)
	v_add_u32_e32 v57, v58, v57                                // 000000008794: 6872733A  // 32-bit unsigned add (address calc)
	v_and_b32_e32 v56, 15, v0                                  // 000000008798: 2670008F  // Bitwise AND (masking)
	v_lshlrev_b32_e32 v56, 1, v56                              // 00000000879C: 24707081  // Left shift (multiply by power of 2)
	v_add_u32_e32 v57, v56, v57                                // 0000000087A0: 68727338  // 32-bit unsigned add (address calc)
	v_lshlrev_b32_e32 v56, 2, v57                              // 0000000087A4: 24707282  // Left shift (multiply by power of 2)
	s_mul_i32 s60, 0x100, s7                                   // 0000000087A8: 923C07FF 00000100
	v_add_u32_e64 v56, v56, s60                                // 0000000087B0: D1340038 00007938  // 32-bit unsigned add (address calc)
	ds_write_b32 v56, v128 offset:20736                        // 0000000087B8: D81A5100 00008038  // Write 32-bit to LDS
	ds_write_b32 v56, v129 offset:24832                        // 0000000087C0: D81A6100 00008138  // Write 32-bit to LDS
	ds_write_b32 v56, v130 offset:21760                        // 0000000087C8: D81A5500 00008238  // Write 32-bit to LDS
	ds_write_b32 v56, v131 offset:25856                        // 0000000087D0: D81A6500 00008338  // Write 32-bit to LDS
	ds_write_b32 v56, v132 offset:22784                        // 0000000087D8: D81A5900 00008438  // Write 32-bit to LDS
	ds_write_b32 v56, v133 offset:26880                        // 0000000087E0: D81A6900 00008538  // Write 32-bit to LDS
	ds_write_b32 v56, v134 offset:23808                        // 0000000087E8: D81A5D00 00008638  // Write 32-bit to LDS
	ds_write_b32 v56, v135 offset:27904                        // 0000000087F0: D81A6D00 00008738  // Write 32-bit to LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 0000000087F8: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 0000000087FC: BF8A0000  // Workgroup barrier synchronization
	v_lshrrev_b32_e32 v56, 4, v0                               // 000000008800: 20700084  // Right shift (divide by power of 2)
	v_lshlrev_b32_e32 v57, 6, v56                              // 000000008804: 24727086  // Left shift (multiply by power of 2)
	v_and_b32_e32 v56, 15, v0                                  // 000000008808: 2670008F  // Bitwise AND (masking)
	v_lshlrev_b32_e32 v56, 1, v56                              // 00000000880C: 24707081  // Left shift (multiply by power of 2)
	v_add_u32_e32 v57, v56, v57                                // 000000008810: 68727338  // 32-bit unsigned add (address calc)
	v_lshlrev_b32_e32 v56, 2, v57                              // 000000008814: 24707282  // Left shift (multiply by power of 2)
	ds_read_b64 v[128:129], v56 offset:20736                   // 000000008818: D8EC5100 80000038  // Read 64-bit from LDS
	ds_read_b64 v[130:131], v56 offset:20864                   // 000000008820: D8EC5180 82000038  // Read 64-bit from LDS
	ds_read_b64 v[132:133], v56 offset:21760                   // 000000008828: D8EC5500 84000038  // Read 64-bit from LDS
	ds_read_b64 v[134:135], v56 offset:21888                   // 000000008830: D8EC5580 86000038  // Read 64-bit from LDS
	ds_read_b64 v[136:137], v56 offset:22784                   // 000000008838: D8EC5900 88000038  // Read 64-bit from LDS
	ds_read_b64 v[138:139], v56 offset:22912                   // 000000008840: D8EC5980 8A000038  // Read 64-bit from LDS
	ds_read_b64 v[140:141], v56 offset:23808                   // 000000008848: D8EC5D00 8C000038  // Read 64-bit from LDS
	ds_read_b64 v[142:143], v56 offset:23936                   // 000000008850: D8EC5D80 8E000038  // Read 64-bit from LDS
	ds_read_b64 v[144:145], v56 offset:24832                   // 000000008858: D8EC6100 90000038  // Read 64-bit from LDS
	ds_read_b64 v[146:147], v56 offset:24960                   // 000000008860: D8EC6180 92000038  // Read 64-bit from LDS
	ds_read_b64 v[148:149], v56 offset:25856                   // 000000008868: D8EC6500 94000038  // Read 64-bit from LDS
	ds_read_b64 v[150:151], v56 offset:25984                   // 000000008870: D8EC6580 96000038  // Read 64-bit from LDS
	ds_read_b64 v[152:153], v56 offset:26880                   // 000000008878: D8EC6900 98000038  // Read 64-bit from LDS
	ds_read_b64 v[154:155], v56 offset:27008                   // 000000008880: D8EC6980 9A000038  // Read 64-bit from LDS
	ds_read_b64 v[156:157], v56 offset:27904                   // 000000008888: D8EC6D00 9C000038  // Read 64-bit from LDS
	ds_read_b64 v[158:159], v56 offset:28032                   // 000000008890: D8EC6D80 9E000038  // Read 64-bit from LDS
	s_add_u32 s12, s56, s12                                    // 000000008898: 800C0C38
	s_addc_u32 s13, 0, s13                                     // 00000000889C: 820D0D80
	s_add_u32 s16, s79, s16                                    // 0000000088A0: 8010104F
	s_addc_u32 s17, 0, s17                                     // 0000000088A4: 82111180
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 0000000088A8: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 0000000088AC: BF8A0000  // Workgroup barrier synchronization
	v_mov_b32_e32 v192, 0                                      // 0000000088B0: 7F800280  // Vector move
	v_mov_b32_e32 v224, 0                                      // 0000000088B4: 7FC00280  // Vector move
	v_mov_b32_e32 v193, 0                                      // 0000000088B8: 7F820280  // Vector move
	v_mov_b32_e32 v225, 0                                      // 0000000088BC: 7FC20280  // Vector move
	v_mov_b32_e32 v194, 0                                      // 0000000088C0: 7F840280  // Vector move
	v_mov_b32_e32 v226, 0                                      // 0000000088C4: 7FC40280  // Vector move
	v_mov_b32_e32 v195, 0                                      // 0000000088C8: 7F860280  // Vector move
	v_mov_b32_e32 v227, 0                                      // 0000000088CC: 7FC60280  // Vector move
	v_mov_b32_e32 v196, 0                                      // 0000000088D0: 7F880280  // Vector move
	v_mov_b32_e32 v228, 0                                      // 0000000088D4: 7FC80280  // Vector move
	v_mov_b32_e32 v197, 0                                      // 0000000088D8: 7F8A0280  // Vector move
	v_mov_b32_e32 v229, 0                                      // 0000000088DC: 7FCA0280  // Vector move
	v_mov_b32_e32 v198, 0                                      // 0000000088E0: 7F8C0280  // Vector move
	v_mov_b32_e32 v230, 0                                      // 0000000088E4: 7FCC0280  // Vector move
	v_mov_b32_e32 v199, 0                                      // 0000000088E8: 7F8E0280  // Vector move
	v_mov_b32_e32 v231, 0                                      // 0000000088EC: 7FCE0280  // Vector move
	v_mov_b32_e32 v200, 0                                      // 0000000088F0: 7F900280  // Vector move
	v_mov_b32_e32 v232, 0                                      // 0000000088F4: 7FD00280  // Vector move
	v_mov_b32_e32 v201, 0                                      // 0000000088F8: 7F920280  // Vector move
	v_mov_b32_e32 v233, 0                                      // 0000000088FC: 7FD20280  // Vector move
	v_mov_b32_e32 v202, 0                                      // 000000008900: 7F940280  // Vector move
	v_mov_b32_e32 v234, 0                                      // 000000008904: 7FD40280  // Vector move
	v_mov_b32_e32 v203, 0                                      // 000000008908: 7F960280  // Vector move
	v_mov_b32_e32 v235, 0                                      // 00000000890C: 7FD60280  // Vector move
	v_mov_b32_e32 v204, 0                                      // 000000008910: 7F980280  // Vector move
	v_mov_b32_e32 v236, 0                                      // 000000008914: 7FD80280  // Vector move
	v_mov_b32_e32 v205, 0                                      // 000000008918: 7F9A0280  // Vector move
	v_mov_b32_e32 v237, 0                                      // 00000000891C: 7FDA0280  // Vector move
	v_mov_b32_e32 v206, 0                                      // 000000008920: 7F9C0280  // Vector move
	v_mov_b32_e32 v238, 0                                      // 000000008924: 7FDC0280  // Vector move
	v_mov_b32_e32 v207, 0                                      // 000000008928: 7F9E0280  // Vector move
	v_mov_b32_e32 v239, 0                                      // 00000000892C: 7FDE0280  // Vector move
	ds_write_b64 v4, v[192:193] offset:20736                   // 000000008930: D89A5100 0000C004  // Write 64-bit to LDS
	ds_write_b64 v4, v[194:195] offset:29440                   // 000000008938: D89A7300 0000C204  // Write 64-bit to LDS
	ds_write_b64 v4, v[196:197] offset:22912                   // 000000008940: D89A5980 0000C404  // Write 64-bit to LDS
	ds_write_b64 v4, v[198:199] offset:31616                   // 000000008948: D89A7B80 0000C604  // Write 64-bit to LDS
	ds_write_b64 v4, v[200:201] offset:25088                   // 000000008950: D89A6200 0000C804  // Write 64-bit to LDS
	ds_write_b64 v4, v[202:203] offset:33792                   // 000000008958: D89A8400 0000CA04  // Write 64-bit to LDS
	ds_write_b64 v4, v[204:205] offset:27264                   // 000000008960: D89A6A80 0000CC04  // Write 64-bit to LDS
	ds_write_b64 v4, v[206:207] offset:35968                   // 000000008968: D89A8C80 0000CE04  // Write 64-bit to LDS
	s_mov_b32 s80, 0                                           // 000000008970: BED00080  // Scalar move (constant/register)
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 000000008974: BF8C0000  // Wait for memory operations (CRITICAL for perf)


// --- LABEL (potential loop target) ---
0000000000008978 <label_16DE>:
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(12) lgkmcnt(0)                             // 000000008978: BF8C007C  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 00000000897C: BF8A0000  // Workgroup barrier synchronization

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[192:195], a[0:1], v[128:129], 0// 000000008980: D3F300C0 0A030100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[64:67], v48, s[12:15], 0 offen       // 000000008988: E05C1000 80834030
	v_mfma_f32_16x16x32_fp8_fp8 v[192:195], a[2:3], v[130:131], v[192:195]// 000000008990: D3F300C0 0F030502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v64, v5 offset:20736                           // 000000008998: D86C5100 40000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v65, v5 offset:25088                           // 0000000089A0: D86C6200 41000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[196:199], a[0:1], v[144:145], 0// 0000000089A8: D3F300C4 0A032100  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v24, v6, s[16:19], 0 offen               // 0000000089B0: E0501000 80041806
	v_mfma_f32_16x16x32_fp8_fp8 v[196:199], a[2:3], v[146:147], v[196:199]// 0000000089B8: D3F300C4 0F132502  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v66, v5 offset:20768                           // 0000000089C0: D86C5120 42000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v67, v5 offset:25120                           // 0000000089C8: D86C6220 43000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[200:203], a[4:5], v[128:129], 0// 0000000089D0: D3F300C8 0A030104  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[68:71], v49, s[12:15], 0 offen       // 0000000089D8: E05C1000 80834431
	v_mfma_f32_16x16x32_fp8_fp8 v[200:203], a[6:7], v[130:131], v[200:203]// 0000000089E0: D3F300C8 0F230506  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v68, v5 offset:20800                           // 0000000089E8: D86C5140 44000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v69, v5 offset:25152                           // 0000000089F0: D86C6240 45000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[204:207], a[4:5], v[144:145], 0// 0000000089F8: D3F300CC 0A032104  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[204:207], a[6:7], v[146:147], v[204:207]// 000000008A00: D3F300CC 0F332506  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v70, v5 offset:20832                           // 000000008A08: D86C5160 46000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v71, v5 offset:25184                           // 000000008A10: D86C6260 47000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[208:211], a[8:9], v[128:129], 0// 000000008A18: D3F300D0 0A030108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[72:75], v50, s[12:15], 0 offen       // 000000008A20: E05C1000 80834832
	v_mfma_f32_16x16x32_fp8_fp8 v[208:211], a[10:11], v[130:131], v[208:211]// 000000008A28: D3F300D0 0F43050A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v72, v5 offset:29440                           // 000000008A30: D86C7300 48000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v73, v5 offset:33792                           // 000000008A38: D86C8400 49000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[212:215], a[8:9], v[144:145], 0// 000000008A40: D3F300D4 0A032108  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[212:215], a[10:11], v[146:147], v[212:215]// 000000008A48: D3F300D4 0F53250A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v74, v5 offset:29472                           // 000000008A50: D86C7320 4A000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v75, v5 offset:33824                           // 000000008A58: D86C8420 4B000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[216:219], a[12:13], v[128:129], 0// 000000008A60: D3F300D8 0A03010C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[76:79], v51, s[12:15], 0 offen       // 000000008A68: E05C1000 80834C33
	s_add_u32 s12, s78, s12                                    // 000000008A70: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000008A74: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[216:219], a[14:15], v[130:131], v[216:219]// 000000008A78: D3F300D8 0F63050E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v76, v5 offset:29504                           // 000000008A80: D86C7340 4C000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v77, v5 offset:33856                           // 000000008A88: D86C8440 4D000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[220:223], a[12:13], v[144:145], 0// 000000008A90: D3F300DC 0A03210C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[220:223], a[14:15], v[146:147], v[220:223]// 000000008A98: D3F300DC 0F73250E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v78, v5 offset:29536                           // 000000008AA0: D86C7360 4E000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v79, v5 offset:33888                           // 000000008AA8: D86C8460 4F000005  // Read 32-bit from LDS (shared memory)
	s_waitcnt vmcnt(13)                                        // 000000008AB0: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[192:195], a[16:17], v[132:133], v[192:195]// 000000008AB4: D3F300C0 0F030910  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[80:83], v48, s[12:15], 0 offen       // 000000008ABC: E05C1000 80835030
	v_mfma_f32_16x16x32_fp8_fp8 v[192:195], a[18:19], v[134:135], v[192:195]// 000000008AC4: D3F300C0 0F030D12  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[196:199], a[16:17], v[148:149], v[196:199]// 000000008ACC: D3F300C4 0F132910  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[196:199], a[18:19], v[150:151], v[196:199]// 000000008AD4: D3F300C4 0F132D12  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[200:203], a[20:21], v[132:133], v[200:203]// 000000008ADC: D3F300C8 0F230914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[84:87], v49, s[12:15], 0 offen       // 000000008AE4: E05C1000 80835431
	v_mfma_f32_16x16x32_fp8_fp8 v[200:203], a[22:23], v[134:135], v[200:203]// 000000008AEC: D3F300C8 0F230D16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[204:207], a[20:21], v[148:149], v[204:207]// 000000008AF4: D3F300CC 0F332914  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[204:207], a[22:23], v[150:151], v[204:207]// 000000008AFC: D3F300CC 0F332D16  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[208:211], a[24:25], v[132:133], v[208:211]// 000000008B04: D3F300D0 0F430918  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[88:91], v50, s[12:15], 0 offen       // 000000008B0C: E05C1000 80835832
	v_mfma_f32_16x16x32_fp8_fp8 v[208:211], a[26:27], v[134:135], v[208:211]// 000000008B14: D3F300D0 0F430D1A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[212:215], a[24:25], v[148:149], v[212:215]// 000000008B1C: D3F300D4 0F532918  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[212:215], a[26:27], v[150:151], v[212:215]// 000000008B24: D3F300D4 0F532D1A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[216:219], a[28:29], v[132:133], v[216:219]// 000000008B2C: D3F300D8 0F63091C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[92:95], v51, s[12:15], 0 offen       // 000000008B34: E05C1000 80835C33
	s_add_u32 s12, s78, s12                                    // 000000008B3C: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000008B40: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[216:219], a[30:31], v[134:135], v[216:219]// 000000008B44: D3F300D8 0F630D1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[220:223], a[28:29], v[148:149], v[220:223]// 000000008B4C: D3F300DC 0F73291C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[220:223], a[30:31], v[150:151], v[220:223]// 000000008B54: D3F300DC 0F732D1E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v23, v32 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000008B5C: 0A7040FA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000008B64: 7E720338  // Vector move
	v_pk_mul_f32 v[192:193], v[56:57], v[192:193]              // 000000008B68: D3B140C0 18038138
	v_pk_mul_f32 v[194:195], v[56:57], v[194:195]              // 000000008B70: D3B140C2 18038538
	v_pk_mul_f32 v[200:201], v[56:57], v[200:201]              // 000000008B78: D3B140C8 18039138
	v_pk_mul_f32 v[202:203], v[56:57], v[202:203]              // 000000008B80: D3B140CA 18039538
	v_mul_f32_dpp v56, v23, v32 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000008B88: 0A7040FA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000008B90: 7E720338  // Vector move
	v_pk_mul_f32 v[208:209], v[56:57], v[208:209]              // 000000008B94: D3B140D0 1803A138
	v_pk_mul_f32 v[210:211], v[56:57], v[210:211]              // 000000008B9C: D3B140D2 1803A538
	v_pk_mul_f32 v[216:217], v[56:57], v[216:217]              // 000000008BA4: D3B140D8 1803B138
	v_pk_mul_f32 v[218:219], v[56:57], v[218:219]              // 000000008BAC: D3B140DA 1803B538
	v_mul_f32_dpp v56, v23, v33 row_newbcast:0 row_mask:0xf bank_mask:0xf// 000000008BB4: 0A7042FA FF015017  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000008BBC: 7E720338  // Vector move
	v_pk_mul_f32 v[196:197], v[56:57], v[196:197]              // 000000008BC0: D3B140C4 18038938
	v_pk_mul_f32 v[198:199], v[56:57], v[198:199]              // 000000008BC8: D3B140C6 18038D38
	v_pk_mul_f32 v[204:205], v[56:57], v[204:205]              // 000000008BD0: D3B140CC 18039938
	v_pk_mul_f32 v[206:207], v[56:57], v[206:207]              // 000000008BD8: D3B140CE 18039D38
	v_mul_f32_dpp v56, v23, v33 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000008BE0: 0A7042FA FF015117  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000008BE8: 7E720338  // Vector move
	v_pk_mul_f32 v[212:213], v[56:57], v[212:213]              // 000000008BEC: D3B140D4 1803A938
	v_pk_mul_f32 v[214:215], v[56:57], v[214:215]              // 000000008BF4: D3B140D6 1803AD38
	v_pk_mul_f32 v[220:221], v[56:57], v[220:221]              // 000000008BFC: D3B140DC 1803B938
	v_pk_mul_f32 v[222:223], v[56:57], v[222:223]              // 000000008C04: D3B140DE 1803BD38
	s_waitcnt vmcnt(13)                                        // 000000008C0C: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[32:33], v[136:137], 0// 000000008C10: D3F300A0 0A031120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[96:99], v48, s[12:15], 0 offen       // 000000008C18: E05C1000 80836030
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[34:35], v[138:139], v[160:163]// 000000008C20: D3F300A0 0E831522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[224:225] offset:38144                   // 000000008C28: D89A9500 0000E004  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[32:33], v[152:153], 0// 000000008C30: D3F300A4 0A033120  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[34:35], v[154:155], v[164:167]// 000000008C38: D3F300A4 0E933522  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[226:227] offset:46848                   // 000000008C40: D89AB700 0000E204  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[36:37], v[136:137], 0// 000000008C48: D3F300A8 0A031124  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[100:103], v49, s[12:15], 0 offen     // 000000008C50: E05C1000 80836431
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[38:39], v[138:139], v[168:171]// 000000008C58: D3F300A8 0EA31526  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[228:229] offset:40320                   // 000000008C60: D89A9D80 0000E404  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[36:37], v[152:153], 0// 000000008C68: D3F300AC 0A033124  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[38:39], v[154:155], v[172:175]// 000000008C70: D3F300AC 0EB33526  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[230:231] offset:49024                   // 000000008C78: D89ABF80 0000E604  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[40:41], v[136:137], 0// 000000008C80: D3F300B0 0A031128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[104:107], v50, s[12:15], 0 offen     // 000000008C88: E05C1000 80836832
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[42:43], v[138:139], v[176:179]// 000000008C90: D3F300B0 0EC3152A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[232:233] offset:42496                   // 000000008C98: D89AA600 0000E804  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[40:41], v[152:153], 0// 000000008CA0: D3F300B4 0A033128  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[42:43], v[154:155], v[180:183]// 000000008CA8: D3F300B4 0ED3352A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[234:235] offset:51200                   // 000000008CB0: D89AC800 0000EA04  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[44:45], v[136:137], 0// 000000008CB8: D3F300B8 0A03112C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[108:111], v51, s[12:15], 0 offen     // 000000008CC0: E05C1000 80836C33
	s_add_u32 s12, s78, s12                                    // 000000008CC8: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000008CCC: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[46:47], v[138:139], v[184:187]// 000000008CD0: D3F300B8 0EE3152E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[236:237] offset:44672                   // 000000008CD8: D89AAE80 0000EC04  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[44:45], v[152:153], 0// 000000008CE0: D3F300BC 0A03312C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[46:47], v[154:155], v[188:191]// 000000008CE8: D3F300BC 0EF3352E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[238:239] offset:53376                   // 000000008CF0: D89AD080 0000EE04  // Write 64-bit to LDS
	s_waitcnt vmcnt(13)                                        // 000000008CF8: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[48:49], v[140:141], v[160:163]// 000000008CFC: D3F300A0 0E831930  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[112:115], v48, s[12:15], 0 offen     // 000000008D04: E05C1000 80837030
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[50:51], v[142:143], v[160:163]// 000000008D0C: D3F300A0 0E831D32  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[48:49], v[156:157], v[164:167]// 000000008D14: D3F300A4 0E933930  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[50:51], v[158:159], v[164:167]// 000000008D1C: D3F300A4 0E933D32  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[52:53], v[140:141], v[168:171]// 000000008D24: D3F300A8 0EA31934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[116:119], v49, s[12:15], 0 offen     // 000000008D2C: E05C1000 80837431
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[54:55], v[142:143], v[168:171]// 000000008D34: D3F300A8 0EA31D36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[52:53], v[156:157], v[172:175]// 000000008D3C: D3F300AC 0EB33934  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[54:55], v[158:159], v[172:175]// 000000008D44: D3F300AC 0EB33D36  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[56:57], v[140:141], v[176:179]// 000000008D4C: D3F300B0 0EC31938  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[120:123], v50, s[12:15], 0 offen     // 000000008D54: E05C1000 80837832
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[58:59], v[142:143], v[176:179]// 000000008D5C: D3F300B0 0EC31D3A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[56:57], v[156:157], v[180:183]// 000000008D64: D3F300B4 0ED33938  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[58:59], v[158:159], v[180:183]// 000000008D6C: D3F300B4 0ED33D3A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[60:61], v[140:141], v[184:187]// 000000008D74: D3F300B8 0EE3193C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[124:127], v51, s[12:15], 0 offen     // 000000008D7C: E05C1000 80837C33
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[62:63], v[142:143], v[184:187]// 000000008D84: D3F300B8 0EE31D3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[60:61], v[156:157], v[188:191]// 000000008D8C: D3F300BC 0EF3393C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[62:63], v[158:159], v[188:191]// 000000008D94: D3F300BC 0EF33D3E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v23, v34 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000008D9C: 0A7044FA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000008DA4: 7E720338  // Vector move
	v_pk_fma_f32 v[192:193], v[160:161], v[56:57], v[192:193]  // 000000008DA8: D3B040C0 1F0271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[194:195], v[162:163], v[56:57], v[194:195]  // 000000008DB0: D3B040C2 1F0A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[200:201], v[168:169], v[56:57], v[200:201]  // 000000008DB8: D3B040C8 1F2271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[202:203], v[170:171], v[56:57], v[202:203]  // 000000008DC0: D3B040CA 1F2A71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v34 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000008DC8: 0A7044FA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000008DD0: 7E720338  // Vector move
	v_pk_fma_f32 v[208:209], v[176:177], v[56:57], v[208:209]  // 000000008DD4: D3B040D0 1F4271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[210:211], v[178:179], v[56:57], v[210:211]  // 000000008DDC: D3B040D2 1F4A71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[216:217], v[184:185], v[56:57], v[216:217]  // 000000008DE4: D3B040D8 1F6271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[218:219], v[186:187], v[56:57], v[218:219]  // 000000008DEC: D3B040DA 1F6A71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v35 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000008DF4: 0A7046FA FF015217  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000008DFC: 7E720338  // Vector move
	v_pk_fma_f32 v[196:197], v[164:165], v[56:57], v[196:197]  // 000000008E00: D3B040C4 1F1271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[198:199], v[166:167], v[56:57], v[198:199]  // 000000008E08: D3B040C6 1F1A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[204:205], v[172:173], v[56:57], v[204:205]  // 000000008E10: D3B040CC 1F3271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[206:207], v[174:175], v[56:57], v[206:207]  // 000000008E18: D3B040CE 1F3A71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v23, v35 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000008E20: 0A7046FA FF015317  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000008E28: 7E720338  // Vector move
	v_pk_fma_f32 v[212:213], v[180:181], v[56:57], v[212:213]  // 000000008E2C: D3B040D4 1F5271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[214:215], v[182:183], v[56:57], v[214:215]  // 000000008E34: D3B040D6 1F5A71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[220:221], v[188:189], v[56:57], v[220:221]  // 000000008E3C: D3B040DC 1F7271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[222:223], v[190:191], v[56:57], v[222:223]  // 000000008E44: D3B040DE 1F7A71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x200, s80                                  // 000000008E4C: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 000000008E54: BF0A513C
	s_cselect_b32 s56, s56, 0                                  // 000000008E58: 85388038
	s_cselect_b32 s78, s78, 0                                  // 000000008E5C: 854E804E
	s_cselect_b32 s79, s79, 0                                  // 000000008E60: 854F804F
	s_add_u32 s12, s56, s12                                    // 000000008E64: 800C0C38
	s_addc_u32 s13, 0, s13                                     // 000000008E68: 820D0D80
	s_add_u32 s16, s79, s16                                    // 000000008E6C: 8010104F
	s_addc_u32 s17, 0, s17                                     // 000000008E70: 82111180
	v_mov_b32_e32 v56, v25                                     // 000000008E74: 7E700319  // Vector move
	v_mov_b32_e32 v57, v25                                     // 000000008E78: 7E720319  // Vector move
	v_pk_mul_f32 v[192:193], v[56:57], v[192:193]              // 000000008E7C: D3B140C0 18038138
	v_pk_mul_f32 v[194:195], v[56:57], v[194:195]              // 000000008E84: D3B140C2 18038538
	v_pk_mul_f32 v[200:201], v[56:57], v[200:201]              // 000000008E8C: D3B140C8 18039138
	v_pk_mul_f32 v[202:203], v[56:57], v[202:203]              // 000000008E94: D3B140CA 18039538
	v_pk_mul_f32 v[208:209], v[56:57], v[208:209]              // 000000008E9C: D3B140D0 1803A138
	v_pk_mul_f32 v[210:211], v[56:57], v[210:211]              // 000000008EA4: D3B140D2 1803A538
	v_pk_mul_f32 v[216:217], v[56:57], v[216:217]              // 000000008EAC: D3B140D8 1803B138
	v_pk_mul_f32 v[218:219], v[56:57], v[218:219]              // 000000008EB4: D3B140DA 1803B538
	v_mov_b32_e32 v56, v26                                     // 000000008EBC: 7E70031A  // Vector move
	v_mov_b32_e32 v57, v26                                     // 000000008EC0: 7E72031A  // Vector move
	v_pk_mul_f32 v[196:197], v[56:57], v[196:197]              // 000000008EC4: D3B140C4 18038938
	v_pk_mul_f32 v[198:199], v[56:57], v[198:199]              // 000000008ECC: D3B140C6 18038D38
	v_pk_mul_f32 v[204:205], v[56:57], v[204:205]              // 000000008ED4: D3B140CC 18039938
	v_pk_mul_f32 v[206:207], v[56:57], v[206:207]              // 000000008EDC: D3B140CE 18039D38
	v_pk_mul_f32 v[212:213], v[56:57], v[212:213]              // 000000008EE4: D3B140D4 1803A938
	v_pk_mul_f32 v[214:215], v[56:57], v[214:215]              // 000000008EEC: D3B140D6 1803AD38
	v_pk_mul_f32 v[220:221], v[56:57], v[220:221]              // 000000008EF4: D3B140DC 1803B938
	v_pk_mul_f32 v[222:223], v[56:57], v[222:223]              // 000000008EFC: D3B140DE 1803BD38
	v_cmp_u_f32_e64 s[48:49], v192, v192                       // 000000008F04: D0480030 000381C0
	v_add3_u32 v52, v192, v55, 1                               // 000000008F0C: D1FF0034 02066FC0
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000008F14: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v193, v193                       // 000000008F1C: D0480030 000383C1
	v_add3_u32 v52, v193, v55, 1                               // 000000008F24: D1FF0034 02066FC1
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000008F2C: D1000039 00C26D34
	v_perm_b32 v192, v57, v56, s52                             // 000000008F34: D1ED00C0 00D27139
	v_cmp_u_f32_e64 s[48:49], v194, v194                       // 000000008F3C: D0480030 000385C2
	v_add3_u32 v52, v194, v55, 1                               // 000000008F44: D1FF0034 02066FC2
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000008F4C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v195, v195                       // 000000008F54: D0480030 000387C3
	v_add3_u32 v52, v195, v55, 1                               // 000000008F5C: D1FF0034 02066FC3
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000008F64: D1000039 00C26D34
	v_perm_b32 v193, v57, v56, s52                             // 000000008F6C: D1ED00C1 00D27139
	v_cmp_u_f32_e64 s[48:49], v196, v196                       // 000000008F74: D0480030 000389C4
	v_add3_u32 v52, v196, v55, 1                               // 000000008F7C: D1FF0034 02066FC4
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000008F84: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v197, v197                       // 000000008F8C: D0480030 00038BC5
	v_add3_u32 v52, v197, v55, 1                               // 000000008F94: D1FF0034 02066FC5
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000008F9C: D1000039 00C26D34
	v_perm_b32 v194, v57, v56, s52                             // 000000008FA4: D1ED00C2 00D27139
	v_cmp_u_f32_e64 s[48:49], v198, v198                       // 000000008FAC: D0480030 00038DC6
	v_add3_u32 v52, v198, v55, 1                               // 000000008FB4: D1FF0034 02066FC6
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000008FBC: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v199, v199                       // 000000008FC4: D0480030 00038FC7
	v_add3_u32 v52, v199, v55, 1                               // 000000008FCC: D1FF0034 02066FC7
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000008FD4: D1000039 00C26D34
	v_perm_b32 v195, v57, v56, s52                             // 000000008FDC: D1ED00C3 00D27139
	v_cmp_u_f32_e64 s[48:49], v200, v200                       // 000000008FE4: D0480030 000391C8
	v_add3_u32 v52, v200, v55, 1                               // 000000008FEC: D1FF0034 02066FC8
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000008FF4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v201, v201                       // 000000008FFC: D0480030 000393C9
	v_add3_u32 v52, v201, v55, 1                               // 000000009004: D1FF0034 02066FC9
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 00000000900C: D1000039 00C26D34
	v_perm_b32 v196, v57, v56, s52                             // 000000009014: D1ED00C4 00D27139
	v_cmp_u_f32_e64 s[48:49], v202, v202                       // 00000000901C: D0480030 000395CA
	v_add3_u32 v52, v202, v55, 1                               // 000000009024: D1FF0034 02066FCA
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 00000000902C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v203, v203                       // 000000009034: D0480030 000397CB
	v_add3_u32 v52, v203, v55, 1                               // 00000000903C: D1FF0034 02066FCB
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009044: D1000039 00C26D34
	v_perm_b32 v197, v57, v56, s52                             // 00000000904C: D1ED00C5 00D27139
	v_cmp_u_f32_e64 s[48:49], v204, v204                       // 000000009054: D0480030 000399CC
	v_add3_u32 v52, v204, v55, 1                               // 00000000905C: D1FF0034 02066FCC
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009064: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v205, v205                       // 00000000906C: D0480030 00039BCD
	v_add3_u32 v52, v205, v55, 1                               // 000000009074: D1FF0034 02066FCD
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 00000000907C: D1000039 00C26D34
	v_perm_b32 v198, v57, v56, s52                             // 000000009084: D1ED00C6 00D27139
	v_cmp_u_f32_e64 s[48:49], v206, v206                       // 00000000908C: D0480030 00039DCE
	v_add3_u32 v52, v206, v55, 1                               // 000000009094: D1FF0034 02066FCE
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 00000000909C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v207, v207                       // 0000000090A4: D0480030 00039FCF
	v_add3_u32 v52, v207, v55, 1                               // 0000000090AC: D1FF0034 02066FCF
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 0000000090B4: D1000039 00C26D34
	v_perm_b32 v199, v57, v56, s52                             // 0000000090BC: D1ED00C7 00D27139
	v_cmp_u_f32_e64 s[48:49], v208, v208                       // 0000000090C4: D0480030 0003A1D0
	v_add3_u32 v52, v208, v55, 1                               // 0000000090CC: D1FF0034 02066FD0
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000090D4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v209, v209                       // 0000000090DC: D0480030 0003A3D1
	v_add3_u32 v52, v209, v55, 1                               // 0000000090E4: D1FF0034 02066FD1
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 0000000090EC: D1000039 00C26D34
	v_perm_b32 v200, v57, v56, s52                             // 0000000090F4: D1ED00C8 00D27139
	v_cmp_u_f32_e64 s[48:49], v210, v210                       // 0000000090FC: D0480030 0003A5D2
	v_add3_u32 v52, v210, v55, 1                               // 000000009104: D1FF0034 02066FD2
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 00000000910C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v211, v211                       // 000000009114: D0480030 0003A7D3
	v_add3_u32 v52, v211, v55, 1                               // 00000000911C: D1FF0034 02066FD3
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009124: D1000039 00C26D34
	v_perm_b32 v201, v57, v56, s52                             // 00000000912C: D1ED00C9 00D27139
	v_cmp_u_f32_e64 s[48:49], v212, v212                       // 000000009134: D0480030 0003A9D4
	v_add3_u32 v52, v212, v55, 1                               // 00000000913C: D1FF0034 02066FD4
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009144: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v213, v213                       // 00000000914C: D0480030 0003ABD5
	v_add3_u32 v52, v213, v55, 1                               // 000000009154: D1FF0034 02066FD5
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 00000000915C: D1000039 00C26D34
	v_perm_b32 v202, v57, v56, s52                             // 000000009164: D1ED00CA 00D27139
	v_cmp_u_f32_e64 s[48:49], v214, v214                       // 00000000916C: D0480030 0003ADD6
	v_add3_u32 v52, v214, v55, 1                               // 000000009174: D1FF0034 02066FD6
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 00000000917C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v215, v215                       // 000000009184: D0480030 0003AFD7
	v_add3_u32 v52, v215, v55, 1                               // 00000000918C: D1FF0034 02066FD7
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009194: D1000039 00C26D34
	v_perm_b32 v203, v57, v56, s52                             // 00000000919C: D1ED00CB 00D27139
	v_cmp_u_f32_e64 s[48:49], v216, v216                       // 0000000091A4: D0480030 0003B1D8
	v_add3_u32 v52, v216, v55, 1                               // 0000000091AC: D1FF0034 02066FD8
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000091B4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v217, v217                       // 0000000091BC: D0480030 0003B3D9
	v_add3_u32 v52, v217, v55, 1                               // 0000000091C4: D1FF0034 02066FD9
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 0000000091CC: D1000039 00C26D34
	v_perm_b32 v204, v57, v56, s52                             // 0000000091D4: D1ED00CC 00D27139
	v_cmp_u_f32_e64 s[48:49], v218, v218                       // 0000000091DC: D0480030 0003B5DA
	v_add3_u32 v52, v218, v55, 1                               // 0000000091E4: D1FF0034 02066FDA
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000091EC: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v219, v219                       // 0000000091F4: D0480030 0003B7DB
	v_add3_u32 v52, v219, v55, 1                               // 0000000091FC: D1FF0034 02066FDB
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009204: D1000039 00C26D34
	v_perm_b32 v205, v57, v56, s52                             // 00000000920C: D1ED00CD 00D27139
	v_cmp_u_f32_e64 s[48:49], v220, v220                       // 000000009214: D0480030 0003B9DC
	v_add3_u32 v52, v220, v55, 1                               // 00000000921C: D1FF0034 02066FDC
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009224: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v221, v221                       // 00000000922C: D0480030 0003BBDD
	v_add3_u32 v52, v221, v55, 1                               // 000000009234: D1FF0034 02066FDD
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 00000000923C: D1000039 00C26D34
	v_perm_b32 v206, v57, v56, s52                             // 000000009244: D1ED00CE 00D27139
	v_cmp_u_f32_e64 s[48:49], v222, v222                       // 00000000924C: D0480030 0003BDDE
	v_add3_u32 v52, v222, v55, 1                               // 000000009254: D1FF0034 02066FDE
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 00000000925C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v223, v223                       // 000000009264: D0480030 0003BFDF
	v_add3_u32 v52, v223, v55, 1                               // 00000000926C: D1FF0034 02066FDF
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009274: D1000039 00C26D34
	v_perm_b32 v207, v57, v56, s52                             // 00000000927C: D1ED00CF 00D27139
	s_cmp_ge_u32 s80, 0x200                                    // 000000009284: BF09FF50 00000200
	s_cselect_b32 s59, 0x200, s59                              // 00000000928C: 853B3BFF 00000200
	s_setvskip s20, 0                                          // 000000009294: BF108014
	global_atomic_pk_add_bf16 v80, v64, s[8:9]                 // 000000009298: DD488000 00084050
	s_setvskip 0, 0                                            // 0000000092A0: BF108080
	s_setvskip s20, 0                                          // 0000000092A4: BF108014
	global_atomic_pk_add_bf16 v80, v65, s[8:9] offset:256      // 0000000092A8: DD488100 00084150
	s_setvskip 0, 0                                            // 0000000092B0: BF108080
	s_setvskip s20, 1                                          // 0000000092B4: BF108114
	global_atomic_pk_add_bf16 v82, v66, s[8:9]                 // 0000000092B8: DD488000 00084252
	s_setvskip 0, 0                                            // 0000000092C0: BF108080
	s_setvskip s20, 1                                          // 0000000092C4: BF108114
	global_atomic_pk_add_bf16 v82, v67, s[8:9] offset:256      // 0000000092C8: DD488100 00084352
	s_setvskip 0, 0                                            // 0000000092D0: BF108080
	s_setvskip s20, 2                                          // 0000000092D4: BF108214
	global_atomic_pk_add_bf16 v84, v68, s[8:9]                 // 0000000092D8: DD488000 00084454
	s_setvskip 0, 0                                            // 0000000092E0: BF108080
	s_setvskip s20, 2                                          // 0000000092E4: BF108214
	global_atomic_pk_add_bf16 v84, v69, s[8:9] offset:256      // 0000000092E8: DD488100 00084554
	s_setvskip 0, 0                                            // 0000000092F0: BF108080
	s_setvskip s20, 3                                          // 0000000092F4: BF108314
	global_atomic_pk_add_bf16 v86, v70, s[8:9]                 // 0000000092F8: DD488000 00084656
	s_setvskip 0, 0                                            // 000000009300: BF108080
	s_setvskip s20, 3                                          // 000000009304: BF108314
	global_atomic_pk_add_bf16 v86, v71, s[8:9] offset:256      // 000000009308: DD488100 00084756
	s_setvskip 0, 0                                            // 000000009310: BF108080
	s_setvskip s20, 4                                          // 000000009314: BF108414
	global_atomic_pk_add_bf16 v88, v72, s[8:9]                 // 000000009318: DD488000 00084858
	s_setvskip 0, 0                                            // 000000009320: BF108080
	s_setvskip s20, 4                                          // 000000009324: BF108414
	global_atomic_pk_add_bf16 v88, v73, s[8:9] offset:256      // 000000009328: DD488100 00084958
	s_setvskip 0, 0                                            // 000000009330: BF108080
	s_setvskip s20, 5                                          // 000000009334: BF108514
	global_atomic_pk_add_bf16 v90, v74, s[8:9]                 // 000000009338: DD488000 00084A5A
	s_setvskip 0, 0                                            // 000000009340: BF108080
	s_setvskip s20, 5                                          // 000000009344: BF108514
	global_atomic_pk_add_bf16 v90, v75, s[8:9] offset:256      // 000000009348: DD488100 00084B5A
	s_setvskip 0, 0                                            // 000000009350: BF108080
	s_setvskip s20, 6                                          // 000000009354: BF108614
	global_atomic_pk_add_bf16 v92, v76, s[8:9]                 // 000000009358: DD488000 00084C5C
	s_setvskip 0, 0                                            // 000000009360: BF108080
	s_setvskip s20, 6                                          // 000000009364: BF108614
	global_atomic_pk_add_bf16 v92, v77, s[8:9] offset:256      // 000000009368: DD488100 00084D5C
	s_setvskip 0, 0                                            // 000000009370: BF108080
	s_setvskip s20, 7                                          // 000000009374: BF108714
	global_atomic_pk_add_bf16 v94, v78, s[8:9]                 // 000000009378: DD488000 00084E5E
	s_setvskip 0, 0                                            // 000000009380: BF108080
	s_setvskip s20, 7                                          // 000000009384: BF108714
	global_atomic_pk_add_bf16 v94, v79, s[8:9] offset:256      // 000000009388: DD488100 00084F5E
	s_setvskip 0, 0                                            // 000000009390: BF108080
	s_add_u32 s8, s59, s8                                      // 000000009394: 8008083B
	s_addc_u32 s9, 0, s9                                       // 000000009398: 82090980
	s_addk_i32 s80, 0x100                                      // 00000000939C: B7500100
	s_cmp_lt_i32 s80, s81                                      // 0000000093A0: BF045150
	s_cbranch_scc0 label_0F49                                  // 0000000093A4: BF84F5DF  // Branch if SCC==0 (condition false)
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(12) lgkmcnt(0)                             // 0000000093A8: BF8C007C  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 0000000093AC: BF8A0000  // Workgroup barrier synchronization
	v_mfma_f32_16x16x32_fp8_fp8 v[224:227], a[64:65], v[128:129], 0// 0000000093B0: D3F300E0 0A030140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[0:3], v48, s[12:15], 0 offen         // 0000000093B8: E05C1000 80830030
	v_mfma_f32_16x16x32_fp8_fp8 v[224:227], a[66:67], v[130:131], v[224:227]// 0000000093C0: D3F300E0 0F830542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v64, v5 offset:38144                           // 0000000093C8: D86C9500 40000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v65, v5 offset:42496                           // 0000000093D0: D86CA600 41000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[228:231], a[64:65], v[144:145], 0// 0000000093D8: D3F300E4 0A032140  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dword v23, v6, s[16:19], 0 offen               // 0000000093E0: E0501000 80041706
	v_mfma_f32_16x16x32_fp8_fp8 v[228:231], a[66:67], v[146:147], v[228:231]// 0000000093E8: D3F300E4 0F932542  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v66, v5 offset:38176                           // 0000000093F0: D86C9520 42000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v67, v5 offset:42528                           // 0000000093F8: D86CA620 43000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[232:235], a[68:69], v[128:129], 0// 000000009400: D3F300E8 0A030144  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[4:7], v49, s[12:15], 0 offen         // 000000009408: E05C1000 80830431
	v_mfma_f32_16x16x32_fp8_fp8 v[232:235], a[70:71], v[130:131], v[232:235]// 000000009410: D3F300E8 0FA30546  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v68, v5 offset:38208                           // 000000009418: D86C9540 44000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v69, v5 offset:42560                           // 000000009420: D86CA640 45000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[236:239], a[68:69], v[144:145], 0// 000000009428: D3F300EC 0A032144  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[236:239], a[70:71], v[146:147], v[236:239]// 000000009430: D3F300EC 0FB32546  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v70, v5 offset:38240                           // 000000009438: D86C9560 46000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v71, v5 offset:42592                           // 000000009440: D86CA660 47000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[240:243], a[72:73], v[128:129], 0// 000000009448: D3F300F0 0A030148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[8:11], v50, s[12:15], 0 offen        // 000000009450: E05C1000 80830832
	v_mfma_f32_16x16x32_fp8_fp8 v[240:243], a[74:75], v[130:131], v[240:243]// 000000009458: D3F300F0 0FC3054A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v72, v5 offset:46848                           // 000000009460: D86CB700 48000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v73, v5 offset:51200                           // 000000009468: D86CC800 49000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[244:247], a[72:73], v[144:145], 0// 000000009470: D3F300F4 0A032148  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[244:247], a[74:75], v[146:147], v[244:247]// 000000009478: D3F300F4 0FD3254A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v74, v5 offset:46880                           // 000000009480: D86CB720 4A000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v75, v5 offset:51232                           // 000000009488: D86CC820 4B000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[248:251], a[76:77], v[128:129], 0// 000000009490: D3F300F8 0A03014C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[12:15], v51, s[12:15], 0 offen       // 000000009498: E05C1000 80830C33
	s_add_u32 s12, s78, s12                                    // 0000000094A0: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 0000000094A4: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[248:251], a[78:79], v[130:131], v[248:251]// 0000000094A8: D3F300F8 0FE3054E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v76, v5 offset:46912                           // 0000000094B0: D86CB740 4C000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v77, v5 offset:51264                           // 0000000094B8: D86CC840 4D000005  // Read 32-bit from LDS (shared memory)
	v_mfma_f32_16x16x32_fp8_fp8 v[252:255], a[76:77], v[144:145], 0// 0000000094C0: D3F300FC 0A03214C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[252:255], a[78:79], v[146:147], v[252:255]// 0000000094C8: D3F300FC 0FF3254E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	ds_read_b32 v78, v5 offset:46944                           // 0000000094D0: D86CB760 4E000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v79, v5 offset:51296                           // 0000000094D8: D86CC860 4F000005  // Read 32-bit from LDS (shared memory)
	s_waitcnt vmcnt(13)                                        // 0000000094E0: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[224:227], a[80:81], v[132:133], v[224:227]// 0000000094E4: D3F300E0 0F830950  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[16:19], v48, s[12:15], 0 offen       // 0000000094EC: E05C1000 80831030
	v_mfma_f32_16x16x32_fp8_fp8 v[224:227], a[82:83], v[134:135], v[224:227]// 0000000094F4: D3F300E0 0F830D52  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[228:231], a[80:81], v[148:149], v[228:231]// 0000000094FC: D3F300E4 0F932950  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[228:231], a[82:83], v[150:151], v[228:231]// 000000009504: D3F300E4 0F932D52  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[232:235], a[84:85], v[132:133], v[232:235]// 00000000950C: D3F300E8 0FA30954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[20:23], v49, s[12:15], 0 offen       // 000000009514: E05C1000 80831431
	v_mfma_f32_16x16x32_fp8_fp8 v[232:235], a[86:87], v[134:135], v[232:235]// 00000000951C: D3F300E8 0FA30D56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[236:239], a[84:85], v[148:149], v[236:239]// 000000009524: D3F300EC 0FB32954  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[236:239], a[86:87], v[150:151], v[236:239]// 00000000952C: D3F300EC 0FB32D56  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[240:243], a[88:89], v[132:133], v[240:243]// 000000009534: D3F300F0 0FC30958  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[24:27], v50, s[12:15], 0 offen       // 00000000953C: E05C1000 80831832
	v_mfma_f32_16x16x32_fp8_fp8 v[240:243], a[90:91], v[134:135], v[240:243]// 000000009544: D3F300F0 0FC30D5A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[244:247], a[88:89], v[148:149], v[244:247]// 00000000954C: D3F300F4 0FD32958  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[244:247], a[90:91], v[150:151], v[244:247]// 000000009554: D3F300F4 0FD32D5A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[248:251], a[92:93], v[132:133], v[248:251]// 00000000955C: D3F300F8 0FE3095C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[28:31], v51, s[12:15], 0 offen       // 000000009564: E05C1000 80831C33
	s_add_u32 s12, s78, s12                                    // 00000000956C: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 000000009570: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[248:251], a[94:95], v[134:135], v[248:251]// 000000009574: D3F300F8 0FE30D5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[252:255], a[92:93], v[148:149], v[252:255]// 00000000957C: D3F300FC 0FF3295C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[252:255], a[94:95], v[150:151], v[252:255]// 000000009584: D3F300FC 0FF32D5E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v24, v32 row_newbcast:0 row_mask:0xf bank_mask:0xf// 00000000958C: 0A7040FA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000009594: 7E720338  // Vector move
	v_pk_mul_f32 v[224:225], v[56:57], v[224:225]              // 000000009598: D3B140E0 1803C138
	v_pk_mul_f32 v[226:227], v[56:57], v[226:227]              // 0000000095A0: D3B140E2 1803C538
	v_pk_mul_f32 v[232:233], v[56:57], v[232:233]              // 0000000095A8: D3B140E8 1803D138
	v_pk_mul_f32 v[234:235], v[56:57], v[234:235]              // 0000000095B0: D3B140EA 1803D538
	v_mul_f32_dpp v56, v24, v32 row_newbcast:1 row_mask:0xf bank_mask:0xf// 0000000095B8: 0A7040FA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000095C0: 7E720338  // Vector move
	v_pk_mul_f32 v[240:241], v[56:57], v[240:241]              // 0000000095C4: D3B140F0 1803E138
	v_pk_mul_f32 v[242:243], v[56:57], v[242:243]              // 0000000095CC: D3B140F2 1803E538
	v_pk_mul_f32 v[248:249], v[56:57], v[248:249]              // 0000000095D4: D3B140F8 1803F138
	v_pk_mul_f32 v[250:251], v[56:57], v[250:251]              // 0000000095DC: D3B140FA 1803F538
	v_mul_f32_dpp v56, v24, v33 row_newbcast:0 row_mask:0xf bank_mask:0xf// 0000000095E4: 0A7042FA FF015018  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000095EC: 7E720338  // Vector move
	v_pk_mul_f32 v[228:229], v[56:57], v[228:229]              // 0000000095F0: D3B140E4 1803C938
	v_pk_mul_f32 v[230:231], v[56:57], v[230:231]              // 0000000095F8: D3B140E6 1803CD38
	v_pk_mul_f32 v[236:237], v[56:57], v[236:237]              // 000000009600: D3B140EC 1803D938
	v_pk_mul_f32 v[238:239], v[56:57], v[238:239]              // 000000009608: D3B140EE 1803DD38
	v_mul_f32_dpp v56, v24, v33 row_newbcast:1 row_mask:0xf bank_mask:0xf// 000000009610: 0A7042FA FF015118  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000009618: 7E720338  // Vector move
	v_pk_mul_f32 v[244:245], v[56:57], v[244:245]              // 00000000961C: D3B140F4 1803E938
	v_pk_mul_f32 v[246:247], v[56:57], v[246:247]              // 000000009624: D3B140F6 1803ED38
	v_pk_mul_f32 v[252:253], v[56:57], v[252:253]              // 00000000962C: D3B140FC 1803F938
	v_pk_mul_f32 v[254:255], v[56:57], v[254:255]              // 000000009634: D3B140FE 1803FD38
	s_waitcnt vmcnt(13)                                        // 00000000963C: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[96:97], v[136:137], 0// 000000009640: D3F300A0 0A031160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[32:35], v48, s[12:15], 0 offen       // 000000009648: E05C1000 80832030
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[98:99], v[138:139], v[160:163]// 000000009650: D3F300A0 0E831562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[192:193] offset:20736                   // 000000009658: D89A5100 0000C004  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[96:97], v[152:153], 0// 000000009660: D3F300A4 0A033160  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[98:99], v[154:155], v[164:167]// 000000009668: D3F300A4 0E933562  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[194:195] offset:29440                   // 000000009670: D89A7300 0000C204  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[100:101], v[136:137], 0// 000000009678: D3F300A8 0A031164  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[36:39], v49, s[12:15], 0 offen       // 000000009680: E05C1000 80832431
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[102:103], v[138:139], v[168:171]// 000000009688: D3F300A8 0EA31566  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[196:197] offset:22912                   // 000000009690: D89A5980 0000C404  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[100:101], v[152:153], 0// 000000009698: D3F300AC 0A033164  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[102:103], v[154:155], v[172:175]// 0000000096A0: D3F300AC 0EB33566  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[198:199] offset:31616                   // 0000000096A8: D89A7B80 0000C604  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[104:105], v[136:137], 0// 0000000096B0: D3F300B0 0A031168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[40:43], v50, s[12:15], 0 offen       // 0000000096B8: E05C1000 80832832
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[106:107], v[138:139], v[176:179]// 0000000096C0: D3F300B0 0EC3156A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[200:201] offset:25088                   // 0000000096C8: D89A6200 0000C804  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[104:105], v[152:153], 0// 0000000096D0: D3F300B4 0A033168  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[106:107], v[154:155], v[180:183]// 0000000096D8: D3F300B4 0ED3356A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[202:203] offset:33792                   // 0000000096E0: D89A8400 0000CA04  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[108:109], v[136:137], 0// 0000000096E8: D3F300B8 0A03116C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[44:47], v51, s[12:15], 0 offen       // 0000000096F0: E05C1000 80832C33
	s_add_u32 s12, s78, s12                                    // 0000000096F8: 800C0C4E
	s_addc_u32 s13, 0, s13                                     // 0000000096FC: 820D0D80
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[110:111], v[138:139], v[184:187]// 000000009700: D3F300B8 0EE3156E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[204:205] offset:27264                   // 000000009708: D89A6A80 0000CC04  // Write 64-bit to LDS

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[108:109], v[152:153], 0// 000000009710: D3F300BC 0A03316C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[110:111], v[154:155], v[188:191]// 000000009718: D3F300BC 0EF3356E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[206:207] offset:35968                   // 000000009720: D89A8C80 0000CE04  // Write 64-bit to LDS
	s_waitcnt vmcnt(13)                                        // 000000009728: BF8C0F7D  // Wait for memory operations (CRITICAL for perf)

// === MFMA COMPUTE BLOCK ===
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[112:113], v[140:141], v[160:163]// 00000000972C: D3F300A0 0E831970  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[48:51], v48, s[12:15], 0 offen       // 000000009734: E05C1000 80833030
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], a[114:115], v[142:143], v[160:163]// 00000000973C: D3F300A0 0E831D72  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[112:113], v[156:157], v[164:167]// 000000009744: D3F300A4 0E933970  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[164:167], a[114:115], v[158:159], v[164:167]// 00000000974C: D3F300A4 0E933D72  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[116:117], v[140:141], v[168:171]// 000000009754: D3F300A8 0EA31974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[52:55], v49, s[12:15], 0 offen       // 00000000975C: E05C1000 80833431
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], a[118:119], v[142:143], v[168:171]// 000000009764: D3F300A8 0EA31D76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[116:117], v[156:157], v[172:175]// 00000000976C: D3F300AC 0EB33974  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[172:175], a[118:119], v[158:159], v[172:175]// 000000009774: D3F300AC 0EB33D76  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[120:121], v[140:141], v[176:179]// 00000000977C: D3F300B0 0EC31978  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[56:59], v50, s[12:15], 0 offen       // 000000009784: E05C1000 80833832
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], a[122:123], v[142:143], v[176:179]// 00000000978C: D3F300B0 0EC31D7A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[120:121], v[156:157], v[180:183]// 000000009794: D3F300B4 0ED33978  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], a[122:123], v[158:159], v[180:183]// 00000000979C: D3F300B4 0ED33D7A  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[124:125], v[140:141], v[184:187]// 0000000097A4: D3F300B8 0EE3197C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	buffer_load_dwordx4 a[60:63], v51, s[12:15], 0 offen       // 0000000097AC: E05C1000 80833C33
	v_mfma_f32_16x16x32_fp8_fp8 v[184:187], a[126:127], v[142:143], v[184:187]// 0000000097B4: D3F300B8 0EE31D7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[124:125], v[156:157], v[188:191]// 0000000097BC: D3F300BC 0EF3397C  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], a[126:127], v[158:159], v[188:191]// 0000000097C4: D3F300BC 0EF33D7E  // MFMA: 16x16x32 FP8*FP8->FP32 matrix multiply
	v_mul_f32_dpp v56, v24, v34 row_newbcast:2 row_mask:0xf bank_mask:0xf// 0000000097CC: 0A7044FA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 0000000097D4: 7E720338  // Vector move
	v_pk_fma_f32 v[224:225], v[160:161], v[56:57], v[224:225]  // 0000000097D8: D3B040E0 1F8271A0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[226:227], v[162:163], v[56:57], v[226:227]  // 0000000097E0: D3B040E2 1F8A71A2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[232:233], v[168:169], v[56:57], v[232:233]  // 0000000097E8: D3B040E8 1FA271A8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[234:235], v[170:171], v[56:57], v[234:235]  // 0000000097F0: D3B040EA 1FAA71AA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v34 row_newbcast:3 row_mask:0xf bank_mask:0xf// 0000000097F8: 0A7044FA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000009800: 7E720338  // Vector move
	v_pk_fma_f32 v[240:241], v[176:177], v[56:57], v[240:241]  // 000000009804: D3B040F0 1FC271B0  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[242:243], v[178:179], v[56:57], v[242:243]  // 00000000980C: D3B040F2 1FCA71B2  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[248:249], v[184:185], v[56:57], v[248:249]  // 000000009814: D3B040F8 1FE271B8  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[250:251], v[186:187], v[56:57], v[250:251]  // 00000000981C: D3B040FA 1FEA71BA  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v35 row_newbcast:2 row_mask:0xf bank_mask:0xf// 000000009824: 0A7046FA FF015218  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 00000000982C: 7E720338  // Vector move
	v_pk_fma_f32 v[228:229], v[164:165], v[56:57], v[228:229]  // 000000009830: D3B040E4 1F9271A4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[230:231], v[166:167], v[56:57], v[230:231]  // 000000009838: D3B040E6 1F9A71A6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[236:237], v[172:173], v[56:57], v[236:237]  // 000000009840: D3B040EC 1FB271AC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[238:239], v[174:175], v[56:57], v[238:239]  // 000000009848: D3B040EE 1FBA71AE  // Packed FMA: 2x FP32 fused multiply-add
	v_mul_f32_dpp v56, v24, v35 row_newbcast:3 row_mask:0xf bank_mask:0xf// 000000009850: 0A7046FA FF015318  // FP32 multiply with data parallel primitives (cross-lane)
	v_mov_b32_e32 v57, v56                                     // 000000009858: 7E720338  // Vector move
	v_pk_fma_f32 v[244:245], v[180:181], v[56:57], v[244:245]  // 00000000985C: D3B040F4 1FD271B4  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[246:247], v[182:183], v[56:57], v[246:247]  // 000000009864: D3B040F6 1FDA71B6  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[252:253], v[188:189], v[56:57], v[252:253]  // 00000000986C: D3B040FC 1FF271BC  // Packed FMA: 2x FP32 fused multiply-add
	v_pk_fma_f32 v[254:255], v[190:191], v[56:57], v[254:255]  // 000000009874: D3B040FE 1FFA71BE  // Packed FMA: 2x FP32 fused multiply-add
	s_add_u32 s60, 0x200, s80                                  // 00000000987C: 803C50FF 00000200
	s_cmp_lt_u32 s60, s81                                      // 000000009884: BF0A513C
	s_cselect_b32 s56, s56, 0                                  // 000000009888: 85388038
	s_cselect_b32 s78, s78, 0                                  // 00000000988C: 854E804E
	s_cselect_b32 s79, s79, 0                                  // 000000009890: 854F804F
	s_add_u32 s12, s56, s12                                    // 000000009894: 800C0C38
	s_addc_u32 s13, 0, s13                                     // 000000009898: 820D0D80
	s_add_u32 s16, s79, s16                                    // 00000000989C: 8010104F
	s_addc_u32 s17, 0, s17                                     // 0000000098A0: 82111180
	v_mov_b32_e32 v56, v25                                     // 0000000098A4: 7E700319  // Vector move
	v_mov_b32_e32 v57, v25                                     // 0000000098A8: 7E720319  // Vector move
	v_pk_mul_f32 v[224:225], v[56:57], v[224:225]              // 0000000098AC: D3B140E0 1803C138
	v_pk_mul_f32 v[226:227], v[56:57], v[226:227]              // 0000000098B4: D3B140E2 1803C538
	v_pk_mul_f32 v[232:233], v[56:57], v[232:233]              // 0000000098BC: D3B140E8 1803D138
	v_pk_mul_f32 v[234:235], v[56:57], v[234:235]              // 0000000098C4: D3B140EA 1803D538
	v_pk_mul_f32 v[240:241], v[56:57], v[240:241]              // 0000000098CC: D3B140F0 1803E138
	v_pk_mul_f32 v[242:243], v[56:57], v[242:243]              // 0000000098D4: D3B140F2 1803E538
	v_pk_mul_f32 v[248:249], v[56:57], v[248:249]              // 0000000098DC: D3B140F8 1803F138
	v_pk_mul_f32 v[250:251], v[56:57], v[250:251]              // 0000000098E4: D3B140FA 1803F538
	v_mov_b32_e32 v56, v26                                     // 0000000098EC: 7E70031A  // Vector move
	v_mov_b32_e32 v57, v26                                     // 0000000098F0: 7E72031A  // Vector move
	v_pk_mul_f32 v[228:229], v[56:57], v[228:229]              // 0000000098F4: D3B140E4 1803C938
	v_pk_mul_f32 v[230:231], v[56:57], v[230:231]              // 0000000098FC: D3B140E6 1803CD38
	v_pk_mul_f32 v[236:237], v[56:57], v[236:237]              // 000000009904: D3B140EC 1803D938
	v_pk_mul_f32 v[238:239], v[56:57], v[238:239]              // 00000000990C: D3B140EE 1803DD38
	v_pk_mul_f32 v[244:245], v[56:57], v[244:245]              // 000000009914: D3B140F4 1803E938
	v_pk_mul_f32 v[246:247], v[56:57], v[246:247]              // 00000000991C: D3B140F6 1803ED38
	v_pk_mul_f32 v[252:253], v[56:57], v[252:253]              // 000000009924: D3B140FC 1803F938
	v_pk_mul_f32 v[254:255], v[56:57], v[254:255]              // 00000000992C: D3B140FE 1803FD38
	v_cmp_u_f32_e64 s[48:49], v224, v224                       // 000000009934: D0480030 0003C1E0
	v_add3_u32 v52, v224, v55, 1                               // 00000000993C: D1FF0034 02066FE0
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009944: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v225, v225                       // 00000000994C: D0480030 0003C3E1
	v_add3_u32 v52, v225, v55, 1                               // 000000009954: D1FF0034 02066FE1
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 00000000995C: D1000039 00C26D34
	v_perm_b32 v224, v57, v56, s52                             // 000000009964: D1ED00E0 00D27139
	v_cmp_u_f32_e64 s[48:49], v226, v226                       // 00000000996C: D0480030 0003C5E2
	v_add3_u32 v52, v226, v55, 1                               // 000000009974: D1FF0034 02066FE2
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 00000000997C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v227, v227                       // 000000009984: D0480030 0003C7E3
	v_add3_u32 v52, v227, v55, 1                               // 00000000998C: D1FF0034 02066FE3
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009994: D1000039 00C26D34
	v_perm_b32 v225, v57, v56, s52                             // 00000000999C: D1ED00E1 00D27139
	v_cmp_u_f32_e64 s[48:49], v228, v228                       // 0000000099A4: D0480030 0003C9E4
	v_add3_u32 v52, v228, v55, 1                               // 0000000099AC: D1FF0034 02066FE4
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000099B4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v229, v229                       // 0000000099BC: D0480030 0003CBE5
	v_add3_u32 v52, v229, v55, 1                               // 0000000099C4: D1FF0034 02066FE5
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 0000000099CC: D1000039 00C26D34
	v_perm_b32 v226, v57, v56, s52                             // 0000000099D4: D1ED00E2 00D27139
	v_cmp_u_f32_e64 s[48:49], v230, v230                       // 0000000099DC: D0480030 0003CDE6
	v_add3_u32 v52, v230, v55, 1                               // 0000000099E4: D1FF0034 02066FE6
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 0000000099EC: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v231, v231                       // 0000000099F4: D0480030 0003CFE7
	v_add3_u32 v52, v231, v55, 1                               // 0000000099FC: D1FF0034 02066FE7
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009A04: D1000039 00C26D34
	v_perm_b32 v227, v57, v56, s52                             // 000000009A0C: D1ED00E3 00D27139
	v_cmp_u_f32_e64 s[48:49], v232, v232                       // 000000009A14: D0480030 0003D1E8
	v_add3_u32 v52, v232, v55, 1                               // 000000009A1C: D1FF0034 02066FE8
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009A24: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v233, v233                       // 000000009A2C: D0480030 0003D3E9
	v_add3_u32 v52, v233, v55, 1                               // 000000009A34: D1FF0034 02066FE9
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009A3C: D1000039 00C26D34
	v_perm_b32 v228, v57, v56, s52                             // 000000009A44: D1ED00E4 00D27139
	v_cmp_u_f32_e64 s[48:49], v234, v234                       // 000000009A4C: D0480030 0003D5EA
	v_add3_u32 v52, v234, v55, 1                               // 000000009A54: D1FF0034 02066FEA
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009A5C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v235, v235                       // 000000009A64: D0480030 0003D7EB
	v_add3_u32 v52, v235, v55, 1                               // 000000009A6C: D1FF0034 02066FEB
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009A74: D1000039 00C26D34
	v_perm_b32 v229, v57, v56, s52                             // 000000009A7C: D1ED00E5 00D27139
	v_cmp_u_f32_e64 s[48:49], v236, v236                       // 000000009A84: D0480030 0003D9EC
	v_add3_u32 v52, v236, v55, 1                               // 000000009A8C: D1FF0034 02066FEC
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009A94: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v237, v237                       // 000000009A9C: D0480030 0003DBED
	v_add3_u32 v52, v237, v55, 1                               // 000000009AA4: D1FF0034 02066FED
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009AAC: D1000039 00C26D34
	v_perm_b32 v230, v57, v56, s52                             // 000000009AB4: D1ED00E6 00D27139
	v_cmp_u_f32_e64 s[48:49], v238, v238                       // 000000009ABC: D0480030 0003DDEE
	v_add3_u32 v52, v238, v55, 1                               // 000000009AC4: D1FF0034 02066FEE
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009ACC: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v239, v239                       // 000000009AD4: D0480030 0003DFEF
	v_add3_u32 v52, v239, v55, 1                               // 000000009ADC: D1FF0034 02066FEF
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009AE4: D1000039 00C26D34
	v_perm_b32 v231, v57, v56, s52                             // 000000009AEC: D1ED00E7 00D27139
	v_cmp_u_f32_e64 s[48:49], v240, v240                       // 000000009AF4: D0480030 0003E1F0
	v_add3_u32 v52, v240, v55, 1                               // 000000009AFC: D1FF0034 02066FF0
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009B04: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v241, v241                       // 000000009B0C: D0480030 0003E3F1
	v_add3_u32 v52, v241, v55, 1                               // 000000009B14: D1FF0034 02066FF1
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009B1C: D1000039 00C26D34
	v_perm_b32 v232, v57, v56, s52                             // 000000009B24: D1ED00E8 00D27139
	v_cmp_u_f32_e64 s[48:49], v242, v242                       // 000000009B2C: D0480030 0003E5F2
	v_add3_u32 v52, v242, v55, 1                               // 000000009B34: D1FF0034 02066FF2
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009B3C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v243, v243                       // 000000009B44: D0480030 0003E7F3
	v_add3_u32 v52, v243, v55, 1                               // 000000009B4C: D1FF0034 02066FF3
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009B54: D1000039 00C26D34
	v_perm_b32 v233, v57, v56, s52                             // 000000009B5C: D1ED00E9 00D27139
	v_cmp_u_f32_e64 s[48:49], v244, v244                       // 000000009B64: D0480030 0003E9F4
	v_add3_u32 v52, v244, v55, 1                               // 000000009B6C: D1FF0034 02066FF4
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009B74: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v245, v245                       // 000000009B7C: D0480030 0003EBF5
	v_add3_u32 v52, v245, v55, 1                               // 000000009B84: D1FF0034 02066FF5
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009B8C: D1000039 00C26D34
	v_perm_b32 v234, v57, v56, s52                             // 000000009B94: D1ED00EA 00D27139
	v_cmp_u_f32_e64 s[48:49], v246, v246                       // 000000009B9C: D0480030 0003EDF6
	v_add3_u32 v52, v246, v55, 1                               // 000000009BA4: D1FF0034 02066FF6
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009BAC: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v247, v247                       // 000000009BB4: D0480030 0003EFF7
	v_add3_u32 v52, v247, v55, 1                               // 000000009BBC: D1FF0034 02066FF7
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009BC4: D1000039 00C26D34
	v_perm_b32 v235, v57, v56, s52                             // 000000009BCC: D1ED00EB 00D27139
	v_cmp_u_f32_e64 s[48:49], v248, v248                       // 000000009BD4: D0480030 0003F1F8
	v_add3_u32 v52, v248, v55, 1                               // 000000009BDC: D1FF0034 02066FF8
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009BE4: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v249, v249                       // 000000009BEC: D0480030 0003F3F9
	v_add3_u32 v52, v249, v55, 1                               // 000000009BF4: D1FF0034 02066FF9
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009BFC: D1000039 00C26D34
	v_perm_b32 v236, v57, v56, s52                             // 000000009C04: D1ED00EC 00D27139
	v_cmp_u_f32_e64 s[48:49], v250, v250                       // 000000009C0C: D0480030 0003F5FA
	v_add3_u32 v52, v250, v55, 1                               // 000000009C14: D1FF0034 02066FFA
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009C1C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v251, v251                       // 000000009C24: D0480030 0003F7FB
	v_add3_u32 v52, v251, v55, 1                               // 000000009C2C: D1FF0034 02066FFB
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009C34: D1000039 00C26D34
	v_perm_b32 v237, v57, v56, s52                             // 000000009C3C: D1ED00ED 00D27139
	v_cmp_u_f32_e64 s[48:49], v252, v252                       // 000000009C44: D0480030 0003F9FC
	v_add3_u32 v52, v252, v55, 1                               // 000000009C4C: D1FF0034 02066FFC
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009C54: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v253, v253                       // 000000009C5C: D0480030 0003FBFD
	v_add3_u32 v52, v253, v55, 1                               // 000000009C64: D1FF0034 02066FFD
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009C6C: D1000039 00C26D34
	v_perm_b32 v238, v57, v56, s52                             // 000000009C74: D1ED00EE 00D27139
	v_cmp_u_f32_e64 s[48:49], v254, v254                       // 000000009C7C: D0480030 0003FDFE
	v_add3_u32 v52, v254, v55, 1                               // 000000009C84: D1FF0034 02066FFE
	v_cndmask_b32_e64 v56, v52, v54, s[48:49]                  // 000000009C8C: D1000038 00C26D34
	v_cmp_u_f32_e64 s[48:49], v255, v255                       // 000000009C94: D0480030 0003FFFF
	v_add3_u32 v52, v255, v55, 1                               // 000000009C9C: D1FF0034 02066FFF
	v_cndmask_b32_e64 v57, v52, v54, s[48:49]                  // 000000009CA4: D1000039 00C26D34
	v_perm_b32 v239, v57, v56, s52                             // 000000009CAC: D1ED00EF 00D27139
	s_cmp_ge_u32 s80, 0x200                                    // 000000009CB4: BF09FF50 00000200
	s_cselect_b32 s59, 0x200, s59                              // 000000009CBC: 853B3BFF 00000200
	s_setvskip s20, 0                                          // 000000009CC4: BF108014
	global_atomic_pk_add_bf16 v80, v64, s[8:9]                 // 000000009CC8: DD488000 00084050
	s_setvskip 0, 0                                            // 000000009CD0: BF108080
	s_setvskip s20, 0                                          // 000000009CD4: BF108014
	global_atomic_pk_add_bf16 v80, v65, s[8:9] offset:256      // 000000009CD8: DD488100 00084150
	s_setvskip 0, 0                                            // 000000009CE0: BF108080
	s_setvskip s20, 1                                          // 000000009CE4: BF108114
	global_atomic_pk_add_bf16 v82, v66, s[8:9]                 // 000000009CE8: DD488000 00084252
	s_setvskip 0, 0                                            // 000000009CF0: BF108080
	s_setvskip s20, 1                                          // 000000009CF4: BF108114
	global_atomic_pk_add_bf16 v82, v67, s[8:9] offset:256      // 000000009CF8: DD488100 00084352
	s_setvskip 0, 0                                            // 000000009D00: BF108080
	s_setvskip s20, 2                                          // 000000009D04: BF108214
	global_atomic_pk_add_bf16 v84, v68, s[8:9]                 // 000000009D08: DD488000 00084454
	s_setvskip 0, 0                                            // 000000009D10: BF108080
	s_setvskip s20, 2                                          // 000000009D14: BF108214
	global_atomic_pk_add_bf16 v84, v69, s[8:9] offset:256      // 000000009D18: DD488100 00084554
	s_setvskip 0, 0                                            // 000000009D20: BF108080
	s_setvskip s20, 3                                          // 000000009D24: BF108314
	global_atomic_pk_add_bf16 v86, v70, s[8:9]                 // 000000009D28: DD488000 00084656
	s_setvskip 0, 0                                            // 000000009D30: BF108080
	s_setvskip s20, 3                                          // 000000009D34: BF108314
	global_atomic_pk_add_bf16 v86, v71, s[8:9] offset:256      // 000000009D38: DD488100 00084756
	s_setvskip 0, 0                                            // 000000009D40: BF108080
	s_setvskip s20, 4                                          // 000000009D44: BF108414
	global_atomic_pk_add_bf16 v88, v72, s[8:9]                 // 000000009D48: DD488000 00084858
	s_setvskip 0, 0                                            // 000000009D50: BF108080
	s_setvskip s20, 4                                          // 000000009D54: BF108414
	global_atomic_pk_add_bf16 v88, v73, s[8:9] offset:256      // 000000009D58: DD488100 00084958
	s_setvskip 0, 0                                            // 000000009D60: BF108080
	s_setvskip s20, 5                                          // 000000009D64: BF108514
	global_atomic_pk_add_bf16 v90, v74, s[8:9]                 // 000000009D68: DD488000 00084A5A
	s_setvskip 0, 0                                            // 000000009D70: BF108080
	s_setvskip s20, 5                                          // 000000009D74: BF108514
	global_atomic_pk_add_bf16 v90, v75, s[8:9] offset:256      // 000000009D78: DD488100 00084B5A
	s_setvskip 0, 0                                            // 000000009D80: BF108080
	s_setvskip s20, 6                                          // 000000009D84: BF108614
	global_atomic_pk_add_bf16 v92, v76, s[8:9]                 // 000000009D88: DD488000 00084C5C
	s_setvskip 0, 0                                            // 000000009D90: BF108080
	s_setvskip s20, 6                                          // 000000009D94: BF108614
	global_atomic_pk_add_bf16 v92, v77, s[8:9] offset:256      // 000000009D98: DD488100 00084D5C
	s_setvskip 0, 0                                            // 000000009DA0: BF108080
	s_setvskip s20, 7                                          // 000000009DA4: BF108714
	global_atomic_pk_add_bf16 v94, v78, s[8:9]                 // 000000009DA8: DD488000 00084E5E
	s_setvskip 0, 0                                            // 000000009DB0: BF108080
	s_setvskip s20, 7                                          // 000000009DB4: BF108714
	global_atomic_pk_add_bf16 v94, v79, s[8:9] offset:256      // 000000009DB8: DD488100 00084F5E
	s_setvskip 0, 0                                            // 000000009DC0: BF108080
	s_add_u32 s8, s59, s8                                      // 000000009DC4: 8008083B
	s_addc_u32 s9, 0, s9                                       // 000000009DC8: 82090980
	s_addk_i32 s80, 0x100                                      // 000000009DCC: B7500100
	s_cmp_lt_i32 s80, s81                                      // 000000009DD0: BF045150
	s_cbranch_scc0 label_0F49                                  // 000000009DD4: BF84F353  // Branch if SCC==0 (condition false)
	s_branch label_16DE                                        // 000000009DD8: BF82FAE7  // Unconditional branch


// --- LABEL (potential loop target) ---
0000000000009ddc <label_1BF7>:
	s_cmp_ge_u32 s59, 0                                        // 000000009DDC: BF09803B
	s_cselect_b32 s59, 0x200, s59                              // 000000009DE0: 853B3BFF 00000200
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 000000009DE8: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 000000009DEC: BF8A0000  // Workgroup barrier synchronization
	s_cmp_eq_u32 s64, 0x100                                    // 000000009DF0: BF06FF40 00000100
	s_cbranch_scc0 label_1C73                                  // 000000009DF8: BF840074  // Branch if SCC==0 (condition false)

// === LDS WRITE (staging data) ===
	ds_write_b64 v4, v[192:193] offset:20736                   // 000000009DFC: D89A5100 0000C004  // Write 64-bit to LDS
	ds_write_b64 v4, v[194:195] offset:29440                   // 000000009E04: D89A7300 0000C204  // Write 64-bit to LDS
	ds_write_b64 v4, v[196:197] offset:22912                   // 000000009E0C: D89A5980 0000C404  // Write 64-bit to LDS
	ds_write_b64 v4, v[198:199] offset:31616                   // 000000009E14: D89A7B80 0000C604  // Write 64-bit to LDS
	ds_write_b64 v4, v[200:201] offset:25088                   // 000000009E1C: D89A6200 0000C804  // Write 64-bit to LDS
	ds_write_b64 v4, v[202:203] offset:33792                   // 000000009E24: D89A8400 0000CA04  // Write 64-bit to LDS
	ds_write_b64 v4, v[204:205] offset:27264                   // 000000009E2C: D89A6A80 0000CC04  // Write 64-bit to LDS
	ds_write_b64 v4, v[206:207] offset:35968                   // 000000009E34: D89A8C80 0000CE04  // Write 64-bit to LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 000000009E3C: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 000000009E40: BF8A0000  // Workgroup barrier synchronization
	ds_read_b32 v64, v5 offset:20736                           // 000000009E44: D86C5100 40000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v65, v5 offset:25088                           // 000000009E4C: D86C6200 41000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v66, v5 offset:20768                           // 000000009E54: D86C5120 42000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v67, v5 offset:25120                           // 000000009E5C: D86C6220 43000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v68, v5 offset:20800                           // 000000009E64: D86C5140 44000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v69, v5 offset:25152                           // 000000009E6C: D86C6240 45000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v70, v5 offset:20832                           // 000000009E74: D86C5160 46000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v71, v5 offset:25184                           // 000000009E7C: D86C6260 47000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v72, v5 offset:29440                           // 000000009E84: D86C7300 48000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v73, v5 offset:33792                           // 000000009E8C: D86C8400 49000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v74, v5 offset:29472                           // 000000009E94: D86C7320 4A000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v75, v5 offset:33824                           // 000000009E9C: D86C8420 4B000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v76, v5 offset:29504                           // 000000009EA4: D86C7340 4C000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v77, v5 offset:33856                           // 000000009EAC: D86C8440 4D000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v78, v5 offset:29536                           // 000000009EB4: D86C7360 4E000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v79, v5 offset:33888                           // 000000009EBC: D86C8460 4F000005  // Read 32-bit from LDS (shared memory)
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 000000009EC4: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_setvskip s20, 0                                          // 000000009EC8: BF108014
	global_atomic_pk_add_bf16 v80, v64, s[8:9]                 // 000000009ECC: DD488000 00084050
	s_setvskip 0, 0                                            // 000000009ED4: BF108080
	s_setvskip s20, 0                                          // 000000009ED8: BF108014
	global_atomic_pk_add_bf16 v80, v65, s[8:9] offset:256      // 000000009EDC: DD488100 00084150
	s_setvskip 0, 0                                            // 000000009EE4: BF108080
	s_setvskip s20, 1                                          // 000000009EE8: BF108114
	global_atomic_pk_add_bf16 v82, v66, s[8:9]                 // 000000009EEC: DD488000 00084252
	s_setvskip 0, 0                                            // 000000009EF4: BF108080
	s_setvskip s20, 1                                          // 000000009EF8: BF108114
	global_atomic_pk_add_bf16 v82, v67, s[8:9] offset:256      // 000000009EFC: DD488100 00084352
	s_setvskip 0, 0                                            // 000000009F04: BF108080
	s_setvskip s20, 2                                          // 000000009F08: BF108214
	global_atomic_pk_add_bf16 v84, v68, s[8:9]                 // 000000009F0C: DD488000 00084454
	s_setvskip 0, 0                                            // 000000009F14: BF108080
	s_setvskip s20, 2                                          // 000000009F18: BF108214
	global_atomic_pk_add_bf16 v84, v69, s[8:9] offset:256      // 000000009F1C: DD488100 00084554
	s_setvskip 0, 0                                            // 000000009F24: BF108080
	s_setvskip s20, 3                                          // 000000009F28: BF108314
	global_atomic_pk_add_bf16 v86, v70, s[8:9]                 // 000000009F2C: DD488000 00084656
	s_setvskip 0, 0                                            // 000000009F34: BF108080
	s_setvskip s20, 3                                          // 000000009F38: BF108314
	global_atomic_pk_add_bf16 v86, v71, s[8:9] offset:256      // 000000009F3C: DD488100 00084756
	s_setvskip 0, 0                                            // 000000009F44: BF108080
	s_setvskip s20, 4                                          // 000000009F48: BF108414
	global_atomic_pk_add_bf16 v88, v72, s[8:9]                 // 000000009F4C: DD488000 00084858
	s_setvskip 0, 0                                            // 000000009F54: BF108080
	s_setvskip s20, 4                                          // 000000009F58: BF108414
	global_atomic_pk_add_bf16 v88, v73, s[8:9] offset:256      // 000000009F5C: DD488100 00084958
	s_setvskip 0, 0                                            // 000000009F64: BF108080
	s_setvskip s20, 5                                          // 000000009F68: BF108514
	global_atomic_pk_add_bf16 v90, v74, s[8:9]                 // 000000009F6C: DD488000 00084A5A
	s_setvskip 0, 0                                            // 000000009F74: BF108080
	s_setvskip s20, 5                                          // 000000009F78: BF108514
	global_atomic_pk_add_bf16 v90, v75, s[8:9] offset:256      // 000000009F7C: DD488100 00084B5A
	s_setvskip 0, 0                                            // 000000009F84: BF108080
	s_setvskip s20, 6                                          // 000000009F88: BF108614
	global_atomic_pk_add_bf16 v92, v76, s[8:9]                 // 000000009F8C: DD488000 00084C5C
	s_setvskip 0, 0                                            // 000000009F94: BF108080
	s_setvskip s20, 6                                          // 000000009F98: BF108614
	global_atomic_pk_add_bf16 v92, v77, s[8:9] offset:256      // 000000009F9C: DD488100 00084D5C
	s_setvskip 0, 0                                            // 000000009FA4: BF108080
	s_setvskip s20, 7                                          // 000000009FA8: BF108714
	global_atomic_pk_add_bf16 v94, v78, s[8:9]                 // 000000009FAC: DD488000 00084E5E
	s_setvskip 0, 0                                            // 000000009FB4: BF108080
	s_setvskip s20, 7                                          // 000000009FB8: BF108714
	global_atomic_pk_add_bf16 v94, v79, s[8:9] offset:256      // 000000009FBC: DD488100 00084F5E
	s_setvskip 0, 0                                            // 000000009FC4: BF108080
	s_branch label_1D49                                        // 000000009FC8: BF8200D6  // Unconditional branch


// --- LABEL (potential loop target) ---
0000000000009fcc <label_1C73>:
	ds_read_b32 v64, v5 offset:20736                           // 000000009FCC: D86C5100 40000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v65, v5 offset:25088                           // 000000009FD4: D86C6200 41000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v66, v5 offset:20768                           // 000000009FDC: D86C5120 42000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v67, v5 offset:25120                           // 000000009FE4: D86C6220 43000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v68, v5 offset:20800                           // 000000009FEC: D86C5140 44000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v69, v5 offset:25152                           // 000000009FF4: D86C6240 45000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v70, v5 offset:20832                           // 000000009FFC: D86C5160 46000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v71, v5 offset:25184                           // 00000000A004: D86C6260 47000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v72, v5 offset:29440                           // 00000000A00C: D86C7300 48000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v73, v5 offset:33792                           // 00000000A014: D86C8400 49000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v74, v5 offset:29472                           // 00000000A01C: D86C7320 4A000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v75, v5 offset:33824                           // 00000000A024: D86C8420 4B000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v76, v5 offset:29504                           // 00000000A02C: D86C7340 4C000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v77, v5 offset:33856                           // 00000000A034: D86C8440 4D000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v78, v5 offset:29536                           // 00000000A03C: D86C7360 4E000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v79, v5 offset:33888                           // 00000000A044: D86C8460 4F000005  // Read 32-bit from LDS (shared memory)
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 00000000A04C: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_setvskip s20, 0                                          // 00000000A050: BF108014
	global_atomic_pk_add_bf16 v80, v64, s[8:9]                 // 00000000A054: DD488000 00084050
	s_setvskip 0, 0                                            // 00000000A05C: BF108080
	s_setvskip s20, 0                                          // 00000000A060: BF108014
	global_atomic_pk_add_bf16 v80, v65, s[8:9] offset:256      // 00000000A064: DD488100 00084150
	s_setvskip 0, 0                                            // 00000000A06C: BF108080
	s_setvskip s20, 1                                          // 00000000A070: BF108114
	global_atomic_pk_add_bf16 v82, v66, s[8:9]                 // 00000000A074: DD488000 00084252
	s_setvskip 0, 0                                            // 00000000A07C: BF108080
	s_setvskip s20, 1                                          // 00000000A080: BF108114
	global_atomic_pk_add_bf16 v82, v67, s[8:9] offset:256      // 00000000A084: DD488100 00084352
	s_setvskip 0, 0                                            // 00000000A08C: BF108080
	s_setvskip s20, 2                                          // 00000000A090: BF108214
	global_atomic_pk_add_bf16 v84, v68, s[8:9]                 // 00000000A094: DD488000 00084454
	s_setvskip 0, 0                                            // 00000000A09C: BF108080
	s_setvskip s20, 2                                          // 00000000A0A0: BF108214
	global_atomic_pk_add_bf16 v84, v69, s[8:9] offset:256      // 00000000A0A4: DD488100 00084554
	s_setvskip 0, 0                                            // 00000000A0AC: BF108080
	s_setvskip s20, 3                                          // 00000000A0B0: BF108314
	global_atomic_pk_add_bf16 v86, v70, s[8:9]                 // 00000000A0B4: DD488000 00084656
	s_setvskip 0, 0                                            // 00000000A0BC: BF108080
	s_setvskip s20, 3                                          // 00000000A0C0: BF108314
	global_atomic_pk_add_bf16 v86, v71, s[8:9] offset:256      // 00000000A0C4: DD488100 00084756
	s_setvskip 0, 0                                            // 00000000A0CC: BF108080
	s_setvskip s20, 4                                          // 00000000A0D0: BF108414
	global_atomic_pk_add_bf16 v88, v72, s[8:9]                 // 00000000A0D4: DD488000 00084858
	s_setvskip 0, 0                                            // 00000000A0DC: BF108080
	s_setvskip s20, 4                                          // 00000000A0E0: BF108414
	global_atomic_pk_add_bf16 v88, v73, s[8:9] offset:256      // 00000000A0E4: DD488100 00084958
	s_setvskip 0, 0                                            // 00000000A0EC: BF108080
	s_setvskip s20, 5                                          // 00000000A0F0: BF108514
	global_atomic_pk_add_bf16 v90, v74, s[8:9]                 // 00000000A0F4: DD488000 00084A5A
	s_setvskip 0, 0                                            // 00000000A0FC: BF108080
	s_setvskip s20, 5                                          // 00000000A100: BF108514
	global_atomic_pk_add_bf16 v90, v75, s[8:9] offset:256      // 00000000A104: DD488100 00084B5A
	s_setvskip 0, 0                                            // 00000000A10C: BF108080
	s_setvskip s20, 6                                          // 00000000A110: BF108614
	global_atomic_pk_add_bf16 v92, v76, s[8:9]                 // 00000000A114: DD488000 00084C5C
	s_setvskip 0, 0                                            // 00000000A11C: BF108080
	s_setvskip s20, 6                                          // 00000000A120: BF108614
	global_atomic_pk_add_bf16 v92, v77, s[8:9] offset:256      // 00000000A124: DD488100 00084D5C
	s_setvskip 0, 0                                            // 00000000A12C: BF108080
	s_setvskip s20, 7                                          // 00000000A130: BF108714
	global_atomic_pk_add_bf16 v94, v78, s[8:9]                 // 00000000A134: DD488000 00084E5E
	s_setvskip 0, 0                                            // 00000000A13C: BF108080
	s_setvskip s20, 7                                          // 00000000A140: BF108714
	global_atomic_pk_add_bf16 v94, v79, s[8:9] offset:256      // 00000000A144: DD488100 00084F5E
	s_setvskip 0, 0                                            // 00000000A14C: BF108080
	s_add_u32 s8, s59, s8                                      // 00000000A150: 8008083B
	s_addc_u32 s9, 0, s9                                       // 00000000A154: 82090980
	ds_write_b64 v4, v[224:225] offset:38144                   // 00000000A158: D89A9500 0000E004  // Write 64-bit to LDS
	ds_write_b64 v4, v[226:227] offset:46848                   // 00000000A160: D89AB700 0000E204  // Write 64-bit to LDS
	ds_write_b64 v4, v[228:229] offset:40320                   // 00000000A168: D89A9D80 0000E404  // Write 64-bit to LDS
	ds_write_b64 v4, v[230:231] offset:49024                   // 00000000A170: D89ABF80 0000E604  // Write 64-bit to LDS
	ds_write_b64 v4, v[232:233] offset:42496                   // 00000000A178: D89AA600 0000E804  // Write 64-bit to LDS
	ds_write_b64 v4, v[234:235] offset:51200                   // 00000000A180: D89AC800 0000EA04  // Write 64-bit to LDS
	ds_write_b64 v4, v[236:237] offset:44672                   // 00000000A188: D89AAE80 0000EC04  // Write 64-bit to LDS
	ds_write_b64 v4, v[238:239] offset:53376                   // 00000000A190: D89AD080 0000EE04  // Write 64-bit to LDS
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 00000000A198: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_barrier                                                  // 00000000A19C: BF8A0000  // Workgroup barrier synchronization
	ds_read_b32 v64, v5 offset:38144                           // 00000000A1A0: D86C9500 40000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v65, v5 offset:42496                           // 00000000A1A8: D86CA600 41000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v66, v5 offset:38176                           // 00000000A1B0: D86C9520 42000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v67, v5 offset:42528                           // 00000000A1B8: D86CA620 43000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v68, v5 offset:38208                           // 00000000A1C0: D86C9540 44000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v69, v5 offset:42560                           // 00000000A1C8: D86CA640 45000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v70, v5 offset:38240                           // 00000000A1D0: D86C9560 46000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v71, v5 offset:42592                           // 00000000A1D8: D86CA660 47000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v72, v5 offset:46848                           // 00000000A1E0: D86CB700 48000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v73, v5 offset:51200                           // 00000000A1E8: D86CC800 49000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v74, v5 offset:46880                           // 00000000A1F0: D86CB720 4A000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v75, v5 offset:51232                           // 00000000A1F8: D86CC820 4B000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v76, v5 offset:46912                           // 00000000A200: D86CB740 4C000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v77, v5 offset:51264                           // 00000000A208: D86CC840 4D000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v78, v5 offset:46944                           // 00000000A210: D86CB760 4E000005  // Read 32-bit from LDS (shared memory)
	ds_read_b32 v79, v5 offset:51296                           // 00000000A218: D86CC860 4F000005  // Read 32-bit from LDS (shared memory)
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt lgkmcnt(0)                                       // 00000000A220: BF8CC07F  // Wait for memory operations (CRITICAL for perf)
	s_setvskip s20, 0                                          // 00000000A224: BF108014
	global_atomic_pk_add_bf16 v80, v64, s[8:9]                 // 00000000A228: DD488000 00084050
	s_setvskip 0, 0                                            // 00000000A230: BF108080
	s_setvskip s20, 0                                          // 00000000A234: BF108014
	global_atomic_pk_add_bf16 v80, v65, s[8:9] offset:256      // 00000000A238: DD488100 00084150
	s_setvskip 0, 0                                            // 00000000A240: BF108080
	s_setvskip s20, 1                                          // 00000000A244: BF108114
	global_atomic_pk_add_bf16 v82, v66, s[8:9]                 // 00000000A248: DD488000 00084252
	s_setvskip 0, 0                                            // 00000000A250: BF108080
	s_setvskip s20, 1                                          // 00000000A254: BF108114
	global_atomic_pk_add_bf16 v82, v67, s[8:9] offset:256      // 00000000A258: DD488100 00084352
	s_setvskip 0, 0                                            // 00000000A260: BF108080
	s_setvskip s20, 2                                          // 00000000A264: BF108214
	global_atomic_pk_add_bf16 v84, v68, s[8:9]                 // 00000000A268: DD488000 00084454
	s_setvskip 0, 0                                            // 00000000A270: BF108080
	s_setvskip s20, 2                                          // 00000000A274: BF108214
	global_atomic_pk_add_bf16 v84, v69, s[8:9] offset:256      // 00000000A278: DD488100 00084554
	s_setvskip 0, 0                                            // 00000000A280: BF108080
	s_setvskip s20, 3                                          // 00000000A284: BF108314
	global_atomic_pk_add_bf16 v86, v70, s[8:9]                 // 00000000A288: DD488000 00084656
	s_setvskip 0, 0                                            // 00000000A290: BF108080
	s_setvskip s20, 3                                          // 00000000A294: BF108314
	global_atomic_pk_add_bf16 v86, v71, s[8:9] offset:256      // 00000000A298: DD488100 00084756
	s_setvskip 0, 0                                            // 00000000A2A0: BF108080
	s_setvskip s20, 4                                          // 00000000A2A4: BF108414
	global_atomic_pk_add_bf16 v88, v72, s[8:9]                 // 00000000A2A8: DD488000 00084858
	s_setvskip 0, 0                                            // 00000000A2B0: BF108080
	s_setvskip s20, 4                                          // 00000000A2B4: BF108414
	global_atomic_pk_add_bf16 v88, v73, s[8:9] offset:256      // 00000000A2B8: DD488100 00084958
	s_setvskip 0, 0                                            // 00000000A2C0: BF108080
	s_setvskip s20, 5                                          // 00000000A2C4: BF108514
	global_atomic_pk_add_bf16 v90, v74, s[8:9]                 // 00000000A2C8: DD488000 00084A5A
	s_setvskip 0, 0                                            // 00000000A2D0: BF108080
	s_setvskip s20, 5                                          // 00000000A2D4: BF108514
	global_atomic_pk_add_bf16 v90, v75, s[8:9] offset:256      // 00000000A2D8: DD488100 00084B5A
	s_setvskip 0, 0                                            // 00000000A2E0: BF108080
	s_setvskip s20, 6                                          // 00000000A2E4: BF108614
	global_atomic_pk_add_bf16 v92, v76, s[8:9]                 // 00000000A2E8: DD488000 00084C5C
	s_setvskip 0, 0                                            // 00000000A2F0: BF108080
	s_setvskip s20, 6                                          // 00000000A2F4: BF108614
	global_atomic_pk_add_bf16 v92, v77, s[8:9] offset:256      // 00000000A2F8: DD488100 00084D5C
	s_setvskip 0, 0                                            // 00000000A300: BF108080
	s_setvskip s20, 7                                          // 00000000A304: BF108714
	global_atomic_pk_add_bf16 v94, v78, s[8:9]                 // 00000000A308: DD488000 00084E5E
	s_setvskip 0, 0                                            // 00000000A310: BF108080
	s_setvskip s20, 7                                          // 00000000A314: BF108714
	global_atomic_pk_add_bf16 v94, v79, s[8:9] offset:256      // 00000000A318: DD488100 00084F5E
	s_setvskip 0, 0                                            // 00000000A320: BF108080


// --- LABEL (potential loop target) ---
000000000000a324 <label_1D49>:
// SYNC: Wait for ALL LDS/scalar ops to complete
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)                    // 00000000A324: BF8C0000  // Wait for memory operations (CRITICAL for perf)

// === KERNEL EPILOGUE ===
	s_endpgm                                                   // 00000000A328: BF810000  // End program
