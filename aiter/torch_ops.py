# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Torch library registration for AITER ops to enable torch.compile compatibility.

This module registers AITER's custom HIP kernels with torch.library so they can
be traced by torch.compile without causing graph breaks.
"""

import torch
from torch.library import Library
from typing import Optional

# Create the AITER library for custom ops
aiter_lib = Library("aiter", "FRAGMENT")

# Flag to track if ops have been registered
_OPS_REGISTERED = False


def _mla_decode_stage1_asm_fwd_impl(
    Q: torch.Tensor,
    KV: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    num_kv_splits_indptr: Optional[torch.Tensor],
    work_meta_data: Optional[torch.Tensor],
    work_indptr: Optional[torch.Tensor],
    work_info_set: Optional[torch.Tensor],
    max_seqlen_q: int,
    softmax_scale: float,
    splitData: torch.Tensor,
    splitLse: torch.Tensor,
    output: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
    kv_scale: Optional[torch.Tensor] = None,
) -> None:
    """Implementation wrapper for mla_decode_stage1_asm_fwd."""
    import aiter_  # The compiled extension module
    
    aiter_.mla_decode_stage1_asm_fwd(
        Q, KV, qo_indptr, kv_indptr, kv_page_indices, kv_last_page_lens,
        num_kv_splits_indptr, work_meta_data, work_indptr, work_info_set,
        max_seqlen_q, softmax_scale, splitData, splitLse, output,
        q_scale, kv_scale
    )


def _mla_decode_stage1_asm_fwd_fake(
    Q: torch.Tensor,
    KV: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    num_kv_splits_indptr: Optional[torch.Tensor],
    work_meta_data: Optional[torch.Tensor],
    work_indptr: Optional[torch.Tensor],
    work_info_set: Optional[torch.Tensor],
    max_seqlen_q: int,
    softmax_scale: float,
    splitData: torch.Tensor,
    splitLse: torch.Tensor,
    output: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
    kv_scale: Optional[torch.Tensor] = None,
) -> None:
    """Fake implementation for torch.compile - output is written in-place."""
    pass


def _mla_reduce_v1_impl(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: Optional[torch.Tensor],
    reduce_partial_map: torch.Tensor,
    max_seqlen_q: int,
    final_output: torch.Tensor,
    final_lse: Optional[torch.Tensor],
) -> None:
    """Implementation wrapper for mla_reduce_v1."""
    import aiter_
    
    aiter_.mla_reduce_v1(
        partial_output, partial_lse, reduce_indptr,
        reduce_final_map, reduce_partial_map, max_seqlen_q,
        final_output, final_lse
    )


def _mla_reduce_v1_fake(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: Optional[torch.Tensor],
    reduce_partial_map: torch.Tensor,
    max_seqlen_q: int,
    final_output: torch.Tensor,
    final_lse: Optional[torch.Tensor],
) -> None:
    """Fake implementation for torch.compile - output is written in-place."""
    pass


def _mla_prefill_asm_fwd_impl(
    Q: torch.Tensor,
    KV: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    splitData: torch.Tensor,
    splitLse: torch.Tensor,
) -> None:
    """Implementation wrapper for mla_prefill_asm_fwd."""
    import aiter_
    
    aiter_.mla_prefill_asm_fwd(
        Q, KV, qo_indptr, kv_indptr, kv_page_indices, kv_last_page_lens,
        max_seqlen_q, softmax_scale, splitData, splitLse
    )


def _mla_prefill_asm_fwd_fake(
    Q: torch.Tensor,
    KV: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    splitData: torch.Tensor,
    splitLse: torch.Tensor,
) -> None:
    """Fake implementation for torch.compile - output is written in-place."""
    pass


def register_mla_ops():
    """Register MLA operations with torch.library for torch.compile compatibility."""
    global _OPS_REGISTERED
    
    if _OPS_REGISTERED:
        return
    
    # Register mla_decode_stage1_asm_fwd
    aiter_lib.define(
        "mla_decode_stage1_asm_fwd("
        "Tensor Q, Tensor KV, Tensor qo_indptr, Tensor kv_indptr, "
        "Tensor kv_page_indices, Tensor kv_last_page_lens, "
        "Tensor? num_kv_splits_indptr, Tensor? work_meta_data, "
        "Tensor? work_indptr, Tensor? work_info_set, "
        "int max_seqlen_q, float softmax_scale, "
        "Tensor(a!) splitData, Tensor(b!) splitLse, Tensor(c!) output, "
        "Tensor? q_scale, Tensor? kv_scale"
        ") -> ()"
    )
    aiter_lib.impl("mla_decode_stage1_asm_fwd", _mla_decode_stage1_asm_fwd_impl, "HIP")
    aiter_lib._register_fake("mla_decode_stage1_asm_fwd", _mla_decode_stage1_asm_fwd_fake)
    
    # Register mla_reduce_v1
    aiter_lib.define(
        "mla_reduce_v1("
        "Tensor partial_output, Tensor partial_lse, Tensor reduce_indptr, "
        "Tensor? reduce_final_map, Tensor reduce_partial_map, int max_seqlen_q, "
        "Tensor(a!) final_output, Tensor?(b!) final_lse"
        ") -> ()"
    )
    aiter_lib.impl("mla_reduce_v1", _mla_reduce_v1_impl, "HIP")
    aiter_lib._register_fake("mla_reduce_v1", _mla_reduce_v1_fake)
    
    # Register mla_prefill_asm_fwd
    aiter_lib.define(
        "mla_prefill_asm_fwd("
        "Tensor Q, Tensor KV, Tensor qo_indptr, Tensor kv_indptr, "
        "Tensor kv_page_indices, Tensor kv_last_page_lens, "
        "int max_seqlen_q, float softmax_scale, "
        "Tensor(a!) splitData, Tensor(b!) splitLse"
        ") -> ()"
    )
    aiter_lib.impl("mla_prefill_asm_fwd", _mla_prefill_asm_fwd_impl, "HIP")
    aiter_lib._register_fake("mla_prefill_asm_fwd", _mla_prefill_asm_fwd_fake)
    
    _OPS_REGISTERED = True


# Auto-register on import
register_mla_ops()

