import torch

def shuffle_weight(x: torch.Tensor, layout=(16, 16), use_int4=False) -> torch.Tensor:
    # Hardcode BLOCK_K and BLOCK_N
    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size() if not use_int4 else 32
    BN = IN
    assert x.shape[-2] % BN == 0, f"{x.shape[-2]} % {BN} == {x.shape[-2] % BN }"
    assert x.shape[-1] % BK == 0, f"{x.shape[-1]} % {BK} == {x.shape[-1] % BK }"

    # Calculate optimal chunk size based on tensor dimensions
    try:
        # Estimate memory needed for one full operation
        tensor_bytes = x.numel() * x.element_size()
        # Try to work in chunks that use ~100MB at a time
        target_chunk_bytes = 100 * 1024 * 1024  # 100MB
        elements_per_chunk = target_chunk_bytes // x.element_size()
        
        # Calculate chunk size based on first dimension
        if x.dim() > 2 and x.shape[0] > 1:
            elements_per_batch = x.shape[-2] * x.shape[-1]
            chunk_size = max(1, elements_per_chunk // elements_per_batch)
            chunk_size = min(chunk_size, x.shape[0])
        else:
            chunk_size = 1
    except:
        chunk_size = 1
    
    # Process in chunks if beneficial
    if x.dim() > 2 and chunk_size > 1 and x.shape[0] > chunk_size:
        batch_size = x.shape[0]
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            try:
                # Work on a slice
                chunk = x[i:end_idx]
                chunk_view = chunk.view(-1, chunk.shape[-2] // BN, BN, 
                                      chunk.shape[-1] // BK, BK // K, K)
                chunk_permuted = chunk_view.permute(0, 1, 3, 4, 2, 5)
                chunk_result = chunk_permuted.contiguous().view(chunk.shape)
                
                # Copy back in-place
                x[i:end_idx].copy_(chunk_result)
                
                # Clean up
                del chunk_view, chunk_permuted, chunk_result
                
            except torch.OutOfMemoryError:
                print(f"[SHUFFLE] OOM on chunk {i}-{end_idx}, trying smaller chunks")
                # Try processing this chunk element by element
                for j in range(i, end_idx):
                    try:
                        single = x[j:j+1]
                        single_view = single.view(-1, single.shape[-2] // BN, BN, 
                                                single.shape[-1] // BK, BK // K, K)
                        single_permuted = single_view.permute(0, 1, 3, 4, 2, 5)
                        single_result = single_permuted.contiguous().view(single.shape)
                        x[j:j+1].copy_(single_result)
                        del single_view, single_permuted, single_result
                    except:
                        print(f"[SHUFFLE] Failed to process element {j}, skipping")
                        continue
                        
        return x
    else:
        # Single pass
        try:
            x_ = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
            x_ = x_.permute(0, 1, 3, 4, 2, 5)
            x_ = x_.contiguous()
            x_ = x_.reshape(*x.shape)
            return x_
        except torch.OutOfMemoryError:
            print("[SHUFFLE] OOM during single-pass, trying chunked approach")
            # Fallback to processing smaller pieces
            if x.dim() > 2:
                # Process one batch element at a time
                for i in range(x.shape[0]):
                    try:
                        single = x[i:i+1]
                        single_view = single.view(-1, single.shape[-2] // BN, BN, 
                                                single.shape[-1] // BK, BK // K, K)
                        single_permuted = single_view.permute(0, 1, 3, 4, 2, 5)
                        single_result = single_permuted.contiguous().view(single.shape)
                        x[i:i+1].copy_(single_result)
                        del single_view, single_permuted, single_result
                    except:
                        print(f"[SHUFFLE] Failed to process batch element {i}")
                        continue
                return x
            else:
                print("[SHUFFLE] Cannot chunk 2D tensor, returning original")
                return x