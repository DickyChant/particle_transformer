"""
Fused Attention with Bias Kernel for Particle Transformer

This module provides a Triton kernel that fuses the following operations:
1. Scaled dot-product attention: scores = Q @ K^T / sqrt(d_k)
2. Bias addition: scores = scores + bias (pairwise particle interaction features)
3. Softmax: attention_weights = softmax(scores)
4. Output projection: output = attention_weights @ V

The bias is the key component of the Particle Transformer that incorporates
pairwise particle interaction features (e.g., delta-R, kt, z, invariant mass)
into the attention mechanism.
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _fused_attention_bias_fwd_kernel(
    # Pointers to matrices
    Q_ptr,
    K_ptr,
    V_ptr,
    Bias_ptr,
    Out_ptr,
    # Matrix dimensions
    N_CTX: tl.constexpr,  # sequence length
    D_HEAD: tl.constexpr,  # head dimension
    # Strides for Q
    stride_qb,  # batch stride
    stride_qh,  # head stride
    stride_qm,  # sequence stride
    stride_qk,  # head dim stride
    # Strides for K
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kk,
    # Strides for V
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vk,
    # Strides for Bias
    stride_bb,  # batch stride
    stride_bh,  # head stride (or 0 if bias is shared across heads)
    stride_bm,  # row stride
    stride_bn,  # col stride
    # Strides for Output
    stride_ob,
    stride_oh,
    stride_om,
    stride_ok,
    # Scale factor
    scale,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused attention kernel with additive bias.
    
    Computes: softmax(Q @ K^T / scale + Bias) @ V
    
    This kernel processes one block of the output at a time.
    """
    # Program IDs
    pid_m = tl.program_id(0)  # which block of rows
    pid_bh = tl.program_id(1)  # batch * num_heads index
    
    # Compute batch and head indices
    # Assuming batch and head are flattened into one dimension
    
    # Offsets for the block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize pointers with batch/head offset
    q_ptrs = Q_ptr + pid_bh * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K_ptr + pid_bh * stride_kh + offs_k[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_ptrs = V_ptr + pid_bh * stride_vh + offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk
    bias_ptrs = Bias_ptr + pid_bh * stride_bh + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    
    # Load Q block (BLOCK_M, D_HEAD)
    mask_m = offs_m < N_CTX
    q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < D_HEAD), other=0.0)
    
    # Initialize accumulators for online softmax
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # max for numerical stability
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # sum of exp
    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)  # accumulated output
    
    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n_curr < N_CTX
        
        # Load K block (D_HEAD, BLOCK_N)
        k_ptrs_curr = K_ptr + pid_bh * stride_kh + offs_k[:, None] * stride_kk + offs_n_curr[None, :] * stride_kn
        k = tl.load(k_ptrs_curr, mask=(offs_k[:, None] < D_HEAD) & mask_n[None, :], other=0.0)
        
        # Compute attention scores: (BLOCK_M, BLOCK_N)
        scores = tl.dot(q, k) * scale
        
        # Load and add bias
        bias_ptrs_curr = Bias_ptr + pid_bh * stride_bh + offs_m[:, None] * stride_bm + offs_n_curr[None, :] * stride_bn
        bias = tl.load(bias_ptrs_curr, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        scores = scores + bias
        
        # Mask out invalid positions
        scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, float("-inf"))
        
        # Online softmax update
        m_i_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(scores - m_i_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # Scale previous accumulator
        acc = acc * alpha[:, None]
        
        # Load V block (BLOCK_N, D_HEAD)
        v_ptrs_curr = V_ptr + pid_bh * stride_vh + offs_n_curr[:, None] * stride_vn + offs_k[None, :] * stride_vk
        v = tl.load(v_ptrs_curr, mask=mask_n[:, None] & (offs_k[None, :] < D_HEAD), other=0.0)
        
        # Accumulate: acc += p @ v
        acc += tl.dot(p.to(v.dtype), v)
        
        m_i = m_i_new
    
    # Final normalization
    acc = acc / l_i[:, None]
    
    # Store output
    out_ptrs = Out_ptr + pid_bh * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & (offs_k[None, :] < D_HEAD))


class FusedAttentionBiasFunction(torch.autograd.Function):
    """
    Autograd function for fused attention with bias.
    
    Forward: softmax(Q @ K^T / sqrt(d_k) + Bias) @ V
    """
    
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of fused attention with bias.
        
        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
            v: Value tensor of shape (batch, num_heads, seq_len, head_dim)
            bias: Bias tensor of shape (batch, num_heads, seq_len, seq_len)
                  or (batch * num_heads, seq_len, seq_len)
        
        Returns:
            Output tensor of shape (batch, num_heads, seq_len, head_dim)
        """
        # Validate inputs
        assert q.dim() == 4, f"Expected 4D query tensor, got {q.dim()}D"
        batch, num_heads, seq_len, head_dim = q.shape
        
        assert k.shape == q.shape, f"Key shape {k.shape} != Query shape {q.shape}"
        assert v.shape == q.shape, f"Value shape {v.shape} != Query shape {q.shape}"
        
        # Handle bias shape - can be (batch, num_heads, seq_len, seq_len) or (batch * num_heads, seq_len, seq_len)
        if bias.dim() == 3:
            bias = bias.view(batch, num_heads, seq_len, seq_len)
        assert bias.shape == (batch, num_heads, seq_len, seq_len), \
            f"Bias shape {bias.shape} != expected ({batch}, {num_heads}, {seq_len}, {seq_len})"
        
        # Ensure contiguous tensors
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        bias = bias.contiguous()
        
        # Allocate output
        out = torch.empty_like(q)
        
        # Scale factor
        scale = 1.0 / (head_dim ** 0.5)
        
        # Block sizes - tuned for typical ParT parameters
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = head_dim  # Full head dimension
        
        # Grid
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * num_heads)
        
        # Reshape for kernel (merge batch and heads)
        q_flat = q.view(batch * num_heads, seq_len, head_dim)
        k_flat = k.view(batch * num_heads, seq_len, head_dim)
        v_flat = v.view(batch * num_heads, seq_len, head_dim)
        bias_flat = bias.view(batch * num_heads, seq_len, seq_len)
        out_flat = out.view(batch * num_heads, seq_len, head_dim)
        
        _fused_attention_bias_fwd_kernel[grid](
            q_flat, k_flat, v_flat, bias_flat, out_flat,
            seq_len, head_dim,
            # Q strides
            q_flat.stride(0), q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            # K strides
            k_flat.stride(0), k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            # V strides
            v_flat.stride(0), v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            # Bias strides
            bias_flat.stride(0), bias_flat.stride(0), bias_flat.stride(1), bias_flat.stride(2),
            # Output strides
            out_flat.stride(0), out_flat.stride(0), out_flat.stride(1), out_flat.stride(2),
            scale,
            BLOCK_M, BLOCK_N, BLOCK_K,
        )
        
        # Save for backward (if needed)
        ctx.save_for_backward(q, k, v, bias, out)
        ctx.scale = scale
        
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass - falls back to PyTorch implementation for now.
        A fully fused backward kernel can be added for further optimization.
        """
        q, k, v, bias, out = ctx.saved_tensors
        scale = ctx.scale
        
        # Recompute attention weights for backward
        batch, num_heads, seq_len, head_dim = q.shape
        
        # scores = q @ k^T * scale + bias
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + bias
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Backward through V projection
        grad_v = torch.matmul(attn_weights.transpose(-2, -1), grad_out)
        grad_attn = torch.matmul(grad_out, v.transpose(-2, -1))
        
        # Backward through softmax
        grad_scores = attn_weights * (grad_attn - (grad_attn * attn_weights).sum(dim=-1, keepdim=True))
        
        # Backward through bias addition
        grad_bias = grad_scores
        
        # Backward through scale
        grad_scores = grad_scores * scale
        
        # Backward through matmul
        grad_q = torch.matmul(grad_scores, k)
        grad_k = torch.matmul(grad_scores.transpose(-2, -1), q)
        
        return grad_q, grad_k, grad_v, grad_bias


def fused_attention_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Fused attention with additive bias.
    
    Computes: softmax(Q @ K^T / sqrt(d_k) + Bias) @ V
    
    This is the key operation in Particle Transformer where pairwise particle
    interaction features are incorporated into the attention mechanism.
    
    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch, num_heads, seq_len, head_dim)
        bias: Bias tensor of shape (batch, num_heads, seq_len, seq_len)
              or (batch * num_heads, seq_len, seq_len)
        use_triton: Whether to use the Triton kernel (default: True)
    
    Returns:
        Output tensor of shape (batch, num_heads, seq_len, head_dim)
    
    Example:
        >>> batch, heads, seq_len, head_dim = 32, 8, 128, 64
        >>> q = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        >>> k = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        >>> v = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        >>> bias = torch.randn(batch, heads, seq_len, seq_len, device='cuda')
        >>> output = fused_attention_bias(q, k, v, bias)
    """
    if use_triton and q.is_cuda:
        return FusedAttentionBiasFunction.apply(q, k, v, bias)
    else:
        # Fallback to PyTorch implementation
        scale = 1.0 / (q.size(-1) ** 0.5)
        
        # Handle bias shape
        if bias.dim() == 3:
            batch = q.size(0)
            num_heads = q.size(1)
            seq_len = q.size(2)
            bias = bias.view(batch, num_heads, seq_len, seq_len)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + bias
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
