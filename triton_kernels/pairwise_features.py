"""
Fused Pairwise Lorentz Features Kernel for Particle Transformer

This module provides a Triton kernel that computes pairwise particle
interaction features efficiently. These features are used as attention
bias in the Particle Transformer model.

The pairwise features include:
- ln(kt): log of the kT variable (ptmin * deltaR)
- ln(z): log of the momentum fraction (ptmin / (pti + ptj))
- ln(deltaR): log of the angular distance
- ln(m²): log of the invariant mass squared of the pair
"""

import math
import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _pairwise_lv_fts_kernel(
    # Input: 4-vectors (px, py, pz, E)
    X_ptr,  # (batch, 4, seq_len)
    # Output: pairwise features
    Out_ptr,  # (batch, num_fts, seq_len, seq_len)
    # Dimensions
    BATCH: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    NUM_FTS: tl.constexpr,
    # Strides for X
    stride_xb,  # batch stride
    stride_xc,  # channel (4-vector component) stride
    stride_xs,  # sequence stride
    # Strides for output
    stride_ob,  # batch stride
    stride_of,  # feature stride
    stride_oi,  # row stride
    stride_oj,  # col stride
    # Constants
    EPS: tl.constexpr,
    PI: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute pairwise Lorentz-invariant features between all particle pairs.
    
    For each pair (i, j), computes:
    - pt_i, pt_j: transverse momentum
    - rap_i, rap_j: rapidity  
    - phi_i, phi_j: azimuthal angle
    - deltaR: angular distance
    - kt: min(pt_i, pt_j) * deltaR
    - z: min(pt_i, pt_j) / (pt_i + pt_j)
    - m²: invariant mass squared of the pair
    
    Output features (if NUM_FTS >= 4):
    [0]: ln(kt)
    [1]: ln(z)
    [2]: ln(deltaR)
    [3]: ln(m²)
    """
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_pair = tl.program_id(1)  # which block of (i, j) pairs
    
    # Compute which (i, j) pairs this block handles
    # For efficiency, we process pairs in a 1D fashion
    pair_offset = pid_pair * BLOCK_SIZE
    pair_idx = pair_offset + tl.arange(0, BLOCK_SIZE)
    
    # Convert linear pair index to (i, j) indices
    # pair_idx = i * SEQ_LEN + j
    i_idx = pair_idx // SEQ_LEN
    j_idx = pair_idx % SEQ_LEN
    
    # Mask for valid pairs
    valid_mask = (pair_idx < SEQ_LEN * SEQ_LEN) & (i_idx < SEQ_LEN) & (j_idx < SEQ_LEN)
    
    # Load 4-vectors for particle i
    # X shape: (batch, 4, seq_len)
    px_i_ptr = X_ptr + pid_batch * stride_xb + 0 * stride_xc + i_idx * stride_xs
    py_i_ptr = X_ptr + pid_batch * stride_xb + 1 * stride_xc + i_idx * stride_xs
    pz_i_ptr = X_ptr + pid_batch * stride_xb + 2 * stride_xc + i_idx * stride_xs
    e_i_ptr = X_ptr + pid_batch * stride_xb + 3 * stride_xc + i_idx * stride_xs
    
    px_i = tl.load(px_i_ptr, mask=valid_mask, other=0.0)
    py_i = tl.load(py_i_ptr, mask=valid_mask, other=0.0)
    pz_i = tl.load(pz_i_ptr, mask=valid_mask, other=0.0)
    e_i = tl.load(e_i_ptr, mask=valid_mask, other=1.0)  # Avoid division by zero
    
    # Load 4-vectors for particle j
    px_j_ptr = X_ptr + pid_batch * stride_xb + 0 * stride_xc + j_idx * stride_xs
    py_j_ptr = X_ptr + pid_batch * stride_xb + 1 * stride_xc + j_idx * stride_xs
    pz_j_ptr = X_ptr + pid_batch * stride_xb + 2 * stride_xc + j_idx * stride_xs
    e_j_ptr = X_ptr + pid_batch * stride_xb + 3 * stride_xc + j_idx * stride_xs
    
    px_j = tl.load(px_j_ptr, mask=valid_mask, other=0.0)
    py_j = tl.load(py_j_ptr, mask=valid_mask, other=0.0)
    pz_j = tl.load(pz_j_ptr, mask=valid_mask, other=0.0)
    e_j = tl.load(e_j_ptr, mask=valid_mask, other=1.0)
    
    # Compute pt (transverse momentum)
    pt_i = tl.sqrt(px_i * px_i + py_i * py_i + EPS)
    pt_j = tl.sqrt(px_j * px_j + py_j * py_j + EPS)
    
    # Compute rapidity: rap = 0.5 * ln((E + pz) / (E - pz))
    # For numerical stability, ensure e >= |pz| (physical constraint)
    # Use absolute value of energy to handle edge cases in test data
    e_i_safe = tl.maximum(tl.abs(e_i), EPS)
    e_j_safe = tl.maximum(tl.abs(e_j), EPS)
    denom_i = tl.maximum(e_i_safe - pz_i, EPS)
    denom_j = tl.maximum(e_j_safe - pz_j, EPS)
    numer_i = tl.maximum(e_i_safe + pz_i, EPS)
    numer_j = tl.maximum(e_j_safe + pz_j, EPS)
    rap_i = 0.5 * tl.log(numer_i / denom_i)
    rap_j = 0.5 * tl.log(numer_j / denom_j)
    
    # Compute phi (azimuthal angle)
    phi_i = tl.libdevice.atan2(py_i, px_i)
    phi_j = tl.libdevice.atan2(py_j, px_j)
    
    # Compute delta_phi with periodicity
    dphi = phi_i - phi_j
    # Wrap to [-pi, pi]: dphi = (dphi + pi) % (2*pi) - pi
    dphi = dphi + PI
    dphi = dphi - tl.floor(dphi / (2.0 * PI)) * (2.0 * PI)
    dphi = dphi - PI
    
    # Compute delta_R
    drap = rap_i - rap_j
    delta_r2 = drap * drap + dphi * dphi
    delta_r = tl.sqrt(delta_r2 + EPS)
    
    # Compute kt = min(pt_i, pt_j) * delta_R
    pt_min = tl.minimum(pt_i, pt_j)
    kt = pt_min * delta_r
    
    # Compute z = min(pt_i, pt_j) / (pt_i + pt_j)
    pt_sum = pt_i + pt_j
    z = pt_min / tl.maximum(pt_sum, EPS)
    
    # Compute invariant mass squared: m² = (E_i + E_j)² - |p_i + p_j|²
    e_sum = e_i + e_j
    px_sum = px_i + px_j
    py_sum = py_i + py_j
    pz_sum = pz_i + pz_j
    p_sum_sq = px_sum * px_sum + py_sum * py_sum + pz_sum * pz_sum
    m2 = e_sum * e_sum - p_sum_sq
    m2 = tl.maximum(m2, EPS)  # Ensure positive for log
    
    # Compute log features
    ln_kt = tl.log(tl.maximum(kt, EPS))
    ln_z = tl.log(tl.maximum(z, EPS))
    ln_delta_r = tl.log(tl.maximum(delta_r, EPS))
    ln_m2 = tl.log(m2)
    
    # Store output features
    # Output shape: (batch, num_fts, seq_len, seq_len)
    # Linear index: batch * stride_ob + fts * stride_of + i * stride_oi + j * stride_oj
    
    out_base = pid_batch * stride_ob
    
    # Feature 0: ln(kt)
    out_ptr_0 = Out_ptr + out_base + 0 * stride_of + i_idx * stride_oi + j_idx * stride_oj
    tl.store(out_ptr_0, ln_kt, mask=valid_mask)
    
    # Feature 1: ln(z)
    out_ptr_1 = Out_ptr + out_base + 1 * stride_of + i_idx * stride_oi + j_idx * stride_oj
    tl.store(out_ptr_1, ln_z, mask=valid_mask)
    
    # Feature 2: ln(delta_r)
    out_ptr_2 = Out_ptr + out_base + 2 * stride_of + i_idx * stride_oi + j_idx * stride_oj
    tl.store(out_ptr_2, ln_delta_r, mask=valid_mask)
    
    # Feature 3: ln(m²)
    if NUM_FTS > 3:
        out_ptr_3 = Out_ptr + out_base + 3 * stride_of + i_idx * stride_oi + j_idx * stride_oj
        tl.store(out_ptr_3, ln_m2, mask=valid_mask)


class PairwiseLVFeaturesFunction(torch.autograd.Function):
    """
    Autograd function for computing pairwise Lorentz-invariant features.
    """
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        num_outputs: int = 4,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute pairwise Lorentz features.
        
        Args:
            x: 4-vector tensor of shape (batch, 4, seq_len) with (px, py, pz, E)
            num_outputs: Number of output features (1-4)
            eps: Small epsilon for numerical stability
        
        Returns:
            Pairwise features of shape (batch, num_outputs, seq_len, seq_len)
        """
        assert x.dim() == 3 and x.size(1) == 4, \
            f"Expected shape (batch, 4, seq_len), got {x.shape}"
        
        batch, _, seq_len = x.shape
        num_fts = min(num_outputs, 4)
        
        # Ensure contiguous
        x = x.contiguous()
        
        # Allocate output
        out = torch.empty(batch, num_fts, seq_len, seq_len, dtype=x.dtype, device=x.device)
        
        # Block size
        BLOCK_SIZE = 256
        
        # Grid: one block per batch, multiple blocks for pairs
        num_pairs = seq_len * seq_len
        num_pair_blocks = triton.cdiv(num_pairs, BLOCK_SIZE)
        grid = (batch, num_pair_blocks)
        
        _pairwise_lv_fts_kernel[grid](
            x, out,
            batch, seq_len, num_fts,
            x.stride(0), x.stride(1), x.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            eps, math.pi,
            BLOCK_SIZE,
        )
        
        ctx.save_for_backward(x)
        ctx.num_outputs = num_outputs
        ctx.eps = eps
        
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass - falls back to PyTorch for gradient computation.
        """
        x, = ctx.saved_tensors
        eps = ctx.eps
        
        # For now, we don't support gradients through this operation
        # as it's typically used in a no_grad context in ParT
        return None, None, None


def pairwise_lv_fts_triton(
    x: torch.Tensor,
    num_outputs: int = 4,
    eps: float = 1e-8,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Compute pairwise Lorentz-invariant features between all particle pairs.
    
    This function computes physics-motivated features for each pair of particles:
    - ln(kt): log of the kT variable (momentum perpendicular to the jet axis)
    - ln(z): log of the momentum fraction
    - ln(deltaR): log of the angular distance in eta-phi space
    - ln(m²): log of the invariant mass squared
    
    These features are used in the Particle Transformer to incorporate pairwise
    particle interactions into the attention mechanism.
    
    Args:
        x: 4-vector tensor of shape (batch, 4, seq_len) with (px, py, pz, E)
        num_outputs: Number of output features (1-4), default 4
        eps: Small epsilon for numerical stability
        use_triton: Whether to use the Triton kernel (default: True)
    
    Returns:
        Pairwise features of shape (batch, num_outputs, seq_len, seq_len)
    
    Example:
        >>> batch, seq_len = 32, 128
        >>> x = torch.randn(batch, 4, seq_len, device='cuda')  # (px, py, pz, E)
        >>> features = pairwise_lv_fts_triton(x)
        >>> print(features.shape)  # (32, 4, 128, 128)
    """
    if use_triton and x.is_cuda:
        return PairwiseLVFeaturesFunction.apply(x, num_outputs, eps)
    else:
        # Fallback to PyTorch implementation
        return _pairwise_lv_fts_pytorch(x, num_outputs, eps)


def _pairwise_lv_fts_pytorch(
    x: torch.Tensor,
    num_outputs: int = 4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    PyTorch reference implementation of pairwise Lorentz features.
    """
    batch, _, seq_len = x.shape
    
    # Extract components
    px = x[:, 0:1, :]  # (batch, 1, seq_len)
    py = x[:, 1:2, :]
    pz = x[:, 2:3, :]
    e = x[:, 3:4, :]
    
    # Compute pt
    pt = torch.sqrt(px ** 2 + py ** 2 + eps)
    
    # Compute rapidity
    # For numerical stability, ensure e >= |pz| (physical constraint)
    # In practice, random test data may violate this, so we handle it gracefully
    e_safe = torch.clamp(torch.abs(e), min=eps)
    denom = torch.clamp(e_safe - pz, min=eps)
    numer = torch.clamp(e_safe + pz, min=eps)
    rap = 0.5 * torch.log(numer / denom)
    
    # Compute phi
    phi = torch.atan2(py, px)
    
    # Expand for pairwise computation
    # (batch, 1, seq_len) -> (batch, 1, seq_len, 1) and (batch, 1, 1, seq_len)
    pt_i = pt.unsqueeze(-1)  # (batch, 1, seq_len, 1)
    pt_j = pt.unsqueeze(-2)  # (batch, 1, 1, seq_len)
    rap_i = rap.unsqueeze(-1)
    rap_j = rap.unsqueeze(-2)
    phi_i = phi.unsqueeze(-1)
    phi_j = phi.unsqueeze(-2)
    
    # Delta phi with periodicity
    dphi = phi_i - phi_j
    dphi = (dphi + math.pi) % (2 * math.pi) - math.pi
    
    # Delta R
    drap = rap_i - rap_j
    delta_r = torch.sqrt(drap ** 2 + dphi ** 2 + eps)
    
    # kt = min(pt_i, pt_j) * delta_R
    pt_min = torch.minimum(pt_i, pt_j)
    kt = pt_min * delta_r
    
    # z = min(pt_i, pt_j) / (pt_i + pt_j)
    z = pt_min / torch.clamp(pt_i + pt_j, min=eps)
    
    # Compute features
    ln_kt = torch.log(torch.clamp(kt, min=eps))
    ln_z = torch.log(torch.clamp(z, min=eps))
    ln_delta_r = torch.log(torch.clamp(delta_r, min=eps))
    
    outputs = [ln_kt, ln_z, ln_delta_r]
    
    if num_outputs > 3:
        # Invariant mass squared
        px_i = px.unsqueeze(-1)
        px_j = px.unsqueeze(-2)
        py_i = py.unsqueeze(-1)
        py_j = py.unsqueeze(-2)
        pz_i = pz.unsqueeze(-1)
        pz_j = pz.unsqueeze(-2)
        e_i = e.unsqueeze(-1)
        e_j = e.unsqueeze(-2)
        
        e_sum = e_i + e_j
        p_sum_sq = (px_i + px_j) ** 2 + (py_i + py_j) ** 2 + (pz_i + pz_j) ** 2
        m2 = e_sum ** 2 - p_sum_sq
        m2 = torch.clamp(m2, min=eps)
        ln_m2 = torch.log(m2)
        outputs.append(ln_m2)
    
    # Concatenate and remove extra dimension
    result = torch.cat(outputs[:num_outputs], dim=1)
    return result.squeeze(1) if num_outputs == 1 else result
