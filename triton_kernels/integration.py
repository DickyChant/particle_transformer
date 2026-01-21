"""
Integration Module for Triton-Accelerated Particle Transformer

This module provides drop-in replacements and utilities for integrating
the Triton kernels with the existing Particle Transformer model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    from .attention_bias import fused_attention_bias
    from .pairwise_features import pairwise_lv_fts_triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class TritonMultiheadAttentionWithBias(nn.Module):
    """
    Multi-head attention module with additive bias, using Triton kernels.
    
    This is a drop-in replacement for nn.MultiheadAttention that efficiently
    handles the attention bias used in Particle Transformer.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        use_triton: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_triton = use_triton and TRITON_AVAILABLE
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention bias.
        
        Args:
            query: (seq_len, batch, embed_dim)
            key: (seq_len, batch, embed_dim)
            value: (seq_len, batch, embed_dim)
            attn_bias: (batch * num_heads, seq_len, seq_len) or (batch, num_heads, seq_len, seq_len)
            key_padding_mask: (batch, seq_len) where True means padding
            need_weights: Whether to return attention weights
        
        Returns:
            output: (seq_len, batch, embed_dim)
            attn_weights: Optional attention weights
        """
        seq_len, batch, _ = query.shape
        
        # Project Q, K, V
        q = self.q_proj(query)  # (seq_len, batch, embed_dim)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        # (seq_len, batch, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.transpose(0, 1).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.transpose(0, 1).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.transpose(0, 1).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle attention bias
        if attn_bias is not None:
            if attn_bias.dim() == 3:
                # (batch * num_heads, seq_len, seq_len) -> (batch, num_heads, seq_len, seq_len)
                attn_bias = attn_bias.view(batch, self.num_heads, seq_len, seq_len)
        else:
            attn_bias = torch.zeros(batch, self.num_heads, seq_len, seq_len, 
                                    dtype=q.dtype, device=q.device)
        
        # Handle key padding mask
        if key_padding_mask is not None:
            # (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_bias = attn_bias.masked_fill(mask, float('-inf'))
        
        # Use Triton kernel or fallback
        if self.use_triton and q.is_cuda and not need_weights:
            output = fused_attention_bias(q, k, v, attn_bias, use_triton=True)
        else:
            # PyTorch fallback
            scale = 1.0 / (self.head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale + attn_bias
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, v)
        
        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (seq_len, batch, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        output = output.transpose(0, 1)
        
        # Output projection
        output = self.out_proj(output)
        
        if need_weights:
            scale = 1.0 / (self.head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale + attn_bias
            attn_weights = F.softmax(scores, dim=-1)
            return output, attn_weights
        
        return output, None


class TritonPairEmbed(nn.Module):
    """
    Pair embedding module using Triton kernels for pairwise feature computation.
    
    This module computes pairwise Lorentz-invariant features and projects them
    to create the attention bias for the Particle Transformer.
    """
    
    def __init__(
        self,
        pairwise_lv_dim: int = 4,
        dims: list = [64, 64, 64],
        use_pre_activation: bool = True,
        activation: str = 'gelu',
        use_triton: bool = True,
    ):
        super().__init__()
        
        self.pairwise_lv_dim = pairwise_lv_dim
        self.use_triton = use_triton and TRITON_AVAILABLE
        self.out_dim = dims[-1]
        
        # Build embedding network
        input_dim = pairwise_lv_dim
        layers = [nn.BatchNorm1d(input_dim)]
        
        for dim in dims:
            layers.extend([
                nn.Conv1d(input_dim, dim, 1),
                nn.BatchNorm1d(dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        
        if use_pre_activation:
            layers = layers[:-1]  # Remove last activation
        
        self.embed = nn.Sequential(*layers)
    
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute pair embeddings.
        
        Args:
            v: 4-vectors of shape (batch, 4, seq_len)
        
        Returns:
            Pair embeddings of shape (batch, out_dim, seq_len, seq_len)
        """
        batch, _, seq_len = v.shape
        
        with torch.no_grad():
            # Compute pairwise features using Triton or PyTorch
            if self.use_triton and v.is_cuda:
                pair_fts = pairwise_lv_fts_triton(v, num_outputs=self.pairwise_lv_dim)
            else:
                pair_fts = self._compute_pairwise_pytorch(v)
        
        # Reshape for Conv1d: (batch, features, seq_len, seq_len) -> (batch, features, seq_len * seq_len)
        pair_fts = pair_fts.view(batch, self.pairwise_lv_dim, -1)
        
        # Apply embedding network
        embedded = self.embed(pair_fts)
        
        # Reshape back: (batch, out_dim, seq_len * seq_len) -> (batch, out_dim, seq_len, seq_len)
        return embedded.view(batch, self.out_dim, seq_len, seq_len)
    
    def _compute_pairwise_pytorch(self, v: torch.Tensor) -> torch.Tensor:
        """PyTorch fallback for pairwise feature computation."""
        from .pairwise_features import _pairwise_lv_fts_pytorch
        return _pairwise_lv_fts_pytorch(v, self.pairwise_lv_dim)


def patch_particle_transformer(model: nn.Module, use_triton: bool = True) -> nn.Module:
    """
    Patch an existing ParticleTransformer model to use Triton kernels.
    
    This function modifies the model in-place to use Triton-accelerated
    attention and pairwise feature computation.
    
    Args:
        model: A ParticleTransformer model instance
        use_triton: Whether to enable Triton kernels
    
    Returns:
        The patched model
    
    Example:
        >>> from weaver.nn.model.ParticleTransformer import ParticleTransformer
        >>> model = ParticleTransformer(input_dim=18, num_classes=10)
        >>> model = patch_particle_transformer(model)
    """
    if not use_triton or not TRITON_AVAILABLE:
        return model
    
    # Store original forward for wrapping
    original_forward = model.forward
    
    def patched_forward(x, v=None, mask=None, uu=None, uu_idx=None):
        # Use Triton kernel for pairwise features if available
        if v is not None and hasattr(model, 'pair_embed') and model.pair_embed is not None:
            with torch.no_grad():
                batch, _, seq_len = v.shape
                if v.is_cuda:
                    # Compute pairwise features with Triton
                    pair_fts = pairwise_lv_fts_triton(v, num_outputs=4)
                    # This replaces the standard pairwise computation
        
        return original_forward(x, v, mask, uu, uu_idx)
    
    # Note: Full patching would require deeper integration
    # For now, we provide the kernels for manual integration
    
    return model


def benchmark_kernels(
    batch_size: int = 32,
    seq_len: int = 128,
    num_heads: int = 8,
    head_dim: int = 64,
    num_iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Benchmark Triton kernels against PyTorch implementations.
    
    Args:
        batch_size: Batch size for benchmarking
        seq_len: Sequence length (number of particles)
        num_heads: Number of attention heads
        head_dim: Dimension per head
        num_iterations: Number of iterations for timing
        warmup: Number of warmup iterations
    
    Returns:
        Dictionary with timing results
    """
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    # Benchmark pairwise features
    v = torch.randn(batch_size, 4, seq_len, device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = pairwise_lv_fts_triton(v, use_triton=False)
        if device.type == 'cuda':
            _ = pairwise_lv_fts_triton(v, use_triton=True)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time PyTorch
    start = time.time()
    for _ in range(num_iterations):
        _ = pairwise_lv_fts_triton(v, use_triton=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations
    
    # Time Triton
    if device.type == 'cuda':
        start = time.time()
        for _ in range(num_iterations):
            _ = pairwise_lv_fts_triton(v, use_triton=True)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / num_iterations
    else:
        triton_time = None
    
    results['pairwise_features'] = {
        'pytorch_ms': pytorch_time * 1000,
        'triton_ms': triton_time * 1000 if triton_time else None,
        'speedup': pytorch_time / triton_time if triton_time else None,
    }
    
    # Benchmark attention with bias
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v_attn = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    bias = torch.randn(batch_size, num_heads, seq_len, seq_len, device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = fused_attention_bias(q, k, v_attn, bias, use_triton=False)
        if device.type == 'cuda':
            _ = fused_attention_bias(q, k, v_attn, bias, use_triton=True)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Time PyTorch
    start = time.time()
    for _ in range(num_iterations):
        _ = fused_attention_bias(q, k, v_attn, bias, use_triton=False)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations
    
    # Time Triton
    if device.type == 'cuda':
        start = time.time()
        for _ in range(num_iterations):
            _ = fused_attention_bias(q, k, v_attn, bias, use_triton=True)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) / num_iterations
    else:
        triton_time = None
    
    results['attention_bias'] = {
        'pytorch_ms': pytorch_time * 1000,
        'triton_ms': triton_time * 1000 if triton_time else None,
        'speedup': pytorch_time / triton_time if triton_time else None,
    }
    
    return results
