"""
cuDNN Frontend Attention Integration for Particle Transformer

This module provides integration with NVIDIA cuDNN Frontend's Attention operation
for accelerating the Particle Transformer model. The cuDNN attention implementation
provides optimized Flash Attention kernels that can be more efficient than custom
Triton kernels on certain hardware configurations.

Reference:
https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.9.0/operations/Attention.html

Requirements:
- PyTorch >= 2.0
- nvidia-cudnn-frontend >= 1.0
- cuDNN >= 9.12.0 (runtime requirement)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Try to import cuDNN frontend
try:
    import cudnn
    CUDNN_FRONTEND_AVAILABLE = True
except ImportError:
    CUDNN_FRONTEND_AVAILABLE = False
    cudnn = None


def check_cudnn_version() -> bool:
    """
    Check if cuDNN version is sufficient for frontend attention operations.
    
    Returns:
        True if cuDNN is available and version >= 9.12.0
    
    Note:
        Returns False if cudnn.backend_version() raises any exception,
        including AttributeError (function not available) or RuntimeError
        (cuDNN library not loaded).
    """
    if not CUDNN_FRONTEND_AVAILABLE:
        return False
    
    try:
        # Get cuDNN backend version
        version = cudnn.backend_version()
        # Version is encoded as major * 10000 + minor * 100 + patch
        # cuDNN 9.12.0 = 91200
        return version >= 91200
    except (AttributeError, RuntimeError, OSError):
        # AttributeError: backend_version not available
        # RuntimeError: cuDNN library issues
        # OSError: library loading issues
        return False


class CuDNNScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention using cuDNN Frontend.
    
    This module provides Flash Attention via NVIDIA cuDNN for efficient
    attention computation with optional additive bias (for ParT pairwise features).
    
    The cuDNN attention supports:
    - Fused QKV computation
    - Flash Attention memory optimization
    - Variable sequence lengths
    - Additive attention bias
    
    Args:
        head_dim: Dimension per attention head
        dropout_p: Dropout probability (default: 0.0)
        is_causal: Whether to use causal masking (default: False)
        scale: Attention scale factor (default: None, uses 1/sqrt(head_dim))
    """
    
    def __init__(
        self,
        head_dim: int,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()
        
        self.head_dim = head_dim
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.scale = scale if scale is not None else (1.0 / (head_dim ** 0.5))
        
        self._cudnn_available = check_cudnn_version()
        self._graph_cache = {}
    
    def _build_cudnn_graph(
        self,
        batch: int,
        num_heads: int,
        seq_len_q: int,
        seq_len_kv: int,
        dtype: torch.dtype,
        has_bias: bool,
    ) -> Optional["cudnn.Graph"]:
        """
        Build a cuDNN graph for the attention operation.
        
        Args:
            batch: Batch size
            num_heads: Number of attention heads
            seq_len_q: Query sequence length
            seq_len_kv: Key/Value sequence length
            dtype: Data type
            has_bias: Whether attention bias is used
        
        Returns:
            cuDNN Graph object or None if cuDNN is not available
        """
        if not self._cudnn_available:
            return None
        
        try:
            graph = cudnn.Graph()
            
            # Define tensor descriptors
            q_desc = graph.create_tensor_descriptor(
                dims=[batch, num_heads, seq_len_q, self.head_dim],
                strides=[num_heads * seq_len_q * self.head_dim,
                        seq_len_q * self.head_dim,
                        self.head_dim, 1],
                data_type=cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.FLOAT,
                name="Q"
            )
            
            k_desc = graph.create_tensor_descriptor(
                dims=[batch, num_heads, seq_len_kv, self.head_dim],
                strides=[num_heads * seq_len_kv * self.head_dim,
                        seq_len_kv * self.head_dim,
                        self.head_dim, 1],
                data_type=cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.FLOAT,
                name="K"
            )
            
            v_desc = graph.create_tensor_descriptor(
                dims=[batch, num_heads, seq_len_kv, self.head_dim],
                strides=[num_heads * seq_len_kv * self.head_dim,
                        seq_len_kv * self.head_dim,
                        self.head_dim, 1],
                data_type=cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.FLOAT,
                name="V"
            )
            
            # Define attention operation
            attn_options = {
                "is_causal": self.is_causal,
                "attn_scale": self.scale,
                "dropout_prob": self.dropout_p if self.training else 0.0,
            }
            
            if has_bias:
                bias_desc = graph.create_tensor_descriptor(
                    dims=[batch, num_heads, seq_len_q, seq_len_kv],
                    strides=[num_heads * seq_len_q * seq_len_kv,
                            seq_len_q * seq_len_kv,
                            seq_len_kv, 1],
                    data_type=cudnn.data_type.HALF if dtype == torch.float16 else cudnn.data_type.FLOAT,
                    name="Bias"
                )
                attn_options["bias"] = bias_desc
            
            # Add SDPA operation
            o_desc, stats_desc = graph.sdpa_forward(
                q=q_desc,
                k=k_desc,
                v=v_desc,
                **attn_options
            )
            
            # Build the graph
            graph.build()
            
            return graph
            
        except (RuntimeError, AttributeError, ValueError) as e:
            # RuntimeError: cuDNN graph building/execution errors
            # AttributeError: API not available in this cuDNN version
            # ValueError: Invalid tensor dimensions or parameters
            import warnings
            warnings.warn(f"cuDNN graph building failed ({type(e).__name__}): {e}. Falling back to PyTorch.")
            return None
            return None
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using cuDNN attention.
        
        Args:
            q: Query tensor (batch, num_heads, seq_len_q, head_dim)
            k: Key tensor (batch, num_heads, seq_len_kv, head_dim)
            v: Value tensor (batch, num_heads, seq_len_kv, head_dim)
            attn_bias: Optional attention bias (batch, num_heads, seq_len_q, seq_len_kv)
        
        Returns:
            Output tensor (batch, num_heads, seq_len_q, head_dim)
        """
        batch, num_heads, seq_len_q, head_dim = q.shape
        seq_len_kv = k.shape[2]
        
        # Try cuDNN first
        if self._cudnn_available and q.is_cuda:
            output = self._forward_cudnn(q, k, v, attn_bias)
            if output is not None:
                return output
        
        # Fallback to PyTorch
        return self._forward_pytorch(q, k, v, attn_bias)
    
    def _forward_cudnn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Execute attention using cuDNN backend.
        
        Returns None if cuDNN execution fails.
        """
        try:
            batch, num_heads, seq_len_q, head_dim = q.shape
            seq_len_kv = k.shape[2]
            has_bias = attn_bias is not None
            
            # Get or build graph
            cache_key = (batch, num_heads, seq_len_q, seq_len_kv, q.dtype, has_bias)
            
            if cache_key not in self._graph_cache:
                graph = self._build_cudnn_graph(
                    batch, num_heads, seq_len_q, seq_len_kv,
                    q.dtype, has_bias
                )
                if graph is None:
                    return None
                self._graph_cache[cache_key] = graph
            
            graph = self._graph_cache[cache_key]
            
            # Ensure contiguous tensors
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            if attn_bias is not None:
                attn_bias = attn_bias.contiguous()
            
            # Allocate output
            output = torch.empty_like(q)
            
            # Execute graph
            workspace_size = graph.get_workspace_size()
            workspace = torch.empty(workspace_size, dtype=torch.uint8, device=q.device)
            
            inputs = {"Q": q, "K": k, "V": v}
            if attn_bias is not None:
                inputs["Bias"] = attn_bias
            
            graph.execute(inputs, {"O": output}, workspace)
            
            return output
            
        except Exception:
            return None
    
    def _forward_pytorch(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        PyTorch fallback implementation.
        """
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_bias is not None:
            scores = scores + attn_bias
        
        if self.is_causal:
            seq_len_q = q.shape[2]
            seq_len_kv = k.shape[2]
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_kv, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        if self.dropout_p > 0.0 and self.training:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout_p)
        
        return torch.matmul(attn_weights, v)


class CuDNNMultiheadAttentionWithBias(nn.Module):
    """
    Multi-head attention with additive bias using cuDNN Frontend.
    
    This is a drop-in replacement for standard multi-head attention that uses
    cuDNN's optimized Flash Attention kernels when available.
    
    Args:
        embed_dim: Total embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in linear projections (default: True)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # cuDNN attention core
        self.attn = CuDNNScaledDotProductAttention(
            head_dim=self.head_dim,
            dropout_p=dropout,
            is_causal=False,
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
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
            need_weights: Whether to return attention weights (not supported with cuDNN)
        
        Returns:
            output: (seq_len, batch, embed_dim)
            attn_weights: None (not supported with cuDNN)
        """
        seq_len, batch, _ = query.shape
        
        # Project Q, K, V
        q = self.q_proj(query)
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
                attn_bias = attn_bias.view(batch, self.num_heads, seq_len, seq_len)
        
        # Handle key padding mask by adding to bias
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            if attn_bias is None:
                attn_bias = torch.zeros(batch, self.num_heads, seq_len, seq_len,
                                       dtype=q.dtype, device=q.device)
            attn_bias = attn_bias.masked_fill(mask, float('-inf'))
        
        # Apply attention
        output = self.attn(q, k, v, attn_bias)
        
        # Reshape back: (batch, num_heads, seq_len, head_dim) -> (seq_len, batch, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        output = output.transpose(0, 1)
        
        # Output projection
        output = self.out_proj(output)
        
        return output, None


# Module-level cache for attention instances (keyed by parameters)
_cudnn_attention_cache: dict = {}


def cudnn_attention_with_bias(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    bias: torch.Tensor,
    scale: Optional[float] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Functional interface for cuDNN attention with additive bias.
    
    This function provides a convenient way to use cuDNN Flash Attention
    with the pairwise particle interaction bias used in Particle Transformer.
    
    Note:
        Attention instances are cached based on (head_dim, dropout_p, is_causal, scale)
        to enable cuDNN graph caching for optimal performance.
    
    Args:
        q: Query tensor (batch, num_heads, seq_len, head_dim)
        k: Key tensor (batch, num_heads, seq_len, head_dim)
        v: Value tensor (batch, num_heads, seq_len, head_dim)
        bias: Attention bias (batch, num_heads, seq_len, seq_len) or
              (batch * num_heads, seq_len, seq_len)
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        dropout_p: Dropout probability (default: 0.0)
        is_causal: Whether to use causal masking (default: False)
    
    Returns:
        Output tensor (batch, num_heads, seq_len, head_dim)
    
    Example:
        >>> batch, heads, seq_len, head_dim = 32, 8, 128, 64
        >>> q = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        >>> k = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        >>> v = torch.randn(batch, heads, seq_len, head_dim, device='cuda')
        >>> bias = torch.randn(batch, heads, seq_len, seq_len, device='cuda')
        >>> output = cudnn_attention_with_bias(q, k, v, bias)
    """
    head_dim = q.shape[-1]
    batch = q.size(0)
    num_heads = q.size(1)
    seq_len = q.size(2)
    
    # Get or create cached attention instance
    cache_key = (head_dim, dropout_p, is_causal, scale)
    if cache_key not in _cudnn_attention_cache:
        _cudnn_attention_cache[cache_key] = CuDNNScaledDotProductAttention(
            head_dim=head_dim,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
    
    attn = _cudnn_attention_cache[cache_key]
    
    # Handle bias shape with validation
    if bias.dim() == 3:
        # Validate that reshaping is valid
        expected_elements = batch * num_heads * seq_len * seq_len
        if bias.numel() != expected_elements:
            raise ValueError(
                f"Cannot reshape bias with {bias.numel()} elements to "
                f"({batch}, {num_heads}, {seq_len}, {seq_len}) = {expected_elements} elements"
            )
        bias = bias.view(batch, num_heads, seq_len, seq_len)
    
    return attn(q, k, v, bias)
