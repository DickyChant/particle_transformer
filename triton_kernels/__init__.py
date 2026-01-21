"""
Triton Kernels for Particle Transformer (ParT) Acceleration

This module provides optimized kernels for key operations in the
Particle Transformer model, including:
- Fused pairwise Lorentz features computation (Triton)
- Fused attention with bias (Triton and cuDNN backends)

These kernels target the most computationally intensive parts of the ParT model,
particularly the pairwise interaction features and their incorporation into
the multi-head attention mechanism.

Backends:
- Triton: Custom kernels for fine-grained control and flexibility
- cuDNN Frontend: NVIDIA's optimized Flash Attention (requires cuDNN >= 9.12.0)
"""

from .attention_bias import (
    fused_attention_bias,
    FusedAttentionBiasFunction,
)
from .pairwise_features import (
    pairwise_lv_fts_triton,
    PairwiseLVFeaturesFunction,
)

# cuDNN backend (optional, requires nvidia-cudnn-frontend and cuDNN >= 9.12.0)
try:
    from .cudnn_attention import (
        cudnn_attention_with_bias,
        CuDNNScaledDotProductAttention,
        CuDNNMultiheadAttentionWithBias,
        check_cudnn_version,
        CUDNN_FRONTEND_AVAILABLE,
    )
except ImportError:
    CUDNN_FRONTEND_AVAILABLE = False
    cudnn_attention_with_bias = None
    CuDNNScaledDotProductAttention = None
    CuDNNMultiheadAttentionWithBias = None
    check_cudnn_version = lambda: False

__all__ = [
    # Triton kernels
    "fused_attention_bias",
    "FusedAttentionBiasFunction",
    "pairwise_lv_fts_triton",
    "PairwiseLVFeaturesFunction",
    # cuDNN backend
    "cudnn_attention_with_bias",
    "CuDNNScaledDotProductAttention",
    "CuDNNMultiheadAttentionWithBias",
    "check_cudnn_version",
    "CUDNN_FRONTEND_AVAILABLE",
]
