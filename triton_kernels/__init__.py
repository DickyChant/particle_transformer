"""
Triton Kernels for Particle Transformer (ParT) Acceleration

This module provides optimized Triton kernels for key operations in the
Particle Transformer model, including:
- Fused pairwise Lorentz features computation
- Fused attention with bias (scaled dot-product attention with additive bias)

These kernels target the most computationally intensive parts of the ParT model,
particularly the pairwise interaction features and their incorporation into
the multi-head attention mechanism.
"""

from .attention_bias import (
    fused_attention_bias,
    FusedAttentionBiasFunction,
)
from .pairwise_features import (
    pairwise_lv_fts_triton,
    PairwiseLVFeaturesFunction,
)

__all__ = [
    "fused_attention_bias",
    "FusedAttentionBiasFunction",
    "pairwise_lv_fts_triton",
    "PairwiseLVFeaturesFunction",
]
