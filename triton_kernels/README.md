# Triton Kernels for Particle Transformer Acceleration

This module provides optimized kernels for accelerating the Particle Transformer (ParT) model, with a focus on the pairwise interaction features and attention bias mechanism.

## Overview

The Particle Transformer uses pairwise particle interaction features incorporated into the multi-head attention as a **bias before softmax**. This module provides multiple backend options for these operations:

- **Triton**: Custom kernels for fine-grained control and flexibility
- **cuDNN Frontend**: NVIDIA's optimized Flash Attention (requires cuDNN >= 9.12.0)

### Key Operations Accelerated

1. **Pairwise Lorentz Features** (`pairwise_features.py`)
   - Computes physics-motivated features for each particle pair:
     - `ln(kt)`: log of the kT variable (`min(pt_i, pt_j) * ΔR`)
     - `ln(z)`: log of the momentum fraction (`min(pt_i, pt_j) / (pt_i + pt_j)`)
     - `ln(ΔR)`: log of the angular distance in η-φ space
     - `ln(m²)`: log of the invariant mass squared of the pair

2. **Fused Attention with Bias** (`attention_bias.py`, `cudnn_attention.py`)
   - Combines QK^T computation, bias addition, softmax, and output projection
   - Implements the key attention bias formula: `softmax(Q @ K^T / √d_k + Bias) @ V`

## Installation

### Basic (Triton backend)

```bash
pip install torch triton
```

### With cuDNN Frontend (optional, for Flash Attention)

```bash
pip install torch nvidia-cudnn-frontend
```

**Note:** cuDNN Frontend requires cuDNN >= 9.12.0 at runtime.

## Usage

### Pairwise Lorentz Features

```python
from triton_kernels import pairwise_lv_fts_triton

# Input: 4-vectors (px, py, pz, E) for each particle
# Shape: (batch, 4, seq_len)
x = torch.randn(32, 4, 128, device='cuda')

# Compute pairwise features
# Output shape: (batch, 4, seq_len, seq_len)
features = pairwise_lv_fts_triton(x, num_outputs=4)
```

### Fused Attention with Bias (Triton)

```python
from triton_kernels import fused_attention_bias

# Standard attention inputs
batch, num_heads, seq_len, head_dim = 32, 8, 128, 64
q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')

# Pairwise interaction bias (from PairEmbed)
bias = torch.randn(batch, num_heads, seq_len, seq_len, device='cuda')

# Fused computation with Triton
output = fused_attention_bias(q, k, v, bias)
```

### Fused Attention with Bias (cuDNN)

```python
from triton_kernels import cudnn_attention_with_bias, check_cudnn_version

# Check if cuDNN is available
if check_cudnn_version():
    output = cudnn_attention_with_bias(q, k, v, bias)
else:
    # Fall back to Triton or PyTorch
    output = fused_attention_bias(q, k, v, bias)
```

### cuDNN Multi-head Attention Module

```python
from triton_kernels import CuDNNMultiheadAttentionWithBias

# Drop-in replacement using cuDNN Flash Attention
attn = CuDNNMultiheadAttentionWithBias(
    embed_dim=256,
    num_heads=8,
    dropout=0.1
).cuda()

# Forward pass (seq_len, batch, embed_dim format)
output, _ = attn(query, key, value, attn_bias=bias)
```

### Integration with Existing Model

```python
from triton_kernels.integration import TritonMultiheadAttentionWithBias, TritonPairEmbed

# Drop-in replacement for nn.MultiheadAttention (Triton backend)
attn = TritonMultiheadAttentionWithBias(
    embed_dim=256,
    num_heads=8,
    use_triton=True
)

# Accelerated pair embedding
pair_embed = TritonPairEmbed(
    pairwise_lv_dim=4,
    dims=[64, 64, 8],
    use_triton=True
)
```

### Benchmarking

```python
from triton_kernels.integration import benchmark_kernels

results = benchmark_kernels(
    batch_size=32,
    seq_len=128,
    num_heads=8,
    head_dim=64,
)

print(f"Pairwise Features Speedup: {results['pairwise_features']['speedup']:.2f}x")
print(f"Attention with Bias Speedup: {results['attention_bias']['speedup']:.2f}x")
```

## Testing

Run the test suite:

```bash
pytest triton_kernels/test_kernels.py -v
```

## API Reference

### `pairwise_lv_fts_triton(x, num_outputs=4, eps=1e-8, use_triton=True)`

Compute pairwise Lorentz-invariant features.

**Args:**
- `x`: 4-vector tensor of shape `(batch, 4, seq_len)` with `(px, py, pz, E)`
- `num_outputs`: Number of output features (1-4)
- `eps`: Small epsilon for numerical stability
- `use_triton`: Whether to use Triton kernel (falls back to PyTorch if False or no GPU)

**Returns:**
- Pairwise features of shape `(batch, num_outputs, seq_len, seq_len)`

### `fused_attention_bias(q, k, v, bias, use_triton=True)`

Fused scaled dot-product attention with additive bias (Triton backend).

**Args:**
- `q`: Query tensor `(batch, num_heads, seq_len, head_dim)`
- `k`: Key tensor `(batch, num_heads, seq_len, head_dim)`
- `v`: Value tensor `(batch, num_heads, seq_len, head_dim)`
- `bias`: Bias tensor `(batch, num_heads, seq_len, seq_len)` or `(batch * num_heads, seq_len, seq_len)`
- `use_triton`: Whether to use Triton kernel

**Returns:**
- Output tensor `(batch, num_heads, seq_len, head_dim)`

### `cudnn_attention_with_bias(q, k, v, bias, scale=None, dropout_p=0.0, is_causal=False)`

Fused scaled dot-product attention with additive bias (cuDNN Flash Attention backend).

**Args:**
- `q`: Query tensor `(batch, num_heads, seq_len, head_dim)`
- `k`: Key tensor `(batch, num_heads, seq_len, head_dim)`
- `v`: Value tensor `(batch, num_heads, seq_len, head_dim)`
- `bias`: Bias tensor `(batch, num_heads, seq_len, seq_len)`
- `scale`: Attention scale factor (default: `1/sqrt(head_dim)`)
- `dropout_p`: Dropout probability
- `is_causal`: Whether to use causal masking

**Returns:**
- Output tensor `(batch, num_heads, seq_len, head_dim)`

### `check_cudnn_version()`

Check if cuDNN version is sufficient for frontend attention operations.

**Returns:**
- `True` if cuDNN >= 9.12.0 is available

## Performance Notes

- The Triton kernels are optimized for the typical ParT parameters:
  - Sequence length: 128 particles
  - Number of heads: 8
  - Head dimension: 64
  - Batch size: 32-512

- For very small batch sizes or sequence lengths, the PyTorch fallback may be comparable or faster.

- The kernels support both FP32 and FP16 (via AMP) computations.

- **cuDNN backend** may offer better performance on NVIDIA GPUs with sufficient cuDNN version,
  especially for larger batch sizes and sequence lengths.

## Architecture

```
triton_kernels/
├── __init__.py              # Package exports
├── attention_bias.py        # Fused attention with bias (Triton)
├── cudnn_attention.py       # cuDNN Frontend attention integration
├── pairwise_features.py     # Pairwise Lorentz features kernel
├── integration.py           # Integration utilities and drop-in modules
├── test_kernels.py          # Test suite
└── README.md                # This file
```

## References

- [Particle Transformer Paper](https://arxiv.org/abs/2202.03772)
- [Triton Documentation](https://triton-lang.org/)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [cuDNN Frontend Attention](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.9.0/operations/Attention.html)
