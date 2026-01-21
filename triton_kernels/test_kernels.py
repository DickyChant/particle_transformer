"""
Tests for Triton Kernels

This module tests the correctness of the Triton kernels against
PyTorch reference implementations.
"""

import math
import pytest
import torch
import torch.nn.functional as F

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for Triton kernel tests"
)


class TestPairwiseLVFeatures:
    """Tests for pairwise Lorentz-invariant features kernel."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        from triton_kernels.pairwise_features import pairwise_lv_fts_triton
        
        batch, seq_len = 4, 32
        x = torch.randn(batch, 4, seq_len, device='cuda')
        
        for num_outputs in [1, 2, 3, 4]:
            out = pairwise_lv_fts_triton(x, num_outputs=num_outputs)
            expected_shape = (batch, num_outputs, seq_len, seq_len)
            assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    
    def test_correctness_vs_pytorch(self):
        """Test that Triton kernel produces same results as PyTorch."""
        from triton_kernels.pairwise_features import pairwise_lv_fts_triton
        
        batch, seq_len = 4, 32
        torch.manual_seed(42)
        x = torch.randn(batch, 4, seq_len, device='cuda')
        
        # Compute with Triton
        triton_out = pairwise_lv_fts_triton(x, num_outputs=4, use_triton=True)
        
        # Compute with PyTorch
        pytorch_out = pairwise_lv_fts_triton(x, num_outputs=4, use_triton=False)
        
        # Compare
        torch.testing.assert_close(triton_out, pytorch_out, rtol=1e-3, atol=1e-3)
    
    def test_numerical_stability(self):
        """Test kernel handles edge cases (very small/large values)."""
        from triton_kernels.pairwise_features import pairwise_lv_fts_triton
        
        batch, seq_len = 2, 16
        
        # Test with small values
        x_small = torch.randn(batch, 4, seq_len, device='cuda') * 1e-6
        out_small = pairwise_lv_fts_triton(x_small, num_outputs=4)
        assert torch.isfinite(out_small).all(), "Output contains NaN/Inf for small inputs"
        
        # Test with large values
        x_large = torch.randn(batch, 4, seq_len, device='cuda') * 1e6
        out_large = pairwise_lv_fts_triton(x_large, num_outputs=4)
        assert torch.isfinite(out_large).all(), "Output contains NaN/Inf for large inputs"
    
    def test_batch_independence(self):
        """Test that batches are processed independently."""
        from triton_kernels.pairwise_features import pairwise_lv_fts_triton
        
        batch, seq_len = 4, 32
        torch.manual_seed(42)
        x = torch.randn(batch, 4, seq_len, device='cuda')
        
        # Process full batch
        full_out = pairwise_lv_fts_triton(x, num_outputs=4)
        
        # Process each sample individually
        for i in range(batch):
            single_out = pairwise_lv_fts_triton(x[i:i+1], num_outputs=4)
            torch.testing.assert_close(full_out[i:i+1], single_out, rtol=1e-4, atol=1e-4)


class TestFusedAttentionBias:
    """Tests for fused attention with bias kernel."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        from triton_kernels.attention_bias import fused_attention_bias
        
        batch, num_heads, seq_len, head_dim = 4, 8, 32, 64
        q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        bias = torch.randn(batch, num_heads, seq_len, seq_len, device='cuda')
        
        out = fused_attention_bias(q, k, v, bias)
        
        assert out.shape == q.shape, f"Expected {q.shape}, got {out.shape}"
    
    def test_correctness_vs_pytorch(self):
        """Test that Triton kernel produces same results as PyTorch."""
        from triton_kernels.attention_bias import fused_attention_bias
        
        batch, num_heads, seq_len, head_dim = 2, 4, 16, 32
        torch.manual_seed(42)
        
        q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        bias = torch.randn(batch, num_heads, seq_len, seq_len, device='cuda')
        
        # Compute with Triton
        triton_out = fused_attention_bias(q, k, v, bias, use_triton=True)
        
        # Compute with PyTorch
        pytorch_out = fused_attention_bias(q, k, v, bias, use_triton=False)
        
        # Compare
        torch.testing.assert_close(triton_out, pytorch_out, rtol=1e-2, atol=1e-2)
    
    def test_bias_effect(self):
        """Test that bias actually affects the output."""
        from triton_kernels.attention_bias import fused_attention_bias
        
        batch, num_heads, seq_len, head_dim = 2, 4, 16, 32
        torch.manual_seed(42)
        
        q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        
        bias_zero = torch.zeros(batch, num_heads, seq_len, seq_len, device='cuda')
        bias_nonzero = torch.randn(batch, num_heads, seq_len, seq_len, device='cuda')
        
        out_zero = fused_attention_bias(q, k, v, bias_zero)
        out_nonzero = fused_attention_bias(q, k, v, bias_nonzero)
        
        # Outputs should be different
        assert not torch.allclose(out_zero, out_nonzero), "Bias should affect output"
    
    def test_3d_bias_input(self):
        """Test that 3D bias (batch * num_heads, seq_len, seq_len) works."""
        from triton_kernels.attention_bias import fused_attention_bias
        
        batch, num_heads, seq_len, head_dim = 2, 4, 16, 32
        torch.manual_seed(42)
        
        q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        bias_4d = torch.randn(batch, num_heads, seq_len, seq_len, device='cuda')
        bias_3d = bias_4d.view(batch * num_heads, seq_len, seq_len)
        
        out_4d = fused_attention_bias(q, k, v, bias_4d)
        out_3d = fused_attention_bias(q, k, v, bias_3d)
        
        torch.testing.assert_close(out_4d, out_3d)
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through the backward pass."""
        from triton_kernels.attention_bias import fused_attention_bias
        
        batch, num_heads, seq_len, head_dim = 2, 4, 16, 32
        
        q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
        bias = torch.randn(batch, num_heads, seq_len, seq_len, device='cuda', requires_grad=True)
        
        out = fused_attention_bias(q, k, v, bias, use_triton=True)
        loss = out.sum()
        loss.backward()
        
        # Check that gradients exist and are finite
        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert bias.grad is not None, "No gradient for bias"
        
        assert torch.isfinite(q.grad).all(), "q.grad contains NaN/Inf"
        assert torch.isfinite(k.grad).all(), "k.grad contains NaN/Inf"
        assert torch.isfinite(v.grad).all(), "v.grad contains NaN/Inf"
        assert torch.isfinite(bias.grad).all(), "bias.grad contains NaN/Inf"


class TestIntegration:
    """Tests for integration utilities."""
    
    def test_triton_multihead_attention(self):
        """Test TritonMultiheadAttentionWithBias module."""
        from triton_kernels.integration import TritonMultiheadAttentionWithBias
        
        embed_dim, num_heads = 256, 8
        seq_len, batch = 32, 4
        
        attn = TritonMultiheadAttentionWithBias(embed_dim, num_heads).cuda()
        
        query = torch.randn(seq_len, batch, embed_dim, device='cuda')
        key = torch.randn(seq_len, batch, embed_dim, device='cuda')
        value = torch.randn(seq_len, batch, embed_dim, device='cuda')
        attn_bias = torch.randn(batch * num_heads, seq_len, seq_len, device='cuda')
        
        output, _ = attn(query, key, value, attn_bias)
        
        assert output.shape == (seq_len, batch, embed_dim)
    
    def test_triton_pair_embed(self):
        """Test TritonPairEmbed module."""
        from triton_kernels.integration import TritonPairEmbed
        
        batch, seq_len = 4, 32
        dims = [64, 64, 8]  # Output 8 heads
        
        pair_embed = TritonPairEmbed(pairwise_lv_dim=4, dims=dims).cuda()
        
        v = torch.randn(batch, 4, seq_len, device='cuda')
        
        out = pair_embed(v)
        
        assert out.shape == (batch, dims[-1], seq_len, seq_len)


class TestBenchmark:
    """Benchmark tests (can be skipped in CI)."""
    
    @pytest.mark.slow
    def test_benchmark(self):
        """Run benchmark and print results."""
        from triton_kernels.integration import benchmark_kernels
        
        results = benchmark_kernels(
            batch_size=32,
            seq_len=128,
            num_heads=8,
            head_dim=64,
            num_iterations=100,
            warmup=10,
        )
        
        print("\n=== Benchmark Results ===")
        for name, metrics in results.items():
            print(f"\n{name}:")
            print(f"  PyTorch: {metrics['pytorch_ms']:.3f} ms")
            if metrics['triton_ms']:
                print(f"  Triton: {metrics['triton_ms']:.3f} ms")
                print(f"  Speedup: {metrics['speedup']:.2f}x")


class TestCuDNNAttention:
    """Tests for cuDNN Frontend attention integration."""
    
    def test_cudnn_availability_check(self):
        """Test cuDNN version checking function."""
        from triton_kernels.cudnn_attention import check_cudnn_version, CUDNN_FRONTEND_AVAILABLE
        
        # Should return a boolean
        result = check_cudnn_version()
        assert isinstance(result, bool)
        
        # If cudnn-frontend is not installed, should return False
        if not CUDNN_FRONTEND_AVAILABLE:
            assert result is False
    
    def test_cudnn_scaled_dot_product_attention_fallback(self):
        """Test that cuDNN attention falls back to PyTorch when unavailable."""
        from triton_kernels.cudnn_attention import CuDNNScaledDotProductAttention
        
        batch, num_heads, seq_len, head_dim = 2, 4, 16, 32
        
        attn = CuDNNScaledDotProductAttention(head_dim=head_dim).cuda()
        
        q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        bias = torch.randn(batch, num_heads, seq_len, seq_len, device='cuda')
        
        # Should work even without cuDNN (falls back to PyTorch)
        output = attn(q, k, v, bias)
        
        assert output.shape == q.shape
        assert torch.isfinite(output).all()
    
    def test_cudnn_attention_correctness(self):
        """Test cuDNN attention produces correct results (vs PyTorch reference)."""
        from triton_kernels.cudnn_attention import CuDNNScaledDotProductAttention
        
        batch, num_heads, seq_len, head_dim = 2, 4, 16, 32
        torch.manual_seed(42)
        
        attn = CuDNNScaledDotProductAttention(head_dim=head_dim).cuda()
        
        q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        bias = torch.randn(batch, num_heads, seq_len, seq_len, device='cuda')
        
        # cuDNN attention (will use PyTorch fallback if cuDNN unavailable)
        cudnn_out = attn(q, k, v, bias)
        
        # PyTorch reference
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale + bias
        pytorch_out = torch.matmul(torch.softmax(scores, dim=-1), v)
        
        torch.testing.assert_close(cudnn_out, pytorch_out, rtol=1e-3, atol=1e-3)
    
    def test_cudnn_multihead_attention_module(self):
        """Test CuDNNMultiheadAttentionWithBias module."""
        from triton_kernels.cudnn_attention import CuDNNMultiheadAttentionWithBias
        
        embed_dim, num_heads = 256, 8
        seq_len, batch = 32, 4
        
        attn = CuDNNMultiheadAttentionWithBias(embed_dim, num_heads).cuda()
        
        query = torch.randn(seq_len, batch, embed_dim, device='cuda')
        key = torch.randn(seq_len, batch, embed_dim, device='cuda')
        value = torch.randn(seq_len, batch, embed_dim, device='cuda')
        attn_bias = torch.randn(batch, num_heads, seq_len, seq_len, device='cuda')
        
        output, _ = attn(query, key, value, attn_bias)
        
        assert output.shape == (seq_len, batch, embed_dim)
        assert torch.isfinite(output).all()
    
    def test_cudnn_attention_with_bias_function(self):
        """Test cudnn_attention_with_bias functional interface."""
        from triton_kernels.cudnn_attention import cudnn_attention_with_bias
        
        batch, num_heads, seq_len, head_dim = 2, 4, 16, 32
        
        q = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        k = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        v = torch.randn(batch, num_heads, seq_len, head_dim, device='cuda')
        bias = torch.randn(batch, num_heads, seq_len, seq_len, device='cuda')
        
        output = cudnn_attention_with_bias(q, k, v, bias)
        
        assert output.shape == q.shape
        assert torch.isfinite(output).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
