import importlib
import sys
import types
import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


def _install_fake_weaver():
    if 'weaver.nn.model.ParticleTransformer' in sys.modules:
        return

    logger_module = types.ModuleType('weaver.utils.logger')

    class _FakeLogger:
        def info(self, *_args, **_kwargs):
            return None

    logger_module._logger = _FakeLogger()

    part_module = types.ModuleType('weaver.nn.model.ParticleTransformer')

    class _FakeTrimmer:
        def __call__(self, x, v=None, mask=None, uu=None):
            if mask is None:
                mask = torch.ones((x.size(0), 1, x.size(2)), dtype=torch.bool, device=x.device)
            return x, v, mask, uu

    class _FakeBlock(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim

        def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
            del padding_mask, attn_mask
            if x_cls is not None:
                return x_cls + x.mean(dim=0, keepdim=True)
            return x + 1

    class ParticleTransformer(nn.Module):
        def __init__(self, input_dim, num_classes=None, embed_dims=None, num_layers=2, num_cls_layers=1,
                     for_inference=False, **kwargs):
            super().__init__()
            del kwargs
            embed_dims = [input_dim] if not embed_dims else embed_dims
            embed_dim = embed_dims[-1]
            self.trimmer = _FakeTrimmer()
            self.for_inference = for_inference
            self.use_amp = False
            self.embed = lambda x: x.permute(2, 0, 1).contiguous()
            self.pair_embed = None
            self.blocks = nn.ModuleList([_FakeBlock(embed_dim) for _ in range(num_layers)])
            self.cls_blocks = nn.ModuleList([_FakeBlock(embed_dim) for _ in range(num_cls_layers)])
            self.norm = nn.Identity()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
            self.fc = None if num_classes is None else nn.Linear(embed_dim, num_classes)

        def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
            del v, uu_idx
            x, _, mask, uu = self.trimmer(x, None, mask, uu)
            del uu
            padding_mask = ~mask.squeeze(1)
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=None)
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)
            x_cls = self.norm(cls_tokens).squeeze(0)
            if self.fc is None:
                return x_cls
            return self.fc(x_cls)

    def build_sparse_tensor(uu, idx, seq_len):
        del idx, seq_len
        return uu

    part_module.ParticleTransformer = ParticleTransformer
    part_module.build_sparse_tensor = build_sparse_tensor

    sys.modules['weaver'] = types.ModuleType('weaver')
    sys.modules['weaver.nn'] = types.ModuleType('weaver.nn')
    sys.modules['weaver.nn.model'] = types.ModuleType('weaver.nn.model')
    sys.modules['weaver.nn.model.ParticleTransformer'] = part_module
    sys.modules['weaver.utils'] = types.ModuleType('weaver.utils')
    sys.modules['weaver.utils.logger'] = logger_module


class _DataConfig:
    def __init__(self):
        self.input_dicts = {'pf_features': ['a', 'b', 'c', 'd']}
        self.label_value = [0, 1, 2]
        self.input_names = ['points', 'features', 'lorentz_vectors', 'mask']
        self.input_shapes = {
            'points': (1, 2, 4),
            'features': (1, 4, 4),
            'lorentz_vectors': (1, 4, 4),
            'mask': (1, 1, 4),
        }


@unittest.skipUnless(torch is not None, "torch is required")
class ResidualAttentionWrapperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_fake_weaver()

    def test_particle_transformer_wrapper_residual_attn(self):
        module = importlib.import_module('networks.example_ParticleTransformer')
        data_config = _DataConfig()
        model, _ = module.get_model(data_config, use_residual_attn=True, num_layers=2)

        self.assertTrue(model.use_residual_attn)
        self.assertEqual(tuple(model.residual_attn_weights.shape), (2, 128))

        n, p = 3, 4
        points = torch.zeros((n, 2, p))
        features = torch.ones((n, 4, p))
        lorentz_vectors = torch.zeros((n, 4, p))
        mask = torch.ones((n, 1, p), dtype=torch.bool)
        out = model(points, features, lorentz_vectors, mask)
        self.assertEqual(tuple(out.shape), (n, 3))

    def test_particle_transformer_finetune_wrapper_residual_attn(self):
        module = importlib.import_module('networks.example_ParticleTransformer_finetune')
        data_config = _DataConfig()
        model, _ = module.get_model(
            data_config,
            use_residual_attn=True,
            num_layers=2,
            fc_params=[(16, 0.0)],
        )

        self.assertTrue(model.use_residual_attn)
        self.assertEqual(tuple(model.residual_attn_weights.shape), (2, 128))

        n, p = 2, 4
        points = torch.zeros((n, 2, p))
        features = torch.ones((n, 4, p))
        lorentz_vectors = torch.zeros((n, 4, p))
        mask = torch.ones((n, 1, p), dtype=torch.bool)
        out = model(points, features, lorentz_vectors, mask)
        self.assertEqual(tuple(out.shape), (n, 3))


if __name__ == '__main__':
    unittest.main()
