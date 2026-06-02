import torch
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger


class ParticleTransformerWrapperV2(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token'}

    def forward(self, points, features, lorentz_vectors, mask):
        mod = self.mod
        with torch.no_grad():
            x, v, mask, uu = mod.trimmer(features, lorentz_vectors, mask, None)
            padding_mask = ~mask.squeeze(1)  # (N, P), True=padded

        x = mod.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)

        attn_mask = None
        if v is not None and mod.pair_embed is not None:
            attn_mask = mod.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*H, P, P)

        P, N, C = x.shape
        cls_tokens = mod.cls_token.expand(1, N, -1)               # (1, N, C)
        x = torch.cat((cls_tokens, x), dim=0)                     # (P+1, N, C)

        cls_pad = padding_mask.new_zeros(N, 1)
        padding_mask = torch.cat((cls_pad, padding_mask), dim=1)  # (N, P+1)

        if attn_mask is not None:
            NH = attn_mask.size(0)
            zeros_col = attn_mask.new_zeros(NH, P, 1)
            zeros_row = attn_mask.new_zeros(NH, 1, P + 1)
            attn_mask = torch.cat((attn_mask, zeros_col), dim=2)  # (NH, P, P+1)
            attn_mask = torch.cat((zeros_row, attn_mask), dim=1)  # (NH, P+1, P+1)

        for block in mod.blocks:
            x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

        x_cls = mod.norm(x[0])   # (N, C)
        return mod.fc(x_cls)


def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[64],
        pair_embed_dims=[32, 32],
        num_heads=4,
        num_layers=4,
        num_cls_layers=0,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapperV2(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {
            **{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names},
            'softmax': {0: 'N'},
        },
    }
    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
