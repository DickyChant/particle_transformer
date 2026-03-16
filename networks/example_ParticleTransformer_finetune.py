import torch
import torch.nn as nn
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py
'''


class ParticleTransformerWrapper(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.use_residual_attn = kwargs.pop('use_residual_attn', False)

        in_dim = kwargs['embed_dims'][-1]
        fc_params = kwargs.pop('fc_params')
        num_classes = kwargs.pop('num_classes')
        self.for_inference = kwargs['for_inference']

        fcs = []
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, num_classes))
        self.fc = nn.Sequential(*fcs)

        kwargs['num_classes'] = None
        kwargs['fc_params'] = None
        self.mod = ParticleTransformer(**kwargs)
        if self.use_residual_attn:
            num_layers = len(self.mod.blocks)
            embed_dim = self.mod.blocks[0].embed_dim
            self.residual_attn_norm = nn.LayerNorm(embed_dim)
            self.residual_attn_weights = nn.Parameter(torch.zeros(num_layers, embed_dim), requires_grad=True)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        if self.use_residual_attn:
            x_cls = self._forward_with_residual_attn(features, v=lorentz_vectors, mask=mask)
        else:
            x_cls = self.mod(features, v=lorentz_vectors, mask=mask)
        output = self.fc(x_cls)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        return output

    def _apply_residual_attn(self, states, layer_idx):
        # s: state-depth index, p: particle/sequence position, n: batch, c: embedding channel
        values = torch.stack(states, dim=0)
        keys = self.residual_attn_norm(values)
        logits = torch.einsum('c,spnc->spn', self.residual_attn_weights[layer_idx], keys)
        weights = torch.softmax(logits, dim=0)
        return torch.einsum('spn,spnc->pnc', weights, values)

    def _forward_with_residual_attn(self, x, v=None, mask=None, uu=None, uu_idx=None):
        with torch.no_grad():
            if not self.mod.for_inference and uu_idx is not None:
                import importlib
                part_module = importlib.import_module(ParticleTransformer.__module__)
                uu = part_module.build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.mod.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)

        with torch.cuda.amp.autocast(enabled=self.mod.use_amp):
            x = self.mod.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)
            attn_mask = None
            if (v is not None or uu is not None) and self.mod.pair_embed is not None:
                attn_mask = self.mod.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))

            states = [x]
            for i, block in enumerate(self.mod.blocks):
                x_in = self._apply_residual_attn(states, i)
                x = block(x_in, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)
                states.append(x)

            cls_tokens = self.mod.cls_token.expand(1, x.size(1), -1)
            for block in self.mod.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            return self.mod.norm(cls_tokens).squeeze(0)


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        use_residual_attn=False,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
