import os
import torch
import torch.nn as nn
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

ParticleTransformer = import_module(
    os.path.join(os.path.dirname(__file__), 'ParticleTransformer.py'), 'ParT').ParticleTransformer


class ParticleTransformerWrapper(torch.nn.Module):
    """
    ParticleTransformer with augmented class for softmax classification.
    Adds an extra "background/null" class to the final softmax output.
    No attention mask (pair_embed disabled) + augmented attention.
    """
    def __init__(self, num_classes, add_augmented_class=True, **kwargs) -> None:
        super().__init__()
        # Store original num_classes
        self.num_classes = num_classes
        self.add_augmented_class = add_augmented_class
        
        # Create model with num_classes + 1 if adding augmented class
        model_num_classes = num_classes + 1 if add_augmented_class else num_classes
        kwargs['num_classes'] = None  # Get features, not logits
        self.mod = ParticleTransformer(**kwargs)
        
        # Create final classification layer with augmented class
        embed_dim = kwargs.get('embed_dims', [128, 512, 128])[-1]
        fc_params = kwargs.get('fc_params', [])
        
        fcs = []
        in_dim = embed_dim
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, model_num_classes))
        self.fc = nn.Sequential(*fcs)
        
        self.for_inference = kwargs.get('for_inference', False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        # Get features from transformer
        x_cls = self.mod(features, v=lorentz_vectors, mask=mask)  # (N, embed_dim)
        
        # Get logits (with augmented class)
        logits = self.fc(x_cls)  # (N, num_classes + 1)
        
        if self.for_inference:
            # Use softmax for classification
            probs = torch.softmax(logits, dim=1)  # (N, num_classes + 1)
            # Remove augmented class and renormalize
            if self.add_augmented_class:
                probs = probs[:, :-1]  # Remove last (augmented) class
                probs = probs / probs.sum(dim=1, keepdim=True)  # Renormalize
            output = probs
        else:
            # For training, return full logits (augmented class included for loss)
            output = logits
            
        return output


def get_model(data_config, **kwargs):

    num_classes = len(data_config.label_value)
    
    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=num_classes,  # Will be passed to wrapper
        # network configurations
        pair_input_dim=4,
        embed_dims=[128, 512, 128],
        pair_embed_dims=None,  # Disable pair embedding (no attention mask)
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # Augmented class options
        add_augmented_class=True,  # Add extra "background" class to softmax
        use_augmented_attention=True,  # Use augmented class in attention softmax (default: True)
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    
    # Handle boolean flags from command line (they might come as strings)
    if 'use_augmented_attention' in kwargs:
        use_augmented_attention_val = kwargs.get('use_augmented_attention')
        if isinstance(use_augmented_attention_val, str):
            use_augmented_attention = use_augmented_attention_val.lower() in ('true', '1', 'yes', 'on')
        else:
            use_augmented_attention = bool(use_augmented_attention_val)
        cfg['use_augmented_attention'] = use_augmented_attention
    else:
        use_augmented_attention = cfg.get('use_augmented_attention', True)
    
    if 'add_augmented_class' in kwargs:
        add_augmented_class_val = kwargs.get('add_augmented_class')
        if isinstance(add_augmented_class_val, str):
            add_augmented_class = add_augmented_class_val.lower() in ('true', '1', 'yes', 'on')
        else:
            add_augmented_class = bool(add_augmented_class_val)
        cfg['add_augmented_class'] = add_augmented_class
    else:
        add_augmented_class = cfg.get('add_augmented_class', True)
    
    _logger.info('Model config (no mask + augmented): %s' % str(cfg))
    
    if use_augmented_attention:
        # Import the augmented block variant
        from weaver.utils.import_tools import import_module
        ParticleTransformerAugmented = import_module(
            os.path.join(os.path.dirname(__file__), 'ParticleTransformer.py'), 'ParT').ParticleTransformerAugmented
        # Create model with augmented attention
        cfg_pt = cfg.copy()
        cfg_pt.pop('add_augmented_class', None)
        cfg_pt.pop('use_augmented_attention', None)
        cfg_pt['num_classes'] = None
        mod = ParticleTransformerAugmented(**cfg_pt)
        
        # Create wrapper with augmented class in final softmax
        embed_dim = cfg.get('embed_dims', [128, 512, 128])[-1]
        fc_params = cfg.get('fc_params', [])
        model_num_classes = num_classes + 1 if cfg.get('add_augmented_class', True) else num_classes
        
        fcs = []
        in_dim = embed_dim
        for out_dim, drop_rate in fc_params:
            fcs.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
            in_dim = out_dim
        fcs.append(nn.Linear(in_dim, model_num_classes))
        fc = nn.Sequential(*fcs)
        
        class AugmentedWrapper(torch.nn.Module):
            def __init__(self, mod, fc, for_inference, add_augmented_class):
                super().__init__()
                self.mod = mod
                self.fc = fc
                self.for_inference = for_inference
                self.add_augmented_class = add_augmented_class
            @torch.jit.ignore
            def no_weight_decay(self):
                return {'mod.cls_token', }
            def forward(self, points, features, lorentz_vectors, mask):
                x_cls = self.mod(features, v=lorentz_vectors, mask=mask)
                logits = self.fc(x_cls)  # (N, num_classes + 1)
                if self.for_inference:
                    probs = torch.softmax(logits, dim=1)  # (N, num_classes + 1)
                    # Remove augmented class and renormalize
                    if self.add_augmented_class:
                        probs = probs[:, :-1]  # Remove last (augmented) class
                        probs = probs / probs.sum(dim=1, keepdim=True)  # Renormalize
                    return probs
                return logits
        
        model = AugmentedWrapper(mod, fc, cfg.get('for_inference', False), add_augmented_class)
    else:
        model = ParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    """
    Loss function for softmax classification with augmented class.
    Uses CrossEntropyLoss - the augmented class is treated as an additional class.
    """
    return torch.nn.CrossEntropyLoss()

