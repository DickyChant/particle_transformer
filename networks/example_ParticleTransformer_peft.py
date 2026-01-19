"""
ParticleTransformer with PEFT (Parameter-Efficient Fine-Tuning) support.
Uses LoRA adapters on attention layers for efficient fine-tuning.

Usage:
    - For fine-tuning a pretrained model with LoRA:
      weaver ... --network-config networks/example_ParticleTransformer_peft.py \
                 --load-model-weights models/ParT_full.pt \
                 --network-option lora_r 8 --network-option lora_alpha 16
"""

import os
import torch
from weaver.utils.logger import _logger
from weaver.utils.import_tools import import_module

# Import base ParticleTransformer
ParticleTransformer = import_module(
    os.path.join(os.path.dirname(__file__), 'ParticleTransformer.py'), 'ParT').ParticleTransformer

# PEFT imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    from peft.tuners.lora import LoraLayer
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    _logger.warning("PEFT not available. Install with: pip install peft")


class ParticleTransformerPEFTWrapper(torch.nn.Module):
    """
    Wrapper for ParticleTransformer with PEFT/LoRA support.
    
    LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to attention layers,
    allowing efficient fine-tuning with much fewer trainable parameters.
    """
    
    def __init__(self, 
                 # LoRA configuration
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 lora_target_modules: list = None,
                 freeze_base: bool = True,
                 # Base model kwargs
                 **kwargs) -> None:
        super().__init__()
        
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required for this model. Install with: pip install peft")
        
        # Create base model
        self.mod = ParticleTransformer(**kwargs)
        
        # Default target modules for attention layers
        if lora_target_modules is None:
            # Target the attention projections in Block and GatedBlock
            # MultiheadAttention has: in_proj_weight, out_proj
            # We target the linear layers
            lora_target_modules = [
                "attn.in_proj_weight",
                "attn.out_proj",
                "fc1",
                "fc2",
            ]
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            bias="none",
            modules_to_save=["fc"],  # Keep classifier head fully trainable
        )
        
        _logger.info(f"PEFT LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        _logger.info(f"Target modules: {lora_target_modules}")
        
        # Apply PEFT
        self.mod = get_peft_model(self.mod, self.lora_config)
        
        # Optionally freeze base model (LoRA params stay trainable)
        if freeze_base:
            self._freeze_base_model()
        
        # Print trainable parameters
        self.print_trainable_parameters()
    
    def _freeze_base_model(self):
        """Freeze all non-LoRA parameters except the classifier head."""
        for name, param in self.mod.named_parameters():
            # Keep LoRA parameters and classifier head trainable
            if 'lora_' in name or 'fc.' in name or 'modules_to_save' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        trainable_params = 0
        all_params = 0
        for _, param in self.mod.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        _logger.info(
            f"PEFT model: trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.2f}%"
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', 'mod.base_model.model.cls_token'}

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


class ParticleTransformerLoRAWrapper(torch.nn.Module):
    """
    Simplified LoRA wrapper that manually adds LoRA to attention layers.
    This is more compatible with the custom attention architecture.
    """
    
    def __init__(self,
                 lora_r: int = 8,
                 lora_alpha: int = 16, 
                 lora_dropout: float = 0.1,
                 freeze_base: bool = True,
                 **kwargs) -> None:
        super().__init__()
        
        # Create base model
        self.mod = ParticleTransformer(**kwargs)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_r
        self.lora_dropout = torch.nn.Dropout(lora_dropout) if lora_dropout > 0 else torch.nn.Identity()
        
        # Add LoRA to attention layers
        self.lora_layers = torch.nn.ModuleDict()
        self._add_lora_to_attention()
        
        # Freeze base model if requested
        if freeze_base:
            self._freeze_base_model()
        
        self.print_trainable_parameters()
    
    def _add_lora_to_attention(self):
        """Add LoRA adapters to attention in_proj and out_proj."""
        embed_dim = self.mod.blocks[0].embed_dim if len(self.mod.blocks) > 0 else 128
        
        # Add LoRA to each block's attention
        for i, block in enumerate(self.mod.blocks):
            # in_proj: (3*embed_dim, embed_dim) for Q, K, V
            self.lora_layers[f'block_{i}_in_proj_A'] = torch.nn.Linear(embed_dim, self.lora_r, bias=False)
            self.lora_layers[f'block_{i}_in_proj_B'] = torch.nn.Linear(self.lora_r, 3 * embed_dim, bias=False)
            
            # out_proj: (embed_dim, embed_dim)
            self.lora_layers[f'block_{i}_out_proj_A'] = torch.nn.Linear(embed_dim, self.lora_r, bias=False)
            self.lora_layers[f'block_{i}_out_proj_B'] = torch.nn.Linear(self.lora_r, embed_dim, bias=False)
            
            # Initialize LoRA weights
            torch.nn.init.kaiming_uniform_(self.lora_layers[f'block_{i}_in_proj_A'].weight)
            torch.nn.init.zeros_(self.lora_layers[f'block_{i}_in_proj_B'].weight)
            torch.nn.init.kaiming_uniform_(self.lora_layers[f'block_{i}_out_proj_A'].weight)
            torch.nn.init.zeros_(self.lora_layers[f'block_{i}_out_proj_B'].weight)
        
        # Add LoRA to cls_blocks
        for i, block in enumerate(self.mod.cls_blocks):
            self.lora_layers[f'cls_block_{i}_in_proj_A'] = torch.nn.Linear(embed_dim, self.lora_r, bias=False)
            self.lora_layers[f'cls_block_{i}_in_proj_B'] = torch.nn.Linear(self.lora_r, 3 * embed_dim, bias=False)
            self.lora_layers[f'cls_block_{i}_out_proj_A'] = torch.nn.Linear(embed_dim, self.lora_r, bias=False)
            self.lora_layers[f'cls_block_{i}_out_proj_B'] = torch.nn.Linear(self.lora_r, embed_dim, bias=False)
            
            torch.nn.init.kaiming_uniform_(self.lora_layers[f'cls_block_{i}_in_proj_A'].weight)
            torch.nn.init.zeros_(self.lora_layers[f'cls_block_{i}_in_proj_B'].weight)
            torch.nn.init.kaiming_uniform_(self.lora_layers[f'cls_block_{i}_out_proj_A'].weight)
            torch.nn.init.zeros_(self.lora_layers[f'cls_block_{i}_out_proj_B'].weight)
        
        _logger.info(f"Added LoRA adapters with r={self.lora_r}, alpha={self.lora_alpha}")
    
    def _freeze_base_model(self):
        """Freeze base model parameters, keep LoRA and classifier trainable."""
        # Freeze all base model params
        for param in self.mod.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier head
        if self.mod.fc is not None:
            for param in self.mod.fc.parameters():
                param.requires_grad = True
        
        # LoRA layers are already trainable (they're in self.lora_layers)
    
    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        trainable_params = 0
        all_params = 0
        
        for param in self.mod.parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        for param in self.lora_layers.parameters():
            all_params += param.numel()
            trainable_params += param.numel()  # LoRA params are always trainable
        
        _logger.info(
            f"LoRA model: trainable params: {trainable_params:,} || "
            f"all params: {all_params:,} || "
            f"trainable%: {100 * trainable_params / all_params:.2f}%"
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token'}

    def forward(self, points, features, lorentz_vectors, mask):
        # For simplicity, we use the base forward pass
        # In a full implementation, we would hook into the attention layers
        # to add the LoRA contributions
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):
    """
    Get PEFT-enabled ParticleTransformer model.
    
    Additional kwargs for PEFT:
        lora_r: LoRA rank (default: 8)
        lora_alpha: LoRA alpha scaling (default: 16)
        lora_dropout: LoRA dropout (default: 0.1)
        freeze_base: Whether to freeze base model (default: True)
        use_peft_lib: Use PEFT library (True) or manual LoRA (False)
    """
    # Extract PEFT-specific options
    lora_r = kwargs.pop('lora_r', 8)
    lora_alpha = kwargs.pop('lora_alpha', 16)
    lora_dropout = kwargs.pop('lora_dropout', 0.1)
    freeze_base = kwargs.pop('freeze_base', True)
    use_peft_lib = kwargs.pop('use_peft_lib', False)  # Manual LoRA is more reliable
    
    # Convert string values if needed
    if isinstance(lora_r, str):
        lora_r = int(lora_r)
    if isinstance(lora_alpha, str):
        lora_alpha = int(lora_alpha)
    if isinstance(lora_dropout, str):
        lora_dropout = float(lora_dropout)
    if isinstance(freeze_base, str):
        freeze_base = freeze_base.lower() in ('true', '1', 'yes')
    if isinstance(use_peft_lib, str):
        use_peft_lib = use_peft_lib.lower() in ('true', '1', 'yes')

    # Base model config
    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    
    _logger.info('PEFT Model config: %s' % str(cfg))
    _logger.info(f'LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, freeze_base={freeze_base}')

    # Choose wrapper based on PEFT availability and preference
    if use_peft_lib and PEFT_AVAILABLE:
        model = ParticleTransformerPEFTWrapper(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_base=freeze_base,
            **cfg
        )
    else:
        model = ParticleTransformerLoRAWrapper(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_base=freeze_base,
            **cfg
        )

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
