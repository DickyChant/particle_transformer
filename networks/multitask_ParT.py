"""
Multi-task Particle Transformer: classification + regression with gradient analysis.

Shared ParT backbone with two heads:
  - cls_head: jet classification (10 classes, CE loss)
  - reg_head: jet pT regression (1 output, MSE loss)

Loss balancing via uncertainty weighting (Kendall et al., 2018).
Gradient cosine similarity monitoring between tasks.

Usage:
  weaver --network-config networks/multitask_ParT.py \
         --data-config data/JetClass/JetClass_multitask.yaml ...
"""

import time
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter, defaultdict

from weaver.nn.model.ParticleTransformer import ParticleTransformer
from weaver.utils.logger import _logger
from weaver.utils.data.tools import _concat


# ---- Model ----

class MultiTaskParTWrapper(nn.Module):
    def __init__(self, num_classes=10, embed_dim=128, **kwargs):
        super().__init__()
        # Backbone: ParT without built-in FC head
        kwargs.pop('fc_params', None)
        self.mod = ParticleTransformer(num_classes=None, fc_params=None, **kwargs)
        # Task-specific heads on CLS token embedding
        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.reg_head = nn.Linear(embed_dim, 1)
        # Learnable loss weights (part of model so optimizer picks them up)
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token'}

    def forward(self, points, features, lorentz_vectors, mask):
        x_cls = self.mod(features, v=lorentz_vectors, mask=mask)  # (batch, embed_dim)
        cls_logits = self.cls_head(x_cls)   # (batch, num_classes)
        reg_pred = self.reg_head(x_cls).squeeze(-1)  # (batch,)
        return cls_logits, reg_pred

    def backbone_parameters(self):
        """Parameters shared between tasks (everything except heads)."""
        head_params = set(id(p) for p in self.cls_head.parameters()) | \
                      set(id(p) for p in self.reg_head.parameters())
        return [p for p in self.parameters() if id(p) not in head_params]


# ---- Loss ----

class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task loss (Kendall et al., 2018).

    L = (1/2σ²_cls) * CE + log(σ_cls) + (1/2σ²_reg) * MSE + log(σ_reg)

    The learnable log_var parameters live on the model (MultiTaskParTWrapper),
    not here, so they are optimized by weaver's optimizer.
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, cls_logits, reg_pred, cls_label, reg_target, log_var_cls, log_var_reg):
        loss_cls = self.ce(cls_logits, cls_label)
        loss_reg = self.mse(reg_pred, reg_target)

        precision_cls = torch.exp(-log_var_cls)
        precision_reg = torch.exp(-log_var_reg)

        loss = precision_cls * loss_cls + log_var_cls + \
               precision_reg * loss_reg + log_var_reg

        return loss, loss_cls.detach(), loss_reg.detach()


# ---- Gradient cosine similarity ----

def compute_grad_cosine(model, cls_logits, reg_pred, cls_label, reg_target, loss_module):
    """Compute cosine similarity between task gradients on shared backbone."""
    model.zero_grad()
    ce_loss = loss_module.ce(cls_logits, cls_label)
    ce_loss.backward(retain_graph=True)

    backbone = model.module if hasattr(model, 'module') else model
    if hasattr(backbone, '_orig_mod'):
        backbone = backbone._orig_mod
    grad_cls = torch.cat([p.grad.flatten() for p in backbone.backbone_parameters()
                          if p.grad is not None])

    model.zero_grad()
    mse_loss = loss_module.mse(reg_pred, reg_target)
    mse_loss.backward(retain_graph=True)

    grad_reg = torch.cat([p.grad.flatten() for p in backbone.backbone_parameters()
                          if p.grad is not None])

    cos_sim = F.cosine_similarity(grad_cls.unsqueeze(0), grad_reg.unsqueeze(0)).item()
    model.zero_grad()
    return cos_sim


# ---- Custom train / evaluate ----

def _get_autocast_config(extra_args):
    args = extra_args.get('args', None) if extra_args else None
    if args is None:
        return False, None
    enable = getattr(args, 'use_amp', False)
    if getattr(args, 'amp_dtype', '') == 'bf16':
        return enable, torch.bfloat16
    return enable, None


def _unwrap(model):
    while hasattr(model, 'module'):
        model = model.module
    while hasattr(model, '_orig_mod'):
        model = model._orig_mod
    return model


def _make_train_fn(grad_cos_freq=50):

    def train_multitask(model, loss_func, opt, scheduler, train_loader, dev, epoch,
                        steps_per_epoch=None, grad_scaler=None, tb_helper=None, extra_args=None):
        model.train()
        data_config = train_loader.dataset.config
        clip_grad_norm = getattr(opt, '_clip_grad_norm', float('inf'))
        enable_autocast, autocast_dtype = _get_autocast_config(extra_args)

        raw_model = _unwrap(model)

        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        total_correct = 0
        num_batches = 0
        count = 0
        entry_count = 0
        cos_sims = []

        start_time = time.time()
        with tqdm.tqdm(train_loader) as tq:
            for X, y, _ in tq:
                inputs = [X[k].to(dev) for k in data_config.input_names]
                cls_label = y['cls_label'].long().to(dev)
                reg_target = y['reg_target'].float().to(dev)
                batch_size = cls_label.shape[0]
                entry_count += batch_size

                if tb_helper:
                    tb_helper.global_step += 1

                opt.zero_grad()
                with torch.autocast('cuda', enabled=enable_autocast, dtype=autocast_dtype):
                    cls_logits, reg_pred = model(*inputs)
                    loss, loss_cls, loss_reg = loss_func(cls_logits, reg_pred, cls_label, reg_target,
                                                        raw_model.log_var_cls, raw_model.log_var_reg)

                if grad_scaler is None:
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad_norm).item()
                    opt.step()
                else:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(opt)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_grad_norm).item()
                    grad_scaler.step(opt)
                    grad_scaler.update()

                if scheduler and getattr(scheduler, '_update_per_step', False):
                    scheduler.step()

                _, preds = cls_logits.max(1)
                correct = (preds == cls_label).sum().item()
                total_correct += correct
                count += batch_size

                loss_val = loss.item()
                total_loss += loss_val
                total_cls_loss += loss_cls.item()
                total_reg_loss += loss_reg.item()
                num_batches += 1

                # Gradient cosine similarity (periodic)
                if grad_cos_freq > 0 and num_batches % grad_cos_freq == 0:
                    with torch.autocast('cuda', enabled=enable_autocast, dtype=autocast_dtype):
                        cls_out, reg_out = model(*inputs)
                    cos_sim = compute_grad_cosine(
                        model, cls_out, reg_out, cls_label, reg_target, _unwrap_loss(loss_func))
                    cos_sims.append(cos_sim)
                    if tb_helper:
                        tb_helper.writer.add_scalar('Gradient/cos_sim', cos_sim, tb_helper.global_step)

                tq.set_postfix({
                    'Loss': '%.5f' % loss_val,
                    'CE': '%.5f' % loss_cls.item(),
                    'MSE': '%.5f' % loss_reg.item(),
                    'Acc': '%.5f' % (correct / batch_size),
                })

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

        if scheduler and not getattr(scheduler, '_update_per_step', False):
            scheduler.step()

        time_diff = time.time() - start_time
        avg_loss = total_loss / num_batches
        avg_cls = total_cls_loss / num_batches
        avg_reg = total_reg_loss / num_batches
        avg_acc = total_correct / count
        avg_cos = np.mean(cos_sims) if cos_sims else float('nan')

        # Get learned weights
        w_cls = torch.exp(-raw_model.log_var_cls).item()
        w_reg = torch.exp(-raw_model.log_var_reg).item()

        _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)',
                      entry_count, entry_count / time_diff)
        _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f', avg_loss, avg_acc)
        _logger.info('  CE: %.5f (w=%.3f), MSE: %.5f (w=%.3f), GradCos: %.4f',
                      avg_cls, w_cls, avg_reg, w_reg, avg_cos)

        if tb_helper:
            tb_helper.write_scalars([
                ("Loss/train (epoch)", avg_loss, epoch),
                ("Loss/train_cls (epoch)", avg_cls, epoch),
                ("Loss/train_reg (epoch)", avg_reg, epoch),
                ("Acc/train (epoch)", avg_acc, epoch),
                ("Weight/cls_precision", w_cls, epoch),
                ("Weight/reg_precision", w_reg, epoch),
                ("Gradient/cos_sim_epoch", avg_cos, epoch),
            ])
            if scheduler:
                tb_helper.write_scalars([
                    ("LR", scheduler.get_last_lr()[0], epoch),
                ])

    return train_multitask


def _unwrap_loss(loss_func):
    """Get the underlying MultiTaskLoss from potential DDP/compile wrappers."""
    if isinstance(loss_func, MultiTaskLoss):
        return loss_func
    if hasattr(loss_func, 'module'):
        return _unwrap_loss(loss_func.module)
    return loss_func


def _make_evaluate_fn():

    def evaluate_multitask(model, test_loader, dev, epoch, for_training=True,
                           loss_func=None, steps_per_epoch=None,
                           eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                           tb_helper=None, extra_args=None):
        model.eval()
        raw_model = _unwrap(model)
        data_config = test_loader.dataset.config

        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0
        total_correct = 0
        num_batches = 0
        count = 0
        entry_count = 0

        scores = []
        # Match weaver's standard contract: labels and observers are dicts
        # keyed by the YAML field names. We always accumulate `y` (so loss
        # computation has cls_label / reg_target); observers are only needed
        # in test/predict mode where we write a ROOT/parquet file.
        labels = defaultdict(list)
        observers = defaultdict(list)
        preds_reg = []  # also kept separately for sklearn regression metrics

        start_time = time.time()
        with torch.no_grad():
            with tqdm.tqdm(test_loader) as tq:
                for X, y, Z in tq:
                    inputs = [X[k].to(dev) for k in data_config.input_names]
                    cls_label = y['cls_label'].long().to(dev)
                    reg_target = y['reg_target'].float().to(dev)
                    batch_size = cls_label.shape[0]
                    entry_count += batch_size

                    cls_logits, reg_pred = model(*inputs)

                    if loss_func is not None:
                        loss, loss_cls, loss_reg = loss_func(cls_logits, reg_pred, cls_label, reg_target,
                                                        raw_model.log_var_cls, raw_model.log_var_reg)
                        total_loss += loss.item()
                        total_cls_loss += loss_cls.item()
                        total_reg_loss += loss_reg.item()

                    _, preds = cls_logits.max(1)
                    correct = (preds == cls_label).sum().item()
                    total_correct += correct
                    count += batch_size
                    num_batches += 1

                    scores.append(torch.softmax(cls_logits.float(), dim=1).cpu().numpy())
                    for k, v in y.items():
                        labels[k].append(v.cpu().numpy())
                    preds_reg.append(reg_pred.cpu().numpy())
                    if not for_training and Z is not None:
                        for k, v in Z.items():
                            observers[k].append(v)

                    if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                        break

        time_diff = time.time() - start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_cls = total_cls_loss / num_batches if num_batches > 0 else 0
        avg_reg = total_reg_loss / num_batches if num_batches > 0 else 0
        avg_acc = total_correct / count if count > 0 else 0

        scores = np.concatenate(scores)
        labels = {k: _concat(v) for k, v in labels.items()}
        preds_reg = np.concatenate(preds_reg)
        # Stash regression predictions alongside labels so they are written
        # to the predict-output ROOT/parquet file like any other column.
        labels['reg_pred'] = preds_reg
        observers = {k: _concat(v) for k, v in observers.items()}

        labels_cls = labels['cls_label']
        targets_reg = labels['reg_target']

        # Classification metrics
        from weaver.utils.nn.metrics import evaluate_metrics
        metric_results = evaluate_metrics(labels_cls, scores, eval_metrics=eval_metrics)

        # Regression metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        reg_mse = mean_squared_error(targets_reg, preds_reg)
        reg_mae = mean_absolute_error(targets_reg, preds_reg)

        _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)',
                      entry_count, entry_count / time_diff)
        _logger.info('Eval AvgLoss: %.5f, AvgAcc: %.5f', avg_loss, avg_acc)
        _logger.info('  CE: %.5f, MSE: %.5f', avg_cls, avg_reg)
        _logger.info('  Regression MSE: %.5f, MAE: %.5f', reg_mse, reg_mae)

        for k, v in metric_results.items():
            if 'matrix' not in k:
                _logger.info('    - %s: \n%s', k, str(v))
            else:
                _logger.info('    - %s: \n%s', k, str(v))

        # Use classification AUC as the main validation metric, fall back to accuracy
        val_metric = metric_results.get('roc_auc_score')
        if val_metric is None:
            val_metric = avg_acc

        if for_training:
            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/eval (epoch)", avg_loss, epoch),
                    ("Loss/eval_cls (epoch)", avg_cls, epoch),
                    ("Loss/eval_reg (epoch)", avg_reg, epoch),
                    ("Acc/eval (epoch)", avg_acc, epoch),
                    ("Regression/MSE (epoch)", reg_mse, epoch),
                    ("Regression/MAE (epoch)", reg_mae, epoch),
                ])
            return val_metric
        else:
            # Test mode: weaver expects (metric, scores, labels_dict, observers_dict).
            # save_root/save_parquet then call output.update(labels) and
            # output.update(observers); both must be plain dicts of arrays.
            return val_metric, scores, labels, observers

    return evaluate_multitask


# ---- Weaver interface ----

def get_model(data_config, **kwargs):
    # Determine num_classes from data config
    # With custom labels, label_value is a dict; we need the number of classes
    num_classes = kwargs.pop('num_classes', 10)
    embed_dims = kwargs.get('embed_dims', [128, 512, 128])
    embed_dim = embed_dims[-1] if embed_dims else 128

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=num_classes,
        embed_dim=embed_dim,
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=embed_dims,
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        activation='gelu',
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    # Pop our custom keys before passing to ParT
    num_classes = cfg.pop('num_classes')
    embed_dim = cfg.pop('embed_dim')
    cfg.pop('grad_cos_freq', None)
    _logger.info('MultiTask ParT config: num_classes=%d, embed_dim=%d, %s', num_classes, embed_dim, str(cfg))

    model = MultiTaskParTWrapper(num_classes=num_classes, embed_dim=embed_dim, **cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['cls_logits', 'reg_pred'],
        'dynamic_axes': {
            **{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names},
            'cls_logits': {0: 'N'}, 'reg_pred': {0: 'N'},
        },
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return MultiTaskLoss()


def get_train_fn(data_config, **kwargs):
    grad_cos_freq = kwargs.pop('grad_cos_freq', 50)
    return _make_train_fn(grad_cos_freq=grad_cos_freq)


def get_evaluate_fn(data_config, **kwargs):
    return _make_evaluate_fn()
