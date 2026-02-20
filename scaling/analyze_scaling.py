#!/usr/bin/env python
"""
Chinchilla Scaling Law Analysis for Particle Transformer on JetClass.

Parses training results from TensorBoard logs and text logs,
fits the scaling law L(N, D) = A * N^(-alpha) + B * D^(-beta) + E,
and produces diagnostic plots.

Usage:
    python analyze_scaling.py [--results-dir DIR] [--output-dir DIR] [--count-params]
"""

import os
import sys
import re
import json
import argparse
import glob
import numpy as np

# ---- Configuration ----

MODEL_SIZES = ['nano', 'micro', 'tiny', 'small', 'base', 'large', 'xlarge']
DATA_BUDGETS = ['5M', '10M', '25M', '50M', '100M', '250M', '500M']

# Model configs for parameter counting
MODEL_CONFIGS = {
    'nano':   dict(embed_dims=[32, 128, 32],     pair_embed_dims=[16, 16, 16],     num_heads=4,  num_layers=2,  num_cls_layers=1),
    'micro':  dict(embed_dims=[48, 192, 48],     pair_embed_dims=[24, 24, 24],     num_heads=4,  num_layers=3,  num_cls_layers=1),
    'tiny':   dict(embed_dims=[64, 256, 64],     pair_embed_dims=[32, 32, 32],     num_heads=4,  num_layers=4,  num_cls_layers=1),
    'small':  dict(embed_dims=[96, 384, 96],     pair_embed_dims=[48, 48, 48],     num_heads=8,  num_layers=6,  num_cls_layers=2),
    'base':   dict(embed_dims=[128, 512, 128],   pair_embed_dims=[64, 64, 64],     num_heads=8,  num_layers=8,  num_cls_layers=2),
    'large':  dict(embed_dims=[192, 768, 192],   pair_embed_dims=[96, 96, 96],     num_heads=8,  num_layers=10, num_cls_layers=2),
    'xlarge': dict(embed_dims=[256, 1024, 256],  pair_embed_dims=[128, 128, 128],  num_heads=16, num_layers=12, num_cls_layers=2),
}

# Data budget -> total training samples
BUDGET_TO_SAMPLES = {
    '5M':   5_120_000,
    '10M':  10_240_000,
    '25M':  25_600_000,
    '50M':  51_200_000,
    '100M': 102_400_000,
    '250M': 256_000_000,
    '500M': 512_000_000,
}

DEFAULT_RESULTS_DIR = '/pscratch/sd/s/sqian/part_training_output/scaling_study'


def count_model_params(model_size):
    """Instantiate a model and count parameters."""
    try:
        import torch
        from weaver.nn.model.ParticleTransformer import ParticleTransformer

        cfg = MODEL_CONFIGS[model_size]
        model = ParticleTransformer(
            input_dim=18,
            num_classes=10,
            pair_input_dim=4,
            use_pre_activation_pair=False,
            embed_dims=cfg['embed_dims'],
            pair_embed_dims=cfg['pair_embed_dims'],
            num_heads=cfg['num_heads'],
            num_layers=cfg['num_layers'],
            num_cls_layers=cfg['num_cls_layers'],
            block_params=None,
            cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
            fc_params=[],
            activation='gelu',
            trim=True,
            for_inference=False,
        )
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return n_params, n_trainable
    except ImportError:
        print("Warning: Cannot import weaver. Using analytical estimates.")
        return estimate_params_analytical(model_size), None


def estimate_params_analytical(model_size):
    """Rough analytical parameter count estimate."""
    cfg = MODEL_CONFIGS[model_size]
    d = cfg['embed_dims'][-1]  # embed_dim
    n_layers = cfg['num_layers']
    n_cls_layers = cfg['num_cls_layers']
    n_heads = cfg['num_heads']

    # Embedding: roughly input_dim*d_0 + d_0*d_1 + d_1*d_2 + LayerNorms
    dims = [18] + cfg['embed_dims']
    embed_params = sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1))

    # Pair embedding: Conv1d layers
    pdims = [4] + cfg['pair_embed_dims'] + [n_heads]
    pair_params = sum(pdims[i] * pdims[i + 1] + pdims[i + 1] for i in range(len(pdims) - 1))

    # Each transformer block:
    # MHA: 4 * d^2 + 4*d (Q, K, V, O projections + biases)
    # FFN: d * 4d + 4d + 4d * d + d
    # LayerNorms: 4 * 2d (pre_attn, post_attn, pre_fc, post_fc)
    # scale params: n_heads + d
    block_params = 4 * d * d + 4 * d + d * 4 * d + 4 * d + 4 * d * d + d + 4 * 2 * d + n_heads + d

    # Classification head: d -> 10
    cls_params = d * 10 + 10

    # cls_token
    cls_token_params = d

    total = embed_params + pair_params + (n_layers + n_cls_layers) * block_params + cls_params + cls_token_params
    return total


def parse_tensorboard_logs(tb_dir):
    """
    Parse TensorBoard event files to extract per-epoch loss and accuracy.

    Returns dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc',
    each mapping to list of (step/epoch, value) tuples.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("Warning: tensorboard not installed. Cannot parse TB logs.")
        return None

    if not os.path.isdir(tb_dir):
        return None

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    results = {}
    tag_mapping = {
        'Loss/train (epoch)': 'train_loss',
        'Loss/eval (epoch)': 'val_loss',
        'Acc/train (epoch)': 'train_acc',
        'Acc/eval (epoch)': 'val_acc',
    }

    available_tags = ea.Tags().get('scalars', [])
    for tb_tag, result_key in tag_mapping.items():
        if tb_tag in available_tags:
            events = ea.Scalars(tb_tag)
            results[result_key] = [(e.step, e.value) for e in events]

    return results if results else None


def parse_text_logs(log_dir, run_name):
    """
    Parse text log files for validation accuracy.
    Falls back to this when TensorBoard logs are unavailable.

    Returns dict with 'val_acc' -> list of (epoch, accuracy) tuples.
    """
    # Find log files matching the run name
    pattern = os.path.join(log_dir, f'{run_name}*.log')
    log_files = glob.glob(pattern)
    # Also try without timestamp suffix
    pattern2 = os.path.join(log_dir, f'{run_name}_*.log')
    log_files.extend(glob.glob(pattern2))
    log_files = sorted(set(log_files))

    if not log_files:
        return None

    results = {'val_acc': [], 'train_loss': [], 'train_acc': []}
    epoch_pattern = re.compile(r'Epoch #(\d+): Current validation metric: ([\d.]+)')
    train_pattern = re.compile(r'Train AvgLoss: ([\d.]+), AvgAcc: ([\d.]+)')

    for log_file in log_files:
        # Only parse the rank-0 log (no .001, .002 suffix, or .000 suffix)
        base = os.path.basename(log_file)
        if re.search(r'\.\d{3}$', base):
            rank = int(base.rsplit('.', 1)[1])
            if rank != 0:
                continue

        with open(log_file, 'r') as f:
            current_epoch = -1
            for line in f:
                m = epoch_pattern.search(line)
                if m:
                    epoch = int(m.group(1))
                    acc = float(m.group(2))
                    results['val_acc'].append((epoch, acc))

                m = train_pattern.search(line)
                if m:
                    loss = float(m.group(1))
                    acc = float(m.group(2))
                    current_epoch += 1
                    results['train_loss'].append((current_epoch, loss))
                    results['train_acc'].append((current_epoch, acc))

    return results if any(v for v in results.values()) else None


def collect_results(results_dir):
    """
    Collect all scaling study results.

    Returns a list of dicts:
    [{'model_size': str, 'data_budget': str, 'n_params': int,
      'total_samples': int, 'best_val_loss': float, 'best_val_acc': float,
      'train_history': {...}, ...}, ...]
    """
    tb_dir = os.path.join(results_dir, 'tensorboard')
    log_dir = os.path.join(results_dir, 'logs')

    results = []
    for model_size in MODEL_SIZES:
        n_params, _ = count_model_params(model_size)
        for budget in DATA_BUDGETS:
            run_name = f'{model_size}_{budget}'
            total_samples = BUDGET_TO_SAMPLES[budget]

            entry = {
                'model_size': model_size,
                'data_budget': budget,
                'n_params': n_params,
                'total_samples': total_samples,
                'best_val_loss': None,
                'best_val_acc': None,
                'val_loss_history': [],
                'val_acc_history': [],
            }

            # Try TensorBoard first
            tb_run_dir = os.path.join(tb_dir, run_name)
            tb_data = parse_tensorboard_logs(tb_run_dir)
            if tb_data:
                if 'val_loss' in tb_data and tb_data['val_loss']:
                    entry['val_loss_history'] = tb_data['val_loss']
                    entry['best_val_loss'] = min(v for _, v in tb_data['val_loss'])
                if 'val_acc' in tb_data and tb_data['val_acc']:
                    entry['val_acc_history'] = tb_data['val_acc']
                    entry['best_val_acc'] = max(v for _, v in tb_data['val_acc'])

            # Fallback to text logs for accuracy
            if entry['best_val_acc'] is None:
                text_data = parse_text_logs(log_dir, run_name)
                if text_data and text_data.get('val_acc'):
                    entry['val_acc_history'] = text_data['val_acc']
                    entry['best_val_acc'] = max(v for _, v in text_data['val_acc'])

            # Only include runs that have results
            if entry['best_val_loss'] is not None or entry['best_val_acc'] is not None:
                results.append(entry)
                print(f"  {run_name}: params={n_params:,}, "
                      f"loss={entry['best_val_loss']}, acc={entry['best_val_acc']}")
            else:
                print(f"  {run_name}: NO RESULTS FOUND")

    return results


def fit_scaling_law(results, metric='loss'):
    """
    Fit L(N, D) = A * N^(-alpha) + B * D^(-beta) + E

    Args:
        results: list of result dicts from collect_results()
        metric: 'loss' or 'acc'

    Returns:
        dict with fitted parameters and fit quality metrics
    """
    from scipy.optimize import curve_fit

    # Filter to runs with valid metric
    if metric == 'loss':
        valid = [r for r in results if r['best_val_loss'] is not None]
        N = np.array([r['n_params'] for r in valid], dtype=np.float64)
        D = np.array([r['total_samples'] for r in valid], dtype=np.float64)
        L = np.array([r['best_val_loss'] for r in valid], dtype=np.float64)
    else:
        valid = [r for r in results if r['best_val_acc'] is not None]
        N = np.array([r['n_params'] for r in valid], dtype=np.float64)
        D = np.array([r['total_samples'] for r in valid], dtype=np.float64)
        # For accuracy, fit 1 - acc (error rate) which should decrease
        L = 1.0 - np.array([r['best_val_acc'] for r in valid], dtype=np.float64)

    if len(valid) < 5:
        print(f"Warning: Only {len(valid)} data points for '{metric}' fit. Need at least 5.")
        return None

    def scaling_law(X, A, alpha, B, beta, E):
        N, D = X
        return A * np.power(N, -alpha) + B * np.power(D, -beta) + E

    # Initial guess
    p0 = [1.0, 0.5, 1.0, 0.5, np.min(L)]

    try:
        popt, pcov = curve_fit(
            scaling_law, (N, D), L, p0=p0,
            bounds=([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, np.max(L)]),
            maxfev=50000,
        )
        perr = np.sqrt(np.diag(pcov))

        L_pred = scaling_law((N, D), *popt)
        ss_res = np.sum((L - L_pred) ** 2)
        ss_tot = np.sum((L - np.mean(L)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        fit_result = {
            'A': popt[0], 'alpha': popt[1],
            'B': popt[2], 'beta': popt[3],
            'E': popt[4],
            'A_err': perr[0], 'alpha_err': perr[1],
            'B_err': perr[2], 'beta_err': perr[3],
            'E_err': perr[4],
            'r_squared': r_squared,
            'n_points': len(valid),
            'metric': metric,
        }

        print(f"\nScaling Law Fit ({metric}):")
        print(f"  L(N,D) = {popt[0]:.4f} * N^(-{popt[1]:.4f}) + {popt[2]:.4f} * D^(-{popt[3]:.4f}) + {popt[4]:.6f}")
        print(f"  R^2 = {r_squared:.6f}")
        print(f"  alpha = {popt[1]:.4f} +/- {perr[1]:.4f}")
        print(f"  beta  = {popt[3]:.4f} +/- {perr[3]:.4f}")

        return fit_result

    except Exception as e:
        print(f"Fitting failed for {metric}: {e}")
        return None


def compute_optimal_allocation(fit_result, compute_budgets=None):
    """
    For a given compute budget C (in units of N * D),
    find the optimal N and D that minimize L(N, D).

    Returns arrays of (C, N_opt, D_opt, L_opt).
    """
    if fit_result is None:
        return None

    A = fit_result['A']
    alpha = fit_result['alpha']
    B = fit_result['B']
    beta = fit_result['beta']

    # Optimal ratio: dL/dN = dL/dD with constraint N*D = C
    # -A*alpha*N^(-alpha-1) = -B*beta*D^(-beta-1) and N*D = C
    # => N_opt = (A*alpha / (B*beta))^(1/(alpha+beta)) * C^(beta/(alpha+beta))
    # => D_opt = C / N_opt

    if compute_budgets is None:
        compute_budgets = np.logspace(10, 18, 50)

    ratio_coeff = (A * alpha / (B * beta)) ** (1.0 / (alpha + beta))
    n_exp = beta / (alpha + beta)

    N_opt = ratio_coeff * np.power(compute_budgets, n_exp)
    D_opt = compute_budgets / N_opt

    def scaling_law(N, D):
        return A * np.power(N, -alpha) + B * np.power(D, -beta) + fit_result['E']

    L_opt = scaling_law(N_opt, D_opt)

    return {
        'compute': compute_budgets,
        'N_opt': N_opt,
        'D_opt': D_opt,
        'L_opt': L_opt,
        'N_exponent': n_exp,
        'D_exponent': 1 - n_exp,
    }


def plot_results(results, fit_result, optimal, output_dir):
    """Generate all scaling law plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    os.makedirs(output_dir, exist_ok=True)

    # Color scheme for model sizes
    colors = {
        'nano': '#1f77b4', 'micro': '#ff7f0e', 'tiny': '#2ca02c',
        'small': '#d62728', 'base': '#9467bd', 'large': '#8c564b',
        'xlarge': '#e377c2',
    }

    # ---- Plot 1: Val Loss vs Data Budget (one curve per model size) ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for model_size in MODEL_SIZES:
        runs = [r for r in results if r['model_size'] == model_size and r['best_val_loss'] is not None]
        if not runs:
            continue
        runs.sort(key=lambda r: r['total_samples'])
        D = [r['total_samples'] for r in runs]
        L = [r['best_val_loss'] for r in runs]
        n_params = runs[0]['n_params']
        ax.plot(D, L, 'o-', color=colors[model_size],
                label=f'{model_size} ({n_params/1e6:.2f}M)', markersize=6)

    ax.set_xscale('log')
    ax.set_xlabel('Total Training Samples (D)', fontsize=14)
    ax.set_ylabel('Best Validation Loss', fontsize=14)
    ax.set_title('Scaling: Loss vs Data Budget', fontsize=16)
    ax.legend(title='Model Size', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_vs_data.pdf'), dpi=150)
    fig.savefig(os.path.join(output_dir, 'loss_vs_data.png'), dpi=150)
    plt.close(fig)

    # ---- Plot 2: Val Loss vs Model Params (one curve per data budget) ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    budget_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DATA_BUDGETS)))
    for i, budget in enumerate(DATA_BUDGETS):
        runs = [r for r in results if r['data_budget'] == budget and r['best_val_loss'] is not None]
        if not runs:
            continue
        runs.sort(key=lambda r: r['n_params'])
        N = [r['n_params'] for r in runs]
        L = [r['best_val_loss'] for r in runs]
        ax.plot(N, L, 'o-', color=budget_colors[i],
                label=f'D={budget}', markersize=6)

    ax.set_xscale('log')
    ax.set_xlabel('Model Parameters (N)', fontsize=14)
    ax.set_ylabel('Best Validation Loss', fontsize=14)
    ax.set_title('Scaling: Loss vs Model Size', fontsize=16)
    ax.legend(title='Data Budget', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'loss_vs_params.pdf'), dpi=150)
    fig.savefig(os.path.join(output_dir, 'loss_vs_params.png'), dpi=150)
    plt.close(fig)

    # ---- Plot 3: Val Accuracy versions ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for model_size in MODEL_SIZES:
        runs = [r for r in results if r['model_size'] == model_size and r['best_val_acc'] is not None]
        if not runs:
            continue
        runs.sort(key=lambda r: r['total_samples'])
        D = [r['total_samples'] for r in runs]
        acc = [r['best_val_acc'] for r in runs]
        n_params = runs[0]['n_params']
        ax1.plot(D, acc, 'o-', color=colors[model_size],
                 label=f'{model_size} ({n_params/1e6:.2f}M)', markersize=6)

    ax1.set_xscale('log')
    ax1.set_xlabel('Total Training Samples (D)', fontsize=14)
    ax1.set_ylabel('Best Validation Accuracy', fontsize=14)
    ax1.set_title('Accuracy vs Data Budget', fontsize=16)
    ax1.legend(title='Model Size', fontsize=10)
    ax1.grid(True, alpha=0.3)

    for i, budget in enumerate(DATA_BUDGETS):
        runs = [r for r in results if r['data_budget'] == budget and r['best_val_acc'] is not None]
        if not runs:
            continue
        runs.sort(key=lambda r: r['n_params'])
        N = [r['n_params'] for r in runs]
        acc = [r['best_val_acc'] for r in runs]
        ax2.plot(N, acc, 'o-', color=budget_colors[i],
                 label=f'D={budget}', markersize=6)

    ax2.set_xscale('log')
    ax2.set_xlabel('Model Parameters (N)', fontsize=14)
    ax2.set_ylabel('Best Validation Accuracy', fontsize=14)
    ax2.set_title('Accuracy vs Model Size', fontsize=16)
    ax2.legend(title='Data Budget', fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'accuracy_plots.pdf'), dpi=150)
    fig.savefig(os.path.join(output_dir, 'accuracy_plots.png'), dpi=150)
    plt.close(fig)

    # ---- Plot 4: Fitted scaling law surface ----
    if fit_result is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        A, alpha, B, beta, E = (fit_result['A'], fit_result['alpha'],
                                 fit_result['B'], fit_result['beta'], fit_result['E'])

        # Scatter actual data
        valid = [r for r in results if r['best_val_loss'] is not None]
        N_data = np.array([r['n_params'] for r in valid])
        D_data = np.array([r['total_samples'] for r in valid])
        L_data = np.array([r['best_val_loss'] for r in valid])

        # Predicted vs actual
        L_pred = A * np.power(N_data, -alpha) + B * np.power(D_data, -beta) + E

        ax.scatter(L_data, L_pred, c=[colors[r['model_size']] for r in valid],
                   s=80, edgecolors='k', linewidth=0.5, zorder=5)
        lims = [min(L_data.min(), L_pred.min()) * 0.95, max(L_data.max(), L_pred.max()) * 1.05]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect fit')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Actual Val Loss', fontsize=14)
        ax.set_ylabel('Predicted Val Loss', fontsize=14)
        ax.set_title(f'Scaling Law Fit (R² = {fit_result["r_squared"]:.4f})', fontsize=16)

        # Add legend for model sizes
        for ms in MODEL_SIZES:
            if any(r['model_size'] == ms for r in valid):
                ax.scatter([], [], c=colors[ms], label=ms, s=60, edgecolors='k', linewidth=0.5)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'scaling_law_fit.pdf'), dpi=150)
        fig.savefig(os.path.join(output_dir, 'scaling_law_fit.png'), dpi=150)
        plt.close(fig)

    # ---- Plot 5: Optimal N/D allocation ----
    if optimal is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        ax1.loglog(optimal['compute'], optimal['N_opt'], 'b-', linewidth=2, label='Optimal N')
        ax1.loglog(optimal['compute'], optimal['D_opt'], 'r-', linewidth=2, label='Optimal D')
        ax1.set_xlabel('Compute Budget (N × D)', fontsize=14)
        ax1.set_ylabel('Optimal N or D', fontsize=14)
        ax1.set_title('Compute-Optimal Allocation', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Add text with exponents
        ax1.text(0.05, 0.95,
                 f'N ∝ C^{optimal["N_exponent"]:.3f}\nD ∝ C^{optimal["D_exponent"]:.3f}',
                 transform=ax1.transAxes, fontsize=14, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.semilogx(optimal['compute'], optimal['L_opt'], 'g-', linewidth=2)
        ax2.set_xlabel('Compute Budget (N × D)', fontsize=14)
        ax2.set_ylabel('Optimal Loss', fontsize=14)
        ax2.set_title('Frontier Loss vs Compute', fontsize=16)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'optimal_allocation.pdf'), dpi=150)
        fig.savefig(os.path.join(output_dir, 'optimal_allocation.png'), dpi=150)
        plt.close(fig)

    # ---- Plot 6: IsoFLOP contours ----
    if fit_result is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        A, alpha, B, beta, E = (fit_result['A'], fit_result['alpha'],
                                 fit_result['B'], fit_result['beta'], fit_result['E'])

        N_range = np.logspace(4, 8, 200)
        D_range = np.logspace(6, 10, 200)
        N_grid, D_grid = np.meshgrid(N_range, D_range)
        L_grid = A * np.power(N_grid, -alpha) + B * np.power(D_grid, -beta) + E

        # IsoFLOP lines (C = N * D)
        C_values = np.logspace(12, 17, 6)
        for C in C_values:
            D_line = C / N_range
            mask = (D_line >= D_range.min()) & (D_line <= D_range.max())
            if mask.any():
                L_line = A * np.power(N_range[mask], -alpha) + B * np.power(D_line[mask], -beta) + E
                ax.plot(N_range[mask], L_line, '--', alpha=0.6,
                        label=f'C = {C:.0e}')

        # Overlay actual data points
        valid = [r for r in results if r['best_val_loss'] is not None]
        for r in valid:
            ax.scatter(r['n_params'], r['best_val_loss'],
                       c=colors[r['model_size']], s=60, edgecolors='k',
                       linewidth=0.5, zorder=5)

        ax.set_xscale('log')
        ax.set_xlabel('Model Parameters (N)', fontsize=14)
        ax.set_ylabel('Validation Loss', fontsize=14)
        ax.set_title('IsoFLOP Profiles', fontsize=16)
        ax.legend(title='Compute Budget', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'isoflop_profiles.pdf'), dpi=150)
        fig.savefig(os.path.join(output_dir, 'isoflop_profiles.png'), dpi=150)
        plt.close(fig)

    print(f"\nPlots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Chinchilla Scaling Law Analysis')
    parser.add_argument('--results-dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help='Base directory with scaling study results')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save plots (default: results-dir/analysis)')
    parser.add_argument('--count-params', action='store_true',
                        help='Only count and print model parameters, then exit')
    parser.add_argument('--save-json', action='store_true',
                        help='Save collected results and fit parameters to JSON')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'analysis')

    # ---- Count parameters for each model size ----
    print("=" * 60)
    print("Model Parameter Counts")
    print("=" * 60)
    param_counts = {}
    for model_size in MODEL_SIZES:
        n_params, n_trainable = count_model_params(model_size)
        param_counts[model_size] = n_params
        trainable_str = f" (trainable: {n_trainable:,})" if n_trainable is not None else ""
        print(f"  {model_size:>8s}: {n_params:>12,} params{trainable_str}")
    print()

    if args.count_params:
        return

    # ---- Collect results ----
    print("=" * 60)
    print("Collecting Results")
    print("=" * 60)
    results = collect_results(args.results_dir)

    if not results:
        print("\nNo results found. Check that training runs have completed.")
        print(f"Expected TensorBoard logs in: {args.results_dir}/tensorboard/")
        print(f"Expected text logs in: {args.results_dir}/logs/")
        return

    print(f"\nFound {len(results)} completed runs out of {len(MODEL_SIZES) * len(DATA_BUDGETS)} total")

    # ---- Fit scaling law ----
    print("\n" + "=" * 60)
    print("Fitting Scaling Law")
    print("=" * 60)

    loss_fit = fit_scaling_law(results, metric='loss')
    acc_fit = fit_scaling_law(results, metric='acc')

    # ---- Compute optimal allocation ----
    optimal = compute_optimal_allocation(loss_fit)
    if optimal is not None:
        print(f"\nOptimal allocation exponents:")
        print(f"  N_opt ∝ C^{optimal['N_exponent']:.4f}")
        print(f"  D_opt ∝ C^{optimal['D_exponent']:.4f}")
        print(f"  (Chinchilla found N ∝ C^0.50, D ∝ C^0.50)")

    # ---- Save JSON ----
    if args.save_json:
        os.makedirs(args.output_dir, exist_ok=True)
        json_path = os.path.join(args.output_dir, 'scaling_results.json')
        json_data = {
            'param_counts': param_counts,
            'results': [{k: v for k, v in r.items() if k not in ('val_loss_history', 'val_acc_history')}
                        for r in results],
            'loss_fit': loss_fit,
            'acc_fit': acc_fit,
            'optimal': {k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in optimal.items()} if optimal else None,
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"\nResults saved to {json_path}")

    # ---- Generate plots ----
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)
    plot_results(results, loss_fit, optimal, args.output_dir)


if __name__ == '__main__':
    main()
