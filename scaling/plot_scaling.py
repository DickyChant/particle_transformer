#!/usr/bin/env python
"""
Plot Chinchilla scaling laws from parsed CSV.

Reads the CSV produced by parse_runs.py and generates scaling law plots.

Usage:
    python plot_scaling.py                           # last epoch per run, best loss
    python plot_scaling.py --epoch 0                  # epoch 0 only (1st pass)
    python plot_scaling.py --epoch 0 --filter sample_type=Pythia
    python plot_scaling.py --best                     # best loss across all epochs (default)
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---- Constants ----

DEFAULT_CSV = '/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/parsed_runs.csv'
DEFAULT_OUTPUT_DIR = '/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/plots'

MODEL_SIZES = ['nano', 'micro', 'tiny', 'small', 'base', 'large', 'xlarge']
DATA_BUDGETS = ['5M', '10M', '25M', '50M', '100M', '250M', '500M']

PARAMS_M = {
    'nano': 0.1, 'micro': 0.2, 'tiny': 0.4,
    'small': 1.0, 'base': 2.14, 'large': 8.5, 'xlarge': 25.0,
}

# Default epoch config: (samples_per_epoch_per_gpu, num_epochs, ngpus=4)
BUDGET_EPOCH_CONFIG = {
    '5M':   (320_000,   4),
    '10M':  (640_000,   4),
    '25M':  (1_600_000, 4),
    '50M':  (1_600_000, 8),
    '100M': (1_600_000, 16),
    '250M': (1_600_000, 40),
    '500M': (2_560_000, 50),
}
DEFAULT_NGPUS = 4

MODEL_COLORS = {
    'nano': '#1f77b4', 'micro': '#ff7f0e', 'tiny': '#2ca02c',
    'small': '#d62728', 'base': '#9467bd', 'large': '#8c564b',
    'xlarge': '#e377c2',
}

UNIQUE_SAMPLES = {
    'Pythia': 100_000_000,
    'Herwig': 100_000_000,
    'Mixed':  200_000_000,
}


# ---- Chinchilla scaling law ----

def chinchilla_loss(X, A, alpha, B, beta, E):
    """L(N, D) = A * N^(-alpha) + B * D^(-beta) + E"""
    N, D = X
    return A * np.power(N, -alpha) + B * np.power(D, -beta) + E


def fit_chinchilla(N, D, L):
    """Fit Chinchilla scaling law. Returns (popt, r2) or (None, None)."""
    if len(L) < 5:
        print(f"  Warning: only {len(L)} data points, need >= 5 for fit.")
        return None, None

    p0 = [1.0, 0.5, 1.0, 0.5, np.min(L)]
    try:
        popt, _ = curve_fit(
            chinchilla_loss, (N, D), L, p0=p0,
            bounds=([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, np.max(L)]),
            maxfev=50000,
        )
        L_pred = chinchilla_loss((N, D), *popt)
        ss_res = np.sum((L - L_pred) ** 2)
        ss_tot = np.sum((L - np.mean(L)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return popt, r2
    except Exception as e:
        print(f"  Chinchilla fit failed: {e}")
        return None, None


# ---- Data selection ----

def select_epoch_data(df, epoch=None):
    """
    Select one row per (model, budget, config) based on epoch choice.

    epoch=None: best (lowest) loss across all epochs per run
    epoch=N:    loss at epoch N (0-indexed)

    Returns df with 'D_M' column = total_samples_seen / 1e6 for that selection.
    """
    group_keys = ['model_size', 'data_budget', 'sample_type', 'feature_type', 'pair_tag']

    if epoch is not None:
        selected = df[df['epoch'] == epoch].copy()
    else:
        # Best loss per run
        selected = df.sort_values('train_loss').drop_duplicates(
            subset=group_keys, keep='first').copy()

    selected['D_M'] = selected['total_samples_seen'] / 1e6
    return selected


# ---- Plotting ----

def make_predictions(popt, observed_df, epoch=None):
    """Generate predictions for missing (model, budget) combos."""
    if popt is None:
        return pd.DataFrame()

    observed_keys = set(zip(observed_df['model_size'], observed_df['data_budget']))
    preds = []
    for model in MODEL_SIZES:
        for budget in DATA_BUDGETS:
            if (model, budget) in observed_keys:
                continue
            n_params = PARAMS_M[model]
            spe, num_ep = BUDGET_EPOCH_CONFIG[budget]
            if epoch is not None:
                d_samples = spe * DEFAULT_NGPUS * (epoch + 1)
            else:
                d_samples = spe * DEFAULT_NGPUS * num_ep
            d_m = d_samples / 1e6
            pred_loss = chinchilla_loss((n_params, d_m), *popt)
            preds.append({
                'model_size': model, 'data_budget': budget,
                'params_M': n_params, 'D_M': d_m,
                'total_samples_seen': d_samples,
                'train_loss': pred_loss, 'predicted': True,
            })
    return pd.DataFrame(preds)


def plot_scaling_pair(df, title_suffix, output_path, unique_samples, pred_df=None):
    """Two-panel: Loss vs Data + Loss vs Model Size, with optional predictions."""
    N = df['params_M'].values
    D = df['D_M'].values
    L = df['train_loss'].values

    popt, r2 = fit_chinchilla(N, D, L)
    if popt is not None:
        A, alpha, B, beta, E = popt
        print(f"  Fit: L = {A:.4f}*N^(-{alpha:.4f}) + {B:.4f}*D^(-{beta:.4f}) + {E:.6f}  [R²={r2:.4f}]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    unique_M = unique_samples / 1e6

    # Combine observed + predicted D values for right panel coloring
    all_d_vals = sorted(df['D_M'].unique())
    if pred_df is not None and not pred_df.empty:
        all_d_vals = sorted(set(all_d_vals) | set(pred_df['D_M'].unique()))
    budget_cmap = {d: plt.cm.viridis(i / max(len(all_d_vals) - 1, 1))
                   for i, d in enumerate(all_d_vals)}

    # ---- Left: Loss vs Data ----
    for model in MODEL_SIZES:
        sub = df[df['model_size'] == model].sort_values('D_M')
        if sub.empty:
            continue
        n_params = sub['params_M'].iloc[0]
        c = MODEL_COLORS.get(model, 'gray')
        reuse_mask = sub['total_samples_seen'] > unique_samples

        no_reuse = sub[~reuse_mask]
        with_reuse = sub[reuse_mask]
        label = f"{model} ({n_params:.2f}M)"
        if not no_reuse.empty:
            ax1.plot(no_reuse['D_M'], no_reuse['train_loss'],
                     marker='o', ls='', color=c, label=label, ms=7)
        if not with_reuse.empty:
            ax1.plot(with_reuse['D_M'], with_reuse['train_loss'],
                     marker='o', ls='', color=c, ms=7,
                     markerfacecolor='none', markeredgewidth=1.5,
                     label=f"{model} (reused)" if no_reuse.empty else None)

        # Plot predictions for this model
        if pred_df is not None and not pred_df.empty:
            psub = pred_df[pred_df['model_size'] == model].sort_values('D_M')
            if not psub.empty:
                ax1.plot(psub['D_M'], psub['train_loss'],
                         marker='x', ls='', color=c, ms=8, markeredgewidth=2, alpha=0.7)

        if popt is not None:
            d_all = list(df['D_M'])
            if pred_df is not None and not pred_df.empty:
                d_all += list(pred_df['D_M'])
            x_fit = np.logspace(np.log10(max(min(d_all) * 0.8, 0.5)),
                                np.log10(max(d_all) * 1.2), 50)
            ax1.plot(x_fit, chinchilla_loss((n_params, x_fit), *popt),
                     color=c, alpha=0.6, ls='--')

    # Add a single legend entry for predictions
    if pred_df is not None and not pred_df.empty:
        ax1.plot([], [], marker='x', ls='', color='gray', ms=8,
                 markeredgewidth=2, label='predicted')

    ax1.axvline(x=unique_M, color='gray', ls=':', alpha=0.5,
                label=f'unique data ({unique_M:.0f}M)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Training Samples Seen D (M)', fontsize=13)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=13)
    ax1.set_title(f'Loss vs Data\n{title_suffix}')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, which='both', ls='-', alpha=0.3)

    if popt is not None:
        eq_str = (f"R²={r2:.4f}\n"
                  f"$A={A:.3f}$, $\\alpha={alpha:.3f}$\n"
                  f"$B={B:.3f}$, $\\beta={beta:.3f}$\n"
                  f"$E={E:.5f}$")
        ax1.text(0.03, 0.03, eq_str, transform=ax1.transAxes, fontsize=9,
                 va='bottom', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # ---- Right: Loss vs Model Size ----
    for d_val in all_d_vals:
        sub = df[df['D_M'] == d_val].sort_values('params_M')
        c = budget_cmap[d_val]
        is_reused = d_val * 1e6 > unique_samples
        label = f"D={d_val:.1f}M" + (" (reused)" if is_reused else "")
        if not sub.empty:
            ax2.plot(sub['params_M'], sub['train_loss'],
                     marker='s', ls='', color=c, label=label, ms=7,
                     markerfacecolor='none' if is_reused else c,
                     markeredgewidth=1.5 if is_reused else 1.0)

        if pred_df is not None and not pred_df.empty:
            psub = pred_df[pred_df['D_M'] == d_val].sort_values('params_M')
            if not psub.empty:
                ax2.plot(psub['params_M'], psub['train_loss'],
                         marker='x', ls='', color=c, ms=8, markeredgewidth=2, alpha=0.7,
                         label=label + ' (pred)' if sub.empty else None)

        if popt is not None:
            x_fit = np.logspace(np.log10(0.05), np.log10(30.0), 50)
            ax2.plot(x_fit, chinchilla_loss((x_fit, d_val), *popt),
                     color=c, alpha=0.6, ls='--')

    ax2.set_xscale('log')
    ax2.set_xlabel('Model Parameters N (M)', fontsize=13)
    ax2.set_ylabel('Cross-Entropy Loss', fontsize=13)
    ax2.set_title(f'Loss vs Model Size\n{title_suffix}')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, which='both', ls='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"  Saved: {output_path}")
    plt.close(fig)
    return popt, r2


def plot_comparison(fits, output_path):
    """Compare scaling law fits across configurations."""
    if len(fits) < 2:
        return

    labels = list(fits.keys())
    alphas = [fits[k]['popt'][1] for k in labels]
    betas = [fits[k]['popt'][3] for k in labels]
    Es = [fits[k]['popt'][4] for k in labels]
    r2s = [fits[k]['r2'] for k in labels]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(labels))
    width = 0.6

    axes[0].barh(x, alphas, height=width, color='steelblue')
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].set_xlabel(r'$\alpha$ (model scaling exponent)')
    axes[0].set_title(r'$\alpha$: how much more params help')

    axes[1].barh(x, betas, height=width, color='coral')
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(labels, fontsize=9)
    axes[1].set_xlabel(r'$\beta$ (data scaling exponent)')
    axes[1].set_title(r'$\beta$: how much more data helps')

    axes[2].barh(x, Es, height=width, color='seagreen')
    axes[2].set_yticks(x)
    axes[2].set_yticklabels(labels, fontsize=9)
    axes[2].set_xlabel('E (irreducible loss)')
    axes[2].set_title('Irreducible loss floor')

    for i, (e, r2) in enumerate(zip(Es, r2s)):
        axes[2].text(e + 0.001, i, f'R²={r2:.3f}', va='center', fontsize=8)

    plt.suptitle(r'Scaling Law Comparison: $L(N,D) = A \cdot N^{-\alpha} + B \cdot D^{-\beta} + E$',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"\nSaved comparison: {output_path}")
    plt.close(fig)


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description='Plot Chinchilla scaling laws from parsed CSV')
    parser.add_argument('--csv', default=DEFAULT_CSV, help='Input CSV from parse_runs.py')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Select specific epoch (0-indexed). Default: best loss across all epochs.')
    parser.add_argument('--filter', nargs='+', default=None,
                        help='Filter: key=value pairs (e.g., sample_type=Pythia)')
    parser.add_argument('--predict', action='store_true',
                        help='Predict missing runs from fitted scaling law')
    parser.add_argument('--complete-only', action='store_true',
                        help='Exclude incomplete runs (max epoch < num_epochs_cfg - 1)')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Output directory for plots')
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}")
        print("Run parse_runs.py first.")
        return

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} epoch entries from {args.csv}")

    # Apply filters
    if args.filter:
        for filt in args.filter:
            key, val = filt.split('=', 1)
            df = df[df[key].astype(str) == val]
            print(f"  Filter {key}={val} -> {len(df)} rows")

    if df.empty:
        print("No data after filtering.")
        return

    # Exclude incomplete runs if requested
    if args.complete_only:
        group_keys = ['model_size', 'data_budget', 'sample_type', 'feature_type', 'pair_tag']
        complete_mask = df.groupby(group_keys).apply(
            lambda g: g['epoch'].max() >= g['num_epochs_cfg'].iloc[0] - 1
        )
        complete_keys = complete_mask[complete_mask].index
        before = df[group_keys].drop_duplicates().shape[0]
        df = df.merge(
            pd.DataFrame(complete_keys.tolist(), columns=group_keys),
            on=group_keys, how='inner')
        after = df[group_keys].drop_duplicates().shape[0]
        print(f"  --complete-only: {before} -> {after} runs ({before - after} incomplete excluded)")

    # Select epoch
    epoch_tag = f'_epoch{args.epoch}' if args.epoch is not None else '_best'
    epoch_label = f' [epoch {args.epoch}]' if args.epoch is not None else ' [best]'
    selected = select_epoch_data(df, epoch=args.epoch)

    print(f"\nSelected {len(selected)} data points{epoch_label}")
    print(selected[['model_size', 'data_budget', 'sample_type', 'feature_type',
                     'pair_tag', 'params_M', 'D_M', 'train_loss', 'source']]
          .sort_values(['sample_type', 'feature_type', 'pair_tag', 'D_M', 'params_M'])
          .to_string(index=False))

    os.makedirs(args.output_dir, exist_ok=True)

    # Plot per config slice
    fits = {}
    config_groups = selected.groupby(['sample_type', 'feature_type', 'pair_tag'])

    for (st, ft, pt), group in config_groups:
        if len(group) < 3:
            print(f"\nSkipping {st}/{ft}/{pt}: only {len(group)} points")
            continue

        unique = UNIQUE_SAMPLES.get(st, 100_000_000)
        config_label = f"{st} / {ft} / {pt}{epoch_label}"
        safe_name = f"{st}_{ft}_{pt}{epoch_tag}"
        output_path = os.path.join(args.output_dir, f'scaling_{safe_name}.png')

        print(f"\n--- {config_label} ({len(group)} points) ---")

        # Generate predictions if requested
        pred_df = None
        if args.predict:
            safe_name += '_pred'
            output_path = os.path.join(args.output_dir, f'scaling_{safe_name}.png')
            N = group['params_M'].values
            D = group['D_M'].values
            L = group['train_loss'].values
            popt_pre, _ = fit_chinchilla(N, D, L)
            pred_df = make_predictions(popt_pre, group, epoch=args.epoch)
            if not pred_df.empty:
                print(f"  Predictions for {len(pred_df)} missing (model, budget) combos:")
                print(pred_df[['model_size', 'data_budget', 'params_M', 'D_M', 'train_loss']]
                      .sort_values(['D_M', 'params_M']).to_string(index=False))

        popt, r2 = plot_scaling_pair(group, config_label, output_path, unique, pred_df=pred_df)

        if popt is not None:
            fits[config_label] = {'popt': popt, 'r2': r2}

    # Comparison
    if len(fits) >= 2:
        plot_comparison(fits, os.path.join(args.output_dir, f'scaling_comparison{epoch_tag}.png'))

    # Combined plot
    if len(config_groups) > 1 and len(selected) >= 5:
        print(f"\n--- All configs combined ({len(selected)} points) ---")
        plot_scaling_pair(selected, f'All configurations{epoch_label}',
                         os.path.join(args.output_dir, f'scaling_all{epoch_tag}.png'),
                         unique_samples=min(UNIQUE_SAMPLES.values()))


if __name__ == '__main__':
    main()
