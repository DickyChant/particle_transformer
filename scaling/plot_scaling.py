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

# Empirical loss-to-metric mappings: log(1-metric) = slope * L + intercept
# Fitted from 1-epoch Pythia/full/pair runs. R² > 0.98 for all.
LOSS_TO_METRIC = {
    'train_acc':   {'slope': 0.8005, 'intercept': -1.8636, 'r2': 0.982, 'label': 'Training Accuracy'},
    'val_metric':  {'slope': 0.7043, 'intercept': -2.0279, 'r2': 0.994, 'label': 'Validation AUC'},
    'val_roc_auc': {'slope': 1.2751, 'intercept': -4.5410, 'r2': 0.994, 'label': 'ROC AUC Score'},
    'bkg_rej':     {'slope': -1.2751, 'intercept': 4.5410, 'r2': 0.994, 'label': 'Background Rejection (1/FPR)'},
}


def loss_to_metric(loss_vals, metric_name):
    """Convert loss to derived metric using empirical mapping.

    For acc/AUC: metric = 1 - exp(slope * L + intercept)
    For bkg_rej: metric = exp(slope * L + intercept)
    """
    m = LOSS_TO_METRIC[metric_name]
    log_vals = m['slope'] * loss_vals + m['intercept']
    if metric_name == 'bkg_rej':
        return np.exp(log_vals)
    return 1.0 - np.exp(log_vals)


# ---- Chinchilla scaling law ----

def chinchilla_loss(X, A, alpha, B, beta, E):
    """L(N, D) = A * N^(-alpha) + B * D^(-beta) + E"""
    N, D = X
    return A * np.power(N, -alpha) + B * np.power(D, -beta) + E


def chinchilla_gain(X, A, alpha, B, beta, E):
    """Inverted Chinchilla for metrics that increase: M(N,D) = E - A*N^(-alpha) - B*D^(-beta)"""
    N, D = X
    return E - A * np.power(N, -alpha) - B * np.power(D, -beta)


def fit_chinchilla(N, D, L, increasing=False):
    """Fit Chinchilla scaling law. Returns (popt, r2, increasing) or (None, None, None).

    For loss (decreasing): L = A*N^(-a) + B*D^(-b) + E
    For acc/AUC (increasing): M = E - A*N^(-a) - B*D^(-b)
    """
    if len(L) < 5:
        print(f"  Warning: only {len(L)} data points, need >= 5 for fit.")
        return None, None

    func = chinchilla_gain if increasing else chinchilla_loss

    if increasing:
        p0 = [0.5, 0.5, 0.5, 0.5, np.max(L)]
        bounds = ([0, 0, 0, 0, np.min(L)], [np.inf, 2, np.inf, 2, 1.0])
    else:
        # E can be negative (e.g. for log(1-x) transforms)
        e_min = min(0, np.min(L) * 2) if np.min(L) < 0 else 0
        p0 = [1.0, 0.5, 1.0, 0.5, np.min(L)]
        bounds = ([0, 0, 0, 0, e_min], [np.inf, 2, np.inf, 2, np.max(L)])

    try:
        popt, _ = curve_fit(func, (N, D), L, p0=p0, bounds=bounds, maxfev=50000)
        L_pred = func((N, D), *popt)
        ss_res = np.sum((L - L_pred) ** 2)
        ss_tot = np.sum((L - np.mean(L)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return popt, r2
    except Exception as e:
        print(f"  Chinchilla fit failed: {e}")
        return None, None


# ---- Data selection ----

def select_epoch_data(df, epoch=None, metric='train_loss'):
    """
    Select one row per (model, budget, config) based on epoch choice.

    epoch=None: best metric across all epochs per run
    epoch=N:    metric at epoch N (0-indexed)

    Returns df with 'D_M' column = total_samples_seen / 1e6 for that selection.
    """
    # `run_type` (multi vs 1ep) and `task_type` (singletask vs multitask) are
    # both part of the run identity. Merging across either silently mixes
    # qualitatively different scaling regimes / loss definitions.
    group_keys = ['model_size', 'data_budget', 'sample_type', 'feature_type',
                  'pair_tag', 'run_type']
    if 'task_type' in df.columns:
        group_keys = group_keys + ['task_type']

    # Drop rows where the metric is NaN
    df = df.dropna(subset=[metric])

    if epoch is not None:
        selected = df[df['epoch'] == epoch].copy()
    else:
        # For loss: lower is better. For acc/auc: higher is better.
        if 'loss' in metric:
            selected = df.sort_values(metric).drop_duplicates(
                subset=group_keys, keep='first').copy()
        else:
            selected = df.sort_values(metric, ascending=False).drop_duplicates(
                subset=group_keys, keep='first').copy()

    selected['D_M'] = selected['total_samples_seen'] / 1e6
    return selected


# ---- Plotting ----

def make_predictions(popt, observed_df, epoch=None, metric='train_loss', increasing=False, transform=None):
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
            func = chinchilla_gain if increasing else chinchilla_loss
            pred_val = func((n_params, d_m), *popt)
            # Skip predictions outside the valid range
            if increasing and (pred_val < 0 or pred_val > 1):
                continue
            if transform == 'log1mx' and pred_val > 0:
                continue
            preds.append({
                'model_size': model, 'data_budget': budget,
                'params_M': n_params, 'D_M': d_m,
                'total_samples_seen': d_samples,
                metric: pred_val, 'predicted': True,
            })
    return pd.DataFrame(preds)


def plot_scaling_pair(df, title_suffix, output_path, unique_samples, pred_df=None,
                     metric='train_loss', increasing=False, transform=None):
    """Two-panel: metric vs Data + metric vs Model Size, with optional predictions."""
    METRIC_LABELS = {
        'train_loss': 'Cross-Entropy Loss',
        'train_acc': 'Training Accuracy',
        'val_metric': 'Validation AUC',
        'val_roc_auc': 'ROC AUC Score',
    }
    ylabel = METRIC_LABELS.get(metric, metric)
    if transform == 'log1mx':
        ylabel = f'log(1 - {ylabel})'

    N = df['params_M'].values
    D = df['D_M'].values
    L = df[metric].values

    popt, r2 = fit_chinchilla(N, D, L, increasing=increasing)
    fit_func = chinchilla_gain if increasing else chinchilla_loss
    if popt is not None:
        A, alpha, B, beta, E = popt
        if increasing:
            print(f"  Fit: M = {E:.6f} - {A:.4f}*N^(-{alpha:.4f}) - {B:.4f}*D^(-{beta:.4f})  [R²={r2:.4f}]")
        else:
            print(f"  Fit: L = {A:.4f}*N^(-{alpha:.4f}) + {B:.4f}*D^(-{beta:.4f}) + {E:.6f}  [R²={r2:.4f}]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    unique_M = unique_samples / 1e6

    # Combine observed + predicted D values for right panel coloring
    all_d_vals = sorted(df['D_M'].unique())
    if pred_df is not None and not pred_df.empty:
        all_d_vals = sorted(set(all_d_vals) | set(pred_df['D_M'].unique()))
    budget_cmap = {d: plt.cm.viridis(i / max(len(all_d_vals) - 1, 1))
                   for i, d in enumerate(all_d_vals)}

    # D range for fit lines (span observed + predicted)
    d_all = list(df['D_M'])
    if pred_df is not None and not pred_df.empty:
        d_all += list(pred_df['D_M'])
    d_fit_range = np.logspace(np.log10(max(min(d_all) * 0.8, 0.5)),
                              np.log10(max(d_all) * 1.2), 50) if d_all else None

    # ---- Left: Loss vs Data ----
    for model in MODEL_SIZES:
        n_params = PARAMS_M[model]
        c = MODEL_COLORS.get(model, 'gray')
        sub = df[df['model_size'] == model].sort_values('D_M')
        has_data = not sub.empty

        label = f"{model} ({n_params:.2f}M)"
        if has_data:
            reuse_mask = sub['total_samples_seen'] > unique_samples
            no_reuse = sub[~reuse_mask]
            with_reuse = sub[reuse_mask]
            if not no_reuse.empty:
                ax1.plot(no_reuse['D_M'], no_reuse[metric],
                         marker='o', ls='', color=c, label=label, ms=7)
            if not with_reuse.empty:
                ax1.plot(with_reuse['D_M'], with_reuse[metric],
                         marker='o', ls='', color=c, ms=7,
                         markerfacecolor='none', markeredgewidth=1.5,
                         label=f"{model} (reused)" if no_reuse.empty else None)

        # Plot predictions for this model
        if pred_df is not None and not pred_df.empty:
            psub = pred_df[pred_df['model_size'] == model].sort_values('D_M')
            if not psub.empty:
                ax1.plot(psub['D_M'], psub[metric],
                         marker='x', ls='', color=c, ms=8, markeredgewidth=2, alpha=0.7,
                         label=label + ' (pred)' if not has_data else None)

        # Always draw fit line for every model
        if popt is not None and d_fit_range is not None:
            ax1.plot(d_fit_range, fit_func((n_params, d_fit_range), *popt),
                     color=c, alpha=0.4 if not has_data else 0.6,
                     ls=':' if not has_data else '--')

    # Add a single legend entry for predictions
    if pred_df is not None and not pred_df.empty:
        ax1.plot([], [], marker='x', ls='', color='gray', ms=8,
                 markeredgewidth=2, label='predicted')

    ax1.axvline(x=unique_M, color='gray', ls=':', alpha=0.5,
                label=f'unique data ({unique_M:.0f}M)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Training Samples Seen D (M)', fontsize=13)
    ax1.set_ylabel(ylabel, fontsize=13)
    ax1.set_title(f'{ylabel} vs Data\n{title_suffix}')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, which='both', ls='-', alpha=0.3)

    if popt is not None:
        eq_str = (f"R²={r2:.4f}\n"
                  f"$A={A:.3f}$, $\\alpha={alpha:.3f}$\n"
                  f"$B={B:.3f}$, $\\beta={beta:.3f}$\n"
                  f"$E={E:.5f}$")
        text_loc = (0.03, 0.97 if increasing else 0.03)
        ax1.text(text_loc[0], text_loc[1], eq_str, transform=ax1.transAxes, fontsize=9,
                 va='top' if increasing else 'bottom',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # ---- Right: Loss vs Model Size ----
    for d_val in all_d_vals:
        sub = df[df['D_M'] == d_val].sort_values('params_M')
        c = budget_cmap[d_val]
        is_reused = d_val * 1e6 > unique_samples
        label = f"D={d_val:.1f}M" + (" (reused)" if is_reused else "")
        if not sub.empty:
            ax2.plot(sub['params_M'], sub[metric],
                     marker='s', ls='', color=c, label=label, ms=7,
                     markerfacecolor='none' if is_reused else c,
                     markeredgewidth=1.5 if is_reused else 1.0)

        if pred_df is not None and not pred_df.empty:
            psub = pred_df[pred_df['D_M'] == d_val].sort_values('params_M')
            if not psub.empty:
                ax2.plot(psub['params_M'], psub[metric],
                         marker='x', ls='', color=c, ms=8, markeredgewidth=2, alpha=0.7,
                         label=label + ' (pred)' if sub.empty else None)

        if popt is not None:
            x_fit = np.logspace(np.log10(0.05), np.log10(30.0), 50)
            ax2.plot(x_fit, fit_func((x_fit, d_val), *popt),
                     color=c, alpha=0.6, ls='--')

    ax2.set_xscale('log')
    ax2.set_xlabel('Model Parameters N (M)', fontsize=13)
    ax2.set_ylabel(ylabel, fontsize=13)
    ax2.set_title(f'{ylabel} vs Model Size\n{title_suffix}')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, which='both', ls='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"  Saved: {output_path}")
    plt.close(fig)
    return popt, r2


def plot_derived(df, popt, title_suffix, output_path, unique_samples, derived_metrics=None):
    """Plot derived metrics (acc, AUC, bkg rejection) from loss scaling law.

    Uses the loss fit L(N,D) and empirical mappings to show how physics metrics scale.
    """
    if popt is None or derived_metrics is None:
        return

    n_metrics = len(derived_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 7))
    if n_metrics == 1:
        axes = [axes]

    unique_M = unique_samples / 1e6

    for ax, dm in zip(axes, derived_metrics):
        m_info = LOSS_TO_METRIC[dm]

        # Plot observed data points
        for model in MODEL_SIZES:
            sub = df[df['model_size'] == model].sort_values('D_M')
            if sub.empty:
                continue
            n_params = sub['params_M'].iloc[0]
            c = MODEL_COLORS.get(model, 'gray')

            # Compute derived metric from observed loss
            obs_metric = loss_to_metric(sub['train_loss'].values, dm)

            ax.plot(sub['D_M'], obs_metric,
                    marker='o', ls='', color=c, ms=7,
                    label=f"{model} ({n_params:.2f}M)")

        # Plot fit curves for all models
        d_range = df['D_M']
        x_fit = np.logspace(np.log10(max(d_range.min() * 0.8, 0.5)),
                            np.log10(d_range.max() * 1.2), 50)
        for model in MODEL_SIZES:
            n_params = PARAMS_M[model]
            c = MODEL_COLORS.get(model, 'gray')
            loss_fit = chinchilla_loss((n_params, x_fit), *popt)
            metric_fit = loss_to_metric(loss_fit, dm)
            has_data = not df[df['model_size'] == model].empty
            ax.plot(x_fit, metric_fit, color=c, alpha=0.4 if not has_data else 0.6,
                    ls=':' if not has_data else '--')

        if dm == 'bkg_rej':
            ax.set_yscale('log')

        ax.axvline(x=unique_M, color='gray', ls=':', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('Training Samples D (M)', fontsize=12)
        ax.set_ylabel(m_info['label'], fontsize=12)
        ax.set_title(f'{m_info["label"]} vs Data\n{title_suffix}')
        ax.legend(fontsize=7, loc='lower right' if dm != 'bkg_rej' else 'upper right')
        ax.grid(True, which='both', ls='-', alpha=0.3)

        # Annotation: mapping info
        ax.text(0.03, 0.97, f"from loss via\nlog(1-m)={m_info['slope']:.3f}L+{m_info['intercept']:.3f}\n(R²={m_info['r2']:.3f})",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.suptitle(f'Derived Metrics from Loss Scaling Law\n{title_suffix}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close(fig)


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


def generate_table(df, output_dir=None):
    """Fit scaling laws for every config and print a comprehensive table."""
    group_keys = ['model_size', 'data_budget']
    config_keys = ['sample_type', 'feature_type', 'pair_tag', 'run_type']

    rows = []
    for keys, gdf in df.groupby(config_keys):
        st, ft, pt, rt = keys

        # Best loss per (model, budget)
        best = gdf.sort_values('train_loss').drop_duplicates(
            subset=group_keys, keep='first').dropna(subset=['train_loss'])

        n_runs = len(best)
        n_models = best['model_size'].nunique()
        n_budgets = best['data_budget'].nunique()
        if n_runs < 5:
            continue

        N = best['params_M'].values
        D = best['total_samples_seen'].values / 1e6
        L = best['train_loss'].values

        popt, r2 = fit_chinchilla(N, D, L, increasing=False)
        if popt is None:
            continue

        A, alpha, B, beta, E = popt
        rows.append({
            'sample': st, 'features': ft, 'pair': pt, 'run_type': rt,
            'n_runs': n_runs, 'n_models': n_models, 'n_budgets': n_budgets,
            'A': A, 'alpha': alpha, 'B': B, 'beta': beta, 'E': E, 'R2': r2,
        })

    if not rows:
        print("No configs with enough data to fit.")
        return

    # Sort: 1ep first, then by sample/features/pair
    rows.sort(key=lambda r: (0 if r['run_type'] == '1ep' else 1,
                             r['sample'], r['features'], r['pair']))

    # Print table
    hdr = f"{'Config':<35s} {'Runs':>4s} {'Mdl':>3s} {'Bdg':>3s}  {'A':>7s} {'α':>7s} {'B':>7s} {'β':>7s} {'E':>8s} {'R²':>6s}"
    sep = '=' * len(hdr)
    print(f"\n{sep}")
    print("  Chinchilla Scaling Law: L(N,D) = A·N^(-α) + B·D^(-β) + E")
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        label = f"{r['sample']}/{r['features']}/{r['pair']} ({r['run_type']})"
        print(f"{label:<35s} {r['n_runs']:4d} {r['n_models']:3d} {r['n_budgets']:3d}  "
              f"{r['A']:7.4f} {r['alpha']:7.4f} {r['B']:7.4f} {r['beta']:7.4f} {r['E']:8.5f} {r['R2']:6.4f}")
    print(sep)
    print(f"{'Vigl et al. (SetTransformer)':<35s} {'':>4s} {'':>3s} {'':>3s}  "
          f"{'':>7s} {'0.4400':>7s} {'':>7s} {'0.2200':>7s} {'0.32000':>8s} {'':>6s}")
    print(sep)

    # Save CSV
    if output_dir:
        csv_path = os.path.join(output_dir, 'scaling_table.csv')
        table_df = pd.DataFrame(rows)
        table_df.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description='Plot Chinchilla scaling laws from parsed CSV')
    parser.add_argument('--csv', default=DEFAULT_CSV, help='Input CSV from parse_runs.py')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Select specific epoch (0-indexed). Default: best loss across all epochs.')
    parser.add_argument('--metric', default='train_loss',
                        choices=['train_loss', 'train_acc', 'val_metric', 'val_roc_auc'],
                        help='Metric to plot (default: train_loss)')
    parser.add_argument('--transform', default=None,
                        choices=['none', 'log1mx'],
                        help='Transform metric before fitting. '
                             'log1mx: log(1-x), maps acc/AUC approaching 1 to decreasing loss-like values. '
                             'Default: auto (log1mx for acc/AUC, none for loss).')
    parser.add_argument('--filter', nargs='+', default=None,
                        help='Filter: key=value pairs (e.g., sample_type=Pythia)')
    parser.add_argument('--predict', action='store_true',
                        help='Predict missing runs from fitted scaling law')
    parser.add_argument('--derived', action='store_true',
                        help='Plot derived metrics (acc, AUC, bkg rejection) from loss fit')
    parser.add_argument('--table', action='store_true',
                        help='Print comprehensive scaling law table across all configs')
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

    # Table mode: fit all configs and print summary
    if args.table:
        generate_table(df, output_dir=args.output_dir)
        return

    # Apply filters
    if args.filter:
        for filt in args.filter:
            key, val = filt.split('=', 1)
            df = df[df[key].astype(str) == val]
            print(f"  Filter {key}={val} -> {len(df)} rows")

    # Build a suffix from filters for output naming
    filter_tag = ''
    if args.filter:
        for filt in args.filter:
            key, val = filt.split('=', 1)
            filter_tag += f'_{val}'

    if df.empty:
        print("No data after filtering.")
        return

    # Exclude incomplete runs if requested
    if args.complete_only:
        group_keys = ['model_size', 'data_budget', 'sample_type', 'feature_type',
                      'pair_tag', 'run_type']
        if 'task_type' in df.columns:
            group_keys = group_keys + ['task_type']
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
    metric = args.metric
    increasing = 'loss' not in metric

    # Auto-select transform
    transform = args.transform
    if transform is None:
        transform = 'log1mx' if increasing else 'none'
    if transform == 'none':
        transform = None

    metric_tag = '' if metric == 'train_loss' else f'_{metric}'
    if transform:
        metric_tag += f'_{transform}'

    selected = select_epoch_data(df, epoch=args.epoch, metric=metric)

    # Apply transform
    if transform and not selected.empty:
        raw_col = metric
        if transform == 'log1mx':
            # log(1-x): negative, decreasing as x->1 (like a loss: lower=better)
            selected[metric] = np.log(np.clip(1.0 - selected[metric], 1e-10, 1.0))
            increasing = False
        print(f"  Applied transform: {transform} to {raw_col}")

    print(f"\nSelected {len(selected)} data points{epoch_label}")
    show_cols = ['model_size', 'data_budget', 'sample_type', 'params_M', 'D_M', metric]
    print(selected[show_cols]
          .sort_values(['sample_type', 'D_M', 'params_M'])
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
        safe_name = f"{st}_{ft}_{pt}{filter_tag}{metric_tag}{epoch_tag}"
        output_path = os.path.join(args.output_dir, f'scaling_{safe_name}.png')

        print(f"\n--- {config_label} ({len(group)} points) ---")

        # Generate predictions if requested
        pred_df = None
        if args.predict:
            safe_name += '_pred'
            output_path = os.path.join(args.output_dir, f'scaling_{safe_name}.png')
            N = group['params_M'].values
            D = group['D_M'].values
            L = group[metric].values
            popt_pre, _ = fit_chinchilla(N, D, L, increasing=increasing)
            pred_df = make_predictions(popt_pre, group, epoch=args.epoch, metric=metric,
                                       increasing=increasing, transform=transform)
            if not pred_df.empty:
                print(f"  Predictions for {len(pred_df)} missing (model, budget) combos:")
                print(pred_df[['model_size', 'data_budget', 'params_M', 'D_M', metric]]
                      .sort_values(['D_M', 'params_M']).to_string(index=False))

        popt, r2 = plot_scaling_pair(group, config_label, output_path, unique, pred_df=pred_df,
                                     metric=metric, increasing=increasing, transform=transform)

        if popt is not None:
            fits[config_label] = {'popt': popt, 'r2': r2}

            # Derived metrics from loss fit
            if args.derived and metric == 'train_loss' and not transform:
                derived_name = f"{st}_{ft}_{pt}{filter_tag}{epoch_tag}_derived"
                derived_path = os.path.join(args.output_dir, f'scaling_{derived_name}.png')
                print(f"  Plotting derived metrics...")
                plot_derived(group, popt, config_label, derived_path, unique,
                             derived_metrics=['train_acc', 'val_metric', 'val_roc_auc', 'bkg_rej'])

    # Comparison
    if len(fits) >= 2:
        plot_comparison(fits, os.path.join(args.output_dir, f'scaling_comparison{filter_tag}{metric_tag}{epoch_tag}.png'))

    # Combined plot
    if len(config_groups) > 1 and len(selected) >= 5:
        print(f"\n--- All configs combined ({len(selected)} points) ---")
        plot_scaling_pair(selected, f'All configurations{epoch_label}',
                         os.path.join(args.output_dir, f'scaling_all{filter_tag}{metric_tag}{epoch_tag}.png'),
                         unique_samples=min(UNIQUE_SAMPLES.values()), metric=metric,
                         increasing=increasing, transform=transform)


if __name__ == '__main__':
    main()
