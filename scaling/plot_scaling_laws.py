#!/usr/bin/env python
"""
Plot Chinchilla scaling laws for Particle Transformer on JetClass.

Fits the standard Chinchilla form using cross-entropy loss:
    L(N, D) = A * N^(-alpha) + B * D^(-beta) + E

where N = model parameters, D = total training samples seen.

Supports multiple study dimensions:
    - Model size:     nano, micro, tiny, small, base, large, xlarge
    - Data budget:    5M .. 500M
    - Sample type:    Pythia, Herwig, Mixed (Pythia+Herwig = 200M unique)
    - Feature type:   kin (7 features), kinpid (13), full (17)
    - Pairwise:       pair (with pairwise attention bias), nopair (without)

Data budget clarification:
    Pythia/Herwig each have 100M unique training samples. Mixed = 200M.
    For budgets exceeding the unique sample count, data is reused across epochs.
    D counts total samples *seen* (with repetition).

Loss source priority:
    1. v2 SLURM logs (structured SCALING_RUN_CONFIG / SCALING_RUN_RESULT blocks)
    2. TensorBoard logs (Loss/eval epoch) -> best validation cross-entropy loss
    3. Weaver text logs (Train AvgLoss) -> best training cross-entropy loss
    4. v1 SLURM logs (Model size: / Data budget: headers)

Usage:
    python plot_scaling_laws.py [--results-dir DIR] [--output-dir DIR]
    python plot_scaling_laws.py --from-slurm /path/to/slurm_logs/slurm-*.out
    python plot_scaling_laws.py --filter sample_type=Mixed feature_type=full
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ---- Constants ----

RESULTS_DIR_V1 = '/pscratch/sd/s/sqian/part_training_output/scaling_study'
RESULTS_DIR_V2 = '/pscratch/sd/s/sqian/part_training_output/scaling_study_v2'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_SIZES = ['nano', 'micro', 'tiny', 'small', 'base', 'large', 'xlarge']
DATA_BUDGETS = ['5M', '10M', '25M', '50M', '100M', '250M', '500M']
SAMPLE_TYPES = ['Pythia', 'Herwig', 'Mixed']
FEATURE_TYPES = ['kin', 'kinpid', 'full']
PAIR_TAGS = ['pair', 'nopair']

# Approximate parameter counts (millions)
PARAMS_M = {
    'nano': 0.1, 'micro': 0.2, 'tiny': 0.4,
    'small': 1.0, 'base': 2.14, 'large': 8.5, 'xlarge': 25.0,
}

# Data budget -> total training samples seen (= samples_per_epoch * 4 GPUs * num_epochs)
BUDGET_TO_TOTAL_SAMPLES = {
    '5M':    5_120_000,
    '10M':  10_240_000,
    '25M':  25_600_000,
    '50M':  51_200_000,
    '100M': 102_400_000,
    '250M': 256_000_000,
    '500M': 512_000_000,
}

# Unique training samples per sample type
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
    """
    Fit the Chinchilla scaling law to observed (N, D, L) data.
    Returns (popt, r_squared) or (None, None) on failure.
    popt = [A, alpha, B, beta, E]
    """
    if len(L) < 5:
        print(f"  Warning: only {len(L)} data points, need >= 5 for a reliable fit.")
        return None, None

    p0 = [1.0, 0.5, 1.0, 0.5, np.min(L)]
    try:
        popt, pcov = curve_fit(
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


# ---- Data collection ----

def _parse_v2_slurm_log(filepath):
    """
    Parse a v2 SLURM log with structured SCALING_RUN_CONFIG/RESULT blocks.
    Returns a dict or None.
    """
    try:
        with open(filepath, 'r') as f:
            text = f.read()
    except Exception:
        return None

    # Parse config block
    config_match = re.search(
        r'===== SCALING_RUN_CONFIG =====\n(.*?)\n===== END_CONFIG =====',
        text, re.DOTALL)
    if not config_match:
        return None

    config = {}
    for line in config_match.group(1).strip().split('\n'):
        if ':' in line:
            k, v = line.split(':', 1)
            config[k.strip()] = v.strip()

    model = config.get('model_size')
    budget = config.get('data_budget')
    sample_type = config.get('sample_type', 'Pythia')
    feature_type = config.get('feature_type', 'full')
    pair_features = config.get('pair_features', '1')
    pair_tag = 'pair' if pair_features == '1' else 'nopair'

    if not model or not budget:
        return None

    # Parse result block for best_train_loss
    best_loss = None
    result_match = re.search(
        r'===== SCALING_RUN_RESULT =====\n(.*?)\n===== END_RESULT =====',
        text, re.DOTALL)
    if result_match:
        for line in result_match.group(1).strip().split('\n'):
            if line.startswith('best_train_loss:'):
                val = line.split(':', 1)[1].strip()
                if val != 'N/A':
                    best_loss = float(val)

    # Also try parsing Train AvgLoss from the log body (more reliable)
    train_losses = [float(m.group(1)) for m in re.finditer(r'Train AvgLoss: ([\d.]+)', text)]

    # Also follow weaver log reference
    weaver_log_match = re.search(r"'log',\s*'([^']+\.log\.\d+)'", text)
    if weaver_log_match:
        weaver_path = weaver_log_match.group(1)
        if os.path.isfile(weaver_path):
            try:
                with open(weaver_path, 'r') as wf:
                    for line in wf:
                        m = re.search(r'Train AvgLoss: ([\d.]+)', line)
                        if m:
                            train_losses.append(float(m.group(1)))
            except Exception:
                pass

    if train_losses:
        best_loss = min(train_losses)
    if best_loss is None:
        return None

    total_samples = BUDGET_TO_TOTAL_SAMPLES.get(budget, float(budget.replace('M', '')) * 1e6)
    return {
        'model': model,
        'budget_label': budget,
        'sample_type': sample_type,
        'feature_type': feature_type,
        'pair_tag': pair_tag,
        'params_M': PARAMS_M.get(model, 1.0),
        'total_samples': total_samples,
        'best_val_loss': best_loss,
        'source': 'v2_slurm',
    }


def _parse_v1_slurm_log(filepath):
    """
    Parse a v1 SLURM log (old format with 'Model size:' / 'Data budget:' headers).
    Returns a dict or None. Assumes Pythia/full/pair defaults.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception:
        return None

    train_loss_re = re.compile(r'Train AvgLoss: ([\d.]+)')
    weaver_log_re = re.compile(r"'log',\s*'([^']+\.log\.\d+)'")

    model = budget = weaver_log_path = None
    losses = []
    for line in lines:
        if 'model_size:' in line or 'Model size:' in line:
            model = line.strip().split()[-1]
        if 'data_budget:' in line or 'Data budget:' in line:
            budget = line.strip().split()[-1]
        m = train_loss_re.search(line)
        if m:
            losses.append(float(m.group(1)))
        m = weaver_log_re.search(line)
        if m:
            weaver_log_path = m.group(1)

    if not losses and weaver_log_path and os.path.isfile(weaver_log_path):
        try:
            with open(weaver_log_path, 'r') as f:
                for line in f:
                    m = train_loss_re.search(line)
                    if m:
                        losses.append(float(m.group(1)))
        except Exception:
            pass

    if not model or not budget or not losses:
        return None

    total_samples = BUDGET_TO_TOTAL_SAMPLES.get(budget, float(budget.replace('M', '')) * 1e6)
    return {
        'model': model,
        'budget_label': budget,
        'sample_type': 'Pythia',
        'feature_type': 'full',
        'pair_tag': 'pair',
        'params_M': PARAMS_M.get(model, 1.0),
        'total_samples': total_samples,
        'best_val_loss': min(losses),
        'source': 'v1_slurm',
    }


def collect_from_slurm_logs(slurm_files):
    """Parse a list of SLURM log files (auto-detects v1 vs v2 format)."""
    results = []
    for f in slurm_files:
        entry = _parse_v2_slurm_log(f)
        if entry is None:
            entry = _parse_v1_slurm_log(f)
        if entry is not None:
            results.append(entry)
    return results


def _find_tb_run_dirs(results_dir, run_name):
    """Find TensorBoard event directories for a given run_name."""
    candidates = []

    tb_dir = os.path.join(results_dir, 'tensorboard', run_name)
    if os.path.isdir(tb_dir):
        candidates.append(tb_dir)

    # Weaver local runs/ directory (nested path structure)
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = os.path.join(repo_dir, 'runs')
    if os.path.isdir(runs_dir):
        for ts_dir in os.listdir(runs_dir):
            nested = glob.glob(os.path.join(runs_dir, ts_dir, '**', run_name), recursive=True)
            for d in nested:
                if os.path.isdir(d) and glob.glob(os.path.join(d, 'events.out.tfevents*')):
                    candidates.append(d)

    return candidates


def collect_from_tensorboard(results_dir, sample_type='Pythia', feature_type='full', pair_tag='pair'):
    """Parse TensorBoard event files for validation loss."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not installed; skipping TensorBoard parsing.")
        return []

    results = []
    for model_size in MODEL_SIZES:
        for budget_label, total_samples in BUDGET_TO_TOTAL_SAMPLES.items():
            # v2 run names include all dimensions; v1 is just model_budget
            run_names = [
                f'{model_size}_{budget_label}_{sample_type}_{feature_type}_{pair_tag}',
                f'{model_size}_{budget_label}',
            ]

            best_val_loss = None
            for run_name in run_names:
                for run_dir in _find_tb_run_dirs(results_dir, run_name):
                    try:
                        ea = EventAccumulator(run_dir)
                        ea.Reload()
                        available = ea.Tags().get('scalars', [])
                        if 'Loss/eval (epoch)' in available:
                            events = ea.Scalars('Loss/eval (epoch)')
                            if events:
                                run_best = min(e.value for e in events)
                                if best_val_loss is None or run_best < best_val_loss:
                                    best_val_loss = run_best
                    except Exception:
                        continue

            if best_val_loss is not None:
                results.append({
                    'model': model_size,
                    'budget_label': budget_label,
                    'sample_type': sample_type,
                    'feature_type': feature_type,
                    'pair_tag': pair_tag,
                    'params_M': PARAMS_M.get(model_size, 1.0),
                    'total_samples': total_samples,
                    'best_val_loss': best_val_loss,
                    'source': 'tensorboard',
                })

    return results


def collect_from_text_logs(results_dir, sample_type='Pythia', feature_type='full', pair_tag='pair'):
    """Parse weaver text logs (.log.000 files) for training loss."""
    log_dir = os.path.join(results_dir, 'logs')
    if not os.path.isdir(log_dir):
        return []

    train_loss_re = re.compile(r'Train AvgLoss: ([\d.]+)')
    results = []

    for model_size in MODEL_SIZES:
        for budget_label, total_samples in BUDGET_TO_TOTAL_SAMPLES.items():
            # Try both v2 and v1 naming conventions
            run_names = [
                f'{model_size}_{budget_label}_{sample_type}_{feature_type}_{pair_tag}',
                f'{model_size}_{budget_label}',
            ]

            losses = []
            for run_name in run_names:
                log_files = sorted(glob.glob(os.path.join(log_dir, f'{run_name}_*.log.000')))
                for lf in log_files:
                    try:
                        with open(lf, 'r') as fh:
                            for line in fh:
                                m = train_loss_re.search(line)
                                if m:
                                    losses.append(float(m.group(1)))
                    except Exception:
                        continue

            if losses:
                results.append({
                    'model': model_size,
                    'budget_label': budget_label,
                    'sample_type': sample_type,
                    'feature_type': feature_type,
                    'pair_tag': pair_tag,
                    'params_M': PARAMS_M.get(model_size, 1.0),
                    'total_samples': total_samples,
                    'best_val_loss': min(losses),
                    'source': 'text_log (train loss)',
                })

    return results


def collect_from_run_dirs(results_dir):
    """
    Scan v2 timestamped run directories under {results_dir}/runs/.

    Each run dir has the structure:
        {results_dir}/runs/{RUN_NAME}_{TIMESTAMP}/
            config.txt          key: value pairs
            logs/*.log.000      weaver rank-0 text log
            tensorboard/        TensorBoard event files
            training_complete.flag  (if finished)

    Tries TensorBoard for val loss first, falls back to text log train loss.
    Returns list of result dicts.
    """
    runs_base = os.path.join(results_dir, 'runs')
    if not os.path.isdir(runs_base):
        return []

    train_loss_re = re.compile(r'Train AvgLoss: ([\d.]+)')
    results = []

    for run_dir_name in sorted(os.listdir(runs_base)):
        run_dir = os.path.join(runs_base, run_dir_name)
        config_file = os.path.join(run_dir, 'config.txt')
        if not os.path.isfile(config_file):
            continue

        # Parse config.txt
        config = {}
        with open(config_file, 'r') as f:
            for line in f:
                if ':' in line:
                    k, v = line.split(':', 1)
                    config[k.strip()] = v.strip()

        model = config.get('model_size')
        budget = config.get('data_budget')
        sample_type = config.get('sample_type', 'Pythia')
        feature_type = config.get('feature_type', 'full')
        pair_tag = config.get('pair_tag', 'pair')

        if not model or not budget:
            continue

        total_samples = BUDGET_TO_TOTAL_SAMPLES.get(
            budget, float(budget.replace('M', '')) * 1e6)

        best_loss = None
        source = None

        # Try TensorBoard (val loss) first
        tb_dir = os.path.join(run_dir, 'tensorboard')
        if os.path.isdir(tb_dir) and glob.glob(os.path.join(tb_dir, 'events.out.tfevents*')):
            try:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                ea = EventAccumulator(tb_dir)
                ea.Reload()
                available = ea.Tags().get('scalars', [])
                if 'Loss/eval (epoch)' in available:
                    events = ea.Scalars('Loss/eval (epoch)')
                    if events:
                        best_loss = min(e.value for e in events)
                        source = 'run_dir (val loss)'
            except Exception:
                pass

        # Fallback: text logs (train loss)
        if best_loss is None:
            log_files = sorted(glob.glob(os.path.join(run_dir, 'logs', '*.log.000')))
            losses = []
            for lf in log_files:
                try:
                    with open(lf, 'r') as fh:
                        for line in fh:
                            m = train_loss_re.search(line)
                            if m:
                                losses.append(float(m.group(1)))
                except Exception:
                    continue
            if losses:
                best_loss = min(losses)
                source = 'run_dir (train loss)'

        if best_loss is not None:
            results.append({
                'model': model,
                'budget_label': budget,
                'sample_type': sample_type,
                'feature_type': feature_type,
                'pair_tag': pair_tag,
                'params_M': PARAMS_M.get(model, 1.0),
                'total_samples': total_samples,
                'best_val_loss': best_loss,
                'source': source,
            })

    return results


def collect_all(results_dirs, from_slurm=None):
    """
    Collect results from all available sources across v1 and v2 directories.
    Returns a DataFrame with columns: model, budget_label, sample_type,
    feature_type, pair_tag, params_M, total_samples, best_val_loss, source.
    """
    all_results = []

    if from_slurm:
        slurm_files = []
        for pattern in from_slurm:
            slurm_files.extend(sorted(glob.glob(pattern)))
        results = collect_from_slurm_logs(slurm_files)
        print(f"Parsed {len(results)} runs from {len(slurm_files)} SLURM log files")
        all_results.extend(results)
    else:
        for rdir in results_dirs:
            if not os.path.isdir(rdir):
                continue
            print(f"\nSearching {rdir}...")

            # Primary: scan timestamped run directories (v2 layout)
            run_results = collect_from_run_dirs(rdir)
            if run_results:
                print(f"  Run dirs: {len(run_results)} runs")
                all_results.extend(run_results)

            # Fallback: TensorBoard in flat layout or local runs/
            for st in SAMPLE_TYPES:
                for ft in FEATURE_TYPES:
                    for pt in PAIR_TAGS:
                        tb_results = collect_from_tensorboard(rdir, st, ft, pt)
                        if tb_results:
                            print(f"  TB: {len(tb_results)} runs ({st}/{ft}/{pt})")
                            all_results.extend(tb_results)

            # Fallback: text logs in flat layout (v1)
            for st in SAMPLE_TYPES:
                for ft in FEATURE_TYPES:
                    for pt in PAIR_TAGS:
                        txt_results = collect_from_text_logs(rdir, st, ft, pt)
                        if txt_results:
                            print(f"  Text: {len(txt_results)} runs ({st}/{ft}/{pt})")
                            all_results.extend(txt_results)

            # Fallback: SLURM logs
            slurm_dir = os.path.join(rdir, 'slurm_logs')
            if os.path.isdir(slurm_dir):
                slurm_files = sorted(glob.glob(os.path.join(slurm_dir, 'slurm-*.out')))
                if slurm_files:
                    slurm_results = collect_from_slurm_logs(slurm_files)
                    if slurm_results:
                        print(f"  SLURM: {len(slurm_results)} runs")
                        all_results.extend(slurm_results)

    if not all_results:
        return pd.DataFrame()

    df = pd.DataFrame(all_results)

    # De-duplicate: keep lowest loss per unique (model, budget, sample_type, feature_type, pair_tag)
    dedup_keys = ['model', 'budget_label', 'sample_type', 'feature_type', 'pair_tag']
    df = df.sort_values('best_val_loss').drop_duplicates(
        subset=dedup_keys, keep='first'
    ).reset_index(drop=True)

    df['total_samples_M'] = df['total_samples'] / 1e6
    return df


# ---- Plotting ----

MODEL_COLORS = {
    'nano': '#1f77b4', 'micro': '#ff7f0e', 'tiny': '#2ca02c',
    'small': '#d62728', 'base': '#9467bd', 'large': '#8c564b',
    'xlarge': '#e377c2',
}


def plot_scaling_pair(df, title_suffix, output_path, unique_samples):
    """
    Two-panel scaling law plot for a single configuration slice:
        Left:  Loss vs Data (one curve per model size, fit overlaid)
        Right: Loss vs Model Parameters (one curve per data budget, fit overlaid)
    """
    N_data = df['params_M'].values
    D_data = df['total_samples_M'].values
    L_data = df['best_val_loss'].values

    popt, r2 = fit_chinchilla(N_data, D_data, L_data)
    if popt is not None:
        A, alpha, B, beta, E = popt
        print(f"  Fit: L = {A:.4f}*N^(-{alpha:.4f}) + {B:.4f}*D^(-{beta:.4f}) + {E:.6f}  [R²={r2:.4f}]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    budget_cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(BUDGET_TO_TOTAL_SAMPLES)))
    unique_M = unique_samples / 1e6

    # ---- Left: Loss vs Data Budget ----
    for model in MODEL_SIZES:
        sub = df[df['model'] == model].sort_values('total_samples_M')
        if sub.empty:
            continue
        n_params = sub['params_M'].iloc[0]
        c = MODEL_COLORS.get(model, 'gray')
        reuse_mask = sub['total_samples_M'] > unique_M

        no_reuse = sub[~reuse_mask]
        with_reuse = sub[reuse_mask]
        label = f"{model} ({n_params:.2f}M)"
        if not no_reuse.empty:
            ax1.plot(no_reuse['total_samples_M'], no_reuse['best_val_loss'],
                     marker='o', ls='', color=c, label=label, ms=7)
        if not with_reuse.empty:
            ax1.plot(with_reuse['total_samples_M'], with_reuse['best_val_loss'],
                     marker='o', ls='', color=c, ms=7,
                     markerfacecolor='none', markeredgewidth=1.5,
                     label=f"{model} (reused)" if no_reuse.empty else None)

        if popt is not None:
            x_fit = np.logspace(np.log10(max(sub['total_samples_M'].min() * 0.8, 1)),
                                np.log10(600), 50)
            ax1.plot(x_fit, chinchilla_loss((n_params, x_fit), *popt),
                     color=c, alpha=0.6, ls='--')

    ax1.axvline(x=unique_M, color='gray', ls=':', alpha=0.5,
                label=f'unique data ({unique_M:.0f}M)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Total Training Samples D (M)', fontsize=13)
    ax1.set_ylabel('Cross-Entropy Loss', fontsize=13)
    ax1.set_title(f'Loss vs Data Budget\n{title_suffix}')
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
    budgets_sorted = sorted(BUDGET_TO_TOTAL_SAMPLES.items(), key=lambda x: x[1])
    for i, (blabel, tsamples) in enumerate(budgets_sorted):
        sub = df[df['budget_label'] == blabel].sort_values('params_M')
        if sub.empty:
            continue
        c = budget_cmap[i % len(budget_cmap)]
        is_reused = tsamples > unique_samples
        label = f"D={blabel}" + (" (reused)" if is_reused else "")
        ax2.plot(sub['params_M'], sub['best_val_loss'],
                 marker='s', ls='', color=c, label=label, ms=7,
                 markerfacecolor='none' if is_reused else c,
                 markeredgewidth=1.5 if is_reused else 1.0)

        if popt is not None:
            x_fit = np.logspace(np.log10(0.05), np.log10(30.0), 50)
            ax2.plot(x_fit, chinchilla_loss((x_fit, tsamples / 1e6), *popt),
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
    """
    Compare scaling law fits across configurations.
    Plots fitted exponents (alpha, beta) and irreducible loss (E) side by side.
    """
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

    # Annotate R² on the rightmost panel
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
    parser = argparse.ArgumentParser(description='Plot Chinchilla scaling laws (loss-based)')
    parser.add_argument('--results-dir', type=str, nargs='+',
                        default=[RESULTS_DIR_V2, RESULTS_DIR_V1],
                        help='Scaling study directories to search (searched in order)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for output plots (default: scaling/)')
    parser.add_argument('--from-slurm', nargs='+', default=None,
                        help='Parse SLURM log files directly (glob patterns)')
    parser.add_argument('--filter', nargs='+', default=None,
                        help='Filter runs: key=value pairs (e.g., sample_type=Mixed feature_type=full)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = OUTPUT_DIR
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Collect all results ----
    df = collect_all(args.results_dir, from_slurm=args.from_slurm)

    if df.empty:
        print("No results found.")
        return

    # Apply filters
    if args.filter:
        for filt in args.filter:
            key, val = filt.split('=', 1)
            df = df[df[key] == val]
            print(f"Filter: {key}={val} -> {len(df)} runs remaining")

    print(f"\nTotal: {len(df)} unique runs")
    print("-" * 80)
    display_cols = ['model', 'budget_label', 'sample_type', 'feature_type',
                    'pair_tag', 'params_M', 'total_samples_M', 'best_val_loss', 'source']
    display_cols = [c for c in display_cols if c in df.columns]
    print(df.sort_values(['sample_type', 'feature_type', 'pair_tag', 'total_samples_M', 'params_M']
                         )[display_cols].to_string(index=False))
    print("-" * 80)

    # ---- Generate plots per configuration slice ----
    fits = {}
    config_groups = df.groupby(['sample_type', 'feature_type', 'pair_tag'])

    for (st, ft, pt), group in config_groups:
        if len(group) < 3:
            print(f"\nSkipping {st}/{ft}/{pt}: only {len(group)} runs")
            continue

        unique = UNIQUE_SAMPLES.get(st, 100_000_000)
        config_label = f"{st} / {ft} / {pt}"
        safe_name = f"{st}_{ft}_{pt}"
        output_path = os.path.join(args.output_dir, f'scaling_laws_{safe_name}.png')

        print(f"\n--- {config_label} ({len(group)} runs) ---")
        popt, r2 = plot_scaling_pair(group, config_label, output_path, unique)

        if popt is not None:
            fits[config_label] = {'popt': popt, 'r2': r2}

    # ---- Comparison plot across configs ----
    if len(fits) >= 2:
        plot_comparison(fits, os.path.join(args.output_dir, 'scaling_laws_comparison.png'))

    # ---- Also generate a combined "all configs" plot if there are multiple ----
    if len(config_groups) > 1 and len(df) >= 5:
        print(f"\n--- All configs combined ({len(df)} runs) ---")
        plot_scaling_pair(df, 'All configurations',
                         os.path.join(args.output_dir, 'scaling_laws_all.png'),
                         unique_samples=min(UNIQUE_SAMPLES.values()))


if __name__ == '__main__':
    main()
