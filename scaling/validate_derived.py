#!/usr/bin/env python
"""
Held-out validation + cross-term comparison for *derived* metrics.

Same workflow as validate_scaling.py, but applied to bounded metrics via the
log(1-m) transform so the Chinchilla form is well-defined:

  y(m) = log(1 - m)   for m in [0, 1)

With A, B > 0 the Chinchilla form y = A N^(-α) + B D^(-β) + E approaches a
finite negative floor E = log(1 - m_max) as N, D → ∞. We therefore relax the
lower bound on E to -∞ for transformed metrics (it remains ≥ 0 for raw loss).

For each metric × config we:
  1. Hold out large + xlarge model points
  2. Fit standard and cross-term on small models (nano - base)
  3. Predict held-out points; report RMSE, bias, R²_val
  4. Plot pred-vs-obs in transformed space, residual vs D in original space

Outputs:
  derived_validation_<metric>_<config>.png      (one per metric × config)
  derived_validation_summary.csv                  (master numbers)
  derived_validation_master.png                  (4-metric × 2-form grid panel)
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

DEFAULT_CSV = '/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/parsed_runs.csv'
OUTPUT_DIR = '/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/plots'

PARAMS_M = {'nano': 0.1, 'micro': 0.2, 'tiny': 0.4, 'small': 1.0,
            'base': 2.14, 'large': 8.5, 'xlarge': 25.0}
SMALL_MODELS = ['nano', 'micro', 'tiny', 'small', 'base']
HELD_OUT = ['large', 'xlarge']

MODEL_COLORS = {
    'nano': '#1f77b4', 'micro': '#ff7f0e', 'tiny': '#2ca02c',
    'small': '#d62728', 'base': '#9467bd', 'large': '#8c564b',
    'xlarge': '#e377c2',
}

# Metric configs: column name → (display, is_loss, transform fn, inverse fn)
# transform fn: raw metric → fit-target y
# is_loss: if True, fit raw value with E ≥ 0; else fit log(1-m) with E free
METRICS = [
    ('train_loss',  'Train CE loss',         True),
    ('train_acc',   'Training accuracy',     False),
    ('val_metric',  'Val. acc / AUC',        False),
    ('val_roc_auc', 'Val. ROC AUC',          False),
]


def chinchilla(X, A, alpha, B, beta, E):
    N, D = X
    return A * np.power(N, -alpha) + B * np.power(D, -beta) + E


def chinchilla_cross(X, A, alpha, B, beta, C, gamma, E):
    N, D = X
    return (A * np.power(N, -alpha)
            + B * np.power(D, -beta)
            + C * np.power(N * D, -gamma)
            + E)


def transform(m, is_loss):
    if is_loss:
        return m
    # clip to avoid log(0)
    m_safe = np.clip(m, 1e-6, 1 - 1e-6)
    return np.log(1 - m_safe)


def fit_form(form_func, N, D, y, p0, bounds):
    popt, pcov = curve_fit(form_func, (N, D), y, p0=p0, bounds=bounds,
                           maxfev=200000)
    y_pred = form_func((N, D), *popt)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return popt, pcov, ss_res, r2


def select_best(df, metric):
    """Best (min train_loss) row per (size, budget, sample, ...) keeps the run.
    Then drop rows where the metric is missing."""
    keys = ['model_size', 'data_budget', 'sample_type', 'feature_type',
            'pair_tag', 'run_type']
    df = df.dropna(subset=['train_loss'])
    df = df.sort_values('train_loss').drop_duplicates(subset=keys, keep='first')
    df = df.dropna(subset=[metric]).copy()
    return df


def evaluate_one(form, train, val, metric_col, is_loss):
    N_tr = train['params_M'].values
    D_tr = train['D_M'].values
    y_tr = transform(train[metric_col].values, is_loss)
    L_min = float(y_tr.min())
    L_max = float(y_tr.max())
    L_pad = (L_max - L_min) * 0.5

    if form == 'standard':
        if is_loss:
            p0 = [1.0, 0.5, 1.0, 0.5, max(L_min, 0.0)]
            bd = ([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, max(L_max, 1e-3)])
        else:
            # E unconstrained for log(1-m): floor is negative
            p0 = [1.0, 0.5, 1.0, 0.5, L_min]
            bd = ([0, 0, 0, 0, L_min - L_pad - 5],
                  [np.inf, 2, np.inf, 2, L_max + L_pad + 5])
        func = chinchilla
    else:  # cross
        if is_loss:
            p0 = [1.0, 0.5, 1.0, 0.5, 0.5, 0.3, max(L_min, 0.0)]
            bd = ([0, 0, 0, 0, 0, 0, 0],
                  [np.inf, 2, np.inf, 2, np.inf, 2, max(L_max, 1e-3)])
        else:
            p0 = [1.0, 0.5, 1.0, 0.5, 0.5, 0.3, L_min]
            bd = ([0, 0, 0, 0, 0, 0, L_min - L_pad - 5],
                  [np.inf, 2, np.inf, 2, np.inf, 2, L_max + L_pad + 5])
        func = chinchilla_cross

    popt, pcov, ssr, r2_fit = fit_form(func, N_tr, D_tr, y_tr, p0, bd)

    N_val = val['params_M'].values
    D_val = val['D_M'].values
    y_val = transform(val[metric_col].values, is_loss)
    y_pred = func((N_val, D_val), *popt)
    res = y_val - y_pred
    rmse = float(np.sqrt(np.mean(res ** 2)))
    bias = float(np.mean(res))
    r2_val = 1 - float(np.sum(res ** 2)) / float(np.sum((y_val - np.mean(y_val)) ** 2))
    return {
        'popt': popt, 'pcov': pcov, 'r2_fit': r2_fit, 'ssr': ssr,
        'rmse': rmse, 'bias': bias, 'r2_val': r2_val,
        'y_val': y_val, 'y_pred': y_pred,
        'func': func,
    }


def per_metric_per_config(df, metric_col, is_loss, sample_type, run_type,
                           output_path):
    sub = select_best(df, metric_col)
    sub = sub[(sub['sample_type'] == sample_type) &
              (sub['run_type'] == run_type)].copy()
    if sub.empty:
        return None
    sub['D_M'] = sub['total_samples_seen'] / 1e6
    train = sub[sub['model_size'].isin(SMALL_MODELS)]
    val = sub[sub['model_size'].isin(HELD_OUT)]
    if len(train) < 5 or len(val) < 2:
        return None

    res_std = evaluate_one('standard', train, val, metric_col, is_loss)
    res_x = evaluate_one('cross', train, val, metric_col, is_loss)

    # Nested F-test on FULL data (small + held-out) — same as significance_tests
    full = pd.concat([train, val])
    Nf = full['params_M'].values
    Df = full['D_M'].values
    yf = transform(full[metric_col].values, is_loss)
    full_std = evaluate_one('standard', full, full[:0].copy() if False else full,
                            metric_col, is_loss)
    full_x = evaluate_one('cross', full, full, metric_col, is_loss)
    n = len(full)
    df1, df2 = 2, n - 7
    if full_x['ssr'] > 0 and df2 > 0:
        F = ((full_std['ssr'] - full_x['ssr']) / df1) / (full_x['ssr'] / df2)
        pF = 1.0 - stats.f.cdf(F, df1, df2)
    else:
        F, pF = np.nan, np.nan

    # ---- 4-panel plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) Pred vs Obs (transformed space)
    ax = axes[0]
    for label, res, marker in [(f'Standard', res_std, 'o'),
                                (f'Cross-term', res_x, 'X')]:
        for model in HELD_OUT:
            s = val[val['model_size'] == model]
            if s.empty: continue
            c = MODEL_COLORS[model]
            y_p = res['func']((s['params_M'].values, s['D_M'].values),
                              *res['popt'])
            y_o = transform(s[metric_col].values, is_loss)
            ax.scatter(y_p, y_o, color=c, s=80, marker=marker,
                       edgecolor='black', linewidth=0.7,
                       label=f"{model} ({label})" if model == HELD_OUT[0] else None)
    all_y = np.concatenate([res_std['y_val'], res_std['y_pred'],
                             res_x['y_val'], res_x['y_pred']])
    pad = (all_y.max() - all_y.min()) * 0.05
    lims = [all_y.min() - pad, all_y.max() + pad]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y = x')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal', 'box')
    if is_loss:
        ax.set_xlabel('Predicted loss'); ax.set_ylabel('Observed loss')
    else:
        ax.set_xlabel(r'Predicted $\log(1-m)$'); ax.set_ylabel(r'Observed $\log(1-m)$')
    title_parts = [f'{sample_type}/{run_type}, {metric_col}']
    title_parts.append(
        f"std: RMSE={res_std['rmse']:.3f}, R²={res_std['r2_val']:+.2f}")
    title_parts.append(
        f"cross: RMSE={res_x['rmse']:.3f}, R²={res_x['r2_val']:+.2f}")
    ax.set_title("\n".join(title_parts), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    # (b) Residuals vs D
    ax = axes[1]
    for model in HELD_OUT:
        s = val[val['model_size'] == model].sort_values('D_M')
        if s.empty: continue
        c = MODEL_COLORS[model]
        y_o = transform(s[metric_col].values, is_loss)
        y_std = res_std['func']((s['params_M'].values, s['D_M'].values),
                                 *res_std['popt'])
        y_x = res_x['func']((s['params_M'].values, s['D_M'].values),
                             *res_x['popt'])
        ax.plot(s['D_M'], y_o - y_std, marker='o', ls='-', color=c, ms=8,
                label=f"{model} std")
        ax.plot(s['D_M'], y_o - y_x, marker='X', ls='--', color=c, ms=10,
                markeredgecolor='black', markeredgewidth=0.7,
                label=f"{model} cross")
    ax.axhline(0, color='black', ls='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Training samples D (M)')
    ax.set_ylabel('Residual (obs − pred)')
    ax.set_title(f'F-test (full data): F(2,{n-7})={F:.2f}, p={pF:.1e}',
                  fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.3)

    plt.suptitle(f'Held-out validation: {metric_col} · {sample_type}/{run_type}',
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)

    return {
        'metric': metric_col, 'sample': sample_type, 'run_type': run_type,
        'n_train': len(train), 'n_val': len(val),
        'std_rmse': res_std['rmse'], 'std_bias': res_std['bias'],
        'std_r2_val': res_std['r2_val'], 'std_r2_fit': res_std['r2_fit'],
        'cross_rmse': res_x['rmse'], 'cross_bias': res_x['bias'],
        'cross_r2_val': res_x['r2_val'], 'cross_r2_fit': res_x['r2_fit'],
        'rmse_drop_pct': 100.0 * (1.0 - res_x['rmse'] / res_std['rmse'])
            if res_std['rmse'] > 0 else 0.0,
        'F': F, 'p_F': pF,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--output-dir', default=OUTPUT_DIR)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} epoch rows from {args.csv}")
    os.makedirs(args.output_dir, exist_ok=True)

    configs = [
        ('Pythia', 'multi'),
        ('Herwig', 'multi'),
        ('Mixed',  'multi'),
        ('Pythia', '1ep'),
    ]
    rows = []
    for metric_col, label, is_loss in METRICS:
        for st, rt in configs:
            out = os.path.join(
                args.output_dir,
                f'derived_validation_{metric_col}_{st}_{rt}.png')
            r = per_metric_per_config(df, metric_col, is_loss,
                                       st, rt, out)
            if r is None:
                continue
            r['display'] = label
            rows.append(r)

    rdf = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'derived_validation_summary.csv')
    rdf.to_csv(csv_path, index=False)

    # Print table
    print("\n" + "=" * 110)
    print(f"{'metric':<14s} {'config':<15s} {'n_tr':>4s} {'n_v':>3s} "
          f"{'RMSE_std':>9s} {'RMSE_x':>8s} {'drop%':>6s} "
          f"{'R²val_std':>10s} {'R²val_x':>8s} {'F':>7s} {'p_F':>9s}")
    print("=" * 110)
    for _, r in rdf.iterrows():
        cfg = f"{r['sample']}/{r['run_type']}"
        ftag = (f"{r['F']:7.2f}" if np.isfinite(r['F']) else f"{'—':>7s}")
        ptag = (f"{r['p_F']:.1e}" if np.isfinite(r['p_F']) else f"{'—':>9s}")
        print(f"{r['metric']:<14s} {cfg:<15s} {int(r['n_train']):>4d} "
              f"{int(r['n_val']):>3d} {r['std_rmse']:>9.4f} "
              f"{r['cross_rmse']:>8.4f} {r['rmse_drop_pct']:>+5.1f}% "
              f"{r['std_r2_val']:>+9.3f} {r['cross_r2_val']:>+7.3f} "
              f"{ftag:>7s} {ptag:>9s}")
    print("=" * 110)
    print(f"\nSaved CSV: {csv_path}")

    # ---- Master grid plot: 4 metrics × 4 configs, RMSE-drop bars ----
    metrics_present = rdf['metric'].unique().tolist()
    configs_present = rdf.apply(lambda r: f"{r['sample']}/{r['run_type']}",
                                 axis=1).unique().tolist()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.18
    x = np.arange(len(configs_present))
    cmap = plt.get_cmap('tab10')
    for i, mc in enumerate(metrics_present):
        rs = rdf[rdf['metric'] == mc]
        vals = []
        for cfg in configs_present:
            sample, rt = cfg.split('/')
            row = rs[(rs['sample'] == sample) & (rs['run_type'] == rt)]
            if row.empty:
                vals.append(0)
            else:
                vals.append(float(row['rmse_drop_pct'].iloc[0]))
        ax.bar(x + (i - 1.5) * width, vals, width=width,
                label=mc, color=cmap(i))
    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(configs_present, rotation=10)
    ax.set_ylabel('Held-out RMSE drop, cross vs standard (%)')
    ax.set_title('Cross-term improvement on held-out (large + xlarge), per metric')
    ax.legend(fontsize=9, loc='best')
    ax.grid(axis='y', alpha=0.3)
    master_path = os.path.join(args.output_dir, 'derived_validation_master.png')
    plt.tight_layout()
    plt.savefig(master_path, dpi=180)
    plt.close(fig)
    print(f"Saved master plot: {master_path}")


if __name__ == '__main__':
    main()
