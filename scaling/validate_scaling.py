#!/usr/bin/env python
"""
Validate scaling law extrapolation by holding out large+xlarge models.

Two improvements over the basic Chinchilla form:

1. Cross-term form: `L = A·N^(-α) + B·D^(-β) + C·(N·D)^(-γ) + E`
   The C·(ND)^(-γ) term captures N-D coupling — getting both bigger together
   helps more than the additive law predicts. This explicitly addresses the
   small positive bias we saw on held-out large models.

2. Shared-E for 1-epoch: the irreducible floor E should be physical and not
   depend on the training regime (1-ep vs multi-epoch). We fix E using the
   multi-epoch fit and refit only the prefactors and exponents for 1-ep.

For each config, we report fit + held-out (large, xlarge) RMSE / bias for both
the standard form and the cross-term form, and the user can compare directly.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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


# ---- Functional forms ----

def chinchilla(X, A, alpha, B, beta, E):
    """Standard Chinchilla: L = A*N^(-α) + B*D^(-β) + E"""
    N, D = X
    return A * np.power(N, -alpha) + B * np.power(D, -beta) + E


def chinchilla_cross(X, A, alpha, B, beta, C, gamma, E):
    """Cross-term Chinchilla: L = A*N^(-α) + B*D^(-β) + C*(N*D)^(-γ) + E

    The C·(ND)^(-γ) term captures coupling: getting both bigger helps more
    than the additive form predicts.
    """
    N, D = X
    return (A * np.power(N, -alpha)
            + B * np.power(D, -beta)
            + C * np.power(N * D, -gamma)
            + E)


def fit_form(form_func, N, D, L, p0, bounds, E_fixed=None):
    """Fit a Chinchilla-like form. If E_fixed is given, keep E pinned to that value."""
    if E_fixed is not None:
        # Remove last param (E) from p0 and bounds, wrap function
        p0 = p0[:-1]
        bounds = (bounds[0][:-1], bounds[1][:-1])

        def wrapped(X, *params):
            return form_func(X, *params, E_fixed)
        popt, _ = curve_fit(wrapped, (N, D), L, p0=p0, bounds=bounds, maxfev=100000)
        popt_full = list(popt) + [E_fixed]
        L_pred = form_func((N, D), *popt_full)
    else:
        popt, _ = curve_fit(form_func, (N, D), L, p0=p0, bounds=bounds, maxfev=100000)
        popt_full = list(popt)
        L_pred = form_func((N, D), *popt_full)

    ss_res = np.sum((L - L_pred) ** 2)
    ss_tot = np.sum((L - np.mean(L)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return np.array(popt_full), r2


def select_best(df, metric='train_loss'):
    keys = ['model_size', 'data_budget', 'sample_type', 'feature_type',
            'pair_tag', 'run_type']
    if 'task_type' in df.columns:
        keys.append('task_type')
    df = df.dropna(subset=[metric])
    return df.sort_values(metric).drop_duplicates(subset=keys, keep='first').copy()


# ---- Validation core ----

def validate_one(form_func, train, val, p0, bounds, E_fixed=None):
    """Fit on train, evaluate on val. Returns (popt, r2_fit, val_metrics)."""
    N_tr = train['params_M'].values
    D_tr = train['D_M'].values
    L_tr = train['train_loss'].values
    popt, r2_fit = fit_form(form_func, N_tr, D_tr, L_tr, p0, bounds, E_fixed=E_fixed)

    N_val = val['params_M'].values
    D_val = val['D_M'].values
    L_val = val['train_loss'].values
    L_pred = form_func((N_val, D_val), *popt)

    res = L_val - L_pred
    rmse = np.sqrt(np.mean(res ** 2))
    bias = np.mean(res)
    mae = np.mean(np.abs(res))
    rel = np.mean(res / L_val)
    r2_val = 1 - np.sum(res ** 2) / np.sum((L_val - np.mean(L_val)) ** 2)
    return {
        'popt': popt, 'r2_fit': r2_fit,
        'rmse': rmse, 'bias': bias, 'mae': mae,
        'mean_rel_err': rel, 'r2_val': r2_val,
        'L_val': L_val, 'L_pred': L_pred, 'val_df': val,
    }


def validate_config(df, sample_type, run_type, output_path, multi_E=None):
    """Run both standard and cross-term validations on one config."""
    sub = df[(df['sample_type'] == sample_type) & (df['run_type'] == run_type)].copy()
    if sub.empty:
        return None
    sub['D_M'] = sub['total_samples_seen'] / 1e6

    train = sub[sub['model_size'].isin(SMALL_MODELS)]
    val = sub[sub['model_size'].isin(HELD_OUT)]
    if len(train) < 5 or len(val) < 3:
        return None

    L_min = train['train_loss'].min()
    L_max = train['train_loss'].max()

    # Standard fit (free E)
    p0_std = [1.0, 0.5, 1.0, 0.5, L_min]
    bd_std = ([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, L_max])
    res_std = validate_one(chinchilla, train, val, p0_std, bd_std)

    # Standard fit with shared E from multi-epoch (only meaningful for 1-ep)
    res_std_sharedE = None
    if multi_E is not None and run_type == '1ep':
        res_std_sharedE = validate_one(
            chinchilla, train, val, p0_std, bd_std, E_fixed=multi_E)

    # Cross-term fit (free E)
    p0_x = [1.0, 0.5, 1.0, 0.5, 0.5, 0.3, L_min]
    bd_x = ([0, 0, 0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, np.inf, 2, L_max])
    res_x = validate_one(chinchilla_cross, train, val, p0_x, bd_x)

    # Cross-term fit with shared E
    res_x_sharedE = None
    if multi_E is not None and run_type == '1ep':
        res_x_sharedE = validate_one(
            chinchilla_cross, train, val, p0_x, bd_x, E_fixed=multi_E)

    print(f"\n=== {sample_type} / {run_type} ===")
    print(f"  Train: {len(train)} small-model points,  Val: {len(val)} held-out")
    A, al, B, be, E = res_std['popt']
    print(f"  Standard:                        L = {A:.3f}N^(-{al:.3f}) + {B:.3f}D^(-{be:.3f}) + {E:.4f}")
    print(f"                                   R²_fit={res_std['r2_fit']:.4f}, RMSE_val={res_std['rmse']:.4f}, "
          f"rel_err={res_std['mean_rel_err']*100:+.2f}%, R²_val={res_std['r2_val']:+.3f}")
    if res_std_sharedE is not None:
        ps = res_std_sharedE['popt']
        print(f"  Standard (shared E={multi_E:.4f}):  L = {ps[0]:.3f}N^(-{ps[1]:.3f}) + {ps[2]:.3f}D^(-{ps[3]:.3f}) + {ps[4]:.4f}*")
        print(f"                                   R²_fit={res_std_sharedE['r2_fit']:.4f}, RMSE_val={res_std_sharedE['rmse']:.4f}, "
              f"rel_err={res_std_sharedE['mean_rel_err']*100:+.2f}%, R²_val={res_std_sharedE['r2_val']:+.3f}")
    Ax, alx, Bx, bex, Cx, gx, Ex = res_x['popt']
    print(f"  Cross-term:                      L = {Ax:.3f}N^(-{alx:.3f}) + {Bx:.3f}D^(-{bex:.3f}) "
          f"+ {Cx:.3f}(ND)^(-{gx:.3f}) + {Ex:.4f}")
    print(f"                                   R²_fit={res_x['r2_fit']:.4f}, RMSE_val={res_x['rmse']:.4f}, "
          f"rel_err={res_x['mean_rel_err']*100:+.2f}%, R²_val={res_x['r2_val']:+.3f}")
    if res_x_sharedE is not None:
        psx = res_x_sharedE['popt']
        print(f"  Cross-term (shared E):           "
              f"L = {psx[0]:.3f}N^(-{psx[1]:.3f}) + {psx[2]:.3f}D^(-{psx[3]:.3f}) + "
              f"{psx[4]:.3f}(ND)^(-{psx[5]:.3f}) + {psx[6]:.4f}*")
        print(f"                                   R²_fit={res_x_sharedE['r2_fit']:.4f}, RMSE_val={res_x_sharedE['rmse']:.4f}, "
              f"rel_err={res_x_sharedE['mean_rel_err']*100:+.2f}%, R²_val={res_x_sharedE['r2_val']:+.3f}")

    # ---- 4-panel comparison plot ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # Pick the best result for each form to display in panels
    use_std = res_std_sharedE if res_std_sharedE is not None else res_std
    use_x = res_x_sharedE if res_x_sharedE is not None else res_x
    label_std = 'Standard' + (' (shared E)' if res_std_sharedE is not None else '')
    label_x = 'Cross-term' + (' (shared E)' if res_x_sharedE is not None else '')

    # (a) Predicted vs Observed: standard
    ax = axes[0, 0]
    for model in HELD_OUT:
        s = val[val['model_size'] == model]
        if s.empty:
            continue
        c = MODEL_COLORS[model]
        L_p = chinchilla((s['params_M'].values, s['D_M'].values), *use_std['popt'])
        ax.scatter(L_p, s['train_loss'], color=c, s=80, edgecolor='black', linewidth=0.7,
                   label=f"{model} ({PARAMS_M[model]:.1f}M)")
    lims = [min(use_std['L_pred'].min(), use_std['L_val'].min()) * 0.95,
            max(use_std['L_pred'].max(), use_std['L_val'].max()) * 1.05]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y = x')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Predicted Loss'); ax.set_ylabel('Observed Loss')
    ax.set_title(f'{label_std}\nRMSE={use_std["rmse"]:.4f}, bias={use_std["bias"]:+.4f}, '
                 f'rel={use_std["mean_rel_err"]*100:+.2f}%')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_aspect('equal', 'box')

    # (b) Predicted vs Observed: cross-term
    ax = axes[0, 1]
    for model in HELD_OUT:
        s = val[val['model_size'] == model]
        if s.empty:
            continue
        c = MODEL_COLORS[model]
        L_p = chinchilla_cross((s['params_M'].values, s['D_M'].values), *use_x['popt'])
        ax.scatter(L_p, s['train_loss'], color=c, s=80, edgecolor='black', linewidth=0.7,
                   label=f"{model} ({PARAMS_M[model]:.1f}M)")
    lims = [min(use_x['L_pred'].min(), use_x['L_val'].min()) * 0.95,
            max(use_x['L_pred'].max(), use_x['L_val'].max()) * 1.05]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='y = x')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Predicted Loss'); ax.set_ylabel('Observed Loss')
    ax.set_title(f'{label_x}\nRMSE={use_x["rmse"]:.4f}, bias={use_x["bias"]:+.4f}, '
                 f'rel={use_x["mean_rel_err"]*100:+.2f}%')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_aspect('equal', 'box')

    # (c) Residuals vs D for both forms
    ax = axes[1, 0]
    for model in HELD_OUT:
        s = val[val['model_size'] == model].sort_values('D_M')
        if s.empty:
            continue
        c = MODEL_COLORS[model]
        L_p_std = chinchilla((s['params_M'].values, s['D_M'].values), *use_std['popt'])
        L_p_x = chinchilla_cross((s['params_M'].values, s['D_M'].values), *use_x['popt'])
        ax.plot(s['D_M'], s['train_loss'].values - L_p_std, marker='o', ls='-',
                color=c, ms=8, label=f"{model} std")
        ax.plot(s['D_M'], s['train_loss'].values - L_p_x, marker='X', ls='--',
                color=c, ms=10, markeredgecolor='black', markeredgewidth=0.7,
                label=f"{model} cross")
    ax.axhline(0, color='black', ls='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Training Samples D (M)')
    ax.set_ylabel('Residual (Observed − Predicted)')
    ax.set_title('Held-out residuals vs D')
    ax.legend(fontsize=8, ncol=2); ax.grid(True, which='both', alpha=0.3)

    # (d) Comparison summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary = [
        f"Config: {sample_type} / {run_type}",
        f"Train (small models): n={len(train)}",
        f"Held-out (large, xlarge): n={len(val)}",
        "",
        f"Standard form: L = A·N^(-α) + B·D^(-β) + E",
        f"  RMSE = {res_std['rmse']:.4f}, bias = {res_std['bias']:+.4f}",
        f"  rel err = {res_std['mean_rel_err']*100:+.2f}%, R²_val = {res_std['r2_val']:+.3f}",
        f"  α = {res_std['popt'][1]:.3f}, β = {res_std['popt'][3]:.3f}, E = {res_std['popt'][4]:.4f}",
    ]
    if res_std_sharedE is not None:
        summary.extend([
            "",
            f"Standard (E pinned to multi-epoch {multi_E:.4f}):",
            f"  RMSE = {res_std_sharedE['rmse']:.4f}, bias = {res_std_sharedE['bias']:+.4f}",
            f"  rel err = {res_std_sharedE['mean_rel_err']*100:+.2f}%, R²_val = {res_std_sharedE['r2_val']:+.3f}",
        ])
    summary.extend([
        "",
        f"Cross-term form: + C·(N·D)^(-γ)",
        f"  RMSE = {res_x['rmse']:.4f}, bias = {res_x['bias']:+.4f}",
        f"  rel err = {res_x['mean_rel_err']*100:+.2f}%, R²_val = {res_x['r2_val']:+.3f}",
        f"  α = {res_x['popt'][1]:.3f}, β = {res_x['popt'][3]:.3f}",
        f"  C = {res_x['popt'][4]:.3f}, γ = {res_x['popt'][5]:.3f}, E = {res_x['popt'][6]:.4f}",
    ])
    if res_x_sharedE is not None:
        summary.extend([
            "",
            f"Cross-term (E pinned):",
            f"  RMSE = {res_x_sharedE['rmse']:.4f}, bias = {res_x_sharedE['bias']:+.4f}",
            f"  rel err = {res_x_sharedE['mean_rel_err']*100:+.2f}%, R²_val = {res_x_sharedE['r2_val']:+.3f}",
        ])
    ax.text(0.02, 0.98, "\n".join(summary), transform=ax.transAxes,
            fontsize=10, va='top', family='monospace',
            bbox=dict(facecolor='#f8f8f8', edgecolor='gray', alpha=0.95))

    plt.suptitle(f'Validation: {sample_type} / {run_type}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"  Saved: {output_path}")
    plt.close(fig)

    return {
        'sample': sample_type, 'run_type': run_type,
        'n_train': len(train), 'n_val': len(val),
        'std': res_std, 'std_sharedE': res_std_sharedE,
        'cross': res_x, 'cross_sharedE': res_x_sharedE,
    }


def main():
    parser = argparse.ArgumentParser(description='Validate scaling law extrapolation')
    parser.add_argument('--csv', default=DEFAULT_CSV)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} epoch entries from {args.csv}")
    selected = select_best(df)
    selected['D_M'] = selected['total_samples_seen'] / 1e6
    print(f"After selecting best-loss per run: {len(selected)} runs")

    os.makedirs(args.output_dir, exist_ok=True)

    # First fit Pythia multi-epoch on ALL points to get the best E estimate
    pythia_multi = selected[(selected['sample_type'] == 'Pythia') &
                             (selected['run_type'] == 'multi')]
    if not pythia_multi.empty:
        N_all = pythia_multi['params_M'].values
        D_all = pythia_multi['D_M'].values
        L_all = pythia_multi['train_loss'].values
        popt_all, _ = fit_form(
            chinchilla, N_all, D_all, L_all,
            p0=[1.0, 0.5, 1.0, 0.5, L_all.min()],
            bounds=([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, L_all.max()]))
        multi_E = float(popt_all[4])
        print(f"\nPythia multi-epoch (full fit): E = {multi_E:.4f} — used as shared floor for 1-ep")
    else:
        multi_E = 0.29

    configs = [
        ('Pythia', 'multi'),
        ('Herwig', 'multi'),
        ('Mixed',  'multi'),
        ('Pythia', '1ep'),
    ]
    results = []
    for st, rt in configs:
        out = os.path.join(args.output_dir, f'validation2_{st}_{rt}.png')
        r = validate_config(selected, st, rt, out, multi_E=multi_E)
        if r is not None:
            results.append(r)

    # Summary table
    print("\n" + "=" * 110)
    print("Summary: scaling law extrapolation to held-out (large, xlarge), n=14 (multi) / 10 (1ep)")
    print("=" * 110)
    header = f"{'Config':22s}  {'form':22s}  {'R²_fit':>7s}  {'RMSE':>7s}  {'bias':>8s}  {'rel%':>7s}  {'R²_val':>7s}"
    print(header)
    print("-" * len(header))
    rows = []
    for r in results:
        cfg = f"{r['sample']}/{r['run_type']}"
        for label, key in [
            ('standard', 'std'),
            ('standard (shared E)', 'std_sharedE'),
            ('cross-term', 'cross'),
            ('cross-term (shared E)', 'cross_sharedE'),
        ]:
            res = r[key]
            if res is None:
                continue
            line = (f"{cfg:22s}  {label:22s}  "
                    f"{res['r2_fit']:7.4f}  {res['rmse']:7.4f}  "
                    f"{res['bias']:+8.4f}  {res['mean_rel_err']*100:+7.2f}  "
                    f"{res['r2_val']:+7.3f}")
            print(line)
            rows.append({
                'config': cfg, 'form': label,
                'R2_fit': res['r2_fit'], 'RMSE': res['rmse'],
                'bias': res['bias'], 'rel_err_pct': res['mean_rel_err'] * 100,
                'R2_val': res['r2_val'],
            })
    print("=" * 110)

    rdf = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, 'validation2_summary.csv')
    rdf.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path}")


if __name__ == '__main__':
    main()
