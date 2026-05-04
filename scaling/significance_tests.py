#!/usr/bin/env python
"""
Statistical significance tests for the Chinchilla scaling-law fits.

For each (sample, run_type) config we fit the full data (no hold-out) with both:
  standard:   L = A·N^(-α) + B·D^(-β) + E                      (5 params)
  cross-term: L = A·N^(-α) + B·D^(-β) + C·(N·D)^(-γ) + E       (7 params)

For each form we report:
  - Wald t-stat and p-value per parameter (t = popt / sqrt(diag(pcov)),
    df = n - p, two-sided).
  - σ_param = sqrt(diag(pcov)) for context.

Then a nested F-test compares standard ⊂ cross-term:
  F = ((SSR_red - SSR_full) / (p_full - p_red)) / (SSR_full / (n - p_full))
  H0: extra terms (C, γ) carry no information — i.e. C=0.
  Reject if p-F < 0.05.

We also print AIC / BIC for the two forms (lower = better) for reference.
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats

DEFAULT_CSV = '/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/parsed_runs.csv'
OUTPUT_DIR = '/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/plots'

PARAMS_M = {'nano': 0.1, 'micro': 0.2, 'tiny': 0.4, 'small': 1.0,
            'base': 2.14, 'large': 8.5, 'xlarge': 25.0}


def chinchilla(X, A, alpha, B, beta, E):
    N, D = X
    return A * np.power(N, -alpha) + B * np.power(D, -beta) + E


def chinchilla_cross(X, A, alpha, B, beta, C, gamma, E):
    N, D = X
    return (A * np.power(N, -alpha)
            + B * np.power(D, -beta)
            + C * np.power(N * D, -gamma)
            + E)


def chinchilla_reduced(X, B, beta, C, gamma, E):
    """Reduced cross-term form (drop pure-N term):
       L = B*D^(-β) + C*(N*D)^(-γ) + E   (5 params, same count as standard)
    """
    N, D = X
    return (B * np.power(D, -beta)
            + C * np.power(N * D, -gamma)
            + E)


def fit_with_cov(form, N, D, L, p0, bounds):
    popt, pcov = curve_fit(form, (N, D), L, p0=p0, bounds=bounds, maxfev=200000)
    L_pred = form((N, D), *popt)
    ssr = float(np.sum((L - L_pred) ** 2))
    return popt, pcov, ssr, L_pred


def select_best(df, metric='train_loss'):
    keys = ['model_size', 'data_budget', 'sample_type', 'feature_type',
            'pair_tag', 'run_type']
    if 'task_type' in df.columns:
        keys.append('task_type')
    df = df.dropna(subset=[metric])
    return df.sort_values(metric).drop_duplicates(subset=keys, keep='first').copy()


def t_table(popt, pcov, names, n):
    se = np.sqrt(np.clip(np.diag(pcov), 0, None))
    df = n - len(popt)
    rows = []
    for nm, est, s in zip(names, popt, se):
        if s == 0 or not np.isfinite(s):
            t = np.inf if est != 0 else 0.0
            p = 0.0
        else:
            t = est / s
            p = 2.0 * (1.0 - stats.t.cdf(abs(t), df))
        rows.append({'param': nm, 'estimate': est, 'se': s, 't': t, 'p': p, 'df': df})
    return rows


def aic_bic(ssr, n, k):
    """Gaussian AIC/BIC up to constant. Lower = better."""
    aic = n * np.log(ssr / n) + 2 * k
    bic = n * np.log(ssr / n) + k * np.log(n)
    return aic, bic


def fmt_p(p):
    if p < 1e-4:
        return f"<1e-4"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.4f}"


def stars(p):
    if p < 1e-3: return '***'
    if p < 1e-2: return '**'
    if p < 5e-2: return '*'
    if p < 1e-1: return '.'
    return ''


def analyze_config(df, sample_type, run_type, fout):
    sub = df[(df['sample_type'] == sample_type) &
             (df['run_type'] == run_type)].copy()
    if len(sub) < 8:
        print(f"\n[skip] {sample_type}/{run_type}: only {len(sub)} points")
        return None
    sub['D_M'] = sub['total_samples_seen'] / 1e6
    N = sub['params_M'].values
    D = sub['D_M'].values
    L = sub['train_loss'].values
    n = len(L)
    L_min = float(L.min()); L_max = float(L.max())

    # Standard fit
    p0_s = [1.0, 0.5, 1.0, 0.5, L_min]
    bd_s = ([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, L_max])
    popt_s, pcov_s, ssr_s, _ = fit_with_cov(chinchilla, N, D, L, p0_s, bd_s)
    rows_s = t_table(popt_s, pcov_s, ['A', 'alpha', 'B', 'beta', 'E'], n)

    # Cross-term fit
    p0_x = [1.0, 0.5, 1.0, 0.5, 0.5, 0.3, L_min]
    bd_x = ([0, 0, 0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, np.inf, 2, L_max])
    popt_x, pcov_x, ssr_x, _ = fit_with_cov(
        chinchilla_cross, N, D, L, p0_x, bd_x)
    rows_x = t_table(popt_x, pcov_x,
                     ['A', 'alpha', 'B', 'beta', 'C', 'gamma', 'E'], n)

    # Reduced fit (drop pure-N term)
    p0_r = [1.0, 0.5, 0.5, 0.3, L_min]
    bd_r = ([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, L_max])
    popt_r, pcov_r, ssr_r, _ = fit_with_cov(
        chinchilla_reduced, N, D, L, p0_r, bd_r)
    rows_r = t_table(popt_r, pcov_r,
                     ['B', 'beta', 'C', 'gamma', 'E'], n)

    # Nested-form comparison via PARAMETRIC BOOTSTRAP LRT.
    # The naive F-test is invalid here: under H0:C=0 (or H0:A=0), the paired
    # exponent (γ or α) is unidentified and C/A sit on a non-negativity
    # bound, so the asymptotic F distribution does not apply. Instead we
    # simulate the null distribution of the LR statistic directly:
    #   LR = n * log(SSR_null / SSR_alt)
    # by drawing y_b = chinchilla(N,D, *popt_null) + N(0, σ̂_null), refitting
    # both forms on the synthetic data, and recording LR_b. The bootstrap
    # p-value is the empirical fraction of LR_b ≥ LR_obs.
    k_s, k_x, k_r = 5, 7, 5
    aic_s, bic_s = aic_bic(ssr_s, n, k_s)
    aic_x, bic_x = aic_bic(ssr_x, n, k_x)
    aic_r, bic_r = aic_bic(ssr_r, n, k_r)

    LR_sx = n * np.log(ssr_s / ssr_x) if ssr_x > 0 else np.nan
    LR_rx = n * np.log(ssr_r / ssr_x) if ssr_x > 0 else np.nan

    F = pF = F_rx = pF_rx = np.nan
    if ssr_x > 0 and n > 7:
        # Residual std of the standard fit (used as null for both tests since
        # cross-term and reduced both nest inside the standard via different
        # parameter constraints; using a common null fixes the noise scale.)
        sigma_s = float(np.sqrt(ssr_s / max(1, n - k_s)))
        sigma_r = float(np.sqrt(ssr_r / max(1, n - k_r)))
        rng = np.random.default_rng(20260512)
        B = 1000
        boot_LR_sx = np.empty(B)
        boot_LR_rx = np.empty(B)
        for b in range(B):
            # std-as-null bootstrap for std⊂cross
            y_b = chinchilla((N, D), *popt_s) + rng.normal(0.0, sigma_s, size=n)
            try:
                ps_b, _ = curve_fit(chinchilla, (N, D), y_b, p0=p0_s, bounds=bd_s, maxfev=20000)
                px_b, _ = curve_fit(chinchilla_cross, (N, D), y_b, p0=p0_x, bounds=bd_x, maxfev=20000)
                ssr_s_b = float(np.sum((y_b - chinchilla((N, D), *ps_b))**2))
                ssr_x_b = float(np.sum((y_b - chinchilla_cross((N, D), *px_b))**2))
                boot_LR_sx[b] = n * np.log(ssr_s_b / ssr_x_b) if ssr_x_b > 0 else 0.0
            except Exception:
                boot_LR_sx[b] = 0.0
            # reduced-as-null bootstrap for reduced⊂cross
            y_b = chinchilla_reduced((N, D), *popt_r) + rng.normal(0.0, sigma_r, size=n)
            try:
                pr_b, _ = curve_fit(chinchilla_reduced, (N, D), y_b, p0=p0_r, bounds=bd_r, maxfev=20000)
                px_b, _ = curve_fit(chinchilla_cross, (N, D), y_b, p0=p0_x, bounds=bd_x, maxfev=20000)
                ssr_r_b = float(np.sum((y_b - chinchilla_reduced((N, D), *pr_b))**2))
                ssr_x_b = float(np.sum((y_b - chinchilla_cross((N, D), *px_b))**2))
                boot_LR_rx[b] = n * np.log(ssr_r_b / ssr_x_b) if ssr_x_b > 0 else 0.0
            except Exception:
                boot_LR_rx[b] = 0.0
        # Bootstrap p-values (with +1 / B+1 correction so we never report exactly 0).
        pF = float((np.sum(boot_LR_sx >= LR_sx) + 1) / (B + 1))
        pF_rx = float((np.sum(boot_LR_rx >= LR_rx) + 1) / (B + 1))
        # Report LR statistics in F's slot for printing compatibility.
        F = float(LR_sx)
        F_rx = float(LR_rx)

    # Bound-pinned warning: if a param sits exactly on a bound, t-stat is meaningless.
    pinned_x = []
    for i, (val, nm) in enumerate(zip(popt_x,
                                       ['A', 'alpha', 'B', 'beta', 'C', 'gamma', 'E'])):
        lo = bd_x[0][i]; hi = bd_x[1][i]
        if (lo != -np.inf and abs(val - lo) < 1e-6) or \
           (hi != np.inf and abs(val - hi) < 1e-6):
            pinned_x.append(nm)

    # ---- print ----
    sep = "=" * 80
    print(f"\n{sep}\n  {sample_type} / {run_type}   (n={n})\n{sep}", file=fout)
    print(f"\n  Standard:   L = A·N^(-α) + B·D^(-β) + E    "
          f"SSR={ssr_s:.5f}  AIC={aic_s:.2f}  BIC={bic_s:.2f}", file=fout)
    print(f"  {'param':>6s} {'estimate':>12s} {'se':>11s} {'t':>9s} "
          f"{'p':>10s}     df={n - k_s}", file=fout)
    for r in rows_s:
        print(f"  {r['param']:>6s} {r['estimate']:>12.5f} {r['se']:>11.5f} "
              f"{r['t']:>9.3f} {fmt_p(r['p']):>10s}  {stars(r['p'])}",
              file=fout)

    print(f"\n  Cross-term: + C·(N·D)^(-γ)                    "
          f"SSR={ssr_x:.5f}  AIC={aic_x:.2f}  BIC={bic_x:.2f}", file=fout)
    print(f"  {'param':>6s} {'estimate':>12s} {'se':>11s} {'t':>9s} "
          f"{'p':>10s}     df={n - k_x}", file=fout)
    for r in rows_x:
        print(f"  {r['param']:>6s} {r['estimate']:>12.5f} {r['se']:>11.5f} "
              f"{r['t']:>9.3f} {fmt_p(r['p']):>10s}  {stars(r['p'])}",
              file=fout)
    if pinned_x:
        print(f"  ⚠ pinned to bound: {', '.join(pinned_x)}  "
              f"— SE/t for these params not meaningful.", file=fout)

    print(f"\n  Reduced (drop A·N^(-α)):  L = B·D^(-β) + C·(N·D)^(-γ) + E    "
          f"SSR={ssr_r:.5f}  AIC={aic_r:.2f}  BIC={bic_r:.2f}", file=fout)
    print(f"  {'param':>6s} {'estimate':>12s} {'se':>11s} {'t':>9s} "
          f"{'p':>10s}     df={n - k_r}", file=fout)
    for r in rows_r:
        print(f"  {r['param']:>6s} {r['estimate']:>12.5f} {r['se']:>11.5f} "
              f"{r['t']:>9.3f} {fmt_p(r['p']):>10s}  {stars(r['p'])}",
              file=fout)

    print(f"\n  Bootstrap LRT (standard ⊂ cross-term):  H0: C = 0", file=fout)
    print(f"    LR = n·log(SSR_std/SSR_cross) = {F:.3f},  "
          f"p_boot = {fmt_p(pF)}  {stars(pF)}  (B=1000)", file=fout)
    print(f"  Bootstrap LRT (reduced ⊂ cross-term):  H0: A = 0", file=fout)
    print(f"    LR = n·log(SSR_red/SSR_cross) = {F_rx:.3f},  "
          f"p_boot = {fmt_p(pF_rx)}  {stars(pF_rx)}  (B=1000)", file=fout)
    dAIC = aic_x - aic_s
    dBIC = bic_x - bic_s
    dAIC_rs = aic_r - aic_s
    dBIC_rs = bic_r - bic_s
    dAIC_rx = aic_r - aic_x
    dBIC_rx = bic_r - bic_x
    print(f"    ΔAIC: cross−std = {dAIC:+.2f},  reduced−std = {dAIC_rs:+.2f},  "
          f"reduced−cross = {dAIC_rx:+.2f}", file=fout)
    print(f"    ΔBIC: cross−std = {dBIC:+.2f},  reduced−std = {dBIC_rs:+.2f},  "
          f"reduced−cross = {dBIC_rx:+.2f}", file=fout)

    return {
        'config': f"{sample_type}/{run_type}", 'n': n,
        'ssr_std': ssr_s, 'ssr_cross': ssr_x, 'ssr_red': ssr_r,
        'F': F, 'p_F': pF, 'F_rx': F_rx, 'p_F_rx': pF_rx,
        'dAIC': dAIC, 'dBIC': dBIC,
        'dAIC_rs': dAIC_rs, 'dBIC_rs': dBIC_rs,
        'dAIC_rx': dAIC_rx, 'dBIC_rx': dBIC_rx,
        'aic_std': aic_s, 'aic_cross': aic_x, 'aic_red': aic_r,
        'bic_std': bic_s, 'bic_cross': bic_x, 'bic_red': bic_r,
        'rows_std': rows_s, 'rows_cross': rows_x, 'rows_red': rows_r,
        'pinned_cross': pinned_x,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--out', default=os.path.join(OUTPUT_DIR, 'significance_report.txt'))
    ap.add_argument('--csv-out', default=os.path.join(OUTPUT_DIR, 'significance_summary.csv'))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sel = select_best(df)
    sel['D_M'] = sel['total_samples_seen'] / 1e6

    configs = [
        ('Pythia', 'multi'),
        ('Herwig', 'multi'),
        ('Mixed',  'multi'),
        ('Pythia', '1ep'),
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results = []
    with open(args.out, 'w') as fout:
        # header
        for sink in (fout, None):
            target = sink if sink is not None else __import__('sys').stdout
            print("Significance tests for Chinchilla scaling-law fits", file=target)
            print(f"CSV: {args.csv}", file=target)
            print(f"Per-param: Wald t-stat = est/SE, two-sided.\n"
                  "Cross-term significance via parametric bootstrap LRT (B=1000) — the\n"
                  "naive F-test is invalid because C and A sit on the parameter bound\n"
                  "under the null and the paired exponent is unidentified there.\n"
                  "Stars: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10",
                  file=target)
        for st, rt in configs:
            r = analyze_config(sel, st, rt, fout)
            # also dump to stdout
            if r is not None:
                results.append(r)
        # rerun dump to stdout by re-opening (simple: read back the file)
    # echo report to stdout
    with open(args.out) as f:
        print(f.read())

    # ---- compact CSV summary ----
    rows_csv = []
    for r in results:
        for rr in r['rows_std']:
            rows_csv.append({'config': r['config'], 'form': 'standard',
                             'param': rr['param'],
                             'estimate': rr['estimate'], 'se': rr['se'],
                             't': rr['t'], 'p': rr['p']})
        for rr in r['rows_cross']:
            rows_csv.append({'config': r['config'], 'form': 'cross-term',
                             'param': rr['param'],
                             'estimate': rr['estimate'], 'se': rr['se'],
                             't': rr['t'], 'p': rr['p']})
        for rr in r['rows_red']:
            rows_csv.append({'config': r['config'], 'form': 'reduced',
                             'param': rr['param'],
                             'estimate': rr['estimate'], 'se': rr['se'],
                             't': rr['t'], 'p': rr['p']})
        rows_csv.append({'config': r['config'], 'form': 'F (std vs cross)',
                         'param': '—',
                         'estimate': r['F'], 'se': np.nan, 't': np.nan,
                         'p': r['p_F']})
        rows_csv.append({'config': r['config'], 'form': 'F (red vs cross)',
                         'param': '—',
                         'estimate': r['F_rx'], 'se': np.nan, 't': np.nan,
                         'p': r['p_F_rx']})
    pd.DataFrame(rows_csv).to_csv(args.csv_out, index=False)
    print(f"\nSaved compact CSV: {args.csv_out}")
    print(f"Saved full report: {args.out}")


if __name__ == '__main__':
    main()
