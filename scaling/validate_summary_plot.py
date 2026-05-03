#!/usr/bin/env python
"""Master summary plot of validation results across all configs."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

CSV = '/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/parsed_runs.csv'
OUT = '/pscratch/sd/s/sqian/part_training_output/scaling_study_v2/plots'

PARAMS_M = {'nano': 0.1, 'micro': 0.2, 'tiny': 0.4, 'small': 1.0,
            'base': 2.14, 'large': 8.5, 'xlarge': 25.0}
SMALL = ['nano', 'micro', 'tiny', 'small', 'base']
HELD = ['large', 'xlarge']

MODEL_COLORS = {
    'nano': '#1f77b4', 'micro': '#ff7f0e', 'tiny': '#2ca02c',
    'small': '#d62728', 'base': '#9467bd', 'large': '#8c564b',
    'xlarge': '#e377c2',
}


def chinchilla(X, A, alpha, B, beta, E):
    N, D = X
    return A * np.power(N, -alpha) + B * np.power(D, -beta) + E


def chinchilla_cross(X, A, alpha, B, beta, C, gamma, E):
    N, D = X
    return (A * np.power(N, -alpha) + B * np.power(D, -beta)
            + C * np.power(N * D, -gamma) + E)


def fit_form(form, N, D, L, p0, bounds):
    popt, _ = curve_fit(form, (N, D), L, p0=p0, bounds=bounds, maxfev=100000)
    return popt


def select(df):
    keys = ['model_size', 'data_budget', 'sample_type', 'feature_type', 'pair_tag', 'run_type']
    return df.dropna(subset=['train_loss']).sort_values('train_loss').drop_duplicates(subset=keys, keep='first').copy()


def main():
    df = pd.read_csv(CSV)
    sel = select(df)
    sel['D_M'] = sel['total_samples_seen'] / 1e6

    configs = [
        ('Pythia', 'multi', 'Pythia (multi-epoch)'),
        ('Herwig', 'multi', 'Herwig (multi-epoch)'),
        ('Mixed',  'multi', 'Mixed (multi-epoch)'),
        ('Pythia', '1ep',   'Pythia (1-epoch)'),
    ]

    # ============================================================
    # Master plot: 4 configs x 2 forms (predicted-vs-observed)
    # ============================================================
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for col, (st, rt, title) in enumerate(configs):
        sub = sel[(sel['sample_type'] == st) & (sel['run_type'] == rt)]
        train = sub[sub['model_size'].isin(SMALL)]
        val = sub[sub['model_size'].isin(HELD)]

        L_min, L_max = train['train_loss'].min(), train['train_loss'].max()

        # Standard fit
        popt_std = fit_form(chinchilla, train['params_M'].values, train['D_M'].values, train['train_loss'].values,
                            p0=[1.0, 0.5, 1.0, 0.5, L_min],
                            bounds=([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, L_max]))
        # Cross-term fit
        popt_x = fit_form(chinchilla_cross, train['params_M'].values, train['D_M'].values, train['train_loss'].values,
                          p0=[1.0, 0.5, 1.0, 0.5, 0.5, 0.3, L_min],
                          bounds=([0, 0, 0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, np.inf, 2, L_max]))

        L_p_std = chinchilla((val['params_M'].values, val['D_M'].values), *popt_std)
        L_p_x = chinchilla_cross((val['params_M'].values, val['D_M'].values), *popt_x)

        rmse_std = np.sqrt(np.mean((val['train_loss'].values - L_p_std) ** 2))
        rmse_x = np.sqrt(np.mean((val['train_loss'].values - L_p_x) ** 2))

        # Top row: standard
        ax = axes[0, col]
        for model in HELD:
            s = val[val['model_size'] == model]
            if s.empty: continue
            c = MODEL_COLORS[model]
            L_p = chinchilla((s['params_M'].values, s['D_M'].values), *popt_std)
            ax.scatter(L_p, s['train_loss'], color=c, s=90, edgecolor='black', linewidth=0.7,
                       label=f"{model}", zorder=3)
        all_vals = np.concatenate([L_p_std, val['train_loss'].values])
        lims = [all_vals.min() * 0.95, all_vals.max() * 1.05]
        ax.plot(lims, lims, 'k--', alpha=0.4)
        ax.fill_between(lims, [l * 0.95 for l in lims], [l * 1.05 for l in lims],
                        alpha=0.1, color='gray', label='±5%')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Predicted Loss', fontsize=11)
        ax.set_ylabel('Observed Loss', fontsize=11)
        ax.set_title(f'{title}\nStandard: RMSE={rmse_std:.4f}', fontsize=11)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')

        # Bottom row: cross-term
        ax = axes[1, col]
        for model in HELD:
            s = val[val['model_size'] == model]
            if s.empty: continue
            c = MODEL_COLORS[model]
            L_p = chinchilla_cross((s['params_M'].values, s['D_M'].values), *popt_x)
            ax.scatter(L_p, s['train_loss'], color=c, s=90, edgecolor='black', linewidth=0.7,
                       label=f"{model}", zorder=3)
        all_vals = np.concatenate([L_p_x, val['train_loss'].values])
        lims = [all_vals.min() * 0.95, all_vals.max() * 1.05]
        ax.plot(lims, lims, 'k--', alpha=0.4)
        ax.fill_between(lims, [l * 0.95 for l in lims], [l * 1.05 for l in lims],
                        alpha=0.1, color='green', label='±5%')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Predicted Loss', fontsize=11)
        ax.set_ylabel('Observed Loss', fontsize=11)
        improvement = (rmse_std - rmse_x) / rmse_std * 100
        ax.set_title(f'Cross-term: RMSE={rmse_x:.4f}  ({improvement:+.0f}% vs std)', fontsize=11)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'box')

    fig.text(0.01, 0.74, 'Standard\n$L = AN^{-α} + BD^{-β} + E$', fontsize=12,
             rotation=90, va='center', ha='center', weight='bold')
    fig.text(0.01, 0.26, 'Cross-term\n$+ C(ND)^{-γ}$', fontsize=12,
             rotation=90, va='center', ha='center', weight='bold', color='darkgreen')

    plt.suptitle('Held-out validation: predict large + xlarge from small-model fit',
                 fontsize=15, weight='bold')
    plt.tight_layout(rect=[0.02, 0, 1, 0.97])
    out_path = os.path.join(OUT, 'validation_master.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)

    # ============================================================
    # 3D surface plot: Pythia multi-epoch standard vs cross-term
    # ============================================================
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(16, 7))

    sub = sel[(sel['sample_type'] == 'Pythia') & (sel['run_type'] == 'multi')]
    train = sub[sub['model_size'].isin(SMALL)]
    val = sub[sub['model_size'].isin(HELD)]

    L_min, L_max = train['train_loss'].min(), train['train_loss'].max()
    popt_std = fit_form(chinchilla, train['params_M'].values, train['D_M'].values, train['train_loss'].values,
                        p0=[1.0, 0.5, 1.0, 0.5, L_min],
                        bounds=([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, L_max]))
    popt_x = fit_form(chinchilla_cross, train['params_M'].values, train['D_M'].values, train['train_loss'].values,
                      p0=[1.0, 0.5, 1.0, 0.5, 0.5, 0.3, L_min],
                      bounds=([0, 0, 0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, np.inf, 2, L_max]))

    N_grid = np.logspace(np.log10(0.05), np.log10(30), 40)
    D_grid = np.logspace(np.log10(3), np.log10(600), 40)
    NN, DD = np.meshgrid(N_grid, D_grid)
    LL_std = chinchilla((NN, DD), *popt_std)
    LL_x = chinchilla_cross((NN, DD), *popt_x)

    for i, (LL, popt, title) in enumerate([(LL_std, popt_std, 'Standard'), (LL_x, popt_x, 'Cross-term')]):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.plot_surface(np.log10(NN), np.log10(DD), LL, alpha=0.4, cmap='viridis',
                        edgecolor='none', rstride=2, cstride=2)
        # Train (small models): blue dots
        ax.scatter(np.log10(train['params_M'].values), np.log10(train['D_M'].values),
                   train['train_loss'].values, color='blue', s=40, edgecolor='black',
                   linewidth=0.5, label='small models (fit)')
        # Held-out: red Xs
        ax.scatter(np.log10(val['params_M'].values), np.log10(val['D_M'].values),
                   val['train_loss'].values, color='red', s=80, marker='X',
                   edgecolor='black', linewidth=0.7, label='large+xlarge (held-out)')
        ax.set_xlabel('log$_{10}$ N (M)')
        ax.set_ylabel('log$_{10}$ D (M)')
        ax.set_zlabel('Loss')
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, loc='upper left')
        ax.view_init(elev=22, azim=-65)

    plt.suptitle('Scaling-law surface — Pythia multi-epoch', fontsize=14, weight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUT, 'validation_surface3d.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)

    # ============================================================
    # RMSE bar comparison across all configs
    # ============================================================
    fig, ax = plt.subplots(figsize=(11, 6))
    cfg_labels = ['Pythia\nmulti', 'Herwig\nmulti', 'Mixed\nmulti', 'Pythia\n1-epoch']
    rmse_std_all = []
    rmse_x_all = []

    for st, rt, _ in configs:
        sub = sel[(sel['sample_type'] == st) & (sel['run_type'] == rt)]
        train = sub[sub['model_size'].isin(SMALL)]
        val = sub[sub['model_size'].isin(HELD)]
        L_min, L_max = train['train_loss'].min(), train['train_loss'].max()
        popt_std = fit_form(chinchilla, train['params_M'].values, train['D_M'].values, train['train_loss'].values,
                            p0=[1.0, 0.5, 1.0, 0.5, L_min],
                            bounds=([0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, L_max]))
        popt_x = fit_form(chinchilla_cross, train['params_M'].values, train['D_M'].values, train['train_loss'].values,
                          p0=[1.0, 0.5, 1.0, 0.5, 0.5, 0.3, L_min],
                          bounds=([0, 0, 0, 0, 0, 0, 0], [np.inf, 2, np.inf, 2, np.inf, 2, L_max]))
        L_p_std = chinchilla((val['params_M'].values, val['D_M'].values), *popt_std)
        L_p_x = chinchilla_cross((val['params_M'].values, val['D_M'].values), *popt_x)
        rmse_std_all.append(np.sqrt(np.mean((val['train_loss'].values - L_p_std) ** 2)))
        rmse_x_all.append(np.sqrt(np.mean((val['train_loss'].values - L_p_x) ** 2)))

    x = np.arange(len(cfg_labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, rmse_std_all, w, label='Standard', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + w/2, rmse_x_all, w, label='Cross-term', color='seagreen', edgecolor='black')

    for i, (s, c) in enumerate(zip(rmse_std_all, rmse_x_all)):
        impr = (s - c) / s * 100
        ax.annotate(f'-{impr:.0f}%', xy=(i, max(s, c) + 0.005),
                    ha='center', fontsize=11, weight='bold', color='darkgreen')

    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels, fontsize=11)
    ax.set_ylabel('RMSE on held-out (large + xlarge)', fontsize=12)
    ax.set_title('Scaling-law extrapolation error: standard vs cross-term form', fontsize=13, weight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(OUT, 'validation_rmse_bars.png')
    plt.savefig(out_path, dpi=200)
    print(f'Saved: {out_path}')
    plt.close(fig)

    print('\nAll plots saved to:', OUT)


if __name__ == '__main__':
    main()
