#!/usr/bin/env python
"""
Parse all scaling study runs and extract per-epoch training loss to CSV.

Scans v1 and v2 run directories, extracts every epoch's train loss,
and writes a structured CSV — a single source of truth for plotting.

Output columns:
    model_size, data_budget, sample_type, feature_type, pair_tag,
    params_M, num_epochs_cfg, epoch, samples_per_epoch, ngpus,
    total_samples_seen, train_loss, source, run_dir

Usage:
    python parse_runs.py [--output PATH] [--results-dir DIR ...]
"""

import argparse
import csv
import glob
import os
import re
import sys

# ---- Constants ----

RESULTS_DIR_V1 = '/pscratch/sd/s/sqian/part_training_output/scaling_study'
RESULTS_DIR_V2 = '/pscratch/sd/s/sqian/part_training_output/scaling_study_v2'
DEFAULT_OUTPUT = os.path.join(RESULTS_DIR_V2, 'parsed_runs.csv')

PARAMS_M = {
    'nano': 0.1, 'micro': 0.2, 'tiny': 0.4,
    'small': 1.0, 'base': 2.14, 'large': 8.5, 'xlarge': 25.0,
}

# Default epoch config for v1 runs: (samples_per_epoch_per_gpu, num_epochs)
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

TRAIN_LOSS_RE = re.compile(r'Train AvgLoss: ([\d.]+)')

CSV_COLUMNS = [
    'model_size', 'data_budget', 'sample_type', 'feature_type', 'pair_tag',
    'params_M', 'num_epochs_cfg', 'epoch', 'samples_per_epoch', 'ngpus',
    'total_samples_seen', 'train_loss', 'source', 'run_dir',
]


def _extract_losses_from_log(filepath):
    """Extract ordered list of Train AvgLoss values from a weaver log file."""
    losses = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                m = TRAIN_LOSS_RE.search(line)
                if m:
                    losses.append(float(m.group(1)))
    except Exception:
        pass
    return losses


def parse_v2_runs(results_dir):
    """Parse v2 timestamped run directories."""
    runs_base = os.path.join(results_dir, 'runs')
    if not os.path.isdir(runs_base):
        return []

    rows = []
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
        if not model or not budget:
            continue

        sample_type = config.get('sample_type', 'Pythia')
        feature_type = config.get('feature_type', 'full')
        pair_tag = config.get('pair_tag', 'pair')
        spe = int(config.get('samples_per_epoch', 0))
        ngpus = int(config.get('ngpus', DEFAULT_NGPUS))
        num_epochs_cfg = int(config.get('num_epochs', 0))

        if spe == 0:
            # Fallback to defaults
            if budget in BUDGET_EPOCH_CONFIG:
                spe, num_epochs_cfg = BUDGET_EPOCH_CONFIG[budget]
            else:
                continue

        # Find the log with the most epoch data
        log_files = sorted(glob.glob(os.path.join(run_dir, 'logs', '*.log.000')))
        best_losses = []
        for lf in log_files:
            losses = _extract_losses_from_log(lf)
            if len(losses) > len(best_losses):
                best_losses = losses

        for epoch_idx, loss in enumerate(best_losses):
            rows.append({
                'model_size': model,
                'data_budget': budget,
                'sample_type': sample_type,
                'feature_type': feature_type,
                'pair_tag': pair_tag,
                'params_M': PARAMS_M.get(model, 0),
                'num_epochs_cfg': num_epochs_cfg,
                'epoch': epoch_idx,
                'samples_per_epoch': spe,
                'ngpus': ngpus,
                'total_samples_seen': spe * ngpus * (epoch_idx + 1),
                'train_loss': loss,
                'source': 'v2_run_dir',
                'run_dir': run_dir,
            })

    return rows


def parse_v1_logs(results_dir):
    """Parse v1 flat log directory. All v1 runs are Pythia/full/pair."""
    log_dir = os.path.join(results_dir, 'logs')
    if not os.path.isdir(log_dir):
        return []

    # Group log files by run prefix (model_budget)
    log_files = sorted(glob.glob(os.path.join(log_dir, '*.log.000')))
    run_groups = {}
    for lf in log_files:
        basename = os.path.basename(lf)
        # Format: model_budget_YYYYMMDD-HHMMSS_...log.000
        m = re.match(r'^([a-z]+)_(\d+M)_', basename)
        if m:
            key = (m.group(1), m.group(2))
            run_groups.setdefault(key, []).append(lf)

    rows = []
    for (model, budget), files in sorted(run_groups.items()):
        if budget not in BUDGET_EPOCH_CONFIG:
            continue
        spe, num_epochs_cfg = BUDGET_EPOCH_CONFIG[budget]

        # Use the file with the most epoch data (latest complete run)
        best_losses = []
        best_file = None
        for lf in files:
            losses = _extract_losses_from_log(lf)
            if len(losses) > len(best_losses):
                best_losses = losses
                best_file = lf

        for epoch_idx, loss in enumerate(best_losses):
            rows.append({
                'model_size': model,
                'data_budget': budget,
                'sample_type': 'Pythia',
                'feature_type': 'full',
                'pair_tag': 'pair',
                'params_M': PARAMS_M.get(model, 0),
                'num_epochs_cfg': num_epochs_cfg,
                'epoch': epoch_idx,
                'samples_per_epoch': spe,
                'ngpus': DEFAULT_NGPUS,
                'total_samples_seen': spe * DEFAULT_NGPUS * (epoch_idx + 1),
                'train_loss': loss,
                'source': 'v1_log',
                'run_dir': os.path.dirname(best_file) if best_file else '',
            })

    return rows


def parse_v1_slurm(results_dir):
    """Parse v1 SLURM logs as fallback for runs without text logs."""
    slurm_dir = os.path.join(results_dir, 'slurm_logs')
    if not os.path.isdir(slurm_dir):
        return []

    weaver_log_re = re.compile(r"'log',\s*'([^']+\.log\.\d+)'")

    rows = []
    for sf in sorted(glob.glob(os.path.join(slurm_dir, 'slurm-*.out'))):
        try:
            with open(sf, 'r') as f:
                text = f.read()
        except Exception:
            continue

        model = budget = None
        for line in text.split('\n'):
            if 'model_size:' in line or 'Model size:' in line:
                model = line.strip().split()[-1]
            if 'data_budget:' in line or 'Data budget:' in line:
                budget = line.strip().split()[-1]

        if not model or not budget or budget not in BUDGET_EPOCH_CONFIG:
            continue

        spe, num_epochs_cfg = BUDGET_EPOCH_CONFIG[budget]

        # Try to get losses from the SLURM log text itself
        losses = [float(m.group(1)) for m in TRAIN_LOSS_RE.finditer(text)]

        # Follow weaver log reference if no losses in SLURM output
        if not losses:
            wm = weaver_log_re.search(text)
            if wm and os.path.isfile(wm.group(1)):
                losses = _extract_losses_from_log(wm.group(1))

        for epoch_idx, loss in enumerate(losses):
            rows.append({
                'model_size': model,
                'data_budget': budget,
                'sample_type': 'Pythia',
                'feature_type': 'full',
                'pair_tag': 'pair',
                'params_M': PARAMS_M.get(model, 0),
                'num_epochs_cfg': num_epochs_cfg,
                'epoch': epoch_idx,
                'samples_per_epoch': spe,
                'ngpus': DEFAULT_NGPUS,
                'total_samples_seen': spe * DEFAULT_NGPUS * (epoch_idx + 1),
                'train_loss': loss,
                'source': 'v1_slurm',
                'run_dir': sf,
            })

    return rows


def deduplicate(rows):
    """Keep the row with the lowest loss for each (model, budget, config, epoch)."""
    best = {}
    for r in rows:
        key = (r['model_size'], r['data_budget'], r['sample_type'],
               r['feature_type'], r['pair_tag'], r['epoch'])
        if key not in best or r['train_loss'] < best[key]['train_loss']:
            best[key] = r
    return sorted(best.values(),
                  key=lambda r: (r['sample_type'], r['feature_type'], r['pair_tag'],
                                 r['model_size'], r['data_budget'], r['epoch']))


def main():
    parser = argparse.ArgumentParser(description='Parse scaling study runs to CSV')
    parser.add_argument('--results-dir', nargs='+',
                        default=[RESULTS_DIR_V2, RESULTS_DIR_V1],
                        help='Directories to scan')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Output CSV path (default: {DEFAULT_OUTPUT})')
    args = parser.parse_args()

    all_rows = []
    for rdir in args.results_dir:
        if not os.path.isdir(rdir):
            print(f"Skipping {rdir} (not found)")
            continue

        print(f"Scanning {rdir}...")

        # v2 run dirs
        v2_rows = parse_v2_runs(rdir)
        if v2_rows:
            print(f"  v2 run dirs: {len(v2_rows)} epoch entries")
            all_rows.extend(v2_rows)

        # v1 text logs
        v1_rows = parse_v1_logs(rdir)
        if v1_rows:
            print(f"  v1 text logs: {len(v1_rows)} epoch entries")
            all_rows.extend(v1_rows)

        # v1 SLURM logs (fallback)
        slurm_rows = parse_v1_slurm(rdir)
        if slurm_rows:
            print(f"  v1 SLURM logs: {len(slurm_rows)} epoch entries")
            all_rows.extend(slurm_rows)

    if not all_rows:
        print("No data found.")
        sys.exit(1)

    rows = deduplicate(all_rows)

    # Summary
    runs = set((r['model_size'], r['data_budget'], r['sample_type'],
                r['feature_type'], r['pair_tag']) for r in rows)
    configs = set((r['sample_type'], r['feature_type'], r['pair_tag']) for r in rows)

    print(f"\nTotal: {len(rows)} epoch entries from {len(runs)} unique runs, {len(configs)} config(s)")
    for cfg in sorted(configs):
        cfg_rows = [r for r in rows if (r['sample_type'], r['feature_type'], r['pair_tag']) == cfg]
        cfg_runs = set((r['model_size'], r['data_budget']) for r in cfg_rows)
        print(f"  {'/'.join(cfg)}: {len(cfg_runs)} runs, {len(cfg_rows)} epochs")

    # Write CSV
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten: {args.output}")


if __name__ == '__main__':
    main()
