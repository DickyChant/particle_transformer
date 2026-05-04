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
    'nano': 0.0497, 'micro': 0.1375, 'tiny': 0.2919,
    'small': 0.9871, 'base': 2.140, 'large': 11.670, 'xlarge': 35.070,
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
TRAIN_ACC_RE = re.compile(r'Train AvgLoss: [\d.]+, AvgAcc: ([\d.]+)')
VAL_METRIC_RE = re.compile(r'Current validation metric: ([\d.]+)')
ROC_AUC_RE = re.compile(r'^\s*- roc_auc_score:\s*$')
ROC_AUC_VAL_RE = re.compile(r'^([\d.]+)\s*$')

# Weaver logs the actual model parameter count at startup as
# "Number of parameters:           VALUE UNIT" (where UNIT is k/M/G or empty).
PARAMS_LOG_RE = re.compile(r'Number of parameters:\s+([\d.]+)\s*([kMG]?)')

# Multitask-specific per-task losses logged by networks/multitask_ParT.py.
# Real weaver logs prefix every line with `[YYYY-MM-DD HH:MM:SS,ms] INFO:`,
# so we use unanchored patterns and re.search() throughout. The `(w=...)`
# clause distinguishes the train line ("CE: x (w=y), MSE: ..., GradCos: ...")
# from the eval line ("CE: x, MSE: y") so the same regex doesn't match both.
TRAIN_CE_RE = re.compile(r'CE:\s*([\d.]+)\s*\(w=([\d.]+)\)')
TRAIN_MSE_RE = re.compile(r'MSE:\s*([\d.]+)\s*\(w=([\d.]+)\)')
TRAIN_GRADCOS_RE = re.compile(r'GradCos:\s*([-\d.eE+]+|nan)')
EVAL_REG_MAE_RE = re.compile(r'Regression MSE:\s*[\d.]+,\s*MAE:\s*([\d.]+)')

CSV_COLUMNS = [
    'model_size', 'data_budget', 'sample_type', 'feature_type', 'pair_tag',
    'run_type', 'task_type', 'params_M', 'num_epochs_cfg', 'epoch',
    'samples_per_epoch', 'ngpus', 'total_samples_seen',
    'train_loss', 'train_acc', 'val_metric', 'val_roc_auc',
    'test_metric', 'test_roc_auc',
    'train_ce', 'train_mse', 'train_grad_cos', 'val_reg_mae',
    'source', 'run_dir',
]

# Weaver logs `Test metric X.XXXXX` once at the end of a run with `--data-test`.
# By default this is test accuracy (the eval function returns total_correct/count).
# Test ROC AUC comes from the LAST `roc_auc_score:` block in the log.
TEST_METRIC_RE = re.compile(r'Test metric\s+([\d.]+)')


def _params_M_from_log(log_path):
    """Read the model parameter count (in M) from a weaver log line.
    Returns None if not found."""
    try:
        with open(log_path, 'r') as f:
            for line in f:
                m = PARAMS_LOG_RE.search(line)
                if m:
                    val = float(m.group(1))
                    unit = m.group(2)
                    if unit == 'k':
                        return val / 1000.0
                    if unit == 'M':
                        return val
                    if unit == 'G':
                        return val * 1000.0
                    return val / 1e6
    except Exception:
        pass
    return None


def _extract_test_metrics_from_log(filepath):
    """Extract end-of-run test-set metrics from a weaver log.

    Returns dict with `test_metric` (Test metric line; usually accuracy) and
    `test_roc_auc` (the LAST `roc_auc_score:` block in the log, which is the
    test-eval ROC AUC since test is run after all training/validation epochs).
    Either may be None if the run didn't reach the test phase.
    """
    out = {'test_metric': None, 'test_roc_auc': None}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
    except Exception:
        return out

    # Test accuracy: parse "Test metric X.XXXXX"
    for line in lines:
        m = TEST_METRIC_RE.search(line)
        if m:
            out['test_metric'] = float(m.group(1))
            # don't break — keep the LAST occurrence in case multiple test
            # phases are present (e.g. requeued runs)

    # Test ROC AUC: only meaningful if the test phase actually ran (i.e., we
    # saw a "Test metric" line). The last roc_auc_score block in the log
    # is then the test-eval one, since weaver runs test after all epochs.
    if out['test_metric'] is not None:
        last_roc = None
        for i, line in enumerate(lines):
            if ROC_AUC_RE.match(line) and 'matrix' not in line:
                if i + 1 < len(lines):
                    m_roc = ROC_AUC_VAL_RE.match(lines[i + 1].strip())
                    if m_roc:
                        last_roc = float(m_roc.group(1))
        out['test_roc_auc'] = last_roc
    return out


EPOCH_TRAIN_RE = re.compile(r'Epoch #(\d+) training')


def _extract_metrics_from_log(filepath):
    """Extract per-epoch metrics from a weaver log file.

    Returns list of dicts indexed by ABSOLUTE epoch number (recovered from
    `Epoch #N training` lines), with keys: epoch, train_loss, train_acc,
    val_metric, val_roc_auc, plus multitask-only fields train_ce, train_mse,
    train_grad_cos, val_reg_mae. The list is sorted by epoch.
    """
    by_epoch = {}
    current = {}
    current_epoch = None
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            # Epoch start: capture the absolute epoch number that the next
            # block of metrics belongs to.
            m_epoch = EPOCH_TRAIN_RE.search(line)
            if m_epoch:
                # Flush any in-progress epoch dict before we move on.
                if current_epoch is not None and current.get('train_loss') is not None:
                    by_epoch[current_epoch] = current
                current_epoch = int(m_epoch.group(1))
                current = {}

            # Train loss + accuracy
            m_loss = TRAIN_LOSS_RE.search(line)
            if m_loss:
                current['train_loss'] = float(m_loss.group(1))
                m_acc = TRAIN_ACC_RE.search(line)
                if m_acc:
                    current['train_acc'] = float(m_acc.group(1))

            # Validation metric (AUC from weaver)
            m_val = VAL_METRIC_RE.search(line)
            if m_val:
                current['val_metric'] = float(m_val.group(1))

            # Multitask training: "[ts] INFO:   CE: x (w=y), MSE: x (w=y), GradCos: ..."
            m_ce_train = TRAIN_CE_RE.search(line)
            if m_ce_train:
                current['train_ce'] = float(m_ce_train.group(1))
                m_mse_train = TRAIN_MSE_RE.search(line)
                if m_mse_train:
                    current['train_mse'] = float(m_mse_train.group(1))
                m_gc = TRAIN_GRADCOS_RE.search(line)
                if m_gc:
                    try:
                        current['train_grad_cos'] = float(m_gc.group(1))
                    except ValueError:
                        current['train_grad_cos'] = float('nan')

            # Multitask validation regression: "  Regression MSE: ..., MAE: ..."
            m_reg_eval = EVAL_REG_MAE_RE.search(line)
            if m_reg_eval:
                current['val_reg_mae'] = float(m_reg_eval.group(1))

            # roc_auc_score (standalone, next line has the value).
            if ROC_AUC_RE.match(line) and 'matrix' not in line:
                if i + 1 < len(lines):
                    m_roc = ROC_AUC_VAL_RE.match(lines[i + 1].strip())
                    if m_roc and current.get('val_roc_auc') is None:
                        current['val_roc_auc'] = float(m_roc.group(1))

            i += 1

        # Flush the final epoch.
        if current_epoch is not None and current.get('train_loss') is not None:
            by_epoch[current_epoch] = current

        # Some legacy logs lack `Epoch #N training` lines; fall back to
        # positional indexing in that case so we don't drop those runs.
        if not by_epoch and current.get('train_loss') is not None:
            by_epoch[0] = current
    except Exception:
        pass

    return [
        {'epoch': ep, **{k: v.get(k) for k in
            ('train_loss', 'train_acc', 'val_metric', 'val_roc_auc',
             'train_ce', 'train_mse', 'train_grad_cos', 'val_reg_mae')}}
        for ep, v in sorted(by_epoch.items())
    ]


def _stitch_run_metrics(log_files):
    """Merge per-epoch metrics across ALL log fragments of a run.

    On requeue, each retry produces its own log file but continues from a
    later epoch (via `--load-epoch`). The fragments together describe the
    full run; we merge them by absolute epoch number, with later fragments
    overriding earlier ones for any shared epoch (the later one is from
    the most recent retry).
    """
    merged = {}
    for lf in sorted(log_files):  # sort so newer fragments come last by mtime/timestamp prefix
        for entry in _extract_metrics_from_log(lf):
            merged[entry['epoch']] = entry
    return [merged[k] for k in sorted(merged.keys())]


def _extract_losses_from_log(filepath):
    """Extract ordered list of Train AvgLoss values from a weaver log file (legacy)."""
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
        # Single-task launchers and the multitask launcher both write task_type;
        # default to 'singletask' for legacy v2 runs that pre-date the field.
        task_type = config.get('task_type', 'singletask')
        spe = int(config.get('samples_per_epoch', 0))
        ngpus = int(config.get('ngpus', DEFAULT_NGPUS))
        num_epochs_cfg = int(config.get('num_epochs', 0))
        run_type = '1ep' if '_1ep_' in run_dir_name else 'multi'

        if spe == 0:
            # Fallback to defaults
            if budget in BUDGET_EPOCH_CONFIG:
                spe, num_epochs_cfg = BUDGET_EPOCH_CONFIG[budget]
            else:
                continue

        # Stitch all log fragments under this run dir (resumed runs produce
        # multiple log files; each contains a contiguous chunk of epochs).
        log_files = sorted(glob.glob(os.path.join(run_dir, 'logs', '*.log.000')))
        stitched = _stitch_run_metrics(log_files)

        # Test-set metrics: take the latest log fragment that ran the test
        # phase (newest timestamp wins).
        test_info = {'test_metric': None, 'test_roc_auc': None}
        for lf in sorted(log_files, reverse=True):
            t = _extract_test_metrics_from_log(lf)
            if t['test_metric'] is not None:
                test_info = t
                break

        # Prefer the parameter count weaver actually logged (any fragment is fine).
        params_log = None
        for lf in log_files:
            params_log = _params_M_from_log(lf)
            if params_log is not None:
                break
        params_M_value = params_log if params_log is not None else PARAMS_M.get(model, 0)

        if not stitched:
            continue
        max_epoch = max(e['epoch'] for e in stitched)
        for m in stitched:
            ep = m['epoch']
            is_last = (ep == max_epoch)
            rows.append({
                'model_size': model,
                'data_budget': budget,
                'sample_type': sample_type,
                'feature_type': feature_type,
                'pair_tag': pair_tag,
                'run_type': run_type,
                'task_type': task_type,
                'params_M': params_M_value,
                'num_epochs_cfg': num_epochs_cfg,
                'epoch': ep,
                'samples_per_epoch': spe,
                'ngpus': ngpus,
                'total_samples_seen': spe * ngpus * (ep + 1),
                'train_loss': m['train_loss'],
                'train_acc': m.get('train_acc'),
                'val_metric': m.get('val_metric'),
                'val_roc_auc': m.get('val_roc_auc'),
                'test_metric': test_info['test_metric'] if is_last else None,
                'test_roc_auc': test_info['test_roc_auc'] if is_last else None,
                'train_ce': m.get('train_ce'),
                'train_mse': m.get('train_mse'),
                'train_grad_cos': m.get('train_grad_cos'),
                'val_reg_mae': m.get('val_reg_mae'),
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
        best_metrics = []
        best_file = None
        for lf in files:
            metrics = _extract_metrics_from_log(lf)
            if len(metrics) > len(best_metrics):
                best_metrics = metrics
                best_file = lf

        # v1 runs share a single flat `logs/` dir so we cannot uniquely associate
        # a test result with one run. Skip test extraction for v1; only v2
        # timestamped run dirs report test-set metrics.
        params_log = _params_M_from_log(best_file) if best_file else None
        params_M_value = params_log if params_log is not None else PARAMS_M.get(model, 0)
        for epoch_idx, met in enumerate(best_metrics):
            rows.append({
                'model_size': model,
                'data_budget': budget,
                'sample_type': 'Pythia',
                'feature_type': 'full',
                'pair_tag': 'pair',
                'run_type': 'multi',
                'task_type': 'singletask',
                'params_M': params_M_value,
                'num_epochs_cfg': num_epochs_cfg,
                'epoch': epoch_idx,
                'samples_per_epoch': spe,
                'ngpus': DEFAULT_NGPUS,
                'total_samples_seen': spe * DEFAULT_NGPUS * (epoch_idx + 1),
                'train_loss': met['train_loss'],
                'train_acc': met.get('train_acc'),
                'val_metric': met.get('val_metric'),
                'val_roc_auc': met.get('val_roc_auc'),
                'test_metric': None,
                'test_roc_auc': None,
                'train_ce': None, 'train_mse': None,
                'train_grad_cos': None, 'val_reg_mae': None,
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
                'run_type': 'multi',
                'task_type': 'singletask',
                'params_M': PARAMS_M.get(model, 0),
                'num_epochs_cfg': num_epochs_cfg,
                'epoch': epoch_idx,
                'samples_per_epoch': spe,
                'ngpus': DEFAULT_NGPUS,
                'total_samples_seen': spe * DEFAULT_NGPUS * (epoch_idx + 1),
                'train_loss': loss,
                'train_acc': None,
                'val_metric': None,
                'val_roc_auc': None,
                'test_metric': None,
                'test_roc_auc': None,
                'train_ce': None, 'train_mse': None,
                'train_grad_cos': None, 'val_reg_mae': None,
                'source': 'v1_slurm',
                'run_dir': sf,
            })

    return rows


def deduplicate(rows):
    """Keep the row with the lowest loss for each unique config × epoch.

    `task_type` is part of the key so single-task and multitask runs that share
    a `model_size` name (but have different architectures and parameter counts)
    cannot accidentally dedup against each other.
    """
    best = {}
    for r in rows:
        key = (r['model_size'], r['data_budget'], r['sample_type'],
               r['feature_type'], r['pair_tag'], r['run_type'],
               r.get('task_type', 'singletask'), r['epoch'])
        if key not in best or r['train_loss'] < best[key]['train_loss']:
            best[key] = r
    return sorted(best.values(),
                  key=lambda r: (r['run_type'], r['sample_type'], r['feature_type'],
                                 r['pair_tag'], r.get('task_type', 'singletask'),
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
