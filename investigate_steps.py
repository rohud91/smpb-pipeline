"""
investigate_steps.py — Step 5 of the smPB pipeline (optional).

Diagnoses traces at a specific step count to distinguish real oligomers
from doublets, over-segmented traces, or other artefacts.

For each scored trace, computes:
  - Initial intensity and total bleach amplitude in unit-step multiples
  - Within-trace step-amplitude consistency (mean and std)
  - Distance to nearest neighbouring detected spot

Set STEPS_TO_INVESTIGATE to the step count you want to scrutinise
(e.g. 4 for suspected tetramers, 3 for suspected trimers, or even 1
to verify that monomers are real).

INPUTS
  - step_counts.csv         from score_steps.py
  - traces_corrected.csv    from extract_traces.py
  - spots_used.csv          from extract_traces.py

OUTPUTS  (all saved to OUTPUT_DIR)
  - investigation_metrics.csv      Per-trace metrics for all good traces
  - {N}_step_traces.png            Gallery of traces at the focal step count
  - investigation_boxplots.png     Metric boxplots grouped by step count
  - investigation_report.txt       Summary with per-trace detail

USAGE
  1. Set STEPS_TO_INVESTIGATE and file paths below.
  2. Run:  python investigate_steps.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

STEP_COUNTS_CSV = "/path/to/step_counts.csv"
TRACES_CSV      = "/path/to/traces_corrected.csv"
SPOTS_CSV       = "/path/to/spots_used.csv"
OUTPUT_DIR      = "/path/to/output_folder"

STEPS_TO_INVESTIGATE = 4     # Which step count to scrutinise
PIXEL_SIZE_UM        = 0.11  # Camera pixel size in µm
N_INIT_FRAMES        = 20    # Frames averaged for initial intensity
N_FINAL_FRAMES       = 50    # Frames averaged for final (baseline) intensity


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def parse_amps(s):
    """Parse pipe-separated amplitude string into numpy array."""
    if pd.isna(s) or s == '':
        return np.array([])
    out = []
    for a in str(s).split('|'):
        try:
            out.append(float(a))
        except ValueError:
            pass
    return np.array(out)


def compute_unit_step(df_good):
    """Fit Gaussian to all pooled step amplitudes. Returns (mu, sigma)."""
    all_amps = []
    for s in df_good['step_amplitudes']:
        all_amps.extend(parse_amps(s).tolist())
    all_amps = np.array(all_amps)
    counts, edges = np.histogram(all_amps, bins=30)
    centers = 0.5 * (edges[:-1] + edges[1:])
    try:
        p0 = [np.median(all_amps), np.std(all_amps), counts.max()]
        popt, _ = curve_fit(gaussian, centers, counts, p0=p0, maxfev=5000)
        return float(popt[0]), float(popt[1])
    except Exception:
        return float(np.median(all_amps)), float(np.std(all_amps))


def compute_metrics(df_good, traces, spots, unit_step):
    """Build a DataFrame of diagnostic metrics for every good trace."""
    trace_cols = set(traces.columns)
    spots_lookup = spots.set_index('_spot_id')
    coords_px = spots[['x_px', 'y_px']].values
    tree = cKDTree(coords_px)

    rows = []
    for _, r in df_good.iterrows():
        sid = str(r['spot_id'])
        if sid not in trace_cols:
            continue
        trace = traces[sid].values
        initial = float(trace[:N_INIT_FRAMES].mean())
        final   = float(trace[-N_FINAL_FRAMES:].mean())
        total_bleach = initial - final
        amps = parse_amps(r['step_amplitudes'])

        # Nearest-neighbour distance from spots_used.csv
        spot_num = int(sid.replace('spot_', ''))
        try:
            x_px = float(spots_lookup.loc[spot_num, 'x_px'])
            y_px = float(spots_lookup.loc[spot_num, 'y_px'])
            dists, _ = tree.query([x_px, y_px], k=2)
            nn_um = float(dists[1]) * PIXEL_SIZE_UM
        except (KeyError, ValueError):
            x_px = y_px = nn_um = np.nan

        rows.append({
            'spot_id': sid,
            'n_steps': int(r['n_steps']),
            'initial': initial,
            'final': final,
            'total_bleach': total_bleach,
            'bleach_units': total_bleach / unit_step,
            'step_mean': float(np.mean(amps)) if len(amps) else np.nan,
            'step_std':  float(np.std(amps))  if len(amps) else np.nan,
            'nn_um': nn_um,
            'x_px': x_px,
            'y_px': y_px,
        })
    return pd.DataFrame(rows)


def plot_trace_gallery(df_metrics, traces, unit_step, out_path):
    """Grid of traces at the focal step count with unit-step reference lines."""
    target = df_metrics[df_metrics['n_steps'] == STEPS_TO_INVESTIGATE]
    if len(target) == 0:
        print(f"No {STEPS_TO_INVESTIGATE}-step traces found.")
        return
    n = len(target)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.5*nrows),
                              sharex=True)
    axes = np.array(axes).ravel() if n > 1 else [axes]
    for ax, (_, row) in zip(axes, target.iterrows()):
        trace = traces[row['spot_id']].values
        ax.plot(trace, lw=0.6, color='gray')
        for k in range(int(row['bleach_units'] + 2)):
            ax.axhline(row['final'] + k * unit_step,
                       color='red', lw=0.5, alpha=0.25, linestyle='--')
        ax.set_title(
            f"{row['spot_id']}  bleach = {row['bleach_units']:.1f} units\n"
            f"step_mean = {row['step_mean']:.2f}, "
            f"step_std = {row['step_std']:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(target):]:
        ax.set_visible(False)
    fig.suptitle(
        f"{STEPS_TO_INVESTIGATE}-step traces — dashed red lines mark "
        f"unit-step multiples", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_boxplots(df_metrics, out_path):
    """Side-by-side boxplots of each metric grouped by step count."""
    metrics = [
        ('initial',      'Initial intensity'),
        ('bleach_units', 'Total bleach / unit_step'),
        ('step_mean',    'Mean step amplitude per trace'),
        ('step_std',     'Step amplitude std (within trace)'),
        ('nn_um',        'Distance to nearest neighbour (\u00b5m)'),
    ]
    groups = sorted(df_metrics['n_steps'].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 4))
    for ax, (col, label) in zip(axes, metrics):
        data = [df_metrics[df_metrics['n_steps'] == n][col].dropna()
                for n in groups]
        ax.boxplot(data, labels=[str(n) for n in groups], showmeans=True)
        ax.set_xlabel('n_steps')
        ax.set_title(label, fontsize=10)
        if col == 'bleach_units':
            for k in groups:
                ax.axhline(k, color='red', lw=0.5, alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(df_metrics, unit_step, unit_sigma, out_path):
    """Write a text report with per-group summary and per-trace detail."""
    lines = [
        "Step investigation report",
        "=" * 50, "",
        f"Focal step count: {STEPS_TO_INVESTIGATE}",
        f"Unitary step (Gaussian mean): {unit_step:.4f}",
        f"Unitary step (Gaussian std):  {unit_sigma:.4f}", "",
        "Per n_steps summary:",
        f"{'n':<4} {'count':<6} {'init mean':<12} "
        f"{'bleach (units) mean':<22} {'step_mean mean':<16} {'nn_um median':<14}",
    ]
    for n in sorted(df_metrics['n_steps'].unique()):
        sub = df_metrics[df_metrics['n_steps'] == n]
        lines.append(
            f"{n:<4} {len(sub):<6} {sub['initial'].mean():<12.3f} "
            f"{sub['bleach_units'].mean():<22.3f} "
            f"{sub['step_mean'].mean():<16.3f} "
            f"{sub['nn_um'].median():<14.3f}")
    lines.append("")
    target = df_metrics[df_metrics['n_steps'] == STEPS_TO_INVESTIGATE]
    if len(target):
        lines.append(f"Per-trace detail of the {STEPS_TO_INVESTIGATE}-step traces:")
        for _, r in target.iterrows():
            lines.append(
                f"  {r['spot_id']:<12}  initial = {r['initial']:.2f}  "
                f"bleach_units = {r['bleach_units']:.2f}  "
                f"step_mean = {r['step_mean']:.2f}  "
                f"step_std = {r['step_std']:.2f}  "
                f"nn_um = {r['nn_um']:.2f}")

    text = '\n'.join(lines)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(STEP_COUNTS_CSV)
    df_good = df[df['status'] == 'good'].copy()
    df_good['n_steps'] = df_good['n_steps'].astype(int)
    print(f"Loaded {len(df_good)} good traces.")

    traces = pd.read_csv(TRACES_CSV)
    spots  = pd.read_csv(SPOTS_CSV)
    if '_spot_id' not in spots.columns and 'Index' in spots.columns:
        spots['_spot_id'] = spots['Index'].astype(int)
    print(f"Loaded traces: {traces.shape[1]-1} spots, {traces.shape[0]} frames.")
    print(f"Loaded spots: {len(spots)} rows.")

    unit_step, unit_sigma = compute_unit_step(df_good)
    print(f"Unit step: {unit_step:.4f}  (sigma: {unit_sigma:.4f})")

    df_metrics = compute_metrics(df_good, traces, spots, unit_step)
    df_metrics.to_csv(out / 'investigation_metrics.csv', index=False)

    plot_trace_gallery(df_metrics, traces, unit_step,
                       out / f'{STEPS_TO_INVESTIGATE}_step_traces.png')
    plot_metric_boxplots(df_metrics, out / 'investigation_boxplots.png')
    write_report(df_metrics, unit_step, unit_sigma,
                 out / 'investigation_report.txt')


if __name__ == "__main__":
    main()
