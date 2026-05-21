"""
quality_filter.py — Step 3 of the smPB pipeline.

Filters scored traces for consistency between the scored step count and
the total bleach amplitude. A trace passes if its total intensity drop
(expressed in multiples of the unitary step size) matches the scored
number of steps within a configurable tolerance.

This catches two common artefacts:
  - Over-segmentation: noise scored as extra steps (bleach_units < n_steps)
  - Missed steps: real steps not scored (bleach_units > n_steps)

The original step_counts.csv is NOT modified. Output is written to
step_counts_filtered.csv with additional columns (bleach_units, filter_result).

INPUTS
  - step_counts.csv         from score_steps.py
  - traces_corrected.csv    from extract_traces.py

OUTPUTS  (saved to OUTPUT_DIR)
  - step_counts_filtered.csv    Same format as input but with excluded traces
                                 marked as 'excluded_quality' or 'excluded_zero_step'

USAGE
  1. Edit USER SETTINGS below.
  2. Run:  python quality_filter.py
  3. Point analyze_stoichiometry.py at step_counts_filtered.csv.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

STEP_COUNTS_CSV = "/path/to/step_counts.csv"
TRACES_CSV      = "/path/to/traces_corrected.csv"
OUTPUT_DIR      = "/path/to/output_folder"

N_INIT_FRAMES    = 20     # Frames averaged for initial intensity estimate
N_FINAL_FRAMES   = 50     # Frames averaged for final (baseline) intensity
BLEACH_TOLERANCE = 0.5    # Keep trace if |bleach_units − n_steps| ≤ this
                          # Start with 0.5; increase to 0.7–1.0 if rejecting
                          # good traces (esp. when step amplitude CV > 0.3)


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def parse_amps(s):
    """Parse pipe-separated amplitude string into a numpy array."""
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
    """Fit a Gaussian to all pooled step amplitudes. Return the mean (mu)."""
    all_amps = []
    for s in df_good['step_amplitudes']:
        all_amps.extend(parse_amps(s).tolist())
    all_amps = np.array(all_amps)
    if len(all_amps) == 0:
        raise RuntimeError("No step amplitudes found in good traces.")
    counts, edges = np.histogram(all_amps, bins=30)
    centers = 0.5 * (edges[:-1] + edges[1:])
    try:
        p0 = [np.median(all_amps), np.std(all_amps), counts.max()]
        popt, _ = curve_fit(gaussian, centers, counts, p0=p0, maxfev=5000)
        return float(popt[0])
    except Exception:
        return float(np.median(all_amps))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    df_orig = pd.read_csv(STEP_COUNTS_CSV)
    traces  = pd.read_csv(TRACES_CSV)
    df_good = df_orig[df_orig['status'] == 'good'].copy()
    df_good['n_steps'] = df_good['n_steps'].astype(int)

    unit_step = compute_unit_step(df_good)
    print(f"Unit step (Gaussian fit): {unit_step:.4f}")

    # Compute bleach_units for each good trace
    bleach_units = {}
    for _, row in df_good.iterrows():
        sid = row['spot_id']
        if sid not in traces.columns:
            continue
        trace = traces[sid].values
        initial = float(trace[:N_INIT_FRAMES].mean())
        final   = float(trace[-N_FINAL_FRAMES:].mean())
        bleach_units[sid] = (initial - final) / unit_step

    # Apply filter: mark traces that fail as excluded
    df_filtered = df_orig.copy()
    df_filtered['bleach_units']  = np.nan
    df_filtered['filter_result'] = ''

    n_zero = n_excluded = n_kept = n_no_trace = 0
    for i, row in df_filtered.iterrows():
        if row['status'] != 'good':
            continue
        sid = row['spot_id']
        n   = int(row['n_steps'])
        bu  = bleach_units.get(sid, np.nan)
        df_filtered.at[i, 'bleach_units'] = bu

        if n == 0:
            df_filtered.at[i, 'status'] = 'excluded_zero_step'
            df_filtered.at[i, 'filter_result'] = 'zero_step_artefact'
            n_zero += 1
        elif np.isnan(bu):
            df_filtered.at[i, 'status'] = 'excluded_no_trace'
            df_filtered.at[i, 'filter_result'] = 'no_trace_data'
            n_no_trace += 1
        elif abs(bu - n) > BLEACH_TOLERANCE:
            df_filtered.at[i, 'status'] = 'excluded_quality'
            df_filtered.at[i, 'filter_result'] = f'bleach_units_off (bu={bu:.2f}, n={n})'
            n_excluded += 1
        else:
            df_filtered.at[i, 'filter_result'] = 'pass'
            n_kept += 1

    out_path = out / 'step_counts_filtered.csv'
    df_filtered.to_csv(out_path, index=False)

    # Print report
    n_good_before = (df_orig['status'] == 'good').sum()
    print(f"\nFilter report")
    print(f"  Tolerance:                  |bleach_units - n_steps| <= {BLEACH_TOLERANCE}")
    print(f"  Good traces before filter:  {n_good_before}")
    print(f"  Good traces after filter:   {n_kept}")
    print(f"  Excluded zero-step:         {n_zero}")
    print(f"  Excluded no-trace:          {n_no_trace}")
    print(f"  Excluded by bleach_units:   {n_excluded}")

    print(f"\n  Per n_steps breakdown (good only):")
    print(f"  {'n':<4} {'before':<8} {'after':<8} {'kept_%':<8}")
    df_g = df_orig[df_orig['status'] == 'good'].copy()
    df_g['n_steps'] = df_g['n_steps'].astype(int)
    for n in sorted(df_g['n_steps'].unique()):
        before = (df_g['n_steps'] == n).sum()
        after  = ((df_filtered['status'] == 'good') &
                  (df_filtered['n_steps'].astype(int) == n)).sum()
        pct = 100 * after / before if before else 0
        print(f"  {n:<4} {before:<8} {after:<8} {pct:<8.0f}")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
