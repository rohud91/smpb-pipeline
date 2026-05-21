"""
diagnose_quality_filter.py — Diagnostic tool for quality_filter.py

Analyzes why traces are being rejected by the quality filter and
recommends an appropriate BLEACH_TOLERANCE value based on your data.

Run this if quality_filter.py rejects >50% of your manually scored traces.

USAGE
  1. Edit USER SETTINGS below (same paths as quality_filter.py).
  2. Run:  python diagnose_quality_filter.py
  3. Check the output report and quality_filter_diagnostics.png.
  4. Adjust BLEACH_TOLERANCE in quality_filter.py based on recommendation.

OUTPUTS
  - Terminal report with rejection statistics
  - quality_filter_diagnostics.png (4-panel diagnostic figure)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path


# ════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ════════════════════════════════════════════════════════════════════════════

STEP_COUNTS_CSV = "/path/to/step_counts.csv"
TRACES_CSV      = "/path/to/traces_corrected.csv"

N_INIT_FRAMES   = 20
N_FINAL_FRAMES  = 50


# ════════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def parse_amps(s):
    """Parse pipe-separated amplitude string."""
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
    """Fit Gaussian to pooled step amplitudes."""
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


def analyze_filter_behavior(step_counts_csv, traces_csv):
    """Run full diagnostic on quality filter rejections."""
    
    df_orig = pd.read_csv(step_counts_csv)
    traces  = pd.read_csv(traces_csv)
    df_good = df_orig[df_orig['status'] == 'good'].copy()
    df_good['n_steps'] = df_good['n_steps'].astype(int)
    
    print(f"Loaded {len(df_good)} good traces from scoring.")
    
    # Compute unit step
    unit_step, unit_sigma = compute_unit_step(df_good)
    print(f"\nUnit step: μ = {unit_step:.4f}, σ = {unit_sigma:.4f}")
    print(f"Coefficient of variation: σ/μ = {unit_sigma/unit_step:.3f}")
    
    # Compute bleach_units for all good traces
    results = []
    for _, row in df_good.iterrows():
        sid = row['spot_id']
        n_steps = int(row['n_steps'])
        
        if sid not in traces.columns:
            continue
        
        trace = traces[sid].values
        initial = float(trace[:N_INIT_FRAMES].mean())
        final   = float(trace[-N_FINAL_FRAMES:].mean())
        total_bleach = initial - final
        bleach_units = total_bleach / unit_step
        
        deviation = bleach_units - n_steps
        abs_dev = abs(deviation)
        
        results.append({
            'spot_id': sid,
            'n_steps': n_steps,
            'initial': initial,
            'final': final,
            'total_bleach': total_bleach,
            'bleach_units': bleach_units,
            'deviation': deviation,
            'abs_deviation': abs_dev,
        })
    
    df_analysis = pd.DataFrame(results)
    
    # Show rejection rates at different tolerance levels
    print("\n" + "="*70)
    print("REJECTION ANALYSIS AT DIFFERENT TOLERANCE LEVELS")
    print("="*70)
    
    tolerances = [0.3, 0.5, 0.7, 1.0, 1.5]
    for tol in tolerances:
        n_pass = (df_analysis['abs_deviation'] <= tol).sum()
        n_reject = len(df_analysis) - n_pass
        pct_keep = 100 * n_pass / len(df_analysis)
        print(f"\nTolerance = ±{tol:.1f}")
        print(f"  Pass:   {n_pass:3d} traces ({pct_keep:5.1f}%)")
        print(f"  Reject: {n_reject:3d} traces ({100-pct_keep:5.1f}%)")
    
    # Show per-n breakdown at current 0.5 tolerance
    print("\n" + "="*70)
    print("REJECTION BY STEP COUNT (tolerance = 0.5)")
    print("="*70)
    print(f"{'n':<4} {'total':<8} {'pass':<8} {'reject':<8} {'keep_%':<8}")
    
    for n in sorted(df_analysis['n_steps'].unique()):
        df_n = df_analysis[df_analysis['n_steps'] == n]
        n_total = len(df_n)
        n_pass = (df_n['abs_deviation'] <= 0.5).sum()
        n_reject = n_total - n_pass
        pct = 100 * n_pass / n_total if n_total else 0
        print(f"{n:<4} {n_total:<8} {n_pass:<8} {n_reject:<8} {pct:<8.0f}")
    
    # Show worst rejections
    df_rejected = df_analysis[df_analysis['abs_deviation'] > 0.5].copy()
    df_rejected = df_rejected.sort_values('abs_deviation', ascending=False)
    
    print("\n" + "="*70)
    print("TOP 10 REJECTIONS (|bleach_units - n_steps| > 0.5)")
    print("="*70)
    print(f"{'spot_id':<12} {'n':<4} {'bleach_u':<10} {'dev':<8} {'issue':<30}")
    
    for _, row in df_rejected.head(10).iterrows():
        issue = "over-bleached" if row['deviation'] > 0 else "under-bleached"
        print(f"{row['spot_id']:<12} {row['n_steps']:<4} "
              f"{row['bleach_units']:<10.2f} {row['deviation']:<8.2f} {issue:<30}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: deviation distribution
    ax = axes[0, 0]
    ax.hist(df_analysis['deviation'], bins=40, edgecolor='black', alpha=0.7)
    ax.axvline(-0.5, color='red', ls='--', lw=1.5, label='tolerance = ±0.5')
    ax.axvline(0.5, color='red', ls='--', lw=1.5)
    ax.axvline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('bleach_units - n_steps')
    ax.set_ylabel('Number of traces')
    ax.set_title('Deviation distribution')
    ax.legend()
    
    # Panel 2: deviation vs n_steps
    ax = axes[0, 1]
    for n in sorted(df_analysis['n_steps'].unique()):
        df_n = df_analysis[df_analysis['n_steps'] == n]
        ax.scatter([n]*len(df_n), df_n['deviation'], alpha=0.5, s=30)
    ax.axhline(-0.5, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.axhline(0.5, color='red', ls='--', lw=1.5, alpha=0.7)
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('n_steps')
    ax.set_ylabel('bleach_units - n_steps')
    ax.set_title('Deviation by step count')
    ax.set_xticks(sorted(df_analysis['n_steps'].unique()))
    
    # Panel 3: bleach_units vs n_steps
    ax = axes[1, 0]
    ax.scatter(df_analysis['n_steps'], df_analysis['bleach_units'], 
               alpha=0.5, s=30)
    n_range = [df_analysis['n_steps'].min() - 0.5, 
               df_analysis['n_steps'].max() + 0.5]
    ax.plot(n_range, n_range, 'k-', lw=1, label='ideal (1:1)')
    ax.plot(n_range, [x - 0.5 for x in n_range], 'r--', lw=1.5, alpha=0.7)
    ax.plot(n_range, [x + 0.5 for x in n_range], 'r--', lw=1.5, alpha=0.7, 
            label='tolerance = ±0.5')
    ax.set_xlabel('n_steps (scored)')
    ax.set_ylabel('bleach_units (intensity)')
    ax.set_title('Scored steps vs bleach magnitude')
    ax.legend()
    
    # Panel 4: cumulative retention vs tolerance
    ax = axes[1, 1]
    tol_range = np.linspace(0, 2, 100)
    retention = [100 * (df_analysis['abs_deviation'] <= t).sum() / len(df_analysis) 
                 for t in tol_range]
    ax.plot(tol_range, retention, 'b-', lw=2)
    ax.axvline(0.5, color='red', ls='--', lw=1.5, alpha=0.7, label='current (0.5)')
    ax.axhline(90, color='gray', ls=':', lw=1)
    ax.axhline(95, color='gray', ls=':', lw=1)
    ax.set_xlabel('Tolerance (|bleach_units - n_steps|)')
    ax.set_ylabel('% traces retained')
    ax.set_title('Retention vs tolerance')
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig('/home/claude/quality_filter_diagnostics.png', dpi=150)
    plt.close()
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Mean |deviation|:   {df_analysis['abs_deviation'].mean():.3f}")
    print(f"Median |deviation|: {df_analysis['abs_deviation'].median():.3f}")
    print(f"95th percentile:    {df_analysis['abs_deviation'].quantile(0.95):.3f}")
    
    # Recommendation
    q95 = df_analysis['abs_deviation'].quantile(0.95)
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"Your current tolerance (0.5) rejects {100 - 100*(df_analysis['abs_deviation'] <= 0.5).sum()/len(df_analysis):.1f}% of scored traces.")
    print(f"\nWith σ/μ = {unit_sigma/unit_step:.3f}, step size variability is substantial.")
    print(f"The 95th percentile deviation is {q95:.2f}.")
    print(f"\nSuggested tolerance: {max(0.7, round(q95, 1)):.1f}")
    print(f"  → Would retain {100*(df_analysis['abs_deviation'] <= max(0.7, round(q95, 1))).sum()/len(df_analysis):.1f}% of traces")
    
    return df_analysis


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    df = analyze_filter_behavior(STEP_COUNTS_CSV, TRACES_CSV)
    print(f"\nDiagnostic plot saved: quality_filter_diagnostics.png")
    print("\nFull analysis table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
