"""
analyze_stoichiometry.py — Step 4 of the smPB pipeline.

Fits truncated binomial distributions to the step-count histogram to
determine protein stoichiometry and the active-fluorophore fraction p.

The binomial is "truncated" because molecules with zero active fluorophores
are invisible and cannot be counted.

INPUTS
  - step_counts.csv or step_counts_filtered.csv (from quality_filter.py)

OUTPUTS  (all saved to OUTPUT_DIR)
  - step_amplitude_histogram.png   Gaussian fit → unitary step size
  - step_count_histogram.png       Raw step-count bar chart
  - binomial_fits.png              One panel per candidate n with fit overlay
  - fit_results.csv                Table of n, p, chi2 for each candidate
  - fit_report.txt                 Human-readable summary

USAGE
  1. Edit USER SETTINGS below.
  2. Run:  python analyze_stoichiometry.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, brentq
from scipy.special import comb
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

STEP_COUNTS_CSV = "/path/to/step_counts_filtered.csv"
OUTPUT_DIR      = "/path/to/output_folder"

CANDIDATE_N = [2, 3, 4, 5]   # Stoichiometries to test
P_FIXED     = None            # Set to a number (e.g. 0.65) to fix p from
                              # a positive control; None = estimate from data


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def truncated_binomial_pmf(x, n, p):
    """P(X = x | X > 0) for X ~ Binomial(n, p).

    The truncation accounts for the fact that molecules with all
    fluorophores dark (X = 0) are invisible.
    """
    pmf = comb(n, x) * p**x * (1 - p)**(n - x)
    truncation = 1 - (1 - p) ** n
    return pmf / truncation


def estimate_p_from_mean(mean_within, n):
    """Find p such that E[X | X>0] = n·p / (1 − (1−p)^n) equals mean_within."""
    def diff(p):
        return n * p / (1 - (1 - p) ** n) - mean_within
    try:
        return brentq(diff, 1e-4, 1 - 1e-4)
    except ValueError:
        return mean_within / n


def collect_amplitudes(df):
    """Pool all per-trace step amplitudes into one flat array."""
    amps = []
    for amp_str in df['step_amplitudes'].dropna().astype(str):
        if not amp_str:
            continue
        for a in amp_str.split('|'):
            try:
                amps.append(float(a))
            except ValueError:
                pass
    return np.array(amps)


def fit_amplitude_histogram(amplitudes, out_path):
    """Plot step amplitudes and fit a Gaussian to determine the unit step."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts, edges, _ = ax.hist(amplitudes, bins=30, alpha=0.6,
                                edgecolor='black', color='steelblue')
    centers = 0.5 * (edges[:-1] + edges[1:])
    try:
        p0 = [np.median(amplitudes), np.std(amplitudes), counts.max()]
        popt, _ = curve_fit(gaussian, centers, counts, p0=p0, maxfev=5000)
        mu, sigma, amp = popt
        x_fit = np.linspace(amplitudes.min(), amplitudes.max(), 300)
        ax.plot(x_fit, gaussian(x_fit, *popt), 'r-', lw=2,
                label=f'Gaussian  \u03bc = {mu:.3f}, \u03c3 = {sigma:.3f}')
        ax.legend()
        unit_step, unit_sigma = mu, sigma
    except Exception as e:
        print(f"  Gaussian fit failed: {e}")
        unit_step = float(np.median(amplitudes))
        unit_sigma = float(np.std(amplitudes))
    ax.set_xlabel('Step amplitude')
    ax.set_ylabel('Number of observations')
    ax.set_title(f'Step amplitudes  (N = {len(amplitudes)})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return unit_step, unit_sigma


def plot_step_count_histogram(step_counts, out_path):
    """Simple bar chart of step counts."""
    fig, ax = plt.subplots(figsize=(8, 5))
    max_x = int(step_counts.max())
    bins = np.arange(0.5, max_x + 1.5, 1)
    ax.hist(step_counts, bins=bins, alpha=0.7, edgecolor='black', color='steelblue')
    ax.set_xticks(np.arange(1, max_x + 1))
    ax.set_xlabel('Step count per molecule')
    ax.set_ylabel('Number of molecules')
    ax.set_title(f'Step counts  (N = {len(step_counts)})')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fit_binomial(step_counts, n, p_fixed=None):
    """Fit truncated binomial for stoichiometry n.

    Observations with step count > n are counted as 'beyond model'.
    Returns a dict with fit parameters and diagnostics.
    """
    within_mask = step_counts <= n
    obs_within = step_counts[within_mask]
    obs_beyond = step_counts[~within_mask]

    counts_within = np.bincount(obs_within, minlength=n + 1)[1:n + 1]
    N_within = counts_within.sum()
    mean_within = obs_within.mean() if N_within > 0 else 0.0

    p = float(p_fixed) if p_fixed is not None else estimate_p_from_mean(mean_within, n)
    x_vals = np.arange(1, n + 1)
    expected = truncated_binomial_pmf(x_vals, n, p) * N_within
    chi2 = np.sum((counts_within - expected) ** 2 / (expected + 1e-9))

    return {
        'n': n, 'p': p,
        'observed_within': counts_within, 'expected_within': expected,
        'N_within': N_within, 'mean_within': mean_within,
        'beyond_count': len(obs_beyond),
        'beyond_pct': 100 * len(obs_beyond) / len(step_counts),
        'chi2': chi2,
    }


def plot_binomial_fits(step_counts, fits, out_path):
    """Multi-panel figure: one panel per candidate n with fit overlay."""
    n_panels = len(fits)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]
    max_x = int(step_counts.max())
    for ax, fit in zip(axes, fits):
        n = fit['n']
        bins = np.arange(0.5, max_x + 1.5, 1)
        ax.hist(step_counts, bins=bins, alpha=0.5,
                edgecolor='black', color='steelblue', label='observed')
        x_vals = np.arange(1, n + 1)
        ax.plot(x_vals, fit['expected_within'], 'rs-',
                markersize=10, lw=1.5, label=f"binomial n={n}")
        title = f"n = {n},  p = {fit['p']:.3f}\n\u03c7\u00b2 = {fit['chi2']:.1f}"
        if fit['beyond_count']:
            title += f"\n{fit['beyond_count']} traces beyond model"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Step count')
        ax.set_xticks(np.arange(1, max_x + 1))
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Number of molecules')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_report(unit_step, unit_sigma, step_counts, fits, out_path):
    """Write a human-readable text report of the analysis."""
    lines = [
        "Stoichiometry analysis report",
        "=" * 50, "",
        f"Number of scored traces (good): {len(step_counts)}",
        f"Mean step count:                {step_counts.mean():.3f}",
        f"Median step count:              {int(np.median(step_counts))}", "",
        "Step amplitude (unitary step size from Gaussian fit):",
        f"  mean (mu):  {unit_step:.4f}",
        f"  width (sigma): {unit_sigma:.4f}", "",
        "Step-count distribution:",
    ]
    max_x = int(step_counts.max())
    for x in range(1, max_x + 1):
        n_x = int((step_counts == x).sum())
        pct = 100 * n_x / len(step_counts)
        lines.append(f"  {x} steps: {n_x:4d}  ({pct:5.1f}%)")
    lines += ["", "Binomial fits (truncated, observed at X >= 1):",
              f"  {'n':<4} {'p':<8} {'mean':<8} {'chi2':<8} {'beyond':<8}"]
    for fit in fits:
        lines.append(f"  {fit['n']:<4} {fit['p']:<8.3f} "
                     f"{fit['mean_within']:<8.3f} {fit['chi2']:<8.1f} "
                     f"{fit['beyond_count']} ({fit['beyond_pct']:.1f}%)")
    lines += ["",
        "Notes:",
        "  - 'beyond' counts observations with x > n; these cannot be",
        "    explained by a pure binomial of stoichiometry n.",
        "  - Lower chi2 is better, but values cannot be directly compared",
        "    across different n.",
        "  - Without a positive control to fix p, the stoichiometry",
        "    cannot be unambiguously identified."]

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
    df = df[df['status'] == 'good'].copy()
    df['n_steps'] = df['n_steps'].astype(int)
    print(f"Loaded {len(df)} good traces.")

    amplitudes = collect_amplitudes(df)
    print(f"Collected {len(amplitudes)} step amplitudes.")

    unit_step, unit_sigma = fit_amplitude_histogram(
        amplitudes, out / 'step_amplitude_histogram.png')
    step_counts = df['n_steps'].values
    plot_step_count_histogram(step_counts, out / 'step_count_histogram.png')

    fits = [fit_binomial(step_counts, n, P_FIXED) for n in CANDIDATE_N]
    plot_binomial_fits(step_counts, fits, out / 'binomial_fits.png')

    pd.DataFrame([{
        'n': f['n'], 'p': f['p'], 'mean_within': f['mean_within'],
        'chi2': f['chi2'], 'N_within': f['N_within'],
        'beyond_count': f['beyond_count'], 'beyond_pct': f['beyond_pct'],
    } for f in fits]).to_csv(out / 'fit_results.csv', index=False)

    write_report(unit_step, unit_sigma, step_counts, fits, out / 'fit_report.txt')


if __name__ == "__main__":
    main()
