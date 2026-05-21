"""
figure_step_bleaching.py — Publication-quality bleaching trace figures.

Plots single-molecule traces with horizontal bars at each plateau mean
(Pan et al. 2019 style). Saves individual and combined multi-panel
figures in PDF, SVG, and PNG formats.

INPUTS
  - traces_corrected.csv     from extract_traces.py
  - step_counts.csv          from score_steps.py (or filtered version)

OUTPUTS  (saved to OUTPUT_DIR)
  - {spot_id}_pan_style.pdf/svg/png   One figure per spot
  - pan_style_combined.pdf/svg/png    All spots in one multi-panel figure

USAGE
  1. Set SPOTS_TO_PLOT to the spot IDs of your best representative traces.
  2. Run:  python figure_step_bleaching.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Editable text in vector outputs (for Inkscape / Illustrator)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


# ═══════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

TRACES_CSV      = "/path/to/traces_corrected.csv"
STEP_COUNTS_CSV = "/path/to/step_counts_filtered.csv"
OUTPUT_DIR      = "/path/to/output_folder/figures"

# Spot IDs to plot — pick your cleanest examples per step count
SPOTS_TO_PLOT = [
    "spot_0506",   # example 1-step
    "spot_2717",   # example 2-step
    "spot_2931",   # example 3-step
]

# Styling
LINE_COLOR    = "green"    # Trace colour
LINE_WIDTH    = 0.6
BAR_COLOR     = "black"    # Horizontal plateau-mean bars
BAR_WIDTH     = 2.0
SHOW_ARROWS   = False      # Add 'step1', 'step2' arrow labels
FIGSIZE_PER   = (5, 3)     # Size per panel
FRAME_RATE_HZ = None       # Set to a number (e.g. 4) for x-axis in seconds


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def parse_clicks(s):
    """Parse pipe-separated frame indices from step_frames column."""
    if pd.isna(s) or s == '':
        return []
    return [int(x) for x in str(s).split('|') if x]


def plateau_means(trace, click_frames):
    """Compute the mean intensity of each plateau between bleaching events.

    Returns list of (start_frame, end_frame, mean_intensity) tuples.
    """
    n = len(trace)
    boundaries = [0] + sorted(click_frames) + [n]
    means = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b - a >= 1:
            means.append((a, b, float(np.mean(trace[a:b]))))
    return means


def plot_one_trace(ax, trace, click_frames, title=""):
    """Draw a trace with horizontal bars at each plateau mean."""
    x = np.arange(len(trace))
    if FRAME_RATE_HZ:
        x = x / FRAME_RATE_HZ
    ax.plot(x, trace, color=LINE_COLOR, lw=LINE_WIDTH)

    # Draw plateau mean bars
    means = plateau_means(trace, click_frames)
    for a, b, m in means:
        xa = a / FRAME_RATE_HZ if FRAME_RATE_HZ else a
        xb = b / FRAME_RATE_HZ if FRAME_RATE_HZ else b
        ax.plot([xa, xb], [m, m], color=BAR_COLOR, lw=BAR_WIDTH,
                solid_capstyle='butt')

    # Optional step arrows
    if SHOW_ARROWS:
        for i, c in enumerate(sorted(click_frames), start=1):
            xc = c / FRAME_RATE_HZ if FRAME_RATE_HZ else c
            y_at_click = trace[max(0, c-1)]
            y_offset = 0.15 * (np.max(trace) - np.min(trace))
            ax.annotate(f"step{i}", xy=(xc, y_at_click),
                        xytext=(xc, y_at_click + y_offset),
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                        fontsize=9, ha='center')

    ax.set_xlabel("Time (s)" if FRAME_RATE_HZ else "Frame")
    ax.set_ylabel("Intensity (a.u.)")
    if title:
        ax.set_title(title, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    traces = pd.read_csv(TRACES_CSV)
    scores = pd.read_csv(STEP_COUNTS_CSV)

    # --- Individual figures per spot ---
    for spot_id in SPOTS_TO_PLOT:
        if spot_id not in traces.columns:
            print(f"Skipping {spot_id}: not in traces CSV.")
            continue
        row = scores[scores['spot_id'] == spot_id]
        if row.empty:
            print(f"Skipping {spot_id}: not in step_counts CSV.")
            continue
        clicks  = parse_clicks(row.iloc[0]['step_frames'])
        n_steps = int(row.iloc[0]['n_steps'])

        fig, ax = plt.subplots(figsize=FIGSIZE_PER)
        plot_one_trace(ax, traces[spot_id].values, clicks,
                       title=f"{n_steps}-step")
        fig.tight_layout()
        stem = out / f"{spot_id}_pan_style"
        fig.savefig(f"{stem}.pdf", bbox_inches='tight')
        fig.savefig(f"{stem}.svg", bbox_inches='tight')
        fig.savefig(f"{stem}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {stem}.pdf/.svg/.png")

    # --- Combined multi-panel ---
    n = len(SPOTS_TO_PLOT)
    fig, axes = plt.subplots(n, 1, figsize=(FIGSIZE_PER[0], FIGSIZE_PER[1] * n))
    if n == 1:
        axes = [axes]
    for ax, spot_id in zip(axes, SPOTS_TO_PLOT):
        if spot_id not in traces.columns:
            ax.set_visible(False)
            continue
        row = scores[scores['spot_id'] == spot_id]
        if row.empty:
            ax.set_visible(False)
            continue
        clicks  = parse_clicks(row.iloc[0]['step_frames'])
        n_steps = int(row.iloc[0]['n_steps'])
        plot_one_trace(ax, traces[spot_id].values, clicks,
                       title=f"{n_steps}-step")
    fig.tight_layout()
    stem = out / "pan_style_combined"
    fig.savefig(f"{stem}.pdf", bbox_inches='tight')
    fig.savefig(f"{stem}.svg", bbox_inches='tight')
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined: {stem}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
