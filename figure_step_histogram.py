"""
figure_step_histogram.py — Publication-quality step-count histogram.

Generates a clean bar chart of bleaching step counts with percentage
labels. Saves in PDF, SVG, and PNG formats.

INPUTS
  - step_counts.csv or step_counts_filtered.csv

OUTPUTS  (saved to OUTPUT_DIR)
  - step_count_histogram.pdf/svg/png

USAGE
  1. Edit USER SETTINGS below.
  2. Run:  python figure_step_histogram.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'


# ═══════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

STEP_COUNTS_CSV = "/path/to/step_counts_filtered.csv"
OUTPUT_DIR      = "/path/to/output_folder/figures"
OUTPUT_STEM     = "step_count_histogram"

BAR_COLOR      = "#4682B4"    # Steel blue
BAR_EDGE_COLOR = "black"
SHOW_PERCENT   = True         # Annotate each bar with percentage
FIGSIZE        = (5, 4)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(STEP_COUNTS_CSV)
    df = df[df['status'] == 'good'].copy()
    df['n_steps'] = df['n_steps'].astype(int)
    df = df[df['n_steps'] >= 1]   # Drop any 0-step artefacts

    counts  = df['n_steps'].value_counts().sort_index()
    n_total = counts.sum()
    print(f"N = {n_total} good traces")
    print(counts)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(counts.index, counts.values,
                  color=BAR_COLOR, edgecolor=BAR_EDGE_COLOR, lw=1)

    if SHOW_PERCENT:
        for b, c in zip(bars, counts.values):
            pct = 100 * c / n_total
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.01 * counts.max(),
                    f"{pct:.1f}%", ha='center', va='bottom', fontsize=9)

    ax.set_xlabel("Number of bleaching steps")
    ax.set_ylabel("Number of molecules")
    ax.set_xticks(counts.index)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.97, 0.97, f"N = {n_total}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10)
    fig.tight_layout()

    stem = out / OUTPUT_STEM
    fig.savefig(f"{stem}.pdf", bbox_inches='tight')
    fig.savefig(f"{stem}.svg", bbox_inches='tight')
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {stem}.pdf/.svg/.png")


if __name__ == "__main__":
    main()
