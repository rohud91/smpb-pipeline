"""
extract_traces.py — Step 1 of the smPB pipeline.

Reads a ComDet spot-detection CSV and a multi-frame TIFF image stack.
For each spot passing size and shape filters, extracts a per-frame
fluorescence trace from a small ROI with annular local background
subtraction.

INPUTS
  - ComDet results CSV (columns: Index, X_(px), Y_(px), xMin, yMin,
    xMax, yMax, NArea)
  - Multi-frame TIFF stack (frames × Y × X)

OUTPUTS  (all saved to OUTPUT_DIR)
  - traces_corrected.csv   Background-subtracted traces (one column per spot)
  - traces_raw.csv         Raw ROI intensities before background subtraction
  - traces_background.csv  Local background estimates per frame
  - spots_used.csv         Metadata for spots that passed all filters
  - example_traces.png     Random selection of traces for visual QC

SPOT ID CONVENTION
  Each spot is named 'spot_NNNN' where NNNN is the ComDet Index value.
  This ensures the same physical spot keeps the same ID across
  re-extractions, regardless of filter changes.

USAGE
  1. Edit the USER SETTINGS block below.
  2. Run:  python extract_traces.py
  3. Inspect example_traces.png before proceeding to score_steps.py.
"""

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# USER SETTINGS — edit these for your dataset
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH   = "/path/to/comdet_results.csv"
STACK_PATH = "/path/to/image_stack.tif"
OUTPUT_DIR = "/path/to/output_folder"

# Spot filter (pixel-based).
# For GFP, 1.4 NA, 0.11 µm/px: Airy disk ≈ 4 px → area ≈ 8–12 px²
AREA_MIN = 5       # Minimum NArea (px²) — excludes noise
AREA_MAX = 15      # Maximum NArea (px²) — excludes clusters
AR_MAX   = 1.5     # Maximum aspect ratio — excludes elongated shapes

# ROI and background geometry (pixels)
ROI_HALF = 2       # ROI = (2×ROI_HALF+1)² = 5×5 pixels
BG_INNER = 4       # Inner radius of background annulus
BG_OUTER = 7       # Outer radius of background annulus

N_EXAMPLE_TRACES = 6

# Columns to try (in order) as stable spot ID
ID_COLUMN_CANDIDATES = [
    'Index', 'ID', 'id', 'TRACK_ID', 'TrackID', 'Label', 'label', 'spot_id'
]


# ═══════════════════════════════════════════════════════════════════════════
# FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def find_id_column(df):
    """Return the first column that is unique and matches a known ID name."""
    for cand in ID_COLUMN_CANDIDATES:
        if cand in df.columns and df[cand].is_unique:
            return cand
    return None


def load_and_filter_spots(csv_path):
    """Load ComDet CSV, assign stable spot IDs, filter by size and shape."""
    df = pd.read_csv(csv_path, sep=None, engine='python')
    n_start = len(df)
    print(f"Loaded {n_start} spots. Columns: {list(df.columns)}")

    # Assign stable spot ID from ComDet's Index column
    id_col = find_id_column(df)
    if id_col is None:
        print("No unique ID column found. Using row number as spot ID.")
        df['_spot_id'] = np.arange(1, len(df) + 1)
    else:
        print(f"Using '{id_col}' as stable spot ID.")
        df['_spot_id'] = df[id_col].astype(int)

    # Pixel coordinates
    if 'X_(px)' in df.columns and 'Y_(px)' in df.columns:
        df['x_px'] = df['X_(px)'].round().astype(int)
        df['y_px'] = df['Y_(px)'].round().astype(int)
    else:
        raise ValueError(
            f"Expected 'X_(px)' and 'Y_(px)' columns. Got: {list(df.columns)}")

    # Shape descriptors from bounding box
    for c in ['xMin', 'xMax', 'yMin', 'yMax', 'NArea']:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    width  = df['xMax'] - df['xMin'] + 1
    height = df['yMax'] - df['yMin'] + 1
    df['_area'] = df['NArea']
    df['_ar']   = np.maximum(width, height) / np.minimum(width, height)

    # Apply filter
    keep = (
        (df['_area'] >= AREA_MIN) & (df['_area'] <= AREA_MAX) &
        (df['_ar']   <= AR_MAX)
    )
    filtered = df[keep].reset_index(drop=True)
    print(f"Spots before filter: {n_start}")
    print(f"Spots after filter:  {len(filtered)} "
          f"({100 * len(filtered) / n_start:.0f}%)")

    if len(filtered) == 0:
        print("\nFilter summary (to help tune cutoffs):")
        print(f"  NArea range: {df['_area'].min():.0f}–{df['_area'].max():.0f}, "
              f"median {df['_area'].median():.1f}")
        print(f"  AR    range: {df['_ar'].min():.2f}–{df['_ar'].max():.2f}, "
              f"median {df['_ar'].median():.2f}")
    return filtered


def extract_traces(stack, spots,
                   roi_half=ROI_HALF, bg_inner=BG_INNER, bg_outer=BG_OUTER):
    """Extract per-spot traces with annular background subtraction.

    Returns (signal, background, corrected, used_ids) where each matrix
    has shape (n_frames, n_spots) and used_ids is a list of spot ID ints.
    """
    n_frames, ny, nx = stack.shape

    # Pre-compute ROI and background masks relative to spot center
    yy, xx = np.mgrid[-bg_outer:bg_outer+1, -bg_outer:bg_outer+1]
    r = np.sqrt(yy**2 + xx**2)
    roi_mask = (np.abs(yy) <= roi_half) & (np.abs(xx) <= roi_half)
    bg_mask  = (r >= bg_inner) & (r <= bg_outer)

    signals, backgrounds, used_ids = [], [], []
    for _, row in spots.iterrows():
        x, y = int(row['x_px']), int(row['y_px'])
        spot_id = int(row['_spot_id'])

        # Skip spots too close to the edge for the background annulus
        if (x - bg_outer < 0 or x + bg_outer >= nx or
            y - bg_outer < 0 or y + bg_outer >= ny):
            continue

        box = stack[:, y-bg_outer:y+bg_outer+1, x-bg_outer:x+bg_outer+1]
        signals.append(box[:, roi_mask].mean(axis=1))
        backgrounds.append(box[:, bg_mask].mean(axis=1))
        used_ids.append(spot_id)

    if not signals:
        raise RuntimeError("No spots survived filter + edge rejection.")

    signal     = np.array(signals).T
    background = np.array(backgrounds).T
    corrected  = signal - background
    print(f"Spots extracted (after edge rejection): {len(used_ids)}")
    return signal, background, corrected, used_ids


def save_traces(matrix, used_ids, output_dir, stem):
    """Save traces matrix as CSV with spot_NNNN column names."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    columns = [f"spot_{sid:04d}" for sid in used_ids]
    df = pd.DataFrame(matrix, columns=columns)
    df.insert(0, "frame", np.arange(len(df)))
    csv_path = out / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def plot_examples(corrected, used_ids, output_dir, n=N_EXAMPLE_TRACES):
    """Plot random traces for visual quality check."""
    out = Path(output_dir)
    n_show = min(n, corrected.shape[1])
    fig, axes = plt.subplots(n_show, 1, figsize=(8, 1.6 * n_show), sharex=True)
    if n_show == 1:
        axes = [axes]
    rng = np.random.default_rng(0)
    chosen = rng.choice(corrected.shape[1], size=n_show, replace=False)
    for ax, idx in zip(axes, chosen):
        ax.plot(corrected[:, idx], lw=0.7, color='k')
        ax.set_ylabel(f"spot_{used_ids[idx]:04d}", fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlabel("Frame")
    fig.suptitle("Example background-subtracted traces", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "example_traces.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out / 'example_traces.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    spots = load_and_filter_spots(CSV_PATH)

    print(f"\nLoading stack: {STACK_PATH}")
    stack = tifffile.imread(STACK_PATH)
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D (frames, Y, X); got shape {stack.shape}")
    print(f"Stack shape: {stack.shape}")

    signal, background, corrected, used_ids = extract_traces(stack, spots)
    save_traces(corrected,  used_ids, OUTPUT_DIR, "traces_corrected")
    save_traces(signal,     used_ids, OUTPUT_DIR, "traces_raw")
    save_traces(background, used_ids, OUTPUT_DIR, "traces_background")
    plot_examples(corrected, used_ids, OUTPUT_DIR)

    # Save spot metadata (used by investigate_steps.py for spatial analysis)
    spots_used = spots[spots['_spot_id'].isin(used_ids)].copy()
    spots_used.to_csv(Path(OUTPUT_DIR) / "spots_used.csv", index=False)
    print(f"Saved: {Path(OUTPUT_DIR) / 'spots_used.csv'}")


if __name__ == "__main__":
    main()
