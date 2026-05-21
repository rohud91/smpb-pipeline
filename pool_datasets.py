"""
utils/pool_datasets.py — Pool step_counts CSVs from multiple fields of view.

When multiple image fields were acquired from the same sample (technical
replicates), this script concatenates their step_counts files into a
single pooled CSV. A 'field' column is added so per-field statistics
can be checked later.

USAGE
  1. Set FILES and OUTPUT_PATH below.
  2. Run:  python utils/pool_datasets.py
"""

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

FILES = [
    "/path/to/field1/step_counts_filtered.csv",
    "/path/to/field2/step_counts_filtered.csv",
    "/path/to/field3/step_counts_filtered.csv",
]

OUTPUT_PATH = "/path/to/output/step_counts_pooled.csv"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    dfs = []
    for i, f in enumerate(FILES, 1):
        df = pd.read_csv(f)
        df['field'] = f"field_{i}"
        dfs.append(df)
        print(f"  field_{i}: {len(df)} rows from {f}")

    pooled = pd.concat(dfs, ignore_index=True)
    pooled.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPooled {len(pooled)} traces across {len(FILES)} fields.")
    print(f"Saved: {OUTPUT_PATH}")
    print("\nPer-field status counts:")
    print(pooled.groupby('field')['status'].value_counts())


if __name__ == "__main__":
    main()
