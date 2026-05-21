"""
utils/fix_csv_separators.py — One-time fix for semicolon locale issues.

Early versions of score_steps.py used semicolons as the inner separator
for step_frames and step_amplitudes columns. This causes problems in
Excel with European locale settings (where semicolons are CSV delimiters).

This script converts semicolons to pipes '|' inside those columns.
Run it once on any affected step_counts.csv files. A backup is created
automatically.

USAGE
  1. Set CSV_PATH below.
  2. Run:  python utils/fix_csv_separators.py
"""

import pandas as pd
import shutil
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

CSV_PATH = "/path/to/step_counts.csv"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    p = Path(CSV_PATH)
    backup = p.with_name(p.stem + "_backup.csv")
    shutil.copy(p, backup)
    print(f"Backup saved to: {backup}")

    df = pd.read_csv(p)
    df['step_frames'] = (df['step_frames'].fillna('').astype(str)
                         .str.replace(';', '|', regex=False))
    df['step_amplitudes'] = (df['step_amplitudes'].fillna('').astype(str)
                             .str.replace(';', '|', regex=False))
    df.to_csv(p, index=False)
    print(f"Converted: {p}")
    print(f"Rows processed: {len(df)}")


if __name__ == "__main__":
    main()
