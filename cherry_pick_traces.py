"""
utils/cherry_pick_traces.py — Find the cleanest traces for figure panels.

Ranks all good traces by a "cleanliness" score that penalises deviation
from the unitary step amplitude and within-trace amplitude variability.
Prints the top candidates for each step count.

USAGE
  1. Set STEP_COUNTS_CSV and UNIT_STEP below.
  2. Run:  python utils/cherry_pick_traces.py
"""

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

STEP_COUNTS_CSV = "/path/to/step_counts_filtered.csv"
UNIT_STEP       = 0.83    # From Gaussian fit (analyze_stoichiometry.py output)
TOP_N           = 3       # How many candidates to show per step count
MAX_STEPS       = 3       # Highest step count to consider


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    df = pd.read_csv(STEP_COUNTS_CSV)
    df = df[df['status'] == 'good'].copy()

    def parse(s):
        return [float(x) for x in str(s).split('|') if x]

    df['amps']     = df['step_amplitudes'].apply(parse)
    df['amp_mean'] = df['amps'].apply(lambda a: sum(a)/len(a) if a else 0)
    df['amp_std']  = df['amps'].apply(
        lambda a: pd.Series(a).std() if len(a) > 1 else 0)

    # Cleanliness = distance from unit step + within-trace variability
    df['cleanliness'] = abs(df['amp_mean'] - UNIT_STEP) + df['amp_std']
    df_sorted = df.sort_values('cleanliness')

    for n in range(1, MAX_STEPS + 1):
        best = df_sorted[df_sorted['n_steps'] == n].head(TOP_N)
        print(f"\nBest {n}-step traces:")
        if len(best) == 0:
            print("  (none)")
        else:
            print(best[['spot_id', 'amp_mean', 'amp_std', 'cleanliness']]
                  .to_string(index=False))


if __name__ == "__main__":
    main()
