"""
score_steps.py — Step 2 of the smPB pipeline.

Interactive matplotlib tool for manually counting bleaching steps in
single-molecule fluorescence traces.

WORKFLOW
  1. Edit the USER SETTINGS block below.
  2. Run:  python score_steps.py
  3. A matplotlib window opens showing one trace at a time.
  4. Click on each bleaching transition, then press Enter to save.
  5. Press q to quit. Progress is saved after every trace.

CONTROLS
  left click    Mark a bleaching step at the clicked frame
  u             Undo last click
  c             Clear all clicks for this trace
  Enter / n     Save trace as "good" and advance to next
  Space         Skip trace (mark as unusable)
  b             Go back to previous trace (clicks restored)
  s             Save current view as PNG to gallery/
  j             Jump to a specific spot ID (prompted in terminal)
  q             Quit (all progress auto-saved)

OUTPUTS  (saved to OUTPUT_DIR)
  step_counts.csv   One row per scored spot with columns:
                      spot_id, n_steps, step_frames, step_amplitudes, status

NOTES
  - The CSV uses pipe '|' as the inner separator for step_frames and
    step_amplitudes to avoid conflicts with CSV commas and Czech Excel
    semicolons.
  - Traces are presented in seeded random order (RANDOM_SEED) so the
    sequence is reproducible across sessions.
  - Pre-screening filters out traces that don't show bleaching behaviour
    (configurable via PRESCREEN_DROP_MIN and PRESCREEN_SNR_MIN).

USAGE
  python score_steps.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from pathlib import Path
import sys


# ═══════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

TRACES_CSV = "/path/to/traces_corrected.csv"
OUTPUT_DIR = "/path/to/output_folder"

TARGET_TRACES      = 300    # Display target in title bar (scoring continues past it)
MEDIAN_WINDOW      = 3      # Median filter for display (must be odd; raw data unchanged)
PRESCREEN_DROP_MIN = 0.5    # Min normalised intensity drop to be a candidate
PRESCREEN_SNR_MIN  = 1.5    # Min drop / baseline-noise ratio
RANDOM_SEED        = 42     # Seed for reproducible random presentation order


# ═══════════════════════════════════════════════════════════════════════════
# SCORER CLASS
# ═══════════════════════════════════════════════════════════════════════════

class StepScorer:
    def __init__(self, traces_csv, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gallery_dir = self.output_dir / "gallery"
        self.gallery_dir.mkdir(exist_ok=True)
        self.results_csv = self.output_dir / "step_counts.csv"

        # ---- Load traces ----
        df = pd.read_csv(traces_csv)
        if 'frame' in df.columns:
            self.frames = df['frame'].values
            self.spot_cols = [c for c in df.columns if c != 'frame']
        else:
            self.frames = np.arange(len(df))
            self.spot_cols = list(df.columns)
        self.traces = df[self.spot_cols].values   # shape: (n_frames, n_spots)
        self.n_frames, self.n_spots = self.traces.shape
        print(f"Loaded {self.n_spots} traces of {self.n_frames} frames each.")

        # ---- Pre-screen: keep only traces showing bleaching ----
        n_init  = min(20, self.n_frames // 10)
        n_final = min(50, self.n_frames // 5)
        init  = self.traces[:n_init].mean(axis=0)
        final = self.traces[-n_final:].mean(axis=0)
        drop  = init - final
        noise = self.traces[-n_final:].std(axis=0) + 1e-10
        snr   = drop / noise
        keep_mask = (drop > PRESCREEN_DROP_MIN) & (snr > PRESCREEN_SNR_MIN)
        candidate_indices = np.where(keep_mask)[0]
        print(f"Pre-screen: {len(candidate_indices)} of {self.n_spots} traces "
              f"show bleaching (drop>{PRESCREEN_DROP_MIN}, SNR>{PRESCREEN_SNR_MIN}).")

        # ---- Randomise presentation order (seeded) ----
        rng = np.random.default_rng(RANDOM_SEED)
        rng.shuffle(candidate_indices)
        self.order = candidate_indices

        # ---- Resume from previous session ----
        self.results = self._load_existing()
        scored_ids = {r['spot_id'] for r in self.results}
        self.position = 0
        for i, idx in enumerate(self.order):
            if self.spot_cols[idx] not in scored_ids:
                self.position = i
                break
        else:
            self.position = len(self.order)
        print(f"Already scored: {len(self.results)}.  Target: {TARGET_TRACES}.")

        # ---- Per-trace state ----
        self.click_frames = []
        self.step_marker_lines = []

        # ---- Set up matplotlib ----
        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.show_current()
        plt.show()

    # ──────────────── I/O ────────────────

    def _load_existing(self):
        if self.results_csv.exists():
            df = pd.read_csv(self.results_csv).fillna('')
            return df.to_dict('records')
        return []

    def save_results(self):
        if self.results:
            pd.DataFrame(self.results).to_csv(self.results_csv, index=False)

    # ──────────────── helpers ────────────────

    def current_spot_idx(self):
        if self.position >= len(self.order):
            return None
        return self.order[self.position]

    def current_traces(self):
        idx = self.current_spot_idx()
        if idx is None:
            return None, None
        raw = self.traces[:, idx]
        smooth = medfilt(raw, kernel_size=MEDIAN_WINDOW)
        return raw, smooth

    def _find_existing_record(self, spot_name):
        for r in self.results:
            if r['spot_id'] == spot_name:
                return r
        return None

    # ──────────────── drawing ────────────────

    def show_current(self):
        """Draw the current trace with any existing click markers."""
        self.ax.clear()
        self.step_marker_lines = []
        idx = self.current_spot_idx()
        if idx is None:
            self.ax.set_title("All candidate traces scored. Press q to quit.")
            self.fig.canvas.draw_idle()
            return
        spot_name = self.spot_cols[idx]

        # Restore clicks from a previous scoring of this trace
        self.click_frames = []
        existing = self._find_existing_record(spot_name)
        if existing and existing.get('status') == 'good':
            sf = str(existing.get('step_frames', ''))
            if sf:
                self.click_frames = [int(f) for f in sf.split('|') if f]

        raw, smooth = self.current_traces()
        self.ax.plot(self.frames, raw,    color='lightgray', lw=0.5, label='raw')
        self.ax.plot(self.frames, smooth, color='black',     lw=1.0,
                     label=f'median-{MEDIAN_WINDOW}')
        self.ax.axhline(0, color='red', lw=0.5, alpha=0.3)
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Intensity (background-subtracted)')
        self.ax.legend(loc='upper right')

        for f in self.click_frames:
            ln = self.ax.axvline(f, color='red', lw=1.2, alpha=0.7)
            self.step_marker_lines.append(ln)
        self._update_title()
        self.fig.canvas.draw_idle()

    def _redraw_clicks(self):
        for ln in self.step_marker_lines:
            try:
                ln.remove()
            except (ValueError, AttributeError):
                pass
        self.step_marker_lines = []
        for f in self.click_frames:
            ln = self.ax.axvline(f, color='red', lw=1.2, alpha=0.7)
            self.step_marker_lines.append(ln)
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        idx = self.current_spot_idx()
        if idx is None:
            return
        spot_name = self.spot_cols[idx]
        n_good = sum(1 for r in self.results if r.get('status') == 'good')
        n_skip = sum(1 for r in self.results if r.get('status') == 'skip')
        n_seen = n_good + n_skip
        n_total = len(self.order)
        self.ax.set_title(
            f"{spot_name}  |  seen: {n_seen}/{n_total}  |  "
            f"good: {n_good}/{TARGET_TRACES}  |  skipped: {n_skip}  |  "
            f"clicks: {len(self.click_frames)}\n"
            f"[click=step  u=undo  c=clear  Enter=save  space=skip  "
            f"b=back  s=PNG  j=jump  q=quit]",
            fontsize=10)

    # ──────────────── jump to spot ────────────────

    def jump_to_spot(self):
        """Prompt for a spot ID in the terminal and jump to that trace."""
        if self.click_frames:
            print("Current trace has unsaved clicks. Press Enter to save "
                  "first, or 'c' to clear, before jumping.")
            return
        target = input("Jump to spot ID (e.g. 1138 or spot_1138): ").strip()
        if not target:
            print("Cancelled.")
            return
        target = target.replace('spot_', '').lstrip('0') or '0'
        try:
            target_int = int(target)
        except ValueError:
            print(f"Invalid spot ID: {target}")
            return
        for i, idx in enumerate(self.order):
            sid = int(self.spot_cols[idx].replace('spot_', ''))
            if sid == target_int:
                self.position = i
                self.show_current()
                print(f"Jumped to spot_{target_int:04d} "
                      f"(position {i+1}/{len(self.order)})")
                return
        print(f"Spot {target} not found in scoring order.")

    # ──────────────── events ────────────────

    def on_click(self, event):
        if event.inaxes != self.ax or event.button != 1 or event.xdata is None:
            return
        f = int(round(event.xdata))
        if 0 <= f < self.n_frames:
            self.click_frames.append(f)
            self.click_frames.sort()
            self._redraw_clicks()

    def on_key(self, event):
        key = event.key
        if key is None:
            return
        if key == 'u':                              # Undo last click
            if self.click_frames:
                self.click_frames.pop()
                self._redraw_clicks()
        elif key == 'c':                            # Clear all clicks
            self.click_frames = []
            self._redraw_clicks()
        elif key in ('enter', 'return', 'n'):       # Save as good
            self._save_current(status='good')
            self.position += 1
            self.show_current()
        elif key == ' ':                            # Skip
            self._save_current(status='skip')
            self.position += 1
            self.show_current()
        elif key == 'b':                            # Go back
            if self.position > 0:
                self.position -= 1
                self.show_current()
        elif key == 's':                            # Save PNG
            self._save_png()
        elif key == 'j':                            # Jump to spot
            self.jump_to_spot()
        elif key == 'q':                            # Quit
            self.save_results()
            n_good = sum(1 for r in self.results if r.get('status') == 'good')
            print(f"\nQuit. {len(self.results)} traces scored "
                  f"({n_good} good, {len(self.results)-n_good} skipped).")
            print(f"Saved to {self.results_csv}")
            plt.close(self.fig)

    # ──────────────── saving ────────────────

    def _compute_step_amplitudes(self):
        """Amplitude of each step = mean(segment before) − mean(segment after)."""
        _, smooth = self.current_traces()
        if smooth is None:
            return []
        boundaries = [0] + sorted(self.click_frames) + [self.n_frames]
        levels = []
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i+1]
            levels.append(float(smooth[a:b].mean()) if b - a >= 3 else np.nan)
        return [levels[i] - levels[i+1] for i in range(len(levels) - 1)]

    def _save_current(self, status='good'):
        idx = self.current_spot_idx()
        if idx is None:
            return
        spot_name = self.spot_cols[idx]
        if status == 'good':
            amps = self._compute_step_amplitudes()
            n_steps = len(self.click_frames)
            sf_str = '|'.join(str(f) for f in self.click_frames)
            sa_str = '|'.join(f'{a:.3f}' for a in amps)
        else:
            n_steps, sf_str, sa_str = -1, '', ''
        record = {
            'spot_id': spot_name,
            'n_steps': n_steps,
            'step_frames': sf_str,
            'step_amplitudes': sa_str,
            'status': status,
        }
        self.results = [r for r in self.results if r['spot_id'] != spot_name]
        self.results.append(record)
        self.save_results()

    def _save_png(self):
        idx = self.current_spot_idx()
        if idx is None:
            return
        spot_name = self.spot_cols[idx]
        n = len(self.click_frames)
        png_path = self.gallery_dir / f"{spot_name}_steps_{n}.png"
        self.fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"Saved PNG: {png_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if not Path(TRACES_CSV).exists():
        print(f"ERROR: TRACES_CSV not found: {TRACES_CSV}")
        sys.exit(1)
    StepScorer(TRACES_CSV, OUTPUT_DIR)


if __name__ == "__main__":
    main()
