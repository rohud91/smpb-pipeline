# smpb-pipeline

End-to-end Python pipeline for determining membrane protein stoichiometry by single-molecule photobleaching (smPB).

## Overview

Single-molecule photobleaching counts the number of fluorescent subunits in individual protein complexes by recording their stepwise fluorescence decay under continuous excitation. Each discrete intensity drop corresponds to one fluorophore bleaching irreversibly, so the number of steps reports the number of labelled subunits.

This pipeline takes fluorescence image stacks and spot-detection output (from [ComDet](https://github.com/ekatrukha/ComDet) in FIJI) through trace extraction, interactive step scoring, automated quality filtering, and statistical analysis including truncated binomial fitting.

Developed for GFP-tagged membrane proteins in plant microsomal fractions imaged on a spinning-disk confocal microscope, but applicable to any smPB experiment with minor parameter adjustments.

## Pipeline

```
FIJI / ComDet                     Python pipeline
─────────────    ┌─────────────────────────────────────────────────────┐
                 │                                                     │
 Image stack ──► │  1. extract_traces.py       Spot → trace extraction │
 + ComDet CSV    │         │                                           │
                 │         ▼                                           │
                 │  2. score_steps.py          Interactive step scoring │
                 │         │                                           │
                 │         ▼                                           │
                 │  3. quality_filter.py       Bleach-amplitude filter  │
                 │         │                                           │
                 │         ▼                                           │
                 │  4. analyze_stoichiometry.py   Binomial fitting     │
                 │         │                                           │
                 │         ▼ (optional)                                │
                 │  5. investigate_steps.py    Outlier diagnostics     │
                 │                                                     │
                 │  Figure scripts:                                    │
                 │  • figure_step_bleaching.py   Pan et al.–style      │
                 │  • figure_step_histogram.py   Publication histogram  │
                 │                                                     │
                 │  Utilities (utils/):                                │
                 │  • pool_datasets.py           Multi-field pooling   │
                 │  • cherry_pick_traces.py       Find cleanest traces │
                 │  • fix_csv_separators.py       Locale fix (one-time)│
                 └─────────────────────────────────────────────────────┘
```

## Requirements

- Python ≥ 3.9
- numpy
- pandas
- scipy
- matplotlib
- tifffile

```bash
pip install -r requirements.txt
```

## Quick start

1. **In FIJI:** open your image stack, verify pixel size, generate a max-intensity projection, run ComDet, and save the results table as CSV.

2. **Run each script in order.** Each has a `USER SETTINGS` block at the top for file paths and parameters:

```bash
python extract_traces.py
python score_steps.py
python quality_filter.py
python analyze_stoichiometry.py
```

3. **Inspect outputs at each stage** before proceeding. Check `example_traces.png` after extraction, review the quality-filter report, and examine the fit report after analysis.

## Scripts

| Script | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `extract_traces.py` | ROI extraction with background subtraction | ComDet CSV, TIFF stack | `traces_corrected.csv`, `spots_used.csv` |
| `score_steps.py` | Interactive manual step counting | `traces_corrected.csv` | `step_counts.csv` |
| `quality_filter.py` | Filter by bleach-amplitude consistency | `step_counts.csv`, `traces_corrected.csv` | `step_counts_filtered.csv` |
| `analyze_stoichiometry.py` | Truncated binomial fitting | `step_counts_filtered.csv` | Histograms, `fit_report.txt` |
| `investigate_steps.py` | Diagnose outlier step counts | `step_counts.csv`, traces, spots | Metrics, gallery, boxplots |
| `figure_step_bleaching.py` | Publication trace figures | traces, step_counts | PDF/SVG/PNG per spot |
| `figure_step_histogram.py` | Publication histogram | step_counts | PDF/SVG/PNG |

## Key parameters

| Parameter | Script | Default | Description |
|---|---|---|---|
| `AREA_MIN` / `AREA_MAX` | extract_traces | 5 / 15 | Spot size filter (px²) |
| `AR_MAX` | extract_traces | 1.5 | Maximum aspect ratio |
| `ROI_HALF` | extract_traces | 2 | ROI = (2n+1)² pixels |
| `BG_INNER` / `BG_OUTER` | extract_traces | 4 / 7 | Background annulus radii (px) |
| `PRESCREEN_DROP_MIN` | score_steps | 0.5 | Min intensity drop for candidates |
| `PRESCREEN_SNR_MIN` | score_steps | 1.5 | Min drop / noise ratio |
| `BLEACH_TOLERANCE` | quality_filter | 0.5 | \|bleach_units − n_steps\| threshold |
| `CANDIDATE_N` | analyze_stoichiometry | [2,3,4,5] | Stoichiometries to test |
| `STEPS_TO_INVESTIGATE` | investigate_steps | 4 | Focal step count for diagnostics |

## Scoring controls

| Key | Action |
|---|---|
| Left click | Mark a bleaching step |
| `u` | Undo last click |
| `c` | Clear all clicks |
| `Enter` / `n` | Save as "good" and advance |
| `Space` | Skip trace |
| `b` | Go back one trace |
| `s` | Save current view as PNG |
| `j` | Jump to specific spot ID |
| `q` | Quit (progress auto-saved) |

## Troubleshooting
 
### Quality filter rejects too many traces
 
If `quality_filter.py` rejects >50% of your manually scored traces:
 
1. **Check the unit step variability** in the output. If σ/μ > 0.3, increase `BLEACH_TOLERANCE` to 0.7–1.0.
2. **Inspect rejected traces visually** — are they genuinely bad, or is the filter too strict?
3. **Common causes of high rejection rates:**
   - ROI too small (increase `ROI_HALF` in `extract_traces.py`)
   - Background annulus eating signal (adjust `BG_INNER`/`BG_OUTER`)
   - Aggregates/clusters passing size filter (tighten `AREA_MAX`, `AR_MAX`)
   
The filter report shows per-n retention rates — use these to identify systematic issues.
 
### Low spot counts after extraction
 
If `extract_traces.py` keeps very few spots:
 
1. Check the filter summary in terminal output (shows NArea and aspect ratio distributions)
2. Relax `AREA_MIN`/`AREA_MAX` if your fluorophore or optics differ from defaults
3. Verify pixel size calibration — incorrectly calibrated images yield wrong spot sizes

### Step amplitude histogram looks bimodal
 
Two peaks in the amplitude histogram usually means:
 
- **Dimers in the population** (second peak at ~2× unit step)
- **Incomplete maturation** (dark GFP subunits, lowering effective n)
- **Aggregates** (tighten size filter)
Consider analyzing subpopulations separately or using a positive control to fix *p*.

## References

- Hallworth R, Bhatt K, Bhatt P (2012). Quantitative measurement of membrane protein subunit stoichiometry using single molecule fluorescence. *Proc SPIE* 8228.
- Nichols MG, Bhatt P, Bhatt K, Hallworth R (2016). Stoichiometry determination of fluorescent protein tagged membrane proteins. In: *Fluorescence Methods for Investigation of Living Cells and Microorganisms.* IntechOpen.
- Dey S, Bhatt S, Bhatt P, Bhatt K, Maiti S (2018). An improved method for determination of membrane protein stoichiometry. *J Phys Chem Lett* 9, 2814–2819.
- Pan Y, Chai X, Gao Q et al. (2019). Dynamic interactions of plant CNGC subunits and calmodulins drive oscillatory Ca²⁺ channel activities. *Dev Cell* 48, 710–725.

## Citation

If you use this pipeline in your work, please cite:

> [Author et al. (Year). Title. Journal. DOI.]

## License

MIT License. See [LICENSE](LICENSE) for details.
