# Results

## `notebook_outputs/`: raw outputs of the run reported in the paper

| Path | Content |
|---|---|
| `results/E3_runs.csv` | One row per (E3 run × held-out dataset): zero-shot accuracy and macro-F1 (with window-level bootstrap CIs), few-shot accuracy and F1 at 1 % and 10 %, number of windows |
| `results/E4_runs.csv` | Same, for every defect × severity × seed |
| `results/summary.json`, `results/RESULTS.md` | Summary written automatically by the notebook (see caveat below) |
| `tables/` | Tables written by the notebook (CSV, Markdown, LaTeX) |
| `audit/audit_extracted.csv` | Probe outcome for every field of every extracted record (10 runs × 9 datasets) |
| `audit/audit_gold.csv` | Probe outcome for every field of the nine gold records |
| `extracted/` | The 90 extracted records (`<model>__<variant>__<dataset>.json`), including the raw LLM generations per field group (`_raw`) |
| `gold/` | The gold records as used by the run (identical to `data/gold_records/`) |
| `runs/manifest.jsonl` | Time-stamped log of every extraction and training run |

**Caveat.** `summary.json` and `RESULTS.md` are generated automatically. Their "verification gain" (≈ 22 points)
averages over all verifiable fields, including placements and modalities, whose probe is the loader's channel
map and therefore agrees with gold by construction. Their "recovery" ratios divide by a gain that is not
statistically reliable. The paper therefore does not use these two summaries; it uses `paper_analysis/`.

## `paper_analysis/`: everything reported in the paper

Produced by `make analysis` from `notebook_outputs/`:

- `stats.json`: all statistics, including verification on signal-verifiable fields only (59.5 % → 95.0 %), contradiction rates, E3 summaries and paired tests, E4 costs, E5.
- `tables/T_E3_summary.csv`, `T_E3_pairwise.csv`, `T_E4_costs.csv`, `T1_datasets.csv`, `verification_physical_rows.csv`.
- `figs/`: data figures (PDF + PNG) and `kbs_fig_stats.json` (verification by model, value-vs-reliability inputs).
