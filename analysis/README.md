# Analysis scripts

Run from the repository root (or use `make figures`):

```bash
python analysis/make_paper_figures.py results/notebook_outputs results/paper_analysis
python analysis/make_extra_figures.py results/notebook_outputs results/paper_analysis results/paper_analysis/figs
python analysis/collect_figures.py
```

| Script | Produces |
|---|---|
| `make_paper_figures.py` | `stats.json`, E3/E4 tables, verification recomputed on signal-verifiable fields, figures: extraction heatmap, conditions, defects, few-shot |
| `make_extra_figures.py` | probe decision space, verification by model, E3 forest plot, value vs reliability |
| `collect_figures.py` | copies the final figures, numbered as in the manuscript, into `figures/` |
| `diagrams/` | HTML sources of Fig. 1 (framework) and Fig. 2 (encoder) with their PNG/PDF renders. Re-render with `make diagrams` (needs Node.js and Playwright with Chromium). |

Bootstrap (10,000 resamples) and permutation (20,000) use fixed seeds, so the outputs are deterministic.
