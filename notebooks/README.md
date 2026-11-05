# Notebook

`M2M_Metadata_to_Model.ipynb` runs the whole study in one place.

| Section | Content |
|---|---|
| 0–1 | Configuration and environment |
| 2 | Dataset registry, download, one loader per dataset, unified session format |
| 3 | Harmonisation-record schema, canonical taxonomy, gold records |
| 4 | Documentation bundles, retrieval, LLM extraction, E1 scoring |
| 5 | Signal probes, verification, documentation–data audit (E2) |
| 6 | Record-driven harmonisation (units, resampling, windows, channel text) |
| 7 | Metadata-conditioned encoder |
| 8 | Experiments E3 (conditions), E4 (defect injection), E5 (label alignment) |
| 9–11 | Figures, summary, export of all outputs |

Run `MODE = "SMOKE"` first (CPU, minutes). For `MODE = "FULL"`, see `docs/reproducibility.md`. The outputs are
cleared, and `HF_TOKEN` is empty by design: add your token only at run time.

`scripts/m2m_pipeline.py` is a read-only export of this notebook for code review. Regenerate it with
`python scripts/export_notebook.py`.
