# Reproducibility guide

## 1. Two levels of reproduction

| Level | What you reproduce | Requirements | Command |
|---|---|---|---|
| **A. Analysis** | Every statistic, table and figure in the paper, computed from the released raw outputs | Python ≥ 3.9, CPU | `make figures` |
| **B. Experiment** | The raw outputs themselves: downloads, LLM extraction, verification, 63 training runs | GPU ≥ 24 GB, Hugging Face token | run the notebook in `FULL` mode |

Level A is deterministic: `results/paper_analysis/stats.json` should match the released file exactly, and
`tests/test_released_artefacts.py` checks the headline result. Level B depends on GPU kernels and library
versions, so expect small differences in individual runs (see Section 5).

## 2. Settings used for the paper

| Component | Setting |
|---|---|
| Datasets | Train: HAPT, PAMAP2, WISDM-19, DSADS, REALDISP. Held out: MHEALTH, USC-HAD, HARTH, MotionSense. HHAR is listed in the notebook but failed to load and is excluded. |
| Extraction models | Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct, Gemma-2-9B-it. 4-bit NF4, double quantisation, bf16 compute, greedy decoding, ≤ 400 new tokens. |
| Context variants | zero-shot (first 5,000 words of documentation); retrieval (220-word chunks, 40-word overlap, top-6 by all-mpnet-base-v2) |
| Record for M2/M3 | Qwen2.5-7B-Instruct, retrieval (fixed in the configuration before extraction) |
| Probes | rate: 1/median Δt, ±10 %; accelerometer unit: resting-norm bands (0.6, 1.5) g, (6, 14) m/s², (600, 1500) milli-g, from ≤ 2,000 samples × ≤ 400 sessions; gyroscope: median over sessions of the 99th percentile of absolute per-axis angular rate > 20 → deg/s |
| Windows | 30 Hz, 5 s, stride 2.5 s, clip ±20, ≤ 20,000 windows per dataset (class-stratified) |
| Encoder | conv stem 1→64→128 (GELU), 10 tokens per channel, 3 × pre-norm Transformer (d = 128, 4 heads, FF 256, dropout 0.1), 0.64 M parameters |
| Training | AdamW (lr 1e-3, wd 1e-2), one-cycle, batch 256, 25 epochs, clip 1.0, seeds 0, 1, 2 |
| Text encoder | sentence-transformers/all-mpnet-base-v2 (768-d) |
| Hardware | 1 × NVIDIA L4 (24 GB), Google Colab; 9–49 s extraction per dataset and variant; 28–42 min per training run |

## 3. Running the notebook (Level B)

1. Open `notebooks/M2M_Metadata_to_Model.ipynb` in Colab and select an L4 (or larger) GPU.
2. In the configuration cell, set `MODE = "FULL"`. Set `WORK_DIR` to a Google Drive folder with about 20 GB free; the defaults mount Drive.
3. Paste your Hugging Face token into `HF_TOKEN`. You must first accept the licences of the gated models on huggingface.co. **Do not commit a notebook that contains the token.**
4. Choose *Runtime → Run all*. If the session disconnects, re-run; finished stages are skipped.
5. The last code cell writes `M2M_paper_assets_FULL.zip`. Unzip it into a new folder and run the analysis scripts on that folder:

```bash
python analysis/make_paper_figures.py <your_outputs> <your_analysis>
python analysis/make_extra_figures.py <your_outputs> <your_analysis> <your_analysis>/figs
```

For a fast check before committing GPU time, run `MODE = "SMOKE"` on CPU. It uses three synthetic datasets with
known units, rates and placements, and a mock extractor.

## 4. Run inventory

- **E1/E2:** 10 extraction runs (5 models × 2 variants) × 9 datasets = 90 records, in `results/notebook_outputs/extracted/`.
- **E3:** 5 conditions × 3 seeds = 15 training runs.
- **E4:** 5 defects × 3 severities × 3 seeds = 45 training runs.
- **Re-runs:** 3 M3 runs after the sampling-rate fix (see `CHANGELOG.md`).

Each run is evaluated on the four held-out datasets, giving 12 (dataset × seed) pairs per comparison. The order
and timestamps of every run are in `results/notebook_outputs/runs/manifest.jsonl`.

## 5. Known issues and how they are handled

- **Sampling-rate fallback (fixed).** In the first full run (v5), the verified condition (M3) resampled sessions without timestamps from 50 Hz instead of from the verified record's rate. This affected DSADS and USC-HAD. The notebook in this repository contains the fix, and the three M3 runs were repeated. All released M3 results come from the corrected runs; no other condition was affected.
- **Gyroscope scoring in E1.** `score_field` counts a correctly empty gyroscope unit as wrong for the two datasets without a gyroscope (WISDM-19, HARTH), which caps that field at 0.78. The paper discloses this.
- **Layout fields.** Placement and modality "verification" compares the extracted values with the loader's channel map. When they disagree, the loader's values are adopted. Because the channel map comes from the same documentation as the gold record, these fields are excluded from verification-accuracy claims.
- **Auto-generated summary.** `results/notebook_outputs/results/RESULTS.md` and `summary.json` are written automatically by the notebook. Their "verification gain" includes layout fields, and their "recovery" ratios are not used in the paper. The paper's numbers come from `results/paper_analysis/stats.json` (see `results/README.md`).
- **Statistical dependence.** The 12 pairs come from only four held-out datasets. Treat p-values as approximate; per-dataset effects are reported throughout.
- **Gold records.** One annotator wrote them, and no inter-annotator agreement is available. To compute Cohen's κ, add a second annotator's records with the same schema.
