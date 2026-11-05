# Changelog

## 1.0.0 (2026-10-25): release accompanying the manuscript

- Full pipeline notebook (v7 of the experiment), with outputs cleared and no credentials.
- `m2m_tools` package: probes, schema, scoring, statistics, harmonisation helpers, with tests.
- Gold harmonisation records for nine datasets, plus a JSON schema.
- Raw outputs of the reported run: 90 extracted records with raw LLM generations, audits, all E3/E4 per-run results, run manifest.
- Analysis scripts that regenerate every statistic, table and figure in the paper; final numbered figures.

### Experiment history

- **v5 (first full run, 22–24 Oct 2026).** All extraction, E3, E4 and E5 runs.
- **v6/v7 (25 Oct 2026).**
  - *Fix:* in the M3 (verified) condition, sessions without timestamps now fall back to the verified record's documented sampling rate instead of 50 Hz. This previously affected DSADS (25 Hz) and USC-HAD (100 Hz). The three M3 runs were re-trained; the other 48 E3 rows and all E4/E5 results are unchanged. Mean M3 zero-shot accuracy moved from 33.2 % to 36.3 %.
  - Extraction extended to the gated models: Mistral-7B, Llama-3.1-8B, Gemma-2-9B.
  - Each trained encoder is now saved to `runs/<run>.pt`.
