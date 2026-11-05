# M2M results (FULL mode, 2026-10-25)

**Data.** 5 training data sets (hapt, pamap2, wisdm19, dsads, realdisp) and 4 held-out (mhealth, uschad, harth, motionsense); 53,309 evaluation windows of 5 s at 30 Hz.

**E1 Extraction.** Best system: Mistral-7B-Instruct-v0.3 / retrieval with composite field accuracy 0.714. Hardest fields: setting, protocol_type, gyroscope_unit.

**E2 Audit.** 1 of 9 data sets have at least one documentation–data contradiction in the gold records; 1.8% of verifiable fields are contradicted. Verification raised extracted-record accuracy on verifiable fields by 22.5 points on average.

**E3 Metadata conditions (zero-shot accuracy, mean over held-out data sets and seeds).** M0 none 28.2%; M1 generic 26.7%; M2 extracted 27.0%; M3 verified 36.3%; M4 gold 35.4%.
Gold metadata adds 7.3 points over no metadata; verified automatic metadata adds 8.1 points and unverified extraction -1.1 points (recovery of the gold gain: 111.2% verified, -15.5% unverified).

**E4 Defect costs at 100% severity (points of zero-shot accuracy).** placement: -0.3; unit: 23.1; rate: 1.9; axis: 3.8; label: 27.6.

**E5 Label alignment accuracy.** embedding: 85.1%; wordnet: 71.4%; llm_constrained: 78.9%.

Figures: F1 audit matrix, F2 extraction heatmap, F3 conditions, F4 defect costs, F5 few-shot, F6 embeddings, F7 label alignment, F8 verification repair (PDF + PNG in `figures/`). Tables T1–T8 in `tables/` (CSV, Markdown, LaTeX).
