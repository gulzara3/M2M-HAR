# The harmonisation record

A harmonisation record is the part of a dataset's metadata that a cross-dataset model consumes. The JSON schema
is in `data/schema/harmonisation_record.schema.json`; the expert (gold) records for the nine datasets are in
`data/gold_records/`.

| Class | Field | Value domain | Verified by |
|---|---|---|---|
| Signal-verifiable | `sampling_rate_hz` | number (Hz) | 1 / median inter-sample interval of timestamps (±10 %) |
| | `accelerometer_unit` | `g`, `m/s^2`, `milli-g` | resting gravity magnitude (bands 0.6–1.5, 6–14, 600–1500) |
| | `gyroscope_unit` | `rad/s`, `deg/s` | 99th percentile of absolute angular rate > 20 → deg/s (medium confidence) |
| File-verifiable | `n_participants` | integer | unique subject identifiers in the files |
| | `activities` | list of raw labels | unique label values in the files (fuzzy set-F1 ≥ 0.9) |
| Layout | `sensor_placements` | list | loader channel map (not independent of gold) |
| | `modalities` | list | loader channel map (not independent of gold) |
| Documentation only | `devices`, `protocol_type`, `setting` | list / enum | none |

Every gold record also has:

- `channels`: one entry per channel the loader uses, `{modality, axis, placement, unit}`. Channel descriptions are generated from these entries, e.g. "accelerometer x-axis, chest, m/s^2".
- `canonical_map`: maps each raw activity label to one of 33 canonical classes (`m2m_tools.schema.CANONICAL_ACTIVITIES`).

## Reconciliation rule

For each verifiable field, the probe value `π(X)` is compared with the documented or extracted value `r̂`:

- probe undefined (e.g. no timestamps) → keep `r̂`, status *unverifiable*;
- consistent → keep `r̂`, status *confirmed*;
- otherwise → adopt `π(X)`, status *contradicted*. A missing `r̂` counts as contradicted, so verification also fills gaps.

The same rule can audit expert records. For the nine gold records, it flagged one candidate discrepancy (the
MHEALTH gyroscope unit).
