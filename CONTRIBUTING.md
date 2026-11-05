# Contributing

Issues and pull requests are welcome, in particular:

- **Gold records for further datasets.** Add a JSON file to `data/gold_records/` that follows `data/schema/harmonisation_record.schema.json`. Cite the documentation you used, and run `pytest`.
- **Second-annotator records.** These let us report inter-annotator agreement.
- **Corrections** to documented units or rates that the probes flag (e.g. the MHEALTH gyroscope unit).
- **New probes**, with unit tests in `tests/`.

Please keep `pytest -q` and `make analysis` passing, and never commit credentials (Hugging Face tokens) or
dataset files.
