"""m2m_tools: reusable components of the Metadata-to-Model (M2M) study.

Modules
-------
schema     Harmonisation-record fields, field groups, verifiability classes, taxonomy, prompt.
probes     Deterministic signal probes (sampling rate, accelerometer unit, gyroscope unit) and reconciliation.
harmonise  Unit conversion, resampling, windowing and channel descriptions.
scoring    Record normalisation and field-level scoring against gold records.
stats      Paired bootstrap, sign-flip permutation, Holm, TOST and effect size.

The full experimental pipeline (data download, LLM extraction, training, all experiments)
lives in ``notebooks/M2M_Metadata_to_Model.ipynb``; these modules mirror its logic.
"""
__version__ = "1.0.0"
