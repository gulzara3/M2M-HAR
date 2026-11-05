"""Harmonisation-record schema, field groups, verifiability classes and the canonical activity taxonomy.

These definitions are copied verbatim from the experiment notebook
(``notebooks/M2M_Metadata_to_Model.ipynb``, Section 3-4) so that the record format
can be reused without running the notebook.
"""
from __future__ import annotations

import json
from typing import Dict, List

#: The ten fields of a harmonisation record (order used in all tables).
RECORD_FIELDS: List[str] = [
    "sampling_rate_hz", "n_participants", "devices", "sensor_placements", "modalities",
    "accelerometer_unit", "gyroscope_unit", "activities", "protocol_type", "setting",
]

#: Field definitions shown to the language model (verbatim from the notebook prompt).
FIELD_HELP: Dict[str, str] = {
    "sampling_rate_hz": "Nominal sampling frequency of the inertial sensors in Hz (a number; use the modal value if several).",
    "n_participants": "Number of human subjects/participants/volunteers who contributed data (integer).",
    "devices": "Recording devices or sensor products (e.g. 'Samsung Galaxy S II smartphone', 'Xsens MTx', 'Shimmer2', 'Axivity AX3').",
    "sensor_placements": "Body locations where sensors were worn (e.g. 'waist', 'chest', 'right wrist', 'ankle', 'front trouser pocket').",
    "modalities": "Sensor modalities recorded, from: accelerometer, gyroscope, magnetometer, heart rate, ECG, temperature, other.",
    "accelerometer_unit": "Unit of the accelerometer values in the data files: one of 'g', 'm/s^2', 'milli-g', or null if not stated.",
    "gyroscope_unit": "Unit of the gyroscope values: one of 'rad/s', 'deg/s', or null if not stated / no gyroscope.",
    "activities": "The list of activity class names exactly as the data set labels them.",
    "protocol_type": "One of 'experimental' (scripted activities), 'observational' (free-living), 'survey', or null.",
    "setting": "One of 'laboratory', 'free-living', 'both', or null.",
}

#: Verifiability classes (paper, Eq. 2 and Table 2).
VERIFIABILITY: Dict[str, List[str]] = {
    "signal": ["sampling_rate_hz", "accelerometer_unit", "gyroscope_unit"],
    "file": ["n_participants", "activities"],
    "layout": ["sensor_placements", "modalities"],
    "documentation_only": ["devices", "protocol_type", "setting"],
}

#: Field groups used for retrieval-augmented extraction (one prompt per group).
FIELD_GROUPS: Dict[str, List[str]] = {
    "acquisition": ["sampling_rate_hz", "n_participants", "devices", "modalities"],
    "placement_units": ["sensor_placements", "accelerometer_unit", "gyroscope_unit"],
    "activities": ["activities"],
    "protocol": ["protocol_type", "setting"],
}

#: Retrieval queries per field group.
GROUP_QUERIES: Dict[str, str] = {
    "acquisition": "sampling rate Hz frequency number of subjects participants volunteers device sensor model accelerometer gyroscope magnetometer",
    "placement_units": "sensor worn on body location wrist chest ankle waist pocket placement units g m/s^2 rad/s deg/s acceleration gyroscope values",
    "activities": "activities performed list of activity classes labels walking sitting standing",
    "protocol": "data collection protocol scripted free-living laboratory experiment observational",
}

#: The 33-class canonical activity taxonomy.
CANONICAL_ACTIVITIES: List[str] = [
    "walking", "running", "jogging", "sitting", "standing", "lying", "ascending stairs", "descending stairs", "cycling",
    "nordic walking", "jumping", "rope jumping", "vacuum cleaning", "ironing", "folding laundry", "house cleaning", "computer work",
    "watching tv", "car driving", "playing sport", "elevator", "exercise (upper body)", "exercise (lower body)", "rowing", "eating",
    "drinking", "writing/typing", "brushing teeth", "clapping", "transition", "sleeping", "shuffling", "other",
]

ACC_UNITS = ("g", "m/s^2", "milli-g")
GYRO_UNITS = ("rad/s", "deg/s")


def chunk_text(text: str, words: int = 220, overlap: int = 40) -> List[str]:
    """Split documentation into overlapping word chunks (220 words, 40-word overlap in the paper)."""
    w = text.split()
    chunks = []
    for i in range(0, max(1, len(w)), words - overlap):
        chunks.append(" ".join(w[i:i + words]))
        if i + words >= len(w):
            break
    return chunks


def build_prompt(group: str, context: str) -> str:
    """Schema-constrained extraction prompt for one field group (verbatim from the notebook)."""
    schema = {f: FIELD_HELP[f] for f in FIELD_GROUPS[group]}
    return (f"You extract metadata about a human activity recognition (HAR) data set from its documentation.\n"
            f"Return ONLY a JSON object with exactly these keys (use null when the documentation does not say):\n"
            f"{json.dumps(schema, indent=1)}\n\nDOCUMENTATION:\n\"\"\"\n{context}\n\"\"\"\n\nJSON:")
