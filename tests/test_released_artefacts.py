"""Integrity checks on the released gold records and results (fast; no GPU, no downloads)."""
import json
from pathlib import Path

import pandas as pd
import pytest

from m2m_tools.probes import accelerometer_unit_from_magnitude
from m2m_tools.schema import CANONICAL_ACTIVITIES, RECORD_FIELDS
from m2m_tools.stats import boot

ROOT = Path(__file__).resolve().parents[1]
GOLD = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "data" / "gold_records").glob("*.json"))}
NB = ROOT / "results" / "notebook_outputs"


def test_nine_gold_records_with_all_fields():
    assert len(GOLD) == 9
    for k, rec in GOLD.items():
        for f in RECORD_FIELDS:
            assert f in rec, (k, f)
        assert rec["accelerometer_unit"] in ("g", "m/s^2", "milli-g")
        assert rec["gyroscope_unit"] in ("rad/s", "deg/s", None)


def test_canonical_maps_use_the_taxonomy():
    for k, rec in GOLD.items():
        for raw, canon in rec.get("canonical_map", {}).items():
            assert canon in CANONICAL_ACTIVITIES, (k, raw, canon)


def test_gold_audit_gravity_values_map_to_documented_units():
    a = pd.read_csv(NB / "audit" / "audit_gold.csv")
    a = a[a.field == "accelerometer_unit"]
    for _, r in a.iterrows():
        m = float(r.data.split("≈")[1].rstrip(")"))
        assert accelerometer_unit_from_magnitude(m) == GOLD[r.dataset]["accelerometer_unit"]


def test_headline_e3_result_reproduces():
    e3 = pd.read_csv(NB / "results" / "E3_runs.csv")
    p = e3.pivot_table(index=["dataset", "seed"], columns="condition", values="zeroshot_acc") * 100
    d = (p["M4_gold"] - p["M3_verified"]).values
    assert len(d) == 12
    assert d.mean() == pytest.approx(-0.818, abs=0.01)
    lo, hi = boot(d)
    assert lo == pytest.approx(-2.35, abs=0.05) and hi == pytest.approx(0.70, abs=0.05)


def test_gold_records_validate_against_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "data" / "schema" / "harmonisation_record.schema.json").read_text())
    for rec in GOLD.values():
        jsonschema.validate(rec, schema)
