import numpy as np
import pytest

from m2m_tools.scoring import composite_score, fuzzy_set_f1, normalise_record, score_field
from m2m_tools.stats import boot, cohens_dz, holm, signflip, tost


def test_fuzzy_set_f1():
    assert fuzzy_set_f1(["Right wrist", "chest"], ["right wrist", "chest"]) == 1.0
    assert fuzzy_set_f1([], []) == 1.0
    assert fuzzy_set_f1(["ankle"], []) == 0.0
    assert 0 < fuzzy_set_f1(["chest", "ankle", "head"], ["chest", "ankle"]) < 1


def test_score_field_rules():
    assert score_field("sampling_rate_hz", 52, 50) == 1.0
    assert score_field("sampling_rate_hz", 9, 100) == 0.0
    assert score_field("accelerometer_unit", "g", "g") == 1.0
    assert score_field("gyroscope_unit", None, None) == 0.0             # documented property (paper, Section 6.1)
    assert score_field("n_participants", 30, 30) == 1.0


def test_normalise_record_units_and_lists():
    r = normalise_record({"accelerometer_unit": "m/s2", "gyroscope_unit": "degrees per second",
                          "sampling_rate_hz": "50 Hz", "n_participants": "30 volunteers", "activities": "walking, sitting"})
    assert r["accelerometer_unit"] == "m/s^2" and r["gyroscope_unit"] == "deg/s"
    assert r["sampling_rate_hz"] == 50.0 and r["n_participants"] == 30
    assert r["activities"] == ["walking", "sitting"]


def test_composite_of_identical_record_is_one_except_null_fields():
    gold = {"sampling_rate_hz": 50, "n_participants": 10, "devices": ["x"], "sensor_placements": ["chest"],
            "modalities": ["accelerometer"], "accelerometer_unit": "g", "gyroscope_unit": "rad/s",
            "activities": ["walking"], "protocol_type": "experimental", "setting": "laboratory"}
    assert composite_score(gold, gold) == 1.0


def test_holm_matches_definition():
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_stats_are_deterministic_and_sensible():
    d = np.array([1.0, 2.0, 1.5, 2.5, 0.5, 1.2, 1.8, 2.2, 0.9, 1.1, 1.4, 1.6])
    lo, hi = boot(d)
    assert lo < d.mean() < hi and boot(d) == [lo, hi]
    assert signflip(d) < 0.01
    assert cohens_dz(d) > 1
    assert tost(d - d.mean(), margin=1.0) < 0.05
