import numpy as np
import pytest

from m2m_tools.probes import (accelerometer_unit_from_magnitude, probe_accelerometer_unit, probe_gyroscope_unit,
                              probe_sampling_rate, rate_from_timestamps, rate_is_consistent, reconcile,
                              resting_gravity_magnitude)

RNG = np.random.default_rng(0)


def _session(scale, n=1500, rest=True):
    """Tri-axial accelerometer: a resting segment (gravity on z) followed by motion, in units of `scale` per g."""
    rest_part = np.c_[RNG.normal(0, 0.01, (n // 3, 2)), 1 + RNG.normal(0, 0.01, n // 3)]
    move = RNG.normal(0, 0.6, (n - n // 3, 3)) + [0, 0, 1]
    return np.vstack([rest_part, move]) * scale


@pytest.mark.parametrize("scale,unit", [(1.0, "g"), (9.81, "m/s^2"), (1000.0, "milli-g")])
def test_accelerometer_unit_recovered_from_gravity(scale, unit):
    u, m = probe_accelerometer_unit([_session(scale) for _ in range(5)])
    assert u == unit
    assert m == pytest.approx(scale, rel=0.05)


def test_resting_magnitude_uses_lowest_variance_window():
    assert resting_gravity_magnitude(_session(9.81)) == pytest.approx(9.81, rel=0.02)


def test_band_edges_are_open_intervals():
    assert accelerometer_unit_from_magnitude(1.0) == "g"
    assert accelerometer_unit_from_magnitude(0.6) is None
    assert accelerometer_unit_from_magnitude(3.0) is None
    assert accelerometer_unit_from_magnitude(None) is None


def test_gyroscope_threshold():
    rad = [RNG.normal(0, 1.0, (1000, 3)) for _ in range(4)]          # p99 ~ 2.6 rad/s
    deg = [r * 180 / np.pi for r in rad]                               # same motion in deg/s
    assert probe_gyroscope_unit(rad)[0] == "rad/s"
    assert probe_gyroscope_unit(deg)[0] == "deg/s"
    assert probe_gyroscope_unit([]) == (None, None)


def test_sampling_rate_from_timestamps():
    t = np.arange(0, 20, 1 / 50.0)
    assert rate_from_timestamps(t) == pytest.approx(50.0)
    assert rate_from_timestamps(t[:5]) is None                          # too short
    med, rates = probe_sampling_rate([50.0, 50.1, None, 100.0])
    assert med == pytest.approx(50.1) and rates == [50, 100]
    assert probe_sampling_rate([None]) == (None, None)


def test_rate_consistency_tolerance():
    assert rate_is_consistent(50, 52.0)
    assert not rate_is_consistent(9, 100.0)                             # PAMAP2 case in the paper
    assert not rate_is_consistent(None, 50.0)


def test_reconcile_statuses():
    assert reconcile("g", None, False) == ("g", "unverifiable")
    assert reconcile("g", "g", True) == ("g", "confirmed")
    assert reconcile(None, "g", False) == ("g", "contradicted")        # missing value is filled
    assert reconcile("g", "m/s^2", False) == ("m/s^2", "contradicted")
