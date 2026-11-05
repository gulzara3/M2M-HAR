"""Audit a documented harmonisation record against raw inertial recordings with the M2M probes.

Run:  python examples/audit_your_dataset.py
Replace the synthetic arrays below with your own sessions (one [N, 3] array per session and placement for
the accelerometer, one [N, k] array per session for the gyroscope channels, timestamps in seconds).
"""
import numpy as np

from m2m_tools.probes import (probe_accelerometer_unit, probe_gyroscope_unit, probe_sampling_rate,
                              rate_from_timestamps, rate_is_consistent, reconcile)

rng = np.random.default_rng(1)
# --- synthetic "dataset": 8 sessions at 25 Hz, accelerometer in m/s^2, gyroscope in rad/s -------------------
sessions = []
for _ in range(8):
    n = 1500
    acc = np.vstack([np.c_[rng.normal(0, .05, (500, 2)), 9.81 + rng.normal(0, .05, 500)], rng.normal(0, 4, (n - 500, 3)) + [0, 0, 9.81]])
    gyr = rng.normal(0, 1.2, (n, 3))
    t = np.arange(n) / 25.0
    sessions.append(dict(acc=acc, gyr=gyr, t=t))

# --- what the documentation (or an LLM reading it) claims ------------------------------------------------------
documented = {"sampling_rate_hz": 50, "accelerometer_unit": "g", "gyroscope_unit": None}

# --- probes ------------------------------------------------------------------------------------------------------
rate_med, rate_set = probe_sampling_rate(rate_from_timestamps(s["t"]) for s in sessions)
acc_unit, g_hat = probe_accelerometer_unit(s["acc"] for s in sessions)
gyr_unit, p99 = probe_gyroscope_unit(s["gyr"] for s in sessions)

rows = [
    ("sampling_rate_hz", *reconcile(documented["sampling_rate_hz"], rate_med,
                                    rate_is_consistent(documented["sampling_rate_hz"], rate_med, rate_set)), f"{rate_med:.1f} Hz"),
    ("accelerometer_unit", *reconcile(documented["accelerometer_unit"], acc_unit, documented["accelerometer_unit"] == acc_unit), f"|a|rest = {g_hat:.2f}"),
    ("gyroscope_unit", *reconcile(documented["gyroscope_unit"], gyr_unit, documented["gyroscope_unit"] == gyr_unit), f"p99 = {p99:.1f}"),
]
print(f"{'field':20s} {'verified value':16s} {'status':14s} evidence")
for f, v, status, ev in rows:
    v = round(v, 1) if isinstance(v, float) else v
    print(f"{f:20s} {str(v):16s} {status:14s} {ev}")
