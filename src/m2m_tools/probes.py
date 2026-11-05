"""Deterministic signal probes and record reconciliation.

Array-based re-implementations of the probes in the experiment notebook
(``probe_rate``, ``probe_acc_unit``, ``probe_gyro_unit``, ``verify_record``). The logic,
thresholds and sampling limits are identical; only the input format differs
(lists of NumPy arrays instead of the notebook's session store), so the probes can be
applied to any inertial dataset.

Paper reference: Section 4.3, Eqs. (6)-(10).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Resting-gravity bands (native units) -> unit. Open intervals, as in the notebook.
ACC_BANDS = (("g", 0.6, 1.5), ("m/s^2", 6.0, 14.0), ("milli-g", 600.0, 1500.0))
#: 99th-percentile angular-rate threshold separating rad/s from deg/s.
GYRO_THRESHOLD = 20.0
#: Relative tolerance for confirming a documented sampling rate.
RATE_TOLERANCE = 0.10


# --------------------------------------------------------------------------- sampling rate
def rate_from_timestamps(t: Sequence[float]) -> Optional[float]:
    """Sampling rate (Hz) from timestamps in seconds: 1 / median positive inter-sample interval."""
    t = np.asarray(t, dtype=float)
    d = np.diff(t)
    d = d[(d > 0) & np.isfinite(d)]
    return float(1.0 / np.median(d)) if len(d) > 10 else None


def probe_sampling_rate(session_rates: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[List[int]]]:
    """Median session rate and the set of distinct rates (rounded to 10 Hz). ``(None, None)`` if no timestamps."""
    rates = [r for r in session_rates if r]
    if not rates:
        return None, None
    return float(np.median(rates)), sorted(set(int(round(r / 10) * 10) for r in rates))


def rate_is_consistent(documented, measured_median: float, measured_set: Optional[List[int]] = None,
                       tol: float = RATE_TOLERANCE) -> bool:
    """True if every documented rate lies within ``tol`` of a measured rate (Eq. 6 consistency rule)."""
    docs = documented if isinstance(documented, list) else [documented]
    docs = [d for d in docs if isinstance(d, (int, float))]
    refs = measured_set or [measured_median]
    return bool(docs) and all(any(abs(d - r) / r <= tol for r in refs) for d in docs)


# --------------------------------------------------------------------------- accelerometer unit
def resting_gravity_magnitude(acc_xyz: np.ndarray, max_samples: int = 2000) -> Optional[float]:
    """Mean |a| inside the lowest-variance rolling window of one tri-axial accelerometer session."""
    v = np.asarray(acc_xyz, dtype=float)[:max_samples]
    n = len(v)
    if n == 0 or v.ndim != 2 or v.shape[1] != 3:
        return None
    norm = np.linalg.norm(v, axis=1)
    w = max(10, n // 20)
    stds = pd.Series(norm).rolling(w).std().values
    means = pd.Series(norm).rolling(w).mean().values
    ok = ~np.isnan(stds)
    return float(means[ok][np.argmin(stds[ok])]) if ok.sum() > 0 else None


def accelerometer_unit_from_magnitude(m: Optional[float]) -> Optional[str]:
    """Map a resting gravity magnitude to g, m/s^2 or milli-g (Eq. 8); ``None`` if outside every band."""
    if m is None:
        return None
    for unit, lo, hi in ACC_BANDS:
        if lo < m < hi:
            return unit
    return None


def probe_accelerometer_unit(sessions: Iterable[np.ndarray], max_sessions: int = 400) -> Tuple[Optional[str], Optional[float]]:
    """Unit and median resting magnitude over sessions.

    ``sessions`` yields ``[N, 3]`` arrays (one tri-axial accelerometer per array; pass one array per
    session *and* placement, as the notebook does).
    """
    mags = []
    for i, s in enumerate(sessions):
        if i >= max_sessions:
            break
        m = resting_gravity_magnitude(s)
        if m is not None:
            mags.append(m)
    if not mags:
        return None, None
    m = float(np.median(mags))
    return accelerometer_unit_from_magnitude(m), m


# --------------------------------------------------------------------------- gyroscope unit
def probe_gyroscope_unit(sessions: Iterable[np.ndarray], max_sessions: int = 400,
                         threshold: float = GYRO_THRESHOLD) -> Tuple[Optional[str], Optional[float]]:
    """deg/s if the median over sessions of the 99th percentile of |omega| (pooled over all gyroscope
    axes and channels of a session) exceeds ``threshold``, else rad/s (Eq. 9). Medium confidence."""
    q = []
    for i, s in enumerate(sessions):
        if i >= max_sessions:
            break
        s = np.asarray(s, dtype=float)
        if s.size:
            q.append(np.percentile(np.abs(s), 99))
    if not q:
        return None, None
    p99 = float(np.median(q))
    return ("deg/s" if p99 > threshold else "rad/s"), p99


# --------------------------------------------------------------------------- reconciliation
def reconcile(documented, probed, consistent: bool):
    """Eq. (10): keep the documented value if the probe is undefined or agrees, otherwise adopt the probe.

    Returns ``(value, status)`` with status in {"unverifiable", "confirmed", "contradicted"}.
    A missing documented value counts as contradicted whenever the probe is defined.
    """
    if probed is None:
        return documented, "unverifiable"
    if consistent:
        return documented, "confirmed"
    return probed, "contradicted"
