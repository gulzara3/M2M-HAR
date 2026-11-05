"""Record compilation helpers: unit conversion, resampling and windowing (paper Section 4.4, Eq. 11).

Constants and logic are identical to the experiment notebook (Section 6).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

#: Accelerometer unit -> factor to g. Unknown units fall back to m/s^2.
ACC_FACTOR = {"g": 1.0, "m/s^2": 1 / 9.81, "milli-g": 1e-3, None: 1 / 9.81}
#: Gyroscope unit -> factor to rad/s (an extra 1/5 is applied when building windows).
GYR_FACTOR = {"rad/s": 1.0, "deg/s": np.pi / 180, None: 1.0}
GYRO_EXTRA_SCALE = 1 / 5.0
DEFAULT_RATE = 50.0
TARGET_HZ = 30
WINDOW_SEC = 5.0
STRIDE_SEC = 2.5
CLIP = 20.0


def channel_scale(modality: str, acc_unit: Optional[str], gyr_unit: Optional[str]) -> float:
    """Multiplicative factor kappa(u_c) for one channel."""
    if modality == "accelerometer":
        return ACC_FACTOR.get(acc_unit, 1 / 9.81)
    if modality == "gyroscope":
        return GYR_FACTOR.get(gyr_unit, 1.0) * GYRO_EXTRA_SCALE
    return 1.0


def resample_to(x: np.ndarray, src_hz: Optional[float], dst_hz: float = TARGET_HZ) -> Optional[np.ndarray]:
    """Linear-interpolation resampling of an ``[N, C]`` array from ``src_hz`` to ``dst_hz``."""
    if src_hz is None or abs(src_hz - dst_hz) < 1e-6:
        return x
    n_out = int(round(len(x) * dst_hz / src_hz))
    if n_out < 2:
        return None
    t_src = np.arange(len(x)) / src_hz
    t_dst = np.arange(n_out) / dst_hz
    return np.stack([np.interp(t_dst, t_src, x[:, c]) for c in range(x.shape[1])], 1).astype(np.float32)


def make_windows(x: np.ndarray, hz: float = TARGET_HZ, window_sec: float = WINDOW_SEC,
                 stride_sec: float = STRIDE_SEC, clip: float = CLIP) -> np.ndarray:
    """Cut an ``[N, C]`` signal into ``[W, C, T]`` windows (5 s, 2.5 s stride), clipped to +/- ``clip``."""
    T, S = int(window_sec * hz), int(stride_sec * hz)
    if x is None or len(x) < T:
        return np.zeros((0, x.shape[1] if x is not None else 0, T), np.float32)
    w = [x[i:i + T].T for i in range(0, len(x) - T + 1, S)]
    return np.nan_to_num(np.clip(np.stack(w).astype(np.float32), -clip, clip))


def channel_description(modality: str, axis: str, placement: str, unit: str) -> str:
    """Natural-language channel description tau_r(c) (Eq. 12)."""
    return f"{modality} {axis}-axis, {placement}, {unit}"
