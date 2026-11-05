"""Paired statistics used in the paper (Section 5.3): bootstrap CI, sign-flip permutation test,
Holm correction, two one-sided tests (TOST) and paired effect size.

``signflip``, ``boot`` and ``holm`` are the exact functions used to produce the paper's tables
(``analysis/make_paper_figures.py``), with the same default seeds.
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats as _st


def signflip(d: Sequence[float], n: int = 20000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p-value for H0: mean(d) = 0."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    obs = d.mean()
    null = (d[None, :] * rng.choice([-1, 1], (n, len(d)))).mean(1)
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))


def boot(d: Sequence[float], n: int = 10000, seed: int = 0) -> list:
    """95 % percentile bootstrap CI of the mean."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    b = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def holm(p: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(1, run)
    return adj


def tost(d: Sequence[float], margin: float) -> float:
    """TOST p-value for equivalence of a paired mean difference within +/- ``margin`` (one-sample t-tests)."""
    d = np.asarray(d, float)
    p_lo = _st.ttest_1samp(d, -margin, alternative="greater").pvalue
    p_hi = _st.ttest_1samp(d, margin, alternative="less").pvalue
    return float(max(p_lo, p_hi))


def cohens_dz(d: Sequence[float]) -> float:
    d = np.asarray(d, float)
    return float(d.mean() / d.std(ddof=1))


def paired_summary(d: Sequence[float]) -> Dict[str, float]:
    """Mean, bootstrap CI, permutation p and d_z of paired differences."""
    lo, hi = boot(d)
    return dict(mean=float(np.mean(d)), ci_lo=lo, ci_hi=hi, p=signflip(d), dz=cohens_dz(d), n=len(d))
