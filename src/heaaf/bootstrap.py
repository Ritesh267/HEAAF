"""Fast exact paired bootstrap for binary-indicator metrics.

Every metric the paper compares between two policies -- TPR, TNR, G-mean,
emergency denial rate, benign interruption rate -- is the mean of a 0/1
indicator over a fixed stratum of the test split.  When two policies are
evaluated on the *same* events, the pair of indicators takes one of four
values per event, so a bootstrap resample of that stratum is described
completely by a multinomial draw over those four cells.

Drawing the cell counts directly is exactly equivalent to resampling event
indices, but turns an O(n_boot * n) computation into O(n_boot).  That is what
makes 4,000 replicates on a 70,000-event split affordable for every contrast
in the table rather than only for the headline one.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def paired_rate_replicates(ind_a: np.ndarray, ind_b: np.ndarray,
                           mask: np.ndarray, n_boot: int,
                           rng: np.random.Generator
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap replicates of (rate_a, rate_b) within one stratum."""
    a = np.asarray(ind_a).astype(bool)[mask]
    b = np.asarray(ind_b).astype(bool)[mask]
    n = len(a)
    if n == 0:
        nan = np.full(n_boot, np.nan)
        return nan, nan
    cells = np.array([int((~a & ~b).sum()), int((a & ~b).sum()),
                      int((~a & b).sum()), int((a & b).sum())], dtype=float)
    draws = rng.multinomial(n, cells / n, size=n_boot)          # (n_boot, 4)
    return ((draws[:, 1] + draws[:, 3]) / n,
            (draws[:, 2] + draws[:, 3]) / n)


def summarise(d: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    """Percentile interval and two-sided bootstrap p-value for a difference."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan")}
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return {"diff": float(d.mean()),
            "lo": float(np.percentile(d, 100 * alpha / 2)),
            "hi": float(np.percentile(d, 100 * (1 - alpha / 2))),
            "p": float(min(1.0, max(p, 1.0 / (len(d) + 1))))}


def paired_metric_bootstrap(metric: str, acts_a: np.ndarray, acts_b: np.ndarray,
                            y: np.ndarray, emergency: np.ndarray,
                            allow_idx: int, deny_idx: int,
                            n_boot: int = 4000, alpha: float = 0.05,
                            seed: int = 0, return_replicates: bool = False):
    """Paired bootstrap of A - B for one metric on the shared test split.

    Positives and negatives are resampled independently, which preserves the
    base rate in every replicate; G-mean therefore combines two independent
    strata, as it does in the point estimate.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    fa = np.asarray(acts_a) != allow_idx
    fb = np.asarray(acts_b) != allow_idx
    da = np.asarray(acts_a) == deny_idx
    db = np.asarray(acts_b) == deny_idx

    if metric == "GMean":
        ta, tb = paired_rate_replicates(fa, fb, y == 1, n_boot, rng)
        na, nb = paired_rate_replicates(fa, fb, y == 0, n_boot, rng)
        va = np.sqrt(np.clip(ta, 0, 1) * np.clip(1 - na, 0, 1))
        vb = np.sqrt(np.clip(tb, 0, 1) * np.clip(1 - nb, 0, 1))
    elif metric == "TPR":
        va, vb = paired_rate_replicates(fa, fb, y == 1, n_boot, rng)
    elif metric == "interrupt_rate_benign":
        va, vb = paired_rate_replicates(fa, fb, y == 0, n_boot, rng)
    elif metric == "deny_rate_emergency":
        m = (y == 0) & (np.asarray(emergency).astype(int) == 1)
        va, vb = paired_rate_replicates(da, db, m, n_boot, rng)
    else:
        raise ValueError(f"unsupported metric {metric}")

    d = va - vb
    d = d[np.isfinite(d)]
    if return_replicates:
        return d
    if len(d) == 0:
        return {"diff": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p": float("nan")}
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return {"diff": float(d.mean()),
            "lo": float(np.percentile(d, 100 * alpha / 2)),
            "hi": float(np.percentile(d, 100 * (1 - alpha / 2))),
            "p": float(min(1.0, max(p, 1.0 / (len(d) + 1))))}
