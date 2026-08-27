"""Uncertainty quantification and significance testing.

A single-seed table is not a result.  Two distinct sources of variability
matter here and they need different instruments:

* **Seed variability** -- the agent, the replay buffer, the forest and the
  generator are all stochastic.  This is handled by repeating the whole
  pipeline over independent seeds and reporting mean +/- standard deviation,
  plus a Student-t interval on the mean.
* **Sampling variability of the test split** -- with a malicious base rate near
  1%, a 70,000-event test split contains only ~750 malicious events, so TPR is
  estimated from a small effective sample.  This is handled by a stratified
  bootstrap over test events.

Comparisons between policies are made *paired* on the same seeds and the same
bootstrap resamples, because the policies share a test split and their errors
are strongly correlated; an unpaired test would be badly under-powered.
Because the main table makes many comparisons at once, p-values are corrected
with Holm-Bonferroni, which controls the family-wise error rate without
assuming independence.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps


# ==========================================================================
# Intervals on the mean across seeds
# ==========================================================================
def mean_ci(values: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    """Mean with a Student-t interval; degenerates gracefully for n < 2."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n": 0}
    if n == 1:
        return {"mean": float(v[0]), "sd": 0.0,
                "lo": float(v[0]), "hi": float(v[0]), "n": 1}
    m = float(v.mean())
    sd = float(v.std(ddof=1))
    half = float(sps.t.ppf(1 - alpha / 2, n - 1) * sd / np.sqrt(n))
    return {"mean": m, "sd": sd, "lo": m - half, "hi": m + half, "n": n}


def fmt_pm(m: float, sd: float, dp: int = 3) -> str:
    """LaTeX-safe 'mean +/- sd' rendering used throughout the tables."""
    if not np.isfinite(m):
        return "--"
    if not np.isfinite(sd) or sd == 0:
        return f"{m:.{dp}f}"
    return f"{m:.{dp}f} $\\pm$ {sd:.{dp}f}"


# ==========================================================================
# Stratified bootstrap over test events
# ==========================================================================
def stratified_bootstrap_idx(y: np.ndarray, n_boot: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Resample positives and negatives separately.

    Stratifying preserves the base rate in every replicate.  Without it, some
    replicates would contain very few malicious events and the TPR estimate
    would acquire variance that has nothing to do with the policy.
    """
    y = np.asarray(y).astype(int)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    out = np.empty((n_boot, len(y)), dtype=np.int64)
    for b in range(n_boot):
        out[b] = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True)])
    return out


def bootstrap_metric(stat_fn: Callable[[np.ndarray], float], y: np.ndarray,
                     n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap interval for a metric computed on test events."""
    rng = np.random.default_rng(seed)
    idx = stratified_bootstrap_idx(y, n_boot, rng)
    vals = np.array([stat_fn(i) for i in idx], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {"mean": float(vals.mean()),
            "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2)))}


def paired_bootstrap_diff(stat_a: Callable[[np.ndarray], float],
                          stat_b: Callable[[np.ndarray], float],
                          y: np.ndarray, n_boot: int = 2000,
                          alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Paired bootstrap on A - B, using the *same* resample for both policies.

    Returns the observed difference, its interval and a two-sided bootstrap
    p-value for the null that the difference is zero.
    """
    rng = np.random.default_rng(seed)
    idx = stratified_bootstrap_idx(y, n_boot, rng)
    d = np.array([stat_a(i) - stat_b(i) for i in idx], dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p": float("nan")}
    obs = float(d.mean())
    # two-sided p from the proportion of replicates on the wrong side of zero
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return {"diff": obs,
            "lo": float(np.percentile(d, 100 * alpha / 2)),
            "hi": float(np.percentile(d, 100 * (1 - alpha / 2))),
            "p": float(min(1.0, max(p, 1.0 / (len(d) + 1))))}


# ==========================================================================
# Paired tests across seeds
# ==========================================================================
def paired_seed_test(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Paired comparison of two policies over matched seeds.

    Reports the mean difference, a Wilcoxon signed-rank p-value (no normality
    assumption, appropriate for the small number of seeds a full pipeline
    repeat allows) and Cohen's d_z as an effect size.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 2 or np.allclose(a, b):
        return {"diff": float(np.mean(a - b)) if n else float("nan"),
                "p": float("nan"), "dz": float("nan"), "n": n}
    d = a - b
    try:
        p = float(sps.wilcoxon(a, b, zero_method="zsplit").pvalue)
    except ValueError:
        p = float("nan")
    sd = float(d.std(ddof=1))
    return {"diff": float(d.mean()), "p": p,
            "dz": float(d.mean() / sd) if sd > 0 else float("inf"), "n": n}


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction.

    Returns (adjusted p-values, reject flags).  Controls the family-wise error
    rate under arbitrary dependence, which is the right guarantee for a table
    whose rows are evaluated on one shared test split.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj, adj <= alpha


def stars(p: float) -> str:
    """Compact significance marker for table cells."""
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "$^{***}$"
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""
