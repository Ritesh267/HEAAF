"""Evaluation metrics.

Four families are reported, because an authentication control that is only
scored on detection is not evaluated at all in a clinical setting:

* **security** -- discrimination and per-scenario detection;
* **friction** -- what the control costs clinicians, expressed in challenges
  per shift and in whole-time-equivalent clinician hours per year;
* **explanation quality** -- fidelity to exact Shapley values, agreement with
  the generator's ground-truth causal drivers, sparsity and stability;
* **calibration** -- whether the number the policy engine thresholds means
  what it claims to mean.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, matthews_corrcoef,
                             roc_auc_score)

from .config import (ACTION_SECONDS, ACTIONS, AIDX, CLINICIAN_RELOGINS_PER_DAY,
                     DEPT_CLINICIANS, SHIFTS_PER_YEAR)


# ==========================================================================
# Security
# ==========================================================================
def gmean(tpr: float, tnr: float) -> float:
    return float(np.sqrt(max(tpr, 0.0) * max(tnr, 0.0)))


def eer(scores: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Equal-error rate for a risk score (higher = more suspicious).

    A request is flagged when ``score >= t``; the crossing point of the false
    positive rate (legitimate access interrupted) and the false negative rate
    (adversary silently admitted) is returned together with its threshold.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    order = np.argsort(scores, kind="mergesort")
    s, yy = scores[order], y[order]
    P, N = int(yy.sum()), int(len(yy) - yy.sum())
    if P == 0 or N == 0:
        return float("nan"), float("nan")
    neg_below = np.cumsum(1 - yy) - (1 - yy)   # negatives strictly before k
    pos_below = np.cumsum(yy) - yy             # positives strictly before k
    fpr = (N - neg_below) / N
    fnr = pos_below / P
    k = int(np.argmin(np.abs(fpr - fnr)))
    return float(0.5 * (fpr[k] + fnr[k])), float(s[k])


def detection_metrics(y: np.ndarray, risk: np.ndarray,
                      flagged: np.ndarray) -> Dict[str, float]:
    """`flagged` is 1 when the system did anything other than silently allow."""
    y = np.asarray(y).astype(int)
    flagged = np.asarray(flagged).astype(int)
    tp = int(((y == 1) & (flagged == 1)).sum())
    fn = int(((y == 1) & (flagged == 0)).sum())
    fp = int(((y == 0) & (flagged == 1)).sum())
    tn = int(((y == 0) & (flagged == 0)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / max(prec + tpr, 1e-12)
    out = {
        "TPR": tpr, "FPR": 1 - tnr, "TNR": tnr, "precision": prec, "F1": f1,
        "GMean": gmean(tpr, tnr),
        "MCC": float(matthews_corrcoef(y, flagged)) if len(set(flagged)) > 1 else 0.0,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    }
    if risk is not None and len(set(y)) > 1:
        out["AUROC"] = float(roc_auc_score(y, risk))
        out["AUPRC"] = float(average_precision_score(y, risk))
        out["EER"] = eer(risk, y)[0]
    return out


def per_scenario_detection(df, flagged: np.ndarray, col: str = "attack"
                           ) -> Dict[str, float]:
    """Detection rate per scenario; `df` must be positionally aligned to `flagged`."""
    flagged = np.asarray(flagged)
    labels = df[col].to_numpy()
    out: Dict[str, float] = {}
    for name in np.unique(labels):
        if name == "":
            continue
        m = labels == name
        out[str(name)] = float(flagged[m].mean())
    return out


# ==========================================================================
# Friction (clinical cost of the control)
# ==========================================================================
def friction_metrics(df, actions: np.ndarray) -> Dict[str, float]:
    y = df["y"].to_numpy()
    emerg = df["emergency"].to_numpy()
    hardneg = (df["hardneg"].to_numpy() != "")
    ben = y == 0
    act = np.asarray(actions)
    is_stepup = np.isin(act, [AIDX["STEP_UP_OTP"], AIDX["STEP_UP_STRONG"]])
    is_deny = act == AIDX["DENY"]
    secs = np.array([ACTION_SECONDS[ACTIONS[a]] for a in act])

    stepup_rate = float(is_stepup[ben].mean())
    deny_rate = float(is_deny[ben].mean())
    emerg_ben = ben & (emerg == 1)
    hard_ben = ben & hardneg
    routine_ben = ben & ~hardneg & (emerg == 0)

    sec_per_event = float(secs[ben].mean())
    per_shift = sec_per_event * CLINICIAN_RELOGINS_PER_DAY
    hours_year = per_shift * SHIFTS_PER_YEAR * DEPT_CLINICIANS / 3600.0
    return {
        "stepup_rate_benign": stepup_rate,
        "deny_rate_benign": deny_rate,
        "interrupt_rate_benign": float((is_stepup | is_deny)[ben].mean()),
        "deny_rate_emergency": float(is_deny[emerg_ben].mean()) if emerg_ben.any() else 0.0,
        "interrupt_rate_emergency": float((is_stepup | is_deny)[emerg_ben].mean()) if emerg_ben.any() else 0.0,
        "interrupt_rate_hardneg": float((is_stepup | is_deny)[hard_ben].mean()) if hard_ben.any() else 0.0,
        "interrupt_rate_routine": float((is_stepup | is_deny)[routine_ben].mean()) if routine_ben.any() else 0.0,
        "auth_seconds_per_access": sec_per_event,
        "auth_seconds_per_clinician_shift": per_shift,
        "dept_clinician_hours_per_year": hours_year,
    }


def containment_metrics(df, actions: np.ndarray) -> Dict[str, float]:
    """How far into a compromised session the adversary gets before contained."""
    act = np.asarray(actions)
    contained_at, uncontained = [], 0
    for _, grp in df[df.y == 1].groupby("session"):
        idx = grp.index.to_numpy()
        a = act[idx]
        hit = np.where(a != AIDX["ALLOW"])[0]
        if len(hit) == 0:
            uncontained += 1
        else:
            contained_at.append(int(hit[0]) + 1)
    n = len(contained_at) + uncontained
    return {
        "sessions_compromised": n,
        "session_containment_rate": float(len(contained_at) / max(n, 1)),
        "median_steps_to_containment": float(np.median(contained_at)) if contained_at else float("nan"),
        "mean_steps_to_containment": float(np.mean(contained_at)) if contained_at else float("nan"),
        "contained_within_1_step": float(np.mean([c <= 1 for c in contained_at])) if contained_at else 0.0,
        "contained_within_3_steps": float(np.mean([c <= 3 for c in contained_at])) if contained_at else 0.0,
    }


# ==========================================================================
# Calibration
# ==========================================================================
def calibration_metrics(y: np.ndarray, p: np.ndarray, bins: int = 15) -> Dict[str, float]:
    y = np.asarray(y).astype(float)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece, mce = 0.0, 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        gap = abs(p[m].mean() - y[m].mean())
        ece += m.mean() * gap
        mce = max(mce, gap)
    return {"brier": brier, "ECE": float(ece), "MCE": float(mce)}


def reliability_curve(y, p, bins: int = 12):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() < 5:
            continue
        xs.append(float(np.mean(p[m])))
        ys.append(float(np.mean(y[m])))
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


# ==========================================================================
# Explanation quality
# ==========================================================================
def topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 3) -> float:
    ta = set(np.argsort(-a)[:k])
    tb = set(np.argsort(-b)[:k])
    return len(ta & tb) / k


def explanation_fidelity(phi_hat: np.ndarray, phi_ref: np.ndarray,
                         ks: Sequence[int] = (1, 3, 5)) -> Dict[str, float]:
    """Agreement between an approximate attribution and the exact Shapley values."""
    out: Dict[str, float] = {}
    rhos, l1s = [], []
    for k in ks:
        out[f"top{k}_overlap"] = float(np.mean(
            [topk_overlap(a, b, k) for a, b in zip(phi_hat, phi_ref)]))
    for a, b in zip(phi_hat, phi_ref):
        rho = spearmanr(a, b).statistic
        rhos.append(0.0 if np.isnan(rho) else float(rho))
        l1s.append(float(np.abs(a - b).sum() / (np.abs(b).sum() + 1e-12)))
    out["spearman"] = float(np.mean(rhos))
    out["spearman_sd"] = float(np.std(rhos))
    out["rel_L1"] = float(np.mean(l1s))
    return out


def completeness_gap(phi: np.ndarray, f_x: np.ndarray, f_ref: float) -> float:
    """|sum(phi) - (f(x) - f(ref))| normalised by the target magnitude."""
    target = f_x - f_ref
    return float(np.mean(np.abs(phi.sum(axis=1) - target) /
                         (np.abs(target) + 1e-9)))


def driver_recall(phi: np.ndarray, drivers: List[List[int]],
                  k: int = 3) -> Dict[str, float]:
    """Does the explanation surface the feature the generator actually moved?"""
    hits_any, hits_all = [], []
    for p, dr in zip(phi, drivers):
        if not dr:
            continue
        top = set(np.argsort(-p)[:k])
        hits_any.append(float(len(top & set(dr)) > 0))
        hits_all.append(len(top & set(dr)) / min(k, len(dr)))
    return {f"driver_hit@{k}": float(np.mean(hits_any)) if hits_any else float("nan"),
            f"driver_coverage@{k}": float(np.mean(hits_all)) if hits_all else float("nan")}


def sparsity(phi: np.ndarray) -> float:
    """Fraction of total attribution mass carried by the top three features."""
    out = []
    for p in phi:
        a = np.abs(p)
        s = a.sum() + 1e-12
        out.append(float(np.sort(a)[-3:].sum() / s))
    return float(np.mean(out))


def stability(f, phi_fn, X: np.ndarray, eps: float = 0.01,
              rng: np.random.Generator | None = None) -> float:
    """Max relative change of the attribution under an eps-perturbation."""
    rng = rng or np.random.default_rng(0)
    Xp = np.clip(X + rng.normal(0, eps, X.shape), 0, 1)
    p0, p1 = phi_fn(X), phi_fn(Xp)
    num = np.linalg.norm(p1 - p0, axis=1)
    den = np.linalg.norm(X - Xp, axis=1) + 1e-12
    return float(np.median(num / den))
