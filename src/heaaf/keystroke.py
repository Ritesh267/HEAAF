"""Behavioural-biometric layer of HEAAF.

The behavioural evidence used in the evaluation is *real*: it comes from the
CMU keystroke-dynamics benchmark of Killourhy and Maxion (DSN 2009), in which
51 subjects each typed the password ``.tie5Roanl`` 400 times across 8 sessions
separated by at least 24 hours.  We use the sessions themselves as a natural
source of template ageing (behavioural drift): early sessions enrol the
template, later sessions are used for testing.

The detector is the *scaled Manhattan* distance, which was the strongest of
the fourteen algorithms benchmarked in the original study.  We reproduce its
equal-error rate as a sanity check (see ``evaluate_eer``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import KEYSTROKE_CSV

META_COLS = ["subject", "sessionIndex", "rep"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_keystroke(path=None) -> pd.DataFrame:
    """Load the CMU benchmark CSV (31 timing features + 3 metadata columns)."""
    path = path or KEYSTROKE_CSV
    df = pd.read_csv(path)
    missing = [c for c in META_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Unexpected keystroke file layout, missing {missing}")
    feat_cols = [c for c in df.columns if c not in META_COLS]
    if len(feat_cols) != 31:
        raise ValueError(f"Expected 31 timing features, found {len(feat_cols)}")
    return df


def timing_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def hold_columns(cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c.startswith("H.")]


def flight_columns(cols: Sequence[str]) -> List[str]:
    # UD.* are up-down (flight) latencies; DD.* are down-down digraph times
    return [c for c in cols if c.startswith("UD.")]


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------
@dataclass
class KeystrokeTemplate:
    """Per-subject enrolment template for the scaled-Manhattan detector."""

    subject: str
    mu: np.ndarray          # (31,) mean timing vector
    mad: np.ndarray         # (31,) mean absolute deviation, floored
    d_mu: float             # mean genuine distance during enrolment
    d_sd: float             # std of genuine distance during enrolment
    speed_mu: float         # mean total typing time
    speed_sd: float
    fvar_mu: float          # mean flight-time dispersion
    fvar_sd: float

    def distance(self, X: np.ndarray) -> np.ndarray:
        """Scaled-Manhattan distance of one or more timing vectors."""
        X = np.atleast_2d(X)
        return np.abs(X - self.mu).__truediv__(self.mad).sum(axis=1)

    def z_distance(self, X: np.ndarray) -> np.ndarray:
        """Distance standardised by the subject's own enrolment spread."""
        return (self.distance(X) - self.d_mu) / self.d_sd

    def z_speed(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return (X.sum(axis=1) - self.speed_mu) / self.speed_sd

    def z_fvar(self, X: np.ndarray, fidx: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return (X[:, fidx].std(axis=1) - self.fvar_mu) / self.fvar_sd


def build_templates(df: pd.DataFrame,
                    enrol_sessions: Sequence[int]) -> Dict[str, KeystrokeTemplate]:
    """Fit one enrolment template per subject from the enrolment sessions."""
    cols = timing_columns(df)
    fidx = np.array([cols.index(c) for c in flight_columns(cols)])
    templates: Dict[str, KeystrokeTemplate] = {}
    enrol = df[df.sessionIndex.isin(enrol_sessions)]
    for subject, grp in enrol.groupby("subject"):
        X = grp[cols].to_numpy(dtype=float)
        mu = X.mean(axis=0)
        mad = np.abs(X - mu).mean(axis=0)
        mad = np.maximum(mad, 1e-4)                 # floor to avoid blow-up
        d = (np.abs(X - mu) / mad).sum(axis=1)
        speeds = X.sum(axis=1)
        fvar = X[:, fidx].std(axis=1)
        templates[subject] = KeystrokeTemplate(
            subject=subject, mu=mu, mad=mad,
            d_mu=float(d.mean()), d_sd=float(d.std() + 1e-9),
            speed_mu=float(speeds.mean()), speed_sd=float(speeds.std() + 1e-9),
            fvar_mu=float(fvar.mean()), fvar_sd=float(fvar.std() + 1e-9),
        )
    return templates


# --------------------------------------------------------------------------
# Sanity check: reproduce the benchmark equal-error rate
# --------------------------------------------------------------------------
def evaluate_eer(df: pd.DataFrame,
                 templates: Dict[str, KeystrokeTemplate],
                 test_sessions: Sequence[int],
                 n_impostor_per_subject: int = 5,
                 rng: np.random.Generator | None = None) -> Dict[str, float]:
    """Equal-error rate of the scaled-Manhattan detector.

    Protocol follows Killourhy and Maxion: genuine scores come from the
    subject's held-out repetitions; impostor scores come from the first
    repetitions of every *other* subject.
    """
    rng = rng or np.random.default_rng(0)
    cols = timing_columns(df)
    test = df[df.sessionIndex.isin(test_sessions)]
    subjects = sorted(templates)
    eers: List[float] = []
    for s in subjects:
        tpl = templates[s]
        gen = tpl.distance(test[test.subject == s][cols].to_numpy(float))
        imp_mask = ((df.subject != s) & (df.sessionIndex == 1) &
                    (df.rep <= n_impostor_per_subject))
        imp = tpl.distance(df[imp_mask][cols].to_numpy(float))
        eers.append(_eer(gen, imp))
    return {"eer_mean": float(np.mean(eers)), "eer_sd": float(np.std(eers))}


def _eer(genuine: np.ndarray, impostor: np.ndarray) -> float:
    """Equal-error rate from genuine (low) and impostor (high) score sets."""
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    best, best_gap = 0.5, np.inf
    for t in thresholds:
        far = float((impostor <= t).mean())   # impostor accepted
        frr = float((genuine > t).mean())     # genuine rejected
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best = gap, 0.5 * (far + frr)
    return best


# --------------------------------------------------------------------------
# Sampling pools used by the simulator
# --------------------------------------------------------------------------
@dataclass
class KeystrokePool:
    """Pre-computed behavioural feature triples per (subject, split)."""

    subjects: List[str]
    genuine: Dict[str, np.ndarray]   # subject -> (n, 3) [z_dist, z_speed, z_fvar]
    impostor: Dict[str, np.ndarray]  # subject -> (m, 3) scored against subject


def build_pool(df: pd.DataFrame,
               templates: Dict[str, KeystrokeTemplate],
               sessions: Sequence[int],
               impostor_reps: int = 8,
               rng: np.random.Generator | None = None) -> KeystrokePool:
    """Score every subject's own samples and a pool of impostor samples.

    ``impostor`` entries are what an adversary typing the victim's password
    would look like to the victim's template.  Impostor material is drawn
    only from *other* subjects, so genuine and impostor identities are
    disjoint by construction.
    """
    rng = rng or np.random.default_rng(0)
    cols = timing_columns(df)
    fidx = np.array([cols.index(c) for c in flight_columns(cols)])
    sub_df = df[df.sessionIndex.isin(sessions)]
    subjects = sorted(templates)
    genuine: Dict[str, np.ndarray] = {}
    impostor: Dict[str, np.ndarray] = {}
    for s in subjects:
        tpl = templates[s]
        Xg = sub_df[sub_df.subject == s][cols].to_numpy(float)
        genuine[s] = np.column_stack(
            [tpl.z_distance(Xg), tpl.z_speed(Xg), tpl.z_fvar(Xg, fidx)])
        others = [o for o in subjects if o != s]
        pick = rng.choice(others, size=min(20, len(others)), replace=False)
        Xi = sub_df[sub_df.subject.isin(pick) &
                    (sub_df.rep <= impostor_reps)][cols].to_numpy(float)
        impostor[s] = np.column_stack(
            [tpl.z_distance(Xi), tpl.z_speed(Xi), tpl.z_fvar(Xi, fidx)])
    return KeystrokePool(subjects=subjects, genuine=genuine, impostor=impostor)
