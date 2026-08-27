"""HEAAF-Bench: a reproducible access-request testbed for a clinical unit.

The testbed is *semi-synthetic*.  The behavioural channel (three keystroke
features) is taken from the CMU keystroke benchmark, so the hardest part of
the problem -- telling two human typists apart -- uses real human data.  The
contextual channel (device, network, roster, resource sensitivity, care
relationship, peer deviation) is generated from a documented parametric model
of a 51-clinician hospital department.

Two properties make the testbed useful for explainability research and are
impossible to obtain from a production access log:

1. every malicious event carries a **ground-truth driver set** -- the features
   the generator actually perturbed -- so an explanation can be scored against
   the causal mechanism rather than against another explainer;
2. the benign class contains **hard negatives** (night cover, replacement
   workstations, locums, code-blue access, cross-ward cover, remote on-call)
   which are unusual *and* legitimate, so a false-positive rate measured here
   reflects real clinical friction rather than an easy benign baseline.

All randomness is drawn from an explicitly seeded ``numpy.random.Generator``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from . import keystroke as ks
from .config import (ATTACKS, ATTACK_NAMES, D, FEATURES, FIDX,
                     HARD_NEGATIVES, HARD_NEGATIVE_NAMES, SimConfig)

ROLES = ["nurse", "physician", "allied", "admin", "sysadmin", "vendor"]
ROLE_MIX = np.array([0.51, 0.24, 0.12, 0.06, 0.04, 0.03])


@dataclass
class Staff:
    uid: int
    subject: str          # CMU subject providing the behavioural identity
    role: str
    shift: int            # 0 day, 1 evening, 2 night
    shared_ws_p: float    # probability of working from a shared workstation
    remote_p: float       # probability of legitimate off-network access
    sens_mean: float      # typical sensitivity of the resources touched
    vol_scale: float      # personal volume dispersion


def build_staff(subjects: Sequence[str], rng: np.random.Generator) -> List[Staff]:
    staff: List[Staff] = []
    roles = rng.choice(ROLES, size=len(subjects), p=ROLE_MIX)
    for i, (s, role) in enumerate(zip(subjects, roles)):
        shift = int(rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2]))
        cfg = {
            "nurse":     (0.80, 0.02, 0.30, 1.0),
            "physician": (0.45, 0.18, 0.45, 1.1),
            "allied":    (0.60, 0.05, 0.35, 0.9),
            "admin":     (0.25, 0.10, 0.55, 1.3),
            "sysadmin":  (0.15, 0.35, 0.80, 1.5),
            "vendor":    (0.10, 0.75, 0.60, 1.2),
        }[role]
        staff.append(Staff(uid=i, subject=s, role=role, shift=shift,
                           shared_ws_p=cfg[0], remote_p=cfg[1],
                           sens_mean=cfg[2], vol_scale=cfg[3]))
    return staff


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


class BenchGenerator:
    """Generates an event trace for one data split."""

    def __init__(self, pool: ks.KeystrokePool, cfg: SimConfig, seed: int):
        self.pool = pool
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.staff = build_staff(pool.subjects, self.rng)
        self.attack_p = np.array([ATTACKS[a]["weight"] for a in ATTACK_NAMES])
        self.attack_p = self.attack_p / self.attack_p.sum()

    # ------------------------------------------------------------------
    # Per-event feature construction
    # ------------------------------------------------------------------
    def _benign_context(self, st: Staff, emergency: bool) -> np.ndarray:
        r = self.rng
        x = np.zeros(D)
        x[FIDX["dev_unknown"]] = _clip01(r.binomial(1, 0.03) * r.uniform(.5, 1.))
        x[FIDX["dev_risk"]] = _clip01(
            r.binomial(1, st.shared_ws_p) * r.uniform(.15, .45) +
            abs(r.normal(0, .06)))
        remote = r.binomial(1, st.remote_p)
        x[FIDX["net_zone_risk"]] = _clip01(
            remote * r.uniform(.45, .70) + (1 - remote) * abs(r.normal(0, .05)))
        x[FIDX["geo_novel"]] = _clip01(remote * r.uniform(0, .35) +
                                       abs(r.normal(0, .04)))
        x[FIDX["impossible_travel"]] = 0.0
        x[FIDX["time_dev"]] = _clip01(abs(r.normal(0, .13)))
        x[FIDX["res_sens"]] = _clip01(r.beta(2.0, 2.0) * .5 + st.sens_mean * .5)
        x[FIDX["vol_z"]] = _clip01(abs(r.normal(0, .16)) * st.vol_scale)
        x[FIDX["no_care_rel"]] = _clip01(r.binomial(1, .07) * r.uniform(.3, .8) +
                                         abs(r.normal(0, .05)))
        x[FIDX["peer_dev"]] = _clip01(abs(r.normal(0, .15)))
        x[FIDX["fail_recent"]] = _clip01(r.binomial(1, .04) * r.uniform(.1, .35))
        x[FIDX["emergency"]] = 1.0 if emergency else 0.0
        return x

    def _apply_hard_negative(self, x: np.ndarray, kind: str) -> np.ndarray:
        r = self.rng
        bump = {
            "H1_night_cover": {"time_dev": (.55, .95), "peer_dev": (.35, .70)},
            "H2_new_device": {"dev_unknown": (.85, 1.0), "dev_risk": (.30, .60)},
            "H3_locum": {"peer_dev": (.55, .90), "no_care_rel": (.45, .80),
                          "dev_unknown": (.60, .95)},
            "H4_code_blue": {"no_care_rel": (.70, 1.0), "vol_z": (.40, .75),
                              "res_sens": (.55, .90)},
            "H5_cross_ward": {"no_care_rel": (.50, .85), "peer_dev": (.30, .65)},
            "H6_remote_oncall": {"net_zone_risk": (.55, .85),
                                  "geo_novel": (.40, .80), "time_dev": (.40, .75)},
        }[kind]
        for f, (lo, hi) in bump.items():
            x[FIDX[f]] = max(x[FIDX[f]], r.uniform(lo, hi))
        if kind == "H4_code_blue":
            x[FIDX["emergency"]] = 1.0
        return x

    def _apply_attack(self, x: np.ndarray, kind: str, escalation: float,
                      stealth: float, active: Sequence[str]) -> np.ndarray:
        """Perturb the declared driver features of an attack scenario.

        Three properties keep the task realistically hard.  ``escalation`` in
        [0,1] grows with the adversary's dwell time, so early containment is
        worth something.  ``stealth`` in [0,1] is drawn once per compromised
        session and shrinks the perturbation, so the corpus contains both loud
        and quiet adversaries.  ``active`` is the subset of the scenario's
        drivers that this particular adversary actually trips -- a competent
        attacker does not light up every indicator at once.  Crucially, no
        perturbation is applied to non-driver features: an attack leaves no
        global fingerprint that a classifier could latch onto instead of the
        mechanism.
        """
        r = self.rng
        lo, hi = {
            "A1_credential_theft": (.42, .88),
            "A2_lateral_movement": (.34, .78),
            "A3_insider_snooping": (.30, .72),
            "A4_vendor_abuse": (.34, .78),
            "A5_iomt_spoof": (.38, .82),
            "A6_session_hijack": (.34, .78),
        }[kind]
        damp = 1.0 - 0.55 * stealth
        for f in active:
            lift = r.uniform(lo, hi) * damp * (0.78 + 0.30 * escalation)
            x[FIDX[f]] = _clip01(max(x[FIDX[f]], lift))
        if kind == "A1_credential_theft" and r.random() < 0.35 * damp:
            x[FIDX["impossible_travel"]] = _clip01(r.uniform(.5, 1.0))
        if kind == "A5_iomt_spoof":
            x[FIDX["emergency"]] = 0.0
        return x

    def _keystroke(self, st: Staff, impostor: bool) -> np.ndarray:
        """Draw a real keystroke observation and map it to three features."""
        src = self.pool.impostor if impostor else self.pool.genuine
        arr = src[st.subject]
        row = arr[self.rng.integers(len(arr))]
        # squash z-scores into [0,1] with a soft logistic so that the feature
        # space is bounded (required for the reference-Shapley baseline)
        return _clip01(1.0 / (1.0 + np.exp(-(row - 1.0) / 1.6)))

    # ------------------------------------------------------------------
    # Trace generation
    # ------------------------------------------------------------------
    def generate(self, n_events: int) -> pd.DataFrame:
        cfg, r = self.cfg, self.rng
        rows: List[Dict] = []
        sid = 0
        while len(rows) < n_events:
            st = self.staff[r.integers(len(self.staff))]
            length = max(1, int(r.poisson(cfg.mean_session_len)))
            # `attack_rate` is specified per *event*; because a compromised
            # session is malicious for almost its whole length, the session
            # level probability is essentially the same quantity.
            is_attack = r.random() < cfg.attack_rate
            atk = ATTACK_NAMES[r.choice(len(ATTACK_NAMES), p=self.attack_p)] \
                if is_attack else ""
            # in a hijack the session starts benign and turns malicious
            turn = int(r.integers(2, max(3, length))) if atk == "A6_session_hijack" else 0
            emergency_sess = r.random() < cfg.emergency_rate
            # per-session adversary competence and the driver subset actually
            # tripped by this particular intrusion
            stealth = float(r.beta(2.0, 2.0)) if is_attack else 0.0
            active: List[str] = []
            if is_attack:
                drv = ATTACKS[atk]["drivers"]
                mask = r.random(len(drv)) < 0.75
                if not mask.any():
                    mask[r.integers(len(drv))] = True
                active = [f for f, m in zip(drv, mask) if m]
            hn = ""
            if not is_attack and r.random() < cfg.hard_negative_rate:
                hn = HARD_NEGATIVE_NAMES[r.integers(len(HARD_NEGATIVE_NAMES))]
            for step in range(length):
                malicious = bool(atk) and step >= turn
                emergency = emergency_sess or (hn == "H4_code_blue")
                x = self._benign_context(st, emergency)
                # open-loop value of "time since last strong authentication";
                # the closed-loop environment overwrites this from the action
                # history, but the static trace needs a well-defined value so
                # that open-loop (shadow-mode) scoring is possible.
                x[FIDX["auth_age"]] = min(1.0, step / 10.0)
                # behavioural channel: impostor material only for identity
                # takeover scenarios; insiders and vendors use their own hands
                impostor = malicious and atk in (
                    "A1_credential_theft", "A6_session_hijack")
                x[[FIDX["kd_dist"], FIDX["kd_speed_dev"], FIDX["kd_flight_var"]]] = \
                    self._keystroke(st, impostor)
                if hn:
                    x = self._apply_hard_negative(x, hn)
                if malicious:
                    esc = (step - turn) / max(1, length - turn)
                    x = self._apply_attack(x, atk, esc, stealth, active)
                rows.append({
                    "session": sid, "step": step, "length": length,
                    "uid": st.uid, "subject": st.subject, "role": st.role,
                    "y": int(malicious), "attack": atk if malicious else "",
                    "hardneg": hn, "emergency": int(emergency),
                    **{f: x[FIDX[f]] for f in FEATURES},
                })
                if len(rows) >= n_events:
                    break
            sid += 1
        df = pd.DataFrame(rows)
        return df


def make_splits(cfg: SimConfig | None = None,
                seed: int | None = None) -> Dict[str, pd.DataFrame]:
    """Build train / validation / test traces with a temporal behavioural split.

    Enrolment sessions (1-4) fit the keystroke templates; sessions 5-6 supply
    the behavioural material for training and validation; sessions 7-8 -- the
    most template-aged -- supply the test material.  The split therefore
    contains genuine behavioural drift rather than a random shuffle.
    """
    cfg = cfg or SimConfig()
    seed = cfg.seed if seed is None else seed
    df = ks.load_keystroke()
    templates = ks.build_templates(df, cfg.enrol_sessions)
    rng = np.random.default_rng(seed)
    pool_tr = ks.build_pool(df, templates, cfg.val_sessions, rng=rng)
    pool_te = ks.build_pool(df, templates, cfg.test_sessions, rng=rng)

    gen_tr = BenchGenerator(pool_tr, cfg, seed)
    gen_va = BenchGenerator(pool_tr, cfg, seed + 101)
    gen_va.staff = gen_tr.staff          # same establishment
    gen_te = BenchGenerator(pool_te, cfg, seed + 202)
    gen_te.staff = gen_tr.staff

    n = cfg.n_events
    return {
        "train": gen_tr.generate(n),
        "val": gen_va.generate(n // 4),
        "test": gen_te.generate(n // 2),
        "templates": templates,
        "staff": gen_tr.staff,
    }


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURES].to_numpy(dtype=np.float64)


def ground_truth_drivers(df: pd.DataFrame) -> List[List[int]]:
    """Indices of the features the generator perturbed, per malicious event."""
    out: List[List[int]] = []
    for atk in df["attack"]:
        if atk:
            out.append([FIDX[f] for f in ATTACKS[atk]["drivers"]])
        else:
            out.append([])
    return out
