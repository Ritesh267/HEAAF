"""Central configuration for the HEAAF reference implementation.

Everything that a reviewer might want to change (cost model, thresholds,
network size, base rates) lives here so that no magic numbers are buried
inside the experiment scripts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
TABDIR = RESULTS / "tables"
LOGDIR = RESULTS / "logs"

for _d in (DATA_PROC, RESULTS, FIGDIR, TABDIR, LOGDIR):
    _d.mkdir(parents=True, exist_ok=True)

KEYSTROKE_CSV = DATA_RAW / "DSL-StrongPasswordData.csv"

# --------------------------------------------------------------------------
# Feature space  (Section IV-A of the paper)
# --------------------------------------------------------------------------
# Every feature is oriented so that *larger = more suspicious*, except
# `emergency`, which is a clinical-context modifier.  Keeping a consistent
# orientation makes the Shapley attributions directly readable by a SOC
# analyst without a sign lookup table.

FEATURES: List[str] = [
    "kd_dist",            # 0  scaled-Manhattan keystroke template distance (z)
    "kd_speed_dev",       # 1  typing-speed deviation from personal baseline (z)
    "kd_flight_var",      # 2  inter-key (flight) timing variability deviation (z)
    "dev_unknown",        # 3  device unseen in this user's device history
    "dev_risk",           # 4  device posture / shared-workstation risk
    "net_zone_risk",      # 5  network zone (clinical LAN -> unknown ISP)
    "geo_novel",          # 6  geolocation novelty w.r.t. user history
    "impossible_travel",  # 7  velocity-based impossible-travel indicator
    "time_dev",           # 8  deviation from the user's rostered shift profile
    "res_sens",           # 9  sensitivity of the requested resource
    "vol_z",              # 10 record-access volume z-score vs personal baseline
    "no_care_rel",        # 11 absence of a clinician-patient care relationship
    "peer_dev",           # 12 deviation from the role peer group
    "auth_age",           # 13 time since last strong authentication (norm.)
    "fail_recent",        # 14 recent failed authentication attempts (norm.)
    "emergency",          # 15 declared clinical emergency context
]
D = len(FEATURES)
FIDX: Dict[str, int] = {f: i for i, f in enumerate(FEATURES)}

# Human-readable names used by the natural-language explanation renderer.
FEATURE_LABELS: Dict[str, str] = {
    "kd_dist": "typing rhythm differs from the enrolled template",
    "kd_speed_dev": "typing speed is unusual for this user",
    "kd_flight_var": "key-transition timing is unusually irregular",
    "dev_unknown": "the device has not been used by this account before",
    "dev_risk": "the endpoint is shared or fails posture checks",
    "net_zone_risk": "the connection originates outside the clinical network",
    "geo_novel": "the access location is new for this account",
    "impossible_travel": "the location is unreachable since the last login",
    "time_dev": "the access falls outside the rostered shift pattern",
    "res_sens": "the requested record or function is highly sensitive",
    "vol_z": "the volume of records accessed is far above normal",
    "no_care_rel": "there is no recorded care relationship with this patient",
    "peer_dev": "the behaviour deviates from the role peer group",
    "auth_age": "strong authentication has not been performed recently",
    "fail_recent": "recent authentication failures on this account",
    "emergency": "an emergency clinical context is declared",
}

# --------------------------------------------------------------------------
# Semantic feature groups (Section IV-D)
# --------------------------------------------------------------------------
# The 16 raw features are organised into 8 semantic groups.  This grouping is
# not cosmetic: it is the unit at which the Explainable Decision Layer reports
# to a clinician, and it is what makes *exact* group-level Shapley values
# affordable inside the decision cycle (2^8 = 256 coalitions rather than
# 2^16 = 65,536).  Groups are mutually exclusive and jointly exhaustive, so
# group attributions inherit efficiency from the feature-level game.

GROUPS: Dict[str, List[str]] = {
    "Keystroke dynamics":   ["kd_dist", "kd_speed_dev", "kd_flight_var"],
    "Device posture":       ["dev_unknown", "dev_risk"],
    "Network and location": ["net_zone_risk", "geo_novel", "impossible_travel"],
    "Temporal context":     ["time_dev"],
    "Access pattern":       ["vol_z", "no_care_rel"],
    "Peer-group behaviour": ["peer_dev"],
    "Session assurance":    ["auth_age", "fail_recent"],
    "Clinical context":     ["res_sens", "emergency"],
}
GROUP_NAMES: List[str] = list(GROUPS)
G = len(GROUP_NAMES)
GROUP_OF: Dict[str, str] = {f: g for g, fs in GROUPS.items() for f in fs}
GROUP_IDX: Dict[str, List[int]] = {g: [FIDX[f] for f in fs] for g, fs in GROUPS.items()}
GIDX: Dict[str, int] = {g: i for i, g in enumerate(GROUP_NAMES)}

# Clinician-facing phrasing, one per group.
GROUP_LABELS: Dict[str, str] = {
    "Keystroke dynamics":   "your typing rhythm differs from the enrolled pattern",
    "Device posture":       "this device is unrecognised or fails a posture check",
    "Network and location": "the connection is coming from an unusual place",
    "Temporal context":     "the access falls outside your rostered shift",
    "Access pattern":       "the records being opened are outside your normal pattern",
    "Peer-group behaviour": "the activity differs from others in your role",
    "Session assurance":    "strong authentication has not been performed recently",
    "Clinical context":     "the record requested is highly sensitive",
}

# Sanity: the grouping must partition the feature space exactly.
assert sorted(f for fs in GROUPS.values() for f in fs) == sorted(FEATURES), (
    "GROUPS must be a partition of FEATURES")
assert len({f for fs in GROUPS.values() for f in fs}) == D, "GROUPS overlap"

# --------------------------------------------------------------------------
# Action space (Section IV-C)
# --------------------------------------------------------------------------
ACTIONS: List[str] = ["ALLOW", "STEP_UP_OTP", "STEP_UP_STRONG", "DENY"]
A = len(ACTIONS)
AIDX: Dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}

# Median clinician-facing latency of each action, in seconds.  Values follow
# published measurements of hospital authentication workflows (badge tap /
# OTP / full re-authentication with a second factor).
ACTION_SECONDS: Dict[str, float] = {
    "ALLOW": 0.0,
    "STEP_UP_OTP": 8.0,
    "STEP_UP_STRONG": 25.0,
    "DENY": 45.0,   # denial forces an alternative workflow / help-desk call
}

# Probability that an adversary who is challenged actually completes the
# challenge (i.e. the residual risk of a step-up).  A stolen password does not
# usually come with the second factor; a session hijack sometimes does.
CHALLENGE_BYPASS_P: Dict[str, float] = {
    "STEP_UP_OTP": 0.18,     # OTP relay / prompt bombing succeeds sometimes
    "STEP_UP_STRONG": 0.04,  # phishing-resistant factor + biometrics
    "DENY": 0.0,
}


@dataclass
class RewardConfig:
    """Cost model of the constrained MDP (Section IV-C)."""

    # security cost of admitting an adversary, scaled by resource sensitivity
    breach_base: float = 6.0
    breach_sens_gain: float = 6.0
    # operational value of a frictionless legitimate access
    allow_benign: float = 0.20
    # friction penalty per action for legitimate users
    friction_weight: float = 1.0
    friction: Dict[str, float] = field(default_factory=lambda: {
        "ALLOW": 0.0, "STEP_UP_OTP": 0.18, "STEP_UP_STRONG": 0.52, "DENY": 1.20})
    # additional patient-safety penalty for obstructing an emergency access
    emergency_penalty: Dict[str, float] = field(default_factory=lambda: {
        "ALLOW": 0.0, "STEP_UP_OTP": 0.20, "STEP_UP_STRONG": 0.80, "DENY": 3.60})
    # reward for correctly challenging / blocking an adversary
    correct_challenge: Dict[str, float] = field(default_factory=lambda: {
        "STEP_UP_OTP": 1.20, "STEP_UP_STRONG": 2.00, "DENY": 2.40})
    # imbalance-correction factor applied to the malicious class (cf. RLAuth)
    kappa: float = 1.0
    gamma: float = 0.90


@dataclass
class AgentConfig:
    hidden: Tuple[int, ...] = (64, 64)
    lr: float = 1e-3
    batch: int = 128
    buffer: int = 60_000
    episodes: int = 90_000        # environment steps
    warmup: int = 3_000
    target_sync: int = 750
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 30_000
    balanced_replay: float = 0.5  # fraction of malicious samples per minibatch
    double_q: bool = True
    grad_clip: float = 5.0
    seed: int = 0


@dataclass
class SimConfig:
    """HEAAF-Bench generator settings (Section V-A)."""

    n_events: int = 150_000
    attack_rate: float = 0.012        # fraction of malicious events
    hard_negative_rate: float = 0.06  # unusual-but-legitimate events
    emergency_rate: float = 0.035     # events raised under emergency context
    mean_session_len: float = 12.0
    enrol_sessions: Tuple[int, ...] = (1, 2, 3, 4)
    val_sessions: Tuple[int, ...] = (5, 6)
    test_sessions: Tuple[int, ...] = (7, 8)
    seed: int = 20260821


@dataclass
class PolicyConfig:
    """Policy-engine thresholds and clinical safety valve (Section IV-E)."""

    tau_low: float = 0.25    # below -> ALLOW
    tau_high: float = 0.60   # above -> DENY (subject to safety valve)
    sens_floor: float = 0.85  # resources above this always get >= OTP
    safety_valve: bool = True  # never hard-DENY a declared emergency
    emergency_relief: float = 0.12  # risk discount granted in emergencies


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=str))


# --------------------------------------------------------------------------
# Attack taxonomy: each scenario is anchored to a documented healthcare
# incident class and declares the features it is *designed* to perturb.  Those
# declared drivers are the ground truth used to score explanation quality.
# --------------------------------------------------------------------------
ATTACKS: Dict[str, Dict] = {
    "A1_credential_theft": {
        "label": "Stolen-credential remote login",
        "anchor": "Change Healthcare (2024) / 23andMe (2023)",
        "drivers": ["kd_dist", "dev_unknown", "geo_novel", "net_zone_risk"],
        "weight": 0.30,
    },
    "A2_lateral_movement": {
        "label": "Post-intrusion lateral movement",
        "anchor": "HSHS (2023) ransomware intrusion",
        "drivers": ["time_dev", "res_sens", "peer_dev", "vol_z"],
        "weight": 0.17,
    },
    "A3_insider_snooping": {
        "label": "Insider privilege misuse",
        "anchor": "NHS ICO enforcement cases",
        "drivers": ["no_care_rel", "vol_z", "peer_dev"],
        "weight": 0.20,
    },
    "A4_vendor_abuse": {
        "label": "Third-party / vendor account abuse",
        "anchor": "MCNA Dental (2023) supply-chain breach",
        "drivers": ["net_zone_risk", "time_dev", "vol_z", "res_sens"],
        "weight": 0.13,
    },
    "A5_iomt_spoof": {
        "label": "IoMT gateway impersonation",
        "anchor": "Philips patient-monitor advisories (ICSMA-20-254-01)",
        "drivers": ["dev_risk", "dev_unknown", "res_sens", "auth_age"],
        "weight": 0.10,
    },
    "A6_session_hijack": {
        "label": "Mid-session takeover (AiTM / token theft)",
        "anchor": "Adversary-in-the-middle MFA bypass",
        "drivers": ["kd_dist", "kd_speed_dev", "peer_dev", "auth_age"],
        "weight": 0.10,
    },
}
ATTACK_NAMES: List[str] = list(ATTACKS.keys())

# Unusual-but-legitimate scenarios ("hard negatives").  These exist to make
# the false-positive analysis clinically meaningful: they are exactly the
# events that naive risk engines punish.
HARD_NEGATIVES: Dict[str, Dict] = {
    "H1_night_cover": {"label": "Unrostered night cover", "drivers": ["time_dev", "peer_dev"]},
    "H2_new_device": {"label": "Replacement workstation", "drivers": ["dev_unknown"]},
    "H3_locum": {"label": "Locum / agency first shift", "drivers": ["peer_dev", "no_care_rel", "dev_unknown"]},
    "H4_code_blue": {"label": "Emergency (code-blue) access", "drivers": ["no_care_rel", "vol_z", "emergency"]},
    "H5_cross_ward": {"label": "Cross-ward cover", "drivers": ["no_care_rel", "peer_dev"]},
    "H6_remote_oncall": {"label": "Remote on-call review", "drivers": ["net_zone_risk", "geo_novel", "time_dev"]},
}
HARD_NEGATIVE_NAMES: List[str] = list(HARD_NEGATIVES.keys())

# Operational constants used by the clinical friction-budget analysis
# (Section VI-C).  Sources are cited in the paper.
CLINICIAN_RELOGINS_PER_DAY = 14.3   # observed EHR re-authentications / clinician / day
SHIFTS_PER_YEAR = 220               # whole-time-equivalent clinical shifts
DEPT_CLINICIANS = 51                # size of the modelled department
