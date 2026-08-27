"""Policy engine: risk score -> authentication action.

The policy engine is deliberately *separate* from the learned agent.  The agent
supplies a calibrated risk score and an attribution vector; the policy engine
applies the deterministic, auditable rules that a hospital's information
governance committee can read, amend and sign off:

* two thresholds define the ALLOW / step-up / DENY bands;
* a sensitivity floor forces at least a one-time code on the most sensitive
  resources regardless of how benign the context looks;
* an emergency relief term discounts the score in a declared clinical
  emergency, and a **safety valve** guarantees that a declared emergency is
  never hard-denied by the automated system -- it is escalated to strong
  authentication with an alert instead.

The safety valve is the direct answer to the objection that adaptive MFA can
lock a clinician out during resuscitation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .config import ACTIONS, AIDX, FIDX, PolicyConfig


@dataclass
class Decision:
    action: str
    risk: float
    adjusted_risk: float
    reason_code: str


class PolicyEngine:
    def __init__(self, cfg: PolicyConfig | None = None):
        self.cfg = cfg or PolicyConfig()

    def decide(self, risk: float, x: np.ndarray) -> Decision:
        c = self.cfg
        emergency = x[FIDX["emergency"]] > 0.5
        sens = x[FIDX["res_sens"]]
        r = risk - (c.emergency_relief if emergency else 0.0)
        r = float(np.clip(r, 0.0, 1.0))

        if r >= c.tau_high:
            action, code = "DENY", "risk_above_deny_threshold"
        elif r >= c.tau_low:
            action = "STEP_UP_STRONG" if r >= 0.5 * (c.tau_low + c.tau_high) \
                else "STEP_UP_OTP"
            code = "risk_in_stepup_band"
        else:
            action, code = "ALLOW", "risk_below_allow_threshold"

        if sens >= c.sens_floor and action == "ALLOW":
            action, code = "STEP_UP_OTP", "sensitivity_floor"

        if c.safety_valve and emergency and action == "DENY":
            action, code = "STEP_UP_STRONG", "emergency_safety_valve"

        return Decision(action=action, risk=float(risk), adjusted_risk=r,
                        reason_code=code)

    def decide_batch(self, risk: np.ndarray, X: np.ndarray):
        return [self.decide(float(r), x) for r, x in zip(risk, X)]

    def actions_batch(self, risk: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Vectorised variant returning action indices."""
        c = self.cfg
        emerg = X[:, FIDX["emergency"]] > 0.5
        sens = X[:, FIDX["res_sens"]]
        r = np.clip(risk - emerg * c.emergency_relief, 0.0, 1.0)
        mid = 0.5 * (c.tau_low + c.tau_high)
        a = np.full(len(r), AIDX["ALLOW"], dtype=np.int64)
        a[(r >= c.tau_low) & (r < mid)] = AIDX["STEP_UP_OTP"]
        a[(r >= mid) & (r < c.tau_high)] = AIDX["STEP_UP_STRONG"]
        a[r >= c.tau_high] = AIDX["DENY"]
        a[(sens >= c.sens_floor) & (a == AIDX["ALLOW"])] = AIDX["STEP_UP_OTP"]
        if c.safety_valve:
            a[emerg & (a == AIDX["DENY"])] = AIDX["STEP_UP_STRONG"]
        return a
