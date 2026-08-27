"""Explainable Decision Layer (EDL).

HEAAF separates two explanation duties that the literature usually conflates:

* **Tier 1 -- inline.** Produced *with* the decision, on the critical path of
  the access request, and shown to the clinician and to the policy engine.  It
  must cost microseconds, so it is a gradient attribution of the risk logit
  against a fixed clinical reference profile.  For the piecewise-linear risk
  network used here, gradient x input is exactly the Shapley value of the local
  linear region, and integrated gradients additionally satisfies completeness.

* **Tier 2 -- forensic.** Produced off the critical path for the audit trail,
  incident response and regulator-facing records.  It may cost milliseconds to
  seconds, so KernelSHAP -- or, for this 16-feature space, the *exact* Shapley
  values over all 2^16 = 65,536 coalitions -- is affordable.

A third option sits between them and is the one actually disclosed to the
clinician: *exact group-level Shapley values* over the 8 semantic groups.
Because the groups partition the feature space, the group game is a
well-defined cooperative game in its own right, and it needs only 2^8 = 256
coalition evaluations -- cheap enough to run inline while remaining exact.

Because Tier 2 exact Shapley values are computable for this feature space, the
paper can measure how much fidelity Tier 1 gives up, rather than assuming it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (A, AIDX, D, G, FEATURES, FEATURE_LABELS, GIDX,
                     GROUP_IDX, GROUP_LABELS, GROUP_NAMES)


# ==========================================================================
# The scalar functional being explained
# ==========================================================================
class RiskFunctional:
    """Risk *logit* as a scalar function of the feature vector.

    Attribution targets the logit rather than the probability so that the
    additive decomposition is on a scale where contributions combine linearly.
    """

    def __init__(self, agent, head):
        self.agent = agent
        self.head = head

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Q = self.agent.q_values(np.atleast_2d(X))
        adv = self.head.advantage(Q)
        z = (adv - self.head.mu) / self.head.sd
        return self.head.w * z + self.head.b

    def prob(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self(X), -30, 30)))

    def grad(self, X: np.ndarray) -> np.ndarray:
        """Exact gradient of the risk logit w.r.t. the inputs."""
        X = np.atleast_2d(X)
        Q = self.agent.q_values(X)
        others = [i for i in range(A) if i != AIDX["ALLOW"]]
        jbest = np.array(others)[np.argmax(Q[:, others], axis=1)]
        G = np.zeros_like(X)
        # group rows that select the same non-ALLOW action so one backward
        # pass serves each group
        for j in np.unique(jbest):
            m = jbest == j
            h = np.zeros(A)
            h[j] = 1.0
            h[AIDX["ALLOW"]] = -1.0
            G[m] = self.agent.q.input_gradient(X[m], h)
        return G * (self.head.w / self.head.sd)


def clinical_reference(X_benign: np.ndarray) -> np.ndarray:
    """Reference profile: the median benign access request.

    Using a single fixed reference (rather than an expectation over a
    background sample) makes the explanation contrastive against a stated
    "routine ward access" and keeps Tier 1 and Tier 2 comparable, since
    integrated gradients and reference Shapley then share a baseline.
    """
    return np.median(np.atleast_2d(X_benign), axis=0)


# ==========================================================================
# Tier 1 -- inline attributions
# ==========================================================================
def grad_x_input(f: RiskFunctional, X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """phi = grad f(x) * (x - ref).  One backward pass, O(d)."""
    X = np.atleast_2d(X)
    return f.grad(X) * (X - ref)


def integrated_gradients(f: RiskFunctional, X: np.ndarray, ref: np.ndarray,
                         m: int = 16) -> np.ndarray:
    """Riemann (midpoint) approximation of integrated gradients."""
    X = np.atleast_2d(X)
    n = X.shape[0]
    alphas = (np.arange(m) + 0.5) / m
    total = np.zeros_like(X)
    for a in alphas:
        total += f.grad(ref + a * (X - ref))
    return (total / m) * (X - ref)


# ==========================================================================
# Tier 2 -- forensic attributions
# ==========================================================================
def kernel_shap(f: RiskFunctional, x: np.ndarray, ref: np.ndarray,
                n_samples: int = 256, rng: np.random.Generator | None = None
                ) -> np.ndarray:
    """KernelSHAP with the Shapley kernel and a single reference point."""
    rng = rng or np.random.default_rng(0)
    d = len(x)
    sizes = np.arange(1, d)
    kern = (d - 1) / (sizes * (d - sizes))
    p = kern / kern.sum()
    ks = rng.choice(sizes, size=n_samples, p=p)
    Z = np.zeros((n_samples, d), dtype=bool)
    for i, k in enumerate(ks):
        Z[i, rng.choice(d, size=int(k), replace=False)] = True
    Xz = np.where(Z, x, ref)
    fz = f(Xz)
    f0, f1 = float(f(ref)[0]), float(f(x)[0])
    # constrain sum(phi) = f1 - f0 by eliminating the last coefficient
    y = fz - f0 - Z[:, -1] * (f1 - f0)
    Zt = Z[:, :-1].astype(float) - Z[:, [-1]].astype(float)
    coef, *_ = np.linalg.lstsq(Zt, y, rcond=None)
    phi = np.append(coef, (f1 - f0) - coef.sum())
    return phi


def exact_shapley(f: RiskFunctional, x: np.ndarray, ref: np.ndarray,
                  chunk: int = 8192) -> np.ndarray:
    """Exact reference Shapley values over all 2^d coalitions.

    Feasible because the HEAAF feature space has d = 16 (65,536 coalitions).
    This is the ground truth against which Tier-1 fidelity is measured.
    """
    d = len(x)
    n = 1 << d
    bits = ((np.arange(n)[:, None] >> np.arange(d)[None, :]) & 1).astype(bool)
    vals = np.empty(n)
    diff = x - ref
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        vals[s:e] = f(ref + bits[s:e] * diff)
    card = bits.sum(axis=1)
    # Shapley weights w(|S|) = |S|! (d-|S|-1)! / d!
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, d + 1)))])
    w = np.exp(logfact[np.arange(d)] + logfact[d - 1 - np.arange(d)] - logfact[d])
    phi = np.zeros(d)
    idx = np.arange(n)
    for i in range(d):
        bit = 1 << i
        without = idx[(idx & bit) == 0]
        with_i = without | bit
        phi[i] = float(np.sum(w[card[without]] * (vals[with_i] - vals[without])))
    return phi


# ==========================================================================
# Group-level attribution (what the clinician actually sees)
# ==========================================================================
_GROUP_MASKS = None


def group_masks() -> np.ndarray:
    """(G, D) boolean matrix; row g selects the features belonging to group g."""
    global _GROUP_MASKS
    if _GROUP_MASKS is None:
        M = np.zeros((G, D), dtype=bool)
        for g, name in enumerate(GROUP_NAMES):
            M[g, GROUP_IDX[name]] = True
        _GROUP_MASKS = M
    return _GROUP_MASKS


def to_groups(phi: np.ndarray) -> np.ndarray:
    """Sum feature attributions within each semantic group.

    Summation is the correct aggregation for an additive attribution: because
    the groups partition the features, the group vector inherits completeness
    from the feature vector exactly, with no residual.
    """
    phi = np.atleast_2d(phi)
    return phi @ group_masks().T.astype(float)


def exact_shapley_groups(f: RiskFunctional, x: np.ndarray,
                         ref: np.ndarray) -> np.ndarray:
    """Exact Shapley values of the *group* game over all 2^G coalitions.

    The characteristic function is v(S) = f(x_S ; ref_{-S}) where a group is
    switched on or off as a unit.  With G = 8 this is 256 evaluations, so the
    value is exact rather than sampled -- the clinician-facing number carries
    no estimation error at all.
    """
    M = group_masks()
    n = 1 << G
    sel = ((np.arange(n)[:, None] >> np.arange(G)[None, :]) & 1).astype(bool)
    on = sel @ M                                  # (n, D) feature-level mask
    vals = f(np.where(on, x, ref))
    card = sel.sum(axis=1)
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, G + 1)))])
    w = np.exp(logfact[np.arange(G)] + logfact[G - 1 - np.arange(G)] - logfact[G])
    idx = np.arange(n)
    phi = np.zeros(G)
    for g in range(G):
        bit = 1 << g
        without = idx[(idx & bit) == 0]
        phi[g] = float(np.sum(w[card[without]] * (vals[without | bit] - vals[without])))
    return phi


def exact_shapley_groups_batch(f: RiskFunctional, X: np.ndarray,
                               ref: np.ndarray) -> np.ndarray:
    """Vectorised group Shapley for a batch of requests.

    All n * 256 coalition states are scored in a single forward pass, which is
    what makes exact group attribution viable on the decision path.
    """
    X = np.atleast_2d(X)
    n_x, M = X.shape[0], group_masks()
    n = 1 << G
    sel = ((np.arange(n)[:, None] >> np.arange(G)[None, :]) & 1).astype(bool)
    on = sel @ M                                            # (256, D)
    states = np.where(on[None, :, :], X[:, None, :], ref[None, None, :])
    vals = f(states.reshape(-1, D)).reshape(n_x, n)
    card = sel.sum(axis=1)
    logfact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, G + 1)))])
    w = np.exp(logfact[np.arange(G)] + logfact[G - 1 - np.arange(G)] - logfact[G])
    idx = np.arange(n)
    phi = np.zeros((n_x, G))
    for g in range(G):
        bit = 1 << g
        without = idx[(idx & bit) == 0]
        phi[:, g] = (w[card[without]] *
                     (vals[:, without | bit] - vals[:, without])).sum(axis=1)
    return phi


# ==========================================================================
# Natural-language rendering
# ==========================================================================
@dataclass
class Explanation:
    risk: float
    action: str
    phi: np.ndarray
    top: List[Tuple[str, float]]
    user_text: str
    analyst_text: str

    def as_audit_record(self, event_id, user_id) -> Dict:
        return {
            "event_id": event_id,
            "subject": user_id,
            "decision": self.action,
            "risk_score": round(float(self.risk), 4),
            "attributions": {FEATURES[i]: round(float(v), 4)
                             for i, v in enumerate(self.phi)},
            "top_factors": [{"feature": f, "contribution": round(float(v), 4)}
                            for f, v in self.top],
            "explanation": self.user_text,
        }


def render(phi: np.ndarray, risk: float, action: str, k: int = 3) -> Explanation:
    order = np.argsort(-phi)
    top = [(FEATURES[i], float(phi[i])) for i in order[:k] if phi[i] > 0]
    if not top:
        top = [(FEATURES[order[0]], float(phi[order[0]]))]
    reasons = [FEATURE_LABELS[f] for f, _ in top]
    verb = {"ALLOW": "Access granted",
            "STEP_UP_OTP": "One-time code requested",
            "STEP_UP_STRONG": "Strong re-authentication requested",
            "DENY": "Access blocked"}[action]
    if len(reasons) == 1:
        why = reasons[0]
    else:
        why = ", ".join(reasons[:-1]) + " and " + reasons[-1]
    user_text = f"{verb}: {why}."
    share = phi[order[:k]] / (np.abs(phi).sum() + 1e-12)
    analyst_text = (f"risk={risk:.3f} -> {action}; " + "; ".join(
        f"{FEATURES[i]}={phi[i]:+.3f} ({100*s:.0f}% of |phi|)"
        for i, s in zip(order[:k], share)))
    return Explanation(risk=float(risk), action=action, phi=phi, top=top,
                       user_text=user_text, analyst_text=analyst_text)


# ==========================================================================
# Role-scoped disclosure (Section VII)
# ==========================================================================
# Disclosing *why* a session was challenged hands information back to whoever
# is at the keyboard, adversary included.  HEAAF therefore does not emit one
# explanation: it emits a different projection of the same attribution vector
# to each audience, and the projection shown to the person being challenged is
# deliberately lossy.

BANDS = ("a minor factor", "a contributing factor", "the main factor")


def _band(share: float) -> str:
    """Quantise a normalised contribution into three coarse magnitude bands."""
    if share >= 0.50:
        return BANDS[2]
    if share >= 0.25:
        return BANDS[1]
    return BANDS[0]


@dataclass
class Disclosure:
    """One attribution vector, projected for three different audiences."""

    action: str
    subject_text: str          # shown to the clinician being challenged
    analyst_text: str          # SOC / information governance
    audit_record: Dict         # immutable log entry
    phi_groups: np.ndarray
    risk: float

    def for_role(self, role: str) -> str:
        return self.analyst_text if role in ("soc", "ig", "dpo") else self.subject_text


def render_disclosure(phi_groups: np.ndarray, risk: float, action: str,
                      event_id: str = "", subject: str = "",
                      reason_code: str = "") -> Disclosure:
    """Project a group attribution vector into role-scoped disclosures.

    * **Subject view.** One dominant group, named in clinical English, with a
      three-band magnitude and *no numeric score*.  An attacker who is told
      only "the main factor was where you are connecting from" learns the sign
      of one coordinate out of eight; they are not handed a descent direction.
    * **Analyst view.** The complete signed vector, the calibrated score and
      the reconciliation against the benign reference.
    * **Audit record.** Everything, timestamped with the decision, so the
      justification cannot be reconstructed later from a retrained model.
    """
    phi = np.asarray(phi_groups, dtype=float).ravel()
    pos = np.clip(phi, 0.0, None)
    total = float(pos.sum()) + 1e-12
    order = np.argsort(-phi)
    top = int(order[0])

    verb = {"ALLOW": "Access granted",
            "STEP_UP_OTP": "A one-time code was requested",
            "STEP_UP_STRONG": "Strong re-authentication was requested",
            "DENY": "Access was blocked"}[action]

    if phi[top] <= 0:
        subject_text = f"{verb}: no single factor stood out; this was a routine check."
    else:
        subject_text = (f"{verb}: {GROUP_LABELS[GROUP_NAMES[top]]}. "
                        f"This was {_band(float(pos[top]) / total)} in the decision.")

    analyst_text = (f"risk={risk:.3f} -> {action} [{reason_code}]; " + "; ".join(
        f"{GROUP_NAMES[i]}={phi[i]:+.3f}" for i in order))

    audit = {
        "event_id": event_id,
        "subject": subject,
        "decision": action,
        "reason_code": reason_code,
        "risk_score": round(float(risk), 4),
        "group_attributions": {GROUP_NAMES[i]: round(float(phi[i]), 4)
                               for i in range(len(phi))},
        "dominant_group": GROUP_NAMES[top],
        "subject_disclosure": subject_text,
    }
    return Disclosure(action=action, subject_text=subject_text,
                      analyst_text=analyst_text, audit_record=audit,
                      phi_groups=phi, risk=float(risk))


def disclosure_leakage(phi_groups: np.ndarray) -> Dict[str, float]:
    """How much of the attribution vector each audience actually receives.

    Reported as the fraction of the (positive) attribution mass and the number
    of real-valued coordinates released.  The subject view releases one
    coarsened coordinate out of ``G``; the analyst view releases all of them.
    This is the quantity Section VII trades against the transparency duty.
    """
    P = np.atleast_2d(np.asarray(phi_groups, dtype=float))
    pos = np.clip(P, 0.0, None)
    tot = pos.sum(axis=1) + 1e-12
    top = pos.max(axis=1)
    return {
        "subject_coords_released": 1.0,
        "subject_bits_per_decision": float(np.log2(len(BANDS) * P.shape[1])),
        "analyst_coords_released": float(P.shape[1]),
        "subject_mass_fraction": float(np.mean(top / tot)),
    }
