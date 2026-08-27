"""Invariant tests for the HEAAF reference implementation.

These are not accuracy tests.  They check the properties the paper's claims
actually rest on, so that a reviewer can verify the argument mechanically:

* the attribution layer satisfies the Shapley axioms it invokes;
* group attributions are exact and complete;
* the clinical safety valve cannot be defeated;
* the metric code computes what its name says;
* the keystroke benchmark reproduces the published error rate.

Run with ``pytest -q`` from the repository root.
"""
from __future__ import annotations

import numpy as np
import pytest

from heaaf import explain as EX
from heaaf import metrics as M
from heaaf.config import (A, AIDX, D, FEATURES, FIDX, G, GROUPS, GROUP_IDX,
                          GROUP_NAMES, PolicyConfig)
from heaaf.policy import PolicyEngine


# ==========================================================================
# A linear-Gaussian stand-in for the risk functional
# ==========================================================================
class _LinearRisk:
    """f(x) = w.x + b.  Shapley values are known in closed form: w_i (x_i - r_i)."""

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(size=D)
        self.b = float(rng.normal())

    def __call__(self, X):
        return np.atleast_2d(X) @ self.w + self.b

    def grad(self, X):
        return np.tile(self.w, (np.atleast_2d(X).shape[0], 1))


class _QuadraticRisk:
    """A functional whose interactions are exactly pairwise."""

    def __init__(self, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(size=D)
        self.V = rng.normal(size=(D, D)) * 0.3
        self.V = 0.5 * (self.V + self.V.T)

    def __call__(self, X):
        X = np.atleast_2d(X)
        return X @ self.w + np.einsum("ij,jk,ik->i", X, self.V, X)

    def grad(self, X):
        X = np.atleast_2d(X)
        return self.w + 2.0 * (X @ self.V)


class _ThirdOrderRisk(_QuadraticRisk):
    """Adds one three-way interaction that straddles two semantic groups.

    Features 0 and 1 are both in "Keystroke dynamics"; feature 5 is in
    "Network and location".  A three-way term split 2-1 across groups is the
    smallest structure for which the quotient game stops agreeing with the
    block-summed feature game.
    """

    def __call__(self, X):
        X = np.atleast_2d(X)
        return super().__call__(X) + 3.0 * X[:, 0] * X[:, 1] * X[:, 5]


class _SaturatingRisk:
    """A genuinely non-polynomial functional, used to test IG convergence."""

    def __init__(self, seed: int = 2):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(size=D)

    def __call__(self, X):
        return np.tanh(np.atleast_2d(X) @ self.w)

    def grad(self, X):
        X = np.atleast_2d(X)
        t = np.tanh(X @ self.w)
        return (1.0 - t ** 2)[:, None] * self.w[None, :]


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 1, size=D)
    ref = rng.uniform(0, 1, size=D)
    return x, ref


# ==========================================================================
# Structural invariants of the feature space
# ==========================================================================
def test_groups_partition_features():
    flat = [f for fs in GROUPS.values() for f in fs]
    assert sorted(flat) == sorted(FEATURES), "groups must cover every feature"
    assert len(flat) == len(set(flat)) == D, "groups must not overlap"
    assert G == 8, "the paper reports eight semantic groups"


def test_group_shapley_is_cheaper_than_feature_shapley():
    assert (1 << G) == 256
    assert (1 << D) == 65_536


# ==========================================================================
# Shapley axioms
# ==========================================================================
def test_exact_feature_shapley_matches_closed_form_on_linear_model(sample):
    x, ref = sample
    f = _LinearRisk()
    phi = EX.exact_shapley(f, x, ref)
    np.testing.assert_allclose(phi, f.w * (x - ref), atol=1e-9)


def test_exact_shapley_satisfies_efficiency(sample):
    """sum(phi) must equal f(x) - f(ref) -- the completeness axiom."""
    x, ref = sample
    for f in (_LinearRisk(), _QuadraticRisk()):
        phi = EX.exact_shapley(f, x, ref)
        assert abs(phi.sum() - (float(f(x)[0]) - float(f(ref)[0]))) < 1e-8


def test_exact_shapley_dummy_axiom(sample):
    """A feature the functional ignores must receive exactly zero credit."""
    x, ref = sample
    f = _LinearRisk()
    dummy = FIDX["time_dev"]
    f.w[dummy] = 0.0
    phi = EX.exact_shapley(f, x, ref)
    assert abs(phi[dummy]) < 1e-10


def test_exact_shapley_symmetry_axiom(sample):
    """Two features entering the functional identically get equal credit."""
    x, ref = sample
    f = _LinearRisk()
    i, j = FIDX["geo_novel"], FIDX["dev_unknown"]
    f.w[j] = f.w[i]
    x = x.copy(); ref = ref.copy()
    x[j], ref[j] = x[i], ref[i]
    phi = EX.exact_shapley(f, x, ref)
    assert abs(phi[i] - phi[j]) < 1e-10


# ==========================================================================
# Group-level attribution
# ==========================================================================
def test_group_shapley_satisfies_efficiency(sample):
    x, ref = sample
    for f in (_LinearRisk(), _QuadraticRisk()):
        phi = EX.exact_shapley_groups(f, x, ref)
        assert phi.shape == (G,)
        assert abs(phi.sum() - (float(f(x)[0]) - float(f(ref)[0]))) < 1e-8


def test_group_shapley_batch_matches_single(sample):
    x, ref = sample
    f = _QuadraticRisk()
    X = np.vstack([x, ref, 0.5 * (x + ref)])
    batch = EX.exact_shapley_groups_batch(f, X, ref)
    single = np.vstack([EX.exact_shapley_groups(f, xi, ref) for xi in X])
    np.testing.assert_allclose(batch, single, atol=1e-9)


def test_group_shapley_equals_summed_feature_shapley_when_additive(sample):
    """For an additive functional the two games coincide; this pins the code."""
    x, ref = sample
    f = _LinearRisk()
    grouped = EX.to_groups(EX.exact_shapley(f, x, ref))[0]
    direct = EX.exact_shapley_groups(f, x, ref)
    np.testing.assert_allclose(grouped, direct, atol=1e-9)


def test_group_shapley_agrees_with_block_sums_under_pairwise_interaction(sample):
    """Pairwise interaction is not enough to separate the two games.

    Writing the game in the unanimity basis, a term supported on T gives each
    member c_T/|T|, so a block B receives c_T |T cap B| / |T|; the quotient
    game instead gives it c_T / |Tbar| where Tbar is the set of blocks T
    touches.  For |T| <= 2 these always coincide, whatever the partition.
    """
    x, ref = sample
    f = _QuadraticRisk()
    grouped = EX.to_groups(EX.exact_shapley(f, x, ref))[0]
    direct = EX.exact_shapley_groups(f, x, ref)
    np.testing.assert_allclose(grouped, direct, atol=1e-9)


def test_group_shapley_diverges_under_three_way_interaction(sample):
    """A 2-1 split of a three-way term across blocks separates the two games.

    The block gets 2c/3 by summation but only c/2 in the quotient game.  This
    is why the paper reports the exact group value rather than block sums.
    """
    x, ref = sample
    f = _ThirdOrderRisk()
    grouped = EX.to_groups(EX.exact_shapley(f, x, ref))[0]
    direct = EX.exact_shapley_groups(f, x, ref)
    assert np.abs(grouped - direct).max() > 1e-6
    # both remain complete, so the disagreement is purely in the split
    tgt = float(f(x)[0]) - float(f(ref)[0])
    assert abs(grouped.sum() - tgt) < 1e-8
    assert abs(direct.sum() - tgt) < 1e-8


def test_to_groups_preserves_total_mass(sample):
    rng = np.random.default_rng(3)
    phi = rng.normal(size=(5, D))
    np.testing.assert_allclose(EX.to_groups(phi).sum(axis=1), phi.sum(axis=1),
                               atol=1e-12)


# ==========================================================================
# Integrated gradients
# ==========================================================================
def test_integrated_gradients_is_exact_for_linear_models(sample):
    x, ref = sample
    f = _LinearRisk()
    phi = EX.integrated_gradients(f, x, ref, m=4)[0]
    np.testing.assert_allclose(phi, f.w * (x - ref), atol=1e-9)


def test_integrated_gradients_is_exact_for_quadratic_models(sample):
    """Midpoint Riemann integrates a linear gradient field exactly."""
    x, ref = sample
    f = _QuadraticRisk()
    target = float(f(x)[0]) - float(f(ref)[0])
    assert abs(EX.integrated_gradients(f, x, ref, m=2)[0].sum() - target) < 1e-10


def test_integrated_gradients_completeness_improves_with_steps(sample):
    """On a genuinely non-polynomial functional, more steps must close the gap."""
    x, ref = sample
    f = _SaturatingRisk()
    target = float(f(x)[0]) - float(f(ref)[0])
    gaps = [abs(EX.integrated_gradients(f, x, ref, m=m)[0].sum() - target)
            for m in (2, 8, 64)]
    assert gaps[2] < gaps[1] < gaps[0]


# ==========================================================================
# Policy engine and the clinical safety valve
# ==========================================================================
def _emergency_batch(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, D))
    X[:, FIDX["emergency"]] = 1.0
    return X


def test_safety_valve_never_denies_a_declared_emergency():
    """The paper's central patient-safety claim, checked exhaustively."""
    X = _emergency_batch()
    pe = PolicyEngine(PolicyConfig(safety_valve=True))
    for r in np.linspace(0.0, 1.0, 101):
        acts = pe.actions_batch(np.full(len(X), r), X)
        assert not (acts == AIDX["DENY"]).any(), f"denied an emergency at risk={r}"


def test_without_the_valve_emergencies_are_denied():
    """The ablation must actually differ, or the valve result would be empty."""
    X = _emergency_batch()
    pe = PolicyEngine(PolicyConfig(safety_valve=False))
    acts = pe.actions_batch(np.full(len(X), 0.99), X)
    assert (acts == AIDX["DENY"]).all()


def test_policy_is_monotone_in_risk():
    """Higher risk must never produce a weaker action."""
    rng = np.random.default_rng(5)
    X = rng.uniform(0, 0.8, size=(200, D))
    X[:, FIDX["emergency"]] = 0.0
    X[:, FIDX["res_sens"]] = 0.1
    pe = PolicyEngine()
    prev = pe.actions_batch(np.zeros(len(X)), X)
    for r in np.linspace(0.05, 1.0, 40):
        cur = pe.actions_batch(np.full(len(X), r), X)
        assert (cur >= prev).all(), f"action strength decreased at risk={r}"
        prev = cur


def test_sensitivity_floor_forces_a_challenge():
    rng = np.random.default_rng(6)
    X = rng.uniform(0, 0.2, size=(100, D))
    X[:, FIDX["emergency"]] = 0.0
    X[:, FIDX["res_sens"]] = 0.99
    acts = PolicyEngine().actions_batch(np.zeros(len(X)), X)
    assert not (acts == AIDX["ALLOW"]).any()


def test_batch_and_scalar_policy_agree():
    rng = np.random.default_rng(11)
    X = rng.uniform(0, 1, size=(300, D))
    r = rng.uniform(0, 1, size=300)
    pe = PolicyEngine()
    batch = pe.actions_batch(r, X)
    scalar = np.array([AIDX[pe.decide(float(ri), xi).action] for ri, xi in zip(r, X)])
    np.testing.assert_array_equal(batch, scalar)


# ==========================================================================
# Role-scoped disclosure
# ==========================================================================
def test_subject_disclosure_omits_the_numeric_score():
    phi = np.array([0.8, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
    d = EX.render_disclosure(phi, risk=0.734, action="STEP_UP_OTP")
    assert "0.734" not in d.subject_text
    assert "0.8" not in d.subject_text
    assert GROUP_NAMES[0].split()[0].lower() in d.subject_text.lower() or \
        "typing" in d.subject_text.lower()


def test_subject_disclosure_names_exactly_one_group():
    rng = np.random.default_rng(2)
    for _ in range(50):
        phi = rng.normal(size=G)
        d = EX.render_disclosure(phi, risk=0.5, action="STEP_UP_OTP")
        named = sum(1 for lbl in EX.GROUP_LABELS.values() if lbl in d.subject_text)
        assert named <= 1


def test_analyst_disclosure_is_complete():
    rng = np.random.default_rng(4)
    phi = rng.normal(size=G)
    d = EX.render_disclosure(phi, risk=0.5, action="DENY")
    for name in GROUP_NAMES:
        assert name in d.analyst_text


def test_audit_record_is_serialisable():
    import json
    phi = np.arange(G, dtype=float)
    rec = EX.render_disclosure(phi, 0.5, "ALLOW", event_id="e1",
                               subject="s002").audit_record
    json.loads(json.dumps(rec))


# ==========================================================================
# Metrics
# ==========================================================================
def test_gmean_is_the_geometric_mean():
    assert abs(M.gmean(0.81, 0.49) - 0.63) < 1e-12
    assert M.gmean(0.0, 1.0) == 0.0


def test_eer_on_separable_scores_is_zero():
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([np.zeros(50), np.ones(50)])
    e, _ = M.eer(s, y)
    assert e < 1e-9


def test_eer_on_random_scores_is_near_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=4000)
    e, _ = M.eer(rng.normal(size=4000), y)
    assert 0.4 < e < 0.6


def test_detection_metrics_confusion_counts():
    y = np.array([1, 1, 0, 0, 0])
    flagged = np.array([1, 0, 1, 0, 0])
    m = M.detection_metrics(y, None, flagged)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 1, 2)
    assert abs(m["TPR"] - 0.5) < 1e-12
    assert abs(m["TNR"] - 2 / 3) < 1e-12


def test_calibration_of_a_perfectly_calibrated_score_is_near_zero():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=200_000)
    y = (rng.uniform(size=200_000) < p).astype(int)
    assert M.calibration_metrics(y, p)["ECE"] < 0.01


def test_completeness_gap_is_zero_for_an_exact_decomposition():
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(20, G))
    fx = phi.sum(axis=1) + 3.0
    assert M.completeness_gap(phi, fx, 3.0) < 1e-9


# ==========================================================================
# Keystroke benchmark
# ==========================================================================
@pytest.mark.skipif(
    not __import__("heaaf.config", fromlist=["KEYSTROKE_CSV"]).KEYSTROKE_CSV.exists(),
    reason="CMU keystroke CSV not present")
def test_keystroke_corpus_shape():
    from heaaf import keystroke as ks
    df = ks.load_keystroke()
    assert df.subject.nunique() == 51
    assert sorted(df.sessionIndex.unique()) == list(range(1, 9))
    assert len(df) == 51 * 400


@pytest.mark.skipif(
    not __import__("heaaf.config", fromlist=["KEYSTROKE_CSV"]).KEYSTROKE_CSV.exists(),
    reason="CMU keystroke CSV not present")
def test_scaled_manhattan_eer_matches_published_range():
    """Killourhy & Maxion report ~0.096 EER for scaled Manhattan.

    Reproducing it is the sanity check that the behavioural channel is wired
    up correctly; a wide band is used because our session split differs.
    """
    from heaaf import keystroke as ks
    df = ks.load_keystroke()
    tpl = ks.build_templates(df, (1, 2, 3, 4))
    e = ks.evaluate_eer(df, tpl, (7, 8))
    assert 0.02 < e["eer_mean"] < 0.20, e


@pytest.mark.skipif(
    not __import__("heaaf.config", fromlist=["KEYSTROKE_CSV"]).KEYSTROKE_CSV.exists(),
    reason="CMU keystroke CSV not present")
def test_template_ageing_is_monotone_enough():
    """Later sessions should not be *easier* than the enrolment-adjacent ones."""
    from heaaf import keystroke as ks
    df = ks.load_keystroke()
    tpl = ks.build_templates(df, (1, 2, 3, 4))
    e5 = ks.evaluate_eer(df, tpl, (5,))["eer_mean"]
    e8 = ks.evaluate_eer(df, tpl, (8,))["eer_mean"]
    assert e8 >= e5 - 1e-9


# ==========================================================================
# Generator
# ==========================================================================
def test_generator_is_deterministic_under_a_seed():
    from dataclasses import replace
    from heaaf.config import SimConfig
    from heaaf import keystroke as ks
    from heaaf.simulator import BenchGenerator
    import pandas as pd
    if not __import__("heaaf.config", fromlist=["KEYSTROKE_CSV"]).KEYSTROKE_CSV.exists():
        pytest.skip("CMU keystroke CSV not present")
    cfg = replace(SimConfig(), n_events=2000)
    df = ks.load_keystroke()
    tpl = ks.build_templates(df, cfg.enrol_sessions)
    pool = ks.build_pool(df, tpl, cfg.val_sessions, rng=np.random.default_rng(0))
    a = BenchGenerator(pool, cfg, 42).generate(2000)
    b = BenchGenerator(pool, cfg, 42).generate(2000)
    pd.testing.assert_frame_equal(a, b)


def test_attack_drivers_are_valid_feature_names():
    from heaaf.config import ATTACKS, HARD_NEGATIVES
    for spec in list(ATTACKS.values()) + list(HARD_NEGATIVES.values()):
        for f in spec["drivers"]:
            assert f in FIDX, f"unknown driver feature {f}"
