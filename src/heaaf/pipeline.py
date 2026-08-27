"""End-to-end experimental pipeline for the HEAAF evaluation."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import baselines as B
from . import explain as EX
from . import keystroke as ks
from . import metrics as M
from .agent import AccessEnv, DDQNAgent, RiskHead
from .config import (A, ACTIONS, AIDX, ATTACKS, D, FEATURES, FIDX, LOGDIR,
                     RESULTS, TABDIR, AgentConfig, Config, PolicyConfig,
                     RewardConfig, SimConfig)
from .policy import PolicyEngine
from .simulator import feature_matrix, ground_truth_drivers, make_splits


# ==========================================================================
# Training
# ==========================================================================
def train_heaaf(splits: Dict, cfg: Config, seed: int, verbose: bool = False):
    ac = AgentConfig(**{**asdict(cfg.agent), "seed": seed})
    env = AccessEnv(splits["train"], cfg.reward, seed=seed)
    agent = DDQNAgent(ac)
    log = agent.train(env, gamma=cfg.reward.gamma, verbose=verbose)
    Xv, yv = feature_matrix(splits["val"]), splits["val"].y.to_numpy()
    head = RiskHead().fit(agent.q_values(Xv), yv)
    return agent, head, log


def train_rlauth(splits: Dict, cfg: Config, seed: int):
    """RLAuth-style baseline: binary action set, bandit episodes, no clinical cost."""
    ac = AgentConfig(**{**asdict(cfg.agent), "seed": seed + 500})
    rc = RewardConfig(gamma=0.0)
    env = AccessEnv(splits["train"], rc, seed=seed + 500,
                    action_names=["ALLOW", "DENY"], clinical_costs=False,
                    bandit=True)
    agent = DDQNAgent(ac, n_actions=2)
    agent.train(env, gamma=0.0)
    Xv, yv = feature_matrix(splits["val"]), splits["val"].y.to_numpy()
    Q = agent.q_values(Xv)
    adv = Q[:, 1] - Q[:, 0]
    mu, sd = float(adv.mean()), float(adv.std() + 1e-9)
    z = (adv - mu) / sd
    w, b = 1.0, float(np.log((yv.mean() + 1e-6) / (1 - yv.mean() + 1e-6)))
    for _ in range(3000):
        p = 1 / (1 + np.exp(-np.clip(w * z + b, -30, 30)))
        w -= 0.5 * float(np.mean((p - yv) * z))
        b -= 0.5 * float(np.mean(p - yv))
    return B.RLAuthStyle(agent, {"mu": mu, "sd": sd, "w": w, "b": b})


def fit_static_baselines(splits: Dict, seed: int):
    Xtr, ytr = feature_matrix(splits["train"]), splits["train"].y.to_numpy()
    Xv, yv = feature_matrix(splits["val"]), splits["val"].y.to_numpy()
    rba = B.FreemanRBA().fit(Xtr, ytr).calibrate(Xv, yv)
    xai = B.XAIStatic(seed=seed).fit(Xtr, ytr)
    return rba, xai


# ==========================================================================
# Evaluation of one policy on the test split
# ==========================================================================
def evaluate(name: str, df: pd.DataFrame, risk: np.ndarray,
             actions: np.ndarray) -> Dict:
    y = df["y"].to_numpy()
    flagged = (actions != AIDX["ALLOW"]).astype(int)
    out = {"policy": name}
    out.update(M.detection_metrics(y, risk, flagged))
    out.update(M.friction_metrics(df, actions))
    out.update(M.containment_metrics(df.reset_index(drop=True), actions))
    out["per_attack"] = M.per_scenario_detection(df, flagged)
    return out


def closed_loop_return(policy_fn, df: pd.DataFrame, reward: RewardConfig,
                       seed: int = 0, n_sessions: int = 4000) -> Dict[str, float]:
    """Average discounted return per session when the policy actually acts."""
    env = AccessEnv(df, reward, seed=seed, shuffle=True)
    rets, undisc, lens = [], [], []
    for _ in range(n_sessions):
        s = env.reset()
        G, R, t = 0.0, 0.0, 0
        while True:
            a = policy_fn(s.reshape(1, -1))[0]
            s, r, done, _ = env.step(int(a))
            G += (reward.gamma ** t) * r
            R += r
            t += 1
            if done:
                break
        rets.append(G)
        undisc.append(R)
        lens.append(t)
    return {"return_discounted": float(np.mean(rets)),
            "return_undiscounted": float(np.mean(undisc)),
            "mean_episode_len": float(np.mean(lens))}


# ==========================================================================
# Experiment 1 -- detection, friction and containment
# ==========================================================================
def experiment_main(splits: Dict, cfg: Config, seeds: Sequence[int],
                    verbose: bool = True) -> Dict:
    df_te = splits["test"].reset_index(drop=True)
    Xte = feature_matrix(df_te)
    rows: List[Dict] = []
    artefacts: Dict[int, Dict] = {}

    for seed in seeds:
        t0 = time.time()
        agent, head, log = train_heaaf(splits, cfg, seed)
        pe = PolicyEngine(cfg.policy)
        risk = head.score(agent, Xte)
        acts = pe.actions_batch(risk, Xte)
        rows.append({**evaluate("HEAAF", df_te, risk, acts), "seed": seed,
                     **M.calibration_metrics(df_te.y.to_numpy(), risk)})

        rl = train_rlauth(splits, cfg, seed)
        r5 = rl.risk(Xte)
        rows.append({**evaluate(rl.name, df_te, r5, rl.act(Xte)), "seed": seed,
                     **M.calibration_metrics(df_te.y.to_numpy(), r5)})

        rba, xai = fit_static_baselines(splits, seed)
        r3 = rba.risk(Xte)
        rows.append({**evaluate(rba.name, df_te, r3, rba.act(Xte)), "seed": seed,
                     **M.calibration_metrics(df_te.y.to_numpy(), r3)})
        r4 = xai.risk(Xte)
        rows.append({**evaluate(xai.name, df_te, r4, xai.act(Xte)), "seed": seed,
                     **M.calibration_metrics(df_te.y.to_numpy(), r4)})

        b1, b2 = B.AlwaysMFA(), B.StaticPassword()
        rows.append({**evaluate(b1.name, df_te, b1.risk(Xte), b1.act(Xte)), "seed": seed})
        rows.append({**evaluate(b2.name, df_te, b2.risk(Xte), b2.act(Xte)), "seed": seed})

        artefacts[seed] = {"agent": agent, "head": head, "rba": rba,
                           "xai": xai, "rl": rl, "log": log}
        if verbose:
            print(f"  [seed {seed}] trained in {time.time()-t0:.1f}s")

    return {"rows": rows, "artefacts": artefacts, "df_test": df_te, "X_test": Xte}


# ==========================================================================
# Experiment 2 -- security / friction Pareto frontier
# ==========================================================================
def experiment_pareto(main: Dict, cfg: Config,
                      taus: Sequence[float] = np.linspace(0.02, 0.95, 40)) -> pd.DataFrame:
    df_te, Xte = main["df_test"], main["X_test"]
    y = df_te.y.to_numpy()
    rows = []
    for seed, art in main["artefacts"].items():
        curves = {
            "HEAAF": art["head"].score(art["agent"], Xte),
            art["rl"].name: art["rl"].risk(Xte),
            art["rba"].name: art["rba"].risk(Xte),
            art["xai"].name: art["xai"].risk(Xte),
        }
        for name, risk in curves.items():
            for t in taus:
                pc = PolicyConfig(tau_low=float(t),
                                  tau_high=float(min(0.999, t + 0.35)),
                                  sens_floor=cfg.policy.sens_floor,
                                  safety_valve=cfg.policy.safety_valve,
                                  emergency_relief=cfg.policy.emergency_relief)
                acts = PolicyEngine(pc).actions_batch(risk, Xte)
                flagged = (acts != AIDX["ALLOW"]).astype(int)
                fr = M.friction_metrics(df_te, acts)
                rows.append({
                    "policy": name, "seed": seed, "tau": float(t),
                    "TPR": float(flagged[y == 1].mean()),
                    "interrupt_rate_benign": fr["interrupt_rate_benign"],
                    "hours_year": fr["dept_clinician_hours_per_year"],
                    "deny_rate_emergency": fr["deny_rate_emergency"],
                })
    return pd.DataFrame(rows)


# ==========================================================================
# Experiment 3 -- explanation fidelity, grounding, cost
# ==========================================================================
def experiment_explanations(main: Dict, n_instances: int = 200,
                            seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    df_te, Xte = main["df_test"], main["X_test"]
    art = main["artefacts"][min(main["artefacts"])]
    f = EX.RiskFunctional(art["agent"], art["head"])
    ref = EX.clinical_reference(Xte[df_te.y.to_numpy() == 0])

    mal = np.where(df_te.y.to_numpy() == 1)[0]
    ben = np.where(df_te.y.to_numpy() == 0)[0]
    n_mal = min(len(mal), int(n_instances * 0.6))
    idx = np.concatenate([rng.choice(mal, n_mal, replace=False),
                          rng.choice(ben, n_instances - n_mal, replace=False)])
    X = Xte[idx]
    drivers = [ground_truth_drivers(df_te)[i] for i in idx]

    methods: Dict[str, np.ndarray] = {}
    timings: Dict[str, float] = {}

    def timed(name, fn, reps=3):
        fn()                       # warm-up
        t0 = time.perf_counter()
        for _ in range(reps):
            out = fn()
        timings[name] = (time.perf_counter() - t0) / (reps * len(X)) * 1e3
        methods[name] = out

    timed("GradxInput", lambda: EX.grad_x_input(f, X, ref))
    for m in (8, 16, 32):
        timed(f"IG-{m}", lambda m=m: EX.integrated_gradients(f, X, ref, m=m))
    for ns in (128, 512):
        timed(f"KernelSHAP-{ns}",
              lambda ns=ns: np.vstack([EX.kernel_shap(f, x, ref, ns, rng) for x in X]),
              reps=1)
    t0 = time.perf_counter()
    exact = np.vstack([EX.exact_shapley(f, x, ref) for x in X])
    timings["ExactShapley"] = (time.perf_counter() - t0) / len(X) * 1e3
    methods["ExactShapley"] = exact

    fx = f(X)
    fref = float(f(ref)[0])
    rows = []
    for name, phi in methods.items():
        r = {"method": name, "ms_per_explanation": timings[name]}
        r.update(M.explanation_fidelity(phi, exact))
        r["completeness_gap"] = M.completeness_gap(phi, fx, fref)
        r["sparsity_top3"] = M.sparsity(phi)
        r.update(M.driver_recall(phi, drivers, k=3))
        r.update(M.driver_recall(phi, drivers, k=1))
        r["stability"] = M.stability(
            f, (lambda XX, name=name: _phi_fn(name, f, XX, ref, rng)), X[:60])
        rows.append(r)

    # per-attack grounding for the deployed Tier-1 explainer
    per_atk = {}
    atk_labels = df_te["attack"].to_numpy()[idx]
    for a in np.unique(atk_labels):
        if a == "":
            continue
        m = atk_labels == a
        per_atk[str(a)] = M.driver_recall(methods["IG-16"][m],
                                          [d for d, k in zip(drivers, m) if k], k=3)
    return {"table": pd.DataFrame(rows), "per_attack": per_atk,
            "phi": methods, "idx": idx, "ref": ref, "X": X,
            "drivers": drivers, "f": f}


def _phi_fn(name, f, XX, ref, rng):
    if name == "GradxInput":
        return EX.grad_x_input(f, XX, ref)
    if name.startswith("IG-"):
        return EX.integrated_gradients(f, XX, ref, m=int(name.split("-")[1]))
    if name.startswith("KernelSHAP"):
        ns = int(name.split("-")[1])
        return np.vstack([EX.kernel_shap(f, x, ref, ns, rng) for x in XX])
    return np.vstack([EX.exact_shapley(f, x, ref) for x in XX])


# ==========================================================================
# Experiment 3b -- group-level explanation (the clinician-facing disclosure)
# ==========================================================================
def experiment_group_explanations(main: Dict, n_instances: int = 400,
                                  seed: int = 0) -> Dict:
    """Fidelity and cost of the disclosure the clinician actually receives.

    Two distinct cooperative games are involved and the paper is careful to
    keep them apart:

    * the **feature game**, with 16 players, whose exact Shapley values cost
      2^16 = 65,536 coalition evaluations; and
    * the **group game**, with 8 players, whose exact Shapley values cost only
      2^8 = 256 evaluations.

    Summing feature-level attributions inside a block is *not* in general the
    Shapley value of that block in the quotient game -- the two coincide only
    under conditions this risk functional does not satisfy.  We therefore
    measure the divergence rather than assume it away, and we report the exact
    group value as the quantity actually disclosed.
    """
    rng = np.random.default_rng(seed)
    df_te, Xte = main["df_test"], main["X_test"]
    art = main["artefacts"][min(main["artefacts"])]
    f = EX.RiskFunctional(art["agent"], art["head"])
    ref = EX.clinical_reference(Xte[df_te.y.to_numpy() == 0])

    y = df_te.y.to_numpy()
    mal, ben = np.where(y == 1)[0], np.where(y == 0)[0]
    n_mal = min(len(mal), int(n_instances * 0.6))
    idx = np.concatenate([rng.choice(mal, n_mal, replace=False),
                          rng.choice(ben, n_instances - n_mal, replace=False)])
    X = Xte[idx]

    # ---- exact group Shapley: the disclosed quantity -------------------
    t0 = time.perf_counter()
    phi_g_exact = EX.exact_shapley_groups_batch(f, X, ref)
    ms_group_exact = (time.perf_counter() - t0) / len(X) * 1e3

    # ---- exact feature Shapley, summed into blocks ---------------------
    t0 = time.perf_counter()
    phi_f_exact = np.vstack([EX.exact_shapley(f, x, ref) for x in X])
    ms_feat_exact = (time.perf_counter() - t0) / len(X) * 1e3
    phi_f_grouped = EX.to_groups(phi_f_exact)

    # ---- cheap estimators, aggregated to groups ------------------------
    rows = []
    fx, fref = f(X), float(f(ref)[0])

    def add(name, phi_groups, ms):
        r = {"method": name, "ms_per_explanation": ms}
        r.update(M.explanation_fidelity(phi_groups, phi_g_exact, ks=(1, 2, 3)))
        r["completeness_gap"] = M.completeness_gap(phi_groups, fx, fref)
        rows.append(r)

    add("Exact group Shapley (disclosed)", phi_g_exact, ms_group_exact)
    add("Exact feature Shapley, summed", phi_f_grouped, ms_feat_exact)
    for m in (8, 16, 32):
        t0 = time.perf_counter()
        ph = EX.integrated_gradients(f, X, ref, m=m)
        ms = (time.perf_counter() - t0) / len(X) * 1e3
        add(f"IG-{m}, summed", EX.to_groups(ph), ms)
    t0 = time.perf_counter()
    ph = EX.grad_x_input(f, X, ref)
    add("GradxInput, summed", EX.to_groups(ph),
        (time.perf_counter() - t0) / len(X) * 1e3)

    # ---- does the disclosure point at the true cause? ------------------
    from .config import GIDX, GROUP_OF, ATTACKS
    drv_groups = []
    for a in df_te["attack"].to_numpy()[idx]:
        drv_groups.append(sorted({GIDX[GROUP_OF[fn]] for fn in ATTACKS[a]["drivers"]})
                          if a else [])
    ground = {"dominant_group_hit": float(np.mean(
        [phi_g_exact[i].argmax() in d for i, d in enumerate(drv_groups) if d])),
        "top2_group_hit": float(np.mean(
            [len(set(np.argsort(-phi_g_exact[i])[:2]) & set(d)) > 0
             for i, d in enumerate(drv_groups) if d]))}

    # ---- how much does each audience actually learn? -------------------
    leak = EX.disclosure_leakage(phi_g_exact)
    return {"table": pd.DataFrame(rows), "grounding": ground, "leakage": leak,
            "phi_group_exact": phi_g_exact, "phi_feat_exact": phi_f_exact,
            "idx": idx, "ref": ref, "X": X}


# ==========================================================================
# Experiment 4 -- decision latency
# ==========================================================================
def experiment_latency(main: Dict, n: int = 4000, reps: int = 25) -> pd.DataFrame:
    art = main["artefacts"][min(main["artefacts"])]
    Xte = main["X_test"][:n]
    f = EX.RiskFunctional(art["agent"], art["head"])
    pe = PolicyEngine()
    ref = EX.clinical_reference(Xte)
    rows = []

    def bench(name, fn, batch):
        fn()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) / batch * 1e3)
        ts = np.array(ts)
        rows.append({"stage": name, "p50_ms": float(np.percentile(ts, 50)),
                     "p95_ms": float(np.percentile(ts, 95)),
                     "p99_ms": float(np.percentile(ts, 99)),
                     "mean_ms": float(ts.mean())})

    bench("risk scoring (batch)", lambda: art["head"].score(art["agent"], Xte), len(Xte))
    bench("exact group Shapley (8 groups)",
          lambda: EX.exact_shapley_groups_batch(f, Xte[:512], ref), 512)
    bench("policy engine", lambda: pe.actions_batch(np.full(len(Xte), .3), Xte), len(Xte))
    bench("Tier-1 explanation (Grad x Input)",
          lambda: EX.grad_x_input(f, Xte, ref), len(Xte))
    bench("Tier-1 explanation (IG-16)",
          lambda: EX.integrated_gradients(f, Xte, ref, m=16), len(Xte))
    single = Xte[:1]
    bench("end-to-end, single request (IG-16)",
          lambda: (art["head"].score(art["agent"], single),
                   EX.integrated_gradients(f, single, ref, m=16),
                   pe.actions_batch(np.array([0.3]), single)), 1)
    bench("end-to-end, single request (exact group Shapley)",
          lambda: (art["head"].score(art["agent"], single),
                   EX.exact_shapley_groups_batch(f, single, ref),
                   pe.actions_batch(np.array([0.3]), single)), 1)
    return pd.DataFrame(rows)


# ==========================================================================
# Experiment 5 -- ablations
# ==========================================================================
ABLATIONS = {
    "HEAAF (full)": {},
    "w/o behavioural channel": {"mask": ["kd_dist", "kd_speed_dev", "kd_flight_var"]},
    "w/o contextual channel": {"mask": ["dev_unknown", "dev_risk", "net_zone_risk",
                                         "geo_novel", "impossible_travel", "time_dev",
                                         "no_care_rel", "peer_dev"]},
    "w/o graded step-up": {"binary": True},
    "w/o clinical cost terms": {"clinical": False},
    "w/o sequential credit (gamma=0)": {"gamma": 0.0},
    "w/o risk calibration": {"nocal": True},
    "w/o clinical safety valve": {"novalve": True},
}


def experiment_ablation(splits: Dict, cfg: Config, seeds: Sequence[int],
                        budget: float = 0.01) -> pd.DataFrame:
    """Ablate one component at a time, always at a matched friction budget.

    An earlier version compared ablations at a *fixed* pair of thresholds.
    That silently conflated two different effects: a component that genuinely
    improves ranking, and a component that merely shifts the score
    distribution relative to a hard-coded threshold.  The "w/o calibration"
    arm made this obvious -- removing Platt scaling moved the scores, not the
    ranking, and the arm then interrupted 98% of benign traffic, which
    measured the threshold rather than the ablation.

    Every arm is therefore given the operating point that spends the same
    clinical friction budget on the validation split, so the comparison is
    "same cost to clinicians, how much security do you get".
    """
    df_te = splits["test"].reset_index(drop=True)
    Xte = feature_matrix(df_te)
    df_va = splits["val"].reset_index(drop=True)
    ref_med = np.median(feature_matrix(splits["train"]), axis=0)

    def tau_at_budget(risk_val, y_val):
        ben = risk_val[np.asarray(y_val) == 0]
        return float(np.quantile(ben, 1.0 - budget))

    rows = []
    for name, spec in ABLATIONS.items():
        for seed in seeds:
            c = Config(sim=cfg.sim,
                       agent=AgentConfig(**{**asdict(cfg.agent), "seed": seed}),
                       reward=RewardConfig(**asdict(cfg.reward)),
                       policy=PolicyConfig(**asdict(cfg.policy)))
            if spec.get("novalve"):
                c.policy.safety_valve = False
                c.policy.emergency_relief = 0.0
            tr, va, te = splits["train"].copy(), splits["val"].copy(), df_te.copy()
            Xte_v = Xte.copy()
            if "mask" in spec:
                for fname in spec["mask"]:
                    for d in (tr, va, te):
                        d[fname] = ref_med[FIDX[fname]]
                    Xte_v[:, FIDX[fname]] = ref_med[FIDX[fname]]
            if spec.get("gamma") is not None:
                c.reward.gamma = spec["gamma"]
            clinical = spec.get("clinical", True)
            if not clinical:
                c.reward.emergency_penalty = {k: 0.0 for k in c.reward.emergency_penalty}
            names = ["ALLOW", "DENY"] if spec.get("binary") else None
            n_act = 2 if spec.get("binary") else A
            env = AccessEnv(tr, c.reward, seed=seed, action_names=names,
                            clinical_costs=clinical)
            agent = DDQNAgent(c.agent, n_actions=n_act)
            agent.train(env, gamma=c.reward.gamma)
            Xv, yv = feature_matrix(va), va.y.to_numpy()
            Xv_masked = Xv.copy()
            if "mask" in spec:
                for fname in spec["mask"]:
                    Xv_masked[:, FIDX[fname]] = ref_med[FIDX[fname]]
            if spec.get("binary"):
                Q = agent.q_values(Xv_masked)
                adv = Q[:, 1] - Q[:, 0]
                mu, sd = adv.mean(), adv.std() + 1e-9
                z = (adv - mu) / sd
                w, b = 1.0, float(np.log((yv.mean() + 1e-6) / (1 - yv.mean() + 1e-6)))
                for _ in range(3000):
                    p = 1 / (1 + np.exp(-np.clip(w * z + b, -30, 30)))
                    w -= 0.5 * float(np.mean((p - yv) * z))
                    b -= 0.5 * float(np.mean(p - yv))

                def _risk(XX):
                    Qx = agent.q_values(XX)
                    return 1 / (1 + np.exp(-np.clip(
                        w * ((Qx[:, 1] - Qx[:, 0]) - mu) / sd + b, -30, 30)))
            else:
                head = RiskHead().fit(agent.q_values(Xv_masked), yv)
                if spec.get("nocal"):
                    def _risk(XX):
                        return 1 / (1 + np.exp(
                            -np.clip(head.advantage(agent.q_values(XX)), -30, 30)))
                else:
                    def _risk(XX):
                        return head.score(agent, XX)
            risk = _risk(Xte_v)
            # Matched friction budget: read each arm's threshold off validation.
            t = tau_at_budget(_risk(Xv_masked), yv)
            pc = PolicyConfig(tau_low=t, tau_high=float(min(0.999, t + 0.35)),
                              sens_floor=c.policy.sens_floor,
                              safety_valve=c.policy.safety_valve,
                              emergency_relief=c.policy.emergency_relief)
            acts = PolicyEngine(pc).actions_batch(risk, Xte_v)
            r = evaluate(name, te, risk, acts)
            r.update(M.calibration_metrics(te.y.to_numpy(), risk))
            r["seed"] = seed
            r["tau"] = float(t)
            rows.append(r)
    return pd.DataFrame(rows)


# ==========================================================================
# Experiment 6 -- behavioural drift (template ageing)
# ==========================================================================
def experiment_drift(cfg: Config, seeds: Sequence[int]) -> pd.DataFrame:
    """Train once on sessions 5-6 and test on progressively aged material."""
    df = ks.load_keystroke()
    templates = ks.build_templates(df, cfg.sim.enrol_sessions)
    rows = []
    eer_rows = []
    for block in [(5,), (6,), (7,), (8,)]:
        e = ks.evaluate_eer(df, templates, block)
        eer_rows.append({"session_block": f"S{block[0]}", **e})

    from .simulator import BenchGenerator
    for seed in seeds:
        rng = np.random.default_rng(cfg.sim.seed + seed)
        pool_tr = ks.build_pool(df, templates, cfg.sim.val_sessions, rng=rng)
        gen_tr = BenchGenerator(pool_tr, cfg.sim, cfg.sim.seed + seed)
        tr = gen_tr.generate(cfg.sim.n_events)
        va = BenchGenerator(pool_tr, cfg.sim, cfg.sim.seed + seed + 77)
        va.staff = gen_tr.staff
        val = va.generate(cfg.sim.n_events // 4)
        splits = {"train": tr, "val": val}
        agent, head, _ = train_heaaf({"train": tr, "val": val}, cfg, seed)
        rba, xai = fit_static_baselines({"train": tr, "val": val}, seed)
        for block in [(5,), (6,), (7,), (8,)]:
            pool = ks.build_pool(df, templates, block, rng=np.random.default_rng(seed))
            g = BenchGenerator(pool, cfg.sim, cfg.sim.seed + 900 + seed)
            g.staff = gen_tr.staff
            te = g.generate(cfg.sim.n_events // 3).reset_index(drop=True)
            X = feature_matrix(te)
            for name, risk, acts in [
                ("HEAAF", head.score(agent, X), None),
                (rba.name, rba.risk(X), rba.act(X)),
                (xai.name, xai.risk(X), xai.act(X))]:
                if acts is None:
                    acts = PolicyEngine(cfg.policy).actions_batch(risk, X)
                flagged = (acts != AIDX["ALLOW"]).astype(int)
                m = M.detection_metrics(te.y.to_numpy(), risk, flagged)
                fr = M.friction_metrics(te, acts)
                rows.append({"policy": name, "seed": seed,
                             "session_block": f"S{block[0]}",
                             "GMean": m["GMean"], "TPR": m["TPR"],
                             "AUROC": m["AUROC"],
                             "interrupt_rate_benign": fr["interrupt_rate_benign"]})
    return pd.DataFrame(rows), pd.DataFrame(eer_rows)


# ==========================================================================
# Aggregation helpers
# ==========================================================================
def aggregate(rows: List[Dict], keys: Sequence[str],
              group: str = "policy") -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = df.groupby(group)[list(keys)].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def save_json(obj, name: str) -> None:
    (LOGDIR / name).write_text(json.dumps(obj, indent=2, default=float))
