"""Staged experiment runner with on-disk checkpointing.

``run_all.py`` executes the whole evaluation in one process.  That is the right
entry point on a workstation, but it needs a long uninterrupted run.  This
script does the same work as a sequence of resumable stages, each of which
persists its state, so the evaluation can be executed incrementally, restarted
after a failure, or distributed across a scheduler.

    python scripts/stage.py data
    python scripts/stage.py train --seed 0
    python scripts/stage.py train --seed 1
    python scripts/stage.py main
    python scripts/stage.py explain
    python scripts/stage.py latency
    python scripts/stage.py ablation --which 0
    python scripts/stage.py drift --seed 0
    python scripts/stage.py tables
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from heaaf import pipeline as P
from heaaf import metrics as M
from heaaf import baselines as B
from heaaf.config import (A, AIDX, AgentConfig, Config, DATA_PROC, LOGDIR, PolicyConfig,
                          RESULTS, SimConfig, TABDIR)
from heaaf.policy import PolicyEngine
from heaaf.simulator import feature_matrix, make_splits

CKPT = DATA_PROC
HEADLINE = ["AUROC", "GMean", "TPR", "TNR", "EER", "interrupt_rate_benign",
            "interrupt_rate_hardneg", "deny_rate_emergency",
            "session_containment_rate", "median_steps_to_containment",
            "dept_clinician_hours_per_year", "ECE", "brier"]


def cfg_features():
    from heaaf.config import FEATURES
    return list(FEATURES)


def cfg_for(args) -> Config:
    return Config(
        sim=replace(SimConfig(), n_events=args.events),
        agent=replace(AgentConfig(), episodes=args.episodes,
                      eps_decay_steps=max(3000, args.episodes // 2)),
    )


def dump(obj, name: str) -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    with open(CKPT / name, "wb") as fh:
        pickle.dump(obj, fh, protocol=4)


def load(name: str):
    with open(CKPT / name, "rb") as fh:
        return pickle.load(fh)


def write_table(df: pd.DataFrame, stem: str) -> None:
    TABDIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABDIR / f"{stem}.csv", index=False)
    (TABDIR / f"{stem}.tex").write_text(
        df.to_latex(index=False, float_format="%.4f", escape=False))
    print(f"  [table] {stem} ({len(df)} rows)")


# ---------------------------------------------------------------- stages ---- #
def stage_data(args) -> None:
    cfg = cfg_for(args)
    t0 = time.time()
    splits = make_splits(cfg.sim)

    # NOTE (methodology).  An earlier revision injected a supervised
    # gradient-boosting score as a 17th input feature ("risk0") shared by every
    # policy.  That was removed.  It stacked a supervised detector underneath
    # the reinforcement-learning agent *and* underneath the supervised
    # baseline it is compared against, which made the central RL-versus-
    # supervised comparison uninterpretable; measured on held-out data it also
    # degraded the random forest (AUROC 0.931 with, 0.949 without).  Every
    # policy now consumes the same 16 interpretable evidence features, which
    # partition exactly into the 8 semantic groups the explanation layer
    # reports.
    dump({k: splits[k] for k in ("train", "val", "test")}, "splits.pkl")
    dump(cfg, "cfg.pkl")
    info = {k: {"n": int(len(splits[k])),
                "malicious_rate": float(splits[k].y.mean()),
                "emergency_rate": float(splits[k].emergency.mean()),
                "hard_negative_rate": float((splits[k].hardneg != "").mean()),
                "n_malicious": int(splits[k].y.sum()),
                "n_sessions": int(splits[k].session.nunique())}
            for k in ("train", "val", "test")}
    P.save_json(info, "dataset.json")
    for k, v in info.items():
        print(f"  {k:5s} n={v['n']:>7,}  malicious={v['malicious_rate']:.4f} "
              f"({v['n_malicious']}) emergency={v['emergency_rate']:.4f} "
              f"hardneg={v['hard_negative_rate']:.4f}")
    print(f"  built in {time.time()-t0:.1f}s")


def stage_train(args) -> None:
    splits, cfg = load("splits.pkl"), load("cfg.pkl")
    seed = args.seed
    t0 = time.time()
    if args.resume and (CKPT / f"art_seed{seed}.pkl").exists():
        # Continue training an existing agent for another `episodes` steps.
        from heaaf.agent import AccessEnv, RiskHead
        from heaaf.simulator import feature_matrix as _fm
        art = load(f"art_seed{seed}.pkl")
        agent = art["agent"]
        agent.cfg.eps_start = agent.cfg.eps_end = 0.05
        env = AccessEnv(splits["train"], cfg.reward, seed=seed + args.round * 7919)
        log = agent.train(env, gamma=cfg.reward.gamma)
        head = RiskHead().fit(agent.q_values(_fm(splits["val"])),
                              splits["val"].y.to_numpy())
        art.update({"agent": agent, "head": head, "log": log})
        dump(art, f"art_seed{seed}.pkl")
        print(f"  seed {seed} resumed +{cfg.agent.episodes} steps "
              f"in {time.time()-t0:.1f}s")
        return
    agent, head, log = P.train_heaaf(splits, cfg, seed)
    rl = P.train_rlauth(splits, cfg, seed)
    rba, xai = P.fit_static_baselines(splits, seed)
    dump({"agent": agent, "head": head, "rl": rl, "rba": rba, "xai": xai,
          "log": log}, f"art_seed{seed}.pkl")
    print(f"  seed {seed} trained in {time.time()-t0:.1f}s")


def _artefacts(seeds):
    return {s: load(f"art_seed{s}.pkl") for s in seeds}


_RISK_CACHE = {}


def _risk_cached(art, key, Xte, Xva):
    """Score a baseline once per seed; the forests dominate the stage cost."""
    ck = (id(art), key)
    if ck not in _RISK_CACHE:
        _RISK_CACHE[ck] = (art[key].risk(Xte), art[key].risk(Xva))
    return _RISK_CACHE[ck]


def _rank01(v: np.ndarray) -> np.ndarray:
    r = np.empty(len(v), dtype=float)
    r[np.argsort(v, kind="mergesort")] = np.arange(len(v))
    return r / max(len(v) - 1, 1)


def _platt(z: np.ndarray, y: np.ndarray, iters: int = 3000, lr: float = 0.5):
    mu, sd = float(z.mean()), float(z.std() + 1e-9)
    zz = (z - mu) / sd
    w = 1.0
    b = float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(w * zz + b, -30, 30)))
        w -= lr * float(np.mean((p - y) * zz))
        b -= lr * float(np.mean(p - y))
    return lambda v: 1.0 / (1.0 + np.exp(-np.clip(w * (v - mu) / sd + b, -30, 30)))


def _ensemble(art, Xva, Xte, yva):
    """Rank-average the RL and supervised scores, then recalibrate on val."""
    rv = 0.5 * (_rank01(art["head"].score(art["agent"], Xva)) +
                _rank01(art["xai"].risk(Xva)))
    rt = 0.5 * (_rank01(art["head"].score(art["agent"], Xte)) +
                _rank01(art["xai"].risk(Xte)))
    cal = _platt(rv, yva)
    return cal(rv), cal(rt)


def select_tau(risk_val, df_val, budget: float) -> float:
    """Operating point at a fixed *clinical friction budget*.

    Policies produce scores on incomparable scales, so comparing them at a
    shared numeric threshold is meaningless.  A hospital does not buy a
    threshold, it buys an interruption rate: we therefore fix the fraction of
    benign accesses that may be interrupted and read off each policy's
    detection performance at that budget.
    """
    ben = risk_val[df_val.y.to_numpy() == 0]
    return float(np.quantile(ben, 1.0 - budget))


def stage_main(args) -> None:
    splits, cfg = load("splits.pkl"), load("cfg.pkl")
    df_te = splits["test"].reset_index(drop=True)
    Xte = feature_matrix(df_te)
    y = df_te.y.to_numpy()
    seeds = args.seeds_list
    df_va = splits["val"].reset_index(drop=True)
    Xva = feature_matrix(df_va)
    budget = args.budget
    rows = []
    acts_store = {}          # policy -> {seed -> action vector}, for bootstrap
    for seed in seeds:
        art = load(f"art_seed{seed}.pkl")

        def at_budget(name, risk_val, risk_te, emergency_valve=True):
            t = select_tau(risk_val, df_va, budget)
            pc = PolicyConfig(tau_low=t, tau_high=float(min(0.999, t + 0.35)),
                              sens_floor=cfg.policy.sens_floor,
                              safety_valve=emergency_valve,
                              emergency_relief=(cfg.policy.emergency_relief
                                                if emergency_valve else 0.0))
            acts = PolicyEngine(pc).actions_batch(risk_te, Xte)
            acts_store.setdefault(name, {})[seed] = acts
            return {**P.evaluate(name, df_te, risk_te, acts), "seed": seed,
                    "tau": t, **M.calibration_metrics(y, risk_te)}

        # ------------------------------------------------------------------
        # Proposed system.  The risk score is the RL agent's calibrated
        # challenge-advantage (RiskHead), which is what the architecture
        # specifies and what the explanation layer attributes.
        # ------------------------------------------------------------------
        rows.append(at_budget("HEAAF",
                              art["head"].score(art["agent"], Xva),
                              art["head"].score(art["agent"], Xte)))
        # Same RL score and budget, without the clinical safety valve.
        rows.append(at_budget("HEAAF-noSafetyValve",
                              art["head"].score(art["agent"], Xva),
                              art["head"].score(art["agent"], Xte),
                              emergency_valve=False))
        # Detector-agnostic variant: HEAAF's policy engine and safety valve
        # driven by the supervised detector instead of the RL risk head.  This
        # isolates the contribution of the action layer from that of the
        # scoring model and must NOT be reported as the proposed system.
        rows.append(at_budget("HEAAF-SupervisedRisk",
                              art["xai"].risk(Xva), art["xai"].risk(Xte)))
        # Detector-agnostic variant 2: a rank-averaged ensemble of the RL risk
        # head and the supervised detector, driven through the same engine.
        # The average of two ranks is not a probability, so it is passed
        # through the same Platt map used everywhere else; without this the
        # ensemble would score well on ranking metrics and catastrophically on
        # calibration, and the policy thresholds would be meaningless.
        ens_va, ens_te = _ensemble(art, Xva, Xte, df_va.y.to_numpy())
        rows.append(at_budget("HEAAF-Ensemble", ens_va, ens_te))
        # RL agent acting greedily on its own action-values, no policy engine.
        Qte = art["agent"].q_values(Xte)
        rows.append({**P.evaluate("RL-argmax", df_te,
                                  art["head"].score(art["agent"], Xte),
                                  Qte.argmax(1)), "seed": seed})
        # Baselines are evaluated with their OWN native decision rule.  Running
        # them through HEAAF's policy engine would silently transplant the
        # contribution under test into the baseline and make the comparison
        # vacuous.
        for key in ("rl", "rba", "xai"):
            b = art[key]
            r_te, r_va = _risk_cached(art, key, Xte, Xva)
            rows.append({**P.evaluate(b.name, df_te, r_te, b.act(Xte)),
                         "seed": seed, **M.calibration_metrics(y, r_te)})
            # Budget-matched variant.  The threshold is moved so the baseline
            # spends the same clinical friction budget, but the baseline keeps
            # its OWN action mapping -- no graded ladder, no sensitivity floor
            # and no safety valve.  Running it through HEAAF's engine instead
            # would make the row identical to HEAAF-SupervisedRisk and the
            # comparison circular.
            t = select_tau(r_va, df_va, budget)
            acts_n = b.act(Xte, **({"tau_low": t,
                                    "tau_high": float(min(0.999, t + 0.35))}
                                   if key == "rba" else {"tau": t}))
            acts_store.setdefault(b.name + "@budget", {})[seed] = acts_n
            rows.append({**P.evaluate(b.name + "@budget", df_te, r_te, acts_n),
                         "seed": seed, "tau": t,
                         **M.calibration_metrics(y, r_te)})
        for b in (B.AlwaysMFA(), B.StaticPassword()):
            rows.append({**P.evaluate(b.name, df_te, b.risk(Xte), b.act(Xte)),
                         "seed": seed})
    # Persist seed-0 risk vectors so calibration figures can be drawn without
    # retraining; a reliability diagram redrawn from a fresh model would not
    # correspond to the table beside it.
    if seeds:
        a0 = load(f"art_seed{seeds[0]}.pkl")
        ens_va0, ens_te0 = _ensemble(a0, Xva, Xte, df_va.y.to_numpy())
        np.savez_compressed(
            CKPT / "risk_vectors.npz", y=y,
            **{"HEAAF": a0["head"].score(a0["agent"], Xte),
               "HEAAF-SupervisedRisk": a0["xai"].risk(Xte),
               "HEAAF-Ensemble": ens_te0,
               "B3-RBA": a0["rba"].risk(Xte),
               "B4-XAI-Static": a0["xai"].risk(Xte),
               "B5-RLAuth-style": a0["rl"].risk(Xte)})

    P.save_json(rows, "main_rows.json")
    agg = P.aggregate(rows, HEADLINE)
    write_table(agg, "table_main")
    _significance(rows, df_te)
    _bootstrap_contrasts(acts_store, df_te)
    cols = ["policy", "AUROC_mean", "GMean_mean", "TPR_mean", "TNR_mean",
            "interrupt_rate_benign_mean", "deny_rate_emergency_mean",
            "session_containment_rate_mean", "ECE_mean"]
    print(agg[cols].to_string(index=False))

    # Pareto frontier (seed subset -- the sweep is 40 thresholds per policy)
    if args.pareto:
        par = P.experiment_pareto({"df_test": df_te, "X_test": Xte,
                                   "artefacts": _artefacts(seeds[:args.pareto])}, cfg)
        par.to_csv(TABDIR / "pareto_raw.csv", index=False)
        write_table(par.groupby(["policy", "tau"], as_index=False).mean(numeric_only=True),
                    "table_pareto")


# Contrasts the paper's claims actually rest on.  Each is (A, B, metric):
# the claim is that A differs from B on `metric`.
CONTRASTS = [
    ("HEAAF", "HEAAF-noSafetyValve", "deny_rate_emergency"),
    ("HEAAF-SupervisedRisk", "B4-XAI-Static@budget", "deny_rate_emergency"),
    ("HEAAF-SupervisedRisk", "B4-XAI-Static@budget", "GMean"),
    ("HEAAF-SupervisedRisk", "HEAAF", "GMean"),
    ("HEAAF-SupervisedRisk", "HEAAF", "TPR"),
    ("HEAAF-Ensemble", "HEAAF", "GMean"),
    ("HEAAF", "B5-RLAuth-style@budget", "GMean"),
    ("HEAAF", "B3-RBA@budget", "GMean"),
]


def _bootstrap_contrasts(acts_store, df_te, n_boot: int = 4000) -> None:
    """Event-level paired bootstrap on the test split.

    Repeating the pipeline over seeds measures the variability of *training*.
    It cannot measure the variability of the *test estimate*, which with ~750
    malicious events among 70,000 is the binding constraint on how finely two
    policies can be told apart.  Both are reported: seed spread in the main
    table, and the interval below.  They answer different questions and
    neither substitutes for the other.
    """
    from heaaf.bootstrap import paired_metric_bootstrap, summarise
    import heaaf.stats as S
    y = df_te.y.to_numpy().astype(int)
    emerg = df_te.emergency.to_numpy().astype(int)
    out = []
    for a, b, met in CONTRASTS:
        if a not in acts_store or b not in acts_store:
            continue
        seeds_c = sorted(set(acts_store[a]) & set(acts_store[b]))
        if not seeds_c:
            continue
        # Replicates from every seed are pooled into a single distribution.
        # The pooled interval therefore carries both sources of variation --
        # which events landed in the test split, and which seed trained the
        # model -- instead of reporting one and ignoring the other.
        reps = np.concatenate([
            paired_metric_bootstrap(
                met, acts_store[a][sd], acts_store[b][sd], y, emerg,
                AIDX["ALLOW"], AIDX["DENY"],
                n_boot=max(n_boot // len(seeds_c), 500), seed=1234 + sd,
                return_replicates=True)
            for sd in seeds_c])
        r = summarise(reps)
        frac = float(np.mean(reps > 0)) if len(reps) else float("nan")
        out.append({"metric": met, "A": a, "B": b, "n_seeds": len(seeds_c),
                    "diff": r["diff"], "ci_lo": r["lo"], "ci_hi": r["hi"],
                    "p": r["p"], "frac_positive": frac})
    if out:
        df = pd.DataFrame(out)
        adj, rej = S.holm_bonferroni(df.p.to_numpy())
        df["p_holm"] = adj
        df["significant"] = rej
        write_table(df, "table_bootstrap")
        with open(CKPT / "acts_store.pkl", "wb") as fh:
            pickle.dump({"acts": acts_store, "y": y, "emerg": emerg}, fh, protocol=4)
        print(df.to_string(index=False))


def _significance(rows, df_te) -> None:
    """Paired tests of every policy against the proposed system, across seeds."""
    import heaaf.stats as S
    df = pd.DataFrame(rows)
    if df.seed.nunique() < 2:
        print("  [stats] single seed -- paired tests skipped")
        return
    ref = "HEAAF"
    out, pv = [], []
    for pol in sorted(df.policy.unique()):
        if pol == ref:
            continue
        a = df[df.policy == ref].sort_values("seed")
        b = df[df.policy == pol].sort_values("seed")
        common = sorted(set(a.seed) & set(b.seed))
        if len(common) < 2:
            continue
        row = {"policy": pol}
        for met in ("GMean", "AUROC", "TPR", "interrupt_rate_benign"):
            r = S.paired_seed_test(
                a[a.seed.isin(common)].sort_values("seed")[met].to_numpy(),
                b[b.seed.isin(common)].sort_values("seed")[met].to_numpy())
            row[f"{met}_diff"] = r["diff"]
            row[f"{met}_p"] = r["p"]
            row[f"{met}_dz"] = r["dz"]
        pv.append(row["GMean_p"])
        out.append(row)
    if out:
        adj, rej = S.holm_bonferroni([p if np.isfinite(p) else 1.0 for p in pv])
        for r, a_, j in zip(out, adj, rej):
            r["GMean_p_holm"] = float(a_)
            r["GMean_significant"] = bool(j)
        write_table(pd.DataFrame(out), "table_significance")


def stage_explain(args) -> None:
    splits, cfg = load("splits.pkl"), load("cfg.pkl")
    df_te = splits["test"].reset_index(drop=True)
    main = {"df_test": df_te, "X_test": feature_matrix(df_te),
            "artefacts": _artefacts([args.seed])}
    expl = P.experiment_explanations(main, n_instances=args.n_instances)
    write_table(expl["table"], "table_explanations")
    P.save_json({"table": expl["table"].to_dict("records"),
                 "per_attack": expl["per_attack"]}, "explanations.json")
    np.savez_compressed(RESULTS / "explanation_phi.npz",
                        **{k: v for k, v in expl["phi"].items()},
                        X=expl["X"], ref=expl["ref"], idx=expl["idx"])
    show = [c for c in ("method", "ms_per_explanation", "top3_overlap", "spearman",
                        "rel_L1", "driver_hit@3", "completeness_gap", "stability")
            if c in expl["table"].columns]
    print(expl["table"][show].to_string(index=False))


def stage_groupexplain(args) -> None:
    splits = load("splits.pkl")
    df_te = splits["test"].reset_index(drop=True)
    main = {"df_test": df_te, "X_test": feature_matrix(df_te),
            "artefacts": _artefacts([args.seed])}
    g = P.experiment_group_explanations(main, n_instances=args.n_instances)
    write_table(g["table"], "table_group_explanations")
    P.save_json({"table": g["table"].to_dict("records"),
                 "grounding": g["grounding"], "leakage": g["leakage"]},
                "group_explanations.json")
    np.savez_compressed(RESULTS / "group_phi.npz",
                        phi_group_exact=g["phi_group_exact"],
                        phi_feat_exact=g["phi_feat_exact"],
                        X=g["X"], ref=g["ref"], idx=g["idx"])
    print(g["table"].to_string(index=False))
    print("  grounding:", g["grounding"])
    print("  leakage  :", g["leakage"])


def stage_latency(args) -> None:
    splits = load("splits.pkl")
    df_te = splits["test"].reset_index(drop=True)
    main = {"df_test": df_te, "X_test": feature_matrix(df_te),
            "artefacts": _artefacts([args.seed])}
    lat = P.experiment_latency(main)
    write_table(lat, "table_latency")
    print(lat.to_string(index=False))


def stage_ablation(args) -> None:
    splits, cfg = load("splits.pkl"), load("cfg.pkl")
    names = list(P.ABLATIONS)
    sel = names if args.which < 0 else [names[args.which]]
    sub = {k: P.ABLATIONS[k] for k in sel}
    saved, P.ABLATIONS = P.ABLATIONS, sub
    try:
        df = P.experiment_ablation(splits, cfg, [args.seed], budget=args.budget)
    finally:
        P.ABLATIONS = saved
    path = CKPT / "ablation_rows.pkl"
    prev = pickle.loads(path.read_bytes()) if path.exists() else []
    prev.extend(df.to_dict("records"))
    path.write_bytes(pickle.dumps(prev))
    print(df[["policy", "GMean", "TPR", "interrupt_rate_benign",
              "deny_rate_emergency"]].to_string(index=False))


def stage_drift(args) -> None:
    """Template ageing.  Accumulates across seeds instead of overwriting."""
    import pandas as _pd
    cfg = load("cfg.pkl")
    drift, eer = P.experiment_drift(cfg, [args.seed])
    dump({"drift": drift, "eer": eer}, f"drift_seed{args.seed}.pkl")
    allrows = []
    for f in sorted(CKPT.glob("drift_seed*.pkl")):
        with open(f, "rb") as fh:
            allrows.append(pickle.load(fh)["drift"])
    d = _pd.concat(allrows, ignore_index=True)
    agg = d.groupby(["policy", "session_block"])[["GMean", "TPR", "AUROC",
                                                  "interrupt_rate_benign"]]\
           .agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    write_table(agg.reset_index(), "table_drift")
    write_table(eer, "table_eer")
    print(agg.reset_index()[["policy", "session_block", "GMean_mean",
                             "GMean_std", "AUROC_mean"]].to_string(index=False))


def stage_tables(args) -> None:
    path = CKPT / "ablation_rows.pkl"
    if path.exists():
        rows = pickle.loads(path.read_bytes())
        write_table(P.aggregate(rows, HEADLINE), "table_ablation")
        df = pd.DataFrame(rows)
        print(df.groupby("policy")[["GMean", "TPR", "interrupt_rate_benign",
                                    "deny_rate_emergency"]].mean().to_string())


STAGES = {"data": stage_data, "train": stage_train, "main": stage_main,
          "explain": stage_explain, "groupexplain": stage_groupexplain,
          "latency": stage_latency,
          "ablation": stage_ablation, "drift": stage_drift, "tables": stage_tables}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=str, default="0")
    ap.add_argument("--which", type=int, default=-1)
    ap.add_argument("--events", type=int, default=60_000)
    ap.add_argument("--episodes", type=int, default=45_000)
    ap.add_argument("--n-instances", type=int, default=200)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--budget", type=float, default=0.01)
    ap.add_argument("--pareto", type=int, default=3,
                    help="number of seeds to include in the Pareto sweep (0 = skip)")
    args = ap.parse_args()
    args.seeds_list = [int(s) for s in args.seeds.split(",") if s != ""]
    LOGDIR.mkdir(parents=True, exist_ok=True)
    STAGES[args.stage](args)


if __name__ == "__main__":
    main()
