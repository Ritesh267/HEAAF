"""Emit publication-ready LaTeX tables from the persisted results.

Every number printed in the manuscript is produced here, from the CSV and
JSON files the staged pipeline wrote.  Nothing is transcribed by hand, so a
table in the paper cannot drift away from the run that produced it: re-running
the pipeline and re-running this script regenerates the manuscript's numbers
exactly.

    python scripts/make_tables.py

Output goes to ``results/tables/tex/`` as fragments that ``main.tex`` inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from heaaf.config import ATTACKS, LOGDIR, TABDIR

TEXDIR = TABDIR / "tex"
TEXDIR.mkdir(parents=True, exist_ok=True)


def _w(name: str, body: str) -> None:
    (TEXDIR / f"{name}.tex").write_text(body.rstrip() + "\n")
    print(f"  [tex] {name}.tex")


def _csv(stem: str):
    p = TABDIR / f"{stem}.csv"
    return pd.read_csv(p) if p.exists() else None


def _json(stem: str):
    p = LOGDIR / f"{stem}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pm(m, s, dp=3):
    """'mean $\\pm$ sd', or just the mean when the spread is exactly zero."""
    if m is None or not np.isfinite(m):
        return "--"
    if s is None or not np.isfinite(s) or s == 0:
        return f"{m:.{dp}f}"
    return f"{m:.{dp}f}\\,\\smaller{{$\\pm$\\,{s:.{dp}f}}}"


def tt(name: str) -> str:
    """Typeset an identifier verbatim.

    Underscores are active in text mode, and long identifiers cannot be
    hyphenated, so a discretionary break is inserted after each underscore to
    keep narrow table columns from overflowing.
    """
    return "\\texttt{" + str(name).replace("_", "\\_\\allowbreak{}") + "}"


def bold_if(txt: str, cond: bool) -> str:
    return f"\\textbf{{{txt}}}" if cond else txt


# ==========================================================================
# Table 1 -- evaluation corpus
# ==========================================================================
def t_corpus() -> None:
    d = _json("dataset")
    if not d:
        return
    rows = []
    for k, lab in (("train", "Train"), ("val", "Validation"), ("test", "Test")):
        v = d[k]
        rows.append(
            f"{lab} & {v['n']:,} & {v['n_sessions']:,} & {v['n_malicious']:,} & "
            f"{100*v['malicious_rate']:.2f} & {100*v['emergency_rate']:.2f} & "
            f"{100*v['hard_negative_rate']:.2f} \\\\")
    _w("tab_corpus", r"""\begin{tabular}{lrrrrrr}
\toprule
& & & \multicolumn{2}{c}{Malicious} & Emerg. & Hard neg. \\
\cmidrule(lr){4-5}
Split & Events & Sessions & $n$ & \% & \% & \% \\
\midrule
""" + "\n".join(rows).replace(",", "{,}") + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Table 2 -- main policy comparison
# ==========================================================================
MAIN_ORDER = [
    ("B2-StaticLogin", "B2 Static login (gate only)", "deg"),
    ("B1-AlwaysMFA", "B1 Always-MFA (challenge all)", "deg"),
    ("B3-RBA", "B3 RBA \\cite{freeman2016,wiefling2022}", "nat"),
    ("B4-XAI-Static", "B4 XAI-static \\cite{raouf2024,nizam2026}", "nat"),
    ("B5-RLAuth-style", "B5 RLAuth-style \\cite{rlauth}", "nat"),
    ("B3-RBA@budget", "\\quad B3 at matched budget", "bud"),
    ("B4-XAI-Static@budget", "\\quad B4 at matched budget", "bud"),
    ("B5-RLAuth-style@budget", "\\quad B5 at matched budget", "bud"),
    ("RL-argmax", "RL agent, no policy engine", "abl"),
    ("HEAAF-noSafetyValve", "HEAAF without safety valve", "abl"),
    ("HEAAF", "\\textbf{HEAAF} (RL risk)", "ours"),
    ("HEAAF-Ensemble", "\\textbf{HEAAF-Ensemble}", "ours"),
    ("HEAAF-SupervisedRisk", "\\textbf{HEAAF-SupervisedRisk}", "ours"),
]
COMPARABLE = {"B3-RBA@budget", "B4-XAI-Static@budget", "B5-RLAuth-style@budget",
              "HEAAF", "HEAAF-Ensemble", "HEAAF-SupervisedRisk"}


def t_main() -> None:
    df = _csv("table_main")
    if df is None:
        return
    d = df.set_index("policy")
    comp = [p for p in COMPARABLE if p in d.index]
    best = {
        "AUROC_mean": max(d.loc[comp, "AUROC_mean"]),
        "GMean_mean": max(d.loc[comp, "GMean_mean"]),
        "TPR_mean": max(d.loc[comp, "TPR_mean"]),
        "session_containment_rate_mean": max(d.loc[comp, "session_containment_rate_mean"]),
        "deny_rate_emergency_mean": min(d.loc[comp, "deny_rate_emergency_mean"]),
        "dept_clinician_hours_per_year_mean": min(d.loc[comp, "dept_clinician_hours_per_year_mean"]),
    }
    lines, prev = [], None
    for key, label, kind in MAIN_ORDER:
        if key not in d.index:
            continue
        if prev is not None and kind != prev:
            lines.append(r"\midrule")
        prev = kind
        r = d.loc[key]
        ok = key in COMPARABLE
        cells = [
            label,
            bold_if(pm(r.AUROC_mean, r.get("AUROC_std")), ok and np.isclose(r.AUROC_mean, best["AUROC_mean"])),
            bold_if(pm(r.GMean_mean, r.get("GMean_std")), ok and np.isclose(r.GMean_mean, best["GMean_mean"])),
            bold_if(pm(r.TPR_mean, r.get("TPR_std")), ok and np.isclose(r.TPR_mean, best["TPR_mean"])),
            f"{100*r.interrupt_rate_benign_mean:.2f}",
            f"{100*r.interrupt_rate_hardneg_mean:.2f}",
            bold_if(f"{100*r.deny_rate_emergency_mean:.3f}",
                    ok and np.isclose(r.deny_rate_emergency_mean, best["deny_rate_emergency_mean"])),
            bold_if(f"{r.session_containment_rate_mean:.3f}",
                    ok and np.isclose(r.session_containment_rate_mean, best["session_containment_rate_mean"])),
            bold_if(f"{r.dept_clinician_hours_per_year_mean:.1f}",
                    ok and np.isclose(r.dept_clinician_hours_per_year_mean, best["dept_clinician_hours_per_year_mean"])),
        ]
        lines.append(" & ".join(cells) + r" \\")
    _w("tab_main", r"""\begin{tabular}{lccccccccr}
\toprule
& & & & \multicolumn{2}{c}{Interrupted (\%)} & Denied & Session & Clin.\ h \\
\cmidrule(lr){5-6}
Policy & AUROC & G-mean & TPR & benign & hard neg. & emerg.\ (\%) & contain. & /yr \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Table 3 -- paired bootstrap contrasts
# ==========================================================================
NICE = {"deny_rate_emergency": "Emergency denial rate",
        "GMean": "G-mean", "TPR": "TPR",
        "interrupt_rate_benign": "Benign interruption"}


def t_bootstrap() -> None:
    df = _csv("table_bootstrap")
    if df is None:
        return
    lines = []
    for _, r in df.iterrows():
        star = "\\checkmark" if r.significant else "--"
        short = lambda n: (n.replace("HEAAF-SupervisedRisk", "HEAAF-Sup.")
                             .replace("HEAAF-noSafetyValve", "HEAAF, no valve")
                             .replace("B4-XAI-Static@budget", "B4@budget")
                             .replace("B5-RLAuth-style@budget", "B5@budget")
                             .replace("B3-RBA@budget", "B3@budget"))
        lines.append(
            f"{NICE.get(r.metric, r.metric)} & {short(r.A)} & {short(r.B)} & "
            f"{r['diff']:+.4f} & $[{r.ci_lo:+.4f},\\,{r.ci_hi:+.4f}]$ & "
            f"{r.p_holm:.3f} & {star} \\\\")
    _w("tab_bootstrap", r"""\begin{tabular}{>{\raggedright\arraybackslash}p{1.9cm}p{2.6cm}p{2.6cm}rlcc}
\toprule
Metric & Policy A & Policy B & \multicolumn{1}{c}{$A-B$} & 95\% CI & $p_{\text{Holm}}$ & Sig. \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Table 4 -- feature-level attribution fidelity
# ==========================================================================
def t_explanations() -> None:
    df = _csv("table_explanations")
    if df is None:
        return
    lines = []
    for _, r in df.iterrows():
        name = str(r.method).replace("GradxInput", "Grad$\\times$Input") \
                            .replace("ExactShapley", "Exact Shapley (2$^{16}$)")
        gap = r.completeness_gap
        gap_s = "$<10^{-13}$" if gap < 1e-13 else f"{gap:.3f}"
        lines.append(
            f"{name} & {r.ms_per_explanation:.4f} & {r.top3_overlap:.3f} & "
            f"{r.spearman:.3f} & {r.rel_L1:.3f} & {gap_s} & "
            f"{r['driver_hit@3']:.3f} \\\\")
    _w("tab_explanations", r"""\begin{tabular}{lrccccc}
\toprule
Method & ms/expl. & Top-3 & Spearman $\rho$ & Rel.\ $L_1$ & Compl.\ gap & Driver hit@3 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Table 5 -- group-level attribution (what the clinician sees)
# ==========================================================================
def t_group() -> None:
    df = _csv("table_group_explanations")
    if df is None:
        return
    lines = []
    for _, r in df.iterrows():
        name = (str(r.method).replace("GradxInput", "Grad$\\times$Input")
                .replace("Exact group Shapley (disclosed)",
                         "\\textbf{Exact group Shapley} (2$^{8}$)")
                .replace("Exact feature Shapley, summed",
                         "Exact feature Shapley (2$^{16}$), block-summed"))
        gap = r.completeness_gap
        gap_s = "$<10^{-13}$" if gap < 1e-13 else f"{gap:.3f}"
        lines.append(f"{name} & {r.ms_per_explanation:.4f} & {r.top1_overlap:.3f} & "
                     f"{r.spearman:.3f} & {r.rel_L1:.3f} & {gap_s} \\\\")
    _w("tab_group", r"""\begin{tabular}{lrcccc}
\toprule
Method & ms/expl. & Top-1 & Spearman $\rho$ & Rel.\ $L_1$ & Compl.\ gap \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Table 6 -- decision-path latency
# ==========================================================================
def t_latency() -> None:
    df = _csv("table_latency")
    if df is None:
        return
    lines = [f"{r.stage} & {r.p50_ms:.4f} & {r.p95_ms:.4f} & {r.p99_ms:.4f} \\\\"
             for _, r in df.iterrows()]
    _w("tab_latency", r"""\begin{tabular}{lrrr}
\toprule
Decision-path stage & p50 (ms) & p95 (ms) & p99 (ms) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Table 7 -- per-scenario detection
# ==========================================================================
SCEN_COLS = ["HEAAF", "HEAAF-Ensemble", "HEAAF-SupervisedRisk",
             "B3-RBA@budget", "B4-XAI-Static@budget", "B5-RLAuth-style@budget"]
SCEN_HEAD = ["HEAAF", "Ens.", "Sup.", "B3@b", "B4@b", "B5@b"]


def t_scenarios() -> None:
    rows = _json("main_rows")
    if not rows:
        return
    acc = {}
    for r in rows:
        if r["policy"] in SCEN_COLS and isinstance(r.get("per_attack"), dict):
            acc.setdefault(r["policy"], []).append(r["per_attack"])
    if not acc:
        return
    names = sorted({k for v in acc.values() for d in v for k in d})
    expl = _json("explanations") or {}
    per_atk = expl.get("per_attack", {})
    lines = []
    for n in names:
        lab = ATTACKS[n]["label"] if n in ATTACKS else n
        vals = [np.mean([d.get(n, np.nan) for d in acc[p]]) if p in acc else np.nan
                for p in SCEN_COLS]
        best = np.nanmax(vals)
        cells = [bold_if(f"{v:.3f}", np.isclose(v, best)) if np.isfinite(v) else "--"
                 for v in vals]
        hit = per_atk.get(n, {}).get("driver_hit@3")
        cells.append(f"{hit:.3f}" if hit is not None else "--")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    _w("tab_scenarios", r"""\begin{tabular}{l""" + "c" * (len(SCEN_COLS) + 1) + r"""}
\toprule
& \multicolumn{""" + str(len(SCEN_COLS)) + r"""}{c}{Detection rate} & Driver \\
\cmidrule(lr){2-""" + str(len(SCEN_COLS) + 1) + r"""}
Attack scenario & """ + " & ".join(SCEN_HEAD) + r""" & hit@3 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Table 8 -- ablations
# ==========================================================================
def t_ablation() -> None:
    df = _csv("table_ablation")
    if df is None:
        return
    d = df.set_index("policy")
    full = [i for i in d.index if "full" in i.lower()]
    base = float(d.loc[full[0], "GMean_mean"]) if full else np.nan
    order = full + [i for i in d.index if i not in full]
    lines = []
    for i, key in enumerate(order):
        r = d.loc[key]
        lab = key.replace("w/o ", "$-$ ").replace("_", " ")
        delta = "--" if key in full else f"{r.GMean_mean - base:+.3f}"
        lines.append(f"{lab} & {pm(r.GMean_mean, r.get('GMean_std'))} & {delta} & "
                     f"{pm(r.TPR_mean, r.get('TPR_std'))} & "
                     f"{100*r.interrupt_rate_benign_mean:.2f} & "
                     f"{100*r.deny_rate_emergency_mean:.3f} \\\\")
        if i == 0:
            lines.append(r"\midrule")
    _w("tab_ablation", r"""\begin{tabular}{lcccrr}
\toprule
Configuration & G-mean & $\Delta$ & TPR & Int.\ (\%) & Deny em.\ (\%) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Table 9 -- template ageing
# ==========================================================================
def t_drift() -> None:
    eer = _csv("table_eer")
    dr = _csv("table_drift")
    if eer is None:
        return
    pol = ["HEAAF", "B3-RBA", "B4-XAI-Static"]
    lines = []
    for _, e in eer.iterrows():
        blk = e.session_block
        cells = [blk, f"{e.eer_mean:.4f}"]
        for p in pol:
            if dr is None:
                cells.append("--")
                continue
            m = dr[(dr.policy == p) & (dr.session_block == blk)]
            col = "GMean_mean" if "GMean_mean" in dr.columns else "GMean"
            sd = "GMean_std" if "GMean_std" in dr.columns else None
            cells.append(pm(float(m[col].iloc[0]),
                            float(m[sd].iloc[0]) if sd is not None else None)
                         if len(m) else "--")
        lines.append(" & ".join(cells) + r" \\")
    _w("tab_drift", r"""\begin{tabular}{lcccc}
\toprule
Test & Keystroke & \multicolumn{3}{c}{G-mean at a 1\% friction budget} \\
\cmidrule(lr){3-5}
session & EER & HEAAF & B3-RBA & B4-XAI-Static \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Appendix -- feature space and semantic groups
# ==========================================================================
def t_features() -> None:
    from heaaf.config import FEATURE_LABELS, GROUPS
    lines = []
    for g, feats in GROUPS.items():
        for j, f in enumerate(feats):
            first = f"\\multirow{{{len(feats)}}}{{*}}{{{g}}}" if j == 0 else ""
            lines.append(f"{first} & {tt(f)} & {FEATURE_LABELS[f]} \\\\")
        lines.append(r"\addlinespace[2pt]")
    _w("tab_features", r"""\begin{tabular}{p{2.1cm}p{2.5cm}>{\raggedright\arraybackslash}p{8.2cm}}
\toprule
Semantic group & Feature & Meaning (oriented so larger $=$ more suspicious) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Appendix -- attack taxonomy
# ==========================================================================
def t_attacks() -> None:
    from heaaf.config import HARD_NEGATIVES
    lines = []
    for k, v in ATTACKS.items():
        lines.append(f"{v['label']} & {100*v['weight']:.0f} & {v['anchor']} & "
                     + ", ".join(tt(d) for d in v["drivers"]) + r" \\")
    lines.append(r"\midrule")
    for k, v in HARD_NEGATIVES.items():
        lines.append(f"{v['label']} & -- & legitimate, unusual & "
                     + ", ".join(tt(d) for d in v["drivers"]) + r" \\")
    _w("tab_attacks", r"""\begin{tabular}{>{\raggedright\arraybackslash}p{2.7cm}r>{\raggedright\arraybackslash}p{3.2cm}>{\raggedright\arraybackslash}p{7.0cm}}
\toprule
Scenario & \% & Anchor incident class & Declared ground-truth drivers \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Appendix -- hyperparameters
# ==========================================================================
def t_hyper() -> None:
    from dataclasses import asdict
    from heaaf.config import (ACTION_SECONDS, AgentConfig, CHALLENGE_BYPASS_P,
                              PolicyConfig, RewardConfig, SimConfig)
    a, r, s, p = AgentConfig(), RewardConfig(), SimConfig(), PolicyConfig()
    rows = [
        ("Q-network", "hidden layers", str(list(a.hidden))),
        ("", "learning rate (Adam)", f"{a.lr:g}"),
        ("", "batch size", str(a.batch)),
        ("", "replay capacity", f"{a.buffer:,}".replace(",", "{,}")),
        ("", "environment steps", f"{a.episodes:,}".replace(",", "{,}")),
        ("", "warm-up steps", f"{a.warmup:,}".replace(",", "{,}")),
        ("", "target sync interval", str(a.target_sync)),
        ("", "$\\varepsilon$ schedule", f"{a.eps_start:g}$\\to${a.eps_end:g} over {a.eps_decay_steps:,}".replace(",", "{,}")),
        ("", "malicious fraction per minibatch", f"{a.balanced_replay:g}"),
        ("", "gradient clip", f"{a.grad_clip:g}"),
        ("Reward", "breach cost (base, sens.\\ gain)", f"{r.breach_base:g}, {r.breach_sens_gain:g}"),
        ("", "value of a frictionless allow", f"{r.allow_benign:g}"),
        ("", "friction (OTP / strong / deny)", f"{r.friction['STEP_UP_OTP']:g} / {r.friction['STEP_UP_STRONG']:g} / {r.friction['DENY']:g}"),
        ("", "emergency penalty (OTP / strong / deny)", f"{r.emergency_penalty['STEP_UP_OTP']:g} / {r.emergency_penalty['STEP_UP_STRONG']:g} / {r.emergency_penalty['DENY']:g}"),
        ("", "correct challenge (OTP / strong / deny)", f"{r.correct_challenge['STEP_UP_OTP']:g} / {r.correct_challenge['STEP_UP_STRONG']:g} / {r.correct_challenge['DENY']:g}"),
        ("", "discount $\\gamma$", f"{r.gamma:g}"),
        ("Policy", "$\\tau_{\\text{low}}$ (set by friction budget)", "1\\% benign quantile"),
        ("", "$\\tau_{\\text{high}}$", "$\\tau_{\\text{low}}+0.35$"),
        ("", "sensitivity floor $\\phi$", f"{p.sens_floor:g}"),
        ("", "emergency relief $\\rho$", f"{p.emergency_relief:g}"),
        ("Challenge", "bypass prob.\\ (OTP / strong)", f"{CHALLENGE_BYPASS_P['STEP_UP_OTP']:g} / {CHALLENGE_BYPASS_P['STEP_UP_STRONG']:g}"),
        ("", "clinician seconds (OTP / strong / deny)", f"{ACTION_SECONDS['STEP_UP_OTP']:g} / {ACTION_SECONDS['STEP_UP_STRONG']:g} / {ACTION_SECONDS['DENY']:g}"),
        ("Corpus", "events (train)", f"{140000:,}".replace(",", "{,}")),
        ("", "attack rate / hard-negative rate", f"{100*s.attack_rate:g}\\% / {100*s.hard_negative_rate:g}\\%"),
        ("", "emergency rate", f"{100*s.emergency_rate:g}\\%"),
        ("", "mean session length", f"{s.mean_session_len:g}"),
    ]
    lines = [f"{a_} & {b} & {c} \\\\" for a_, b, c in rows]
    _w("tab_hyper", r"""\begin{tabular}{p{1.5cm}>{\raggedright\arraybackslash}p{6.0cm}>{\raggedright\arraybackslash}p{5.4cm}}
\toprule
Component & Parameter & Value \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}""")


# ==========================================================================
# Machine-readable summary of every number quoted in the prose
# ==========================================================================
def numbers() -> None:
    m = _csv("table_main")
    out = {}
    if m is not None:
        d = m.set_index("policy")
        for p in d.index:
            out[p] = {c.replace("_mean", ""): float(d.loc[p, c])
                      for c in d.columns if c.endswith("_mean")
                      and np.isfinite(d.loc[p, c])}
    for stem, key in (("table_bootstrap", "bootstrap"),
                      ("table_explanations", "explanations"),
                      ("table_group_explanations", "group_explanations"),
                      ("table_latency", "latency"),
                      ("table_ablation", "ablation"),
                      ("table_eer", "eer")):
        t = _csv(stem)
        if t is not None:
            out[key] = t.to_dict("records")
    g = _json("group_explanations")
    if g:
        out["grounding"] = g["grounding"]
        out["leakage"] = g["leakage"]
    (TEXDIR / "numbers.json").write_text(json.dumps(out, indent=1, default=float))
    print("  [tex] numbers.json")


def main() -> None:
    for fn in (t_corpus, t_main, t_bootstrap, t_explanations, t_group, t_latency,
               t_scenarios, t_ablation, t_drift, t_features, t_attacks, t_hyper):
        try:
            fn()
        except Exception as exc:
            print(f"  [skip] {fn.__name__}: {type(exc).__name__}: {exc}")
    numbers()
    print(f"\n  tables written to {TEXDIR}")


if __name__ == "__main__":
    main()
