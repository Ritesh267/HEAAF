"""Regenerate every figure in the manuscript from the persisted results.

The script reads only what the staged pipeline has already written to
``results/``.  It never retrains anything, so figures cannot silently drift
away from the tables they sit next to: if a figure is missing, the stage that
produces its inputs has not been run.

    python scripts/make_figures.py            # all available figures
    python scripts/make_figures.py --only pareto reliability

Output is vector PDF at a column width that matches IEEEtran, with type-1
fonts embedded, so the figures drop straight into the LaTeX build.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

import _bootstrap  # noqa: F401

from heaaf.config import (ATTACKS, DATA_PROC, FIGDIR, GROUP_NAMES, LOGDIR,
                          RESULTS, TABDIR)

# -- house style -----------------------------------------------------------
COL_W = 3.45          # IEEE single-column width, inches
COL2_W = 7.16         # IEEE double-column width
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "legend.fontsize": 6.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

PALETTE = {
    "HEAAF": "#1b3a6b",
    "HEAAF-SupervisedRisk": "#0f7b6c",
    "HEAAF-Ensemble": "#7d4a9e",
    "B3-RBA": "#c8791a",
    "B4-XAI-Static": "#b3283a",
    "B5-RLAuth-style": "#5b6770",
}
MARKER = {"HEAAF": "o", "HEAAF-SupervisedRisk": "s", "HEAAF-Ensemble": "D",
          "B3-RBA": "^", "B4-XAI-Static": "v", "B5-RLAuth-style": "x"}


def _save(fig, stem: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / f"{stem}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [figure] {out.name}")


def _csv(stem: str):
    p = TABDIR / f"{stem}.csv"
    return pd.read_csv(p) if p.exists() else None


def _json(stem: str):
    p = LOGDIR / f"{stem}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ==========================================================================
# 1. Security / friction Pareto frontier
# ==========================================================================
def fig_pareto() -> bool:
    df = _csv("pareto_raw")
    if df is None:
        return False
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for pol, grp in df.groupby("policy"):
        g = grp.groupby("tau", as_index=False)[["interrupt_rate_benign", "TPR"]].mean()
        g = g.sort_values("interrupt_rate_benign")
        ax.plot(g.interrupt_rate_benign, g.TPR,
                color=PALETTE.get(pol, "#666"), marker=MARKER.get(pol, "."),
                markersize=2.4, markevery=3, label=pol)
    ax.axvline(0.01, color="k", ls=":", lw=0.9)
    ax.annotate("1\\% friction\nbudget", xy=(0.01, 0.99), xytext=(0.0135, 0.99),
                fontsize=6, va="top", ha="left", color="#333")
    ax.set_xscale("log")
    ax.set_xlabel("Benign interruption rate (log scale)")
    ax.set_ylabel("True positive rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    _save(fig, "fig_pareto")
    return True


# ==========================================================================
# 2. Reliability diagram
# ==========================================================================
def fig_reliability() -> bool:
    rows = _json("main_rows")
    risks = DATA_PROC / "risk_vectors.npz"
    if rows is None or not risks.exists():
        return False
    z = np.load(risks, allow_pickle=True)
    y = z["y"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.plot([0, 1], [0, 1], color="k", ls="--", lw=0.8, label="perfect")
    for pol in ("HEAAF", "HEAAF-SupervisedRisk", "B3-RBA"):
        if pol not in z:
            continue
        from heaaf.metrics import reliability_curve
        xs, ys, ns = reliability_curve(y, z[pol], bins=12)
        ax.plot(xs, ys, marker=MARKER.get(pol, "o"), markersize=3,
                color=PALETTE.get(pol, "#666"), label=pol)
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed malicious fraction")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    _save(fig, "fig_reliability")
    return True


# ==========================================================================
# 3. Explanation fidelity versus cost
# ==========================================================================
def fig_explanation_frontier() -> bool:
    """Cost against fidelity, at both levels of the explanation.

    Panel (a) is the 16-player feature game, where exact Shapley is a 65,536
    coalition computation and every practical method is an approximation.
    Panel (b) is the 8-player group game -- the level actually disclosed to a
    clinician -- where exactness costs 256 evaluations and is therefore an
    available option rather than a reference point.
    """
    df = _csv("table_explanations")
    dg = _csv("table_group_explanations")
    if df is None and dg is None:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 2.45))

    def scatter(ax, frame, xcol, ycol, exact_marker):
        offs = {}
        for _, r in frame.iterrows():
            name = str(r["method"])
            exact = "Exact" in name
            ax.scatter(r[xcol], r[ycol], s=34 if exact else 24, zorder=3,
                       marker="*" if exact else "o",
                       color="#b3283a" if exact else "#1b3a6b")
            key = round(float(r[ycol]), 2)
            k = offs.get(key, 0)
            offs[key] = k + 1
            ax.annotate(name.replace(", summed", "").replace(" (disclosed)", ""),
                        (r[xcol], r[ycol]), textcoords="offset points",
                        xytext=(-6 if r[xcol] > 8 else 5, -2 + 7 * k),
                        ha="right" if r[xcol] > 8 else "left", fontsize=5.6)

    if df is not None:
        ax = axes[0]
        scatter(ax, df, "ms_per_explanation", "spearman", True)
        ax.set_title("(a) Feature level (16 players)")
        ax.set_ylabel(r"Spearman $\rho$ vs exact feature Shapley")
        ax.set_ylim(0.45, 1.06)
    if dg is not None:
        ax = axes[1]
        scatter(ax, dg, "ms_per_explanation", "spearman", True)
        ax.set_title("(b) Group level (8 players, disclosed)")
        ax.set_ylabel(r"Spearman $\rho$ vs exact group Shapley")
        ax.set_ylim(0.45, 1.06)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Cost per explanation (ms, log scale)")
        ax.set_xlim(8e-4, 2e3)
    fig.tight_layout()
    _save(fig, "fig_explanation_frontier")
    return True


# ==========================================================================
# 4. Per-scenario detection
# ==========================================================================
def fig_per_scenario() -> bool:
    rows = _json("main_rows")
    if rows is None:
        return False
    want = ["HEAAF", "HEAAF-SupervisedRisk", "B3-RBA"]
    acc = {}
    for r in rows:
        if r["policy"] in want and r.get("per_attack"):
            acc.setdefault(r["policy"], []).append(r["per_attack"])
    if not acc:
        return False
    names = sorted({k for v in acc.values() for d in v for k in d})
    short = {n: ATTACKS[n]["label"] if n in ATTACKS else n for n in names}
    fig, ax = plt.subplots(figsize=(COL2_W, 2.4))
    w = 0.8 / max(len(acc), 1)
    for i, (pol, ds) in enumerate(acc.items()):
        m = [np.mean([d.get(n, np.nan) for d in ds]) for n in names]
        e = [np.std([d.get(n, np.nan) for d in ds]) for n in names]
        ax.bar(np.arange(len(names)) + i * w, m, width=w, yerr=e, capsize=1.6,
               color=PALETTE.get(pol, "#666"), label=pol,
               error_kw={"lw": 0.6})
    ax.set_xticks(np.arange(len(names)) + w * (len(acc) - 1) / 2)
    ax.set_xticklabels([short[n] for n in names], fontsize=6.2)
    ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=len(acc), loc="upper center", bbox_to_anchor=(0.5, 1.22))
    _save(fig, "fig_per_scenario")
    return True


# ==========================================================================
# 5. Template ageing
# ==========================================================================
def fig_drift() -> bool:
    eer = _csv("table_eer")
    drift = _csv("table_drift")
    if eer is None:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 2.3))
    ax = axes[0]
    ax.errorbar(range(len(eer)), eer.eer_mean, yerr=eer.eer_sd, marker="o",
                markersize=3.5, color="#1b3a6b", capsize=2, lw=1.2)
    ax.set_xticks(range(len(eer)))
    ax.set_xticklabels(eer.session_block)
    ax.set_xlabel("Test session (later = more template ageing)")
    ax.set_ylabel("Keystroke EER")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("(a) Behavioural channel degrades with age")
    ax = axes[1]
    if drift is not None:
        col = "GMean_mean" if "GMean_mean" in drift.columns else "GMean"
        sd = "GMean_std" if "GMean_std" in drift.columns else None
        for pol, g in drift.groupby("policy"):
            g = g.sort_values("session_block")
            ax.errorbar(g.session_block, g[col],
                        yerr=(g[sd] if sd else None), capsize=2,
                        marker=MARKER.get(pol, "o"), markersize=3.5,
                        color=PALETTE.get(pol, "#666"), label=pol)
        ax.legend(loc="best")
    ax.set_xlabel("Test session")
    ax.set_ylabel("G-mean")
    ax.set_title("(b) End-to-end effect on detection")
    fig.tight_layout()
    _save(fig, "fig_drift")
    return True


# ==========================================================================
# 6. RL training curve
# ==========================================================================
def fig_training() -> bool:
    paths = sorted(DATA_PROC.glob("art_seed*.pkl"))
    if not paths:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 2.2))
    got = False
    for p in paths:
        try:
            with open(p, "rb") as fh:
                art = pickle.load(fh)
            log = art.get("log")
            if log is None or not log.steps:
                continue
        except Exception:
            continue
        got = True
        lab = p.stem.replace("art_", "")
        axes[0].plot(log.steps, log.ret, lw=0.9, alpha=0.85, label=lab)
        axes[1].plot(log.steps, log.loss, lw=0.9, alpha=0.85, label=lab)
    if not got:
        plt.close(fig)
        return False
    axes[0].set_xlabel("Environment steps"); axes[0].set_ylabel("Mean episode return")
    axes[0].set_title("(a) Return")
    axes[1].set_xlabel("Environment steps"); axes[1].set_ylabel("TD loss")
    axes[1].set_yscale("log"); axes[1].set_title("(b) Temporal-difference loss")
    axes[1].legend(ncol=2, fontsize=5.6)
    fig.tight_layout()
    _save(fig, "fig_training")
    return True


# ==========================================================================
# 7. Group attribution signature per attack scenario
# ==========================================================================
def fig_group_signature() -> bool:
    npz = RESULTS / "group_phi.npz"
    splits = DATA_PROC / "splits.pkl"
    if not npz.exists() or not splits.exists():
        return False
    z = np.load(npz, allow_pickle=True)
    with open(splits, "rb") as fh:
        te = pickle.load(fh)["test"].reset_index(drop=True)
    phi, idx = z["phi_group_exact"], z["idx"]
    atk = te["attack"].to_numpy()[idx]
    names = [a for a in sorted(set(atk)) if a]
    if not names:
        return False
    Mrows = []
    for a in names:
        m = phi[atk == a].mean(axis=0)
        s = np.abs(m).sum() + 1e-12
        Mrows.append(m / s)
    Mx = np.array(Mrows)
    fig, ax = plt.subplots(figsize=(COL2_W, 0.42 * len(names) + 1.5))
    v = np.abs(Mx).max()
    im = ax.imshow(Mx, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(GROUP_NAMES)))
    ax.set_xticklabels(GROUP_NAMES, rotation=32, ha="right", fontsize=6.2)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([ATTACKS[a]["label"] if a in ATTACKS else a for a in names],
                       fontsize=6.4)
    ax.grid(False)
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            if abs(Mx[i, j]) > 0.12:
                ax.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center",
                        fontsize=5.4,
                        color="white" if abs(Mx[i, j]) > 0.55 * v else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cb.set_label("Normalised group attribution", fontsize=6.4)
    cb.ax.tick_params(labelsize=6)
    _save(fig, "fig_group_signature")
    return True


# ==========================================================================
# 8. Ablation effects
# ==========================================================================
def fig_ablation() -> bool:
    df = _csv("table_ablation")
    if df is None or "GMean_mean" not in df:
        return False
    full = df[df.policy.str.contains("full", case=False)]
    if full.empty:
        return False
    base = float(full.GMean_mean.iloc[0])
    d = df[~df.policy.str.contains("full", case=False)].copy()
    d["delta"] = d.GMean_mean - base
    d = d.sort_values("delta")
    fig, ax = plt.subplots(figsize=(COL_W, 0.30 * len(d) + 1.0))
    cols = ["#b3283a" if v < 0 else "#0f7b6c" for v in d.delta]
    err = d.GMean_std.fillna(0.0) if "GMean_std" in d else None
    ax.barh(range(len(d)), d.delta, color=cols, height=0.62,
            xerr=err, capsize=1.8, error_kw={"lw": 0.6})
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([s.replace("w/o ", "− ") for s in d.policy], fontsize=6.2)
    ax.set_xlabel(r"$\Delta$ G-mean versus full HEAAF")
    _save(fig, "fig_ablation")
    return True


FIGURES = {
    "pareto": fig_pareto,
    "reliability": fig_reliability,
    "explanation": fig_explanation_frontier,
    "scenario": fig_per_scenario,
    "drift": fig_drift,
    "training": fig_training,
    "signature": fig_group_signature,
    "ablation": fig_ablation,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, choices=list(FIGURES))
    args = ap.parse_args()
    todo = args.only or list(FIGURES)
    made, skipped = 0, []
    for name in todo:
        try:
            if FIGURES[name]():
                made += 1
            else:
                skipped.append(name)
        except Exception as exc:                      # keep going
            skipped.append(f"{name} ({type(exc).__name__}: {exc})")
    print(f"\n  {made} figure(s) written to {FIGDIR}")
    if skipped:
        print("  skipped (inputs not present): " + ", ".join(skipped))


if __name__ == "__main__":
    main()
