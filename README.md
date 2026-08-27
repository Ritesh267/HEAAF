# HEAAF — Hybrid Explainable Adaptive Authentication Framework

Reference implementation and evaluation harness for *Explainable Adaptive
Continuous Authentication for Smart Healthcare: Exact Inline Attribution and a
Clinical Safety Valve under Zero Trust*.

Everything in the paper — every table, every figure, every number quoted in the
prose — is produced by the code in this repository. Nothing is transcribed by
hand.

---

## What this is

An adaptive continuous-authentication system with four layers:

1. **Behavioural and contextual evidence** — 16 features that partition
   exactly into 8 semantic groups.
2. **Adaptive risk evaluation** — a Double DQN whose challenge advantage is
   Platt-calibrated into a risk probability.
3. **Explainable Decision Layer** — *exact* Shapley attribution over the 8
   semantic groups, computed inside the decision cycle.
4. **Adaptive MFA trigger** — a four-action graded ladder with a clinical
   safety valve that cannot deny a declared emergency.

The behavioural channel is the real CMU keystroke-dynamics benchmark
(Killourhy & Maxion, DSN 2009). Clinical context, roles, shifts, device estate,
resource sensitivity and labelled attacks are generated, because no public
corpus couples keystroke biometrics to clinical role and labelled insider
compromise. See "Honest limitations" below.

### Headline findings

| | |
|---|---|
| Exact **group** Shapley (2^8 = 256 coalitions) | **0.44 ms** p50 |
| Exact **feature** Shapley (2^16 = 65,536) | 53.3 ms |
| End-to-end decision + **exact** explanation | **0.611 ms** |
| End-to-end decision + *approximate* IG-16 | 1.214 ms |

The exact explanation is *cheaper* than the approximate one, because 256
coalition states are one batched forward pass while integrated gradients need
16 sequential ones.

The safety valve, holding detector and friction budget fixed, drops emergency
denials from 1.099% to 0.000% for a G-mean cost of 0.005 [0.002, 0.009].

The reinforcement-learning risk score is **not** the strongest detector: a
supervised detector inside the same policy engine gains 0.378 G-mean
[0.283, 0.469]. We report this negative result in full; it relocates the
contribution of RL to the action layer.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+. CPU only — no GPU is used or needed. The neural network is a
small NumPy MLP; there is no deep-learning framework dependency.

The CMU keystroke CSV is expected at `data/raw/DSL-StrongPasswordData.csv`
(included). Source: <https://www.cs.cmu.edu/~keystroke/>.

---

## Reproducing the paper

The pipeline is **staged and resumable**: each stage reads persisted artefacts
from the previous one, so you can re-run a single experiment without repeating
everything.

```bash
# 1. Build the corpus (~25 s)
python scripts/stage.py data --events 140000

# 2. Train one seed (~155 s each). The paper uses seeds 0-7.
for s in 0 1 2 3 4 5 6 7; do
  python scripts/stage.py train --seed $s --events 140000 --episodes 90000
done

# 3. Main comparison + significance testing (~10 min)
python scripts/stage.py main --seeds 0,1,2,3,4,5,6,7 --budget 0.01 --pareto 3

# 4. Explanation benchmarks
python scripts/stage.py explain      --seed 0 --n-instances 300
python scripts/stage.py groupexplain --seed 0 --n-instances 500
python scripts/stage.py latency      --seed 0

# 5. Ablations (paper uses seeds 0-4; ~9 min per seed)
for s in 0 1 2 3 4; do
  python scripts/stage.py ablation --which -1 --seed $s \
      --events 140000 --episodes 90000 --budget 0.01
done
python scripts/stage.py tables

# 6. Template ageing (paper uses seeds 0-2)
for s in 0 1 2; do
  python scripts/stage.py drift --seed $s --events 140000 --episodes 90000
done

# 7. Figures and LaTeX tables
python scripts/make_figures.py
  python scripts/make_tables.py
python scripts/make_tables.py
```

Total wall time is roughly 90 minutes on a single modern CPU core.

For a fast smoke test of the whole path, shrink everything:

```bash
python scripts/stage.py data  --events 15000
python scripts/stage.py train --seed 0 --episodes 8000
python scripts/stage.py main  --seeds 0 --budget 0.01 --pareto 0
```

### Outputs

| Path | Contents |
|---|---|
| `results/tables/*.csv` | raw result tables |
| `results/tables/tex/*.tex` | publication-ready LaTeX fragments |
| `results/tables/tex/numbers.json` | every number quoted in the paper |
| `results/figures/*.pdf` | all eight figures, vector PDF |
| `results/logs/*.json` | per-seed records and configuration |
| `data/processed/` | trained agents, splits, cached artefacts |

`paper/main.tex` inputs the LaTeX fragments directly, so regenerating the
tables regenerates the manuscript's numbers.

---

## Tests

```bash
pytest -q          # 35 invariant tests, ~4 s
```

These check the properties the paper's argument rests on, not merely that the
code runs:

- exact Shapley satisfies **efficiency, symmetry and the dummy axiom**, and
  matches the closed form on a linear model;
- **group and feature games coincide** under pairwise interaction and
  **diverge** under three-way interaction — the claim in Section IV-D;
- the **safety valve never denies a declared emergency**, checked exhaustively
  across the risk range;
- the policy is **monotone in risk**, and batch and scalar paths agree;
- the **subject disclosure never contains the numeric score** and never names
  more than one group;
- integrated gradients are exact for quadratic models and converge on a
  non-polynomial one;
- the scaled-Manhattan verifier **reproduces the published EER range**;
- the generator is **deterministic under a seed**.

---

## Layout

```
src/heaaf/
  config.py      features, the 8-group partition, attacks, all hyperparameters
  simulator.py   clinical context generator and attack injection
  keystroke.py   CMU corpus loading, templates, scaled-Manhattan verifier
  agent.py       Double DQN, replay, environment, calibrated risk head
  nn.py          minimal NumPy MLP (forward, backward, Adam)
  policy.py      graded ladder, sensitivity floor, clinical safety valve
  explain.py     exact feature/group Shapley, KernelSHAP, IG, disclosure
  baselines.py   B3-RBA, B4-XAI-Static, B5-RLAuth-style
  metrics.py     detection, friction, containment, calibration, fidelity
  stats.py       bootstrap CIs, paired Wilcoxon, Holm-Bonferroni
  bootstrap.py   exact multinomial paired bootstrap for binary metrics
  pipeline.py    the experiments
scripts/
  stage.py         staged runner (data|train|main|explain|groupexplain|
                   latency|ablation|drift|tables)
  make_figures.py  all eight figures
  make_tables.py   all LaTeX tables + numbers.json
tests/             invariant test suite
paper/             manuscript, bibliography, figures, tables
```

---

## Design notes worth knowing

**Why 16 features and not 17.** An earlier revision injected a supervised
gradient-boosting score as a 17th feature shared by every policy. That is
withdrawn. It stacked a supervised detector underneath both the RL agent *and*
the supervised baseline it is compared against, making the central comparison
uninterpretable; measured out-of-sample it also *degraded* the random forest
(AUROC 0.931 with, 0.949 without). Removing it also restores an exact 16 -> 8
partition, which is what makes group-level exactness affordable.

**Why baselines keep their own action rule.** Running a baseline through
HEAAF's policy engine transplants the contribution under test into the
baseline. It also makes `B4-XAI-Static@budget` numerically identical to
`HEAAF-SupervisedRisk`. Budget-matched variants therefore move only the
threshold and keep the baseline's native action mapping.

**Why ablations use a matched friction budget.** Comparing arms at a fixed
threshold conflates components that improve ranking with components that merely
shift the score distribution. Every ablation arm is given the threshold that
spends the same 1% budget on validation.

**Why the bootstrap is multinomial.** Every metric compared is the mean of a
0/1 indicator over a fixed stratum, and both policies see the same events, so a
resample is described completely by a multinomial draw over four joint cells.
This is exactly equivalent to resampling indices and reduces cost from
O(n_boot * n) to O(n_boot).

---

## Honest limitations

- **Clinical context is generated.** The behavioural channel is real and the
  session split contains genuine template ageing, but roles, shifts, devices,
  geography, sensitivity and attack labels come from a parameterised generator.
  Absolute rates are properties of the generator, not estimates of hospital
  reality. The testbed supports *relative* comparison under identical
  conditions, which is what every claim in the paper is.
- **Generator-detector circularity.** Detectors are evaluated on data whose
  generative process is known. Per-session stealth, partial driver activation
  and a hard-negative class mitigate this only partly.
- **One behavioural modality.** Fixed-text password typing on a desktop
  keyboard. Touch, free-text and voice have different error characteristics.
- **No adaptive adversary.** Attacks do not respond to the defence. Given that
  the system returns explanations to the person being challenged, this is the
  most important missing experiment.
- **Cost parameters are illustrative.** Clinician-seconds per action and breach
  costs are plausible published estimates, not measurements from a partner
  site. They are all in `config.py` so they can be varied.
- **Eight seeds is a small sample for a rank test.** With n = 8 the smallest
  attainable two-sided Wilcoxon p is 0.0078, so after Holm correction that test
  cannot reach significance for any effect. No claim rests on it; all
  significance claims come from the event-level bootstrap.

---

## Citation

```bibtex
@inproceedings{mukherjee2026heaaf,
  author = {Mukherjee, Ritesh},
  title  = {Explainable Adaptive Continuous Authentication for Smart
            Healthcare: Exact Inline Attribution and a Clinical Safety
            Valve under Zero Trust},
  year   = {2026}
}
```

## License

Code released for academic use. The CMU keystroke-dynamics dataset is subject
to its own terms; see <https://www.cs.cmu.edu/~keystroke/>.
