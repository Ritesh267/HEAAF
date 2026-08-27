# Changelog

## Revision 2 — submission revision

This revision corrects several methodological defects in revision 1. Three of
them affected reported numbers, so **every result was regenerated from
scratch**; no figure from revision 1 was carried forward.

### Corrections that changed results

**1. Removed the stacked Tier-0 feature (`risk0`).**
Revision 1 fitted a supervised gradient-boosting detector on the training split
and injected its out-of-sample score as a 17th input feature consumed by *every*
policy. This was wrong for two independent reasons:

- It placed a supervised detector underneath both the reinforcement-learning
  agent and the supervised baseline the agent is compared against, so the
  paper's central "RL versus supervised" comparison measured nothing
  interpretable.
- Measured on held-out data it *degraded* the supervised detector
  (AUROC 0.931 with the feature, 0.949 without).

Removing it also restores an exact 16-feature → 8-group partition, which is
what makes exact group-level Shapley affordable. It weakens the RL score
(AUROC ≈ 0.84 → 0.72), which makes the paper's negative result harder for us,
not easier.

**2. Budget-matched baselines no longer run inside HEAAF's policy engine.**
Revision 1 evaluated `B4-XAI-Static@budget` by routing the baseline's score
through HEAAF's graded ladder, sensitivity floor and safety valve. That
transplants the contribution under test into the baseline, and produced a row
numerically identical to `HEAAF-SupervisedRisk`. Budget-matched variants now
move only the threshold and keep each baseline's native action mapping.

**3. Ablations now use a matched friction budget.**
Revision 1 compared ablation arms at a *fixed* threshold pair, which conflated
components that improve ranking with components that merely shift the score
distribution. The "w/o risk calibration" arm made this visible: it appeared to
interrupt 98% of benign traffic, which measured the threshold rather than the
ablation. Each arm now receives the threshold that spends the same 1%
validation friction budget.

### Corrections that did not change results

**4. Documentation/implementation mismatch on feature count.**
The paper claimed 8 semantic groups and 2^8 coalitions; `explain.py` documented
16 features and 2^16; the code actually had 17 features and 2^17. All three now
agree at 16 features / 8 groups.

**5. `scripts/make_figures.py` did not exist** despite being referenced in the
README. The repository contained no figures at all. Eight figures are now
generated.

**6. No tests existed** despite the README advertising `pytest -q`. There are
now 35 invariant tests.

### Additions

- **Exact group-level Shapley** (`explain.exact_shapley_groups`,
  `exact_shapley_groups_batch`) over the 8 semantic groups, 256 coalitions,
  computed as one batched forward pass. This is the quantity now disclosed.
- **Role-scoped disclosure** (`explain.render_disclosure`,
  `disclosure_leakage`) — subject / analyst / audit projections of the same
  attribution vector, with the leakage quantified. The paper claimed this
  behaviour in revision 1; it was not implemented.
- **`stats.py`** — bootstrap confidence intervals, paired Wilcoxon signed-rank
  tests, Holm–Bonferroni correction.
- **`bootstrap.py`** — exact paired bootstrap for binary-indicator metrics via
  a multinomial draw over the four joint outcome cells. Equivalent to
  resampling indices, but O(n_boot) instead of O(n_boot·n), which is what makes
  4,000 replicates affordable for every contrast.
- **`scripts/make_tables.py`** — emits every LaTeX table in the manuscript plus
  `numbers.json`, directly from the persisted results, so the paper's numbers
  cannot drift from the run that produced them.
- **`HEAAF-Ensemble`** — rank-averaged RL + supervised score, recalibrated;
  demonstrates the architecture is detector-agnostic.
- **`w/o clinical safety valve`** ablation arm.
- **Exact group Shapley** added to the latency benchmark.
- Seed count raised from 1 to 8 for the main table (5 for ablations, 3 for
  drift), with the drift stage accumulating across seeds instead of
  overwriting.

### Fixed defects

- `HEAAF-Ensemble` initially produced ECE 0.494 because a rank average is not a
  probability; it is now passed through the same Platt map as every other
  score (ECE 0.005).
- The `main` stage re-scored the random forest several times per seed; baseline
  risk vectors are now cached per seed.
- `stage.py drift` overwrote `table_drift.csv` on each invocation instead of
  accumulating across seeds.
- Long identifiers and justified text overflowed narrow table columns in the
  generated LaTeX; identifiers now carry break opportunities and description
  columns are ragged-right.

### Known non-defects, documented rather than fixed

- **The safety-valve effect is not significant when contrasted against HEAAF's
  own no-valve ablation** (−0.0012, CI [−0.0052, 0.000]). This is expected: a
  valve that forbids denial only matters when the detector reaches the denial
  band, and the weak RL score rarely does. The effect is large and significant
  against the strong supervised detector (−0.0109, CI [−0.0149, −0.0072]).
  HEAAF's own emergency denial rate is 0 *by construction* and is proved
  exhaustively by test, not estimated.
- **Removing the graded ladder improves G-mean** (+0.034). Collapsing to
  allow/deny concentrates the friction budget into denials, which contain more
  sessions. G-mean does not see the cost; the clinician-hours and
  emergency-denial columns do.
- **Removing risk calibration changes G-mean by −0.000.** Once thresholds are
  budget-matched, calibration is a monotone reparameterisation and cannot
  change ranking. Its value is that the budget becomes expressible at all.

## Revision 1 — initial submission

Initial framework, generator, baselines and evaluation.
