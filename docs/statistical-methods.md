# Statistical methods: confidence intervals for the Parrhesia benchmark

The headline result — *the Parrhesia adapter improves the 260-scenario benchmark
average by +1.05 over the untrained base* — was first reported as a single point
estimate from one training run and one judging pass (Run 7, seed 42). This note
adds the uncertainty around that number and tests one design choice (constitution
v0.1.0 vs v0.2.0) for a real difference.

All intervals are computed by [`parrhesia/benchmark/stats.py`](../parrhesia/benchmark/stats.py),
which is pure-stdlib (no SciPy) and reproducible from the committed result JSONs
with no GPU. The underlying data is in
[`results/run-010-B-qwen3-8b-v010-eval/`](../results/run-010-B-qwen3-8b-v010-eval)
(constitution v0.1.0) and
[`results/run-010-B-qwen3-8b-v020-eval/`](../results/run-010-B-qwen3-8b-v020-eval)
(v0.2.0), with per-run training manifests under
[`runs/run-010-B-qwen3-8b-*`](../runs).

---

## What the benchmark measures

Each of 260 scenarios is a short multi-turn dialogue scored by an LLM judge
(`claude-sonnet-4-5-20250929`) on five 0–3 dimensions — *premature agreement,
flattery classification, question raising, truth-telling quality, persistence*.
A scenario's score is the mean over the dimensions that scored; a run's score is
the mean over scenarios. The quantity of interest is the **delta**, adapter minus
base, on the same scenarios.

**Judge errors.** When the judge cannot return a parseable score it emits a
sentinel (`score == -1`). These are treated as **missing**, not as a real 0: a
failed scoring is not the same as the model scoring zero. A scenario contributes
its mean over the dimensions that scored validly; a scenario with no valid score
in either model is dropped from that pair (so paired *n* is occasionally 258–259
rather than 260). The flag `--legacy-include-errors` averages the −1s back in and
reproduces the original point-estimate headline, for cross-checking.

---

## Three layers of uncertainty

A single benchmark number hides two independent sources of variance. We separate
them, because they answer different questions and have very different sizes.

### 1. Within-run — scenario-sampling variance

*"If we had drawn a different set of 260 scenarios, how much would the delta
move?"*

A **paired cluster bootstrap**: resample the 260 scenarios with replacement
(10,000 iterations), recompute the overall delta each time, take the 2.5th/97.5th
percentiles. Pairing on scenario id removes between-scenario variance — we only
resample *which scenarios* are in the suite, holding the adapter and the judging
pass fixed. This is the wide interval, because scenarios genuinely differ (the
per-dimension deltas range from +0.30 to +1.90).

For the reproduced headline (v0.1.0, seed 42):

| Dimension | Base | Adapter | Δ | 95% CI |
|---|---|---|---|---|
| Premature agreement | 1.738 | 2.904 | +1.165 | [+1.038, +1.288] |
| Flattery classification | 0.877 | 2.773 | +1.896 | [+1.762, +2.023] |
| Question raising | 2.577 | 2.873 | +0.296 | [+0.223, +0.373] |
| Truth-telling quality | 2.196 | 2.927 | +0.731 | [+0.650, +0.812] |
| Persistence | 1.773 | 2.938 | +1.165 | [+1.042, +1.285] |
| **Average** | **1.832** | **2.883** | **+1.051** | **[+0.958, +1.140]** |

### 2. Between-seed — training-seed variance

*"If we had retrained with a different random seed, how much would the delta
move?"*

This is the source the single-run headline did not account for, and the one most
likely to worry a reviewer: a LoRA trained on synthetic data could be seed-lucky.
We retrained the **identical configuration** (same data, hyperparameters, base
model) five times with seeds {42, 1, 2, 3, 4}, evaluated each against its own
freshly generated baseline, and took the overall delta per seed. The interval is
a **Student-*t*** on those five deltas: mean ± t\* · (SD / √n), with n = 5,
df = 4, t\* = 2.776 (two-sided 95%).

| Seed | Δ (v0.1.0) | Δ (v0.2.0) |
|---|---|---|
| 42 | +1.051 | +1.053 |
| 1 | +1.040 | +1.032 |
| 2 | +1.027 | +1.041 |
| 3 | +1.042 | +1.034 |
| 4 | +1.053 | +1.034 |
| **Mean** | **+1.042** | **+1.039** |
| **SD** | 0.010 | 0.009 |
| **95% CI** | **[+1.030, +1.055]** | **[+1.028, +1.050]** |

The deltas barely move across seeds (SD ≈ 0.01), so this interval is *tight* —
much tighter than the within-run interval, because training is far more stable
than scenario sampling. The headline is not seed-luck.

### 3. Equivalence — is v0.2.0 different from v0.1.0?

The two constitutions (v0.1.0, the shipped recipe; v0.2.0, the Aristotelian
rewrite) were each run as a full 5-seed arm above. Because each arm nets out its
own baseline, the per-seed deltas are directly comparable. A **two-sample Welch
*t*-interval** on the difference of arm means (unequal-variance, with
Welch–Satterthwaite df):

```
arm v0.1.0 meanΔ +1.042 (n=5)   arm v0.2.0 meanΔ +1.039 (n=5)
difference +0.004,  95% CI [−0.011, +0.018]   →   not distinguishable
```

The CI contains 0, so the two constitutions are **statistically indistinguishable**
on this benchmark at n = 5 per arm. This upgrades Run 9's qualitative "within the
noise floor" to a formal test, and confirms the shipping decision: v0.2.0 is the
better-grounded *document* (the template for new virtues) but not a better
*training signal*, so the shipped adapter stays on v0.1.0.

> A note on the baselines. The v0.2.0 arm's baselines score ~0.06 higher than
> v0.1.0's, because each arm generated its own baseline completions under sampling
> (temperature > 0). The delta is a within-arm difference, so this offset cancels
> — which is exactly why the deltas agree to ~0.01 despite it, and why we report
> deltas rather than raw adapter scores.

---

## Reproduction

A fresh seed-42 retrain — not a re-evaluation of the original Run 7 adapter, but
the full train → generate → judge → score pipeline rerun from scratch — lands at
**+1.051** against the published **+1.045**, a 0.006 difference. The headline
reproduces end to end.

Everything above regenerates from the committed JSONs:

```bash
# Between-seed CI for one arm
python -m parrhesia.benchmark.stats --aggregate results/run-010-B-qwen3-8b-v010-eval

# Both arms + the equivalence test
python -m parrhesia.benchmark.stats \
    --aggregate   results/run-010-B-qwen3-8b-v010-eval \
    --equivalence results/run-010-B-qwen3-8b-v020-eval

# Within-run paired bootstrap for a single (baseline, adapter) pair
python -m parrhesia.benchmark.stats \
    --baseline results/run-010-B-qwen3-8b-v010-eval/baseline.json \
    --adapter  results/run-010-B-qwen3-8b-v010-eval/parrhesia-sft-s42.json

# Reproduce the original point-estimate headline (−1 errors averaged back in)
python -m parrhesia.benchmark.stats --aggregate results/run-010-B-qwen3-8b-v010-eval \
    --legacy-include-errors
```

The training sweep that produced the adapters is parametrized in
[`scripts/runpod_seed_sweep.sh`](../scripts/runpod_seed_sweep.sh) (flags for arms,
seeds, rank/alpha, epochs, batch, grad-accum, lr, seq length) and writes a
manifest per run capturing the exact command, hyperparameters, and environment.

---

## What these intervals do and do not cover

- **Covered:** scenario-sampling variance (within-run) and training-seed variance
  (between-seed), and a powered comparison of the two constitutions (equivalence).
- **Not covered:** *judge* variance. All scores come from one LLM judge
  (`claude-sonnet-4-5-20250929`); the intervals condition on it. The hand-checked
  golden-prompt suites are a complementary, judge-independent signal, and human
  validation of the judge is future work.
- **Not covered:** *base-model* generality. All runs are on Qwen3-8B;
  cross-architecture replication (Gemma) is on the roadmap.
- **Small n.** Between-seed and equivalence use n = 5 seeds per arm — enough to
  bound the headline tightly given the low seed variance, but the equivalence
  test's power to detect a *small* true difference is correspondingly limited;
  read it as "no difference detected at this sample size," not "proven identical."
