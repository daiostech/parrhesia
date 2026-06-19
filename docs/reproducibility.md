# Reproducibility ledger

What "reproducible" means here, honestly — and exactly where the seams are. The
strongest guarantee is **artifact preservation**, not bit-identical re-runs, because
every result touches an LLM (teacher, judge, user-sim) and a GPU, neither of which is
bit-deterministic. So we state three tiers and an explicit gap list rather than claim
a precision we don't have.

## Three tiers

| Tier | Claim | Status |
|---|---|---|
| **1 — re-evaluate the exact artifact** | Re-judge the committed generations, or re-eval the preserved adapter → the same scores within judge variance | ✅ holds for **every** run |
| **2 — re-train / re-generate from scratch** | Same data + hyperparameters + code + env → result within the **measured** variance band | ✅ given the inputs; ⚠ exact *code* only where the SHA is clean (see below) |
| **3 — bit-identical** | Byte-for-byte identical weights/scores | ✗ never — LLM APIs and GPU kernels aren't bit-stable |

**The measured band.** Tier 2 is meaningful because we *quantified* the residual
nondeterminism: across 5 retrains of the headline config the per-seed delta has
SD ≈ 0.01 (see [`docs/statistical-methods.md`](statistical-methods.md)). "Reproducible"
means "inside that band," and the band is published.

## What is preserved (Tier 1, all runs)

- **Trained adapters** — on the HuggingFace Hub (+ local, md5-verified).
- **Exact generations** — the model responses are committed inside each `results/<run>/…json`, so re-judging is exact.
- **Training data** + the **8 curated revision examples** (`data/examples/phronesis_revisions.json`), the **benchmark scenarios**, and the **rubric** — all committed.
- **Hyperparameters, seed, and resolved env versions** — in each `runs/<run>/manifest.yaml`.

## The gap ledger (five classes)

1. **Code provenance.** `git_sha` is **blank** for `run-001` and `run-010` (they ran from an scp'd, non-git working tree). Separately, six runs are **`git_dirty: true`** — a SHA *was* recorded but there were **uncommitted local edits at run time**, including **`run-007`, the +1.045 headline**. Only `run-006` and `run-009` are cleanly pinned (SHA set, not dirty). Prompts, the judge, and the rubric live in code, so they inherit this gap — and the judge-caching incident proved prompt structure moves scores.
2. **Base model.** Manifests record the *name* (`Qwen/Qwen3-8B`) with **no revision SHA**, and with 4-bit loading Unsloth actually pulls `unsloth/qwen3-8b-unsloth-bnb-4bit` — so the recorded base was the *request*, not the *resolved artifact*.
3. **Environment.** Versions were recorded for ~9 packages only, and `requirements-train.txt` is all **ranges, no pins** — so `pip install` resolves different versions over time and the exact env wasn't recreatable from the repo.
4. **Prompt/rubric hashing.** Inputs are versioned only via the (gappy) `git_sha`; nothing hashed them independently.
5. **Inherent nondeterminism.** Claude teacher (data-gen) + Claude judge/user-sim/scorer (eval) + GPU/vLLM are not bit-stable → caps everything at Tier 2.

## Per-run status

| Run | Code pinned | Base rev | Tier 1 (re-eval) | Tier 2 (re-train) |
|---|---|---|---|---|
| run-001-A | blank SHA | — | ✅ | ⚠ code unknown |
| run-002/003/004/005-A | SHA **dirty** | — | ✅ | ⚠ SHA + lost local diff |
| run-006-B | SHA clean | — | ✅ | ✅ (code clean) |
| **run-007-B** (headline) | SHA **dirty** | — | ✅ | ⚠ SHA + lost local diff |
| run-008-A | SHA **dirty** | — | ✅ | ⚠ SHA + lost local diff |
| run-009-B (v0.2.0) | SHA clean | — | ✅ | ✅ (code clean) |
| run-010-B (multi-seed) | blank SHA (scp'd) | — | ✅ | ⚠ code ≈ `ae39535`–`77eaee5` era |
| **run-011-B** (Gemma, this run) | **`4c2b068`, clean clone** | **pinned** | ✅ | ✅ **first fully-pinned run** |

The historical manifests are left **as captured** — they're the authentic record of
what was and wasn't recorded at run time; reconstructed values live here in the ledger,
never silently written back into them.

## What the hardening changes (run-011 onward)

The manifest system was hardened so this never recurs:

- **`base_model_revision`** resolved from HF and recorded (`manifest.py:resolve_model_revision`).
- **`runs/<id>/requirements.lock`** — a full `pip freeze` (`manifest.py:write_env_lock`) — the exact, recreatable env, not just ~9 summary versions; env capture also expanded (driver, OS, bitsandbytes, accelerate, vllm, …).
- **Clean git clone on the pod** so `git_sha` is captured and `git_dirty` is honest (run-011 = `4c2b068`, clean).
- **Taxonomy + constitution parametrized and md5-recorded** (`generate_sft.py --taxonomy/--constitution`), so the data-generation step pins its exact inputs by content.

Net: every run is **re-evaluable exactly today**; run-011 is the first that is also **re-trainable to within the measured band from a fully pinned recipe**.
