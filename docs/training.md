# Training & evaluation guide

Operational runbook for training a Parrhesia adapter and running the benchmark on GPU. For the conceptual overview and results, see the main [README](../README.md). The canonical pipeline is **Approach B (direct SFT)**; Approach A (OCT) and Approach C (DPO triplets) are documented in [`log.md`](../log.md).

## Prerequisites

- **Anthropic API key** in `.env` (`ANTHROPIC_API_KEY=...`) — data generation and the LLM judge.
- **HuggingFace token** in `.env` (`HF_TOKEN=...`) — push/pull adapters and eval results.
- **A GPU host.** Reference config: RunPod RTX 4090, **50 GB container disk + 100 GB volume**. Run 8+ keeps both the train and serve venvs on the pod simultaneously, so 50 GB of volume tips over to "Disk quota exceeded"; 100 GB has headroom.

## Dependency separation

Training (Unsloth) and serving (vLLM) require incompatible torch versions and cannot coexist in one environment. Each phase installs from its own requirements file, and `runpod_setup.sh` creates both venvs:

| File | Phase | Key deps |
|---|---|---|
| `requirements.txt` | Local data gen / reporting | anthropic, datasets |
| `requirements-train.txt` | SFT/DPO training | unsloth, peft, trl |
| `requirements-serve.txt` | Evaluation | vllm |

## Step 1 — Generate SFT data (local)

Uses Claude Sonnet to generate parrhesia-demonstrating conversation pairs across the 10 Aristotelian scenario categories, then filters with a Claude judge.

```bash
python -m parrhesia.data.generate_sft \
  --num-per-category 200 \
  --output-dir data/generated/sft \
  --filter
```

Produces ~1,300 pairs (10 categories × ~130 each, after the quality filter) plus 100 examples targeting specific failure modes from earlier-run analysis. Cost: ~$8 in Claude API calls.

## Step 2 — Phronesis revision (local, recommended)

Run 7's headline result came from this step. Raw SFT data is correct in *content* but the trained model over-corrects in emotionally sensitive contexts ("your father was wrong" about a deceased parent's advice). This pass scores every pair on a 1–3 delivery rubric, then revises score-≤2 pairs to acknowledge the user's situation before delivering the same truth.

```bash
python -m parrhesia.data.revise_sft \
  --input data/generated/sft/training_pairs_filtered.jsonl \
  --output data/generated/sft/training_pairs_revised.jsonl \
  --scores data/generated/sft/scores.jsonl \
  --examples data/examples/phronesis_revisions.json \
  --threshold 2 --concurrency 10
```

In Run 7 this revised 899 of 1,334 examples (67%). The threshold matters: revising only score-1 examples (Run 6, 17%) didn't shift the model; revising score-2 lighter-touch grafts as well (Run 7, 67%) did. Cost: ~$2.

## Step 3 — SFT training (GPU)

On a RunPod pod (RTX 4090, 50 GB container + 100 GB volume):

```bash
# Clone and set up both venvs
cd /workspace && git clone https://github.com/daiostech/parrhesia.git && cd parrhesia
bash scripts/runpod_setup.sh

# Upload .env and revised training data from your machine:
#   scp -P <PORT> path/to/.env root@<pod-ip>:/workspace/parrhesia/
#   mkdir -p first if needed, then:
#   scp -P <PORT> data/generated/sft/training_pairs_revised.jsonl \
#       root@<pod-ip>:/workspace/parrhesia/data/generated/sft/

source .venv-train/bin/activate
python -m parrhesia.train.sft \
  --model Qwen/Qwen3-8B \
  --data data/generated/sft/training_pairs_revised.jsonl \
  --output adapters/parrhesia-sft-8b \
  --lora-r 64 --lora-alpha 128 \
  --epochs 3 --batch-size 2 --grad-accum 16 \
  --lr 2e-4 --max-seq-length 2048 \
  --run-id run-00X-B-qwen3-8b --push
```

Training takes ~13 min on a 4090. `--run-id` records hyperparameters and metrics into `runs/<id>/manifest.yaml`; `--push` uploads the adapter to `daios/parrhesia-sft-8b` on Hub under a branch named after the run ID.

## Step 4 — Evaluate (GPU)

The eval script serves base + adapter via vLLM, runs the 260-scenario quantitative benchmark and the standard + hard golden-prompt suites against both, and judges with the Claude API.

```bash
# Generate scenarios first (local, one-time):
python -m parrhesia.benchmark.generate_scenarios --num-per-category 25

# On the pod:
bash scripts/runpod_eval.sh \
  --run-id run-00X-B-qwen3-8b \
  --base-model Qwen/Qwen3-8B \
  --adapters "parrhesia-sft=./adapters/parrhesia-sft-8b" \
  --training-data data/generated/sft/training_pairs_revised.jsonl \
  --prompt-key messages.0.content
```

Evaluates 260 quantitative scenarios + standard + hard golden prompts × 2 models. Takes ~2–3 hours. Results land in `results/<run-id>/` and auto-push to `daios/parrhesia-eval-results`.

**Contamination filtering.** The eval script applies the LLM contamination check to adapter iterations (their training data may topically overlap golden prompts) and skips it for the baseline (it has no training data to be contaminated by). See `log.md` → "Methodology footnote: contamination-check bug fix (2026-04-30)".

## Step 5 — Qualitative diff and reporting (local)

```bash
scp -r -P <PORT> root@<pod-ip>:/workspace/parrhesia/results/run-00X-B-qwen3-8b/ ./results/

parrhesia report results/run-00X-B-qwen3-8b/baseline.json
parrhesia report results/run-00X-B-qwen3-8b/parrhesia-sft.json --format html

# Compare your run against Run 7 (the published headline)
parrhesia qualitative-diff \
  results/run-007-B-qwen3-8b/qualitative-hard-parrhesia-sft.json \
  results/run-00X-B-qwen3-8b/qualitative-hard-parrhesia-sft.json
```

The two qualitative suites:
- **Standard** (`golden_prompts.jsonl`, 10 prompts) — regression tests the base model should pass. Catches obvious breakage.
- **Hard** (`golden_prompts_hard.jsonl`, 10 prompts) — sycophancy-research patterns (are-you-sure flips, fabricated citations, stated preferences, multi-turn escalation, the universal `should_distinguish_watchful_waiting` failure) that reliably break models. This is where training should show improvement.

## RunPod tips

- **Container disk: 50 GB minimum.** Model weights (~16 GB) + pip packages (~15–20 GB) fill the default 20 GB.
- **Volume disk: 100 GB for Run 8+.** Both `.venv-train` (~13 GB) and `.venv-serve` (~12 GB) plus adapters, generated data, and the HF model cache (~16 GB for Qwen3-8B) fit comfortably; 50 GB tips over to "Disk quota exceeded".
- **`mkdir -p /workspace/parrhesia/adapters`** before `scp`-ing the adapter — the directory doesn't exist after a fresh clone.
- **Stop vs terminate.** If you'll reuse the pod soon, *stop* (preserves state) instead of *terminate* (destroys everything). Otherwise `scp` results back and terminate.
